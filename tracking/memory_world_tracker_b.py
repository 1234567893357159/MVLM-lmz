"""Full tracker adaptation for B, isolated from the running A/fusion-only code.

Histories are detached online observations. Backpropagate each search frame before
encoding the next, then synchronize the summed gradients once per clip batch.
"""
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lib.models.mvlm.mvlm import MVLM, build_mvlm
from lib.models.mvlm.temporal_core import TemporalState, predict_prior_batch, finish_location, observe
from lib.models.mvlm.world_state import search_crop, xywh_to_center, center_to_xywh, image_to_crop, extract_roi
from lib.test.tracker.utils import Preprocessor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, giou_loss
from lib.utils.heapmap_utils import generate_heatmap


class TrackerStageB(MVLM):
    @classmethod
    def from_base(cls, base):
        return cls(base.text_encoder, base.encoder, base.head, base.num_frames,
                   base.num_template, base.head_type, base.cfg)

    def configure_world_stage(self, stage):
        if stage != 'B' or not (self.world_enabled and self.memory_enabled):
            raise ValueError('Full tracker adaptation requires joint memory/world stage B')
        if self.cfg.TRAIN.FREEZE_ENCODER or not self.cfg.TRAIN.WORLD.TRAIN_HEAD:
            raise ValueError('Full tracker B must explicitly unfreeze encoder and CENTER')
        self.world_stage = stage
        self.requires_grad_(False)
        for module in (self.encoder, self.text_encoder.text_proj, self.head, self.future_fusion):
            module.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        # B here trains the visual encoder; the legacy MVLM.train freezes it.
        torch.nn.Module.train(self, mode)
        if self.world_stage is not None:
            for module in (self.text_encoder.clip, self.memory_reader, self.world_model, self.quality_head):
                if module is not None:
                    module.eval()
            if self.cfg.TRAIN.WORLD.FREEZE_BN_STATS:
                for module in self.head.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
        return self


def build_tracker_stage_b(cfg):
    return TrackerStageB.from_base(build_mvlm(cfg, training=False))


@torch.no_grad()
def frozen_text_tokens(text_encoder, texts):
    """Exactly TextEncoder.forward's frozen CLIP portion, before text_proj."""
    token = text_encoder.clip_tokenizer(texts).to(text_encoder.text_proj.weight.device)
    if token.ndim == 1:
        token = token[None]
    tower = text_encoder.clip
    value = tower.token_embedding(token).type(tower.dtype) + tower.positional_embedding.type(tower.dtype)
    value = tower.transformer(value.permute(1, 0, 2)).permute(1, 0, 2)
    return text_encoder.slice_token(value, token).type(text_encoder.dtype).detach()


def cm_per_example(language, visual, boxes):
    """Vectorized official CM objective (including its integer token mask).

    Return zero plus a validity mask for examples lacking positive/background
    tokens. Valid examples match lib.utils.cm_loss.CMloss(top_k=10).
    """
    side = math.isqrt(visual.shape[1])
    if side * side != visual.shape[1]:
        raise ValueError('CM requires a square search grid')
    language = F.normalize(language.float().mean(1), dim=-1, eps=1e-8)
    visual = F.normalize(visual.float(), dim=-1, eps=1e-8)
    similarity = (visual * language[:, None]).sum(-1)
    start = boxes[:, :2].clamp(0, 1)
    end = torch.minimum(torch.ones_like(start), start + boxes[:, 2:])
    bounds = torch.cat((start, end), -1).mul(side).long()
    positions = torch.arange(side, device=boxes.device)
    positive = ((positions[None, None, :] >= bounds[:, 0, None, None])
                & (positions[None, None, :] < bounds[:, 2, None, None])
                & (positions[None, :, None] >= bounds[:, 1, None, None])
                & (positions[None, :, None] < bounds[:, 3, None, None])).flatten(1)
    npos, nneg = positive.sum(1), (~positive).sum(1)
    rho = (similarity * positive).sum(1) / npos.clamp_min(1)
    margin = rho[:, None] - similarity
    exponent = (-margin * margin.abs()).masked_fill(positive, -float('inf'))
    hard = exponent.topk(min(10, visual.shape[1]), dim=1).values
    values = hard.exp().sum(1) / nneg.clamp(min=1, max=10)
    valid = (npos > 0) & (nneg > 0)
    return torch.where(valid, values, values * 0), valid


MEANS = ('loss', 'tracking_loss', 'giou_loss', 'l1_loss', 'focal_loss', 'cm_loss', 'cm_weighted', 'iou')
COUNTS = ('supervised_frames', 'predictions', 'active', 'memory_writes', 'cm_valid_frames')


def loss_sum(head, encoded, target, cfg):
    n = len(target)
    predicted = box_cxcywh_to_xyxy(head['pred_boxes'].reshape(-1, 4)).float()
    truth = box_xywh_to_xyxy(target).clamp(0, 1)
    giou, iou = giou_loss(predicted, truth)
    l1 = F.l1_loss(predicted, truth)
    heatmap = generate_heatmap(target[None], cfg.DATA.SEARCH.SIZE, cfg.MODEL.ENCODER.STRIDE)[0][:, None]
    score = head['score_map'].float().clamp(1e-6, 1 - 1e-6)
    positive = heatmap.eq(1)
    negative = heatmap.lt(1)
    focal = (-(score.log() * (1 - score).pow(2) * positive
               + (1 - score).log() * score.pow(2) * (1 - heatmap).pow(4) * negative)
             .flatten(1).sum(1) / positive.flatten(1).sum(1).clamp_min(1)).mean()
    tracking = cfg.TRAIN.GIOU_WEIGHT * giou + cfg.TRAIN.L1_WEIGHT * l1 + focal
    cm, cm_valid = cm_per_example(encoded['tae'][:, -77:], encoded['tae'][:, :-77], target)
    cm_mean = cm.sum() / cm_valid.sum().clamp_min(1)
    weighted = cfg.TRAIN.CM_WEIGHT * cm_mean if cfg.TRAIN.CM_LOSS else cm_mean * 0
    total = tracking + weighted
    metrics = dict(zip(MEANS, (total, tracking, giou, l1, focal, cm_mean, weighted, iou.mean())))
    return total * n, {key: float(value.detach()) for key, value in metrics.items()}, int(cm_valid.sum())


class StreamingTrackerBActor:
    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg
        self.device = next(model.parameters()).device
        self.preprocessor = Preprocessor(self.device)

    def __call__(self, clips, backward=None):
        """backward receives each frame's loss SUM; no retained multi-frame graph."""
        model, cfg = self.model, self.cfg
        totals = {key: 0 for key in COUNTS}
        sums = {key: 0. for key in MEANS}
        states = [TemporalState(cfg) for _ in clips]
        cached = frozen_text_tokens(model.text_encoder, [clip['language'] for clip in clips])
        for t in range(len(clips[0]['images'])):
            learning = t >= cfg.TRAIN.WORLD.WARMUP_FRAMES
            frames = [clip['frame_ids'][t:t + 1].to(self.device) for clip in clips]
            patches, regions = [], []
            with torch.no_grad():
                # Project anew per frame so its graph can be released by backward.
                prior_language = model.text_encoder.text_proj(cached)
                prediction = predict_prior_batch(model, states, prior_language, frames)
                for clip, state in zip(clips, states):
                    patch, region = search_crop(clip['images'][t], cfg.TEST.SEARCH_SIZE, cfg.TEST.SEARCH_FACTOR,
                                               None if state.global_search else state.previous_box)
                    patches.append(patch)
                    regions.append(region[None].to(self.device))
                search = self.preprocessor.process_batch(patches)
                if cfg.DATA.MULTI_MODAL_VISION:
                    search = torch.cat((search, search), 1)
            with torch.set_grad_enabled(backward is not None and learning):
                language = model.text_encoder.text_proj(cached)
                encoded = model.forward_encoder([], [search], language)
                raw = encoded['feat_x']
                fused, extra = model.fuse_features(raw, language, prediction, torch.cat(regions), None, torch.cat(frames))
                head = model.forward_head(fused)
                targets, indices = [], []
                for i, (clip, state, frame) in enumerate(zip(clips, states, frames)):
                    prior = None if prediction is None else {key: value[i:i + 1] for key, value in prediction.items()}
                    result = finish_location(model, state, clip['images'][t], language[i:i + 1], cfg, frame,
                        patches[i], regions[i], raw[i:i + 1], prior,
                        {key: value[i:i + 1] for key, value in head.items()},
                        {key: value[i:i + 1] for key, value in extra.items()})
                    with torch.no_grad():
                        gt = clip['boxes'][t:t + 1].to(self.device) / result['scale']
                        target = center_to_xywh(image_to_crop(xywh_to_center(gt), result['region']))
                        _, roi_valid = extract_roi(raw[i:i + 1], target)
                        center = target[:, :2] + target[:, 2:] / 2
                        valid = (clip['valid'][t:t + 1].to(self.device).bool()
                                 & clip['visible'][t:t + 1].to(self.device).bool() & roi_valid
                                 & ((center > 0) & (center < 1)).all(-1))
                        if learning and bool(valid.all()):
                            indices.append(i)
                            targets.append(target)
                        if prior is not None:
                            totals['predictions'] += 1
                            totals['active'] += int(prior['valid'].sum())
                        totals['memory_writes'] += sum(observe(model, state, result, frame))
                if indices:
                    value, metrics, cm_valid = loss_sum(
                        {key: tensor[indices] for key, tensor in head.items()},
                        {'tae': encoded['tae'][indices]}, torch.cat(targets), cfg)
                    if not bool(torch.isfinite(value)):
                        raise FloatingPointError('Non-finite B frame loss at frame {}'.format(t))
                    for key in MEANS:
                        sums[key] += metrics[key] * len(indices)
                    totals['supervised_frames'] += len(indices)
                    totals['cm_valid_frames'] += cm_valid
                    if backward is not None:
                        backward(value)
            # Explicitly drop the last per-video view and this frame's graph.
            del result, head, encoded, raw, fused, extra, language
        n = totals['supervised_frames']
        return dict(totals, **{key: value / max(1, n) for key, value in sums.items()})


def global_metrics(metrics, device):
    n = metrics['supervised_frames']
    values = [metrics.get(key, 0) for key in COUNTS] + [metrics.get(key, 0.) * n for key in MEANS]
    packed = torch.tensor(values, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(packed)
    values = packed.tolist()
    result = dict(zip(COUNTS, map(int, values[:len(COUNTS)])))
    result.update({key: value / max(1, result['supervised_frames'])
                   for key, value in zip(MEANS, values[len(COUNTS):])})
    return result


def synchronize_gradients(model, supervised_frames, bucket_bytes=32 * 1024 ** 2):
    """SUM local frame gradients, divide by global actual supervised-frame count.

    All ranks, including one with zero supervision, enter identical collectives.
    This is explicit data parallelism, not an additional DDP average.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    used = torch.tensor([p.grad is not None for p in params], dtype=torch.int32, device=params[0].device)
    if dist.is_initialized():
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
    bucket, size = [], 0

    def flush():
        if not bucket:
            return
        flat = torch._utils._flatten_dense_tensors([p.grad for p in bucket])
        if dist.is_initialized():
            dist.all_reduce(flat)
        flat.div_(max(1, supervised_frames))
        for p, reduced in zip(bucket, torch._utils._unflatten_dense_tensors(flat, [p.grad for p in bucket])):
            p.grad.copy_(reduced)

    for p, active in zip(params, used.tolist()):
        if not active:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if size + p.numel() * p.element_size() > bucket_bytes:
            flush()
            bucket, size = [], 0
        bucket.append(p)
        size += p.numel() * p.element_size()
    flush()


def module_gradient_norms(model):
    return {'gradient_norm_' + name.replace('.', '_'): float(torch.stack([
        p.grad.detach().float().norm().square() for p in module.parameters() if p.grad is not None
    ]).sum().sqrt()) if any(p.grad is not None for p in module.parameters()) else 0.
        for name, module in (('encoder', model.encoder), ('text_projection', model.text_encoder.text_proj),
                             ('head', model.head), ('fusion', model.future_fusion))}
