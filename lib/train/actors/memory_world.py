"""Stage A/B/C learning over the exact online temporal core."""
import torch
import torch.nn.functional as F
from lib.models.mvlm.temporal_core import TemporalState, locate, locate_batch_stage_a, observe
from lib.models.mvlm.world_state import (center_to_xywh, xywh_to_center, image_to_crop,
                                         extract_roi, motion_target)
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, giou_loss
from lib.utils.heapmap_utils import generate_heatmap
from lib.utils.focal_loss import FocalLoss
from lib.test.tracker.utils import Preprocessor


def tracking_loss(output, target, cfg):
    predicted = box_cxcywh_to_xyxy(output['pred_boxes'].reshape(-1, 4))
    truth = box_xywh_to_xyxy(target).clamp(0, 1)
    giou, _ = giou_loss(predicted, truth)
    heatmap = generate_heatmap(target[None], cfg.DATA.SEARCH.SIZE, cfg.MODEL.ENCODER.STRIDE)[0][:, None]
    return (cfg.TRAIN.GIOU_WEIGHT * giou + cfg.TRAIN.L1_WEIGHT * F.l1_loss(predicted, truth)
            + FocalLoss()(output['score_map'].clamp(1e-6, 1 - 1e-6), heatmap))


@torch.no_grad()
def candidate_iou_targets(boxes, truth_xywh):
    boxes = box_cxcywh_to_xyxy(boxes.detach())
    truth = box_xywh_to_xyxy(truth_xywh.detach())
    intersection = (torch.minimum(boxes[:, 2:], truth[:, 2:]) - torch.maximum(boxes[:, :2], truth[:, :2])).clamp_min(0).prod(-1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1)
    target_area = (truth[:, 2:] - truth[:, :2]).clamp_min(0).prod(-1)
    return (intersection / (area + target_area - intersection).clamp_min(1e-8)).clamp(0, 1)


class MemoryWorldActor:
    def __init__(self, model, cfg, stage):
        self.model, self.cfg, self.stage = model, cfg, stage
        self.device = next(model.parameters()).device
        self.preprocessor = Preprocessor(self.device)
        self.step = 0

    def frames(self, clips):
        """Yield causal frame results; the caller commits observations before advancing."""
        cfg, model = self.cfg, self.model
        if cfg.TRAIN.WORLD.BATCHED_ENCODER:
            if self.stage != 'A':
                raise ValueError('BATCHED_ENCODER currently supports stage A only')
            if len({len(clip['images']) for clip in clips}) != 1:
                raise ValueError('Batched clips must have the same temporal length')
            states = [TemporalState(cfg) for _ in clips]
            with torch.no_grad():
                language = model.forward_textencoder([clip['language'] for clip in clips])
            for t in range(len(clips[0]['images'])):
                frames = [clip['frame_ids'][t:t + 1].to(self.device) for clip in clips]
                learning = t >= cfg.TRAIN.WORLD.WARMUP_FRAMES
                with torch.set_grad_enabled(torch.is_grad_enabled() and learning):
                    results = locate_batch_stage_a(model, states, [clip['images'][t] for clip in clips],
                                                   language, cfg, frames, self.preprocessor)
                for index, (clip, state, frame, result) in enumerate(zip(clips, states, frames, results)):
                    yield clip, t, frame, state, result
        else:
            for clip in clips:
                state = TemporalState(cfg)
                with torch.no_grad():
                    language = model.forward_textencoder([clip['language']])
                for t, image in enumerate(clip['images']):
                    frame = clip['frame_ids'][t:t + 1].to(self.device)
                    learning = t >= cfg.TRAIN.WORLD.WARMUP_FRAMES
                    with torch.set_grad_enabled(torch.is_grad_enabled() and learning):
                        result = locate(model, state, image, language, cfg, frame, self.preprocessor)
                    yield clip, t, frame, state, result

    def __call__(self, clips, validation=False):
        cfg, model = self.cfg, self.model
        losses, appearance_errors, motion_errors, quality_errors = [], [], [], []
        predictions, active, writes = 0, 0, 0
        world = cfg.TRAIN.WORLD
        ramp = min(1., self.step / world.HISTORY_RAMP_STEPS) if world.HISTORY_RAMP_STEPS else 1.
        probability = world.PREDICTED_HISTORY_PROB * ramp
        for clip, t, frame, state, result in self.frames(clips):
            history = state.history.pack()
            learning = t >= world.WARMUP_FRAMES
            output, prior = result['output'], result['prior']
            gt = clip['boxes'][t:t + 1].to(self.device) / result['scale']
            gt_center = xywh_to_center(gt)
            crop_gt = center_to_xywh(image_to_crop(gt_center, result['region']))
            target_tokens, roi_valid = extract_roi(output['feat_x'], crop_gt)
            gt_valid = clip['valid'][t:t + 1].to(self.device).bool()
            visible = clip['visible'][t:t + 1].to(self.device).bool()
            center = crop_gt[:, :2] + crop_gt[:, 2:] / 2
            supervised = gt_valid & visible & roi_valid & ((center > 0) & (center < 1)).all(-1)
            terms = []
            if self.stage in ('B', 'C') and supervised.any() and learning:
                terms.append(tracking_loss(output, crop_gt, cfg))
            if prior is not None:
                predictions += 1
                active += int(prior['valid'].sum())
                if self.stage in ('A', 'C') and learning:
                    geometry_valid = gt_valid & history['valid'][:, -1]
                    if geometry_valid.any():
                        target_delta = motion_target(history['boxes'][:, -1][geometry_valid], gt_center[geometry_valid])
                        motion = F.smooth_l1_loss(prior['delta'][geometry_valid], target_delta)
                        terms.append(world.MOTION_WEIGHT * motion)
                        motion_errors.append(float(motion.detach()))
                        valid = geometry_valid & visible & roi_valid
                        if valid.any():
                            predicted, target = prior['tokens'][valid], target_tokens[valid]
                            cosine = (1 - F.cosine_similarity(predicted, target, dim=-1)).mean()
                            regression = F.smooth_l1_loss(F.layer_norm(predicted, (predicted.shape[-1],)),
                                                          F.layer_norm(target, (target.shape[-1],)))
                            terms.append(world.APPEARANCE_WEIGHT * (cosine + world.FEATURE_WEIGHT * regression))
                            appearance_errors.append(float(cosine.detach()))
                    if supervised.any() and 'quality_logits' in output:
                        targets = candidate_iou_targets(result['selection']['boxes'], crop_gt)
                        quality_loss = F.binary_cross_entropy_with_logits(output['quality_logits'][0], targets)
                        terms.append(world.QUALITY_LOSS_WEIGHT * quality_loss)
                        quality_errors.append(float(quality_loss.detach()))
            if terms:
                value = torch.stack(terms).sum()
                if validation or value.requires_grad:
                    losses.append(value)
            teacher = (self.stage == 'A' and not validation and bool((gt_valid & visible & roi_valid).all())
                       and torch.rand((), device=self.device).item() >= probability)
            if teacher:
                written = observe(model, state, result, frame, gt_center,
                                  torch.ones(1, device=self.device), gt_valid & visible & roi_valid)
            else:
                written = observe(model, state, result, frame)
            writes += sum(written)
        metrics = dict(supervised_frames=len(losses), predictions=predictions, active=active,
                       memory_writes=writes, predicted_history_probability=probability)
        for name, values in [('appearance_cosine_error', appearance_errors), ('motion_error', motion_errors),
                             ('quality_loss', quality_errors)]:
            if values:
                metrics[name] = sum(values) / len(values)
        if not losses:
            return None, metrics
        loss = torch.stack(losses).mean()
        metrics['loss'] = float(loss.detach())
        return loss, metrics
