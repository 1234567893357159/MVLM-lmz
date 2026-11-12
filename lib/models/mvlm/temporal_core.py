"""Shared causal localization/observation updates for training and online tracking.

No labels enter locate(). Observations are committed only after localization;
training may explicitly teacher-force a past observation in stage A.
"""
import math
import torch
import torch.nn.functional as F
from .memory import TargetMemoryBank
from .world_state import (WorldHistory, search_crop, extract_roi, image_to_crop,
                          crop_to_image, xywh_to_center, center_to_xywh)
from lib.utils.box_ops import clip_box


class TemporalState:
    def __init__(self, cfg, batch_size=1):
        self.cfg = cfg
        self.history = WorldHistory(cfg.MODEL.WORLD.HISTORY)
        m = cfg.MODEL.MEMORY
        self.bank = TargetMemoryBank(batch_size, m.SHORT_CAPACITY, m.LONG_CAPACITY,
                                     m.MIN_QUALITY, m.NOVELTY_THRESHOLD) if m.ENABLED else None
        self.reset()

    def reset(self):
        self.history.clear()
        if self.bank is not None:
            self.bank.reset()
        self.last_visual = None
        self.kmem = None
        self.previous_box = None
        self.global_search = True


def predict_prior(model, state, language, frame_ids):
    history = state.history.pack()
    if not model.world_enabled or history is None or model.cfg.TRAIN.WORLD.HEAD_ONLY:
        return None
    context = None
    if model.memory_enabled:
        context = model.memory_reader(history['tokens'][:, -1], language, state.bank, frame_ids)
    return model.world_model(history, language, frame_ids, context, state.last_visual)


class IndependentMemoryView:
    """Read-only padded batch of independent banks; no cross-video entries."""
    def __init__(self, states):
        self.states = states

    def pack(self, features, frames, tokens_per_entry, include_metadata=False):
        rows = [state.bank.pack(features[i:i + 1], frames[i:i + 1], tokens_per_entry,
                                include_metadata=include_metadata)
                for i, state in enumerate(self.states)]
        count = max(row[0].shape[1] for row in rows)
        return tuple(torch.cat([F.pad(row[column], (0, 0, 0, count - row[column].shape[1]))
                                if row[column].ndim == 3 else
                                F.pad(row[column], (0, count - row[column].shape[1]))
                                for row in rows], dim=0)
                     for column in range(len(rows[0])))


def predict_prior_batch(model, states, language, frame_ids):
    histories = [state.history.pack() for state in states]
    if not model.world_enabled or all(history is None for history in histories):
        return None
    if any(history is None for history in histories):
        raise ValueError('Batched histories must advance at the same temporal index')
    history = {key: torch.cat([row[key] for row in histories]) for key in histories[0]}
    frames = torch.cat(frame_ids)
    context = model.memory_reader(history['tokens'][:, -1], language,
                                  IndependentMemoryView(states), frames) if model.memory_enabled else None
    visual = torch.cat([state.last_visual for state in states])
    return model.world_model(history, language, frames, context, visual)


@torch.no_grad()
def select_candidates(model, output, language, kmem, cfg):
    # Reuse the exact baseline margin helpers, including historical ROI pooling.
    from lib.test.tracker.mvlm import batched_slice_mean_from_grid, select_per_box_distractor
    from lib.test.utils.hann import hann2d
    response = output['score_map']
    if cfg.TEST.WINDOW:
        window = hann2d(torch.tensor([response.shape[-1]] * 2), centered=True).to(response)
        response = response * window
    boxes, scores = model.head.cal_bboxv2(response, output['size_map'], output['offset_map'], return_score=True)
    boxes, scores = boxes[0], scores[0]
    roi = F.normalize(batched_slice_mean_from_grid(output['feat_x'], boxes), dim=-1)
    text = F.normalize(language.mean(1)[0], dim=-1)
    correlations = (roi * text).sum(-1)
    p = cfg.TEST.MVLM
    dcorr, dcls = select_per_box_distractor(boxes, correlations, scores, p.PSI_OUT)
    confidence = p.ALPHA_CORR * (correlations - dcorr).tanh() + (1 - p.ALPHA_CORR) * (scores - dcls).tanh()
    kmem = confidence if kmem is None else (1 - p.LAMBDA) * confidence + p.LAMBDA * kmem
    reliable = ((1 - p.OMEGA) * confidence + p.OMEGA * kmem) >= p.TAU
    ranking = scores
    if 'quality_logits' in output and model.world_stage != 'A' and not cfg.TRAIN.WORLD.HEAD_ONLY:
        strength = cfg.MODEL.WORLD.QUALITY_WEIGHT * output['prediction_reliability'][0]
        ranking = scores * ((1 - strength) + strength * output['quality_logits'][0].sigmoid())
    allowed = reliable.nonzero().flatten()
    # Keep the baseline global-search fallback when no candidate is reliable.
    index = allowed[ranking[allowed].argmax()] if allowed.numel() else scores.argmax()
    return dict(boxes=boxes, scores=scores, correlations=correlations, ranking=ranking,
                index=index, reliable=reliable, kmem=kmem)


@torch.no_grad()
def select_stage_a_batch(model, output, language, states, cfg):
    """Vectorized baseline margin rule with the original grid pooling convention."""
    from lib.test.utils.hann import hann2d
    response = output['score_map']
    if cfg.TEST.WINDOW:
        response = response * hann2d(torch.tensor([response.shape[-1]] * 2), centered=True).to(response)
    boxes, scores = model.head.cal_bboxv2(response, output['size_map'], output['offset_map'], return_score=True)
    boxes, scores = torch.stack(boxes), torch.stack(scores)
    batch, count = scores.shape
    raw = output['feat_x']
    side = math.isqrt(raw.shape[1])
    cx, cy, width, height = boxes.unbind(-1)
    x1 = ((cx - width / 2) * (side - 1)).round().long().clamp_min(0)
    y1 = ((cy - height / 2) * (side - 1)).round().long().clamp_min(0)
    x2 = torch.maximum(((cx + width / 2) * (side - 1)).round().long(), x1 + 1).clamp_max(side)
    y2 = torch.maximum(((cy + height / 2) * (side - 1)).round().long(), y1 + 1).clamp_max(side)
    x1, y1 = x1.clamp_max(side - 1), y1.clamp_max(side - 1)
    integral = raw.reshape(batch, side, side, -1).cumsum(1).cumsum(2)
    integral = F.pad(integral, (0, 0, 1, 0, 1, 0))
    sample = torch.arange(batch, device=raw.device)[:, None]
    pooled = (integral[sample, y2, x2] - integral[sample, y1 + 1, x2]
              - integral[sample, y2, x1 + 1] + integral[sample, y1 + 1, x1 + 1])
    pooled = pooled / ((y2 - y1) * (x2 - x1)).clamp_min(1)[..., None]
    correlations = (F.normalize(pooled, dim=-1) * F.normalize(language.mean(1), dim=-1)[:, None]).sum(-1)
    corners = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                         boxes[..., :2] + boxes[..., 2:] / 2), -1)
    intersection = (torch.minimum(corners[:, :, None, 2:], corners[:, None, :, 2:])
                    - torch.maximum(corners[:, :, None, :2], corners[:, None, :, :2])).clamp_min(0).prod(-1)
    area = (corners[..., 2:] - corners[..., :2]).clamp_min(0).prod(-1)
    iou = intersection / (area[:, :, None] + area[:, None, :] - intersection + 1e-12)
    p = cfg.TEST.MVLM
    outside = (iou <= p.PSI_OUT) & ~torch.eye(count, dtype=torch.bool, device=raw.device)[None]
    dcorr = torch.where(outside.any(-1), correlations[:, None].masked_fill(~outside, -float('inf')).max(-1).values,
                        correlations.min(-1).values[:, None])
    dcls = torch.where(outside.any(-1), scores[:, None].masked_fill(~outside, -float('inf')).max(-1).values,
                       scores.min(-1).values[:, None])
    confidence = p.ALPHA_CORR * (correlations - dcorr).tanh() + (1 - p.ALPHA_CORR) * (scores - dcls).tanh()
    kmem = torch.stack([confidence[i] if state.kmem is None else
                        (1 - p.LAMBDA) * confidence[i] + p.LAMBDA * state.kmem
                        for i, state in enumerate(states)])
    reliable = ((1 - p.OMEGA) * confidence + p.OMEGA * kmem) >= p.TAU
    indices = torch.where(reliable.any(-1), scores.masked_fill(~reliable, -float('inf')).argmax(-1), scores.argmax(-1))
    values = dict(boxes=boxes, scores=scores, correlations=correlations, ranking=scores,
                  index=indices, reliable=reliable, kmem=kmem)
    return [{key: value[i] for key, value in values.items()} for i in range(batch)]


def locate(model, state, image, language, cfg, frame_ids, preprocessor):
    """One frame, before any memory/history write. Shared by Actor and tracker."""
    prior = predict_prior(model, state, language, frame_ids)
    patch, region = search_crop(image, cfg.TEST.SEARCH_SIZE, cfg.TEST.SEARCH_FACTOR,
                               None if state.global_search else state.previous_box)
    region = region[None].to(next(model.parameters()).device)
    search = preprocessor.process(patch)
    if cfg.DATA.MULTI_MODAL_VISION:
        search = torch.cat((search, search), 1)
    with torch.no_grad():
        encoded = model.forward_encoder([], [search], language)
    raw = encoded['feat_x']
    return finish_location(model, state, image, language, cfg, frame_ids,
                           patch, region, raw, prior)


def finish_location(model, state, image, language, cfg, frame_ids,
                    patch, region, raw, prior, head_output=None, temporal_output=None, selection_output=None):
    """Shared per-video selection and geometry after an individual or batched encoding."""
    if temporal_output is None:
        features, extra = model.fuse_features(raw, language, prior, region, state.bank, frame_ids)
    else:
        features, extra = None, dict(temporal_output)
    output = model.forward_head(features) if head_output is None else dict(head_output)
    output.update(extra, feat_x=raw)
    selection = select_candidates(model, output, language, state.kmem, cfg) if selection_output is None else selection_output
    index = selection['index']
    full = crop_to_image(selection['boxes'][index:index + 1], region)
    height, width = image.shape[:2]
    scale = full.new_tensor([width, height, width, height])
    pixels = clip_box((center_to_xywh(full)[0] * scale).tolist(), height, width, margin=10)
    stored = xywh_to_center(full.new_tensor(pixels)[None] / scale)
    return dict(output=output, prior=prior, region=region, selection=selection,
                box=stored, pixels=pixels, scale=scale, patch=patch)


def locate_batch_stage_a(model, states, images, language, cfg, frame_ids, preprocessor):
    """Batch frozen encoding/CENTER at one time step; each video keeps its own history.

    Predictions read only prior frames. No GT is accepted here. Stage-A bypasses
    future injection, so frozen head computation can also be shared safely.
    """
    if model.world_stage != 'A':
        raise ValueError('Batched frozen-head localization is specific to stage A')
    if not len(states) == len(images) == language.shape[0] == len(frame_ids):
        raise ValueError('Batched localization requires one state per video')
    prediction = predict_prior_batch(model, states, language, frame_ids)
    priors = [None if prediction is None else {key: value[i:i + 1] for key, value in prediction.items()}
              for i in range(len(states))]
    patches, regions = [], []
    for index, (state, image) in enumerate(zip(states, images)):
        patch, region = search_crop(image, cfg.TEST.SEARCH_SIZE, cfg.TEST.SEARCH_FACTOR,
                                   None if state.global_search else state.previous_box)
        patches.append(patch)
        regions.append(region[None].to(language.device))
    with torch.no_grad():
        search = preprocessor.process_batch(patches)
        if cfg.DATA.MULTI_MODAL_VISION:
            search = torch.cat((search, search), 1)
        raw = model.forward_encoder([], [search], language)['feat_x']
        features = model.future_fusion.absent(raw)
        head = model.forward_head(features)
        selections = select_stage_a_batch(model, dict(head, feat_x=raw), language, states, cfg)
    extra = dict(feat_x_temporal=features, prediction_reliability=raw.new_zeros(len(states)))
    if prediction is not None and model.quality_head is not None:
        extra['quality_logits'] = model.quality_head(prediction, torch.cat(regions), model.fx_sz)
    return [finish_location(model, state, image, language[index:index + 1], cfg, frame_ids[index],
                            patches[index], regions[index], raw[index:index + 1], priors[index],
                            {key: value[index:index + 1] for key, value in head.items()},
                            {key: value[index:index + 1] for key, value in extra.items()}, selections[index])
            for index, (state, image) in enumerate(zip(states, images))]


@torch.no_grad()
def observe(model, state, result, frame_ids, box=None, quality=None, valid=None):
    """Commit detached ORIGINAL observations. Never write generated features."""
    raw, prior = result['output']['feat_x'], result['prior']
    selected, index = result['selection'], result['selection']['index']
    teacher = box is not None
    box = result['box'] if box is None else box
    quality = selected['scores'][index:index + 1].clamp(0, 1) if quality is None else quality
    roi, roi_valid = extract_roi(raw, center_to_xywh(image_to_crop(box, result['region'])))
    valid = selected['reliable'][index:index + 1] if valid is None else valid
    valid = valid & roi_valid
    state.history.append(roi, box, quality, valid, frame_ids)
    side = math.isqrt(raw.shape[1])
    state.last_visual = F.adaptive_avg_pool2d(raw.detach().transpose(1, 2).reshape(
        raw.shape[0], raw.shape[-1], side, side), (4, 4)).flatten(2).transpose(1, 2)
    written = []
    if state.bank is not None:
        write_valid = valid.clone()
        if prior is not None and not teacher:
            agreement = F.cosine_similarity(roi.mean(1), prior['tokens'].mean(1), dim=-1)
            write_valid &= ~prior['valid'] | (agreement >= state.cfg.MODEL.MEMORY.WRITE_CONSISTENCY)
        written = state.bank.write(roi, center_to_xywh(box), frame_ids, quality, write_valid)
    state.kmem = selected['kmem']
    state.global_search = not bool(valid.all()) if teacher else not bool(selected['reliable'][index])
    state.previous_box = (center_to_xywh(box)[0] * result['scale']).tolist() if teacher else result['pixels']
    return written
