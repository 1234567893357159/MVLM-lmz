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
    features, extra = model.fuse_features(raw, language, prior, region, state.bank, frame_ids)
    output = model.forward_head(features)
    output.update(extra, feat_x=raw)
    selection = select_candidates(model, output, language, state.kmem, cfg)
    index = selection['index']
    full = crop_to_image(selection['boxes'][index:index + 1], region)
    height, width = image.shape[:2]
    scale = full.new_tensor([width, height, width, height])
    pixels = clip_box((center_to_xywh(full)[0] * scale).tolist(), height, width, margin=10)
    stored = xywh_to_center(full.new_tensor(pixels)[None] / scale)
    return dict(output=output, prior=prior, region=region, selection=selection,
                box=stored, pixels=pixels, scale=scale, patch=patch)


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
