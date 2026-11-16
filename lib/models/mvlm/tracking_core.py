"""One causal frame loop for inference and prediction-driven training.

Only the caller's loss/scorer receives annotations. All learned operations use
model.forward (including a DDP wrapper); discrete state updates are detached.
"""
import copy
import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.test.tracker.utils import Preprocessor, sample_target
from lib.test.utils.hann import hann2d
from lib.utils.box_ops import clip_box
from .memory import extract_target_tokens
from .memory_policy import Observation, WriteDecision
from .reliable_memory import ReliableMemoryBank
from .tracking_ops import pairwise_iou, select_per_box_distractor, batched_slice_mean_from_grid


@dataclass
class CropTransform:
    origin_x: float
    origin_y: float
    width: float
    height: float

    def to_image(self, box):
        scale = box.new_tensor([self.width, self.height, self.width, self.height])
        return box * scale + box.new_tensor([self.origin_x, self.origin_y, 0., 0.])

    def to_crop(self, box):
        scale = box.new_tensor([self.width, self.height, self.width, self.height])
        return (box - box.new_tensor([self.origin_x, self.origin_y, 0., 0.])) / scale


@dataclass
class TrackingState:
    video_id: str
    language: str
    memory: object
    frame_id: int = -1
    box: object = None
    global_search: bool = True
    kmem: object = None


def prepare_crop(image, box, global_search, size, factor):
    h, w = image.shape[:2]
    if global_search or box is None:
        return cv2.resize(image, (size, size)), CropTransform(0., 0., float(w), float(h))
    x, y, bw, bh = box
    side = math.ceil(math.sqrt(bw * bh) * factor)
    if side < 1:
        raise ValueError('Invalid predicted crop')
    origin_x, origin_y = round(x+bw/2-side/2), round(y+bh/2-side/2)
    crop, _ = sample_target(image, box, factor, output_sz=size)
    return crop, CropTransform(origin_x, origin_y, side, side)


@torch.no_grad()
def select_candidates(boxes, scores, similarities, previous_kmem, cfg):
    """Exact legacy MVLM selection on fixed candidates, plus explicit diagnostics."""
    corr_out, cls_out = select_per_box_distractor(boxes, similarities, scores, cfg.PSI_OUT)
    corr_margin, cls_margin = torch.tanh(similarities-corr_out), torch.tanh(scores-cls_out)
    confidence = cfg.ALPHA_CORR * corr_margin + (1-cfg.ALPHA_CORR) * cls_margin
    kmem = confidence.clone() if previous_kmem is None else (1-cfg.LAMBDA)*confidence + cfg.LAMBDA*previous_kmem
    combined = (1-cfg.OMEGA)*confidence + cfg.OMEGA*kmem
    passed = combined >= cfg.TAU
    if bool(passed.any()):
        ids = passed.nonzero(as_tuple=False).squeeze(1)
        index = int(ids[scores[ids].argmax()])
    else:
        index = int(scores.argmax())
    outside = pairwise_iou(boxes)[index] <= cfg.PSI_OUT
    outside[index] = False
    return dict(index=index, passed=passed, next_global=not bool(passed.any()), kmem=kmem,
                cls_margin=float(cls_margin[index]) if bool(outside.any()) else None,
                language_margin=float(corr_margin[index]) if bool(outside.any()) else None)


@torch.no_grad()
def competitor_descriptors(features, boxes, scores, similarities, index, cfg, roi_size):
    if cfg.MAX_COMPETITORS < 1:
        return features.new_empty((0, features.shape[-1]))
    iou = pairwise_iou(boxes)
    allowed = (iou[index] <= cfg.MAX_TARGET_IOU)
    allowed[index] = False
    # Stable CPU ordering also defines deterministic ties on older torch versions.
    by_score = sorted(range(len(boxes)), key=lambda j: (-float(scores[j]), j))
    by_language = sorted(range(len(boxes)), key=lambda j: (-float(similarities[j]), j))
    chosen = []
    for pair in zip(by_score, by_language):
        for j in pair:
            if not bool(allowed[j]) or j in chosen:
                continue
            if any(float(iou[j, k]) > cfg.NMS_IOU for k in chosen):
                continue
            xywh = torch.cat((boxes[j, :2]-boxes[j, 2:]/2, boxes[j, 2:]))[None]
            tokens, valid = extract_target_tokens(features, xywh, roi_size)
            if not bool(valid[0]):
                continue
            chosen.append(j)
            if len(chosen) >= cfg.MAX_COMPETITORS:
                break
        if len(chosen) >= cfg.MAX_COMPETITORS:
            break
    descriptors = []
    for j in chosen:
        xywh = torch.cat((boxes[j, :2]-boxes[j, 2:]/2, boxes[j, 2:]))[None]
        tokens, _ = extract_target_tokens(features, xywh, roi_size)
        descriptors.append(F.normalize(tokens[0].float().mean(0), dim=-1))
    return torch.stack(descriptors) if descriptors else features.new_empty((0, features.shape[-1]))


class TrackingCore:
    def __init__(self, model, cfg, device):
        if not cfg.MODEL.ENCODER.TEMPLATE_FREE:
            raise ValueError('Reliable core supports template-free tracking only')
        if cfg.DATA.SEARCH.SIZE != cfg.TEST.SEARCH_SIZE:
            raise ValueError('Train/test search sizes must match the model feature grid')
        self.model = model
        self.network = model.module if hasattr(model, 'module') else model
        self.cfg, self.device = cfg, torch.device(device)
        self.preprocessor = Preprocessor(self.device)
        self.window = None
        if cfg.TEST.WINDOW:
            grid = cfg.TEST.SEARCH_SIZE // cfg.MODEL.ENCODER.STRIDE
            self.window = hann2d(torch.tensor([grid, grid]), centered=True).to(self.device)
        self.state = None
        self.transform = None

    def reset(self, language, video_id='video'):
        if not isinstance(language, str) or not language.strip():
            raise ValueError('Strict language tracking requires a nonempty query')
        self.state = TrackingState(video_id, language, self.network.new_memory(1))
        with torch.no_grad():
            self.text_emb = self.network.forward_textencoder([language]).detach()

    def state_dict(self):
        return copy.deepcopy(dict(version=1, state=self.state, transform=self.transform,
                                  text_emb=self.text_emb, cfg=self.cfg))

    def load_state_dict(self, saved):
        if saved['version'] != 1 or saved['cfg'] != self.cfg:
            raise ValueError('Incompatible tracking snapshot')
        saved = copy.deepcopy(saved)
        self.state, self.transform, self.text_emb = saved['state'], saved['transform'], saved['text_emb']

    def step(self, image, frame_id=None):
        if self.state is None:
            raise RuntimeError('Call reset(language) before step')
        state = self.state
        frame_id = state.frame_id+1 if frame_id is None else int(frame_id)
        if frame_id <= state.frame_id:
            raise ValueError('Frame IDs must increase')
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected uint8 RGB image')
        crop, transform = prepare_crop(image, state.box, state.global_search,
                                       self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_FACTOR)
        self.transform = transform
        search = self.preprocessor.process(crop)
        if self.cfg.DATA.MULTI_MODAL_VISION:
            search = torch.cat((search, search), dim=1)
        # TF encoder ignores this API placeholder. It contains no GT crop.
        template = search[:, :, :self.cfg.TEST.TEMPLATE_SIZE, :self.cfg.TEST.TEMPLATE_SIZE]
        output = self.model(template_list=[template], search_list=[search], text_emb=self.text_emb,
                            memory_state=state.memory,
                            frame_ids=torch.tensor([frame_id], device=self.device))
        with torch.no_grad():
            response = output['score_map'].detach()
            if self.window is not None:
                response = response * self.window
            boxes, scores = self.network.head.cal_bboxv2(response, output['size_map'].detach(),
                                                        output['offset_map'].detach(), return_score=True)
            boxes, scores = boxes[0], scores[0]
            if not all(bool(torch.isfinite(x).all()) for x in (boxes, scores, output['feat_x'])):
                raise ValueError('Nonfinite tracking output')
            features = batched_slice_mean_from_grid(output['feat_x'].detach(), boxes)
            similarities = F.normalize(features, dim=-1) @ F.normalize(self.text_emb.mean(1)[0], dim=-1)
            selected = select_candidates(boxes, scores, similarities, state.kmem, self.cfg.TEST.MVLM)
            index = selected['index']
            box_crop = torch.cat((boxes[index, :2]-boxes[index, 2:]/2, boxes[index, 2:]))
            mapped = transform.to_image(box_crop)
            h, w = image.shape[:2]
            valid_box = bool((mapped[2:] > 0).all())
            result_box = clip_box(mapped.tolist(), h, w, margin=min(10, h, w)) if valid_box else [0., 0., 0., 0.]
            roi_tokens, roi_valid = extract_target_tokens(output['feat_x'], box_crop[None], self.cfg.MODEL.MEMORY.ROI_SIZE)
            normalized = mapped.new_tensor(result_box) / mapped.new_tensor([w, h, w, h])
            obs = Observation(frame_id, mapped.new_tensor(result_box), normalized, roi_tokens[0],
                              F.normalize(roi_tokens[0].float().mean(0), dim=-1), float(scores[index]),
                              selected['cls_margin'], selected['language_margin'],
                              bool(selected['passed'][index]), bool(roi_valid[0]) and valid_box, state.global_search)
            decision = WriteDecision('disabled', (), 0, 'disabled')
            if isinstance(state.memory, ReliableMemoryBank):
                competitors = None
                if self.cfg.MODEL.MEMORY.DISTRACTOR_RETENTION:
                    competitors = competitor_descriptors(output['feat_x'], boxes, scores, similarities,
                                                        index, self.cfg.MODEL.MEMORY.RETENTION,
                                                        self.cfg.MODEL.MEMORY.ROI_SIZE)
                decision = state.memory.update(obs, competitors)
            elif state.memory is not None:
                accepted = state.memory.write(roi_tokens, normalized[None], [frame_id], scores[index:index+1],
                                              torch.tensor([obs.roi_valid and obs.vlm_pass], device=self.device))[0]
                decision = WriteDecision('commit' if accepted else 'reject', (), int(accepted), 'legacy')
            state.box = result_box if valid_box else None
            state.global_search = selected['next_global'] or not valid_box
            state.kmem, state.frame_id = selected['kmem'], frame_id
            diagnostics = dict(frame_id=frame_id, candidate_index=index, used_global_search=obs.used_global_search,
                               next_global=state.global_search, score=obs.score, cls_margin=obs.cls_margin,
                               language_margin=obs.language_margin, write=vars(decision),
                               short_size=len(state.memory.short[0]) if state.memory else 0,
                               long_size=len(state.memory.long[0]) if state.memory else 0)
            if isinstance(state.memory, ReliableMemoryBank):
                diagnostics['memory_event'] = state.memory.events[0]
            if self.network.memory_enabled:
                diagnostics['fusion'] = dict(self.network.memory_fusion.last_diagnostics)
        return output, dict(target_bbox=result_box, failed=not valid_box, diagnostics=diagnostics)
