"""Shared geometry and detached, sequence-local history for world prediction.

State boxes: full-image normalized cxcywh. Crop regions: full-image normalized
xywh. ROI boxes: crop-normalized xywh. Pixel crops use the actual rounded origin.
"""
import math
from collections import deque

import torch
from torchvision.ops import roi_align


def xywh_to_center(boxes):
    return torch.cat((boxes[..., :2] + boxes[..., 2:] / 2, boxes[..., 2:]), -1)


def center_to_xywh(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., 2:]), -1)


def image_to_crop(boxes, regions):
    """Normalized full-image cxcywh -> crop-normalized cxcywh."""
    return torch.cat(((boxes[..., :2] - regions[..., :2]) / regions[..., 2:],
                      boxes[..., 2:] / regions[..., 2:]), -1)


def crop_to_image(boxes, regions):
    return torch.cat((boxes[..., :2] * regions[..., 2:] + regions[..., :2],
                      boxes[..., 2:] * regions[..., 2:]), -1)


def motion_target(previous, future):
    size = previous[..., 2:].clamp_min(1e-5)
    return torch.cat(((future[..., :2] - previous[..., :2]) / size,
                      (future[..., 2:].clamp_min(1e-5) / size).log()), -1)


def apply_motion(previous, delta):
    return torch.cat((previous[..., :2] + delta[..., :2] * previous[..., 2:],
                      previous[..., 2:] * delta[..., 2:].clamp(-4, 4).exp()), -1)


def extract_roi(features, boxes):
    """Detached 2x2 observations; return validity separately, never learn from NaN."""
    batch, count, channels = features.shape
    side = math.isqrt(count)
    if side * side != count or boxes.shape != (batch, 4):
        raise ValueError('Expected a square feature grid and [B,4] ROI boxes')
    corners = torch.cat((boxes[:, :2], boxes[:, :2] + boxes[:, 2:]), -1)
    valid = torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > 0).all(-1)
    corners = torch.nan_to_num(corners).clamp(0, 1)
    valid &= (corners[:, 2:] > corners[:, :2]).all(-1)
    indices = torch.arange(batch, device=features.device).float()[:, None]
    regions = torch.cat((indices, corners * side), -1)
    grid = features.detach().transpose(1, 2).reshape(batch, channels, side, side)
    tokens = roi_align(grid.float(), regions.float(), (2, 2), aligned=True,
                       sampling_ratio=2).flatten(2).transpose(1, 2).to(features)
    return tokens.masked_fill(~valid[:, None, None], 0), valid


def search_crop(image, size, factor, state=None):
    """Reuse baseline pixel crop; also return its exact normalized affine region."""
    from lib.test.tracker.utils import sample_target, resize_sample_target
    height, width = image.shape[:2]
    if state is None:
        patch, _ = resize_sample_target(image, output_sz=size)
        region = [0., 0., 1., 1.]
    else:
        x, y, w, h = map(float, state)
        side = math.ceil(math.sqrt(w * h) * factor)
        if side < 1:
            raise ValueError('Invalid search box')
        x0, y0 = round(x + w / 2 - side / 2), round(y + h / 2 - side / 2)
        patch, _ = sample_target(image, list(state), factor, output_sz=size)
        region = [x0 / width, y0 / height, side / width, side / height]
    return patch, torch.tensor(region, dtype=torch.float32)


class WorldHistory:
    def __init__(self, length):
        self.length = length
        self.items = deque(maxlen=length)

    def clear(self):
        self.items.clear()

    def append(self, tokens, boxes, quality, valid, frame_ids):
        if self.items and not (frame_ids > self.items[-1]['frame_ids']).all():
            raise ValueError('History frame IDs must increase strictly')
        finite = torch.isfinite(tokens).all(-1).all(-1) & torch.isfinite(boxes).all(-1)
        valid = valid.bool() & finite & (boxes[:, 2:] > 0).all(-1)
        self.items.append(dict(tokens=torch.nan_to_num(tokens.detach()),
                               boxes=torch.nan_to_num(boxes.detach()),
                               quality=torch.nan_to_num(quality.detach()).clamp(0, 1),
                               valid=valid.detach(), frame_ids=frame_ids.detach().clone()))

    def pack(self):
        if len(self.items) < self.length:
            return None
        return {key: torch.stack([item[key] for item in self.items], 1)
                for key in self.items[0]}
