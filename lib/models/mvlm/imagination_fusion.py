"""Spatially aligned prediction residual; raw search features remain untouched."""
import math
import torch
import torch.nn.functional as F
from torch import nn
from .world_state import image_to_crop, center_to_xywh, extract_roi


class ImaginationFusion(nn.Module):
    def __init__(self, channels, hidden=256, heads=4, strength=1., min_quality=.0):
        super().__init__()
        self.query = nn.Linear(channels, hidden)
        self.memory = nn.Linear(channels, hidden)
        self.position = nn.Parameter(torch.randn(4, hidden) * .02)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.gate = nn.Linear(hidden * 2, 1)
        self.output = nn.Linear(hidden, channels)
        self.strength = strength
        self.min_quality = min_quality
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, prediction, regions):
        if prediction is None:
            return features
        boxes = image_to_crop(prediction['boxes'], regions)
        observation, roi_valid = extract_roi(features, center_to_xywh(boxes))
        consistency = F.cosine_similarity(observation.mean(1), prediction['tokens'].mean(1), dim=-1)
        valid = prediction['valid'] & roi_valid & (prediction['quality'] >= self.min_quality)
        if not valid.any():
            return features
        query = self.query(features)
        memory = self.memory(prediction['tokens']) + self.position
        retrieved, _ = self.attention(query, memory, memory, need_weights=False)
        gate = torch.sigmoid(self.gate(torch.cat((query, retrieved), -1)))
        side = math.isqrt(features.shape[1])
        if side * side != features.shape[1]:
            raise ValueError('Expected square search features')
        axis = (torch.arange(side, device=features.device, dtype=features.dtype) + .5) / side
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        grid = torch.stack((xx.flatten(), yy.flatten()), -1)
        distance = (grid[None] - boxes[:, None, :2]) / boxes[:, None, 2:].clamp_min(1 / side)
        spatial = (-.5 * distance.square().sum(-1)).exp()[..., None]
        reliability = (consistency.clamp(0, 1) * prediction['quality'] * valid).detach()
        delta = self.output(gate * retrieved) * spatial * reliability[:, None, None]
        return features + self.strength * delta
