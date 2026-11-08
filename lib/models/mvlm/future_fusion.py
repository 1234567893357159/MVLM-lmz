"""Observation-preserving future injection with explicit geometric alignment."""
import math
import torch
import torch.nn.functional as F
from torch import nn
from .world_state import image_to_crop, center_to_xywh, extract_roi


class FutureFusion(nn.Module):
    def __init__(self, channels, hidden, heads, mode='cross_attention', strength=1., min_quality=0.):
        super().__init__()
        if mode not in ('cross_attention', 'concat'):
            raise ValueError('Unknown future fusion mode: ' + str(mode))
        self.mode, self.hidden = mode, hidden
        self.strength, self.min_quality = strength, min_quality
        self.query = nn.Linear(channels, hidden)
        self.value = nn.Linear(channels, hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.gate = nn.Linear(hidden * 2, 1)
        if mode == 'cross_attention':
            self.output = nn.Linear(hidden, channels)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def absent(self, features):
        if self.mode == 'concat':
            return torch.cat((features, features.new_zeros(*features.shape[:2], self.hidden)), -1)
        return features

    def from_memory(self, features, context):
        if context is None:
            return self.absent(features)
        readout = context['tokens'] * context['valid'][:, None, None].to(features)
        if self.mode == 'concat':
            return torch.cat((features, readout), -1)
        return features + self.strength * self.output(readout) * context['valid'][:, None, None].to(features)

    def forward(self, features, prediction, regions):
        batch, count, _ = features.shape
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError('Expected square search features')
        if prediction is None:
            return self.absent(features), features.new_zeros(batch)
        if regions is None:
            raise ValueError('Future fusion requires exact crop geometry')
        boxes = image_to_crop(prediction['boxes'], regions)
        observation, roi_valid = extract_roi(features, center_to_xywh(boxes))
        agreement = F.cosine_similarity(observation.mean(1), prediction['tokens'].mean(1), dim=-1).clamp(0, 1)
        reliable = prediction['valid'] & roi_valid & (prediction['quality'] >= self.min_quality)
        reliability = (agreement * prediction['quality'] * reliable).detach()
        axis = (torch.arange(side, device=features.device, dtype=features.dtype) + .5) / side
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        grid = torch.stack((xx.flatten(), yy.flatten()), -1)
        distance = (grid[None] - boxes[:, None, :2]) / boxes[:, None, 2:].clamp_min(1 / side)
        spatial = (-.5 * distance.square().sum(-1)).exp()[..., None]
        values = self.value(prediction['tokens'])
        query = self.query(features)
        if self.mode == 'concat':
            # Geometric ROI warp: no attention in the concatenation ablation.
            sampling_grid = 2 * distance.reshape(batch, side, side, 2)
            patch = values.transpose(1, 2).reshape(batch, self.hidden, 2, 2)
            retrieved = F.grid_sample(patch, sampling_grid, align_corners=False).flatten(2).transpose(1, 2)
        else:
            retrieved, _ = self.attention(query, values, values, need_weights=False)
        gate = torch.sigmoid(self.gate(torch.cat((query, retrieved), -1)))
        readout = gate * retrieved
        if self.mode == 'concat':
            return torch.cat((features, self.strength * readout * spatial * reliability[:, None, None]), -1), reliability
        return features + self.strength * self.output(readout) * spatial * reliability[:, None, None], reliability
