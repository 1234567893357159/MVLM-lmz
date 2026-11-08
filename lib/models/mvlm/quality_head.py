"""Causal grid-quality prior: uses past latent state and crop geometry only."""
import torch
from torch import nn


class CandidateQualityHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.geometry = nn.Linear(6, hidden)
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, prediction, regions, side):
        axis = (torch.arange(side, device=regions.device, dtype=regions.dtype) + .5) / side
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        grid = torch.stack((xx.flatten(), yy.flatten()), -1)[None].expand(regions.shape[0], -1, -1)
        full = regions[:, None, :2] + grid * regions[:, None, 2:]
        boxes = prediction['boxes']
        relative = (full - boxes[:, None, :2]) / boxes[:, None, 2:].clamp_min(1e-5)
        geometry = torch.cat((grid, relative, boxes[:, None, 2:].expand(-1, side * side, -1)), -1)
        latent = prediction['latent'][:, None] + self.geometry(geometry)
        return self.output(latent).squeeze(-1)
