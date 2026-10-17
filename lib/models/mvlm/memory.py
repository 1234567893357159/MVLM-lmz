import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align


@dataclass
class MemoryEntry:
    tokens: torch.Tensor
    frame_id: int
    quality: float
    box: torch.Tensor


def extract_target_tokens(features, boxes, roi_size=2):
    batch_size, num_tokens, channels = features.shape
    grid_size = math.isqrt(num_tokens)
    if grid_size * grid_size != num_tokens:
        raise ValueError('Memory requires a square search feature grid')
    if boxes.shape != (batch_size, 4):
        raise ValueError('Memory boxes must have shape [batch, 4] in normalized xywh')
    boxes = boxes.detach().to(device=features.device, dtype=torch.float32)
    corners = torch.cat((boxes[:, :2], boxes[:, :2] + boxes[:, 2:]), dim=-1)
    valid = torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > 0).all(-1)
    corners = corners.clamp(0, 1)
    valid &= (corners[:, 2:] > corners[:, :2]).all(-1)
    corners = torch.nan_to_num(corners) * grid_size
    batch_indices = torch.arange(batch_size, device=features.device).float().unsqueeze(1)
    regions = torch.cat((batch_indices, corners), dim=-1)
    feature_map = features.detach().transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)
    pooled = roi_align(feature_map.float(), regions, output_size=roi_size,
                       spatial_scale=1.0, sampling_ratio=2, aligned=True)
    return pooled.flatten(2).transpose(1, 2).to(features.dtype), valid


def crop_boxes_to_image(boxes, transforms, image_sizes):
    transforms = transforms.to(boxes)
    image_sizes = image_sizes.to(boxes)
    full_boxes = torch.cat((boxes[:, :2] * transforms[:, 2:] + transforms[:, :2],
                            boxes[:, 2:] * transforms[:, 2:]), -1)
    scales = image_sizes[:, [1, 0, 1, 0]]
    return full_boxes / scales


class TargetMemoryBank:
    def __init__(self, batch_size, short_capacity=8, long_capacity=16,
                 min_quality=0.5, novelty_threshold=0.1):
        if batch_size < 1 or short_capacity < 0 or long_capacity < 0:
            raise ValueError('Invalid memory batch size or capacity')
        if short_capacity + long_capacity == 0:
            raise ValueError('At least one memory store must be enabled')
        self.batch_size = batch_size
        self.short_capacity = short_capacity
        self.long_capacity = long_capacity
        self.min_quality = min_quality
        self.novelty_threshold = novelty_threshold
        self.reset()

    def reset(self):
        self.short = [[] for _ in range(self.batch_size)]
        self.long = [[] for _ in range(self.batch_size)]

    def entries(self, batch_index):
        combined = {entry.frame_id: entry for entry in self.long[batch_index]}
        combined.update({entry.frame_id: entry for entry in self.short[batch_index]})
        return sorted(combined.values(), key=lambda entry: entry.frame_id)

    @torch.no_grad()
    def write(self, tokens, boxes, frame_ids, qualities, valid=None):
        if tokens.shape[0] != self.batch_size or boxes.shape != (self.batch_size, 4):
            raise ValueError('Memory writes must match the bank batch size')
        if valid is None:
            valid = torch.ones(self.batch_size, dtype=torch.bool, device=tokens.device)
        written = []
        for batch_index in range(self.batch_size):
            quality = float(qualities[batch_index])
            frame_id = int(frame_ids[batch_index])
            acceptable = (bool(valid[batch_index]) and math.isfinite(quality)
                          and quality >= self.min_quality
                          and bool(torch.isfinite(tokens[batch_index]).all())
                          and bool(torch.isfinite(boxes[batch_index]).all())
                          and bool((boxes[batch_index, 2:] > 0).all()))
            existing = self.entries(batch_index)
            if existing and frame_id <= existing[-1].frame_id:
                raise ValueError('Memory writes must be strictly chronological')
            written.append(acceptable)
            if not acceptable:
                continue
            entry = MemoryEntry(tokens[batch_index].detach().clone(), frame_id,
                                min(quality, 1.0), boxes[batch_index].detach().clone())
            if self.short_capacity:
                self.short[batch_index].append(entry)
                self.short[batch_index] = self.short[batch_index][-self.short_capacity:]
            if not self.long_capacity:
                continue
            long_store = self.long[batch_index]
            if long_store:
                descriptors = F.normalize(torch.stack([item.tokens.float().mean(0) for item in long_store]), dim=-1)
                descriptor = F.normalize(entry.tokens.float().mean(0), dim=-1)
                similarity, nearest = (descriptors @ descriptor).max(0)
                if 1.0 - float(similarity) < self.novelty_threshold:
                    nearest = int(nearest)
                    if entry.quality > long_store[nearest].quality:
                        long_store[nearest] = entry
                    continue
            long_store.append(entry)
            if len(long_store) > self.long_capacity:
                descriptors = F.normalize(torch.stack([item.tokens.float().mean(0) for item in long_store]), dim=-1)
                similarities = descriptors @ descriptors.T
                similarities.fill_diagonal_(-1)
                redundancy = similarities.max(-1).values
                utility = torch.tensor([item.quality for item in long_store], device=redundancy.device) - redundancy
                del long_store[int(utility.argmin())]
        return written

    def pack(self, features, frame_ids, tokens_per_entry):
        if features.shape[0] != self.batch_size:
            raise ValueError('Memory reads must match the bank batch size')
        entries = [self.entries(batch_index) for batch_index in range(self.batch_size)]
        max_entries = max(len(items) for items in entries)
        token_count = max_entries * tokens_per_entry
        tokens = features.new_zeros(self.batch_size, token_count, features.shape[-1])
        mask = torch.zeros(self.batch_size, token_count, dtype=torch.bool, device=features.device)
        quality = features.new_zeros(self.batch_size, token_count)
        age = features.new_zeros(self.batch_size, token_count)
        long_term = torch.zeros_like(mask)
        for batch_index, items in enumerate(entries):
            short_ids = {entry.frame_id for entry in self.short[batch_index]}
            for entry_index, entry in enumerate(items):
                if entry.frame_id >= int(frame_ids[batch_index]):
                    raise ValueError('Memory cannot read current or future frame entries')
                start = entry_index * tokens_per_entry
                end = start + tokens_per_entry
                tokens[batch_index, start:end] = entry.tokens.to(features)
                mask[batch_index, start:end] = True
                quality[batch_index, start:end] = entry.quality
                age[batch_index, start:end] = int(frame_ids[batch_index]) - entry.frame_id
                long_term[batch_index, start:end] = entry.frame_id not in short_ids
        return tokens, mask, quality, age, long_term
