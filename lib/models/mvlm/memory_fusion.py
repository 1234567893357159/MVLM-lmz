import torch
import torch.nn.functional as F
from torch import nn


class MemoryFusion(nn.Module):
    def __init__(self, channels, hidden_dim=256, key_dim=128, roi_size=2,
                 use_semantic=True, short_decay=0.1, long_decay=0.01):
        super().__init__()
        self.roi_size = roi_size
        self.use_semantic = use_semantic
        self.short_decay = short_decay
        self.long_decay = long_decay
        self.visual_proj = nn.Linear(channels, hidden_dim)
        self.text_proj = nn.Linear(channels, hidden_dim)
        self.appearance_key = nn.Linear(hidden_dim, key_dim)
        self.semantic_key = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim),
                                          nn.GELU(), nn.Linear(hidden_dim, key_dim))
        self.value_proj = nn.Linear(channels, hidden_dim)
        self.roi_position = nn.Parameter(torch.zeros(roi_size * roi_size, hidden_dim))
        self.stream_gate = nn.Linear(hidden_dim * 2, 1)
        self.output_gate = nn.Linear(hidden_dim * 2, 1)
        self.output_proj = nn.Linear(hidden_dim, channels)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _semantic(self, visual, language):
        language = language.expand_as(visual)
        return F.normalize(self.semantic_key(torch.cat((visual, language, visual * language), -1)), dim=-1)

    def _read(self, query, key, values, bias):
        logits = 10.0 * (query @ key.transpose(-1, -2)) + bias.unsqueeze(1)
        null_logits = logits.new_zeros(*logits.shape[:-1], 1)
        weights = torch.cat((logits, null_logits), -1).softmax(-1)[..., :-1]
        return weights @ values

    def forward(self, features, text_emb, memory, frame_ids):
        tokens, valid, quality, age, long_term = memory.pack(features, frame_ids, self.roi_size ** 2)
        visual = self.visual_proj(features)
        language = visual.new_zeros(features.shape[0], 1, visual.shape[-1])
        if text_emb is not None:
            language = self.text_proj(text_emb.mean(1)).unsqueeze(1)
        history = self.visual_proj(tokens)
        positions = self.roi_position.repeat(tokens.shape[1] // (self.roi_size ** 2), 1)
        values = self.value_proj(tokens) + positions.unsqueeze(0)
        decay = torch.where(long_term, age.new_tensor(self.long_decay), age.new_tensor(self.short_decay))
        bias = quality.clamp_min(1e-6).log() - decay * torch.log1p(age)
        bias = bias.masked_fill(~valid, float('-inf'))
        appearance = self._read(F.normalize(self.appearance_key(visual), dim=-1),
                                F.normalize(self.appearance_key(history), dim=-1), values, bias)
        if self.use_semantic and text_emb is not None:
            semantic = self._read(self._semantic(visual, language), self._semantic(history, language), values, bias)
            stream_gate = torch.sigmoid(self.stream_gate(torch.cat((appearance, semantic), -1)))
            retrieved = stream_gate * appearance + (1 - stream_gate) * semantic
        else:
            retrieved = appearance
        gate = torch.sigmoid(self.output_gate(torch.cat((visual, retrieved), -1)))
        has_memory = valid.any(-1).view(-1, 1, 1)
        delta = self.output_proj(gate * retrieved) * has_memory.to(features.dtype)
        return features + delta, gate * has_memory.to(features.dtype)
