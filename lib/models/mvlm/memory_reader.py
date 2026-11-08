"""Language-conditioned content readout, independent of head injection."""
import torch
from torch import nn
import torch.nn.functional as F


class MemoryReader(nn.Module):
    def __init__(self, channels, cfg):
        super().__init__()
        hidden, key = cfg.HIDDEN_DIM, cfg.KEY_DIM
        self.cfg = cfg
        self.visual_proj = nn.Linear(channels, hidden)
        self.language_proj = nn.Linear(channels, hidden)
        self.appearance_key = nn.Linear(hidden, key)
        self.semantic_key = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, key))
        self.value_proj = nn.Linear(channels, hidden)
        # Full-image xywh, log age, quality, long-term membership.
        self.state_proj = nn.Linear(7, hidden)
        self.roi_position = nn.Parameter(torch.randn(cfg.ROI_SIZE ** 2, hidden) * .02)
        self.stream_gate = nn.Linear(hidden * 2, 1)
        self.output_gate = nn.Linear(hidden * 2, 1)

    def semantic(self, visual, language):
        language = language.expand_as(visual)
        return F.normalize(self.semantic_key(torch.cat((visual, language, visual * language), -1)), dim=-1)

    @staticmethod
    def attend(query, keys, values, bias):
        logits = 10 * (query @ keys.transpose(-1, -2)) + bias[:, None]
        # A null slot keeps empty/fully rejected banks finite.
        weights = torch.cat((logits, logits.new_zeros(*logits.shape[:-1], 1)), -1).softmax(-1)[..., :-1]
        return weights @ values

    def forward(self, queries, language, bank, frame_ids):
        tokens, valid, quality, age, long_term, boxes = bank.pack(
            queries, frame_ids, self.cfg.ROI_SIZE ** 2, include_metadata=True)
        current, history = self.visual_proj(queries), self.visual_proj(tokens)
        lang = current.new_zeros(current.shape[0], 1, current.shape[-1])
        if language is not None:
            lang = self.language_proj(language.mean(1))[:, None]
        metadata = torch.cat((boxes, age.log1p()[..., None], quality[..., None],
                              long_term.to(tokens)[..., None]), -1)
        positions = self.roi_position.repeat(tokens.shape[1] // (self.cfg.ROI_SIZE ** 2), 1)
        values = self.value_proj(tokens) + self.state_proj(metadata) + positions[None]
        decay = torch.where(long_term, age.new_tensor(self.cfg.LONG_DECAY), age.new_tensor(self.cfg.SHORT_DECAY))
        bias = (quality.clamp_min(1e-6).log() - decay * age.log1p()).masked_fill(~valid, float('-inf'))
        appearance = self.attend(F.normalize(self.appearance_key(current), dim=-1),
                                 F.normalize(self.appearance_key(history), dim=-1), values, bias)
        retrieved = appearance
        if self.cfg.USE_SEMANTIC and language is not None:
            semantic = self.attend(self.semantic(current, lang), self.semantic(history, lang), values, bias)
            stream = torch.sigmoid(self.stream_gate(torch.cat((appearance, semantic), -1)))
            retrieved = stream * appearance + (1 - stream) * semantic
        available = valid.any(-1)
        gate = torch.sigmoid(self.output_gate(torch.cat((current, retrieved), -1)))
        gate = gate * available[:, None, None].to(gate)
        return dict(tokens=gate * retrieved, valid=available, gate=gate)
