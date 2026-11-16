"""Causal write confirmation. Model observations only; no ground-truth inputs."""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class Observation:
    frame_id: int
    box_image_xywh: torch.Tensor
    box_normalized_xywh: torch.Tensor
    roi_tokens: torch.Tensor
    descriptor: torch.Tensor
    score: float
    cls_margin: Optional[float]
    language_margin: Optional[float]
    vlm_pass: bool
    roi_valid: bool
    used_global_search: bool = False


@dataclass
class WriteDecision:
    action: str
    reasons: Tuple[str, ...]
    streak: int
    transition: str


class ReliableWritePolicy:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.CONFIRM_FRAMES < 1 or cfg.RECOVERY_CONFIRM_FRAMES < 1:
            raise ValueError('Confirmation lengths must be positive')
        self.reset()

    def reset(self):
        self.previous = None
        self.streak = 0
        self.recovery = True
        self.last_frame = -1

    @torch.no_grad()
    def observe(self, obs):
        if obs.frame_id <= self.last_frame:
            raise ValueError('Observations must be strictly chronological')
        self.last_frame = obs.frame_id
        reasons = []
        tensors = (obs.box_image_xywh, obs.box_normalized_xywh, obs.roi_tokens, obs.descriptor)
        if not all(bool(torch.isfinite(x).all()) for x in tensors) or not math.isfinite(obs.score):
            reasons.append('nonfinite')
        if not obs.roi_valid or not bool((obs.box_image_xywh[2:] > 0).all()):
            reasons.append('invalid_roi')
        if obs.score < self.cfg.MIN_SCORE:
            reasons.append('score')
        if not obs.vlm_pass:
            reasons.append('vlm_margin')
        for value, threshold, name in (
                (obs.cls_margin, self.cfg.MIN_CLS_MARGIN, 'cls_margin'),
                (obs.language_margin, self.cfg.MIN_LANGUAGE_MARGIN, 'language_margin')):
            if value is not None and (not math.isfinite(value) or value < threshold):
                reasons.append(name)
        if reasons:
            self.previous = None
            self.streak = 0
            self.recovery |= obs.used_global_search
            return WriteDecision('reject', tuple(reasons), 0, 'recovery' if self.recovery else 'normal')

        # Enter recovery once, rather than restarting on every global-search frame.
        if obs.used_global_search and not self.recovery:
            self.recovery = True
            self.streak = 0
            self.previous = None
        consistent = True
        if self.previous is not None:
            previous = self.previous
            similarity = float(F.cosine_similarity(obs.descriptor[None], previous['descriptor'][None]))
            consistent = similarity >= self.cfg.MIN_DESCRIPTOR_SIM
            if not self.recovery:
                box = obs.box_image_xywh.float()
                old = previous['box'].float()
                elapsed = max(1, obs.frame_id - previous['frame_id'])
                displacement = ((box[:2] + box[2:] / 2) - (old[:2] + old[2:] / 2)).norm()
                speed = float(displacement / old[2:].prod().sqrt().clamp_min(1e-6)) / elapsed
                scale = float((box[2:] / old[2:].clamp_min(1e-6)).log().abs().max()) / elapsed
                if speed > self.cfg.MAX_CENTER_SPEED or scale > self.cfg.MAX_LOG_SCALE_RATE:
                    self.recovery = True
                    consistent = False
        self.streak = self.streak + 1 if consistent else 1
        self.previous = dict(descriptor=obs.descriptor.detach().clone(),
                             box=obs.box_image_xywh.detach().clone(), frame_id=obs.frame_id)
        required = self.cfg.RECOVERY_CONFIRM_FRAMES if self.recovery else self.cfg.CONFIRM_FRAMES
        transition = 'recovery' if self.recovery else 'normal'
        if self.streak >= required:
            self.recovery = False
            return WriteDecision('commit', (), self.streak, transition)
        return WriteDecision('pending', ('confirmation',) if consistent else ('inconsistent',),
                             self.streak, transition)
