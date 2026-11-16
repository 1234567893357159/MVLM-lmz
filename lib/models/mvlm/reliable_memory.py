"""Small runtime bank; learnable memory reading remains in MemoryFusion."""
import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .memory import MemoryEntry, TargetMemoryBank
from .memory_policy import ReliableWritePolicy, WriteDecision


@dataclass
class ReliableEntry(MemoryEntry):
    committed_at: int = -1
    disc_ema: float = 0.0
    disc_count: int = 0


class ReliableMemoryBank(TargetMemoryBank):
    def __init__(self, batch_size, cfg):
        self.cfg = copy.deepcopy(cfg)
        super().__init__(batch_size, cfg.SHORT_CAPACITY, cfg.LONG_CAPACITY,
                         cfg.MIN_QUALITY, cfg.NOVELTY_THRESHOLD)

    def reset(self):
        super().reset()
        self.policies = [ReliableWritePolicy(self.cfg.POLICY) for _ in range(self.batch_size)]
        self.last_observed = [-1] * self.batch_size
        self.events = [{} for _ in range(self.batch_size)]

    def write(self, *args, **kwargs):
        raise RuntimeError('Reliable memory requires update(Observation), not legacy GT/mixed writes')

    @torch.no_grad()
    def update(self, obs, competitors=None, batch_index=0):
        i = batch_index
        if obs.frame_id <= self.last_observed[i]:
            raise ValueError('Memory updates must be strictly chronological')
        self.last_observed[i] = obs.frame_id
        if self.cfg.RELIABLE_WRITE:
            decision = self.policies[i].observe(obs)
        else:
            valid = (obs.roi_valid and obs.vlm_pass and math.isfinite(obs.score)
                     and obs.score >= self.min_quality
                     and bool(torch.isfinite(obs.roi_tokens).all())
                     and bool(torch.isfinite(obs.box_normalized_xywh).all())
                     and bool((obs.box_normalized_xywh[2:] > 0).all()))
            decision = WriteDecision('commit' if valid else 'reject', () if valid else ('legacy_filter',),
                                     1 if valid else 0, 'legacy')
        event = dict(frame_id=obs.frame_id, action=decision.action, reasons=list(decision.reasons),
                     streak=decision.streak, transition=decision.transition, evicted=None)
        self.events[i] = event
        if decision.action != 'commit':
            return decision
        entry = ReliableEntry(obs.roi_tokens.detach().clone(), obs.frame_id, min(obs.score, 1.),
                              obs.box_normalized_xywh.detach().clone(), obs.frame_id)
        if self.short_capacity:
            self.short[i].append(entry)
            self.short[i] = self.short[i][-self.short_capacity:]
        if not self.long_capacity:
            return decision
        store = self.long[i]
        # Stats use only accepted observations. Unknown competitor evidence stays unknown.
        if self.cfg.DISTRACTOR_RETENTION and competitors is not None and competitors.numel():
            competitors = F.normalize(competitors.detach().float(), dim=-1)
            if not bool(torch.isfinite(competitors).all()):
                raise ValueError('Nonfinite competitor descriptors')
            target = F.normalize(obs.descriptor.detach().float(), dim=-1)
            for item in store + [entry]:
                descriptor = F.normalize(item.tokens.float().mean(0), dim=-1)
                margin = float(descriptor @ target - (competitors @ descriptor).max())
                decay = self.cfg.RETENTION.EMA_DECAY
                item.disc_ema = margin if not item.disc_count else decay * item.disc_ema + (1-decay) * margin
                item.disc_count += 1
        if store:
            descriptors = F.normalize(torch.stack([e.tokens.float().mean(0) for e in store]), dim=-1)
            descriptor = F.normalize(entry.tokens.float().mean(0), dim=-1)
            similarity, nearest = (descriptors @ descriptor).max(0)
            if 1 - float(similarity) < self.novelty_threshold:
                nearest = int(nearest)
                old = store[nearest]
                replace = entry.quality > old.quality
                if (self.cfg.DISTRACTOR_RETENTION
                        and abs(entry.quality-old.quality) <= self.cfg.RETENTION.UTILITY_TOLERANCE
                        and min(entry.disc_count, old.disc_count) >= self.cfg.RETENTION.MIN_OBSERVATIONS):
                    replace = (entry.disc_ema, entry.frame_id) > (old.disc_ema, old.frame_id)
                if replace:
                    event['evicted'] = old.frame_id
                    store[nearest] = entry
                return decision
        store.append(entry)
        if len(store) > self.long_capacity:
            descriptors = F.normalize(torch.stack([e.tokens.float().mean(0) for e in store]), dim=-1)
            sims = descriptors @ descriptors.T
            sims.fill_diagonal_(-1)
            utility = torch.tensor([e.quality for e in store], device=sims.device) - sims.max(-1).values
            index = int(utility.argmin())
            if self.cfg.DISTRACTOR_RETENTION:
                eligible = [j for j, item in enumerate(store)
                            if float(utility[j]-utility.min()) <= self.cfg.RETENTION.UTILITY_TOLERANCE
                            and item.disc_count >= self.cfg.RETENTION.MIN_OBSERVATIONS]
                if len(eligible) >= 2:
                    index = min(eligible, key=lambda j: (store[j].disc_ema, store[j].frame_id))
            event['evicted'] = store[index].frame_id
            del store[index]
        return decision

    def state_dict(self):
        return copy.deepcopy(dict(version=1, cfg=self.cfg, batch_size=self.batch_size,
                                  short=self.short, long=self.long, last_observed=self.last_observed,
                                  policies=[vars(p) for p in self.policies], events=self.events))

    def load_state_dict(self, state):
        if state['version'] != 1 or state['batch_size'] != self.batch_size or state['cfg'] != self.cfg:
            raise ValueError('Incompatible reliable memory snapshot')
        state = copy.deepcopy(state)
        self.short, self.long = state['short'], state['long']
        self.last_observed, self.events = state['last_observed'], state['events']
        for policy, values in zip(self.policies, state['policies']):
            policy.__dict__.update(values)
