"""One-step, language-conditioned target dynamics in frozen MVLM feature space."""
import torch
from torch import nn
from .world_state import apply_motion, motion_target


class WorldModel(nn.Module):
    def __init__(self, channels, history=4, hidden=256, layers=2, heads=4):
        super().__init__()
        if history < 2 or hidden % heads:
            raise ValueError('History >= 2 and hidden divisible by heads required')
        self.history = history
        self.visual_proj = nn.Linear(channels, hidden)
        self.language_proj = nn.Linear(channels, hidden)
        # cx,cy,log(w),log(h), velocity(4), quality, valid
        self.state_proj = nn.Linear(10, hidden)
        self.time_proj = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.roi_position = nn.Parameter(torch.randn(4, hidden) * .02)
        self.types = nn.Parameter(torch.randn(3, hidden) * .02)
        self.future_queries = nn.Parameter(torch.randn(5, hidden) * .02)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.,
                                            activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(hidden))
        self.appearance_head = nn.Linear(hidden, channels)
        self.motion_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 4))
        # Start close to appearance persistence / stationary geometry.
        nn.init.zeros_(self.appearance_head.weight)
        nn.init.zeros_(self.appearance_head.bias)
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)

    def forward(self, history, language, target_frame):
        tokens, boxes, valid = history['tokens'], history['boxes'], history['valid']
        frames = history['frame_ids']
        batch, length, roi, _ = tokens.shape
        if length != self.history or roi != 4 or not (frames[:, 1:] > frames[:, :-1]).all():
            raise ValueError('Malformed or unordered history')
        if not (target_frame > frames[:, -1]).all():
            raise ValueError('Future prediction must follow all history frames')
        # Remove invalid observations before any learned projection (not afterwards):
        # inf * zero gradients can otherwise contaminate an otherwise masked sample.
        tokens = tokens.masked_fill(~valid[..., None, None], 0)
        fallback = boxes.new_tensor([0., 0., 1., 1.])
        boxes = torch.where(valid[..., None], boxes, fallback)
        safe_size = boxes[..., 2:].clamp_min(1e-5)
        geometry = torch.cat((boxes[..., :2], safe_size.log()), -1)
        velocity = torch.zeros_like(boxes)
        dt = (frames[:, 1:] - frames[:, :-1]).clamp_min(1).to(boxes)[..., None]
        velocity[:, 1:] = motion_target(boxes[:, :-1], boxes[:, 1:]) / dt
        velocity[:, 1:] *= (valid[:, 1:] & valid[:, :-1])[..., None]
        states = torch.cat((geometry, velocity, history['quality'][..., None],
                            valid.to(boxes)[..., None]), -1)
        time = (frames - target_frame[:, None]).to(boxes).unsqueeze(-1) / self.history
        time = self.time_proj(time)
        visual = (self.visual_proj(tokens) + self.roi_position + self.types[0]
                  + time.unsqueeze(2))
        state = self.state_proj(states) + time + self.types[1]
        past = torch.cat((visual, state.unsqueeze(2)), 2).flatten(1, 2)
        past = past.masked_fill(~valid.repeat_interleave(5, 1)[..., None], 0)
        text = self.language_proj(language.mean(1)).unsqueeze(1) + self.types[2]
        queries = self.future_queries.unsqueeze(0).expand(batch, -1, -1)
        sequence = torch.cat((past, text, queries), 1)
        padding = torch.cat((~valid.repeat_interleave(5, 1),
                             torch.zeros(batch, 6, dtype=torch.bool, device=tokens.device)), 1)
        future = self.transformer(sequence, src_key_padding_mask=padding)[:, -5:]
        delta = self.motion_head(future[:, 4])
        return dict(tokens=tokens[:, -1] + self.appearance_head(future[:, :4]),
                    boxes=apply_motion(boxes[:, -1], delta), delta=delta,
                    valid=valid[:, -1] & (valid.sum(1) >= 2),
                    quality=history['quality'][:, -1], frame_ids=target_frame)
