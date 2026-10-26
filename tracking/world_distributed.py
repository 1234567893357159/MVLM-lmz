"""Torchrun data parallelism for the world actor, with shared data auditing."""
import datetime
import hashlib
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler


class RankClipSampler(Sampler):
    """Assign disjoint, contiguous per-rank clips within each global batch."""
    def __init__(self, start, stop, batch_size, rank, world_size):
        self.start, self.stop, self.batch = start, stop, batch_size
        self.rank, self.world_size = rank, world_size

    def __iter__(self):
        for base in range(self.start, self.stop, self.batch * self.world_size):
            yield from range(base + self.rank * self.batch,
                             min(base + (self.rank + 1) * self.batch, self.stop))

    def __len__(self):
        return sum(1 for _ in self)


class ActorStep(torch.nn.Module):
    def __init__(self, model, actor, cover_unused=True):
        super().__init__()
        self.model, self.actor = model, actor
        self.cover_unused = cover_unused

    def forward(self, clips, step):
        self.actor.step = step
        loss, metrics = self.actor(clips)
        if not self.cover_unused:
            return loss, metrics
        # Every rank must participate even if its clips have no valid targets.
        # Also cover conditionally unused appearance/motion/fusion parameters.
        zero = torch.stack([p.reshape(-1)[0] for p in self.model.parameters()
                            if p.requires_grad]).sum() * 0
        return (zero if loss is None else loss + zero), metrics


class WorldDistributed:
    def setup(self, requested_device):
        self.rank = int(os.environ.get('RANK', 0))
        self.size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.primary = self.rank == 0
        self.device = torch.device(requested_device)
        if self.size > 1:
            if self.device.type == 'cuda':
                self.device = torch.device('cuda', self.local_rank)
                torch.cuda.set_device(self.device)
            dist.init_process_group('nccl' if self.device.type == 'cuda' else 'gloo',
                                    timeout=datetime.timedelta(minutes=20))
        return self.device

    def primary_call(self, function):
        result, failure = None, [None]
        if self.primary:
            try:
                result = function()
            except Exception as error:
                failure[0] = '{}: {}'.format(type(error).__name__, error)
                if self.size == 1:
                    raise
        if self.size > 1:
            dist.broadcast_object_list(failure, src=0)
        if failure[0]:
            raise RuntimeError('Rank 0 failed: ' + failure[0])
        return result

    def wrap(self, model, actor, broadcast_buffers=False):
        module = ActorStep(model, actor, cover_unused=self.size > 1)
        if self.size > 1:
            return DistributedDataParallel(module,
                device_ids=[self.local_rank] if self.device.type == 'cuda' else None,
                broadcast_buffers=broadcast_buffers)
        return module

    def all_true(self, value):
        tensor = torch.tensor(int(value), device=self.device)
        if self.size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return bool(tensor.item())

    def metrics(self, values):
        counts = ('supervised_frames', 'predictions', 'active')
        means = ('loss', 'appearance_cosine_error', 'motion_error', 'copy_cosine_error',
                 'stationary_motion_error', 'constant_velocity_error', 'predicted_history_probability')
        n = values.get('supervised_frames', 0)
        packed = [values.get(key, 0) for key in counts]
        for key in means:
            packed.extend([values.get(key, 0) * n, n if key in values else 0])
        tensor = torch.tensor(packed, dtype=torch.float64, device=self.device)
        if self.size > 1:
            dist.all_reduce(tensor)
        packed = tensor.tolist()
        result = {key: int(packed[i]) for i, key in enumerate(counts)}
        for i, key in enumerate(means):
            total, weight = packed[len(counts) + 2 * i:len(counts) + 2 * i + 2]
            if weight:
                result[key] = total / weight
        return result

    def gather(self, value):
        if self.size == 1:
            return [value]
        values = [None] * self.size
        dist.all_gather_object(values, value)
        return values

    def barrier(self):
        if self.size > 1:
            dist.barrier()

    def sync_buffers(self, model):
        if self.size > 1:
            for value in model.buffers():
                dist.broadcast(value, src=0)

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()


def trainable_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
