"""Two-rank CPU regression for sparse supervision and disjoint clip sampling."""
import json
import os
from pathlib import Path
import socket
import tempfile

import torch
import torch.multiprocessing as mp
from world_distributed import WorldDistributed, RankClipSampler, trainable_digest


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.unused = torch.nn.Parameter(torch.tensor(3.))
        self.frozen = torch.nn.Parameter(torch.tensor(7.), requires_grad=False)


class ToyActor:
    def __init__(self, model, rank):
        self.model, self.rank = model, rank

    def __call__(self, clips):
        if self.step == 1 or (self.step == 0 and self.rank == 0):
            return None, dict(supervised_frames=0, predictions=0, active=0)
        target = 5. if self.step == 0 else (3. if self.rank == 0 else 7.)
        count = 2 if self.step == 0 else (1 if self.rank == 0 else 3)
        loss = (self.model.weight - target).square()
        return loss, dict(loss=float(loss.detach()), supervised_frames=count, predictions=count, active=0)


def worker(rank, port, destination):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    torch.set_num_threads(1)
    parallel = WorldDistributed()
    parallel.setup('cpu')
    try:
        model = ToyModel()
        training = parallel.wrap(model, ToyActor(model, rank))
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss, local = training([], step)
            metrics = parallel.metrics(local)
            count = metrics['supervised_frames']
            loss = loss * (local['supervised_frames'] * 2 / max(1, count))
            loss.backward()
            if count:
                optimizer.step()
            expected = 1.8 if step < 2 else 2.64
            assert abs(model.weight.item() - expected) < 1e-6, (step, model.weight)
            assert model.unused.item() == 3 and model.frozen.item() == 7
            assert model.frozen.grad is None
        indices = list(RankClipSampler(0, 32, 2, rank, 2))
        resumed = list(RankClipSampler(8, 32, 2, rank, 2))
        assert resumed == [i for i in indices if i >= 8]
        rows = parallel.gather(dict(rank=rank, digest=trainable_digest(model), indices=indices))
        assert rows[0]['digest'] == rows[1]['digest']
        assert not set(rows[0]['indices']).intersection(rows[1]['indices'])
        assert sorted(rows[0]['indices'] + rows[1]['indices']) == list(range(32))
        model.register_buffer('running_stat', torch.tensor(float(rank + 10)))
        parallel.sync_buffers(model)
        assert model.running_stat.item() == 10
        if rank == 0:
            Path(destination).write_text(json.dumps(dict(sparse_rank_backward=True,
                all_empty_batch_skipped=True, supervised_frame_weighting=True,
                unused_parameters_safe=True, replicas_identical=True,
                disjoint_sampler_and_resume=True, validation_buffers_synchronized=True)))
    finally:
        parallel.close()


if __name__ == '__main__':
    with socket.socket() as endpoint:
        endpoint.bind(('127.0.0.1', 0))
        port = endpoint.getsockname()[1]
    with tempfile.TemporaryDirectory() as temporary:
        destination = str(Path(temporary) / 'result.json')
        mp.spawn(worker, args=(port, destination), nprocs=2, join=True)
        print(Path(destination).read_text())
