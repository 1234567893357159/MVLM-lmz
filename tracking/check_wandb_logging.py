"""Offline checks for training logging, checkpoint resume and rank-zero ownership."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.train.admin.wandb_logger import WandbLogger
from lib.train.trainers.ltr_trainer import LTRTrainer


def settings(folder):
    return SimpleNamespace(save_dir=folder, project_path='train/check', local_rank=-1,
                           env=SimpleNamespace(workspace_dir=folder, tensorboard_dir=folder + '/tb'),
                           device=torch.device('cpu'), use_gpu=False, move_data_to_gpu=False,
                           scheduler_type='step', grad_clip_norm=1., print_interval=1,
                           script_name='mvlm', config_name='wandb_check', description='offline check',
                           log_file=folder + '/train.log', save_every_epoch=True,
                           wandb_config={'ENABLED': True, 'MODE': 'offline', 'LOG_INTERVAL': 1},
                           training_config={'purpose': 'synthetic logging check, not model training'})


class Actor:
    def __init__(self):
        self.net = torch.nn.Linear(2, 1)

    def to(self, device):
        self.net.to(device)

    def train(self, training):
        self.net.train(training)

    def __call__(self, data):
        loss = self.net(data['x']).square().mean()
        return loss, {'Loss/total': loss.item(), 'IoU': .25}


class Loader:
    name = 'train'
    training = True
    epoch_interval = 1
    stack_dim = 1
    sampler = None

    def __len__(self):
        return 2

    def __iter__(self):
        for _ in range(2):
            yield {'x': torch.ones(2, 2), 'template_images': torch.zeros(1, 2, 1)}


def make_trainer(folder):
    actor = Actor()
    optimizer = torch.optim.AdamW(actor.net.parameters(), lr=1e-4)
    return LTRTrainer(actor, [Loader()], optimizer, settings(folder),
                      torch.optim.lr_scheduler.StepLR(optimizer, step_size=2))


def distributed_worker(rank, init_path, folder):
    dist.init_process_group('gloo', init_method='file://' + init_path, rank=rank, world_size=2)
    try:
        logger = WandbLogger(settings(folder))
        logger.start(torch.nn.Linear(2, 1))
        assert (logger.run is not None) == (rank == 0)
        if rank == 0:
            with patch.object(logger.run, 'log', wraps=logger.run.log) as log:
                logger.log_metrics({'train/Loss/total': float(1 + 2 * rank)}, 1)
                assert log.call_args[0][0]['train/Loss/total'] == 2.
        else:
            logger.log_metrics({'train/Loss/total': float(1 + 2 * rank)}, 1)
        logger.finish()
    finally:
        dist.destroy_process_group()


def main():
    os.environ['WANDB_MODE'] = 'offline'
    with tempfile.TemporaryDirectory(prefix='mvlm-wandb-check-') as folder:
        trainer = make_trainer(folder)
        trainer.wandb_logger.start(trainer.actor.net)
        with patch.object(trainer.wandb_logger.run, 'log', wraps=trainer.wandb_logger.run.log) as log:
            with patch('torch.cuda.synchronize'):
                trainer.train(1, fail_safe=False)
            payloads = [call.args[0] for call in log.call_args_list]
            assert any('train/Loss/total' in payload for payload in payloads)
            assert any('epoch/train/IoU' in payload for payload in payloads)
            assert any('lr/group0' in payload for payload in payloads)
            assert any('runtime/train/seconds_per_batch_rank0' in payload for payload in payloads)
        assert trainer.global_step == 2
        trainer.wandb_logger.finish()
        metadata = Path(folder) / 'wandb_tracking/train/check/run.json'
        assert json.loads(metadata.read_text())['mode'] == 'offline'
        checkpoint = Path(folder) / 'checkpoints/train/check/Linear_ep0001.pth.tar'
        assert torch.load(checkpoint, map_location='cpu')['global_step'] == 2
        original_id = json.loads(metadata.read_text())['id']
        resumed = make_trainer(folder)
        assert resumed.load_checkpoint()
        assert resumed.global_step == 2 and resumed.epoch == 1
        with patch('torch.cuda.synchronize'):
            resumed.train(2, fail_safe=False)
        assert resumed.global_step == 4
        assert json.loads(metadata.read_text())['id'] == original_id
        resumed.wandb_logger.finish()
        disabled = settings(folder)
        disabled.wandb_config = {'ENABLED': False}
        logger = WandbLogger(disabled)
        logger.start(torch.nn.Linear(2, 1))
        assert logger.run is None
        failed = make_trainer(folder + '/failed')
        with patch.object(failed, 'train_epoch', side_effect=RuntimeError('synthetic failure')):
            try:
                failed.train(1, fail_safe=True)
            except RuntimeError as exc:
                assert str(exc) == 'synthetic failure'
            else:
                raise AssertionError('Training failures must propagate to W&B lifecycle handling')
        mp.spawn(distributed_worker, args=(folder + '/dist-init', folder + '/distributed'), nprocs=2, join=True)
    print('PASS: offline logging, disabled mode, checkpoint steps, resume, two-rank metric averaging and rank-zero ownership')


if __name__ == '__main__':
    main()
