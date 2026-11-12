"""Measure real stage-B clips on an explicitly selected device; never resumes a run."""
import argparse
import json
from pathlib import Path
import random
import time

import _init_paths
import numpy as np
import torch

from lib.config.mvlm.config import load_config
from lib.models.mvlm import build_mvlm
from lib.models.mvlm.joint_checkpoint import load_baseline, load_joint
from lib.train.actors.memory_world import MemoryWorldActor
from lib.train.dataset.world_clips import WorldClips, collate_world_clips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--data-cache', required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batched-stage-b', action='store_true')
    args = parser.parse_args()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    if cfg.TRAIN.WORLD.STAGE != 'B' or cfg.TRAIN.WORLD.BATCHED_ENCODER:
        raise ValueError('This profile uses the existing stage-B sequential actor')
    random.seed(cfg.TRAIN.WORLD.SEED)
    np.random.seed(cfg.TRAIN.WORLD.SEED)
    torch.manual_seed(cfg.TRAIN.WORLD.SEED)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report = dict(status='starting', batch_per_gpu=args.batch, steps=[], parent=args.parent)

    def record():
        temporary = target / 'profile.json.tmp'
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(target / 'profile.json')

    record()
    try:
        model = build_mvlm(cfg, training=False)
        load_baseline(model, torch.load(args.baseline, map_location='cpu'))
        parent = torch.load(args.parent, map_location='cpu')
        if parent.get('world_stage') != 'A':
            raise ValueError('Expected a stage-A parent')
        load_joint(model, parent)
        report['parent_step'] = parent['step']
        del parent
        model.configure_world_stage('B')
        model.cuda().train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        report['trainable_parameters'] = sum(p.numel() for p in trainable)
        initial = [p.detach().clone() for p in trainable]
        frozen = {name: p._version for name, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW(trainable, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        if args.batched_stage_b:
            from memory_world_stage_b_actor import BatchedStageBActor
            actor = BatchedStageBActor(model, cfg, 'B')
        else:
            actor = MemoryWorldActor(model, cfg, 'B')
        report['batch_execution'] = 'batched' if args.batched_stage_b else 'sequential'
        cache = torch.load(args.data_cache, map_location='cpu')
        aliases = {'LASOT': 'lasot', 'VASTTRACK': 'vasttrack', 'TNL2K_train': 'tnl2k'}
        weights = {aliases[name]: ratio for name, ratio in zip(cfg.DATA.TRAIN.DATASETS_NAME, cfg.DATA.TRAIN.DATASETS_RATIO)}
        dataset = WorldClips(cache['partitions']['train'], cfg.TRAIN.WORLD.CLIP_LENGTH,
                             args.batch * args.steps, cfg.TRAIN.WORLD.SEED,
                             stride=cfg.TRAIN.WORLD.MAX_STRIDE, dataset_weights=weights)
        del cache
        report.update(status='running', current_step=0)
        record()
        for step in range(args.steps):
            data_start = time.monotonic()
            clips = collate_world_clips([dataset[step * args.batch + i] for i in range(args.batch)])
            data_seconds = time.monotonic() - data_start
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            actor.step = step
            loss, metrics = actor(clips)
            if loss is None or not torch.isfinite(loss).item():
                raise ValueError('Missing or non-finite profile loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.TRAIN.GRAD_CLIP_NORM)
            if not torch.isfinite(norm).item():
                raise ValueError('Non-finite profile gradient')
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            row = dict(step=step + 1, data_seconds=data_seconds, compute_seconds=seconds,
                       clips_per_second=args.batch / seconds, grad_norm=float(norm),
                       peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20, **metrics)
            report['steps'].append(row)
            report['current_step'] = step + 1
            record()
            print(json.dumps(row), flush=True)
            del clips, loss
        report.update(status='completed', trainable_updated=any(not torch.equal(a, b) for a, b in zip(initial, trainable)),
                      frozen_parameters_unchanged=all(p._version == frozen[name] and p.grad is None
                          for name, p in model.named_parameters() if not p.requires_grad))
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error),
                      peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        record()
        raise
    record()


if __name__ == '__main__':
    main()
