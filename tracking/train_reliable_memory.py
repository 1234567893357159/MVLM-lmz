"""Explicit closed-loop training/engineering short runs; no automatic experiments."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from lib.config.mvlm.config import load_config
from lib.train.actors.reliable_memory import ReliableMemoryActor
from lib.train.data.reliable_memory_dataset import read_manifest, ReliableClips
from reliable_memory_runtime import (atomic_json, atomic_save, sha256, seed_all, build_loaded_model,
                                     source_hashes, snapshot_sources, configure_group,
                                     parameter_digest, bn_digest, revision)
from reliable_memory_wandb import WandbMonitor, monitoring_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='mvlm_TF_reliable_memory')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--global-batch', type=int, default=16)
    parser.add_argument('--clip-frames', type=int)
    parser.add_argument('--warmup-frames', type=int)
    parser.add_argument('--engineering-subset', type=int, default=0, help='Videos/source/split; labels run as engineering only')
    parser.add_argument('--seed', type=int, default=20261116)
    parser.add_argument('--group', choices=['M1','M2','M3','M4','M5','M6'], default='M6')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Checkpoint and stop at this absolute update, preserving total schedule')
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--wandb', action='store_true', help='Enable optional W&B monitoring')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb-project')
    parser.add_argument('--wandb-entity')
    parser.add_argument('--wandb-name')
    parser.add_argument('--wandb-log-interval', type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg.TRAIN.RELIABLE_CLOSED_LOOP or cfg.TRAIN.CM_LOSS:
        parser.error('Require the closed-loop config with CM disabled')
    if args.wandb or args.wandb_mode is not None:
        cfg.WANDB.ENABLED = True
    for key in ('MODE', 'PROJECT', 'ENTITY', 'NAME'):
        value = getattr(args, 'wandb_' + key.lower())
        if value is None:
            value = os.environ.get('WANDB_' + key)
        if value is not None:
            cfg.WANDB[key] = value
    if args.wandb_log_interval is not None:
        cfg.WANDB.LOG_INTERVAL = args.wandb_log_interval
    if cfg.WANDB.MODE == 'disabled':
        cfg.WANDB.ENABLED = False
    if cfg.WANDB.MODE not in ('online', 'offline', 'disabled') or cfg.WANDB.LOG_INTERVAL < 1:
        parser.error('Invalid W&B mode or log interval')
    if cfg.TRAIN.OPTIMIZER != 'ADAMW' or cfg.TRAIN.SCHEDULER.TYPE != 'cosine':
        parser.error('This trainer requires explicit AdamW and cosine scheduling')
    if not cfg.TRAIN.FIX_BN or cfg.TRAIN.MEMORY_FREEZE_HEAD:
        parser.error('Require frozen BN running statistics and a trainable CENTER')
    group = args.group
    configure_group(cfg, group)
    if args.max_updates is not None:
        cfg.TRAIN.MAX_UPDATES = args.max_updates
    if args.clip_frames is not None:
        cfg.TRAIN.CLIP_FRAMES = args.clip_frames
    if args.warmup_frames is not None:
        cfg.TRAIN.WARMUP_FRAMES = args.warmup_frames
    if min(cfg.TRAIN.MAX_UPDATES, args.global_batch, args.save_every) < 1 or args.engineering_subset < 0:
        parser.error('Invalid budget/batch/subset')
    if args.stop_after is not None and not 0 < args.stop_after <= cfg.TRAIN.MAX_UPDATES:
        parser.error('stop-after must be inside the total budget')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if args.global_batch % world:
        parser.error('Global batch must be divisible by world size')
    cfg.TRAIN.BATCH_SIZE = args.global_batch//world
    if world > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group('nccl')
    else:
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    output = Path(args.out).resolve()
    if rank == 0:
        if not args.resume:
            output.mkdir(parents=True, exist_ok=False)
        elif not (output/'last.pth.tar').is_file():
            raise ValueError('No checkpoint to resume')
    if world > 1:
        dist.barrier()
    status = dict(status='starting', group=group, engineering_only=bool(args.engineering_subset), step=0)
    def write_status(**kwargs):
        status.update(kwargs)
        if rank == 0:
            atomic_json(output/'training_status.json', status)
    monitor = None
    try:
        write_status()
        seed_all(args.seed)
        partitions, manifest_hash = read_manifest(args.manifest, cfg.TRAIN.CLIP_FRAMES, args.engineering_subset)
        model, _, audit = build_loaded_model(cfg, args.checkpoint, device, training=True)
        frozen_before, bn_before = parameter_digest(model, True), bn_digest(model)
        trainable_before = parameter_digest(model, False)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        total = cfg.TRAIN.MAX_UPDATES
        def lr_scale(step):
            warmup = min(200, total//10)
            if step < warmup:
                return (step+1)/warmup
            return .5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        hashes = source_hashes()
        metadata = dict(config=cfg, manifest_sha256=manifest_hash, checkpoint_sha256=sha256(args.checkpoint),
                        source_hashes=hashes, world_size=world, global_batch=args.global_batch, group=group,
                        engineering_subset=args.engineering_subset, seed=args.seed,
                        clip_length=cfg.TRAIN.CLIP_FRAMES, protocol='strict_language',
                        data_signature=sha256(args.manifest))
        step, attempt = 0, 0
        if args.resume:
            saved = torch.load(output/'last.pth.tar', map_location='cpu')
            if saved['metadata'] != metadata:
                raise ValueError('Resume config/data/code/budget changed')
            incompatible = model.load_state_dict(saved['net'], strict=False)
            if any(not name.startswith('text_encoder.clip.') for name in incompatible.missing_keys) or incompatible.unexpected_keys:
                raise ValueError('Invalid resumed model')
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            step, attempt = saved['step'], saved['attempt']
            rng = saved['rng'][rank]
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state(rng['cuda'], device)
        else:
            seed_all(args.seed + rank)
            if rank == 0:
                snapshot_sources(output/'source', hashes)
                atomic_json(output/'run.json', dict(metadata, git_revision=revision(), audit=audit,
                           train_videos=len(partitions['train']), val_videos=len(partitions['val']),
                           precision='FP32 learned modules; CLIP '+audit['clip_dtype'],
                           notes='Subset audits all identities and resolved paths/signatures of selected videos only.'))
        if world > 1:
            wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                               find_unused_parameters=True)
        else:
            wrapped = model
        actor = ReliableMemoryActor(wrapped, cfg, device)
        samples = max(total*args.global_batch*3, 100)
        aliases = {'LASOT': 'lasot', 'VASTTRACK': 'vasttrack', 'TNL2K_train': 'tnl2k'}
        names, ratios = cfg.DATA.TRAIN.DATASETS_NAME, cfg.DATA.TRAIN.DATASETS_RATIO
        if (len(names) != len(ratios) or len(names) != len(set(names))
                or any(name not in aliases for name in names)):
            raise ValueError('Invalid configured training source names/ratios')
        weights = {aliases[name]: ratio for name, ratio in zip(names, ratios)}
        if any(seq['dataset'] not in weights for seq in partitions['train']):
            raise ValueError('Manifest source is missing from configured source ratios')
        dataset = ReliableClips(partitions['train'], cfg.TRAIN.CLIP_FRAMES, samples, args.seed,
                                max_stride=cfg.DATA.MEMORY_MAX_STRIDE, dataset_weights=weights)
        per_rank = args.global_batch//world
        stop = args.stop_after or total
        monitor = WandbMonitor(output, cfg.WANDB, dict(metadata, audit=audit, git_revision=revision()),
                               resume=args.resume, rank=rank)
        while step < stop:
            started = time.monotonic()
            clips = [dataset[attempt*args.global_batch + rank*per_rank+i] for i in range(per_rank)]
            attempt += 1
            count = torch.tensor(float(actor.supervision_count(clips)), device=device)
            if world > 1:
                dist.all_reduce(count)
            if float(count) == 0:
                if attempt > total*3:
                    raise RuntimeError('Too many empty-supervision batches')
                continue
            optimizer.zero_grad(set_to_none=True)
            # DDP averages gradients, so local losses divide by global_count/world.
            _, metrics, results = actor(clips, backward=True, normalization=float(count)/world)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.TRAIN.GRAD_CLIP_NORM)
            if not bool(torch.isfinite(gradient_norm)):
                raise ValueError('Nonfinite gradient')
            lr_used = optimizer.param_groups[0]['lr']
            optimizer.step()
            scheduler.step()
            step += 1
            row = dict(metrics, step=step, attempt=attempt, gradient_norm=float(gradient_norm),
                       lr=optimizer.param_groups[0]['lr'], lr_used=lr_used, rank=rank)
            with (output/('metrics_rank%d.jsonl'%rank)).open('a') as handle:
                handle.write(json.dumps(row, allow_nan=False)+'\n')
            print(json.dumps(row), flush=True)
            write_status(status='running', step=step)
            if cfg.WANDB.ENABLED and (step == 1 or step % cfg.WANDB.LOG_INTERVAL == 0 or step == stop):
                # Losses already divide by global_count/world; counts sum across ranks.
                keys = sorted(metrics)
                values = torch.tensor([metrics[key] for key in keys], device=device, dtype=torch.float64)
                if world > 1:
                    dist.all_reduce(values)
                global_row = dict(row, **dict(zip(keys, values.cpu().tolist())))
                global_row['loss'] /= world
                if rank == 0:
                    payload = monitoring_metrics(global_row, results, time.monotonic()-started, per_rank)
                    if device.type == 'cuda':
                        payload.update({'gpu/rank0/allocated_mib': torch.cuda.memory_allocated(device)/2**20,
                                        'gpu/rank0/peak_allocated_mib': torch.cuda.max_memory_allocated(device)/2**20})
                    monitor.log(payload, step, force=step == stop)
            if step%args.save_every == 0 or step == stop:
                rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                           cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
                rngs = [None]*world
                if world > 1:
                    dist.all_gather_object(rngs, rng)
                else:
                    rngs[0] = rng
                if rank == 0:
                    state = {k: v.detach().cpu() for k,v in model.state_dict().items()
                             if not k.startswith('text_encoder.clip.')}
                    atomic_save(output/'last.pth.tar', dict(net=state, optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(), step=step, attempt=attempt,
                                metadata=metadata, rng=rngs))
        report = dict(rank=rank, frozen_unchanged=parameter_digest(model, True)==frozen_before,
                      bn_unchanged=bn_digest(model)==bn_before,
                      trainable_changed=parameter_digest(model, False)!=trainable_before,
                      trainable_sha256=parameter_digest(model, False), step=step)
        atomic_json(output/('audit_rank%d.json'%rank), report)
        if not report['frozen_unchanged'] or not report['bn_unchanged']:
            raise AssertionError('Frozen parameters or BN statistics changed')
        if world > 1:
            reports = [None]*world
            dist.all_gather_object(reports, report)
            if len({r['trainable_sha256'] for r in reports}) != 1:
                raise AssertionError('DDP replicas differ')
        write_status(status='completed' if step == total else 'checkpointed', step=step, audit=report)
    except BaseException as error:
        write_status(status='failed', error=repr(error))
        raise
    finally:
        if monitor is not None:
            monitor.close(status['status'])
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
