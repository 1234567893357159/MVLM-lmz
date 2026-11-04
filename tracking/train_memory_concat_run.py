"""One bounded DDP segment of the audited, step-based memory concat experiment."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import cv2
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from lib.config.mvlm.config import cfg, update_config_from_file
from lib.models.mvlm import build_mvlm
from lib.train.admin.settings import Settings
from lib.train.base_functions import update_settings, build_dataloaders
from lib.train.actors.mvlm import MVLM_Actor
from lib.train.memory_run_utils import (write_json, memory_optimizer, set_step_lr,
                                        frozen_digest, bn_state, assert_bn_unchanged,
                                        validate_step_budget, optimizer_audit)
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def save_checkpoint(model, optimizer, step, loader_length, out, config, total_steps=4000):
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, rng_state())
    if dist.get_rank() == 0:
        weights = {key: value.detach().cpu() for key, value in model.state_dict().items()
                   if not key.startswith('text_encoder.clip.')}
        checkpoint = dict(net=weights, optimizer=optimizer.state_dict(), global_step=step,
                          epoch=step // loader_length, batch_in_epoch=step % loader_length,
                          scheduler=dict(type='memory_step', last_update=step,
                                         warmup_updates=200, decay_after_update=3000),
                          total_step_budget=total_steps,
                          rng_states=states, config=dict(config),
                          replay_limit='Worker RNG/prefetch state is not captured; not bitwise replay')
        path = out / 'checkpoints' / ('MVLM_step%06d.pth.tar' % step)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        torch.save(checkpoint, temp)
        temp.replace(path)
        write_json(out/'latest_checkpoint.json', dict(step=step, path=str(path), updated_at=now()))
    dist.barrier()


def run(args):
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    dist.init_process_group('nccl', timeout=dt.timedelta(minutes=15))
    device = torch.device('cuda', local)
    seed = 20261018 + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    update_config_from_file(str(args.config))
    if args.batch:
        cfg.TRAIN.BATCH_SIZE = args.batch
    cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = cfg.TRAIN.BATCH_SIZE * dist.get_world_size() * 1000
    settings = Settings()
    settings.local_rank = local
    settings.save_dir = str(args.out)
    settings.env.tensorboard_dir = str(args.out/'tensorboard')
    settings.config_name = 'mvlm_TF_memory_concat_2gpu'
    settings.script_name = 'mvlm'
    settings.project_path = 'train/mvlm/' + settings.config_name
    update_settings(settings, cfg)
    loader = build_dataloaders(cfg, settings)
    model = build_mvlm(cfg, training=True).to(device)
    original_digest = frozen_digest(model)
    original_bn = bn_state(model)
    optimizer = memory_optimizer(model, cfg)
    step = 0
    rng_restored = False
    if args.resume:
        checkpoint = torch.load(str(args.resume), map_location='cpu')
        missing = model.load_state_dict(checkpoint['net'], strict=False)
        assert not missing.unexpected_keys and all(k.startswith('text_encoder.clip.') for k in missing.missing_keys)
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['global_step']
        assert frozen_digest(model) == original_digest, 'Resumed frozen encoder/text differ from initialization'
        assert_bn_unchanged(model, original_bn)
        if len(checkpoint.get('rng_states', [])) == dist.get_world_size():
            state = checkpoint['rng_states'][rank]
            random.setstate(state['python']); np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])
            rng_restored = True
        del checkpoint
    validate_step_budget(step, args.stop_step, args.total_steps, bool(args.resume))
    if args.total_steps > 4000:
        assert rng_restored, 'Continuation requires the saved RNG state for each rank'
        assert optimizer.state, 'Continuation requires AdamW moments, not just model weights'
    net = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=True)
    actor = MVLM_Actor(net, dict(giou=giou_loss, l1=torch.nn.functional.l1_loss, focal=FocalLoss()),
                       dict(giou=2., l1=5., focal=1.), settings, cfg)
    actor.train(True)
    assert not model.encoder.training and not model.text_encoder.training
    assert all(not m.training for m in model.head.modules() if isinstance(m, torch.nn.BatchNorm2d))
    args.out.mkdir(parents=True, exist_ok=True)
    initial_step = step
    memory_before = {k: p.detach().cpu().clone() for k, p in model.memory_fusion.named_parameters()}
    torch.cuda.reset_peak_memory_stats()
    start_audit = dict(rank=rank, step=step, target_steps=args.total_steps,
        resume_checkpoint=str(args.resume) if args.resume else None,
        rank_rng_restored=rng_restored, optimizer=optimizer_audit(optimizer),
        batch_per_rank=cfg.TRAIN.BATCH_SIZE, global_batch=cfg.TRAIN.BATCH_SIZE*dist.get_world_size(),
        clip_frames=cfg.DATA.SEARCH.NUMBER, frozen_sha256=original_digest, bn_frozen=True,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        groups=[dict(name=g['name'], peak_lr=g['peak_lr'], parameters=sum(p.numel() for p in g['params']))
                for g in optimizer.param_groups])
    write_json(args.out/('rank%d_start_audit.json'%rank), start_audit)
    write_json(args.out/'audits'/('rank%d_from%06d_start.json'%(rank, step)), start_audit)
    interval_stats, interval_count = {}, 0
    interval_start = time.monotonic()
    try:
        while step < args.stop_step:
            epoch, offset = divmod(step, len(loader))
            loader.sampler.set_epoch(epoch)
            for batch_index, data in enumerate(loader):
                if batch_index < offset:
                    continue
                set_step_lr(optimizer, step + 1)
                data = data.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, stats = actor(data)
                finite = torch.tensor(int(torch.isfinite(loss)), device=device)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError('Nonfinite loss at update %d' % (step+1))
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1., error_if_nonfinite=True)
                optimizer.step()
                step += 1
                stats['GradNorm'] = float(grad_norm)
                for key, value in stats.items():
                    interval_stats[key] = interval_stats.get(key, 0.) + float(value)
                interval_count += 1
                if step % 20 == 0 or step == args.stop_step:
                    torch.cuda.synchronize()
                    keys = sorted(interval_stats)
                    values = torch.tensor([interval_stats[k]/interval_count for k in keys], device=device, dtype=torch.float64)
                    dist.all_reduce(values); values /= dist.get_world_size()
                    seconds = time.monotonic() - interval_start
                    if rank == 0:
                        payload = dict(global_step=step, epoch=step/len(loader), updated_at=now(),
                            **{'train/'+k: v for k, v in zip(keys, values.cpu().tolist())})
                        payload.update({'lr/'+g['name']:g['lr'] for g in optimizer.param_groups})
                        payload.update(seconds_per_update=seconds/interval_count,
                            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                            allocated_mib=torch.cuda.memory_allocated()/2**20,
                            clips_per_second_global=cfg.TRAIN.BATCH_SIZE*dist.get_world_size()*interval_count/seconds)
                        with (args.out/'training_metrics.jsonl').open('a') as stream:
                            stream.write(json.dumps(payload)+'\n')
                        write_json(args.out/'train_status.json', dict(status='training', **payload))
                        print(json.dumps(payload), flush=True)
                    interval_stats, interval_count = {}, 0
                    interval_start = time.monotonic()
                if not args.probe and (step % 500 == 0 or step == args.stop_step):
                    save_checkpoint(model, optimizer, step, len(loader), args.out, cfg, args.total_steps)
                if step >= args.stop_step:
                    break
        assert_bn_unchanged(model, original_bn)
        assert frozen_digest(model) == original_digest, 'Frozen encoder/text changed'
        assert all(p.grad is None for p in model.encoder.parameters())
        changed = [k for k,p in model.memory_fusion.named_parameters()
                   if not torch.equal(p.detach().cpu(), memory_before[k])]
        new_channel_norms = {name:float(getattr(model.head,name)[0].weight[:,512:].norm())
                             for name in ('conv1_ctr','conv1_offset','conv1_size')}
        assert changed and all(v > 0 for v in new_channel_norms.values())
        audit = dict(status='passed', rank=rank, start_step=initial_step, final_step=step,
            target_steps=args.total_steps, optimizer=optimizer_audit(optimizer),
            frozen_encoder_text_unchanged=True, bn_statistics_unchanged=True,
            changed_memory_parameters=changed, new_channel_norms=new_channel_norms,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, completed_at=now())
        write_json(args.out/('rank%d_final_audit.json'%rank), audit)
        write_json(args.out/'audits'/('rank%d_from%06d_to%06d_final.json'%(rank, initial_step, step)), audit)
        if rank == 0:
            write_json(args.out/'train_status.json', dict(status='segment_completed', global_step=step, updated_at=now()))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stop-step', type=int, required=True)
    parser.add_argument('--total-steps', type=int, default=4000)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--probe', action='store_true')
    run(parser.parse_args())
