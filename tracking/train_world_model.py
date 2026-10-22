"""Explicit stage A/B trainer. Does not download data or auto-start experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

import _init_paths
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.config.mvlm.config import load_config
from lib.models.mvlm import build_mvlm
from lib.models.mvlm.world_checkpoint import load_tracker_weights, load_predictor, world_spec
from lib.train.actors.world_model import WorldModelActor
from lib.train.dataset.world_clips import read_manifest, WorldClips, collate_world_clips


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, str(temporary))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Path to stage A or B YAML')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--world-checkpoint', help='Required stage-A checkpoint for stage B')
    parser.add_argument('--output', required=True, help='New directory, or existing directory with --resume')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--validation-clips', type=int, default=32)
    args = parser.parse_args()
    cfg = load_config(args.config)
    stage = cfg.TRAIN.WORLD.STAGE
    if not cfg.MODEL.WORLD.ENABLED or stage not in ('A', 'B'):
        parser.error('An enabled world model and stage A/B config are required')
    if stage == 'B' and not args.world_checkpoint and not args.resume:
        parser.error('Stage B requires --world-checkpoint from stage A')
    if cfg.MODEL.PRETRAINED_MODEL is not None:
        parser.error('Use --baseline for complete, explicitly validated tracker weights')
    world = cfg.TRAIN.WORLD
    if not 0 <= world.PREDICTED_HISTORY_PROB <= 1 or world.CLIP_LENGTH <= cfg.MODEL.WORLD.HISTORY:
        parser.error('Invalid history probability or clip length')
    if min(world.MAX_STEPS, world.SAVE_INTERVAL, cfg.TRAIN.BATCH_SIZE, args.validation_clips) < 1:
        parser.error('Step counts, batch size and validation clips must be positive')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training started. Use CPU only for smoke checks.')
    output = Path(args.output).resolve()
    if args.resume:
        if not (output / 'last.pth.tar').is_file():
            raise ValueError('Resume requires last.pth.tar in the existing output directory')
    elif output.exists():
        raise FileExistsError('Use a fresh output directory; existing artifacts will not be overwritten')
    random.seed(world.SEED)
    np.random.seed(world.SEED)
    torch.manual_seed(world.SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(world.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    partitions, manifest_hash = read_manifest(args.manifest, world.CLIP_LENGTH)
    data_signature = hashlib.sha256(json.dumps({
        key: [seq['fingerprint'] for seq in sequences] for key, sequences in partitions.items()
    }, sort_keys=True).encode()).hexdigest()
    baseline_hash = sha256(args.baseline)
    # Build from the complete tracking checkpoint, without backbone-pretraining download.
    model = build_mvlm(cfg, training=False)
    load_tracker_weights(model, torch.load(args.baseline, map_location='cpu'))
    if stage == 'B' and not args.resume:
        checkpoint = torch.load(args.world_checkpoint, map_location='cpu')
        if checkpoint.get('manifest_sha256') != manifest_hash or checkpoint.get('baseline_sha256') != baseline_hash:
            raise ValueError('Stage A/B must share the same data split and baseline weights')
        if checkpoint.get('data_signature') != data_signature:
            raise ValueError('Training annotations or resolved frame paths changed after stage A')
        if checkpoint['config']['DATA'] != cfg.DATA or checkpoint['config']['TEST'] != cfg.TEST:
            raise ValueError('Stage A/B encoding/crop/confidence settings must match')
        load_predictor(model, checkpoint)
    model.configure_world_stage(stage)
    model.to(device)
    optimizer = torch.optim.AdamW([
        dict(params=[p for name, p in model.named_parameters() if p.requires_grad and not name.startswith('head.')], lr=cfg.TRAIN.LR),
        dict(params=[p for name, p in model.named_parameters() if p.requires_grad and name.startswith('head.')], lr=world.HEAD_LR),
    ], weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, world.MAX_STEPS)
    actor = WorldModelActor(model, cfg, stage)
    step, sample_index, best = 0, 0, float('inf')
    metadata = dict(world_stage=stage, world_spec=world_spec(model), config=cfg,
                    manifest_sha256=manifest_hash, baseline_sha256=baseline_hash, data_signature=data_signature,
                    world_source_sha256=sha256(args.world_checkpoint) if args.world_checkpoint else None)
    if args.resume:
        checkpoint = torch.load(str(output / 'last.pth.tar'), map_location='cpu')
        for key in ('world_stage', 'world_spec', 'config', 'manifest_sha256', 'baseline_sha256', 'data_signature'):
            if checkpoint[key] != metadata[key]:
                raise ValueError('Resume metadata differs: ' + key)
        metadata['world_source_sha256'] = checkpoint.get('world_source_sha256')
        run_spec = json.loads((output / 'run.json').read_text())
        source_root = Path(__file__).resolve().parents[1]
        for relative, expected in run_spec['source_hashes'].items():
            if sha256(source_root / relative) != expected:
                raise ValueError('Source changed since training; cannot silently resume: ' + relative)
        if stage == 'A':
            load_predictor(model, checkpoint)
        else:
            load_tracker_weights(model, checkpoint, require_world=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        step, sample_index, best = checkpoint['step'], checkpoint['sample_index'], checkpoint['best_clip_loss']
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    else:
        output.mkdir(parents=True)
        (output / 'config.yaml').write_bytes(Path(args.config).read_bytes())
        (output / 'split.json').write_bytes(Path(args.manifest).read_bytes())
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(Path(__file__).resolve().parents[1]), text=True).strip()
        diff = subprocess.check_output(['git', 'diff', '--binary'], cwd=str(Path(__file__).resolve().parents[1]))
        (output / 'tracked_changes.patch').write_bytes(diff)
        source_paths = [Path(__file__).resolve().parents[1] / path for path in (
            'lib/models/mvlm', 'lib/train/actors/world_model.py',
            'lib/train/dataset/world_clips.py', 'lib/test/tracker/mvlm.py', 'lib/test/tracker/utils.py',
            'lib/config/mvlm/config.py', 'tracking/train_world_model.py')]
        import shutil
        source_hashes = {}
        root = Path(__file__).resolve().parents[1]
        for source in source_paths:
            for file in (source.rglob('*.py') if source.is_dir() else [source]):
                relative = file.relative_to(root)
                destination = output / 'source' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(file), str(destination))
                source_hashes[str(relative)] = sha256(file)
        (output / 'run.json').write_text(json.dumps(dict(
            stage=stage, git_revision=revision, manifest_sha256=manifest_hash,
            torch_version=torch.__version__, numpy_version=np.__version__, device=str(device),
            baseline_sha256=baseline_hash, source_hashes=source_hashes,
            train_videos=len(partitions['train']), val_videos=len(partitions['val']),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            validation_protocol='fixed online-history clips; not full-sequence AUC'), indent=2))
    # Reserve extra indices for explicitly skipped clips with no usable supervision.
    train = WorldClips(partitions['train'], world.CLIP_LENGTH,
                       world.MAX_STEPS * cfg.TRAIN.BATCH_SIZE * 4, world.SEED)
    validation = WorldClips(partitions['val'], world.CLIP_LENGTH,
                            args.validation_clips, world.SEED + 1, augment=False)
    loader_generator = torch.Generator().manual_seed(world.SEED)
    loader = DataLoader(train, batch_size=cfg.TRAIN.BATCH_SIZE,
                        sampler=range(sample_index, len(train)), num_workers=cfg.TRAIN.NUM_WORKER,
                        collate_fn=collate_world_clips, generator=loader_generator)
    model.train()
    skipped = 0
    for clips in loader:
        if step >= world.MAX_STEPS:
            break
        sample_index += len(clips)
        optimizer.zero_grad(set_to_none=True)
        actor.step = step
        loss, metrics = actor(clips)
        if loss is None:
            skipped += 1
            if skipped > 100:
                raise RuntimeError('Too many clips without supervision; inspect data and history validity')
            continue
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.TRAIN.GRAD_CLIP_NORM)
        if not torch.isfinite(norm):
            raise FloatingPointError('Non-finite gradients')
        optimizer.step()
        scheduler.step()
        step += 1
        row = dict(step=step, split='train', skipped_clips=skipped, **metrics)
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        if step == 1 or step % cfg.TRAIN.PRINT_INTERVAL == 0:
            print(json.dumps(row), flush=True)
        if step % world.SAVE_INTERVAL == 0 or step == world.MAX_STEPS:
            model.eval()
            rows = []
            with torch.no_grad():
                for i in range(len(validation)):
                    _, values = actor([validation[i]], validation=True)
                    rows.append(values)
            usable = [row for row in rows if row['supervised_frames']]
            val_loss = sum(row['loss'] * row['supervised_frames'] for row in usable) / max(1, sum(row['supervised_frames'] for row in usable)) if usable else None
            val = dict(step=step, split='fixed_validation_clips', loss=val_loss,
                       clips=len(rows), usable_clips=len(usable), rows=rows)
            (output / ('validation_%06d.json' % step)).write_text(json.dumps(val, indent=2))
            improved = val_loss is not None and val_loss < best
            if improved:
                best = val_loss
            checkpoint = dict(metadata, step=step, sample_index=sample_index, best_clip_loss=best,
                              optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                              python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                              torch_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
            if stage == 'A':
                checkpoint['world_model'] = model.world_model.state_dict()
            else:
                checkpoint['net'] = {key: value for key, value in model.state_dict().items()
                                     if not key.startswith('text_encoder.clip.')}
            atomic_save(checkpoint, output / ('step_%06d.pth.tar' % step))
            atomic_save(checkpoint, output / 'last.pth.tar')
            if improved:
                atomic_save(checkpoint, output / 'best_clip_loss.pth.tar')
            print(json.dumps({key: value for key, value in val.items() if key != 'rows'}), flush=True)
            model.train()
    if step < world.MAX_STEPS:
        raise RuntimeError('Input sampling budget exhausted before planned optimizer steps')
    print('Completed stage {}: {} optimizer steps; full-sequence evaluation is separate.'.format(stage, step))


if __name__ == '__main__':
    main()
