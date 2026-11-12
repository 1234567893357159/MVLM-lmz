"""Full visual/projection/CENTER/fusion adaptation from A with streaming backward."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from world_monitor import WorldMonitor
from world_distributed import WorldDistributed, RankClipSampler, trainable_digest

import _init_paths
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.config.mvlm.config import load_config
from memory_world_tracker_b import (build_tracker_stage_b, StreamingTrackerBActor,
    global_metrics, synchronize_gradients, module_gradient_norms, MEANS, COUNTS)
from lib.models.mvlm.joint_checkpoint import load_baseline, load_joint, joint_spec
from lib.train.dataset.world_clips import WorldClips, collate_world_clips
from world_manifest_fast import read_manifest

monitor = WorldMonitor()
parallel = WorldDistributed()


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
    parser.add_argument('--config', required=True, help='Path to a joint stage A, B or C YAML')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--world-checkpoint', help='Compatible joint stage-A checkpoint for stage B')
    parser.add_argument('--output', required=True, help='New directory, or existing directory with --resume')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--validation-clips', type=int, default=32)
    parser.add_argument('--wandb-project', help='Enable online W&B logging to this project')
    parser.add_argument('--wandb-entity', help='W&B user or team; defaults to the logged-in account')
    parser.add_argument('--wandb-name', help='Run display name, default: output directory name')
    parser.add_argument('--verify-distributed', action='store_true', help='Audit final replica equality and frozen parameters')
    parser.add_argument('--trial-data-cache', help='Reuse an audited cache ONLY for a <=3-step capacity smoke')
    parser.add_argument('--initial-validation', action='store_true', help='Measure the unchanged A parent on fixed validation clips')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.MODEL.TEMPORAL_BASELINE:
        parser.error('TEMPORAL_BASELINE is an evaluation-only geometry control')
    stage = cfg.TRAIN.WORLD.STAGE
    if not (cfg.MODEL.WORLD.ENABLED or cfg.MODEL.MEMORY.ENABLED) or stage not in ('A', 'B', 'C'):
        parser.error('An enabled temporal model and stage A/B/C config are required')
    if ((stage == 'B' and cfg.MODEL.WORLD.ENABLED and not cfg.TRAIN.WORLD.HEAD_ONLY) or stage == 'C') and not args.world_checkpoint and not args.resume:
        parser.error('Stage B needs stage-A weights; stage C needs adapted stage-B/C weights')
    if cfg.MODEL.PRETRAINED_MODEL is not None:
        parser.error('Use --baseline for complete, explicitly validated tracker weights')
    world = cfg.TRAIN.WORLD
    if stage != 'B' or not cfg.TRAIN.CM_LOSS or cfg.TRAIN.WORLD.HEAD_ONLY:
        parser.error('This entry requires full tracker B and the paper CM objective')
    if args.trial_data_cache and world.MAX_STEPS > 3:
        parser.error('Cached data is allowed only for <=3-step capacity smokes; formal runs re-audit')
    if not 0 <= world.PREDICTED_HISTORY_PROB <= 1 or world.CLIP_LENGTH <= cfg.MODEL.WORLD.HISTORY:
        parser.error('Invalid history probability or clip length')
    if min(world.MAX_STEPS, world.SAVE_INTERVAL, cfg.TRAIN.BATCH_SIZE, args.validation_clips) < 1:
        parser.error('Step counts, batch size and validation clips must be positive')
    if not 0 <= world.WARMUP_FRAMES < world.CLIP_LENGTH:
        parser.error('WARMUP_FRAMES must be nonnegative and smaller than CLIP_LENGTH')
    if cfg.TRAIN.NUM_WORKER < 0:
        parser.error('NUM_WORKER must be nonnegative')
    if world.BATCHED_ENCODER and stage != 'A':
        parser.error('BATCHED_ENCODER currently supports only stage A')
    if cfg.TRAIN.OPTIMIZER != 'ADAMW' or cfg.TRAIN.SCHEDULER.TYPE != 'cosine':
        parser.error('Joint training supports explicit AdamW and cosine step scheduling')
    device = parallel.setup(args.device)
    args.device = str(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training started. Use CPU only for smoke checks.')
    output = Path(args.output).resolve()
    def prepare_output():
        if args.resume:
            if not (output / 'last.pth.tar').is_file():
                raise ValueError('Resume requires last.pth.tar in the existing output directory')
        elif output.exists():
            raise FileExistsError('Use a fresh output directory; existing artifacts will not be overwritten')
        else:
            output.mkdir(parents=True)
    parallel.primary_call(prepare_output)
    random.seed(world.SEED)
    np.random.seed(world.SEED)
    torch.manual_seed(world.SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed(world.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    def prepare_data():
        if args.trial_data_cache:
            payload = torch.load(args.trial_data_cache, map_location='cpu')
            if payload['manifest_hash'] != sha256(args.manifest) or payload['baseline_hash'] != sha256(args.baseline):
                raise ValueError('Capacity smoke cache does not match the supplied files')
        else:
            partitions, manifest_hash = read_manifest(args.manifest, world.CLIP_LENGTH)
            payload = dict(partitions=partitions, manifest_hash=manifest_hash,
                data_signature=hashlib.sha256(json.dumps({
                    key: [seq['fingerprint'] for seq in sequences] for key, sequences in partitions.items()
                }, sort_keys=True).encode()).hexdigest(), baseline_hash=sha256(args.baseline))
        if parallel.size > 1:
            # Rank 0 audits current files on every launch/resume. Other ranks read
            # only this freshly generated shared cache, avoiding eight full scans.
            atomic_save(payload, output / 'data_cache.pth')
        return payload
    payload = parallel.primary_call(prepare_data)
    if not parallel.primary:
        payload = torch.load(str(output / 'data_cache.pth'), map_location='cpu')
    partitions, manifest_hash = payload['partitions'], payload['manifest_hash']
    data_signature, baseline_hash = payload['data_signature'], payload['baseline_hash']
    del payload
    # Build from the complete tracking checkpoint, without backbone-pretraining download.
    model = build_tracker_stage_b(cfg)
    load_baseline(model, torch.load(args.baseline, map_location='cpu'))
    if stage == 'A' and args.world_checkpoint:
        parser.error('Stage A initializes from --baseline; use --resume for interrupted training')
    if stage in ('B', 'C') and args.world_checkpoint and not args.resume:
        checkpoint = torch.load(args.world_checkpoint, map_location='cpu')
        expected_stages = ('A',) if stage == 'B' else ('B', 'C')
        if checkpoint.get('world_stage') not in expected_stages:
            raise ValueError('Wrong parent checkpoint stage')
        if checkpoint.get('manifest_sha256') != manifest_hash or checkpoint.get('baseline_sha256') != baseline_hash:
            raise ValueError('Joint stages must share the same data split and baseline weights')
        if checkpoint.get('data_signature') != data_signature:
            raise ValueError('Training annotations or resolved frame paths changed after stage A')
        if any(checkpoint['config'][section] != cfg[section] for section in ('MODEL', 'DATA', 'TEST')):
            raise ValueError('Stage A/B encoding/crop/confidence settings must match')
        load_joint(model, checkpoint)
    model.configure_world_stage(stage)
    model.to(device)
    optimizer = torch.optim.AdamW([
        dict(name='encoder_and_projection', params=list(model.encoder.parameters()) + list(model.text_encoder.text_proj.parameters()),
             lr=cfg.TRAIN.LR * cfg.TRAIN.ENCODER_MULTIPLIER),
        dict(name='head', params=model.head.parameters(), lr=world.HEAD_LR),
        dict(name='fusion', params=model.future_fusion.parameters(), lr=cfg.TRAIN.LR),
    ], weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, world.MAX_STEPS)
    actor = StreamingTrackerBActor(model, cfg)
    step, sample_index, best = 0, 0, float('inf')
    metadata = dict(world_stage=stage, joint_spec=joint_spec(model), config=cfg,
                    manifest_sha256=manifest_hash, baseline_sha256=baseline_hash, data_signature=data_signature,
                    world_size=parallel.size, global_batch_size=cfg.TRAIN.BATCH_SIZE * parallel.size,
                    world_source_sha256=sha256(args.world_checkpoint) if args.world_checkpoint else None,
                    batch_execution='full-tracker-b-streaming-frame-backward-fp32-v1')
    if args.resume:
        checkpoint = torch.load(str(output / 'last.pth.tar'), map_location='cpu')
        for key in ('world_stage', 'joint_spec', 'config', 'manifest_sha256', 'baseline_sha256', 'data_signature', 'batch_execution'):
            if checkpoint[key] != metadata[key]:
                raise ValueError('Resume metadata differs: ' + key)
        metadata['world_source_sha256'] = checkpoint.get('world_source_sha256')
        if checkpoint.get('world_size', 1) != parallel.size:
            raise ValueError('Exact resume requires the same world size')
        run_spec = json.loads((output / 'run.json').read_text())
        source_root = Path(__file__).resolve().parents[1]
        for relative, expected in run_spec['source_hashes'].items():
            if sha256(source_root / relative) != expected:
                raise ValueError('Source changed since training; cannot silently resume: ' + relative)
        load_joint(model, checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        step, sample_index, best = checkpoint['step'], checkpoint['sample_index'], checkpoint['best_clip_loss']
        rng = checkpoint.get('distributed_rng', [checkpoint])[parallel.rank]
        random.setstate(rng['python_rng'])
        np.random.set_state(rng['numpy_rng'])
        torch.set_rng_state(rng['torch_rng'])
        if device.type == 'cuda':
            if isinstance(rng['cuda_rng'], list):
                torch.cuda.set_rng_state_all(rng['cuda_rng'])
            else:
                torch.cuda.set_rng_state(rng['cuda_rng'], device)
    elif parallel.primary:
        (output / 'config.yaml').write_bytes(Path(args.config).read_bytes())
        (output / 'split.json').write_bytes(Path(args.manifest).read_bytes())
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(Path(__file__).resolve().parents[1]), text=True).strip()
        diff = subprocess.check_output(['git', 'diff', '--binary'], cwd=str(Path(__file__).resolve().parents[1]))
        (output / 'tracked_changes.patch').write_bytes(diff)
        source_paths = [Path(__file__).resolve().parents[1] / path for path in (
            'lib/models/mvlm', 'lib/train/actors/memory_world.py',
            'lib/train/dataset/world_clips.py', 'lib/test/tracker/mvlm.py', 'lib/test/tracker/utils.py',
            'lib/config/mvlm/config.py', 'tracking/train_memory_world.py', 'tracking/world_monitor.py',
            'tracking/world_distributed.py', 'tracking/train_memory_world_tracker_b.py', 'tracking/memory_world_tracker_b.py',
            'tracking/world_manifest_fast.py')]
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
            world_size=parallel.size, per_gpu_batch=cfg.TRAIN.BATCH_SIZE,
            global_batch_size=cfg.TRAIN.BATCH_SIZE * parallel.size,
            baseline_sha256=baseline_hash, source_hashes=source_hashes,
            train_videos=len(partitions['train']), val_videos=len(partitions['val']),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            batch_execution=metadata['batch_execution'], parent_checkpoint_step=checkpoint['step'],
            training_scope=['encoder', 'text_encoder.text_proj', 'head', 'future_fusion'],
            precision='fp32', gradient_reduction='sum local frame gradients / global supervised frames',
            capacity_smoke=bool(args.trial_data_cache),
            validation_protocol='fixed online-history clips; not full-sequence AUC'), indent=2))
    parallel.primary_call(lambda: monitor.start(args, cfg, step))
    if not args.resume:
        # Different stochastic histories per rank; model initialization used the
        # same seed. Clip indices are disjoint independently of these RNGs.
        random.seed(world.SEED + parallel.rank)
        np.random.seed(world.SEED + parallel.rank)
        torch.manual_seed(world.SEED + parallel.rank)
        if device.type == 'cuda':
            torch.cuda.manual_seed(world.SEED + parallel.rank)
    aliases = {'LASOT': 'lasot', 'VASTTRACK': 'vasttrack', 'TNL2K_train': 'tnl2k'}
    names, ratios = cfg.DATA.TRAIN.DATASETS_NAME, cfg.DATA.TRAIN.DATASETS_RATIO
    if len(names) != len(ratios) or len(set(names)) != len(names) or any(n not in aliases for n in names):
        raise ValueError('Joint training needs explicit LaSOT/VastTrack/TNL2K names and ratios')
    dataset_weights = {aliases[name]: ratio for name, ratio in zip(names, ratios)}
    if any(seq['dataset'] not in dataset_weights for seq in partitions['train'] + partitions['val']):
        raise ValueError('Manifest contains a dataset missing from the configured sampling ratios')
    # Reserve extra indices for explicitly skipped clips with no usable supervision.
    train = WorldClips(partitions['train'], world.CLIP_LENGTH,
                       world.MAX_STEPS * cfg.TRAIN.BATCH_SIZE * parallel.size * 4, world.SEED,
                       stride=world.MAX_STRIDE, dataset_weights=dataset_weights)
    validation = WorldClips(partitions['val'], world.CLIP_LENGTH,
                            args.validation_clips, world.SEED + 1, augment=False,
                            stride=world.MAX_STRIDE, dataset_weights=dataset_weights)
    loader_generator = torch.Generator().manual_seed(world.SEED)
    sampler = RankClipSampler(sample_index, len(train), cfg.TRAIN.BATCH_SIZE, parallel.rank, parallel.size)
    loader = DataLoader(train, batch_size=cfg.TRAIN.BATCH_SIZE,
                        sampler=sampler, num_workers=cfg.TRAIN.NUM_WORKER,
                        collate_fn=collate_world_clips, generator=loader_generator,
                        **({'multiprocessing_context': 'spawn'} if parallel.size > 1 and cfg.TRAIN.NUM_WORKER else {}),
                        **({'prefetch_factor': 1} if cfg.TRAIN.NUM_WORKER else {}))
    model.train()
    if args.verify_distributed:
        initial_digest = trainable_digest(model)
        frozen_versions = {name: p._version for name, p in model.named_parameters() if not p.requires_grad}
        first_indices = list(range(sample_index + parallel.rank * cfg.TRAIN.BATCH_SIZE,
                                   sample_index + (parallel.rank + 1) * cfg.TRAIN.BATCH_SIZE))
        initial_module_samples = {name: parameter.detach().flatten()[:128].clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad}
    if args.initial_validation and not args.resume:
        model.eval()
        rows = []
        with torch.no_grad():
            for i in range(parallel.rank, len(validation), parallel.size):
                rows.append(dict(clip_index=i, **actor([validation[i]])))
        rows = sorted([row for shard in parallel.gather(rows) for row in shard], key=lambda row: row['clip_index'])
        n = sum(row['supervised_frames'] for row in rows)
        val = dict(step=0, split='fixed_validation_clips', clips=len(rows),
                   usable_clips=sum(bool(row['supervised_frames']) for row in rows), rows=rows,
                   **{key: sum(row[key] * row['supervised_frames'] for row in rows) / max(1, n) for key in MEANS})
        val.update({key: sum(row[key] for row in rows) for key in COUNTS})
        def save_initial():
            monitor.validation(val)
            (output / 'validation_000000.json').write_text(json.dumps(val, indent=2))
            print(json.dumps({key: value for key, value in val.items() if key != 'rows'}), flush=True)
        parallel.primary_call(save_initial)
        model.train()
    skipped = 0
    for clips in loader:
        if step >= world.MAX_STEPS:
            break
        sample_index += len(clips) * parallel.size
        optimizer.zero_grad(set_to_none=True)
        parallel.sync_buffers(model)
        local_metrics = actor(clips, backward=lambda value: value.backward())
        metrics = global_metrics(local_metrics, device)
        supervised = metrics['supervised_frames']
        if not supervised:
            skipped += 1
            if skipped > 100:
                raise RuntimeError('Too many clips without supervision; inspect data and history validity')
            continue
        synchronize_gradients(model, supervised)
        module_norms = module_gradient_norms(model)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.TRAIN.GRAD_CLIP_NORM)
        if not parallel.all_true(torch.isfinite(norm).item()):
            raise FloatingPointError('Non-finite gradients')
        optimizer.step()
        scheduler.step()
        step += 1
        row = dict(step=step, split='train', skipped_clips=skipped,
                   world_size=parallel.size, global_batch_size=cfg.TRAIN.BATCH_SIZE * parallel.size, **metrics)
        row.update(module_norms)
        row.update({'learning_rate_' + group['name']: group['lr'] for group in optimizer.param_groups})
        if parallel.primary:
            monitor.train(row, optimizer, norm)
            with (output / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            if step == 1 or step % cfg.TRAIN.PRINT_INTERVAL == 0:
                print(json.dumps(row), flush=True)
        if step % world.SAVE_INTERVAL == 0 or step == world.MAX_STEPS:
            if stage in ('B', 'C'):
                # Validate the same running BN statistics that rank 0 saves.
                parallel.sync_buffers(model)
            model.eval()
            rows = []
            with torch.no_grad():
                for i in range(parallel.rank, len(validation), parallel.size):
                    values = actor([validation[i]])
                    rows.append(dict(clip_index=i, **values))
            rows = sorted([row for shard in parallel.gather(rows) for row in shard], key=lambda row: row['clip_index'])
            usable = [row for row in rows if row['supervised_frames']]
            val_loss = sum(row['loss'] * row['supervised_frames'] for row in usable) / max(1, sum(row['supervised_frames'] for row in usable)) if usable else None
            val = dict(step=step, split='fixed_validation_clips', loss=val_loss,
                       clips=len(rows), usable_clips=len(usable), rows=rows)
            n = sum(row['supervised_frames'] for row in usable)
            val.update({key: sum(row[key] * row['supervised_frames'] for row in usable) / max(1, n) for key in MEANS})
            val.update({key: sum(row[key] for row in rows) for key in COUNTS})
            improved = val_loss is not None and val_loss < best
            if improved:
                best = val_loss
            rng = dict(python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                       torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
            rngs = parallel.gather(rng)
            checkpoint = dict(metadata, step=step, sample_index=sample_index, best_clip_loss=best,
                              optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                              distributed_rng=rngs, **rng)
            checkpoint['net'] = {key: value for key, value in model.state_dict().items()
                                 if not key.startswith('text_encoder.clip.')}
            def save_checkpoint():
                monitor.validation(val)
                (output / ('validation_%06d.json' % step)).write_text(json.dumps(val, indent=2))
                atomic_save(checkpoint, output / ('step_%06d.pth.tar' % step))
                atomic_save(checkpoint, output / 'last.pth.tar')
                if improved:
                    atomic_save(checkpoint, output / 'best_clip_loss.pth.tar')
                print(json.dumps({key: value for key, value in val.items() if key != 'rows'}), flush=True)
            parallel.primary_call(save_checkpoint)
            if args.verify_distributed:
                digest = trainable_digest(model)
                digests = parallel.gather(digest)
                if len(set(digests)) != 1:
                    raise RuntimeError('Replica parameters differ after explicit gradient synchronization')
            model.train()
    if step < world.MAX_STEPS:
        raise RuntimeError('Input sampling budget exhausted before planned optimizer steps')
    if args.verify_distributed:
        digest = trainable_digest(model)
        audit = dict(rank=parallel.rank, device=str(device), step=step, first_clip_indices=first_indices,
            trainable_sha256=digest, trainable_updated=digest != initial_digest,
            frozen_parameters_unchanged=all(p._version == frozen_versions[name] and p.grad is None
                for name, p in model.named_parameters() if not p.requires_grad))
        audit['module_sample_delta_l2'] = {prefix: sum(float((parameter.detach().flatten()[:128]
            - initial_module_samples[name]).square().sum()) for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.requires_grad) ** .5
            for prefix in ('encoder.', 'text_encoder.text_proj.', 'head.', 'future_fusion.')}
        audits = parallel.gather(audit)
        if len({row['trainable_sha256'] for row in audits}) != 1 or not all(
                row['trainable_updated'] and row['frozen_parameters_unchanged'] for row in audits):
            raise RuntimeError('Distributed replica/freeze audit failed')
        parallel.primary_call(lambda: (output / 'distributed_audit.json').write_text(json.dumps(audits, indent=2)))
    if parallel.primary:
        print('Completed stage {}: {} optimizer steps on {} ranks; full-sequence evaluation is separate.'.format(stage, step, parallel.size))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        monitor.finish(error)
        raise
    else:
        monitor.finish()
    finally:
        parallel.close()
