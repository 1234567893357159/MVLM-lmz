"""Strict first-frame prediction with independent annotation scoring."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from lib.config.mvlm.config import load_config
from lib.models.mvlm.tracking_core import TrackingCore
from lib.test.analysis.extract_results import calc_err_center, calc_iou_overlap
from lib.train.data.reliable_memory_dataset import read_manifest
from reliable_memory_runtime import (atomic_json, sha256, build_loaded_model, source_hashes,
                                     seed_all, configure_group)


def score_predictions(boxes, sequence):
    # This function is called after predictions are complete. No scorer inputs
    # are passed into TrackingCore, including visibility or frame-zero GT.
    pred = torch.as_tensor(boxes, dtype=torch.float64)
    truth = torch.as_tensor(sequence['boxes'][:len(boxes)], dtype=torch.float64)
    valid = torch.isfinite(truth).all(1) & (truth[:, 2:] > 0).all(1)
    valid &= torch.isfinite(pred).all(1) & (pred[:, 2:] > 0).all(1)
    if sequence['dataset'] == 'lasot':
        valid &= torch.as_tensor(sequence['visible'][:len(boxes)]).bool()
    # Use repository geometry, but never its frame-zero GT substitution or
    # previous-frame filling. Invalid/invisible/failed frames fail all metrics.
    overlap, center = torch.full((len(boxes),), -1., dtype=torch.float64), torch.full((len(boxes),), float('inf'), dtype=torch.float64)
    norm = center.clone()
    overlap[valid] = calc_iou_overlap(pred[valid], truth[valid])
    center[valid] = calc_err_center(pred[valid], truth[valid])
    norm[valid] = calc_err_center(pred[valid], truth[valid], normalized=True)
    thresholds = torch.arange(0, 1.01, .05, dtype=torch.float64)
    return dict(AUC=float((overlap[:, None] > thresholds).double().mean()*100),
                PRE=float((center <= 20).double().mean()*100),
                NPR=float((norm <= .2).double().mean()*100), first_frame_IoU=float(overlap[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='mvlm_TF_reliable_memory')
    parser.add_argument('--checkpoint', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest')
    source.add_argument('--dataset', choices=['otb99_lang', 'lasot', 'tnl2k'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--protocol', default='strict_language', choices=['strict_language'])
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--allow-untrained-memory', action='store_true', help='Engineering only')
    parser.add_argument('--baseline', action='store_true', help='Disable content memory with these weights')
    parser.add_argument('--group', default='M6', choices=['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.max_sequences, args.max_frames) < 0:
        parser.error('Limits cannot be negative')
    cfg = load_config(args.config)
    configure_group(cfg, args.group)
    if cfg.TEST.FIRST_FRAME_PROTOCOL != 'strict_language':
        parser.error('Require the strict-language config')
    if args.baseline:
        cfg.MODEL.MEMORY.ENABLED = False
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    seed_all(20261116)
    output = Path(args.out).resolve()
    if args.resume:
        if not (output/'manifest.json').is_file():
            parser.error('Resume requires the existing manifest.json')
    else:
        output.mkdir(parents=True, exist_ok=False)
    metadata = dict(config=cfg, checkpoint_sha256=sha256(args.checkpoint),
                    dataset=args.dataset, manifest_sha256=sha256(args.manifest) if args.manifest else None,
                    protocol=args.protocol, max_sequences=args.max_sequences, max_frames=args.max_frames,
                    allow_untrained_memory=args.allow_untrained_memory,
                    engineering_only=bool(args.max_sequences or args.max_frames or args.allow_untrained_memory),
                    baseline=args.baseline, group=args.group, source_hashes=source_hashes(),
                    metrics_protocol='strict-v1: float boxes; actual frame zero; invalid/invisible/failed frames count failure in AUC/PRE/NPR',
                    evaluation_source_sha256=sha256(__file__))
    if args.resume:
        import json
        if json.loads((output/'manifest.json').read_text()) != metadata:
            raise ValueError('Evaluation inputs/config/code changed')
    else:
        atomic_json(output/'manifest.json', metadata)
    try:
        if args.manifest:
            partitions, _ = read_manifest(args.manifest, 1, args.max_sequences)
            sequences = partitions['val']
            if args.max_sequences:
                # Interleave sources for a small engineering run, rather than
                # accidentally taking only the first source in manifest order.
                groups = defaultdict(list)
                for seq in sequences:
                    groups[seq['dataset']].append(seq)
                sequences = [groups[ds][i] for i in range(max(map(len, groups.values())))
                             for ds in sorted(groups) if i < len(groups[ds])]
        else:
            from lib.test.evaluation import get_dataset
            sequences = [dict(dataset=seq.dataset, name=seq.name, frames=seq.frames,
                              language=seq.language_query, boxes=seq.ground_truth_rect,
                              visible=seq.target_visible if seq.target_visible is not None
                              else np.ones(len(seq.frames), dtype=bool))
                         for seq in get_dataset(args.dataset)]
        if args.max_sequences:
            sequences = sequences[:args.max_sequences]
        if not sequences:
            raise ValueError('Empty evaluation')
        # Missing language is explicit failure, never silent 700 -> 699 filtering.
        for seq in sequences:
            if not isinstance(seq['language'], str) or not seq['language'].strip():
                raise ValueError('Empty language: '+seq['name'])
        model, _, audit = build_loaded_model(cfg, args.checkpoint, device,
                                            allow_untrained=args.allow_untrained_memory)
        atomic_json(output/'loading_audit.json', audit)
        rows = []
        for seq in sequences:
            destination = output/'results'/seq['dataset']/(seq['name']+'.json')
            if destination.exists():
                import json
                rows.append(json.loads(destination.read_text()))
                continue
            core = TrackingCore(model, cfg, device)
            core.reset(seq['language'], seq['dataset']+'/'+seq['name'])
            paths = seq['frames'][:args.max_frames] if args.max_frames else seq['frames']
            predictions, diagnostics = [], []
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.monotonic()
            with torch.no_grad():
                for frame_id, path in enumerate(paths):
                    image = cv2.imread(str(path))
                    if image is None:
                        raise ValueError('Cannot decode '+str(path))
                    _, result = core.step(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), frame_id)
                    predictions.append(result['target_bbox'])
                    diagnostics.append(result['diagnostics'])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.monotonic()-start
            # Preserve floats: integer rounding is separately labelled legacy behavior.
            boxes = np.asarray(predictions, dtype=np.float64)
            if boxes.shape != (len(paths), 4) or not np.isfinite(boxes).all():
                raise ValueError('Invalid/incomplete predictions')
            metrics = score_predictions(boxes, seq)
            actions = Counter(row['write']['action'] for row in diagnostics)
            row = dict(dataset=seq['dataset'], name=seq['name'], frames=len(paths), metrics=metrics,
                       seconds=elapsed, timing='RGB decode + core; language/reset excluded; GPU synchronized',
                       write_counts=dict(actions), boxes=boxes.tolist(), diagnostics=diagnostics)
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(destination, row)
            rows.append(row)
            atomic_json(output/'status.json', dict(status='running', completed=len(rows), total=len(sequences)))
            print(seq['name'], metrics, dict(actions), flush=True)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['dataset']].append(row)
        metrics = {ds: {key: float(np.mean([r['metrics'][key] for r in values]))
                         for key in ['AUC', 'PRE', 'NPR', 'first_frame_IoU']}
                   for ds, values in grouped.items()}
        atomic_json(output/'summary.json', dict(metrics=metrics, sequences=len(rows),
                    engineering_only=metadata['engineering_only'], protocol=args.protocol,
                    mean_AUC=float(np.mean([value['AUC'] for value in metrics.values()])),
                    note='Subset/untrained runs establish execution only, not model quality.'))
        atomic_json(output/'status.json', dict(status='completed', completed=len(rows), total=len(sequences)))
    except BaseException as error:
        atomic_json(output/'status.json', dict(status='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
