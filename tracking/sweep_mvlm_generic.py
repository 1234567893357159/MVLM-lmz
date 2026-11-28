"""Configurable full-video MVLM parameter sweep; see reports/reproduction/GENERIC_SWEEP.md."""
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import _init_paths
import cv2
import numpy as np
import torch
import yaml
from easydict import EasyDict

from lib.config.mvlm import config as config_module
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.evaluation import get_dataset
from lib.test.evaluation.data import Sequence
from lib.test.tracker.mvlm import MVLM
from lib.test.utils import TrackerParams

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    if hasattr(config_module, 'load_config'):
        return config_module.load_config(str(path))
    cfg = copy.deepcopy(config_module.cfg)
    with open(path) as f:
        config_module._update_config(cfg, EasyDict(yaml.safe_load(f)))
    return cfg


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError('Not JSON serializable: ' + type(value).__name__)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False, default=convert)


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def path_at_root(path):
    path = Path(path).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def safe_name(name):
    if not isinstance(name, str) or not name or name in ('.', '..') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in name):
        raise ValueError('Unsafe name: ' + repr(name))
    return name


def apply_overrides(cfg, overrides):
    for path, value in overrides.items():
        keys = path.split('.')
        memory = {'MIN_QUALITY', 'NOVELTY_THRESHOLD', 'SHORT_DECAY', 'LONG_DECAY', 'WRITE_CONSISTENCY'}
        world = {'QUALITY_WEIGHT', 'FUSION_STRENGTH', 'MIN_QUALITY'}
        allowed = (len(keys) == 3 and (keys[:2] == ['TEST', 'MVLM'] or
                   (keys[:2] == ['MODEL', 'MEMORY'] and keys[2] in memory) or
                   (keys[:2] == ['MODEL', 'WORLD'] and keys[2] in world)))
        allowed = allowed or (len(keys) == 4 and keys[:2] == ['MODEL', 'MEMORY'] and
                              keys[2] in ('POLICY', 'RETENTION', 'FUSION'))
        if not allowed:
            raise ValueError('Unsupported inference parameter: ' + path)
        node = cfg
        try:
            for key in keys[:-1]:
                node = node[key]
        except KeyError as exc:
            raise ValueError('Parameter not available in this model: ' + path) from exc
        if keys[-1] not in node or isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError('Invalid parameter: ' + path)
        if isinstance(node[keys[-1]], int) and not isinstance(value, int):
            raise ValueError('Integer parameter required: ' + path)
        node[keys[-1]] = value


def sequence_hash(seq):
    h = hashlib.sha256()
    h.update(canonical([seq.dataset, seq.name, list(seq.frames), seq.init_info()]).encode())
    h.update(np.asarray(seq.ground_truth_rect).tobytes())
    if seq.target_visible is not None:
        h.update(np.asarray(seq.target_visible).tobytes())
    return h.hexdigest()


def select_sequences(spec):
    kind = spec['kind']
    if kind == 'benchmark':
        sequences = list(get_dataset(*spec['datasets']))
        provenance = {'kind': kind, 'selection_scope': 'standard_test_exploratory'}
    elif kind == 'heldout':
        try:
            from lib.train.dataset.world_clips import describe_sequence
        except ImportError as exc:
            raise ValueError('This branch has no world_clips heldout reader; use benchmark data or add a branch-specific reader') from exc
        manifest_path = path_at_root(spec['manifest'])
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('protocol') != 'official-train-video-split-v1':
            raise ValueError('Unsupported heldout manifest')
        sequences = []
        for row in manifest['val']:
            if spec.get('datasets') and row['dataset'] not in spec['datasets']:
                continue
            if spec.get('sequences') is not None and row['dataset'] + '/' + row['name'] not in spec['sequences']:
                continue
            seq = describe_sequence(row['dataset'], manifest['roots'][row['dataset']], row['name'])
            if row.get('fingerprint', seq['fingerprint']) != seq['fingerprint']:
                raise ValueError('Manifest fingerprint mismatch: ' + row['name'])
            sequences.append(Sequence(seq['name'], seq['frames'], seq['dataset'], seq['boxes'],
                                      target_visible=seq['visible'] if seq['dataset'] == 'lasot' else None,
                                      language_query=seq['language']))
        provenance = {'kind': kind, 'selection_scope': 'heldout_validation',
                      'manifest': str(manifest_path), 'manifest_sha256': digest(manifest_path),
                      'audit': 'selected val videos: annotation, language, visibility and frame paths; no image-content hashes or train-split audit'}
    else:
        raise ValueError('Unknown dataset kind: ' + str(kind))
    wanted = spec.get('sequences')
    if wanted is not None:
        if not isinstance(wanted, list) or not wanted or len(set(wanted)) != len(wanted):
            raise ValueError('sequences must be a nonempty list of unique dataset/name strings')
        sequences = [s for s in sequences if s.dataset + '/' + s.name in set(wanted)]
        if {s.dataset + '/' + s.name for s in sequences} != set(wanted):
            raise ValueError('Some requested sequences are absent')
    keys = [s.dataset + '/' + s.name for s in sequences]
    if not sequences or len(set(keys)) != len(keys):
        raise ValueError('Empty or duplicate sequence selection')
    for s in sequences:
        safe_name(s.dataset)
        safe_name(s.name)
        if len(s.frames) < 2 or len(s.frames) != len(s.ground_truth_rect):
            raise ValueError('Invalid frame/GT count: ' + s.name)
    return sorted(sequences, key=lambda s: (s.dataset, s.name)), provenance


def metric(boxes, seq):
    raw = np.asarray(boxes, dtype=np.float64)
    if raw.shape != (len(seq.frames), 4) or not np.isfinite(raw).all() or not (raw[:, 2:] > 0).all():
        raise ValueError('Invalid prediction: ' + seq.name)
    integer = raw.astype(np.int64)
    if not (integer[:, 2:] > 0).all():
        raise ValueError('Integer conversion produced invalid box: ' + seq.name)
    overlap, center, norm, _ = calc_seq_err_robust(torch.as_tensor(integer, dtype=torch.float64),
        torch.as_tensor(np.asarray(seq.ground_truth_rect), dtype=torch.float64), seq.dataset,
        torch.as_tensor(seq.target_visible) if seq.target_visible is not None else None)
    thresholds = torch.arange(0., 1.01, .05, dtype=torch.float64)
    return integer, {'AUC': float((overlap[:, None] > thresholds).double().mean() * 100),
                     'PRE': float((center <= 20).double().mean() * 100),
                     'NPR': float((norm <= .2).double().mean() * 100)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, help='JSON experiment specification')
    parser.add_argument('--output', required=True, help='New output directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan', action='store_true', help='Audit data and print plan without creating output or loading models')
    parser.add_argument('--smoke-frames', type=int, help='Diagnostic only: truncate each video to this many frames')
    args = parser.parse_args()
    spec = json.loads(path_at_root(args.spec).read_text())
    models, variants = spec['models'], spec['variants']
    if not models or not variants:
        raise ValueError('At least one model and variant are required')
    for name in list(models) + list(variants):
        safe_name(name)
    if len(set(models)) != len(models) or len(set(variants)) != len(variants):
        raise ValueError('Duplicate name')
    seqs, provenance = select_sequences(spec['data'])
    if args.smoke_frames is not None:
        if args.smoke_frames < 2:
            raise ValueError('Smoke test requires at least two frames')
        for seq in seqs:
            seq.frames = seq.frames[:args.smoke_frames]
            seq.ground_truth_rect = seq.ground_truth_rect[:args.smoke_frames]
            if seq.target_visible is not None:
                seq.target_visible = seq.target_visible[:args.smoke_frames]
        provenance['smoke_frames'] = args.smoke_frames
        provenance['selection_scope'] = 'diagnostic_partial_video_not_evaluation'
    model_info = {}
    for name, model in models.items():
        cfg_path, checkpoint = path_at_root(model['config']), path_at_root(model['checkpoint'])
        cfg = load_config(str(cfg_path))
        for values in variants.values():
            apply_overrides(load_config(str(cfg_path)), values)
        model_info[name] = {'config': str(cfg_path), 'config_sha256': digest(cfg_path),
                            'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint)}
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--binary', '--', 'lib', 'tracking', 'experiments'], cwd=ROOT)
    identity = {'spec': spec, 'models': model_info, 'data': provenance,
                'sequences': [{'dataset': s.dataset, 'name': s.name, 'frames': len(s.frames), 'sha256': sequence_hash(s)} for s in seqs],
                'git_head': revision, 'tracked_code_diff_sha256': hashlib.sha256(diff).hexdigest(),
                'sweep_sha256': digest(__file__), 'device': args.device,
                'protocol': 'first frame GT; {} videos; integer xywh; success AUC 21 thresholds, PRE@20, NPR@0.2; video mean per dataset'.format('partial diagnostic' if args.smoke_frames else 'full'),
                'seed': int(spec.get('seed', 0)), 'torch_version': torch.__version__}
    print(json.dumps({'sequences': len(seqs), 'frames_per_variant_model': sum(len(s.frames) for s in seqs),
                      'models': list(models), 'variants': list(variants), 'video_evaluations': len(seqs)*len(models)*len(variants),
                      'data': provenance}, indent=2), flush=True)
    if args.plan:
        return
    out = path_at_root(args.output)
    if args.resume:
        if not (out / 'manifest.json').is_file():
            raise ValueError('Resume requires an existing manifest')
    else:
        out.mkdir(parents=True, exist_ok=False)
    with (out / 'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = out / 'manifest.json'
        if args.resume:
            if canonical(json.loads(manifest.read_text())['identity']) != canonical(identity):
                raise ValueError('Cannot resume: experiment identity changed')
        else:
            write(manifest, {'identity': identity, 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        seed = identity['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rows = []
        try:
            for model_name, model in model_info.items():
                for variant_name, overrides in variants.items():
                    cfg = load_config(model['config'])
                    apply_overrides(cfg, overrides)
                    params = TrackerParams()
                    params.cfg, params.checkpoint, params.debug, params.device = cfg, model['checkpoint'], 0, args.device
                    params.template_factor, params.template_size = cfg.TEST.TEMPLATE_FACTOR, cfg.TEST.TEMPLATE_SIZE
                    params.search_factor, params.search_size = cfg.TEST.SEARCH_FACTOR, cfg.TEST.SEARCH_SIZE
                    tracker, current_dataset = None, None
                    for seq in seqs:
                        dest = out / 'results' / model_name / variant_name / seq.dataset / (seq.name + '.json')
                        if dest.exists():
                            row = json.loads(dest.read_text())
                            if row.get('dataset') != seq.dataset or row.get('sequence') != seq.name or row.get('frames') != len(seq.frames):
                                raise ValueError('Invalid existing result: ' + str(dest))
                        else:
                            if current_dataset != seq.dataset:
                                del tracker
                                tracker = MVLM(params, seq.dataset)
                                current_dataset = seq.dataset
                            start = time.monotonic()
                            image = cv2.imread(seq.frames[0])
                            if image is None: raise ValueError('Cannot decode ' + seq.frames[0])
                            init_info = seq.init_info()
                            init_info['seq_name'] = seq.name
                            tracker.initialize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), init_info)
                            boxes = [list(seq.ground_truth_rect[0])]
                            for frame in seq.frames[1:]:
                                image = cv2.imread(frame)
                                if image is None: raise ValueError('Cannot decode ' + frame)
                                boxes.append(tracker.track(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))['target_bbox'])
                            boxes, metrics = metric(boxes, seq)
                            row = {'dataset': seq.dataset, 'sequence': seq.name, 'frames': len(seq.frames),
                                   'metrics': metrics, 'boxes': boxes.tolist(), 'wall_seconds_including_decode': time.monotonic()-start}
                            write(dest, row)
                        rows.append((model_name, variant_name, row))
                        write(out / 'status.json', {'status': 'running', 'completed': len(rows), 'total': len(seqs)*len(models)*len(variants),
                                                   'model': model_name, 'variant': variant_name, 'sequence': seq.dataset+'/'+seq.name})
                    del tracker
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            summary = {}
            for model_name in models:
                summary[model_name] = {}
                for variant_name in variants:
                    group = [r for m, v, r in rows if m == model_name and v == variant_name]
                    per = {}
                    for ds in sorted({r['dataset'] for r in group}):
                        subset = [r for r in group if r['dataset'] == ds]
                        per[ds] = {'sequences': len(subset), **{k: float(np.mean([r['metrics'][k] for r in subset])) for k in ('AUC', 'PRE', 'NPR')}}
                    summary[model_name][variant_name] = {'per_dataset': per, 'macro_AUC': float(np.mean([x['AUC'] for x in per.values()]))}
            write(out / 'summary.json', summary)
            write(out / 'status.json', {'status': 'completed', 'completed': len(rows), 'total': len(rows)})
        except BaseException as exc:
            write(out / 'status.json', {'status': 'failed', 'completed': len(rows), 'error': repr(exc)})
            raise


if __name__ == '__main__':
    main()
