"""Evaluate an MVLM checkpoint on the frozen VastTrack holdout without GT initialization."""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lib.local_config import EnvSettings
from lib.test.evaluation.data import Sequence, SequenceList
from lib.test.evaluation.running import run_dataset
from lib.test.analysis.extract_results import calc_err_center, calc_iou_overlap
from tracking.strict_language_eval import StrictLanguageTracker

# Use the versioned copy; the bytes/hash match the original frozen run split.
SPLIT = ROOT / 'reports/reproduction/vasttrack_split_20261026/val.txt'
EXPECTED_SHA256 = '90457546060ba767d57fa93be6c1a8f2bdb619ea960ec2a3fe11ccd5d4f05e4f'


def load_holdout():
    raw = SPLIT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise ValueError('Frozen validation split hash does not match')
    names = raw.decode().splitlines()
    if len(names) != 500 or len(set(names)) != 500:
        raise ValueError('Expected 500 unique validation sequences')
    base = Path(EnvSettings().vasttrack_dir) / 'train'
    dataset = SequenceList()
    for name in names:
        cls = name.rsplit('-', 1)[0]
        folder = base / cls / name
        gt = np.loadtxt(folder / 'Groundtruth.txt', delimiter=',', ndmin=2)
        if gt.shape[1] != 4 or not np.isfinite(gt).all():
            raise ValueError('Invalid bounding boxes: ' + name)
        language = (folder / 'nlp.txt').read_text().strip().lower()
        if not language:
            raise ValueError('Empty language query: ' + name)
        frames = [str(folder / 'imgs' / ('%05d.jpg' % i)) for i in range(1, len(gt)+1)]
        if len(frames) < 2 or any(not Path(frame).is_file() for frame in frames):
            raise ValueError('Missing or insufficient images: ' + name)
        visible = (gt[:, 2:] > 0).all(axis=1)
        if not visible.any():
            raise ValueError('No valid target boxes: ' + name)
        dataset.append(Sequence(name, frames, 'vasttrack', gt,
                                object_class=cls, target_visible=visible, language_query=language))
    return dataset


def calculate(dataset, tracker, folder):
    rows, errors = [], []
    thresholds = torch.arange(0., 1.01, .05, dtype=torch.float64)
    for seq in dataset:
        base = Path(tracker.results_dir) / seq.dataset / seq.name
        try:
            boxes = np.loadtxt(str(base) + '.txt', delimiter='\t', ndmin=2)
            times = np.loadtxt(str(base) + '_time.txt', ndmin=1)
            gt = np.asarray(seq.ground_truth_rect)
            if boxes.shape != (len(seq.frames), 4) or times.shape != (len(seq.frames),):
                raise ValueError('Prediction or time length mismatch')
            if not np.isfinite(boxes).all() or not (boxes[:, 2:] > 0).all():
                raise ValueError('Invalid predicted boxes')
            if not np.isfinite(times).all() or not (times >= 0).all():
                raise ValueError('Invalid frame times')
            pred, anno = torch.as_tensor(boxes, dtype=torch.float64), torch.as_tensor(gt, dtype=torch.float64)
            valid = torch.as_tensor(seq.target_visible).bool()
            pred, anno = pred[valid], anno[valid]
            overlap = calc_iou_overlap(pred, anno)
            center = calc_err_center(pred, anno)
            normalized = calc_err_center(pred, anno, normalized=True)
            rows.append({'sequence': seq.name, 'frames': len(seq.frames), 'valid_frames': int(valid.sum()),
                         'AUC': (overlap[:, None] > thresholds[None, :]).double().mean(0).mean().item()*100,
                         'PRE': (center <= 20).double().mean().item()*100,
                         'NPR': (normalized <= .20).double().mean().item()*100,
                         'first_frame_prediction': boxes[0].tolist()})
        except Exception as exc:
            errors.append({'sequence': seq.name, 'error': repr(exc)})
    (folder/'per_sequence.json').write_text(json.dumps(rows, indent=2))
    (folder/'prediction_validation.json').write_text(json.dumps({'expected':len(dataset), 'validated':len(rows), 'errors':errors},indent=2))
    if errors:
        raise RuntimeError('Missing or invalid results: ' + repr(errors[:3]))
    return {'metrics': {key: float(np.mean([row[key] for row in rows])) for key in ('AUC','PRE','NPR')},
            'sequences': len(rows), 'frames': sum(r['frames'] for r in rows),
            'valid_frames': sum(r['valid_frames'] for r in rows)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--weights',required=True)
    parser.add_argument('--exp-id',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--threads',type=int,default=16);parser.add_argument('--gpus',type=int,default=8)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args();folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
    status={'status':'starting','dataset':'vasttrack_holdout','weights':args.weights,'config':args.config,
            'validation_split_sha256':EXPECTED_SHA256,'started_at':time.time(),
            'protocol':'language-only; first frame predicted; no ground-truth initialization',
            'metric_policy':'equal sequence weighting, exclude invalid GT frames; AUC thresholds 0:0.05:1; PRE@20px; NPR@0.20'}
    def save():
        tmp=folder/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(folder/'status.json')
    save()
    try:
        dataset=load_holdout()
        if args.limit:dataset=dataset[:args.limit]
        status.update(status='running',sequences=len(dataset));save()
        tracker=StrictLanguageTracker('mvlm',args.config,args.exp_id,'vasttrack',args.weights)
        run_dataset(dataset,[tracker],debug=False,threads=args.threads,num_gpus=args.gpus,parallel_mode='mp')
        status.update(calculate(dataset,tracker,folder),status='completed',completed_at=time.time());save()
        print('RESULT',json.dumps(status),flush=True)
    except BaseException as exc:
        status.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),completed_at=time.time());save()
        raise

if __name__=='__main__':main()
