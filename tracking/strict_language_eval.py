"""Evaluate MVLM with a predicted first frame and no ground-truth initialization."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lib.test.evaluation import get_dataset
from lib.test.evaluation.data import SequenceList
from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.running import run_dataset
from lib.test.analysis.extract_results import calc_err_center, calc_iou_overlap


class StrictLanguageTracker(Tracker):
    def _track_sequence(self, tracker, seq, init_info):
        if not tracker.template_free or not tracker.multi_modal_language or not tracker.use_nlp:
            raise ValueError('Strict language-only requires template-free model and language')
        if not str(init_info.get('init_nlp', '')).strip():
            raise ValueError('Missing language query')
        # The original wrapper passes GT and saves it as the first prediction.
        # Template-free inference ignores the template; initialize with a full-image
        # placeholder and infer frame 0 before processing the remaining frames.
        original = tracker.initialize
        def initialize_first_frame(image, info):
            safe_info = dict(info)
            safe_info['init_bbox'] = [0, 0, image.shape[1], image.shape[0]]
            safe_info.pop('gt_bbox', None)
            original(image, safe_info)
            return tracker.track(image, {})
        tracker.initialize = initialize_first_frame
        safe_info = dict(init_info)
        safe_info.pop('init_bbox', None)
        return super()._track_sequence(tracker, seq, safe_info)


def load_sequences(dataset_name):
    dataset = get_dataset(dataset_name)
    excluded = [{'sequence': seq.name, 'reason': 'empty language query'}
                for seq in dataset if not str(getattr(seq, 'language_query', '') or '').strip()]
    usable = SequenceList([seq for seq in dataset
                           if str(getattr(seq, 'language_query', '') or '').strip()])
    return usable, excluded


def calculate(dataset, tracker, folder):
    rows, errors = [], []
    overlap_thresholds = torch.arange(0., 1.01, .05, dtype=torch.float64)
    for seq in dataset:
        base = Path(tracker.results_dir) / seq.dataset / seq.name
        try:
            boxes = np.loadtxt(str(base) + '.txt', delimiter='\t', ndmin=2)
            times = np.loadtxt(str(base) + '_time.txt', ndmin=1)
            gt = np.asarray(seq.ground_truth_rect)
            assert boxes.shape == (len(seq.frames), 4), ('boxes', boxes.shape, len(seq.frames))
            assert len(gt) == len(seq.frames), ('annotations', len(gt), len(seq.frames))
            assert times.shape == (len(seq.frames),) and np.isfinite(times).all() and (times >= 0).all()
            assert np.isfinite(boxes).all() and (boxes[:,2:] > 0).all()
            pred = torch.as_tensor(boxes, dtype=torch.float64)
            anno = torch.as_tensor(gt, dtype=torch.float64)
            valid = (anno[:,2:] > 0).all(1)
            if seq.target_visible is not None:
                valid &= torch.as_tensor(seq.target_visible).bool()
            iou = calc_iou_overlap(pred, anno)
            center = calc_err_center(pred, anno)
            ncenter = calc_err_center(pred, anno, normalized=True)
            iou[~valid] = -1
            center[~valid] = float('inf')
            # Preserve the historical evaluator's NPR convention. Non-LaSOT
            # invalid GT frames count as NPR successes; see the archived report.
            # AUC and PRE count invalid frames as failures for all datasets.
            ncenter[~valid] = float('inf') if seq.dataset == 'lasot' else -1
            auc = (iou[:,None] > overlap_thresholds[None,:]).double().mean(0).mean().item()*100
            pre = (center <= 20).double().mean().item()*100
            npre = (ncenter <= .20).double().mean().item()*100
            rows.append({'sequence':seq.name,'frames':len(seq.frames),'AUC':auc,'PRE':pre,'NPR':npre,
                         'total_inference_seconds':float(times.sum()), 'first_frame_prediction':boxes[0].tolist()})
        except Exception as exc:
            errors.append({'sequence':seq.name,'error':repr(exc)})
    (folder/'per_sequence.json').write_text(json.dumps(rows,indent=2))
    (folder/'prediction_validation.json').write_text(json.dumps({'expected':len(dataset),'validated':len(rows),'errors':errors},indent=2))
    if errors:
        raise RuntimeError('Missing or invalid results: '+repr(errors[:3]))
    metrics={key:float(np.mean([row[key] for row in rows])) for key in ('AUC','PRE','NPR')}
    return {'metrics':metrics,'sequences':len(rows),'frames':sum(r['frames'] for r in rows),
            'fps_including_initialization':sum(r['frames'] for r in rows)/sum(r['total_inference_seconds'] for r in rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--config',required=True)
    p.add_argument('--weights',required=True);p.add_argument('--out',required=True)
    p.add_argument('--threads',type=int,default=16);p.add_argument('--gpus',type=int,default=8)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--exp-id',default='');args=p.parse_args()
    folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
    status={'status':'starting','dataset':args.dataset,'weights':args.weights,'config':args.config,
            'protocol':'language-only: first frame predicted from full image; no GT initialization',
            'note':'Historical parameters were tuned on these test sets; exploratory results.', 'started_at':time.time()}
    def save():
        tmp=folder/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(folder/'status.json')
    save()
    try:
        dataset,excluded=load_sequences(args.dataset)
        if args.limit: dataset=dataset[:args.limit]
        status.update(status='running',sequences=len(dataset),excluded=excluded);save()
        tracker=StrictLanguageTracker('mvlm',args.config,args.exp_id or args.config,args.dataset,args.weights)
        run_dataset(dataset,[tracker],debug=False,threads=args.threads,num_gpus=args.gpus,parallel_mode='mp')
        status.update(calculate(dataset,tracker,folder),status='completed',completed_at=time.time());save()
        print('RESULT',json.dumps(status),flush=True)
    except BaseException as exc:
        status.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),completed_at=time.time());save()
        raise

if __name__ == '__main__':main()
