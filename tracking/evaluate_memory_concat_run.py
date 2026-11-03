"""Evaluate the frozen training-side holdout using the run's exact config."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from lib.config.mvlm.config import cfg, update_config_from_file
from lib.test.utils import TrackerParams
from lib.test.tracker.mvlm import MVLM as OnlineTracker
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.train.memory_run_utils import write_json


def worker(index, tasks, config_path, weight, out, baseline, smoke):
    if not tasks[index]:
        return
    torch.set_num_threads(1); cv2.setNumThreads(0)
    device = index % 2
    torch.cuda.set_device(device)
    update_config_from_file(config_path)
    config = copy.deepcopy(cfg)
    config.MODEL.MEMORY.ENABLED = not baseline
    config.MODEL.PRETRAINED_MODEL = None
    # These three training sources intentionally share every inference setting.
    # Reuse one network per worker rather than keeping three CLIP copies alive.
    params = TrackerParams()
    params.cfg, params.checkpoint = config, weight
    params.device, params.debug = 'cuda:'+str(device), False
    params.template_factor, params.template_size = config.TEST.TEMPLATE_FACTOR, config.TEST.TEMPLATE_SIZE
    params.search_factor, params.search_size = config.TEST.SEARCH_FACTOR, config.TEST.SEARCH_SIZE
    tracker = OnlineTracker(params, 'vasttrack')
    for record in tasks[index]:
        destination = Path(out)/record['dataset']/(record['name']+'.txt')
        paths = Path(record['frame_list']).read_text().splitlines()
        with np.load(record['annotations']) as source:
            boxes = source['boxes']
        if smoke:
            paths, boxes = paths[:30], boxes[:30]
        if destination.exists():
            previous = np.loadtxt(destination, ndmin=2)
            if previous.shape == boxes.shape and np.isfinite(previous).all():
                continue
            destination.rename(destination.with_suffix('.invalid'))
        predictions, telemetry = [], []
        with torch.no_grad():
            for frame, path in enumerate(paths):
                image = cv2.imread(path)
                if image is None:
                    raise ValueError('Undecodable validation image: '+path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if frame == 0:
                    tracker.initialize(image, dict(init_bbox=boxes[0].tolist(), init_nlp=record['language']))
                    predictions.append(boxes[0].tolist())
                else:
                    result = tracker.track(image)
                    predictions.append(result['target_bbox'])
                    if 'memory_gate' in result:
                        telemetry.append([result['memory_gate'],result['memory_short_size'],result['memory_long_size']])
        values = np.asarray(predictions)
        if values.shape != boxes.shape or not np.isfinite(values).all() or not (values[:,2:]>0).all():
            raise RuntimeError('Invalid predictions: '+record['key'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix('.tmp')
        np.savetxt(temp, values, fmt='%d', delimiter='\t'); temp.replace(destination)
        if telemetry:
            write_json(destination.with_suffix('.memory.json'),dict(
                mean_gate=float(np.mean(np.asarray(telemetry)[:,0])),
                max_short=int(np.max(np.asarray(telemetry)[:,1])), max_long=int(np.max(np.asarray(telemetry)[:,2]))))
        print('VALIDATED', record['key'],len(paths),flush=True)


def evaluate(args):
    manifest = json.loads(args.manifest.read_text())
    records = manifest['records']
    if args.smoke:
        records = [min((r for r in records if r['dataset']==name),key=lambda r:r['frames'])
                   for name in ('lasot','vasttrack','tnl2k')]
    assert all(r['split']=='val' for r in records)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tasks, loads = [[] for _ in range(args.workers)], [0]*args.workers
    for record in sorted(records,key=lambda r:r['frames'],reverse=True):
        index = min(range(args.workers), key=lambda i:loads[i])
        tasks[index].append(record); loads[index] += min(record['frames'],30) if args.smoke else record['frames']
    mp.spawn(worker,args=(tasks,str(args.config),str(args.weights),str(args.out),args.baseline,args.smoke),
             nprocs=args.workers,join=True)
    grouped = defaultdict(list)
    for record in records:
        with np.load(record['annotations']) as source:
            boxes, visible = source['boxes'].copy(), source['visible'].copy()
        if args.smoke:
            boxes, visible = boxes[:30], visible[:30]
        predictions = np.loadtxt(args.out/record['dataset']/(record['name']+'.txt'),ndmin=2)
        assert predictions.shape == boxes.shape
        overlap,center,norm,_ = calc_seq_err_robust(torch.tensor(predictions,dtype=torch.float64),
            torch.tensor(boxes,dtype=torch.float64),record['dataset'],
            torch.tensor(visible) if record['dataset']=='lasot' else None)
        thresholds = torch.arange(0,1.01,.05,dtype=torch.float64)
        grouped[record['dataset']].append(dict(name=record['name'],
            AUC=float((overlap[:,None]>thresholds).double().mean()*100),
            PRE=float((center<=20).double().mean()*100), NPR=float((norm<=.2).double().mean()*100)))
    metrics = {name:{key:float(np.mean([r[key] for r in rows])) for key in ('AUC','PRE','NPR')}
               for name,rows in grouped.items()}
    result = dict(scope='engineering_3x30' if args.smoke else args.scope,baseline=args.baseline,
        sequences=len(records),weights=str(args.weights),manifest=str(args.manifest),
        metrics=metrics,mean_AUC=float(np.mean([r['AUC'] for r in metrics.values()])),
        seconds=time.monotonic()-started,per_sequence=dict(grouped))
    write_json(args.out/'metrics.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_sequence'}),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    for name in ('config','manifest','weights','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--scope',choices=['full150','quick30'],default='full150')
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    evaluate(parser.parse_args())
