"""Explore transfer of the historical OTB tuned parameters to B checkpoints and baseline."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'output/memory_world_b_otb_historical_params_20261112_run01'
TRAIN = ROOT / 'output/memory_world_tracker_b_2gpu_b80_20261111_run01'
CONFIG = 'mvlm_TF_memory_world_tracker_b_2gpu'
DATASETS = {'otb99_lang': 48}
MODELS = {
    'baseline': (ROOT / 'models/MVLM_TF.pth.tar', [0, 1]),
    'b1025': (TRAIN / 'step_001025.pth.tar', [2, 3]),
    'b4000': (TRAIN / 'step_004000.pth.tar', [4, 5]),
}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')


def worker(args):
    import cv2
    import numpy as np
    import torch
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.tracker import Tracker
    from lib.test.analysis.extract_results import calc_seq_err_robust
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    status_path = OUT / 'workers' / f'{args.model}_{args.shard}.json'
    status = dict(status='starting', model=args.model, shard=args.shard, started_at=now(), pid=os.getpid())
    write(status_path, status)
    try:
        for ds, expected in DATASETS.items():
            dataset = get_dataset(ds)
            assert len(dataset) == expected, (ds, len(dataset), expected)
            wrapper = Tracker('mvlm', CONFIG, 'benchmark', ds, str(MODELS[args.model][0]))
            params = wrapper.get_parameters()
            params.debug = False
            # OTB test-set-tuned historical parameters, fixed for all models.
            p = params.cfg.TEST.MVLM
            p.ALPHA_CORR, p.TAU, p.OMEGA, p.LAMBDA, p.PSI_OUT = 0.5, 0.03, 0.4, 0.1, 0.1
            if args.model == 'baseline':
                params.cfg.MODEL.MEMORY.ENABLED = False
                params.cfg.MODEL.WORLD.ENABLED = False
                params.cfg.MODEL.WORLD.QUALITY_ENABLED = False
                params.cfg.MODEL.TEMPORAL_BASELINE = True
            tracker = wrapper.create_tracker(params)
            assert tracker.template_free and tracker.use_nlp and tracker.multi_modal_language
            for idx, seq in enumerate(dataset):
                if idx % args.shards != args.shard:
                    continue
                if args.smoke and idx >= args.shards:
                    continue
                result_path = OUT / ('smoke' if args.smoke else 'results') / args.model / ds / (seq.name + '.json')
                if result_path.exists() and not args.smoke:
                    continue
                assert len(seq.frames) == len(seq.ground_truth_rect), (seq.name, len(seq.frames), len(seq.ground_truth_rect))
                if args.smoke:
                    seq.frames = seq.frames[:40]
                    seq.ground_truth_rect = seq.ground_truth_rect[:40]
                    if seq.target_visible is not None:
                        seq.target_visible = seq.target_visible[:40]
                status.update(status='running', dataset=ds, sequence=seq.name, frame_count=len(seq.frames), updated_at=now())
                write(status_path, status)
                start = time.monotonic()
                output = wrapper._track_sequence(tracker, seq, seq.init_info())
                boxes = np.asarray(output['target_bbox']).astype(int)
                timings = np.asarray(output['time'])
                assert boxes.shape == (len(seq.frames), 4) and timings.shape == (len(seq.frames),), (seq.name, boxes.shape, timings.shape)
                assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all() and np.isfinite(timings).all()
                pred = torch.as_tensor(boxes, dtype=torch.float64)
                anno = torch.as_tensor(seq.ground_truth_rect, dtype=torch.float64)
                visible = torch.as_tensor(seq.target_visible) if seq.target_visible is not None else None
                overlap, center, norm, valid = calc_seq_err_robust(pred, anno, seq.dataset, visible)
                thresholds = torch.arange(0, 1.01, .05, dtype=torch.float64)
                row = dict(sequence=seq.name, dataset=ds, frames=len(seq.frames),
                           AUC=float((overlap[:, None] > thresholds).double().mean() * 100),
                           PRE=float((center <= 20).double().mean() * 100),
                           NPR=float((norm <= .2).double().mean() * 100),
                           wall_seconds=time.monotonic()-start, boxes=boxes.tolist())
                write(result_path, row)
                print(json.dumps({k: v for k, v in row.items() if k != 'boxes'}), flush=True)
            del tracker
            torch.cuda.empty_cache()
        status.update(status='completed', completed_at=now())
    except BaseException as exc:
        status.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), completed_at=now())
        raise
    finally:
        write(status_path, status)


def suite(args):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(started_at=now(), config=str(ROOT / 'experiments/mvlm' / (CONFIG + '.yaml')),
                    description='Exploratory: historical parameters previously tuned on OTB test videos; identical fixed parameters for all three models; not an independent final test.', inference_parameters=dict(ALPHA_CORR=0.5, TAU=0.03, OMEGA=0.4, LAMBDA=0.1, PSI_OUT=0.1),
                    datasets=DATASETS, workers_per_gpu=args.workers_per_gpu, models={})
    for name, (path, gpus) in MODELS.items():
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b''):
                h.update(chunk)
        manifest['models'][name] = dict(checkpoint=str(path), sha256=h.hexdigest(), physical_gpus=gpus)
    write(OUT/'manifest.json', manifest)
    processes = []
    status = dict(status='starting', pid=os.getpid(), started_at=now(), completed={})
    try:
        for name, (_, gpus) in MODELS.items():
            shards = len(gpus) * args.workers_per_gpu
            for shard in range(shards):
                gpu = gpus[shard % len(gpus)]
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), MVLM_DEVICE='cuda', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                log = OUT / 'logs' / f'{name}_{shard}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                f = log.open('w')
                cmd = [sys.executable, __file__, 'worker', '--model', name, '--shard', str(shard), '--shards', str(shards)]
                p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
                f.close()
                processes.append((name, shard, gpu, p))
                time.sleep(2)
        status['status'] = 'running'
        while True:
            errors = [dict(model=n, shard=s, gpu=g, returncode=p.poll()) for n,s,g,p in processes if p.poll() not in (None, 0)]
            status.update(updated_at=now(), workers=[dict(model=n, shard=s, gpu=g, pid=p.pid, returncode=p.poll()) for n,s,g,p in processes])
            for name in MODELS:
                status['completed'][name] = {ds: len(list((OUT/'results'/name/ds).glob('*.json'))) for ds in DATASETS}
            write(OUT/'status.json', status)
            if errors:
                raise RuntimeError(f'Worker failure: {errors}')
            if all(p.poll() is not None for _,_,_,p in processes):
                break
            time.sleep(20)
        results = {}
        for name in MODELS:
            results[name] = {}
            for ds, expected in DATASETS.items():
                rows = [json.loads(p.read_text()) for p in (OUT/'results'/name/ds).glob('*.json')]
                assert len(rows) == expected, (name, ds, len(rows), expected)
                results[name][ds] = dict(sequences=len(rows), frames=sum(r['frames'] for r in rows),
                    **{k: sum(r[k] for r in rows)/len(rows) for k in ('AUC','PRE','NPR')})
        write(OUT/'summary.json', results)
        status.update(status='completed', completed_at=now(), results=results)
    except BaseException as exc:
        for _,_,_,p in processes:
            if p.poll() is None:
                p.terminate()
        status.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), completed_at=now())
        raise
    finally:
        write(OUT/'status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('worker')
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--smoke', action='store_true')
    p = sub.add_parser('suite')
    p.add_argument('--workers-per-gpu', type=int, default=2)
    args = parser.parse_args()
    (worker if args.command == 'worker' else suite)(args)
