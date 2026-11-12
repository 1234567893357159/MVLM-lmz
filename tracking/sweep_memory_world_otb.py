"""Exploratory fixed-grid OTB test-set sweep, not an independent final evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tracking'))
import evaluate_memory_world_otb_historical_params as benchmark

OUT = ROOT / 'output/memory_world_b_otb_sweep_20261112_run01'
ANCHOR = dict(ALPHA_CORR=0.5, TAU=0.03, OMEGA=0.4, LAMBDA=0.1, PSI_OUT=0.1, QUALITY_WEIGHT=0.25)
# One change at a time around the historic OTB parameter combination.
VARIANTS = {
    'alpha03': {'ALPHA_CORR': 0.3}, 'alpha07': {'ALPHA_CORR': 0.7},
    'alpha09': {'ALPHA_CORR': 0.9}, 'tau001': {'TAU': 0.01},
    'tau005': {'TAU': 0.05}, 'tau010': {'TAU': 0.10},
    'omega01_lambda04': {'OMEGA': 0.1, 'LAMBDA': 0.4},
    'omega02_lambda04': {'OMEGA': 0.2, 'LAMBDA': 0.4},
    'psi005': {'PSI_OUT': 0.05}, 'psi020': {'PSI_OUT': 0.20},
    'quality000': {'QUALITY_WEIGHT': 0.0}, 'quality050': {'QUALITY_WEIGHT': 0.5},
}
MODELS = ('b1025', 'b4000')
SHARDS = 4

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def worker(args):
    values = dict(ANCHOR, **VARIANTS[args.variant])
    # Load an isolated in-memory copy of the verified benchmark worker, replacing
    # only its parameter assignment; avoid modifying the original experiment.
    import types
    source = (ROOT / 'tracking/evaluate_memory_world_otb_historical_params.py').read_text()
    original = '            p.ALPHA_CORR, p.TAU, p.OMEGA, p.LAMBDA, p.PSI_OUT = 0.5, 0.03, 0.4, 0.1, 0.1'
    replacement = "            p.ALPHA_CORR, p.TAU, p.OMEGA, p.LAMBDA, p.PSI_OUT = SWEEP_VALUES['ALPHA_CORR'], SWEEP_VALUES['TAU'], SWEEP_VALUES['OMEGA'], SWEEP_VALUES['LAMBDA'], SWEEP_VALUES['PSI_OUT']\n            params.cfg.MODEL.WORLD.QUALITY_WEIGHT = SWEEP_VALUES['QUALITY_WEIGHT']"
    if source.count(original) != 1:
        raise ValueError('Benchmark parameter assignment changed')
    module = types.ModuleType('sweep_worker')
    module.__file__ = str(ROOT / 'tracking/evaluate_memory_world_otb_historical_params.py')
    module.SWEEP_VALUES = values
    exec(compile(source.replace(original, replacement), module.__file__, 'exec'), module.__dict__)
    module.OUT = (OUT.parent / (OUT.name + '_smoke') if args.smoke else OUT) / args.variant
    print('SWEEP_PARAMETERS', json.dumps(values, sort_keys=True), flush=True)
    module.worker(SimpleNamespace(model=args.model, shard=args.shard, shards=SHARDS, smoke=args.smoke))


def suite():
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = dict(started_at=benchmark.now(), protocol='exploratory OTB test-set tuning; not independent final evaluation',
        dataset='otb99_lang', sequences=48, shards=SHARDS, models={m: dict(checkpoint=str(benchmark.MODELS[m][0]), sha256=sha(benchmark.MODELS[m][0])) for m in MODELS},
        source={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__).resolve(), ROOT/'tracking/evaluate_memory_world_otb_historical_params.py',ROOT/'experiments/mvlm/mvlm_TF_memory_world_tracker_b_2gpu.yaml')},
        anchor=ANCHOR, variants={name:dict(ANCHOR,**v) for name,v in VARIANTS.items()},
        reference='output/memory_world_b_otb_historical_params_20261112_run01', max_workers=8)
    benchmark.write(OUT/'manifest.json',manifest)
    jobs=[(v,m,s) for v in VARIANTS for m in MODELS for s in range(SHARDS)]
    running={}; status={'status':'running','pid':os.getpid(),'started_at':benchmark.now()}
    try:
        while jobs or running:
            for gpu,(proc,handle,job) in list(running.items()):
                code=proc.poll()
                if code is not None:
                    handle.close();del running[gpu]
                    if code: raise RuntimeError('worker failed: {} gpu={} exit={}'.format(job,gpu,code))
            for gpu in range(8):
                if gpu in running or not jobs: continue
                v,m,shard=jobs.pop(0)
                log=OUT/'logs'/('{}-{}-{}.log'.format(v,m,shard));log.parent.mkdir(exist_ok=True)
                handle=log.open('w');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MVLM_DEVICE='cuda',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--variant',v,'--model',m,'--shard',str(shard)]
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                running[gpu]=(p,handle,(v,m,shard))
            status.update(updated_at=benchmark.now(), queued_jobs=len(jobs),running=[dict(gpu=g,pid=p.pid,variant=j[0],model=j[1],shard=j[2]) for g,(p,h,j) in running.items()],completed={v:{m:len(list((OUT/v/'results'/m/'otb99_lang').glob('*.json'))) for m in MODELS} for v in VARIANTS})
            benchmark.write(OUT/'status.json',status)
            if running: time.sleep(10)
        results={}
        for v in VARIANTS:
            results[v]={}
            for m in MODELS:
                rows=[json.loads(p.read_text()) for p in (OUT/v/'results'/m/'otb99_lang').glob('*.json')]
                if len(rows)!=48: raise ValueError((v,m,len(rows)))
                results[v][m]=dict(sequences=48, **{k:sum(r[k] for r in rows)/48 for k in ('AUC','PRE','NPR')})
        benchmark.write(OUT/'summary.json',results)
        status.update(status='completed',completed_at=benchmark.now(),running=[])
    except BaseException as e:
        for p,h,j in running.values(): p.terminate();h.close()
        status.update(status='failed',error=repr(e),traceback=traceback.format_exc(),completed_at=benchmark.now())
        raise
    finally: benchmark.write(OUT/'status.json',status)

if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True);sub.add_parser('suite')
    w=sub.add_parser('worker');w.add_argument('--variant',choices=VARIANTS,required=True);w.add_argument('--model',choices=MODELS,required=True);w.add_argument('--shard',type=int,required=True);w.add_argument('--smoke',action='store_true')
    args=p.parse_args();worker(args) if args.command=='worker' else suite()
