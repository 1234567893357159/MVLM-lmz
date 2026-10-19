"""Controlled validation-prefix diagnostics. Oracle runs are diagnostic only."""
import argparse
import fcntl
from collections import defaultdict
import copy
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.data import Sequence
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.models.mvlm.memory import extract_target_tokens
import lib.test.tracker.mvlm as tracker_module

RUN=Path('/root/MVLM/output/memory_three_custom_20261018_run01')
VARIANTS={
 'unchanged':dict(checkpoint=20),
 'disabled':dict(checkpoint=20,disabled=True),
 'half':dict(checkpoint=20,scale=.5),
 'quarter':dict(checkpoint=20,scale=.25),
 'short_only':dict(checkpoint=20,long_capacity=0),
 'strict_quality':dict(checkpoint=20,min_quality=.9),
 'no_margin_write_gate':dict(checkpoint=20,no_margin=True),
 'oracle_gt_write':dict(checkpoint=20,oracle=True),
 'enhanced_correlation':dict(checkpoint=20,enhanced=True),
 'best_half':dict(checkpoint=2,scale=.5),
 'best_short_only':dict(checkpoint=2,long_capacity=0),
}


def score(pred,boxes,visible,dataset):
    overlap,center,norm,_=calc_seq_err_robust(torch.tensor(pred,dtype=torch.float64),torch.tensor(boxes,dtype=torch.float64),dataset,
               torch.tensor(visible) if dataset=='lasot' else None)
    return dict(AUC=float((overlap[:,None]>torch.arange(0.,1.0001,.05,dtype=torch.float64)[None,:]).float().mean()*100),
                PRE=float((center<=20).float().mean()*100),NPR=float((norm<=.2).float().mean()*100))


def one_sequence(tracker,record,variant,out,frames_limit):
    paths=Path(record['frame_list']).read_text().splitlines()[:frames_limit or None]
    with np.load(record['annotations']) as a:boxes=a['boxes'][:len(paths)].copy();visible=a['visible'][:len(paths)].copy()
    options=VARIANTS[variant];net=tracker.network
    net.memory_enabled=not options.get('disabled',False)
    net.memory_cfg.LONG_CAPACITY=options.get('long_capacity',16)
    net.memory_cfg.MIN_QUALITY=options.get('min_quality',.5)
    cache={}; diagnostics=defaultdict(list)
    original_slice=tracker_module.batched_slice_mean_from_grid
    original_extract=tracker_module.extract_target_tokens
    def capture(module,args,output):
        cache['out']=output
        if options.get('enhanced') and 'feat_x_mem' in output:cache['enhanced']=output['feat_x_mem']
    def fusion_hook(module,args,output):
        features=args[0];fused,gate=output
        diagnostics['delta_to_feature_norm'].append(float((fused-features).norm()/features.norm().clamp_min(1e-8)))
        if options.get('scale',1.) == 1.: return output
        return features+options.get('scale',1.)*(fused-features),gate
    def sliced(tokens,boxes):return original_slice(cache.get('enhanced',tokens),boxes)
    def extracted(*args,**kwargs):
        result=original_extract(*args,**kwargs);cache['roi_valid']=result[1].clone();return result
    tracker_module.batched_slice_mean_from_grid=sliced
    tracker_module.extract_target_tokens=extracted
    h1=net.register_forward_hook(capture);h2=net.memory_fusion.register_forward_hook(fusion_hook)
    predictions=[];started=time.time()
    try:
        for i,path in enumerate(paths):
            image=cv2.imread(path)
            if image is None:raise ValueError(path)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);H,W=image.shape[:2]
            cache['frame']=i;cache['image_size']=(H,W);cache['gt']=boxes[i];cache['visible']=bool(visible[i])
            if i==0:
                seq=Sequence(record['name'],paths,record['dataset'],boxes,language_query=record['language'])
                tracker.initialize(image,seq.init_info());pred=boxes[0]
                if tracker.content_memory is not None:
                    bank=tracker.content_memory;original_write=bank.write
                    def write(tokens,full_box,frame_ids,quality,valid=None):
                        i=cache['frame'];gt=cache['gt'];H,W=cache['image_size']
                        if options.get('oracle'):
                            if cache['global_before']:
                                roi=np.asarray(gt)/np.array([W,H,W,H])
                            else:
                                x,y,w,h=cache['state_before'];side=math.ceil(math.sqrt(w*h)*tracker.params.search_factor)
                                left=round(x+.5*w-.5*side);top=round(y+.5*h-.5*side)
                                roi=(np.asarray(gt)-np.array([left,top,0,0]))/side
                            roi=tokens.new_tensor(roi).reshape(1,4)
                            tokens,valid=extract_target_tokens(cache['out']['feat_x'],roi,tracker.cfg.MODEL.MEMORY.ROI_SIZE)
                            valid &= cache['visible'];full_box=tokens.new_tensor(gt).reshape(1,4)/tokens.new_tensor([W,H,W,H])
                            quality=torch.ones_like(quality)
                        elif options.get('no_margin'):
                            valid=cache['roi_valid']
                        true=tokens.new_tensor(gt).reshape(1,4)/tokens.new_tensor([W,H,W,H])
                        tl=torch.maximum(full_box[:,:2],true[:,:2]);br=torch.minimum(full_box[:,:2]+full_box[:,2:],true[:,:2]+true[:,2:])
                        inter=(br-tl).clamp_min(0).prod(1);iou=inter/(full_box[:,2:].prod(1)+true[:,2:].prod(1)-inter).clamp_min(1e-8)
                        written=original_write(tokens,full_box,frame_ids,quality,valid)
                        diagnostics['write_accepted'].append(float(written[0]))
                        if written[0]:diagnostics['accepted_write_iou'].append(float(iou[0]))
                        return written
                    bank.write=write
            else:
                cache['global_before']=tracker.flag;cache['state_before']=list(tracker.state)
                with torch.no_grad():result=tracker.track(image)
                pred=result['target_bbox']
                if 'memory_gate' in result:diagnostics['gate'].append(result['memory_gate'])
            predictions.append(pred)
        values=np.asarray(predictions).astype(np.int64)
        assert values.shape==boxes.shape and np.isfinite(values).all()
        destination=out/variant/record['dataset'];destination.mkdir(parents=True,exist_ok=True)
        np.savetxt(destination/(record['name']+'.txt'),values,fmt='%d',delimiter='\t')
        result=dict(variant=variant,key=record['key'],dataset=record['dataset'],name=record['name'],frames=len(paths),
                    metrics=score(values,boxes,visible,record['dataset']),seconds=time.time()-started,
                    diagnostics={k:dict(mean=float(np.mean(v)),count=len(v)) for k,v in diagnostics.items() if v})
        (destination/(record['name']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        return result
    finally:
        h1.remove();h2.remove();tracker_module.batched_slice_mean_from_grid=original_slice;tracker_module.extract_target_tokens=original_extract


def worker(index,jobs,out,limit):
    torch.set_num_threads(1);cv2.setNumThreads(0);torch.cuda.set_device(index%8)
    tracker=None;current=None
    while True:
        task=jobs.get()
        if task is None:break
        variant,record=task
        dest=Path(out)/variant/record['dataset']/(record['name']+'.json')
        if dest.exists():continue
        checkpoint=VARIANTS[variant]['checkpoint']
        # All dataset-specific settings in this frozen config are identical.
        if current!=checkpoint:
            if tracker is not None:del tracker;torch.cuda.empty_cache()
            path=RUN/f'checkpoints/train/mvlm/mvlm_TF_memory_three_custom/MVLM_ep{checkpoint:04d}.pth.tar'
            wrapper=Tracker('mvlm','mvlm_TF_memory_three_custom','diagnostic',record['dataset'],str(path))
            params=wrapper.get_parameters();params.cfg.MODEL.PRETRAINED_MODEL=None
            params.device='cuda:'+str(index%8);params.debug=False
            tracker=wrapper.create_tracker(params);current=checkpoint
        result=one_sequence(tracker,record,variant,Path(out),limit)
        print(json.dumps({'worker':index,'variant':variant,'key':record['key'],'AUC':result['metrics']['AUC'],'seconds':result['seconds']}),flush=True)


def main(out,workers,limit,full_variant=None):
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((RUN/'split/manifest.json').read_text())
    records=[r for r in manifest['records'] if r['split']=='val']
    final=json.loads((RUN/'validation/epoch_0020/metrics.json').read_text());base=json.loads((RUN/'validation/baseline/metrics.json').read_text())
    if not full_variant:
        names=set()
        for d,rows in final['per_sequence'].items():
            ref={s['name']:s['AUC'] for s in base['per_sequence'][d]}
            ordered=sorted(rows,key=lambda s:s['AUC']-ref[s['name']])
            for x in [ordered[0],ordered[len(ordered)//2],ordered[-1]]:names.add((d,x['name']))
        records=[r for r in records if (r['dataset'],r['name']) in names]
    selection=out/'selection.json'
    if not selection.exists():selection.write_text(json.dumps({'records':[r['key'] for r in records],'prefix_limit':limit,'selection':('full frozen validation' if full_variant else 'worst/median/best final-checkpoint change per dataset; diagnostic selection bias; not representative performance')},indent=2)+'\n')
    variants=[full_variant] if full_variant else list(VARIANTS)
    context=mp.get_context('spawn');queue=context.Queue()
    for variant in variants:
        for record in sorted(records,key=lambda r:r['frames'],reverse=True):queue.put((variant,record))
    for _ in range(workers):queue.put(None)
    processes=[context.Process(target=worker,args=(i,queue,str(out),limit)) for i in range(workers)]
    for p in processes:p.start()
    for p in processes:
        p.join()
        if p.exitcode:raise RuntimeError('Diagnostic worker failed: '+str(p.exitcode))
    report={}
    for variant in variants:
        values=[json.loads((out/variant/r['dataset']/(r['name']+'.json')).read_text()) for r in records]
        report[variant]={'mean_AUC':float(np.mean([np.mean([v['metrics']['AUC'] for v in values if v['dataset']==d]) for d in ['otb99_lang','lasot','tnl2k']])),
                         'datasets':{d:float(np.mean([v['metrics']['AUC'] for v in values if v['dataset']==d])) for d in ['otb99_lang','lasot','tnl2k']},'sequences':len(values)}
    for label,folder in [('cached_baseline','baseline'),('cached_final','epoch_0020'),('cached_best','epoch_0002')]:
        values=[]
        for r in records:
            with np.load(r['annotations']) as a:boxes=a['boxes'][:limit or None];visible=a['visible'][:limit or None]
            pred=np.loadtxt(RUN/'validation'/folder/r['dataset']/(r['name']+'.txt'))[:len(boxes)]
            values.append((r['dataset'],score(pred,boxes,visible,r['dataset'])['AUC']))
        report[label]={'mean_AUC':float(np.mean([np.mean([v for d,v in values if d==dataset]) for dataset in ['otb99_lang','lasot','tnl2k']])),
                       'datasets':{dataset:float(np.mean([v for d,v in values if d==dataset])) for dataset in ['otb99_lang','lasot','tnl2k']}}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--limit',type=int,default=512);p.add_argument('--full-variant',choices=list(VARIANTS))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'.execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main(a.out.resolve(),a.workers,a.limit,a.full_variant)
