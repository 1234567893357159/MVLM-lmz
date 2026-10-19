"""Evaluate only the frozen validation split; test split is never read here."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lib.test.evaluation.data import Sequence
from lib.test.evaluation.tracker import Tracker
from lib.test.analysis.extract_results import calc_seq_err_robust


def worker(gpu, assignments, config_name, weights, baseline, out, devices=None):
    physical_gpu = gpu % (devices or torch.cuda.device_count())
    torch.set_num_threads(1); cv2.setNumThreads(0); torch.cuda.set_device(physical_gpu)
    trackers={}
    for record in assignments[gpu]:
        folder=Path(out)/record['dataset']; folder.mkdir(parents=True,exist_ok=True)
        dest=folder/(record['name']+'.txt')
        if dest.exists(): continue
        dataset=record['dataset']
        if dataset not in trackers:
            wrapper=Tracker('mvlm',config_name,'custom-validation',dataset,str(weights))
            params=wrapper.get_parameters();params.cfg.MODEL.MEMORY.ENABLED=not baseline
            params.cfg.MODEL.PRETRAINED_MODEL=None;params.device='cuda:'+str(physical_gpu);params.debug=False
            trackers[dataset]=wrapper.create_tracker(params)
        tracker=trackers[dataset]
        paths=Path(record['frame_list']).read_text().splitlines()
        with np.load(record['annotations']) as a: boxes=a['boxes'].copy()
        preds=[]
        with torch.no_grad():
            for i,path in enumerate(paths):
                image=cv2.imread(path)
                if image is None: raise ValueError('Undecodable validation frame: '+path)
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                if i==0:
                    seq=Sequence(record['name'],paths,dataset,boxes,language_query=record['language'])
                    info=seq.init_info();info['seq_name']=seq.name
                    tracker.initialize(image,info);pred=boxes[0].tolist()
                else: pred=tracker.track(image)['target_bbox']
                preds.append(pred)
        values=np.asarray(preds)
        if values.shape!=boxes.shape or not np.isfinite(values).all() or not (values[:,2:]>0).all():
            raise ValueError('Invalid prediction '+record['key'])
        # Match repository prediction files: integer bounding boxes.
        temp=dest.with_suffix('.tmp');np.savetxt(temp,values,fmt='%d',delimiter='\t');temp.replace(dest)
        print('VALIDATED',gpu,record['key'],len(paths),flush=True)


def evaluate(manifest_path,weights,out,config_name,gpus,baseline=False,limit=0,workers=None,scope='full'):
    manifest=json.loads(Path(manifest_path).read_text())
    records=[r for r in manifest['records'] if r['split']=='val']
    if limit:
        # Only for engineering checks; never used for best-checkpoint selection.
        selected=[]
        for dataset in sorted({r['dataset'] for r in records}):
            selected.append(min((r for r in records if r['dataset']==dataset),key=lambda r:r['frames']))
        records=selected[:limit]
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    workers = workers or gpus
    if gpus < 1 or gpus > torch.cuda.device_count() or workers < 1: raise ValueError("Invalid GPU/worker count")
    tasks=[[] for _ in range(workers)];loads=[0]*workers
    for r in sorted(records,key=lambda r:r['frames'],reverse=True):
        i=min(range(workers),key=lambda i:loads[i]);tasks[i].append(r);loads[i]+=r['frames']
    started=time.time()
    mp.spawn(worker,args=(tasks,config_name,str(weights),baseline,str(out),gpus),nprocs=workers,join=True)
    grouped=defaultdict(list)
    for r in records:
        with np.load(r['annotations']) as a:
            anno=torch.tensor(a['boxes'],dtype=torch.float64)
            visible=torch.tensor(a['visible']) if r['dataset']=='lasot' else None
        pred=torch.tensor(np.loadtxt(out/r['dataset']/(r['name']+'.txt')),dtype=torch.float64)
        overlap,center,norm,_=calc_seq_err_robust(pred,anno,r['dataset'],visible)
        auc=(overlap[:,None]>torch.arange(0.,1.0001,.05,dtype=torch.float64)[None,:]).float().mean().item()*100
        grouped[r['dataset']].append({'name':r['name'],'PRE':(center<=20).float().mean().item()*100,
                                      'NPR':(norm<=.2).float().mean().item()*100,'AUC':auc})
    metrics={d:{k:float(np.mean([r[k] for r in rows])) for k in ['PRE','NPR','AUC']} for d,rows in grouped.items()}
    result={'metrics':metrics,'mean_AUC':float(np.mean([m['AUC'] for m in metrics.values()])),
            'sequences':len(records),'per_sequence':dict(grouped),'baseline':baseline,'weights':str(weights),
            'seconds':time.time()-started,'engineering_subset':bool(limit or scope!='full'),
            'evaluation_scope':scope}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_sequence'},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--out',required=True);p.add_argument('--config',default='mvlm_TF_memory_three_custom')
    p.add_argument('--gpus',type=int,default=8);p.add_argument('--baseline',action='store_true');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--workers',type=int,default=None)
    p.add_argument('--scope',choices=['full','fixed_subset'],default='full')
    a=p.parse_args();evaluate(a.manifest,a.weights,a.out,a.config,a.gpus,a.baseline,a.limit,a.workers,a.scope)
