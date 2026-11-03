"""Evaluate each E61-E70 checkpoint on free GPUs6,7 with fixed main params."""
import argparse
import csv
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/root/MVLM')
TRAIN=ROOT/'output/paper_continuation_e60_e70_20261102_run01'
OUT=ROOT/'output/paper_continuation_eval_20261102_run01'
BASE=ROOT/'output/paper_checkpoint_grid_20261101_run01'
REPORT=ROOT/'reports/reproduction/PAPER_CONTINUATION_TESTS_20261102.md'
DATASETS=('otb99_lang','lasot','tnl2k')
EPOCHS=tuple(range(61,71))


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def render(state):
    state['updated_at']=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
    save(OUT/'status.json',state)
    rows=[]
    for ep in EPOCHS:
        for ds in DATASETS:
            p=OUT/f'epoch_{ep:04d}'/ds/'status.json'
            if p.exists():rows.append(json.loads(p.read_text()))
    lines=['# E60之后续训：固定参数测试','',f"状态：{state['status']}；更新时间：{state['updated_at']}。",'',
           '使用空闲物理GPU4、5和6、7两组并行评测，每卡12个视频进程，各进程batch1。E61～E70每个检查点测试三个数据集，共30组；评测忙时后续检查点排队，不跳过。main首帧真实框计分，固定各数据集历史参数。属于探索性延长实验。','',
           '| Epoch | OTB AUC / PRE / NPR | LaSOT AUC / PRE / NPR | TNL2K AUC / PRE / NPR |','| ---: | --- | --- | --- |',
           '| 论文参考 | 60.70 / 84.30 / 75.40 | 60.70 / 65.50 / 68.30 | 57.80 / — / 74.10 |']
    for ep in (56,60)+EPOCHS:
        cells=[]
        for ds in DATASETS:
            p=(BASE if ep in (56,60) else OUT)/f'epoch_{ep:04d}'/ds/'status.json'
            r=json.loads(p.read_text()) if p.exists() else {}
            cells.append(' / '.join(f"{r['metrics'][k]:.2f}" for k in ('AUC','PRE','NPR')) if r.get('status')=='completed' else r.get('status','等待检查点'))
        lines.append(f'| {ep} | '+' | '.join(cells)+' |')
    lines+=['','E56为此前测试引导的高分对照，E60为原配方预先指定最终模型；新增结果不作为独立测试选模型证据。']
    if state.get('error'):lines+=['',state['error']]
    REPORT.write_text('\n'.join(lines)+'\n')


def run_job(args):
    spec=importlib.util.spec_from_file_location('grid',str(OUT/'source/tracking/evaluate_checkpoint_grid.py'))
    grid=importlib.util.module_from_spec(spec);spec.loader.exec_module(grid)
    grid.CKPTS=TRAIN/'checkpoints/train/mvlm/mvlm_TF_continue_e60_e70'
    grid.OUT=OUT
    args.workers=24;args.probe=False;args.config='papergrid_20261101_'+args.dataset
    args.exp_id=f'papercontinue_20261102_e{args.epoch:04d}_{args.dataset}'
    args.job_out=str(OUT/f'epoch_{args.epoch:04d}'/args.dataset)
    grid.job(args)


def idle(devices):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows={int(r.split(',')[0]):[int(x) for x in r.split(',')[1:]] for r in raw.splitlines()}
    return all(rows[i][0]<100 and rows[i][1]<5 for i in devices)


def suite():
    import fcntl
    OUT.mkdir(exist_ok=True);lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
    groups=((4,5),(6,7));active={};children={};failures=[]
    def alive(pid):
        try:
            stat=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1]
            return stat[0]!='Z'
        except FileNotFoundError:return False
    # Adopt the running job when replacing a supervisor; its video workers continue.
    old_jobs=previous.get('active_jobs',[])
    if not old_jobs and previous.get('worker_pid') and previous.get('status')=='testing':
        old_jobs=[dict(epoch=previous['epoch'],dataset=previous['dataset'],pid=previous['worker_pid'],physical_gpus=[6,7])]
    for job in old_jobs:
        if alive(job['pid']):
            cmd=Path(f"/proc/{job['pid']}/cmdline").read_bytes()
            assert b'continuation_checkpoint_eval.py' in cmd and b'job' in cmd,job
            active[tuple(job['physical_gpus'])]=job
    state=dict(status='testing',pid=os.getpid(),completed_jobs=0,total_jobs=len(EPOCHS)*len(DATASETS),physical_gpus=[4,5,6,7],gpu_groups=[list(g) for g in groups],evaluation_epochs=list(EPOCHS))
    while True:
        for group,job in list(active.items()):
            child=children.get(job['pid'])
            if (child.poll() is None) if child else alive(job['pid']):continue
            path=OUT/f"epoch_{job['epoch']:04d}"/job['dataset']/'status.json'
            result=json.loads(path.read_text()) if path.exists() else {}
            if result.get('status')!='completed':failures.append(dict(job=job,result=result))
            del active[group]
        completed=[];pending=[]
        assigned={(j['epoch'],j['dataset']) for j in active.values()}
        for ep in EPOCHS:
            for ds in DATASETS:
                path=OUT/f'epoch_{ep:04d}'/ds/'status.json'
                result=json.loads(path.read_text()) if path.exists() else {}
                if result.get('status')=='completed':completed.append((ep,ds))
                elif (ep,ds) not in assigned:pending.append((ep,ds))
        if not failures:
            for group in groups:
                if group in active or not pending:continue
                ep,ds=pending[0]
                checkpoint=TRAIN/f'checkpoints/train/mvlm/mvlm_TF_continue_e60_e70/MVLM_ep{ep:04d}.pth.tar'
                if not checkpoint.exists():continue
                if not idle(group):continue
                time.sleep(3)
                if not idle(group):continue
                pending.pop(0)
                folder=OUT/f'epoch_{ep:04d}'/ds;folder.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,group)),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OPENCV_FOR_THREADS_NUM='1',MPLBACKEND='Agg')
                with (folder/'test.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'job','--epoch',str(ep),'--dataset',ds],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children[child.pid]=child
                active[group]=dict(epoch=ep,dataset=ds,pid=child.pid,physical_gpus=list(group))
        state.update(completed_jobs=len(completed),active_jobs=list(active.values()),pending_jobs=len(pending),status='testing' if active else 'waiting_free_gpus')
        if failures:state.update(status='draining_after_failure' if active else 'failed',error=str(failures))
        elif len(completed)==state['total_jobs']:state['status']='completed'
        render(state)
        if state['status'] in ('completed','failed'):return
        time.sleep(20)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['suite','job']);p.add_argument('--epoch',type=int);p.add_argument('--dataset');args=p.parse_args()
    if args.mode=='job':run_job(args)
    else:suite()
