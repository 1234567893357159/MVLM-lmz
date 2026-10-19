"""Detached supervisor for one authorized 20k-step custom memory experiment."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(run_dir):
    run_dir.mkdir(parents=True,exist_ok=True)
    with (run_dir/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status_path=run_dir/'training_status.json'
        status={'status':'starting','supervisor_pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),
                'planned_optimizer_steps':20000,'world_size':8,'per_gpu_batch':2,'effective_batch':16,
                'monitor_url':'https://wandb.ai/2896287267-hit/mvlm-memory/runs/nie0ya10',
                'save_dir':str(run_dir),'config':'mvlm_TF_memory_three_custom'}
        def save():
            temp=status_path.with_suffix('.tmp');temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(status_path)
        save()
        command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                 '--module','lib.train.run_training','--script','mvlm','--config',status['config'],
                 '--save_dir',str(run_dir),'--seed','20261018']
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',MVLM_DISTRIBUTED_TIMEOUT_SECONDS='10800')
        with (run_dir/'train.log').open('a') as log:
            process=subprocess.Popen(command,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            status.update(status='running',training_pid=process.pid,command=command);save()
            result=process.wait()
        status.update(exit_code=result,finished_at=datetime.now(timezone.utc).isoformat())
        if result==0:
            import torch
            final=run_dir/'checkpoints/train/mvlm/mvlm_TF_memory_three_custom/MVLM_ep0020.pth.tar'
            try:
                checkpoint=torch.load(final,map_location='cpu')
                if checkpoint['global_step']!=20000: raise ValueError('Unexpected final update count')
                if not (run_dir/'validation/best.json').exists(): raise ValueError('No best validation checkpoint')
                status.update(status='completed',completed_optimizer_steps=checkpoint['global_step'],final_checkpoint=str(final))
            except Exception as exc: status.update(status='failed',error=repr(exc))
        else: status['status']='failed'
        save()
        if status['status']!='completed': raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    main(p.parse_args().run_dir.resolve())
