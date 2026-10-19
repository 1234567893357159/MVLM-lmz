"""Four-thousand-step head-unfreezing experiment with deferred validation."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
REFERENCE=Path('/root/MVLM/output/memory_three_custom_20261018_run01')
CONFIG='mvlm_TF_memory_head_joint_short_20261019'


def run(out):
    import torch
    out.mkdir(parents=True,exist_ok=True)
    with (out/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'status':'starting','planned_optimizer_steps':4000,'supervisor_pid':os.getpid(),
               'started_at':datetime.now(timezone.utc).isoformat(),'config':CONFIG,
               'monitor_url':'https://wandb.ai/2896287267-hit/mvlm-memory/runs/hafdntia',
               'world_size':8,'per_gpu_batch':2,'effective_batch':16,
               'evaluation_schedule':'After continuous training, evaluate saved 1000/2000/3000/4000-step checkpoints.'}
        def save():
            p=out/'training_status.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(p)
        save();wb=None
        try:
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                     NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',MVLM_DISTRIBUTED_TIMEOUT_SECONDS='10800')
            command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                     '--module','lib.train.run_training','--script','mvlm','--config',CONFIG,
                     '--save_dir',str(out),'--seed','20261018']
            with (out/'train.log').open('a') as log:
                p=subprocess.Popen(command,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state.update(status='running',training_pid=p.pid,command=command);save();code=p.wait()
            if code:raise RuntimeError('Training subprocess exit '+str(code))
            checkpoints=out/'checkpoints/train/mvlm'/CONFIG
            final=checkpoints/'MVLM_ep0004.pth.tar'
            ckpt=torch.load(final,map_location='cpu');assert ckpt['global_step']==4000
            base=torch.load('/root/MVLM/models/MVLM_TF.pth.tar',map_location='cpu')['net']
            frozen_changed=[k for k,v in base.items() if not k.startswith('head.') and not torch.equal(v,ckpt['net'][k])]
            assert not frozen_changed,frozen_changed[:5]
            changed_head=[k for k,v in base.items() if k.startswith('head.') and not torch.equal(v,ckpt['net'][k])]
            assert changed_head,'Head did not update'
            assert ckpt['net']['memory_fusion.output_proj.weight'].norm()>0
            assert all(bool(torch.isfinite(v).all()) for v in ckpt['net'].values() if v.is_floating_point())
            state.update(status='validating',completed_optimizer_steps=4000,training_finished_at=datetime.now(timezone.utc).isoformat(),
                         final_checkpoint=str(final),changed_head_tensors=len(changed_head),frozen_encoder_text_changed=0);save()
            del ckpt,base
            import wandb
            metadata=out/'wandb_tracking/train/mvlm'/CONFIG/'run.json';saved=json.loads(metadata.read_text())
            wb=wandb.init(project=saved['project'],entity=saved['entity'],id=saved['id'],resume='must',dir=str(metadata.parent))
            wb.summary['status']='validating_saved_checkpoints'
            wb.define_metric('val/global_step');wb.define_metric('val/*',step_metric='val/global_step')
            wb.define_metric('baseline/*',step_metric='val/global_step')
            wb.define_metric('reference/global_step');wb.define_metric('reference/frozen_head/*',step_metric='reference/global_step')
            baseline=json.loads((REFERENCE/'validation/baseline/metrics.json').read_text())
            folder=out/'validation';folder.mkdir(exist_ok=True)
            (folder/'baseline_reference.json').write_text(json.dumps({'source':str(REFERENCE/'validation/baseline/metrics.json'),
                       'metrics':baseline['metrics'],'mean_AUC':baseline['mean_AUC']},indent=2)+'\n')
            all_metrics=[]
            for epoch in range(1,5):
                step=epoch*1000;dest=folder/('epoch_%04d'%epoch);dest.mkdir(exist_ok=True)
                state.update(validating_epoch=epoch,validating_step=step);save()
                weights=checkpoints/('MVLM_ep%04d.pth.tar'%epoch)
                if not (dest/'metrics.json').exists():
                    cmd=[sys.executable,'-u',str(ROOT/'tracking/evaluate_memory_custom.py'),
                         '--manifest',str(REFERENCE/'split/manifest.json'),'--weights',str(weights),
                         '--out',str(dest),'--config',CONFIG,'--gpus','8','--workers','16']
                    with (dest/'evaluate.log').open('a') as log:
                        subprocess.run(cmd,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                result=json.loads((dest/'metrics.json').read_text());assert result['sequences']==159 and not result['engineering_subset']
                payload={'global_step':4000,'val/global_step':step,'val/mean_AUC':result['mean_AUC'],'baseline/mean_AUC':baseline['mean_AUC']}
                for d,metrics in result['metrics'].items():
                    for k,v in metrics.items():payload['val/'+d+'/'+k]=v
                wb.log(payload)
                if epoch in [2,4]:
                    ref=json.loads((REFERENCE/('validation/epoch_%04d'%epoch)/'metrics.json').read_text())
                    wb.log({'global_step':4000,'reference/global_step':step,'reference/frozen_head/mean_AUC':ref['mean_AUC']})
                entry={'epoch':epoch,'global_step':step,'metrics':result['metrics'],'mean_AUC':result['mean_AUC'],'checkpoint':str(weights)}
                all_metrics.append(entry);best=max(all_metrics,key=lambda x:x['mean_AUC'])
                (folder/'best.json').write_text(json.dumps(best,indent=2)+'\n')
                if best is entry:shutil.copy2(weights,folder/'best.pth.tar')
                (out/'validation_summary.json').write_text(json.dumps({'baseline_mean_AUC':baseline['mean_AUC'],'checkpoints':all_metrics,'best':best},indent=2)+'\n')
                wb.summary['best_val_mean_AUC']=best['mean_AUC'];wb.summary['best_val_step']=best['global_step']
            state.update(status='completed',best=best,finished_at=datetime.now(timezone.utc).isoformat());save()
            wb.summary['status']='completed';wb.finish();wb=None
        except BaseException as exc:
            state.update(status='failed',error=repr(exc),finished_at=datetime.now(timezone.utc).isoformat());save()
            if wb:wb.summary['status']='failed';wb.finish(exit_code=1)
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    run(p.parse_args().run_dir.resolve())
