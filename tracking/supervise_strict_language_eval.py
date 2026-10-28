"""Run fixed strict language-only comparisons, recording each process exit."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/vasttrack_language_eval_20261028_run01'
trained=ROOT/'output/vasttrack_baseline_20261026_run01/checkpoints/train/mvlm/mvlm_TF_vasttrack/MVLM_ep0060.pth.tar'
official=ROOT/'models/MVLM_TF.pth.tar'

def main():
    if (OUT / 'suite_status.json').exists():
        raise FileExistsError('Historical suite already exists: ' + str(OUT))
    OUT.mkdir(parents=True, exist_ok=True)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    status={'status':'running','started_at':time.time(),'pid':os.getpid(),'completed':[]}
    def save():
     p=OUT/'suite_status.tmp';p.write_text(json.dumps(status,indent=2));p.replace(OUT/'suite_status.json')
    save()
    try:
     for ds in ('otb99_lang','tnl2k','lasot'):
      config='vasttrack_e60_strict_'+ds+'_20261028'
      for label,weights in (('trained_epoch60',trained),('official_baseline',official)):
       exp=config+'_'+label
       folder=OUT/(ds+'_'+label);folder.mkdir(exist_ok=True)
       cmd=[sys.executable,'-u','-m','tracking.strict_language_eval','--dataset',ds,'--config',config,'--exp-id',exp,'--weights',str(weights),'--out',str(folder),'--threads','16','--gpus','8']
       status.update(current_dataset=ds,current_model=label,current_output=str(folder),current_command=cmd);save()
       with (folder/'test.log').open('w') as log:
        result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
       info=json.loads((folder/'status.json').read_text())
       if result.returncode or info.get('status')!='completed':raise RuntimeError(f'{ds} {label} failed ({result.returncode}): {info.get("error")}')
       status['completed'].append({'dataset':ds,'model':label,'metrics':info['metrics'],'sequences':info['sequences'],'frames':info['frames'],'excluded':info['excluded'],'result':str(folder)})
       save()
     status.update(status='completed',finished_at=time.time());save()
    except BaseException as exc:
     status.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),finished_at=time.time());save();raise


if __name__ == '__main__':
    main()
