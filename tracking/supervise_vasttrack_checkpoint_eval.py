"""Queue fixed VastTrack checkpoint validation after the current test suite."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/vasttrack_checkpoint_validation_20261028_run01'
PRIOR=ROOT/'output/vasttrack_language_eval_20261028_run01/suite_status.json'
CKPT=ROOT/'output/vasttrack_baseline_20261026_run01/checkpoints/train/mvlm/mvlm_TF_vasttrack'
CONFIG='vasttrack_holdout_strict_common_20261028'
EPOCHS=(10,20,30,40,50,60)

def main():
    status={'status':'waiting_for_test_suite','started_at':time.time(),'pid':os.getpid(),'epochs':list(EPOCHS),
            'completed':[],'prior_suite':str(PRIOR),'current':None,
            'parameters':{'ALPHA_CORR':.5,'TAU':.03,'OMEGA':.1,'LAMBDA':.4,'PSI_OUT':.1},
            'parameter_note':'fixed across all epochs; transferred from previous LaSOT best, not tuned on VastTrack'}
    def save():
        tmp=OUT/'queue_status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(OUT/'queue_status.json')

    def alive(pid):
        try:
            stat=Path('/proc/%s/stat'%pid).read_text().split()
            return len(stat)>2 and stat[2]!='Z'
        except (FileNotFoundError,ProcessLookupError):return False

    def run(epoch,smoke=False):
        label='smoke' if smoke else 'epoch_%04d'%epoch
        folder=OUT/label;folder.mkdir(exist_ok=True)
        weight=CKPT/('MVLM_ep%04d.pth.tar'%epoch)
        exp='vasttrack_holdout_strict_20261028_'+label
        cmd=[sys.executable,'-u','-m','tracking.strict_vasttrack_holdout_eval','--config',CONFIG,'--weights',str(weight),
             '--exp-id',exp,'--out',str(folder),'--threads',str(1 if smoke else 16),'--gpus',str(1 if smoke else 8)]
        if smoke:cmd+=['--limit','1']
        status.update(status='running',current=label,command=cmd);save()
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        with (folder/'test.log').open('w') as log:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        info=json.loads((folder/'status.json').read_text())
        if result.returncode or info.get('status')!='completed':
            raise RuntimeError('%s failed (%s): %s'%(label,result.returncode,info.get('error')))
        if smoke:status['smoke_result']=str(folder)
        else:status['completed'].append({'epoch':epoch,'metrics':info['metrics'],'sequences':info['sequences'],'frames':info['frames'],'valid_frames':info['valid_frames'],'result':str(folder)})
        save()

    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'queue_status.json').exists():
        previous=json.loads((OUT/'queue_status.json').read_text())
        if previous.get('status') != 'waiting_for_test_suite' or previous.get('completed'):
            raise RuntimeError('Existing validation queue has already started: '+str(OUT))
        (OUT/'queue_status_before_priority_change.json').write_text(json.dumps(previous,indent=2))
    for epoch in EPOCHS:
        if not (CKPT/('MVLM_ep%04d.pth.tar'%epoch)).is_file():raise FileNotFoundError('Missing checkpoint %d'%epoch)
    save()
    try:
        while True:
            prior=json.loads(PRIOR.read_text())
            status['prior_status']=prior.get('status');status['prior_current_dataset']=prior.get('current_dataset');status['prior_current_model']=prior.get('current_model');save()
            if prior.get('status') in ('completed','failed','cancelled'):
                break
            if not alive(prior.get('pid')):
                raise RuntimeError('Existing test suite stopped without recording a terminal status')
            time.sleep(30)
        run(10,smoke=True)
        for epoch in EPOCHS:run(epoch)
        status.update(validation_completed=True, status='running_deferred_lasot', current='lasot_epoch60');save()
        # Preserve the originally requested trained-model LaSOT test after the
        # six validation checkpoints. No further official-weight tests are run.
        folder=ROOT/'output/vasttrack_language_eval_20261028_run01/lasot_trained_epoch60'
        folder.mkdir(parents=True,exist_ok=True)
        cfg='vasttrack_e60_strict_lasot_20261028'
        exp=cfg+'_trained_epoch60'
        cmd=[sys.executable,'-u','-m','tracking.strict_language_eval','--dataset','lasot','--config',cfg,
             '--exp-id',exp,'--weights',str(CKPT/'MVLM_ep0060.pth.tar'),'--out',str(folder),'--threads','16','--gpus','8']
        status['command']=cmd;save()
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        with (folder/'test.log').open('w') as log:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        info=json.loads((folder/'status.json').read_text())
        if result.returncode or info.get('status')!='completed':
            raise RuntimeError('Deferred LaSOT evaluation failed: '+repr(info.get('error')))
        status['deferred_lasot']={'metrics':info['metrics'],'sequences':info['sequences'],'frames':info['frames'],'result':str(folder)}
        status.update(status='completed',finished_at=time.time(),current=None);save()
    except BaseException as exc:
        status.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),finished_at=time.time());save();raise


if __name__ == '__main__':
    main()
