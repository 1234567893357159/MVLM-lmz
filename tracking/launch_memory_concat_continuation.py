"""Resume the completed 4,000-step experiment to 6,000 without changing its policy."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from lib.train.memory_run_utils import write_json


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def launch(parent, out):
    if out.exists():
        raise RuntimeError('Continuation directory already exists; inspect it before resuming')
    previous=json.loads((parent/'run_spec.json').read_text())
    status=json.loads((parent/'training_status.json').read_text())
    if status['status']!='completed' or status['global_step']!=4000:
        raise RuntimeError('Expected the completed 4000-step parent run')
    checkpoint=parent/'checkpoints/MVLM_step004000.pth.tar'
    for filename,expected in previous['input_hashes'].items():
        if sha(filename)!=expected:
            raise RuntimeError('Parent experiment input changed: '+filename)
    weights=torch.load(str(checkpoint),map_location='cpu')
    states=weights['optimizer']['state']
    steps=[float(state['step']) for state in states.values() if 'step' in state]
    if weights['global_step']!=4000 or not steps or min(steps)!=4000 or max(steps)!=4000:
        raise RuntimeError('Expected complete AdamW state at update 4000')
    if len(weights.get('rng_states',[]))!=2:
        raise RuntimeError('Expected both saved rank RNG states')
    for group in weights['optimizer']['param_groups']:
        if abs(group['lr']-group['peak_lr']*.1)>1e-12:
            raise RuntimeError('Parent checkpoint does not use the expected tail LR')
    checkpoint_audit=dict(global_step=4000,optimizer_state_entries=len(states),
        optimizer_step_min=min(steps),optimizer_step_max=max(steps),rng_ranks=2,
        groups=[{key:value for key,value in group.items() if key!='params'}
                for group in weights['optimizer']['param_groups']])
    del weights,states
    resources=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'],text=True)
    available={int(row.split(',')[0]):int(row.split(',')[1]) for row in resources.splitlines()}
    if any(available.get(index,999999)>256 for index in (4,5)):
        raise RuntimeError('Physical GPU4 and GPU5 must be free before continuation')

    out.mkdir(parents=True)
    shutil.copy2(parent/'config.yaml',out/'config.yaml')
    shutil.copytree(parent/'split',out/'split')
    shutil.copy2(parent/'data_preparation.json',out/'data_preparation.json')
    control=out/'validation/step004000_full150/metrics.json'
    control.parent.mkdir(parents=True)
    shutil.copy2(parent/'validation/step004000_full150/metrics.json',control)
    baseline=parent/'validation/baseline_full150/metrics.json'
    source=out/'source'
    source.mkdir()
    # Copy model, actor, sampler and evaluator from the actual parent snapshot.
    # Only run control, budget handling and audit code are replaced.
    for folder in ('lib','tracking','experiments'):
        shutil.copytree(Path(previous['source'])/folder,source/folder,
            ignore=shutil.ignore_patterns('__pycache__','node_modules','web','*.pyc'))
    replacements=('lib/train/memory_run_utils.py','tracking/train_memory_concat_run.py',
        'tracking/supervise_memory_concat_run.py','tracking/launch_memory_concat_continuation.py',
        'tracking/log_memory_concat_wandb.py')
    for filename in replacements:
        shutil.copy2(ROOT/filename,source/filename)
    manifest={str(path.relative_to(source)):sha(path) for path in source.rglob('*') if path.is_file()}
    write_json(out/'source_manifest.json',manifest)
    identifier=uuid.uuid4().hex[:8]
    wandb=dict(id=identifier,entity=previous['wandb']['entity'],project=previous['wandb']['project'])
    wandb['url']='https://wandb.ai/{entity}/{project}/runs/{id}'.format(**wandb)
    checkpoint_sha=sha(checkpoint)
    spec=dict(previous)
    spec.update(name=out.name,source=str(source),repository=str(ROOT),
        created_at=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_manifest_sha256=sha(out/'source_manifest.json'),
        source_changes_from_parent=list(replacements),
        config_sha256=sha(out/'config.yaml'),start_step=4000,optimizer_updates=6000,
        additional_optimizer_updates=2000,sampled_clips=480000,search_frame_draws=1920000,
        additional_sampled_clips=160000,additional_search_frame_draws=640000,wandb=wandb,
        validation=dict(full150_at=[5000,6000],workers=8,control_step=4000,
            selection='best of the fixed 4000 control and the 5000/6000 checkpoints'),
        continuation=dict(parent_out=str(parent),parent_wandb_url=previous['wandb']['url'],
            parent_git_commit=previous['git_commit'],checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_sha,optimizer_audit=checkpoint_audit,
            control_mean_AUC=json.loads(control.read_text())['mean_AUC'],
            baseline_mean_AUC=json.loads(baseline.read_text())['mean_AUC'],
            policy='Only extend update budget; restore model, AdamW and both rank RNG states',
            learning_rates=dict(memory=1e-5,center=1e-6)))
    spec.pop('startup_attempts',None)
    frozen_paths=[out/'config.yaml',out/'split/validation_manifest.json',out/'split/quick_manifest.json',control]
    spec['input_hashes']=dict(previous['input_hashes'])
    spec['input_hashes'].update({str(path):sha(path) for path in frozen_paths})
    spec['input_hashes'][str(checkpoint)]=checkpoint_sha
    write_json(out/'run_spec.json',spec)
    write_json(out/'parent_checkpoint_audit.json',checkpoint_audit)
    write_json(out/'latest_checkpoint.json',dict(step=4000,path=str(checkpoint),sha256=checkpoint_sha))
    command=[sys.executable,'-u',str(source/'tracking/supervise_memory_concat_run.py'),'--out',str(out)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',WANDB_CONSOLE='off',PYTHONUNBUFFERED='1')
    with (out/'supervisor.log').open('a',buffering=1) as stream:
        child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'supervisor_launch.json',dict(pid=child.pid,command=command,cwd=str(source),wandb=wandb))
    print(json.dumps(dict(supervisor_pid=child.pid,wandb=wandb,out=str(out)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    launch(args.parent.resolve(),args.out.resolve())
