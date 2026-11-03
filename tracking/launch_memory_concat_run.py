"""Freeze code/config provenance and detach the already-authorized run supervisor."""
import argparse
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
import yaml
from lib.train.memory_run_utils import write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def launch(out):
    if (out/'run_spec.json').exists():
        raise RuntimeError('Run already frozen; inspect its supervisor before resuming')
    for rank in (0,1):
        audit=json.loads((out/'preflight_b40'/('rank%d_final_audit.json'%rank)).read_text())
        assert audit['status']=='passed' and audit['final_step']==10
        assert audit['frozen_encoder_text_unchanged'] and audit['bn_statistics_unchanged']
    source=out/'source';source.mkdir()
    for folder in ('lib','tracking','experiments'):
        shutil.copytree(ROOT/folder,source/folder,ignore=shutil.ignore_patterns(
            '__pycache__','node_modules','web','*.pyc'))
    files={str(path.relative_to(source)):sha(path) for path in source.rglob('*') if path.is_file()}
    write_json(out/'source_manifest.json',files)
    config=yaml.safe_load((out/'config.yaml').read_text())
    config['TRAIN']['SCHEDULER']['TYPE']='memory_step4000'
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    identifier=uuid.uuid4().hex[:8]
    wandb=dict(id=identifier,project='mvlm-memory',entity='2896287267-hit',
               url='https://wandb.ai/2896287267-hit/mvlm-memory/runs/'+identifier)
    frozen_paths=[out/'config.yaml',out/'split/validation_manifest.json',out/'split/quick_manifest.json']
    frozen_paths += [Path(path) for path in config['DATA']['TRAIN']['SEQUENCE_LISTS'].values()]
    spec=dict(name=out.name,source=str(source),repository=str(ROOT),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch='codex/xmem-concat-center',source_manifest_sha256=sha(out/'source_manifest.json'),
        config_sha256=sha(out/'config.yaml'),input_hashes={str(path):sha(path) for path in frozen_paths},
        baseline_checkpoint=config['MODEL']['PRETRAINED_MODEL'],
        data=json.loads((out/'data_preparation.json').read_text()),
        physical_gpus=[4,5],batch_per_rank=40,global_batch=80,clip_frames=4,
        optimizer_updates=4000,sampled_clips=320000,search_frame_draws=1280000,
        optimizer='AdamW',memory_peak_lr=1e-4,center_peak_lr=1e-5,weight_decay=1e-4,
        schedule=dict(warmup_updates=200,decay_after_update=3000,decay_factor=.1),
        frozen=dict(encoder=True,text_encoder_including_projection=True,center_bn_running_stats=True),
        precision='FP32',accumulation_steps=1,seed=20261018,wandb=wandb,
        validation=dict(quick30_at=[1000,2000,3000,4000],full150_at=[2000,4000],
            extra_full='best quick checkpoint if different',workers=8,
            baseline_timing='fixed before training; evaluated after first segment on the same two GPUs'),
        known_limits=['Adaptation holdout may have been seen by official base pretraining',
                      'Four-frame clips do not fully exercise long-capacity memory',
                      'Worker/prefetch state is not saved, so resume is not bitwise replay'])
    write_json(out/'run_spec.json',spec)
    command=[sys.executable,'-u',str(source/'tracking/supervise_memory_concat_run.py'),'--out',str(out)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',WANDB_CONSOLE='off',PYTHONUNBUFFERED='1')
    with (out/'supervisor.log').open('a',buffering=1) as stream:
        child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,
                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'supervisor_launch.json',dict(pid=child.pid,command=command,cwd=str(source),wandb=wandb))
    print(json.dumps(dict(supervisor_pid=child.pid,wandb=wandb,out=str(out)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    launch(parser.parse_args().out.resolve())
