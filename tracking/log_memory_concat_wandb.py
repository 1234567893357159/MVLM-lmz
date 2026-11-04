"""Durable local W&B backup and independently retrying live W&B upload."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, payload):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload,indent=2)+'\n');temp.replace(path)


def run_online_worker(command, env):
    # The SDK can disable upload permanently while run.log keeps succeeding.
    # Restart that client from the durable queue rather than waiting for exit.
    child=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    for line in child.stdout:
        print(line,end='',flush=True)
        if 'Fatal error while uploading data' in line:
            print('Restarting online W&B client after fatal upload error; local queue retained.',flush=True)
            try:
                os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=10)
            return 1
    return child.wait()


def worker(out,mode):
    import wandb
    spec=json.loads((out/'run_spec.json').read_text())
    meta=spec['wandb']
    folder=out/('wandb_'+mode);folder.mkdir(exist_ok=True)
    os.environ['WANDB_MODE']=mode
    run=wandb.init(project=meta['project'],entity=meta['entity'],id=meta['id'],name=out.name,
        mode=mode,dir=str(folder),resume='allow' if mode=='online' else None,config=spec,
        settings=wandb.Settings(mode=mode,init_timeout=45))
    run.define_metric('global_step');run.define_metric('*',step_metric='global_step')
    run.define_metric('val/global_step');run.define_metric('val/*',step_metric='val/global_step')
    run.define_metric('baseline/*',step_metric='val/global_step')
    status=out/('wandb_status.json' if mode=='online' else 'wandb_offline_status.json')
    cursor=0
    write(status,dict(mode=mode,id=meta['id'],url=meta['url'],local_dir=run.dir,queued_step=0))
    try:
        while True:
            queue=out/'wandb_queue.jsonl'
            if queue.exists():
                with queue.open() as stream:
                    stream.seek(cursor)
                    while True:
                        position=stream.tell();line=stream.readline()
                        if not line or not line.endswith('\n'):
                            stream.seek(position);break
                        payload=json.loads(line);run.log(payload)
                        step=payload.get('global_step')
                        if step is not None:
                            write(status,dict(mode=mode,id=meta['id'],url=meta['url'],local_dir=run.dir,
                                queued_step=step,updated_at=dt.datetime.now(dt.timezone.utc).isoformat()))
                    cursor=stream.tell()
            state=json.loads((out/'training_status.json').read_text())
            run.summary['status']=state['status']
            if state['status'] in ('completed','failed'):
                summary=out/'wandb_summary.json'
                if summary.exists():run.summary.update(json.loads(summary.read_text()))
                run.finish(exit_code=0 if state['status']=='completed' else 1)
                write(status,dict(mode=mode,id=meta['id'],url=meta['url'],completed=True,
                    training_status=state['status'],local_dir=run.dir))
                return
            time.sleep(5)
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=['online','offline'],required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker(args.out,args.mode)
    else:
        for attempt in range(1,1441):
            env=dict(os.environ,WANDB_MODE=args.mode,WANDB_CONSOLE='off',WANDB_HTTP_TIMEOUT='15')
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--out',str(args.out),
                     '--mode',args.mode,'--worker']
            returncode=(run_online_worker(command,env) if args.mode=='online'
                        else subprocess.run(command,env=env).returncode)
            if returncode==0:break
            payload=dict(mode='pending_'+args.mode,id=json.loads((args.out/'run_spec.json').read_text())['wandb']['id'],
                         attempt=attempt,last_exit_code=returncode,
                         local_queue=str(args.out/'wandb_queue.jsonl'))
            write(args.out/('wandb_status.json' if args.mode=='online' else 'wandb_offline_status.json'),payload)
            time.sleep(30)
