"""Check this continuation's online metrics and reconcile only after it exits."""
from pathlib import Path
import datetime as dt
import json
import os
import subprocess
import sys
import time

OUT=Path('/root/MVLM/output/paper_continuation_e60_e70_20261102_run01')
META=OUT/'wandb_tracking/train/mvlm/mvlm_TF_continue_e60_e70/run.json'


def save(data):
    data['checked_at']=dt.datetime.now(dt.timezone.utc).isoformat()
    p=OUT/'wandb_sync_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)


def main():
    import wandb
    attempts=0
    while True:
        if not META.exists():
            save(dict(status='waiting_for_run_metadata'))
            time.sleep(15);continue
        meta=json.loads(META.read_text());train=json.loads((OUT/'training_status.json').read_text())
        report=dict(status='checking',run_url=meta['url'],run_id=meta['id'],training_status=train['status'],
                    local_step=train.get('approximate_global_step',75000),sync_attempts=attempts)
        try:
            run=wandb.Api(timeout=20).run(f"{meta['entity']}/{meta['project']}/{meta['id']}")
            summary=dict(run.summary);step=summary.get('global_step',0)
            report.update(online_state=run.state,online_global_step=step,online_summary_status=summary.get('status'),
                          status='online_current' if step>=report['local_step']-200 else 'online_lagging')
            if train['status']=='completed' and step>=87500 and run.state=='finished':
                report['status']='complete_and_synced';save(report);return
            if train['status'] in ('failed','stopped'):
                report['status']='training_ended_unsuccessfully';save(report);return
            if train['status']=='completed':
                folders=sorted(META.parent.glob('wandb/run-*'+meta['id']))
                if folders:
                    attempts+=1;report.update(status='completion_reconciliation',sync_attempts=attempts);save(report)
                    cmd=[sys.executable,'-m','wandb','sync','--include-online','--include-synced','--skip-console','--id',meta['id'],
                         '--project',meta['project'],'--entity',meta['entity'],str(folders[-1])]
                    with (OUT/'wandb_reconciliation.log').open('a') as log:
                        try:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=240);report['last_sync_exit_code']=r.returncode
                        except subprocess.TimeoutExpired:report['last_sync_exit_code']='timeout'
        except Exception as exc:
            report.update(status='online_unreachable_local_history_preserved',error_type=type(exc).__name__)
        save(report)
        time.sleep(300)

if __name__=='__main__':main()
