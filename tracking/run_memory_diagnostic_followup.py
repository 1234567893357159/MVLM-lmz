"""Resume-safe follow-up: control reproduction then two full validation checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/root/MVLM/output/memory_three_custom_20261018_run01/diagnostics')
state={'stage':'waiting_for_screen','status':'running'}
def save(): (OUT/'followup_status.json').write_text(json.dumps(state,indent=2)+'\n')
def run(label,args):
    state.update(stage=label);save()
    with (OUT/(label+'.log')).open('a') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'tracking/diagnose_memory_training.py')]+args,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,check=True)
try:
    save()
    while not (OUT/'screen/summary.json').exists():time.sleep(5)
    run('screen_control',['--out',str(OUT/'screen'),'--workers','8','--limit','512'])
    for variant in ['quarter','strict_quality']:
        if not (OUT/('full_'+variant)/'summary.json').exists():
            run('full_'+variant,['--out',str(OUT/('full_'+variant)),'--workers','8','--limit','0','--full-variant',variant])
    state.update(status='completed',stage='complete');save()
except BaseException as exc:
    state.update(status='failed',error=repr(exc));save();raise
