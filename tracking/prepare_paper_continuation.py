"""Prepare an isolated E60->E70 continuation and one-update restore check."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import yaml

ROOT=Path('/root/MVLM')
PARENT=ROOT/'output/paper_reproduction_20261030_run01'
OUT=ROOT/'output/paper_continuation_e60_e70_20261102_run01'
PROBE=ROOT/'output/paper_continuation_preflight_20261102_run01'
NAME='mvlm_TF_continue_e60_e70'
PROBE_NAME='mvlm_TF_continue_restore_probe'


def main():
    assert not OUT.exists() and not PROBE.exists(), 'Refusing to overwrite continuation outputs'
    OUT.mkdir();PROBE.mkdir()
    source=OUT/'source'
    shutil.copytree(PARENT/'source',source,ignore=shutil.ignore_patterns('__pycache__','*.pyc','node_modules','.git'))
    config=yaml.safe_load((PARENT/'config.yaml').read_text())
    config['TRAIN']['EPOCH']=70
    config['WANDB'].update(NAME='paper-continue-e60-e70-b80-2gpu-20261102',ENABLED=True,MODE='online')
    for folder,name,epochs,samples,enabled in ((OUT,NAME,70,100000,True),(PROBE,PROBE_NAME,61,80,False)):
        c=yaml.safe_load(yaml.safe_dump(config));c['TRAIN']['EPOCH']=epochs;c['DATA']['TRAIN']['SAMPLE_PER_EPOCH']=samples;c['WANDB']['ENABLED']=enabled
        text=yaml.safe_dump(c,sort_keys=False)
        (folder/'config.yaml').write_text(text)
        (source/'experiments/mvlm'/f'{name}.yaml').write_text(text)
        (ROOT/'experiments/mvlm'/f'{name}.yaml').write_text(text)
        dest=folder/'checkpoints/train/mvlm'/name;dest.mkdir(parents=True)
        shutil.copy2(PARENT/'checkpoints/train/mvlm/mvlm_TF_paper_reproduction/MVLM_ep0060.pth.tar',dest/'MVLM_ep0060.pth.tar')
    shutil.copytree(PARENT/'split',OUT/'split')
    p=source/'lib/train/trainers/base_trainer.py';s=p.read_text()
    anchor='    def load_state_dict('
    pos=s.index(anchor)
    audit='''        # Continuation-only audit after restoring model and AdamW state.
        import json
        from pathlib import Path
        assert self.epoch >= 60 and self.global_step >= 75000
        if self.epoch == 60:
            assert self.global_step == 75000
        assert [group['lr'] for group in self.optimizer.param_groups] == [5e-5, 5e-6]
        assert self.lr_scheduler.last_epoch == self.epoch
        current = net.state_dict()
        names = ['encoder.body.cls_token', 'text_encoder.text_proj.weight', 'head.conv1_ctr.0.weight']
        checks = {name: bool(torch.equal(current[name].detach().cpu(), checkpoint_dict['net'][name])) for name in names}
        assert all(checks.values()), checks
        restored = self.optimizer.state_dict()
        expected = checkpoint_dict['optimizer']
        assert len(restored['state']) == len(expected['state']) > 0
        moment_checks = []
        for key in list(expected['state'])[:2]:
            for field in ('exp_avg', 'exp_avg_sq'):
                equal = bool(torch.equal(restored['state'][key][field].detach().cpu(), expected['state'][key][field].detach().cpu()))
                assert equal, (key, field)
                moment_checks.append(dict(parameter_id=key, field=field, equal=equal))
        rank = self.settings.local_rank
        record = dict(checkpoint=checkpoint_path, restored_epoch=self.epoch, restored_global_step=self.global_step,
                      optimizer_lrs=[g['lr'] for g in self.optimizer.param_groups], optimizer_state_entries=len(restored['state']),
                      scheduler_last_epoch=self.lr_scheduler.last_epoch, parameter_checks=checks, optimizer_moment_checks=moment_checks,
                      first_expected_step=self.global_step+1, rank=rank,
                      note='RNG/data-loader states were not saved by parent; not bitwise continuous replay.')
        Path(self.settings.save_dir, f'resume_audit_rank{rank}.json').write_text(json.dumps(record, indent=2))
        print('CONTINUATION RESTORE AUDIT', json.dumps(record), flush=True)

'''
    s=s[:pos]+audit+s[pos:]
    s=s.replace('        return True\n\n        # Continuation-only audit','\n        # Continuation-only audit',1)
    s=s.replace("        print('CONTINUATION RESTORE AUDIT', json.dumps(record), flush=True)\n\n    def load_state_dict","        print('CONTINUATION RESTORE AUDIT', json.dumps(record), flush=True)\n        return True\n\n    def load_state_dict",1)
    p.write_text(s)
    p=source/'lib/train/train_script.py';s=p.read_text().replace("'initialization.json'","'pre_resume_initialization.json'");p.write_text(s)
    p=source/'lib/train/admin/wandb_logger.py';s=p.read_text();anchor="        config['monitoring']['seed'] = getattr(self.settings, 'base_seed', None)"
    s=s.replace(anchor,anchor+"\n        parent_record = Path(self.settings.save_dir) / 'continuation.json'\n        if parent_record.exists():\n            config['continuation'] = json.loads(parent_record.read_text())")
    p.write_text(s)
    p=ROOT/'tracking/supervise_continuation_training.py';s=(ROOT/'tracking/supervise_reproduction_training.py').read_text()
    s=s.replace("'Supervise one frozen-source MVLM training run on explicitly selected GPUs.'", "'Supervise one frozen-source MVLM continuation.'")
    s=s.replace("    env = os.environ.copy()", "    env = os.environ.copy()\n    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):\n        env.pop(key, None)\n    state.update(resumed_epoch=60, resumed_global_step=75000, epoch=60, approximate_global_step=75000, target_global_step=87500 if epochs==70 else 75001)")
    s=s.replace("approximate_global_step=(epoch - 1) * per_epoch + batch_index", "approximate_global_step=75000 + (epoch - 61) * per_epoch + batch_index")
    s=s.replace("    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):\n        env.pop(key, None)\n", '')
    p.write_text(s);shutil.copy2(p,source/'tracking'/p.name)
    parent_summary=json.loads((ROOT/'reports/reproduction/paper_training_results_20261101/summary.json').read_text())
    metadata=dict(prepared_at=dt.datetime.now(dt.timezone.utc).isoformat(),parent_run='2896287267-hit/mvlm-reproduction/7ln28ook',parent_output=str(PARENT),
                  parent_checkpoint=parent_summary['final_checkpoint'],parent_checkpoint_sha256=parent_summary['final_checkpoint_sha256'],
                  start_epoch=60,target_epoch=70,start_global_step=75000,target_global_step=87500,
                  physical_gpus=[4,5],world_size=2,batch_per_rank=40,global_batch=80,
                  resumed_learning_rates=[5e-5,5e-6],samples_per_epoch=100000,additional_samples=1000000,
                  optimizer='AdamW (state restored)',scheduler='Original StepLR, last_epoch restored to60; no decay during61-70',
                  rng_state_restored=False,monitoring_proxy_policy='Direct HTTPS unavailable; use the reachable inherited proxy and retain complete local W&B history.',
                  planned_evaluation_epochs=[65,70],source=str(source))
    (OUT/'continuation.json').write_text(json.dumps(metadata,indent=2))
    (OUT/'run_spec.json').write_text(json.dumps(metadata,indent=2))
    manifest={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')}
    (OUT/'source_sha256.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(output=str(OUT),probe=str(PROBE),source=str(source),config=NAME,probe_config=PROBE_NAME),indent=2))

if __name__=='__main__':main()
