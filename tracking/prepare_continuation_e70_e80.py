"""Prepare isolated E70->E80 training and E71-E80 evaluation outputs."""
from pathlib import Path
import copy
import hashlib
import json
import py_compile
import shutil
import datetime as dt
import sys

import torch
import yaml

ROOT = Path('/root/MVLM')
sys.path.insert(0, str(ROOT))
PARENT = ROOT / 'output/paper_continuation_e60_e70_20261102_run01'
OUT = ROOT / 'output/paper_continuation_e70_e80_20261103_run01'
EVAL_PARENT = ROOT / 'output/paper_continuation_eval_20261102_run01'
EVAL = ROOT / 'output/paper_continuation_eval_20261103_run01'
NAME = 'mvlm_TF_continue_e70_e80'


def replace(text, old, new):
    assert old in text, old
    return text.replace(old, new)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert not OUT.exists() and not EVAL.exists(), 'Refusing to overwrite outputs'
    checkpoint = PARENT / 'checkpoints/train/mvlm/mvlm_TF_continue_e60_e70/MVLM_ep0070.pth.tar'
    state = torch.load(checkpoint, map_location='cpu')
    assert state['epoch'] == 70 and state['global_step'] == 87500
    assert [g['lr'] for g in state['optimizer']['param_groups']] == [5e-5, 5e-6]
    assert len(state['optimizer']['state']) == 622
    assert all(not v.is_floating_point() or torch.isfinite(v).all() for v in state['net'].values())
    parent_hash = sha(checkpoint)
    del state
    OUT.mkdir(); EVAL.mkdir()
    source = OUT / 'source'
    shutil.copytree(PARENT / 'source', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    config = yaml.safe_load((PARENT / 'config.yaml').read_text())
    original = copy.deepcopy(config)
    config['TRAIN']['EPOCH'] = 80
    config['WANDB']['NAME'] = 'paper-continue-e70-e80-b80-2gpu-20261103'
    check = copy.deepcopy(config)
    check['TRAIN']['EPOCH'] = original['TRAIN']['EPOCH']
    check['WANDB']['NAME'] = original['WANDB']['NAME']
    assert check == original
    text = yaml.safe_dump(config, sort_keys=False)
    for path in (OUT / 'config.yaml', source / f'experiments/mvlm/{NAME}.yaml', ROOT / f'experiments/mvlm/{NAME}.yaml'):
        path.write_text(text)
    dest = OUT / f'checkpoints/train/mvlm/{NAME}/MVLM_ep0070.pth.tar'
    dest.parent.mkdir(parents=True)
    shutil.copy2(checkpoint, dest)
    assert sha(dest) == parent_hash
    shutil.copytree(PARENT / 'split', OUT / 'split')

    path = source / 'lib/train/trainers/base_trainer.py'
    text = replace(path.read_text(), 'assert self.epoch >= 60 and self.global_step >= 75000', 'assert self.epoch >= 70 and self.global_step >= 87500')
    text = replace(text, 'if self.epoch == 60:\n            assert self.global_step == 75000', 'if self.epoch == 70:\n            assert self.global_step == 87500')
    path.write_text(text)

    text = (ROOT / 'tracking/supervise_continuation_training.py').read_text()
    text = replace(text, 'resumed_epoch=60, resumed_global_step=75000, epoch=60, approximate_global_step=75000, target_global_step=87500 if epochs==70 else 75001', 'resumed_epoch=70, resumed_global_step=87500, epoch=70, approximate_global_step=87500, target_global_step=100000')
    text = replace(text, 'approximate_global_step=75000 + (epoch - 61) * per_epoch + batch_index', 'approximate_global_step=87500 + (epoch - 71) * per_epoch + batch_index')
    supervisor = source / 'tracking/supervise_continuation_e70_e80.py'
    supervisor.write_text(text)

    text = (ROOT / 'tracking/monitor_continuation_wandb.py').read_text()
    text = text.replace(str(PARENT), str(OUT)).replace('mvlm_TF_continue_e60_e70', NAME)
    text = replace(text, "get('approximate_global_step',75000)", "get('approximate_global_step',87500)")
    text = replace(text, 'step>=87500', 'step>=100000')
    monitor = source / 'tracking/monitor_continuation_e70_e80.py'
    monitor.write_text(text)

    (EVAL / 'source/tracking').mkdir(parents=True)
    shutil.copy2(EVAL_PARENT / 'source/tracking/evaluate_checkpoint_grid.py', EVAL / 'source/tracking/evaluate_checkpoint_grid.py')
    (EVAL / 'source/lib').symlink_to(EVAL_PARENT / 'source/lib', target_is_directory=True)
    shutil.copy2(EVAL_PARENT / 'configuration_manifest.json', EVAL / 'configuration_manifest.json')
    manifest = json.loads((EVAL / 'configuration_manifest.json').read_text())
    for ds, record in manifest.items():
        assert sha(ROOT / f'experiments/mvlm/papergrid_20261101_{ds}.yaml') == record['sha256']
    text = (ROOT / 'tracking/continuation_checkpoint_eval.py').read_text()
    text = replace(text, "TRAIN=ROOT/'output/paper_continuation_e60_e70_20261102_run01'", "TRAIN=ROOT/'output/paper_continuation_e70_e80_20261103_run01'")
    text = replace(text, "OUT=ROOT/'output/paper_continuation_eval_20261102_run01'", "OUT=ROOT/'output/paper_continuation_eval_20261103_run01'")
    text = text.replace('mvlm_TF_continue_e60_e70', NAME)
    text = replace(text, "BASE=ROOT/'output/paper_checkpoint_grid_20261101_run01'", "BASE=ROOT/'output/paper_checkpoint_grid_20261101_run01'\nPREVIOUS=ROOT/'output/paper_continuation_eval_20261102_run01'")
    text = replace(text, "REPORT=ROOT/'reports/reproduction/PAPER_CONTINUATION_TESTS_20261102.md'", "REPORT=ROOT/'reports/reproduction/PAPER_CONTINUATION_TESTS_20261103.md'")
    text = replace(text, 'EPOCHS=tuple(range(61,71))', 'EPOCHS=tuple(range(71,81))')
    text = replace(text, 'for ep in (56,60)+EPOCHS:', 'for ep in (56,60,66,70)+EPOCHS:')
    text = replace(text, '(BASE if ep in (56,60) else OUT)', '(BASE if ep<=60 else PREVIOUS if ep<=70 else OUT)')
    text = text.replace('E60之后续训：固定参数测试', 'E70之后续训：固定参数测试')
    text = replace(text, '使用空闲物理GPU4、5和6、7两组并行评测', '使用空闲物理GPU6、7评测；GPU4、5用于训练')
    text = text.replace('E61～E70', 'E71～E80').replace('papercontinue_20261102_', 'papercontinue_20261103_')
    text = replace(text, 'groups=((4,5),(6,7));', 'groups=((6,7),);')
    text = replace(text, 'physical_gpus=[4,5,6,7]', 'physical_gpus=[6,7]')
    text = replace(text, 'E56为此前测试引导的高分对照，E60为原配方预先指定最终模型', 'E56、E66为此前探索性高分对照，E60、E70为前两阶段固定终点')
    evaluator = EVAL / 'source/tracking/continuation_e70_e80_eval.py'
    # Process adoption validates this same filename if a supervisor is restarted.
    text = text.replace("b'continuation_checkpoint_eval.py'", "b'continuation_e70_e80_eval.py'")
    evaluator.write_text(text)
    parent_run = json.loads((PARENT / 'wandb_tracking/train/mvlm/mvlm_TF_continue_e60_e70/run.json').read_text())
    metadata = dict(prepared_at=dt.datetime.now(dt.timezone.utc).isoformat(), parent_output=str(PARENT),
                    parent_checkpoint=str(checkpoint), parent_checkpoint_sha256=parent_hash, parent_run=parent_run,
                    start_epoch=70, target_epoch=80, start_global_step=87500, target_global_step=100000,
                    physical_gpus=[4,5], world_size=2, batch_per_rank=40, global_batch=80,
                    resumed_learning_rates=[5e-5,5e-6], samples_per_epoch=100000, additional_samples=1000000,
                    optimizer='AdamW state restored', scheduler='StepLR last_epoch restored to70; no further decay through80',
                    rng_state_restored=False, planned_evaluation_epochs=list(range(71,81)), evaluation_gpus=[6,7],
                    source=str(source), config=NAME, inherited_proxy_preserved=True)
    for filename in ('continuation.json','run_spec.json'):
        (OUT / filename).write_text(json.dumps(metadata,indent=2))
    (OUT / 'checkpoint_preflight.json').write_text(json.dumps(dict(epoch=70,global_step=87500,all_stored_model_floats_finite=True,optimizer_state_entries=622,learning_rates=[5e-5,5e-6],copied_checkpoint_sha256=parent_hash,config_changed_fields=['TRAIN.EPOCH','WANDB.NAME']),indent=2))
    for path in (source / 'lib/train/trainers/base_trainer.py',supervisor,monitor,evaluator):
        py_compile.compile(str(path),doraise=True)
    (OUT / 'source_sha256.json').write_text(json.dumps({str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},indent=2))
    print(json.dumps(dict(output=str(OUT),evaluation=str(EVAL),supervisor=str(supervisor),monitor=str(monitor),evaluator=str(evaluator)),indent=2))


if __name__ == '__main__':
    main()
