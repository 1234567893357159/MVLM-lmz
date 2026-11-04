import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from lib.train.memory_run_utils import memory_optimizer, set_step_lr, validate_step_budget
from tracking.supervise_memory_concat_run import Supervisor


class ContinuationOptimizerTests(unittest.TestCase):
    def test_resume_preserves_adamw_moments_and_tail_lr(self):
        model=torch.nn.Module()
        model.memory_fusion=torch.nn.Linear(2,2)
        model.head=torch.nn.Linear(2,1)
        config=SimpleNamespace(TRAIN=SimpleNamespace(LR=1e-4,MEMORY_HEAD_LR=1e-5,WEIGHT_DECAY=1e-4))
        optimizer=memory_optimizer(model,config)
        def update(net,opt,step):
            set_step_lr(opt,step)
            opt.zero_grad()
            net.head(net.memory_fusion(torch.ones(2,2))).square().sum().backward()
            opt.step()
        update(model,optimizer,4000)
        resumed=copy.deepcopy(model)
        resumed_optimizer=memory_optimizer(resumed,config)
        resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        validate_step_budget(4000,5000,6000,resuming=True)
        update(model,optimizer,4001)
        update(resumed,resumed_optimizer,4001)
        for expected,actual in zip(model.parameters(),resumed.parameters()):
            self.assertTrue(torch.equal(expected,actual))
        self.assertAlmostEqual(resumed_optimizer.param_groups[0]['lr'],1e-5)
        self.assertAlmostEqual(resumed_optimizer.param_groups[1]['lr'],1e-6)
        self.assertTrue(all(float(state['step'])==2 for state in resumed_optimizer.state.values()))

    def test_extension_requires_explicit_budget_and_completed_parent(self):
        for arguments in ((4000,5000,4000,True),(0,6000,6000,False),(2000,6000,6000,True)):
            with self.assertRaises(ValueError):
                validate_step_budget(*arguments)
        validate_step_budget(4500,5000,6000,resuming=True)


class ContinuationSupervisorTests(unittest.TestCase):
    def make_supervisor(self,out):
        (out/'run_spec.json').write_text(json.dumps(dict(source=str(out),start_step=4000,
            optimizer_updates=6000,batch_per_rank=40,global_batch=80,wandb=dict(url='new-run'),
            validation=dict(full150_at=[5000,6000]),continuation=dict(parent_out='parent',
                parent_wandb_url='parent-run',checkpoint='parent4000',checkpoint_sha256='sha',
                baseline_mean_AUC=78.4331))))
        return Supervisor(out)

    def test_4000_control_is_retained_when_continuation_regresses(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            supervisor=self.make_supervisor(out)
            supervisor.run=SimpleNamespace(summary={},finish=mock.Mock())
            results={4000:dict(mean_AUC=78.08),5000:dict(mean_AUC=77.5),6000:dict(mean_AUC=77.9)}
            with mock.patch.object(supervisor,'validation',side_effect=lambda step,scope:results[step]), \
                 mock.patch.object(supervisor,'train_continuation_segment'), \
                 mock.patch.object(supervisor,'status'):
                supervisor.continue_main()
            best=json.loads((out/'best.json').read_text())
            self.assertEqual(best['step'],4000)
            self.assertEqual(best['checkpoint'],'parent4000')
            comparison=json.loads((out/'continuation_comparison.json').read_text())
            self.assertAlmostEqual(comparison['steps']['6000']['delta_vs_4000'],-.18)

    def test_restart_uses_latest_saved_optimizer_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            supervisor=self.make_supervisor(out)
            (out/'latest_checkpoint.json').write_text(json.dumps(dict(step=4000,path='parent4000')))
            commands=[]
            def execute(command,label):
                commands.append(command)
                if len(commands)==1:
                    (out/'latest_checkpoint.json').write_text(json.dumps(dict(step=4500,path='saved4500')))
                    raise RuntimeError('simulated interrupted trainer')
                (out/'checkpoints').mkdir()
                (out/'checkpoints/MVLM_step005000.pth.tar').touch()
            with mock.patch.object(supervisor,'execute',side_effect=execute), \
                 mock.patch.object(supervisor,'status'), \
                 mock.patch('tracking.supervise_memory_concat_run.time.sleep'):
                supervisor.train_continuation_segment(5000)
            self.assertEqual(len(commands),2)
            self.assertEqual(commands[0][commands[0].index('--resume')+1],'parent4000')
            self.assertEqual(commands[1][commands[1].index('--resume')+1],'saved4500')
            self.assertEqual(commands[1][commands[1].index('--total-steps')+1],'6000')
            self.assertIn('automatic_training_retry',(out/'events.jsonl').read_text())


if __name__=='__main__':
    unittest.main()
