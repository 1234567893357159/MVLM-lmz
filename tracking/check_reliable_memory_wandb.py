"""No-network checks of the closed-loop trainer's optional W&B sidecar."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reliable_memory_wandb import WandbMonitor, monitoring_metrics, worker


class MonitorChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.options = dict(ENABLED=True, MODE='offline', PROJECT='check', ENTITY='', NAME='', LOG_INTERVAL=2)
        self.child = Mock(returncode=0)

    def tearDown(self):
        self.temporary.cleanup()

    def start(self, **kwargs):
        with patch('reliable_memory_wandb.subprocess.Popen', return_value=self.child):
            return WandbMonitor(self.root, self.options, {'purpose': 'unit check'}, **kwargs)

    def test_disabled_and_nonprimary_do_not_start_sdk(self):
        self.options['ENABLED'] = False
        self.start().close('completed')
        self.options['ENABLED'] = True
        self.start(rank=1).close('completed')
        self.assertFalse((self.root/'wandb_monitor').exists())

    def test_local_events_interval_finish_and_nonfinite(self):
        monitor = self.start()
        for step in (1, 2, 3):
            monitor.log({'loss': .5, 'invalid': float('nan')}, step)
        monitor.log({'loss': .4}, 3, force=True)
        monitor.log({'loss': .4}, 3, force=True)
        monitor.close('checkpointed')
        monitor.close('completed')
        rows = [json.loads(line) for line in (monitor.session/'events.jsonl').read_text().splitlines()]
        self.assertEqual([row['step'] for row in rows[:-1]], [1, 2, 3])
        self.assertNotIn('invalid', rows[0]['metrics'])
        self.assertEqual(rows[-1], dict(kind='finish', status='checkpointed'))

    def test_resume_preserves_identity_and_new_session(self):
        first = self.start()
        first.close('checkpointed')
        second = self.start(resume=True)
        second.close('completed')
        load = lambda m: json.loads((m.session/'request.json').read_text())
        self.assertEqual(load(first)['identity'], load(second)['identity'])
        self.assertNotEqual(first.session, second.session)

    def test_spawn_failure_is_nonfatal(self):
        with patch('reliable_memory_wandb.subprocess.Popen', side_effect=OSError('no worker')):
            monitor = WandbMonitor(self.root, self.options, {})
        monitor.log({'loss': 1}, 1)
        monitor.close('completed')

    def test_sdk_failure_is_recorded_without_involving_trainer(self):
        monitor = self.start()
        monitor.log({'loss': .5}, 1)
        monitor.close('completed')
        with patch.dict(sys.modules, {'wandb': None}):
            with self.assertRaises(ModuleNotFoundError):
                worker(monitor.session)
        state = json.loads((monitor.session/'status.json').read_text())
        self.assertEqual(state['status'], 'unavailable')
        self.assertTrue((monitor.session/'events.jsonl').is_file())

    def test_worker_logs_actual_steps_and_finish_status(self):
        monitor = self.start()
        monitor.log({'train/loss': .5}, 1)
        monitor.log({'train/loss': .4}, 2)
        monitor.close('checkpointed')
        run = SimpleNamespace(step=0, summary={}, define_metric=Mock(), log=Mock(), finish=Mock())
        sdk = SimpleNamespace(init=Mock(return_value=run), Settings=Mock())
        with patch.dict(sys.modules, {'wandb': sdk}):
            worker(monitor.session)
        self.assertEqual([call.kwargs['step'] for call in run.log.call_args_list], [1, 2])
        self.assertEqual(run.summary['training_status'], 'checkpointed')
        run.finish.assert_called_once_with(exit_code=0)
        self.assertIsNone(sdk.init.call_args.kwargs['resume'])
        self.assertEqual(json.loads((monitor.session/'status.json').read_text())['status'], 'finished')

    def test_global_metrics_and_rank0_diagnostics_are_separate(self):
        row = dict(loss=2., supervised=8, regression=4, outside=1, partial=1, background=2,
                   committed=3, pending=1, rejected=2, attempt=1, lr_used=1e-4, lr=5e-5, gradient_norm=4.)
        results = [dict(diagnostics=dict(score=.9, short_size=3, long_size=1, used_global_search=False,
                                        write=dict(reasons=['low_score']),
                                        fusion=dict(raw_ratio=.2, clipped_fraction=.1,
                                                    readout_ratio=.3, gate=.4, memory_fraction=1.)))]
        payload = monitoring_metrics(row, results, seconds=2., batch=1)
        self.assertEqual(payload['train/loss'], 2.)
        self.assertEqual(payload['memory/commit_rate'], .5)
        self.assertEqual(payload['memory/rank0/reasons/low_score'], 1)
        self.assertEqual(payload['fusion/rank0/raw_ratio_mean'], .2)
        self.assertEqual(payload['fusion/rank0/readout_ratio_mean'], .3)
        self.assertEqual(payload['fusion/rank0/gate_mean'], .4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
