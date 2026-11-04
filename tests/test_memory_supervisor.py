import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tracking.supervise_memory_concat_run import Supervisor
from tracking.log_memory_concat_wandb import run_online_worker


class SupervisorUploadTests(unittest.TestCase):
    def test_partial_metric_line_is_retried_without_duplicate_upload(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            (out/'run_spec.json').write_text(json.dumps(dict(source=folder,
                batch_per_rank=40,global_batch=80,wandb=dict(url='test'))))
            supervisor=Supervisor(out)
            received=[]
            supervisor.run=SimpleNamespace(log=received.append)
            path=out/'training_metrics.jsonl'
            payload=json.dumps(dict(global_step=20,**{'train/Loss':.5}))
            path.write_text(payload)
            with mock.patch('tracking.supervise_memory_concat_run.subprocess.check_output',return_value=''):
                supervisor.upload()
                self.assertEqual(supervisor.offset,0)
                with path.open('a') as stream:stream.write('\n')
                supervisor.upload();supervisor.upload()
            training=[item for item in received if 'train/Loss' in item]
            self.assertEqual(training,[dict(global_step=20,**{'train/Loss':.5})])
            self.assertEqual(supervisor.offset,path.stat().st_size)


class OnlineUploaderTests(unittest.TestCase):
    def test_fatal_upload_warning_restarts_only_the_client_process_group(self):
        child=mock.Mock(pid=123456,stdout=iter([
            'wandb: syncing\n','wandb: WARNING Fatal error while uploading data.\n']))
        child.wait.return_value=-15
        with mock.patch('tracking.log_memory_concat_wandb.subprocess.Popen',return_value=child) as launch, \
             mock.patch('tracking.log_memory_concat_wandb.os.killpg') as kill, \
             mock.patch('builtins.print'):
            self.assertEqual(run_online_worker(['uploader'],{'WANDB_MODE':'online'}),1)
        self.assertTrue(launch.call_args.kwargs['start_new_session'])
        kill.assert_called_once_with(123456,15)
        child.wait.assert_called_once_with(timeout=10)

    def test_successful_upload_completion_does_not_restart_the_client(self):
        child=mock.Mock(pid=123456,stdout=iter(['wandb: synced successfully\n']))
        child.wait.return_value=0
        with mock.patch('tracking.log_memory_concat_wandb.subprocess.Popen',return_value=child), \
             mock.patch('tracking.log_memory_concat_wandb.os.killpg') as kill, \
             mock.patch('builtins.print'):
            self.assertEqual(run_online_worker(['uploader'],{}),0)
        kill.assert_not_called()
