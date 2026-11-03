import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tracking.supervise_memory_concat_run import Supervisor


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
