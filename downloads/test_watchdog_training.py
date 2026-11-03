"""Checks for watchdog decisions during processing and transfer stalls."""
import contextlib
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import watchdog_training as w


class WatchdogChecks(unittest.TestCase):
    def check_state(self, stages, before, after):
        state = {'_runtime': {'pid': 123}}
        snapshots = [(state, stages, before), (state, stages, after)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(w, 'OUT', Path(tmp)), \
                patch.object(w, 'snapshot', side_effect=snapshots), \
                patch.object(w, 'alive', return_value=True), \
                patch.object(w.time, 'sleep'), \
                patch.object(w.subprocess, 'run') as launch, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            w.main(True)
            launch.assert_not_called()
            return output.getvalue()

    def test_verifying_does_not_restart(self):
        self.assertIn('PROCESSING', self.check_state({'lasot': 'downloading', 'tnl2k': 'verifying'}, 10, 10))

    def test_repair_does_not_restart(self):
        self.assertIn('PROCESSING', self.check_state({'lasot': 'downloading', 'tnl2k': 'repairing'}, 10, 10))

    def test_progress_of_current_archive_is_counted(self):
        self.assertIn('"status": "OK"', self.check_state({'lasot': 'downloading', 'tnl2k': 'failed'}, 10, 30))

    def test_one_idle_sample_does_not_restart(self):
        self.assertIn('WAITING', self.check_state({'lasot': 'downloading', 'tnl2k': 'failed'}, 10, 10))

    def test_failed_tasks_do_not_loop_on_restart(self):
        self.assertIn('NEEDS_ATTENTION', self.check_state({'lasot': 'failed', 'tnl2k': 'failed'}, 10, 10))


if __name__ == '__main__':
    unittest.main()
