"""Synthetic checks for frozen validation selection and checkpoint retention."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from reliable_memory_runtime import atomic_save, source_hashes
from reliable_validation import selected_manifest
from train_reliable_memory import save_candidate
from watch_reliable_training import choose_best, main as watch_main


class OrchestrationChecks(unittest.TestCase):
    def test_candidate_survives_last_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            atomic_save(root/'last.pth.tar', {'step': 1})
            first = save_candidate(root, 1)
            atomic_save(root/'last.pth.tar', {'step': 2})
            self.assertEqual(torch.load(first)['step'], 1)
            with self.assertRaises(ValueError):
                save_candidate(root, 1)
            self.assertEqual(torch.load(save_candidate(root, 2))['step'], 2)

    def test_fixed_selection_and_tie_break(self):
        sources = ('lasot', 'tnl2k', 'vasttrack')
        original = dict(protocol='official-train-video-split-v1', partial=False,
                        train=[dict(dataset=s, name='train') for s in sources],
                        val=[dict(dataset=s, name='val%d' % n) for s in sources for n in range(35)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'manifest.json'
            path.write_text(json.dumps(original))
            first = selected_manifest(path, 30, 123)
            second = selected_manifest(path, 30, 123)
            self.assertEqual(first, second)
            self.assertEqual(len(first['val']), 90)
            self.assertEqual(len(first['train']), 3)
        self.assertEqual(choose_best([dict(step=2000, mean_AUC=70.),
                                      dict(step=1000, mean_AUC=70.)])['step'], 1000)

    def test_watcher_chooses_and_links_best(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'checkpoints').mkdir()
            for step in (1, 2):
                (root/'checkpoints'/('step_%06d.pth.tar' % step)).write_bytes(('checkpoint%d' % step).encode())
            (root/'run.json').write_text(json.dumps(dict(source_hashes=source_hashes(), group='M6',
                                                        candidate_every=1, config={'TRAIN': {'MAX_UPDATES': 2}})))
            (root/'training_status.json').write_text(json.dumps(dict(status='completed')))
            manifest = root/'fixed.json'
            manifest.write_text(json.dumps(dict(val=[dict(dataset=s) for s in ('lasot','tnl2k','vasttrack')])))
            def fake_run(command, **kwargs):
                out = Path(command[command.index('--out')+1])
                out.mkdir()
                step = int(out.name.split('_')[-1])
                summary = dict(engineering_only=False, sequences=3, mean_AUC=float(step),
                               metrics={s: {'AUC': float(step)} for s in ('lasot','tnl2k','vasttrack')})
                (out/'summary.json').write_text(json.dumps(summary))
                return SimpleNamespace(returncode=0)
            args = ['watch', '--run-dir', str(root), '--validation-manifest', str(manifest),
                    '--candidate-every', '1', '--total-updates', '2', '--no-wandb']
            with patch('sys.argv', args), patch('watch_reliable_training.subprocess.run', side_effect=fake_run):
                watch_main()
            self.assertEqual(json.loads((root/'validation/best.json').read_text())['best']['step'], 2)
            self.assertEqual((root/'best.pth.tar').read_bytes(), b'checkpoint2')
            self.assertEqual(json.loads((root/'validation/watcher_status.json').read_text())['status'], 'completed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
