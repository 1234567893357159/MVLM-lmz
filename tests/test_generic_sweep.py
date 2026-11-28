import sys
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tracking'))
import sweep_mvlm_generic as sweep


class GenericSweepTests(unittest.TestCase):
    def test_yaml_configs_are_isolated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yaml'
            path.write_text('TEST:\n  MVLM:\n    ALPHA_CORR: 0.321\n')
            cfg = sweep.load_config(str(path))
            self.assertEqual(cfg.TEST.MVLM.ALPHA_CORR, 0.321)
            self.assertNotEqual(sweep.config_module.cfg.TEST.MVLM.ALPHA_CORR, 0.321)

    def test_config_isolation_and_invalid_override(self):
        first, second = copy.deepcopy(sweep.config_module.cfg), copy.deepcopy(sweep.config_module.cfg)
        original = second.TEST.MVLM.ALPHA_CORR
        sweep.apply_overrides(first, {'TEST.MVLM.ALPHA_CORR': 0.123})
        self.assertEqual(first.TEST.MVLM.ALPHA_CORR, 0.123)
        self.assertEqual(second.TEST.MVLM.ALPHA_CORR, original)
        with self.assertRaises(ValueError):
            sweep.apply_overrides(first, {'MODEL.WORLD.ENABLED': False})
        with self.assertRaises(ValueError):
            sweep.apply_overrides(first, {'TEST.MVLM.UNKNOWN': 1})

    def test_metric_perfect_and_invalid_predictions(self):
        seq = SimpleNamespace(name='sample', dataset='tnl2k', frames=['1', '2'],
                              ground_truth_rect=np.array([[5, 5, 10, 10], [6, 5, 10, 10]]),
                              target_visible=None)
        boxes, metrics = sweep.metric(seq.ground_truth_rect, seq)
        self.assertEqual(boxes.shape, (2, 4))
        self.assertAlmostEqual(metrics['AUC'], 100 * 20 / 21)
        self.assertEqual(metrics['PRE'], 100)
        self.assertEqual(metrics['NPR'], 100)
        with self.assertRaises(ValueError):
            sweep.metric([[1, 2, 0, 3], [1, 2, 4, 3]], seq)

    def test_numpy_fingerprint_and_name_safety(self):
        seq = SimpleNamespace(dataset='lasot', name='video-1', frames=['one', 'two'],
                              init_info=lambda: {'init_bbox': [np.float32(1)]},
                              ground_truth_rect=np.ones((2, 4)), target_visible=None)
        self.assertEqual(sweep.sequence_hash(seq), sweep.sequence_hash(seq))
        with self.assertRaises(ValueError):
            sweep.safe_name('../other')

    def test_model_specific_runtime_thresholds(self):
        cfg = sweep.EasyDict({'MODEL': {'MEMORY': {'MIN_QUALITY': 0.5,
                              'POLICY': {'CONFIRM_FRAMES': 2}},
                              'WORLD': {'FUSION_STRENGTH': 1.0}}})
        sweep.apply_overrides(cfg, {'MODEL.MEMORY.MIN_QUALITY': 0.7,
                                  'MODEL.MEMORY.POLICY.CONFIRM_FRAMES': 3,
                                  'MODEL.WORLD.FUSION_STRENGTH': 0.5})
        self.assertEqual(cfg.MODEL.MEMORY.POLICY.CONFIRM_FRAMES, 3)
        with self.assertRaises(ValueError):
            sweep.apply_overrides(cfg, {'MODEL.MEMORY.POLICY.CONFIRM_FRAMES': 2.5})


if __name__ == '__main__':
    unittest.main()
