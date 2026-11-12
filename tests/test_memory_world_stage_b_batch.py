"""Check B batching preserves causal per-video states and gradients through frozen CENTER."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

from test_memory_world import config, tiny_model, clip
from lib.train.actors.memory_world import MemoryWorldActor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tracking'))
from memory_world_stage_b_actor import BatchedStageBActor


class StageBBatchTests(unittest.TestCase):
    def test_loss_gradients_and_independent_histories_match_serial(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        cfg = config()
        cfg.TRAIN.WORLD.STAGE = 'B'
        cfg.TRAIN.WORLD.FREEZE_BN_STATS = True
        cfg.TRAIN.WORLD.PREDICTED_HISTORY_PROB = 1.
        one, two = clip(), clip()
        two['images'] = [np.flip(image, axis=1).copy() for image in two['images']]
        two['boxes'] = torch.tensor([[8., 12., 16., 20.]] * 6)
        two['frame_ids'] = torch.arange(6) * 3
        two['valid'][4] = False
        two['visible'][4] = False
        serial = tiny_model(cfg)
        with torch.no_grad():
            # Nonzero output ensures all attention/gate gradients are exercised.
            serial.future_fusion.output.weight.normal_(std=.02)
        batched = tiny_model(copy.deepcopy(cfg))
        batched.load_state_dict(serial.state_dict())
        for model in (serial, batched):
            model.configure_world_stage('B')
            model.train()
        loss, metrics = MemoryWorldActor(serial, cfg, 'B')([one, two])
        other, batched_metrics = BatchedStageBActor(batched, cfg, 'B')([one, two])
        torch.testing.assert_close(other, loss, rtol=2e-5, atol=2e-6)
        self.assertAlmostEqual(batched_metrics['loss'], cfg.TRAIN.GIOU_WEIGHT * batched_metrics['giou_loss']
                               + cfg.TRAIN.L1_WEIGHT * batched_metrics['l1_loss']
                               + batched_metrics['focal_loss'], places=5)
        for key in ('supervised_frames', 'predictions', 'active', 'memory_writes'):
            self.assertEqual(metrics[key], batched_metrics[key])
        loss.backward()
        other.backward()
        nonzero = 0
        for (name, parameter), (other_name, other_parameter) in zip(serial.named_parameters(), batched.named_parameters()):
            self.assertEqual(name, other_name)
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, other_parameter.grad, rtol=2e-4, atol=5e-6)
                nonzero += int(parameter.grad.abs().sum() > 0)
            else:
                self.assertIsNone(parameter.grad)
                self.assertIsNone(other_parameter.grad)
        self.assertGreater(nonzero, 5)


if __name__ == '__main__':
    unittest.main()
