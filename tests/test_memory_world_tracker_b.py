"""Streaming gradient equivalence, paper CM equivalence, and empty-rank reduction."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from test_memory_world import config, tiny_model, clip
from lib.utils.cm_loss import CMloss
from lib.train.actors.memory_world import tracking_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tracking'))
from memory_world_tracker_b import (TrackerStageB, StreamingTrackerBActor,
                                   cm_per_example, synchronize_gradients, loss_sum)


def reduce_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    model = nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.fill_(.3)
        model.bias.zero_()
    if rank == 0:
        # A sum of two frame losses; the other rank has no supervised frames.
        (model(torch.tensor([[1., 2.], [3., 4.]])).square().sum()).backward()
    synchronize_gradients(model, supervised_frames=2, bucket_bytes=4)
    torch.testing.assert_close(model.weight.grad, torch.tensor([[7.2, 10.2]]))
    torch.testing.assert_close(model.bias.grad, torch.tensor([3.]))
    dist.destroy_process_group()


class TrackerBTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(22)

    def test_vectorized_cm_matches_official_and_gradients(self):
        visual = torch.randn(3, 16, 32, requires_grad=True)
        language = torch.randn(3, 77, 32, requires_grad=True)
        boxes = torch.tensor([[.1, .1, .6, .6], [.2, .2, .7, .5], [-.1, .1, .7, .4]])
        actual, valid = cm_per_example(language, visual, boxes)
        expected = CMloss()(language, visual, boxes, 1., 1.)
        self.assertTrue(valid.all())
        torch.testing.assert_close(actual.mean(), expected)
        grads = torch.autograd.grad(actual.mean(), (language, visual), retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (language, visual))
        for one, two in zip(grads, expected_grads):
            torch.testing.assert_close(one, two, atol=1e-7, rtol=2e-4)

    def test_streaming_matches_retained_graph_and_freeze_scope(self):
        cfg = config()
        cfg.TRAIN.WORLD.STAGE = 'B'
        cfg.TRAIN.WORLD.TRAIN_HEAD = True
        cfg.TRAIN.FREEZE_ENCODER = False
        cfg.TRAIN.WORLD.FREEZE_BN_STATS = True
        cfg.TRAIN.CM_LOSS = True
        cfg.TRAIN.CM_WEIGHT = .5
        base = tiny_model(cfg)
        base.text_encoder.text_proj = nn.Linear(32, 32)
        base.text_encoder.clip = nn.Linear(32, 32)
        original_forward = base.encoder.forward
        def encode(templates, searches, language):
            out = original_forward(templates, searches, language)
            out['tae'] = torch.cat((out['feat_x'], language), 1)
            return out
        base.encoder.forward = encode
        model = TrackerStageB.from_base(base)
        model.configure_world_stage('B')
        model.train()
        self.assertTrue(model.encoder.training)
        self.assertFalse(model.text_encoder.clip.training)
        self.assertFalse(model.world_model.training)
        for name, p in model.named_parameters():
            self.assertEqual(p.requires_grad, name.startswith(('encoder.', 'text_encoder.text_proj.', 'head.', 'future_fusion.')))
        actor = StreamingTrackerBActor(model, cfg)
        clips = [clip(), clip()]
        clips[1]['valid'][3] = False
        cached = torch.randn(2, 77, 32)
        retained = []
        with mock.patch('memory_world_tracker_b.frozen_text_tokens', return_value=cached):
            reference = actor(clips, backward=retained.append)
            (torch.stack(retained).sum() / reference['supervised_frames']).backward()
            gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            streamed = actor(clips, backward=lambda value: value.backward())
            synchronize_gradients(model, streamed['supervised_frames'])
        self.assertEqual(reference['supervised_frames'], streamed['supervised_frames'])
        self.assertAlmostEqual(reference['loss'], streamed['loss'], places=6)
        for name, p in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(p.grad, gradients[name], rtol=2e-4, atol=3e-6)
            elif not p.requires_grad:
                self.assertIsNone(p.grad)

    def test_tracking_formula_matches_legacy_including_boundary_heatmap(self):
        cfg = config()
        cfg.TRAIN.CM_LOSS = True
        cfg.TRAIN.CM_WEIGHT = .5
        boxes = torch.tensor([[.1, .1, .5, .5], [.95, .95, .08, .08]])
        head = dict(pred_boxes=torch.tensor([[[.4, .4, .3, .3]], [[.9, .9, .2, .2]]], requires_grad=True),
                    score_map=torch.randn(2, 1, 4, 4).sigmoid().requires_grad_())
        encoded = {'tae': torch.randn(2, 16 + 77, 32, requires_grad=True)}
        actual, metrics, _ = loss_sum(head, encoded, boxes, cfg)
        tracking = torch.stack([tracking_loss({key: value[i:i + 1] for key, value in head.items()},
                                              boxes[i:i + 1], cfg) for i in range(2)]).mean()
        cm = CMloss()(encoded['tae'][:, -77:], encoded['tae'][:, :-77], boxes, 1., 1.)
        expected = tracking + .5 * cm
        torch.testing.assert_close(actual / 2, expected)
        self.assertAlmostEqual(metrics['tracking_loss'], float(tracking), places=6)

    def test_two_ranks_with_empty_supervision(self):
        with tempfile.TemporaryDirectory() as temp:
            mp.spawn(reduce_worker, args=(os.path.join(temp, 'rendezvous'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
