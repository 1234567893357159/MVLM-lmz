"""Focused causal and numerical checks for the reliable-memory implementation."""
import argparse
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import cv2
from torch import nn
from lib.config.mvlm.config import load_config
from lib.models.mvlm.memory import TargetMemoryBank
from lib.models.mvlm.memory_fusion import MemoryFusion
from lib.models.mvlm.memory_policy import Observation, ReliableWritePolicy
from lib.models.mvlm.reliable_memory import ReliableMemoryBank, ReliableEntry
from lib.models.mvlm.mvlm import MVLM
from lib.models.mvlm.head import CenterPredictor
from lib.models.mvlm.tracking_core import TrackingCore, prepare_crop, select_candidates
from lib.train.actors.reliable_memory import ReliableMemoryActor, frame_loss
from check_memory_module import TinyEncoder
from evaluate_reliable_memory import score_predictions
from reliable_memory_runtime import configure_group
from lib.train.data.reliable_memory_dataset import ReliableClips


def config(permissive=False):
    cfg = load_config('mvlm_TF_reliable_memory')
    cfg.DATA.SEARCH.SIZE = cfg.TEST.SEARCH_SIZE = 64
    cfg.DATA.TEMPLATE.SIZE = cfg.TEST.TEMPLATE_SIZE = 32
    cfg.DATA.MULTI_MODAL_VISION = False
    cfg.MODEL.MEMORY.HIDDEN_DIM = 16
    cfg.MODEL.MEMORY.KEY_DIM = 8
    cfg.TRAIN.CLIP_FRAMES = 5
    cfg.TRAIN.WARMUP_FRAMES = 1
    if permissive:
        policy = cfg.MODEL.MEMORY.POLICY
        policy.MIN_SCORE = 0.
        policy.MIN_CLS_MARGIN = policy.MIN_LANGUAGE_MARGIN = -2.
        policy.MIN_DESCRIPTOR_SIM = -1.
        policy.MAX_CENTER_SPEED = policy.MAX_LOG_SCALE_RATE = 1e9
        cfg.TEST.MVLM.TAU = -2.
        cfg.MODEL.MEMORY.MIN_QUALITY = 0.
    return cfg


def observation(t, basis=0, **kwargs):
    tokens = torch.eye(32)[basis].repeat(4, 1)
    values = dict(frame_id=t, box_image_xywh=torch.tensor([10., 10., 20., 20.]),
                  box_normalized_xywh=torch.tensor([.1, .1, .2, .2]), roi_tokens=tokens,
                  descriptor=tokens.mean(0), score=.95, cls_margin=.2, language_margin=.2,
                  vlm_pass=True, roi_valid=True, used_global_search=False)
    values.update(kwargs)
    return Observation(**values)


class TinyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('embedding', torch.arange(32).float().view(1, 1, 32)/32)

    def forward(self, text):
        return self.embedding.expand(len(text), 3, 32)


def tiny_model(cfg):
    model = MVLM(TinyText(), TinyEncoder(), CenterPredictor(32, 32, 4, 16),
                 memory_cfg=cfg.MODEL.MEMORY)
    model.freeze_memory_backbone = model.freeze_head_bn = True
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    return model.train()


class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.cfg = config()

    def test_confirmation_causality_isolation(self):
        bank = ReliableMemoryBank(2, self.cfg.MODEL.MEMORY)
        actions = ['pending', 'pending', 'commit', 'commit', 'reject', 'pending', 'commit']
        for t, action in enumerate(actions):
            self.assertEqual(bank.update(observation(t, score=.1 if t == 4 else .95)).action, action)
            if t < 2:
                self.assertFalse(bank.entries(0))
        self.assertFalse(bank.entries(1))
        with self.assertRaises(ValueError):
            bank.pack(torch.ones(2, 16, 32), [6, 7], 4)
        mask = bank.pack(torch.ones(2, 16, 32), [7, 7], 4)[1]
        self.assertTrue(mask[0].all())
        self.assertFalse(mask[1].any())
        with self.assertRaises(ValueError):
            bank.update(observation(6))

    def test_recovery_motion_and_missing_margin(self):
        policy = ReliableWritePolicy(self.cfg.MODEL.MEMORY.POLICY)
        for t, action in enumerate(['pending', 'pending', 'commit']):
            self.assertEqual(policy.observe(observation(t, used_global_search=True,
                             cls_margin=None, language_margin=None)).action, action)
        jump = torch.tensor([1000., 1000., 20., 20.])
        for t, action in zip(range(3, 6), ['pending', 'pending', 'commit']):
            self.assertEqual(policy.observe(observation(t, box_image_xywh=jump)).action, action)
        self.assertEqual(policy.observe(observation(6, score=float('nan'))).action, 'reject')

    def test_retention_snapshot_capacity(self):
        cfg = self.cfg.MODEL.MEMORY
        cfg.RELIABLE_WRITE = False
        cfg.LONG_CAPACITY = 2
        bank = ReliableMemoryBank(1, cfg)
        for i, margin in enumerate([.5, -.5]):
            obs = observation(i, basis=i)
            bank.long[0].append(ReliableEntry(obs.roi_tokens, i, .95,
                                obs.box_normalized_xywh, i, margin, 3))
        bank.update(observation(2, basis=2))
        self.assertEqual([e.frame_id for e in bank.long[0]], [0, 2])
        clone = ReliableMemoryBank(1, cfg)
        clone.load_state_dict(bank.state_dict())
        for t in range(3, 30):
            for item in [bank, clone]:
                item.update(observation(t, basis=t % 10), torch.eye(32)[[20, 21]])
        self.assertEqual([e.frame_id for e in bank.entries(0)], [e.frame_id for e in clone.entries(0)])
        self.assertLessEqual(len(bank.short[0]), 8)
        self.assertLessEqual(len(bank.long[0]), 2)
        self.assertFalse(any(e.tokens.requires_grad for e in bank.entries(0)))

    def test_legacy_bank_equivalence(self):
        cfg = self.cfg.MODEL.MEMORY
        cfg.RELIABLE_WRITE = cfg.DISTRACTOR_RETENTION = False
        old, new = TargetMemoryBank(1), ReliableMemoryBank(1, cfg)
        for t in range(40):
            obs = observation(t, roi_tokens=torch.randn(4, 32), score=.95 if t % 3 else .1)
            old.write(obs.roi_tokens[None], obs.box_normalized_xywh[None], [t], [obs.score], [True])
            new.update(obs)
        self.assertEqual([e.frame_id for e in old.entries(0)], [e.frame_id for e in new.entries(0)])

    def test_fusion_bounds_zero_and_gradients(self):
        bank = TargetMemoryBank(2)
        bank.write(torch.randn(2, 4, 32), torch.tensor([[.1, .1, .3, .3]]*2), [0, 0], [1., 0.])
        features, text = torch.randn(2, 16, 32), torch.randn(2, 3, 32)
        fusion = MemoryFusion(32, 16, 8, bounded=True, max_residual_ratio=.25)
        self.assertTrue(torch.equal(fusion(features, text, bank, [1, 1])[0], features))
        optimizer = torch.optim.AdamW(fusion.parameters(), lr=.01)
        for step in range(2):
            optimizer.zero_grad()
            result, _ = fusion(features, text, bank, [1, 1])
            (result-1).square().mean().backward()
            self.assertGreater(float(fusion.output_proj.weight.grad.abs().sum()), 0.)
            if step:
                self.assertGreater(float(fusion.appearance_key.weight.grad.abs().sum()), 0.)
            optimizer.step()
        nn.init.normal_(fusion.output_proj.weight, std=10)
        result, _ = fusion(features, text, bank, [1, 1])
        self.assertTrue(((result-features).norm(dim=-1) <= .25*features.norm(dim=-1)+1e-5).all())
        self.assertTrue(torch.equal(result[1], features[1]))
        fusion.max_residual_ratio = 0
        self.assertTrue(torch.equal(fusion(features, text, bank, [1, 1])[0], features))
        fusion.max_residual_ratio = .25
        self.assertTrue(torch.equal(fusion(features*0, text, bank, [1, 1])[0], features*0))

    def test_coordinate_roundtrip_and_candidates(self):
        image = np.zeros((90, 170, 3), dtype=np.uint8)
        for global_search in [True, False]:
            _, transform = prepare_crop(image, [13.2, 9.1, 35., 25.], global_search, 64, 4)
            box = torch.tensor([30., 20., 25., 15.])
            self.assertTrue(torch.allclose(transform.to_image(transform.to_crop(box)), box, atol=1e-5))
        _, transform = prepare_crop(image, None, True, 64, 4)
        self.assertTrue(torch.allclose(transform.to_image(torch.tensor([0., 0., 1., 1.])),
                                      torch.tensor([0., 0., 170., 90.])))
        cfg = self.cfg.TEST.MVLM
        cfg.ALPHA_CORR = .5
        cfg.TAU = .01
        selected = select_candidates(torch.tensor([[.2, .2, .1, .1], [.8, .8, .1, .1]]),
                                     torch.tensor([.9, .2]), torch.tensor([.8, .1]), None, cfg)
        self.assertEqual(selected['index'], 0)
        self.assertFalse(selected['next_global'])

    def test_closed_loop_gt_independence_backward_and_resume(self):
        cfg = config(True)
        model = tiny_model(cfg)
        actor = ReliableMemoryActor(model, cfg, 'cpu')
        image = np.random.RandomState(9).randint(0, 255, (90, 170, 3), dtype=np.uint8)
        clip = dict(images=[image]*5, boxes=torch.tensor([[30., 20., 25., 15.]]*5),
                    valid=torch.ones(5, dtype=torch.bool), visible=torch.ones(5, dtype=torch.bool),
                    known_absent=torch.zeros(5, dtype=torch.bool), frame_ids=list(range(5)),
                    language='a person', name='test')
        loss, _, results = actor([clip])
        other_loss, _, other = actor([dict(clip, boxes=clip['boxes']+15)])
        self.assertEqual(results, other)
        self.assertNotEqual(float(loss), float(other_loss))
        model.zero_grad(set_to_none=True)
        _, stats, _ = actor([clip], backward=True)
        self.assertGreater(stats['supervised'], 0)
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.head.parameters()))
        self.assertIsNotNone(model.memory_fusion.output_proj.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        core = TrackingCore(model, cfg, 'cpu')
        core.reset('a person', 'one')
        for _ in range(4):
            core.step(image)
        self.assertTrue(core.state.memory.entries(0))
        saved = core.state_dict()
        _, expected = core.step(image)
        core.load_state_dict(saved)
        _, actual = core.step(image)
        self.assertEqual(expected, actual)
        core.reset('a different target', 'two')
        self.assertFalse(core.state.memory.entries(0))
        self.assertTrue(core.state.global_search)
        self.assertEqual(core.state.frame_id, -1)
        self.assertFalse(any('short' in k or 'long' in k for k in model.state_dict()))

    def test_unknown_outside_absent_loss_and_config_isolation(self):
        cfg = config(True)
        core = TrackingCore(tiny_model(cfg), cfg, 'cpu')
        core.reset('a person')
        output, _ = core.step(np.zeros((90, 170, 3), dtype=np.uint8))
        box = torch.tensor([1000., 1000., 20., 20.])
        loss, stats = frame_loss(output, core.transform, box, True, True, False, cfg)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats['outside'], 1)
        loss, stats = frame_loss(output, core.transform, box, False, False, False, cfg)
        self.assertEqual(float(loss), 0.)
        self.assertEqual(stats['supervised'], 0)
        loss, stats = frame_loss(output, core.transform, box, False, False, True, cfg)
        self.assertEqual(stats['background'], 1)
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(load_config('mvlm_TF_memory').MODEL.MEMORY.RELIABLE_WRITE)
        self.assertFalse(load_config('mvlm_TF_memory').TEST.RELIABLE_CORE)

    def test_strict_scoring_never_substitutes_gt(self):
        seq = dict(dataset='tnl2k', boxes=np.array([[100., 100., 20., 20.]]*2),
                   visible=np.ones(2, dtype=bool))
        original = np.array([[0., 0., 20., 20.], [0., 0., 0., 0.]])
        scores = score_predictions(original, seq)
        self.assertEqual(scores['AUC'], 0.)
        self.assertEqual(scores['PRE'], 0.)
        self.assertEqual(scores['NPR'], 0.)
        self.assertEqual(scores['first_frame_IoU'], 0.)
        self.assertTrue(np.array_equal(original[0], [0., 0., 20., 20.]))
        seq['boxes'][1] = 0
        self.assertEqual(score_predictions(original, seq)['NPR'], 0.)

    def test_all_paired_group_paths(self):
        image = np.zeros((90, 170, 3), dtype=np.uint8)
        for group in ('M1', 'M2', 'M3', 'M4', 'M5', 'M6'):
            cfg = configure_group(config(True), group)
            model = tiny_model(cfg)
            core = TrackingCore(model, cfg, 'cpu')
            core.reset('a person', group)
            for _ in range(5):
                output, result = core.step(image)
            self.assertTrue(torch.isfinite(output['score_map']).all())
            if group == 'M1':
                self.assertIsNone(core.state.memory)
                self.assertFalse(hasattr(model, 'memory_fusion'))
            else:
                self.assertTrue(core.state.memory.entries(0))
            if group in ('M2', 'M5'):
                self.assertIsInstance(core.state.memory, TargetMemoryBank)

    def test_raw_sampling_keeps_invisible_interior(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(6):
                path = str(Path(directory)/('%d.png' % i))
                cv2.imwrite(path, np.zeros((30, 40, 3), dtype=np.uint8))
                paths.append(path)
            seq = dict(dataset='lasot', name='synthetic', frames=paths,
                       boxes=np.array([[2., 2., 10., 10.]]*6, dtype=np.float32),
                       valid=np.ones(6, dtype=bool), visible=np.array([1, 0, 0, 0, 0, 1], dtype=bool),
                       language='a person')
            dataset = ReliableClips([seq], length=5, samples=2, max_stride=1, augment=False)
            clip = dataset[0]
            self.assertEqual(clip['frame_ids'], [0, 1, 2, 3, 4])
            self.assertEqual(clip['visible'].tolist(), [True, False, False, False, False])
            self.assertEqual(clip['known_absent'].tolist(), [False, True, True, True, True])
            seq['visible'][:] = False
            self.assertEqual(len(dataset[1]['images']), 5)

    def test_schedule_config_and_duplicate_rejection(self):
        cfg = load_config('mvlm_TF_reliable_memory')
        self.assertEqual(cfg.TRAIN.SCHEDULER.TYPE, 'cosine')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'duplicate.yaml'
            path.write_text('TRAIN:\n  LR: 0.1\n  LR: 0.2\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate configuration key'):
                load_config(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=['cpu'])
    parser.parse_args()
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    sys.exit(0 if result.wasSuccessful() else 1)
