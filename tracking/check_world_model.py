"""CPU checks for causal state prediction, geometry, stage freezing and rollout."""
import json
import sys
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path

import _init_paths
import cv2
import yaml
import numpy as np
import torch
from torch import nn

from lib.config.mvlm.config import load_config
from lib.models.mvlm.mvlm import MVLM
from lib.models.mvlm.world_model import WorldModel
from lib.models.mvlm.imagination_fusion import ImaginationFusion
from lib.models.mvlm.world_checkpoint import load_tracker_weights, load_predictor, world_spec
from lib.models.mvlm.world_state import (WorldHistory, image_to_crop, crop_to_image,
    extract_roi, search_crop, apply_motion, motion_target)
from lib.train.actors.world_model import WorldModelActor

ROOT = Path(__file__).resolve().parents[1]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_channels = 8
        self.body = nn.Linear(8, 8)
        self.body.num_patches_search = 16
        self.body.num_patches_template = 4

    def forward(self, templates, searches, language):
        value = searches[0].mean((1, 2, 3)).view(-1, 1, 1)
        features = self.body(torch.ones(value.shape[0], 16, 8)) + value
        return dict(feat_x=features)


class Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.proj(torch.ones(len(texts), 3, 8))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 1, 1)

    def forward(self, feature, target=None):
        score = self.conv(feature).sigmoid()
        mean = score.mean((1, 2, 3))
        box = torch.stack((mean, mean, mean, mean), -1)
        return score, box, score.repeat(1, 2, 1, 1), score.repeat(1, 2, 1, 1)

    def cal_bboxv2(self, response, size, offset, return_score=True):
        box = response.new_tensor([.5, .5, .4, .4]).view(1, 1, 4)
        return box, response.amax((2, 3))


def configuration():
    cfg = load_config(ROOT / 'experiments/mvlm/mvlm_TF_world_stage_a.yaml')
    cfg.MODEL.WORLD.HIDDEN = 16
    cfg.MODEL.WORLD.LAYERS = 1
    cfg.DATA.SEARCH.SIZE = 64
    cfg.MODEL.ENCODER.STRIDE = 16
    cfg.TEST.MVLM.TAU = -.1
    cfg.TEST.WINDOW = False
    return cfg


def model_and_cfg():
    cfg = configuration()
    return MVLM(Text(), Encoder(), Head(), head_type='CENTER', world_cfg=cfg.MODEL.WORLD), cfg


def make_history(batch=2):
    history = WorldHistory(4)
    for t in range(4):
        history.append(torch.rand(batch, 4, 8), torch.tensor([[.4 + t * .01, .5, .2, .3]]).repeat(batch, 1),
                       torch.ones(batch), torch.ones(batch, dtype=torch.bool), torch.full((batch,), t))
    return history


class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)

    def test_geometry_and_rounded_crop(self):
        image = np.zeros((71, 113, 3), dtype=np.uint8)
        patch, region = search_crop(image, 64, 4., [13.2, 7.7, 15., 11.])
        boxes = torch.tensor([[.3, .4, .12, .15]])
        self.assertEqual(patch.shape, (64, 64, 3))
        self.assertTrue(torch.allclose(crop_to_image(image_to_crop(boxes, region), region), boxes))
        expected_side = 52
        self.assertAlmostEqual(float(region[2]), expected_side / 113, places=6)
        future = boxes + torch.tensor([[.02, -.03, .01, .02]])
        self.assertTrue(torch.allclose(apply_motion(boxes, motion_target(boxes, future)), future))

    def test_history_causality_and_isolation(self):
        history = make_history()
        world = WorldModel(8, hidden=16, layers=1, heads=4).eval()
        language = torch.rand(2, 3, 8)
        prediction = world(history.pack(), language, torch.tensor([4, 4]))
        self.assertEqual(prediction['tokens'].shape, (2, 4, 8))
        with self.assertRaises(ValueError):
            world(history.pack(), language, torch.tensor([3, 4]))
        with self.assertRaises(ValueError):
            history.append(torch.rand(2, 4, 8), torch.rand(2, 4), torch.ones(2),
                           torch.ones(2, dtype=torch.bool), torch.tensor([3, 3]))
        # Move beyond the initial zero output heads to make isolation meaningful.
        nn.init.normal_(world.appearance_head.weight, std=.1)
        first = world(history.pack(), language, torch.tensor([4, 4]))['tokens'][0]
        altered = history.pack()
        altered['tokens'][1] += 100
        second = world(altered, language, torch.tensor([4, 4]))['tokens'][0]
        self.assertTrue(torch.equal(first, second))
        history.clear()
        self.assertIsNone(history.pack())

    def test_invalid_history_has_finite_gradients(self):
        history = make_history()
        history.items[0]['valid'][1] = False
        history.items[0]['tokens'][1] = float('inf')
        history.items[0]['boxes'][1] = float('nan')
        world = WorldModel(8, hidden=16, layers=1, heads=4)
        result = world(history.pack(), torch.rand(2, 3, 8), torch.tensor([4, 4]))
        loss = result['tokens'].square().mean() + result['delta'].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in world.parameters() if p.grad is not None))

    def test_roi_invalid_and_fusion_bypass(self):
        features = torch.rand(2, 16, 8)
        roi, valid = extract_roi(features, torch.tensor([[.2, .3, .4, .4], [float('nan'), 0., 1., 1.]]))
        self.assertEqual(valid.tolist(), [True, False])
        self.assertTrue(torch.isfinite(roi).all())
        fusion = ImaginationFusion(8, hidden=16, heads=4)
        prediction = dict(tokens=torch.rand(2, 4, 8), boxes=torch.tensor([[.5, .5, .3, .3]]).repeat(2, 1),
                          valid=torch.ones(2, dtype=torch.bool), quality=torch.ones(2))
        regions = torch.tensor([[0., 0., 1., 1.]]).repeat(2, 1)
        self.assertTrue(torch.equal(features, fusion(features, prediction, regions)))
        nn.init.normal_(fusion.output.weight, std=.1)
        prediction['valid'].zero_()
        self.assertTrue(torch.equal(features, fusion(features, prediction, regions)))

    def test_stages_gradients_rollout_and_checkpoint(self):
        model, cfg = model_and_cfg()
        clip = dict(images=[np.full((80, 100, 3), 100 + t, dtype=np.uint8) for t in range(8)],
                    boxes=torch.tensor([[35., 25., 30., 30.]]).repeat(8, 1),
                    valid=torch.ones(8, dtype=torch.bool), visible=torch.ones(8, dtype=torch.bool),
                    frame_ids=torch.arange(8), language='a target')
        for stage in ('A', 'B'):
            model.configure_world_stage(stage)
            model.train()
            self.assertFalse(model.encoder.training)
            self.assertEqual(model.world_model.training, stage == 'A')
            baseline = {key: value.clone() for key, value in model.state_dict().items()}
            actor = WorldModelActor(model, cfg, stage)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = actor([clip])
                self.assertIsNotNone(loss)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        self.assertIsNone(parameter.grad, name)
                optimizer.step()
            updated = [key for key, value in model.state_dict().items() if not torch.equal(value, baseline[key])]
            expected = ('world_model.',) if stage == 'A' else ('head.', 'imagination_fusion.')
            self.assertTrue(updated)
            self.assertTrue(all(key.startswith(expected) for key in updated), updated)
            if stage == 'A':
                self.assertTrue(any(key.startswith('world_model.transformer.') for key in updated))
            else:
                self.assertTrue(any(key.startswith('imagination_fusion.') for key in updated))
            model.zero_grad(set_to_none=True)
        clone, _ = model_and_cfg()
        checkpoint = dict(net=model.state_dict(), world_stage='B', world_spec=world_spec(model))
        load_tracker_weights(clone, checkpoint, require_world=True)
        for key, value in clone.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))
        del checkpoint['net']['world_model.future_queries']
        with self.assertRaises(ValueError):
            load_tracker_weights(clone, checkpoint, require_world=True)
        with self.assertRaises(ValueError):
            load_predictor(clone, dict(world_stage='B'))

    def test_manifest_training_cli_resume_and_stage_b(self):
        import train_world_model as trainer
        from lib.train.dataset.world_clips import read_manifest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data'
            for name in ('video_a', 'video_b'):
                directory = data / 'train' / name
                (directory / 'imgs').mkdir(parents=True)
                for frame in range(10):
                    cv2.imwrite(str(directory / 'imgs' / ('%04d.jpg' % frame)),
                                np.full((80, 100, 3), 120 + frame, dtype=np.uint8))
                np.savetxt(str(directory / 'groundtruth.txt'), np.tile([35, 25, 30, 30], (10, 1)), delimiter=',')
                (directory / 'language.txt').write_text('a target')
            manifest = dict(protocol='official-train-video-split-v1', roots=dict(tnl2k=str(data)),
                            train=[dict(dataset='tnl2k', name='video_a')],
                            val=[dict(dataset='tnl2k', name='video_b')])
            manifest_path = root / 'split.json'
            manifest_path.write_text(json.dumps(manifest))
            partitions, digest = read_manifest(manifest_path, 8)
            self.assertEqual(len(partitions['train']), 1)
            duplicate = dict(manifest, val=manifest['train'])
            manifest_path.write_text(json.dumps(duplicate))
            with self.assertRaises(ValueError):
                read_manifest(manifest_path, 8)
            manifest_path.write_text(json.dumps(manifest))
            model, cfg = model_and_cfg()
            cfg.TRAIN.BATCH_SIZE = 1
            cfg.TRAIN.NUM_WORKER = 0
            cfg.TRAIN.WORLD.MAX_STEPS = 2
            cfg.TRAIN.WORLD.SAVE_INTERVAL = 1
            cfg.TRAIN.WORLD.PREDICTED_HISTORY_PROB = 0.
            baseline = root / 'baseline.pth.tar'
            torch.save(dict(net={key: value for key, value in model.state_dict().items()
                                 if not key.startswith(('world_model.', 'imagination_fusion.'))}), str(baseline))
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump(json.loads(json.dumps(cfg))))
            output = root / 'stage_a'
            arguments = ['train_world_model.py', '--config', str(config_path), '--manifest', str(manifest_path),
                         '--baseline', str(baseline), '--output', str(output), '--device', 'cpu', '--validation-clips', '1']
            factory = lambda cfg, training=False: model_and_cfg()[0]
            with patch.object(trainer, 'build_mvlm', side_effect=factory), patch.object(sys, 'argv', arguments):
                trainer.main()
            expected = torch.load(str(output / 'last.pth.tar'))
            self.assertEqual(expected['step'], 2)
            self.assertNotIn('net', expected)
            # Replay from a genuine intermediate checkpoint, including optimizer and RNG states.
            torch.save(torch.load(str(output / 'step_000001.pth.tar')), str(output / 'last.pth.tar'))
            with patch.object(trainer, 'build_mvlm', side_effect=factory), patch.object(sys, 'argv', arguments + ['--resume']):
                trainer.main()
            resumed = torch.load(str(output / 'last.pth.tar'))
            for key, value in expected['world_model'].items():
                self.assertTrue(torch.equal(value, resumed['world_model'][key]), key)
            cfg.TRAIN.WORLD.STAGE = 'B'
            config_path.write_text(yaml.safe_dump(json.loads(json.dumps(cfg))))
            stage_b = list(arguments)
            stage_b[stage_b.index('--output') + 1] = str(root / 'stage_b')
            stage_b += ['--world-checkpoint', str(output / 'last.pth.tar')]
            with patch.object(trainer, 'build_mvlm', side_effect=factory), patch.object(sys, 'argv', stage_b):
                trainer.main()
            adapted = torch.load(str(root / 'stage_b' / 'last.pth.tar'))
            self.assertEqual(adapted['world_stage'], 'B')
            for key, value in resumed['world_model'].items():
                self.assertTrue(torch.equal(value, adapted['net']['world_model.' + key]), key)
            import evaluate_world_model as evaluator
            eval_output = root / 'full_validation'
            evaluation_arguments = ['evaluate_world_model.py', '--config', str(config_path),
                '--manifest', str(manifest_path), '--checkpoint', str(root / 'stage_b' / 'last.pth.tar'),
                '--output', str(eval_output), '--device', 'cpu']
            with patch('lib.test.tracker.mvlm.build_mvlm', side_effect=factory), patch.object(sys, 'argv', evaluation_arguments):
                evaluator.main()
            summary = json.loads((eval_output / 'summary.json').read_text())
            self.assertEqual(summary['protocol'], 'full_validation_sequences')
            self.assertEqual(summary['sequences'][0]['frames'], 10)
            self.assertTrue(np.isfinite(summary['macro_AUC']))

    def test_disabled_path_and_config_isolation(self):
        model, cfg = model_and_cfg()
        data = torch.randn(1, 6, 64, 64)
        text = model.forward_textencoder(['target'])
        expected = model.forward_head(model.forward_encoder(None, [data], text)['feat_x'])
        actual = model(search_list=[data], text_emb=text)
        self.assertTrue(torch.equal(expected['score_map'], actual['score_map']))
        from lib.test.parameter.mvlm import parameters
        enabled = parameters('mvlm_TF_world_stage_b', 'unused')
        disabled = parameters('mvlm_TF', 'unused')
        self.assertTrue(enabled.cfg.MODEL.WORLD.ENABLED)
        self.assertFalse(disabled.cfg.MODEL.WORLD.ENABLED)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
