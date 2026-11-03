"""CPU regression checks for concat memory, initialization and tracking gradients."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from lib.config.mvlm.config import cfg
from lib.local_config import EnvSettings
from lib.models.mvlm.checkpoint import CENTER_INPUT_WEIGHTS, prepare_memory_checkpoint
from lib.models.mvlm.memory import TargetMemoryBank
from lib.models.mvlm.memory_fusion import MemoryFusion
from lib.models.mvlm.mvlm import build_mvlm
from lib.test.parameter.mvlm import parameters
from lib.test.tracker.mvlm import MVLM as OnlineTracker
from lib.train.actors.mvlm import MVLM_Actor
from lib.utils import TensorDict
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss
from lib.train.memory_run_utils import memory_optimizer, set_step_lr, bn_state, assert_bn_unchanged


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = copy.deepcopy(cfg)
OUTPUT_KEYS = ('pred_boxes', 'score_map', 'size_map', 'offset_map')


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_channels = 32
        self.body = nn.Conv2d(3, 32, 16, stride=16)
        self.body.num_patches_search = 16
        self.body.num_patches_template = 1

    def forward(self, template_list, search_list, text_emb):
        return {'feat_x': self.body(search_list[0]).flatten(2).transpose(1, 2)}


def config_for(mode='concat'):
    config = copy.deepcopy(DEFAULTS)
    config.MODEL.MEMORY.ENABLED = True
    config.MODEL.MEMORY.FUSION_MODE = mode
    config.MODEL.MEMORY.HIDDEN_DIM = 16
    config.MODEL.MEMORY.KEY_DIM = 8
    config.MODEL.HEAD.NUM_CHANNELS = 32
    config.DATA.MULTI_MODAL_LANGUAGE = False
    config.DATA.MULTI_MODAL_VISION = False
    config.DATA.SEARCH.SIZE = 64
    config.DATA.SEARCH.NUMBER = 3
    config.TRAIN.MEMORY_FREEZE_HEAD = False
    config.TRAIN.CM_LOSS = False
    return config


def build_tiny(config, training=False):
    # Exercise the real builder/head/checkpoint path, replacing only the costly
    # visual backbone. All memory, tracking heads and losses are real modules.
    with mock.patch('lib.models.mvlm.mvlm.build_encoder', side_effect=lambda *a, **k: TinyEncoder()):
        return build_mvlm(config, training=training)


class MemoryConcatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.baseline_cfg = config_for()
        self.baseline_cfg.MODEL.MEMORY.ENABLED = False
        self.baseline = build_tiny(self.baseline_cfg).eval()
        self.baseline_path = str(Path(self.temp.name) / 'baseline.pth.tar')
        torch.save({'net': self.baseline.state_dict()}, self.baseline_path)
        self.config = config_for()
        self.config.MODEL.PRETRAINED_MODEL = self.baseline_path

    def initialized_model(self):
        return build_tiny(self.config, training=True)

    def test_concat_preserves_visual_and_masks_empty_samples(self):
        fusion = MemoryFusion(32, 16, 8, fusion_mode='concat')
        self.assertFalse(hasattr(fusion, 'output_proj'))
        features = torch.randn(2, 16, 32)
        bank = TargetMemoryBank(2)
        empty, gate = fusion(features, None, bank, [1, 1])
        self.assertEqual(tuple(empty.shape), (2, 16, 48))
        self.assertTrue(torch.equal(empty[..., :32], features))
        self.assertEqual(int(torch.count_nonzero(empty[..., 32:])), 0)
        self.assertEqual(int(torch.count_nonzero(gate)), 0)
        bank.write(torch.randn(2, 4, 32), torch.tensor([[.2, .2, .4, .4]] * 2),
                   [1, 1], [1., 0.])
        read, gate = fusion(features, torch.randn(2, 3, 32), bank, [2, 2])
        self.assertTrue(torch.equal(read[..., :32], features))
        self.assertGreater(float(read[0, :, 32:].abs().sum()), 0)
        self.assertEqual(int(torch.count_nonzero(read[1, :, 32:])), 0)
        self.assertEqual(int(torch.count_nonzero(gate[1])), 0)
        read[0, :, 32:].square().mean().backward()
        for parameter in (fusion.semantic_key[0].weight, fusion.stream_gate.weight,
                          fusion.text_proj.weight):
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_baseline_migration_and_initial_prediction_equivalence(self):
        model = self.initialized_model().eval()
        self.assertEqual(model.head.memory_channels, 16)
        base_state, state = self.baseline.state_dict(), model.state_dict()
        for name in CENTER_INPUT_WEIGHTS:
            self.assertEqual(state[name].shape[1], 48)
            self.assertTrue(torch.equal(state[name][:, :32], base_state[name]))
            self.assertEqual(int(torch.count_nonzero(state[name][:, 32:])), 0)
        for name in base_state:
            if name not in CENTER_INPUT_WEIGHTS:
                self.assertTrue(torch.equal(state[name], base_state[name]), name)
        images = [torch.randn(2, 3, 64, 64)]
        bank = model.new_memory(2)
        bank.write(torch.randn(2, 4, 32), torch.tensor([[.2, .2, .4, .4]] * 2),
                   [1, 1], [1., 1.])
        with torch.no_grad():
            expected = self.baseline(search_list=images)
            actual = model(search_list=images, memory_state=bank, frame_ids=torch.tensor([2, 2]))
            for name in OUTPUT_KEYS:
                torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)
            self.assertEqual(actual['feat_x'].shape[-1], 32)
            self.assertEqual(actual['feat_x_mem'].shape[-1], 48)
            model.memory_enabled = False
            disabled = model(search_list=images)
            for name in OUTPUT_KEYS:
                torch.testing.assert_close(disabled[name], expected[name], rtol=1e-5, atol=1e-6)

    def test_sequence_loss_trains_expanded_head_and_retrieval(self):
        model = self.initialized_model().train()
        self.assertFalse(model.encoder.training)
        encoder_state = {k: v.clone() for k, v in model.encoder.state_dict().items()}
        actor = MVLM_Actor(model, {'giou': giou_loss, 'l1': nn.functional.l1_loss, 'focal': FocalLoss()},
                           {'giou': 2., 'l1': 5., 'focal': 1.},
                           SimpleNamespace(batchsize=2), self.config)
        images = torch.randn(3, 2, 3, 64, 64)
        data = TensorDict(search_images=images, template_images=images[:1],
                          search_anno=torch.tensor([[[.2, .2, .4, .4]] * 2] * 3),
                          search_frame_ids=torch.tensor([[1, 1], [3, 3], [5, 5]]),
                          search_box_transforms=torch.tensor([[[0., 0., 64., 64.]] * 2] * 3),
                          search_source_sizes=torch.tensor([[[64., 64.]] * 2] * 3),
                          search_visible=torch.ones(3, 2, dtype=torch.bool))
        self.config.TRAIN.MEMORY_PREDICTED_PROB = 0.
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
        for step in range(2):
            optimizer.zero_grad()
            loss, _ = actor(data)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            for name in CENTER_INPUT_WEIGHTS:
                self.assertGreater(float(dict(model.named_parameters())[name].grad[:, 32:].abs().sum()), 0, name)
            if step:
                self.assertGreater(float(model.memory_fusion.appearance_key.weight.grad.abs().sum()), 0)
            optimizer.step()
        for name, value in encoder_state.items():
            self.assertTrue(torch.equal(value, model.encoder.state_dict()[name]), name)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))

    def test_concat_checkpoint_roundtrip_and_online_loading(self):
        model = self.initialized_model().eval()
        with torch.no_grad():
            model.head.conv1_ctr[0].weight[:, 32:].normal_(std=.02)
        checkpoint = str(Path(self.temp.name) / 'concat.pth.tar')
        torch.save({'net': model.state_dict()}, checkpoint)
        params = SimpleNamespace(cfg=copy.deepcopy(self.config), checkpoint=checkpoint, device='cpu', debug=False)
        with mock.patch('lib.test.tracker.mvlm.build_mvlm', side_effect=lambda config, **k: build_tiny(config)):
            tracker = OnlineTracker(params, 'otb99_lang')
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, tracker.network.state_dict()[name]), name)

    def test_untrained_online_guard_and_explicit_baseline_smoke(self):
        params = SimpleNamespace(cfg=copy.deepcopy(self.config), checkpoint=self.baseline_path, device='cpu', debug=False)
        with mock.patch('lib.test.tracker.mvlm.build_mvlm', side_effect=lambda config, **k: build_tiny(config)):
            with self.assertRaisesRegex(ValueError, 'trained memory module'):
                OnlineTracker(params, 'otb99_lang')
            params.cfg.TEST.MEMORY_ALLOW_UNTRAINED = True
            with self.assertWarnsRegex(UserWarning, 'smoke check'):
                tracker = OnlineTracker(params, 'otb99_lang')
        self.assertEqual(tracker.network.head.conv1_ctr[0].in_channels, 48)

    def test_rejects_residual_checkpoint_and_frozen_concat_head(self):
        residual = build_tiny(config_for('residual'))
        with self.assertRaisesRegex(ValueError, 'residual-memory checkpoint'):
            prepare_memory_checkpoint(self.initialized_model(), residual.state_dict(), allow_baseline=True)
        self.config.TRAIN.MEMORY_FREEZE_HEAD = True
        with self.assertRaisesRegex(ValueError, 'MEMORY_FREEZE_HEAD=False'):
            build_tiny(self.config, training=True)
        with self.assertRaisesRegex(ValueError, 'Unknown memory fusion mode'):
            MemoryFusion(32, fusion_mode='typo')

    def test_old_yaml_restores_residual_architecture_without_config_leakage(self):
        concat = parameters('mvlm_TF_memory_concat', self.baseline_path)
        residual = parameters('mvlm_TF_memory', self.baseline_path)
        baseline = parameters('mvlm_TF', self.baseline_path)
        self.assertEqual(concat.cfg.MODEL.MEMORY.FUSION_MODE, 'concat')
        self.assertEqual(residual.cfg.MODEL.MEMORY.FUSION_MODE, 'residual')
        self.assertFalse(baseline.cfg.MODEL.MEMORY.ENABLED)
        self.assertEqual(baseline.cfg.MODEL.MEMORY.FUSION_MODE, 'residual')

    def test_worktree_paths_keep_existing_evaluation_outputs_separate(self):
        env = EnvSettings()
        self.assertEqual(Path(env.workspace_dir), ROOT)
        self.assertEqual(Path(env.results_dir), ROOT / 'output/test/tracking_results')
        self.assertEqual(Path(env.lasot_dir), Path('/root/MVLM/data/lasot'))

    def test_frozen_bn_affine_training_and_optimizer_groups(self):
        self.config.TRAIN.MEMORY_FREEZE_BN_STATS = True
        model = self.initialized_model().train()
        model.train()
        before = bn_state(model)
        for module in model.head.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertFalse(module.training)
                self.assertTrue(module.weight.requires_grad and module.bias.requires_grad)
        optimizer = memory_optimizer(model, self.config)
        groups = [{id(p) for p in g['params']} for g in optimizer.param_groups]
        self.assertFalse(groups[0] & groups[1])
        self.assertEqual(groups[0] | groups[1], {id(p) for p in model.parameters() if p.requires_grad})
        output = model(search_list=[torch.randn(2, 3, 64, 64)])
        output['score_map'].mean().backward()
        self.assertIsNotNone(model.head.conv1_ctr[1].weight.grad)
        optimizer.step()
        assert_bn_unchanged(model, before)

    def test_step_learning_rates_at_schedule_boundaries(self):
        optimizer = memory_optimizer(self.initialized_model(), self.config)
        for step, factor in ((1,.1045),(200,1.),(201,1.),(3000,1.),(3001,.1),(4000,.1)):
            set_step_lr(optimizer, step)
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'], self.config.TRAIN.LR*factor)
            self.assertAlmostEqual(optimizer.param_groups[1]['lr'], self.config.TRAIN.MEMORY_HEAD_LR*factor)

    def test_run_specific_bn_and_sequence_lists_do_not_leak_to_old_yaml(self):
        from lib.config.mvlm.config import update_config_from_file
        path=Path(self.temp.name)/'run.yaml'
        path.write_text('TRAIN:\n  MEMORY_FREEZE_BN_STATS: true\nDATA:\n  TRAIN:\n    SEQUENCE_LISTS:\n      LASOT: frozen-list.txt\n')
        update_config_from_file(str(path))
        old=parameters('mvlm_TF_memory',self.baseline_path).cfg
        self.assertFalse(old.TRAIN.MEMORY_FREEZE_BN_STATS)
        self.assertTrue(all(not value for value in old.DATA.TRAIN.SEQUENCE_LISTS.values()))


if __name__ == '__main__':
    unittest.main()
