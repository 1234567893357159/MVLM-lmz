"""Behavior checks for causal memory-conditioned forecasting and adaptation."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
import yaml

from lib.config.mvlm.config import _DEFAULT_CFG, load_config
from lib.models.mvlm.mvlm import build_mvlm
from lib.models.mvlm.memory import TargetMemoryBank
from lib.models.mvlm.memory_reader import MemoryReader
from lib.models.mvlm.world_state import crop_to_image, image_to_crop
from lib.models.mvlm.temporal_core import TemporalState, predict_prior, locate, observe, select_candidates
from lib.models.mvlm.joint_checkpoint import load_baseline, load_joint, joint_spec
from lib.train.actors.memory_world import MemoryWorldActor, candidate_iou_targets
from lib.test.tracker.utils import Preprocessor

ROOT = Path(__file__).resolve().parents[1]


class TinyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_channels = 32
        self.body = nn.Conv2d(3, 32, 16, stride=16)
        self.body.num_patches_search = (config.DATA.SEARCH.SIZE // 16) ** 2
        self.body.num_patches_template = 4

    def forward(self, templates, searches, text):
        return {'feat_x': self.body(searches[0]).flatten(2).transpose(1, 2)}


class TinyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(77, 32)

    def forward(self, text):
        count = 1 if isinstance(text, str) else len(text)
        return self.embedding(torch.arange(77, device=self.embedding.weight.device))[None].expand(count, -1, -1)


def config(memory=True, world=True, mode='cross_attention'):
    c = copy.deepcopy(_DEFAULT_CFG)
    c.MODEL.MEMORY.ENABLED, c.MODEL.WORLD.ENABLED = memory, world
    c.MODEL.MEMORY.HIDDEN_DIM = c.MODEL.FUSION.CHANNELS = c.MODEL.WORLD.HIDDEN = 16
    c.MODEL.MEMORY.KEY_DIM = 8
    c.MODEL.MEMORY.MIN_QUALITY = 0.
    c.MODEL.MEMORY.WRITE_CONSISTENCY = 0.
    c.MODEL.WORLD.HISTORY = 2
    c.MODEL.WORLD.LAYERS = 1
    c.MODEL.WORLD.HEADS = 4
    c.MODEL.WORLD.QUALITY_ENABLED = world
    c.MODEL.FUSION.MODE = mode
    c.MODEL.HEAD.NUM_CHANNELS = 32
    c.MODEL.ENCODER.STRIDE = 16
    c.DATA.SEARCH.SIZE = c.TEST.SEARCH_SIZE = 64
    c.DATA.TEMPLATE.SIZE = c.TEST.TEMPLATE_SIZE = 32
    c.TEST.SEARCH_FACTOR = 8.
    c.TEST.MVLM.TAU = -2.  # Deterministic acceptance for synthetic pipeline checks.
    c.DATA.MULTI_MODAL_VISION = False
    c.DATA.MULTI_MODAL_LANGUAGE = True
    c.TEST.MULTI_MODAL_LANGUAGE.DEFAULT = True
    c.TEST.USE_NLP.DEFAULT = True
    c.TRAIN.WORLD.CLIP_LENGTH = 6
    c.TRAIN.WORLD.WARMUP_FRAMES = 2
    c.TRAIN.WORLD.HISTORY_RAMP_STEPS = 0
    c.TRAIN.WORLD.PREDICTED_HISTORY_PROB = 0.
    c.TRAIN.WORLD.TRAIN_HEAD = mode == 'concat'
    c.DATA.TRAIN.DATASETS_NAME = ['TNL2K_train']
    c.DATA.TRAIN.DATASETS_RATIO = [1]
    c.TRAIN.BATCH_SIZE = 1
    c.TRAIN.NUM_WORKER = 0
    c.TRAIN.SCHEDULER.TYPE = 'cosine'
    return c


def tiny_model(c):
    with mock.patch('lib.models.mvlm.mvlm.build_encoder', side_effect=lambda *a, **k: TinyEncoder(c)), \
         mock.patch('lib.models.mvlm.mvlm.build_clip_textencoder', side_effect=lambda *a, **k: TinyText()):
        return build_mvlm(c, training=False)


def clip(length=6):
    rng = np.random.RandomState(1)
    return dict(images=[rng.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(length)],
                frame_ids=torch.arange(length), boxes=torch.tensor([[20., 20., 20., 20.]] * length),
                valid=torch.ones(length, dtype=torch.bool), visible=torch.ones(length, dtype=torch.bool),
                language='the target')


class MemoryWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(11)

    def test_config_isolation(self):
        one = load_config(ROOT / 'experiments/mvlm/mvlm_TF_memory_world_stage_a.yaml')
        two = load_config(ROOT / 'experiments/mvlm/mvlm_TF.yaml')
        self.assertTrue(one.MODEL.MEMORY.ENABLED)
        self.assertFalse(two.MODEL.MEMORY.ENABLED or two.MODEL.WORLD.ENABLED)

    def test_baseline_initialization_and_empty_forecast_equivalence(self):
        baseline = tiny_model(config(False, False)).eval()
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            reference = baseline([], [image], raw_text=['target'])
        for mode in ('cross_attention', 'concat'):
            model = tiny_model(config(mode=mode)).eval()
            load_baseline(model, {'net': baseline.state_dict()})
            with torch.no_grad():
                output = model([], [image], raw_text=['target'])
            for key in ('score_map', 'pred_boxes', 'size_map', 'offset_map'):
                torch.testing.assert_close(output[key], reference[key], rtol=1e-6 if mode == 'concat' else 0, atol=2e-7 if mode == 'concat' else 0)

    def test_bank_batch_isolation_and_causality(self):
        c = config().MODEL.MEMORY
        bank = TargetMemoryBank(2, short_capacity=2, long_capacity=2, min_quality=0.)
        bank.write(torch.randn(2, 4, 32), torch.tensor([[.1,.1,.2,.2]] * 2), [1,1], [1.,1.], torch.tensor([True,False]))
        reader = MemoryReader(32, c)
        result = reader(torch.randn(2, 4, 32), torch.randn(2, 77, 32), bank, [2,2])
        self.assertEqual(result['valid'].tolist(), [True,False])
        self.assertEqual(result['tokens'][1].count_nonzero().item(), 0)
        with self.assertRaises(ValueError):
            reader(torch.randn(2,4,32), None, bank, [1,1])
        result['tokens'].square().mean().backward()
        self.assertGreater(reader.state_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(reader.semantic_key[0].weight.grad.abs().sum().item(), 0)

    def ready_state(self, model, c):
        state = TemporalState(c)
        for t in range(2):
            tokens = torch.randn(1,4,32)
            box = torch.tensor([[.5,.5,.3,.3]])
            state.history.append(tokens, box, torch.ones(1), torch.ones(1,dtype=torch.bool), torch.tensor([t]))
            state.bank.write(tokens, torch.tensor([[.35,.35,.3,.3]]), [t], [1.])
        state.last_visual = torch.randn(1,16,32)
        return state

    def test_memory_conditions_forecast_without_future_observation(self):
        c = config(); model = tiny_model(c)
        state = self.ready_state(model,c)
        language = model.forward_textencoder(['target'])
        filled = predict_prior(model,state,language,torch.tensor([2]))
        state.bank.reset()
        empty = predict_prior(model,state,language,torch.tensor([2]))
        self.assertFalse(torch.allclose(filled['latent'],empty['latent']))
        with self.assertRaises(ValueError):
            predict_prior(model,state,language,torch.tensor([1]))
        state.reset()
        self.assertIsNone(predict_prior(model,state,language,torch.tensor([2])))
        self.assertIsNone(state.last_visual)

    def test_quality_head_is_geometry_conditioned_not_observation_conditioned(self):
        c=config(); model=tiny_model(c); state=self.ready_state(model,c)
        language=model.forward_textencoder(['target'])
        prior=predict_prior(model,state,language,torch.tensor([2]))
        region=torch.tensor([[0.,0.,1.,1.]])
        logits=model.quality_head(prior,region,4)
        self.assertEqual(tuple(logits.shape),(1,16))
        shifted=model.quality_head(prior,torch.tensor([[.2,0.,.8,1.]]),4)
        self.assertFalse(torch.allclose(logits,shifted))
        targets=candidate_iou_targets(torch.tensor([[.5,.5,.2,.2],[.1,.1,.1,.1]],requires_grad=True),torch.tensor([[.4,.4,.2,.2]]))
        torch.testing.assert_close(targets,torch.tensor([1.,0.]))
        self.assertFalse(targets.requires_grad)

    def test_frame_step_does_not_write_until_commit(self):
        c=config(); model=tiny_model(c).eval(); state=TemporalState(c)
        lang=model.forward_textencoder(['target'])
        sample=clip(); result=locate(model,state,sample['images'][0],lang,c,torch.tensor([0]),Preprocessor('cpu'))
        self.assertEqual(len(state.history.items),0)
        self.assertEqual(len(state.bank.entries(0)),0)
        observe(model,state,result,torch.tensor([0]))
        self.assertEqual(len(state.history.items),1)
        self.assertEqual(len(state.bank.entries(0)),1)
        self.assertFalse(state.bank.entries(0)[0].tokens.requires_grad)
        with self.assertRaises(ValueError):
            observe(model,state,result,torch.tensor([0]))

    def test_quality_ranking_preserves_confidence_and_fallback(self):
        c=config();model=tiny_model(c).eval();model.world_stage='B'
        x=torch.randn(1,3,64,64);lang=model.forward_textencoder(['target'])
        output=model([], [x], text_emb=lang)
        a=select_candidates(model,output,lang,None,c)
        output['quality_logits']=torch.linspace(-20,20,16)[None]
        output['prediction_reliability']=torch.ones(1)
        c.MODEL.WORLD.QUALITY_WEIGHT=0.
        b=select_candidates(model,output,lang,None,c)
        self.assertEqual(a['index'].item(),b['index'].item())
        torch.testing.assert_close(a['kmem'],b['kmem'])
        c.TEST.MVLM.TAU=2.;c.MODEL.WORLD.QUALITY_WEIGHT=1.
        fallback=select_candidates(model,output,lang,None,c)
        self.assertEqual(fallback['index'].item(),fallback['scores'].argmax().item())

    def test_stage_a_updates_reader_world_and_quality_not_baseline(self):
        c=config();model=tiny_model(c);model.configure_world_stage('A')
        before={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('encoder.','text_encoder.','head.'))}
        actor=MemoryWorldActor(model,c,'A');loss,metrics=actor([clip()])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in (model.memory_reader,model.world_model,model.quality_head):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None),0)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        optimizer.step()
        for k,v in before.items(): torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
        self.assertGreater(metrics['memory_writes'],0)

    def test_stage_b_and_memory_only_have_tracking_gradients(self):
        for world,mode in ((True,'cross_attention'),(True,'concat'),(False,'cross_attention')):
            c=config(world=world,mode=mode);model=tiny_model(c);model.configure_world_stage('B')
            actor=MemoryWorldActor(model,c,'B');loss,_=actor([clip()])
            self.assertIsNotNone(loss)
            loss.backward()
            if mode == 'concat':
                self.assertGreater(model.head.conv1_ctr[0].weight.grad[:,32:].abs().sum().item(),0)
            else:
                self.assertGreater(sum(p.grad.abs().sum().item() for p in model.future_fusion.parameters() if p.grad is not None),0)
            self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
            if world: self.assertTrue(all(p.grad is None for p in model.world_model.parameters()))

    def test_strict_checkpoint_roundtrip_and_guards(self):
        c=config();model=tiny_model(c)
        ckpt=dict(net=model.state_dict(),joint_spec=joint_spec(model),world_stage='A')
        other=tiny_model(c);load_joint(other,ckpt)
        with self.assertRaises(ValueError):load_joint(other,ckpt,inference=True)
        ckpt['world_stage']='B';load_joint(other,ckpt,inference=True)
        broken=copy.deepcopy(ckpt);del broken['net']['quality_head.output.2.weight']
        with self.assertRaises(ValueError):load_joint(other,broken)
        with self.assertRaises(ValueError):load_baseline(other,ckpt)
        mismatch=tiny_model(config(mode='concat'))
        with self.assertRaises(ValueError):load_joint(mismatch,ckpt)

    def test_head_only_control_updates_only_center(self):
        c=config(memory=False);c.TRAIN.WORLD.HEAD_ONLY=True;c.TRAIN.WORLD.TRAIN_HEAD=True
        model=tiny_model(c);model.configure_world_stage('B')
        actor=MemoryWorldActor(model,c,'B');loss,_=actor([clip()]);loss.backward()
        self.assertTrue(all(name.startswith('head.') for name,p in model.named_parameters() if p.requires_grad))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.head.parameters() if p.grad is not None),0)

    def test_dataset_weights_are_not_video_count_weights(self):
        from lib.train.dataset.world_clips import WorldClips
        sequence=dict(frames=['frame']*6, boxes=np.asarray([[1.,1.,10.,10.]]*6,dtype=np.float32),
                      valid=np.ones(6,dtype=bool),visible=np.ones(6,dtype=bool),language='target')
        sequences=[dict(sequence,dataset='vasttrack',name=str(i)) for i in range(40)]
        sequences.append(dict(sequence,dataset='lasot',name='one'))
        dataset=WorldClips(sequences,2,200,17,augment=False,dataset_weights={'vasttrack':1,'lasot':1})
        with mock.patch('lib.train.dataset.world_clips.cv2.imread',return_value=np.zeros((16,16,3),dtype=np.uint8)):
            names=[dataset[i]['name'].split('/')[0] for i in range(200)]
            first,second=dataset[5],dataset[5]
        self.assertGreater(names.count('lasot'),70)
        self.assertLess(names.count('lasot'),130)
        self.assertEqual(first['name'],second['name'])
        torch.testing.assert_close(first['frame_ids'],second['frame_ids'])
        with self.assertRaises(ValueError):
            WorldClips(sequences,2,10,17,dataset_weights={'vasttrack':0,'lasot':1})

    def test_concat_fusion_receives_gradients_after_input_channels_update(self):
        c=config(mode='concat');model=tiny_model(c);model.configure_world_stage('B')
        actor=MemoryWorldActor(model,c,'B')
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        maximum=0.
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss,_=actor([clip()]);loss.backward()
            maximum=max(maximum,sum(p.grad.abs().sum().item() for p in model.future_fusion.parameters() if p.grad is not None))
            optimizer.step()
        self.assertGreater(maximum,0)

    def test_common_modules_share_initialization_across_ablation_flags(self):
        combined=tiny_model(config())
        world=tiny_model(config(memory=False))
        memory=tiny_model(config(world=False))
        for a,b in ((combined.world_model,world.world_model),
                    (combined.quality_head,world.quality_head),
                    (combined.memory_reader,memory.memory_reader)):
            for name,value in a.state_dict().items():
                torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)

    def test_shared_geometry_baseline_matches_disabled_injection_trajectory(self):
        c=config();c.MODEL.WORLD.QUALITY_WEIGHT=0.
        combined=tiny_model(c).eval()
        reference_cfg=config(False,False);reference_cfg.MODEL.TEMPORAL_BASELINE=True
        reference=tiny_model(reference_cfg).eval()
        baseline={key:value for key,value in reference.state_dict().items() if not key.startswith('future_fusion.')}
        load_baseline(combined,{'net':baseline})
        load_baseline(reference,{'net':baseline})
        states=[TemporalState(c),TemporalState(reference_cfg)]
        models=[combined,reference]
        with torch.no_grad():
            languages=[m.forward_textencoder(['target']) for m in models]
            for frame,image in enumerate(clip()['images']):
                results=[locate(m,s,image,l,m.cfg,torch.tensor([frame]),Preprocessor('cpu'))
                         for m,s,l in zip(models,states,languages)]
                self.assertEqual(results[0]['pixels'],results[1]['pixels'])
                for m,s,result in zip(models,states,results):observe(m,s,result,torch.tensor([frame]))

    def test_warmup_populates_history_without_gradients_or_loss(self):
        c=config();c.TRAIN.WORLD.WARMUP_FRAMES=4
        model=tiny_model(c);model.configure_world_stage('A')
        actor=MemoryWorldActor(model,c,'A');enabled=[]
        def recording(*args,**kwargs):
            enabled.append(torch.is_grad_enabled())
            return locate(*args,**kwargs)
        with mock.patch('lib.train.actors.memory_world.locate',side_effect=recording):
            loss,metrics=actor([clip()])
        self.assertEqual(enabled,[False]*4+[True]*2)
        self.assertEqual(metrics['supervised_frames'],2)
        self.assertEqual(metrics['memory_writes'],6)
        self.assertTrue(loss.requires_grad)

    def test_geometry_roundtrip(self):
        box=torch.tensor([[.55,.35,.2,.1]])
        region=torch.tensor([[-.1,.2,.8,.6]])
        torch.testing.assert_close(crop_to_image(image_to_crop(box,region),region),box)


if __name__=='__main__': unittest.main()
