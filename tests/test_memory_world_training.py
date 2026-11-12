"""CPU integration: staged CLI, data audit, checkpoint restore and online loading."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import torch
import yaml

from test_memory_world import config, tiny_model, ROOT
from lib.config.mvlm.config import load_config
from lib.models.mvlm.joint_checkpoint import load_joint
from lib.train.dataset.world_clips import collate_world_clips
from lib.test.tracker.mvlm import MVLM
from lib.test.utils import TrackerParams

sys.path.insert(0,str(ROOT/'tracking'))
import train_memory_world as trainer


class AnnotationClipFixture(torch.utils.data.Dataset):
    def __len__(self):
        return 512

    def __getitem__(self, index):
        return dict(images=[np.full((8, 8, 3), index % 256, dtype=np.uint8)],
                    boxes=torch.tensor([[float(index), 1., 2., 3.]]),
                    valid=torch.ones(1, dtype=torch.bool), visible=torch.ones(1, dtype=torch.bool),
                    frame_ids=torch.tensor([index]), language='target')


class TrainingIntegrationTests(unittest.TestCase):
    def test_large_batch_worker_transfers_annotations_without_fd_exhaustion(self):
        loader = torch.utils.data.DataLoader(AnnotationClipFixture(), batch_size=256, num_workers=1,
            prefetch_factor=1, multiprocessing_context='spawn', collate_fn=collate_world_clips, timeout=30)
        for batch_index, clips in enumerate(loader):
            self.assertEqual(len(clips), 256)
            for sample in (0, 255):
                expected = batch_index * 256 + sample
                self.assertEqual(clips[sample]['frame_ids'].item(), expected)
                self.assertEqual(clips[sample]['boxes'][0, 0].item(), float(expected))
                self.assertEqual(clips[sample]['valid'].dtype, torch.bool)
                self.assertEqual(int(clips[sample]['images'][0][0, 0, 0]), expected % 256)

    def test_staged_cli_restore_and_online_reset(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data'
            rng=np.random.RandomState(8)
            sequences=[]
            for name in ('train_video','val_video'):
                folder=data/'train'/name
                (folder/'imgs').mkdir(parents=True)
                for i in range(6):
                    cv2.imwrite(str(folder/'imgs'/('%04d.jpg'%i)),rng.randint(0,255,(64,64,3),dtype=np.uint8))
                (folder/'groundtruth.txt').write_text('20,20,20,20\n'*6)
                (folder/'language.txt').write_text('a target')
                sequences.append(dict(dataset='tnl2k',name=name))
            manifest=root/'split.json'
            manifest.write_text(json.dumps(dict(protocol='official-train-video-split-v1',roots={'tnl2k':str(data)},train=sequences[:1],val=sequences[1:])))
            c=config();c.TRAIN.WORLD.MAX_STEPS=2;c.TRAIN.WORLD.SAVE_INTERVAL=1
            c.TRAIN.PRINT_INTERVAL=1
            baseline=root/'baseline.pth.tar'
            torch.save({'net':tiny_model(config(False,False)).state_dict()},baseline)
            def save_config(stage):
                d=json.loads(json.dumps(c));d['TRAIN']['WORLD']['STAGE']=stage
                p=root/(stage+'.yaml');p.write_text(yaml.safe_dump(d));return p
            a,b,c_path=save_config('A'),save_config('B'),save_config('C')
            def run(stage,output,parent=None,resume=False):
                argv=['train_memory_world.py','--config',str(stage),'--manifest',str(manifest),
                      '--baseline',str(baseline),'--output',str(output),'--device','cpu','--validation-clips','1']
                if parent:argv+=['--world-checkpoint',str(parent)]
                if resume:argv+=['--resume']
                with mock.patch.object(sys,'argv',argv),mock.patch.object(trainer,'build_mvlm',side_effect=lambda cfg,training=False:tiny_model(cfg)):
                    trainer.main()
            with mock.patch('builtins.print'):
                run(a,root/'stage_a')
                # Resume completed A: exercises model, optimizer, RNG, source/data guards.
                run(a,root/'stage_a',resume=True)
                run(b,root/'stage_b',root/'stage_a'/'last.pth.tar')
                run(c_path,root/'stage_c',root/'stage_b'/'last.pth.tar')
            self.assertEqual(torch.load(root/'stage_c'/'last.pth.tar',map_location='cpu')['world_stage'],'C')
            checkpoint=torch.load(root/'stage_b'/'last.pth.tar',map_location='cpu')
            self.assertEqual(checkpoint['step'],2)
            self.assertEqual(checkpoint['world_stage'],'B')
            cfg=load_config(b);model=tiny_model(cfg);load_joint(model,checkpoint,inference=True)
            params=TrackerParams();params.cfg=cfg;params.checkpoint=str(root/'stage_b'/'last.pth.tar')
            params.device='cpu';params.debug=0
            params.template_factor=2.;params.template_size=32;params.search_factor=8.;params.search_size=64
            with mock.patch('lib.test.tracker.mvlm.build_mvlm',side_effect=lambda *a,**k:tiny_model(cfg)):
                tracker=MVLM(params,'tnl2k')
            image=rng.randint(0,255,(64,64,3),dtype=np.uint8)
            tracker.initialize(image,dict(init_bbox=[1,1,10,10],init_nlp='a target'))
            first=tracker.track(image,{'gt_bbox':[0,0,1,1]})['target_bbox']
            for _ in range(3):tracker.track(image)
            self.assertGreater(len(tracker.temporal_state.history.items),0)
            tracker.set_language('another target')
            self.assertEqual(len(tracker.temporal_state.history.items),0)
            self.assertEqual(len(tracker.temporal_state.bank.entries(0)),0)
            self.assertIsNone(tracker.temporal_state.last_visual)
            self.assertIsNone(tracker.kmem)
            # Template-free first prediction never consumes init GT as content memory.
            tracker.initialize(image,dict(init_bbox=[40,40,15,15],init_nlp='a target'))
            second=tracker.track(image,{'gt_bbox':[50,50,9,9]})['target_bbox']
            self.assertEqual(first,second)
            self.assertTrue(np.isfinite(second).all())


if __name__=='__main__':unittest.main()
