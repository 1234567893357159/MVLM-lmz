"""Paired, fixed-training-crop diagnostic; not online validation performance."""
import copy
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.config.mvlm.config import cfg,update_config_from_file
from lib.train.admin.settings import Settings
from lib.train.base_functions import update_settings,build_dataloaders
from lib.models.mvlm import build_mvlm
from lib.train.actors.mvlm import MVLM_Actor
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss
from torch.nn.functional import l1_loss

random.seed(20261019);np.random.seed(20261019);torch.manual_seed(20261019);torch.set_num_threads(1)
update_config_from_file(str(ROOT/'experiments/mvlm/mvlm_TF_memory_three_custom.yaml'))
cfg=copy.deepcopy(cfg);cfg.DATA.SEARCH.NUMBER=12;cfg.TRAIN.BATCH_SIZE=2;cfg.TRAIN.NUM_WORKER=0;cfg.DATA.TRAIN.SAMPLE_PER_EPOCH=2
net=build_mvlm(cfg);state=torch.load('/root/MVLM/output/memory_three_custom_20261018_run01/checkpoints/train/mvlm/mvlm_TF_memory_three_custom/MVLM_ep0020.pth.tar',map_location='cpu')
net.load_state_dict(state['net'],strict=False);net.cuda().eval()
results={}
for dataset in ['OTB99_train','LASOT','TNL2K_train']:
    config=copy.deepcopy(cfg);config.DATA.TRAIN.DATASETS_NAME=[dataset];config.DATA.TRAIN.DATASETS_RATIO=[1]
    settings=Settings();settings.local_rank=-1;update_settings(settings,config)
    loader=build_dataloaders(config,settings);data=next(iter(loader)).to(torch.device('cuda:0'))
    actor=MVLM_Actor(net,{'giou':giou_loss,'l1':l1_loss,'focal':FocalLoss()},
                    {'giou':config.TRAIN.GIOU_WEIGHT,'l1':config.TRAIN.L1_WEIGHT,'focal':1.},settings,config)
    values={}
    for mode,prob in [('gt_history',0.),('mixed_history',.5),('predicted_history',1.),('fusion_disabled',1.)]:
        hook=None;config.TRAIN.MEMORY_PREDICTED_PROB=prob;torch.manual_seed(77)
        if mode=='fusion_disabled':
            hook=net.memory_fusion.register_forward_hook(lambda module,args,output:(args[0],torch.zeros_like(output[1])))
        with torch.no_grad():loss,stats=actor(data)
        if hook:hook.remove()
        values[mode]={k:float(v) for k,v in stats.items()}
    results[dataset]=values;print(dataset,json.dumps(values),flush=True)
out=Path('/root/MVLM/output/memory_three_custom_20261018_run01/diagnostics/training_crop_gap.json')
out.write_text(json.dumps({'results':results,'clips_per_dataset':2,'frames_per_clip':12,
                          'limitations':'One fixed augmented training batch per dataset. Ground-truth-centered crops, not recursive online predictions; exploratory evidence only.'},indent=2)+'\n')
