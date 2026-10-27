import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.mvlm import build_mvlm
from lib.train.actors import MVLM_Actor
from lib.utils.focal_loss import FocalLoss
from lib.utils.cm_loss import CMloss
# for import modules
import importlib
import numpy as np
import torch
import random

def init_seeds(seed): # it is not worked!
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def run(settings):
    settings.description = 'Training script for Goku series'


    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg # generate cfg from lib.config
    config_module.update_config_from_file(settings.cfg_file) #update cfg from experiments
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    base_seed = 1234567890
    if settings.local_rank != -1:
        init_seeds(base_seed + settings.local_rank)
    else:
        init_seeds(base_seed)

    # update settings based on cfg
    update_settings(settings, cfg)
    settings.vasttrack_split = cfg.DATA.TRAIN.VASTTRACK_SPLIT
    settings.save_every_epoch = cfg.TRAIN.SAVE_EVERY_EPOCH
    settings.base_seed = base_seed
    settings.wandb_config = dict(cfg.WANDB)
    settings.training_config = cfg

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_type = getattr(cfg.DATA, "LOADER", "tracking")
    if loader_type == "tracking":
        loader_train = build_dataloaders(cfg, settings)
    else:
        raise ValueError("illegal DATA LOADER")


    # Create network
    if settings.script_name == "mvlm":
        net = build_mvlm(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, broadcast_buffers=False, device_ids=[settings.local_rank], find_unused_parameters=True) # modify the find_unused_parameters to False to skip a runtime error of twice variable ready
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    # Loss functions and Actors
    if  settings.script_name == "mvlm":
        focal_loss = FocalLoss()
        if cfg.TRAIN.CM_LOSS:
            cm_loss = CMloss()
        else:
            cm_loss = None # dummy loss
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(), 'cm': cm_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT, 'cm': cfg.TRAIN.CM_WEIGHT if cfg.TRAIN.CM_LOSS else 0.}
        actor = MVLM_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    if settings.local_rank in [-1, 0]:
        os.makedirs(settings.save_dir, exist_ok=True)
        import hashlib
        import json
        model = getattr(net, 'module', net)
        groups = {}
        initial = {}
        for name, parameter in model.named_parameters():
            group = ('clip' if name.startswith('text_encoder.clip.') else
                     'text_projection' if name.startswith('text_encoder.text_proj.') else
                     'encoder' if name.startswith('encoder.') else 'head')
            entry = groups.setdefault(group, {'total': 0, 'trainable': 0})
            entry['total'] += parameter.numel()
            if parameter.requires_grad:
                entry['trainable'] += parameter.numel()
                if group not in initial:
                    initial[group] = {'name': name, 'sha256': hashlib.sha256(
                        parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
        if cfg.TRAIN.TYPE == 'text_frozen' and cfg.DATA.MULTI_MODAL_LANGUAGE:
            assert groups['clip']['trainable'] == 0
            assert groups['text_projection']['trainable'] > 0
        with open(os.path.join(settings.save_dir, 'initialization.json'), 'w') as handle:
            json.dump({'parameter_groups': groups, 'initial_parameter_samples': initial,
                       'backbone_initialization': cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                       'tracker_initialization': cfg.MODEL.PRETRAINED_MODEL}, handle, indent=2)
        print('Parameter groups:', groups, flush=True)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train
    try:
        trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
    except BaseException:
        trainer.wandb_logger.finish(exit_code=1)
        raise
    else:
        trainer.wandb_logger.finish()
