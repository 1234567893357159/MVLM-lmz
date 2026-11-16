"""Shared loading, hashing and experiment metadata for reliable-memory CLIs."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def atomic_save(path, value):
    temporary = Path(str(path)+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_loaded_model(cfg, checkpoint, device, training=False, allow_untrained=False):
    from lib.models.mvlm import build_mvlm
    if cfg.MODEL.PRETRAINED_MODEL is not None:
        raise ValueError('Set MODEL.PRETRAINED_MODEL=null and supply --checkpoint explicitly')
    if not cfg.MODEL.ENCODER.TEMPLATE_FREE or not cfg.TEST.RELIABLE_CORE:
        raise ValueError('Require template-free shared reliable core')
    os.environ['MVLM_DEVICE'] = str(device)
    model = build_mvlm(cfg, training=False)
    loaded = torch.load(checkpoint, map_location='cpu')
    state = loaded['net'] if 'net' in loaded else loaded
    state = {k: v for k, v in state.items() if not k.startswith('text_encoder.clip.')}
    if not model.memory_enabled:
        state = {k: v for k, v in state.items() if not k.startswith('memory_fusion.')}
    incompatible = model.load_state_dict(state, strict=False)
    missing_base = [k for k in incompatible.missing_keys if not k.startswith(('memory_fusion.', 'text_encoder.clip.'))]
    missing_memory = [k for k in incompatible.missing_keys if k.startswith('memory_fusion.')]
    if missing_base or incompatible.unexpected_keys:
        raise ValueError('Incompatible checkpoint: '+str((missing_base, incompatible.unexpected_keys)))
    if missing_memory and not (training or allow_untrained):
        raise ValueError('Missing trained memory weights; --allow-untrained-memory is engineering only')
    model.to(device)
    model.freeze_memory_backbone = True
    model.freeze_memory_head = False
    model.freeze_head_bn = True
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(training and name.startswith(('head.', 'memory_fusion.')))
    model.train(training)
    audit = dict(missing_memory=missing_memory, initialized_memory=bool(missing_memory),
                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 trainable_groups=[name for name, p in model.named_parameters() if p.requires_grad],
                 clip_dtype=str(next(model.text_encoder.clip.parameters()).dtype))
    return model, loaded, audit


def source_hashes():
    paths = list((ROOT/'lib/models/mvlm').rglob('*.py')) + [
        ROOT/'lib/config/mvlm/config.py', ROOT/'lib/train/actors/reliable_memory.py',
        ROOT/'lib/train/data/reliable_memory_dataset.py', ROOT/'lib/test/tracker/utils.py',
        ROOT/'lib/utils/focal_loss.py', ROOT/'lib/utils/heapmap_utils.py', ROOT/'lib/utils/box_ops.py',
        ROOT/'tracking/train_reliable_memory.py', ROOT/'tracking/reliable_memory_runtime.py',
        ROOT/'tracking/reliable_memory_wandb.py']
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def snapshot_sources(destination, hashes):
    for relative, expected in hashes.items():
        if sha256(ROOT/relative) != expected:
            raise ValueError('Source changed while snapshotting: '+relative)
        path = Path(destination)/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, path)


def configure_group(cfg, group):
    if group not in ('M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'):
        raise ValueError('Unknown experiment group')
    cfg.MODEL.MEMORY.ENABLED = group not in ('M0', 'M1')
    cfg.MODEL.MEMORY.RELIABLE_WRITE = group in ('M3', 'M6')
    cfg.MODEL.MEMORY.DISTRACTOR_RETENTION = group in ('M4', 'M6')
    cfg.MODEL.MEMORY.BOUNDED_FUSION = group in ('M5', 'M6')
    return cfg


def parameter_digest(model, frozen):
    result = hashlib.sha256()
    for name, value in model.named_parameters():
        if (not value.requires_grad) == frozen:
            result.update(name.encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def bn_digest(model):
    result = hashlib.sha256()
    for name, value in model.head.named_buffers():
        result.update(name.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def revision():
    return subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
