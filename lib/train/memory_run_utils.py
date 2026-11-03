"""Step-based optimization and atomic records for the two-GPU memory experiment."""
import hashlib
import json
from pathlib import Path

import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def lr_factor(step):
    """LR for the next optimizer update numbered step (one-based)."""
    if step <= 200:
        return .1 + .9 * step / 200
    return 1. if step <= 3000 else .1


def memory_optimizer(model, config):
    memory, head, unexpected = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('memory_fusion.'):
            memory.append(parameter)
        elif name.startswith('head.'):
            head.append(parameter)
        else:
            unexpected.append(name)
    if not memory or not head or unexpected:
        raise ValueError('Expected only memory and CENTER trainable: ' + str(unexpected))
    return torch.optim.AdamW([
        dict(params=memory, lr=config.TRAIN.LR, peak_lr=config.TRAIN.LR, name='memory'),
        dict(params=head, lr=config.TRAIN.MEMORY_HEAD_LR,
             peak_lr=config.TRAIN.MEMORY_HEAD_LR, name='center'),
    ], weight_decay=config.TRAIN.WEIGHT_DECAY)


def set_step_lr(optimizer, step):
    for group in optimizer.param_groups:
        group['lr'] = group['peak_lr'] * lr_factor(step)


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(('encoder.', 'text_encoder.')):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def bn_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.head.state_dict().items()
            if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


def assert_bn_unchanged(model, expected):
    actual = bn_state(model)
    if not all(torch.equal(value, actual[name]) for name, value in expected.items()):
        raise RuntimeError('Frozen CENTER BatchNorm statistics changed')
