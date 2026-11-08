"""Strict architecture and baseline boundaries for memory/world checkpoints."""
import torch

PREFIXES = ('memory_reader.', 'world_model.', 'future_fusion.', 'quality_head.')
INPUT_WEIGHTS = ('head.conv1_ctr.0.weight', 'head.conv1_offset.0.weight', 'head.conv1_size.0.weight')


def joint_spec(model):
    c = model.cfg
    return dict(version=1, head_only=bool(c.TRAIN.WORLD.HEAD_ONLY), memory=bool(model.memory_enabled), world=bool(model.world_enabled),
        channels=model.encoder.num_channels, history=c.MODEL.WORLD.HISTORY,
        memory_config={k: c.MODEL.MEMORY[k] for k in ('HIDDEN_DIM', 'KEY_DIM', 'ROI_SIZE', 'USE_SEMANTIC')},
        world_config={k: c.MODEL.WORLD[k] for k in ('HISTORY', 'HIDDEN', 'LAYERS', 'HEADS', 'QUALITY_ENABLED')},
        fusion_config=dict(c.MODEL.FUSION), search_size=c.DATA.SEARCH.SIZE,
        state_format='full_normalized_cxcywh;bank_xywh;crop_xywh:v1')


def load_baseline(model, checkpoint):
    state = dict(checkpoint.get('net', checkpoint))
    if checkpoint.get('joint_spec') or any(k.startswith(PREFIXES) for k in state):
        raise ValueError('Baseline initialization requires a plain MVLM checkpoint')
    expected = model.state_dict()
    if model.cfg.MODEL.FUSION.MODE == 'concat' and model.temporal_enabled:
        for key in INPUT_WEIGHTS:
            if key not in state:
                raise ValueError('Baseline missing CENTER input: ' + key)
            value, target = state[key], expected[key]
            if value.shape[1] != model.encoder.num_channels or value.shape[0] != target.shape[0] or value.shape[2:] != target.shape[2:]:
                raise ValueError('Baseline CENTER shape mismatch: ' + key)
            expanded = torch.zeros_like(target)
            expanded[:, :value.shape[1]] = value
            state[key] = expanded
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith(PREFIXES + ('text_encoder.clip.',))]
    if missing or result.unexpected_keys:
        raise ValueError('Incompatible baseline: missing={}, unexpected={}'.format(missing, result.unexpected_keys))


def load_joint(model, checkpoint, inference=False):
    if checkpoint.get('joint_spec') != joint_spec(model):
        raise ValueError('Joint checkpoint architecture/configuration differs')
    if inference and checkpoint.get('world_stage') not in ('B', 'C'):
        raise ValueError('Online tracking requires an adapted stage B/C checkpoint')
    result = model.load_state_dict(checkpoint['net'], strict=False)
    missing = [k for k in result.missing_keys if not k.startswith('text_encoder.clip.')]
    if missing or result.unexpected_keys:
        raise ValueError('Incomplete joint weights: missing={}, unexpected={}'.format(missing, result.unexpected_keys))
