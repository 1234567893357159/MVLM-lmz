"""Explicit checkpoint boundaries: baseline, stage-A predictor, stage-B tracker."""
import torch


def load_tracker_weights(model, checkpoint, require_world=False):
    state = checkpoint.get('net', checkpoint)
    if not require_world and (checkpoint.get('world_stage') is not None
                              or any(key.startswith(('world_model.', 'imagination_fusion.')) for key in state)):
        raise ValueError('Baseline initialization must use weights without world-model modules')
    if require_world and checkpoint.get('world_stage') != 'B':
        raise ValueError('Online world tracking requires a stage-B checkpoint')
    result = model.load_state_dict(state, strict=False)
    allowed = ('text_encoder.clip.',)
    if not require_world:
        allowed += ('world_model.', 'imagination_fusion.')
    missing = [key for key in result.missing_keys if not key.startswith(allowed)]
    if missing or result.unexpected_keys:
        raise ValueError('Incompatible tracker weights: missing={}, unexpected={}'.format(
            missing, result.unexpected_keys))
    if require_world and checkpoint.get('world_spec') != world_spec(model):
        raise ValueError('World checkpoint structure/configuration does not match tracker')


def world_spec(model):
    world = model.world_model
    return dict(history=world.history, hidden=world.visual_proj.out_features,
                channels=world.visual_proj.in_features,
                layers=len(world.transformer.layers),
                heads=world.transformer.layers[0].self_attn.num_heads,
                roi=4, state_format='full_normalized_cxcywh_v1')


def load_predictor(model, checkpoint):
    if checkpoint.get('world_stage') != 'A' or checkpoint.get('world_spec') != world_spec(model):
        raise ValueError('Expected a compatible stage-A world checkpoint')
    model.world_model.load_state_dict(checkpoint['world_model'], strict=True)
