"""Explicit baseline-to-concat initialization; trained architectures stay strict."""

CENTER_INPUT_WEIGHTS = (
    'head.conv1_ctr.0.weight',
    'head.conv1_offset.0.weight',
    'head.conv1_size.0.weight',
)


def prepare_memory_checkpoint(model, state_dict, allow_baseline=False):
    """Zero-pad only the three baseline CENTER inputs for concat initialization.

    This never converts a trained residual-memory checkpoint or resizes a
    trained concat checkpoint. Optimizer state is not migrated between modes.
    """
    state_dict = dict(state_dict)
    if not (model.memory_enabled and model.memory_fusion.fusion_mode == 'concat'):
        return state_dict
    if any(name.startswith('memory_fusion.') for name in state_dict):
        if any(name.startswith('memory_fusion.output_proj.') for name in state_dict):
            raise ValueError('A residual-memory checkpoint cannot initialize a concat model; '
                             'use baseline MVLM weights or a trained concat checkpoint')
        return state_dict
    if not allow_baseline:
        raise ValueError('Concat evaluation requires a trained memory module and expanded CENTER head; '
                         'TEST.MEMORY_ALLOW_UNTRAINED is only for an explicit smoke check')
    expected = model.state_dict()
    visual_channels = model.encoder.num_channels
    for name in CENTER_INPUT_WEIGHTS:
        if name not in state_dict:
            continue # The caller reports missing baseline weights.
        weight, target = state_dict[name], expected[name]
        baseline_shape = (target.shape[0], visual_channels, *target.shape[2:])
        if tuple(weight.shape) != baseline_shape:
            raise ValueError('Expected baseline CENTER input {} with shape {}, got {}'.format(
                name, baseline_shape, tuple(weight.shape)))
        expanded = weight.new_zeros(target.shape)
        expanded[:, :visual_channels] = weight
        state_dict[name] = expanded
    return state_dict
