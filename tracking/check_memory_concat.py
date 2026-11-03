"""CPU checks for direct memory concatenation into the expanded CENTER head."""
import argparse
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_full_model(checkpoint):
    import os
    os.environ['MVLM_DEVICE'] = 'cpu'
    import torch
    from lib.config.mvlm.config import cfg, update_config_from_file
    from lib.models.mvlm import build_mvlm
    from lib.models.mvlm.checkpoint import CENTER_INPUT_WEIGHTS
    from lib.models.mvlm.head import build_head
    from lib.models.mvlm.memory import extract_target_tokens

    torch.set_num_threads(1)
    torch.manual_seed(17)
    update_config_from_file(str(ROOT / 'experiments/mvlm/mvlm_TF_memory_concat.yaml'))
    config = copy.deepcopy(cfg)
    config.MODEL.PRETRAINED_MODEL = str(checkpoint)
    model = build_mvlm(config, training=True).cpu().eval()
    original = torch.load(str(checkpoint), map_location='cpu')
    original = original['net'] if 'net' in original else original
    for name in CENTER_INPUT_WEIGHTS:
        weight = model.state_dict()[name]
        assert tuple(weight.shape) == (256, 768, 3, 3), (name, weight.shape)
        assert torch.equal(weight[:, :512], original[name]), name
        assert torch.count_nonzero(weight[:, 512:]) == 0, name
    baseline_config = copy.deepcopy(config)
    baseline_config.MODEL.MEMORY.ENABLED = False
    baseline_head = build_head(baseline_config, model.encoder).cpu().eval()
    baseline_head.load_state_dict({key[len('head.'):]: value for key, value in original.items()
                                   if key.startswith('head.')}, strict=True)
    extra_head_parameters = (sum(p.numel() for p in model.head.parameters())
                             - sum(p.numel() for p in baseline_head.parameters()))
    del original
    bank = model.new_memory(1)
    boxes = torch.tensor([[.25, .25, .3, .3]])
    errors = {}
    with torch.no_grad():
        text = model.forward_textencoder(['a person'])
        for frame in (1, 2):
            output = model(template_list=[torch.zeros(1, 6, 112, 112)],
                           search_list=[torch.randn(1, 6, 224, 224)], text_emb=text,
                           memory_state=bank, frame_ids=torch.tensor([frame]))
            assert tuple(output['feat_x'].shape) == (1, 196, 512)
            assert tuple(output['feat_x_mem'].shape) == (1, 196, 768)
            if frame == 1:
                assert torch.count_nonzero(output['feat_x_mem'][..., 512:]) == 0
            else:
                assert torch.count_nonzero(output['feat_x_mem'][..., 512:]) > 0
            feature_map = output['feat_x'].transpose(1, 2).reshape(1, 512, 14, 14)
            score, bbox, size, offset = baseline_head(feature_map)
            expected = dict(score_map=score, pred_boxes=bbox.view(1, 1, 4),
                            size_map=size, offset_map=offset)
            for name, value in expected.items():
                errors[f'frame{frame}/{name}'] = float((output[name] - value).abs().max())
                torch.testing.assert_close(output[name], value, rtol=1e-5, atol=1e-6)
            tokens, valid = extract_target_tokens(output['feat_x'], boxes)
            bank.write(tokens, boxes, [frame], [1.], valid)
    assert extra_head_parameters == 1769472, extra_head_parameters
    assert not hasattr(model.memory_fusion, 'output_proj')
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.text_encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    print('PASS: official MVLM initialization; three 512->768 CENTER input migrations')
    print('PASS: actual FastITPN+CLIP forward, empty/nonempty memory, baseline prediction agreement')
    print('PASS: original 512-channel features retained; encoder/text frozen; expanded head trainable')
    print('Extra CENTER parameters:', extra_head_parameters)
    print('Maximum prediction differences:', errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-model', action='store_true', help='Check official weights and full CPU forward')
    parser.add_argument('--checkpoint', type=Path, default=Path('/root/MVLM/models/MVLM_TF.pth.tar'))
    args = parser.parse_args()
    if args.full_model:
        check_full_model(args.checkpoint)
    else:
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_memory_concat.py')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)


if __name__ == '__main__':
    main()
