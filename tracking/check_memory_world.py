"""Real FastITPN/CLIP CPU preflight, without optimizer updates or accuracy claims."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path

import _init_paths
import torch
from lib.config.mvlm.config import load_config
from lib.models.mvlm import build_mvlm
from lib.models.mvlm.head import build_head
from lib.models.mvlm.joint_checkpoint import load_baseline
from lib.models.mvlm.temporal_core import TemporalState, predict_prior
from lib.models.mvlm.world_state import extract_roi


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default=str(root / 'models/MVLM_TF.pth.tar'))
    parser.add_argument('--output', help='Optional JSON evidence path')
    args = parser.parse_args()
    os.environ['MVLM_DEVICE'] = 'cpu'
    torch.set_num_threads(1)
    torch.manual_seed(20261108)
    checkpoint = torch.load(args.baseline, map_location='cpu')
    records = []
    for mode in ('cross_attention', 'concat'):
        cfg = load_config(root / 'experiments/mvlm/mvlm_TF_memory_world_stage_a.yaml')
        cfg.MODEL.FUSION.MODE = mode
        cfg.TRAIN.WORLD.TRAIN_HEAD = mode == 'concat'
        model = build_mvlm(cfg, training=False).eval()
        load_baseline(model, checkpoint)
        reference_cfg = copy.deepcopy(cfg)
        reference_cfg.MODEL.MEMORY.ENABLED = False
        reference_cfg.MODEL.WORLD.ENABLED = False
        reference = build_head(reference_cfg, model.encoder).eval()
        reference.load_state_dict({
            key[len('head.'):]: value for key, value in checkpoint['net'].items()
            if key.startswith('head.')
        })
        image = torch.rand(1, 6, 224, 224)
        with torch.no_grad():
            language = model.forward_textencoder(['a person'])
            raw = model.forward_encoder([], [image], language)['feat_x']
            expected = reference(raw.transpose(1, 2).reshape(1, 512, 14, 14))
            empty = model([], [image], text_emb=language)
            targets = (expected[0], expected[1][:, None], expected[2], expected[3])
            error = max(float((empty[key] - target).abs().max()) for key, target in zip(
                ('score_map', 'pred_boxes', 'size_map', 'offset_map'), targets))
            assert error < 2e-5, error
        state = TemporalState(cfg)
        roi, _ = extract_roi(raw, raw.new_tensor([[.25, .25, .5, .5]]))
        for frame in range(4):
            state.history.append(roi, raw.new_tensor([[.5, .5, .5, .5]]), torch.ones(1),
                                 torch.ones(1, dtype=torch.bool), torch.tensor([frame]))
            state.bank.write(roi, raw.new_tensor([[.25, .25, .5, .5]]), [frame], [1.])
        state.last_visual = raw[:, :16].detach()
        model.configure_world_stage('A')
        prior = predict_prior(model, state, language.detach(), torch.tensor([4]))
        _, extra = model.fuse_features(raw.detach(), language.detach(), prior,
                                       torch.tensor([[0., 0., 1., 1.]]))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            extra['quality_logits'], torch.ones(1, 196))
        loss.backward()
        norms = {
            name: sum(float(parameter.grad.abs().sum()) for parameter in module.parameters()
                      if parameter.grad is not None)
            for name, module in (('reader', model.memory_reader),
                                 ('world', model.world_model), ('quality', model.quality_head))
        }
        assert all(value > 0 for value in norms.values()), norms
        assert all(parameter.grad is None for parameter in model.encoder.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                   if parameter.grad is not None)
        records.append(dict(mode=mode, raw_shape=list(raw.shape),
                            predicted_roi_shape=list(prior['tokens'].shape),
                            quality_shape=list(extra['quality_logits'].shape),
                            baseline_max_abs_error=error, stage_a_gradient_l1=norms,
                            encoder_frozen=True))
        print(json.dumps(records[-1]), flush=True)
        del model, reference, prior, extra, loss, raw, language, state, roi, empty, expected, targets
        gc.collect()
    if args.output:
        Path(args.output).write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
