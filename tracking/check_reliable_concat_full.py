"""Check the real checkpoint and concat forward on CPU, without an optimizer."""
import argparse
import copy
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['MVLM_DEVICE'] = 'cpu'
import torch
from lib.config.mvlm.config import load_config
from lib.models.mvlm.head import build_head
from lib.models.mvlm.memory import TargetMemoryBank, extract_target_tokens
from reliable_memory_runtime import build_loaded_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = load_config('mvlm_TF_reliable_memory')
    model, loaded, audit = build_loaded_model(cfg, args.checkpoint, torch.device('cpu'), training=True)
    assert len(audit['expanded_center_weights']) == 3
    state = loaded.get('net', loaded)
    baseline_cfg = copy.deepcopy(cfg)
    baseline_cfg.MODEL.MEMORY.ENABLED = False
    head = build_head(baseline_cfg, model.encoder).eval()
    head.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith('head.')}, strict=True)
    model.eval()
    bank = TargetMemoryBank(1)
    with torch.no_grad():
        text = model.forward_textencoder(['a person'])
        template, search = torch.zeros(1, 6, 112, 112), torch.randn(1, 6, 224, 224)
        run = lambda frame: model(template_list=[template], search_list=[search], text_emb=text,
                                  memory_state=bank, frame_ids=torch.tensor([frame]))
        output = run(0)
        assert output['feat_x'].shape == (1, 196, 512)
        assert output['feat_x_mem'].shape == (1, 196, 768)
        assert torch.count_nonzero(output['feat_x_mem'][..., 512:]) == 0
        expected = head(output['feat_x'].transpose(1, 2).reshape(1, 512, 14, 14))
        for key, value in zip(('score_map', 'pred_boxes', 'size_map', 'offset_map'), expected):
            torch.testing.assert_close(output[key], value.view(1, 1, 4) if key == 'pred_boxes' else value,
                                       rtol=1e-5, atol=1e-6)
        box = torch.tensor([[.25, .25, .3, .3]])
        tokens, valid = extract_target_tokens(output['feat_x'], box)
        bank.write(tokens, box, [0], [1.], valid)
        assert torch.count_nonzero(run(1)['feat_x_mem'][..., 512:]) > 0
        for name in audit['expanded_center_weights']:
            weight = model.state_dict()[name]
            assert torch.equal(weight[:, :512], state[name]) and torch.count_nonzero(weight[:, 512:]) == 0
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.startswith(('head.', 'memory_fusion.')), name
    print('PASS: official checkpoint, 512+256 concat, empty/populated bank and original-head parity')
    print('PASS: encoder/text frozen, head/reader trainable; no optimizer update')
    print('Trainable parameters:', audit['trainable_parameters'])


if __name__ == '__main__':
    main()
