import argparse
import copy
import gc
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config.mvlm.config import cfg, update_config_from_file
from lib.models.mvlm.head import CenterPredictor
from lib.models.mvlm.memory import TargetMemoryBank, extract_target_tokens, crop_boxes_to_image
from lib.models.mvlm.memory_fusion import MemoryFusion
from lib.models.mvlm.mvlm import MVLM, build_mvlm
from lib.train.actors.mvlm import MVLM_Actor
from lib.train.data.sampler import TrackingSampler
from lib.train.data.processing import SeqTrackProcessing
from lib.train.data.loader import ltr_collate_stack1
from lib.train.data import transforms as tfm
from lib.utils import TensorDict
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_channels = 32
        self.body = nn.Conv2d(3, 32, 16, stride=16)
        self.body.num_patches_search = 16
        self.body.num_patches_template = 1

    def forward(self, template_list, search_list, text_emb):
        return {'feat_x': self.body(search_list[0]).flatten(2).transpose(1, 2)}


def check_bank_and_roi():
    features = torch.randn(2, 196, 32)
    boxes = torch.tensor([[0.2, 0.2, 0.4, 0.4], [0.3, 0.3, 0.3, 0.3]])
    tokens, valid = extract_target_tokens(features, boxes)
    require(tokens.shape == (2, 4, 32) and bool(valid.all()), 'ROI extraction failed')
    constant = torch.ones(2, 196, 32)
    require(torch.allclose(extract_target_tokens(constant, boxes)[0], torch.ones(2, 4, 32)),
            'ROI coordinates are incorrect')
    invalid_boxes = torch.tensor([[float('nan'), 0, 0.1, 0.1], [0, 0, -1, 0.1]])
    require(not bool(extract_target_tokens(features, invalid_boxes)[1].any()), 'Invalid ROI accepted')
    bank = TargetMemoryBank(2, short_capacity=2, long_capacity=2)
    bank.write(tokens, boxes, [1, 1], [1.0, 0.1])
    require(len(bank.entries(0)) == 1 and not bank.entries(1), 'Quality gate or batch isolation failed')
    require(not bank.entries(0)[0].tokens.requires_grad, 'Memory was not detached')
    for frame_id in range(2, 8):
        bank.write(torch.randn_like(tokens), boxes, [frame_id, frame_id], [1.0, 0.1])
    require(len(bank.short[0]) <= 2 and len(bank.long[0]) <= 2, 'Capacity limit failed')
    try:
        bank.pack(features, [7, 7], 4)
    except ValueError:
        pass
    else:
        raise AssertionError('Current-frame leakage was not rejected')
    fusion = MemoryFusion(32, 16, 8)
    text = torch.randn(2, 3, 32)
    result, gate = fusion(features, text, bank, [8, 8])
    require(torch.equal(result, features), 'Zero residual changed the baseline')
    nn.init.normal_(fusion.output_proj.weight, std=0.05)
    result, gate = fusion(features, text, bank, [8, 8])
    require(torch.equal(result[1], features[1]) and not bool(gate[1].any()), 'Empty batch member changed')
    require(not torch.equal(result[0], features[0]), 'Populated memory was not used')
    bank.reset()
    result, gate = fusion(features, text, bank, [9, 9])
    require(torch.equal(result, features) and bool(torch.isfinite(result).all()), 'Empty-bank bypass failed')
    print('PASS: ROI validity, quality gate, capacity, causal reads, batch isolation and empty bypass')


def check_sampling_and_processing():
    sampler = object.__new__(TrackingSampler)
    sampler.num_search_frames = 4
    sampler.num_template_frames = 1
    sampler.memory_max_stride = 4
    visible = torch.ones(40, dtype=torch.bool)
    template_ids, search_ids = sampler.get_frame_ids_memory(visible)
    require(all(later > earlier for earlier, later in zip(search_ids, search_ids[1:])),
            'Memory sampling is not chronological')
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    image[:, 25:55] = 255
    box = torch.tensor([25., 20., 30., 30.])
    transform = tfm.Transform(tfm.ToTensorAndJitter(0), tfm.Normalize([0, 0, 0], [1, 1, 1]))
    processing = SeqTrackProcessing(
        {'template': 2, 'search': 4}, {'template': 32, 'search': 64},
        {'template': 0, 'search': 0}, {'template': 0, 'search': 0},
        mode='sequence', transform=transform, joint_sequence=True,
        joint_transform=tfm.Transform(tfm.RandomHorizontalFlip(probability=1.0)))
    data = TensorDict(template_images=[image.copy()], template_anno=[box.clone()],
                      template_masks=[torch.zeros(96, 128)], search_images=[image.copy() for _ in search_ids],
                      search_anno=[box.clone() for _ in search_ids],
                      search_masks=[torch.zeros(96, 128) for _ in search_ids],
                      search_frame_ids=torch.tensor(search_ids), search_visible=visible[search_ids])
    result = processing(data)
    require(result['valid'] and result['search_images'].shape == (4, 3, 64, 64), 'Clip processing failed')
    require(result['search_crop_regions'].shape == (4, 4), 'Crop metadata was lost')
    require(torch.equal(result['search_anno'][0], result['search_anno'][-1]), 'Clip geometry is inconsistent')
    recovered = crop_boxes_to_image(result['search_anno'], result['search_box_transforms'], result['search_source_sizes'])
    expected = torch.tensor([[(128 - 25 - 30 - 1) / 128, 20 / 96, 30 / 128, 30 / 96]]).expand(4, -1)
    require(torch.allclose(recovered, expected, atol=1e-5), 'Crop-to-image box transform failed')
    for workers in (0, 1):
        loader = torch.utils.data.DataLoader([result, result], batch_size=2, num_workers=workers,
                                             collate_fn=ltr_collate_stack1)
        batch = next(iter(loader))
        require(batch['search_images'].shape == (4, 2, 3, 64, 64), 'Clip batch collation failed')
    print('PASS: chronological clips, joint augmentation, coordinate transforms and zero/one-worker collation')


def make_actor(model, config):
    return MVLM_Actor(model, {'giou': giou_loss, 'l1': torch.nn.functional.l1_loss,
                             'focal': FocalLoss()},
                      {'giou': 2., 'l1': 5., 'focal': 1.}, SimpleNamespace(batchsize=1), config)


def check_model_and_gradients():
    config = copy.deepcopy(cfg)
    config.MODEL.MEMORY.ENABLED = True
    config.MODEL.MEMORY.HIDDEN_DIM = 16
    config.MODEL.MEMORY.KEY_DIM = 8
    config.DATA.SEARCH.SIZE = 64
    config.DATA.SEARCH.NUMBER = 3
    config.DATA.MULTI_MODAL_LANGUAGE = False
    config.TRAIN.CM_LOSS = False
    config.TRAIN.MEMORY_PREDICTED_PROB = 0.0
    model = MVLM(None, TinyEncoder(), CenterPredictor(inplanes=32, channel=32, feat_sz=4, stride=16),
                 memory_cfg=config.MODEL.MEMORY)
    model.freeze_memory_backbone = True
    model.freeze_memory_head = True
    for module in (model.encoder, model.head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    model.train()
    require(not model.encoder.training and not model.head.training, 'Frozen modules re-entered train mode')
    images = torch.randn(3, 1, 3, 64, 64)
    with torch.no_grad():
        enabled = model(search_list=[images[0]])
        model.memory_enabled = False
        disabled = model(search_list=[images[0]])
        model.memory_enabled = True
    for name in ('pred_boxes', 'score_map', 'size_map', 'offset_map', 'feat_x'):
        require(torch.equal(enabled[name], disabled[name]), 'Disabled baseline mismatch: ' + name)
    data = TensorDict(search_images=images, template_images=images[:1],
                      search_anno=torch.tensor([[[0.2, 0.2, 0.4, 0.4]]] * 3),
                      search_frame_ids=torch.tensor([[1], [3], [5]]),
                      search_box_transforms=torch.tensor([[[0., 0., 64., 64.]]] * 3),
                      search_source_sizes=torch.tensor([[[64., 64.]]] * 3),
                      search_visible=torch.ones(3, 1, dtype=torch.bool))
    actor = make_actor(model, config)
    optimizer = torch.optim.AdamW(model.memory_fusion.parameters(), lr=0.01)
    frozen_weights = {name: value.clone() for name, value in model.head.state_dict().items()}
    for step in range(2):
        optimizer.zero_grad()
        loss, status = actor(data)
        require(bool(torch.isfinite(loss)), 'Nonfinite sequence loss')
        loss.backward()
        require(model.memory_fusion.output_proj.weight.grad.abs().sum() > 0, 'Tracking gradient did not reach fusion')
        if step:
            require(model.memory_fusion.appearance_key.weight.grad.abs().sum() > 0, 'Retrieval has no gradient')
        optimizer.step()
    require(all(torch.equal(value, model.head.state_dict()[name]) for name, value in frozen_weights.items()),
            'Frozen head weights or running statistics changed')
    require(all(parameter.grad is None for parameter in model.encoder.parameters()), 'Frozen encoder received gradients')
    require(not any('short' in name or 'long' in name for name in model.state_dict()),
            'Runtime video memory leaked into checkpoint weights')
    print('PASS: baseline equality, sequential Actor loss, retrieval gradients and frozen head/encoder')


def check_config_isolation():
    from lib.test.parameter.mvlm import parameters
    memory_params = parameters('mvlm_TF_memory', str(ROOT / 'models/MVLM_TF.pth.tar'))
    baseline_params = parameters('mvlm_TF', str(ROOT / 'models/MVLM_TF.pth.tar'))
    require(memory_params.cfg.MODEL.MEMORY.ENABLED and not baseline_params.cfg.MODEL.MEMORY.ENABLED,
            'Memory configuration leaked into the baseline tracker')
    print('PASS: tracker configurations are independent and baseline memory defaults to disabled')


def check_full_model(device):
    update_config_from_file(str(ROOT / 'experiments/mvlm/mvlm_TF_memory.yaml'))
    config = copy.deepcopy(cfg)
    config.TRAIN.MEMORY_PREDICTED_PROB = 0.0
    model = build_mvlm(config, training=True).to(device)
    model.train()
    images = torch.randn(3, 1, 6, 224, 224, device=device)
    data = TensorDict(search_images=images, template_images=images[:1, :, :, :112, :112],
                      search_anno=torch.tensor([[[0.25, 0.25, 0.3, 0.3]]] * 3, device=device),
                      search_frame_ids=torch.tensor([[1], [2], [3]], device=device),
                      search_box_transforms=torch.tensor([[[0., 0., 224., 224.]]] * 3, device=device),
                      search_source_sizes=torch.tensor([[[224., 224.]]] * 3, device=device),
                      search_visible=torch.ones(3, 1, dtype=torch.bool, device=device),
                      nlp_real=['a person'])
    actor = make_actor(model, config)
    loss, status = actor(data)
    loss.backward()
    require(torch.isfinite(loss) and model.memory_fusion.output_proj.weight.grad.abs().sum() > 0,
            'Full model backward failed')
    require(all(parameter.grad is None for parameter in model.encoder.parameters()), 'Full encoder was not frozen')
    print('PASS: official-checkpoint full-model sequence forward/backward:', status)
    del actor, model, data, images, loss
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    from lib.test.parameter.mvlm import parameters
    from lib.test.tracker.mvlm import MVLM as OnlineTracker
    from lib.test.evaluation import get_dataset
    import cv2
    params = parameters('mvlm_TF_memory', str(ROOT / 'models/MVLM_TF.pth.tar'))
    params.cfg = copy.deepcopy(params.cfg)
    params.device = device
    params.debug = False
    try:
        OnlineTracker(params, 'otb99_lang')
    except ValueError as error:
        require('trained memory module' in str(error), 'Unexpected checkpoint rejection: ' + str(error))
    else:
        raise AssertionError('Official checkpoint silently accepted as trained memory weights')
    gc.collect()
    params.cfg.TEST.MEMORY_ALLOW_UNTRAINED = True
    tracker = OnlineTracker(params, 'otb99_lang')
    sequence = get_dataset('otb99_lang')[0]
    frames = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in sequence.frames[:4]]
    initialization = {'init_bbox': sequence.ground_truth_rect[0].tolist(), 'init_nlp': sequence.language_query}
    tracker.network.memory_enabled = False
    tracker.initialize(frames[0], initialization)
    baseline = [tracker.track(frame)['target_bbox'] for frame in frames[1:]]
    tracker.network.memory_enabled = True
    tracker.initialize(frames[0], initialization)
    enhanced = [tracker.track(frame)['target_bbox'] for frame in frames[1:]]
    require(np.array_equal(baseline, enhanced), 'Zero-initialized memory changed real tracking predictions')
    tracker.initialize(frames[0], initialization)
    tracker.content_memory.min_quality = 0.0
    tracker.tau = -1.0
    tracker.track(frames[1])
    require(len(tracker.content_memory.short[0]) == 1, 'Global-view memory write failed')
    tracker.track(frames[2])
    require(len(tracker.content_memory.short[0]) == 2, 'Local-view memory write failed')
    tracker.flag = True
    tracker.track(frames[3])
    require(len(tracker.content_memory.short[0]) == 3, 'Return to global view broke memory')
    tracker.set_language('a different target')
    require(not tracker.content_memory.entries(0) and tracker.kmem is None and tracker.flag,
            'Target switch failed to reset memory or global search')
    print('PASS: untrained checkpoint guard, real-frame equality, global/local writes and target reset')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-model', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    check_bank_and_roi()
    check_sampling_and_processing()
    check_model_and_gradients()
    check_config_isolation()
    if args.full_model:
        check_full_model(args.device)


if __name__ == '__main__':
    main()
