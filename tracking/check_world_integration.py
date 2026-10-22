"""Optional real-checkpoint smoke check; no training run or saved model artifacts."""
import argparse
from pathlib import Path
import json
from unittest.mock import patch

import _init_paths
import cv2
import torch

from lib.config.mvlm.config import load_config
from lib.models.mvlm import build_mvlm
from lib.models.mvlm.world_checkpoint import load_tracker_weights
from lib.models.mvlm.world_state import WorldHistory, extract_roi
from lib.test.tracker.utils import Preprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261022)
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / 'experiments/mvlm/mvlm_TF_world_stage_a.yaml')
    model = build_mvlm(cfg, training=False).to(args.device).eval()
    load_tracker_weights(model, torch.load(args.baseline, map_location='cpu'))
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError('Cannot read image')
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (224, 224))
    search = Preprocessor(torch.device(args.device)).process(image)
    search = torch.cat((search, search), 1)
    with torch.no_grad():
        language = model.forward_textencoder(['the target in the image'])
        original = model(template_list=[], search_list=[search], text_emb=language)
    roi, valid = extract_roi(original['feat_x'], original['feat_x'].new_tensor([[.25, .25, .5, .5]]))
    history = WorldHistory(4)
    for frame in range(4):
        history.append(roi, roi.new_tensor([[.5, .5, .5, .5]]), roi.new_ones(1), valid,
                       torch.tensor([frame], device=args.device))
    packed = history.pack()
    future_frame = torch.tensor([4], device=args.device)
    with torch.no_grad():
        prediction = model.world_model(packed, language, future_frame)
        fused = model(template_list=[], search_list=[search], text_emb=language, world_prediction=prediction,
                      crop_regions=roi.new_tensor([[0., 0., 1., 1.]]))
    assert torch.equal(original['score_map'], fused['score_map']), 'Zero residual must reproduce baseline'
    model.configure_world_stage('A')
    model.train()
    prediction = model.world_model(packed, language, future_frame)
    loss = (prediction['tokens'] - roi - .01).square().mean() + prediction['delta'].square().mean()
    loss.backward()
    assert model.world_model.appearance_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.encoder.parameters())
    model.zero_grad(set_to_none=True)
    model.configure_world_stage('B')
    model.train()
    with torch.no_grad():
        prediction = model.world_model(packed, language, future_frame)
    fused = model.imagination_fusion(original['feat_x'], prediction, roi.new_tensor([[0., 0., 1., 1.]]))
    output = model.forward_head(fused)
    loss = output['score_map'].mean() + output['pred_boxes'].mean()
    loss.backward()
    assert model.imagination_fusion.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.world_model.parameters())
    assert all(p.grad is None for p in model.encoder.parameters())
    assert torch.isfinite(loss)
    model.zero_grad(set_to_none=True)
    # Exercise the actual actor and candidate decoder, not only isolated modules.
    from lib.train.actors.world_model import WorldModelActor
    cfg.TEST.MVLM.TAU = -1.  # Diagnostic override to exercise warm-up and fusion reliably.
    cfg.TRAIN.WORLD.PREDICTED_HISTORY_PROB = 0.
    clip = dict(images=[image.copy() for _ in range(5)],
                boxes=torch.tensor([[56., 56., 112., 112.]]).repeat(5, 1),
                valid=torch.ones(5, dtype=torch.bool), visible=torch.ones(5, dtype=torch.bool),
                frame_ids=torch.arange(5), language='the target in the image')
    for stage in ('A', 'B'):
        model.configure_world_stage(stage)
        model.train()
        rollout_loss, metrics = WorldModelActor(model, cfg, stage)([clip])
        assert rollout_loss is not None and torch.isfinite(rollout_loss), (stage, metrics)
        rollout_loss.backward()
        assert all(p.grad is None for p in model.encoder.parameters())
        if stage == 'B':
            assert all(p.grad is None for p in model.world_model.parameters())
        model.zero_grad(set_to_none=True)
    # Reuse this in-memory model to exercise tracker initialization, full/local
    # crops, history warm-up and language reset without writing a checkpoint.
    from lib.test.tracker.mvlm import MVLM as Tracker
    from lib.test.utils import TrackerParams
    from lib.models.mvlm.world_checkpoint import world_spec
    params = TrackerParams()
    params.cfg, params.checkpoint, params.debug, params.device = cfg, 'in-memory-smoke', 0, args.device
    params.template_factor, params.template_size = cfg.TEST.TEMPLATE_FACTOR, cfg.TEST.TEMPLATE_SIZE
    params.search_factor, params.search_size = cfg.TEST.SEARCH_FACTOR, cfg.TEST.SEARCH_SIZE
    checkpoint = dict(net=model.state_dict(), world_stage='B', world_spec=world_spec(model))
    with patch('lib.test.tracker.mvlm.build_mvlm', return_value=model), patch('torch.load', return_value=checkpoint):
        tracker = Tracker(params, 'lasot')
    tracker.initialize(image, dict(init_bbox=[56., 56., 112., 112.], init_nlp='the target in the image'))
    assert tracker.world_history.pack() is None
    for _ in range(5):
        result = tracker.track(image)
        assert all(torch.isfinite(torch.tensor(result['target_bbox'])))
    assert tracker.world_history.pack() is not None
    tracker.set_language('another target')
    assert tracker.world_history.pack() is None and tracker.flag
    print(json.dumps(dict(checkpoint_loaded=True, search_shape=list(original['feat_x'].shape),
                          zero_residual_baseline_equal=True, stage_a_gradients=True,
                          stage_b_gradients=True, frozen_gradients_absent=True,
                          actual_actor_rollouts=True, actual_tracker_rollout_and_reset=True,
                          world_parameters=sum(p.numel() for p in model.world_model.parameters()),
                          fusion_parameters=sum(p.numel() for p in model.imagination_fusion.parameters()))))


if __name__ == '__main__':
    main()
