"""Real-weight GPU adapter/parity and causal state reload engineering check."""
import argparse
import gc
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from lib.config.mvlm.config import load_config
from lib.models.mvlm.tracking_core import TrackingCore
from lib.train.data.reliable_memory_dataset import read_manifest
from reliable_memory_runtime import build_loaded_model, seed_all, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    seed_all(20261116)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    cfg = load_config('mvlm_TF_reliable_memory')
    model, _, _ = build_loaded_model(cfg, args.checkpoint, device)
    partitions, _ = read_manifest(args.manifest, 6, 1)
    seq = partitions['val'][0]
    images = []
    for path in seq['frames'][:6]:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(path)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    core = TrackingCore(model, cfg, device)
    core.reset(seq['language'], seq['name'])
    with torch.no_grad():
        reference = [core.step(image)[1] for image in images[:4]]
        torch.save(core.state_dict(), destination/'video_state.pth')
        continuation = [core.step(image)[1] for image in images[4:]]
        core.load_state_dict(torch.load(destination/'video_state.pth', map_location=device))
        reloaded = [core.step(image)[1] for image in images[4:]]
    if continuation != reloaded:
        raise AssertionError('Real-weight video state reload changed predictions')
    expected = [row['target_bbox'] for row in reference + continuation]
    del core, model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    from lib.test.tracker.mvlm import MVLM as OnlineTracker
    params = SimpleNamespace(cfg=cfg, checkpoint=args.checkpoint, debug=False,
                             device=str(device), template_factor=cfg.TEST.TEMPLATE_FACTOR,
                             template_size=cfg.TEST.TEMPLATE_SIZE, search_factor=cfg.TEST.SEARCH_FACTOR,
                             search_size=cfg.TEST.SEARCH_SIZE)
    tracker = OnlineTracker(params, 'lasot')
    outputs = []
    for fake_gt in ([0, 0, 1, 1], [10000, 10000, 500, 500]):
        first = tracker.initialize(images[0], dict(init_nlp=seq['language'], init_bbox=fake_gt,
                                                   seq_name=seq['name']))
        outputs.append([first['target_bbox']] + [tracker.track(image, {'gt_bbox': fake_gt})['target_bbox']
                                                 for image in images[1:]])
    if outputs[0] != expected or outputs[1] != expected:
        raise AssertionError('Tracker adapter differs from core or reads supplied GT')
    report = dict(status='passed', frames=6, sequence=seq['dataset']+'/'+seq['name'],
                  adapter_matches_core=True, altered_gt_does_not_change_predictions=True,
                  video_state_reload_matches=True, engineering_only=True,
                  write_actions=[row['diagnostics']['write']['action'] for row in reference+continuation])
    atomic_json(destination/'report.json', report)
    print(report)


if __name__ == '__main__':
    main()
