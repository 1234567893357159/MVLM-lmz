"""Evaluate final image-space boxes on the fixed validation clips of a B run.

All visible and valid annotated frames after warmup count, including frames
whose target is outside the tracker's search crop. The annotations are read
only after localization and do not affect the model's online state.
"""
import argparse
import json
from pathlib import Path

import _init_paths
import torch

from lib.config.mvlm.config import load_config
from lib.models.mvlm import build_mvlm
from lib.models.mvlm.joint_checkpoint import load_joint, joint_spec
from lib.train.dataset.world_clips import WorldClips
import memory_world_tracker_b as tracker_b


def image_iou(predicted, target):
    """IoU of two full-image pixel boxes in xywh format."""
    x, y, w, h = map(float, predicted)
    a, b, c, d = map(float, target)
    intersection = max(0., min(x + w, a + c) - max(x, a)) * max(0., min(y + h, b + d) - max(y, b))
    return intersection / max(1e-8, w * h + c * d - intersection)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-output', required=True, help='Existing full-tracker B run directory')
    parser.add_argument('--checkpoints', nargs='+', required=True, help='Compatible A or B checkpoints')
    parser.add_argument('--output', required=True, help='New JSON report path')
    parser.add_argument('--max-clips', type=int, help='Evaluate a prefix only, for smoke tests')
    parser.add_argument('--reset-head-bn-from', help='Read-only counterfactual using head BN buffers from this checkpoint')
    parser.add_argument('--disable-temporal', action='store_true', help='Bypass future fusion and quality reranking')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    output, destination = Path(args.run_output).resolve(), Path(args.output).resolve()
    if destination.exists():
        parser.error('Use a new report path; existing results are not overwritten')
    cfg = load_config(output / 'config.yaml')
    run = json.loads((output / 'run.json').read_text())
    initial = json.loads((output / 'validation_000000.json').read_text())
    count = initial['clips'] if args.max_clips is None else args.max_clips
    if not 1 <= count <= initial['clips']:
        parser.error('max-clips must be within the original fixed validation set')
    cache = torch.load(str(output / 'data_cache.pth'), map_location='cpu')
    if cache['manifest_hash'] != run['manifest_sha256'] or cache['baseline_hash'] != run['baseline_sha256']:
        raise ValueError('Cached validation split does not match the training run')
    expected = {key: cache[key] for key in ('data_signature',)}
    expected.update(manifest_sha256=cache['manifest_hash'], baseline_sha256=cache['baseline_hash'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.TRAIN.WORLD.SEED)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(cfg.TRAIN.WORLD.SEED)
    model = build_mvlm(cfg, training=False).to(device).eval()
    aliases = {'LASOT': 'lasot', 'VASTTRACK': 'vasttrack', 'TNL2K_train': 'tnl2k'}
    weights = {aliases[name]: ratio for name, ratio in zip(cfg.DATA.TRAIN.DATASETS_NAME, cfg.DATA.TRAIN.DATASETS_RATIO)}
    validation = WorldClips(cache['partitions']['val'], cfg.TRAIN.WORLD.CLIP_LENGTH, count,
        cfg.TRAIN.WORLD.SEED + 1, augment=False, stride=cfg.TRAIN.WORLD.MAX_STRIDE, dataset_weights=weights)
    actor = tracker_b.StreamingTrackerBActor(model, cfg)
    original = tracker_b.finish_location
    bn_state = None
    if args.reset_head_bn_from:
        source = torch.load(args.reset_head_bn_from, map_location='cpu')
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError('BN source checkpoint data or baseline signature differs')
        if source.get('joint_spec') != joint_spec(model):
            raise ValueError('BN source checkpoint architecture differs')
        bn_state = {key: value for key, value in source['net'].items()
                    if key.startswith('head.') and key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
        expected_bn = {name for name, _ in model.named_buffers() if name.startswith('head.')
                       and name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
        if not bn_state or set(bn_state) != expected_bn:
            raise ValueError('BN source must contain all and only the CENTER running statistics')
        del source
    report = dict(protocol='all checkpoints use B inference; fixed clips; final selected full-image boxes; visible valid frames after warmup; '
                  'includes targets outside the search crop; not a full-sequence benchmark',
                  clips=count, disable_temporal=args.disable_temporal,
                  bn_source=args.reset_head_bn_from, results=[])
    try:
        for filename in args.checkpoints:
            checkpoint = torch.load(filename, map_location='cpu')
            if any(checkpoint.get(key) != value for key, value in expected.items()):
                raise ValueError('Checkpoint data or baseline signature differs: ' + filename)
            load_joint(model, checkpoint)
            step = checkpoint['step']
            del checkpoint
            model.world_stage = 'A' if args.disable_temporal else 'B'
            model.eval()
            if bn_state is not None:
                with torch.no_grad():
                    for name, buffer in model.named_buffers():
                        if name in bn_state:
                            buffer.copy_(bn_state[name])
            rows = []
            with torch.no_grad():
                for index in range(count):
                    clip = validation[index]
                    frame_index, frames = [0], []

                    def capture(*capture_args, **capture_kwargs):
                        result = original(*capture_args, **capture_kwargs)
                        t = frame_index[0]
                        if t >= cfg.TRAIN.WORLD.WARMUP_FRAMES and bool(clip['valid'][t] & clip['visible'][t]):
                            frames.append(dict(frame_id=int(clip['frame_ids'][t]),
                                               iou=image_iou(result['pixels'], clip['boxes'][t])))
                        frame_index[0] += 1
                        return result

                    tracker_b.finish_location = capture
                    values = actor([clip])
                    if frame_index[0] != len(clip['images']):
                        raise RuntimeError('Unexpected localization count')
                    rows.append(dict(clip_index=index, frames=frames,
                                     loss_supervised_frames=values['supervised_frames'],
                                     loss=values['loss'], tracking_loss=values['tracking_loss']))
            tracker_b.finish_location = original
            overlaps = [frame['iou'] for row in rows for frame in row['frames']]
            if not overlaps:
                raise ValueError('No visible valid frames in the selected clips')
            n = sum(row['loss_supervised_frames'] for row in rows)
            result = dict(checkpoint=str(Path(filename).resolve()), checkpoint_step=step,
                visible_frames=len(overlaps), selected_full_image_iou=sum(overlaps) / len(overlaps),
                selected_success_at_0_5=sum(value >= .5 for value in overlaps) / len(overlaps),
                loss_supervised_frames=n,
                crop_supervised_loss=(sum(row['loss'] * row['loss_supervised_frames'] for row in rows) / n if n else None),
                rows=rows)
            report['results'].append(result)
            print(json.dumps({key: value for key, value in result.items() if key != 'rows'}), flush=True)
    finally:
        tracker_b.finish_location = original
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(destination)


if __name__ == '__main__':
    main()
