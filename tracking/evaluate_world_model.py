"""Full online evaluation on the fixed validation videos, with MVLM metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import _init_paths
import cv2
import numpy as np
import torch

from lib.config.mvlm.config import load_config
from lib.test.tracker.mvlm import MVLM
from lib.test.utils import TrackerParams
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.train.dataset.world_clips import read_manifest
from train_world_model import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = load_config(args.config)
    partitions, manifest_hash = read_manifest(args.manifest, cfg.MODEL.WORLD.HISTORY + 1)
    if cfg.MODEL.WORLD.ENABLED:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if checkpoint.get('manifest_sha256') != manifest_hash:
            raise ValueError('Evaluation split differs from training split')
        signature = hashlib.sha256(json.dumps({
            key: [seq['fingerprint'] for seq in sequences] for key, sequences in partitions.items()
        }, sort_keys=True).encode()).hexdigest()
        if checkpoint.get('data_signature') != signature:
            raise ValueError('Validation data annotations or resolved paths changed since training')
        for section in ('DATA', 'TEST'):
            if checkpoint['config'][section] != cfg[section]:
                raise ValueError('Evaluation settings differ from the trained config: ' + section)
        del checkpoint
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    params = TrackerParams()
    params.cfg, params.checkpoint, params.debug, params.device = cfg, args.checkpoint, 0, args.device
    params.template_factor, params.template_size = cfg.TEST.TEMPLATE_FACTOR, cfg.TEST.TEMPLATE_SIZE
    params.search_factor, params.search_size = cfg.TEST.SEARCH_FACTOR, cfg.TEST.SEARCH_SIZE
    tracker = MVLM(params, 'lasot')
    results = []
    for sequence in partitions['val']:
        image = cv2.imread(sequence['frames'][0])
        if image is None:
            raise ValueError('Cannot decode ' + sequence['frames'][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Original template-free benchmark initialization convention. World history stays empty.
        tracker.initialize(image, dict(init_bbox=sequence['boxes'][0].tolist(), init_nlp=sequence['language']))
        predictions = [sequence['boxes'][0].tolist()]
        start = time.perf_counter()
        for path in sequence['frames'][1:]:
            image = cv2.imread(path)
            if image is None:
                raise ValueError('Cannot decode ' + path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictions.append(tracker.track(image)['target_bbox'])
        elapsed = time.perf_counter() - start
        predicted = torch.tensor(predictions, dtype=torch.float64)
        truth = torch.tensor(sequence['boxes'], dtype=torch.float64)
        visible = torch.from_numpy(sequence['visible']) if sequence['dataset'] == 'lasot' else None
        overlap, center, normalized, _ = calc_seq_err_robust(predicted, truth, sequence['dataset'], visible)
        thresholds = torch.arange(0., 1.05, .05, dtype=torch.float64)
        # Match extract_results / print_results: sequence mean success curve, PRE@20, NPR@0.2.
        metrics = dict(AUC=float((overlap[:, None] > thresholds).double().mean(0).mean() * 100),
                       PRE=float((center <= 20).double().mean() * 100),
                       NPR=float((normalized <= .2).double().mean() * 100))
        name = sequence['dataset'] + '_' + sequence['name']
        np.savetxt(str(destination / (name + '.txt')), np.asarray(predictions), delimiter='\t', fmt='%.6f')
        result = dict(dataset=sequence['dataset'], name=sequence['name'], frames=len(predictions),
                      elapsed_seconds=elapsed, metrics=metrics)
        results.append(result)
        print(json.dumps(result), flush=True)
    datasets = sorted(set(row['dataset'] for row in results))
    means = {dataset: {key: float(np.mean([row['metrics'][key] for row in results if row['dataset'] == dataset]))
                       for key in ('AUC', 'PRE', 'NPR')} for dataset in datasets}
    summary = dict(protocol='full_validation_sequences', manifest_sha256=manifest_hash,
                   checkpoint_sha256=sha256(args.checkpoint), config=cfg,
                   per_dataset=means, macro_AUC=float(np.mean([value['AUC'] for value in means.values()])),
                   sequences=results, timing='wall time including decoding; not model-only FPS')
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
