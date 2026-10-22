"""Build an explicit validation split exclusively from official training videos."""
import argparse
import json
from pathlib import Path
import random

import _init_paths
from lib.train.dataset.world_clips import describe_sequence, official_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lasot-root')
    parser.add_argument('--tnl2k-root')
    parser.add_argument('--vasttrack-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--val-fraction', type=float, default=.1)
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Explicitly permit a partial training corpus; omissions are recorded')
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error('val-fraction must be between zero and one')
    result = dict(protocol='official-train-video-split-v1', seed=args.seed,
                  roots={}, train=[], val=[], omissions=[], partial=args.allow_incomplete)
    for dataset in ('lasot', 'tnl2k', 'vasttrack'):
        root = getattr(args, dataset + '_root')
        if root is None:
            continue
        root = Path(root).resolve()
        if dataset == 'tnl2k':
            # Never scan root/test or accept it as a train directory.
            directory = root / 'train'
            if not directory.is_dir():
                raise ValueError('Missing official training directory: ' + str(directory))
            names = sorted(p.name for p in directory.iterdir() if p.is_dir())
        else:
            names = sorted(official_names(dataset))
        rows = []
        for name in names:
            try:
                seq = describe_sequence(dataset, root, name)
                if len(seq['frames']) < 8:
                    raise ValueError('Fewer than 8 frames')
                rows.append(dict(dataset=dataset, name=name, fingerprint=seq['fingerprint']))
            except (OSError, ValueError) as error:
                if not args.allow_incomplete:
                    raise ValueError('Incomplete corpus; fix data or explicitly use --allow-incomplete: '
                                     + str(error)) from error
                result['omissions'].append(dict(dataset=dataset, name=name, reason=str(error)))
        if len(rows) < 2:
            raise ValueError('Need at least two valid training videos for ' + dataset)
        random.Random(args.seed).shuffle(rows)
        count = max(1, min(len(rows) - 1, round(len(rows) * args.val_fraction)))
        result['val'].extend(rows[:count])
        result['train'].extend(rows[count:])
        result['roots'][dataset] = str(root)
    if not result['roots']:
        parser.error('Provide at least one training dataset root')
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print('Saved {} train / {} validation videos; {} omissions to {}'.format(
        len(result['train']), len(result['val']), len(result['omissions']), destination))


if __name__ == '__main__':
    main()
