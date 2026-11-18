"""Create a deterministic, auditable three-source validation manifest."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def selected_manifest(source, per_source=30, seed=20261116):
    raw = Path(source).read_bytes()
    original = json.loads(raw)
    if original.get('protocol') != 'official-train-video-split-v1' or original.get('partial'):
        raise ValueError('Expected complete official training split manifest')
    result = dict(original)
    sources = ('lasot', 'tnl2k', 'vasttrack')
    chosen, train = [], []
    for dataset in sources:
        candidates = [r for r in original['val'] if r['dataset'] == dataset]
        candidates.sort(key=lambda r: hashlib.sha256(
            ('{}|{}|{}'.format(seed, dataset, r['name'])).encode()).hexdigest())
        if len(candidates) < per_source:
            raise ValueError('Insufficient validation videos for ' + dataset)
        chosen.extend(candidates[:per_source])
        train.append(next(r for r in original['train'] if r['dataset'] == dataset))
    if len({(r['dataset'], r['name']) for r in original['train'] + original['val']}) != len(original['train']) + len(original['val']):
        raise ValueError('Duplicate or overlapping train/validation video IDs')
    result.update(train=train, val=chosen, partial=False,
                  validation_selection=dict(seed=seed, per_source=per_source,
                                            source_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                                            rule='lowest SHA256 of seed|dataset|name per source',
                                            note='train rows are included only for manifest audit, never evaluation'))
    assert Counter(r['dataset'] for r in chosen) == Counter({source: per_source for source in sources})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-source', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20261116)
    args = parser.parse_args()
    if args.per_source < 1:
        parser.error('per-source must be positive')
    result = selected_manifest(args.source, args.per_source, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to change an already frozen validation set.
    with args.out.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print('Prepared {} validation videos: {}'.format(len(result['val']), args.out))


if __name__ == '__main__':
    main()
