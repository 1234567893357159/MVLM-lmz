"""Freeze official-training-only adaptation splits without changing source data."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml
from lib.train.dataset import Lasot, VastTrack, Tnl2k
from lib.train.memory_run_utils import write_json

CANONICAL = Path('/root/MVLM/output/paper_reproduction_20261030_run01/split')
DATA = Path('/root/MVLM/data')
MAPPING = dict(LASOT='lasot', VASTTRACK='vasttrack', TNL2K_train='tnl2k')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def group_key(dataset, name):
    if dataset == 'tnl2k':
        name = re.sub(r'(?i)[_-]done$', '', name)
        name = re.sub(r'(?i)(?:[_-]video)?[_-][a-z]?\d+$', '', name)
    return name


def select_groups(names, dataset, count, rng):
    groups = defaultdict(list)
    for name in names:
        groups[group_key(dataset, name)].append(name)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected = []
    for key in keys:
        if len(selected) + len(groups[key]) <= count:
            selected.extend(groups[key])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError('Cannot fill grouped holdout: ' + dataset)
    return sorted(selected)


def prepare(out):
    if out.exists():
        raise RuntimeError('Refusing to replace an existing experiment: ' + str(out))
    out.mkdir(parents=True)
    split = out / 'split'
    split.mkdir()
    rng = random.Random(20261018)
    datasets = dict(LASOT=Lasot(split='train', multi_modal_vision=True, multi_modal_language=True, use_nlp=True),
                    VASTTRACK=VastTrack(split='train', multi_modal_vision=True, multi_modal_language=True, use_nlp=True),
                    TNL2K_train=Tnl2k(split=None, multi_modal_vision=True, multi_modal_language=True, use_nlp=True))
    records, counts, sources, lists = [], {}, {}, {}
    for key, dataset in datasets.items():
        name = MAPPING[key]
        source = CANONICAL / (name + '_train.txt')
        names = source.read_text().splitlines()
        expected = dict(lasot=1120, vasttrack=46825, tnl2k=1299)[name]
        assert len(names) == expected and len(set(names)) == expected
        assert 'Fight_game_004-Done' not in names if name == 'tnl2k' else True
        ids = {n: i for i, n in enumerate(dataset.sequence_list)}
        assert not set(names) - set(ids), (name, sorted(set(names) - set(ids))[:5])
        val = select_groups(names, name, 30 if name == 'lasot' else 60, rng)
        train = [n for n in names if n not in set(val)]
        assert not {group_key(name, n) for n in train} & {group_key(name, n) for n in val}
        train_path = split / (name + '_train.txt')
        val_path = split / (name + '_val.txt')
        train_path.write_text('\n'.join(train) + '\n')
        val_path.write_text('\n'.join(val) + '\n')
        lists[key] = str(train_path)
        sources[name] = dict(path=str(source), sha256=sha(source),
                             training_sha256=sha(train_path), validation_sha256=sha(val_path))
        # Audit selected videos fully at the path/annotation level; decode first,
        # middle and last frames. Canonical training inventory was audited earlier.
        for sequence in val:
            i = ids[sequence]
            info = dataset.get_sequence_info(i)
            boxes = info['bbox'].numpy()
            language = str(info['nlp']).strip()
            if name == 'tnl2k':
                folder = DATA / name / 'train' / sequence
                frames = sorted(p for p in (folder / 'imgs').iterdir() if p.suffix.lower() in ('.jpg', '.png', '.jpeg'))
                assert len(frames) == len(boxes), sequence
            else:
                folder = Path(dataset._get_sequence_path(i))
                frames = [Path(dataset._get_frame_path(str(folder), j)) for j in range(len(boxes))]
            assert boxes.shape == (len(frames), 4) and np.isfinite(boxes).all() and language
            assert all(p.is_file() and p.stat().st_size for p in frames), sequence
            sample_hashes = []
            for j in sorted({0, len(frames)//2, len(frames)-1}):
                assert cv2.imread(str(frames[j])) is not None, frames[j]
                sample_hashes.append(sha(frames[j]))
            dest = split / 'validation' / name / sequence
            dest.mkdir(parents=True)
            np.savez_compressed(dest / 'annotations.npz', boxes=boxes, visible=info['visible'].numpy().astype(bool))
            (dest / 'frames.txt').write_text('\n'.join(map(str, frames)) + '\n')
            records.append(dict(key=name+'/'+sequence, dataset=name, name=sequence, split='val',
                                language=language, frames=len(frames), category=sequence.rsplit('-',1)[0],
                                source_group=group_key(name, sequence), sampled_frame_hashes=sample_hashes,
                                annotations=str(dest/'annotations.npz'), frame_list=str(dest/'frames.txt')))
        counts[name] = dict(source=len(names), train=len(train), val=len(val))
        print(name, counts[name], flush=True)
    # Standard tests are never used to choose these training-side holdouts.
    test_lasot = set(Path('/root/MVLM/data/lasot/testing_set.txt').read_text().splitlines())
    assert not test_lasot & set((split/'lasot_train.txt').read_text().splitlines())
    assert not test_lasot & set((split/'lasot_val.txt').read_text().splitlines())
    test_tnl2k = {p.name for p in (DATA/'tnl2k/test').iterdir() if p.is_dir()}
    assert not test_tnl2k & set((split/'tnl2k_train.txt').read_text().splitlines())
    assert not test_tnl2k & set((split/'tnl2k_val.txt').read_text().splitlines())
    manifest = dict(seed=20261018, protocol='official-training-only memory adaptation holdout',
                    limitations='Base checkpoint may have seen these training videos; same-source grouping '
                    'and sampled hashes do not exclude every near duplicate. Not an unseen final test.',
                    counts=counts, sources=sources, records=records)
    write_json(split/'validation_manifest.json', manifest)
    quick = copy.deepcopy(manifest)
    quick['records'] = [r for name in MAPPING.values() for r in sorted(
        (r for r in records if r['dataset']==name), key=lambda r:r['name'])[:10]]
    quick['scope'] = 'fixed_30_sequence_subset'
    write_json(split/'quick_manifest.json', quick)
    config = yaml.safe_load((ROOT/'experiments/mvlm/mvlm_TF_memory_concat.yaml').read_text())
    config['DATA']['TRAIN'].update(CUSTOM_MANIFEST='', DATASETS_NAME=list(MAPPING),
        DATASETS_RATIO=[1,1,1], SEQUENCE_LISTS=lists, SAMPLE_PER_EPOCH=80000)
    config['DATA']['LONG_CLIP_START_EPOCH'] = 0
    config['TRAIN'].update(BATCH_SIZE=40, EPOCH=4, MEMORY_FREEZE_BN_STATS=True,
                           MEMORY_HEAD_LR=1e-5, LR=1e-4, CHECKPOINT_INTERVAL_STEPS=500)
    config['WANDB'].update(ENABLED=True, NAME=out.name, PROJECT='mvlm-memory', MODE='online')
    (out/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(out/'data_preparation.json', dict(counts=counts, sources=sources,
        validation_manifest_sha256=sha(split/'validation_manifest.json'),
        quick_manifest_sha256=sha(split/'quick_manifest.json'), decoded_validation_images=450,
        baseline_checkpoint_sha256=sha(config['MODEL']['PRETRAINED_MODEL']),
        known_exclusions=dict(tnl2k=['Fight_game_004-Done'], vasttrack='11 missing-frame videos absent from canonical list')))
    print('PREPARED', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    prepare(parser.parse_args().out.resolve())
