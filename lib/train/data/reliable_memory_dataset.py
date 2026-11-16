"""Audited official-train manifests and raw-frame clips for closed-loop training."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

SPECS = Path(__file__).resolve().parents[1] / 'data_specs'

@lru_cache(maxsize=3)
def official_names(dataset):
    return set((SPECS / (dataset + '_train_split.txt')).read_text().splitlines())


def describe_sequence(dataset, root, name):
    root = Path(root).resolve()
    if dataset in ('lasot', 'vasttrack'):
        if name not in official_names(dataset):
            raise ValueError('Sequence is outside official training list: ' + name)
        category = name.rsplit('-', 1)[0]
        directory = root / ('train' if dataset == 'vasttrack' else '') / category / name
    elif dataset == 'tnl2k':
        directory = root / 'train' / name
    else:
        raise ValueError('Only lasot, tnl2k and vasttrack official training data supported')
    if dataset == 'tnl2k' and name == 'Fight_game_004-Done':
        raise ValueError('Excluded annotation/frame mismatch: ' + name)
    if Path(name).name != name:
        raise ValueError('Sequence names cannot contain paths')
    # Resolve shared ancestors once; still resolve individual symlink aliases.
    # Repeating realpath for millions of regular frames makes startup very slow.
    image_dir = (directory / ('img' if dataset == 'lasot' else 'imgs')).resolve()
    frames = sorted(p.resolve() if p.is_symlink() else p for p in image_dir.iterdir()
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    annotation = directory / ('Groundtruth.txt' if dataset == 'vasttrack' else 'groundtruth.txt')
    boxes = np.loadtxt(str(annotation), delimiter=',', dtype=np.float32).reshape(-1, 4)
    language = directory / ('language.txt' if dataset == 'tnl2k' else 'nlp.txt')
    text = language.read_text().strip()
    if not text or len(boxes) != len(frames) or len(frames) < 5:
        raise ValueError('Invalid frames/annotations/language: ' + str(directory))
    valid = np.isfinite(boxes).all(1) & (boxes[:, 2:] > 0).all(1)
    visible = valid.copy()
    if dataset == 'lasot':
        for filename in ('full_occlusion.txt', 'out_of_view.txt'):
            path = directory / filename
            if not path.exists():
                raise ValueError('Missing LaSOT visibility: ' + str(path))
            values = np.loadtxt(str(path), delimiter=',').reshape(-1)
            if len(values) != len(frames):
                raise ValueError('Visibility length mismatch: ' + str(path))
            visible &= values == 0
    if not valid.any():
        raise ValueError('No valid boxes: ' + str(directory))
    signature = hashlib.sha256(boxes.tobytes() + valid.tobytes() + visible.tobytes()
                               + text.encode() + '\n'.join(map(str, frames)).encode()).hexdigest()
    return dict(dataset=dataset, name=name, fingerprint=signature, directory=str(directory.resolve()),
                frames=[str(p) for p in frames], boxes=boxes, valid=valid,
                visible=visible, language=text)



def read_manifest(path, clip_length=1, limit_per_dataset=0):
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    if manifest.get('protocol') != 'official-train-video-split-v1':
        raise ValueError('Require an official-train-video-split-v1 manifest')
    if clip_length < 1 or limit_per_dataset < 0:
        raise ValueError('Invalid clip length/subset size')
    seen = set()
    # Audit all identities, even for an explicitly labelled engineering subset.
    for split in ('train', 'val'):
        for row in manifest[split]:
            identity = (row['dataset'], row['name'])
            if identity in seen:
                raise ValueError('Overlapping or duplicate video: ' + str(identity))
            if Path(row['name']).name != row['name']:
                raise ValueError('Names cannot contain paths')
            if row['dataset'] in ('lasot', 'vasttrack') and row['name'] not in official_names(row['dataset']):
                raise ValueError('Nontraining video: ' + str(identity))
            if row['dataset'] not in ('lasot', 'vasttrack', 'tnl2k'):
                raise ValueError('Unknown source')
            seen.add(identity)
    result, paths = {}, set()
    for split in ('train', 'val'):
        result[split] = []
        counts = {}
        for row in manifest[split]:
            ds = row['dataset']
            if limit_per_dataset and counts.get(ds, 0) >= limit_per_dataset:
                continue
            seq = describe_sequence(ds, manifest['roots'][ds], row['name'])
            if row.get('fingerprint', seq['fingerprint']) != seq['fingerprint']:
                raise ValueError('Sequence signature changed: ' + row['name'])
            if len(seq['frames']) < clip_length:
                raise ValueError('Sequence shorter than clip: ' + row['name'])
            if paths.intersection(seq['frames']):
                raise ValueError('Resolved frame overlap')
            paths.update(seq['frames'])
            result[split].append(seq)
            counts[ds] = counts.get(ds, 0) + 1
        if not result[split]:
            raise ValueError('Empty split: ' + split)
    return result, hashlib.sha256(raw).hexdigest()


class ReliableClips(Dataset):
    def __init__(self, sequences, length=12, samples=100, seed=20261116,
                 max_stride=4, augment=True, dataset_weights=None):
        if length < 2 or max_stride < 1 or samples < 1:
            raise ValueError('Invalid clip sampling configuration')
        self.groups = {}
        for seq in sequences:
            if len(seq['frames']) < length:
                raise ValueError('Short sequence')
            self.groups.setdefault(seq['dataset'], []).append(seq)
        self.sources = sorted(self.groups)
        self.weights = np.asarray([1. if dataset_weights is None else dataset_weights[d]
                                   for d in self.sources], dtype=float)
        if not np.isfinite(self.weights).all() or (self.weights <= 0).any():
            raise ValueError('Source sampling weights must be finite and positive')
        self.probabilities = self.weights/self.weights.sum()
        self.length, self.samples, self.seed = length, samples, seed
        self.max_stride, self.augment = max_stride, augment

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = np.random.RandomState((self.seed + int(index)) % (2**32))
        source = self.sources[rng.randint(len(self.sources)) if np.all(self.weights == self.weights[0])
                              else rng.choice(len(self.sources), p=self.probabilities)]
        group = self.groups[source]
        seq = group[rng.randint(len(group))]
        maximum = min(self.max_stride, (len(seq['frames'])-1)//(self.length-1))
        stride = rng.randint(1, maximum+1)
        starts = np.arange(len(seq['frames'])-(self.length-1)*stride)
        # Only choose an annotated visible start; retain every sampled interior frame.
        starts = starts[seq['visible'][starts]]
        if not len(starts):
            # No visible bootstrap with this stride: retain the video and its
            # unknown/invisible interior rather than selecting a different video.
            starts = np.arange(len(seq['frames'])-(self.length-1)*stride)
        ids = starts[rng.randint(len(starts))] + stride*np.arange(self.length)
        gain = rng.uniform(.9, 1.1) if self.augment else 1.
        frames = []
        for frame_id in ids:
            image = cv2.imread(seq['frames'][frame_id])
            if image is None:
                raise ValueError('Cannot decode ' + seq['frames'][frame_id])
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(np.clip(rgb.astype(np.float32)*gain, 0, 255).astype(np.uint8))
        return dict(images=frames, boxes=torch.from_numpy(seq['boxes'][ids].copy()),
                    valid=torch.from_numpy(seq['valid'][ids].copy()),
                    visible=torch.from_numpy(seq['visible'][ids].copy()),
                    known_absent=torch.from_numpy((~seq['visible'][ids] & seq['valid'][ids]).copy())
                        if source == 'lasot' else torch.zeros(len(ids), dtype=torch.bool),
                    frame_ids=ids.tolist(), language=seq['language'], name=source+'/'+seq['name'])
