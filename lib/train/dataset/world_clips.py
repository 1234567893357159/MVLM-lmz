"""Explicit video-disjoint train/validation manifests, never benchmark test splits."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


SPECS = Path(__file__).resolve().parents[1] / 'data_specs'


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
    if Path(name).name != name:
        raise ValueError('Sequence names cannot contain paths')
    image_dir = directory / ('img' if dataset == 'lasot' else 'imgs')
    frames = sorted(p.resolve() for p in image_dir.iterdir()
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


def read_manifest(path, clip_length):
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    if manifest.get('protocol') != 'official-train-video-split-v1':
        raise ValueError('Explicit official training video split manifest required')
    datasets = manifest['roots']
    partitions, identities, directories, images = {}, set(), set(), set()
    for split in ('train', 'val'):
        sequences = []
        for row in manifest[split]:
            identity = (row['dataset'], row['name'])
            if identity in identities:
                raise ValueError('Duplicate or overlapping video: ' + str(identity))
            identities.add(identity)
            seq = describe_sequence(row['dataset'], datasets[row['dataset']], row['name'])
            if row.get('fingerprint', seq['fingerprint']) != seq['fingerprint']:
                raise ValueError('Sequence annotations/paths changed since split preparation')
            if len(seq['frames']) < clip_length:
                raise ValueError('Sequence shorter than clip: ' + seq['name'])
            # Resolved paths also catch symlink aliases across videos or partitions.
            if seq['directory'] in directories or images.intersection(seq['frames']):
                raise ValueError('Resolved video/image path overlap')
            directories.add(seq['directory'])
            images.update(seq['frames'])
            sequences.append(seq)
        if not sequences:
            raise ValueError('Both train and validation must contain videos')
        partitions[split] = sequences
    return partitions, hashlib.sha256(raw).hexdigest()


class WorldClips(Dataset):
    """Stateless sampling by index: reproducible across workers and resumptions."""
    def __init__(self, sequences, length, samples, seed, stride=1, augment=True):
        self.sequences, self.length = sequences, length
        self.samples, self.seed, self.stride, self.augment = samples, seed, stride, augment

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = np.random.RandomState((self.seed + int(index)) % (2 ** 32))
        seq = self.sequences[rng.randint(len(self.sequences))]
        stride = min(self.stride, (len(seq['frames']) - 1) // (self.length - 1))
        start = rng.randint(len(seq['frames']) - (self.length - 1) * stride)
        ids = np.arange(self.length) * stride + start
        boxes = torch.from_numpy(seq['boxes'][ids].copy())
        valid = torch.from_numpy(seq['valid'][ids].copy())
        visible = torch.from_numpy(seq['visible'][ids].copy())
        # One photometric transform for the full clip. No random flips of directional language.
        gain = rng.uniform(.9, 1.1) if self.augment else 1.
        frames = []
        for frame_id in ids:
            image = cv2.imread(seq['frames'][frame_id])
            if image is None:
                raise ValueError('Cannot decode ' + seq['frames'][frame_id])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            frames.append(image)
        return dict(images=frames, boxes=boxes, valid=valid, visible=visible,
                    frame_ids=torch.from_numpy(ids), language=seq['language'],
                    name=seq['dataset'] + '/' + seq['name'])


def collate_world_clips(items):
    # Keep full-resolution images as lists: images can differ in shape across videos.
    return items
