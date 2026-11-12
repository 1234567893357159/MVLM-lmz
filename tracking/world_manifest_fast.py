"""Keep the original manifest audit, using directory-entry metadata for regular frames."""
import hashlib
import os
from pathlib import Path

import numpy as np
from lib.train.dataset import world_clips


def describe_sequence(dataset, root, name):
    # Match world_clips.describe_sequence's checks and fingerprint byte-for-byte.
    # scandir avoids a separate lstat for every regular image on network storage.
    root = Path(root).resolve()
    if dataset in ('lasot', 'vasttrack'):
        if name not in world_clips.official_names(dataset):
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
    image_dir = (directory / ('img' if dataset == 'lasot' else 'imgs')).resolve()
    with os.scandir(str(image_dir)) as entries:
        frames = sorted(Path(entry.path).resolve() if entry.is_symlink() else Path(entry.path)
                        for entry in entries if Path(entry.name).suffix.lower() in ('.jpg', '.jpeg', '.png'))
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
    original = world_clips.describe_sequence
    count = [0]

    def audited(*args, **kwargs):
        result = describe_sequence(*args, **kwargs)
        count[0] += 1
        if count[0] % 5000 == 0:
            print('Audited {} video annotations and frame paths.'.format(count[0]), flush=True)
        return result

    world_clips.describe_sequence = audited
    try:
        return world_clips.read_manifest(path, clip_length)
    finally:
        world_clips.describe_sequence = original
