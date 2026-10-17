"""Resume VastTrack volumes, verify SHA256, extract and validate MVLM's train split."""
import datetime
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import threading
import time
import traceback

ROOT = Path('/root/MVLM')
DOWNLOADS = ROOT / 'downloads/vasttrack'
DESTINATION = ROOT / 'data/vasttrack/train'
STATUS = DOWNLOADS / 'status.json'
state = {}
state_lock = threading.Lock()


def update(**values):
    with state_lock:
        state.update(values)
        state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary = STATUS.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n')
        temporary.replace(STATUS)


def saved_bytes(files):
    total = 0
    for row in files:
        archive = DOWNLOADS / row['path']
        if archive.exists() and archive.stat().st_size == row['size']:
            total += row['size']
            continue
        folder = DOWNLOADS / (row['path'] + '.parts')
        if folder.exists():
            count = 0
            try:
                entries = list(folder.iterdir())
            except FileNotFoundError:
                entries = []
            for p in entries:
                try:
                    if p.name.isdigit():
                        count += p.stat().st_size
                except FileNotFoundError:
                    pass
            total += min(count, row['size'])
    return total


class Volumes(io.RawIOBase):
    def __init__(self, paths):
        super().__init__()
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = path.open('rb')
            size = self.current.readinto(buffer)
            if size:
                return size
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def validate(wanted):
    import numpy as np
    missing = []
    frames = 0
    for name in sorted(wanted):
        directory = DESTINATION / name.rsplit('-', 1)[0] / name
        annotation = directory / 'Groundtruth.txt'
        language = directory / 'nlp.txt'
        if not annotation.is_file() or not language.is_file() or not language.read_text(errors='replace').strip():
            missing.append(name + ': annotation or language missing')
            continue
        try:
            boxes = np.loadtxt(str(annotation), delimiter=',', ndmin=2)
            if boxes.shape[1] != 4 or not np.isfinite(boxes).all():
                raise ValueError('invalid boxes')
            for frame_id in range(1, len(boxes) + 1):
                if not (directory / 'imgs' / ('%05d.jpg' % frame_id)).is_file():
                    raise ValueError('missing frame %d' % frame_id)
            frames += len(boxes)
        except Exception as exc:
            missing.append(name + ': ' + str(exc))
    report = dict(expected_sequences=len(wanted), total_frames=frames,
                  problems=missing, checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (DOWNLOADS / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    if missing:
        raise RuntimeError('%d sequences failed validation; see validation.json' % len(missing))
    sys.path.insert(0, str(ROOT))
    from lib.train.dataset.vasttrack import VastTrack
    from lib.train.data import opencv_loader
    dataset = VastTrack(split='train', image_loader=opencv_loader,
                        multi_modal_language=True, use_nlp=True)
    assert len(dataset.sequence_list) == len(wanted)
    info = dataset.get_sequence_info(0)
    images, _, _ = dataset.get_frames(0, [0], info)
    assert images[0] is not None and images[0].size > 0
    return frames


def main():
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DOWNLOADS / 'source_manifest.json').read_text())
    files = sorted(manifest['files'], key=lambda x: x['path'])
    wanted = set((ROOT / 'lib/train/data_specs/vasttrack_train_split.txt').read_text().splitlines())
    wanted.discard('')
    sys.path.insert(0, str(ROOT / 'downloads'))
    import prepare_test_datasets as downloader
    downloader.DOWNLOADS = DOWNLOADS
    update(status='downloading', source=manifest['source_url'], total_bytes=manifest['total_bytes'],
           downloaded_bytes=saved_bytes(files), target_dir=str(DESTINATION),
           expected_sequences=len(wanted), pid=os.getpid())
    finished = threading.Event()

    def monitor():
        while not finished.wait(10):
            update(downloaded_bytes=saved_bytes(files))

    threading.Thread(target=monitor, daemon=True).start()
    try:
        for number, row in enumerate(files, 1):
            update(archive=row['path'], archive_index=number, archives_total=len(files))
            print('DOWNLOAD', number, '/', len(files), row['path'], flush=True)
            # Existing downloader verifies the LFS SHA256 when assembling each volume.
            downloader.download(manifest['repo'], row)
            print('VERIFIED', row['path'], flush=True)
        update(status='extracting', downloaded_bytes=manifest['total_bytes'])
        extracted = 0
        with io.BufferedReader(Volumes([DOWNLOADS / row['path'] for row in files]), 4 * 1024 * 1024) as stream:
            with tarfile.open(fileobj=stream, mode='r|gz') as archive:
                for member in archive:
                    parts = Path(member.name).parts
                    if member.name.startswith('/') or '..' in parts:
                        raise RuntimeError('Unsafe archive member')
                    if parts and parts[0] == 'vasttrack':
                        parts = parts[1:]
                    if parts and parts[0] in ('train', 'training'):
                        parts = parts[1:]
                    if len(parts) < 3 or parts[1] not in wanted or not member.isfile():
                        continue
                    if parts[0] != parts[1].rsplit('-', 1)[0]:
                        raise RuntimeError('Unexpected class/sequence layout: ' + member.name)
                    destination = DESTINATION.joinpath(*parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists() or destination.stat().st_size != member.size:
                        temporary = destination.with_name(destination.name + '.vasttrack.tmp')
                        with archive.extractfile(member) as source, temporary.open('wb') as out:
                            shutil.copyfileobj(source, out, 1024 * 1024)
                        temporary.replace(destination)
                    extracted += 1
                    if extracted % 10000 == 0:
                        update(extracted_files=extracted)
        update(status='validating', extracted_files=extracted)
        frames = validate(wanted)
        update(status='ready', validated_sequences=len(wanted), validated_frames=frames)
        print('READY', len(wanted), 'sequences', frames, 'frames', flush=True)
    except Exception as exc:
        update(status='failed', error=str(exc))
        traceback.print_exc()
        raise
    finally:
        finished.set()


if __name__ == '__main__':
    main()
