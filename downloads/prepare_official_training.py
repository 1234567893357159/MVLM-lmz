"""Resume mirrored official training data; never extract over benchmark tests."""
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'downloads/official_training'
LOCK = threading.Lock()
STATE = {}


def source_rows(name):
    rows = json.loads((OUT / (name + '_sources.json')).read_text())
    if name == 'lasot':
        categories = {s.split('-')[0] for s in
                      (ROOT / 'lib/train/data_specs/lasot_train_split.txt').read_text().splitlines()}
        return [r for r in rows if r['path'].endswith('.zip') and Path(r['path']).stem in categories]
    return sorted((r for r in rows if '.tar.gz.part.' in r['path']), key=lambda r: r['path'])


def saved_bytes(rows):
    total = 0
    for row in rows:
        target = OUT / row['path']
        try:
            if target.is_file() and target.stat().st_size == row['size']:
                total += row['size']
                continue
            folder = OUT / (row['path'] + '.parts')
            total += min(row['size'], sum(p.stat().st_size for p in folder.glob('*') if p.name.isdigit()))
        except FileNotFoundError:
            # Assembly may remove chunk files while this snapshot is collected.
            continue
    return total


def monitor(stop):
    previous = {}
    while not stop.is_set():
        for name in ('lasot', 'tnl2k'):
            now = time.monotonic()
            rows = source_rows(name)
            saved = saved_bytes(rows)
            before, stamp = previous.get(name, (saved, now))
            speed = max(0, saved - before) / max(now - stamp, 1)
            previous[name] = saved, now
            verified = sum((OUT / (r['path'] + '.verified.json')).exists() for r in rows)
            record(name, downloaded_bytes=saved, speed_bytes_per_second=speed,
                   verified_archives=verified, archives_total=len(rows))
        stop.wait(10)


def record(name, **values):
    with LOCK:
        if values.get('status') in ('downloading', 'repairing', 'ready'):
            for key in ('error', 'pause_reason', 'paused_at'):
                STATE.get(name, {}).pop(key, None)
        STATE.setdefault(name, {}).update(values, updated_at=time.time())
        temp = OUT / 'status.tmp'
        temp.write_text(json.dumps(STATE, indent=2))
        temp.replace(OUT / 'status.json')
    print(name, values, flush=True)


def verified_download(repo, row):
    name = 'lasot' if repo == 'l-lt/LaSOT' else 'tnl2k'
    archive = DOWNLOADER.download(repo, row)
    record(name, status='verifying')
    marker = archive.with_name(archive.name + '.verified.json')
    expected = row['lfs']['oid']
    stat = archive.stat()
    if marker.exists():
        old = json.loads(marker.read_text())
        if old == dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=expected):
            return archive
    digest = hashlib.sha256()
    with archive.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError('Archive SHA256 mismatch: ' + str(archive))
    marker.write_text(json.dumps(dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=expected)))
    return archive


def copy_member(source, target, size):
    if target.exists() and target.stat().st_size == size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.extracting')
    with source, partial.open('wb') as dest:
        shutil.copyfileobj(source, dest)
    if partial.stat().st_size != size:
        raise RuntimeError('Incomplete extraction: ' + str(target))
    partial.replace(target)


def parts(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe archive path: ' + name)
    return path.parts


def validate(name, sequences, root):
    import cv2
    import numpy as np
    total = 0
    for index, seq in enumerate(sequences):
        folder = root / (seq.split('-')[0] if name == 'lasot' else '') / seq
        language = folder / ('nlp.txt' if name == 'lasot' else 'language.txt')
        if not language.is_file() or not language.read_text(errors='replace').strip():
            raise RuntimeError('Missing language: ' + str(folder))
        boxes = np.loadtxt(str(folder / 'groundtruth.txt'), delimiter=',', ndmin=2)
        if boxes.shape[1] != 4 or not np.isfinite(boxes).all():
            raise RuntimeError('Invalid annotations: ' + str(folder))
        images = folder / ('img' if name == 'lasot' else 'imgs')
        files = sorted(p for p in images.iterdir() if p.suffix.lower() in ('.jpg', '.png', '.jpeg'))
        if len(files) != len(boxes) or any(p.stat().st_size == 0 for p in files):
            raise RuntimeError('Image count/size mismatch: %s (images=%d, annotations=%d)' %
                               (folder, len(files), len(boxes)))
        if name == 'lasot':
            for label in ('full_occlusion.txt', 'out_of_view.txt'):
                if np.loadtxt(str(folder / label), delimiter=',').size != len(boxes):
                    raise RuntimeError('Visibility count mismatch: ' + str(folder))
        for p in (files[0], files[len(files)//2], files[-1]):
            if cv2.imread(str(p)) is None:
                raise RuntimeError('Image decode failed: ' + str(p))
        total += len(files)
        if index % 50 == 0:
            record(name, status='validating', validated_sequences=index + 1, validated_frames=total)
    return total


def lasot():
    name = 'lasot'
    sequences = (ROOT / 'lib/train/data_specs/lasot_train_split.txt').read_text().splitlines()
    if len(set(sequences)) != 1120:
        raise RuntimeError('Expected 1120 official LaSOT training sequences')
    sequence_set = set(sequences)
    test_list = ROOT / 'data/lasot/testing_set.txt'
    if sequence_set & set(test_list.read_text().splitlines()):
        raise RuntimeError('LaSOT training and testing sequence lists overlap')
    categories = {s.split('-')[0] for s in sequences}
    rows = source_rows(name)
    if {Path(r['path']).stem for r in rows} != categories:
        raise RuntimeError('LaSOT category archive list incomplete')
    record(name, status='downloading', total_bytes=sum(r['size'] for r in rows),
           downloaded_bytes=saved_bytes(rows), expected_sequences=len(sequences))
    extracted = 0
    for row in rows:
        record(name, status='downloading', current_archive=row['path'])
        archive = verified_download('l-lt/LaSOT', row)
        marker = archive.with_name(archive.name + '.extracted')
        if marker.exists():
            continue
        record(name, status='extracting')
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                pp = parts(member.filename)
                matches = [i for i, p in enumerate(pp) if p in sequence_set]
                if member.is_dir() or not matches:
                    continue
                i = matches[0]
                seq = pp[i]
                target = ROOT / 'data/lasot' / seq.split('-')[0] / Path(*pp[i:])
                copy_member(z.open(member), target, member.file_size)
                extracted += 1
                if extracted % 10000 == 0:
                    record(name, extracted_files=extracted)
        marker.write_text('complete\n')
        record(name, status='downloading')
    frames = validate(name, sequences, ROOT / 'data/lasot')
    (OUT / 'lasot_validation.json').write_text(json.dumps(dict(
        sequences=len(sequences), frames=frames, problems=[],
        image_check='All file counts and sizes; first/middle/last decode per sequence',
        train_list_sha256=hashlib.sha256(('\n'.join(sequences) + '\n').encode()).hexdigest()), indent=2))
    record(name, status='ready', validated_sequences=len(sequences), validated_frames=frames)


TNL2K_TRAIN_RENAMES = {'monitor_manypeople': 'monitor_manypeople_train'}
TNL2K_REVISIONS_SHA256 = '2a2a2c48268c8ea14b3ec8d33753a1a4cfdb727fb3e6b701d86f43c6e93f0d7c'
TNL2K_REVISIONS_URL = ('https://raw.githubusercontent.com/wangxiao5791509/'
                       'TNL2K_evaluation_toolkit/main/annos/revised_annotations.zip')


def apply_tnl2k_revisions(root):
    """Apply the authors' corrections, retaining original mirrored annotations."""
    archive = OUT / 'tnl2k_revised_annotations.zip'
    if not archive.exists():
        partial = archive.with_suffix('.zip.downloading')
        subprocess.run(['curl', '-fLsS', '--connect-timeout', '30', '--max-time', '180',
                        TNL2K_REVISIONS_URL, '-o', str(partial)], check=True)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != TNL2K_REVISIONS_SHA256:
            raise RuntimeError('Official TNL2K annotation revision SHA256 mismatch')
        partial.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != TNL2K_REVISIONS_SHA256:
        raise RuntimeError('Official TNL2K annotation revision SHA256 mismatch')
    applied = []
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            pp = parts(member.filename)
            if len(pp) != 2 or pp[1] != 'groundtruth.txt':
                continue
            target = root / Path(*pp)
            if not target.is_file():
                raise RuntimeError('Missing sequence for official correction: ' + str(target))
            data = z.read(member)
            if target.read_bytes() != data:
                backup = OUT / 'tnl2k_original_annotations' / Path(*pp)
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    shutil.copy2(target, backup)
                partial = target.with_suffix('.txt.revising')
                partial.write_bytes(data)
                partial.replace(target)
            applied.append(dict(sequence=pp[0], sha256=hashlib.sha256(data).hexdigest()))
    (OUT / 'tnl2k_annotation_revisions.json').write_text(json.dumps(dict(
        source=TNL2K_REVISIONS_URL, archive_sha256=TNL2K_REVISIONS_SHA256,
        annotations=applied), indent=2))


def check_tnl2k_renamed_frames(root, test_root):
    """Recheck content isolation for each explicitly renamed sequence."""
    for original, renamed in TNL2K_TRAIN_RENAMES.items():
        test = test_root / original / 'imgs'
        if not test.is_dir():
            continue
        def frame_hashes(folder):
            return {hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()
                    if p.suffix.lower() in ('.jpg', '.png', '.jpeg')}
        if frame_hashes(root / renamed / 'imgs') & frame_hashes(test):
            raise RuntimeError('TNL2K renamed training sequence shares test frames: ' + original)


def tnl2k_member_path(member, test_names):
    """Rename a content-reviewed collision; reject other test sequence names."""
    pp = parts(member.name)
    if not pp or pp[0] != 'tnl2k_train':
        raise RuntimeError('Unexpected TNL2K archive root: ' + member.name)
    if member.isdir():
        return None
    if not member.isfile():
        raise RuntimeError('Unsupported TNL2K archive member: ' + member.name)
    if len(pp) < 3:
        raise RuntimeError('Invalid TNL2K archive path: ' + member.name)
    # Reviewed locally: this is a 262-frame white-car clip, while the test
    # namesake is a different 618-frame pedestrian clip. No frame hashes match.
    seq = TNL2K_TRAIN_RENAMES.get(pp[1], pp[1])
    if seq in test_names:
        raise RuntimeError('TNL2K train/test overlap: ' + member.name)
    return Path(seq, *pp[2:])


def tnl2k():
    name = 'tnl2k'
    rows = source_rows(name)
    record(name, status='downloading', total_bytes=sum(r['size'] for r in rows),
           downloaded_bytes=saved_bytes(rows), expected_sequences=1300)
    archives = []
    for row in rows:
        record(name, status='downloading', current_archive=row['path'])
        archives.append(verified_download('xche32/tnl2k_train', row))
    marker = OUT / 'tnl2k.extracted'
    test_names = {p.name for p in (ROOT / 'data/tnl2k/test').iterdir() if p.is_dir()}
    if not marker.exists():
        record(name, status='extracting')
        extracted = 0
        process = subprocess.Popen(['cat'] + [str(p) for p in archives], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout, mode='r|gz') as tar:
                for member in tar:
                    relative = tnl2k_member_path(member, test_names)
                    if relative is None:
                        continue
                    copy_member(tar.extractfile(member), ROOT / 'data/tnl2k/train' / relative, member.size)
                    extracted += 1
                    if extracted % 10000 == 0:
                        record(name, extracted_files=extracted)
            process.stdout.close()
            if process.wait() != 0:
                raise RuntimeError('Archive concatenation failed')
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
        marker.write_text('complete\n')
    root = ROOT / 'data/tnl2k/train'
    sequences = sorted(p.name for p in root.iterdir() if p.is_dir())
    if len(sequences) != 1300 or set(sequences) & test_names:
        raise RuntimeError('Expected 1300 disjoint TNL2K training sequences')
    check_tnl2k_renamed_frames(root, ROOT / 'data/tnl2k/test')
    apply_tnl2k_revisions(root)
    (OUT / 'tnl2k_sequence_renames.json').write_text(json.dumps(TNL2K_TRAIN_RENAMES, indent=2))
    (OUT / 'tnl2k_train_sequences.txt').write_text('\n'.join(sequences) + '\n')
    frames = validate(name, sequences, root)
    (OUT / 'tnl2k_validation.json').write_text(json.dumps(dict(
        sequences=len(sequences), frames=frames, problems=[],
        image_check='All file counts and sizes; first/middle/last decode per sequence',
        test_sequence_overlap=0), indent=2))
    record(name, status='ready', validated_sequences=len(sequences), validated_frames=frames)


def guarded(name, func):
    try:
        func()
    except BaseException as exc:
        if DOWNLOADER.CANCELLED.is_set():
            record(name, status='paused', pause_reason='Process received stop signal')
        else:
            record(name, status='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    with (OUT / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        STATE['_runtime'] = dict(pid=os.getpid(), started_at=time.time(),
                                 proxy=os.environ.get('HTTPS_PROXY', ''),
                                 workers_per_dataset=int(os.environ.get('MVLM_DOWNLOAD_WORKERS', '10')),
                                 segment_mib=int(os.environ.get('MVLM_DOWNLOAD_SEGMENT_MIB', '8')),
                                 http2=os.environ.get('MVLM_DOWNLOAD_HTTP2') == '1',
                                 training_auto_start=False)
        (OUT / 'prepare.pid').write_text(str(os.getpid()) + '\n')

        def pause(signum, frame):
            # Signal handlers must not acquire STATE's non-reentrant lock.
            # Workers stop and reap their own curl before releasing flock.
            if 'DOWNLOADER' in globals():
                DOWNLOADER.CANCELLED.set()

        signal.signal(signal.SIGTERM, pause)
        spec = importlib.util.spec_from_file_location('existing_downloader', ROOT / 'downloads/training_range_download.py')
        DOWNLOADER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(DOWNLOADER)
        DOWNLOADER.DOWNLOADS = OUT
        def download_progress(row, **values):
            record('lasot' if row['path'].endswith('.zip') else 'tnl2k', **values)
        DOWNLOADER.PROGRESS_CALLBACK = download_progress
        stopped = threading.Event()
        watcher = threading.Thread(target=monitor, args=(stopped,), daemon=True)
        watcher.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(guarded, 'lasot', lasot), pool.submit(guarded, 'tnl2k', tnl2k)]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        finally:
            stopped.set()
            watcher.join()
        for name in ('lasot', 'tnl2k'):
            record(name, downloaded_bytes=saved_bytes(source_rows(name)), speed_bytes_per_second=0)
