"""Resume public test-set downloads, extract, and validate repository loaders."""
import ast
import concurrent.futures as cf
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import threading
import zipfile

ROOT = Path('/root/MVLM')
DOWNLOADS = ROOT / 'downloads'
DATA = ROOT / 'data'
STATUS = DOWNLOADS / 'dataset_status.json'
CHUNK = 256 * 1024 * 1024
state = json.loads(STATUS.read_text()) if STATUS.exists() else {}
lock = threading.Lock()

def record(name, **values):
    with lock:
        state.setdefault(name, {}).update(values)
        if values.get('status') == 'downloading':
            state[name].pop('error', None)
        temp = STATUS.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        temp.replace(STATUS)
        print(name, values, flush=True)

def fetch_part(url, target, start, end):
    size = end - start + 1
    if target.exists() and target.stat().st_size == size:
        return
    failures = 0
    while failures < 40:
        saved = target.stat().st_size if target.exists() else 0
        if saved == size:
            return
        if saved > size:
            raise RuntimeError('Oversized download chunk: ' + target.name)
        first = start + saved
        last = min(first + 16 * 1024 * 1024 - 1, end)
        segment = target.with_suffix('.segment')
        headers = target.with_suffix('.headers')
        download_url = url + '?download=true&range_start=' + str(first)
        subprocess.run(['curl', '-fLsS', '--http1.1', '--connect-timeout', '30',
            '--max-time', '180', '--speed-limit', '1024', '--speed-time', '60',
            '--range', '%s-%s' % (first, last), '-D', str(headers),
            download_url, '-o', str(segment)], stdout=subprocess.DEVNULL)
        ranges = re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',
                            headers.read_text(errors='replace').lower()) if headers.exists() else []
        received = segment.stat().st_size if segment.exists() else 0
        if ranges and int(ranges[-1][0]) == first and 0 < received <= last - first + 1:
            with target.open('ab') as out, segment.open('rb') as src:
                shutil.copyfileobj(src, out)
            failures = 0
        else:
            failures += 1
            time.sleep(min(30, 2 ** min(failures, 5)))
    raise RuntimeError('Download failed after retries: ' + target.name)

def download(repo, row):
    target = DOWNLOADS / row['path']
    size = row['size']
    if target.exists() and target.stat().st_size == size:
        return target
    chunks = DOWNLOADS / (target.name + '.parts')
    chunks.mkdir(exist_ok=True)
    url = 'https://huggingface.co/datasets/%s/resolve/main/%s' % (repo, row['path'])
    jobs = [(url, chunks / ('%06d' % i), start, min(start + CHUNK, size) - 1)
            for i, start in enumerate(range(0, size, CHUNK))]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch_part, *job) for job in jobs]
        for future in cf.as_completed(futures):
            future.result()
    partial = target.with_suffix(target.suffix + '.assembling')
    digest = hashlib.sha256()
    with partial.open('wb') as out:
        for _, part, _, _ in jobs:
            with part.open('rb') as src:
                while True:
                    block = src.read(8 * 1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    out.write(block)
    if partial.stat().st_size != size:
        raise RuntimeError('Incorrect assembled size')
    expected_hash = row.get('lfs', {}).get('oid')
    if expected_hash and digest.hexdigest() != expected_hash:
        raise RuntimeError('SHA256 mismatch: ' + target.name)
    partial.replace(target)
    # Keep original archives; discard only our redundant completed chunks.
    for _, part, _, _ in jobs:
        part.unlink()
        for suffix in ('.segment', '.headers'):
            auxiliary = part.with_suffix(suffix)
            if auxiliary.exists():
                auxiliary.unlink()
    chunks.rmdir()
    return target

def extract_tar(archive, destination, mapper):
    with tarfile.open(archive, 'r|gz') as tar:
        for entry in tar:
            parts = Path(entry.name).parts
            if '..' in parts or entry.name.startswith('/'):
                raise RuntimeError('Unsafe archive path')
            relative = mapper(parts)
            if relative is None or not entry.isfile():
                continue
            path = destination.joinpath(*relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(entry) as src, path.open('wb') as out:
                shutil.copyfileobj(src, out)

def lasot():
    rows = json.loads((DOWNLOADS / 'lasot_test_tree.json').read_text())
    source = ast.parse((ROOT / 'lib/test/evaluation/lasotdataset.py').read_text())
    sequences = next(ast.literal_eval(n.value) for n in ast.walk(source)
                     if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'sequence_list' for t in n.targets))
    wanted = set(sequences)
    destination = DATA / 'lasot'
    record('lasot', status='downloading', expected_sequences=len(wanted))
    for row in rows:
        if not row['path'].endswith('.tar.gz'):
            continue
        archive = download('ZhengGuangze/LaSOTTest', row)
        record('lasot', status='extracting', archive=archive.name)
        def mapper(parts):
            for i, part in enumerate(parts):
                if part in wanted:
                    return (part.split('-')[0],) + tuple(parts[i:])
            return None
        marker = archive.with_suffix('.extracted')
        if not marker.exists():
            extract_tar(archive, destination, mapper)
            marker.write_text('done')
    (destination / 'testing_set.txt').write_text('\n'.join(sequences) + '\n')
    record('lasot', status='extracted')

def tnl2k():
    rows = json.loads((DOWNLOADS / 'xche32_tnl2k_tree.json').read_text())
    row = next(x for x in rows if x['path'] == 'tnl2k.tar.gz')
    record('tnl2k', status='downloading', expected_sequences=700)
    archive = download('xche32/tnl2k', row)
    record('tnl2k', status='extracting')
    def mapper(parts):
        if parts and parts[0] == 'tnl2k':
            parts = parts[1:]
        if parts and parts[0] == 'test':
            parts = parts[1:]
        return parts or None
    marker = archive.with_suffix('.extracted')
    if not marker.exists():
        extract_tar(archive, DATA / 'tnl2k/test', mapper)
        marker.write_text('done')
    count = sum(p.is_dir() for p in (DATA / 'tnl2k/test').iterdir())
    if count != 700:
        raise RuntimeError('TNL2K sequence count is %s, expected 700' % count)
    record('tnl2k', status='extracted', sequences=count)

def otb():
    archive = DOWNLOADS / 'OTB_sentences.zip'
    record('otb99_lang', status='downloading')
    expected_size = 2798845728
    for attempt in range(20):
        if archive.exists() and archive.stat().st_size == expected_size:
            with zipfile.ZipFile(archive) as z:
                z.infolist()
            break
        result = subprocess.run(['curl', '-fLsS', '-C', '-', '--connect-timeout', '30',
            '--speed-limit', '1024', '--speed-time', '90', '--max-time', '3600',
            'https://isis-data.science.uva.nl/zhenyang/cvpr17-langtracker/data/OTB_sentences.zip',
            '-o', str(archive)])
        if result.returncode:
            time.sleep(10)
    else:
        raise RuntimeError('Official OTB archive download failed')
    destination = DATA / 'otb99_lang'
    record('otb99_lang', status='extracting')
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            parts = Path(entry.filename).parts
            if '..' in parts or entry.filename.startswith('/'):
                raise RuntimeError('Unsafe ZIP path')
            anchor = next((i for i, p in enumerate(parts) if p in ('OTB_videos', 'OTB_query_test', 'OTB_query_train')), None)
            if anchor is None or entry.is_dir():
                continue
            path = destination.joinpath(*parts[anchor:])
            path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(entry) as src, path.open('wb') as out:
                shutil.copyfileobj(src, out)
    record('otb99_lang', status='extracted')

def validate():
    import sys
    sys.path.insert(0, str(ROOT))
    from lib.test.evaluation import get_dataset
    for name in ('otb99_lang', 'tnl2k', 'lasot'):
        sequences = get_dataset(name)
        frames = 0
        for seq in sequences:
            if not seq.language_query:
                raise RuntimeError('Missing language: ' + seq.name)
            if len(seq.frames) != len(seq.ground_truth_rect):
                raise RuntimeError('Frame/annotation mismatch: ' + seq.name)
            for frame in seq.frames:
                if not Path(frame).is_file():
                    raise RuntimeError('Missing frame: ' + frame)
            frames += len(seq.frames)
        record(name, status='validated', sequences=len(sequences), frames=frames)

if __name__ == '__main__':
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn): name for fn, name in [(otb, 'otb99_lang'), (tnl2k, 'tnl2k'), (lasot, 'lasot')]}
        failed = False
        for future in cf.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:
                failed = True
                record(name, status='failed', error=str(exc))
    if not failed:
        validate()
        print('All three test datasets validated.', flush=True)
