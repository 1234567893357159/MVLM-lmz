"""Range downloader extracted from the verified VastTrack implementation.
Preserves strict HTTP ranges, resumable chunks and archive SHA256 checks.
"""
import ast
import fcntl
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
import urllib.parse
import uuid
CHUNK = 256 * 1024 * 1024
CANCELLED = threading.Event()
PROGRESS_CALLBACK = None

def progress(row, status, **values):
    if PROGRESS_CALLBACK is not None:
        PROGRESS_CALLBACK(row, status=status, **values)

def check_cancelled():
    if CANCELLED.is_set():
        raise RuntimeError('Download cancelled')

def run_curl(command):
    # Never leave curl writing after the parent releases its archive lock.
    with subprocess.Popen(command, stdout=subprocess.DEVNULL) as process:
        try:
            while True:
                check_cancelled()
                try:
                    return process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

def append_segment(target, segment, headers, first, last, total_size=None, allow_partial=False):
    if not segment.exists() or not headers.exists():
        return 0
    response = headers.read_text(errors='replace').lower()
    blocks = re.split(r'(?=^http/)', response, flags=re.MULTILINE)
    response = blocks[-1]
    status = re.match(r'http/\S+\s+(\d+)', response)
    ranges = re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)', response)
    encoding = re.search(r'^content-encoding:\s*([^\r\n]+)', response, re.MULTILINE)
    received = segment.stat().st_size
    if not status or status.group(1) != '206' or not ranges:
        return 0
    range_start, range_end, range_total = map(int, ranges[-1])
    # 严格校验：必须精确覆盖请求范围 [first, last]
    expected = last - first + 1
    if (range_start != first
            or range_end != last
            or not range_end < range_total
            or (total_size is not None and range_total != total_size)
            or not 0 < received <= expected
            or (not allow_partial and received != expected)
            or (encoding and encoding.group(1).strip() != 'identity')):
        return 0
    # A truncated identity response contains a prefix of this exact range.
    # Preserve only that prefix, then request its missing suffix.
    content_length = re.search(r'^content-length:\s*(\d+)\s*$', response, re.MULTILINE)
    if content_length and int(content_length.group(1)) != expected:
        return 0
    with target.open('ab') as out, segment.open('rb') as source:
        shutil.copyfileobj(source, out)
    return received
_cdn_cache = {}
_cdn_seeded = set()
_cdn_cache_lock = threading.Lock()
def cdn_url_from_headers(headers):
    try:
        text = headers.read_text(errors='replace')
    except OSError:
        return None
    statuses = re.findall(r'^HTTP/\S+\s+(\d+)', text, re.I | re.M)
    if not statuses or statuses[-1] != '206':
        return None
    locations = re.findall(r'^location:\s*(https://[^\r\n]+)', text, re.I | re.M)
    for value in reversed(locations):
        parsed = urllib.parse.urlparse(value.strip())
        if (parsed.scheme != 'https' or not parsed.hostname
                or not parsed.hostname.endswith('.hf.co')
                or parsed.username is not None or parsed.password is not None):
            continue
        query = urllib.parse.parse_qs(parsed.query)
        try:
            expires = float(query.get('Expires', query.get('expires', ['0']))[0])
        except (TypeError, ValueError, IndexError):
            continue
        if expires > time.time() + 300:
            return value.strip(), expires
    return None
def cached_download_url(url, folder):
    with _cdn_cache_lock:
        cached = _cdn_cache.get(url)
        if cached and cached[1] > time.time() + 300:
            return cached[0]
        _cdn_cache.pop(url, None)
        if url not in _cdn_seeded:
            _cdn_seeded.add(url)
            # This folder belongs to exactly one source volume. Stale or failed
            # responses cannot seed a URL; expiry and final status are checked.
            entries = []
            for headers in folder.glob('*.headers'):
                try:
                    entries.append((headers.stat().st_mtime, headers))
                except FileNotFoundError:
                    pass
            for _, headers in sorted(entries, reverse=True):
                candidate = cdn_url_from_headers(headers)
                if candidate:
                    _cdn_cache[url] = candidate
                    return candidate[0]
    return None
def remember_download_url(url, headers):
    candidate = cdn_url_from_headers(headers)
    if candidate:
        with _cdn_cache_lock:
            _cdn_cache[url] = candidate
def invalidate_download_url(url):
    with _cdn_cache_lock:
        _cdn_cache.pop(url, None)
        _cdn_seeded.add(url)
def fetch_part(url, target, start, end, total_size=None):
    check_cancelled()
    size = end - start + 1
    if target.exists() and target.stat().st_size == size:
        return
    failures = 0
    while failures < 40:
        check_cancelled()
        saved = target.stat().st_size if target.exists() else 0
        if saved == size:
            return
        if saved > size:
            raise RuntimeError('Oversized download chunk: ' + target.name)
        first = start + saved
        segment_bytes = int(os.environ.get('MVLM_DOWNLOAD_SEGMENT_MIB', '16')) * 1048576
        last = min(first + segment_bytes - 1, end)
        # Each request owns its files. A curl orphaned by an older launcher must
        # never share an inode with a new request, even after SIGKILL/restart.
        request_id = '%s.%s' % (os.getpid(), uuid.uuid4().hex)
        segment = target.with_name(target.name + '.' + request_id + '.segment')
        headers = segment.with_suffix('.headers')
        original_url = url + '?download=true&range_start=' + str(first)
        cdn_url = cached_download_url(url, target.parent)
        download_url = cdn_url or original_url
        http_version = '--http2' if os.environ.get('MVLM_DOWNLOAD_HTTP2') == '1' else '--http1.1'
        command = ['curl', '-fLsS', http_version, '--connect-timeout', '30',
            '--max-time', '180', '--speed-limit', '1024', '--speed-time', '60',
            '--max-filesize', str(last - first + 1), '-H', 'Accept-Encoding: identity',
            '--range', '%s-%s' % (first, last), '-D', str(headers),
            download_url, '-o', str(segment)]
        # curl 18/28/56/92 denote incomplete transfer, timeout or transport failure.
        # Headers must still describe exactly the requested range and total.
        try:
            returncode = run_curl(command)
            check_cancelled()
            remember_download_url(url, headers)
            if cdn_url and returncode == 22:
                invalidate_download_url(url)
            accepted = 0
            if returncode in (0, 18, 28, 56, 92):
                accepted = append_segment(target, segment, headers, first, last, total_size,
                                          allow_partial=returncode != 0)
        finally:
            segment.unlink(missing_ok=True)
            headers.unlink(missing_ok=True)
        if accepted:
            failures = 0
        else:
            failures += 1
            CANCELLED.wait(min(30, 2 ** min(failures, 5)))
    raise RuntimeError('Download failed after retries: ' + target.name)
def download(repo, row):
    # Never unlink a lock file: replacing its inode defeats flock protection.
    lockfile = DOWNLOADS / (Path(row['path']).name + '.download.lock')
    with lockfile.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return download_locked(repo, row)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            check_cancelled()
            digest.update(block)
    return digest.hexdigest()

def repair_chunk(url, chunks, index, size):
    """Replace a known bad 256 MiB chunk using parallel, independent ranges."""
    folder = chunks / ('.repair-%06d' % index)
    folder.mkdir(exist_ok=True)
    start = index * CHUNK
    stop = min(start + CHUNK, size)
    step = 4 * 1024**2
    jobs = [(url, folder / ('%06d' % i), pos, min(pos + step, stop) - 1, size)
            for i, pos in enumerate(range(start, stop, step))]
    with cf.ThreadPoolExecutor(max_workers=int(os.environ.get('MVLM_DOWNLOAD_WORKERS', '4'))) as pool:
        for future in cf.as_completed([pool.submit(fetch_part, *job) for job in jobs]):
            future.result()
    replacement = folder / 'replacement'
    with replacement.open('wb') as out:
        for _, part, first, last, _ in jobs:
            if part.stat().st_size != last - first + 1:
                raise RuntimeError('Incorrect repair range size')
            with part.open('rb') as src:
                shutil.copyfileobj(src, out)
    if replacement.stat().st_size != stop - start:
        raise RuntimeError('Incorrect repaired chunk size')
    target = chunks / ('%06d' % index)
    if target.exists():
        target.replace(folder / 'original.corrupt')
    replacement.replace(target)
    print('Replaced corrupt chunk', index, 'of', chunks.name, flush=True)

def download_locked(repo, row):
    target = DOWNLOADS / row['path']
    size = row['size']
    expected_hash = row.get('lfs', {}).get('oid')
    if target.exists() and target.stat().st_size == size:
        marker = target.with_name(target.name + '.verified.json')
        stat = target.stat()
        wanted = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=expected_hash)
        if marker.exists() and json.loads(marker.read_text()) == wanted:
            return target
        if not expected_hash or sha256_file(target) == expected_hash:
            return target
        target.replace(target.with_name(target.name + '.corrupt-' + uuid.uuid4().hex[:8]))
        marker.unlink(missing_ok=True)
    chunks = DOWNLOADS / (target.name + '.parts')
    chunks.mkdir(exist_ok=True)
    hf_base = os.environ.get('MVLM_HF_BASE', 'https://huggingface.co')
    revision = row.get('revision', 'main')
    url = hf_base + '/datasets/%s/resolve/%s/%s' % (repo, revision, row['path'])
    planfile = DOWNLOADS / (target.name + '.repair.json')
    if planfile.exists():
        plan = json.loads(planfile.read_text())
        if plan['sha256'] != expected_hash:
            raise RuntimeError('Repair plan source hash mismatch')
        for index in plan.get('pending_chunks', [])[:]:
            if not 0 <= index < (size + CHUNK - 1) // CHUNK:
                raise RuntimeError('Invalid repair chunk index')
            progress(row, 'repairing', repair_chunk=index)
            repair_chunk(url, chunks, index, size)
            plan['pending_chunks'].remove(index)
            plan.setdefault('repaired_chunks', []).append(index)
            temp = planfile.with_suffix('.tmp')
            temp.write_text(json.dumps(plan, indent=2))
            temp.replace(planfile)
    jobs = [(url, chunks / ('%06d' % i), start, min(start + CHUNK, size) - 1, size)
            for i, start in enumerate(range(0, size, CHUNK))]
    partial = target.with_suffix(target.suffix + '.assembling')
    for attempt in range(2):
        progress(row, 'downloading')
        with cf.ThreadPoolExecutor(max_workers=int(os.environ.get("MVLM_DOWNLOAD_WORKERS", "4"))) as pool:
            futures = [pool.submit(fetch_part, *job) for job in jobs]
            for future in cf.as_completed(futures):
                future.result()
        progress(row, 'verifying')
        digest = hashlib.sha256()
        with partial.open('wb') as out:
            for _, part, first, last, _ in jobs:
                check_cancelled()
                if part.stat().st_size != last - first + 1:
                    raise RuntimeError('Incorrect chunk size before assembly')
                with part.open('rb') as src:
                    for block in iter(lambda: src.read(8 * 1024**2), b''):
                        check_cancelled()
                        digest.update(block)
                        out.write(block)
        if partial.stat().st_size != size:
            raise RuntimeError('Incorrect assembled size')
        actual_hash = digest.hexdigest()
        if not expected_hash or actual_hash == expected_hash:
            break
        report = dict(archive=target.name, expected_sha256=expected_hash,
                      actual_sha256=actual_hash, size=size, timestamp=time.time())
        tag = uuid.uuid4().hex[:8]
        quarantine = DOWNLOADS / (target.name + '.corrupt-' + tag)
        quarantine.mkdir()
        (quarantine / 'failure.json').write_text(json.dumps(report, indent=2))
        chunks.replace(quarantine / 'chunks')
        partial.replace(quarantine / 'assembled')
        invalidate_download_url(url)
        if attempt:
            raise RuntimeError('SHA256 mismatch after clean retry: ' + target.name +
                               '; evidence: ' + str(quarantine))
        print('SHA256 mismatch; quarantined suspect data; downloading fresh:', target.name,
              'expected', expected_hash, 'actual', actual_hash, flush=True)
        chunks.mkdir()
    partial.replace(target)
    stat = target.stat()
    target.with_name(target.name + '.verified.json').write_text(json.dumps(dict(
        size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=actual_hash)))
    # Keep original archives; discard only our redundant completed chunks.
    for _, part, _, _, _ in jobs:
        part.unlink()
        for suffix in ('.segment', '.headers'):
            auxiliary = part.with_suffix(suffix)
            if auxiliary.exists():
                auxiliary.unlink()
    # Includes old request files and the targeted repair evidence. Preserve
    # evidence, but remove temporary responses only after whole-file SHA passes.
    for auxiliary in chunks.iterdir():
        if auxiliary.is_file() and auxiliary.suffix in ('.segment', '.headers'):
            auxiliary.unlink()
    if not any(chunks.iterdir()):
        chunks.rmdir()
    return target

DOWNLOADS = Path(__file__).resolve().parent / "official_training"
