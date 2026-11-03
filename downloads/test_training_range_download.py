"""Fault-injection checks for range acceptance, restart and checksum recovery."""
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('downloader', Path(__file__).with_name('training_range_download.py'))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class DownloadChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        d.DOWNLOADS = Path(self.temp.name)
        d.CHUNK = 64 * 1024
        d.CANCELLED.clear()
        d._cdn_cache.clear()
        d._cdn_seeded.clear()
        self.payload = os.urandom(200_000)
        self.row = dict(path='sample.tar.gz.part.aa', size=len(self.payload),
                        revision='pinned-commit', lfs=dict(oid=hashlib.sha256(self.payload).hexdigest()))
        self.requests = []
        self.truncate_once = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                owner.requests.append(self.path)
                first, last = map(int, self.headers['Range'][6:].split('-'))
                body = owner.payload[first:last + 1]
                self.send_response(206)
                self.send_header('Content-Range', 'bytes %d-%d/%d' % (first, last, len(owner.payload)))
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                if owner.truncate_once:
                    owner.truncate_once = False
                    body = body[:len(body) // 2]
                    self.close_connection = True
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.addCleanup(self.server.server_close)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        self.env = patch.dict(os.environ, MVLM_HF_BASE='http://127.0.0.1:%d' % self.server.server_port,
                              MVLM_DOWNLOAD_WORKERS='4', MVLM_DOWNLOAD_SEGMENT_MIB='1',
                              MVLM_DOWNLOAD_HTTP2='0', NO_PROXY='127.0.0.1', no_proxy='127.0.0.1')
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_truncated_response_resumes_byte_exactly(self):
        self.truncate_once = True
        result = d.download('fixture/data', self.row)
        self.assertEqual(result.read_bytes(), self.payload)
        self.assertTrue(all('/resolve/pinned-commit/' in url for url in self.requests))

    def test_wrong_existing_chunks_are_quarantined_and_redownloaded(self):
        chunks = d.DOWNLOADS / (self.row['path'] + '.parts')
        chunks.mkdir()
        for i, start in enumerate(range(0, len(self.payload), d.CHUNK)):
            (chunks / ('%06d' % i)).write_bytes(b'x' * min(d.CHUNK, len(self.payload) - start))
        result = d.download('fixture/data', self.row)
        self.assertEqual(result.read_bytes(), self.payload)
        evidence = list(d.DOWNLOADS.glob('*.corrupt-*/failure.json'))
        self.assertEqual(len(evidence), 1)
        self.assertNotEqual(json.loads(evidence[0].read_text())['actual_sha256'], self.row['lfs']['oid'])

    def test_targeted_repair_preserves_good_chunks(self):
        chunks = d.DOWNLOADS / (self.row['path'] + '.parts')
        chunks.mkdir()
        for i, start in enumerate(range(0, len(self.payload), d.CHUNK)):
            (chunks / ('%06d' % i)).write_bytes(self.payload[start:start + d.CHUNK])
        (chunks / '000001').write_bytes(b'x' * d.CHUNK)
        plan = dict(sha256=self.row['lfs']['oid'], pending_chunks=[1])
        (d.DOWNLOADS / (self.row['path'] + '.repair.json')).write_text(json.dumps(plan))
        result = d.download('fixture/data', self.row)
        self.assertEqual(result.read_bytes(), self.payload)
        self.assertEqual(len(self.requests), 1)
        self.assertTrue((chunks / '.repair-000001/original.corrupt').exists())

    def test_full_size_corrupt_archive_is_not_trusted(self):
        (d.DOWNLOADS / self.row['path']).write_bytes(b'x' * len(self.payload))
        result = d.download('fixture/data', self.row)
        self.assertEqual(result.read_bytes(), self.payload)

    def test_repeated_source_corruption_stops_without_publishing(self):
        # The source keeps returning correctly sized, wrong content. Neither
        # HTTP range validation nor a retry may bypass the expected SHA256.
        self.payload = b'x' * len(self.payload)
        with self.assertRaisesRegex(RuntimeError, 'mismatch after clean retry'):
            d.download('fixture/data', self.row)
        self.assertFalse((d.DOWNLOADS / self.row['path']).exists())
        self.assertEqual(len(list(d.DOWNLOADS.glob('*.corrupt-*/failure.json'))), 2)

    def test_archive_lock_rejects_second_writer(self):
        with (d.DOWNLOADS / (self.row['path'] + '.download.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.assertRaises(BlockingIOError):
                d.download('fixture/data', self.row)

    def test_incorrect_response_range_is_rejected(self):
        target, segment, headers = (d.DOWNLOADS / s for s in ('chunk', 'body', 'headers'))
        segment.write_bytes(b'abcd')
        headers.write_text('HTTP/1.1 206 Partial Content\nContent-Range: bytes 1-4/10\nContent-Length: 4\n\n')
        self.assertEqual(d.append_segment(target, segment, headers, 0, 3, 10), 0)
        self.assertFalse(target.exists())

    def test_stale_response_from_old_process_is_not_reused(self):
        chunks = d.DOWNLOADS / (self.row['path'] + '.parts')
        chunks.mkdir()
        (chunks / '000000.segment').write_bytes(b'x' * d.CHUNK)
        (chunks / '000000.headers').write_text(
            'HTTP/1.1 206 Partial Content\nContent-Range: bytes 0-%d/%d\n\n' %
            (d.CHUNK - 1, len(self.payload)))
        self.assertEqual(d.download('fixture/data', self.row).read_bytes(), self.payload)
        self.assertFalse(list(d.DOWNLOADS.glob('*.corrupt-*')))

    def test_cancel_reaps_subprocess(self):
        processes = []
        original = subprocess.Popen

        def tracked(*args, **kwargs):
            child = original(*args, **kwargs)
            processes.append(child)
            return child

        timer = threading.Timer(0.2, d.CANCELLED.set)
        timer.start()
        try:
            with patch.object(d.subprocess, 'Popen', tracked):
                with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                    d.run_curl(['python', '-c', 'import time; time.sleep(30)'])
            self.assertIsNotNone(processes[0].poll())
        finally:
            timer.cancel()
            d.CANCELLED.clear()


if __name__ == '__main__':
    unittest.main()
