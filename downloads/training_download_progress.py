#!/usr/bin/env python3
"""Read-only progress display for LaSOT and TNL2K training data."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import time
from prepare_official_training import source_rows

ROOT = Path(__file__).resolve().parent / 'official_training'
LABELS = {'downloading': '下载中', 'verifying': 'SHA256校验中',
          'repairing': '损坏分块重新下载中',
          'extracting': '解压中', 'validating': '数据完整性检查中',
          'ready': '可用于训练', 'failed': '失败', 'paused': '已暂停'}


def byte_counts(rows):
    """Count accepted prefixes plus nonoverlapping in-flight response bytes."""
    saved = inflight = 0
    for row in rows:
        target = ROOT / row['path']
        try:
            if target.is_file() and target.stat().st_size == row['size']:
                saved += row['size']
                continue
            folder = ROOT / (row['path'] + '.parts')
            for index, start in enumerate(range(0, row['size'], 256 * 1024**2)):
                chunk_size = min(256 * 1024**2, row['size'] - start)
                part = folder / ('%06d' % index)
                accepted = min(chunk_size, part.stat().st_size) if part.exists() else 0
                saved += accepted
                in_chunk = 0
                for headers in folder.glob(part.name + '.*headers'):
                    segment = headers.with_suffix('.segment')
                    if not segment.exists():
                        continue
                    response = re.split(r'(?=^http/)', headers.read_text(errors='replace').lower(), flags=re.M)[-1]
                    match = re.search(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)', response)
                    if not re.match(r'http/\S+\s+206\b', response) or not match:
                        continue
                    first, last, total = map(int, match.groups())
                    received = segment.stat().st_size
                    if (total == row['size'] and start <= first <= start + accepted
                            and first <= last < start + chunk_size and 0 < received <= last - first + 1):
                        in_chunk = max(in_chunk, max(0, first + received - (start + accepted)))
                inflight += in_chunk
        except FileNotFoundError:
            # Chunk files can disappear while a completed archive is published.
            continue
    return saved, inflight


class SpeedTracker:
    def __init__(self):
        self.previous = {}

    def sample(self, name, count, active):
        now = time.monotonic()
        previous = self.previous.get(name)
        # Keep a high-water mark across transient file-renaming races.
        count = max(count, previous[1]) if previous else count
        self.previous[name] = (now, count)
        if not active:
            return 0.0
        if previous is None:
            return None
        return (count - previous[1]) / max(now - previous[0], 0.001)


def human(value):
    return '%.2f GiB' % (value / 1024**3)


def duration(seconds):
    seconds = int(max(0, seconds))
    return '%d小时%02d分%02d秒' % (seconds // 3600, seconds % 3600 // 60, seconds % 60)


def show(tracker):
    path = ROOT / 'status.json'
    if not path.exists():
        print('尚无下载状态文件：' + str(path))
        return
    state = json.loads(path.read_text())
    runtime = state.get('_runtime', {})
    pid = runtime.get('pid')
    active = False
    if pid:
        try:
            cmd = Path('/proc/%s/cmdline' % pid).read_bytes()
            active = b'prepare_official_training.py' in cmd
        except FileNotFoundError:
            pass
    print('训练数据下载进度  ' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S 北京时间'))
    print('进程：%s%s；此任务不会启动模型训练' % ('运行中' if active else '未运行', ('，PID %s' % pid) if active else ''))
    if runtime:
        print('连接：每套数据%s路并发，%s MiB分段，HTTP/%s' % (
            runtime.get('workers_per_dataset', '?'), runtime.get('segment_mib', '?'),
            '2' if runtime.get('http2') else '1.1'))
    total = done = 0
    total_speed = 0.0
    sampling = False
    for name, label in [('lasot', 'LaSOT'), ('tnl2k', 'TNL2K')]:
        entry = state.get(name, {})
        size = entry.get('total_bytes', 0)
        rows = source_rows(name)
        saved, inflight = byte_counts(rows)
        stage = entry.get('status', 'unknown')
        speed = tracker.sample(name, saved + inflight, active and stage == 'downloading')
        if speed is None:
            sampling = True
        else:
            total_speed += speed
        total += size
        done += saved
        print('\n%s：%s%s' % (label, LABELS.get(stage, stage),
              '（进程已退出，状态可能未更新）' if not active and stage not in ('ready', 'failed', 'paused') else ''))
        print('  已落盘 %s / %s（%.2f%%）' % (human(saved), human(size), saved / size * 100 if size else 0))
        print('  当前文件：%s' % entry.get('current_archive', '尚未开始'))
        print('  在途临时数据：%.2f MiB（尚未验收，未计入百分比）' % (inflight / 1024**2))
        speed_text = '采样中，下次刷新显示' if speed is None else '%.2f MiB/s（%.2f MB/s）' % (speed / 1024**2, speed / 10**6)
        print('  当前下载速度：' + speed_text)
        print('  SHA256通过：%s/%s个压缩包' % (entry.get('verified_archives', 0), entry.get('archives_total', '?')))
        if stage == 'downloading' and speed is not None and speed > 0:
            print('  按当前速度估计剩余下载：%s（不含解压和校验）' % duration((size - saved) / speed))
        if entry.get('extracted_files') is not None:
            print('  已扫描提取文件：%s' % entry['extracted_files'])
        if entry.get('validated_sequences') is not None:
            print('  已检查序列：%s/%s；帧数：%s' % (entry['validated_sequences'],
                  entry.get('expected_sequences', '?'), entry.get('validated_frames', '?')))
        if entry.get('error'):
            print('  错误：' + entry['error'])
    print('\n合计：%s / %s（%.2f%%）' % (human(done), human(total), done / total * 100 if total else 0))
    print('当前总下载速度：' + ('采样中，下次刷新显示' if sampling else
          '%.2f MiB/s（%.2f MB/s）' % (total_speed / 1024**2, total_speed / 10**6)))
    print('速度按两次采样间新增接收字节计算，包含在途数据。')
    print('说明：进度统计已接受的落盘分块；在途数据未计入。100%下载后仍需校验、解压，状态ready才表示准备完成。')
    print('日志：' + str(ROOT / 'prepare.log'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true', help='持续刷新，Ctrl+C退出；不会停止下载')
    parser.add_argument('--interval', type=float, default=3, help='刷新间隔秒数，默认3秒')
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval必须至少为1秒')
    try:
        tracker = SpeedTracker()
        if not args.watch:
            for name in ('lasot', 'tnl2k'):
                accepted, inflight = byte_counts(source_rows(name))
                tracker.sample(name, accepted + inflight, True)
            time.sleep(1)
        while True:
            if args.watch:
                print('\033[2J\033[H', end='', flush=True)
            show(tracker)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n退出进度查看，下载任务继续运行。')
