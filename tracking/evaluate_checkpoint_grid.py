"""Resumable all-epoch evaluation using the historical main-branch protocol."""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('MVLM_WORKSPACE', '/root/MVLM'))
sys.path.insert(0, str(SOURCE_ROOT))
DATASETS = ('otb99_lang', 'lasot', 'tnl2k')
EXPECTED = dict(otb99_lang=48, lasot=280, tnl2k=700)
CONFIGS = dict(otb99_lang='otb_tune_03_prod04_c', lasot='lasot_tune_05_mem004', tnl2k='tnl2k_tune_05_psi012')
TRAIN = ROOT / 'output/paper_reproduction_20261030_run01'
CKPTS = TRAIN / 'checkpoints/train/mvlm/mvlm_TF_paper_reproduction'
OUT = ROOT / 'output/paper_checkpoint_grid_20261101_run01'
REPORT = ROOT / 'reports/reproduction/PAPER_CHECKPOINT_TESTS_20261101.md'


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def gpu_stats():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    return {int(r.split(',')[0]): {'memory_mib': int(r.split(',')[1]), 'utilization': int(r.split(',')[2])} for r in raw.strip().splitlines()}


def job(args):
    import numpy as np
    import torch
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.tracker import Tracker
    from lib.test.evaluation.running import run_dataset
    from lib.test.analysis.extract_results import calc_seq_err_robust
    import cv2
    cv2.setNumThreads(1)
    # Keep the existing main protocol and model loader; preflight checks every
    # non-frozen model key before starting the multiprocessing workers.
    folder = Path(args.job_out)
    folder.mkdir(parents=True, exist_ok=True)
    status = dict(status='starting', epoch=args.epoch, dataset=args.dataset, started_at=now(), workers=args.workers,
                  weight=str(CKPTS / f'MVLM_ep{args.epoch:04d}.pth.tar'), config=args.config,
                  exp_id=args.exp_id, probe=args.probe)
    write_json(folder/'status.json', status)
    try:
        manifest=json.loads((OUT/'configuration_manifest.json').read_text())
        config_path=ROOT/'experiments/mvlm'/f'{args.config}.yaml'
        assert sha(config_path)==manifest[args.dataset]['sha256'], 'Evaluation config changed'
        dataset = get_dataset(args.dataset)
        assert len(dataset) == EXPECTED[args.dataset], len(dataset)
        if args.probe:
            dataset = dataset[:24]
            for seq in dataset:
                seq.frames = seq.frames[:300]
                seq.ground_truth_rect = seq.ground_truth_rect[:300]
                if seq.target_visible is not None:
                    seq.target_visible = seq.target_visible[:300]
        tracker = Tracker('mvlm', args.config, args.exp_id, args.dataset, status['weight'])
        weight = torch.load(status['weight'], map_location='cpu')
        assert weight['epoch'] == args.epoch
        # Loading a real model once catches silent strict=False mismatches.
        params = tracker.get_parameters()
        params.debug = False
        model = tracker.create_tracker(params)
        expected = model.network.state_dict()
        missing = [k for k in expected if k not in weight['net'] and not k.startswith('text_encoder.clip.')]
        unexpected = [k for k in weight['net'] if k not in expected]
        assert not missing and not unexpected, (missing, unexpected)
        assert model.template_free and model.multi_modal_language and model.use_nlp
        write_json(folder/'loading_audit.json', dict(missing_trainable_keys=missing, unexpected_keys=unexpected, epoch=weight['epoch'], checkpoint_sha256=sha(status['weight'])))
        del model, expected, weight
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        status.update(status='running', sequences=len(dataset), results_dir=tracker.results_dir)
        write_json(folder/'status.json',status)
        for seq in dataset:
            base=Path(tracker.results_dir)/seq.dataset/seq.name
            boxes=Path(str(base)+'.txt');times=Path(str(base)+'_time.txt')
            if not boxes.exists():continue
            try:
                b=np.loadtxt(boxes,ndmin=2);t=np.loadtxt(times,ndmin=1)
                assert b.shape==(len(seq.frames),4) and t.shape==(len(seq.frames),)
                assert np.isfinite(b).all() and (b[:,2:]>0).all() and np.isfinite(t).all() and (t>=0).all()
            except Exception:
                archive=folder/'interrupted_predictions';archive.mkdir(exist_ok=True)
                for path in (boxes,times):
                    if path.exists():path.replace(archive/(str(time.time_ns())+'_'+path.name))
        start = time.monotonic()
        run_dataset(dataset, [tracker], debug=False, threads=args.workers, num_gpus=2, parallel_mode='mp')
        elapsed = time.monotonic()-start
        errors, rows = [], []
        thresholds = torch.arange(0,1.01,.05,dtype=torch.float64)
        for seq in dataset:
            try:
                base = Path(tracker.results_dir)/seq.dataset/seq.name
                boxes = np.loadtxt(str(base)+'.txt',ndmin=2)
                timings = np.loadtxt(str(base)+'_time.txt',ndmin=1)
                assert boxes.shape == (len(seq.frames),4), boxes.shape
                assert timings.shape == (len(seq.frames),), timings.shape
                assert np.isfinite(boxes).all() and (boxes[:,2:]>0).all()
                assert np.isfinite(timings).all() and (timings>=0).all()
                pred = torch.as_tensor(boxes, dtype=torch.float64)
                anno = torch.as_tensor(seq.ground_truth_rect, dtype=torch.float64)
                visible = torch.as_tensor(seq.target_visible) if seq.target_visible is not None else None
                overlap, center, norm, valid = calc_seq_err_robust(pred,anno,seq.dataset,visible)
                rows.append(dict(sequence=seq.name, frames=len(seq.frames),
                                 AUC=float((overlap[:,None]>thresholds).double().mean()*100),
                                 PRE=float((center<=20).double().mean()*100),
                                 NPR=float((norm<=.2).double().mean()*100)))
            except Exception as exc:
                errors.append(dict(sequence=seq.name,error=repr(exc)))
        write_json(folder/'prediction_validation.json',dict(expected=len(dataset),validated=len(rows),errors=errors))
        if errors:
            raise RuntimeError(str(errors[:5]))
        write_json(folder/'per_sequence.json',rows)
        metrics = {k:float(np.mean([r[k] for r in rows])) for k in ('AUC','PRE','NPR')}
        status.update(status='completed',completed_at=now(),metrics=metrics,wall_seconds=elapsed,
                      frames=sum(r['frames'] for r in rows),aggregate_fps=sum(r['frames'] for r in rows)/elapsed)
    except BaseException as exc:
        status.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),completed_at=now())
        raise
    finally:
        write_json(folder/'status.json',status)


def render(state):
    rows=[]
    for epoch in range(1,61):
        for ds in DATASETS:
            p=OUT/f'epoch_{epoch:04d}'/ds/'status.json'
            if p.exists():
                row=json.loads(p.read_text());rows.append(row)
    planned=state.get('planned_epochs',list(range(5,61,5)))
    completed=[r for r in rows if r['status']=='completed' and r['epoch'] in planned]
    state['historical_completed_jobs']=sum(r['status']=='completed' and r['epoch'] not in planned for r in rows)
    state['completed_jobs']=len(completed)
    state['updated_at']=now()
    if state.get('current_exp_id'):
        folder=ROOT/'output/test/tracking_results/mvlm'/state['current_exp_id']/state.get('current_dataset','')
        state['current_sequences_written']=len([p for p in folder.glob('*.txt') if not p.name.endswith('_time.txt')])
    write_json(OUT/'status.json',state)
    csvpath=OUT/'results.csv'
    with csvpath.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','dataset','status','AUC','PRE','NPR','wall_seconds','aggregate_fps'])
        writer.writeheader()
        for r in rows:
            writer.writerow({k:r.get('metrics',{}).get(k,r.get(k,'')) for k in writer.fieldnames})
    selection_path=OUT/'resource_selection.json'
    selection=json.loads(selection_path.read_text()) if selection_path.exists() else {}
    lines=['# 检查点粗测与局部细测：main 首帧真实框协议评测', '', f'更新时间：{now()}。运行状态：{state["status"]}；当前计划完成 {len(completed)}/{state.get("total_jobs",36)} 组。', '',
           '运行目录：`'+str(OUT)+'`。仅使用物理 GPU 4、5、6、7，分成4/5与6/7两组并行。每个进程 batch=1；通过并行视频提高吞吐量。', '',
           '**协议说明**：保留 template-free 模型架构，采用 main 首帧真实框初始化和首帧 GT 计分，并保留语言输入；不等于重新训练或启用视觉模板模型。', '',
           '使用 main 已归档历史最优参数。它们来自测试集搜索；根据这些测试成绩选择高分轮次及邻域属于探索性模型选择，最高分不能当作独立最终测试成绩。第60轮仍保留为预先指定的最终模型参考。', '',
           '| 数据集 | 序列数 | α | τ | ω | λ | ψ | main 官方权重历史 AUC |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
           '| OTB99-Lang | 48 | .5 | .03 | .4 | .1 | .1 | 60.56 |',
           '| LaSOT | 280 | .5 | .03 | .1 | .4 | .1 | 61.06 |',
           '| TNL2K | 700 | .5 | .03 | .1 | .4 | .12 | 57.62 |', '',
           'TNL2K 保留 main 的700条口径，包括空语言序列 Fight_video_6-Done。AUC为21个阈值的成功率均值，PRE@20px，NPR@0.2，按序列等权；可见性和无效帧处理保留原 main 规则。', '',
           f'并发设置：总计 {state.get("workers", "待短测确定")} 个视频进程。当前任务：{state.get("current_job", "准备中")}；已写入 {state.get("current_sequences_written", 0)} 条视频预测。', '',
           '策略：先粗测第5、10、15、20、25、30、35、40、45、50、55、60轮，每轮完整测试三个集合，共36组。粗测后按三数据集平均AUC最高的轮次选中心，在中心前后各4轮内以步长1细测；已测过的轮次不重复。最多补8个轮次，共不超过60组。分数相同优先较晚轮次。', '',
           '当前阶段：'+state.get('phase','coarse')+'；细测轮次：'+str(state.get('fine_epochs','粗测完成后确定'))+'。此前第1轮的完整或部分预测保留为历史记录，不计入粗测进度。', '',
           '| Epoch | OTB AUC / PRE / NPR | LaSOT AUC / PRE / NPR | TNL2K AUC / PRE / NPR |',
           '| ---: | --- | --- | --- |',
           '| **论文（表1）** | **60.70 / 84.30 / 75.40** | **60.70 / 65.50 / 68.30** | **57.80 / — / 74.10** |']
    lookup={(r['epoch'],r['dataset']):r for r in rows}
    for ep in range(1,61):
        cells=[]
        for ds in DATASETS:
            r=lookup.get((ep,ds),{})
            if r.get('status')=='completed':
                m=r['metrics'];cells.append(' / '.join(f'{m[k]:.2f}' for k in ('AUC','PRE','NPR')))
            else: cells.append(r.get('status','排队' if ep in planned else '未安排'))
        lines.append(f'| {ep} | '+ ' | '.join(cells)+' |')
    if state.get('active_jobs'):
        lines.extend(['', '| GPU组 | 当前任务 | 已写入视频 | 进程数 |', '| --- | --- | ---: | ---: |'])
        for active_job in state['active_jobs']:
            group='/'.join(map(str,active_job['physical_gpus']))
            lines.append(f"| {group} | Epoch {active_job['epoch']} / {active_job['dataset']} | {active_job['sequences_written']}/{active_job['expected_sequences']} | {active_job['workers']} |")
    if selection:
        lines.extend(['', '资源短测记录：`'+str(selection_path)+'`。每组短测使用同一批视频比较总吞吐量；短测成绩不替代完整测试成绩。', '', '原双卡组短测所选并发：'+str(selection.get('selected_workers'))+' 个进程（每卡12个）；当前两个双卡组最多48个进程。短测原因：'+selection.get('reason','')])
    if state.get('error'): lines.extend(['', '错误：'+state['error']])
    tmp=REPORT.with_suffix('.tmp');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(REPORT)


def job_alive(pid, epoch, dataset):
    try:
        raw = Path(f'/proc/{pid}/cmdline').read_bytes()
        return (b'evaluate_checkpoint_grid.py\x00job' in raw
                and f'--epoch\x00{epoch}\x00'.encode() in raw
                and f'--dataset\x00{dataset}\x00'.encode() in raw)
    except FileNotFoundError:
        return False


def suite(args):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
    coarse = list(range(5, 61, 5))
    epochs = list(coarse)
    groups = ((4, 5), (6, 7))
    state = dict(status='running', phase='coarse', started_at=now(), pid=os.getpid(),
                 physical_gpus=[4,5,6,7], workers=args.workers*2, workers_per_group=args.workers,
                 total_jobs=36, coarse_epochs=coarse, planned_epochs=list(epochs))
    write_json(OUT/'adaptive_plan.json', dict(coarse_epochs=coarse, selection='highest equal-weight mean AUC over all three datasets; later epoch breaks ties', radius=4, fine_stride=1, max_total_jobs=60, gpu_groups=groups, workers_per_group=args.workers))
    active = {}
    failed = set()

    def row_for(epoch, ds):
        p = OUT/f'epoch_{epoch:04d}'/ds/'status.json'
        return json.loads(p.read_text()) if p.exists() else {}

    def completed(epoch, ds):
        return row_for(epoch, ds).get('status') == 'completed'

    def launch(epoch, ds, group, attempt=1):
        count = args.workers if attempt == 1 else max(2, args.workers//2)
        folder = OUT/f'epoch_{epoch:04d}'/ds
        folder.mkdir(parents=True, exist_ok=True)
        config = 'papergrid_20261101_'+ds
        exp = f'papergrid_20261101_e{epoch:04d}_{ds}'
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'job', '--epoch', str(epoch), '--dataset', ds,
               '--config', config, '--exp-id', exp, '--job-out', str(folder), '--workers', str(count)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, group)), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', OPENCV_FOR_THREADS_NUM='1', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
        write_json(folder/'command.json', dict(command=cmd, physical_gpus=group, attempt=attempt))
        with (folder/'test.log').open('a') as log:
            log.write(f'\nATTEMPT {attempt}, workers={count}, physical GPUs={group}\n'); log.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return dict(epoch=epoch, dataset=ds, group=group, pid=proc.pid, handle=proc,
                    attempt=attempt, workers=count, exp_id=exp, adopted=False)

    # Carry live video workers across a supervisor-only restart.
    prior = previous.get('active_jobs', [])
    if not prior and previous.get('worker_pid') and previous.get('current_job', '').startswith('epoch '):
        prior = [dict(epoch=int(previous['current_job'].split()[1]), dataset=previous['current_dataset'],
                      pid=previous['worker_pid'], physical_gpus=[6,7], workers=args.workers,
                      exp_id=previous['current_exp_id'], attempt=previous.get('attempt',1))]
    for r in prior:
        group = tuple(r['physical_gpus'])
        if group in groups and job_alive(r['pid'], r['epoch'], r['dataset']):
            active[group] = dict(epoch=r['epoch'], dataset=r['dataset'], group=group, pid=r['pid'], handle=None,
                                 attempt=r.get('attempt',1), workers=r.get('workers',args.workers), exp_id=r['exp_id'], adopted=True)
    try:
        while True:
            for group, job in list(active.items()):
                proc = job['handle']
                alive = proc.poll() is None if proc is not None else job_alive(job['pid'], job['epoch'], job['dataset'])
                if alive:
                    continue
                if completed(job['epoch'], job['dataset']):
                    del active[group]
                elif job['attempt'] < 2:
                    active[group] = launch(job['epoch'], job['dataset'], group, attempt=2)
                else:
                    failed.add((job['epoch'], job['dataset']))
                    state.setdefault('failed_jobs', []).append(dict(epoch=job['epoch'], dataset=job['dataset'], physical_gpus=group))
                    del active[group]
            busy = {(r['epoch'], r['dataset']) for r in active.values()}
            pending = [(ep,ds) for ep in epochs for ds in DATASETS
                       if not completed(ep,ds) and (ep,ds) not in busy and (ep,ds) not in failed]
            if not active and not pending:
                if not state.get('refinement_selected'):
                    ranking = []
                    for ep in coarse:
                        rows = [row_for(ep,ds) for ds in DATASETS]
                        if all(r.get('status') == 'completed' for r in rows):
                            ranking.append(dict(epoch=ep, mean_auc=sum(r['metrics']['AUC'] for r in rows)/3))
                    if not ranking:
                        raise RuntimeError('No complete coarse checkpoint available for refinement')
                    ranking.sort(key=lambda r:(r['mean_auc'],r['epoch']), reverse=True)
                    best = ranking[0]['epoch']
                    fine = [ep for ep in range(max(1,best-4),min(60,best+4)+1) if ep not in coarse]
                    epochs.extend(fine)
                    state.update(phase='fine', refinement_selected=True, best_coarse_epoch=best, coarse_ranking=ranking,
                                 fine_epochs=fine, planned_epochs=list(epochs), total_jobs=len(epochs)*3)
                    write_json(OUT/'refinement_selection.json', dict(selected_at=now(), ranking=ranking, center_epoch=best,
                               fine_epochs=fine, scope='test-guided exploratory checkpoint selection'))
                    render(state)
                    continue
                state.update(status='completed_with_errors' if failed else 'completed', completed_at=now(), active_jobs=[])
                render(state)
                break
            for group in groups:
                if group in active or not pending:
                    continue
                ep, ds = pending.pop(0)
                weight = CKPTS/f'MVLM_ep{ep:04d}.pth.tar'
                if not weight.exists():
                    ts = json.loads((TRAIN/'training_status.json').read_text())
                    if ts['status'] in ('failed','completed','stopped'):
                        raise RuntimeError(f'Missing checkpoint {weight}; training={ts["status"]}')
                    continue
                active[group] = launch(ep, ds, group)
            summaries = []
            for group, job in active.items():
                folder = ROOT/'output/test/tracking_results/mvlm'/job['exp_id']/job['dataset']
                count = sum(not p.name.endswith('_time.txt') for p in folder.glob('*.txt'))
                summaries.append(dict(epoch=job['epoch'], dataset=job['dataset'], pid=job['pid'], physical_gpus=list(group),
                                      workers=job['workers'], exp_id=job['exp_id'], attempt=job['attempt'],
                                      adopted=job['adopted'], sequences_written=count, expected_sequences=EXPECTED[job['dataset']]))
            state['active_jobs'] = summaries
            if summaries:
                first = summaries[0]
                state.update(current_job=f"epoch {first['epoch']} / {first['dataset']}", current_exp_id=first['exp_id'],
                             current_dataset=first['dataset'], worker_pid=first['pid'])
            state['gpu'] = gpu_stats()
            render(state)
            time.sleep(15)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc)); render(state); raise

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['suite','job'])
    p.add_argument('--workers',type=int,default=8);p.add_argument('--epoch',type=int,default=59)
    p.add_argument('--dataset',default='otb99_lang');p.add_argument('--config',default='papergrid_20261101_otb99_lang')
    p.add_argument('--exp-id',default='');p.add_argument('--job-out',default='');p.add_argument('--probe',action='store_true')
    args=p.parse_args()
    if args.mode=='suite':suite(args)
    else:job(args)

if __name__=='__main__':main()
