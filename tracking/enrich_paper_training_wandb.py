"""Enrich the completed report from archived online and local W&B records."""
import csv
import datetime as dt
import json
from pathlib import Path
import re

ROOT=Path('/root/MVLM')
ART=ROOT/'reports/reproduction/paper_training_results_20261101'
REPORT=ROOT/'reports/reproduction/PAPER_TRAINING_RESULTS_20261101.md'


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    online=json.loads((ART/'wandb_run.json').read_text())
    remote=json.loads((ART/'wandb_history.json').read_text())
    history=json.loads((ART/'wandb_local_history.json').read_text())
    system=json.loads((ART/'wandb_local_system_history.json').read_text())
    records=json.loads((ART/'wandb_local_records.json').read_text())
    base=json.loads((ART/'summary.json').read_text())
    zone=dt.timezone(dt.timedelta(hours=8))
    epochrows={int(r['global_step'])//1250:r for r in history if r.get('epoch/train/Loss/Total') is not None}
    assert sorted(epochrows)==list(range(1,61)) and records['exit_code']==0
    logrows=list(csv.DictReader((ART/'epoch_metrics_log_rounded.csv').open()))
    keys={'total_loss':'epoch/train/Loss/Total','tracking_loss':'epoch/train/Loss/Tracking','weighted_cm_loss':'epoch/train/Loss/CM','train_iou':'epoch/train/IoU'}
    diffs={k:max(abs(float(r[k])-epochrows[int(r['epoch'])][v]) for r in logrows) for k,v in keys.items()}
    assert max(diffs.values())<5.1e-6,diffs
    runtime=[r for r in history if r.get('runtime/train/seconds_per_batch_rank0') is not None]
    batches=[r for r in history if r.get('train/Loss/Total') is not None]
    assert len(runtime)==len(batches)==3780
    assert all(np.isfinite(r[k]) for r in history for k in ('train/Loss/Total','epoch/train/Loss/Total') if r.get(k) is not None)
    lrchanges=[];last=None
    for r in runtime:
        value=(r['lr/group0'],r['lr/group1'])
        if value!=last:
            lrchanges.append(dict(global_step=r['global_step'],epoch=r['epoch'],head_lr=value[0],encoder_projection_lr=value[1]));last=value
    def stat(values):
        a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
        return dict(count=len(a),mean=float(a.mean()),p05=float(np.percentile(a,5)),median=float(np.median(a)),p95=float(np.percentile(a,95)),min=float(a.min()),max=float(a.max()))
    gpu={}
    for index in (4,5):
        gpu[str(index)]={}
        for field in ('gpu','memoryAllocatedBytes','temp','powerWatts'):
            values=[r[f'gpu.{index}.{field}'] for r in system if r.get(f'gpu.{index}.{field}') is not None]
            if field=='memoryAllocatedBytes':values=[v/2**20 for v in values]
            gpu[str(index)][field]=stat(values)
    sampled_time=stat([r['runtime/train/seconds_per_batch_rank0'] for r in runtime])
    online_epochs=[r for r in remote if r.get('epoch/train/Loss/Total') is not None]
    maxonline=max(r['global_step'] for r in remote if r.get('global_step') is not None)
    online_time=max(r['_timestamp'] for r in remote if r.get('_timestamp') is not None)
    crosscheck=dict(online_state=online['state'],online_max_global_step=maxonline,online_complete_epochs=len(online_epochs),
                    online_last_beijing=dt.datetime.fromtimestamp(online_time,zone).isoformat(),local_history_rows=len(history),online_history_rows=len(remote),
                    local_complete_epochs=len(epochrows),local_max_global_step=max(r['global_step'] for r in history if r.get('global_step') is not None),
                    local_exit_code=records['exit_code'],batch_metric_samples=len(batches),runtime_samples=len(runtime),system_records=len(system),
                    max_log_rounding_differences=diffs,learning_rate_changes=lrchanges,gpu_resources=gpu,sampled_rank0_batch_seconds=sampled_time,
                    full_run_global_samples_per_second=6000000/base['wall_seconds'])
    (ART/'wandb_crosscheck.json').write_text(json.dumps(crosscheck,ensure_ascii=False,indent=2))
    with (ART/'wandb_epoch_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','global_step','total_loss','tracking_loss','weighted_cm_loss','train_iou','timestamp_beijing']);writer.writeheader()
        for epoch,r in epochrows.items():writer.writerow(dict(epoch=epoch,global_step=r['global_step'],**{k:r[v] for k,v in keys.items()},timestamp_beijing=dt.datetime.fromtimestamp(r['_timestamp'],zone).isoformat()))
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    steps=np.array([r['global_step'] for r in batches]);loss=np.array([r['train/Loss/Total'] for r in batches]);iou=np.array([r['train/IoU']*100 for r in batches])
    for ax,values,key,title in ((axes[0,0],loss,'epoch/train/Loss/Total','Sampled batch loss vs full-epoch mean'),(axes[0,1],iou,'epoch/train/IoU','Sampled batch IoU vs full-epoch mean')):
        ax.plot(steps,values,alpha=.18,lw=.6,label='Sampled batch (both ranks averaged)')
        er=list(epochrows.values());factor=100 if key.endswith('IoU') else 1
        ax.plot([r['global_step'] for r in er],[factor*r[key] for r in er],color='black',lw=1.7,label='Full-epoch mean')
        ax.axvline(67500,color='gray',ls='--');ax.set(title=title,xlabel='Optimizer updates',ylabel='IoU (%)' if factor==100 else 'Loss');ax.legend(fontsize=8)
    times=np.array([r['_timestamp'] for r in system]);origin=times.min()
    for index in (4,5):
        rr=[r for r in system if r.get(f'gpu.{index}.gpu') is not None]
        t=np.array([r['_timestamp'] for r in rr]);v=np.array([r[f'gpu.{index}.gpu'] for r in rr]);window=40
        if len(v)>=window:
            axes[1,0].plot((t[window-1:]-origin)/3600,np.convolve(v,np.ones(window)/window,mode='valid'),label=f'Physical GPU {index}')
        rr=[r for r in system if r.get(f'gpu.{index}.memoryAllocatedBytes') is not None]
        axes[1,1].plot([(r['_timestamp']-origin)/3600 for r in rr],[r[f'gpu.{index}.memoryAllocatedBytes']/2**20 for r in rr],label=f'Physical GPU {index}',alpha=.8)
    axes[1,0].set(title='System GPU activity (40-sample moving mean)',xlabel='Hours since W&B system monitoring began',ylabel='GPU utilization (%)');axes[1,0].legend()
    axes[1,1].set(title='System GPU framebuffer memory',xlabel='Hours since W&B system monitoring began',ylabel='MiB',ylim=(0,24576));axes[1,1].ticklabel_format(axis='y',style='plain',useOffset=False);axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Complete local W&B records; online history is truncated')
    fig.savefig(ART/'wandb_training_details.png',dpi=170);plt.close(fig)
    r54,r60=epochrows[54],epochrows[60]
    drop=(1-r60['epoch/train/Loss/Total']/r54['epoch/train/Loss/Total'])*100
    gain=(r60['epoch/train/IoU']-r54['epoch/train/IoU'])*100
    section=['<!-- WANDB-DETAILS-BEGIN -->','## W&B历史核对与资源分析','',
             f'核对在线run `{online["id"]}` 与原始本地 `run-7ln28ook.wandb`；在线快照取于 {online["fetched_at"]}。','',
             '| 来源 | 状态 | 完整轮平均记录 | 最大优化步数 |','| --- | --- | ---: | ---: |',
             f'| W&B在线API | {online["state"]}；summary仍为training | {len(online_epochs)}轮 | {maxonline:,} |',
             f'| W&B本地原始文件 | exit_code={records["exit_code"]}，最终summary为completed | 60轮 | 75,000 |',
             '| 训练监督程序和最终检查点 | completed，退出码0 | 60轮 | 75,000 |','',
             f'在线历史最后时间为北京时间 {crosscheck["online_last_beijing"]}，已经进入第42轮，但只有前41轮的完整轮均值。不能用在线最后的Loss或crashed标签代表训练最终结果。同步日志反复出现代理 `127.0.0.1:17891` 连接被拒绝，支持“上传中断”的解释；线上crashed状态具体如何形成未进一步核定。本次只读核对和归档，未重传或改写线上run。',
             f'本地共有{len(history):,}条history记录，其中60条完整轮均值、{len(batches):,}个抽样batch指标、{len(runtime):,}个单步资源记录。抽样间隔为20次更新且每轮末尾另记一次；train指标经过两卡平均，runtime和PyTorch显存仅由rank0记录。',
             f'60轮W&B精确均值与两卡训练日志核对，四项指标最大绝对差为{max(diffs.values()):.2e}，在日志5位小数的舍入误差范围内。主报告指标CSV和轮平均曲线使用完整本地W&B值，耗时仍来自训练日志。','',
             f'第54轮至第60轮，总Loss从{r54["epoch/train/Loss/Total"]:.6f}降到{r60["epoch/train/Loss/Total"]:.6f}，下降{drop:.2f}%；训练IoU提高{gain:.2f}个百分点。降学习率后训练目标仍有改善，不能由此推断测试指标一定提高。',
             f'W&B学习率采样确认：首次低学习率记录出现在第{lrchanges[-1]["epoch"]}轮、第{lrchanges[-1]["global_step"]:,}次更新；两组从5e-4/5e-5变为5e-5/5e-6，与最终优化器一致。','',
             f'![W&B完整训练与资源曲线]({ART}/wandb_training_details.png)','',
             '| 物理训练卡 | GPU利用率均值 / 中位数 / P95 (%) | 系统显存P95 / 最大值 (MiB) | 温度均值 / 最大值 (°C) | 功率均值 (W) |',
             '| --- | ---: | ---: | ---: | ---: |']
    for index,g in gpu.items():
        u=g['gpu'];m=g['memoryAllocatedBytes'];t=g['temp'];p=g['powerWatts']
        section.append(f'| GPU {index} | {u["mean"]:.1f} / {u["median"]:.1f} / {u["p95"]:.1f} | {m["p95"]:.0f} / {m["max"]:.0f} | {t["mean"]:.1f} / {t["max"]:.0f} | {p["mean"]:.1f} |')
    section.extend(['','资源统计来自完整本地W&B系统采样，覆盖W&B监控时段中的数据等待、通信和检查点保存，不等于覆盖模型初始化及监督程序收尾的全部阶段；均值是有效采样点的算术平均。系统显存包含驱动缓存等开销，与PyTorch allocated不同。GPU利用率表示采样期间GPU工作的时间比例，不代表计算单元数量或峰值算力使用率；NCCL通信也可能贡献利用率。仅统计本次物理4、5号训练卡，不把其他任务所在的卡计入训练。',
                    f'抽样rank0单步耗时：中位数{sampled_time["median"]:.3f}秒、P95 {sampled_time["p95"]:.3f}秒、均值{sampled_time["mean"]:.3f}秒，最后一次{runtime[-1]["runtime/train/seconds_per_batch_rank0"]:.3f}秒。该字段是采样当前batch的墙钟耗时，不是整轮平均；clips_per_second_rank0也只按每卡batch40计算，不能当全局吞吐量。',
                    f'总墙钟口径的全局平均样本速度为6,000,000 / {base["wall_seconds"]:.1f} ≈ {crosscheck["full_run_global_samples_per_second"]:.2f}个训练样本/秒。不能把抽样单步耗时直接用于完整任务ETA。','',
                    f'补充证据：[W&B精确逐轮CSV]({ART}/wandb_epoch_metrics.csv)、[核对与资源统计JSON]({ART}/wandb_crosscheck.json)、[在线run快照]({ART}/wandb_run.json)、[完整本地history]({ART}/wandb_local_history.json)。','<!-- WANDB-DETAILS-END -->',''])
    text=REPORT.read_text()
    text=re.sub(r'<!-- WANDB-DETAILS-BEGIN -->.*?<!-- WANDB-DETAILS-END -->\n?', '', text, flags=re.S)
    text=re.sub(r'<!-- WANDB-NOTICE-BEGIN -->.*?<!-- WANDB-NOTICE-END -->\n?', '', text, flags=re.S)
    notice='<!-- WANDB-NOTICE-BEGIN -->\n**监控同步说明：W&B在线历史仅到第42轮中段，页面为crashed；本地W&B记录、监督状态和最终检查点一致确认60轮正常完成。以下最终指标使用完整本地记录，详情见W&B核对章节。**\n<!-- WANDB-NOTICE-END -->\n\n'
    text=text.replace('## 完成情况',notice+'## 完成情况',1)
    text=text.replace('## 检查点与异常','\n'.join(section)+'\n## 检查点与异常',1)
    REPORT.write_text(text)
    base['wandb_crosscheck']=crosscheck
    (ART/'summary.json').write_text(json.dumps(base,ensure_ascii=False,indent=2))
    print(json.dumps(crosscheck,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
