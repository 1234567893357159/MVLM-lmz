"""Generate the completed three-dataset training report from frozen run evidence."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT=Path('/root/MVLM')
sys.path.insert(0,str(ROOT))
RUN=ROOT/'output/paper_reproduction_20261030_run01'
REPORT=ROOT/'reports/reproduction/PAPER_TRAINING_RESULTS_20261101.md'
ART=ROOT/'reports/reproduction/paper_training_results_20261101'


def main():
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ART.mkdir(exist_ok=True)
    status=json.loads((RUN/'training_status.json').read_text())
    spec=json.loads((RUN/'run_spec.json').read_text())
    initialization=json.loads((RUN/'initialization.json').read_text())
    log=(RUN/'train.log').read_text(errors='replace')
    records={}
    for line in log.splitlines():
        match=re.search(r'\[train: (\d+), (\d+) / (\d+)\]',line)
        if not match or match[2]!=match[3]:continue
        values={k:float(v) for k,v in re.findall(r'(Loss/Total|Loss/Tracking|Loss/CM|IoU|TotalTime): ([\d.eE+-]+)',line)}
        assert len(values)==5,line
        records.setdefault(int(match[1]),[]).append(values)
    assert sorted(records)==list(range(1,61)),sorted(records)
    rows=[]
    for epoch, values in sorted(records.items()):
        assert len(values)==2,(epoch,len(values))
        means={k:sum(v[k] for v in values)/2 for k in values[0]}
        rows.append(dict(epoch=epoch,total_loss=means['Loss/Total'],tracking_loss=means['Loss/Tracking'],
                         weighted_cm_loss=means['Loss/CM'],train_iou=means['IoU'],seconds_per_update=means['TotalTime']))
    local_history=ART/'wandb_local_history.json'
    if local_history.exists():
        with (ART/'epoch_metrics_log_rounded.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        history=json.loads(local_history.read_text())
        epoch_rows={int(r['global_step'])//1250:r for r in history if r.get('epoch/train/Loss/Total') is not None}
        assert sorted(epoch_rows)==list(range(1,61))
        mapping={'total_loss':'epoch/train/Loss/Total','tracking_loss':'epoch/train/Loss/Tracking','weighted_cm_loss':'epoch/train/Loss/CM','train_iou':'epoch/train/IoU'}
        for row in rows:
            for key,metric in mapping.items():
                assert abs(row[key]-epoch_rows[row['epoch']][metric])<5.1e-6
                row[key]=epoch_rows[row['epoch']][metric]
    ckdir=RUN/'checkpoints/train/mvlm/mvlm_TF_paper_reproduction'
    files=sorted(ckdir.glob('MVLM_ep*.pth.tar'))
    assert [int(re.search(r'ep(\d+)',f.name)[1]) for f in files]==list(range(1,61))
    final=ckdir/'MVLM_ep0060.pth.tar'
    checkpoint=torch.load(final,map_location='cpu')
    assert checkpoint['epoch']==60 and checkpoint['global_step']==75000 and status['exit_code']==0
    nonfinite=[k for k,v in checkpoint['net'].items() if torch.is_floating_point(v) and not bool(torch.isfinite(v).all())]
    assert not nonfinite,nonfinite
    groups=[dict(lr=g['lr'],initial_lr=g['initial_lr'],weight_decay=g['weight_decay'],parameter_tensors=len(g['params'])) for g in checkpoint['optimizer']['param_groups']]
    assert [g['lr'] for g in groups]==[5e-5,5e-6],groups
    h=hashlib.sha256()
    with final.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    start=dt.datetime.fromisoformat(status['started_at']);end=dt.datetime.fromisoformat(status['updated_at'])
    zone=dt.timezone(dt.timedelta(hours=8));seconds=(end-start).total_seconds()
    errors={'sampler_retries':log.count("data sampler bug: 'float' object is not iterable"),'training_crashes':log.count('Training crashed at epoch'),'tracebacks':log.count('Traceback (most recent call last)')}
    first,last=rows[0],rows[-1]
    summary=dict(run_directory=str(RUN),status=status,start_beijing=start.astimezone(zone).isoformat(),end_beijing=end.astimezone(zone).isoformat(),wall_seconds=seconds,
                 initial_metrics=first,final_metrics=last,best_training_loss_epoch=min(rows,key=lambda r:r['total_loss'])['epoch'],
                 best_training_iou_epoch=max(rows,key=lambda r:r['train_iou'])['epoch'],optimizer_groups=groups,
                 checkpoint_count=len(files),checkpoint_bytes=sum(f.stat().st_size for f in files),final_checkpoint=str(final),final_checkpoint_sha256=h.hexdigest(),
                 final_net_all_finite=not nonfinite,anomalies=errors,metrics_source='Complete local W&B cross-rank full-epoch history, checked against rounded logs; time from logs.' if local_history.exists() else 'Mean of the two rank-local full-epoch training log averages; printed to five decimal places.',
                 trainable_parameters=sum(g['trainable'] for g in initialization['parameter_groups'].values()))
    (ART/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    with (ART/'epoch_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name in ('run_spec.json','initialization.json','config.yaml','training_status.json'):
        shutil.copy2(RUN/name,ART/name)
    readiness=Path('/tmp/mvlm_training_data_recheck_20261030_132358.json')
    if readiness.exists():shutil.copy2(readiness,ART/'pretraining_data_readiness.json')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    x=[r['epoch'] for r in rows]
    for key,label in [('total_loss','Total'),('tracking_loss','Tracking'),('weighted_cm_loss','Weighted CM')]:axes[0,0].plot(x,[r[key] for r in rows],label=label)
    axes[0,0].set(title='Full-epoch training losses',xlabel='Epoch',ylabel='Loss');axes[0,0].legend()
    axes[0,1].plot(x,[100*r['train_iou'] for r in rows]);axes[0,1].set(title='Training IoU (not test AUC)',xlabel='Epoch',ylabel='IoU (%)')
    for g,label in zip(groups,('Head','Encoder + text projection')):
        axes[1,0].step(x,[g['initial_lr'] if e<=54 else g['lr'] for e in x],where='mid',label=label)
    axes[1,0].set_yscale('log');axes[1,0].set(title='Configured schedule, checked against final optimizer',xlabel='Epoch',ylabel='Learning rate');axes[1,0].legend()
    axes[1,1].plot(x,[r['seconds_per_update'] for r in rows]);axes[1,1].set(title='Epoch-average batch time from training log',xlabel='Epoch',ylabel='Seconds / optimizer update')
    for ax in axes.flat:ax.axvline(54.5,color='gray',ls='--',alpha=.5);ax.grid(alpha=.2)
    fig.suptitle('MVLM three-dataset training: 60 epochs, global batch 80')
    fig.savefig(ART/'training_curves.png',dpi=170);plt.close(fig)
    lines=['# MVLM 三数据集训练完成报告','',f'整理时间：{dt.datetime.now(zone).isoformat(timespec="seconds")}。本报告对应 `paper_reproduction_20261030_run01`，不是此前 VastTrack-only 实验。','',
           '## 完成情况','',f'- 训练正常完成：60/60轮，75,000次优化器更新，退出码0；状态文件为 completed。',
           f'- 开始：{start.astimezone(zone):%Y-%m-%d %H:%M:%S}；结束：{end.astimezone(zone):%Y-%m-%d %H:%M:%S}（均为北京时间）。',
           f'- 监督程序记录的总墙钟时间：{seconds/3600:.2f}小时，包含初始化、检查点保存和监控收尾；不是纯GPU计算时间。',
           '- 物理GPU 4、5；每卡batch 40，全局batch 80；每卡2个数据加载worker，总计4个；无梯度累积，未启用AMP。',
           f'- [W&B曲线]({status["wandb"]["url"]})；[训练原始日志]({RUN}/train.log)；[最终状态]({RUN}/training_status.json)。','',
           '## 训练数据','', '| 数据集 | 实际训练视频数 | 准备审计中的图像帧数 | 抽样比例 |','| --- | ---: | ---: | ---: |',
           '| LaSOT | 1,120 | 2,831,982 | 1 |','| VastTrack | 46,825 | 3,805,290 | 1 |','| TNL2K | 1,299 | 727,345 | 1 |','',
           '每轮100,000个随机抽样训练样本，60轮共6,000,000次样本抽取，不代表600万个不同视频或不同帧；1:1:1为抽样概率，未逐数据集累计实际抽样次数。',
           'TNL2K原始1,300条中跳过 `Fight_game_004-Done`，原文件保留；VastTrack有效清单在准备阶段排除了11条缺帧序列。实际序列顺序及哈希见运行目录 `split/` 与归档 `run_spec.json`。准备审计检查了目录、标注和语言，并抽查图像解码；未逐帧完整解码全部图像。',
           '本次没有独立验证集。此前VastTrack-only留出的500条已参与本次训练，不能再用作独立验证。','',
           '## 模型、初始化与优化参数','',
           '- Template-free FastITPN-B + CLIP ViT-L/14 + CENTER；加载预训练FastITPN和CLIP，语言投影与CENTER随机初始化；没有加载论文发布的MVLM或旧自训练模型。',
           f'- CLIP冻结；可训练参数共{summary["trainable_parameters"]:,}个，视觉骨干、文本投影和预测头参与优化。',
           '- AdamW；预测头初始学习率5e-4，视觉编码器及文本投影5e-5；第54轮结束后乘0.1，第55～60轮分别使用5e-5和5e-6。已直接核对最终检查点，W&B终端简写的数值可能被舍入。',
           '- weight decay=1e-4；梯度裁剪范数=0.1；drop path=0.1；基础seed=1234567890，并按rank偏移。',
           '- 搜索输入224×224，搜索区域因子4.0，位置抖动3.5、尺度抖动0.5；训练使用真实框裁剪/监督，不能把训练IoU直接当作纯语言在线追踪成绩。',
           '- 总损失：2×GIoU + 5×L1 + Focal + 0.5×CM；日志中的CM已经乘过0.5。','',
           '## 训练结果与曲线','',
           f'![60轮训练曲线]({ART}/training_curves.png)','',
           f'第1轮到第60轮：总Loss由{first["total_loss"]:.5f}下降到{last["total_loss"]:.5f}，下降{100*(1-last["total_loss"]/first["total_loss"]):.2f}%；训练IoU由{first["train_iou"]*100:.2f}%上升至{last["train_iou"]*100:.2f}%，提高{100*(last["train_iou"]-first["train_iou"]):.2f}个百分点。',
           f'最终Tracking Loss={last["tracking_loss"]:.5f}，加权CM Loss={last["weighted_cm_loss"]:.5f}。训练Loss最低和训练IoU最高的完整轮次分别为第{summary["best_training_loss_epoch"]}、{summary["best_training_iou_epoch"]}轮；这不是按测试性能选出的最优模型。',
           ('下表的Loss/IoU采用完整本地W&B记录的两卡平均精确值，并与每轮两卡训练日志交叉核对；耗时采用训练日志。显示值保留5位小数。' if local_history.exists() else '下表将两卡在每轮末尾打印的累计均值等权平均；日志原值只有5位小数。')+'未记录按LaSOT/VastTrack/TNL2K分别拆分的训练Loss，也未记录可用于分析的梯度范数曲线。','',
           '| Epoch | 总Loss | Tracking Loss | 加权CM Loss | 训练IoU (%) | 秒/更新 |','| ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append(f'| {r["epoch"]} | {r["total_loss"]:.5f} | {r["tracking_loss"]:.5f} | {r["weighted_cm_loss"]:.5f} | {r["train_iou"]*100:.2f} | {r["seconds_per_update"]:.3f} |')
    lines.extend(['','## 检查点与异常','',
                  f'- 已核对第1～60轮检查点连续完整，共60个，合计{summary["checkpoint_bytes"]/1024**3:.2f} GiB。检查点含模型、优化器及训练状态；冻结CLIP权重省略，加载时使用相同预训练CLIP。',
                  f'- [最终第60轮检查点]({final})；其中epoch=60、global_step=75000；全部已保存浮点模型张量检查为有限值。',
                  f'- 第60轮检查点SHA256：`{summary["final_checkpoint_sha256"]}`。',
                  '- 训练过程中多次观测每卡约23,251 MiB/24,576 MiB显存占用；W&B主卡最终记录的PyTorch峰值分配约22,303 MiB。两者统计口径不同，也不是双卡全程采样的显存峰值。',
                  f'- 日志共{errors["sampler_retries"]}次 `data sampler bug: float object is not iterable`；采样器捕获异常后重新抽样，未中断训练。原日志没有序列名/堆栈，无法据此定位具体样本，完整性准备检查也不能排除所有运行时采样问题。',
                  f'- 日志训练崩溃标记{errors["training_crashes"]}次、Traceback标记{errors["tracebacks"]}次，最终正常退出。','',
                  '## 结果边界与后续评测','',
                  'Loss和训练IoU改善说明优化目标得到学习；没有独立验证曲线，无法仅凭这些指标判断泛化、过拟合或论文复现成功。训练参数中的全局batch80是对论文未明确全局/每卡含义的解释，两卡的BatchNorm条件也与论文四卡不同。',
                  f'当前外部评测、论文参考行和逐轮三集AUC/PRE/NPR见[持续更新的测试报告]({ROOT}/reports/reproduction/PAPER_CHECKPOINT_TESTS_20261101.md)。测试采用main首帧真实框协议和历史测试集调参配置；现在按每5轮粗测、最高三集平均AUC附近细测，属于测试引导的探索性模型选择，不能把所选最高分称为独立测试成绩。','',
                  '## 机器可读数据与原始证据','',
                  f'- [60轮指标CSV]({ART}/epoch_metrics.csv)；[汇总JSON]({ART}/summary.json)。',
                  f'- [实际配置]({ART}/config.yaml)；[资源和数据清单审计]({ART}/run_spec.json)；[初始化审计]({ART}/initialization.json)。',
                  f'- 完整源码快照及SHA256：`{RUN}/source/`、`{RUN}/source_sha256.json`。',
                  f'- 早期训练启动记录：[PAPER_REPRODUCTION_TRAINING_20261030.md]({ROOT}/reports/reproduction/PAPER_REPRODUCTION_TRAINING_20261030.md)。其中启动步数和早期W&B状态保留历史含义，最终结果以本报告和最终状态为准。'])
    REPORT.write_text('\n'.join(lines)+'\n')
    if local_history.exists():
        import enrich_paper_training_wandb
        enrich_paper_training_wandb.main()
    print(json.dumps(dict(report=str(REPORT),final=last,duration_hours=seconds/3600,anomalies=errors,checkpoint_count=len(files),optimizer_groups=groups),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
