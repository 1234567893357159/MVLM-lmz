"""Generate a bounded-evidence report from completed diagnostic artifacts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/root/MVLM/output/memory_three_custom_20261018_run01');OUT=RUN/'diagnostics'
def read(name):
    p=OUT/name
    return json.loads(p.read_text()) if p.exists() else None
screen=read('screen/summary.json');audit=read('data_audit.json');archive=read('tnl2k_archive_audit.json')
full_quarter=read('full_quarter/summary.json');full_strict=read('full_strict_quality/summary.json');gap=read('training_crop_gap.json')
report={'status':'completed' if all([screen,audit,archive,full_quarter,full_strict,gap]) else 'in_progress',
        'screen':screen,'full_quarter':full_quarter,'full_strict_quality':full_strict,
        'archive_audit':archive,'training_crop_gap':gap,
        'scope':'Existing checkpoints; nine selected validation prefixes for screening, full frozen 159-sequence validation for two prespecified follow-ups; test metrics never evaluated.',
        'limitations':['Selected diagnostic prefixes are not representative aggregate performance.',
                       'Oracle GT writes are not deployable results and change boxes, visibility and write quality together.',
                       'One training batch per dataset is exploratory paired evidence, not a full retraining comparison.',
                       'No new training/seed comparison; head unfreezing and online-crop retraining remain untested.',
                       'Current archive checksum and sampled image decoding do not certify every extracted frame or semantic annotation.']}
(OUT/'diagnostic_summary.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# 记忆模块逐项排查（2026-10-19）','',
'状态：'+report['status']+'。本轮不重训，不评估保留测试集，不覆盖原训练或预测结果。','',
'## 1. 优化器、冻结和实现检查','',
'- 最终检查点为20,000步；19个记忆参数张量均有非零优化器一阶矩；冻结基线参数变化数为0，权重均为有限值。',
'- 既有ROI/坐标、严格时间顺序采样、片段一致增强、顺序Actor损失、梯度传播及配置隔离检查通过。',
'- 9条诊断序列关闭记忆后与原基线逐帧整数预测完全一致；不改融合的复现对照也与原最终模型逐帧一致。训练/推理均使用RGB，未发现颜色通道错误。','',
'## 2. 数据检查','']
if audit:
 lines += ['- 复核全部冻结标注的长度与有限值；未发现空语言进入新划分；训练/验证/测试路径和已定义同源分组交集均为0。',
           '- 每个序列首/中/末和一个固定随机图像共抽样解码%d张，失败%d张；不是全量逐图解码。'%(audit['decoded_images'],len(audit['decode_errors'])),
           '- TNL2K训练部分有52,687帧宽高非正，其中52,677帧为全零框；当前有效性掩码会排除这些帧。它们不能一概视为损坏，但说明当前训练没有覆盖这些无有效目标框时的写入拒绝情形。']
if archive:lines += ['- 当前TNL2K压缩包SHA256与下载时保存的第三方镜像LFS元数据%s；校验值`%s`。先前未确认事项已关闭，不等于和官方源做了等价性验证。'%('一致' if archive['matches'] else '不一致',archive['sha256'])]
lines += ['', '## 3. 验证子集受控诊断','',
'固定选择每数据集最终模型相对基线变化的最差、中位和最好各一条，共9条；每条最多512帧。选择偏差较强，仅用于定位机制，不能与完整验证集的60.73直接比较。','',
'| 设置 | 子集三集平均AUC | 作用 |','| --- | ---: | --- |']
labels={'cached_baseline':('官方基线','子集参照'), 'cached_final':('最终模型原配置','原始新增记忆'),
 'disabled':('关闭新增记忆','控制基线一致性'), 'half':('融合乘0.5','只减弱残差'), 'quarter':('融合乘0.25','只减弱残差'),
 'short_only':('仅短期记忆','长期容量设0'), 'strict_quality':('写入门槛0.9','原为0.5，其余不变'),
 'no_margin_write_gate':('取消候选margin写入门控','保留ROI有效性和质量门槛'),
 'oracle_gt_write':('真实框写入（仅诊断）','使用当前真实框、可见性和质量1；不是实际模型成绩'),
 'enhanced_correlation':('相关性改用增强特征','原始特征仍用于记忆写入'),
 'best_half':('2000步模型融合乘0.5','检查早期检查点'), 'best_short_only':('2000步仅短期记忆','检查早期检查点')}
if screen:
 for key,(label,meaning) in labels.items():
  if key in screen:lines.append('| %s | %.2f | %s |'%(label,screen[key]['mean_AUC'],meaning))
lines += ['',
'在选定子集上，单独去掉长期库或改用增强特征做相关评分没有恢复至基线。降低融合强度、严格写入和真实框写入有恢复案例；尚不能唯一归因，因为阈值还改变了记忆数量和总修正量。','',
'## 4. 完整159序列验证','',
'以下均为验证集，不是测试集；使用相同最终检查点，不再改变训练。','',
'| 设置 | OTB AUC | LaSOT AUC | TNL2K AUC | 三集平均 | 相对基线 |',
'| --- | ---: | ---: | ---: | ---: | ---: |']
base=json.loads((RUN/'validation/baseline/metrics.json').read_text());final=json.loads((RUN/'validation/epoch_0020/metrics.json').read_text())
def row(label,metrics,mean):
 lines.append('| %s | %.2f | %.2f | %.2f | %.2f | %+.2f |'%(label,metrics['otb99_lang'],metrics['lasot'],metrics['tnl2k'],mean,mean-base['mean_AUC']))
for label,data in [('官方基线',base),('最终原配置',final)]:row(label,{d:v['AUC'] for d,v in data['metrics'].items()},data['mean_AUC'])
for label,variant,data in [('融合乘0.25','quarter',full_quarter),('写入门槛0.9','strict_quality',full_strict)]:
 if data:row(label,data[variant]['datasets'],data[variant]['mean_AUC'])
 else:lines.append('| %s | 待完成 | — | — | — | — |'%label)
paired=read('strict_quality_paired_analysis.json')
if paired:
 lines += ['', '严格写入门槛相对基线平均差异约+0.34个百分点；按数据集内序列重采样的探索性95%%区间为[%.2f, %.2f]，包含0。该区间未计入设置选择和多随机种子不确定性，不能宣称可靠测试集提升。'%tuple(paired['mean_delta_bootstrap_95_interval'])]
lines += ['', '## 5. 固定训练片段配对检查','',
'每个数据集仅用同一批2个12帧训练片段，保持裁剪/增强相同，分别改变历史写入方式。没有更新权重；这不能代替多批次或重新训练实验。','',
'| 数据 | 真框历史损失 | 混合历史损失 | 预测历史损失 | 关闭融合损失 |','| --- | ---: | ---: | ---: | ---: |']
if gap:
 for dataset,modes in gap['results'].items():
  lines.append('| %s | %.4f | %.4f | %.4f | %.4f |'%(dataset,*[modes[k]['Loss/Tracking'] for k in ['gt_history','mixed_history','predicted_history','fusion_disabled']]))
lines += ['',
'这些配对样本说明部分训练裁剪上记忆确能降低定位损失，且OTB/TNL2K样本对历史是否正确较敏感；LaSOT样本未改善。当前训练混合真框/预测框，且裁剪依赖真框，与完全预测驱动的在线闭环不同。这支持继续研究训练/推理落差，不构成已确定唯一原因的证据。','',
'## 6. 尚未被本轮因果验证的解释','',
'- 源码确认训练预测写入质量来自原始score map最大值，在线写入质量来自经过Hann窗的最终候选分数，在线还要求候选margin门控通过；这也是训练/推理写入条件差异。',
'- 优先方向是约束不可靠历史的写入及修正强度，并让训练覆盖预测误差/目标缺失，使用一致门控；不是已证明唯一根因。',
'- 解冻预测头是否更好，需要同预算重训对照；当前只排除了冻结阻断梯度，并没有验证解冻有收益。',
'- 数据量、数据集采样比例及学习率是否最优，需要独立训练对照；不能通过一次成功哈希或一次短片段试验排除。',
'- 图像抽样和路径检查不能证明每帧语义标注都正确；已知同源分组也不能排除全部近重复视频。',
'- 本轮完整验证仍属于在验证集上探索；后续应冻结方案，再评估保留测试集。','',
'可视化：`qualitative_examples.png`；原始设置、逐序列框和统计在本目录各实验子目录。']
text='\n'.join(lines)+'\n';(OUT/'DIAGNOSTIC_REPORT.md').write_text(text)
(ROOT/'reports/reproduction/MEMORY_DIAGNOSTICS_20261019.md').write_text(text)
print(report['status'])
