# 轻量可靠记忆库：首版实现与跑通检查

日期：2026-11-16（北京时间）。分支：`codex/reliable-memory`，基点：`a115a99`。

结论：首版可靠写入、干扰感知保留、受限融合，以及共享逐帧训练/推理路径已实现；工程检查、真实单卡更新/恢复、双卡同步、状态重载和三来源短视频评测均已跑通。这里只确认代码可执行，不能据此认定跟踪精度提升。

后续已接入可选 W&B 监控并完成离线单卡恢复、双卡检查，见 [监控验证报告](RELIABLE_MEMORY_WANDB_20261116.md)。本报告配套 JSON 的源码哈希仍对应首版检查时的快照。

## 1. 本次实现

| 内容 | 文件 |
|---|---|
| 连续确认、拒绝原因、快速运动和全图恢复 | `lib/models/mvlm/memory_policy.py` |
| 长短期管理、干扰辨识统计、受控淘汰和状态快照 | `lib/models/mvlm/reliable_memory.py` |
| 投影后残差范数限幅，保留旧参数名称和旧路径 | `lib/models/mvlm/memory_fusion.py` |
| 精确裁剪变换、共享候选决策、先读后写的逐帧核心 | `lib/models/mvlm/tracking_core.py`、`tracking_ops.py` |
| 新 tracker 路径，首帧返回实际预测，忽略 GT 框 | `lib/test/tracker/mvlm.py` |
| 官方训练清单、完整 RGB 片段、保留遮挡内部帧 | `lib/train/data/reliable_memory_dataset.py` |
| 预测裁剪、GT 仅参与损失、预热与逐帧梯度累积 | `lib/train/actors/reliable_memory.py` |
| M1–M6 开关、AdamW/cosine、冻结审计、源码快照与恢复 | `tracking/train_reliable_memory.py` |
| 首帧实际预测、独立严格计分和逐帧结果导出 | `tracking/evaluate_reliable_memory.py` |
| 新配置默认隔离，重复 YAML 键报错 | `lib/config/mvlm/config.py`、`experiments/mvlm/mvlm_TF_reliable_memory.yaml` |

第一版确认后只提交当前帧，不回填待确认帧。干扰候选用于长期条目的保留统计，不另建可读取的负样本库。读取器仍为旧双路注意力，不新增大型网络。

## 2. 检查结果与证据

### CPU 检查

`tracking/check_reliable_memory.py` 的 12 项检查通过，覆盖：确认与因果隔离、恢复状态机、缺失竞争证据、长期淘汰和快照、旧管理等价、残差上限与梯度、非方图坐标、GT 改变不影响预测、未知/出裁剪/不可见监督、严格首帧计分、M1–M6 路径、遮挡片段采样，以及重复配置拒绝。

旧 `tracking/check_memory_module.py` 的四组轻量检查也通过，覆盖原 ROI/记忆管理、采样处理、梯度和配置隔离。旧新开关关闭时没有引入已检测到的旧路径回归。

日志：

- `output/reliable_memory_checks_final.log`
- `output/legacy_memory_checks.log`

### 真实单卡训练与断点恢复

- 官方权重：`/root/MVLM/models/MVLM_TF.pth.tar`。
- 清单：`/root/MVLM/output/memory_world_split/manifest.json`。
- 工程子集：每来源、每划分取 1 条视频，共训练 3 条、留出 3 条；不改原清单。
- M6，GPU0，全局 batch 1，每片段 12 帧、前 4 帧预热。
- 总预算 2 次更新；先在第 1 次更新保存并停止，再以相同预算恢复到第 2 次。
- 每次 8 帧提供监督；第一步写入 9 次，待确认 2 次，拒绝 1 次；第二个片段 12 次写入均被拒绝。
- 两步损失、梯度有限；可训练参数更新，冻结参数和 CENTER BN 统计未变。
- 每个 run 保存有效配置、父权重/清单/源码哈希及源码快照。恢复会拒绝数据、配置、代码、卡数或预算改变。

最终输出：`output/reliable_smoke_single_20261116_run03/`，重点查看 `training_status.json`、`metrics_rank0.jsonl`、`audit_rank0.json`、`last.pth.tar` 和 `source/`。

### 真实双卡更新

GPU2/3，M6，全局 batch 2，每卡 1 个四帧片段，前 1 帧预热，实际完成 1 次更新。两卡梯度有限、可训练参数哈希一致，冻结参数和 BN 统计不变。

最终输出：`output/reliable_smoke_ddp_20261116_run03/`；两张卡分别保存 `audit_rank0.json`、`audit_rank1.json`。短片段只验证 DDP 更新，不代表验证了长期保留效果。

### 真实权重 tracker、状态重载与 GT 独立性

在 LaSOT 留出视频 `cat-4` 的前 6 帧检查：

1. tracker 新入口与直接调用共享核心的预测相同。
2. 向初始化和跟踪接口传入两组不同的伪 GT 框，预测相同。
3. 视频状态存盘、重载后，后续预测和写入决策一致。
4. 实际出现拒绝、待确认和提交三种写入行为。

最终输出：`output/reliable_adapter_check_20261116_run02/report.json`。视频状态重载检查使用相同模型权重和相同设备；跨权重、跨设备继续视频不在此次验收范围内。

### 三来源评测入口

使用两步工程短训检查点，在留出视频 LaSOT `cat-4`、TNL2K `CM_graycar2_done---`、VastTrack `Wallaby-29` 各测试前 12 帧，共 36 帧。首帧由整图和语言实际预测；每条视频均完整导出浮点框、写入事件及指标。

最终输出：`output/reliable_smoke_eval_20261116_run03/`，其中 `summary.json` 明确标记 `engineering_only=true`。这些不是标准测试集结果，也不是完整视频 AUC，不用于模型选择或提升结论。

TNL2K 和 VastTrack 的这两个短片段没有发生正式写入，LaSOT 有写入。TNL2K 片段存在实际定位失败。不能将“程序执行成功”解释为“所有视频跟踪成功”，也不能认为当前写入阈值已调好。

## 3. 计分协议

新入口使用 `strict-v1`：

- TF 架构，首帧实际预测；运行时状态不接收 GT。
- 保存浮点框，不做旧结果文件的整数舍入。
- 使用项目的框几何与中心误差定义，AUC 采用 21 个 IoU 阈值的严格大于成功率，PRE@20、NPR@0.2，逐视频平均。
- 无效标注、不可见或失败帧在三项指标中均计为失败，分母保留完整帧数。
- 不执行旧 scorer 的首帧 GT 替换、零框前帧填充，也不沿用非 LaSOT 无效帧可被计为 NPR 成功的旧边界规则。

因此新指标不能直接与历史各表相减；比较时应对所有模型使用同一新协议重测。通用旧 `analysis_results` 仍有其历史行为，本分支主结果使用专用新入口。

## 4. 可复用命令

在本 worktree 下执行，按实际环境替换 Python 和资源路径；所有新输出目录必须不存在。

```bash
/root/miniconda3/envs/mvlm/bin/python tracking/check_reliable_memory.py --device cpu

/root/miniconda3/envs/mvlm/bin/python tracking/train_reliable_memory.py \
  --config mvlm_TF_reliable_memory --group M6 \
  --manifest /root/MVLM/output/memory_world_split/manifest.json \
  --checkpoint /root/MVLM/models/MVLM_TF.pth.tar \
  --device cuda:0 --max-updates 2 --stop-after 1 --global-batch 1 \
  --engineering-subset 1 --save-every 1 --out /absolute/path/to/new-smoke
```

恢复时保持其他参数一致，删除 `--stop-after 1` 并加 `--resume`，输出目录沿用已存在的同一目录。不能同时修改 `--max-updates`，因为它决定调度预算。

```bash
/root/miniconda3/envs/mvlm/bin/python tracking/evaluate_reliable_memory.py \
  --config mvlm_TF_reliable_memory --group M6 \
  --checkpoint /absolute/path/to/new-smoke/last.pth.tar \
  --manifest /root/MVLM/output/memory_world_split/manifest.json \
  --max-sequences 3 --max-frames 12 --device cuda:0 \
  --out /absolute/path/to/new-evaluation
```

`--dataset otb99_lang/lasot/tnl2k` 可替代 `--manifest`，用于准备好的标准测试；本次没有启动完整标准测试。语言为空会明确报错，不自动删视频。官方初始化权重缺少记忆参数时，评测默认拒绝；只有显式 `--allow-untrained-memory` 才允许做工程检查。

M1–M6 由训练和评测共同的 `--group` 选择。评测时使用与训练相应的组；有意改变组属于同权重推理干预，不能当作该组配对训练成绩。`--baseline` 使用当前提供权重关闭内容记忆，不会将微调头还原为官方权重。

## 5. 当前边界和下一步

- 新方案可执行，尚未完成 4000 更新正式训练、M1–M6 配对精度比较或多种子实验。
- 真实短测未充分覆盖长期库填满后的淘汰；该机制目前主要通过合成满库检查，需要长视频及共同长片段训练补证据。
- 部分片段全程不写入，需要先在固定开发/验证视频检查 score、候选 margin、确认延迟和写入覆盖率，再确定阈值。
- 已有写入原因、候选索引、库大小和限幅诊断；读取熵、条目生存分析、遮挡事件恢复统计等扩展分析尚未补齐。
- 训练器当前保存最新 `last.pth.tar`，尚未接入定期完整验证、保留全部候选检查点和自动最佳模型选择；正式配对长训前补齐这些调度功能。
- 工程子集审计全量清单中的身份重复，并核验选中视频的路径和签名，不等于全量图像解码或近重复内容审计。
- 新旧计分和闭环训练有变化，必须保留 M2“旧记忆规则＋新闭环”的重训对照。

最终结构化证据见同目录 `RELIABLE_MEMORY_SMOKE_20261116.json`。中间 run01/run02 产物保留；最终实现核验以本文列出的 run03 训练/评测及 run02 适配器检查为准。
