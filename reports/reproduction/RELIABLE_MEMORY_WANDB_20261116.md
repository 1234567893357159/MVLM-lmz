# 可靠记忆库训练：W&B 监控接入与验证

日期：2026-11-16 UTC。分支：`codex/reliable-memory`。

新的逐帧训练入口已提供可选 W&B 监控。所有本次检查使用离线模式，没有上传到云端，也没有运行正式精度实验。结构化证据见 [RELIABLE_MEMORY_WANDB_20261116.json](RELIABLE_MEMORY_WANDB_20261116.json)。

## 使用

在 `tracking/train_reliable_memory.py` 的训练命令后追加：

```bash
--wandb --wandb-project mvlm-reliable-memory --wandb-name M6-seed20261116
```

离线验证再追加 `--wandb-mode offline`。监控默认关闭，默认每 20 次更新记录；`--wandb-log-interval 1` 可逐次记录。首步和停止步始终记录。`--resume` 保持相同输出目录及监控参数。

监控实现：`tracking/reliable_memory_wandb.py`。配置：`experiments/mvlm/mvlm_TF_reliable_memory.yaml` 中的 `WANDB` 段。详细参数与恢复限制见 [实现方案](../../RELIABLE_MEMORY_IMPLEMENTATION.md#可选-wb-监控)。

## 记录内容与故障处理

- 全局损失、监督/回归/出裁剪/部分可见/背景帧数、提交/待确认/拒绝次数及比例。
- 本步使用的学习率、下一步学习率、裁剪前梯度范数、实际更新步数。
- rank0 的记忆容量、候选分数、拒绝原因、全图搜索比例、原始融合残差比和限幅比例、耗时和显存。
- 配置、清单与初始权重哈希、源码哈希、版本、种子、分组和工程子集标记。

双卡损失按训练归一化方式取跨卡均值，计数跨卡求和；逐帧诊断明确标记为 rank0。没有新增验证循环，因此没有验证集精度曲线。

训练进程先保存本地事件，独立进程运行 W&B SDK。缺依赖、认证或网络失败不终止优化；本地训练指标始终写入 `metrics_rank*.jsonl`。每次监控启动单独保存 `worker.log`、`status.json`、`events.jsonl`，在训练输出目录的 `wandb_monitor/session-*` 下。监控退出等待有时间上限。

联网恢复沿用持久化 ID；本次仅验证 ID 持久化和离线恢复，没有验证云端服务续接。离线 SDK 每次启动创建独立目录，即使 ID 相同也不能据此认定已经在云端合并。

## 检查结果

环境：`/root/miniconda3/envs/mvlm/bin/python`，W&B 0.22.3，官方 `MVLM_TF.pth.tar`，原 `memory_world_split/manifest.json`。真实训练均使用 M6、每来源每划分 1 条视频的工程子集、12 帧片段和 4 帧预热。

| 检查 | 结果 |
|---|---|
| 关闭监控/非主进程不创建记录、间隔/末步、非有限监控值过滤、ID 恢复、启动/SDK 失败、实际步数与结束状态、指标口径 | 7 项通过 |
| GPU0，batch 1，总预算 2 步；第 1 步停止，再恢复到第 2 步 | 通过；同一 ID `20a97180`，事件步数为 1、2；SDK 均正常结束 |
| GPU1/2，双卡 batch 2，一步训练 | 通过；仅一份 W&B 记录，ID `3f9a26db` |
| 双卡监督/提交帧数 | 16 / 9，与两份本地日志之和一致 |
| 双卡全局损失 | 2.189360645134002，与两卡归一化损失均值一致 |
| 冻结参数、CENTER BN、双卡训练参数一致性 | 全部通过 |
| 源码快照 | 训练记录哈希与验证时文件一致，包含监控模块 |

两步单卡损失分别为 0.1464738780632615、0.5973819978535175。两步来自不同视频片段，仅用于检查记录一致性，不能解读为模型收敛趋势。

本地证据：

- `output/reliable_wandb_checks_20261116.log`
- `output/reliable_wandb_offline_smoke_20261116_run01/`
- `output/reliable_wandb_offline_smoke_20261116_start.log`
- `output/reliable_wandb_offline_smoke_20261116_resume.log`
- `output/reliable_wandb_ddp_smoke_20261116_run01/`
- `output/reliable_wandb_ddp_smoke_20261116.log`

原首版跑通报告及其源码哈希保留为当时的记录；接入监控后的代码证据以本报告配套 JSON 为准。
