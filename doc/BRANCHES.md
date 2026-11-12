# 分支与文档导航

本文件的本机路径是 2026-11-12 的工作树快照。不同机器或工作树清理后路径可能变化；先用 `git worktree list`、`git branch -a` 核对。表中的仓库内路径以对应分支为准；不必为读文档切换当前分支，可用 `git show <分支>:<路径>`。

| 分支 | 用途 | 本机工作目录 |
|---|---|---|
| `codex/memory-world` | 内容记忆、世界模型、融合与质量头联合开发 | `/root/MVLM` |
| `codex/world-model` | 不含内容记忆的独立世界模型实验 | `/root/MVLM/work/world-model` |
| `codex/xmem-memory` | 目标内容短/长期记忆实验 | `/root/MVLM/work/xmem-wandb` |
| `codex/xmem-concat-center` | 内容记忆拼接扩展 CENTER 实验 | `/root/.codex/worktrees/xmem-concat-center/MVLM` |
| `codex/mvlm-reproduction` | 原模型复现、VastTrack 和三数据集训练 | 未单独挂载；部分资料在当前分支 |
| `main` | 原始项目和早期官方权重参数实验 | 未单独挂载 |

## 联合模型 `codex/memory-world`

- [分支说明](../BRANCH_MEMORY_WORLD.md)、[联合架构](../reports/reproduction/MEMORY_WORLD_IMPLEMENTATION.md)、[训练指南](../reports/reproduction/MEMORY_WORLD_TRAINING.md)。
- [A 阶段](../reports/reproduction/MEMORY_WORLD_STAGE_A_20261109.md)、[当前完整跟踪器 B](../reports/reproduction/MEMORY_WORLD_TRACKER_B_20261111.md)、[训练与结果总记录](../reports/reproduction/MEMORY_WORLD_TRAINING_AND_RESULTS_20261112.md)。
- [完整视频分析](../reports/reproduction/MEMORY_WORLD_B_VIDEO_ANALYSIS_20261112.md)、[历史参数迁移](../reports/reproduction/MEMORY_WORLD_OTB_HISTORICAL_PARAMS_20261112.md)、[OTB 多参数搜索](../reports/reproduction/MEMORY_WORLD_OTB_SWEEP_20261112.md)。

## 独立世界模型 `codex/world-model`

分支内路径：`WORLD_MODEL_HANDOFF.md`（主要交接）、`分支说明.md`、`reports/reproduction/WORLD_MODEL_IMPLEMENTATION.md`、`WORLD_CROSSDATASET_EVALUATION_20261026.md`、`WORLD_PARAMETER_TRANSFER_20261026.md`。后两个同样位于 `reports/reproduction/`。

## 内容记忆 `codex/xmem-memory`

分支内路径：`codex-xmem-memory分支说明.md`；`reports/reproduction/MEMORY_IMPLEMENTATION.md`、`MEMORY_CUSTOM_TRAINING_20261018.md`、`MEMORY_CUSTOM_FINAL_RESULTS.md`、`MEMORY_DIAGNOSTICS_20261019.md`。后面三个与第一个均位于 `reports/reproduction/`。自定义验证集结果不能直接与标准测试集混比。

## 记忆拼接 `codex/xmem-concat-center`

分支内路径：`codex-xmem-concat-center分支说明.md`；`reports/reproduction/MEMORY_CONCAT_CENTER.md`、`MEMORY_CONCAT_TRAINING_20261103.md`、`MEMORY_CONCAT_ANALYSIS_20261104.md`、`MEMORY_CONCAT_RESUME6000_20261104.md`。后面三个与第一个均位于 `reports/reproduction/`。

## 原模型复现及早期文档

- `codex/mvlm-reproduction`：`BRANCH_MVLM_REPRODUCTION.md`、`reports/reproduction/README.md`；当前分支也保留[复现分支说明](../BRANCH_MVLM_REPRODUCTION.md)及[实验索引](../reports/reproduction/README.md)。
- `main`：`README.md`、`REPRODUCTION.md`、`SESSION_HANDOFF.md`，及 `reports/reproduction/` 下三个数据集的参数搜索记录。
- [历史会话交接](../SESSION_HANDOFF.md)属于早期快照；世界模型分支的 `WORLD_MODEL_HANDOFF.md` 也是该分支历史记录，不能替代[当前交接](HANDOFF.md)。

跨分支训练和结果比较先看[实验总览](EXPERIMENTS.md)，不同训练数据、首帧协议和测试集调参结果不能直接混排。
