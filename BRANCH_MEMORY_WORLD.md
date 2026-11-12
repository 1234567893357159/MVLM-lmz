# 内容记忆与世界模型联合开发分支

更新日期：2026-11-11。分支：`codex/memory-world`；基于 `codex/mvlm-reproduction@ef589b4`，首版联合实现为 `076e29c`。

本分支对应开题报告图 3-2，将内容记忆、未来目标表征预测和候选质量先验接入 MVLM 跟踪过程。2026-11-09 已增加真正批量的 A 阶段路径，并启动两卡正式运行流程；进度以[本次训练记录](reports/reproduction/MEMORY_WORLD_STAGE_A_20261109.md)中的服务器状态入口为准。尚无联合模型的基准精度或提升结论。

本次 A 阶段的 [W&B实时监控](https://wandb.ai/2896287267-hit/mvlm-reproduction/runs/ngk6l0q4)已启用，包含训练/验证指标与两张GPU的资源曲线，前序训练记录已补传。

2026-11-11 根据训练范围审计，将 B 从仅融合层适配改为视觉编码器、语言投影、CENTER 和融合层共同训练；CLIP 本体及 A 的读取/预测/质量模块冻结。两卡各40、全局80、FP32，使用独立入口和输出，不改变运行中 A 的源码签名。旧 B 保留第500步权重作为对照，当前试训和监控入口见[完整跟踪器 B 记录](reports/reproduction/MEMORY_WORLD_TRACKER_B_20261111.md)。A 继续运行，从 A/3000 初始化新 B。

## 阅读入口

| 内容 | 文档 |
|---|---|
| 数据流架构图、模块职责、训练与在线接口 | [联合实现说明](reports/reproduction/MEMORY_WORLD_IMPLEMENTATION.md) |
| 数据划分、损失、参数、A/B/C 命令、续训、多卡与评测 | [训练指南](reports/reproduction/MEMORY_WORLD_TRAINING.md) |
| 已完成检查、维度、梯度与源码签名 | [CPU 预检记录](reports/reproduction/memory_world_preflight_20261108.json) |
| 之前的基线训练与评测归档 | [实验文档索引](reports/reproduction/README.md) |
| 两卡大batch的参数、预检与实际运行入口 | [A阶段训练记录](reports/reproduction/MEMORY_WORLD_STAGE_A_20261109.md) |
| A/B并行、固定A/3000父权重、B批量路径与监控 | [B阶段方案与运行记录](reports/reproduction/MEMORY_WORLD_STAGE_B_PLAN_20261111.md) |
| 当前完整跟踪器 B、损失、显存与验证观察 | [完整跟踪器 B 记录](reports/reproduction/MEMORY_WORLD_TRACKER_B_20261111.md) |

## 相对 MVLM 基线的改动

1. **内容记忆**：保存可靠目标的原始视觉 ROI、位置、帧号和质量；外观与语言双路检索，区分短期与长期记忆。
2. **记忆条件世界模型**：使用截至上一帧的视觉与运动历史、语言及记忆读出，预测下一时刻目标 ROI 表征和框变化。
3. **未来信息融合**：将预测位置映射到当前搜索区域，通过几何约束、外观一致性与门控融合到当前视觉特征。
4. **候选质量先验**：预测 196 个候选网格位置的质量，只在原 MVLM 可靠候选集合内调整排序。
5. **统一时序流程**：训练与在线跟踪共用先定位、后观测写回的流程；定位函数不读取 GT，语言切换重置历史与内容库。

默认融合模式为 `cross_attention`，保持原 512 维 CENTER 输入。另提供 `concat` 成对实验配置，将 512 维当前特征与 256 维读出拼接，并适配扩展 CENTER。

## 训练和配置入口

训练入口为 `tracking/train_memory_world.py`，数据划分入口为 `tracking/prepare_world_split.py`，完整留出视频在线评测入口为 `tracking/evaluate_memory_world.py`。

从完整官方 MVLM 跟踪权重初始化。A 冻结基线学习新增预测模块；当前 B 解冻视觉编码器、语言投影和 CENTER，同时学习融合。记忆库是运行时状态，不是可训练参数。

| 阶段 | 更新模块（默认交叉注意力） | 初始学习率 | 更新次数 |
|---|---|---:|---:|
| A | 记忆读取器、世界模型、质量头 | `1e-4` | 4000 |
| B（当前专用入口） | 视觉编码器、语言投影、CENTER、融合层 | 编码/投影 `1e-5`；头 `5e-5`；融合 `1e-4` | 上限4000 |
| C（原开发配置） | 全部新增模块，编码器与头冻结 | `1e-5` | 4000 |

A → B 构成基本训练流程；B 是首个可在线评测的联合检查点。当前 B 使用 `tracking/train_memory_world_tracker_b.py` 与 `mvlm_TF_memory_world_tracker_b_2gpu.yaml`，恢复原论文的定位 + CM 损失。旧适配层 B/C 配置保留为对照，不能视作当前 B 的训练范围；是否以及如何继续 C，待当前 B 验证后决定。完整命令和参数以[训练指南](reports/reproduction/MEMORY_WORLD_TRAINING.md)为准。

配置位于 `experiments/mvlm/`：

- 默认联合：`mvlm_TF_memory_world_stage_a.yaml` → `mvlm_TF_memory_world_stage_b.yaml` → 可选 `mvlm_TF_memory_world_stage_c.yaml`。
- 通道拼接：`mvlm_TF_memory_world_concat_stage_a.yaml` → `mvlm_TF_memory_world_concat_stage_b.yaml`。
- 无质量头、仅世界模型、仅记忆、仅 CENTER 及两种基线配置，见训练指南的对照实验表。

不同结构的 A/B 配置和检查点必须配套使用。各阶段共享相同数据清单、基线及 MODEL/DATA/TEST 设置；原独立分支的训练权重不能直接拼接为联合检查点。

## 验证与当前边界

- 已有 22 项 CPU 回归检查通过，覆盖小模型 A → 恢复 → B → C、图像片段加载及在线状态重置。
- 实际 FastITPN/CLIP 与官方权重 CPU 预检覆盖两种融合结构；预检条件下初始化输出与基线的最大绝对差为 0，新增模块梯度有效。
- 两进程 CPU/Gloo 检查覆盖 A/B/C，包含一侧缺少有效监督的情况，副本保持一致。
- 2026-11-09：26项CPU回归检查、两卡GPU/NCCL真实大batch短测通过；A阶段正式流程已启动，完整训练和在线精度尚未完成。预检初始化输出一致不代表整段在线预测与旧坐标路径逐帧相同。

当前世界模型实现单步潜空间回归。训练片段长 12 帧，前 4 帧无梯度建库，片段间不继承记忆；这不足以验证长期库的全部作用。片段验证损失最低的检查点还需完整视频在线评测。

## 文档和实验记录维护

- 模块接口、架构与因果读写规则更新到联合实现说明；可运行命令、默认参数及检查点衔接更新到训练指南。
- 新实验单独记录日期、代码版本、实际配置、清单签名、全局 batch、更新次数、评测协议和结果，并从实验索引添加入口。
- 数据、权重、完整训练输出保留在服务器的 `data/`、`models/`、`pretrained/`、`output/`，由 Git 忽略；GitHub 保存代码、文档和小型证据。
- [旧复现分支说明](BRANCH_MVLM_REPRODUCTION.md)及其评测成绩作为历史记录保留，不能作为本联合模型的实验结果。
