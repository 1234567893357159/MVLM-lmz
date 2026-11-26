# MVLM 工具与技能目录

整理时间：2026-11-26（北京时间）。本文是跨分支导航，不代表所有分支都包含全部工具；表中列出的路径相对于对应分支仓库根目录。运行前检查 `git branch --show-current`、实际脚本参数、数据与权重路径、GPU 和输出目录。历史文档和运行状态是快照。

## 项目技能

以下六个技能位于 `.agents/skills/<名称>/SKILL.md`，七个项目分支均有。可以直接向 Codex 描述任务或指定技能名；它们是研究工作流程，不是自动启动实验的命令。

| 技能 | 用途 |
|---|---|
| `project-understanding` | 梳理项目、配置、模块与调用流程。 |
| `paper-reading` | 阅读可核验的论文，区分论文贡献、项目实现与研究假设。 |
| `experiment-design` | 设计假设、对照、消融、数据划分和预算。 |
| `training-analysis` | 分析训练损失、学习率、梯度和验证趋势。 |
| `evaluation` | 核对权重、数据集、首帧、坐标和指标协议后比较实验。 |
| `code-review` | 审查代码正确性、时序因果性、梯度、性能及训练推理一致性。 |

## 各分支共有的入口

| 路径 | 用途 | 使用边界 |
|---|---|---|
| `tracking/train.py` | 原 MVLM 训练入口。 | 配置、初始化、资源与输出目录按分支确认；长训需明确授权。 |
| `tracking/test.py` | 运行跟踪并保存结果。 | 先指定数据集、权重、参数和新的实验 ID。 |
| `tracking/analysis_results.py` | 分析跟踪结果、汇总 AUC/PRE/NPR。 | 先核对首帧、坐标、无效帧和聚合方式；测试集选参属于探索。 |
| `tracking/demo.py` | 视频与自然语言描述的跟踪演示。 | 确认配置与检查点兼容；各分支的功能可能不同。 |

参数和环境准备以各分支 `README.md`、`REPRODUCTION.md` 和脚本 `--help` 为准。`tracking/create_default_local_file.py` 用于生成本机路径，但会覆盖 `lib/local_config.py`，运行前须检查和保留既有配置。

## 各分支专用工具

| 分支 | 主要入口 | 可以复用的任务与限制 |
|---|---|---|
| `codex/mvlm-reproduction` | `tracking/strict_language_eval.py`、`tracking/evaluate_checkpoint_grid.py`、`tracking/check_wandb_logging.py` | 严格语言首帧评测、历史检查点评测与日志检查；部分脚本针对历史运行，先核查路径、检查点与协议。 |
| `codex/world-model` | `tracking/train_world_model.py`、`tracking/evaluate_world_model.py`、`tracking/prepare_world_split.py`、`tracking/check_world_model.py` | 独立世界模型训练、评测、划分和预检；不等同于记忆世界联合模型。 |
| `codex/xmem-memory` | `tracking/prepare_memory_custom_split.py`、`tracking/evaluate_memory_custom.py`、`tracking/check_memory_module.py` | 目标内容记忆自定义划分、评测与模块检查；自定义验证不能与标准测试混用。 |
| `codex/xmem-concat-center` | `tracking/train_memory_concat_run.py`、`tracking/evaluate_memory_concat_run.py`、`tracking/check_memory_concat.py` | 记忆拼接 CENTER 的训练、评测和检查；先确认对应运行配置及输出。 |
| `codex/reliable-memory` | `tracking/train_reliable_memory.py`、`tracking/evaluate_reliable_memory.py`、`tracking/check_reliable_memory.py` | 可靠记忆方案的训练、评测和检查；M6 实际训练使用独立批处理副本及运行源码快照，不能以当前分支源码代替。 |
| `codex/memory-world` | 见下一节 | 内容记忆、世界模型、融合及质量先验的联合训练与评测。 |

`main` 主要使用上面的共有入口。工具存在不等于已在当前机器或当前检查点验证；按需阅读对应分支的说明、脚本和实验记录。

## 记忆与世界模型联合分支

以下路径属于 `codex/memory-world`，不要在其他分支假定它们存在。

| 路径 | 用途与复用边界 |
|---|---|
| `tracking/train_memory_world.py` | A/B/C 通用训练入口；阶段、父检查点、清单及结构须匹配，旧 B/C 配置与完整跟踪器 B 的适配需核实。 |
| `tracking/train_memory_world_tracker_b.py` | 完整跟踪器 B 训练；更新视觉编码器、语言投影、CENTER 和融合层的具体范围以配置及代码为准。 |
| `tracking/train_memory_world_c_audited.py` | 2026-11-26 的 C 训练入口：调用通用训练器并使用 `world_manifest_fast.py` 做完整清单审计；配合本次 C 配置仅描述该次方案。 |
| `tracking/evaluate_memory_world.py` | 固定留出视频的完整在线评测；检查清单/签名，使用项目首帧 GT 输出和整数框协议。先做少量视频或等价预检并确认检查点兼容。 |
| `tracking/evaluate_memory_world_fixed_clips.py` | 固定短片段诊断，主要针对完整 B；短片段损失不等于完整视频 AUC，不应直接假定兼容 C。 |
| `tracking/strict_language_eval.py` | 首帧预测的严格语言评测；对联合模型及特定权重的兼容性应先验证。 |
| `tracking/prepare_world_split.py`、`tracking/world_manifest_fast.py` | 准备官方训练来源的留出清单、加速完整审计；可比实验优先使用相同清单。 |
| `tracking/check_memory_world.py`、`tests/` | CPU 模型预检及自动测试；预检覆盖范围以实际测试为准，不可把 A 预检当作 C 验证。 |
| `tracking/world_monitor.py` | 训练入口的本地状态与可选 W&B 记录；网络或代理是否可用需单独检查。 |
| `tracking/monitor_memory_world_wandb.py` | 为特定已有运行桥接同步 W&B；依赖预期控制目录和日志格式，不能直接用于已有不同记录方式的运行。 |
| `tracking/supervise_memory_world.py`、`tracking/stop_memory_world_at_step.py` | 实验监督和按检查点停止；使用前确认进程、配置及输出，避免影响其他运行。 |

`tracking/evaluate_memory_world_benchmark.py` 写死了历史 B 检查点及输出位置；各 `sweep_*`、特定 `run_*trial*` 和历史续训脚本也可能绑定测试集、GPU 或历史目录。它们适合阅读和按需改造，不应原样用于新实验。训练器的 `training_status.json` 若与实际进程或终端完成日志冲突，应并列核实，不只依赖单一状态文件。

当前分支详细入口见 `BRANCH_MEMORY_WORLD.md` 和 `reports/reproduction/MEMORY_WORLD_TRAINING.md`；实验和评测边界见 `doc/EXPERIMENTS.md`。所有实验须记录代码版本、配置、数据划分、权重、种子、预算、协议及状态；不要将数据、权重或完整运行输出提交 Git。
