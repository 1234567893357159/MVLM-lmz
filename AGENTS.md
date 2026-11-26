# 项目协作规则

默认使用中文回复。此仓库基于原始 MVLM，用于无模板自然语言目标跟踪的复现和研究；当前代码还探索内容记忆、未来表征预测、融合及候选质量先验。区分原论文贡献、个人实现、实验观察和研究假设。先读 [文档入口](doc/README.md)、[实验总览](doc/EXPERIMENTS.md)；联合模型按需读 [分支说明](BRANCH_MEMORY_WORLD.md) 和 [实现说明](reports/reproduction/MEMORY_WORLD_IMPLEMENTATION.md)。文档中的日期和运行状态是快照，冲突时并列列明来源，结合代码和实际运行证据核实，不猜测。

## 代码与运行

- Python、PyTorch；依赖见 [requirements.txt](requirements.txt) 和 [requirements-wandb.txt](requirements-wandb.txt)，本机环境及路径见 [复现说明](REPRODUCTION.md) 和 `lib/local_config.py`；迁移机器时先核实路径、数据和权重是否存在。
- `lib/models/mvlm/` 为模型、记忆和世界模型；`lib/train/` 为数据、actor 和训练；`lib/test/` 为跟踪器、数据集和指标；`experiments/mvlm/` 为配置；`tracking/` 为入口；`tests/` 为测试；`reports/reproduction/` 为实验文档和小型证据。数据、权重及完整输出通常在 `data/`、`models/`、`pretrained/`、`output/`，不要扫描其大文件内容。
- 常规训练入口 `tracking/train.py`，跟踪推理入口 `tracking/test.py`，结果分析入口 `tracking/analysis_results.py`，命令和参数见 [README](README.md)。联合模型训练入口 `tracking/train_memory_world.py`、当前完整跟踪器 B 入口 `tracking/train_memory_world_tracker_b.py`，评测入口及阶段配置见 [训练指南](reports/reproduction/MEMORY_WORLD_TRAINING.md) 和 [分支说明](BRANCH_MEMORY_WORLD.md)。实际配置、权重、数据和版本须逐次核实。

## 修改与验证

- 修改前先阅读相关模块、调用链及配置，保持既有接口和实验可复现性；避免无关重构。按改动范围选择轻量测试或静态检查，优先参考 `tests/`；报告实际运行的命令、结果及未验证部分。不得伪造测试通过、指标或科研结论。
- 默认不执行长时间训练、大规模评测或实验；先明确资源、输出及用户授权。未经允许不得删除数据、权重、实验结果或覆盖既有运行输出。
- 记录实验的代码版本、时间、配置及其改动、数据集和划分/清单、权重及初始化、种子、硬件、训练预算、指标、评测协议、运行状态、证据路径及限制；缺失项标注未知。沿用 [实验索引](reports/reproduction/README.md)，不要将计划、部分结果或历史快照写成已完成。
- 比较前核对数据集/划分、序列数、首帧处理、权重、配置、坐标路径、指标与聚合方式。测试集选参是探索性结果，不等同独立泛化；短片段损失不等同完整视频指标。
- 操作 Git 前检查状态，保留用户已有修改及历史；不擅自重置、清理、覆盖或提交。不要将数据、权重或运行输出加入版本控制。

## 项目 Skills

按任务读取 `.agents/skills/` 下最相关的 `SKILL.md`：项目理解、论文阅读、实验设计、训练分析、评测和代码审查分别有对应技能；用户明确指定时优先使用。按需读取所引用的项目资料，不把某技能当作无关任务的强制流程。项目文件若缺失或记录冲突，说明证据与不确定性。

## 开题报告保护

- `tex/` 是用户的本科毕业论文开题报告 LaTeX 源文件，题为《基于记忆与想像建模的视觉-语言目标跟踪方法研究》。这是重要的个人研究资料，永久保留；未经用户明确要求，不得删除或覆盖。目录包含报告源文件、模板、参考文献、插图及原有文档；编译中间产物不应加入版本控制。报告的研究方案是计划，不等同于已完成的实验结果。
