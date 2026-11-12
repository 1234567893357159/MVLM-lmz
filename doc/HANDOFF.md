# MVLM 当前项目交接

整理时间：2026-11-12（北京时间）。当前工作目录 `/root/MVLM`，开发分支 `codex/memory-world`。开始工作先检查 `git status`、现有修改、进程、GPU 和输出；不要覆盖或清理用户已有改动、权重、预测或实验目录。

## 目标与设计

以 MVLM 为基础，研究语言条件目标跟踪：使用每视频独立的目标内容记忆与历史，由世界模型预测下一时刻目标信息，将预测与当前视觉特征融合，通过定位头和质量先验选择目标框。当前记忆读出送入世界模型，再通过预测影响定位；历史和记忆在定位后由可靠的原始观测更新。实现细节见[联合架构](../reports/reproduction/MEMORY_WORLD_IMPLEMENTATION.md)。

开题报告正文、PDF 和架构原图目前不在 `tex/`，该目录仅见编译日志。现有架构对应关系来自已有项目记录；不能声称已重新核对原图。

## 当前训练与评测

- A 阶段完成4000步；完整跟踪器 B 从 **A/3000** 初始化并完成4000步。旧仅融合 B 是独立的已停止试验，不能与完整 B 混淆。
- 当前 B 更新视觉编码器、语言投影、CENTER 和未来融合，冻结 CLIP 本体和 A 的记忆读取器、世界模型、质量头。C 尚未训练，旧开发版 C 不能直接视为适配了当前 B。
- 基线、B1025、B4000 的 OTB 48 / LaSOT 280 / TNL2K 700 完整视频测试均已完成。相同原始参数下 AUC：基线 56.84 / 57.45 / 55.31；B1025 56.28 / 57.65 / 56.25；B4000 55.54 / 57.33 / 56.56。协议、PRE/NPR 及结果限制见[训练与结果](../reports/reproduction/MEMORY_WORLD_TRAINING_AND_RESULTS_20261112.md)和[逐视频分析](../reports/reproduction/MEMORY_WORLD_B_VIDEO_ANALYSIS_20261112.md)。
- OTB 12组参数、两个 B 权重各48条的探索搜索已完成；B1025 最佳 AUC 58.96（alpha09），B4000 最佳 59.07（alpha07）。这是使用 OTB 测试集选参的有限探索，不是独立最终成绩；其他参数组合没有全部重测同参数官方基线。见[完整搜索](../reports/reproduction/MEMORY_WORLD_OTB_SWEEP_20261112.md)。
- 测试结果不能独立证明记忆或世界预测器的因果贡献；B 也更新了原跟踪器。后续优先在固定留出视频分析逐帧表现、做仅原跟踪器训练和各模块消融，再决定是否适配 C。

主要产物：`output/memory_world_stage_a_2gpu_20261109_run02/`、`output/memory_world_tracker_b_2gpu_b80_20261111_run01/`、`output/memory_world_b_benchmark_20261112_run01/` 和 `output/memory_world_b_otb_sweep_20261112_run01/`。输出、数据和权重不随 Git 保存。

## 工作区与监控边界

- 历史实验文档 `reports/reproduction/PAPER_CONTINUATION_TESTS_20261103.md` 在整理前已有用户本地修改，应保留；另有本次会话新增的分析、脚本和文档改动，提交前按实际 `git status` 确认。
- B 原 W&B 页面曾因网络问题只同步早期数据；恢复运行是否真正传齐需核对远端，不能仅凭本地队列判断。既有授权只覆盖数值指标、超参数和 GPU 资源；W&B 客户端可能自动生成运行日志和依赖列表，应检查实际上传文件。
- 已有 `SESSION_HANDOFF.md` 是早期历史快照，不代表当前状态。分支入口见[分支导航](BRANCHES.md)，历史和当前实验结果见[实验总览](EXPERIMENTS.md)。
