# MVLM 当前项目交接

整理时间：2026-11-12；可靠记忆及状态补充：2026-11-24（北京时间）。当前工作目录 `/root/MVLM`，开发分支 `codex/memory-world`。开始工作先检查 `git status`、现有修改、进程、GPU 和输出；不要覆盖或清理用户已有改动、权重、预测或实验目录。

## 最新补充：可靠记忆 M6（2026-11-24）

- 从官方权重重新开始的 `reliable_m6_candidate_20261121_run02` 已完成4000步。GPU4–7、全局batch80、每卡20、FP32/关闭TF32；仅训练记忆读取/融合与扩展CENTER，编码器和文本侧冻结，BN统计冻结。四卡步数、有限值、冻结审计和参数哈希一致。
- 实际训练目录 `/root/MVLM/work/reliable-batched-prototype` 不是Git工作树；25项源码哈希与训练目录source快照匹配。`codex/reliable-memory` 工作树及memory-world当前代码都不能代替实际快照。
- score门槛0.7、语言margin下限−0.05；90条strict_language开发验证四候选已完成，按三来源mean AUC选4000步75.4383。1000步有3个损坏逐视频JSON，原日志和摘要仍在，尚未补跑；详情见正式记录。
- 4000步权重的OTB48/LaSOT280/TNL2K700旧协议搜索于2026-11-23 01:05:55完成，共12336次视频评测、576个分片，相关进程已退出。各集最高AUC为59.24/61.64/56.93；这不是一套统一参数，也不是独立最终测试。
- 测试曾因调度器停止及TNL2K `Fight_video_6-Done` 空语言中断，已保存故障并恢复。唯一空语言视频按旧空字符串编码处理、结果明确标记；不改写数据，不删减700条。最终以测试目录 `resume_status.json`、`resume_summary.json`、`README.md` 为准，早期状态/两份Markdown是历史快照。
- 当前规则仍会接纳错误目标，例 `chameleon-6` 全部1800帧有效可见，仅首帧IoU≥0.5但有966次提交。尚不能证明新阈值、长期库或记忆本身的因果收益。
- 同旧XMem159条视频离线重汇总：M6 mean AUC60.04，旧仅记忆最佳60.52、记忆＋头最佳60.80；参数选择及训练不同，不是受控消融。旧XMem训练与当前完整标准测试有LaSOT196/TNL2K489条交集，不可直接全量排名。
- 后续优先同协议官方基线/无记忆微调对照、2×2写入阈值完整视频对照与错误写入诊断。`work/m6_first_frame_development`仅准备了隔离副本，没有据此重训。W&B上传未确认。

正式证据：[可靠记忆M6实验记录](../reports/reproduction/RELIABLE_MEMORY_M6_RESULTS_20261124.md)；服务器结果表：[测试README](/root/MVLM/output/reliable_m6_legacy_sweep_20261122_run01/README.md)。下文联合模型A/B/C保留原实验范围，不等于M6。

## 目标与设计

以 MVLM 为基础，研究语言条件目标跟踪：使用每视频独立的目标内容记忆与历史，由世界模型预测下一时刻目标信息，将预测与当前视觉特征融合，通过定位头和质量先验选择目标框。当前记忆读出送入世界模型，再通过预测影响定位；历史和记忆在定位后由可靠的原始观测更新。实现细节见[联合架构](../reports/reproduction/MEMORY_WORLD_IMPLEMENTATION.md)。

开题报告正文、PDF 和架构原图目前不在 `tex/`，该目录仅见编译日志。现有架构对应关系来自已有项目记录；不能声称已重新核对原图。

## 当前训练与评测

- A 阶段完成4000步；完整跟踪器 B 从 **A/3000** 初始化并完成4000步。旧仅融合 B 是独立的已停止试验，不能与完整 B 混淆。
- 当前 B 更新视觉编码器、语言投影、CENTER 和未来融合，冻结 CLIP 本体和 A 的记忆读取器、世界模型、质量头。C 尚未训练，旧开发版 C 不能直接视为适配了当前 B。
- 基线、B1025、B4000 的 OTB 48 / LaSOT 280 / TNL2K 700 完整视频测试均已完成。相同原始参数下 AUC：基线 56.84 / 57.45 / 55.31；B1025 56.28 / 57.65 / 56.25；B4000 55.54 / 57.33 / 56.56。协议、PRE/NPR 及结果限制见[训练与结果](../reports/reproduction/MEMORY_WORLD_TRAINING_AND_RESULTS_20261112.md)和[逐视频分析](../reports/reproduction/MEMORY_WORLD_B_VIDEO_ANALYSIS_20261112.md)。
- OTB 12组参数、两个 B 权重各48条的探索搜索已完成；B1025 最佳 AUC 58.96（alpha09），B4000 最佳 59.07（alpha07）。这是使用 OTB 测试集选参的有限探索，不是独立最终成绩；其他参数组合没有全部重测同参数官方基线。见[完整搜索](../reports/reproduction/MEMORY_WORLD_OTB_SWEEP_20261112.md)。
- 2026-11-12 23:51（北京时间）启动 LaSOT/TNL2K 参数搜索：每数据集12组、两个B权重，每组完整评测280/700条；共23,520次视频测试，8张GPU每卡两个工作进程。输出 `output/memory_world_b_lasot_tnl2k_sweep_20261112_run01/`，实时状态与完成组结果见[搜索报告](../reports/reproduction/MEMORY_WORLD_LASOT_TNL2K_SWEEP_20261112.md)。这是启动快照；后续已于2026-11-13完成，最终结果见上述搜索报告。当时参考为E66历史成绩；E66后续独立12组参数搜索于11月14日完成，见[结果](../reports/reproduction/E66_PARAMETER_SWEEP_20261113.md)。
- 2026-11-13 增加长数据集搜索并发：原16个工作进程接续运行，新调度器 `tracking/accelerate_memory_world_long_sweep.py` 将每卡并发由2增至4，总32个；原权重、推理代码、参数网格和已有视频结果保留。实际并发见输出目录 `concurrency.json` 和 `status.json`，调整记录见 `concurrency_events.jsonl`。
- 测试结果不能独立证明记忆或世界预测器的因果贡献；B 也更新了原跟踪器。后续优先在固定留出视频分析逐帧表现、做仅原跟踪器训练和各模块消融，再决定是否适配 C。

主要产物：`output/memory_world_stage_a_2gpu_20261109_run02/`、`output/memory_world_tracker_b_2gpu_b80_20261111_run01/`、`output/memory_world_b_benchmark_20261112_run01/` 和 `output/memory_world_b_otb_sweep_20261112_run01/`。输出、数据和权重不随 Git 保存。

## 工作区与监控边界

- 历史实验文档 `reports/reproduction/PAPER_CONTINUATION_TESTS_20261103.md` 在整理前已有用户本地修改，应保留；另有本次会话新增的分析、脚本和文档改动，提交前按实际 `git status` 确认。
- B 原 W&B 页面曾因网络问题只同步早期数据；恢复运行是否真正传齐需核对远端，不能仅凭本地队列判断。既有授权只覆盖数值指标、超参数和 GPU 资源；W&B 客户端可能自动生成运行日志和依赖列表，应检查实际上传文件。
- 已有 `SESSION_HANDOFF.md` 是早期历史快照，不代表当前状态。分支入口见[分支导航](BRANCHES.md)，历史和当前实验结果见[实验总览](EXPERIMENTS.md)。
