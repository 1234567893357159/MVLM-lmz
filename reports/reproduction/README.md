# 实验与文档索引

项目交接和各分支导航见 [doc 文档入口](../../doc/README.md)，跨分支结果见 [实验总览](../../doc/EXPERIMENTS.md)。

本索引区分可靠记忆M6、联合模型和历史基线实验。文档中的进度、PID 和显卡分配均为带日期的快照；实时状态需同时检查状态文件、进程与 GPU。

## 可靠记忆库 M6（2026-11-24 核实）

[训练、验证、完整测试与比较记录](RELIABLE_MEMORY_M6_RESULTS_20261124.md) 为正式入口：从官方权重完成4000步，四候选90条strict_language验证选4000步75.4383；旧协议三测试集各12组搜索已于2026-11-23 01:05:55完成，最优AUC为OTB59.24、LaSOT61.64、TNL2K56.93。两套结果不混排，测试集选参属于探索。

文档同时记录实际批处理副本/源码哈希、训练曲线、1000步3条损坏产物、调度恢复、唯一空语言视频、错误记忆接纳案例及旧XMem159条同视频重汇总（M6平均60.04；参数/训练未完全对齐）。最新服务器完整表：[测试README](/root/MVLM/output/reliable_m6_legacy_sweep_20261122_run01/README.md)。数据、权重和输出不加入Git。

## 当前：记忆库与世界模型

截至 2026-11-12，A 完成 4000 步，完整跟踪器 B 从 A/3000 初始化并完成 4000 步，C 尚未训练。基线、B1025、B4000 的三数据集完整视频评测已完成；结果和逐视频分析见下方记录。

| 内容 | 入口 |
| --- | --- |
| 实际训练设置、诊断和测试结果 | [训练与结果总记录](MEMORY_WORLD_TRAINING_AND_RESULTS_20261112.md) |
| 完整视频结果及逐视频差异 | [B 视频分析](MEMORY_WORLD_B_VIDEO_ANALYSIS_20261112.md) |
| OTB 历史参数迁移（探索性） | [参数对照](MEMORY_WORLD_OTB_HISTORICAL_PARAMS_20261112.md) |
| OTB 多参数搜索（探索性） | [完整搜索结果](MEMORY_WORLD_OTB_SWEEP_20261112.md) |
| LaSOT / TNL2K 多参数搜索（探索性，结果随进度更新） | [参数搜索报告](MEMORY_WORLD_LASOT_TNL2K_SWEEP_20261112.md) |
| 长数据集搜索并发调整 | [并发与负载记录](MEMORY_WORLD_SWEEP_CONCURRENCY_20261113.md) |
| 模块、数据流和实现边界 | [联合实现说明](MEMORY_WORLD_IMPLEMENTATION.md)、[分支说明](../../BRANCH_MEMORY_WORLD.md) |
| 数据、损失、命令和对照方案 | [训练指南](MEMORY_WORLD_TRAINING.md)；其中原开发版 C 尚未适配当前 B |
| A 的配置、预检和运行记录 | [A 阶段记录](MEMORY_WORLD_STAGE_A_20261109.md) |
| 当前完整跟踪器 B 的设置和诊断 | [B 阶段记录](MEMORY_WORLD_TRACKER_B_20261111.md) |
| 旧仅融合 B 的方案和范围审计 | [旧 B 方案](MEMORY_WORLD_STAGE_B_PLAN_20261111.md)、[范围审计](MEMORY_WORLD_STAGE_B_SCOPE_AUDIT_20261111.md) |
| 检查证据 | [CPU 预检](memory_world_preflight_20261108.json)、[B 两卡预检](memory_world_stage_b_preflight_20261111.json) |

OTB 搜索的参考配置为 `ALPHA_CORR=0.5`、`TAU=0.03`、`OMEGA=0.4`、`LAMBDA=0.1`、`PSI_OUT=0.1`、`QUALITY_WEIGHT=0.25`。除两组 OMEGA/LAMBDA 联合变化外，各候选每次改变一项；QUALITY_WEIGHT 控制候选质量重排，不是训练损失权重。各权重测试 12 组参数，另有历史参考组；每组 48 条视频，采用项目首帧 GT 计分。搜索及汇总入口分别为 `tracking/sweep_memory_world_otb.py` 和 `tracking/report_memory_world_otb_sweep.py`；输出目录中的 manifest、status、summary 和逐视频结果记录完整配置及证据。本轮未向 W&B 上传。测试集选参仅为探索，其他参数组没有同步重测同参数基线。

2026-11-12 21:23（北京时间）的中途快照已被同日 21:48 的完整结果取代；旧快照中记录的 GPU 和进程状态不代表当前状态。

## 历史：三数据集复现与续训

- [60 轮训练结果](PAPER_TRAINING_RESULTS_20261101.md)、[E61–E70 分析](PAPER_CONTINUATION_ANALYSIS_20261103.md)、[E71–E80 训练](PAPER_CONTINUATION_TRAINING_20261103.md)和[逐轮评测记录](PAPER_CONTINUATION_TESTS_20261103.md)。
- [检查点测试](PAPER_CHECKPOINT_TESTS_20261101.md)、[论文差距分析](PAPER_TEST_GAP_ANALYSIS_20261101.md)、[训练方案](PAPER_REPRODUCTION_PLAN_20261030.md)。
- 这些历史实验使用过测试集参数探索；结果及权重选择不能作为独立测试证据，亦不能直接与当前联合实验作同协议比较。

## 历史：官方权重与测试集参数搜索

[OTB](otb_parameter_trials/OTB_TUNING_RESULTS.md)、[LaSOT](lasot_parameter_trials/LASOT_TUNING_RESULTS.md)、[TNL2K](tnl2k_parameter_trials/TNL2K_TUNING_RESULTS.md) 的历史最优 AUC 分别为 60.56、61.06、57.62。参数经过对应测试集搜索，不能作为独立最终成绩。原始配置快照见 [original_config.yaml](original_config.yaml)。

## 数据和服务器产物

数据集、权重、完整训练输出及帧级预测保留在服务器，不随 Git 上传。文档中的 `/root/MVLM/output/` 路径是本地证据入口，不是 GitHub 附件；历史下载状态也不代表当前状态。连接与代理说明见 [SSH 文档](../../SSH_DESKTOP_PROXY.md)。
