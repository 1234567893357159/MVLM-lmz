# 实验与文档索引

本索引区分当前联合模型和历史基线实验。文档中的进度、PID 和显卡分配均为带日期的快照；实时状态需同时检查状态文件、进程与 GPU。

## 当前：记忆库与世界模型

截至 2026-11-12，A 完成 4000 步，完整跟踪器 B 从 A/3000 初始化并完成 4000 步，C 尚未训练。基线、B1025、B4000 的完整视频评测正在进行；OTB 的三组已完成，LaSOT 和 TNL2K 尚无三组完整结果。

| 内容 | 入口 |
| --- | --- |
| 实际训练设置、诊断和测试结果 | [训练与结果总记录](MEMORY_WORLD_TRAINING_AND_RESULTS_20261112.md) |
| 模块、数据流和实现边界 | [联合实现说明](MEMORY_WORLD_IMPLEMENTATION.md)、[分支说明](../../BRANCH_MEMORY_WORLD.md) |
| 数据、损失、命令和对照方案 | [训练指南](MEMORY_WORLD_TRAINING.md)；其中原开发版 C 尚未适配当前 B |
| A 的配置、预检和运行记录 | [A 阶段记录](MEMORY_WORLD_STAGE_A_20261109.md) |
| 当前完整跟踪器 B 的设置和诊断 | [B 阶段记录](MEMORY_WORLD_TRACKER_B_20261111.md) |
| 旧仅融合 B 的方案和范围审计 | [旧 B 方案](MEMORY_WORLD_STAGE_B_PLAN_20261111.md)、[范围审计](MEMORY_WORLD_STAGE_B_SCOPE_AUDIT_20261111.md) |
| 检查证据 | [CPU 预检](memory_world_preflight_20261108.json)、[B 两卡预检](memory_world_stage_b_preflight_20261111.json) |

## 历史：三数据集复现与续训

- [60 轮训练结果](PAPER_TRAINING_RESULTS_20261101.md)、[E61–E70 分析](PAPER_CONTINUATION_ANALYSIS_20261103.md)、[E71–E80 训练](PAPER_CONTINUATION_TRAINING_20261103.md)和[逐轮评测记录](PAPER_CONTINUATION_TESTS_20261103.md)。
- [检查点测试](PAPER_CHECKPOINT_TESTS_20261101.md)、[论文差距分析](PAPER_TEST_GAP_ANALYSIS_20261101.md)、[训练方案](PAPER_REPRODUCTION_PLAN_20261030.md)。
- 这些历史实验使用过测试集参数探索；结果及权重选择不能作为独立测试证据，亦不能直接与当前联合实验作同协议比较。

## 历史：VastTrack 单数据集

训练完成 60 轮、75,000 次更新；它不是论文三数据集训练的完整复现。入口：[分支说明](../../BRANCH_MVLM_REPRODUCTION.md)、[训练与评测归档](vasttrack_run_20261026/README.md)、[数据划分审计](VASTTRACK_PROTOCOL_AUDIT_20261028.md)、[差距分析](VASTTRACK_RESULT_GAP_ANALYSIS_20261028.md)、[W&B 说明](VASTTRACK_WANDB.md)。

## 历史：官方权重与测试集参数搜索

[OTB](otb_parameter_trials/OTB_TUNING_RESULTS.md)、[LaSOT](lasot_parameter_trials/LASOT_TUNING_RESULTS.md)、[TNL2K](tnl2k_parameter_trials/TNL2K_TUNING_RESULTS.md) 的历史最优 AUC 分别为 60.56、61.06、57.62。参数经过对应测试集搜索，不能作为独立最终成绩。原始配置快照见 [original_config.yaml](original_config.yaml)。

## 数据和服务器产物

数据集、权重、完整训练输出及帧级预测保留在服务器，不随 Git 上传。文档中的 `/root/MVLM/output/` 路径是本地证据入口，不是 GitHub 附件；历史下载状态也不代表当前状态。连接与代理说明见 [SSH 文档](../../SSH_DESKTOP_PROXY.md)。
