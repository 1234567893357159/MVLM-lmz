# 训练、评测和数据文档索引

更新日期：2026-10-28（北京时间）。本目录区分自训模型与历史官方权重实验，避免混用数据划分、首帧协议和测试集调参成绩。

## 当前：VastTrack-only 自训模型

| 内容 | 入口 |
| --- | --- |
| 分支范围与训练模块 | [分支说明](../../BRANCH_MVLM_REPRODUCTION.md) |
| 完成状态、数据清单、评测命令 | [本轮归档](vasttrack_run_20261026/README.md) |
| 十组完整评测汇总 | [CSV](vasttrack_run_20261026/metrics.csv)、[JSON](vasttrack_run_20261026/summary.json) |
| 数据集可用状态快照 | [数据清单](vasttrack_run_20261026/data_inventory.json) |
| 60 个检查点位置、大小及最终权重 SHA256 | [检查点清单](vasttrack_run_20261026/checkpoint_manifest.json) |
| 训练启动、预检与结束记录 | [训练说明](VASTTRACK_TRAINING_20261026.md)、[原始训练元数据](vasttrack_run_20261026/training/) |
| W&B 指标含义及训练页面 | [监控说明](VASTTRACK_WANDB.md) |
| 训练/验证划分与哈希 | [固定划分](vasttrack_split_20261026/)、[划分审计](VASTTRACK_PROTOCOL_AUDIT_20261028.md) |
| 与论文的差距及证据边界 | [分析报告](VASTTRACK_RESULT_GAP_ANALYSIS_20261028.md)、[诊断统计](VASTTRACK_RESULT_GAP_ANALYSIS_20261028.json) |
| 原始报告文件完整性 | [归档 SHA256 清单](vasttrack_run_20261026/archive_manifest.json) |

训练完成 60 轮、75,000 次更新。六个轮次的内部验证和第 60 轮三个外部测试均已完成。该实验只用 VastTrack；内部验证 AUC 70.43 与论文外部测试不可横比。OTB 官方同严格协议对照已完成，TNL2K 官方对照中途取消，没有本轮官方完整成绩；官方 LaSOT 本轮未重跑。

## 历史：官方权重与测试集参数搜索

| 数据集 | 记录 | 历史最佳 AUC |
| --- | --- | ---: |
| OTB99-Lang | [参数搜索](otb_parameter_trials/OTB_TUNING_RESULTS.md) | 60.56 |
| LaSOT | [参数搜索](lasot_parameter_trials/LASOT_TUNING_RESULTS.md)、[早期基线](lasot_baseline/结果说明.md) | 61.06 |
| TNL2K | [参数搜索](tnl2k_parameter_trials/TNL2K_TUNING_RESULTS.md) | 57.62 |

历史最佳参数来自测试集搜索，且首帧处理与本轮严格评测不同，不能作为独立的完整训练复现证据。本轮按用户要求固定迁移这些参数，没有再次扫描测试集。`experiments/mvlm/mvlm_TF.yaml` 是较早的调参配置，不能据文件名把它视为官方默认；原始配置快照见 [original_config.yaml](original_config.yaml)。

## 服务器产物和独立下载任务

原始数据、预训练权重、全部训练权重及帧级预测保留于服务器 `/root/MVLM` 下的 `data/`、`pretrained/`、`models/`、`output/`，由 Git 忽略。本次只归档小型配置、状态、逐序列成绩和诊断统计，没有移动这些目录。

LaSOT/TNL2K 训练数据由独立下载任务管理；归档中的下载百分比只是带时间的快照，下载完成、校验和解压后状态 `ready` 才代表可用于训练。相应下载工具及另一实验的 memory 配置不属于本次归档提交。
