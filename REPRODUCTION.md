# MVLM 复现工作区

本仓库保留官方 MVLM 的 Git 历史，保存环境适配、训练与评测工具、实验配置和结果证据。历史 `codex/mvlm-reproduction` 分支的实验导航见[实验与文档索引](reports/reproduction/README.md)。

2026-11-03 更新：三数据集训练已完成60轮，并分两阶段续训至80轮（100,000次更新）。LaSOT/VastTrack/TNL2K 按1:1:1抽样，GPU4、5每卡batch40、全局80；当时 GPU6、7 正在逐轮评测 E71～E80（历史快照）。详见[训练与监控记录](reports/reproduction/PAPER_CONTINUATION_TRAINING_20261103.md)、[评测进度快照](reports/reproduction/PAPER_CONTINUATION_TESTS_20261103.md)及[新W&B运行](https://wandb.ai/2896287267-hit/mvlm-reproduction/runs/jrhbvcza)。E61～E70完整评测中，E66三集平均AUC最高（58.381）；[分析报告](reports/reproduction/PAPER_CONTINUATION_ANALYSIS_20261103.md)记录了小幅续训收益和测试集选择的限制。在线W&B曾中断，训练完成情况以本地日志和检查点核验为准。

## 可靠记忆库 M6 的可复现记录

2026-11-24复核：可靠记忆M6完成4000步、四候选90视频验证及三测试集36组参数搜索；运行代码来自批处理副本与source快照，初始化/清单/源码/最佳权重哈希、冻结范围、空语言例外和恢复经过见[正式实验记录](reports/reproduction/RELIABLE_MEMORY_M6_RESULTS_20261124.md)。本分支当前代码与工作树HEAD都不能单独标识训练版本。

90视频strict_language开发验证与首帧GT/整数框的旧协议测试分开报告。测试目录早期状态文件为历史快照，最终以 `resume_status.json`、`resume_summary.json`、`README.md` 为准；W&B上传未获确认。

## 历史官方权重实验

| 数据集 | 较早默认/基线 AUC | 测试集调参最佳 AUC |
| --- | ---: | ---: |
| LaSOT | 57.77 | 61.06 |
| OTB99-Lang | 56.48 | 60.56 |
| TNL2K | 55.42 | 57.62 |

这些历史结果采用原有首帧处理方式，且最佳值来自测试集调参；与自训结果的直接对照需要说明协议差异。历史记录入口见[总索引](reports/reproduction/README.md)。

## 数据、路径与大文件

数据集、模型权重和完整训练输出不加入 Git；各实验具体状态见对应记录。

本机根目录 `/root/MVLM`，Python 环境 `/root/miniconda3/envs/mvlm/bin/python`。换机器时先调整 `lib/local_config.py` 中的数据和输出路径，准备预训练权重及检查点。依赖见 `requirements.txt`、`requirements-wandb.txt`。

原始运行配置与状态保留当时绝对路径；可复用评测配置与验证入口引用仓库内版本化清单，清单哈希保持不变。历史下载任务的状态以实时记录为准；归档清单只表示采集时的快照。当前联合实验的数据划分见[训练与结果记录](reports/reproduction/MEMORY_WORLD_TRAINING_AND_RESULTS_20261112.md)。
