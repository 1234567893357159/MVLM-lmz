# 训练、评测和数据文档索引

2026-11-12 更新：A、B 均已完成 4000 步，B 使用 A 第 3000 步初始化；三组模型的完整视频测试正在进行。[训练过程与测试结果总记录](MEMORY_WORLD_TRAINING_AND_RESULTS_20261112.md)汇总了实际参数、数据划分、验证结果和 OTB 测试快照；实时进度以测试输出中的 `status.json` 为准。下方 2026-11-11 文字保留为当时的历史状态。

## 当前联合开发分支（2026-11-11）

`codex/memory-world` 已完成联合实现、批量训练改造及两卡GPU/NCCL短测，已启动A阶段正式运行流程，尚无联合模型精度结论。

2026-11-11：A 在 GPU0、1 继续运行。旧仅融合 B 已保存第500步并停止；新完整跟踪器 B 从 A/3000 初始化，使用 GPU2、3、每卡40、全局80，训练视觉编码器、语言投影、CENTER 与融合层，恢复定位 + CM 损失。两卡 FP32 短测已通过，当前记录和监控见[完整跟踪器 B](MEMORY_WORLD_TRACKER_B_20261111.md)。

- [分支说明](../../BRANCH_MEMORY_WORLD.md)：分支范围、默认流程、完成状态与维护约定。
- [联合实现说明](MEMORY_WORLD_IMPLEMENTATION.md)：开题架构对应、数据流、模块接口和验证边界。
- [训练指南](MEMORY_WORLD_TRAINING.md)：数据划分、损失与参数、A/B/C 命令、断点恢复、多卡及评测。
- [CPU 预检记录](memory_world_preflight_20261108.json)：22 项回归检查、实际骨干接口与双进程检查。
- [两卡A阶段训练记录](MEMORY_WORLD_STAGE_A_20261109.md)：26项回归检查、batch扫描、真实GPU短测、正式配置和实时状态入口。
- [B阶段并行训练方案与记录](MEMORY_WORLD_STAGE_B_PLAN_20261111.md)：固定A/3000父权重、B批量计算、batch80、监控与运行入口。
- [B阶段两卡预检证据](memory_world_stage_b_preflight_20261111.json)：真实更新、GIoU/L1/Focal、冻结检查、副本一致性与父权重签名。
- [完整跟踪器 B 重新试训](MEMORY_WORLD_TRACKER_B_20261111.md)：当前训练范围、batch80、逐帧反向、约22GB显存与正式验证观察。

下方三数据集和 VastTrack-only 成绩均为此前基线实验，不代表本联合模型的结果。

## 历史三数据集实验（2026-11-03）

- [数据、参数与60轮训练结果](PAPER_TRAINING_RESULTS_20261101.md)。
- [E61～E70最终分析](PAPER_CONTINUATION_ANALYSIS_20261103.md)：E66三集平均AUC最高，58.381。
- [E71～E80训练记录](PAPER_CONTINUATION_TRAINING_20261103.md)：E80 / 100,000次更新已完成。
- [E71～E80逐轮评测进度快照](PAPER_CONTINUATION_TESTS_20261103.md)：评测继续在服务器运行，GitHub文档截至提交时。

仓库包含代码、配置、指标汇总和图表；数据集、权重、帧级预测及完整W&B逐步/系统采样记录保留在服务器。指向`output/`或服务器绝对路径的链接是本地运行证据，不是GitHub附件。下方VastTrack-only内容为此前独立实验归档。


2026-11-02：已从E60继续训练到E70，物理GPU4、5、每卡40、全局80；[续训运行记录](PAPER_CONTINUATION_TRAINING_20261102.md)、[新W&B页面](https://wandb.ai/2896287267-hit/mvlm-reproduction/runs/lu6xjpyl)、[E65/E70评测记录](PAPER_CONTINUATION_TESTS_20261102.md)。

阶段性分析：[三数据集测试差距与同视频官方权重对照](PAPER_TEST_GAP_ANALYSIS_20261101.md)，附AUC曲线和首个实际预测/严重定位失败统计。

训练完成报告已补充W&B在线/本地历史核对：在线记录止于第42轮中段，本地保留完整60轮；资源统计、精确逐轮数据和同步差异均已归档。

2026-11-01：三数据集训练已正常完成60轮；[训练完成报告](PAPER_TRAINING_RESULTS_20261101.md)汇总数据、实际参数、60轮Loss/IoU曲线、耗时和最终检查点审计，附[逐轮CSV](paper_training_results_20261101/epoch_metrics.csv)。

2026-11-01：用户已改为对三数据集训练检查点每5轮粗测，再在高分邻域逐轮细测；使用 main 首帧真实框协议和历史最优参数评测，现使用 GPU 4、5、6、7 两组并行；进度和每轮三集结果持续更新于 [全轮次评测记录](PAPER_CHECKPOINT_TESTS_20261101.md)。

更新日期：2026-10-28（北京时间）。本目录区分自训模型与历史官方权重实验，避免混用数据划分、首帧协议和测试集调参成绩。

2026-10-30：[三数据集重新训练方案](PAPER_REPRODUCTION_PLAN_20261030.md)已进入[正式训练](PAPER_REPRODUCTION_TRAINING_20261030.md)，仅使用 GPU 4、5、每卡 batch 40、全局 80，总计 4 个加载 worker。短测后重新初始化，W&B 已核实实际指标上传：[监控链接](https://wandb.ai/2896287267-hit/mvlm-reproduction/runs/7ln28ook)。配置为 `mvlm_TF_paper_reproduction`，跳过 1 条异常 TNL2K 序列，保留源文件；[早期准备报告](paper_reproduction_preparation_20261030.json)是调整卡数前的历史快照。

## 历史：VastTrack-only 自训模型

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

LaSOT/TNL2K 训练数据由独立下载任务管理；归档中的下载百分比只是带时间的快照，下载完成、校验和解压后状态 `ready` 才代表可用于训练。本次同步还包括下载准备工具、断点续传测试及相关探测配置；它们不改变此前 VastTrack-only 归档的训练结果。

- [最终续训收益分析](PAPER_CONTINUATION_ANALYSIS_20261103.md)：E61～E70最终对照与是否继续训练的判断。

- [E70→E80续训实验](PAPER_CONTINUATION_TRAINING_20261103.md)：两卡训练、两卡逐轮评测，独立W&B运行。
