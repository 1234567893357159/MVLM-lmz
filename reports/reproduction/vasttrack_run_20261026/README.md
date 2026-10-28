# VastTrack-only：2026-10-26 训练运行归档

记录更新：2026-10-28。采集时间：2026-10-28T20:44:42+08:00。原始输出目录为服务器 `/root/MVLM/output/vasttrack_baseline_20261026_run01`。

## 完成状态和实验范围

- 训练：60 轮、每轮 100,000 个样本、每轮 1250 次更新，共 75,000 次更新，退出码 0；8 卡 × 每卡 10 = 全局 batch 80，无梯度累积。
- 初始化：FastITPN 预训练、CLIP ViT-L/14 冻结，语言投影和 CENTER 头随机初始化；没有加载已训练 MVLM 跟踪器。
- 数据：仅 VastTrack 46,325 条训练序列；预先留出 500 条内部验证，序列名无交集，尚未完成同源视频近重复审计。
- 训练开始：北京时间 2026-10-26 23:31；监督程序完成记录：2026-10-28 10:10，包含训练后收尾时间。
- [W&B 页面](https://wandb.ai/2896287267-hit/mvlm-reproduction/runs/0jox3xdf)只记录训练统计，后续评测没有上传到该 run。

这是单数据集基线。官方训练配方使用 LaSOT、VastTrack、TNL2K_train 三者 1:1:1 抽样，本次不能称为完整论文训练复现。

## 外部测试与官方同协议对照

| 数据集/模型 | 序列 | 帧数 | AUC | PRE | NPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| otb99_lang / 自训 E60 | 48 | 29077 | 52.6971 | 71.0009 | 63.3101 |
| tnl2k / 自训 E60 | 699 | 515708 | 37.1414 | 32.5292 | 49.3768 |
| lasot / 自训 E60 | 280 | 685360 | 51.5193 | 50.2574 | 54.7242 |
| otb99_lang / 官方权重同协议对照 | 48 | 29077 | 60.5249 | 84.2023 | 75.0193 |

TNL2K 下载测试序列 700 条，其中 `Fight_video_6-Done` 语言为空，排除后评测 699 条。OTB99-Lang 48 条对应标准测试划分，另 51 条为原始训练划分。全部完整结果的 `prediction_validation.json` 均无错误。

官方 TNL2K 任务按用户要求中途取消，[取消状态](evaluation/tnl2k_official_baseline/status.json)没有全测试指标；已有部分预测只用于差距诊断。本轮没有重跑官方 LaSOT。总 suite 的取消状态保留历史事实；后来六轮验证和延期的自训 LaSOT 已由[独立队列](validation/queue_status.json)完成。

## 固定内部验证

| Epoch | AUC | PRE | NPR |
| --- | ---: | ---: | ---: |
| 10 | 66.4442 | 59.8978 | 74.4790 |
| 20 | 67.6362 | 61.8952 | 75.5285 |
| 30 | 68.6320 | 63.5398 | 76.5937 |
| 40 | 68.3181 | 63.6076 | 76.0448 |
| 50 | 69.1332 | 64.4868 | 76.4301 |
| 60 | 70.4254 | 66.8746 | 77.8809 |

每轮相同 500 条、39,893 帧；排除 603 个无效 GT 帧，实际指标分母为 39,290 个有效帧。六个抽样轮次中 E60 最高，尚未评测全部 60 轮，也未对其余轮次做三个外部完整测试。内部验证来自 VastTrack 的训练部分，平均序列约 80 帧，其分数不能与外部数据或论文直接比较。

## 推理和计分口径

首帧整图根据语言产生实际预测，不使用真实框初始化。输入模板仅使用整图占位框，template-free 骨干不使用模板像素；后续按原跟踪逻辑运行。

| 数据集 | α | τ | ω | λ | ψ |
| --- | ---: | ---: | ---: | ---: | ---: |
| OTB99-Lang | 0.5 | 0.03 | 0.4 | 0.1 | 0.1 |
| LaSOT | 0.5 | 0.03 | 0.1 | 0.4 | 0.1 |
| TNL2K | 0.5 | 0.03 | 0.1 | 0.4 | 0.12 |
| VastTrack 所有验证轮次 | 0.5 | 0.03 | 0.1 | 0.4 | 0.1 |

前三组参数来自历史官方权重的测试集调参，按要求原样迁移。验证参数固定借用 LaSOT 最优值，未在 VastTrack 搜索，不能声称是新权重的最优参数。

指标以百分数记录，每个序列等权平均。AUC 使用 0:0.05:1 的 21 个 IoU 阈值及严格大于；PRE@20px；NPR@0.20。外部测试无效/不可见帧按失败计入 AUC/PRE，内部验证排除无效帧。外部 NPR 保留原工具约定：非 LaSOT 的无效帧归一化中心误差设为 -1，会被计为 NPR 成功，因此 NPR 有已知边界，不能当作统一无效帧规则的指标。为保留实验原貌，本次归档不改写已有成绩。

主要差距及尚未核定的作者实际 batch 定义见[差距报告](../VASTTRACK_RESULT_GAP_ANALYSIS_20261028.md)。

## 数据状态快照

| 数据 | 已具备 | 尚未完成 |
| --- | --- | --- |
| VastTrack | 本轮使用的 46,825 条可用序列；46,325 训练 + 500 验证 | 同源视频近重复审计 |
| LaSOT | 280 条测试序列，已评测 | 1120 条训练序列准备中，下载 53.66% |
| TNL2K | 700 条测试序列，699 条有效语言已评测 | 1300 条训练序列准备中，下载 77.56% |
| OTB99-Lang | 本地 99 个视频目录，标准测试 48 条已评测 | 本实验未使用其训练部分 |
| MGIT | 未准备、未评测 | 数据准备与测试 |

下载进度只代表采集时刻；没有修改或重启下载任务。只有独立下载程序完成校验和解压并写入 `ready`，才可开始三数据集训练。完整字节数、路径与状态见[data_inventory.json](data_inventory.json)。

## 归档内容与服务器保留内容

- [summary.json](summary.json)、[metrics.csv](metrics.csv)：十组完整评测；取消任务单列，无伪造的最终成绩。
- [evaluation/](evaluation/)：自训三套测试和官方 OTB 对照的状态、逐序列成绩、预测完整性检查；官方 TNL2K 仅取消状态。
- [validation/](validation/)：六个轮次各自的状态、逐序列成绩和完整性检查，以及已完成队列状态。
- [training/](training/)：原始配置、运行规格、完成状态、初始化和来源、运行源码文件 SHA256。配置里的 Git 提交是运行前基础提交；实际训练存在未提交适配，因此应结合冻结源码哈希，而不能仅凭该提交号重建训练。
- [checkpoint_manifest.json](checkpoint_manifest.json)：本地 60 个检查点、字节数与保存时间；本次重新计算 E60 SHA256，其余轮次未计算哈希。
- [archive_manifest.json](archive_manifest.json)：复制的原始记录及汇总的 SHA256；该清单不含自身和本 README。

完整检查点在本机 `output/vasttrack_baseline_20261026_run01/checkpoints/train/mvlm/mvlm_TF_vasttrack/`，逐帧框与时间在 `output/test/tracking_results/mvlm/`。数据、权重、完整帧级预测及日志不上传 GitHub。原始运行 JSON 保留当时绝对路径；这些路径是证据，不代表换机器后自动有效。

## 复评入口

在仓库根目录执行，先激活 MVLM 环境，并配置 `lib/local_config.py`、准备权重和数据。下面命令会实际运行评测；使用新的 exp-id 和输出目录，避免与原始结果混用。内部验证入口读取版本化清单并核对其 SHA256，和原运行冻结清单逐字节相同。

```bash
conda activate mvlm
python -m tracking.strict_language_eval \
  --dataset otb99_lang \
  --config vasttrack_e60_strict_otb99_lang_20261028 \
  --weights output/vasttrack_baseline_20261026_run01/checkpoints/train/mvlm/mvlm_TF_vasttrack/MVLM_ep0060.pth.tar \
  --exp-id replay_e60_otb99_lang \
  --out output/replay_e60_otb99_lang --threads 16 --gpus 8

python -m tracking.strict_vasttrack_holdout_eval \
  --config vasttrack_holdout_strict_common_20261028 \
  --weights output/vasttrack_baseline_20261026_run01/checkpoints/train/mvlm/mvlm_TF_vasttrack/MVLM_ep0060.pth.tar \
  --exp-id replay_e60_vasttrack_holdout \
  --out output/replay_e60_vasttrack_holdout --threads 16 --gpus 8
```

LaSOT/TNL2K 分别替换 `--dataset` 为 `lasot`/`tnl2k`，配置替换为 `vasttrack_e60_strict_lasot_20261028`/`vasttrack_e60_strict_tnl2k_20261028`，同时更换 exp-id 和 out。复评其他轮次时替换检查点并使用独立输出；这些配置用于评测，不用于重启训练。

两个 `tracking/supervise_*` 评测监督脚本保存了本机一次性队列的流程，固定引用历史目录；保留为记录，日常复评使用上述独立入口。本次仅增加导入保护、引用版本化划分和说明，不改变预测或计分逻辑。
