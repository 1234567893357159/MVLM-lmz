# MVLM VastTrack 单数据集复现分支

本分支基于 `main`，记录一次只用 VastTrack 重新训练原始 MVLM 的实验，及其数据划分、W&B 监控和检查点支持。它是单数据集基线：官方配置使用 LaSOT、VastTrack 和 TNL2K，不能把本实验称为完全复现论文的三数据集训练。

截至 2026-10-28：训练已完成 60 轮、75,000 次更新，60 个检查点均保留在本地。第 10/20/30/40/50/60 轮的 500 条内部验证已完成，第 60 轮已完成 OTB99-Lang、LaSOT、TNL2K 的严格 language-only 评测。统一入口见[实验索引](reports/reproduction/README.md)，可随 Git 浏览的运行证据见[本轮归档](reports/reproduction/vasttrack_run_20261026/README.md)。

## 分支内容

- `experiments/mvlm/mvlm_TF_vasttrack.yaml`：VastTrack 训练配置；另外两个 `mvlm_TF_vasttrack_probe_*.yaml` 用于单卡和八卡短测。
- `reports/reproduction/vasttrack_split_20261026/`：固定的视频序列级训练/验证划分及 SHA256。46,825 条可用序列中，46,325 条用于训练、500 条留作后续验证；尚未做近重复视频审计。
- `lib/train/data_specs/vasttrack_train_split.txt`：排除 11 条本地镜像缺帧序列后的 46,825 条完整可用清单。本轮实际训练使用独立的 46,325 条清单，预留另外 500 条验证；它不同于原项目清单，实验结果需说明这一差异。
- 训练器支持可指定的 VastTrack 清单、每 epoch 检查点、优化器更新步数保存、损失非有限检查，以及 W&B 指标记录；修复零数据加载 worker 时的 batch 合并。
- `lib/train/admin/wandb_logger.py`：可选 W&B 监控，八卡标量跨进程平均，只由主进程上传。默认配置不启用；VastTrack 配置启用。
- `tracking/supervise_vasttrack_training.py`：本次八卡运行的后台监督和状态记录工具；实际使用前应依据机器资源及环境检查其固定的八卡参数。
- `reports/reproduction/VASTTRACK_TRAINING_20261026.md` 和 `VASTTRACK_WANDB.md`：本机运行记录和监控说明；文档中的 `/root/MVLM` 路径是当时的运行环境示例。
- `tracking/strict_language_eval.py`：首帧整图预测、无 GT 初始化的外部评测入口；三个 `vasttrack_e60_strict_*_20261028.yaml` 固定历史调参值。
- `tracking/strict_vasttrack_holdout_eval.py`：固定 500 条内部验证入口，读取版本化验证清单并核对 SHA256；所有已抽样轮次使用同一份参数。
- `tracking/supervise_strict_language_eval.py`、`tracking/supervise_vasttrack_checkpoint_eval.py`：本次一次性队列的历史记录，含固定运行目录；日常复评使用归档说明中的独立入口，不直接重启历史队列。

## 训练设置

从 FastITPN-Base 预训练骨干与 CLIP ViT-L/14 初始化，不加载训练好的 MVLM 跟踪权重。CLIP 本体冻结，视觉骨干、语言投影层和 CENTER 预测头参与训练。本次设置为 8 卡、每卡 batch 10、全局 batch 80，无梯度累积；每 epoch 采样 100,000 个训练样本，60 个 epoch。AdamW 主学习率 `5e-4`，编码器参数组倍率 `0.1`，第 54 个 epoch 衰减。

训练配置中的清单路径相对于仓库根目录解析；使用前需在本机配置数据路径，并准备 `pretrained/itpn/fast_itpn_base_clipl_e1600.pt` 及 CLIP 权重。骨干下载来源及本次 SHA256 见训练记录。W&B 使用环境变量或 `wandb login` 管理认证信息，可选依赖在 `requirements-wandb.txt`。

本机实验目录为 `/root/MVLM/output/vasttrack_baseline_20261026_run01`，其中保留运行时独立源码快照、配置、清单、状态、日志和检查点。模型文件、数据集和实验输出由 `.gitignore` 排除，不随代码推送。

## 监控与结果边界

每 20 次更新记录损失、训练 IoU、学习率和主卡资源，epoch 结束记录均值；每个 epoch 保存模型和优化器状态。训练 IoU 是训练样本上的指标，不能等同于在线跟踪测试 AUC。当前训练循环不自动评测留出的 500 条验证序列，也不自动在 LaSOT/TNL2K 测试集上测试或选取最佳模型；可在后续使用已保存检查点单独评测。

训练结束后已经通过独立脚本完成上述验证与测试，结果未回填到训练 W&B 曲线。第 60 轮内部验证 AUC 70.43，是抽样的六个轮次中最高；外部 OTB/LaSOT/TNL2K AUC 分别为 52.70/51.52/37.14。内部验证来自 VastTrack 训练部分，不能与论文外部测试横比。历史参数曾在测试集调优，本次仅固定迁移，结果属于探索性实验。

OTB99-Lang 的 48 条是标准测试划分；TNL2K 700 条测试数据中有 1 条空语言，本轮评测 699 条。官方 OTB 同协议对照已完成，官方 TNL2K 中途取消，官方 LaSOT 没有在本轮重跑。完整成绩、协议及限制见[归档](reports/reproduction/vasttrack_run_20261026/README.md)和[差距分析](reports/reproduction/VASTTRACK_RESULT_GAP_ANALYSIS_20261028.md)。

训练启动前通过单卡 20 步、八卡 200 步真实数据短测，并检查损失及权重有限性、模块更新与 CLIP 冻结。W&B 监控通过离线日志、检查点恢复和双进程 Gloo 检查。实际训练状态应查看实验目录的 `training_status.json` 和 `train.log`，历史报告中的状态不代表实时状态。
