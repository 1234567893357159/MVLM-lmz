# MVLM 三数据集正式训练：2026-10-30

完成状态更新（2026-11-01）：本次训练已正常完成60轮、75,000次更新；完整结果、曲线和最终检查点核验见[训练完成报告](PAPER_TRAINING_RESULTS_20261101.md)。下文保留启动时的历史记录。

用户授权按原论文复现方案从预训练骨干重新训练，并通过 W&B 监控。随后要求尽量少占共享服务器资源、保持全局 batch 80。正式运行使用物理 GPU **4、5**，每卡 **40**、全局 **80**；GPU 0–3 的其他任务未停止，GPU 6、7 保留空闲。每个训练进程仅 2 个加载 worker，总计 4 个；OMP/MKL/OpenBLAS 每进程 1 个计算线程。

正式输出：`/root/MVLM/output/paper_reproduction_20261030_run01`。监督进程在该目录的 `source/` 独立源码快照中运行；后续修改主工作区不改变本次运行。

## 配置与初始化

- 分支 `codex/mvlm-reproduction`，配置 `mvlm_TF_paper_reproduction`。
- LaSOT 1,120 条、VastTrack 46,825 条、TNL2K 1,299 条，按 1:1:1 概率抽样。TNL2K 训练排除 `Fight_game_004-Done`，原文件保留；VastTrack 清单已排除 11 条缺帧序列。
- FastITPN 骨干预训练及 CLIP 初始化；语言投影和 CENTER 随机初始化；不加载官方 MVLM、此前自训权重或预检权重。正式新目录中没有旧检查点，`MODEL.PRETRAINED_MODEL=null`。
- 60 轮，每轮 100,000 样本、1,250 次更新，总计 75,000 次更新。AdamW、主学习率 5e-4、编码器组倍率 0.1、第 54 轮后衰减 0.1、weight decay 1e-4、梯度裁剪 0.1；损失 2×GIoU＋5×L1＋Focal＋0.5×CM。
- 每轮保存检查点和优化器。全局 batch 80 是对论文未明确全局/每卡 batch 的显式解释；两卡每卡 40 的 BatchNorm 条件与论文四卡存在差异。

## 已完成的最短必要预检

预检目录：`/root/MVLM/output/paper_reproduction_preflight_20261030_run01`。

| 检查 | 更新次数 | 结果 |
| --- | ---: | --- |
| 单卡 batch 20 | 3 | 正常完成 |
| 四卡 × batch 20 | 4 | 正常完成 |
| 最终两卡 × batch 40 | 2 | 正常完成，使用正式的每卡 2 个加载 worker |

三个检查点均完成全量已保存张量有限性检查；视觉骨干、语言投影和预测头的抽样参数实际更新；CLIP 可训练参数数目为 0。短训不能证明最终收敛，只验证运行链路与资源可行性。预检结果写入正式输出 `preflight_validation.json`。

## 监控与文件

W&B：[paper-three-dataset-b80-2gpu-20261030](https://wandb.ai/2896287267-hit/mvlm-reproduction/runs/7ln28ook)，run ID `7ln28ook`。2026-10-30 13:48 后在线 API 已确认状态 `running`，实际上传至至少第 40 次更新，记录的 world size=2、global batch=80，最新该次训练损失为 6.2856555。此为启动核验快照，不是实时进度；证据保存在输出 `wandb_live_verification.json`。

项目为 `2896287267-hit/mvlm-reproduction`。准确链接也保存在运行目录 `wandb_tracking/train/mvlm/mvlm_TF_paper_reproduction/run.json`；只由主进程上传两卡平均损失、训练 IoU、学习率、更新步数及资源指标，不上传原始视频和模型权重。训练 IoU 不等于完整视频测试 AUC。

- `training_status.json`：监督/训练 PID、实际配置、进度、退出状态与 W&B 元数据。
- `train.log`、`supervisor.log`：训练及监督日志。
- `config.yaml`、`run_spec.json`、`source_sha256.json`：正式配置、初始化/资源/数据证据与源码哈希。
- `split/`：实际训练序列顺序、哈希及 TNL2K 修订/重命名记录。
- `initialization.json`：实际可训练参数组及初始参数哈希。
- `checkpoints/train/mvlm/mvlm_TF_paper_reproduction/`：每轮模型和优化器检查点。

独立重训必须使用新的输出目录；中断恢复才复用本次冻结源码和输出目录。不要重复启动相同训练任务。实时状态应读取本次状态文件，历史日志中的 PID/步数不作为实时依据。
