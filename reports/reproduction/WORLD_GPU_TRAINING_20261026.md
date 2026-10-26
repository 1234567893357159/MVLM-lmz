# 世界模型 GPU 训练记录

后续：用户已授权启动阶段 B，8 卡真实预检通过并已后台启动，详见 [阶段 B 训练记录](WORLD_STAGE_B_TRAINING_20261026.md)。本文主要保存阶段 A 的过程和最终结果。

日期：2026-10-26（北京时间）。工作区 `/root/MVLM/work/world-model`，分支 `codex/world-model`，基点 `e867ccb`；本次监控接入等改动尚未提交。

## 最新：切换 8 卡同步训练

**最终完成：2026-10-26 13:22:32（北京时间）**。阶段 A 正常完成 4000/4000 步、20 次固定片段验证；4000 条训练损失均有限，最终权重有限，8 卡 RNG/优化器等状态已保存。训练进程已退出，8 张 GPU 显存释放。阶段 B 未自动启动。

验证：首次第 200 步损失 0.2746081，最佳第 3200 步 0.2412911（相对首次下降约 12.13%），最终第 4000 步 0.2431688；每轮 32 个固定片段，29 个存在可计算的监督损失。`best_clip_loss.pth.tar` 对应 3200 步，`step_004000.pth.tar` / `last.pth.tar` 为最终状态。这些仅是片段预测损失，未完成阶段 B 或完整在线视频 PRE/NPR/AUC，不能据此宣称跟踪精度提升。

**本轮 W&B：[world-stage-a-vasttrack-8gpu-20261026](https://wandb.ai/2896287267-hit/mvlm-world-model/runs/s8tibgnj)**，run ID `s8tibgnj`。12:06 的启动快照为 72/4000 步、GPU0–7 各占约 2987 MiB；现已完成，实时结论以上方最终记录为准。

用户随后要求像 memory 分支一样使用全部显卡。已实现 `torchrun` / NCCL / DistributedDataParallel，单卡预跑主动停止于 402 步，200/400 步检查点保留；其本地状态和 W&B summary 标为 `stopped_for_8gpu_transition`。旧 W&B `rkov29g0` 不再代表正在运行的训练。

新输出 `/root/MVLM/output/world_stage_a_vasttrack_8gpu_20261026_run01`，torchrun 启动 PID 12870，日志为同名 `.log`，启动命令在同名 `.launch.json`。每卡 batch 2、全局 batch 16、8 帧片段、4000 步、每 200 步检查点与 32 片段验证；学习率仍为 1e-4，从官方权重重新初始化，不继承单卡预跑优化器。相对单卡每步采样量增大 8 倍，是独立实验。

```bash
cd /root/MVLM/work/world-model
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /root/miniconda3/envs/mvlm/bin/python -u -m torch.distributed.run \
  --standalone --nproc_per_node=8 tracking/train_world_model.py \
  --config experiments/mvlm/mvlm_TF_world_stage_a_vasttrack_8gpu_20261026.yaml \
  --manifest /root/MVLM/output/world_vasttrack_split_20261026/split.json \
  --baseline /root/MVLM/models/MVLM_TF.pth.tar \
  --output /root/MVLM/output/world_stage_a_vasttrack_8gpu_20261026_run01 \
  --device cuda --validation-clips 32 \
  --wandb-project mvlm-world-model --wandb-entity 2896287267-hit \
  --wandb-name world-stage-a-vasttrack-8gpu-20261026
```

运行中不要重复执行。W&B 仅 rank 0 建立一个在线 run，指标为全卡汇总；新网址写入新输出目录 `wandb_run.json`。启动会先由 rank 0 复核全量数据，生成共享元数据缓存，其余 rank 读取本轮刚生成的缓存；每次恢复重新核验，不跳过数据签名检查。

验证证据：

- 原有 7 项 CPU 检查通过，单卡 A/B 与恢复行为保留。
- 双进程 CPU 检查通过：某卡无监督、全卡无监督、未使用参数、按监督帧加权的梯度、采样隔离/恢复、验证缓冲区同步。
- 8 卡真实数据阶段 A 预检完成 4 次更新，每次全局 16 个片段；8 卡可训练权重 SHA256 相同且已更新，冻结参数版本计数未变、无冻结参数梯度，首批采样索引覆盖 0–15 且无交叉；两个验证片段有效。证据 `/root/MVLM/output/world_gpu_8_preflight_20261026/run/distributed_audit_uninterrupted.json`。
- 从第 2 步恢复到第 4 步成功；每次运行内部 8 副本逐位一致。恢复后第 3 步 loss 相同、第 4 步差约 1.18e-9；NCCL 浮点归约存在微小差异，不能声称跨启动逐位重现。每卡 RNG、优化器和全局采样位置均存入检查点；恢复要求同卡数及源码/配置/数据一致。
- 阶段 B 使用 DDP 缓冲区广播，并在并行验证前统一 rank 0 的 BatchNorm 统计；本次 8 卡真实数据验证针对阶段 A，未启动阶段 B 训练。

下面单卡启动记录为本日早先的预跑历史。

## 授权及环境

用户授权在 GPU 恢复后尝试世界模型训练，并明确确认 W&B 实体 `2896287267-hit`、项目 `mvlm-world-model`，上传配置、损失与运行指标，不上传原始视频或模型权重。旧记忆分支的暂停任务不恢复。

环境 Python `/root/miniconda3/envs/mvlm/bin/python`；PyTorch 1.13.0+cu117，CUDA 11.7，驱动 575.57.08，8×RTX 3090 / 24 GiB。阶段 A 冻结 MVLM，仅更新世界预测器；初次预跑为单卡，后续已支持 8 卡 DDP，仍未启用 AMP。

## 启动前验证

- `tracking/check_world_model.py -v`：7 项通过，包括 A/B 冻结范围、清单隔离及中间检查点恢复。
- `tracking/check_world_integration.py --device cuda:0`：真实官方权重、零残差旁路、A/B 梯度、Actor 和在线 tracker 检查通过。
- 本地 GPU0 真实视频预检：8 条训练视频、2 条验证视频，4 次参数更新；最终检查点步数为 4，优化器状态一致，世界模型外观预测头已更新，全部权重有限，2/2 验证片段有效。产物 `/root/MVLM/output/world_gpu_preflight_20261026/run`。预检模型不作为后续训练初始化。
- W&B 模拟检查：训练/验证指标、链接保存、步数与失败状态记录通过；在线 API 已成功读取现有登录用户。
- 在线 GPU 预检也完成 4 步及 2 个有效验证片段，W&B 已确认同步训练/验证曲线，0 media、0 artifact 文件；[在线预检](https://wandb.ai/2896287267-hit/mvlm-world-model/runs/60r7myng)，本地产物 `/root/MVLM/output/world_gpu_preflight_20261026/online_run`。
- 数据读取优化：`official_names()` 增加进程内缓存，避免逐视频重复读取数万行训练清单；清单在单次运行中固定，后续新进程重新读取。图像目录统一解析实际路径，再单独解析软链接帧，避免百万级普通帧重复解析共同祖先；目录和单帧软链接路径等价检查通过，清单隔离及训练恢复检查再次通过。初次只读审计尚未输出清单时停止，使用该优化重新审计。

## 数据及训练方案

仅使用 `/root/MVLM/data/vasttrack` 的官方训练清单视频，独立生成视频级固定训练/验证划分，验证比例 10%、种子 20261022。显式记录缺帧、过短或无效标注等遗漏。LaSOT/TNL2K 的现有测试数据不参与训练。

清单目标 `/root/MVLM/output/world_vasttrack_split_20261026/split.json`。这是 VastTrack 单数据集实验，不是原论文完整三数据集训练配方。

完整审计结果：41,737 条训练视频、4,637 条验证视频，共 46,374 条。官方原始清单中的 462 条被显式排除：451 条不足 8 帧，另 11 条帧/标注等检查不通过，与主目录已记录的 11 条缺帧序列完全一致；具体名称及原因保存在清单 `omissions`。该清单不修改原始数据或主目录的训练清单。

阶段 A 配置 `experiments/mvlm/mvlm_TF_world_stage_a_vasttrack_20261026.yaml`：全局 batch 2、8 帧片段、学习率 1e-4、计划 4000 次优化器更新、每 200 步保存并验证 32 个固定片段；从官方 `/root/MVLM/models/MVLM_TF.pth.tar` 初始化。阶段 B 需加载阶段 A 结果后另行运行，本次不自动串联。

```bash
cd /root/MVLM/work/world-model
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /root/miniconda3/envs/mvlm/bin/python -u tracking/train_world_model.py \
  --config experiments/mvlm/mvlm_TF_world_stage_a_vasttrack_20261026.yaml \
  --manifest /root/MVLM/output/world_vasttrack_split_20261026/split.json \
  --baseline /root/MVLM/models/MVLM_TF.pth.tar \
  --output /root/MVLM/output/world_stage_a_vasttrack_20261026_run01 \
  --device cuda:0 --validation-clips 32 \
  --wandb-project mvlm-world-model --wandb-entity 2896287267-hit \
  --wandb-name world-stage-a-vasttrack-20261026
```

输出目录必须为新目录；恢复须带 `--resume`，且源码、配置、数据与基线校验一致。禁止在已有任务运行时重复启动以上命令。

## 进度与监控

4000 步阶段 A 已后台启动，初始 PID 7105，输出 `/root/MVLM/output/world_stage_a_vasttrack_20261026_run01`。已完成全量元数据复核并在 GPU0 实际更新参数；2026-10-26 11:49（北京时间）复核状态为 `running`，已完成 33 步，前 33 步损失均有限，GPU0 显存约 1.7 GiB。此为启动快照，实际步数必须以状态和指标文件为准。启动记录 `/root/MVLM/output/world_stage_a_vasttrack_20261026_run01.launch.json`，完整控制台日志 `/root/MVLM/output/world_stage_a_vasttrack_20261026_run01.log`。

**本轮在线监控：[world-stage-a-vasttrack-20261026](https://wandb.ai/2896287267-hit/mvlm-world-model/runs/rkov29g0)**，run ID `rkov29g0`。与已完成的 4 步预检 `60r7myng` 区分。

11:50（北京时间）再次复核：本地已完成 103 步、状态 `running`；W&B 在线历史已读取到第 90 步的实际损失，确认训练指标已同步到服务端。

- `<output>/wandb_run.json`：run ID 和具体网页地址。
- `<output>/training_status.json`：运行状态、PID、实际优化器更新次数。
- `<output>/metrics.jsonl`：逐步损失、外观/运动误差及复制/静止/恒速对照。
- `<output>/validation_*.json`：固定片段验证，**不是完整视频 PRE/NPR/AUC**。
- W&B `train/*`：损失、梯度范数、学习率、步时、峰值显存等；`validation/*`：固定片段损失及有效片段数量。W&B 默认系统监控提供设备利用率等信息。

阶段 A 尚不注入融合模块，`train/active` 为 0 属于预期；验证曲线在首次 200 步验证后出现，不应将训练损失当作完整跟踪精度。

片段损失与短训通过只证明训练流程可运行，不能证明跟踪精度提升。完整在线视频验证需单独使用 `tracking/evaluate_world_model.py`。
