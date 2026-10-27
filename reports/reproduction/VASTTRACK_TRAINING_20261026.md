# VastTrack 单数据集原始 MVLM 训练

用户授权：在 `codex/mvlm-reproduction` 分支，以每卡 batch 10、8 卡全局 batch 80，从骨干预训练权重重新训练原始 MVLM，并提供在线 W&B 监控。

## 运行位置

- 正式目录：`/root/MVLM/output/vasttrack_baseline_20261026_run01`
- 源码快照：正式目录的 `source/`；实际训练使用此快照，后续切换仓库分支不会改变这次运行的代码。
- 配置与元数据：`config.yaml`、`run_spec.json`、`source_sha256.json`、`backbone_source.json`、`initialization.json`。
- 实时状态：`training_status.json`；训练日志：`train.log`；监督日志：`supervisor.log`。
- 在线链接和 run ID：`wandb_tracking/train/mvlm/mvlm_TF_vasttrack/run.json`。应实际读取后再提供链接，不凭项目名猜测。
- 检查点：`checkpoints/train/mvlm/mvlm_TF_vasttrack/MVLM_epXXXX.pth.tar`。

## 实验设置

- 仅 VastTrack：46,325 条训练序列，预先保留 500 条完整序列，清单与哈希冻结在正式目录 `split/`。原始可用 46,825 条清单不覆盖。
- 该划分按序列名的固定 SHA256 排序选择，与训练按目录无交集；尚未审计近重复视频，不声称语义完全独立。
- 当前训练器没有自动验证入口；保留验证序列供后续固定评测使用，不会自动上传验证 AUC 或选出最佳验证模型。
- 每 epoch 随机采样 100,000 个样本，1250 次更新；60 epoch，目标 75,000 次更新。每卡 batch 10，无梯度累积。
- AdamW，主学习率 5e-4，编码器参数组倍率 0.1，第 54 epoch 衰减 0.1；权重衰减 1e-4、梯度裁剪 0.1、CM loss 权重 0.5。语言投影层在当前原始优化器中随编码器名称匹配进入较低学习率组。
- FastITPN 使用官方 iTPN 仓库 Google Drive 权重，完整 SHA256 `626fd426dd89b2681d8b3942fa00e05fffb467af111c7bbd6a0a4b8bc0afc388`，与镜像 LFS 哈希相同。
- 不加载 `/root/MVLM/models/MVLM_TF.pth.tar`；CLIP 冻结，骨干微调，语言投影层和 CENTER 预测头随机初始化后训练。不添加记忆分支的新增网络。
- 推理设置采用公开官方 `mvlm_TF.yaml` 的默认 MVLM 参数，不沿用当前主分支的历史调参值。
- 每 epoch 保存模型、优化器和 global_step。新实验目录没有旧检查点；监督程序依据子进程真实退出码记录完成或失败。

## 启动预检

- 8 张 RTX 3090 均完成实际 CUDA 运算检查。
- 实际 VastTrack batch 的图像、目标框、语言及 224×224 裁剪检查通过。
- 单卡 20 步短测完成；全部保存的模型张量有限，视觉骨干、语言投影、预测头抽样参数均发生更新，CLIP 可训练参数为 0。
- 8 卡 200 步短测及检查点审计必须通过后，`start_after_preflight.py` 才会启动正式监督程序；短测权重不用于正式训练初始化。
- 原始数据 loader 的无 worker 堆叠兼容问题已修复；训练器不再吞掉最终异常，非有限损失会明确停止。

## 监控与恢复

训练 IoU 是裁剪训练样本上的定位统计，不能当作在线跟踪 AUC。W&B 记录跨卡平均损失、训练 IoU、epoch、更新步数和主卡速度/显存，冻结的 CLIP 权重不写入检查点。

在执行恢复前先读状态、检查监督 PID 和训练进程，避免重复启动；用户要求暂停时不得自动重启。恢复应复用冻结源码、相同配置和输出目录，以恢复检查点及在线 run ID。仅有 W&B 页面不能证明训练已开始，应同时检查真实更新步数与日志。

## 正式启动核验

{
  "checked_at": "2026-10-26T23:34:41.487419+08:00",
  "status": "training",
  "global_step_at_least": 100,
  "wandb_url": "https://wandb.ai/2896287267-hit/mvlm-reproduction/runs/0jox3xdf",
  "supervisor_pid": 105725,
  "launcher_pid": 105862,
  "note": "Formal run reinitialized from pretrained backbone after successful 20-step single GPU and 200-step 8 GPU checks."
}
