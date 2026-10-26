# 世界模型阶段 B：8 卡训练记录

日期：2026-10-26（北京时间）。用户明确授权启动 B 阶段、保存记录并开启 W&B。工作区 `/root/MVLM/work/world-model`，分支 `codex/world-model`；沿用已有未提交实现，不覆盖主目录改动。

## 最终结果（14:37:13，北京时间）

阶段 B 正常完成 4000/4000 步、20 次固定片段验证，W&B 为 `finished` / `completed`，进程退出且 8 张 GPU 显存已释放。全部 4000 条训练损失有限；最终权重有限，最终世界模型权重与 A 来源逐项 `torch.equal`，8 卡 RNG 等恢复状态齐全。审计保存在正式输出 `final_training_audit.json`。

最佳固定片段损失为第 1000 步 0.6265464913，最终第 4000 步 0.7281143064，首次第 200 步 1.1104249714。后续训练未超过第 1000 步最佳值；`best_clip_loss.pth.tar` 对应 1000 步，`last.pth.tar` / `step_004000.pth.tar` 为最终状态。它们不是完整视频 AUC，尚未执行与原 MVLM 的同集 PRE/NPR/AUC 对照，不能宣称超过基线。

以上为最终状态；下方启动步数和进程为历史快照。

## 初始化与训练范围

- 世界模型加载阶段 A 的 `best_clip_loss.pth.tar`：`/root/MVLM/output/world_stage_a_vasttrack_8gpu_20261026_run01/best_clip_loss.pth.tar`，第 3200 步，A 固定片段验证损失 0.2412910639。
- A 检查点 SHA256：`9b1e8fa91e242602da7af35d4fc8c9d3a7245204802355b7587ed00e19a184f8`。
- 其余跟踪模型从官方 `/root/MVLM/models/MVLM_TF.pth.tar` 初始化。B 新建优化器，不继承 A 优化器，也不继承 B 预检权重。
- **冻结**视觉编码器、文本编码器和 A 训练好的世界模型；它们保持 eval。**训练** `imagination_fusion` 和原 MVLM `head`（CENTER），包括 CENTER 的 BatchNorm 参数及运行统计。
- DDP 训练广播缓冲区；并行验证前将 BatchNorm 等缓冲区统一为 rank 0 的统计，保证各卡验证使用保存检查点对应的统计。

## 配置与数据

- 配置：`experiments/mvlm/mvlm_TF_world_stage_b_vasttrack_8gpu_20261026.yaml`，取自 A 检查点中包含默认值的完整配置，仅切换到阶段 B；`DATA` / `TEST` 与 A 逐项相同。
- 8×RTX 3090，DDP/NCCL；每卡 2、全局 batch 16，8 帧片段；不启用 AMP，数据加载 worker 为 0。
- 融合模块初始学习率 1e-4，CENTER 1e-5，余弦衰减；4000 次优化器更新，每 200 步保存并验证 32 个固定片段。
- 与 A 使用同一清单 `/root/MVLM/output/world_vasttrack_split_20261026/split.json`：41,737 个 VastTrack 训练视频、4,637 个验证视频。不会使用 LaSOT/TNL2K 当前已准备的测试集训练。
- 启动时重新审计数据及清单指纹，检查 A/B 的基线 SHA256、清单 SHA256、数据签名和数据/推理配置一致。权重/清单/配置哈希及完整命令保存在启动记录中。

## 真实 8 卡启动前检查

预检目录 `/root/MVLM/output/world_stage_b_8gpu_preflight_20261026_run01`。使用同一完整清单与 A 最佳权重，4 步训练、每 2 步保存、8 个验证片段；未创建线上预检 run。

`preflight_validation.json` 记录：

- 完成 4 步，8 卡 / 全局 batch 16，全部保存权重有限。
- 世界模型权重与 A 检查点逐项 `torch.equal`，未更新。
- 与官方基线可直接比较的 568 个编码器/文本存储项逐项未变；CLIP 本体不写入 B 检查点，另由冻结参数版本及无梯度检查覆盖。
- 54 个 CENTER 权重/偏置张量发生变化；融合输出层从零初始化变为非零，两个优化器组的 Adam 一阶矩分别有 13 / 54 项非零，证实两组均实际参与训练。
- 8 卡可训练参数 SHA256 一致，冻结参数版本/梯度检查通过。
- 两次验证均有 8/8 个可用片段，最后片段跟踪损失 1.1018413。该值只验证流程，不作为正式实验成绩。

## 正式启动与监控

正式输出 `/root/MVLM/output/world_stage_b_vasttrack_8gpu_20261026_run01`，torchrun 初始 PID 17080。控制台日志为同名 `.log`，完整启动记录为同名 `.launch.json`；后者包含初始化来源、哈希、配置、冻结范围及预检报告。

```bash
cd /root/MVLM/work/world-model
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /root/miniconda3/envs/mvlm/bin/python -u -m torch.distributed.run \
  --standalone --nproc_per_node=8 tracking/train_world_model.py \
  --config experiments/mvlm/mvlm_TF_world_stage_b_vasttrack_8gpu_20261026.yaml \
  --manifest /root/MVLM/output/world_vasttrack_split_20261026/split.json \
  --baseline /root/MVLM/models/MVLM_TF.pth.tar \
  --world-checkpoint /root/MVLM/output/world_stage_a_vasttrack_8gpu_20261026_run01/best_clip_loss.pth.tar \
  --output /root/MVLM/output/world_stage_b_vasttrack_8gpu_20261026_run01 \
  --device cuda --validation-clips 32 \
  --wandb-project mvlm-world-model --wandb-entity 2896287267-hit \
  --wandb-name world-stage-b-vasttrack-8gpu-20261026
```

任务运行中不要重复执行。启动后保存 `wandb_run.json`（网址和 run ID）、`training_status.json`（PID/实际步数/状态）、逐步 `metrics.jsonl`、源码快照与哈希、配置及清单副本、验证报告和检查点。W&B 仅 rank 0 建立一个新 run，上传全卡汇总指标，不上传原始视频或权重。

**W&B：[world-stage-b-vasttrack-8gpu-20261026](https://wandb.ai/2896287267-hit/mvlm-world-model/runs/6kaexdvm)**，run ID `6kaexdvm`，与 A 阶段 `s8tibgnj` 分开记录。

2026-10-26 13:32:25（北京时间）启动复核：状态 `running`，已完成 52/4000 步，rank 0 PID 17086；前批训练损失有限，融合路径已有激活。GPU0–7 各约 3033 MiB；W&B 在线历史已读取到实际训练数据（含 world size 8、global batch 16），确认不是仅创建了网页。此为快照，后续以状态文件和进程为准。

正式输出还保存 `stage_b_provenance.json` 和 `preflight_validation.json`，本说明同步为 `TRAINING_RECORD.md`。A 来源检查点、SHA256、步数、固定划分哈希、新优化器和冻结/训练范围已加入 W&B config 的 `stage_b_initialization`。

## 指标解释

B 使用跟踪损失（GIoU/L1/中心响应损失），与 A 的外观/运动预测损失不能直接比较数值。`validation/*` 为固定片段验证，不是完整视频 PRE/NPR/AUC。`best_clip_loss.pth.tar` 只按片段损失选取，后续需完整在线验证才能判断跟踪效果。
