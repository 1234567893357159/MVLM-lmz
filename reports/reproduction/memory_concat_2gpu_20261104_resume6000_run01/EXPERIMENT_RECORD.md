# 记忆拼接 CENTER 双卡实验档案

状态：training_to_5000_attempt01；更新时间：2026-11-04T18:09:11.264652+08:00

W&B：https://wandb.ai/2896287267-hit/mvlm-memory/runs/15139aa0

GPU4、5；每卡40个四帧片段，全局80；本轮从4000步训练到6000步。
冻结视觉/文本编码器与CENTER BN运行统计，训练记忆读取与CENTER。
学习率峰值记忆1e-4、CENTER1e-5；前200步预热，第3000步后乘0.1。

## 当前记录

{
  "status": "training_to_5000_attempt01",
  "supervisor_pid": 794625,
  "started_at": "2026-11-04T18:03:42.869527+08:00",
  "global_step": 4060,
  "target_steps": 6000,
  "physical_gpus": [
    4,
    5
  ],
  "batch_per_rank": 40,
  "global_batch": 80,
  "wandb_url": "https://wandb.ai/2896287267-hit/mvlm-memory/runs/15139aa0",
  "updated_at": "2026-11-04T18:09:11.264652+08:00",
  "wandb_mode": "online",
  "worker_pid": 794640
}

## 档案入口

- run_spec.json：初始化、数据/源码哈希及资源预算
- config.yaml / split/：冻结配置和训练/验证清单
- training_metrics.jsonl：跨卡平均训练指标
- events.jsonl / resource_metrics.jsonl：阶段事件与资源
- checkpoints/：每500步检查点、优化器、随机状态及步数日程
- rank*_start_audit.json / rank*_final_audit.json：逐段冻结、BN与更新审计
- validation/：基线、固定30条子集及完整150条验证结果
- validation_summary.json / best.json：完整验证的候选模型
- logs/ / wandb_offline/ / wandb_online/：训练验证日志与W&B本地文件
- audits/：保留每段开始与结束的优化器、随机状态和冻结审计

验证集来自本轮训练部分，官方初始化可能已见过。子集与完整验证分数分开记录。
未评估标准测试集。
检查点保存各rank随机状态；DataLoader预取/worker状态未完整捕获，不保证位级恢复。

## 续训来源

父实验：/root/.codex/worktrees/xmem-concat-center/MVLM/output/memory_concat_2gpu_20261103_run01
父实验W&B：https://wandb.ai/2896287267-hit/mvlm-memory/runs/65be0b58
起点检查点：/root/.codex/worktrees/xmem-concat-center/MVLM/output/memory_concat_2gpu_20261103_run01/checkpoints/MVLM_step004000.pth.tar
起点SHA256：005807b6de6ae0391b18d097c495c0b33022182bc7bdea69148df6238f52478e
恢复模型、AdamW状态与双卡随机状态；不重新预热。
续训学习率：记忆1e-5、CENTER1e-6；训练模块、四帧片段、数据与推理设置不变。
5000、6000步完成同一150条验证，并与原4000步结果比较。
训练进程异常时最多自动重试2次，从最新完整检查点继续；失败记录保留。
