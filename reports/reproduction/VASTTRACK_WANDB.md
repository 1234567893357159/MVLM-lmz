# VastTrack 原始 MVLM 训练监控

当前分支：`codex/mvlm-reproduction`。复用记忆分支的 W&B 日志组件，只接入原始训练循环，不引入新增记忆网络或旧实验评测逻辑。

## 配置

`experiments/mvlm/mvlm_TF_vasttrack.yaml` 仅加载 VASTTRACK，启用在线监控，项目为 `mvlm-reproduction`，名称为 `vasttrack-from-pretrained-b80-20261026`。原有配置默认不启用 W&B。

- 使用现有 `wandb login` 登录信息，不在配置或代码中保存密钥。
- 可选依赖：在训练环境安装 `requirements-wandb.txt`；当前环境已有 W&B 0.22.3。
- 可通过 `WANDB_PROJECT`、`WANDB_ENTITY`、`WANDB_NAME` 和 `WANDB_MODE` 环境变量调整目标或切换离线模式。
- 每 20 次成功优化器更新及每 epoch 末尾记录一次批次指标，并记录 epoch 均值。
- 仅全局 rank 0 创建和上传实验；训练标量跨卡取均值。速度和显存明确标为 rank 0 指标。

## 可查看的指标

- `train/Loss/Total`、`train/Loss/Tracking`、`train/Loss/CM`：总损失、跟踪损失及加权 CM 损失。
- `train/IoU`：训练样本上的框 IoU，不能解释为在线跟踪验证 AUC。
- `epoch/train/*`：每个 epoch 的均值。
- `lr/group*`：优化器各参数组学习率。
- `runtime/train/*`、`gpu/*`：主卡每批耗时、样本吞吐量及分配显存。
- 实验配置：代码分支与提交、工作区是否有修改、随机种子、训练参数、每卡/全局 batch、可训练参数量及 VastTrack 清单 SHA256。

训练启动后控制台输出 `W&B monitor: <网址>`；同一网址和 run ID 保存在：

`<save_dir>/wandb_tracking/train/mvlm/mvlm_TF_vasttrack/run.json`

使用同一 save_dir 和配置恢复时复用 run ID，并从 checkpoint 恢复 global_step；新实验必须使用新的 save_dir。旧 checkpoint 缺少步数时依据已完成 epoch 和当前 loader 长度估算。离线模式保留 ID，但 SDK 会创建新的本地离线运行目录，不能视为在线续传已验证。

异常会向上传播并以失败结束 W&B；不会再因原训练器吞掉最后一次异常而误报完成。

## 使用边界

此配置已用于一次真实训练，运行详情见 `VASTTRACK_TRAINING_20261026.md`。BATCH_SIZE=10 表示每卡 batch，8 卡全局 batch=80，不使用梯度累积。FastITPN 权重已从官方链接下载并校验。当前配置使用固定 46,325 条训练序列，另保留 500 条视频级验证序列，清单与哈希在 `reports/reproduction/vasttrack_split_20261026/`。仅记录现有训练统计，不会自动运行验证或上传 AUC、视频。

## 已验证

运行 `/root/miniconda3/envs/mvlm/bin/python tracking/check_wandb_logging.py`，检查真实 W&B SDK 离线日志、合成训练循环、批次和 epoch 指标、断点恢复步数和 ID、关闭监控、训练异常传播，以及双进程 Gloo 的跨卡均值与主进程隔离。此检查不加载 MVLM、不使用实际数据、不启动正式训练。
