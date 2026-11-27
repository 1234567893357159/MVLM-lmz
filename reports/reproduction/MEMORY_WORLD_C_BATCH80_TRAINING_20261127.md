# 记忆世界模型 C 阶段 batch 80 训练

本记录创建于 2026-11-27 20:43（北京时间），启动快照：两个训练进程正在审计数据清单，优化步数与结果尚未确认；以运行目录、训练日志、进程及 W&B 云端为准，不将计划当作完成。

- 目的：从同一 B/4000 出发，训练新增记忆读取、世界预测、质量头和未来融合；后续与 B/4000、C batch 1 在同协议留出集比较。尚不能归因各模块或宣称精度提升。
- 代码：分支 `codex/memory-world`，启动时 HEAD `264290c4d2a65eb89b68636ceddfcf822e8fa01d`。配置当时未提交，原件在 `experiments/mvlm/mvlm_TF_memory_world_c_b80_b4000_20261127.yaml`，SHA256 `dab400b12b2726f03a228cbbc6bcc3d5a04aea28c4bff6718a48c639ba159957`；运行目录保存配置、代码快照与源码哈希。
- 硬件与预算：物理 GPU 5、6，RTX 3090；每卡 40 片段、全局 batch 80；计划 1000 次优化器更新，12 帧/片段，前 4 帧预热；AdamW、初始 LR 1e-5、余弦调度，每 250 步保存及 32 个固定片段验证；种子 `20261108`。设备实际可用性随时变化。
- 初始化：官方基线 `models/MVLM_TF.pth.tar`，SHA256 `91c31230f3d355f71104df8191e386632a39ce9d3bbc1f7ce5bf3772b369cc4f`；父检查点 `output/memory_world_tracker_b_2gpu_b80_20261111_run01/step_004000.pth.tar`，SHA256 `e43b09b95ec3a2118f2f4734c04a4517ed2dffb0ca70524c4aa710101a33d788`。
- 数据与边界：LaSOT、VastTrack、TNL2K 训练来源 1:1:1 采样，清单 `output/memory_world_split/manifest.json`，SHA256 `fb9dd296c54d620075cbe7aa6813f6d931502263f6e109f60515d2ccdd27b0f9`；按清单训练/留出。官方初始化可能见过留出视频；片段 loss 不是完整视频 AUC。
- 输入输出：独立目录 `output/memory_world_c_b80_b4000_20261127_run01/`；日志 `/tmp/mvlm_c_b80_b4000_20261127_run01.log`；[W&B](https://wandb.ai/2896287267-hit/mvlm-reproduction/runs/f7v0cdun)。完成状态、指标、模型选择和完整视频在线测试结果：待核实。
- 启动：`CUDA_VISIBLE_DEVICES=5,6 OMP_NUM_THREADS=2 WANDB_RUN_ID=f7v0cdun WANDB_RESUME=allow WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true WANDB_DISABLE_JOB_CREATION=true WANDB_CONSOLE=off /root/miniconda3/envs/mvlm/bin/torchrun --standalone --nproc_per_node=2 tracking/train_memory_world_c_audited.py --config /root/MVLM/experiments/mvlm/mvlm_TF_memory_world_c_b80_b4000_20261127.yaml --manifest /root/MVLM/output/memory_world_split/manifest.json --baseline /root/MVLM/models/MVLM_TF.pth.tar --world-checkpoint /root/MVLM/output/memory_world_tracker_b_2gpu_b80_20261111_run01/step_004000.pth.tar --output /root/MVLM/output/memory_world_c_b80_b4000_20261127_run01 --device cuda --validation-clips 32 --wandb-project mvlm-reproduction --wandb-entity 2896287267-hit --wandb-name memory-world-C-b80-b4000-20261127 --verify-distributed`。
- 前置容量短测见 [两卡 batch 80 短测](MEMORY_WORLD_C_BATCH80_CAPACITY_20261127.md)。

## 启动后核验快照

2026-11-27 约 20:58（北京时间）：完整清单审计及父权重加载通过，两卡训练已完成至少 2 / 1000 步；两步 loss 分别为 0.72653、0.91463，均为有限值。W&B API 确认运行状态为 running 且收到第 1、2 步。两卡系统显存当时分别约 5241 / 5303 MiB；后续状态与结果以实时运行证据为准。

## 中断核查快照

2026-11-27 21:48（北京时间）核查：训练进程已退出，GPU 5/6 已释放；本地 `metrics.jsonl` 和状态快照最后为第 20/1000 步，最后一条指标时间约 21:08:50。状态文件仍显示 `running`，已过期。日志未记录训练完成、异常堆栈或退出信号，具体终止原因未确认。W&B 日志在 21:08 前后出现代理 127.0.0.1:17891 连接重置及拒绝，确认在线同步受影响，但时序接近不足以证明网络直接终止训练。输出目录没有 `last.pth.tar` 等检查点；原配置每 250 步保存，因此无法从第 20 步精确恢复。保留运行目录和日志；新训练如需重启应从 B/4000 使用新目录并重新记录。

## 使用审计缓存重新训练：run02

2026-11-27 21:57（北京时间）启动快照：从 B/4000 重新开始，非从 run01 第 20 步续训；物理 GPU 5、6，每卡 batch 40，全局 batch 80，计划 1000 步，12 帧/片段、前 4 帧预热，种子 20261108，AdamW 初始学习率 1e-5、余弦调度。每 50 步在固定 32 片段验证并保存。编码器、语言侧和 CENTER 冻结；训练记忆读取、世界模型、质量头及融合层。数据来源和采样比例与上文一致，清单和基线哈希及 B/4000 父检查点与上文相同。

按用户要求不重新逐视频审计，复用 run01 完整审计产生的 `data_cache.pth`（SHA256 `7580ca86c3494b6a51703cf6de8c21c5f3301b3b6ca33b37209ae4fe8294cd02`），核对缓存与清单、基线及父检查点的数据签名；这不证明目前磁盘上的视频、标注或语言未改变。缓存来源、签名和限制详见新输出的 `cache_provenance.json`。

配置 `experiments/mvlm/mvlm_TF_memory_world_c_b80_cached_20261127.yaml` SHA256 `6f3d03086365d692772c68792f6b4feaf7ed75d2d25cb7dddccd5e8a7e16e4d8`；缓存入口 `tracking/train_memory_world_c_cached.py` SHA256 `75f8999c5980b594f1c3cc5f153139b139642bf7f69b934ad4aada3aabe77dfd`，启动时为本地未提交代码，快照与启动参数保存在控制目录。训练输出 `output/memory_world_c_b80_b4000_cached_20261127_run02/`；独立进程监督、GPU 资源及日志位于 `output/memory_world_c_b80_b4000_cached_20261127_run02_control/`。

W&B 同步通过独立桥接器 [run02](https://wandb.ai/2896287267-hit/mvlm-reproduction/runs/0tw0edil)，网络中断时本地指标保留并由桥接器重试；这不保证任意中断后自动恢复。启动快照已完成至少第 1 步，桥接器状态 streaming 且已提交第 1 步；最终完成状态、云端同步及完整视频评测待核实。

2026-11-27 23:08（北京时间）只读核查：run02 进程在运行，本地记录完成 119/1000 步，W&B 桥接器处于 streaming 且记录同步到第 119 步；第 50、100 步固定片段验证已完成。此为过程快照，不代表训练完成或完整视频跟踪效果。
