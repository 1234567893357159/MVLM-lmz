# 联合模型 A 阶段两卡训练记录

日期：2026-11-09（北京时间）。分支 `codex/memory-world`。

用户要求用两张卡训练 A 阶段并增加有效显存使用。本文记录本次配置、预检和运行入口；实时状态以服务器状态文件为准。正式运行输出目录为 `/root/MVLM/output/memory_world_stage_a_2gpu_20261109_run02`，不混用短测权重。

## 批量处理调整与验证

原入口逐视频、逐帧运行编码器。新增 `TRAIN.WORLD.BATCHED_ENCODER=true` 后，A 阶段在同一时刻批量执行文本编码、视觉编码、冻结 CENTER、记忆读取、世界模型、质量头和候选评分。每个视频独立维护历史、可靠性、搜索框及内容库，帧间仍顺序展开；GT 只在定位后用于监督和允许的历史写入。

在线单视频定位和训练批量路径共用定位后处理。原开发配置默认关闭该开关；B/C 继续使用原路径。新增检查覆盖不同视频、不同帧间隔、无效标注、候选可靠性回退、窗口以及批量/单样本损失与梯度对应。

- 最终 CPU 回归：26 项通过。完整小型证据见[预检与启动快照](memory_world_stage_a_preflight_20261109.json)。
- 两卡 GPU/NCCL 短测：每卡 256 个片段、2 次真实优化器更新，通过检查点保存与固定片段验证。
- 两卡可训练参数 SHA256 相同，冻结参数未更新，各 rank 的样本索引不重叠。
- 短测目录：`/root/MVLM/output/memory_world_stage_a_setup_20261109/smoke_2gpu_batch256`。

## Batch 实测与选择

以下为 GPU 0 上预加载真实图像后的单批前向/反向检查，未执行优化器更新；不含数据加载、通信和保存时间。新批量实现各档位只做一次测量，用于选择起点，不是长期吞吐结论。

| 每卡片段 batch | PyTorch 峰值分配 MiB | 峰值缓存 MiB | 片段/秒 |
|---:|---:|---:|---:|
| 16 | 1905 | 1980 | 7.67 |
| 32 | 2405 | 2578 | 8.08 |
| 64 | 3519 | 3700 | 8.41 |
| 128 | 5757 | 6390 | 8.62 |
| 256 | 10242 | 11030 | 8.79 |
| 384 | 14726 | 16378 | 8.65 |

选择每卡 256：比开发起点 batch2 增加实际批量与显存使用，384 没有进一步提升观测吞吐。真实两卡短测 rank0 峰值分配约 10266 MiB、缓存约 11560 MiB。系统显存还包含驱动与通信开销，不能与 PyTorch 分配值直接混用。

原始结果在服务器 `output/memory_world_stage_a_setup_20261109/` 下的 `vectorized_profile.json`、`large_batch_profile.json`、短测日志与 `distributed_audit.json` 中。

## 首次正式启动的数据加载修复

`run01` 在完成首次更新前遇到数据加载进程文件句柄上限（1024），已停止并保留日志，未用于后续初始化。修复为每批只跨进程传输四个完整标注张量，在接收端按视频生成视图；同时仅为本训练进程树提高文件句柄软上限到32768，不修改系统全局设置。新增每批256样本的多进程传输回归检查通过。当前正式运行是从同一基线重新初始化的 `run02`。

## 正式配置

配置：[mvlm_TF_memory_world_stage_a_2gpu.yaml](../../experiments/mvlm/mvlm_TF_memory_world_stage_a_2gpu.yaml)。

| 项目 | 设置 |
|---|---|
| 物理 GPU | 0、1，均为 RTX 3090 24 GB |
| 每卡 / 全局 batch | 256 / 512 个 12 帧片段，无梯度累积 |
| 更新次数 | 4000；名义采样 2,048,000 个片段，跳过的无监督批另计 |
| 优化器 | AdamW，学习率 `1e-4`，weight decay `1e-4` |
| 调度与梯度裁剪 | 按更新次数 cosine 至 0；范数上限 `1.0` |
| 可训练模块 | 记忆读取器、世界模型、质量头，共 2,908,167 个参数 |
| 冻结模块 | 视觉/文本编码器、CENTER、融合层 |
| 精度 | 视觉编码器与新增模块 FP32；冻结 CLIP 沿用其库的 CUDA 加载精度；无 AMP |
| 片段 / 无梯度前缀 / 运动历史 | 12 / 4 / 4 帧；采样间隔随机 1～4 |
| 预测历史比例 | 前 1000 步从 0 升至 0.5，验证全部用预测历史 |
| 损失 | 外观 + 运动 + `0.5 ×`候选质量；外观内部特征回归权重 `0.1` |
| 数据来源比例 | LaSOT / VastTrack / TNL2K = 1 / 1 / 1 |
| 加载 worker | 每卡 4，总计 8；每 worker 预取 1 个批次，OpenCV 单线程 |
| 验证和保存 | 每 50 步，固定 32 个验证片段 |
| 监控 | 本地逐步指标、状态文件、10 秒间隔显卡资源采样及 W&B 实时同步 |

大 batch 改变了每次更新的数据量，4000 步的样本预算显著高于早期全局 batch8 的方案。它不能作为仅改变显卡数量的等预算对照；后续消融需记录实际全局 batch 和更新次数。

## 数据划分

固定清单：`/root/MVLM/output/memory_world_split/manifest.json`。审计未启用 `--allow-incomplete`，没有未处理的缺失项。

| 来源 | 训练视频 | 验证视频 |
|---|---:|---:|
| LaSOT | 1008 | 112 |
| TNL2K | 1169 | 130 |
| VastTrack | 41082 | 4565 |
| 合计 | 43259 | 4807 |

已知 TNL2K 异常序列和不足 12 帧的视频在清单中记录为排除项。训练入口再次核对标注、路径签名与视频隔离；元数据完整性检查不等同于逐张图像内容去重或解码检查。

## 启动与监控

本次由 `tracking/supervise_memory_world.py` 启动两卡进程并记录资源。控制目录：

`/root/MVLM/output/memory_world_stage_a_setup_20261109/formal_run02`

监督程序执行的训练命令如下；现有实验运行时不要重复启动。新的复现实验需换成全新输出目录。

```bash
CUDA_VISIBLE_DEVICES=0,1 MVLM_DEVICE=cuda OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/root/miniconda3/envs/mvlm/bin/python -u -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  /root/MVLM/tracking/train_memory_world.py \
  --config /root/MVLM/experiments/mvlm/mvlm_TF_memory_world_stage_a_2gpu.yaml \
  --manifest /root/MVLM/output/memory_world_split/manifest.json \
  --baseline /root/MVLM/models/MVLM_TF.pth.tar \
  --output /root/MVLM/output/memory_world_stage_a_2gpu_20261109_run02 \
  --device cuda --validation-clips 32 --verify-distributed
```

- `launch.json`：准确命令、环境和显卡分配。
- `supervisor_status.json`：进程、完成状态和最近显卡采样。
- `train.log`：训练器完整输出。
- `gpu_resources.jsonl`：两张训练卡的显存、利用率和功率记录。

训练输出目录中的 `run.json`、`config.yaml`、`split.json`、`source/` 和源码哈希记录实际运行来源；`metrics.jsonl` 与 `training_status.json` 记录逐步训练状态及 rank0 的峰值显存。

### W&B 实时监控

运行：[memory-world-A-b512-2gpu-20261109](https://wandb.ai/2896287267-hit/mvlm-reproduction/runs/ngk6l0q4)。北京时间2026-11-09 23:31已通过 W&B API 核验云端状态为 running、前43步已同步，并读取到从第1步开始的损失和两张卡的资源历史。此为连接核验快照，后续进度以运行页面为准。

本次在训练开始后，通过独立进程 `tracking/monitor_memory_world_wandb.py` 补传历史并持续读取现有 JSON 日志，训练未重启。只上传训练/验证数值指标、GPU资源和模型/训练超参数，关闭代码、Git、自动系统元数据采集，不上传数据集、图片和模型权重。代理沿用 `127.0.0.1:17891`；首次连接失败时每30秒重试，连接后的短暂网络中断由 W&B SDK 本地缓存和重试处理。

- `train/*`：总损失、外观误差、运动误差、质量损失、学习率、梯度范数、记忆写入、有效预测比例和吞吐等，横轴为 `train/step`。
- `validation/*`：每50步的固定片段验证损失与有效片段数，横轴为 `validation/step`；首次验证完成前没有曲线。
- `gpu/0/*`、`gpu/1/*`：两张物理GPU的显存、利用率和功率，横轴为从训练启动计算的 `gpu/elapsed_seconds`。

同步器的状态、日志和运行身份均保存在训练输出目录：`wandb_bridge_status.json`、`wandb_bridge.log`、`wandb_run.json`；`wandb_upload_preview.json` 记录上传字段范围。W&B 内部 `_step` 同时计数训练和资源记录，查看训练进度应使用 `train/step` 或 summary 中的 `completed_optimizer_steps`。

需要恢复同步器时，先确认原同步进程已经退出，再单独运行下面的命令；不重新启动训练、不删除 `wandb_run.json` 或同步状态文件。同步器沿用同一 run ID 和本地读取进度，本命令应在独立终端或持久会话中执行。

```bash
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/mvlm/bin/python -u \
  /root/MVLM/tracking/monitor_memory_world_wandb.py \
  --output /root/MVLM/output/memory_world_stage_a_2gpu_20261109_run02 \
  --control /root/MVLM/output/memory_world_stage_a_setup_20261109/formal_run02 \
  --project mvlm-reproduction --entity 2896287267-hit \
  --name memory-world-A-b512-2gpu-20261109
```

中断后使用相同配置、清单、基线、输出目录及两卡环境，在准确训练命令末尾添加 `--resume`，从最近 `last.pth.tar` 恢复。严格恢复禁止更改相关源码和参数。该监督程序记录成功或失败，不会自行改参数或无限重试。

训练到 4000 步并正常退出才算 A 阶段完成；`best_clip_loss.pth.tar` 用于后续 B 阶段初始化，A 本身不作为完整在线精度结论。正式训练耗时应根据持续实测估计，不能沿用预加载单卡片段速度。

## 启动核验快照

北京时间2026-11-09 22:55，`run02` 已连续完成前4次正式更新，Loss依次为0.61864、0.48861、0.43472、0.43351，监督帧数和梯度均有效，无跳过批次。两卡系统显存约11985 MiB/卡；rank0 PyTorch峰值分配约10271 MiB，缓存约11564 MiB。首步含加载初始化约204秒，后续三步约51、68、42秒；这只是早期速度，不能作为精确完成时间。训练仍在后台运行，尚未完成4000步，也没有在线精度结论。
