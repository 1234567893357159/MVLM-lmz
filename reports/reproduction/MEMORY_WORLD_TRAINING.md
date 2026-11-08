# 记忆库与世界模型联合版本：训练方案和参数

更新日期：2026-11-08。适用分支：`codex/memory-world`，首版实现提交：`076e29c`。

导航：[分支总览](../../BRANCH_MEMORY_WORLD.md) · [架构与实现说明](MEMORY_WORLD_IMPLEMENTATION.md) · [实验索引](README.md)。

本文对应当前代码和 YAML 的实际行为。默认方案为交叉注意力融合，按 **A：预测学习 → B：跟踪适配 → C：联合微调** 执行。A、B 是得到可在线评测模型的基本流程，C 用于进一步实验；是否采用 C 的结果由同一留出集的完整视频在线评测决定。

当前只完成了代码与预检验证，尚未进行正式训练。下面的 4000 步等数值是开发起点，不能当作已经验证的最优参数或精度结论。本文仅说明运行方式，不启动训练。

## 运行顺序与路径约定

先准备固定清单，再执行 A → B，对 B 做完整视频在线评测，之后决定是否投入 C 或其他消融。联合训练使用 `tracking/train_memory_world.py`，本指南的参数对应这一入口。

命令使用当前服务器的绝对路径：仓库 `/root/MVLM`，解释器 `/root/miniconda3/envs/mvlm/bin/python`。换机器时需替换这些路径，并按根目录 README 安装依赖；基线权重、数据和 CLIP 缓存需要单独准备，不随 GitHub 仓库提供。

首次 GPU 运行前，先用独立的临时配置和新输出目录做短测，确认有效监督、梯度、显存与保存/重载正常，再用未改动的正式配置和全新目录开始实验。短测不得通过修改正式运行配置后续训，混入正式预算。

## 1. 训练什么

从 `/root/MVLM/models/MVLM_TF.pth.tar` 初始化完整 MVLM 基线，而不是从头训练编码器，也不是直接拼接两个独立分支已经训练的权重。

视觉编码器 FastITPN-B 和文本编码器 CLIP ViT-L/14 在 A/B/C 中都冻结。默认交叉注意力配置中的原 CENTER 头也全程冻结。训练的重点是记忆读取器、世界模型、候选质量头和融合适配层。

记忆库本身是每个视频片段的运行时状态，没有需要优化的“库参数”。需要学习的是如何检索记忆、利用记忆预测未来、把预测融入当前观测。写入库的是定位后提取的原始视觉 ROI，预测生成的特征不会写回库。

| 阶段 | 更新的模块 | 主要冻结模块 | 新增模块初始学习率 | 默认更新步数 |
|---|---|---|---|---|
| A：预测学习 | 记忆读取器、世界模型、候选质量头 | 编码器、CENTER、融合层 | `1e-4` | 4000 |
| B：跟踪适配 | 融合适配层 | 编码器、CENTER、记忆读取器、世界模型、质量头 | `1e-4` | 4000 |
| C：联合微调 | 记忆读取器、世界模型、质量头、融合层 | 编码器、CENTER | `1e-5` | 4000 |

B 阶段即使冻结 CENTER，跟踪损失仍可通过 CENTER 反向传到融合层。C 阶段不是解冻整网训练。

## 2. 数据划分与片段采样

使用 LaSOT、VastTrack、TNL2K 的训练视频。划分工具默认从每个来源的训练视频中留出约 10% 作为验证集，其余用于适配训练；训练和验证按视频隔离，不把官方测试视频混入训练。

- 数据来源采样比例：LaSOT : VastTrack : TNL2K = `1 : 1 : 1`。先按比例选择来源，再在来源内部按视频均匀抽样。
- 每个样本是一个 12 帧片段；片段内部使用固定间隔，间隔在 1～4 中随机选取，并受视频长度约束。
- 每进程每步处理 2 个独立片段，每个片段都新建记忆库和历史状态。
- 前 4 帧只定位并建立历史，不计算训练损失，也不保留训练计算图；后 8 帧在监督有效时计算损失。
- 同一片段使用一致的亮度增益增强，范围为 `0.9～1.1`；不随机翻转方向性语言对应的图像。
- 标注、可见性、裁剪后 ROI 等条件决定监督是否有效，实际有损失的帧可能少于 8 帧。
- 已知帧数与标注不一致的 `Fight_game_004-Done` 被明确排除；缺失或异常数据默认报错。

```bash
cd /root/MVLM

/root/miniconda3/envs/mvlm/bin/python /root/MVLM/tracking/prepare_world_split.py \
  --lasot-root /root/MVLM/data/lasot \
  --vasttrack-root /root/MVLM/data/vasttrack \
  --tnl2k-root /root/MVLM/data/tnl2k \
  --seed 20261022 \
  --val-fraction 0.1 \
  --clip-length 12 \
  --output /root/MVLM/output/memory_world_split/manifest.json
```

数据根目录和官方基线权重在当前工作区存在，但本文没有执行全量扫描；完整性由上述命令检查。清单只需生成一次，A/B/C 和对照实验使用同一份文件。输出清单已存在时不会覆盖。

`--allow-incomplete` 会显式允许部分语料，并把缺失项写进清单；仅在有意使用子集时添加，不应为绕过数据问题默认开启。官方 MVLM 初始化权重可能已经见过适配留出的训练视频，因此该留出集用于选适配方案，最终结论仍需独立测试集支持。

## 3. 每个片段如何展开

对片段按时间顺序执行：

1. 初始状态为空；第一个处理帧先进行全局定位，不用 GT 预填记忆。
2. 历史达到 4 帧后，用最近的 ROI、位置、质量、上一已完成帧的池化视觉特征、语言及更久的内容记忆预测下一采样时刻。
3. 编码当前图像。A 阶段旁路预测注入和质量重排；B/C 阶段融合预测与当前观测，再输出候选框。
4. 根据当前帧 GT 计算允许的监督损失；GT 不进入定位函数。
5. 定位结束后，提取当前原始视觉 ROI，按可靠性规则更新历史和记忆，然后处理下一帧。

当前历史和库中的观测都停止梯度，不做跨整段历史的反向传播。片段独立，状态不跨训练样本继承。默认 12 帧只能提供较短的建库过程，无法充分检验容量为 16 的长期库；研究长时记忆时，应单独做更长片段、更长预热的实验，并先测显存与吞吐。

### A 阶段的历史过渡

A 阶段训练允许在“完成当前帧预测之后”用 GT 框提取该帧观测，供后续帧读取。对于 GT 和 ROI 有效的帧，使用预测观测的概率为：

`p_pred(step) = 0.5 × min(step / 1000, 1)`

也就是从 0 逐步升到 0.5，1000 步后保持 0.5；其余概率使用 GT 观测。无法使用有效 GT 时走预测观测路径。B、C 和所有验证都使用模型自己的预测历史。

`HISTORY_RAMP_STEPS=1000` 调的是历史来源比例；`WARMUP_FRAMES=4` 指片段前缀建库。这两项都不是学习率预热。

## 4. 损失函数

### A：未来预测监督

未来外观的目标，是冻结编码器在当前帧 GT 区域提取的 `2×2` ROI 特征，不是重建 RGB 图像。

```text
L_app = mean(1 - cosine(pred_roi, target_roi))
        + 0.1 × SmoothL1(LayerNorm(pred_roi), LayerNorm(target_roi))

L_motion = SmoothL1(pred_delta, target_delta)

L_quality = BCEWithLogits(grid_quality_logits, detached_candidate_IoU)

L_A = 1.0 × L_app + 1.0 × L_motion + 0.5 × L_quality
```

运动目标为相对上一观测框的归一化中心位移和宽高对数比例：`Δcx/w_prev, Δcy/h_prev, log(w/w_prev), log(h/h_prev)`。世界模型接收真实采样帧号，支持片段内不等于 1 的帧间隔。

质量头输出 196 个网格位置的事前质量分数，监督目标是实际候选框与当前 GT 的 IoU，IoU 标签停止梯度。质量头没有提前读取当前候选宽高来产生预测。

### B：跟踪监督

```text
L_B = L_track = 2.0 × L_GIoU + 5.0 × L1 + 1.0 × L_focal
```

用当前有效 GT 监督融合后的定位输出，使适配层学会何时、在何处、以多大强度使用预测信息。B 不继续优化未来外观、运动或质量头。

### C：联合监督

```text
L_C = L_track + L_app + L_motion + 0.5 × L_quality
```

各项按有效监督条件启用，片段中的有效帧损失平均后反向传播。`MODEL.WORLD.QUALITY_WEIGHT=0.25` 是推理候选重排强度，与训练质量损失权重 `TRAIN.WORLD.QUALITY_LOSS_WEIGHT=0.5` 是不同参数。

## 5. 当前默认参数

### 优化与训练预算

| 参数 | 默认值 / 行为 |
|---|---|
| 优化器 | AdamW，`betas=(0.9, 0.999)`、`eps=1e-8`，使用 PyTorch 默认值 |
| Weight decay | `1e-4` |
| 学习率 | A/B：`1e-4`；C：`1e-5` |
| CENTER 学习率 | `HEAD_LR=1e-5`；默认交叉注意力配置冻结头，因此不生效 |
| 调度 | 每次更新后执行 cosine，`T_max=MAX_STEPS=4000`，最低学习率为 0 |
| 梯度裁剪 | 全部可训练参数的范数上限 `1.0` |
| 每进程 batch | `2` 个片段；全局 batch = `2 × 进程数` |
| 片段长度 / 历史长度 / 无梯度前缀 | `12 / 4 / 4` |
| 最大采样间隔 | `4` |
| 每阶段更新预算 | `4000` 次有效优化器更新 |
| 保存与片段验证间隔 | 每 `200` 步，以及阶段最后一步 |
| 固定验证片段 | 默认 `32`，由命令行 `--validation-clips` 控制 |
| 控制台输出间隔 | `50` 步；首步也输出 |
| DataLoader workers | 每进程 `2` |
| 训练随机种子 | `20261108`；数据划分种子独立，为 `20261022` |
| BN 运行统计 | 固定；默认头冻结，concat 更新头时其 BN 仿射参数可训练 |
| 数值精度 | 当前入口使用常规 FP32，没有 AMP 或梯度累积开关 |

这里按更新步数训练。`TRAIN.EPOCH`、`DATA.TRAIN.SAMPLE_PER_EPOCH`、`DATA.MAX_SAMPLE_INTERVAL`、`TEST.EPOCH` 等沿用字段不控制本入口的更新预算或片段间隔；应修改 `TRAIN.WORLD.MAX_STEPS` 和 `TRAIN.WORLD.MAX_STRIDE`。

### 参数修改位置

| 要调整的内容 | 配置键或命令行参数 |
|---|---|
| 新增模块学习率、头学习率 | `TRAIN.LR`、`TRAIN.WORLD.HEAD_LR` |
| 每进程片段数、加载进程数 | `TRAIN.BATCH_SIZE`、`TRAIN.NUM_WORKER` |
| 片段、无梯度前缀、最大间隔 | `TRAIN.WORLD.CLIP_LENGTH`、`WARMUP_FRAMES`、`MAX_STRIDE` |
| 更新预算、保存周期 | `TRAIN.WORLD.MAX_STEPS`、`SAVE_INTERVAL` |
| 预测历史比例及过渡长度 | `TRAIN.WORLD.PREDICTED_HISTORY_PROB`、`HISTORY_RAMP_STEPS` |
| 外观、运动、特征回归、质量损失权重 | `TRAIN.WORLD.APPEARANCE_WEIGHT`、`MOTION_WEIGHT`、`FEATURE_WEIGHT`、`QUALITY_LOSS_WEIGHT` |
| 结构与融合模式 | `MODEL.MEMORY`、`MODEL.WORLD`、`MODEL.FUSION` |
| 固定验证片段数 | `--validation-clips`（默认 32，属于命令行参数） |
| 数据留出比例与划分种子 | 划分工具的 `--val-fraction`、`--seed` |

表中缩写键与同一行首个键共享前缀。新实验开始前修改相应 YAML，并保存实验配置副本；A/B/C 的 MODEL/DATA/TEST 必须与父阶段一致。增大片段长度时还需让清单覆盖足够长的视频。参数更改属于新实验，不能覆盖已完成实验目录。

### 结构与记忆参数

| 参数 | 当前值 |
|---|---|
| 当前搜索图像 | `224×224`，局部搜索因子 `4.0` |
| 原始视觉网格 | `14×14`，即 `196×512` |
| 单帧目标 ROI | `2×2`，即 `4×512` |
| 世界模型 | 隐藏维度 `256`，Transformer `2` 层、`4` 个头，FFN 维度 `1024` |
| 记忆读取隐藏维度 / key 维度 | `256 / 128` |
| 短期 / 长期记忆容量 | `8 / 16` 条目标观测，读取时按帧号去重 |
| 最低入库质量 | `0.5`，另受候选可靠性与 ROI 有效性约束 |
| 长期库新颖性阈值 | `0.1` |
| 短期 / 长期时间衰减系数 | `0.1 / 0.01` |
| 写入外观一致性阈值 | `0.2`，作用于已有有效预测时的普通预测写入 |
| 融合模式 / 强度 | `cross_attention / 1.0` |
| 候选质量重排权重 | `0.25`，只调整原可靠候选集合内的排序 |

默认沿用 MVLM 候选可靠性参数：`ALPHA_CORR=1.0`、`TAU=0.1`、`OMEGA=0.4`、`LAMBDA=0.4`、`PSI_OUT=0.0`。成对比较时保持不变。

## 6. 单卡训练命令

使用当前已有的 `mvlm` Python 环境及相同 CLIP 预训练缓存。下列每个输出目录必须是新目录，避免覆盖已有实验。

### A：从官方基线训练预测模块

```bash
/root/miniconda3/envs/mvlm/bin/python /root/MVLM/tracking/train_memory_world.py \
  --config /root/MVLM/experiments/mvlm/mvlm_TF_memory_world_stage_a.yaml \
  --manifest /root/MVLM/output/memory_world_split/manifest.json \
  --baseline /root/MVLM/models/MVLM_TF.pth.tar \
  --output /root/MVLM/output/memory_world_stage_a \
  --device cuda:0
```

### B：从 A 的检查点训练融合层

```bash
/root/miniconda3/envs/mvlm/bin/python /root/MVLM/tracking/train_memory_world.py \
  --config /root/MVLM/experiments/mvlm/mvlm_TF_memory_world_stage_b.yaml \
  --manifest /root/MVLM/output/memory_world_split/manifest.json \
  --baseline /root/MVLM/models/MVLM_TF.pth.tar \
  --world-checkpoint /root/MVLM/output/memory_world_stage_a/best_clip_loss.pth.tar \
  --output /root/MVLM/output/memory_world_stage_b \
  --device cuda:0
```

B 是首个适合完整视频在线跟踪评测的阶段。A 检查点用于后续训练，不用于联合模型在线精度结论。

### C：从 B 的检查点小学习率联合微调

```bash
/root/miniconda3/envs/mvlm/bin/python /root/MVLM/tracking/train_memory_world.py \
  --config /root/MVLM/experiments/mvlm/mvlm_TF_memory_world_stage_c.yaml \
  --manifest /root/MVLM/output/memory_world_split/manifest.json \
  --baseline /root/MVLM/models/MVLM_TF.pth.tar \
  --world-checkpoint /root/MVLM/output/memory_world_stage_b/best_clip_loss.pth.tar \
  --output /root/MVLM/output/memory_world_stage_c \
  --device cuda:0
```

默认 A+B 共 8000 步，A+B+C 共 12000 步。还没有测过真实 GPU 的显存、单步耗时和完整训练时长，不能据此保证某张显卡可运行或承诺训练小时数。显存不足时可在新实验开始前将各阶段 `TRAIN.BATCH_SIZE` 降为 1，并记录实际全局 batch；不要在严格续训途中修改配置。

## 7. 恢复、多卡和监控

### 断点恢复

在原阶段的原命令后加 `--resume`，使用相同输出目录。入口从 `last.pth.tar` 恢复模型、优化器、学习率状态、采样位置和各进程随机状态。

例如恢复上面的 B 阶段：

```bash
/root/miniconda3/envs/mvlm/bin/python /root/MVLM/tracking/train_memory_world.py \
  --config /root/MVLM/experiments/mvlm/mvlm_TF_memory_world_stage_b.yaml \
  --manifest /root/MVLM/output/memory_world_split/manifest.json \
  --baseline /root/MVLM/models/MVLM_TF.pth.tar \
  --world-checkpoint /root/MVLM/output/memory_world_stage_a/best_clip_loss.pth.tar \
  --output /root/MVLM/output/memory_world_stage_b \
  --device cuda:0 --resume
```

`last.pth.tar` 每次保存间隔更新，并非每步保存，因此恢复的是最近已保存状态。严格恢复要求配置、数据清单及签名、基线、相关源码和进程数保持一致；不能通过修改 `MAX_STEPS` 再加 `--resume` 静默延长原实验。

也应保持 `--validation-clips` 等命令行设置一致，避免改变选取 best 的比较条件；这一参数不在 YAML 的严格恢复检查范围内。

### 多卡

把单卡命令的 Python 启动部分改为：

```bash
/root/miniconda3/envs/mvlm/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  /root/MVLM/tracking/train_memory_world.py \
  --config /root/MVLM/experiments/mvlm/mvlm_TF_memory_world_stage_a.yaml \
  --manifest /root/MVLM/output/memory_world_split/manifest.json \
  --baseline /root/MVLM/models/MVLM_TF.pth.tar \
  --output /root/MVLM/output/memory_world_stage_a_2gpu \
  --device cuda
```

默认 2 卡的全局 batch 为 4 个片段，固定 4000 步会比单卡看到更多采样片段，不能直接当作相同训练预算。多卡 GPU/NCCL 尚未验证；已有双进程 CPU/Gloo 一致性检查通过。

### 监控与产物

默认写本地日志；可在训练命令追加 `--wandb-project mvlm-memory-world --wandb-name memory-world-A` 启用已登录账号的 W&B。B、C 使用不同运行名。仅设置配置里的 `WANDB` 字段不会启用此训练入口的上传。

- `metrics.jsonl`：训练损失、监督帧数、有效预测数、记忆写入数等。
- `training_status.json`：当前阶段状态和已完成更新数。
- `run.json`、`config.yaml`、`split.json`、`source/`：运行、参数、数据和源码记录。
- `step_000200.pth.tar` 等：周期检查点。
- `last.pth.tar`：最近保存状态，用于恢复。
- `best_clip_loss.pth.tar`：固定验证片段损失最低的状态。

完整检查点包含冻结视觉骨干；只有冻结 CLIP 本体被省略。默认每阶段保存约 20 个周期检查点，并另外保存 last/best，应在训练前核实磁盘空间。

## 8. 如何判断训练有效

A 重点观察外观余弦误差、运动误差、质量损失，以及记忆写入和有效预测是否持续出现。当前入口没有自动计算“复制上一帧特征”“静止框”等简单预测基准，不能仅凭训练损失下降声称世界模型优于这些基准。

B/C 的固定片段验证用于检查训练过程，但 `best_clip_loss` 不等于完整视频 AUC 最优。应分别对 B、C 检查点做完整视频在线评测，比较 AUC、PRE、NPR，并查看遮挡、重现和干扰场景。

```bash
/root/miniconda3/envs/mvlm/bin/python /root/MVLM/tracking/evaluate_memory_world.py \
  --config /root/MVLM/experiments/mvlm/mvlm_TF_memory_world_stage_b.yaml \
  --manifest /root/MVLM/output/memory_world_split/manifest.json \
  --checkpoint /root/MVLM/output/memory_world_stage_b/best_clip_loss.pth.tar \
  --output /root/MVLM/output/memory_world_validation_b \
  --device cuda:0
```

评测 C 时对应替换配置、检查点和新输出目录。汇总写入 `summary.json`。默认沿用项目首帧 GT 计分口径，首帧 GT 不预填新增内容库；这与严格首帧预测计分应区分。

使用同一清单评测同坐标基线，作为首轮对照：

```bash
/root/miniconda3/envs/mvlm/bin/python /root/MVLM/tracking/evaluate_memory_world.py \
  --config /root/MVLM/experiments/mvlm/mvlm_TF_memory_world_coordinate_baseline.yaml \
  --manifest /root/MVLM/output/memory_world_split/manifest.json \
  --checkpoint /root/MVLM/models/MVLM_TF.pth.tar \
  --output /root/MVLM/output/memory_world_validation_coordinate_baseline \
  --device cuda:0
```

比较各数据集的 `per_dataset` 指标以及等权数据集平均的 `macro_AUC`。默认脚本计时包含读图，不能直接作为纯模型 FPS。正式结果应记录检查点 SHA256、配置、数据清单、初始化来源和评测协议。

## 9. 对照实验与 concat 版本

所有对照使用相同数据清单、初始化、采样规则和明确的更新预算，并记录完整视频验证结果。

| 实验 | 配置名（位于 `/root/MVLM/experiments/mvlm/`） | 流程 |
|---|---|---|
| 同坐标基线 | `mvlm_TF_memory_world_coordinate_baseline.yaml` | 官方权重直接评测 |
| 只更新原 CENTER | `mvlm_TF_head_only_stage_b.yaml` | 从官方权重直接 B |
| 仅记忆 | `mvlm_TF_memory_only.yaml` | 从官方权重直接 B |
| 仅世界模型 | `mvlm_TF_world_only_stage_a.yaml` → `mvlm_TF_world_only.yaml` | A → B |
| 记忆 + 世界模型 | `mvlm_TF_memory_world_stage_a.yaml` → `mvlm_TF_memory_world_stage_b.yaml` | A → B，可加 C |
| 去质量头 | `mvlm_TF_memory_world_no_quality_stage_a.yaml` → `mvlm_TF_memory_world_no_quality_stage_b.yaml` | A → B |
| 通道拼接融合 | `mvlm_TF_memory_world_concat_stage_a.yaml` → `mvlm_TF_memory_world_concat_stage_b.yaml` | A → B |

默认交叉注意力通过残差保持 512 维。concat 版本把原始 512 维与 256 维预测读出拼为 768 维，扩展 CENTER 的输入；原基线通道权重复制，新增输入通道权重零初始化。concat B 同时训练融合层和扩展 CENTER，后者学习率为 `1e-5`。

切换 concat 时必须成对使用 concat A/B，并使用独立输出目录；不能直接加载 cross-attention A 的检查点。当前没有现成的 concat C 配置，不能把默认 C 直接接到 concat B 后；若另做 concat C，需复制 C 配置并保持 `MODEL.FUSION.MODE=concat`、`TRAIN.WORLD.TRAIN_HEAD=true`，以及与父阶段一致的 MODEL/DATA/TEST 设置。

若只想测已有联合检查点的质量重排影响，可另做 `QUALITY_WEIGHT=0` 的同权重评测；这与移除质量头并重新训练是两种不同实验。第一轮先完成默认 A→B 和同坐标基线评测，再判断是否投入 C、concat 及更长历史实验。
