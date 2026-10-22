# 世界模型实现与运行说明

日期：2026-10-22（北京时间）。分支 `codex/world-model`，工作目录 `/root/MVLM/work/world-model`。

## 已实现范围

输入最近四帧目标状态及语言，使用隐藏维度 256、2 层、4 注意力头 Transformer，配合四个未来外观查询和一个未来状态查询，单步预测目标 ROI 表征与相对框变化。预测器 2,113,284 个参数；交叉注意力、空间先验和门控残差融合模块 658,945 个参数。合计新增约 277 万参数，不含已有 CENTER 头。

世界模型阶段 A 只更新预测器；阶段 B 冻结视觉／文本编码器及世界模型，只更新融合模块和 CENTER 头。冻结模块固定为 eval，阶段 B 的 CENTER 参数及其 BatchNorm 统计均更新。独立训练入口隔离原训练器；误用原训练器加载开启世界模型的 YAML 会报错并提示正确入口。

不包含内容记忆库、去噪扩散、未来视频生成、多步自回归或候选框权值头。历史不足四帧时不使用世界预测。最新历史无效或有效历史少于两条时不注入；这是一种保守起点，不支持凭推演无限维持遮挡期跟踪。

## 文件与接口

| 文件 | 职责 |
| --- | --- |
| `lib/models/mvlm/world_state.py` | ROI、运动参数化、坐标转换及每序列历史 |
| `lib/models/mvlm/world_model.py` | 状态与语言编码、轻量 Transformer、未来输出 |
| `lib/models/mvlm/imagination_fusion.py` | 空间对齐、观测一致性与门控残差 |
| `lib/models/mvlm/world_checkpoint.py` | 基线／阶段 A／阶段 B 权重校验 |
| `lib/models/mvlm/mvlm.py` | 可选装配及阶段冻结模式 |
| `lib/train/dataset/world_clips.py` | 官方训练视频清单审计与片段采样 |
| `lib/train/actors/world_model.py` | 两阶段因果展开、损失及基线候选选择 |
| `tracking/prepare_world_split.py` | 从训练视频构造固定验证划分 |
| `tracking/train_world_model.py` | 独立 A/B 训练、保存和恢复 |
| `tracking/evaluate_world_model.py` | 固定验证集的完整在线 PRE/NPR/AUC |
| `lib/test/tracker/mvlm.py` | 在线预测、融合、历史更新及语言切换重置 |
| `lib/test/parameter/mvlm.py` | 从当前代码目录加载 YAML，各实例配置隔离 |

公共几何约定：历史状态为全图归一化 cxcywh；裁剪区域为全图归一化 xywh；ROI 接口为裁剪归一化 xywh。局部裁剪坐标使用实际四舍五入后的像素起点。世界模型开启时，预测框回映射采用这一约定；关闭时保留原 tracker 路径。因此“零融合与基线特征／预测头输出一致”不等于声称开启世界模式后的全序列坐标与原 tracker 逐帧完全一致。

## 数据准备

本轮没有启动下载或修改数据。本机历史上主要准备了测试数据，正式训练数据是否齐备应单独检查。LaSOT 必须使用仓库官方训练名单，TNL2K 只读取根目录下 `train/`，VastTrack 只读取官方训练名单对应的 `train/`。禁止把现有测试视频改名或软链成训练数据。

在新工作目录运行，使用项目 Python：

```bash
cd /root/MVLM/work/world-model
/root/miniconda3/envs/mvlm/bin/python tracking/prepare_world_split.py \
  --lasot-root /root/MVLM/data/lasot \
  --tnl2k-root /root/MVLM/data/tnl2k \
  --vasttrack-root /root/MVLM/data/vasttrack \
  --output /root/MVLM/output/world_split/split.json
```

命令默认要求提供的数据集完整，不完整时报错。可以只提供已准备好的训练数据根目录；`--allow-incomplete` 显式允许部分训练语料并记录遗漏，只适用于预实验，不能描述成完整官方训练配方。

每个数据集按视频独立划分训练／验证，默认验证比例 10%，种子 20261022。加载时检查序列身份、真实目录和图像路径不交叉，验证标注长度、语言、有效框及 LaSOT 遮挡标记。数据签名覆盖标注、可见性、语言及真实图像路径；不对全部图像内容做 SHA256，不保证消除语义近重复视频。每帧实际使用时解码，失败立即报错。

## 阶段 A：预测预训练

```bash
/root/miniconda3/envs/mvlm/bin/python tracking/train_world_model.py \
  --config experiments/mvlm/mvlm_TF_world_stage_a.yaml \
  --manifest /root/MVLM/output/world_split/split.json \
  --baseline /root/MVLM/models/MVLM_TF.pth.tar \
  --output /root/MVLM/output/world_stage_a_run01 \
  --device cuda:0
```

八帧片段，前四帧形成历史，后续每一帧均先从过去预测，再以当前真实观测构造监督，最后更新历史。外观目标取冻结编码器的真值 ROI，使用余弦误差及归一化特征 SmoothL1；运动为相对中心与对数尺度变化 SmoothL1。外观按可见性和 ROI 有效性屏蔽；运动按标注和上一历史有效性屏蔽。

训练历史从真值／观测混合开始，在前 1000 个优化器步中将预测历史选择概率由 0 线性增至 0.5。搜索裁剪由上一历史状态和原置信度门控驱动，首帧全局搜索，不用当前真值决定当前裁剪。片段使用一致的亮度增强，不做可能改变语言左右语义的随机翻转。验证始终使用预测历史。

记录外观复制、静止运动及恒速运动的预测误差对照。当前训练采样为固定间隔一帧、按视频均匀抽样；尚未实现困难属性定向采样和数据集配比控制。

## 阶段 B：跟踪适配

```bash
/root/miniconda3/envs/mvlm/bin/python tracking/train_world_model.py \
  --config experiments/mvlm/mvlm_TF_world_stage_b.yaml \
  --manifest /root/MVLM/output/world_split/split.json \
  --baseline /root/MVLM/models/MVLM_TF.pth.tar \
  --world-checkpoint /root/MVLM/output/world_stage_a_run01/best_clip_loss.pth.tar \
  --output /root/MVLM/output/world_stage_b_run01 \
  --device cuda:0
```

阶段 B 重新加载同一官方基线，并加载和冻结阶段 A 预测器；对基线权重哈希、数据清单、数据签名、编码与推理设置进行一致性检查。只优化融合模块和 CENTER，使用原 GIoU/L1/热图定位损失。历史来自预测，不从真值写入；按在线候选置信度选择局部／全局裁剪，跨帧观测停止梯度。真值只监督定位，不参与预测状态输入。

起始学习率：世界模型／融合 1e-4，CENTER 1e-5；默认全局 batch 2、4000 步、每 200 步保存，余弦衰减。这些仅为开发配置，尚未验证收敛。当前入口为单进程单设备，batch 内片段逐条展开，不提供 DDP 或 AMP；多卡正式规模训练需后续扩展。

三套新 YAML 的置信度参数均明确取自 `reports/reproduction/original_config.yaml`，不使用旧 OTB 调参配置。比较时应使用 `mvlm_TF_world_baseline.yaml`，不要把不同推理设置的成绩直接相减。

## 验证、保存与恢复

训练保存固定随机种子、清单和标注／路径签名、官方权重 SHA256、配置、源码快照／哈希、优化器、学习率状态、采样位置及 Python/NumPy/PyTorch RNG。阶段 A 保存独立预测器；阶段 B 保存可直接推理的完整跟踪权重，省略可从缓存加载的冻结 CLIP 本体。输出目录必须是新目录；恢复时显式增加 `--resume`，并使用相同配置及数据。源码变化时拒绝静默恢复。

训练过程的 `validation_*.json` 是固定短片段验证，`best_clip_loss.pth.tar` 仅按片段损失筛选，不代表完整视频最优 AUC。正式候选应逐个进行完整验证，再按验证指标选择，最后才评测标准测试集。

```bash
/root/miniconda3/envs/mvlm/bin/python tracking/evaluate_world_model.py \
  --config experiments/mvlm/mvlm_TF_world_stage_b.yaml \
  --manifest /root/MVLM/output/world_split/split.json \
  --checkpoint /root/MVLM/output/world_stage_b_run01/step_002000.pth.tar \
  --output /root/MVLM/output/world_validation_run01_step2000 \
  --device cuda:0
```

评测使用原 `calc_seq_err_robust` 规则，PRE@20、NPR@0.2、21 个重叠阈值的成功曲线均值 AUC，先按序列平均，再分别报告数据集均值及数据集宏平均。保留原首帧初始化／评分约定；无模板世界历史不写入首帧真值。耗时包含解码等开销，不冒充模型纯推理 FPS。

基线完整验证使用同一命令，将配置改为 `mvlm_TF_world_baseline.yaml`、权重改为官方权重，并使用独立输出目录。标准测试仍可使用原 `tracking/test.py` 和 `tracking/analysis_results.py`，配置名为 `mvlm_TF_world_stage_b`，权重必须是阶段 B 检查点。

## 已完成验证及限制

```bash
/root/miniconda3/envs/mvlm/bin/python tracking/check_world_model.py -v
/root/miniconda3/envs/mvlm/bin/python tracking/check_world_integration.py \
  --baseline /root/MVLM/models/MVLM_TF.pth.tar \
  --image /root/MVLM/data/otb99_lang/OTB_videos/BlurBody/img/0133.jpg --device cpu
```

轻量检查覆盖几何、因果约束、序列隔离、无效 ROI、融合旁路、梯度与冻结范围、权重校验、配置隔离、清单重叠拒绝、两个阶段训练入口，以及阶段 A 从中间检查点恢复后最终参数逐项一致。

官方权重 CPU 检查确认实际特征 `[1,196,512]`、零残差预测头输出逐项相同、A/B 梯度路径正确，并通过真实 Actor 短片段与 tracker 五帧历史／语言重置检查。该检查使用同一现有图像重复构成短片段，并临时放宽可靠性阈值以覆盖融合路径；它是接口烟雾检查，不是训练、数据协议验证或精度实验，不保存模型权重。

本轮 CUDA 初始化失败，未验证 GPU 训练、显存、速度或完整数据训练收敛；未产生正式训练结果。仅微调 CENTER、简单运动先验注入等完整实验对照尚未运行，也未提供其专用训练模式。
