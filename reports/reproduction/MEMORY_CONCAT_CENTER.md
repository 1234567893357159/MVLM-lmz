# 记忆读出直接拼接到扩展 CENTER 头

日期：2026-11-03；分支：`codex/xmem-concat-center`。
基于 `codex/xmem-memory` 的 `a115a9942d37448d11a58faec74f594d0a6a28ca` 创建独立 worktree。
分支总览和文件导航见 [中文分支说明](../../codex-xmem-concat-center分支说明.md)。

## 改动目标与数据流

保留原有短期/长期记忆、外观/语言条件两路检索、空匹配槽、读取门控和在线写入规则。
新增 `MODEL.MEMORY.FUSION_MODE: concat`，将当前特征与门控后的记忆读出沿通道拼接：

```text
FastITPN 搜索特征 X [B,196,512] ──────────┐
                                         ├─ concat [B,196,768] ─→ CENTER
两路历史检索 R [B,196,256] ─→ 门控 g×R ──┘
```

拼接路径不再经过 `output_proj`，不做 768→512 投影，也不与当前特征做残差相加。
CENTER 中心、尺寸、偏移三个分支的第一层从 512→256 改为 768→256，后续层保持原结构。
当前 3×3 配置下，CENTER 增加 1,769,472 个参数；同时移除残差路径的 256→512 输出投影。
编码器输出仍为 512 维。图文相关性计算与历史 ROI 写入继续使用原始 `feat_x`，
仅送入 CENTER 的 `feat_x_mem` 变为 768 维；每个当前空间位置拼接的是其注意力读出，
不把原始历史条目直接附加到空间 token 序列。

默认值仍为 `residual`，旧 YAML 保持原架构；同进程切换配置时会重置融合模式，防止串用。

## 初始化、训练和检查点

- 从官方或相同基础 MVLM 架构的无内容记忆检查点初始化：复制三个输入卷积的原 512 通道，
  新增 256 通道置零，其余权重原样加载。这样初始化时即使存在记忆，也保留原预测。
- 拼接的记忆值来自非零初始化的检索模块，绕过原零初始化 `output_proj`。
  首次更新可训练新增输入卷积，随后梯度可进入检索模块，不会形成双重零初始化的阻断。
- 空库样本的记忆通道和门控为零。临时关闭 `model.memory_enabled` 时，同样补零记忆通道，
  以支持已扩展预测头上的无读取消融；训练后它不等同于原始未微调的 CENTER 基线。
- 编码器/文本冻结，记忆模块和 CENTER 联合训练。若 concat 配置冻结 CENTER，构建时会报错，
  因为冻结的零权重会让新增记忆无法影响预测。
- 初始化只迁移基线的三个输入卷积，不迁移残差记忆模型，也不迁移跨架构优化器状态。
  训练好的 concat 检查点按其真实维度重载；正式推理拒绝无完整记忆权重的基线。
  `TEST.MEMORY_ALLOW_UNTRAINED` 仅供显式工程检查。

## 独立配置和路径

入口：`experiments/mvlm/mvlm_TF_memory_concat.yaml`。
它复制已有 4,000 步联合训练配置，仅选择 concat 并默认关闭 W&B，没有启动任何训练。
数据仍为旧自定义 OTB/LaSOT/TNL2K 划分、采样比例 1:3:6，不能称为论文标准训练配方。
原始数据清单和官方权重可共享读取，现有实验目录不作为新实验的续训目录。

此 worktree 的 `lib/local_config.py` 从实际代码目录计算工作区、预测结果、图表、
TensorBoard 及默认保存路径；数据集仍共享 `/root/MVLM/data/`。
训练时应显式提供新的 `--save_dir` 和可用 GPU；代码实现不授权恢复已暂停实验。

## 验证命令

在本分支 worktree 根目录执行，均可仅用 CPU：

```bash
CUDA_VISIBLE_DEVICES='' MVLM_DEVICE=cpu OMP_NUM_THREADS=1 \
  /root/miniconda3/envs/mvlm/bin/python tracking/check_memory_concat.py

CUDA_VISIBLE_DEVICES='' MVLM_DEVICE=cpu OMP_NUM_THREADS=1 \
  /root/miniconda3/envs/mvlm/bin/python tracking/check_memory_concat.py --full-model

CUDA_VISIBLE_DEVICES='' MVLM_DEVICE=cpu OMP_NUM_THREADS=1 \
  /root/miniconda3/envs/mvlm/bin/python tracking/check_memory_module.py
```

八项轻量回归覆盖：混合 batch 空库掩码、基线权重迁移/初始预测一致性、真实时序跟踪损失
对新增通道和检索参数的反传、冻结编码器、concat 检查点推理重载、未训练推理保护、
残差模型/冻结头错误配置拒绝、旧配置隔离及独立输出路径。
完整模型检查使用本机官方权重和 CLIP 缓存，验证真实 FastITPN 前向、512/768 维接口和初始化一致性。

2026-11-03 本机检查结果：8 项 CPU 回归全部通过，原残差路径的四组检查全部通过；
官方权重完整模型在空库/非空库两帧上，中心图、预测框、尺寸图、偏移图相对原 CENTER
的最大绝对差均为 0。编码器/文本冻结与预测头可训练状态检查通过，未占用评测 GPU。

本次仅实现与工程验证，没有训练新的模型或测量跟踪 AUC，不能据此判断 concat 的精度收益。
后续对照应保持初始化、数据划分、预算与写入门槛一致，比较残差、较弱残差及 concat；
各自训练新的检查点，不直接以原残差记忆权重充当 concat 模型。
