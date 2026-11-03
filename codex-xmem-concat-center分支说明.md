# codex/xmem-concat-center 分支说明

更新日期：2026-11-03（北京时间）。

## 分支用途与状态

本分支从 `codex/xmem-memory` 的 `a115a9942d37448d11a58faec74f594d0a6a28ca` 创建，
实现“当前图像特征与历史记忆检索结果直接按通道拼接，再由扩展后的 CENTER 头预测”。
这是 MVLM 的内容记忆融合方式对照实验。原 MVLM 方法、视觉/语言编码器和基础跟踪代码
来自原作者，本分支贡献为拼接路径、预测头输入扩展、权重迁移和相应验证。

当前完成代码实现与 CPU 工程验证，尚未训练 concat 模型，也没有 concat 的跟踪精度结果。
原 xmem 分支及主目录 `/root/MVLM` 的训练检查点评测保持独立。

## 与原记忆分支的区别

| 项目 | 原残差模式 `residual` | 本分支新增模式 `concat` |
| --- | --- | --- |
| 当前特征 | 512 维 | 512 维 |
| 门控记忆读出 | 256 维 | 256 维 |
| 融合 | 记忆投影到 512 维后与当前特征相加 | 两路直接拼接为 768 维 |
| CENTER 输入 | 三个分支均为 512→256 | 三个分支均为 768→256 |
| 输出投影 | 保留 `output_proj` | 移除，不增加 768→512 线性层 |
| 初始化 | 输出投影零初始化 | 原 512 通道权重复制，新增 256 通道权重置零 |

检索、门控、容量管理、候选框筛选及写入规则沿用原实现。原始 `feat_x` 继续用于图文评分
和 ROI 提取，只有送入 CENTER 的 `feat_x_mem` 扩大。空库的记忆通道为零，严格先读取历史
再写入当前观测。默认融合模式仍为 `residual`，旧配置不会隐式切换到 concat。

CENTER 增加 1,769,472 个参数，同时移除原融合模块 131,584 个输出投影参数，
相同设置下模型参数净增 1,637,888 个。

## 配置及主要代码

配置入口：[mvlm_TF_memory_concat.yaml](experiments/mvlm/mvlm_TF_memory_concat.yaml)。

| 文件 | 改动 |
| --- | --- |
| `lib/models/mvlm/memory_fusion.py` | 新增 concat 路径，直接输出当前特征与门控读出的拼接 |
| `lib/models/mvlm/head.py` | 扩展三个 CENTER 输入卷积，新增通道零初始化 |
| `lib/models/mvlm/checkpoint.py` | 显式迁移无内容记忆的基线权重，拒绝跨架构误加载 |
| `lib/models/mvlm/mvlm.py` | 接入模式、初始化迁移、无读取消融及冻结头配置校验 |
| `lib/test/tracker/mvlm.py` | 支持 concat 检查点与显式未训练工程检查 |
| `lib/config/mvlm/config.py` | `MODEL.MEMORY.FUSION_MODE` 开关及旧配置隔离 |
| `lib/local_config.py` | 工作区及输出路径跟随当前 worktree，数据集共享读取 |
| `tests/test_memory_concat.py` | 八项 CPU 回归验证 |
| `tracking/check_memory_concat.py` | 轻量回归和真实官方权重完整模型检查入口 |

## 初始化与实验使用

训练从无内容记忆的完整 MVLM 跟踪权重初始化。当前配置指向本机官方
`/root/MVLM/models/MVLM_TF.pth.tar`，编码器和文本冻结，记忆与 CENTER 联合训练。
`TRAIN.MEMORY_FREEZE_HEAD` 必须为 `false`，否则新增的零权重无法学习。
不将原残差记忆检查点或其优化器状态直接迁移为 concat 检查点。
正式推理必须加载已训练的 concat 权重；未训练诊断需要显式打开
`TEST.MEMORY_ALLOW_UNTRAINED`，其输出不作为训练后成绩。

现有配置沿用历史自定义 OTB/LaSOT/TNL2K 划分、1:3:6 抽样、每卡 2 个四帧片段和
4,000 步预算，仅作为开发起点。它依赖旧实验的 `split/manifest.json`，不等于论文标准
三训练集方案；换机器或建立新划分时须调整清单及模型、数据路径。
W&B 默认关闭。后续启动实验应选择空闲 GPU，显式指定新的 `--save_dir` 和独立 `exp_id`，
不复用正在评测或已归档实验的输出目录。

## 已完成验证与限制

2026-11-03 已通过：

- 八项 CPU 回归：空库/混合 batch、迁移与初始化一致性、时序定位损失反传、语言检索梯度、
  编码器冻结、concat 推理重载、错误检查点/冻结头拒绝、配置与输出路径隔离。
- 原残差路径四组工程检查，验证既有功能仍可运行。
- 本机官方权重的真实 FastITPN+CLIP CPU 前向。空库和非空库两帧的中心图、框、尺寸图、
  偏移图相对原 CENTER 的最大绝对差均为 0；三个输入层均为 `[256,768,3,3]`。
- 补丁格式和新增 Python 代码语法检查。

检查命令、完整架构与检查点说明见
[MEMORY_CONCAT_CENTER.md](reports/reproduction/MEMORY_CONCAT_CENTER.md)。
验证仅说明接口、初始化、读写及梯度链路可用，没有测量新结构训练后的 AUC 或推理速度。
后续应在相同数据、初始化、预算和门槛下比较残差、较弱残差及 concat，再冻结方案评测。

本分支仅发布代码、配置和说明；数据、模型、CLIP 缓存、运行日志及大体积结果不加入 Git。
