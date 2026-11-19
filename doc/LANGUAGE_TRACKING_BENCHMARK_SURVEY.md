# 语言驱动目标跟踪模型与基准结果调研

整理日期：2026-11-19。本文整理本项目讨论中查阅的公开论文结果，为毕业设计实验设计提供参考；并非覆盖所有历年模型的完整排行榜，也不代表当前所有数据集的最新最佳成绩。

## 1. 任务和指标

语言驱动目标跟踪根据视频和目标的自然语言描述，持续预测目标位置。比较模型时首先区分两种输入条件：

- **纯语言（NL）**：只提供目标描述，不提供首帧真实框或视觉模板。
- **首帧框 + 语言（NL + BBox）**：除语言描述外，还提供首帧目标框；不能把这一条件的结果直接当成纯语言方法的公平比较。

以下结果均以百分数表示，越高越好。AUC 是目标框重叠率成功曲线下的面积；PRE 是基于中心位置误差的精度，具体阈值和聚合方式应以对应评测协议为准。本项目共同评测路径使用 21 个 IoU 阈值的成功率均值、20 像素 PRE 和 0.2 阈值 NPR，指标先逐视频计算再等权平均。数据集划分、初始化、训练数据、首帧计分、不可见帧处理及推理参数均可能影响比较。

## 2. 纯语言跟踪：论文中报告的对比

下表全部摘自 [MVLM 论文表 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Park_MVLM_Template-Free_Tracking_via_Vision-Language_Margin_Confidence_and_Memory-Gated_Tracking_CVPR_2026_paper.pdf)。每格为 **AUC / PRE**；“—”表示原表未报告。年代只是发表时间，这些方法不是严格的代际继承关系。MGIT 的 JointNLT 与 UVLTrack-B 结果在原论文中标记为使用公开代码得到。

| 模型 | 发表 | TNL2K | LaSOT | OTB99 | MGIT |
|---|---|---:|---:|---:|---:|
| TNL2K-I | CVPR 2021 | 11.0 / 6.0 | 51.0 / 49.0 | 19.0 / 24.0 | — |
| CTRNLT | CVPRW 2022 | 14.0 / 9.0 | 52.0 / 51.0 | 53.0 / 72.0 | — |
| JointNLT | CVPR 2023 | 54.6 / 55.0 | 56.9 / 59.3 | 59.2 / 77.6 | 59.2 / 43.8 |
| QueryNLT | CVPR 2024 | 53.3 / 53.0 | 54.2 / 55.0 | 61.2 / 81.0 | — |
| UVLTrack-B | AAAI 2024 | 55.7 / 57.2 | 57.2 / 61.0 | 60.1 / 79.1 | 56.1 / 44.6 |
| MambaVLT | CVPR 2025 | 58.4 / 58.9 | 55.8 / 57.2 | 58.9 / 79.2 | 64.6 / 50.3 |
| **MVLM（本项目基线）** | **CVPR 2026** | **57.8 / 60.9** | **60.7 / 65.5** | **60.7 / 84.3** | **63.5 / 55.5** |

MVLM 相对 MambaVLT 的 AUC 变化：TNL2K −0.6、LaSOT +4.9、OTB99 +1.8、MGIT −1.1 个百分点；PRE 分别为 +2.0、+8.3、+5.1、+5.2 个百分点。这反映出不同数据集和指标可能出现不同趋势，不能推断每个新模型在所有条件下都优于旧模型。

## 3. 首帧框 + 语言：另一种评测条件

同样摘自 [MVLM 论文表 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Park_MVLM_Template-Free_Tracking_via_Vision-Language_Margin_Confidence_and_Memory-Gated_Tracking_CVPR_2026_paper.pdf)。每格为 **AUC / PRE**；不要与上表纯语言数字直接排名。

| 模型 | 发表 | TNL2K | LaSOT | OTB99 | MGIT |
|---|---|---:|---:|---:|---:|
| SUTrack-B224 | AAAI 2025 | 65.0 / 67.9 | 73.2 / 80.5 | 70.8 / 93.4 | — |
| MambaVLT | CVPR 2025 | 66.5 / 69.9 | 66.6 / 71.0 | 72.2 / 94.4 | 69.9 / 58.9 |
| DUTrack-256 | CVPR 2025 | 64.9 / 70.6 | 73.0 / 81.1 | 70.9 / 93.9 | 70.0 / 65.4 |
| MVLM | CVPR 2026 | 66.9 / 73.0 | 72.7 / 80.4 | 70.6 / 94.0 | 71.7 / 66.3 |

其他对比可参考 [DUTrack 论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Dynamic_Updates_for_Language_Adaptation_in_Visual-Language_Tracking_CVPR_2025_paper.pdf)及 [SVLTrack 论文表 2](https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_Learning_to_Track_Instance_from_Single_Nature_Language_Description_CVPR_2026_paper.pdf)；后者包含不同监督条件，引用时须核对具体设置。

## 4. 本项目的本地结果

下表来自 [项目实验总览](EXPERIMENTS.md)，不是上面论文表的直接复现排名。三组采用相同原始推理参数和共同坐标路径，项目采用首帧 GT 计分；联合基线关闭新增内容记忆、世界模型和质量分支。OTB 为本地 48 条，LaSOT 280 条，TNL2K 700 条。B1025 由留出短片段验证损失选择，B4000 是预定终点。具体初始化和时序协议以[训练与结果记录](../reports/reproduction/MEMORY_WORLD_TRAINING_AND_RESULTS_20261112.md)为准。

| 模型 | OTB AUC / PRE / NPR | LaSOT AUC / PRE / NPR | TNL2K AUC / PRE / NPR |
|---|---:|---:|---:|
| 官方权重，共同路径基线 | 56.84 / 79.38 / 70.02 | 57.45 / 61.05 / 63.75 | 55.31 / 57.67 / 70.74 |
| 联合 B1025 | 56.28 / 78.66 / 68.75 | 57.65 / 62.14 / 63.94 | 56.25 / 59.68 / 71.47 |
| 联合 B4000 | 55.54 / 77.61 / 67.48 | 57.33 / 62.03 / 63.31 | 56.56 / 60.02 / 71.76 |

B4000 相对共同路径基线的 AUC 变化分别为 OTB −1.30、LaSOT −0.12、TNL2K +1.25 个百分点。B 阶段也更新了原跟踪器的视觉编码器、语言投影和 CENTER；因此该系统对照不能单独证明新增模块的因果贡献。历史测试集参数搜索属于探索，不是独立最终测试结果。

## 5. 如何理解提升幅度

不存在每一代模型必须提升多少的统一标准。以 [MVLM 论文表 2](https://openaccess.thecvf.com/content/CVPR2026/papers/Park_MVLM_Template-Free_Tracking_via_Vision-Language_Margin_Confidence_and_Memory-Gated_Tracking_CVPR_2026_paper.pdf)为例，论文的 A3 到 A4 设置在 TNL2K、LaSOT、OTB99 上 AUC 分别提高 0.6、0.1、0.9 个百分点；这是该论文特定消融的结果，不是行业门槛。

作为本项目的经验性工作目标，强基线上稳定提升约 0.5–2 个 AUC 百分点值得进一步分析；更小提升也可能有价值，但需要特别关注逐视频差异、重复训练和不确定性。更大数字也不能替代公平的训练数据、初始化、预算及评测协议。以上范围既不是本科毕业标准，也不是 CCF-C 的录用标准，不能保证发表。

建议优先比较：同初始化、同训练预算的原 MVLM 微调；仅记忆；仅世界模型；两者联合。结合遮挡或重现等场景、速度、显存和参数量分析。模型参数量应注明是否包含冻结的文本编码器；目前本调研未核实统一口径的参数量，不作数值比较。

## 参考资料

- [MVLM 论文（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Park_MVLM_Template-Free_Tracking_via_Vision-Language_Margin_Confidence_and_Memory-Gated_Tracking_CVPR_2026_paper.html)与[官方代码](https://github.com/inha-vllab/MVLM)。
- [DUTrack 论文（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Dynamic_Updates_for_Language_Adaptation_in_Visual-Language_Tracking_CVPR_2025_paper.pdf)。
- [SVLTrack 论文（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_Learning_to_Track_Instance_from_Single_Nature_Language_Description_CVPR_2026_paper.pdf)。
- [本项目实验总览](EXPERIMENTS.md)及[训练与结果记录](../reports/reproduction/MEMORY_WORLD_TRAINING_AND_RESULTS_20261112.md)。
