# MVLM 复现工作区

世界模型分支的最新环境和 GPU 训练记录见 [世界模型交接](WORLD_MODEL_HANDOFF.md) 与 [2026-10-26 训练记录](reports/reproduction/WORLD_GPU_TRAINING_20261026.md)。下方结果为继承的基线历史，不能作为世界模型训练结果。

2026-10-26 最新：世界模型 A/B 训练及跨数据集测试均已完成，随后按用户要求完成 13 组本地参数评测。当前世界模型最佳 OTB/LaSOT/TNL2K AUC 为 59.61/60.36/57.16，尚未超过论文或此前调参官方基线；详见 [参数调试最终记录](reports/reproduction/WORLD_PARAMETER_TRANSFER_20261026.md)。用户要求后续调试/训练关闭 W&B，仅保存本地产物。

本仓库保留官方 MVLM 的 Git 历史，保存当前复现适配、评测脚本与实验配置。

- `SESSION_HANDOFF.md`：详细交接历史，下载进度仅为历史快照。
- `experiments/mvlm/`：独立实验配置。`mvlm_TF.yaml` 为较早的 OTB 调参配置，并非官方默认或最新最优。
- `reports/reproduction/original_config.yaml`：原始参数快照。
- `reports/reproduction/`：精简指标、实验说明和验证记录；其中已有报告沿用原 output 路径，完整产物保留在服务器。
- `downloads/*.py`、`downloads/*.sh`：数据准备和监控脚本；公开来源元数据随代码保存。

## 当前结果

| 数据集 | 原始 AUC | 测试集调参最佳 AUC |
| --- | ---: | ---: |
| LaSOT | 57.77 | 61.06 |
| OTB99-Lang（48 条测试序列） | 56.48 | 60.56 |

测试集调参结果不作为独立复现证据。上表为继承的官方基线历史；世界模型训练与最新结果见文首链接。

## 大文件与环境

数据集、模型权重、检查点、下载分块、输出视频、预测结果和缓存不加入 Git，原有本地文件保留。
作者原始历史中的示例视频和框架图沿用上游历史。
本机路径为 `/root/MVLM`，环境 Python 为 `/root/miniconda3/envs/mvlm/bin/python`；换机器时需调整本地配置与实验脚本路径。
依赖见 `requirements.txt`，环境与已知协议差异见交接文档。

## 后续实验

结构改进从固定的复现提交创建独立分支。正式实验保存代码提交号、配置、数据划分、随机种子、权重 SHA256、命令和指标；大体积产物另外备份。
