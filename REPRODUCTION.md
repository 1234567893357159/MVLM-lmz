# MVLM 复现工作区

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

测试集调参结果不作为独立复现证据。官方原论文配置的正式复现实验尚未启动；记忆分支的训练和验证集实验记录见 `work/xmem-wandb/reports/reproduction/` 与该分支交接文件。两类实验的训练目标和证据范围不同，不应混为一谈。

## 大文件与环境

数据集、模型权重、检查点、下载分块、输出视频、预测结果和缓存不加入 Git，原有本地文件保留。
作者原始历史中的示例视频和框架图沿用上游历史。
本机路径为 `/root/MVLM`，环境 Python 为 `/root/miniconda3/envs/mvlm/bin/python`；换机器时需调整本地配置与实验脚本路径。
依赖见 `requirements.txt`，环境与已知协议差异见交接文档。

## 后续实验

结构改进从固定的复现提交创建独立分支。正式实验保存代码提交号、配置、数据划分、随机种子、权重 SHA256、命令和指标；大体积产物另外备份。
