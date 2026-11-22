---
name: project-understanding
description: 理解 MVLM 项目的研究目标、模块、配置、依赖和执行流程，适用于项目导览或修改前定位代码。
---

# 项目理解

1. 从仓库根目录读取 [文档入口](../../../REPRODUCTION.md)、[README](../../../REPRODUCTION.md) 和 [实验总览](../../../REPRODUCTION.md)；本分支存在对应资料时按需读 [分支说明](../../../REPRODUCTION.md) 与 [实现说明](../../../REPRODUCTION.md)。确认这些资料的日期和适用分支。
2. 根据问题检查 `lib/models/mvlm/`、`lib/train/`、`lib/test/`、`tracking/`、`experiments/mvlm/`、`requirements.txt` 中相关文件，追踪调用、配置、输入输出和依赖。用文件清单和代码搜索，避开数据、权重、缓存和二进制内容。
3. 交叉核对代码和文档；对过期快照、缺失资料或冲突记录给出来源和待核实问题。区分官方 MVLM 与个人扩展。

输出：研究目标、相关模块及数据流、入口/依赖、证据路径、未确认事项；引用具体文件。验证路径存在，说明哪些流程只是静态阅读而未运行。

本分支资料入口：[README](../../../README.md)、[复现说明](../../../REPRODUCTION.md)、[本分支研究记录](../../../reports/reproduction/README.md)；历史报告需核对日期、配置和适用范围。
