# MVLM 文档入口

本目录集中保存当前项目的导航与交接；详细实验文档和历史证据继续保留在 `reports/reproduction/` 和相应分支中。

建议新会话依次阅读：

1. [当前项目交接](HANDOFF.md)：目标、实现、训练状态、结果和注意事项。
2. [分支与文档导航](BRANCHES.md)：各分支职责、Git 内路径及本机工作目录。
3. [各分支实验结果总览](EXPERIMENTS.md)：训练数据、参数、测试结果与协议边界。
4. [实验文档索引](../reports/reproduction/README.md)：详细训练、评测、消融及历史记录。
5. [毕设完成计划](THESIS_COMPLETION_PLAN.md)：剩余工作、核心对照与消融、场景分析、C 阶段决策及论文交付清单。

2026-11-24新增：[可靠记忆库M6正式实验记录](../reports/reproduction/RELIABLE_MEMORY_M6_RESULTS_20261124.md)，包含4000步训练、90视频候选验证、三测试集完整参数搜索、错误写入案例与旧XMem同视频参考。其标准测试搜索已完成，不能把开发验证75.4383与标准测试最高59.24/61.64/56.93混为同一口径。

工具导航：[项目工具与技能目录](TOOLS.md)。

其他入口：[项目首页](../README.md) · [联合模型分支说明](../BRANCH_MEMORY_WORLD.md) · [SSH 与代理](../SSH_DESKTOP_PROXY.md) · [历史会话交接](../SESSION_HANDOFF.md)。

各记录中的运行状态和进程号是带时间的快照；操作前重新检查 Git、进程、GPU 和实际输出。数据、权重、视频预测及完整日志通常保存在服务器，不随 Git 上传。
