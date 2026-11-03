# MVLM 第60轮至第70轮续训运行记录

正式续训已于北京时间 **2026-11-02 11:45:33** 启动。从原实验的第60轮、75,000次更新恢复，训练第61～70轮，目标87,500次更新。当前状态以运行目录的 `training_status.json` 为准。

## 监控与实际参数

- [本次续训 W&B 页面](https://wandb.ai/2896287267-hit/mvlm-reproduction/runs/lu6xjpyl)，独立run ID：`lu6xjpyl`。
- 父训练：`7ln28ook`，原实验的完整60轮和评测记录继续保留。
- 物理GPU **4、5**，每卡batch **40**，全局batch **80**，无梯度累积，未启用AMP。
- AdamW状态恢复；预测头学习率 **5e-5**，编码器和文本投影 **5e-6**；调度器恢复到第60轮，第61～70轮不再衰减。
- LaSOT/VastTrack/TNL2K_train维持1:1:1；每轮100,000样本、1,250次更新；数据清单、增强和损失保持原配方。
- 正式输出：`/root/MVLM/output/paper_continuation_e60_e70_20261102_run01`。训练从该目录的冻结源码启动。

## 恢复与启动核验

单步预检在独立目录完成：原E60的75,000步恢复后，完成一次更新至75,001；文本投影和预测头样本参数发生更新，全部保存的模型浮点张量为有限值。正式运行重新读取原E60副本，没有使用预检后的检查点。

两张卡均生成恢复审计，确认epoch=60、global_step=75,000、调度器last_epoch=60，622项AdamW参数状态恢复；抽样模型张量及AdamW一阶/二阶动量与原检查点一致。

- [预检结果](/root/MVLM/output/paper_continuation_e60_e70_20261102_run01/preflight_validation.json)
- [GPU rank0恢复审计](/root/MVLM/output/paper_continuation_e60_e70_20261102_run01/resume_audit_rank0.json)
- [GPU rank1恢复审计](/root/MVLM/output/paper_continuation_e60_e70_20261102_run01/resume_audit_rank1.json)

W&B在线API启动核验已读到第 **75,060** 步：采样batch的Loss约0.69040、IoU约0.86953，run状态running；API配置确认每卡40、全局80及父run来源。这是启动核验快照，不是实时进度，也不是整轮平均或测试成绩。

- [在线上传核验快照](/root/MVLM/output/paper_continuation_e60_e70_20261102_run01/wandb_live_verification.json)
- [实时训练状态](/root/MVLM/output/paper_continuation_e60_e70_20261102_run01/training_status.json)
- [训练日志](/root/MVLM/output/paper_continuation_e60_e70_20261102_run01/train.log)

服务器本次直连W&B不可达，使用已经验证可用的代理。另有低频在线状态检查；如果训练正常结束但在线指标未完整同步，会针对这个新run进行完成后的补同步。本地完整W&B历史继续保留。状态见运行目录的 `wandb_sync_status.json`，不以网页状态单独判断训练是否完成。

## 检查点与评测

每轮保存E61～E70。达到第70轮即结束，不自动扩展到更晚轮次。原E60副本只作为恢复种子，不覆盖父实验。

第61～70轮每个检查点保存后，评测队列将在物理GPU4、5和6、7两组空闲时并行依次测试三个数据集，共30组；忙时后续检查点排队，维持每卡12个视频进程、每进程batch1及原main固定参数。等待期间不占用这两张卡。结果维护在[续训评测记录](PAPER_CONTINUATION_TESTS_20261102.md)。

## 解释边界

这是原60轮配方之后的探索性延长实验，不替代原配方的正式结果。E56为已观察测试集上的高分参考，E60为续训起点；新模型仍应分别报告与两者的差值，不能把测试引导的最高分称为独立测试。

父检查点未保存完整随机数和数据加载器状态，恢复不等于位级连续重放。这里明确恢复模型、优化器、轮次、步数和学习率。

[原训练完成报告](PAPER_TRAINING_RESULTS_20261101.md)；[续训方案](PAPER_CONTINUATION_PLAN_20261102.md)。
