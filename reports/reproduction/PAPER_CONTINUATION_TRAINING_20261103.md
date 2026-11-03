> 2026-11-03完成更新：本阶段已于北京时间20:16正常结束（exit_code=0），达到E80 / global_step100,000，E80检查点已保存。训练总耗时约9小时13分钟，GPU4、5已释放。逐轮评测仍在GPU6、7运行。W&B在线摘要曾停在step90,960并标记crashed，后台正在补同步；这不是本地训练失败。下方启动核验保留为历史快照。

# MVLM 第70轮至第80轮续训实验

正式启动：北京时间 **2026-11-03 11:03:48**。从E70、global_step 87,500恢复，到E80、100,000次更新结束；额外10轮、12,500次更新、1,000,000次样本抽取。该实验检验相同配方继续低学习率训练是否有收益，不自动扩展到80轮之后。

## 训练与资源

- 物理GPU4、5训练，每卡batch40，全局batch80；每卡2个数据加载进程；无梯度累积，无AMP。
- 从E70完整检查点恢复模型和AdamW状态，622个优化器参数状态；学习率预测头5e-5，编码器和文本投影5e-6。
- 保持原StepLR设置（衰减周期54），恢复last_epoch70，至80轮不再衰减。
- LaSOT / VastTrack / TNL2K训练采样比例1:1:1，每轮100,000条，1,250次更新；保留原异常序列排除规则。
- 除TRAIN.EPOCH和WANDB.NAME外，配置与上一阶段一致。原检查点不覆盖，输出冻结源码运行。
- 两卡恢复审计均通过：E70 / step87,500，抽样权重一致、抽样AdamW动量一致。父检查点未保存RNG和数据加载器状态，因此不保证逐位连续重放。

## 在线监控与记录

- [新W&B窗口](https://wandb.ai/2896287267-hit/mvlm-reproduction/runs/jrhbvcza)，run ID `jrhbvcza`；旧运行独立保留。
- [实时训练状态](/root/MVLM/output/paper_continuation_e70_e80_20261103_run01/training_status.json)
- [训练日志](/root/MVLM/output/paper_continuation_e70_e80_20261103_run01/train.log)
- [第70轮检查点校验](/root/MVLM/output/paper_continuation_e70_e80_20261103_run01/checkpoint_preflight.json)
- [rank0恢复检查](/root/MVLM/output/paper_continuation_e70_e80_20261103_run01/resume_audit_rank0.json)；[rank1恢复检查](/root/MVLM/output/paper_continuation_e70_e80_20261103_run01/resume_audit_rank1.json)
- [W&B上传状态](/root/MVLM/output/paper_continuation_e70_e80_20261103_run01/wandb_sync_status.json)。保留已有可用代理与完整本地W&B记录；训练结束后检查在线完整性并按需补同步。
- 输出目录：`/root/MVLM/output/paper_continuation_e70_e80_20261103_run01`。

## 两卡评测

物理GPU6、7专用于评测，每卡12个视频进程、每进程batch1；GPU4、5保持训练用途。E71～E80逐轮保存后加入队列，各测OTB99-Lang、LaSOT、TNL2K，共30组；评测忙时排队，显卡等待检查点时不占用显存。

继续使用原main首帧真实框接口/计分与固定的各数据集历史参数；TNL2K保持700条，不重新调参。E56、E60、E66、E70列为历史对照，重点比较预先指定终点E80与E70，同时完整记录所有轮次。

[逐轮评测文档](PAPER_CONTINUATION_TESTS_20261103.md)；[实时队列状态](/root/MVLM/output/paper_continuation_eval_20261103_run01/status.json)。

此前三集平均最佳E66为58.381；E70为58.216。新增检查点属于已反复观察测试集上的探索性结果；选出的最好分数不能替代独立验证或重复种子实验。

## 时间参考

上一阶段同样两卡训练10轮约10小时；本阶段可先按约10小时估计，实际以新增日志为准。单组双卡完成30组评测约20小时，自E71开始与训练重叠。此次按用户要求只用两卡训练、两卡测试，不自动将4、5卡转为额外测试组。

## 启动核验快照

第71轮前50次更新已完成（global_step87,550）：两卡日志loss均值约0.5925，训练IoU约87.27%，均为有限值。这只是前50批统计，不能与整轮均值直接比较。两卡显存各约23,577MiB；6、7号卡等待E71期间约3MiB。

W&B在线运行创建成功，初次远程检查状态为running；随后SDK查询认证核验和直接摘要请求出现网络超时，暂未确认在线指标已追上本地步数。本地W&B记录已验证包含恢复后的正常指标。后台监控继续重试，并在训练完成后按需补同步；不以网页暂时滞后判断训练失败。
