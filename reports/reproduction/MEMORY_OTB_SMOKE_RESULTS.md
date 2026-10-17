# 记忆模块 OTB 小规模真实训练与评测

日期：2026-10-17（Asia/Shanghai）。分支：`codex/xmem-memory`。实验运行时为未提交工作区，运行目录内保存了源码快照与差异；后续代码通过本分支提交发布。

## 结论

真实数据训练、优化器更新、checkpoint 保存和重载、在线记忆读写、预测导出、PRE/NPR/AUC 计算均已跑通。本次仅 20 次更新，测试 3 条短序列，没有精度提升结论；短训模型在该子集上略低于基线。

## 数据与训练设置

- 使用已经下载的 OTB99-Lang，不新增数据下载。
- 训练语言划分来自包内 `OTB_query_train/*.txt`，排除 `._` 文件，共 51 条；与测试语言划分无重叠。已检查 51 条的 29,491 行框标注和对应图像路径，没有发现缺失或非有限框数值；这不是全部图像逐一解码验证。
- 本次只用 Basketball（725 帧）、Bolt（350 帧）、Boy（602 帧）、Car4（659 帧），共 2,336 帧可采样训练图像。
- 测试 Biker（142 帧）、Bird2（99 帧）、Dancer2（150 帧），共 391 帧；与训练图像路径交集为零。两个模型分别完整预测这 391 帧。
- 配置：`experiments/mvlm/mvlm_TF_memory_otb_smoke.yaml`；1 epoch、40 个四帧片段采样、每批 2 个片段，共 20 次更新。单张 RTX 3090，学习率 0.0001，随机种子 1234567890。
- 冻结编码器、文本编码器和 CENTER 头，只更新记忆检索／融合模块；原定位损失按片段各帧平均。训练 epoch 日志耗时约 8.36 秒，不含模型初始化、checkpoint 写入及评测。
- 训练初始化使用现有官方跟踪权重，推理保持与当前工作基线相同的置信度超参数。此处基线配置并非作者原始默认参数。

## 检查点验证

运行目录：`/root/MVLM/output/memory_otb_smoke_20261017_run02/`。

检查点：`checkpoints/train/mvlm/mvlm_TF_memory_otb_smoke/MVLM_ep0001.pth.tar`（相对运行目录）。

- 保存的 epoch 为 1，优化器状态记录 20 次更新。
- 官方 checkpoint 中的基线参数在训练后逐项完全相同。
- 记忆输出投影权重范数从零初始化变为约 0.280003，证实新增参数实际更新。
- 官方初始权重 SHA256：`91c31230f3d355f71104df8191e386632a39ce9d3bbc1f7ce5bf3772b369cc4f`。
- 本次检查点 SHA256：`6fa4da2200706bf081100463be92ac8489730ba64a0046e5c3041902e2699b64`。

## 三序列子集结果

沿用仓库语言评测协议，包括首帧预测设为真值。指标按序列平均；NPR 为归一化误差阈值 0.2 处精确率。

| 模型 | PRE | NPR | AUC |
| --- | ---: | ---: | ---: |
| 相同推理配置的原 MVLM | 91.15 | 64.97 | 58.16 |
| 记忆模块短训 20 次更新 | 90.14 | 64.61 | 57.97 |

这里是 3 条序列的工程验证，不是 48 条 OTB 测试集的完整成绩，不能直接与已有完整 OTB 实验或论文比较。结果没有证明记忆模块有效，也不足以证明该方案无效，需要更充分的训练与固定协议下的消融。

线上实际短期／长期最大条目数：Biker 为 8／6，Bird2 为 8／8，Dancer2 为 8／12，确认使用了正常置信度门控下的历史内容写入。

预测分别保存在：

- `output/test/tracking_results/mvlm/memory_otb_smoke_20261017_run02_baseline/otb99_lang/`
- `output/test/tracking_results/mvlm/memory_otb_smoke_20261017_run02_memory/otb99_lang/`

独立指标缓存分别在 `output/test/result_plots/` 下同名实验目录中，没有覆盖原完整评测缓存。

## 本次发现并修复的问题

1. OTB 训练加载器依赖缺失的划分文件：增加按下载包训练语言文件读取训练划分，并支持显式指定训练序列子集；不允许子集中出现非训练划分序列。
2. 修正该训练加载器默认根路径、语言读取、非语言模式变量初始化、Jogging 双目标标注映射、David 的帧起点及隐藏图像文件过滤。
3. 原批量拼接在 `NUM_WORKER=0` 时调用空对象的 `resize_()`，导致首次训练失败；修正为仅在共享输出张量存在时调整大小。基础检查已覆盖 worker 为 0 和 1 的路径。

首次失败日志保存在 `output/memory_otb_smoke_20261017_run01/train.log`；成功实验使用新目录 run02，没有复用失败结果。

## 复查入口

训练命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/miniconda3/envs/mvlm/bin/python -u -m lib.train.run_training \
  --script mvlm --config mvlm_TF_memory_otb_smoke \
  --save_dir /root/MVLM/output/memory_otb_smoke_NEW_RUN --seed 1234567890
```

训练后验证与评测：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/miniconda3/envs/mvlm/bin/python -u tracking/evaluate_memory_otb_smoke.py \
  --run-dir /root/MVLM/output/memory_otb_smoke_NEW_RUN
```

复跑需使用新的运行目录；评测脚本拒绝覆盖同名预测。日志、配置副本、机器可读指标、权重哈希和逐序列统计见成功运行目录内的 `train.log`、`evaluate.log`、`config.yaml`、`summary.json` 与 `结果说明.md`。
