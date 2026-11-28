# MVLM 参数搜索使用指南

本指南适用于当前分支能够通过 `lib/test/tracker/mvlm.py` 加载的模型。切换分支时，须使用该分支的模型代码、配置和权重；工具不会自动移植其他分支的模型。

## 工具和实验边界

`tracking/sweep_mvlm_generic.py` 单进程执行完整视频搜索并保存预测、状态和汇总。`tracking/sweep_mvlm_parallel.py` 使用相同实验规格，以模型与参数组的组合为任务，支持指定物理 GPU、每卡多进程及实时 Markdown；单个任务内部的视频仍顺序执行。

`heldout` 使用训练来源留出视频，用于开发选参；`benchmark` 使用标准测试视频，在测试集上选参数或检查点的结果属于探索性结果，不能当作独立最终测试。初始化权重可能见过留出视频，应保留此限制。短片段 loss 和短测结果不等于完整视频 AUC。

运行前检查 Git 状态及版本、模型配置、权重哈希、数据路径与划分、GPU、磁盘空间，并确定实验预算。新实验使用新目录，不覆盖已有实验。明确记录参数、视频、首帧规则、坐标路径、指标、种子、硬件和运行状态。比较不同模型时，要核对未被搜索项覆盖的配置是否一致。

## 实验规格

保存 UTF-8 JSON。以下为 memory-world 分支接口示例，不是推荐的选参视频集；相对路径从仓库根目录解析：

```json
{
  "seed": 0,
  "data": {
    "kind": "heldout",
    "manifest": "output/memory_world_split/manifest.json",
    "datasets": ["lasot"],
    "sequences": ["lasot/cat-4"]
  },
  "models": {
    "c1000": {
      "config": "experiments/mvlm/mvlm_TF_memory_world_c_b80_cached_20261127.yaml",
      "checkpoint": "output/memory_world_c_b80_b4000_cached_20261127_run02/step_001000.pth.tar"
    }
  },
  "variants": {
    "original": {},
    "alpha05": {"TEST.MVLM.ALPHA_CORR": 0.5},
    "quality0": {"MODEL.WORLD.QUALITY_WEIGHT": 0.0}
  }
}
```

`models` 和 `variants` 至少各有一项；每组候选应用于每个模型。名称只能含英文、数字、下划线、连字符和点，不能含路径分隔符。空对象沿用各模型自己的参数，不保证跨模型参数相同。支持当前分支已有的 `TEST.MVLM.*`、允许的 `MODEL.WORLD` 和 `MODEL.MEMORY` 推理数值；可靠记忆分支还支持已有的 `POLICY`、`RETENTION`、`FUSION` 数值项。具体允许项以脚本 `apply_overrides` 为准；整数配置要求整数，架构开关和维度不能通过候选修改。不同模型是否实际使用某参数，需要核对其推理代码。

标准测试集可用 `"data": {"kind": "benchmark", "datasets": ["otb99_lang", "lasot", "tnl2k"]}`。可通过 `sequences` 列表指定 `数据集/视频名`；省略则使用所选来源的全部视频。`heldout` 要求本分支有 `lib/train/dataset/world_clips.py`，只检查选中留出视频的标注、语言、可见性与路径指纹，不重新审计训练分区或对图像内容计算哈希。标准测试不会自动排除空语言视频；例如 TNL2K 历史 700 条口径中的 `Fight_video_6-Done` 使用空字符串时，须在报告中明确标注。

## 计划与预检

```bash
/root/miniconda3/envs/mvlm/bin/python tracking/sweep_mvlm_generic.py \
  --spec /tmp/sweep.json --output /tmp/unused-plan --plan
```

`--plan` 不创建输出、不加载模型，但会读取选定视频并计算检查点哈希。正式运行前应使用少量视频测试模型加载、预测有效性和实际吞吐量：

```bash
/root/miniconda3/envs/mvlm/bin/python tracking/sweep_mvlm_parallel.py \
  --spec /tmp/sweep.json --output /tmp/mvlm-parallel-smoke \
  --gpus 5 --workers-per-gpu 2 --smoke-frames 6 --interval 10
```

`--smoke-frames` 截断每条视频，是诊断而不是完整视频评测。每卡两个进程只是示例；应实测显存、GPU 利用率、CPU 和数据读取吞吐量，不要假设进程越多越快。

## 正式运行和实时记录

```bash
/root/miniconda3/envs/mvlm/bin/python -u tracking/sweep_mvlm_parallel.py \
  --spec /tmp/sweep.json --output output/NEW_PARALLEL_SWEEP \
  --gpus 5,6,7 --workers-per-gpu 2 --interval 10
```

`--gpus` 为物理 GPU 编号；工具不会自动寻找空闲卡。按实际资源授权和预算运行，并确认进程、日志和已完成视频数持续增长。只有一个模型及一个参数组时只有一个任务，不会自动将视频分片到多个进程。

输出目录包括：

| 路径 | 内容 |
| --- | --- |
| `LIVE.md` | 周期更新的排队、运行、完成与失败状态；仅完整参数组显示 AUC/PRE/NPR |
| `parallel_manifest.json`、`parallel_status.json` | 规格、代码/配置/权重身份、并发设置及运行状态 |
| `specs/模型/参数组.json`、`logs/模型/参数组.log` | 子任务规格和日志 |
| `jobs/模型/参数组/` | 子任务清单、状态、逐视频结果和完成后的汇总 |
| `summary.json` | 所有子任务完成后的合并结果 |

进程运行期间约按 `--interval` 周期更新 Markdown；监督器停止后文件不会自行更新。实际已记录的结果、日志和进程状态均应共同检查，不能将中途结果写成完整实验。

## 中断、续跑和故障

保持同一规格、Git HEAD、已跟踪代码差异、监督器/工作进程脚本、模型配置和权重、GPU 及并发设置，在原并行命令末尾加入 `--resume`。监督器会核对身份；已完成组跳过，已有合法清单的未完成组由工作进程进一步核对视频等信息后续跑。留出清单哈希也会检查；已完成的标准测试组不会重新审计视频，续跑前须自行确认数据、标注和语言未变。

若任务失败，监督器停止启动新任务并等待已启动任务结束。查看 `parallel_status.json` 和对应日志；若输出目录存在但缺少清单、代码身份不匹配或有残留进程，应先查明状态，不要删除或覆盖结果。随机种子不保证跨硬件或 CUDA 版本逐位一致；随机推理中断后重新设种也可能与连续执行不同。

## 指标与报告

工具采用项目当前分支的 `calc_seq_err_robust`：首帧 GT 初始化并输出，后续在线预测，评分前将 xywh 预测框转换为整数；AUC 为 IoU 0 到 1、间隔 0.05 的 21 个阈值成功率均值，PRE 为中心误差 20 像素，NPR 为归一化中心误差 0.2。数据集内逐视频等权，`macro_AUC` 对数据集等权；具体可见性和无效标注处理以当前分支代码为准。视频计时包含读图，不是纯模型 FPS。

报告时给出每组完整配置、数据集及视频数、权重/哈希、协议、各数据集 AUC/PRE/NPR、逐视频结果、完成状态与限制。跨模型比较需相同视频、首帧规则、坐标路径和参数；在标准测试集上搜索所得最高值是探索性结果。
