# 通用 MVLM 推理参数搜索

`tracking/sweep_mvlm_generic.py` 适用于**当前工作树的 `lib/test/tracker/mvlm.py` 能加载的模型与权重**：提供每个模型的 YAML 和检查点，以及一组共享候选参数。不同分支的架构不自动移植；联合权重由现有 `load_joint` 检查架构和阶段，其他权重沿用跟踪器原加载规则。运行前需核对模型架构、权重、依赖及所用分支。固定候选先由研究者明确指定，脚本不自动寻找全局最优。

## 实验文件

保存为 JSON（路径相对仓库根目录或绝对路径）：

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
    "b4000": {
      "config": "experiments/mvlm/mvlm_TF_memory_world_c_b80_cached_20261127.yaml",
      "checkpoint": "output/memory_world_tracker_b_2gpu_b80_20261111_run01/step_004000.pth.tar"
    },
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

该例针对 memory-world 分支，仅展示接口，不是推荐的选参样本；其他分支须替换为该分支的配置及权重。`variants` 的空对象沿用各模型配置原值；若要同参数比较不同模型，必须显式覆盖所有相关参数，或先核对各配置的实际值。支持覆盖当前分支已存在的 `TEST.MVLM.*` 数值、记忆质量/新颖度/衰减/一致性阈值、世界模型质量和融合强度，以及可靠记忆 `POLICY`、`RETENTION`、`FUSION` 下的数值；具体允许项见 `apply_overrides`，整数项须使用整数。不允许用搜索项更改模型架构。不同模型配置里的其余推理选项也需自行核对。每组均使用同一视频清单。`heldout` 依赖当前分支具有 `lib/train/dataset/world_clips.py`，缺少该模块的分支可用 `benchmark`，不可直接套用该留出清单。`datasets` 和 `sequences` 可省略；省略后分别选择全部来源和视频。`sequences` 是 `数据集/视频名` 列表，指定时必须全部存在且不重复。它逐条检查所选留出视频的标注、语言、可见性和路径指纹，不重新审计训练分区，不计算图像内容哈希，也不自动断言初始化权重未见过验证视频。显式提供完整来源可能需要读取大量数据。

标准测试集可用 `"data": {"kind": "benchmark", "datasets": ["otb99_lang", "lasot", "tnl2k"]}`；同样可选 `sequences`。在标准测试集搜索或选择检查点是探索性结果，不能作为独立最终测试。各视频集必须存在且可由当前环境加载；不自动排除空语言视频。

## 使用与输出

```bash
# 只审计选定视频、打印任务规模；不创建输出、不加载模型
/root/miniconda3/envs/mvlm/bin/python tracking/sweep_mvlm_generic.py \
  --spec /tmp/sweep.json --output /tmp/mvlm-sweep --plan

# 仅用于验证加载和数据路径：截断视频，指标不得当完整视频结果
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/mvlm/bin/python tracking/sweep_mvlm_generic.py \
  --spec /tmp/sweep.json --output /tmp/mvlm-sweep-smoke --device cuda:0 --smoke-frames 6

# 完整视频；必须使用新目录，提前确认任务量、GPU 和授权
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/mvlm/bin/python tracking/sweep_mvlm_generic.py \
  --spec /tmp/sweep.json --output output/NEW_SWEEP_DIRECTORY --device cuda:0
```

中断后用同一命令、同一设备及 `--resume` 恢复；脚本核对实验文件内容、选定视频数据指纹、权重与配置哈希、Git HEAD、跟踪代码的已跟踪差异哈希和入口脚本哈希，再跳过已完成视频。不要手工修改结果文件；恢复时仅核对结果的身份与帧数，不验证结果文件的内容哈希。运行期间不要修改代码、输入或结果。随机种子已记录，但不保证跨硬件及 CUDA 版本逐位确定；运行环境、实际 GPU 和必要的依赖版本需另行记录。断点续跑后随机状态从初始种子重新设定，若使用随机推理实现，与不中断运行未必相同。

`manifest.json` 记录实验身份及选中的逐视频指纹，`status.json` 标记进度或错误，`results/模型/参数/数据集/视频.json` 包含每帧预测和指标，`summary.json` 按视频等权计算各数据集 AUC/PRE/NPR，另给各数据集等权的 macro AUC。首帧 GT、整数 xywh、AUC 21 阈值、PRE@20、NPR@0.2，使用项目 `calc_seq_err_robust`。`--smoke-frames` 产生的是诊断片段，清楚标为非完整视频。帧读取包含在 wall time 内，不能视为纯模型 FPS。

每次只启动一个设备上的单进程顺序评测，不接管其他 GPU；运行时间和资源需求按实际输入测量。新目录使用独占创建；不删除旧结果。模型各参数组在同一设备顺序运行；未实现跨 GPU 自动调度或自动排名。
