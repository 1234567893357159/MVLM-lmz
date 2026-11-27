# 记忆世界模型 C 阶段 batch 80 容量短测

- 时间：2026-11-27（北京时间）；性质：容量和多卡路径预检，不是正式训练或精度实验。
- 代码：`codex/memory-world`，Git `264290c4d2a65eb89b68636ceddfcf822e8fa01d`；配置为原 C 配置调整为每卡 batch 40、两卡全局 batch 80、两步、每步 12 帧、前 4 帧预热。优化器、损失和种子沿用原 C 配置，种子 `20261108`。
- 初始化：官方基线 `models/MVLM_TF.pth.tar` 与 B/4000 `output/memory_world_tracker_b_2gpu_b80_20261111_run01/step_004000.pth.tar`。清单 `output/memory_world_split/manifest.json`，使用 B 运行的 `data_cache.pth` 加快短测，并检查清单哈希；未重新逐视频审计，不能代替正式训练的数据核验。
- 硬件：RTX 3090，物理 GPU 5、6；每个进程 PyTorch 显存上限设为卡总容量的 85%。短测启动时两卡空闲。
- 结果：两卡完成 2 次优化器更新并正常退出；训练损失分别约 0.72653、0.91463；每步有效监督帧分别为 620、623。第 1、2 步分别约 50.04、39.38 秒；短测包含首次初始化，且只有两步，不能代表长期吞吐。32 片段正式验证未执行；本次仅验证 2 个固定片段，不作为效果结论。
- 显存：rank 0 PyTorch 峰值 allocated 3537.20 MiB、reserved 4796 MiB；rank 1 峰值未单独记录。两卡未触发 85% 限额或 OOM，但不能据此保证长期稳定。
- 多卡审计：两卡完成第 2 步，采样索引分别 0–39 和 40–79；可训练参数哈希一致、可训练参数发生变化、冻结参数未变化。审计为总体参数变化，未分别证明每个模块的梯度；短测不证明未来训练收敛或跟踪精度提升。
- 证据目录：`output/memory_world_c_batch80_capacity_20261127_run01/`，包含 `run.json`、`metrics.jsonl`、`distributed_audit.json`、`validation_000002.json` 和检查点。该入口使用 `/tmp/mvlm_c_batch80_capacity_20261127.py` 与 `/tmp/mvlm_c_batch80_capacity_20261127.yaml`，临时文件不保证长期保留。输出的状态快照停留在 `running`，终端输出显示 `Completed stage C: 2 optimizer steps on 2 ranks` 且进程退出码为 0。
- 启动命令：`CUDA_VISIBLE_DEVICES=5,6 OMP_NUM_THREADS=2 /root/miniconda3/envs/mvlm/bin/torchrun --standalone --nproc_per_node=2 /tmp/mvlm_c_batch80_capacity_20261127.py --config /tmp/mvlm_c_batch80_capacity_20261127.yaml --manifest /root/MVLM/output/memory_world_split/manifest.json --baseline /root/MVLM/models/MVLM_TF.pth.tar --world-checkpoint /root/MVLM/output/memory_world_tracker_b_2gpu_b80_20261111_run01/step_004000.pth.tar --output /root/MVLM/output/memory_world_c_batch80_capacity_20261127_run01 --device cuda --validation-clips 2 --verify-distributed`。
