# XMem-inspired target content memory

## Status and scope

Implemented on `codex/xmem-memory`. This is a target-level short/long-term memory inspired by XMem, not a port of its dense video segmentation model. The residual Transformer world model and candidate quality head are not implemented in this stage. No formal training or dataset-wide memory evaluation has been run, and no accuracy improvement is claimed.

A subsequent real-data smoke experiment completed 20 optimizer updates on four OTB training sequences and evaluated three disjoint test sequences with the saved checkpoint. See `MEMORY_OTB_SMOKE_RESULTS.md`; this verifies the training/evaluation pipeline, not convergence. The OTB training loader now supports the downloaded training-query split and explicit subsets; zero/one-worker batch collation is covered by the checks.

The original `mvlm_TF.yaml` is unchanged. Memory defaults to disabled. `mvlm_TF_memory.yaml` is a separate starting configuration, using the same inference confidence parameters as the existing working baseline; those parameters are not the original author defaults or the later tuning optimum. Compare experiments using identical fixed settings.

## Forward and update order

1. Encode the current search crop and language with the existing MVLM encoder.
2. Read only entries whose frame IDs precede the current frame. Each sequence/batch member owns a separate runtime bank.
3. Project current search tokens to queries, read appearance and language-conditioned memory streams, and gate the retrieved values into a residual adapter.
4. Pass enhanced search features to the unchanged CENTER head. Preserve original `feat_x` for VL correlation and memory writes; expose enhanced `feat_x_mem` separately.
5. Decode and select candidates with the existing MVLM confidence/search rules.
6. Extract a 2x2 ROI from original search features at the selected crop-relative box. Write only a valid, sufficiently confident observation. Online writes also require the selected candidate to pass the original MVLM confidence threshold.

For the current 224-pixel FastITPN-B configuration, features are `[B,196,512]`; internal adapter width is 256 and key width is 128. The adapter output projection is zero-initialized. Empty banks bypass the adapter exactly, even after training. A null attention slot permits rejection rather than forcing a match.

## Memory management

- Short-term capacity: 8 reliable observations. Long-term capacity: 16 representative observations. Both are configurable; either store can be disabled by setting its capacity to zero.
- Store detached original ROI tokens, frame ID, quality and box metadata. Recompute trainable keys/values on every read, avoiding stale keys as projection weights change.
- Short-term eviction is FIFO. Long-term admission checks appearance novelty; a similar observation can replace an existing one if quality improves. When full, remove the entry with low quality relative to appearance redundancy.
- Language-conditioned keys depend on both visual content and projected language; they are not copies of a fixed text vector.
- Retrieval adds log-quality and log-age biases. Entries retained only in the long-term store decay more slowly. ROI tokens have learned within-ROI position embeddings.
- Boxes supplied to ROIAlign are normalized crop-relative xywh. Stored box metadata is normalized full-image xywh in the training data path and online tracker. Training retains an explicit inverse annotation transform, accounting for the crop-center convention and joint flip. Box metadata does not enter attention in this first-stage implementation. Synthetic checks without image geometry may use unit-image coordinates.
- `initialize()` creates an empty bank. `set_language()` clears content and confidence memory and requests global re-localization. No first-frame ground-truth ROI is inserted into the new bank.
- Banks are runtime state, not model parameters or checkpoint buffers. DDP ranks and different videos do not share them.

## Training path

The memory configuration initializes from `models/MVLM_TF.pth.tar`, including the learned text projection. It skips the extra backbone pretraining file because the complete baseline tracking checkpoint supplies the backbone. Missing baseline weights are rejected. CLIP still uses the existing cached pretrained model.

The baseline encoder/text encoder are frozen and kept in eval mode. The CENTER head is also frozen/eval by default, but its forward remains differentiable with respect to adapter inputs. Only the new fusion/retrieval parameters are optimized initially. Set `TRAIN.MEMORY_FREEZE_HEAD: False` for a separately recorded joint-head adaptation experiment; provide an equal-training-budget baseline comparison.

`DATA.SAMPLER_MODE: memory` samples strictly increasing search frame IDs. It first attempts visible clips with frame stride 1--4; when none is found after bounded attempts, it uses consecutive entries from the visible-frame list, retaining their actual IDs. This initial sampler uses visible history and does not yet train on fully occluded frames.

The processing path applies consistent clip augmentation, keeps frame IDs/visibility and source crop regions/sizes, and returns crop-relative annotations. Crops are still generated from jittered ground truth, not recursively from online predicted states. Online-crop training and occluded-history sampling are follow-up work before claiming full train/inference alignment.

One model forward unrolls the clip, reads before writing, detaches past observations, and computes the existing tracking losses at each frame. The Actor averages the frame losses. Warm-up writes may use past ground-truth boxes; `TRAIN.MEMORY_PREDICTED_PROB` chooses predicted boxes per sample/frame (initially 0.5). Predicted writes are gated by score and visibility. Training does not yet reproduce the online per-candidate margin gate; this difference must be included when studying memory pollution. Ground truth never enters online memory writes.

With the encoder frozen, the original CM loss on raw encoder output cannot train the adapter, so it is disabled in the memory starting configuration. No auxiliary identity loss or prediction-consistency loss is implemented yet. Initial learning is through tracking losses. Zero initialization means the output projection learns first; deeper retrieval parameters start receiving gradients once the projection changes.

The provided 10-epoch, learning-rate 1e-4, per-GPU batch 2, four-frame, VastTrack-only configuration is a development starting point, not a validated or paper-reproduction training recipe. VastTrack must be downloaded, extracted and validated before training. No test split may be used for training. The official three-training-dataset recipe requires preparing the corresponding training splits separately.

## Checks and commands

Run from `/root/MVLM` with the existing environment:

```bash
/root/miniconda3/envs/mvlm/bin/python tracking/check_memory_module.py
/root/miniconda3/envs/mvlm/bin/python tracking/check_memory_module.py --full-model --device cuda:0
```

The lightweight check covers ROI validity, bank quality/capacity, causal reads, batch isolation, empty-bank behavior, zero-residual baseline equality, ordered sampling, joint augmentation, sequence losses, retrieval gradients, frozen weights/statistics and config isolation.

The optional full-model check loads existing official weights, runs a synthetic three-frame forward/backward, and checks a few real OTB frames for zero-initialization equality and global/local memory writes. It also checks target reset and rejection of missing memory weights. Synthetic backward is a smoke check, not training or an accuracy evaluation. It does not export predictions or overwrite existing experiment outputs.

After training-data validation, the training entry point is:

```bash
/root/miniconda3/envs/mvlm/bin/python tracking/train.py \
  --script mvlm --config mvlm_TF_memory \
  --save_dir /root/MVLM/output/memory_training --mode single
```

The existing trainer resumes checkpoints under the same save directory/config. Use a new directory/config for an independent run. Record the code commit, seed, checkpoint source, exact training split and all configuration changes. The batch size is per GPU; four-frame clips multiply image work and memory use.

Evaluate only a checkpoint containing trained `memory_fusion.*` weights:

```bash
/root/miniconda3/envs/mvlm/bin/python tracking/test.py mvlm mvlm_TF_memory \
  --dataset_name otb99_lang --weight_path /absolute/path/to/memory_checkpoint.pth.tar \
  --exp_id memory_otb_new_run --mode single --num_gpus 1
```

The tracker rejects missing memory parameters by default. `TEST.MEMORY_ALLOW_UNTRAINED: True` is an explicit diagnostic override and emits a warning; it must not be presented as a trained memory experiment. Keep this false for reported results.

## Files

- `lib/models/mvlm/memory.py`: ROI extraction and runtime banks.
- `lib/models/mvlm/memory_fusion.py`: trainable two-stream read and gated residual adapter.
- `lib/models/mvlm/mvlm.py`: model attachment, clip unrolling, frozen modes and checkpoint initialization.
- `lib/test/tracker/mvlm.py`: online read/write, checkpoint guard, reset and diagnostics.
- `lib/config/mvlm/config.py`, `experiments/mvlm/mvlm_TF_memory.yaml`: switches and starting parameters.
- `lib/train/data/sampler.py`, `lib/train/data/processing.py`: ordered clips and consistent transforms.
- `lib/train/actors/mvlm.py`, `lib/train/base_functions.py`, `lib/train/train_script.py`: losses, optimizer and training entry.
- `lib/test/parameter/mvlm.py`: separate tracker config snapshots so memory settings do not mutate earlier trackers.
- `tracking/check_memory_module.py`: standalone checks.

Reference: [XMem](https://arxiv.org/abs/2207.07115). The memory/fusion architecture here is a task-specific proposal; effectiveness requires controlled experiments.
