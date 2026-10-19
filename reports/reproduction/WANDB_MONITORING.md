# W&B training monitoring

Implemented on `codex/xmem-memory`; working checkout: `/root/MVLM/work/xmem-wandb`.
The original `/root/MVLM` checkout remains on `main` with its local changes intact.

The two memory YAML configurations enable W&B by default. Other configurations
retain disabled monitoring. Authentication uses the existing `wandb login`; no
API keys belong in YAML, source, or run metadata. Dependency: `wandb==0.22.3`.

## Recorded metrics

- Every 20 optimizer updates, plus the last batch: actual Actor statistics,
  including total/component losses, training IoU and `train/Memory/Gate`.
- Epoch averages under `epoch/train/*` (or the actual loader name).
- Learning rates, epoch, rank-zero batch duration and clips/second, allocated and
  peak CUDA memory. Runtime measurements are explicitly rank-zero measurements.
- Full resolved configuration, source commit/branch and parameter counts.

Scalar losses and IoU are averaged across distributed ranks, with only global
rank zero owning a W&B run. Epoch averages assume equally sized rank shards,
which the existing distributed sampler and `drop_last=True` provide.
`global_step` counts successful optimizer updates, excluding AMP overflow skips;
checkpoints now retain it. Legacy checkpoint steps are estimated from the saved
epoch and current loader length (not reliable after changing world size/batch).
No model gradient hooks, dataset images or checkpoints are uploaded by this code.

The current training pipeline has no full tracking-validation stage. It does not
compute or display validation PRE/NPR/AUC automatically. Training IoU is a
training statistic, not tracking AUC. Memory occupancy, retrieval statistics and
tracking videos are also not included in this first logging integration.

## Running

From this checkout, after training data/configuration have been prepared:

```bash
cd /root/MVLM/work/xmem-wandb
/root/miniconda3/envs/mvlm/bin/python tracking/train.py \
  --script mvlm --config mvlm_TF_memory \
  --save_dir /root/MVLM/output/memory_training_NEW --mode single
```

This starting configuration requires validated VastTrack training data. It is
not the proposed new three-dataset split; data preparation remains separate.
For the existing four-sequence engineering check, use
`--config mvlm_TF_memory_otb_smoke` and a different new save directory.
No training was launched as part of this integration.

`WANDB.PROJECT` defaults to `mvlm-memory`. `ENTITY` can be set to an account or
team; blank uses the logged-in account's default entity. `NAME` defaults to the
training configuration name. Environment overrides `WANDB_PROJECT`,
`WANDB_ENTITY`, `WANDB_NAME` and `WANDB_MODE` are supported. To disable monitoring,
set `WANDB.ENABLED: False`; for disconnected training use `WANDB_MODE=offline`.

At startup, the console prints `W&B monitor: <URL>`. A small metadata file stores
the run ID and URL at:

```text
<save_dir>/wandb_tracking/train/mvlm/<config>/run.json
```

Reusing a save directory resumes the model's latest saved epoch and, in online
mode, the saved W&B run ID. Work performed after the last saved epoch can be
replayed following a crash. Start independent experiments with new save
folders. Offline sessions are separate local sessions and need `wandb sync`
when connectivity returns; offline mode does not provide online run resume.

## Verification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/miniconda3/envs/mvlm/bin/python tracking/check_wandb_logging.py
```

This check uses temporary files and real W&B offline logging. It exercises the
trainer on a tiny synthetic CPU model (CUDA synchronization is mocked), saves
and reloads a checkpoint, continues the step count, checks disabled mode and
starts two Gloo ranks to verify scalar averaging and rank-zero run ownership.
It neither runs MVLM training nor contacts W&B. The integration passed this check
and Python compilation / `git diff --check` on 2026-10-18.
