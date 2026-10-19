# Three-dataset custom memory training — 2026-10-18

Authorized by the user, launched in the background on eight RTX 3090 GPUs.
This is a custom-split adaptation experiment using previously evaluated data,
not an independent reproduction of the original benchmark protocol.

Monitor: https://wandb.ai/2896287267-hit/mvlm-memory/runs/nie0ya10

Working checkout: `/root/MVLM/work/xmem-wandb`, branch `codex/xmem-memory`.
Output: `/root/MVLM/output/memory_three_custom_20261018_run01`.
Code is based on `8bf42d2` plus uncommitted changes. Source archive, per-file
SHA256 hashes and tracked diff are saved with the run before launch.

## Frozen data

| Dataset | Train sequences / frames | Validation sequences / frames | Test sequences / frames |
| --- | --- | --- | --- |
| OTB99-Lang | 51 / 29,491 | 12 / 8,593 | 36 / 20,484 |
| LaSOT | 196 / 474,828 | 42 / 100,476 | 42 / 110,056 |
| TNL2K | 489 / 390,486 | 105 / 66,287 | 105 / 58,935 |

The TNL2K `Fight_video_6-Done` sequence is excluded because its language file is
empty. Original dataset files are not changed. Split seed: 20261018.
`split/manifest.json` SHA256:
`593170c87cefcc4e9d4d089b7d7c85d28dae88ff03ba5c8026e6fd2139b89289`.

OTB's original training queries define its training side. Remaining sequences
are assigned as whole source groups. TNL2K naming families and identical sampled
image hashes are grouped conservatively. Exact image paths do not overlap across
splits. All indexed image paths are present/nonempty; first/middle/last images
are decoded. This does not establish absence of all duplicate or near-duplicate
footage, and it is not full image decoding. No claim of strict class or length
stratification is made. LaSOT training covers 68 of its 70 categories.
Official-weight pretraining provenance still limits interpretation of custom
held-out splits. TNL2K archive SHA256 remains unconfirmed as recorded in handoff.

## Training schedule

- Initialize from official `models/MVLM_TF.pth.tar`; do not initialize from pilots.
- Freeze baseline encoder, text encoder and CENTER head; train memory parameters.
- Eight GPUs × batch 2 clips/GPU = global batch 16, full precision.
- 16,000 sampled clips/epoch = 1,000 optimizer updates/epoch; 20 epochs total.
- Epochs 1–5: four frames/clip, learning rate 1e-4.
- Epochs 6–15: twelve frames/clip, learning rate 5e-5.
- Epochs 16–20: twelve frames/clip, learning rate 1e-5.
- AdamW, weight decay 1e-4, gradient norm clipping 1.0.
- Training dataset sampling ratio OTB:LaSOT:TNL2K = 1:3:6.
- Fixed inference parameters: alpha=.5, tau=.03, omega=.1, lambda=.4, psi=.1.
- Save a checkpoint each epoch. Validate every two epochs / 2,000 updates.

The full 159-sequence validation set is evaluated using repository first-frame
handling, integer prediction files and repository center/overlap definitions.
PRE/NPR/AUC are sequence averages; best selection uses equal mean AUC across
three datasets. The frozen official baseline with identical inference settings
is evaluated and cached at the first validation. No test-set evaluation occurs
during training. The W&B step axis counts optimizer updates rather than frames.
Validation can pause training updates while keeping the W&B run active.

Best selection: `validation/best.json`, `validation/best.pth.tar`.
Epoch predictions and metrics: `validation/epoch_XXXX/`.

## Verification and monitoring

Before launch, 200 four-frame single-GPU updates and 20 twelve-frame updates
completed on real custom training data. Both checkpoints had unchanged baseline
weights and nonzero learned memory output projection. A separate eight-GPU pilot
completed four synchronized updates and saved a checkpoint. Three short
validation sequences (one per dataset) passed online tracking and metric checks;
these engineering-subset metrics are not reported as benchmark results.
Offline W&B logging, disabled mode, checkpoint step resume and two-rank metric
averaging checks passed. The learning-rate schedule was checked at all 20 epochs.

Online W&B logs include scalar losses, training IoU, gates, short/long memory
occupancy, accepted-write fraction, clip length, learning rates and runtime/GPU
memory. Full validation and cached-baseline metrics are uploaded at validation.
Only rank zero uploads; training scalars are averaged across ranks.

Detached supervisor PID at launch: 1067381; torchrun PID: 1067382.
Inspect current state instead of assuming these PIDs or progress remain valid:

```bash
cat /root/MVLM/output/memory_three_custom_20261018_run01/training_status.json
tail -n 20 /root/MVLM/output/memory_three_custom_20261018_run01/train.log
```

`training_status.json` reports running/completed/failed. Do not launch a second
copy. The supervisor uses a lock. A deliberate restart uses the same command
and directory to load the latest epoch checkpoint and resume the W&B run; work
since the latest saved epoch can be replayed. Three-dataset data preparation and
formal training changes have not been committed or pushed.
