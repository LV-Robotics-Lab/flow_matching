# Embodiment training profiles

Checked-in embodiment profiles are hardware-free templates. They define the
policy-side data and action schema but never import a robot SDK, start a
driver, or authorize rollout. Dataset and output roots are explicit
`/absolute/path/to/...` placeholders so a formal run cannot silently write
inside this source checkout.

Copy a profile outside the repository, replace the dataset/output root, retain
its `profile.provenance`, and validate the corresponding Prometheus training
dataset and robot-schema digests before running it.

## Profiles

| Profile | Schema | Provenance |
| --- | --- | --- |
| `configs/train/embodiments/cobot_magic_v1.yaml` | 3 ordered RGB views, 8-frame window, 32-step 14D absolute joint action at 25 Hz | PrometheusV4 `hardware/cobot-daimon@b5e1552`, source SHA-256 embedded in config |
| `configs/train/embodiments/cobot_magic_v1_smoke.yaml` | Bounded two-batch variant of `cobot_magic_v1` | Same Cobot revision, source SHA-256 embedded in config |
| `configs/train/embodiments/cobot_magic_v1_joint30_legacy.yaml` | Legacy 30 Hz key layout retained for migration reference | Same Cobot revision; `status: legacy_reference` |
| `configs/train/embodiments/franka_wuji_v1_smoke.yaml` | 3 ordered RGB views, 8-frame window, 16-step 54D absolute joint action at 30 Hz | PrometheusV4 `hardware/franka-wuji@a9d4292`, source SHA-256 embedded in config |

The 54D profile uses the generic `data.action_dim` contract. Custom dimensions
are accepted only for `action_type: joint`; EEF remains the fixed 20D native
representation. Deployment of an absolute joint checkpoint reports the native
dimension rather than assuming 14D.

## Fail-closed launcher

Use the generic embodiment launcher with an externalized config:

```bash
cp configs/train/embodiments/cobot_magic_v1.yaml /absolute/path/to/run-config.yaml
# Edit paths.data_root while preserving profile.provenance.

PYTHON=/absolute/path/to/python \
  ./scripts/train_embodiment.sh \
  --config /absolute/path/to/run-config.yaml \
  --resume none \
  --resume-mode full \
  --gpus 0
```

The launcher requires an explicit `--resume` decision, rejects `auto`, rejects
an unchanged placeholder or repository-local output, and refuses a nonempty or
symlinked fresh-run directory. `weights-only` is an explicit non-lossless mode.
Resume paths must name an existing checkpoint.

## Cobot historical continuation

The Cobot canonical and smoke profiles are fresh-run recipes. They must not be
used to full-resume the historical epoch-161 checkpoint because that run used
FP32, optimizer learning rate `5.0e-5`, cosine scheduling,
`open_loop_test_every: 10`, `plot_samples: 8`, and `resume_mode: full`.

Create an external continuation config with a distinct run name such as
`cobot_magic_v1_resume161`, bind it to the immutable Prometheus run-evidence
manifests from `hardware/cobot-daimon@b5e1552`, and pass the exact checkpoint to
`train_embodiment.sh`. The legacy RTX 5090 lock is preserved under
`environments/legacy/` for provenance, with its current-source compatibility
warning documented beside it.

These profiles and launcher are training contracts only. They do not prove
dataset quality, checkpoint quality, inference compatibility, or hardware
readiness.
