# Prometheus source adapter

This directory is the hardware-free boundary between the pinned Flow Matching
training source and PrometheusV4's `prometheus_policy_adapter_v1` runner.

The adapter consumes an explicit `prometheus_training_dataset_v1` YAML file
and writes the resolved native training config into the external run
directory. It never writes datasets, caches, checkpoints, or generated config
into this source checkout and never authorizes hardware rollout.

The required contract field `robot.embodiment_schema` selects one exact native
layout:

- `arx_bimanual_v1`: legacy 34D native arrays, sliced into dual-arm joint 14D
  or EEF 20D (`xyz + rot6d + gripper` per arm);
- `cobot_magic_v1`: native 14D joint arrays;
- `franka_wuji_v1`: native 54D joint arrays.

All require RGB views in this order: `base_0`, `left_wrist_0`,
`right_wrist_0`. A contract that omits `robot.embodiment_schema` is rejected —
`arx_bimanual_v1` is the name of the historical layout, not a fallback for an
unlabelled dataset — and no embodiment is ever inferred from robot id or action
dimension. New embodiments require an explicit named schema and contract tests.

`resume=full_state_non_bit_exact` means the native checkpoint restores model,
normalizer, optimizer, scheduler, AMP scaler when present, epoch, and global
step. It does not currently persist every RNG and dataloader state, so the
adapter does not claim bit-exact continuation.

Run a dependency-free structural probe:

```bash
python .prometheus/adapter.py doctor
```

Print a native command without executing it:

```bash
python .prometheus/adapter.py train \
  --dataset-contract /managed/contracts/dataset.yaml \
  --run-dir /managed/runs/fm-001 \
  --plan
```
