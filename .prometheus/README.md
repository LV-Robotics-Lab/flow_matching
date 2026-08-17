# Prometheus source adapter

This directory is the hardware-free boundary between the pinned Flow Matching
training source and PrometheusV4's `prometheus_policy_adapter_v1` runner.

The adapter consumes an explicit `prometheus_training_dataset_v1` YAML file,
rejects datasets outside the source's current legacy ARX-bimanual contract,
and writes the resolved native training config into the external run
directory. It never writes datasets, caches, checkpoints, or generated config
into this source checkout and never authorizes hardware rollout.

The current source supports exactly two action layouts:

- dual-arm joint, 14 values;
- dual-arm EEF, 20 values (`xyz + rot6d + gripper` per arm).

Both require RGB views in this order: `base_0`, `left_wrist_0`,
`right_wrist_0`. This is deliberately named `arx_bimanual_v1`; it is not a
generic robot contract. A future embodiment must first make the native dataset
and model dimensions configurable and add its own contract tests.

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
