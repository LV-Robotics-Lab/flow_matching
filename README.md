# LV Robotics Flow Matching

This repository owns Flow Matching policy training for the PrometheusV4
ecosystem. It also exposes the inference runtime used by Prometheus adapters.
Prometheus hardware branches own collection and robot-side inference;
`policy/flow_matching` owns the training integration and pins this repository by
an exact submodule commit.

The Python import is `lv_flow_matching`. The repository keeps the historical
name `flow_matching`, but does not use the bare `flow_matching` import because
that namespace is already used by another public Python project.

## Install

The inference package supports Python 3.10. Training requires Python 3.11
because its Zarr 3 dependency no longer supports Python 3.10. Always use an
isolated training environment; do not combine it with a Prometheus
preprocessing environment that pins Zarr 2.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,dev]'
```

When checked out through PrometheusV4:

```bash
git submodule update --init third_party/flow_matching
python -m pip install -e 'third_party/flow_matching[train,dev]'
```

## Train

The checked-in YAML files are reproducible recipes; update their dataset and
output paths for the target host. Keep local data, pretrained DINO weights,
latent caches, logs, and checkpoints outside Git.

```bash
./scripts/precompute.sh --config /absolute/path/train.yaml --gpus 0
./scripts/train.sh --config /absolute/path/train.yaml --gpus 0

# Multi-GPU DDP
./scripts/train.sh --config /absolute/path/train.yaml --gpus 0,1,2,3

# Strict continuation, including optimizer/scheduler/AMP state
./scripts/train.sh --config /absolute/path/train.yaml \
  --resume /absolute/path/latest.pt --resume-mode full
```

Finetuning runs precompute first unless `--skip-precompute` is passed:

```bash
./scripts/finetune.sh --config /absolute/path/finetune.yaml --gpus 0
```

`HF_HUB_OFFLINE=1` remains the launcher default. A fresh machine must provide
the configured DINO weights in `pretrained_weights/` or an existing Hugging
Face cache before training.

## Validate

```bash
python -m compileall -q lv_flow_matching tests
python -m pytest -q tests --ignore=tests/fm/test_forward.py
```

The larger forward test is kept for an explicit CPU/GPU model smoke. See
[README_INTEGRATION.md](README_INTEGRATION.md) for source provenance and
[README_DINO_MEMORY.md](README_DINO_MEMORY.md) for Memory-specific behavior.
