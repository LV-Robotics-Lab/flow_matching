#!/usr/bin/env bash
set -euo pipefail

CALLER_DIR="$(pwd -P)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOW_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$FLOW_ROOT"

CONFIG="${FLOW_ROOT}/configs/train/config.yaml"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON:-python}"
RESUME=""
RESUME_SET=0
RESUME_MODE=""
RESUME_MODE_SET=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/train.sh [options]

Options:
  --config PATH   Config yaml (default: configs/train/config.yaml)
  --gpus IDS      CUDA_VISIBLE_DEVICES (default: 0)
  --resume VALUE  Checkpoint path, auto, or none (default: config value)
  --resume-mode MODE
                  full or weights-only (default: config value, then full)
  -h, --help      Show this help

Examples:
  ./scripts/train.sh
  ./scripts/train.sh --gpus 0
  PYTHON=/path/to/env/bin/python ./scripts/train.sh --gpus 0,1,2,3,4,5,6,7
  ./scripts/train.sh --config configs/train/smoke_mem.yaml

Edit training hyperparameters in the config yaml (data, train, models, output, checkpoint).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --config=*)
      CONFIG="${1#*=}"
      shift
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --gpus=*)
      GPUS="${1#*=}"
      shift
      ;;
    --resume)
      RESUME="$2"
      RESUME_SET=1
      shift 2
      ;;
    --resume=*)
      RESUME="${1#*=}"
      RESUME_SET=1
      shift
      ;;
    --resume-mode)
      RESUME_MODE="$2"
      RESUME_MODE_SET=1
      shift 2
      ;;
    --resume-mode=*)
      RESUME_MODE="${1#*=}"
      RESUME_MODE_SET=1
      shift
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPUS"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
if [[ -z "${HF_HOME:-}" && -e "$FLOW_ROOT/.hf_cache" ]]; then
  export HF_HOME="$FLOW_ROOT/.hf_cache"
fi

if [[ "$CONFIG" != /* ]]; then
  CONFIG="${FLOW_ROOT}/${CONFIG}"
fi
if [[ "$RESUME_SET" -eq 1 && "$RESUME" != /* ]]; then
  case "$RESUME" in
    ""|[Aa][Uu][Tt][Oo]|[Nn][Oo][Nn][Ee]) ;;
    *) RESUME="${CALLER_DIR}/${RESUME}" ;;
  esac
fi

# Always launch through PYTHON_BIN so DDP workers use the same environment.
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU="${#GPU_ARR[@]}"
TRAIN_ARGS=(--config "$CONFIG")
if [[ "$RESUME_SET" -eq 1 ]]; then
  TRAIN_ARGS+=(--resume "$RESUME")
fi
if [[ "$RESUME_MODE_SET" -eq 1 ]]; then
  TRAIN_ARGS+=(--resume-mode "$RESUME_MODE")
fi

if [[ "$NGPU" -gt 1 ]]; then
  MASTER_PORT="${MASTER_PORT:-29500}"
  exec "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    -m lv_flow_matching.train "${TRAIN_ARGS[@]}"
else
  exec "$PYTHON_BIN" -m lv_flow_matching.train "${TRAIN_ARGS[@]}"
fi
