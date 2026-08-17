#!/usr/bin/env bash
set -euo pipefail

CALLER_DIR="$(pwd -P)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOW_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG_SPEC=""
RESUME_SPEC=""
RESUME_MODE="full"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  cat >&2 <<'EOF'
Usage:
  ./scripts/train_embodiment.sh --config PATH --resume {none|CHECKPOINT} [options]

Options:
  --config PATH       Externalized embodiment config or checked-in template
  --resume VALUE      Exact checkpoint path, or none for a fresh run
  --resume-mode MODE  full or weights-only (default: full)
  --gpus IDS          CUDA_VISIBLE_DEVICES (default: 0)

The resolved output root must be absolute, outside this source checkout, and
must not contain the checked-in /absolute/path/to placeholder.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      CONFIG_SPEC="$2"
      shift 2
      ;;
    --resume)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      RESUME_SPEC="$2"
      shift 2
      ;;
    --resume-mode)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      RESUME_MODE="$2"
      shift 2
      ;;
    --gpus)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      GPUS="$2"
      shift 2
      ;;
    --)
      echo "argument passthrough is disabled" >&2
      exit 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown launcher argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${CONFIG_SPEC}" ]]; then
  echo "--config is required" >&2
  exit 2
fi
if [[ -z "${RESUME_SPEC}" ]]; then
  echo "--resume is required; pass 'none' explicitly for a fresh run" >&2
  exit 2
fi
case "${RESUME_SPEC}" in
  [Aa][Uu][Tt][Oo])
    echo "--resume auto is forbidden; pass an exact checkpoint or 'none'" >&2
    exit 2
    ;;
esac
if [[ "${RESUME_MODE}" != "full" && "${RESUME_MODE}" != "weights-only" ]]; then
  echo "invalid --resume-mode: ${RESUME_MODE}; expected full or weights-only" >&2
  exit 2
fi

if [[ "${CONFIG_SPEC}" = /* ]]; then
  CONFIG_PATH="${CONFIG_SPEC}"
else
  CONFIG_PATH="${FLOW_ROOT}/${CONFIG_SPEC}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "training config does not exist: ${CONFIG_PATH}" >&2
  exit 3
fi

if ! PYTHON_PATH="$(command -v "${PYTHON_BIN}")"; then
  echo "Flow Matching Python is not executable: ${PYTHON_BIN}" >&2
  exit 4
fi
PYTHON_BIN="${PYTHON_PATH}"

if ! RUN_DIR="$(${PYTHON_BIN} - "${CONFIG_PATH}" "${FLOW_ROOT}" <<'PY'
import sys
from pathlib import Path

from lv_flow_matching.utils.train_utils import load_config

config_path = Path(sys.argv[1]).resolve()
source_root = Path(sys.argv[2]).resolve()
cfg = load_config(str(config_path))
output = cfg.get("output")
if not isinstance(output, dict):
    raise SystemExit("config requires an output mapping")
root_dir = Path(str(output.get("root_dir") or "")).expanduser()
run_name = str(output.get("run_name") or "").strip()
if not root_dir.is_absolute():
    raise SystemExit("output.root_dir must be absolute and outside the source checkout")
if "/absolute/path/to/" in str(root_dir):
    raise SystemExit("replace the checked-in /absolute/path/to output placeholder")
if not run_name:
    raise SystemExit("output.run_name is required")
run_dir = root_dir / run_name
resolved_run_dir = run_dir.resolve()
try:
    resolved_run_dir.relative_to(source_root)
except ValueError:
    pass
else:
    raise SystemExit("output run directory must be outside the source checkout")
print(run_dir)
PY
)"; then
  echo "unable to resolve an external collision-safe run directory" >&2
  exit 6
fi

case "${RESUME_SPEC}" in
  [Nn][Oo][Nn][Ee]) RESUME_IS_NONE=1 ;;
  *) RESUME_IS_NONE=0 ;;
esac

if [[ "${RESUME_IS_NONE}" -eq 1 ]]; then
  if [[ -L "${RUN_DIR}" ]]; then
    echo "fresh run directory must not be a symbolic link: ${RUN_DIR}" >&2
    exit 6
  fi
  if [[ -e "${RUN_DIR}" ]] && {
    [[ ! -d "${RUN_DIR}" ]] ||
      find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .
  }; then
    echo "fresh run directory is not empty: ${RUN_DIR}" >&2
    exit 6
  fi
  RESUME_VALUE="none"
else
  if [[ "${RESUME_SPEC}" = /* ]]; then
    RESUME_VALUE="${RESUME_SPEC}"
  else
    RESUME_VALUE="${CALLER_DIR}/${RESUME_SPEC}"
  fi
  if [[ ! -f "${RESUME_VALUE}" ]]; then
    echo "resume checkpoint does not exist: ${RESUME_VALUE}" >&2
    exit 5
  fi
fi

exec "${SCRIPT_DIR}/train.sh" \
  --config "${CONFIG_PATH}" \
  --gpus "${GPUS}" \
  --resume "${RESUME_VALUE}" \
  --resume-mode "${RESUME_MODE}"
