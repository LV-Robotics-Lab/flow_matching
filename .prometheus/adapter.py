#!/usr/bin/env python3
"""Hardware-free Prometheus adapter for LV Robotics Flow Matching.

The adapter translates a versioned dataset contract into the native YAML
configuration and dispatches native Python modules as argv arrays. It does not
import the model stack while planning, invoke a shell, or authorize inference
on a robot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

import yaml


ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES_PATH = Path(__file__).with_name("capabilities.json")
DEFAULT_CONFIG = ROOT / "configs" / "train" / "config.yaml"
CAMERA_ORDER = ("base_0", "left_wrist_0", "right_wrist_0")
ACTION_CONTRACTS = {
    "abs_qpos": ("joint", "absolute", "joint", 14, "abs_qpos"),
    "relative_qpos": ("joint", "relative", "joint", 14, "abs_qpos"),
    "abs_eef_rot6d": ("eef", "absolute", "dual_eef_rot6d", 20, "abs_eef"),
    "relative_eef_rot6d": ("eef", "relative", "dual_eef_rot6d", 20, "abs_eef"),
}
REQUIRED_PATHS = (
    Path("pyproject.toml"),
    Path("configs/train/config.yaml"),
    Path("lv_flow_matching/train.py"),
    Path("lv_flow_matching/tools/precompute_policy_latents.py"),
    Path("lv_flow_matching/tools/eval_deploy_open_loop.py"),
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _feature_size(value: Any, label: str) -> int:
    feature = _mapping(value, label)
    shape = _list(feature.get("shape"), f"{label}.shape")
    if not shape or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in shape
    ):
        raise ValueError(f"{label}.shape must contain positive integers")
    size = 1
    for item in shape:
        size *= item
    return size


def capabilities() -> dict[str, Any]:
    payload = json.loads(CAPABILITIES_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "prometheus_source_adapter_v1":
        raise RuntimeError("unsupported Prometheus source-adapter schema")
    if payload.get("capabilities", {}).get("hardware_rollout_authorized") is not False:
        raise RuntimeError("training source must not authorize hardware rollout")
    return payload


def doctor() -> dict[str, Any]:
    declared = capabilities()
    missing = [path.as_posix() for path in REQUIRED_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"missing required Flow Matching paths: {missing}")
    if declared["dataset"]["legacy_embodiment_schema"] != "arx_bimanual_v1":
        raise RuntimeError("the fixed 14D/20D source contract must remain explicitly legacy")
    return {
        "ok": True,
        "policy_id": declared["policy_id"],
        "checked_paths": [path.as_posix() for path in REQUIRED_PATHS],
        "environment_reproducible": False,
        "environment_note": "pyproject.toml contains bounded specs, not a lockfile",
        "imports_model_stack": False,
        "hardware_touched": False,
    }


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(_mapping(payload, label))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError("Flow Matching requires an absolute local file:// dataset URI")
    path = Path(unquote(parsed.path)).expanduser()
    if not path.is_absolute():
        raise ValueError("dataset.uri must resolve to an absolute path")
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"dataset path does not exist: {resolved}")
    return resolved


def _camera_names(observation: Mapping[str, Any]) -> tuple[str, ...]:
    images = _list(observation.get("images"), "observation.images")
    names: list[str] = []
    for index, item in enumerate(images):
        feature = _mapping(item, f"observation.images[{index}]")
        name = _string(feature.get("name"), f"observation.images[{index}].name")
        names.append(name.rsplit(".", 1)[-1])
    selected = tuple(names)
    if selected != CAMERA_ORDER:
        raise ValueError(
            "arx_bimanual_v1 requires camera order "
            f"{list(CAMERA_ORDER)}, got {list(selected)}"
        )
    color_order = _string(observation.get("color_order"), "observation.color_order").upper()
    if color_order != "RGB":
        raise ValueError("Flow Matching training boundary requires RGB images")
    return selected


def validate_dataset_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "prometheus_training_dataset_v1":
        raise ValueError("unsupported training dataset schema")
    dataset = _mapping(payload.get("dataset"), "dataset")
    if _string(dataset.get("format"), "dataset.format").lower() != (
        "prometheus_flow_matching_zarr_v1"
    ):
        raise ValueError("Flow Matching accepts only prometheus_flow_matching_zarr_v1")
    dataset_root = _dataset_path(_string(dataset.get("uri"), "dataset.uri"))
    digest = _string(dataset.get("digest"), "dataset.digest").lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("dataset.digest must be a 64-character SHA-256 digest")

    robot = _mapping(payload.get("robot"), "robot")
    schema_sources = _list(robot.get("schema_sources"), "robot.schema_sources")
    if not schema_sources:
        raise ValueError("robot.schema_sources must be non-empty")
    robot_schema_digests: list[str] = []
    for index, item in enumerate(schema_sources):
        source = _mapping(item, f"robot.schema_sources[{index}]")
        source_digest = _string(
            source.get("digest"), f"robot.schema_sources[{index}].digest"
        ).lower()
        if len(source_digest) != 64 or any(
            char not in "0123456789abcdef" for char in source_digest
        ):
            raise ValueError(
                f"robot.schema_sources[{index}].digest must be a SHA-256 digest"
            )
        robot_schema_digests.append(source_digest)

    observation = _mapping(payload.get("observation"), "observation")
    camera_views = _camera_names(observation)
    state = _list(observation.get("state"), "observation.state")
    state_dim = sum(
        _feature_size(item, f"observation.state[{index}]")
        for index, item in enumerate(state)
    )
    tactile = _list(observation.get("tactile"), "observation.tactile")

    action = _mapping(payload.get("action"), "action")
    action_space = _string(action.get("space"), "action.space").lower()
    if action_space not in ACTION_CONTRACTS:
        raise ValueError(
            f"unsupported action.space={action_space!r}; "
            f"expected one of {sorted(ACTION_CONTRACTS)}"
        )
    action_type, representation, expected_frame, expected_dim, deploy_process = (
        ACTION_CONTRACTS[action_space]
    )
    frame = _string(action.get("frame"), "action.frame").lower()
    if frame != expected_frame:
        raise ValueError(
            f"action.frame={frame!r} does not match {action_space!r}; "
            f"expected {expected_frame!r}"
        )
    action_dim = _positive_int(action.get("dim"), "action.dim")
    features = _list(action.get("features"), "action.features")
    measured_dim = sum(
        _feature_size(item, f"action.features[{index}]")
        for index, item in enumerate(features)
    )
    if action_dim != measured_dim or action_dim != expected_dim:
        raise ValueError(
            f"{action_space} requires {expected_dim} values; "
            f"action.dim={action_dim}, features={measured_dim}"
        )
    if state_dim != expected_dim:
        raise ValueError(
            f"{action_space} requires a matching {expected_dim}D state; got {state_dim}"
        )

    language = _mapping(payload.get("language"), "language")
    if _string(language.get("mode"), "language.mode").lower() != "none":
        raise ValueError("this Flow Matching source does not consume language")
    normalization = _mapping(payload.get("normalization"), "normalization")
    if _string(normalization.get("owner"), "normalization.owner").lower() != "trainer":
        raise ValueError("Flow Matching owns and checkpoints its fitted normalizer")
    if _string(normalization.get("method"), "normalization.method").lower() != "none":
        raise ValueError(
            "the dataset must be raw at this boundary; native trainer normalization is internal"
        )

    sampling = _mapping(payload.get("sampling"), "sampling")
    rate_hz = sampling.get("rate_hz")
    if not isinstance(rate_hz, (int, float)) or isinstance(rate_hz, bool) or rate_hz <= 0:
        raise ValueError("sampling.rate_hz must be positive")
    horizon = _positive_int(action.get("horizon"), "action.horizon")
    return {
        "dataset_root": dataset_root,
        "dataset_digest": digest,
        "robot_schema_digests": robot_schema_digests,
        "action_type": action_type,
        "action_representation": representation,
        "action_space": action_space,
        "action_dim": expected_dim,
        "action_horizon": horizon,
        "deploy_process": deploy_process,
        "camera_views": camera_views,
        "use_tactile": bool(tactile),
        "rate_hz": float(rate_hz),
    }


def _ensure_external_run_dir(run_dir: Path) -> Path:
    resolved = run_dir.expanduser().resolve()
    if resolved == ROOT or resolved.is_relative_to(ROOT):
        raise ValueError("run directory must be outside the immutable source checkout")
    return resolved


def build_resolved_config(
    base_config: Path,
    dataset_contract: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], Path]:
    run_dir = _ensure_external_run_dir(run_dir)
    base = _load_yaml(base_config.expanduser().resolve(), "base config")
    contract_path = dataset_contract.expanduser().resolve()
    contract = _load_yaml(contract_path, "dataset contract")
    selected = validate_dataset_contract(contract)
    cfg = deepcopy(base)
    cfg.setdefault("data", {})
    cfg.setdefault("models", {}).setdefault("fm", {})
    cfg.setdefault("output", {})
    cfg.setdefault("precompute", {})
    cfg.setdefault("deploy", {})

    cfg["data"].update(
        {
            "root_dir": str(selected["dataset_root"]),
            "action_type": selected["action_type"],
            "action_representation": selected["action_representation"],
            "action_horizon": selected["action_horizon"],
            "camera_views": list(selected["camera_views"]),
            "use_tactile": selected["use_tactile"],
            "latent_cache_root_dir": str(run_dir / "latent_cache"),
        }
    )
    cfg["models"]["fm"].update(
        {
            "n_image_views": len(selected["camera_views"]),
            "use_tactile": selected["use_tactile"],
        }
    )
    cfg["output"].update({"root_dir": str(run_dir.parent), "run_name": run_dir.name})
    cfg["precompute"]["output_path"] = str(run_dir / "latent_cache" / "frame_backbone.zarr")
    cfg["deploy"].update(
        {"action_process": selected["deploy_process"], "action_hz": selected["rate_hz"]}
    )
    cfg["prometheus_contract"] = {
        "schema": "prometheus_flow_matching_resolved_v1",
        "dataset_contract": str(contract_path),
        "dataset_contract_digest": _sha256_file(contract_path),
        "dataset_digest": selected["dataset_digest"],
        "robot_schema_digests": selected["robot_schema_digests"],
        "legacy_embodiment_schema": "arx_bimanual_v1",
        "action_space": selected["action_space"],
        "action_dim": selected["action_dim"],
        "hardware_rollout_authorized": False,
    }
    return cfg, run_dir / "prometheus_train_config.yaml"


def validate_run_contract(run_dir: Path, dataset_contract: Path) -> dict[str, Any]:
    """Bind resume/eval to the exact dataset and robot schema used by the run."""

    run_dir = _ensure_external_run_dir(run_dir)
    contract_path = dataset_contract.expanduser().resolve()
    selected = validate_dataset_contract(
        _load_yaml(contract_path, "dataset contract")
    )
    resolved_path = run_dir / "resolved_config.yaml"
    if not resolved_path.is_file():
        raise FileNotFoundError(f"missing native resolved config: {resolved_path}")
    resolved = _load_yaml(resolved_path, "native resolved config")
    recorded = _mapping(resolved.get("prometheus_contract"), "prometheus_contract")
    expected = {
        "dataset_contract_digest": _sha256_file(contract_path),
        "dataset_digest": selected["dataset_digest"],
        "robot_schema_digests": selected["robot_schema_digests"],
        "action_space": selected["action_space"],
        "action_dim": selected["action_dim"],
        "legacy_embodiment_schema": "arx_bimanual_v1",
    }
    mismatches = {
        key: {"recorded": recorded.get(key), "requested": value}
        for key, value in expected.items()
        if recorded.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "dataset contract does not match the existing training run: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if recorded.get("hardware_rollout_authorized") is not False:
        raise ValueError("existing run does not retain the hardware rollout prohibition")
    return selected


def write_resolved_config(config: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _gpu_env(gpus: str) -> tuple[dict[str, str], int]:
    selected = [item.strip() for item in gpus.split(",") if item.strip()]
    if not selected:
        raise ValueError("--gpus must contain at least one CUDA device id")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    environment.setdefault("HF_HUB_OFFLINE", "1")
    return environment, len(selected)


def build_native_argv(
    stage: str,
    *,
    config_path: Path | None,
    run_dir: Path,
    gpus: str,
    checkpoint: Path | None,
    native_args: Sequence[str],
) -> tuple[list[str], dict[str, str]]:
    environment, gpu_count = _gpu_env(gpus)
    extra = list(native_args)
    if extra[:1] == ["--"]:
        extra.pop(0)
    if stage in {"prepare", "train", "resume"} and config_path is None:
        raise ValueError(f"{stage} requires a generated config")
    if stage == "prepare":
        return [
            sys.executable,
            "-m",
            "lv_flow_matching.tools.precompute_policy_latents",
            "--config",
            str(config_path),
            *extra,
        ], environment
    if stage in {"train", "resume"}:
        module_args = ["-m", "lv_flow_matching.train", "--config", str(config_path)]
        if stage == "resume":
            if checkpoint is None:
                raise ValueError("full-state resume requires --checkpoint")
            resolved_checkpoint = checkpoint.expanduser().resolve()
            if not resolved_checkpoint.is_file():
                raise FileNotFoundError(f"resume checkpoint does not exist: {resolved_checkpoint}")
            module_args.extend(
                ["--resume", str(resolved_checkpoint), "--resume-mode", "full"]
            )
        module_args.extend(extra)
        if gpu_count > 1:
            port = environment.get("MASTER_PORT", "29500")
            return [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc_per_node={gpu_count}",
                f"--master_port={port}",
                *module_args,
            ], environment
        return [sys.executable, *module_args], environment
    if stage == "eval":
        command = [
            sys.executable,
            "-m",
            "lv_flow_matching.tools.eval_deploy_open_loop",
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(run_dir / "offline_eval"),
        ]
        if checkpoint is not None:
            command.extend(["--checkpoint", str(checkpoint.expanduser().resolve())])
        command.extend(extra)
        return command, environment
    raise ValueError(f"unsupported executable stage: {stage}")


def _source_revision() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def _write_artifact_manifest(run_dir: Path, config_path: Path) -> None:
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"native training completed without expected checkpoint: {checkpoint}")
    cfg = _load_yaml(config_path, "resolved training config")
    contract = _mapping(cfg.get("prometheus_contract"), "prometheus_contract")
    parent_adapter_digest = os.environ.get("PROMETHEUS_POLICY_ADAPTER_DIGEST", "").lower()
    if len(parent_adapter_digest) != 64 or any(
        char not in "0123456789abcdef" for char in parent_adapter_digest
    ):
        raise RuntimeError(
            "PROMETHEUS_POLICY_ADAPTER_DIGEST is required to emit a promotable artifact"
        )
    payload = {
        "schema": "prometheus_policy_artifact_v1",
        "policy_type": "lv_flow_matching",
        "action_space": contract["action_space"],
        "action_dim": contract["action_dim"],
        "adapter_digest": parent_adapter_digest,
        "checkpoint": str(checkpoint),
        "checkpoint_digest": _sha256_file(checkpoint),
        "dataset_digest": contract["dataset_digest"],
        "resolved_config_digest": _sha256_file(config_path),
        "robot_schema_digests": contract["robot_schema_digests"],
        "source_revision": _source_revision(),
        "source_adapter_digest": _sha256_file(Path(__file__).resolve()),
        "normalization_owner": "trainer",
        "legacy_embodiment_schema": contract["legacy_embodiment_schema"],
        "hardware_rollout_authorized": False,
    }
    path = run_dir / "prometheus_artifact.json"
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _stage_parser(subparsers: argparse._SubParsersAction, name: str) -> None:
    parser = subparsers.add_parser(name)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("native_args", nargs=argparse.REMAINDER)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("capabilities")
    subparsers.add_parser("doctor")
    for stage in ("prepare", "train", "resume", "eval"):
        _stage_parser(subparsers, stage)
    args = parser.parse_args(argv)

    if args.command == "capabilities":
        _print_json(capabilities())
        return 0
    if args.command == "doctor":
        _print_json(doctor())
        return 0

    run_dir = _ensure_external_run_dir(args.run_dir)
    config_path: Path | None = None
    if args.command in {"prepare", "train", "resume"}:
        if args.command == "train" and (
            (run_dir / "resolved_config.yaml").exists()
            or (run_dir / "checkpoints").exists()
        ):
            raise FileExistsError(
                "train refuses an existing native run; use the explicit resume stage"
            )
        if args.command == "resume":
            validate_run_contract(run_dir, args.dataset_contract)
        base_config = args.base_config
        if base_config is None:
            base_config = (
                run_dir / "resolved_config.yaml"
                if args.command == "resume"
                else DEFAULT_CONFIG
            )
        config, config_path = build_resolved_config(
            base_config, args.dataset_contract, run_dir
        )
    else:
        validate_run_contract(run_dir, args.dataset_contract)

    command, environment = build_native_argv(
        args.command,
        config_path=config_path,
        run_dir=run_dir,
        gpus=args.gpus,
        checkpoint=args.checkpoint,
        native_args=args.native_args,
    )
    if args.plan:
        _print_json(
            {
                "argv": command,
                "config_path": None if config_path is None else str(config_path),
                "hardware_rollout_authorized": False,
                "shell": False,
            }
        )
        return 0

    if config_path is not None:
        write_resolved_config(config, config_path)
    process = subprocess.run(command, cwd=run_dir, env=environment, check=False)
    if process.returncode != 0:
        return process.returncode
    if args.command in {"train", "resume"}:
        assert config_path is not None
        _write_artifact_manifest(run_dir, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
