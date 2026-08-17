from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = Path(__file__).with_name("adapter.py")
SPEC = importlib.util.spec_from_file_location(
    "flow_matching_prometheus_adapter", ADAPTER_PATH
)
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def _contract(
    dataset_root: Path,
    *,
    action_space: str = "abs_qpos",
    embodiment_schema: str | None = adapter.LEGACY_EMBODIMENT_SCHEMA,
) -> dict:
    if "eef" in action_space:
        dim = 20
        frame = "dual_eef_rot6d"
    elif embodiment_schema == "franka_wuji_v1":
        dim = 54
        frame = "joint"
    else:
        dim = 14
        frame = "joint"
    payload = {
        "schema": "prometheus_training_dataset_v1",
        "dataset": {
            "id": "fixture",
            "format": "prometheus_flow_matching_zarr_v1",
            "uri": dataset_root.as_uri(),
            "digest": "1" * 64,
        },
        "robot": {
            "id": "fixture_robot",
            "schema_sources": [
                {"name": "fixture", "uri": "file:///fixture", "digest": "2" * 64}
            ],
        },
        "sampling": {"rate_hz": 30.0},
        "observation": {
            "color_order": "RGB",
            "state": [
                {
                    "name": "observation.state",
                    "shape": [dim],
                    "dtype": "float32",
                    "unit": "rad",
                }
            ],
            "images": [
                {
                    "name": f"observation.images.{name}",
                    "shape": [8, 8, 3],
                    "dtype": "uint8",
                }
                for name in adapter.CAMERA_ORDER
            ],
            "tactile": [],
        },
        "action": {
            "space": action_space,
            "frame": frame,
            "dim": dim,
            "horizon": 32,
            "features": [
                {"name": "action", "shape": [dim], "dtype": "float32", "unit": "rad"}
            ],
        },
        "language": {"mode": "none"},
        "normalization": {"method": "none", "owner": "trainer"},
    }
    if embodiment_schema is not None:
        payload["robot"]["embodiment_schema"] = embodiment_schema
    return payload


def _write_contract(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_doctor_is_hardware_free() -> None:
    result = adapter.doctor()
    assert result["ok"] is True
    assert result["hardware_touched"] is False
    assert result["imports_model_stack"] is False
    assert result["environment_reproducible"] is False


@pytest.mark.parametrize(
    ("action_space", "expected_type", "expected_representation", "expected_dim"),
    [
        ("abs_qpos", "joint", "absolute", 14),
        ("relative_qpos", "joint", "relative", 14),
        ("abs_eef_rot6d", "eef", "absolute", 20),
        ("relative_eef_rot6d", "eef", "relative", 20),
    ],
)
def test_resolved_config_binds_contract_and_external_run_dir(
    tmp_path: Path,
    action_space: str,
    expected_type: str,
    expected_representation: str,
    expected_dim: int,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    contract_path = _write_contract(
        tmp_path, _contract(dataset_root, action_space=action_space)
    )
    run_dir = tmp_path / "run"

    config, path = adapter.build_resolved_config(
        ROOT / "configs/train/config.yaml", contract_path, run_dir
    )

    assert path == run_dir / "prometheus_train_config.yaml"
    assert config["data"]["root_dir"] == str(dataset_root)
    assert config["data"]["action_type"] == expected_type
    assert config["data"]["action_representation"] == expected_representation
    assert config["data"]["action_horizon"] == 32
    assert "action_dim" not in config["data"]
    assert config["output"] == {"root_dir": str(tmp_path), "run_name": "run"}
    assert config["prometheus_contract"]["action_dim"] == expected_dim
    assert config["prometheus_contract"]["robot_schema_digests"] == ["2" * 64]
    assert config["prometheus_contract"]["hardware_rollout_authorized"] is False
    assert not path.exists()


@pytest.mark.parametrize(
    ("embodiment_schema", "expected_dim"),
    [("cobot_magic_v1", 14), ("franka_wuji_v1", 54)],
)
def test_named_native_joint_schema_binds_exact_dimension(
    tmp_path: Path, embodiment_schema: str, expected_dim: int
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    contract_path = _write_contract(
        tmp_path,
        _contract(dataset_root, embodiment_schema=embodiment_schema),
    )

    config, _ = adapter.build_resolved_config(
        ROOT / "configs/train/config.yaml", contract_path, tmp_path / "run"
    )

    assert config["data"]["action_dim"] == expected_dim
    assert config["prometheus_contract"]["action_dim"] == expected_dim
    assert config["prometheus_contract"]["embodiment_schema"] == embodiment_schema


def test_unlabelled_dataset_is_rejected_instead_of_assumed_legacy(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    payload = _contract(dataset_root, embodiment_schema=None)
    assert "embodiment_schema" not in payload["robot"]

    with pytest.raises(ValueError, match="robot.embodiment_schema is required"):
        adapter.validate_dataset_contract(payload)

    payload["robot"]["embodiment_schema"] = adapter.LEGACY_EMBODIMENT_SCHEMA
    selected = adapter.validate_dataset_contract(payload)
    assert selected["embodiment_schema"] == adapter.LEGACY_EMBODIMENT_SCHEMA


def test_named_schema_is_not_inferred_from_dimension(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    payload = _contract(dataset_root)
    payload["action"]["dim"] = 54
    payload["action"]["features"][0]["shape"] = [54]
    payload["observation"]["state"][0]["shape"] = [54]

    with pytest.raises(ValueError, match="requires 14 values"):
        adapter.validate_dataset_contract(payload)

    payload = _contract(dataset_root, embodiment_schema="franka_wuji_v1")
    payload["robot"]["embodiment_schema"] = "unknown_robot_v1"
    with pytest.raises(ValueError, match="unsupported robot.embodiment_schema"):
        adapter.validate_dataset_contract(payload)


def test_rejects_wrong_dimension_camera_order_and_remote_uri(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    payload = _contract(dataset_root)
    payload["action"]["dim"] = 13
    with pytest.raises(ValueError, match="requires 14 values"):
        adapter.validate_dataset_contract(payload)

    payload = _contract(dataset_root)
    payload["observation"]["images"].reverse()
    with pytest.raises(ValueError, match="camera order"):
        adapter.validate_dataset_contract(payload)

    payload = _contract(dataset_root)
    payload["dataset"]["uri"] = "https://example.invalid/dataset.zarr"
    with pytest.raises(ValueError, match="local file"):
        adapter.validate_dataset_contract(payload)


def test_plan_emits_argv_without_writing_or_importing_models(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    contract_path = _write_contract(tmp_path, _contract(dataset_root))
    run_dir = tmp_path / "run"
    process = subprocess.run(
        [
            sys.executable,
            str(ADAPTER_PATH),
            "train",
            "--dataset-contract",
            str(contract_path),
            "--run-dir",
            str(run_dir),
            "--plan",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(process.stdout)
    assert plan["shell"] is False
    assert plan["hardware_rollout_authorized"] is False
    assert plan["argv"][:3] == [sys.executable, "-m", "lv_flow_matching.train"]
    assert not run_dir.exists()


def test_multi_gpu_and_resume_are_explicit(tmp_path: Path) -> None:
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"fixture")
    command, environment = adapter.build_native_argv(
        "resume",
        config_path=tmp_path / "config.yaml",
        run_dir=tmp_path,
        gpus="2,5",
        checkpoint=checkpoint,
        native_args=(),
    )
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc_per_node=2" in command
    assert command[-4:] == ["--resume", str(checkpoint), "--resume-mode", "full"]
    assert environment["CUDA_VISIBLE_DEVICES"] == "2,5"


def test_existing_run_is_bound_to_exact_dataset_contract(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    contract_path = _write_contract(tmp_path, _contract(dataset_root))
    run_dir = tmp_path / "run"
    config, _ = adapter.build_resolved_config(
        ROOT / "configs/train/config.yaml", contract_path, run_dir
    )
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    selected = adapter.validate_run_contract(run_dir, contract_path)
    assert selected["action_dim"] == 14

    changed = _contract(dataset_root)
    changed["dataset"]["digest"] = "9" * 64
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        adapter.validate_run_contract(run_dir, changed_path)


def test_artifact_manifest_contains_promotion_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    resolved_config = run_dir / "resolved_config.yaml"
    resolved_config.write_text(
        yaml.safe_dump(
            {
                "prometheus_contract": {
                    "action_space": "abs_qpos",
                    "action_dim": 14,
                    "dataset_digest": "1" * 64,
                    "robot_schema_digests": ["2" * 64],
                    "embodiment_schema": "arx_bimanual_v1",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("PROMETHEUS_POLICY_ADAPTER_DIGEST", "3" * 64)
    adapter._write_artifact_manifest(run_dir)

    payload = json.loads((run_dir / "prometheus_artifact.json").read_text())
    required = {
        "policy_type",
        "action_space",
        "action_dim",
        "adapter_digest",
        "checkpoint_digest",
        "dataset_digest",
        "resolved_config_digest",
        "normalization_owner",
        "robot_schema_digests",
        "source_revision",
        "embodiment_schema",
    }
    assert required <= payload.keys()
    assert payload["adapter_digest"] == "3" * 64
    assert payload["embodiment_schema"] == "arx_bimanual_v1"
    assert len(payload["source_adapter_digest"]) == 64
    assert payload["hardware_rollout_authorized"] is False
