from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from lv_flow_matching.infer.config import action_dim_for_config
from lv_flow_matching.infer.postprocess import processed_action_dim
from lv_flow_matching.utils.train_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "configs/train/embodiments"
LAUNCHER = ROOT / "scripts/train_embodiment.sh"


def _source_status() -> str:
    return subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _assert_prometheus_provenance(
    profile: dict[str, object], *, branch: str, source_path: str
) -> None:
    provenance = profile["profile"]["provenance"]
    assert provenance["repository"] == (
        "https://github.com/LV-Robotics-Lab/PrometheusV4.git"
    )
    assert provenance["branch"] == branch
    assert provenance["source_path"] == source_path
    assert re.fullmatch(r"[0-9a-f]{40}", provenance["revision"])
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["source_sha256"])


@pytest.mark.parametrize(
    ("name", "schema", "dim", "state_key", "action_key", "action_hz"),
    [
        (
            "cobot_magic_v1.yaml",
            "cobot_magic_v1",
            14,
            "state_25hz",
            "action_25hz",
            25.0,
        ),
        (
            "cobot_magic_v1_smoke.yaml",
            "cobot_magic_v1",
            14,
            "state_25hz",
            "action_25hz",
            25.0,
        ),
        (
            "cobot_magic_v1_daimon25_remote.yaml",
            "cobot_magic_v1",
            14,
            "state_25hz",
            "action_25hz",
            25.0,
        ),
        (
            "cobot_magic_v1_joint30_legacy.yaml",
            "cobot_magic_v1",
            14,
            "state_30hz",
            "action_30hz",
            25.0,
        ),
        (
            "franka_wuji_v1_smoke.yaml",
            "franka_wuji_v1",
            54,
            "state_30hz",
            "action_30hz",
            30.0,
        ),
    ],
)
def test_every_profile_has_an_exact_hardware_free_training_contract(
    name: str,
    schema: str,
    dim: int,
    state_key: str,
    action_key: str,
    action_hz: float,
) -> None:
    path = PROFILE_DIR / name
    profile = load_config(str(path))

    assert profile["profile"]["embodiment"] == schema
    assert profile["profile"]["hardware_imports_allowed"] is False
    assert profile["data"]["action_type"] == "joint"
    # The native joint action dimension is also the state dimension enforced
    # against both Zarr arrays by ZarrDataset when action_dim is explicit.
    assert profile["data"]["action_dim"] == dim
    assert profile["data"]["state_key"] == state_key
    assert profile["data"]["action_key"] == action_key
    assert profile["data"]["camera_views"] == [
        "base_0",
        "left_wrist_0",
        "right_wrist_0",
    ]
    assert profile["deploy"] == {
        "action_process": "abs_qpos",
        "action_hz": action_hz,
    }
    assert "preprocess" not in profile["deploy"]
    assert "import prometheus" not in path.read_text(encoding="utf-8")


def test_cobot_magic_v1_profiles_are_external_and_schema_bound() -> None:
    canonical_path = PROFILE_DIR / "cobot_magic_v1.yaml"
    smoke_path = PROFILE_DIR / "cobot_magic_v1_smoke.yaml"
    canonical = load_config(str(canonical_path))
    smoke = load_config(str(smoke_path))

    assert canonical["profile"]["hardware_imports_allowed"] is False
    assert canonical["data"]["camera_views"] == [
        "base_0",
        "left_wrist_0",
        "right_wrist_0",
    ]
    assert canonical["data"]["action_type"] == "joint"
    assert canonical["data"]["action_dim"] == 14
    assert canonical["data"]["state_key"] == "state_25hz"
    assert canonical["data"]["action_key"] == "action_25hz"
    assert canonical["deploy"] == {"action_process": "abs_qpos", "action_hz": 25.0}

    for profile_path, profile in ((canonical_path, canonical), (smoke_path, smoke)):
        text = profile_path.read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "import prometheus" not in text
        assert Path(profile["output"]["root_dir"]).is_absolute()
        assert "/absolute/path/to/" in profile["output"]["root_dir"]
        assert Path(profile["data"]["root_dir"]).is_absolute()


def test_cobot_smoke_profile_is_bounded_and_schema_compatible() -> None:
    canonical = load_config(str(PROFILE_DIR / "cobot_magic_v1.yaml"))
    smoke = load_config(str(PROFILE_DIR / "cobot_magic_v1_smoke.yaml"))

    assert smoke["profile"]["embodiment"] == canonical["profile"]["embodiment"]
    assert smoke["data"]["action_dim"] == canonical["data"]["action_dim"]
    assert smoke["data"]["state_key"] == canonical["data"]["state_key"]
    assert smoke["data"]["action_key"] == canonical["data"]["action_key"]
    assert smoke["deploy"] == canonical["deploy"]
    assert smoke["data"]["max_windows"] == 64
    assert smoke["train"]["epochs"] == 1
    assert smoke["train"]["max_train_batches"] == 2
    assert smoke["train"]["num_workers"] == 0
    assert smoke["train"]["open_loop_test_every"] == 0


def test_cobot_daimon25_remote_profile_preserves_migrated_recipe() -> None:
    path = PROFILE_DIR / "cobot_magic_v1_daimon25_remote.yaml"
    profile = load_config(str(path))

    _assert_prometheus_provenance(
        profile,
        branch="hardware/cobot-daimon",
        source_path=(
            "prometheus/policy/flow_matching/configs/train/"
            "cobot_magic_joint_unet_teleop_daimon25_remote.yaml"
        ),
    )
    assert profile["profile"]["status"] == "maintained"
    assert profile["output"]["run_name"] == "teleop_daimon_unet25"
    assert profile["train"]["use_amp"] is True
    assert profile["precompute"]["batch_size"] == 64
    assert "auxiliary_loss" not in profile["models"]["fm"]
    text = path.read_text(encoding="utf-8")
    assert "/home/" not in text
    assert "/absolute/path/to/" in profile["output"]["root_dir"]
    assert Path(profile["data"]["root_dir"]).is_absolute()


def test_franka_wuji_v1_profile_uses_generic_54d_joint_contract() -> None:
    profile = load_config(str(PROFILE_DIR / "franka_wuji_v1_smoke.yaml"))

    _assert_prometheus_provenance(
        profile,
        branch="hardware/franka-wuji",
        source_path=(
            "prometheus/policy/flow_matching/configs/train/"
            "franka_wuji_smoke.yaml"
        ),
    )
    assert profile["profile"]["hardware_imports_allowed"] is False
    assert profile["data"]["action_type"] == "joint"
    assert profile["data"]["action_dim"] == 54
    assert profile["deploy"] == {"action_process": "abs_qpos", "action_hz": 30.0}
    assert action_dim_for_config(profile) == 54
    assert processed_action_dim("abs_qpos", native_action_dim=54) == 54


def test_legacy_rtx5090_lock_is_preserved_byte_for_byte() -> None:
    lock = ROOT / "environments/legacy/cobot_magic_v1_rtx5090_20260815.txt"
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()

    assert digest == "d0a744c2cc662bad6666551ce016c28e2269f7a89aa67ebc01536cc09f43f34c"
    text = lock.read_text(encoding="utf-8")
    assert "torch==2.11.0+cu128" in text
    assert "torchvision==0.26.0+cu128" in text
    assert "zarr==2.18.3" in text


def test_embodiment_launcher_requires_explicit_fail_closed_inputs() -> None:
    before = _source_status()
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    missing = subprocess.run(
        ["bash", str(LAUNCHER)], check=False, capture_output=True, text=True
    )
    assert missing.returncode == 2
    assert "--config is required" in missing.stderr

    auto = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--config",
            str(PROFILE_DIR / "cobot_magic_v1.yaml"),
            "--resume",
            "auto",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert auto.returncode == 2
    assert "--resume auto is forbidden" in auto.stderr
    assert _source_status() == before


def test_embodiment_launcher_rejects_placeholder_and_nonempty_outputs(tmp_path: Path) -> None:
    before = _source_status()
    env = dict(os.environ, PYTHON=sys.executable)
    placeholder = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--config",
            str(PROFILE_DIR / "cobot_magic_v1.yaml"),
            "--resume",
            "none",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert placeholder.returncode == 6
    assert "replace the checked-in /absolute/path/to" in placeholder.stderr

    output_root = tmp_path / "artifacts"
    run_dir = output_root / "collision"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "latest.pt"
    sentinel.write_bytes(b"do not overwrite")
    config = tmp_path / "collision.yaml"
    config.write_text(
        yaml.safe_dump(
            {"output": {"root_dir": str(output_root), "run_name": "collision"}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    collision = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--config",
            str(config),
            "--resume",
            "none",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert collision.returncode == 6
    assert "fresh run directory is not empty" in collision.stderr
    assert sentinel.read_bytes() == b"do not overwrite"
    assert _source_status() == before


def test_embodiment_launcher_rejects_symlinked_fresh_run(tmp_path: Path) -> None:
    before = _source_status()
    historical = tmp_path / "historical"
    historical.mkdir()
    sentinel = historical / "latest.pt"
    sentinel.write_bytes(b"historical checkpoint")
    output_root = tmp_path / "artifacts"
    output_root.mkdir()
    (output_root / "fresh").symlink_to(historical, target_is_directory=True)
    config = tmp_path / "symlink.yaml"
    config.write_text(
        yaml.safe_dump(
            {"output": {"root_dir": str(output_root), "run_name": "fresh"}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--config",
            str(config),
            "--resume",
            "none",
        ],
        env=dict(os.environ, PYTHON=sys.executable),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 6
    assert "must not be a symbolic link" in result.stderr
    assert sentinel.read_bytes() == b"historical checkpoint"
    assert _source_status() == before


def test_embodiment_launcher_exact_resume_restores_nonzero_optimizer_state(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from lv_flow_matching.tools.normalizer import DatasetNormalizer, FieldNormalizer
    from lv_flow_matching.trainers.policy_trainer import get_checkpoint_state

    class Dataset:
        normalizer = DatasetNormalizer(
            state=FieldNormalizer.identity(2),
            action=FieldNormalizer.identity(2),
            tactile=None,
            action_type="joint",
            action_representation="absolute",
        )

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones((2, 2))).square().mean()
    loss.backward()
    optimizer.step()
    checkpoint = tmp_path / "resume.pt"
    torch.save(
        get_checkpoint_state(
            model,
            optimizer,
            Dataset(),
            epoch=2,
            global_step=7,
            cfg={"train": {}},
        ),
        checkpoint,
    )

    output_root = tmp_path / "artifacts"
    config = tmp_path / "resume.yaml"
    config.write_text(
        yaml.safe_dump(
            {"output": {"root_dir": str(output_root), "run_name": "resume-smoke"}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "resume-evidence.json"
    probe = tmp_path / "resume_probe.py"
    probe.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys

            import torch

            from lv_flow_matching.trainers.policy_trainer import (
                preflight_resume_checkpoint,
                restore_checkpoint_state,
            )

            resume = sys.argv[sys.argv.index("--resume") + 1]
            mode_arg = sys.argv[sys.argv.index("--resume-mode") + 1]
            cfg = {
                "train": {
                    "epochs": 4,
                    "resume_path": resume,
                    "resume_mode": mode_arg,
                    "optimizer": {"scheduler": "none"},
                    "use_amp": False,
                }
            }
            state, mode = preflight_resume_checkpoint(cfg)
            assert state is not None
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

            class Dataset:
                normalizer = None

            restore_checkpoint_state(state, model, optimizer, Dataset())
            steps_before = [
                int(value["step"].item()) for value in optimizer.state.values()
            ]
            assert mode == "full"
            assert int(state["global_step"]) == 7
            assert steps_before and min(steps_before) > 0
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones((2, 2))).square().mean().backward()
            optimizer.step()
            steps_after = [
                int(value["step"].item()) for value in optimizer.state.values()
            ]
            assert all(after == before + 1 for before, after in zip(steps_before, steps_after))
            with open(os.environ["RESUME_PROBE_EVIDENCE"], "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "global_step": int(state["global_step"]),
                        "mode": mode,
                        "optimizer_steps_before": steps_before,
                        "optimizer_steps_after": steps_after,
                    },
                    handle,
                )
            """
        ),
        encoding="utf-8",
    )
    wrapper = tmp_path / "python-probe"
    wrapper.write_text(
        "#!/bin/sh\n"
        f"if [ \"$1\" = \"-\" ]; then exec {shlex.quote(sys.executable)} \"$@\"; fi\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(probe))} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    before = _source_status()

    subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--config",
            str(config),
            "--resume",
            str(checkpoint),
            "--resume-mode",
            "full",
        ],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHON=str(wrapper),
            PYTHONPATH=str(ROOT),
            RESUME_PROBE_EVIDENCE=str(evidence),
        ),
        check=True,
    )

    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload == {
        "global_step": 7,
        "mode": "full",
        "optimizer_steps_before": [1, 1],
        "optimizer_steps_after": [2, 2],
    }
    assert _source_status() == before
