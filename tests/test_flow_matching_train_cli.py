from __future__ import annotations

from pathlib import Path

import pytest

from lv_flow_matching.train import (
    normalize_resume_mode,
    requested_resume_mode,
    requested_resume_value,
    resolve_resume_path,
)


def _config(root: Path, *, resume_path: str | None = None) -> dict:
    return {
        "output": {"root_dir": str(root), "run_name": "generic_fm_run"},
        "train": {"resume_path": resume_path},
    }


def test_resume_auto_starts_fresh_then_selects_latest(tmp_path: Path):
    cfg = _config(tmp_path)
    assert resolve_resume_path(cfg, "auto") is None

    latest = tmp_path / "generic_fm_run" / "checkpoints" / "latest.pt"
    latest.parent.mkdir(parents=True)
    latest.touch()

    assert resolve_resume_path(cfg, "auto") == latest.resolve()


def test_resume_auto_requires_stable_run_name(tmp_path: Path):
    cfg = _config(tmp_path)
    cfg["output"]["run_name"] = None

    with pytest.raises(ValueError, match="requires output.run_name"):
        resolve_resume_path(cfg, "auto")


def test_explicit_resume_requires_existing_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_resume_path(_config(tmp_path), str(checkpoint))

    checkpoint.touch()
    assert resolve_resume_path(_config(tmp_path), str(checkpoint)) == checkpoint.resolve()


def test_cli_override_can_disable_or_replace_config_resume(tmp_path: Path):
    configured = tmp_path / "configured.pt"
    configured.touch()
    cfg = _config(tmp_path, resume_path=str(configured))

    assert requested_resume_value(cfg, None) == str(configured)
    assert requested_resume_value(cfg, "none") == "none"
    assert resolve_resume_path(cfg, requested_resume_value(cfg, "none")) is None


def test_requested_resume_requires_train_mapping(tmp_path: Path):
    cfg = _config(tmp_path)
    cfg["train"] = None

    with pytest.raises(TypeError, match="train config must be a mapping"):
        requested_resume_value(cfg, None)


def test_resume_mode_defaults_full_and_requires_explicit_weights_only(tmp_path: Path):
    cfg = _config(tmp_path)

    assert requested_resume_mode(cfg, None) == "full"
    assert requested_resume_mode(cfg, "weights-only") == "weights-only"


def test_resume_mode_rejects_ambiguous_values():
    with pytest.raises(ValueError, match="expected 'full' or 'weights-only'"):
        normalize_resume_mode("weights")
