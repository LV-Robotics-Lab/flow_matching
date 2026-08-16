from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def resolve_resume_path(cfg: dict[str, Any], value: str | None) -> Path | None:
    """Resolve an explicit or canonical latest checkpoint without guessing."""
    if value is None or str(value).strip().lower() in {"", "none"}:
        return None

    text = str(value).strip()
    if text.lower() == "auto":
        output = dict(cfg.get("output") or {})
        run_name = str(output.get("run_name") or "").strip()
        if not run_name:
            raise ValueError("--resume auto requires output.run_name")
        root = Path(str(output.get("root_dir", "outputs"))).expanduser()
        candidate = root / run_name / "checkpoints" / "latest.pt"
        return candidate.resolve() if candidate.is_file() else None

    candidate = Path(text).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {candidate}")
    return candidate


def requested_resume_value(cfg: dict[str, Any], cli_value: str | None) -> str | None:
    """Use the config value unless the caller explicitly supplied a CLI override."""
    if cli_value is not None:
        return cli_value
    train_cfg = cfg.get("train")
    if not isinstance(train_cfg, dict):
        raise TypeError("train config must be a mapping")
    return train_cfg.get("resume_path")


def normalize_resume_mode(value: str | None) -> str:
    mode = str(value or "full").strip().lower()
    if mode not in {"full", "weights-only"}:
        raise ValueError(
            f"invalid resume mode {value!r}; expected 'full' or 'weights-only'"
        )
    return mode


def requested_resume_mode(cfg: dict[str, Any], cli_value: str | None) -> str:
    train_cfg = cfg.get("train")
    if not isinstance(train_cfg, dict):
        raise TypeError("train config must be a mapping")
    return normalize_resume_mode(
        train_cfg.get("resume_mode") if cli_value is None else cli_value
    )


def main() -> None:
    from lv_flow_matching.tools.latent_cache import (
        apply_resolved_latent_cache_root_dir,
    )
    from lv_flow_matching.trainers.policy_trainer import main as train_main
    from lv_flow_matching.utils.train_utils import load_config

    parser = argparse.ArgumentParser(description="Train flow-matching policy")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/config.yaml",
        help="Path to training config yaml",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path, 'auto' for output/run_name/checkpoints/latest.pt, or 'none'",
    )
    parser.add_argument(
        "--resume-mode",
        choices=("full", "weights-only"),
        default=None,
        help="full restores all training state; weights-only must be explicitly requested",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        policy_root = Path(__file__).resolve().parents[1]
        config_path = policy_root / config_path

    cfg = load_config(str(config_path))
    cfg = apply_resolved_latent_cache_root_dir(cfg)
    resume_value = requested_resume_value(cfg, args.resume)
    resume_path = resolve_resume_path(cfg, resume_value)
    cfg["train"]["resume_path"] = None if resume_path is None else str(resume_path)
    cfg["train"]["resume_mode"] = requested_resume_mode(cfg, args.resume_mode)
    if resume_value is not None and str(resume_value).strip().lower() == "auto":
        if resume_path is None:
            print("[train] resume=auto found no checkpoint; starting a fresh run")
        else:
            print(f"[train] resume=auto selected {resume_path}")
    train_main(cfg)


if __name__ == "__main__":
    main()
