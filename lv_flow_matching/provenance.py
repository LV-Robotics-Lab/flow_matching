from __future__ import annotations

from functools import lru_cache
import subprocess
from pathlib import Path
from typing import Any

from lv_flow_matching import __version__


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _git_value(
    args: list[str],
    cwd: Path,
    *,
    allow_empty: bool = False,
) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value if value or allow_empty else None


@lru_cache(maxsize=1)
def source_provenance() -> dict[str, Any]:
    """Return reproducible source identity for manifests and checkpoints."""
    commit = _git_value(["rev-parse", "HEAD"], _REPOSITORY_ROOT)
    status = _git_value(
        ["status", "--porcelain", "--untracked-files=normal"],
        _REPOSITORY_ROOT,
        allow_empty=True,
    )
    superproject_text = _git_value(
        ["rev-parse", "--show-superproject-working-tree"],
        _REPOSITORY_ROOT,
    )
    superproject = Path(superproject_text) if superproject_text else None
    return {
        "distribution": "lv-robotics-flow-matching",
        "package_version": __version__,
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "prometheus_git_commit": (
            _git_value(["rev-parse", "HEAD"], superproject)
            if superproject is not None
            else None
        ),
    }
