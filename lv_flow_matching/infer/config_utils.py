"""Config helpers the inference path needs, with no training dependency.

``utils.train_utils`` imports torch, yaml, and numpy because training needs
them. Inference does not, and a deployment checkout should not have to install
a trainer to read a resolved config, so these two helpers live here and
``utils.train_utils`` re-exports them rather than keeping a second copy.
"""

from __future__ import annotations

from typing import Any, Mapping

_MISSING = object()


def cfg_get(cfg: Mapping[str, Any], path: str, default: Any = _MISSING) -> Any:
    current: Any = cfg
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            if default is _MISSING:
                raise KeyError(path)
            return default
        current = current[part]
    return current


def sync_fm_action_horizon_from_data(
    fm_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Single source of truth: ``data.action_horizon`` (and optional ``data.n_action_steps``).

    Injects into a copy of ``models.fm`` so Dataset and Policy stay aligned.
    Legacy resolved configs that only set ``models.fm.action_horizon`` still work.
    """
    resolved = dict(fm_cfg)
    if "action_horizon" in data_cfg and data_cfg["action_horizon"] is not None:
        horizon = int(data_cfg["action_horizon"])
    elif "action_horizon" in resolved and resolved["action_horizon"] is not None:
        horizon = int(resolved["action_horizon"])
    else:
        horizon = 32
    if horizon < 1:
        raise ValueError(f"action_horizon must be >= 1, got {horizon}")

    if "n_action_steps" in data_cfg and data_cfg["n_action_steps"] is not None:
        n_steps = int(data_cfg["n_action_steps"])
    elif "action_horizon" not in data_cfg and resolved.get("n_action_steps") is not None:
        # Legacy resolved_config: horizon lived only under models.fm
        n_steps = int(resolved["n_action_steps"])
    else:
        n_steps = horizon

    if n_steps < 1:
        raise ValueError(f"n_action_steps must be >= 1, got {n_steps}")
    if n_steps > horizon:
        raise ValueError(
            f"n_action_steps ({n_steps}) cannot exceed action_horizon ({horizon})"
        )

    resolved["action_horizon"] = horizon
    resolved["n_action_steps"] = n_steps
    return resolved
