from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lv_flow_matching.infer.config import (
    default_action_process,
    parse_deploy_config,
)
from lv_flow_matching.infer.postprocess import apply_action_process, processed_action_dim
from lv_flow_matching.infer.preprocess import (
    build_obs_from_frames,
    parse_preprocess_config,
)
from lv_flow_matching.infer.types import (
    DEFAULT_JOINT_NAMES,
    PreprocessConfig,
)


class _IdentityNormalizer:
    def normalize_state_np(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)


def test_default_action_process_for_joint():
    cfg = {"data": {"action_type": "joint"}}
    assert default_action_process(cfg) == "abs_qpos"
    deploy = parse_deploy_config(cfg)
    assert deploy.action_process == "abs_qpos"


def test_build_obs_from_frames_joint_state_shape():
    cfg = PreprocessConfig(action_type="joint")
    frames = [_joint_frame(idx) for idx in range(8)]
    obs, state_raw = build_obs_from_frames(
        frames,
        cfg,
        _IdentityNormalizer(),
        window_size=8,
    )
    assert state_raw.shape == (8, 14)
    assert obs["state"].shape == (8, 14)
    assert obs["image"].shape == (1, 1, 3, 3, 224, 224)


def test_apply_action_process_abs_qpos_pass_through():
    traj = np.arange(32 * 14, dtype=np.float32).reshape(32, 14)
    out = apply_action_process(traj, "abs_qpos")
    np.testing.assert_array_equal(out, traj)


def test_apply_action_process_abs_eef_converts_20d_rot6d_to_14d_rpy():
    traj = np.zeros((4, 20), dtype=np.float32)
    identity_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    traj[:, 3:9] = identity_rot6d
    traj[:, 13:19] = identity_rot6d

    out = apply_action_process(traj, "abs_eef")

    assert out.shape == (4, 14)
    assert np.isfinite(out).all()
    assert processed_action_dim("abs_eef") == 14
    assert processed_action_dim("abs_qpos") == 14


def test_parse_preprocess_config_reads_joint_action_type():
    cfg = {"data": {"action_type": "joint", "image_size": 224}}
    parsed = parse_preprocess_config(cfg)
    assert parsed.action_type == "joint"
    assert parsed.state_dim == 14
    assert parsed.joint_names == DEFAULT_JOINT_NAMES


def _joint_frame(idx: int) -> Any:
    positions = np.arange(14, dtype=np.float32) + float(idx)
    return SimpleNamespace(
        samples={
            "robot_state": SimpleNamespace(
                msg=SimpleNamespace(
                    name=list(DEFAULT_JOINT_NAMES),
                    position=positions.tolist(),
                )
            ),
            "base_0_color": _image_msg(np.zeros((4, 4, 3), dtype=np.uint8), "rgb8"),
            "left_wrist_0_color": _image_msg(np.zeros((4, 4, 3), dtype=np.uint8), "rgb8"),
            "right_wrist_0_color": _image_msg(np.zeros((4, 4, 3), dtype=np.uint8), "rgb8"),
        }
    )


def _image_msg(array: np.ndarray, encoding: str) -> Any:
    array = np.ascontiguousarray(array)
    return SimpleNamespace(
        msg=SimpleNamespace(
            height=array.shape[0],
            width=array.shape[1],
            encoding=encoding,
            step=array.strides[0],
            data=array.tobytes(),
        )
    )
