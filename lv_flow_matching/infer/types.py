from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lv_flow_matching.tools.tactile_feat import TACTILE_BUNDLE_ORDER, TACTILE_FEATURE_DIM


LEGACY_DUAL_ARM_JOINT_NAMES: tuple[str, ...] = tuple(
    [f"left_joint_{idx}" for idx in range(1, 7)] + ["left_gripper"]
    + [f"right_joint_{idx}" for idx in range(1, 7)] + ["right_gripper"]
)
LEGACY_DUAL_ARM_JOINT_DIM = len(LEGACY_DUAL_ARM_JOINT_NAMES)
LEGACY_DUAL_ARM_EEF_ROT6D_DIM = 20
LEGACY_DUAL_ARM_DEPLOY_DIM = 14
LEGACY_THREE_CAMERA_VIEWS = (
    "base_0_color",
    "left_wrist_0_color",
    "right_wrist_0_color",
)
LEGACY_DUAL_ARM_EEF_STREAMS = ("robot_state", "left_eef", "right_eef")

# Compatibility alias for downstream deployments. New code should use the
# explicit legacy-contract name above.
DEFAULT_JOINT_NAMES = LEGACY_DUAL_ARM_JOINT_NAMES

DEFAULT_TACTILE_POINTCLOUD_SHAPE: tuple[int, int, int] = (35, 20, 6)


def tactile_flow_stream_name(bundle: str) -> str:
    return f"{bundle}_tactile_flow"


DEFAULT_TACTILE_STREAMS: tuple[str, ...] = tuple(
    tactile_flow_stream_name(bundle) for bundle in TACTILE_BUNDLE_ORDER
)


@dataclass(frozen=True)
class PreprocessConfig:
    action_type: str = "eef"
    camera_views: tuple[str, ...] = LEGACY_THREE_CAMERA_VIEWS
    gripper_names: dict[str, str] = field(
        default_factory=lambda: {"left": "left_gripper", "right": "right_gripper"}
    )
    joint_names: tuple[str, ...] = DEFAULT_JOINT_NAMES
    image_size: int = 224
    gripper_width_m: float = 0.082
    use_tactile: bool = False
    tactile_streams: tuple[str, ...] = DEFAULT_TACTILE_STREAMS
    tactile_pointcloud_shape: tuple[int, int, int] = DEFAULT_TACTILE_POINTCLOUD_SHAPE

    @property
    def state_dim(self) -> int:
        if self.action_type == "joint":
            if not self.joint_names:
                raise ValueError("joint preprocessing requires at least one joint name")
            return len(self.joint_names)
        if self.action_type == "eef":
            return LEGACY_DUAL_ARM_EEF_ROT6D_DIM
        raise ValueError(f"unsupported action_type={self.action_type!r}")

    @property
    def state_stream_names(self) -> tuple[str, ...]:
        if self.action_type == "joint":
            return ("robot_state",)
        return LEGACY_DUAL_ARM_EEF_STREAMS

    @property
    def tactile_feature_shape(self) -> tuple[int, int, int]:
        height, width, _channels = self.tactile_pointcloud_shape
        return height, width, TACTILE_FEATURE_DIM


@dataclass(frozen=True)
class InferenceChunk:
    actions: np.ndarray
    action_space: str
    hz: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        object.__setattr__(self, "actions", actions)
