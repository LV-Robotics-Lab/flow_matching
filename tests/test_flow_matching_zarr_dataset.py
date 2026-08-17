from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
zarr = pytest.importorskip("zarr")

from lv_flow_matching.datasets.zarr_dataset import (
    ZarrDataset,
    require_complete_zarr_source,
)


@pytest.mark.parametrize("complete", [None, False, 0, 1, "true", "false"])
def test_prometheus_fm_reader_rejects_explicit_nontrue_completion_marker(complete):
    root = SimpleNamespace(
        attrs={"format": "prometheus_flow_matching_v1", "complete": complete}
    )

    with pytest.raises(ValueError, match="source is incomplete"):
        require_complete_zarr_source(root, zarr_path="/tmp/incomplete.zarr")


def test_prometheus_fm_reader_rejects_missing_completion_marker():
    root = SimpleNamespace(attrs={"format": "prometheus_flow_matching_v1"})

    with pytest.raises(ValueError, match="source is incomplete"):
        require_complete_zarr_source(root, zarr_path="/tmp/incomplete.zarr")


def test_prometheus_fm_reader_accepts_complete_source():
    root = SimpleNamespace(
        attrs={"format": "prometheus_flow_matching_v1", "complete": True}
    )

    require_complete_zarr_source(root, zarr_path="/tmp/complete.zarr")


def test_marker_free_legacy_source_remains_compatible():
    require_complete_zarr_source(SimpleNamespace(attrs={}), zarr_path="/tmp/legacy.zarr")


@pytest.mark.parametrize("complete", [None, False, 0, 1, "true", "false"])
def test_reader_rejects_explicit_nontrue_legacy_marker(complete):
    root = SimpleNamespace(attrs={"complete": complete})

    with pytest.raises(ValueError, match="source is incomplete"):
        require_complete_zarr_source(root, zarr_path="/tmp/incomplete-legacy.zarr")


def test_dataset_constructor_applies_completion_gate_before_reading_groups(tmp_path):
    replay_buffer = tmp_path / "replay_buffer.zarr"
    root = zarr.open_group(replay_buffer, mode="w")
    root.attrs.update({"format": "prometheus_flow_matching_v1", "complete": False})

    with pytest.raises(ValueError, match="source is incomplete"):
        ZarrDataset(str(tmp_path), use_tactile=False)


def _write_joint_zarr(root, *, state_dim: int) -> None:
    replay = zarr.open_group(root / "replay_buffer.zarr", mode="w")
    data = replay.require_group("data")
    meta = replay.require_group("meta")
    frames = 24

    def write(group, name, values):
        create_array = getattr(group, "create_array", None)
        if callable(create_array):
            create_array(name, data=values)
        else:
            group.create_dataset(name, data=values)

    write(data, "camera", np.zeros((frames, 4, 4, 9), dtype=np.uint8))
    state = np.arange(frames * state_dim, dtype=np.float32).reshape(frames, state_dim)
    write(data, "state_30hz", state)
    write(data, "action_30hz", state + 0.5)
    write(meta, "episode_ends", np.asarray([frames], dtype=np.int64))


@pytest.mark.parametrize("action_dim", [14, 54])
def test_joint_dataset_enforces_explicit_native_state_and_action_dim(
    tmp_path, action_dim
):
    _write_joint_zarr(tmp_path, state_dim=action_dim)
    dataset = ZarrDataset(
        str(tmp_path),
        window_size=2,
        n_image_steps=1,
        action_horizon=4,
        action_type="joint",
        action_dim=action_dim,
        action_representation="absolute",
        use_tactile=False,
        image_size=4,
        camera_views=["base_0", "left_wrist_0", "right_wrist_0"],
        normalizer_max_windows=4,
        max_windows=4,
    )

    assert dataset.action_dim == action_dim
    item = dataset[0]
    assert item["obs"]["state"].shape == (2, action_dim)
    assert item["action"].shape == (4, action_dim)


def test_explicit_action_dim_rejects_mismatched_zarr(tmp_path):
    _write_joint_zarr(tmp_path, state_dim=14)
    with pytest.raises(ValueError, match="configured action_dim=54"):
        ZarrDataset(
            str(tmp_path),
            window_size=2,
            n_image_steps=1,
            action_horizon=4,
            action_type="joint",
            action_dim=54,
            use_tactile=False,
            image_size=4,
        )


def test_custom_action_dim_is_rejected_for_eef_data():
    with pytest.raises(ValueError, match="only supported for joint"):
        ZarrDataset._resolve_action_dim("eef", 54)


@pytest.mark.parametrize("value", [True, 54.0, "54"])
def test_action_dim_must_be_an_integer(value):
    with pytest.raises(TypeError, match="must be an integer"):
        ZarrDataset._resolve_action_dim("joint", value)
