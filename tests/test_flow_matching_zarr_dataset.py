from __future__ import annotations

from types import SimpleNamespace

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
