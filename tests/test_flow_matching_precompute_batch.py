from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zarr")

from lv_flow_matching.tools.precompute_policy_latents import (
    build_frame_image_batch,
    build_remove_hand_image_batch,
)


class _Dataset:
    n_image_views = 3
    image_size = 2

    def __init__(self, *, processed_shape: tuple[int, ...] | None = None):
        self.calls: list[tuple[int, int]] = []
        self.processed_shape = processed_shape

    def get_camera(self, start: int, end: int):
        self.calls.append((start, end))
        return np.zeros((end - start, 2, 2, 9), dtype=np.uint8)

    def _process_image(self, camera):
        shape = self.processed_shape or (len(camera), 3, 3, 2, 2)
        return torch.zeros(shape, dtype=torch.uint8)


def test_precompute_reads_a_contiguous_camera_batch_once():
    dataset = _Dataset()

    batch = build_frame_image_batch(dataset, [4, 5, 6])

    assert tuple(batch.shape) == (3, 3, 3, 2, 2)
    assert dataset.calls == [(4, 7)]


@pytest.mark.parametrize("indices", [[], [4, 6], [[4, 5]]])
def test_precompute_rejects_invalid_frame_index_batches(indices):
    dataset = _Dataset()

    with pytest.raises(ValueError, match="non-empty vector|contiguous"):
        build_frame_image_batch(dataset, indices)

    assert dataset.calls == []


def test_precompute_rejects_processed_shape_mismatch():
    dataset = _Dataset(processed_shape=(3, 2, 3, 2, 2))

    with pytest.raises(ValueError, match="expected processed frame image"):
        build_frame_image_batch(dataset, [4, 5, 6])

    assert dataset.calls == [(4, 7)]


class _SliceRecorder:
    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def __getitem__(self, item):
        self.calls.append((item.start, item.stop))
        return np.zeros((item.stop - item.start, 2, 2, 3), dtype=np.uint8)


class _RemoveHandDataset:
    image_size = 2

    def __init__(self):
        self.source = _SliceRecorder()
        self.data_group = {"camera_base_remove_hand": self.source}

    def _process_image(self, camera):
        return torch.zeros((len(camera), 1, 3, 2, 2), dtype=torch.uint8)


def test_precompute_reads_a_contiguous_remove_hand_batch_once():
    dataset = _RemoveHandDataset()

    batch = build_remove_hand_image_batch(dataset, [8, 9, 10])

    assert tuple(batch.shape) == (3, 1, 3, 2, 2)
    assert dataset.source.calls == [(8, 11)]
