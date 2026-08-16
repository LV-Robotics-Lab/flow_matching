from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("zarr")

from lv_flow_matching.tools import eval_deploy_open_loop


def test_deploy_eval_dataset_reads_rgb_instead_of_training_latents(monkeypatch):
    captured = {}

    class FakeDataset:
        @classmethod
        def from_config(cls, config):
            captured.update(config)
            return "dataset"

    monkeypatch.setattr(eval_deploy_open_loop, "ZarrDataset", FakeDataset)
    cfg = {
        "data": {
            "root_dir": "/tmp/data",
            "fit_normalizer": True,
            "use_camera_latent": True,
            "latent_cache_root_dir": "/tmp/data/latent_cache",
        }
    }

    assert eval_deploy_open_loop.build_eval_dataset(cfg) == "dataset"
    assert captured["fit_normalizer"] is False
    assert captured["use_camera_latent"] is False
    assert captured["latent_cache_root_dir"] is None
    assert cfg["data"]["fit_normalizer"] is True
    assert cfg["data"]["use_camera_latent"] is True
    assert cfg["data"]["latent_cache_root_dir"] == "/tmp/data/latent_cache"
