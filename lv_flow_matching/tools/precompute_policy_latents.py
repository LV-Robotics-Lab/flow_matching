from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
import zarr
from tqdm import tqdm

_FLOW_MATCHING_ROOT = Path(__file__).resolve().parents[2]

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lv_flow_matching.datasets import ZarrDataset  # noqa: E402
from lv_flow_matching.models.fm.encoders.dino_v2 import DinoV2SmallEncoder, resolve_dino_model_name  # noqa: E402
from lv_flow_matching.tools.latent_cache import (  # noqa: E402
    CAMERA_BASE_REMOVE_HAND_KEY,
    DINOV2_NUM_TOKENS,
    FRAME_CACHE_VERSION,
    apply_resolved_latent_cache_root_dir,
    frame_cache_matches,
    normalize_token_mode,
    remove_hand_frame_cache_matches,
    resolve_frame_backbone_base_remove_hand_zarr_path,
    resolve_frame_backbone_zarr_path,
    token_mode_num_tokens,
    write_latent_cache_identity_attrs,
    write_token_mode_attrs,
)
from lv_flow_matching.utils.train_utils import cfg_get, load_config  # noqa: E402


def build_dataset(cfg: dict) -> ZarrDataset:
    """RGB-only dataset for frame encoding (ignores train window / memory / latent / mix)."""
    data_cfg = dict(cfg["data"])
    data_cfg["use_camera_latent"] = False
    data_cfg["latent_cache_root_dir"] = None
    data_cfg["fit_normalizer"] = False
    data_cfg["camera_augmentation"] = False
    data_cfg["mix_base_remove_hand"] = False
    # Always encode all zarr views; train slices later.
    data_cfg.pop("camera_views", None)
    # Frame SSOT: never truncate for partial smoke encodes.
    data_cfg.pop("max_windows", None)
    data_cfg.pop("memory", None)
    return ZarrDataset.from_config(data_cfg)


def resolve_output_path_from_cfg(cfg: dict, output_path: str | None = None) -> str:
    if output_path:
        return str(output_path)
    cfg = apply_resolved_latent_cache_root_dir(dict(cfg))
    root = cfg_get(cfg, "data.latent_cache_root_dir", None) or cfg_get(cfg, "data.root_dir")
    if root is None:
        raise KeyError("data.root_dir is required to resolve precompute output path")
    return resolve_frame_backbone_zarr_path(str(root))


def resolve_remove_hand_output_path(cfg: dict) -> str:
    cfg = apply_resolved_latent_cache_root_dir(dict(cfg))
    root = cfg_get(cfg, "data.latent_cache_root_dir", None) or cfg_get(
        cfg, "data.root_dir"
    )
    if root is None:
        raise KeyError("data.root_dir is required to resolve remove-hand cache path")
    return resolve_frame_backbone_base_remove_hand_zarr_path(str(root))


def build_frame_image_batch(dataset: ZarrDataset, frame_indices: list[int]) -> torch.Tensor:
    indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("frame_indices must be a non-empty vector")
    if len(indices) > 1 and np.any(np.diff(indices) != 1):
        raise ValueError("frame_indices must be contiguous")

    # Read a contiguous Zarr slice once. Reading one frame at a time repeatedly
    # decompresses the same camera chunk and leaves the accelerator idle.
    camera = dataset.get_camera(int(indices[0]), int(indices[-1]) + 1)
    processed = dataset._process_image(camera)
    expected_shape = (
        len(indices),
        dataset.n_image_views,
        3,
        dataset.image_size,
        dataset.image_size,
    )
    if tuple(processed.shape) != expected_shape:
        raise ValueError(
            f"expected processed frame image {expected_shape}, got {tuple(processed.shape)}"
        )
    return processed


def build_remove_hand_image_batch(
    dataset: ZarrDataset,
    frame_indices: list[int],
) -> torch.Tensor:
    indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("frame_indices must be a non-empty vector")
    if len(indices) > 1 and np.any(np.diff(indices) != 1):
        raise ValueError("frame_indices must be contiguous")

    source = dataset.data_group[CAMERA_BASE_REMOVE_HAND_KEY]
    camera = np.asarray(source[int(indices[0]) : int(indices[-1]) + 1])
    processed = dataset._process_image(camera)
    expected_shape = (
        len(indices),
        1,
        3,
        dataset.image_size,
        dataset.image_size,
    )
    if tuple(processed.shape) != expected_shape:
        raise ValueError(
            f"expected processed remove-hand image {expected_shape}, "
            f"got {tuple(processed.shape)}"
        )
    return processed


def _build_encoder(fm_cfg: dict, device: torch.device) -> DinoV2SmallEncoder:
    model_name = resolve_dino_model_name(
        fm_cfg.get("image_encoder_name"),
        fm_cfg.get("dino_model_name"),
    )
    fm_cfg["dino_model_name"] = model_name
    encoder = DinoV2SmallEncoder(
        out_dim=int(fm_cfg.get("image_feat_dim", 256)),
        pretrained=bool(fm_cfg.get("image_pretrained", True)),
        freeze=True,
        model_name=model_name,
    ).to(device)
    encoder.eval()
    return encoder


def _tokens_to_stored_feat(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
    token_mode: str,
) -> np.ndarray:
    if tokens.ndim != 3:
        raise ValueError(f"expected tokens (B*V,N,D), got {tuple(tokens.shape)}")
    if token_mode == "cls":
        feature = tokens[:, 0].reshape(batch_size, num_views, tokens.shape[-1])
    else:
        feature = tokens.reshape(
            batch_size, num_views, tokens.shape[1], tokens.shape[2]
        )
    return feature.detach().cpu().numpy().astype(np.float32, copy=False)


def precompute_image_latents(
    cfg: dict,
    *,
    force: bool = False,
    dataset: ZarrDataset | None = None,
) -> str:
    """Write frame-only DINO backbone cache (scheme A). Independent of train windows."""
    cfg = apply_resolved_latent_cache_root_dir(dict(cfg))
    pre_cfg = dict(cfg.get("precompute", {}))
    output_path = resolve_output_path_from_cfg(cfg, pre_cfg.get("output_path"))
    # yaml overwrite=true kept as force alias for backward compat
    force = bool(force) or bool(pre_cfg.get("overwrite", False))

    batch_size = max(1, int(pre_cfg.get("batch_size", 256)))
    token_mode = normalize_token_mode(pre_cfg.get("token_mode"), default="all")
    device = torch.device(str(pre_cfg.get("device", cfg_get(cfg, "runtime.device", "cuda"))))
    fm_cfg = dict(cfg["models"]["fm"])
    if not bool(fm_cfg.get("freeze_image_encoder", True)):
        raise ValueError("Precompute requires models.fm.freeze_image_encoder=true.")

    if dataset is None:
        dataset = build_dataset(cfg)
    total_frames = int(dataset.ram_data[dataset.camera_key].shape[0])
    state_frames = int(dataset.ram_data[dataset.state_key].shape[0])
    if total_frames != state_frames:
        raise ValueError(
            f"camera/state frame count mismatch before encode: camera={total_frames}, state={state_frames}"
        )

    if (not force) and frame_cache_matches(
        output_path,
        fm_cfg=fm_cfg,
        source_zarr_path=dataset.zarr_path,
        image_size=int(dataset.image_size),
        camera_views=dataset.camera_views,
        total_frames=total_frames,
        token_mode=token_mode,
        color_order="rgb",
    ):
        print(f"[precompute] frame cache identity match, skipping: {output_path}")
        return output_path

    if os.path.isdir(output_path):
        print(f"[precompute] removing existing cache (force={force}): {output_path}")
        shutil.rmtree(output_path)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    image_encoder = _build_encoder(fm_cfg, device)
    model_name = str(fm_cfg["dino_model_name"])

    out_root = zarr.open_group(output_path, mode="w")
    out_root.attrs["cache_version"] = int(FRAME_CACHE_VERSION)
    out_root.attrs["source_zarr_path"] = dataset.zarr_path
    out_root.attrs["image_size"] = int(dataset.image_size)
    out_root.attrs["color_order"] = "rgb"
    out_root.attrs["frame_image_selection"] = "all_frames"
    write_token_mode_attrs(out_root, token_mode)
    write_latent_cache_identity_attrs(out_root, fm_cfg)
    out_root.attrs["camera_views"] = ",".join(dataset.camera_views)

    data_group = out_root.create_group("data")
    # empty meta group kept for zarr layout stability
    out_root.create_group("meta")

    # Token cache is large (T,V,257,D); keep chunks modest for OSS/CPFS writes.
    chunk_bsz = max(1, min(batch_size, 64))
    frame_arr = None
    print(
        f"[precompute] encoding all frames: T={total_frames}, views={list(dataset.camera_views)}, "
        f"model={model_name}, token_mode={token_mode}, "
        f"batch_size={batch_size}, device={device}, out={output_path}"
    )

    for start_idx in tqdm(
        range(0, total_frames, batch_size),
        desc="precompute:frame_image_backbone_feat",
        unit="batch",
    ):
        frame_indices = list(range(start_idx, min(start_idx + batch_size, total_frames)))
        image_batch = build_frame_image_batch(dataset, frame_indices).to(device, non_blocking=True)

        with torch.inference_mode():
            bsz, num_views = image_batch.shape[:2]
            flat = image_batch.reshape(bsz * num_views, *image_batch.shape[2:])
            tokens = image_encoder.extract_backbone_feat(flat)  # (B*V, 257, D)
            img = _tokens_to_stored_feat(
                tokens,
                batch_size=bsz,
                num_views=num_views,
                token_mode=token_mode,
            )
        if frame_arr is None:
            frame_arr = data_group.create_array(
                "frame_image_backbone_feat",
                shape=(total_frames,) + img.shape[1:],
                chunks=(chunk_bsz,) + img.shape[1:],
                dtype="f4",
            )
            out_root.attrs["image_backbone_dim"] = int(img.shape[-1])
            out_root.attrs["n_image_views"] = int(img.shape[1])
            out_root.attrs["image_num_tokens"] = token_mode_num_tokens(token_mode)
        frame_arr[start_idx : start_idx + len(frame_indices)] = img

    if frame_arr is None:
        raise RuntimeError("no frames were encoded")

    print(f"[precompute] saved frame backbone cache: {output_path}")
    print(
        f"[precompute] frame_image_backbone_feat shape={tuple(frame_arr.shape)}, "
        f"token_mode={token_mode}"
    )
    return output_path


def precompute_base_remove_hand_latents(
    cfg: dict,
    *,
    force: bool = False,
    dataset: ZarrDataset | None = None,
) -> str | None:
    """Encode compact camera_base_remove_hand frames into their own cache."""
    cfg = apply_resolved_latent_cache_root_dir(dict(cfg))
    pre_cfg = dict(cfg.get("precompute", {}))
    output_path = resolve_remove_hand_output_path(cfg)
    force = bool(force) or bool(pre_cfg.get("overwrite", False))

    batch_size = max(1, int(pre_cfg.get("batch_size", 256)))
    token_mode = normalize_token_mode(pre_cfg.get("token_mode"), default="all")
    device = torch.device(
        str(pre_cfg.get("device", cfg_get(cfg, "runtime.device", "cuda")))
    )
    fm_cfg = dict(cfg["models"]["fm"])
    if not bool(fm_cfg.get("freeze_image_encoder", True)):
        raise ValueError("Precompute requires models.fm.freeze_image_encoder=true.")

    if dataset is None:
        dataset = build_dataset(cfg)
    if CAMERA_BASE_REMOVE_HAND_KEY not in dataset.data_group:
        print(
            f"[precompute] no data/{CAMERA_BASE_REMOVE_HAND_KEY} in "
            f"{dataset.zarr_path}; skipping remove-hand cache"
        )
        return None

    total_frames = int(dataset.data_group[CAMERA_BASE_REMOVE_HAND_KEY].shape[0])
    if total_frames == 0:
        print("[precompute] remove-hand array is empty; skipping remove-hand cache")
        return None

    model_name = resolve_dino_model_name(
        fm_cfg.get("image_encoder_name"),
        fm_cfg.get("dino_model_name"),
    )
    fm_cfg["dino_model_name"] = model_name
    if (not force) and remove_hand_frame_cache_matches(
        output_path,
        fm_cfg=fm_cfg,
        source_zarr_path=dataset.zarr_path,
        image_size=int(dataset.image_size),
        total_frames=total_frames,
        token_mode=token_mode,
        color_order="rgb",
    ):
        print(f"[precompute] remove-hand cache identity match, skipping: {output_path}")
        return output_path

    if os.path.isdir(output_path):
        print(f"[precompute] removing existing remove-hand cache: {output_path}")
        shutil.rmtree(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    image_encoder = _build_encoder(fm_cfg, device)
    out_root = zarr.open_group(output_path, mode="w")
    out_root.attrs["cache_version"] = int(FRAME_CACHE_VERSION)
    out_root.attrs["source_zarr_path"] = dataset.zarr_path
    out_root.attrs["image_size"] = int(dataset.image_size)
    out_root.attrs["color_order"] = "rgb"
    out_root.attrs["frame_image_selection"] = CAMERA_BASE_REMOVE_HAND_KEY
    out_root.attrs["compact"] = True
    out_root.attrs["ties_to"] = CAMERA_BASE_REMOVE_HAND_KEY
    out_root.attrs["base_remove_hand"] = "present"
    out_root.attrs["camera_views"] = "base_0"
    write_token_mode_attrs(out_root, token_mode)
    write_latent_cache_identity_attrs(out_root, fm_cfg)

    data_group = out_root.create_group("data")
    out_root.create_group("meta")
    chunk_size = max(1, min(batch_size, 64))
    frame_arr = None
    print(
        f"[precompute] encoding remove-hand frames: T={total_frames}, "
        f"model={model_name}, token_mode={token_mode}, batch_size={batch_size}, "
        f"device={device}, out={output_path}"
    )

    for start_idx in tqdm(
        range(0, total_frames, batch_size),
        desc="precompute:remove_hand_backbone_feat",
        unit="batch",
    ):
        frame_indices = list(
            range(start_idx, min(start_idx + batch_size, total_frames))
        )
        image_batch = build_remove_hand_image_batch(dataset, frame_indices).to(
            device, non_blocking=True
        )
        with torch.inference_mode():
            bsz, num_views = image_batch.shape[:2]
            flat = image_batch.reshape(bsz * num_views, *image_batch.shape[2:])
            tokens = image_encoder.extract_backbone_feat(flat)
            img = _tokens_to_stored_feat(
                tokens,
                batch_size=bsz,
                num_views=num_views,
                token_mode=token_mode,
            )

        if frame_arr is None:
            frame_arr = data_group.create_array(
                "frame_image_backbone_feat",
                shape=(total_frames,) + img.shape[1:],
                chunks=(chunk_size,) + img.shape[1:],
                dtype="f4",
            )
            out_root.attrs["image_backbone_dim"] = int(img.shape[-1])
            out_root.attrs["n_image_views"] = int(img.shape[1])
            out_root.attrs["image_num_tokens"] = token_mode_num_tokens(token_mode)
        frame_arr[start_idx : start_idx + len(frame_indices)] = img

    if frame_arr is None:
        raise RuntimeError("no remove-hand frames were encoded")
    print(
        f"[precompute] saved remove-hand cache: {output_path}, "
        f"shape={tuple(frame_arr.shape)}, token_mode={token_mode}"
    )
    return output_path


def precompute_all(cfg: dict, *, force: bool = False) -> dict[str, str | None]:
    dataset = build_dataset(cfg)
    frame_path = precompute_image_latents(cfg, force=force, dataset=dataset)
    remove_hand_path = precompute_base_remove_hand_latents(
        cfg, force=force, dataset=dataset
    )
    return {
        "frame_backbone": frame_path,
        "frame_backbone_base_remove_hand": remove_hand_path,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Precompute frame-only frozen DINOv2 backbone features (scheme A)."
    )
    parser.add_argument("--config", type=str, default="configs/train/config.yaml")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when an identity-matching frame cache already exists.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _FLOW_MATCHING_ROOT / config_path

    with open(config_path, encoding="utf-8") as handle:
        peek = yaml.safe_load(handle)
    if isinstance(peek, dict) and peek.get("finetune"):
        from lv_flow_matching.utils.finetune_config import resolve_full_config

        cfg = resolve_full_config(config_path, policy_root=_FLOW_MATCHING_ROOT)
    else:
        cfg = load_config(str(config_path))
    cfg = apply_resolved_latent_cache_root_dir(cfg)
    paths = precompute_all(cfg, force=bool(args.force))
    print(f"[precompute] done: {paths}")


if __name__ == "__main__":
    main()
