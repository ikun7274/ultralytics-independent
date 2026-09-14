"""Online-augmentation constants and memory-budget helpers (mirrored from the forked BaseDataset).

Single source of truth for the online-augmentation fallbacks. ``default.yaml`` remains the
authoritative user-facing default; this table exists so the fallbacks cannot drift between the
many ``getattr(self, <online key>, <literal>)`` call sites.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Single source of truth for the online-augmentation defaults used when the dataset was NOT assembled
# by ``v8_transforms`` (direct construction, tests, offline reuse).
_ONLINE_DEFAULTS: dict[str, Any] = {
    "slice_ratio": 1.0,
    "slice_prob": 0.0,
    "slice_overlap_ratio": 0.2,
    "slice_min_tile_area_ratio": 0.005,
    "slice_min_box_retain_ratio": 0.4,
    "slice_min_center_retain_ratio": 0.6,
    "slice_center_constraint": False,
    "slice_center_bias": False,
    "slice_bias_margin": 0.25,
    "slice_bias_jitter": 0.05,
    "slice_full_box_only": False,
    "slice_background_ratio": -1,
    "ratio_pad_ratio": 1.0,
    "blur_ratio": 1.0,
    "compose_ratio": 1.0,
    "weather_ratio": 0.5,
    "occlusion_ratio": 0.5,
    "slice_all_tiles": False,
    "slice_keep_origin": False,
    "ratio_pad_keep": False,
    "blur_keep": False,
    "compose_keep": False,
    "weather_keep": False,
    "occlusion_keep": False,
    "blur_short_len_min": 5,
    "blur_short_len_max": 12,
    "blur_long_len_min": 20,
    "blur_long_len_max": 35,
    "blur_long_defocus_sigma": 1.0,
    "blur_axis_aligned": True,
    "weather_types": "rain,haze,noise",
    "weather_rain_density": 0.15,
    "weather_rain_length": 15.0,
    "weather_haze_beta": 0.4,
    "weather_noise_std": 15.0,
    "occlusion_types": "rect,stripe",
    "occlusion_blocks": 1,
    "occlusion_size_ratio": 0.1,
    "occlusion_color": "auto",
    "occlusion_max_cover": 0.95,
    "ratio_pad_target": "auto",
    "ratio_pad_color": "black",
    "compose_max_side": 0,
    "degrade_max_side": 0,
    "degrade_resample": "linear",
    "slice_grouped_sampler": True,
    "slice_raw_cache_size": 2,
    "prefetch_factor": 2,
    "ims_cache_frames": 0,
    "ims_cache_mb": 1024,
    "slice_save_annotated": True,
    "slice_save_max": 0,
    "slice_save_exist_ok": True,
    "slice_save_max_tile": None,
    "slice_save_max_blur": None,
    "slice_save_max_ratio": None,
    "slice_save_max_compose": None,
    "slice_save_max_weather": None,
    "slice_save_max_occlusion": None,
    "slice_save_dir": None,
    "compose_save_dir": "",
    "ratio_pad_save_dir": "",
    "blur_save_dir": "",
    "weather_save_dir": "",
    "occlusion_save_dir": "",
    # mosaic annotated-save knobs (fork-only, read via _hyp_get; upstream has none of these)
    "mosaic_save_dir": "",
    "mosaic_save_max": 0,
    "mosaic_save_annotated": False,
    "mosaic_save_exist_ok": False,
    "task": 'detect',
    "mode": 'train',
    "epochs": 100,
    "patience": 100,
    "batch": 16,
    "imgsz": 640,
    "save": True,
    "save_period": -1,
    "cache": False,
    "workers": 8,
    "exist_ok": False,
    "pretrained": True,
    "cls_remap": True,
    "optimizer": 'auto',
    "verbose": True,
    "seed": 0,
    "deterministic": True,
    "single_cls": False,
    "rect": False,
    "cos_lr": False,
    "close_mosaic": 10,
    "resume": False,
    "resume_extend_epochs": 0,
    "amp": True,
    "fraction": 1.0,
    "profile": False,
    "multi_scale": 0.0,
    "compile": False,
    "overlap_mask": True,
    "mask_ratio": 4,
    "dropout": 0.0,
    "val": True,
    "split": 'val',
    "save_json": False,
    "iou": 0.7,
    "max_det": 300,
    "dnn": False,
    "plots": True,
    "vid_stride": 1,
    "stream_buffer": False,
    "visualize": False,
    "augment": False,
    "agnostic_nms": False,
    "retina_masks": False,
    "show": False,
    "save_frames": False,
    "save_txt": False,
    "save_conf": False,
    "save_crop": False,
    "show_labels": True,
    "show_conf": True,
    "show_boxes": True,
    "format": 'torchscript',
    "keras": False,
    "optimize": False,
    "dynamic": False,
    "simplify": True,
    "nms": False,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "dis": 6.0,
    "box": 7.5,
    "cls": 0.5,
    "cls_pw": 0.0,
    "dfl": 1.5,
    "pose": 12.0,
    "kobj": 1.0,
    "rle": 1.0,
    "angle": 1.0,
    "dlog": 1.0,
    "dgrad": 0.5,
    "dlam": 1.0,
    "nbs": 64,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "val_slice_enable": False,
    "val_slice_all_tiles": False,
    "val_slice_ratio": 1.0,
    "val_slice_overlap_ratio": 0.2,
    "val_slice_nms_iou": 0.5,
    "val_slice_dual_metric": False,
    "compose_save": False,
    "close_aug_epoch": 0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": 'flip',
    "auto_augment": 'randaugment',
    "erasing": 0.4,
    "tracker": 'tracktrack.yaml',
}


def _online_default(key: str) -> Any:
    """Return the package fallback for ``key``. Unknown keys (e.g. upstream mosaic_save_* knobs the
    forked code reads through _hyp_get but this package does not ship) return None, which the call
    sites' ``or ""`` / ``or 0`` / ``or False`` turn into a safe empty default."""
    return _ONLINE_DEFAULTS.get(key)


def get_split_fraction(fraction, split: str):
    """Return a split ratio/count, normalizing boundary values to 0.0 (none) or 1.0 (all).

    Mirrored from the forked ``ultralytics.data.utils`` (added upstream after 8.4.126).
    """
    if isinstance(fraction, list) and split in (splits := ("train", "val", "test")):
        index = splits.index(split)
        fraction = fraction[index] if index < len(fraction) else 1.0
    elif split != "train":
        fraction = 1.0
    fraction = float(fraction) if fraction in {0, 1} else fraction
    if split in {"train", "val"} and fraction == 0:
        raise ValueError(f"{split} fraction must select at least one image")
    return fraction


def _legacy_ims_cap(ni: int, batch_size: int) -> int:
    """Upstream's steady-state bound on ``self.ims``: ``min(ni, batch*8, 1000) - 1`` frames."""
    return max(1, min(ni, batch_size * 8, 1000) - 1)


def _ims_frame_bytes(imgsz: int | list[int], channels: int) -> int:
    """Bytes of ONE memoised ``self.ims`` frame (stored resized to ``imgsz``)."""
    side = int(max(imgsz)) if isinstance(imgsz, (list, tuple)) else int(imgsz)
    return max(1, side * side * int(channels))


def _ims_cap_for_budget(budget_mb: float, frame_bytes: int) -> int:
    """Frames of ``frame_bytes`` that fit in ``budget_mb`` MiB; ``0`` when there is no budget."""
    if budget_mb <= 0:
        return 0
    return max(1, int(budget_mb * (1 << 20)) // max(1, int(frame_bytes)))


def _resolve_ims_cap(hyp: Any, ni: int, batch_size: int, imgsz: int | list[int], channels: int, augment: bool) -> int:
    """Resolve the ``self.ims`` frame cap. ``0`` (default) = auto; ``> 0`` = exact frames; ``< 0`` = upstream formula."""
    if not augment:
        return 0
    frames = int(getattr(hyp, "ims_cache_frames", _online_default("ims_cache_frames")) or 0)
    if frames > 0:
        return frames
    legacy = _legacy_ims_cap(ni, batch_size)
    if frames < 0:
        return legacy
    budget_mb = float(getattr(hyp, "ims_cache_mb", _online_default("ims_cache_mb")) or 0)
    budgeted = _ims_cap_for_budget(budget_mb, _ims_frame_bytes(imgsz, channels))
    return min(legacy, budgeted) if budgeted else legacy


def _describe_ims_cap(hyp: Any, cap: int, imgsz: int | list[int], channels: int) -> str:
    """One-line, greppable report of the resolved ``self.ims`` budget, per worker."""
    frame = _ims_frame_bytes(imgsz, channels)
    frames = int(getattr(hyp, "ims_cache_frames", _online_default("ims_cache_frames")) or 0)
    if frames > 0:
        source = "explicit"
    elif frames < 0:
        source = "upstream formula"
    else:
        source = "auto=min(upstream, budget)"
    return (
        f"self.ims whole-image cache: {cap} frames x {frame / (1 << 20):.2f} MiB = "
        f"{cap * frame / (1 << 20):.0f} MiB/worker [ims_cache_frames={source}, imgsz={imgsz}, "
        f"channels={channels}]; tune ims_cache_frames / ims_cache_mb to trade re-decodes for memory"
    )
