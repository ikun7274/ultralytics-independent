"""Online-augmentation constants and memory-budget helpers (mirrored from the forked BaseDataset).

Single source of truth for the PROJECT-ONLY fallbacks. ``default.yaml`` remains the authoritative
user-facing default for every key upstream defines; this table exists so the fallbacks cannot drift
between the many ``getattr(self, <online key>, <literal>)`` call sites.

The table deliberately holds ONLY keys upstream does not have. It used to restate 96 upstream keys
(``optimizer`` / ``lr0`` / ``close_mosaic`` / ``patience`` / ...) verbatim. Those rows could never take
effect on ``DEFAULT_CFG`` -- ``install()`` only writes a key when ``not hasattr(DEFAULT_CFG, k)``, which
is always False for an upstream key -- yet they WERE live as the ``_online_default()`` fallback behind
~120 ``getattr(self, key, _online_default(key))`` sites, so any upstream version bump that changed one of
those defaults would have diverged silently through the fallback path.
``check_online_defaults_are_project_only`` is asserted by ``install()`` so the reverse drift (upstream
later adding one of our keys) surfaces immediately instead of silently.
"""

from __future__ import annotations

from typing import Any

# Single source of truth for the ONLINE-AUGMENTATION defaults used when the dataset was NOT assembled
# by ``v8_transforms`` (direct construction, tests, offline reuse). Project-only keys; upstream keys
# resolve through DEFAULT_CFG / DEFAULT_CFG_DICT.
_ONLINE_DEFAULTS: dict[str, Any] = {
    # --- online slicing (SAHI) ---
    # slice_prob is the slicing MASTER GATE: bool True/False is a supported spelling (== 1.0/0.0) and is
    # normalised by augment_setup._resolve_slice_prob, which also range-checks it and warns for 0<p<1
    # (a per-slot coin flip, not a strength knob -- use slice_ratio for that).
    "slice_prob": 0.0,
    "slice_ratio": 1.0,
    "slice_all_tiles": False,
    # slice_target_tiles: with slice_all_tiles=False the base segment holds ONE slot per selected item,
    # and this knob turns that item into a target-bearing (image, tile) pair instead of a blind random
    # tile. Each epoch takes the next block of K units from a fixed shuffled queue of every
    # target-bearing tile, so no unit repeats until the whole queue has been used, then a fresh pass
    # starts. See BaseDataset._target_tile_queue / _apply_target_tile_schedule.
    "slice_target_tiles": False,
    "slice_overlap_ratio": 0.2,
    # img_origin: unified "put EVERY original image into the pool as one whole-frame slot" coverage
    # knob. Replaces the old slice_keep_origin (which only covered the SLICED images). With it on, every
    # image appears at least once per epoch; with it off, an image no augmentation branch selected is
    # simply absent from the pool (the "discard the un-selected" behaviour the ratio-sized layout enables).
    #
    # DEFAULT TRUE, and that default carries the zero-intrusion contract: with the knob off and no other
    # online switch set, EVERY segment computes to 0, so len(dataset) == 0 -- which is not an empty but
    # harmless dataset. build_dataloader opens with `batch = min(batch, len(dataset))`, so a 0-length
    # train set rewrites its own batch size to 0 and torch raises "batch_size should be a positive
    # integer value, but got batch_size=0" before epoch 1. Defaulting ON keeps an all-switches-off
    # install inert (pool == N whole frames == the plain dataset), which is what lets the package sit on
    # a pristine Ultralytics. See tests/test_ooo_branches.py::test_no_switches_means_no_expansion_at_all.
    "img_origin": True,
    "slice_min_tile_area_ratio": 0.005,
    "slice_min_box_retain_ratio": 0.4,
    "slice_min_center_retain_ratio": 0.6,
    "slice_center_constraint": False,
    "slice_center_bias": False,
    "slice_bias_margin": 0.25,
    "slice_bias_jitter": 0.05,
    "slice_full_box_only": False,
    "slice_background_ratio": -1,
    # --- per-epoch branch ratios (mask = round(ratio * count) random positions) ---
    "ratio_pad_ratio": 1.0,
    "blur_ratio": 1.0,
    "compose_ratio": 1.0,
    "weather_ratio": 0.5,
    "occlusion_ratio": 0.5,
    # --- independent branch switches ---
    "ratio_pad_keep": False,
    "blur_keep": False,
    "compose_keep": False,
    "weather_keep": False,
    "occlusion_keep": False,
    "close_aug_epoch": 0,
    # --- aspect-ratio pad ---
    "ratio_pad_target": "auto",
    "ratio_pad_color": "black",
    # --- motion blur ---
    "blur_short_len_min": 5,
    "blur_short_len_max": 12,
    "blur_long_len_min": 20,
    "blur_long_len_max": 35,
    "blur_long_defocus_sigma": 1.0,
    "blur_axis_aligned": True,
    # --- weather degradation ---
    "weather_types": "rain,haze,noise",
    "weather_rain_density": 0.15,
    "weather_rain_length": 15.0,
    "weather_haze_beta": 0.4,
    "weather_noise_std": 15.0,
    # --- occlusion ---
    "occlusion_types": "rect,stripe",
    "occlusion_blocks": 1,
    "occlusion_size_ratio": 0.1,
    "occlusion_color": "auto",
    "occlusion_max_cover": 0.95,
    # --- compose + working-resolution caps ---
    "compose_max_side": 0,
    "degrade_max_side": 0,
    "degrade_resample": "linear",
    # Capacity of the DEGRADED-FRAME cache, in frames -- the "F5" fix. It memoises what
    # ``_cap_long_side`` PRODUCED, so the many reads one source image gets per epoch do not each
    # repeat the same downscale: the mixed pool reads one image 11-12 times per epoch (mosaic mixes 4
    # images per sample across 8 branch segments) and MEASURED 88% of the capped reads are repeats of
    # a key that was already computed. Per hit it saves the resample itself: 0.87 ms at cap=640,
    # 5.46 ms at cap=1280 (4K source).
    #   0  -> auto: 2 * _legacy_ims_cap(ni, batch) = 2 * (min(ni, batch*8, 1000) - 1).
    #         The factor 2 is because TWO cap values coexist in one run: the degradation branches cap
    #         at 2*imgsz and compose caps its sources at imgsz. The capacity deliberately tracks
    #         ``batch`` and NOT the dataset: the reuse comes from the mosaic mix pool, whose window
    #         upstream already bounds at ``_legacy_ims_cap``, and the measured FIFO hit-rate curve is
    #         flat in dataset size over W=16..128 (400 images vs 2000 images, work set 1.5 GB vs
    #         7.6 GB, same curve; they only diverge at W>=256, i.e. when the whole work set fits).
    #         W=auto already buys ~88% of the unbounded hit rate.
    #   >0  -> that many frames.
    #   <0  -> cache disabled. This is a pure A/B switch: the pipeline is byte-identical either way,
    #         so it exists precisely so the equivalence can be re-measured.
    # SCOPE: only a frame the cap actually PRODUCED is ever stored. When the long side is already
    # within the cap (or the cap is disabled) ``_cap_long_side`` hands its input straight back, so
    # there is no work to save -- and storing it would pin a raw-LRU frame in memory for nothing.
    "degrade_frame_cache_size": 0,
    # Byte budget for that cache, per worker (MiB; 0 = no byte limit, the frame count alone decides).
    # A capped frame is bounded by the CAP, not by the source: at imgsz=640 the branches cap at 1280
    # px, so a 16:9 frame is 2.76 MiB and the auto 126 frames are ~348 MiB/worker (@320 it is 84 MiB).
    # The budget is what protects a large imgsz -- imgsz=1280 caps at 2560 px, i.e. 19.7 MiB/frame,
    # where the frame count alone would be 2.5 GB/worker. It is enforced per insert and never evicts
    # the frame it is about to store, so it can only shrink the frame count, never empty the cache.
    "degrade_frame_cache_mb": 512,
    # --- sampling / caching budget ---
    "slice_grouped_sampler": True,
    # Per-worker raw-image LRU capacity, in FRAMES of the original-resolution image. Values in (0, 4)
    # are floored to 4 by the dataset, so 4 is the smallest usable cache; 0 disables the LRU entirely.
    #
    # The capacity has to cover the images that are actually re-read close together, and there are two
    # regimes, so 16 is chosen to cover BOTH:
    #   * grouped sampler (the default): a unit holds <= 4 images and is walked round-robin, so 4 frames
    #     already hold a whole unit;
    #   * plain shuffle (slice_grouped_sampler=False, or no image has 2+ slots): an image's slots are
    #     spread across the whole pool, and the only window that repeats is Mosaic's, i.e.
    #     ``max_buffer_length`` pool SLOTS. At the shipped batch sizes that is <= 127 slots, which at
    #     ~5 slots/image is ~13-25 distinct images.
    # MEASURED (240 images, 1280x720, slice_all_tiles=True + slice_ratio=1.0, real DataLoader, no
    # worker procs): cap 4 -> 20.4 items/s, cap 16 -> 41.0, cap 32 -> 44.3, i.e. +85% from 4 to 32, and
    # the gain is already ~80% of that by 16. Per-sample view (same data): cap 4 49.90 ms / 2.33 decodes
    # per sample, cap 16 32.07 ms / 1.38, cap 32 31.08 ms / 0.93 -- the curve saturates at 16-32.
    # The old default of 4 was set from a 24-image pool where the whole dataset fits in the Mosaic window
    # and capacity cannot matter, so it did not extrapolate to real dataset sizes.
    #
    # EFFECTIVE CAPACITY (important): the dataset caps whatever you set here at upstream's own
    # whole-image cache bound, ``min(ni, batch*8, 1000) - 1`` (constants._legacy_ims_cap). That is NOT a
    # throughput loss -- the bound is >= 17 for any dataset with more than 17 images or a batch of 3+,
    # i.e. it sits ABOVE this default (a 8520-image set at batch 16 gets 127) and only shrinks on the
    # tiny sets where every frame is resident anyway. The cap exists because the mosaic mix pool is fed
    # BY decode events (_touch_buffer_for_decode reproduces upstream's "append on decode" rule), so a
    # cache larger than upstream's suppresses those events and freezes the mosaic window. Measured with
    # every project switch at its default (8 images, batch 4, mosaic=1.0): epoch 0 was byte-identical to
    # pristine upstream but epoch 1 differed on 7/8 samples with max pixel delta 255. With the cap --
    # and with the cache evicting FIFO-by-first-decode instead of LRU, so its resident set matches
    # upstream's self.ims -- both epochs are byte-identical again.
    "slice_raw_cache_size": 16,
    # Byte budget for the same LRU, per worker (MiB; 0 = no byte limit, frames alone decide). The cache
    # stores ORIGINAL-resolution frames, whose size is not known before the first decode, so the frame
    # count above cannot be converted to memory up front. This budget is enforced on every insert: a
    # 1280x720 frame is 2.6 MiB, but a 4000x3000 one is 34 MiB, where 16 frames would be 550 MiB/worker.
    # With the default 256 MiB: 720p keeps all 16 frames, 4000x3000 keeps 7. 0 = trust
    # slice_raw_cache_size (only do that when you know the frame size).
    "slice_raw_cache_mb": 256,
    "ims_cache_frames": 0,
    "ims_cache_mb": 1024,
    # DataLoader prefetch depth, in BATCHES per worker. Stock build.py hardcodes 4 and offers no
    # knob; the grouped loader tail copied that literal, so this key exists to make it settable.
    #
    # Why it matters: the prefetched batches are COLLATED float32 tensors, so the cost is
    #     prefetch_factor x batch x channels x imgsz^2 x 4 B   per worker
    # = 4 x 8 x 3 x 640^2 x 4 = 157 MiB at batch=8/imgsz=640, on top of the ~340 MiB of fixed
    # per-worker overhead (spawn re-imports torch + cv2 and unpickles the dataset). Measured on a
    # 7.9 GB box with ~0.4 GB free, that is what turns workers=4 into a DataLoader-worker
    # MemoryError: 4 x (340 + 157) = ~2.0 GiB of workers alone, plus 240 MiB/worker of raw LRU.
    # 4 is the stock value, so leaving the key alone reproduces upstream behaviour exactly.
    "prefetch_factor": 4,
    # --- annotated-save knobs ---
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
    "compose_save": False,
    "compose_save_dir": "",
    "ratio_pad_save_dir": "",
    "blur_save_dir": "",
    "weather_save_dir": "",
    "occlusion_save_dir": "",
    # mosaic annotated-save knobs: NO-OP on a pristine Ultralytics. The stock ``Mosaic`` is
    # ``Mosaic(dataset, imgsz, p, n)`` -- it takes no save argument -- so ``_compat`` in
    # ``augment_setup`` strips all four at construction. They are kept registered ONLY so an existing
    # args.yaml/checkpoint keeps loading instead of raising "not a valid YOLO argument"; setting one
    # away from its default now warns once (see ``_warn_mosaic_save_is_a_noop``). Use the per-branch
    # ``*_save_dir`` knobs below instead.
    "mosaic_save_dir": "",
    "mosaic_save_max": 0,
    "mosaic_save_annotated": False,
    "mosaic_save_exist_ok": False,
    # --- resume-extension ---
    "resume_extend_epochs": 0,
    # --- validation-side slicing (SAHI eval) ---
    "val_slice_enable": False,
    "val_slice_all_tiles": False,
    "val_slice_ratio": 1.0,
    "val_slice_overlap_ratio": 0.2,
    "val_slice_nms_iou": 0.5,
    "val_slice_dual_metric": False,
}


def _online_default(key: str) -> Any:
    """Return the package fallback for ``key``; unknown keys return None, which the call sites'
    ``or ""`` / ``or 0`` / ``or False`` turn into a safe empty default.

    Upstream keys are NOT meant to be looked up here -- ``_hyp_get`` already prefers
    ``DEFAULT_CFG_DICT`` and only lands here for keys upstream does not define.
    """
    return _ONLINE_DEFAULTS.get(key)


def check_online_defaults_are_project_only(upstream_keys=None) -> list[str]:
    """Return the config-table keys upstream ALSO defines (should always be an empty list).

    The invariant is the reverse of the one the old table relied on: the table must contain no upstream
    key at all. A key present in both places is skipped by ``install()``'s ``if not hasattr(DEFAULT_CFG,
    k)`` guard -- so the package value is never applied -- while remaining live as the
    ``_online_default()`` fallback, i.e. a silent divergence from upstream defaults waiting for a
    version bump.

    Args:
        upstream_keys: the pristine upstream key set. ``install()`` MUST pass the snapshot it took
            BEFORE registering our keys, because it writes them into ``DEFAULT_CFG_DICT`` itself --
            comparing against the live dict afterwards reports every project key as a duplicate. Pass
            an explicit set (e.g. parsed from ``cfg/default.yaml``) for an install-order-independent check.
    """
    if upstream_keys is None:
        from ultralytics.utils import DEFAULT_CFG_DICT

        upstream_keys = set(DEFAULT_CFG_DICT)
    return sorted(k for k in _ONLINE_DEFAULTS if k in upstream_keys)


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
