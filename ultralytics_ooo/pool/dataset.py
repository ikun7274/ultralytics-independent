# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""OnlinePoolDataset: mixed virtual-sample pool as a subclass of a stock Ultralytics BaseDataset.

Mirrors the forked BaseDataset extensions verbatim. Installed onto a pristine Ultralytics by
``ultralytics_ooo.installer``; on its own it needs nothing from the forked tree except the kernels
already in ``ultralytics_ooo.core`` / ``.pool``.
"""

from __future__ import annotations

import bisect
import math
import multiprocessing
import os
import random
import time
import zlib
from collections import OrderedDict
from collections import deque
from copy import deepcopy
from typing import Any, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.data.base import BaseDataset
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.patches import imread

from ultralytics_ooo.core import (
    _RATIO_PAD_COLORS,
    _apply_motion_blur,
    _apply_occlusion,
    _apply_weather,
    _ratio_pad_params,
    _union_area,
    _cap_long_side,
)
from ultralytics_ooo.core.saver import _ensure_dir, _imwrite, _save_cap
from ultralytics_ooo.pool.constants import (
    _online_default,
    _resolve_ims_cap,
    _describe_ims_cap,
    get_split_fraction,
)
from ultralytics_ooo.pool.sampler import SegmentBases


class OnlinePoolDataset(BaseDataset):
    """BaseDataset subclass that adds the mixed virtual-sample pool.

    The original BaseDataset from a stock Ultralytics is inherited unchanged; every method below is
    the mirrored extension. ``installer`` swaps the trainer's dataset factory to this class.
    """

    def __init__(self, *args, **kwargs):
        # Cooperative multiple-inheritance: forward every arg (img_path, hyp, data, task, ...) up the
        # MRO to YOLODataset -> BaseDataset, then run the extension initialisation.
        super().__init__(*args, **kwargs)
        hyp = kwargs.get("hyp", DEFAULT_CFG)
        fraction = kwargs.get("fraction", 1.0)
        # --- extension initialisation (mirrored from the forked BaseDataset.__init__) ---
        self.fraction = get_split_fraction(fraction, "train")
        # Mosaic buffer: a deque with an explicit maxlen gives O(1) eviction.
        self.buffer = deque(maxlen=max(1, self.max_buffer_length - 1)) if self.augment else []
        self._ims_keys: dict[int, None] = {}
        self._ims_cap = _resolve_ims_cap(hyp, self.ni, self.batch_size, self.imgsz, self.channels, self.augment)
        if self._ims_cap > 0:
            LOGGER.info(f"{self.prefix}{_describe_ims_cap(hyp, self._ims_cap, self.imgsz, self.channels)}")
        self._raw_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        _raw_cache_size = int(getattr(hyp, "slice_raw_cache_size", _online_default("slice_raw_cache_size")) or 0)
        self._raw_cache_size = max(4, _raw_cache_size) if _raw_cache_size > 0 else 0
        self._slice_mask = None
        self._ratio_mask = None
        self._blur_mask = None
        self._weather_mask = None
        self._occlusion_mask = None
        self._compose_mask = None
        try:
            self._mp_epoch = multiprocessing.Value("i", -1)
            self._mp_epochs = multiprocessing.Value("i", -1)
        except Exception:
            self._mp_epoch = None
            self._mp_epochs = None
        self._mask_stamp = -1
        self._sync_every = 16
        self._sync_tick = 0
        self._mask_seed = zlib.crc32("\n".join(self.im_files).encode("utf-8", "ignore"))
        self._seg_cache: "SegmentBases | None" = None
        self._seg_key: tuple[int, ...] | None = None
        self._compose_warned = False
        self._save_state: dict[str, list] = {}
        self.prefetch_factor = max(1, int(getattr(hyp, "prefetch_factor", _online_default("prefetch_factor")) or 2))
        self.slice_grouped_sampler = bool(
            getattr(hyp, "slice_grouped_sampler", _online_default("slice_grouped_sampler"))
        )
        self._raw_hits = 0
        self._raw_misses = 0
        try:
            self._mp_raw_hits = multiprocessing.Value("q", 0)
            self._mp_raw_misses = multiprocessing.Value("q", 0)
        except Exception:
            self._mp_raw_hits = None
            self._mp_raw_misses = None
        self._raw_reported = (0, 0)
        # cache='ram' cannot serve the online branches (they read ORIGINAL-resolution frames).
        if self.cache == "ram" and float(getattr(hyp, "slice_prob", 0.0) or 0.0) > 0.0:
            LOGGER.warning(
                f"{self.prefix}cache='ram' cannot accelerate online slicing: degrading to cache=False. "
                "Raise slice_raw_cache_size to speed up decoding instead."
            )
            self.cache = None
        # Report the expanded sample count (train only).
        if self.augment:
            _n_total = len(self)
            _n_origin = self.ni
            _parts = []
            if getattr(self, "slice_transform", None) is not None:
                _parts.append("4 slices" if bool(getattr(self, "slice_all_tiles", _online_default("slice_all_tiles"))) else "random 1 slice")
            if self._keep_origin_on():
                _parts.append("1 origin")
            if bool(getattr(self, "ratio_pad_keep", _online_default("ratio_pad_keep"))):
                _parts.append("1 ratio")
            if bool(getattr(self, "blur_keep", _online_default("blur_keep"))):
                _parts.append("2 blur")
            if self._compose_on():
                _parts.append("N/4 compose")
            if self._weather_on():
                _parts.append("1 weather")
            if self._occlusion_on():
                _parts.append("1 occlusion")
            if _parts:
                LOGGER.info(
                    f"{self.prefix}Online augment: {_n_total} training samples from {_n_origin} images "
                    f"({' + '.join(_parts)} per image)"
                )


    def load_image(
        self, i: int, rect_mode: bool = True, resize_short: bool = False
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load an image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool): Whether to use rectangular resizing (long side to imgsz).
            resize_short (bool): Whether to resize the shorter side to imgsz while maintaining aspect ratio. Overrides
                rect_mode when True.

        Returns:
            im (np.ndarray): Loaded image as a NumPy array.
            hw_original (tuple[int, int]): Original image dimensions in (height, width) format.
            hw_resized (tuple[int, int]): Resized image dimensions in (height, width) format.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                    npy_channels = im.shape[-1] if im.ndim >= 3 else 1
                    if npy_channels != self.channels:
                        LOGGER.warning(
                            f"{self.prefix}Removing stale *.npy image file {fn} with {npy_channels} channels, expected {self.channels}"
                        )
                        Path(fn).unlink(missing_ok=True)
                        im = imread(f, flags=self.cv2_flag)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)  # BGR
            else:  # read image
                im = imread(f, flags=self.cv2_flag)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                if resize_short:  # resize short side to imgsz while maintaining aspect ratio
                    r = self.imgsz / min(h0, w0)  # ratio
                    if r != 1:  # if sizes are not equal
                        w, h = (math.ceil(w0 * r), self.imgsz) if h0 < w0 else (self.imgsz, math.ceil(h0 * r))
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    r = self.imgsz / max(h0, w0)  # ratio
                    if r != 1:  # if sizes are not equal
                        w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]

            # Add to buffer if training with augmentations
            if self.augment and self.cache != "ram":
                if getattr(self, "slice_transform", None) is None:
                    # Without slicing, load_image's index is the dataset index, so the ims cache and
                    # the mosaic buffer are both managed here. The ims entry is ALWAYS bounded by
                    # _remember_ims: without that bound this branch grew to one frame per image in
                    # the dataset (~29 GB/worker for 8520 images at imgsz=1280) whenever the
                    # extended pool was on, because the only eviction used to sit behind the
                    # `not self._extended_pool_on` guard below.
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                    self._remember_ims(i)
                    if not self._extended_pool_on():
                        # Pure-ultralytics mode (no slicing, no project extension): the buffer can only
                        # contain original-image indices, so self-managed append is safe -- the deque
                        # maxlen evicts the oldest entry in O(1).
                        self.buffer.append(i)
                    # Extended pool on: the buffer is bookkept centrally by get_image_and_label with
                    # EXPANDED indices; load_image must not feed it original indices, or the buffer
                    # would mix two index spaces -- e.g. slicing off + compose/blur/ratio/keep_origin on.
                # With slicing: do NOT cache in self.ims here. get_image_and_label centrally manages the buffer
                # with expanded indices; caching origin-indexed images here would leak memory (never released).

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def set_epoch(self, epoch: int = 0, epochs: int | None = None) -> None:
        """Publish the epoch and rebuild this process's per-epoch masks (trainer / main-process entry).

        Called by ``trainer.py`` at each epoch start. The epoch (and the total ``epochs``, needed by
        ``close_aug_epoch``) is published through shared memory so ALREADY-RUNNING DataLoader workers
        can pick it up -- see ``_sync_epoch_masks``. This process's masks are rebuilt immediately
        (the workers=0 / direct-iteration paths read them right here).

        Mask semantics per branch (unchanged): exactly ``round(x * N)`` ORIGINAL images are randomly
        chosen each epoch (``round(x * ceil(N/4))`` groups for compose); un-selected slots keep their
        segment position but fall back to the ORIGINAL full image (``len`` constant). Mask = None
        (all augmented / branch off) when the branch is off or its ratio >= 1. ``close_aug_epoch``:
        during the final N epochs all masks become all-False so every segment falls back to the
        original image.
        """
        if self._mp_epoch is not None:
            try:
                self._mp_epoch.value = int(epoch)
                self._mp_epochs.value = int(epochs) if epochs is not None else -1
            except Exception as e:  # e.g. called inside a worker process; workers rebuild via _sync instead
                LOGGER.debug(f"set_epoch: shared-memory epoch update skipped in this process: {e}")
        self._log_raw_cache_stats()
        self._rebuild_epoch_masks(epoch, epochs)

    def _rebuild_epoch_masks(self, epoch: int, epochs: int | None = None) -> None:
        """(Re)build the per-epoch masks and reset the OnlineSlice counters for ``epoch``.

        Deterministic per (dataset, epoch, epochs): masks are drawn from an RNG seeded ONLY by
        ``(_mask_seed, epoch, epochs)`` -- NOT from the global ``random`` stream -- so the main
        process and every DataLoader worker rebuilding the same epoch derive IDENTICAL masks with no
        cross-process mask transport . A worker that missed epochs and rebuilds late therefore
        still produces exactly the masks of the current epoch. ``_mask_stamp`` is bumped first: the
        rebuild is idempotent for a given (epoch, epochs) pair.
        """
        self._mask_stamp = int(epoch)
        rng = random.Random(f"{self._mask_seed}:{int(epoch)}:{-1 if epochs is None else int(epochs)}")
        # Reset OnlineSlice's positive/background counters every epoch so the neg_ratio background
        # quota restarts per epoch instead of accumulating monotonically. The reset must NOT assume
        # it "propagates to workers via fork inheritance" -- it does not: InfiniteDataLoader
        # forks/spawns workers once at construction, BEFORE any set_epoch. The reset therefore runs
        # in EVERY process that rebuilds (main via set_epoch, each worker via _sync_epoch_masks).
        st = getattr(self, "slice_transform", None)
        if st is not None and hasattr(st, "reset_counters"):
            st.reset_counters()
            # Pin the current epoch for the deterministic seam jitter. Must live in this function
            # (not in set_epoch's main-process path) for the same reason the counters do: workers rebuild
            # via _sync_epoch_masks and would otherwise keep epoch 0 forever, so the 4 tiles of one image
            # would share one grid in the main process and a different one in every worker.
            if hasattr(st, "set_epoch"):
                st.set_epoch(epoch)
        n = len(self.labels)
        aug_on = bool(self.augment)
        # --- close_aug_epoch: final N epochs disable all online augmentation (fallback to origin) ---
        close_epoch = int(getattr(self, "close_aug_epoch", 0))
        if aug_on and close_epoch > 0 and epochs is not None and epoch >= epochs - close_epoch:
            self._slice_mask = np.zeros(n, dtype=bool)
            self._ratio_mask = np.zeros(n, dtype=bool)
            self._blur_mask = np.zeros(n, dtype=bool)
            self._weather_mask = np.zeros(n, dtype=bool)
            self._occlusion_mask = np.zeros(n, dtype=bool)
            self._compose_mask = np.zeros((n + 3) // 4, dtype=bool)
            return
        # --- 表驱动 : 六条增强分支共享同构的"掩码 = round(x*count) 个随机位"逻辑。
        # 差异点只有: 开关谓词 / 比例属性 / 样本数 (compose 是组级 (n+3)//4, 其余原图级 n)。
        # 未选中位保留区段位置但回退原图 (len 恒定); 分支关闭或比例 >=1 时掩码为 None (全增强)。
        for attr, ratio_attr, on_fn, count in self._mask_specs(n):
            x = float(getattr(self, ratio_attr, _online_default(ratio_attr)))
            mask = None
            if aug_on and on_fn() and 0.0 <= x < 1.0:
                mask = np.zeros(count, dtype=bool)
                if x > 0:
                    mask[rng.sample(range(count), int(round(x * count)))] = True
            setattr(self, f"_{attr}_mask", mask)

    def _mask_specs(self, n: int) -> list[tuple[str, str, Callable[[], bool], int]]:
        """Table-driven per-epoch mask construction (refactor).

        Each row: (mask attr suffix, ratio attr, branch-on predicate, sample count). compose is
        GROUP-level -- ``ceil(N/4)`` groups -- while every other branch is original-level (``N``).
        Ratios fall back to ``_ONLINE_DEFAULTS`` only when the dataset was built without
        ``v8_transforms`` (training always copies them from default.yaml).
        """
        return [
            ("slice", "slice_ratio", lambda: getattr(self, "slice_transform", None) is not None, n),
            ("ratio", "ratio_pad_ratio", lambda: bool(getattr(self, "ratio_pad_keep", _online_default("ratio_pad_keep"))), n),
            ("blur", "blur_ratio", lambda: bool(getattr(self, "blur_keep", _online_default("blur_keep"))), n),
            ("compose", "compose_ratio", self._compose_on, (n + 3) // 4),
            ("weather", "weather_ratio", self._weather_on, n),
            ("occlusion", "occlusion_ratio", self._occlusion_on, n),
        ]

    def _sync_epoch_masks(self) -> None:
        """Rebuild this process's masks when the trainer published a NEW epoch through shared memory.

        This is what makes ``set_epoch`` effective inside DataLoader workers at all (fix):
        InfiniteDataLoader spawns workers once at loader construction -- before any set_epoch -- and
        reuses them for the whole run, so they can never observe the trainer's main-process calls.
        Each worker polls the shared epoch periodically and rebuilds only on change; the rebuild is
        deterministic per epoch, so the worker's masks always match the main process's. Same-process
        callers (workers=0 / direct iteration) pay the int read and never rebuild here because
        ``set_epoch`` already rebuilt them.

        The poll runs once every ``_sync_every`` samples rather than on every sample: the read takes
        a lock on a ``multiprocessing.Value``, which is not free at 8 workers x tens of thousands of
        samples per epoch, while an epoch boundary only concerns the handful of samples a worker
        draws right after it. At most ``_sync_every - 1`` samples therefore use the previous epoch's
        masks, and this process's masks would otherwise already be stale by up to ``num_workers``.
        """
        v = self._mp_epoch
        self._sync_tick += 1
        if self._sync_tick % self._sync_every:
            return
        # Same amortization as the epoch poll below: the raw-LRU counters are worker-local, so they can
        # only be observed if they are flushed, but a locked add per read would be worse than the
        # misses it reports.
        self._publish_raw_cache_stats()
        if v is None or not self.augment:
            return
        try:
            epoch = v.value
        except Exception:
            return
        if epoch < 0 or epoch == self._mask_stamp:
            return
        ev = self._mp_epochs
        try:
            epochs = ev.value if ev is not None else -1
        except Exception:
            epochs = -1
        self._rebuild_epoch_masks(epoch, epochs if epochs >= 0 else None)

    def _n_per(self) -> int:
        """Single source of truth: how many samples each ORIGINAL image expands to in the BASE segment.

        The base segment contains ONLY the slicing pipeline output (slice_prob / slice_all_tiles);
        the un-sliced origin copy is now an INDEPENDENT segment (see _segment_bases), laid out after
        the base segment just like ratio / blur / compose:
            slicing off       -> 1  (no expansion, plain originals)
            slice_all_tiles   -> 4  (the 4 sliced tiles)

        EVERY site that needs this number (_segment_bases / get_image_and_label / __len__)
        must call this instead of re-deriving it inline. The formula used to be copy-pasted in 4
        places; adding a new online branch and missing one of them desynchronises __len__ from the
        decodable index range and drops samples with no error at all.
        """
        if not (bool(getattr(self, "slice_all_tiles", _online_default("slice_all_tiles"))) and getattr(self, "slice_transform", None) is not None):
            return 1
        return 4

    def _keep_origin_on(self) -> bool:
        """True when the un-sliced ORIGIN segment allocates samples (slice_keep_origin independent switch).

        ``slice_keep_origin`` is now an independent switch -- it is no longer forced off when
        ``slice_prob == 0`` in ``augment.py``, and it no longer changes ``_n_per``. It only makes
        sense together with slicing though: without slicing the originals are already the whole
        sample pool, so keeping them again would duplicate every image. Auto-suppressed here when
        the slicing pipeline is off.
        """
        return (
            bool(getattr(self, "slice_keep_origin", _online_default("slice_keep_origin")))
            and bool(getattr(self, "slice_all_tiles", _online_default("slice_all_tiles")))
            and getattr(self, "slice_transform", None) is not None
        )

    def _touch_buffer(self, index: int) -> None:
        """Record ``index`` in the mosaic-buffer FIFO, evicting the oldest entry past capacity.

        Single home for the buffer bookkeeping that used to be copy-pasted across every pool
        branch: the capacity rule and the eviction now change in one place. ``self.buffer`` is a
        ``deque(maxlen=...)``, so the append itself evicts the oldest entry in O(1) -- the old
        explicit ``pop(0)`` on a list was O(n).
        """
        if self.augment and self.cache != "ram":
            self.buffer.append(index)

    def _remember_ims(self, i: int) -> None:
        """Record original index ``i`` in the ``self.ims`` FIFO, evicting the oldest entries.

        ``self.ims`` is keyed by ORIGINAL index, but was historically evicted by the mosaic buffer,
        which only holds original indices in pure-ultralytics mode. With the extended pool on the
        buffer holds EXPANDED indices, so it could no longer release anything and the cache grew
        without bound -- one imgsz-sized frame per image in the dataset, i.e. ~29 GB per worker for
        8520 images at imgsz=1280, with no error and no warning. This FIFO restores the legacy bound
        (``_ims_cap``) in every mode.

        Only ever called for a freshly decoded frame (``cache == "ram"`` is excluded by the caller),
        so a key that is already resident is left in place instead of being re-ordered: the set of
        resident keys is what matters here, not strict LRU order.
        """
        if self._ims_cap <= 0 or i in self._ims_keys:
            return
        self._ims_keys[i] = None
        while len(self._ims_keys) > self._ims_cap:
            j = next(iter(self._ims_keys))
            del self._ims_keys[j]
            self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

    def _weather_on(self) -> bool:
        """True when the weather branch allocates samples (weather_keep independent switch)."""
        return bool(getattr(self, "weather_keep", _online_default("weather_keep")))

    def _occlusion_on(self) -> bool:
        """True when the occlusion branch allocates samples (occlusion_keep independent switch)."""
        return bool(getattr(self, "occlusion_keep", _online_default("occlusion_keep")))

    def _save_annotated(self, branch: str, branch_dir, img: np.ndarray, boxes, cls, file_stem: str,
                        key: tuple, save_annotated: bool) -> None:
        """Shared annotated-save block for the online branches.

        Draws green boxes + class labels when ``save_annotated`` and boxes exist, then writes
        ``<branch_dir>/<file_stem>_{seq:05d}_n{len(boxes)}.jpg``. Deduplicated per ``key`` across epochs
        and capped by the per-branch save cap (``_save_cap``). The counter advances only when the
        write actually succeeds.

        Counter state lives in ``self._save_state[branch]`` -- an explicit dict rather than
        ``getattr(self, f"_{branch}_saved")``, which silently created a brand-new counter (and thus
        a broken cap) on a misspelled ``branch`` instead of failing.
        """
        if branch not in _SAVE_BRANCHES:
            raise ValueError(f"_save_annotated: unknown branch {branch!r}; expected one of {sorted(_SAVE_BRANCHES)}.")
        state = self._save_state.setdefault(branch, [0, set()])  # [n_saved, dedup keys]
        cnt, keys = state
        save_cap = _save_cap(self, branch)
        if key in keys or (save_cap != 0 and cnt >= save_cap):
            return
        out = img
        boxes_arr = np.asarray(boxes, dtype=np.float64)
        if save_annotated and len(boxes_arr):
            out = img.copy()
            H2, W2 = img.shape[:2]
            cls_arr = np.asarray(cls).reshape(-1)
            if len(boxes_arr) != len(cls_arr):
                LOGGER.warning(
                    f"_save_annotated({branch}): {len(boxes_arr)} boxes vs {len(cls_arr)} cls for "
                    f"'{file_stem}' -- drawing only the aligned prefix."
                )
            for b, c in zip(boxes_arr, cls_arr):
                cx, cy, bw, bh = (float(v) for v in b)
                x0 = int(round((cx - bw / 2) * W2))
                y0 = int(round((cy - bh / 2) * H2))
                x1 = int(round((cx + bw / 2) * W2))
                y1 = int(round((cy + bh / 2) * H2))
                cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(out, f"cls{int(c)}", (x0, max(0, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        _ensure_dir(branch_dir)
        if _imwrite(branch_dir / f"{file_stem}_{cnt:05d}_n{len(boxes_arr)}.jpg", out):
            state[0] = cnt + 1
            keys.add(key)

    def _finalize_label(self, label: dict[str, Any], img: np.ndarray) -> dict[str, Any]:
        """Resize ``img`` to the training size and attach ori_shape / resized_shape / ratio_pad.

        Shared tail of every online-augmentation branch (blur / weather / occlusion / ratio /
        compose). Kept in one place so resizing semantics cannot drift between branches .
        """
        h1, w1 = img.shape[:2]
        r = self.imgsz / max(h1, w1)
        if r != 1:
            # clamp to imgsz exactly like load_image does -- float error could otherwise
            # produce resized_shape == imgsz+1 and diverge from the vanilla-path semantics.
            img = cv2.resize(img, (min(math.ceil(w1 * r), self.imgsz), min(math.ceil(h1 * r), self.imgsz)),
                             interpolation=cv2.INTER_LINEAR)
        if img.ndim == 2:
            img = img[..., None]
        label["img"] = np.ascontiguousarray(img)
        label["ori_shape"] = (h1, w1)
        label["resized_shape"] = img.shape[:2]
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )
        return self.update_labels_info(label)

    def _segment_lengths(self) -> list[int]:
        """Per-segment sample counts in layout order (base first). SINGLE SOURCE OF TRUTH.

        Layout (each optional branch is an independent, contiguous segment gated ONLY by its own
        switch; slicing lives entirely inside the base segment):
            [0, base_len)                 base:      _n_per samples per original (slicing pipeline)
            [base_len, +N)                origin:    1 un-sliced original per image (keep_origin)
            [base_len+N, +2N)             ratio:     1 aspect-ratio-padded image per original
            [base_len+2N, +4N)            blur:      short + long motion-blurred images per original
            [base_len+4N, +ceil(N/4))     compose:   one 2x2 stitched image per group of 4 originals
            [base_len+4N+ceil(N/4), +N)   weather:   1 rain/haze/noise-degraded image per original
            [base_len+4N+ceil(N/4)+N, +N) occlusion: 1 rect/stripe-occluded image per original

        ``_segment_bases`` derives BOTH the cumulative boundaries and ``total`` from this list, and
        uses ``tuple(...)`` of it as the ``_seg_cache`` version stamp. Every input of the layout is
        therefore read exactly here -- adding a segment is a one-line change that invalidates the
        cache automatically, with no second switch list to keep in sync (see ``_segment_bases``).
        """
        n = len(self.labels)
        return [
            n * self._n_per(),  # base segment: the slicing pipeline's samples
            n if self._keep_origin_on() else 0,  # origin
            n if bool(getattr(self, "ratio_pad_keep", _online_default("ratio_pad_keep"))) else 0,  # ratio
            2 * n if bool(getattr(self, "blur_keep", _online_default("blur_keep"))) else 0,  # blur (short + long)
            (n + 3) // 4 if self._compose_on() else 0,  # compose (groups of 4)
            n if self._weather_on() else 0,  # weather
            n if self._occlusion_on() else 0,  # occlusion
        ]

    def _segment_bases(self) -> SegmentBases:
        """Return the seven segment boundaries of the mixed sample pool, plus the pool total.

        Returns a :class:`SegmentBases` named tuple (base, origin, ratio, blur, compose, weather,
        occlusion, total). ``total`` is derived from the SAME per-segment lengths as the
        boundaries, and ``__len__`` returns it verbatim -- the pool length and the decodable
        index range therefore share one source of truth and can never drift apart.

        The result is cached, but the cache is VERSION-STAMPED on ``tuple(self._segment_lengths())``
        -- its own input -- instead of "first call wins".

        Why that matters: the branch switches are NOT settled by ``__init__``. The normal path is
        ``__init__`` -> ``build_transforms`` -> ``v8_transforms`` attaching ``slice_transform`` /
        ``slice_all_tiles`` / ``*_keep`` onto the object afterwards, and this module already relies
        on that order (``len(self)`` is only logged after ``build_transforms``). A first-call-wins
        cache made that order a silent correctness requirement: anything that touched ``len()``
        earlier -- a partially configured dataset, a test/script that flips a switch after
        constructing one, or a future edit that logs ``len()`` mid-``__init__`` -- froze the
        boundaries on the OLD switches, after which the pool length disagreed with the decodable
        index range and samples were dropped / indexed out of range with no error at all.

        Cost (measured, 5-original all-branches pool): 2.5 us/call vs 0.1 us for the first-call-wins
        accessor and 3.6 us for no cache at all -- i.e. it hands back about two thirds of the L8
        change. That is deliberate and cheap in absolute terms: at 8520 images/epoch the pool is
        ~85k samples, so the stamp costs ~0.2 s CPU per epoch (~0.01% of an epoch that decodes every
        sample at original resolution) and it retires a whole silent-failure class. Keep the stamp
        DERIVED from ``_segment_lengths``; never hand-maintain a second list of switches here -- that
        would move the drift, not remove it.
        """
        seg_lens = self._segment_lengths()
        key = tuple(seg_lens)
        if getattr(self, "_seg_cache", None) is None or getattr(self, "_seg_key", None) != key:
            # Cumulative starts: bases[i] is the START of segment i, bases[-1] the pool total.
            bases = [seg_lens[0]]
            for seg in seg_lens[1:]:
                bases.append(bases[-1] + seg)
            *boundaries, total = bases
            self._seg_cache = SegmentBases(seg_lens[0], *boundaries, total)
            self._seg_key = key
        return self._seg_cache

    def grouped_sample_units(self) -> list[list[list[int]]] | None:
        """Lay the sample pool out as "units of <= 4 source images" for decode-locality sampling.

        The pool ALREADY clusters sub-samples of one image in the index dimension (``index // n_per``
        in the base segment, contiguous runs in the other segments), but the trainer's global shuffle
        destroys that adjacency: with N images the siblings of a sub-sample end up ~``6.5 * N``
        samples away, far beyond the raw LRU's capacity of 4, so every sub-sample re-decodes its
        original JPEG. This method rebuilds the adjacency in the ORDER dimension: it returns units
        where ``units[u][j]`` is the list of pool indices that decode one source image, so a sampler
        walking a unit round-robin keeps all of its source images resident.

        Layout rules -- these MUST mirror ``_segment_bases`` / ``get_image_and_label`` (the returned
        layout is validated against ``total`` below, and the guard turns a mismatch into a warning
        instead of silently dropping samples):
            base      ``n_per`` consecutive indices per image
            origin    one index per image
            ratio     one index per image
            blur      two indices per image (short, long)
            weather   one index per image
            occlusion one index per image
            compose   ONE index per GROUP of four images; it reads all four, so it is emitted first
                      inside its unit and thereby primes the LRU for the whole unit

        Units are consecutive blocks of four images, which is also the compose group size, so a compose
        sample never straddles a unit boundary and every image owns exactly one block.

        Returns ``None`` when grouping cannot pay off, in which case the caller keeps the plain
        shuffle:
            * the dataset is not augmenting, or the raw LRU is off -- there is nothing to reuse;
            * ``n_per == 1`` and no extended segment -- every image maps to exactly ONE pool index, so
              there is no repeated decode to absorb, and a block-ordered stream would only narrow
              Mosaic's recent-sample window for no gain.

        Grouping is a pure reordering: the multiset of indices is unchanged, which the final check
        enforces.
        """
        if not self.augment or self._raw_cache_size <= 0:
            return None
        n = len(self.labels)
        n_per = self._n_per()
        if n < 1 or (n_per <= 1 and not self._extended_pool_on()):
            return None
        segment_bases = self._segment_bases()
        # Only worth it when an original is re-read at least as often as the LRU capacity: below that
        # the saved decodes do not pay for narrowing Mosaic's recent-sample window (e.g. ratio_pad_keep
        # alone = 2 pool samples per image, where the 127-deep ``ims`` cache already absorbs the reuse).
        if segment_bases.total < 4 * n:
            return None
        per_image: list[list[int]] = [[] for _ in range(n)]
        for index in range(segment_bases.origin):  # base segment: [0, base_len) == n * n_per
            per_image[index // n_per].append(index)
        if self._keep_origin_on():
            for i in range(n):
                per_image[i].append(segment_bases.origin + i)
        if bool(getattr(self, "ratio_pad_keep", _online_default("ratio_pad_keep"))):
            for i in range(n):
                per_image[i].append(segment_bases.ratio + i)
        if bool(getattr(self, "blur_keep", _online_default("blur_keep"))):
            for i in range(n):
                per_image[i].extend((segment_bases.blur + 2 * i, segment_bases.blur + 2 * i + 1))
        if self._weather_on():
            for i in range(n):
                per_image[i].append(segment_bases.weather + i)
        if self._occlusion_on():
            for i in range(n):
                per_image[i].append(segment_bases.occlusion + i)

        units: list[list[list[int]]] = []
        # Units are ALWAYS consecutive blocks of four images, so every image owns exactly one block
        # (no index can be emitted twice -- the check below enforces that).
        compose_base = segment_bases.compose
        n_compose = segment_bases.weather - compose_base  # ceil(n / 4) compose samples, one per group
        for start in range(0, n, 4):
            blocks = [per_image[i] for i in range(start, min(start + 4, n))]
            group = start // 4
            if group < n_compose:
                # The compose sample reads its whole group at once, so it is emitted FIRST inside the
                # unit that owns image ``start``: for a complete group its four decodes are exactly
                # this unit's images and prime the LRU for every block here (an unselected compose
                # group -- compose_ratio < 1 -- falls back to the single image ``start``). The tail
                # group wraps around to earlier images, which is harmless: it costs those decodes once
                # per epoch and the unit it belongs to only holds the images that are left.
                blocks[0] = [compose_base + group, *blocks[0]]
            units.append(blocks)

        flat = sorted(index for unit in units for block in unit for index in block)
        if flat != list(range(segment_bases.total)):
            # Falling back to the plain shuffle is always CORRECT (it still visits every index once),
            # it only loses the decode-locality win -- so degrade loudly instead of crashing a run.
            LOGGER.warning(
                f"{self.prefix}grouped_sample_units produced {len(flat)} indices but the pool holds "
                f"{segment_bases.total}: the pool layout and the grouping rules have drifted apart. "
                f"Falling back to the plain shuffle; fix the grouping rules in base.py."
            )
            return None
        return units

    def raw_cache_stats(self) -> tuple[int, int]:
        """Return cumulative ``(hits, misses)`` of the worker-local raw-image LRU.

        Workers flush their counters into shared memory every ``_sync_every`` samples
        (see ``_publish_raw_cache_stats``), so in the main process this reports the aggregate of all
        workers, up to the last flush. ``(0, 0)`` when the LRU is disabled or shared memory is
        unavailable.
        """
        if getattr(self, "_raw_cache_size", 0) <= 0 or getattr(self, "_mp_raw_hits", None) is None:
            return (0, 0)
        try:
            return (int(self._mp_raw_hits.value), int(self._mp_raw_misses.value))
        except Exception:  # shared memory already closed (e.g. loader torn down)
            return (0, 0)

    def _publish_raw_cache_stats(self) -> None:
        """Add this process's pending LRU counters into the shared totals (worker-side, amortized).

        Called from ``_sync_epoch_masks`` on the same ``_sync_every`` tick as the epoch poll, so the
        cost is two locked adds per ``_sync_every`` samples rather than two per read -- the same
        reasoning that keeps the epoch poll out of the per-sample path.
        """
        shared = getattr(self, "_mp_raw_hits", None)
        if shared is None or not (self._raw_hits or self._raw_misses):
            return
        try:
            with shared.get_lock():
                shared.value += self._raw_hits
            with self._mp_raw_misses.get_lock():
                self._mp_raw_misses.value += self._raw_misses
        except Exception:
            return
        self._raw_hits = 0
        self._raw_misses = 0

    def _log_raw_cache_stats(self) -> None:
        """Log the previous epoch's raw-image LRU hit rate (main process, once per epoch).

        This is the regression alarm for the sampling order: the LRU's capacity cannot cover a global
        shuffle, so a hit rate that collapses toward 0 means the grouped sampler is off, was disabled
        by a guard, or the pool layout changed underneath it. Logged as a per-epoch delta so the
        effect of ``close_mosaic`` / branch ratios (which change the read mix) stays visible.
        """
        if getattr(self, "_raw_cache_size", 0) <= 0 or not getattr(self, "augment", False):
            return
        stats = self.raw_cache_stats()
        reads = (stats[0] - self._raw_reported[0]) + (stats[1] - self._raw_reported[1])
        if reads <= 0:
            return
        hits = stats[0] - self._raw_reported[0]
        self._raw_reported = stats
        order = "grouped" if self.slice_grouped_sampler else "globally shuffled"
        LOGGER.info(
            f"{self.prefix}raw-image LRU (slice_raw_cache_size={self._raw_cache_size}, {order} sampler): "
            f"{hits / reads:.1%} hit on {reads} reads last epoch"
        )

    def _build_origin_sample(self, index: int, img_index: int) -> dict[str, Any]:
        """Build one un-sliced ORIGINAL-resolution sample (slice_keep_origin independent segment).

        The origin segment is laid out right after the base slicing segment (see _segment_bases) and
        behaves exactly like a plain full image in the mixed pool: load_image + training resize +
        ratio_pad, and it enters the Mosaic mix pool (dataset.buffer) like every other sample.
        Nothing is written to disk.

        ``index`` is the EXPANDED mixed-pool index (for correct Mosaic buffer bookkeeping);
        ``img_index`` is the ORIGINAL image index this sample derives from.
        """
        label = deepcopy(self.labels[img_index])
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(img_index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        # ``self.batch`` is indexed by IMAGE index (0..ni-1) and yields a batch id, so the ORIGINAL
        # index is the correct key here -- passing the expanded mixed-pool index would read past the
        # end of ``self.batch`` (and pick the wrong batch) as soon as online augmentation and rect
        # were ever allowed to coexist.
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[img_index]]
        # Keep the original on the same Mosaic mix pool as every other sample (cache != 'ram')
        self._touch_buffer(index)
        return self.update_labels_info(label)

    def _compose_on(self) -> bool:
        """True when the compose branch allocates samples (switch on AND >= 4 originals).

        With fewer than 4 originals a group would have to reuse the same image in 2+ quadrants,
        duplicating its targets and skewing the label distribution, so compose is switched off
        entirely in that case.
        """
        return bool(getattr(self, "compose_keep", _online_default("compose_keep"))) and len(self.labels) >= 4

    def _extended_pool_on(self) -> bool:
        """True when any project extension allocates samples beyond plain originals.

        ``load_image``'s self-managed buffer (pure-ultralytics path) is only safe when the buffer
        can only contain original-image indices. Once any extended segment (keep_origin / ratio /
        blur / compose) writes EXPANDED indices into the same buffer, ``load_image`` must not
        pop/clear ``ims`` with them; buffer bookkeeping is then entirely owned by
        ``get_image_and_label`` (expanded indices).
        """
        return (
            self._keep_origin_on()
            or bool(getattr(self, "ratio_pad_keep", _online_default("ratio_pad_keep")))
            or bool(getattr(self, "blur_keep", _online_default("blur_keep")))
            or self._weather_on()
            or self._occlusion_on()
            or self._compose_on()
        )

    def _load_image_cached(self, img_index: int, *, copy: bool = True) -> np.ndarray:
        """Load original-resolution image, with a tiny per-worker memory LRU (no .npy disk cache).

        Unified read path used by all online-augmentation branches (slice / blur / ratio /
        compose). Reads the original JPEG directly via ``imread``; a small in-memory LRU
        (``slice_raw_cache_size``) absorbs repeated decoding of the same image across its
        sub-samples. The custom .npy disk cache was removed -- it measured no benefit.

        Returns the image as a contiguous uint8 array with at least 3 dims (H, W, C); grayscale
        is expanded to (H, W, 1).

        Caching: by default a hit returns a COPY, not the cached array. Callers legitimately take
        ownership of the result (e.g. ``_build_ratio_sample`` keeps ``big = im`` when no padding is
        needed, and downstream affine / mosaic transforms write in place), so handing out the cached
        buffer would corrupt it for the next sub-sample of the same image. A memcpy is still
        ~10-30x cheaper than decoding a large JPEG.

        ``copy=False`` opts out and may return the shared LRU buffer itself (a full-resolution
        memcpy is ~20 ms on a 4000x3000 frame, so it is worth skipping when it is provably safe).
        The contract: the caller must only READ the returned array -- never write into it, never
        hand it to something that writes. compose uses this, because it only copies its sources
        into a freshly allocated canvas.
        """
        # Per-worker LRU lookup (see __init__). Disabled when slice_raw_cache_size <= 0.
        size = self._raw_cache_size
        cache = self._raw_cache
        if size > 0:
            hit = cache.get(img_index)
            if hit is not None:
                # Refresh the order on hit, so hot entries are not evicted purely by first-load time
                # (a plain dict hit used to leave the entry in its original slot).
                cache.move_to_end(img_index)
                self._raw_hits += 1
                return hit.copy() if copy else hit

        f = self.im_files[img_index]
        im = imread(f, flags=self.cv2_flag)
        if im is None:
            raise FileNotFoundError(f"Image Not Found {f}")
        if im.ndim == 2:
            im = im[..., None]

        if size > 0:
            self._raw_misses += 1
            # Evict one entry at a time from the LRU end; no key-list materialisation per miss.
            while len(cache) >= size:
                cache.popitem(last=False)
            cache[img_index] = im
            return im.copy() if copy else im
        return im

    def _degrade_max_side(self) -> float:
        """Pixel cap applied to the degradation branches before they run.

        ``degrade_max_side`` semantics:
            0 (default) -> auto: ``2 * imgsz`` (a 2x oversampled working resolution, so the final
                           resize to ``imgsz`` still downsamples instead of upsampling);
            > 0         -> that many pixels on the long side;
            < 0         -> disabled: degrade at the original resolution (legacy behaviour).
        """
        v = float(getattr(self, "degrade_max_side", _online_default("degrade_max_side")) or 0)
        if v < 0:
            return 0.0
        # 640 = BaseDataset.__init__'s own default, so the fallback cannot disagree with the class.
        return v if v > 0 else 2.0 * float(getattr(self, "imgsz", 640))

    def _degrade_frame(self, img_index: int) -> tuple[np.ndarray, float]:
        """Load an original frame for a degradation branch, downscaled to ``degrade_max_side``.

        blur / weather / occlusion are pixel-scale operations whose cost is O(pixels), yet the
        sample is resized to ``imgsz`` immediately afterwards -- running them at full sensor
        resolution is therefore pure waste (measured 11.5x across the three branches on a
        4000x3000 frame) and inflates each kernel's transient allocation (824 MB for weather noise
        at that size). Slicing keeps running at the original resolution: there it is a genuine
        semantic requirement (small objects must be enlarged, not shrunk).

        Returns ``(img, scale)`` where ``scale`` is the factor actually applied to the long side, so
        pixel-typed parameters (PSF length, defocus sigma, rain-line length) can be scaled with it.
        Scaling those parameters keeps the post-resize result near-identical to the uncapped path
        (measured mean|diff| = 2.11 / corr = 0.9982), because the final resize preserves the
        RELATIVE scale of the degradation; ``scale == 1.0`` means the frame was left untouched.
        """
        # The cap itself lives in _cap_long_side so compose can apply the same rule.
        # The cap itself lives in _cap_long_side so compose can apply the same rule. The kernel is
        # configurable (degrade_resample: "area" = antialiased/slower, "linear" = faster/softer);
        # geometry is identical either way, only the resampling filter differs.
        resample = str(getattr(self, "degrade_resample", _online_default("degrade_resample")) or "linear")
        interp = cv2.INTER_LINEAR if resample == "linear" else cv2.INTER_AREA
        return _cap_long_side(self._load_image_cached(img_index), self._degrade_max_side(), interp=interp)

    def _build_blur_sample(self, index: int, img_index: int, long: bool = False) -> dict[str, Any]:
        """Build one in-memory motion-blurred image from a single original image (online port of the offline
        motion_blur tool).

        Two tiers per image: ``short`` (light, length in [blur_short_len_min, blur_short_len_max], no
        defocus) and ``long`` (heavy, length in [blur_long_len_min, blur_long_len_max], optional defocus
        sigma up to blur_long_defocus_sigma). The blur angle is sampled uniformly in [0, 180) per call
        (augmentation randomness, consistent with fliplr etc.), unless ``blur_axis_aligned`` is set, in
        which case it is restricted to 0/90 deg -- the faithful model when the camera mounting is fixed
        and the motion maps to the image axes (see ``online_degrade._apply_motion_blur``; that path is
        also bit-identical to the general one and ~1.8x faster over both tiers). Labels are UNCHANGED
        (blur does not move targets). The blurred image is resized to the training size like the other
        branches and enters the Mosaic mix pool (dataset.buffer). Nothing is written to disk (save via
        blur_save_dir).

        ``index`` is the EXPANDED mixed-pool index (for correct Mosaic buffer bookkeeping);
        ``img_index`` is the ORIGINAL image index this blurred sample derives from.
        """
        f = self.im_files[img_index]
        # Degrade at the capped resolution (_degrade_frame), scaling the PSF with it so the result
        # after the resize to imgsz matches the original-resolution path (see _degrade_frame).
        im, scale = self._degrade_frame(img_index)
        if long:
            lo = float(getattr(self, "blur_long_len_min", _online_default("blur_long_len_min")))
            hi = float(getattr(self, "blur_long_len_max", _online_default("blur_long_len_max")))
            sigma = float(getattr(self, "blur_long_defocus_sigma", _online_default("blur_long_defocus_sigma")))
            tier = "long"
        else:
            lo = float(getattr(self, "blur_short_len_min", _online_default("blur_short_len_min")))
            hi = float(getattr(self, "blur_short_len_max", _online_default("blur_short_len_max")))
            sigma = 0.0
            tier = "short"
        # blur_ratio: 未选中原图跳过模糊, 整图直通 (位保留, len 恒定); 短+长同命运
        if self._blur_mask is not None and not self._blur_mask[img_index]:
            blur = im
        else:
            length = random.uniform(lo, hi) * scale
            # blur_axis_aligned: 拖影只取像面轴向 (0=水平 / 90=垂直, 二选一), 用于运动方向在像面上
            # 恒定映射到水平或垂直的场景 (相机固定安装, 车载/航拍沿航迹方向)。这类域里轴对齐是更
            # 贴合的建模而非近似: 轴对齐的 PSF 精确等于均匀箱式, 输出与"任意角下恰好取到 0/90"
            # 逐位相同, 只是更快 (见 online_degrade._apply_motion_blur)。
            # 刻意仍然只消费 random 流的一格 —— random.random() 与 random.uniform(0.0, 180.0) 同为
            # 一次抽样, 因此开关翻转不会让下游所有随机决策整体错位, A/B 对比才可解释。
            axis_aligned = bool(getattr(self, "blur_axis_aligned", _online_default("blur_axis_aligned")))
            angle = (0.0 if random.random() < 0.5 else 90.0) if axis_aligned else random.uniform(0.0, 180.0)
            blur = _apply_motion_blur(
                im, length=length, angle=angle, defocus_sigma=sigma * scale, axis_aligned=axis_aligned
            )

        label = deepcopy(self.labels[img_index])
        label.pop("shape", None)
        label["im_file"] = f
        label["img"] = np.ascontiguousarray(blur)

        # Keep the blurred image on the same Mosaic mix pool as every other sample (cache != 'ram')
        self._touch_buffer(index)

        # Optional save for visual inspection (blur_save_dir set). Annotated per
        # slice_save_annotated, capped by slice_save_max_blur (per-branch override; falls back to
        # slice_save_max when the per-branch cap is not set), deduplicated per (image, tier) across epochs.
        save_dir = str(getattr(self, "blur_save_dir", _online_default("blur_save_dir")) or "")
        if save_dir:
            branch_dir = Path(save_dir)
            key = ("blur", tier, index)
            # 画框/去重/限额/命名抽到 _save_annotated; 计数仅在写入成功时前进。
            self._save_annotated(
                "blur", branch_dir, blur,
                label.get("bboxes", np.empty((0, 4))), label.get("cls", np.empty((0, 1))),
                f"blur_{tier}_p{os.getpid()}_img{img_index}", key,
                bool(getattr(self, "slice_save_annotated", _online_default("slice_save_annotated"))),
            )

        # Resize to the training size (shared tail)
        return self._finalize_label(label, blur)

    def _build_weather_sample(self, index: int, img_index: int) -> dict[str, Any]:
        """Build one in-memory weather-degraded image (rain / haze / Gaussian noise) from an original.

        Online port of the weather-degradation proposal: each sample picks ONE type randomly from
        ``weather_types`` (comma-separated, e.g. "rain,haze,noise") and applies the degradation with
        randomly sampled intensity (see ``_apply_weather``). Labels are UNCHANGED (degradation never
        moves targets). ``weather_ratio``: un-selected originals pass through as the full image (slot
        kept, content replaced, len constant -- same pattern as blur_ratio). The image is resized to
        the training size like every other branch and enters the Mosaic mix pool. Nothing is written
        to disk unless ``weather_save_dir`` is set (visual inspection).

        ``index`` is the EXPANDED mixed-pool index (for correct Mosaic buffer bookkeeping);
        ``img_index`` is the ORIGINAL image index this sample derives from.
        """
        f = self.im_files[img_index]
        # Degrade at the capped resolution; only the PIXEL-typed rain-line length scales with it
        # (haze_beta / noise_std are intensity quantities and therefore resolution-independent).
        im, scale = self._degrade_frame(img_index)
        # weather_ratio: 未选中原图整图直通 (位保留, len 恒定)
        if self._weather_mask is not None and not self._weather_mask[img_index]:
            out = im
            weather_type = "none"
        else:
            types = [t.strip() for t in str(getattr(self, "weather_types", _online_default("weather_types"))).split(",") if t.strip()]
            weather_type = random.choice(types) if types else "haze"
            out = _apply_weather(
                im,
                weather_type,
                rain_density=float(getattr(self, "weather_rain_density", _online_default("weather_rain_density"))),
                rain_length=float(getattr(self, "weather_rain_length", _online_default("weather_rain_length"))) * scale,
                haze_beta=float(getattr(self, "weather_haze_beta", _online_default("weather_haze_beta"))),
                noise_std=float(getattr(self, "weather_noise_std", _online_default("weather_noise_std"))),
            )

        label = deepcopy(self.labels[img_index])
        label.pop("shape", None)
        label["im_file"] = f
        label["img"] = np.ascontiguousarray(out)

        # Keep the degraded image on the same Mosaic mix pool as every other sample (cache != 'ram')
        self._touch_buffer(index)

        # Optional save for visual inspection (weather_save_dir set). Annotated per
        # slice_save_annotated, capped by slice_save_max_weather (falls back to slice_save_max),
        # deduplicated per (image, weather_type) across epochs. Saving does NOT depend on the slicing
        # pipeline (slice_transform may be None when slice_prob=0): weather_save_dir alone enables it.
        save_dir = str(getattr(self, "weather_save_dir", _online_default("weather_save_dir")) or "")
        if save_dir:
            branch_dir = Path(save_dir)
            key = ("weather", weather_type, index)
            # 画框/去重/限额/命名抽到 _save_annotated。保存不依赖切片管线
            # (slice_transform 可能为 None), save_annotated 此时按 True 兜底。
            self._save_annotated(
                "weather", branch_dir, out,
                label.get("bboxes", np.empty((0, 4))), label.get("cls", np.empty((0, 1))),
                f"weather_{weather_type}_p{os.getpid()}_img{img_index}", key,
                bool(getattr(self, "slice_save_annotated", _online_default("slice_save_annotated"))),
            )

        # Resize to the training size (shared tail)
        return self._finalize_label(label, out)

    def _build_occlusion_sample(self, index: int, img_index: int) -> dict[str, Any]:
        """Build one in-memory occluded image (rect / stripe blocks) from an original.

        Online port of the occlusion proposal: each selected original gets ``occlusion_blocks``
        semantic blocks (type picked randomly from ``occlusion_types``) drawn on the ORIGINAL
        resolution image. Labels stay UNCHANGED -- the point is occlusion-robust detection -- except
        a box whose covered area ratio exceeds ``occlusion_max_cover`` is dropped from the label
        (a fully-hidden target is pure noise). ``occlusion_ratio``: un-selected originals pass
        through as the full image (slot kept, len constant, same pattern as weather_ratio). The
        image is resized to the training size like every other branch and enters the Mosaic mix
        pool. Nothing is written to disk unless ``occlusion_save_dir`` is set.

        ``index`` is the EXPANDED mixed-pool index; ``img_index`` is the ORIGINAL image index.
        """
        f = self.im_files[img_index]
        # Degrade at the capped resolution. occlusion_size_ratio is a FRACTION of the image, so the
        # parameters need no rescaling -- the occluders keep their relative size, and the boxes
        # returned in pixel space stay consistent because the coverage maths below uses this frame's
        # h/w.
        im = self._degrade_frame(img_index)[0]
        h, w = im.shape[:2]
        # occlusion_ratio: 未选中原图整图直通 (位保留, len 恒定)
        if self._occlusion_mask is not None and not self._occlusion_mask[img_index]:
            out = im
            occluder_boxes = []
            occlusion_type = "none"  # defensive: keep the name defined on every path
        else:
            types = [t.strip() for t in str(getattr(self, "occlusion_types", _online_default("occlusion_types"))).split(",") if t.strip()]
            occlusion_type = random.choice(types) if types else "rect"
            out, occluder_boxes = _apply_occlusion(
                im,
                occlusion_type,
                blocks=int(getattr(self, "occlusion_blocks", _online_default("occlusion_blocks")) or 1),
                size_ratio=float(getattr(self, "occlusion_size_ratio", _online_default("occlusion_size_ratio"))),
                color=str(getattr(self, "occlusion_color", _online_default("occlusion_color")) or "auto"),
            )

        label = deepcopy(self.labels[img_index])
        label.pop("shape", None)
        label["im_file"] = f
        # max_cover: 目标被遮挡面积占比超过阈值 -> 从标签剔除 (完全被盖住的目标=纯噪声)
        max_cover = float(getattr(self, "occlusion_max_cover", _online_default("occlusion_max_cover")))
        if occluder_boxes and len(label.get("bboxes", [])):
            boxes = np.asarray(label["bboxes"], dtype=np.float64).copy()  # normalized xywh
            # covered 改为"块与目标框交集的并集面积"。旧实现逐块累加交叉面积,
            # 多个遮挡块相互重叠时重叠区被重复计入, 覆盖率虚高, 目标可能被提前按 max_cover
            # 误剔除。块数通常 1~3 且只对含目标的图执行, 布尔掩码开销可忽略。
            # (顺带删除从未被使用的 oc_area_sum 死代码)
            keep = np.ones(len(boxes), dtype=bool)
            for bi, b in enumerate(boxes):
                cx, cy, bw, bh = b[0] * w, b[1] * h, b[2] * w, b[3] * h
                x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                area = max(1.0, (x1 - x0) * (y1 - y0))
                # clip every occluder to the target box and union their areas with exact
                # rectangle math (no per-sample h*w bool mask; union semantics preserved).
                inter = []
                for (ox0, oy0, ox1, oy1) in occluder_boxes:
                    ix0, iy0 = max(ox0, x0), max(oy0, y0)
                    ix1, iy1 = min(ox1, x1), min(oy1, y1)
                    if ix1 > ix0 and iy1 > iy0:
                        inter.append((int(math.floor(ix0)), int(math.floor(iy0)),
                                      int(math.ceil(ix1)), int(math.ceil(iy1))))
                covered = _union_area(inter)
                if covered / area >= max_cover:
                    keep[bi] = False
            if not keep.all():
                label["bboxes"] = boxes[keep].astype(np.float32)
                cls = np.asarray(label.get("cls", np.empty((0, 1))))
                label["cls"] = np.asarray(cls).reshape(-1, 1)[keep].astype(np.float32) if len(cls) else cls
                segs = label.get("segments")
                if segs:
                    # Index the segments with the SAME boolean mask as the boxes. The previous
                    # ``zip(keep, segments)`` silently truncated whenever the two lengths differed
                    # (malformed labels, or any future filter touching only one of them), producing
                    # boxes-without-segments misalignment and no error anywhere -- while the cls
                    # path right above already used an explicit mask. Warn instead of raising: one
                    # bad annotation should not abort a multi-hour run, but it must not be silent.
                    if len(segs) != len(keep):
                        LOGGER.warning(
                            f"occlusion: {len(segs)} segments vs {len(keep)} boxes for '{f}' -- "
                            "aligning by position and keeping only mask-true entries."
                        )
                    label["segments"] = [s for s, k in zip(segs, keep) if k]

        label["img"] = np.ascontiguousarray(out)

        # Keep the occluded image on the same Mosaic mix pool as every other sample (cache != 'ram')
        self._touch_buffer(index)

        # Optional save for visual inspection (occlusion_save_dir set). Annotated per
        # slice_save_annotated, capped by slice_save_max_occlusion (falls back to slice_save_max),
        # deduplicated per (image, occlusion_type) across epochs.
        save_dir = str(getattr(self, "occlusion_save_dir", _online_default("occlusion_save_dir")) or "")
        if save_dir:
            branch_dir = Path(save_dir)
            key = ("occlusion", occlusion_type if occluder_boxes else "none", index)
            # 画框/去重/限额/命名抽到 _save_annotated; 保存不依赖切片管线。
            self._save_annotated(
                "occlusion", branch_dir, out,
                label.get("bboxes", np.empty((0, 4))), label.get("cls", np.empty((0, 1))),
                f"occlusion_{occlusion_type if occluder_boxes else 'none'}_p{os.getpid()}_img{img_index}", key,
                bool(getattr(self, "slice_save_annotated", _online_default("slice_save_annotated"))),
            )

        # Resize to the training size (shared tail)
        return self._finalize_label(label, out)

    def _build_ratio_sample(self, index: int, img_index: int) -> dict[str, Any]:
        """Build one in-memory aspect-ratio-padded image from a single original image (online port of the
        offline change_image_resolution tool).

        The original image is read at its ORIGINAL resolution, padded with borders (left/right or top/bottom,
        symmetric) to its target aspect ratio -- ``auto``: 4:3 <-> 16:9 bidirectional, any other ratio goes to
        its nearest of 4:3 or 16:9; ``4:3``/``16:9``: unify every image to that ratio -- and every annotation
        (bboxes/segments/keypoints) is remapped by the pad offset: ``xc' = (xc*W + pad_left)/new_w``,
        ``yc' = (yc*H + pad_top)/new_h``, ``w' = w*W/new_w``, ``h' = h*H/new_h``, then boundary-clamped and
        invalid boxes dropped (identical to the offline tool). The padded image is resized to the training size
        like the other branches and enters the Mosaic mix pool (dataset.buffer). Nothing is written to disk.

        ``index`` is the EXPANDED mixed-pool index (for correct Mosaic buffer bookkeeping);
        ``img_index`` is the ORIGINAL image index this padded sample derives from.
        """
        f = self.im_files[img_index]
        # Cap the working resolution BEFORE padding. ``_ratio_pad_params`` can otherwise inflate the
        # canvas enormously (an 8000x1000 frame aligned to 16:9 becomes 8000x4500, 4.5x the original
        # pixel count and a 108 MB allocation) only for the result to be resized straight back down
        # to imgsz by ``_finalize_label``. All the pad maths is normalised, and the pad offset is
        # derived from this frame's w/h below, so padding a capped frame is equivalent.
        im = self._degrade_frame(img_index)[0]
        h, w = im.shape[:2]
        # ratio_pad_ratio: 未选中原图跳过加框, 整图直通 (位保留, len 恒定)
        skip_pad = self._ratio_mask is not None and not self._ratio_mask[img_index]
        target = str(getattr(self, "ratio_pad_target", _online_default("ratio_pad_target")) or "auto")
        color_key = str(getattr(self, "ratio_pad_color", _online_default("ratio_pad_color")) or "black")
        # validate up front. Previously an unknown color raised a bare KeyError halfway through
        # training, and an unknown target silently fell back to 16:9 (any non-"4:3" string did).
        if color_key not in _RATIO_PAD_COLORS:
            raise ValueError(f"ratio_pad_color must be one of {sorted(_RATIO_PAD_COLORS)}, got '{color_key}'.")
        if target not in ("auto", "4:3", "16:9"):
            raise ValueError(f"ratio_pad_target must be one of 'auto', '4:3', '16:9', got '{target}'.")
        pad = None if skip_pad else _ratio_pad_params(w, h, target, auto=(target == "auto"))
        if pad is None:
            # Already at the target ratio: no padding needed (use the original as-is)
            big = im
            new_w, new_h, pad_left, pad_top = w, h, 0, 0
        else:
            new_w, new_h, pad_left, pad_top = pad
            C = im.shape[2]
            big = np.full((new_h, new_w, C), _RATIO_PAD_COLORS[color_key], dtype=im.dtype)
            big[pad_top:pad_top + h, pad_left:pad_left + w] = im
            del im

        lb = self.labels[img_index]
        boxes = np.asarray(lb.get("bboxes", np.empty((0, 4))), dtype=np.float64).copy()
        # explicit .copy. np.asarray returns the SAME array when the dtype already matches, so this
        # used to hand out a live view of self.labels[i]["cls"] -- and augment._update_label_text writes
        # label["cls"] in place, which would permanently corrupt the cached label for all later epochs.
        # _build_blur_sample and the main path already deepcopy; this branch and _build_compose_sample did not.
        cls = np.asarray(lb.get("cls", np.empty((0, 1))), dtype=np.float32).copy()
        keep = None
        if len(boxes) and (pad_left or pad_top):
            boxes[:, 0] = (boxes[:, 0] * w + pad_left) / new_w
            boxes[:, 1] = (boxes[:, 1] * h + pad_top) / new_h
            boxes[:, 2] = (boxes[:, 2] * w) / new_w
            boxes[:, 3] = (boxes[:, 3] * h) / new_h
            boxes[:, 0] = np.clip(boxes[:, 0], 0.0, 1.0)
            boxes[:, 1] = np.clip(boxes[:, 1], 0.0, 1.0)
            boxes[:, 2] = np.clip(boxes[:, 2], 0.0, 1.0 - boxes[:, 0])
            boxes[:, 3] = np.clip(boxes[:, 3], 0.0, 1.0 - boxes[:, 1])
            keep = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
            boxes = boxes[keep]
            cls = cls[keep]
        # segments: normalized polys remapped by the pad offset (no clipping needed: padding only enlarges)
        segs = []
        for si, s in enumerate(lb.get("segments", []) or []):
            # apply the same `keep` mask as the boxes. Padding never drops anything today, so this is
            # dormant -- but the moment any box is filtered, len(bboxes) != len(segments) would silently
            # mislabel every seg/pose task with no error anywhere.
            if keep is not None and not (si < len(keep) and bool(keep[si])):
                continue
            s = np.asarray(s, dtype=np.float64).copy()
            if pad_left or pad_top:
                s[..., 0] = (s[..., 0] * w + pad_left) / new_w
                s[..., 1] = (s[..., 1] * h + pad_top) / new_h
            segs.append(s.astype(np.float32))
        label = {
            "im_file": f,
            "img": np.ascontiguousarray(big),
            "bboxes": boxes.astype(np.float32),
            "bbox_format": "xywh",
            "normalized": True,
            "cls": cls,
            "segments": segs,
        }
        kpts = lb.get("keypoints", None)
        if kpts is not None:
            k = np.asarray(kpts, dtype=np.float64).copy()
            if pad_left or pad_top:
                k[..., 0] = (k[..., 0] * w + pad_left) / new_w
                k[..., 1] = (k[..., 1] * h + pad_top) / new_h
            if keep is not None and k.shape[0] == len(keep):
                k = k[keep]  # keep keypoints aligned with the filtered boxes
            label["keypoints"] = k.astype(np.float32)

        # Keep the ratio-padded image on the same Mosaic mix pool as every other sample (cache != 'ram')
        self._touch_buffer(index)

        # Optional save of the ratio-padded image for visual inspection (ratio_pad_save_dir set).
        # Annotated per slice_save_annotated, capped by slice_save_max_ratio (per-branch override,
        # falls back to slice_save_max), deduplicated per (image) across epochs/mix visits.
        # Disabled by default (empty dir).
        save_dir = str(getattr(self, "ratio_pad_save_dir", _online_default("ratio_pad_save_dir")) or "")
        if save_dir:
            branch_dir = Path(save_dir)
            key = ("ratio", index)
            # 画框/去重/限额/命名抽到 _save_annotated。
            self._save_annotated(
                "ratio", branch_dir, big,
                label.get("bboxes", np.empty((0, 4))), label.get("cls", np.empty((0, 1))),
                f"ratio_p{os.getpid()}_img{img_index}", key,
                bool(getattr(self, "slice_save_annotated", _online_default("slice_save_annotated"))),
            )

        # Resize to the training size (shared tail)
        return self._finalize_label(label, big)

    def _build_compose_sample(self, index: int) -> dict[str, Any]:
        """Build a 2x2 composed image from 4 original images (online port of the offline compose tool).

        Each group of 4 original images (``[group*4, group*4+3]``, wrapped for the tail group) is stitched
        into one larger image (2 columns x 2 rows) in memory, and every annotation (bboxes/segments/keypoints)
        is remapped from each sub-image's normalized coords to the composed-image coordinate system:
        ``xc' = (xc + col) / 2, yc' = (yc + row) / 2, w' = w / 2, h' = h / 2`` (col/row = 0/1). The composed
        image is then resized to the training size like the other branches. It enters the Mosaic mix pool
        (dataset.buffer) so it participates in mosaic stitching. Nothing is written to disk.

        Each source is capped to half of ``compose_max_side`` BEFORE the canvas is allocated (see the
        comment in the body), so the canvas itself is bounded by ``compose_max_side``.
        """
        n_origin = len(self.labels)
        # The compose segment starts right after the base + ratio + blur segments (_segment_bases);
        # group = index - compose_base selects the group of 4 originals. (named access)
        segment_bases = self._segment_bases()
        group = index - segment_bases.compose
        base = group * 4
        # with fewer than 4 originals the modulo wrap would put the SAME image in 2+ quadrants,
        # duplicating its targets and skewing the label distribution. _compose_on() never allocates
        # compose indices in that case, so this is a defensive guard for direct calls only.
        if n_origin < 4:
            raise IndexError(
                f"Compose samples are disabled with fewer than 4 images (got {n_origin}); index {index} is "
                f"outside the valid range."
            )
        idxs = [(base + j) % n_origin for j in range(4)]  # wrap the tail group to always have 4 images
        # compose_ratio: 未选中组退回组内第 1 张原图整图 (位保留, len 恒定)
        if self._compose_mask is not None and not self._compose_mask[group]:
            return self._build_origin_sample(index, base)
        # Working-resolution cap, the same rule the degradation branches apply through
        # _degrade_frame: bring every source down to at most HALF the compose canvas side BEFORE the
        # canvas is allocated. compose labels are normalized coords that only depend on the relative
        # 2x2 geometry, so capping the pixels does not touch the labels -- and it replaces the old
        # "allocate the canvas at full sensor resolution, fill it, then downscale the whole thing",
        # measured at 5.6x the time and 8.7x the peak memory on 4x 4000x3000 sources.
        # compose_max_side semantics now mirror degrade_max_side:
        #   0 -> auto: 2 * imgsz      > 0 -> that many pixels on the long side      < 0 -> disabled
        max_side = getattr(self, "compose_max_side", _online_default("compose_max_side"))
        max_side = int(max_side) if max_side is not None else 0
        if max_side == 0:
            max_side = 2 * int(getattr(self, "imgsz", 640))
        half = max_side / 2.0 if max_side > 0 else 0.0  # <= 0 disables the cap (see _cap_long_side)
        imgs = []
        for i in idxs:
            # copy=False: each source is only READ here (it is memcpy'd into a canvas slice, or handed
            # to cv2.resize which returns a new array), so the worker-local LRU buffer it may alias is
            # never written to. This drops 4 full-resolution memcpys (~20 ms each at 4000x3000).
            # INTER_LINEAR (not the degradation branches' INTER_AREA): see _cap_long_side -- the box
            # path costs 40 ms/source here against 1.7 ms, and the pre-fix code was bilinear too.
            im, _ = _cap_long_side(self._load_image_cached(i, copy=False), half, interp=cv2.INTER_LINEAR)
            imgs.append(im)
        # Unify sub-image size to the max in this group (stretching preserves normalized coords
        # linearly). W/H are the max over the ALREADY CAPPED group, so 2*W and 2*H are within
        # max_side by construction and the composed image never needs a post-hoc downscale.
        W = max(im.shape[1] for im in imgs)
        H = max(im.shape[0] for im in imgs)
        # Allocate the composed image ONCE and fill each 2x2 sub-block in place (memory-friendly: avoids the
        # intermediate row1/row2 hstack buffers, ~2x peak reduction on a large composed image).
        C = imgs[0].shape[2]
        big = np.empty((2 * H, 2 * W, C), dtype=imgs[0].dtype)
        for j, im in enumerate(imgs):
            r, c = divmod(j, 2)
            if im.shape[1] != W or im.shape[0] != H:
                # never a downscale (W/H are the group max), so INTER_LINEAR is the right kernel here
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
            big[r * H:(r + 1) * H, c * W:(c + 1) * W] = im
            imgs[j] = None  # free each source the moment it is copied, instead of holding all 4
        del imgs  # release the 4 source images as early as possible

        boxes_all, cls_all, segs_all, kpts_all, has_kpts = [], [], [], [], False
        for j, i in enumerate(idxs):
            row, col = divmod(j, 2)
            lb = self.labels[i]
            boxes = np.asarray(lb.get("bboxes", np.empty((0, 4))), dtype=np.float64).copy()
            # explicit .copy (np.asarray is a no-op view when the dtype already matches), otherwise
            # the composed label shares cls storage with self.labels and in-place text updates corrupt it.
            cls = np.asarray(lb.get("cls", np.empty((0, 1))), dtype=np.float32).copy()
            if len(boxes):
                boxes[:, 0] = (boxes[:, 0] + col) / 2.0
                boxes[:, 1] = (boxes[:, 1] + row) / 2.0
                boxes[:, 2] = boxes[:, 2] / 2.0
                boxes[:, 3] = boxes[:, 3] / 2.0
                boxes_all.append(boxes)
                cls_all.append(cls)
            # segments: list of normalized polys -> composed-image normalized
            for s in lb.get("segments", []) or []:
                s = np.asarray(s, dtype=np.float64).copy()
                s[..., 0] = (s[..., 0] + col) / 2.0
                s[..., 1] = (s[..., 1] + row) / 2.0
                segs_all.append(s.astype(np.float32))
            # keypoints: (N, K, 3) normalized x,y + visibility
            kpts = lb.get("keypoints", None)
            if kpts is not None:
                has_kpts = True
                k = np.asarray(kpts, dtype=np.float64).copy()
                k[..., 0] = (k[..., 0] + col) / 2.0
                k[..., 1] = (k[..., 1] + row) / 2.0
                kpts_all.append(k)

        bboxes = np.concatenate(boxes_all, 0).astype(np.float32) if boxes_all else np.empty((0, 4), np.float32)
        cls = np.concatenate(cls_all, 0).astype(np.float32) if cls_all else np.empty((0, 1), np.float32)
        label = {
            "im_file": self.im_files[idxs[0]],
            "img": np.ascontiguousarray(big),
            "bboxes": bboxes,
            "bbox_format": "xywh",
            "normalized": True,
            "cls": cls,
            "segments": segs_all,
        }
        if has_kpts:
            label["keypoints"] = np.concatenate(kpts_all, 0).astype(np.float32)

        # Optional save of the composed 2x2 image for visual inspection (compose_save=True).
        # Save dir: compose_save_dir (custom) if set, else slice_save_dir/compose/. Annotated per
        # slice_save_annotated, capped by slice_save_max_compose (per-branch override, falls back to
        # slice_save_max), deduplicated per (group) across epochs/mix visits.
        st = getattr(self, "slice_transform", None)
        if getattr(self, "compose_save", False):
            comp_dir = str(getattr(self, "compose_save_dir", "") or "")
            branch_dir = Path(comp_dir) if comp_dir else (st.save_dir / "compose" if st is not None and st.save_dir is not None else None)
            if branch_dir is None and not getattr(self, "_compose_warned", False):
                # args.yaml ships compose_save=True with an empty compose_save_dir and no
                # slice_save_dir -> nothing was ever written and nothing said why.
                self._compose_warned = True
                LOGGER.warning(
                    f"{self.prefix}compose_save=True but no output directory is configured: set "
                    f"compose_save_dir, or slice_save_dir (images then go to <slice_save_dir>/compose). "
                    f"Skipping compose save."
                )
            if branch_dir is not None:
                key = ("compose", group)
                # 画框/去重/限额/命名抽到 _save_annotated (成功写盘才计数)。
                self._save_annotated(
                    "compose", branch_dir, big,
                    bboxes, cls,
                    f"compose_p{os.getpid()}_g{group}", key,
                    bool(getattr(self, "slice_save_annotated", _online_default("slice_save_annotated"))),
                )

        # Keep the composed image on the same Mosaic mix pool as every other sample (cache != 'ram')
        self._touch_buffer(index)

        # Resize to the training size (shared tail)
        return self._finalize_label(label, big)

    def get_image_and_label(self, index: int, count_slice: bool = True) -> dict[str, Any]:
        """Get and return label information from the dataset.

        Args:
            index (int): Index of the image to retrieve.
            count_slice (bool): Whether OnlineSlice updates its positive/background counters and saves tiles.
                Auxiliary "mix" samples requested by Mosaic/CutMix/MixUp pass ``False`` so they slice normally
                but do not inflate the ``neg_ratio`` quota or duplicate saved slices.
        """
        # pick up the trainer's latest set_epoch publish. No-op (one locked int read) in the
        # main process and whenever the epoch is unchanged; in a DataLoader worker with a stale mask
        # set this deterministically rebuilds the masks for the published epoch.
        self._sync_epoch_masks()
        # Mixed-pool layout (see _segment_bases): a base segment holding ONLY the slicing pipeline's
        # samples (4 tiles per image), followed by SIX INDEPENDENT segments gated only by their own
        # switches -- origin (1 un-sliced original per image, keep_origin), ratio (1 per image),
        # blur (2 per image), compose (1 per 4 images), weather (1 per image), occlusion (1 per
        # image). None of them requires slicing or keep_origin.
        emit_all = getattr(self, "slice_all_tiles", _online_default("slice_all_tiles"))
        origin_on = self._keep_origin_on()
        ratio_on = bool(getattr(self, "ratio_pad_keep", _online_default("ratio_pad_keep")))
        blur_on = bool(getattr(self, "blur_keep", _online_default("blur_keep")))
        compose_on = self._compose_on()
        weather_on = self._weather_on()
        occlusion_on = self._occlusion_on()
        # Named field access: keep the SegmentBases object and read ``segment_bases.origin`` / ``segment_bases.ratio`` ...
        # Unpacking it into seven positional locals (base_len, o_base, r_base, ...) would throw away
        # exactly the readability the NamedTuple was introduced for.
        segment_bases = self._segment_bases()

        if occlusion_on and index >= segment_bases.occlusion:
            return self._build_occlusion_sample(index, index - segment_bases.occlusion)  # origin index = offset in the occlusion segment
        if weather_on and index >= segment_bases.weather:
            return self._build_weather_sample(index, index - segment_bases.weather)  # origin index = offset in the weather segment
        if compose_on and index >= segment_bases.compose:
            return self._build_compose_sample(index)  # composed 2x2 sample from 4 original images
        if blur_on and index >= segment_bases.blur:
            j = index - segment_bases.blur  # 0..2N-1: even -> short tier, odd -> long tier
            return self._build_blur_sample(index, j // 2, long=(j % 2 == 1))
        if ratio_on and index >= segment_bases.ratio:
            return self._build_ratio_sample(index, index - segment_bases.ratio)  # origin index = offset in the ratio segment
        if origin_on and index >= segment_bases.origin:
            return self._build_origin_sample(index, index - segment_bases.origin)  # un-sliced original (keep_origin segment)

        # ---- base segment (slicing pipeline) ----
        n_per = self._n_per()
        if n_per > 1:  # slicing is active and expands each original image
            img_index = index // n_per
            k = index % n_per  # tile index for emit_all mode
        else:  # no expansion: sample index == original image index
            img_index = index
            k = None
        label = deepcopy(self.labels[img_index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        # Online slicing runs on the ORIGINAL-resolution image (before any training resize) so small
        # objects are genuinely enlarged when the sliced sub-image is resized to the training size.
        # slice_ratio: each epoch set_epoch() rebuilds a mask that picks exactly round(slice_ratio*N)
        # ORIGINAL images to slice (original-level decision: with emit_all all 4 tiles share one fate);
        # the rest are fed as un-sliced full images. Mask None = pure slicing (default) or slicing off.
        # The un-sliced originals live in their own independent segment (_build_origin_sample) and never slice.
        slice_t = getattr(self, "slice_transform", None)
        # Lazy init (standalone / direct-iteration use, no trainer ever calling set_epoch): rebuild
        # once on first access so 0 <= *_ratio < 1 still works. MAIN process only -- DataLoader
        # workers (InfiniteDataLoader spawns them before any set_epoch) must NOT self-build here:
        # they would stamp epoch 0 without the total `epochs` (breaking close_aug_epoch) and pin the
        # shared channel; they wait for the trainer's publish via _sync_epoch_masks instead. This
        # once-per-stamp form also fixes the old refire-per-sample bug (mask None + ratio < 1 used
        # to re-run the whole rebuild on every single sample).
        if (
            self._mask_stamp < 0
            and 0.0 <= float(getattr(self, "slice_ratio", _online_default("slice_ratio"))) < 1.0
            and multiprocessing.parent_process() is None
        ):
            self._rebuild_epoch_masks(int(getattr(self, "epoch", 0)))
        slice_ok = self._slice_mask is None or bool(self._slice_mask[img_index])
        # The mosaic buffer is maintained centrally here using DATASET (expanded) indices whenever
        # load_image does NOT self-manage it: slicing on, or any project extension on (load_image's
        # self-managed path is only active in pure-ultralytics mode; mixing original indices from
        # load_image with expanded indices here would corrupt the buffer).
        if (
            self.augment
            and self.cache != "ram"
            and (slice_t is not None or self._extended_pool_on())
        ):
            self._touch_buffer(index)
        if slice_t is not None and self.augment and slice_ok:
            # _load_image_cached: 直接读原图 jpg + worker 内存 LRU (.npy 磁盘缓存已移除)
            im = self._load_image_cached(img_index)
            im, label = (
                slice_t.slice_at(im, label, k, src=(img_index, k), count=count_slice, key=img_index)
                if emit_all
                else slice_t(im, label, src=img_index, count=count_slice, key=img_index)
            )
            # reuse the shared tail instead of a third hand-rolled resize. The sliced sub-image
            # becomes the new "original" for downstream transforms; _finalize_label applies the exact same
            # resize + imgsz clamp as load_image and every other online branch, so resized_shape/ratio_pad
            # can no longer drift (the old inline resize omitted the clamp).
            return self._finalize_label(label, im)
        # load_image indexes the ORIGINAL image files, so always use img_index (== index in the non-sliced
        # case); in emit_all mode index is the expanded (4N) sample index and would overflow.
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(img_index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            # ``self.batch`` is keyed by IMAGE index (see _build_origin_sample); `index` may be an expanded
            # mixed-pool index here (emit_all with this original not selected for slicing), which
            # would overflow ``self.batch``. rect and online augmentation are mutually exclusive
            # today (v8_transforms disables the online path when rect is on), but relying on that
            # cross-file invariant without a key that is correct on its own is how this project got
            # its earlier buffer-accounting bugs.
            label["rect_shape"] = self.batch_shapes[self.batch[img_index]]
        return self.update_labels_info(label)

    def __len__(self) -> int:
        """Return the number of samples in the mixed pool.

        Base segment (``_n_per`` samples per original) plus six independent segments: +N origin
        (keep_origin), +N ratio, +2N blur, +ceil(N/4) compose, +N weather, +N occlusion -- each
        present only when its own switch is on.

        Single source of truth (fix): the total comes from ``_segment_bases``, which derives
        it from the same per-segment lengths as every boundary. The accumulation used to be
        re-implemented here; adding a new branch and missing this copy would desynchronise
        ``len(dataset)`` from the decodable index range and silently drop samples.
        """
        return self._segment_bases().total

def patch_fraction_guard() -> None:
    """Survive fraction rounding to zero on a tiny dataset.

    Stock get_img_files does im_files[:round(len * fraction)]; on a tiny set that rounds to 0 and
    leaves an empty dataset (later IndexError). When the stock result is empty and fraction<1, re-run
    the lookup with fraction forced to 1.0 to keep at least the source image(s).
    """
    from ultralytics.data.base import BaseDataset
    from ultralytics.utils import LOGGER

    if getattr(BaseDataset, "_ooo_fraction_guard", False):
        return
    _orig = BaseDataset.get_img_files

    def get_img_files(self, img_path):
        files = _orig(self, img_path)
        if self.fraction < 1 and len(files) == 0:
            old = self.fraction
            self.fraction = 1.0
            try:
                files = _orig(self, img_path)
            finally:
                self.fraction = old
            LOGGER.warning(
                f"ooo fraction guard: fraction={old} selected 0 images (rounded to zero); retained {len(files)} source image(s)."
            )
        return files

    BaseDataset.get_img_files = get_img_files
    BaseDataset._ooo_fraction_guard = True

