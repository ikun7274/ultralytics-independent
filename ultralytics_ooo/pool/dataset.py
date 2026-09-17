# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""OnlinePoolDataset: mixed virtual-sample pool as a subclass of a stock Ultralytics BaseDataset.

Mirrors the forked BaseDataset extensions verbatim. Installed onto a pristine Ultralytics by
``ultralytics_ooo.installer``; on its own it needs nothing from the forked tree except the kernels
already in ``ultralytics_ooo.core`` / ``.pool``.
"""

from __future__ import annotations

import inspect
import math
import multiprocessing
import os
import random
import zlib
from collections import OrderedDict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ultralytics.data.base import BaseDataset
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.patches import imread
from ultralytics_ooo.core import (
    _RATIO_PAD_COLORS,
    _apply_motion_blur,
    _apply_occlusion,
    _apply_weather,
    _cap_long_side,
    _ratio_pad_params,
    _union_area,
)
from ultralytics_ooo.core.saver import _ensure_dir, _imwrite
from ultralytics_ooo.pool.constants import (
    _describe_ims_cap,
    _online_default,
    _resolve_ims_cap,
    get_split_fraction,
)
from ultralytics_ooo.pool.sampler import SegmentBases

# Annotated-save branches accepted by _save_annotated, and the per-branch save-cap lookup.
# Mirrored from the fork; slice_save_max_<branch> overrides the global slice_save_max (0 = unlimited).
_SAVE_BRANCHES = {"blur", "weather", "occlusion", "ratio", "compose", "slice"}


def _resolve_init_knobs(args: tuple, kwargs: dict) -> tuple[Any, Any]:
    """Resolve the ``hyp`` / ``fraction`` this class needs from the forwarded ``*args``/``**kwargs``.

    Reading them out of ``kwargs`` alone (the previous behaviour) silently substituted the defaults for
    any POSITIONAL construction -- ``OnlinePoolDataset(img_path, 640, False, True, hyp_ns, ...)``, or a
    third-party fork's ``build_yolo_dataset``. Nothing raised: the online hyperparameters simply never
    reached the dataset, so the ims-cache budget, the raw-image LRU, grouped sampling and the
    ``cache='ram'`` guard all quietly fell back to their defaults.

    Bound against ``BaseDataset.__init__`` because that is where the real defaults for both live, no
    matter which subclass (YOLODataset / the installed mixin) sits in the MRO. Unbindable calls fall
    back to the keyword lookup rather than raising.
    """
    hyp = kwargs.get("hyp", DEFAULT_CFG)
    fraction = kwargs.get("fraction", 1.0)
    if "hyp" in kwargs and "fraction" in kwargs:
        return hyp, fraction
    try:
        bound = inspect.signature(BaseDataset.__init__).bind_partial(None, *args, **kwargs)
    except (TypeError, ValueError):
        return hyp, fraction
    bound.apply_defaults()
    return bound.arguments.get("hyp", hyp), bound.arguments.get("fraction", fraction)


def _save_cap(dataset, branch: str) -> int:
    """Per-branch save cap: ``slice_save_max_<branch>`` overrides the global ``slice_save_max``.

    A missing or ``None`` per-branch value falls back to the global key; ``0`` means unlimited. The
    first parameter is ``dataset`` rather than ``self`` on purpose -- this is a free function, and
    naming it ``self`` invited it to be read as a method.
    """
    v = getattr(dataset, f"slice_save_max_{branch}", _online_default(f"slice_save_max_{branch}"))
    if v is None:
        v = getattr(dataset, "slice_save_max", _online_default("slice_save_max"))
    return int(v) if v is not None else 0


class OnlinePoolDataset(BaseDataset):
    """BaseDataset subclass that adds the mixed virtual-sample pool.

    The original BaseDataset from a stock Ultralytics is inherited unchanged; every method below is
    the mirrored extension. ``installer`` swaps the trainer's dataset factory to this class.
    """

    def __init__(self, *args, **kwargs):
        # Cooperative multiple-inheritance: forward every arg (img_path, hyp, data, task, ...) up the
        # MRO to YOLODataset -> BaseDataset, then run the extension initialisation.
        super().__init__(*args, **kwargs)
        # Resolved against BaseDataset's real signature, so a positional call cannot silently hand the
        # extension defaults instead of the caller's hyp/fraction (see _resolve_init_knobs).
        hyp, fraction = _resolve_init_knobs(args, kwargs)
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
        # Byte budget for the same LRU (0 = unlimited). The cache holds ORIGINAL-resolution frames, so
        # the frame count above cannot be turned into memory before the first decode; the budget is
        # therefore enforced per insert in ``_load_image_cached``. It only ever BINDS the frame count
        # from above -- 16 frames of a 4000x3000 frame would be 550 MiB/worker.
        self._raw_cache_mb = float(getattr(hyp, "slice_raw_cache_mb", _online_default("slice_raw_cache_mb")) or 0)
        self._raw_cache_bytes = 0
        self._raw_cache_budget = int(self._raw_cache_mb * (1 << 20)) if self._raw_cache_mb > 0 else 0
        # One-time report of the EFFECTIVE frame capacity (see _load_image_cached). The two caps are
        # independent and only the byte one learns the real frame size, so the mismatch is otherwise
        # invisible: at 4000x3000 a frame is 34.3 MiB, so the shipped 256 MiB budget holds 7 of the 16
        # requested frames -- and raising slice_raw_cache_size to 32 would still hold exactly 7.
        self._raw_cache_eff_logged = False
        # Per-epoch branch SELECTIONS (see _rebuild_epoch_masks). Each is an ordered list of the
        # ORIGINAL image indices (group indices for compose) that this branch augments this epoch, with
        # its length -- K -- fixed by the CONFIGURED ratio, not by the draw. ``None`` is the "all slots"
        # sentinel used by ratio >= 1: it keeps the no-mask fast path (no list to allocate, no RNG to
        # draw) exactly like the boolean-mask design did, and ``_sel_indices`` maps it to ``range(n)``.
        self._sel_slice = None
        self._sel_ratio = None
        self._sel_blur = None
        self._sel_weather = None
        self._sel_occlusion = None
        self._sel_compose = None
        # True inside close_aug_epoch's window: the line-up keeps its shape but every branch builder
        # emits the plain original instead of its transform (see _rebuild_epoch_masks).
        self._closing = False
        # Target-tile schedule (slice_target_tiles, slice_all_tiles=False only): ``_tile_units`` is this
        # epoch's list of (image, tile) pairs -- one per BASE slot -- with the pairs drawn from
        # ``_target_tile_queue_cache``. ``None`` means "the schedule is not in use this epoch", and the
        # base segment then maps slots to images with a blind random tile exactly as before. Both are
        # rebuilt in every process that rebuilds the masks (main via set_epoch, workers via
        # _sync_epoch_masks) from the same seed, so they never need transporting (see
        # _apply_target_tile_schedule).
        self._tile_units: list[tuple[int, int]] | None = None
        self._target_tile_queue_cache: list[tuple[int, int]] | None = None
        self._target_tile_pass_cache: tuple[int, list[tuple[int, int]]] | None = None
        self._target_tile_warned = False
        self._target_tile_empty_warned = False
        try:
            self._mp_epoch = multiprocessing.Value("i", -1)
            self._mp_epochs = multiprocessing.Value("i", -1)
        except Exception:  # noqa: BLE001 -- shared memory may be unavailable; degrade to a no-op epoch sync
            self._mp_epoch = None
            self._mp_epochs = None
        self._mask_stamp = -1
        self._sync_every = 16
        self._sync_tick = 0
        self._sync_fail_warned = False
        # Seed for the per-epoch selections. Derived from the image-file LIST so that (a) it is stable
        # across processes -- main and every worker rebuild the same selection with no transport -- and
        # (b) two datasets over the same files agree. Consequence worth knowing: MOVING or renaming a
        # dataset changes the seed, and therefore which images each epoch selects. That is harmless (it
        # is a seed, not a contract) but it does mean a snapshot-diff tool has to rebuild its dataset at
        # the same path to be reproducible -- see tools/ooo_layout_snapshot.py::_dataset_root.
        self._mask_seed = zlib.crc32("\n".join(self.im_files).encode("utf-8", "ignore"))
        self._seg_cache: SegmentBases | None = None
        self._seg_key: tuple[int, ...] | None = None
        self._compose_warned = False
        self._cap_warned: set[str] = set()
        self._ratio_zero_warned: set[str] = set()
        self._ratio_negative_warned: set[str] = set()
        self._save_state: dict[str, list] = {}
        # NOTE: `self.prefetch_factor` is deliberately NOT set here. One knob, one writer: the augment
        # assembly mirrors it onto the dataset (`augment_setup.v8_transforms` ->
        # `dataset.prefetch_factor = _hyp_get(hyp, "prefetch_factor")`), and that has already run by now
        # because it lives in build_transforms, called from super().__init__(). The grouped loader tail in
        # pool/sampler.py is the only READER (`getattr(dataset, "prefetch_factor", ...)`). Scope reminder:
        # it only takes effect on the grouped path -- stock build_dataloader hardcodes 4 and takes no
        # argument for it -- so the sampler warns once when a non-default value cannot be applied instead
        # of dropping it silently.
        self.slice_grouped_sampler = bool(
            getattr(hyp, "slice_grouped_sampler", _online_default("slice_grouped_sampler"))
        )
        self._raw_hits = 0
        self._raw_misses = 0
        try:
            self._mp_raw_hits = multiprocessing.Value("q", 0)
            self._mp_raw_misses = multiprocessing.Value("q", 0)
        except Exception:  # noqa: BLE001 -- shared memory may be unavailable; LRU stats become local-only
            self._mp_raw_hits = None
            self._mp_raw_misses = None
        self._raw_reported = (0, 0)
        # cache='ram' cannot serve the online branches (they read ORIGINAL-resolution frames).
        # ``float(... or 0.0)`` also covers a bool (``True`` == 1.0, ``False`` -> ``0.0`` via ``or``); the
        # AUTHORITATIVE read of slice_prob -- type normalisation, range check and the "0<p<1" warning --
        # is ``augment_setup._resolve_slice_prob``, which has already run by now (it lives in
        # ``v8_transforms``, called from ``build_transforms`` inside ``super().__init__()``). This site
        # only needs the "is slicing on at all" bit, so it stays a plain comparison.
        if self.cache == "ram" and float(getattr(hyp, "slice_prob", 0.0) or 0.0) > 0.0:
            LOGGER.warning(
                f"{self.prefix}cache='ram' cannot accelerate online slicing: degrading to cache=False. "
                "Raise slice_raw_cache_size to speed up decoding instead."
            )
            self.cache = None
        # Report the expanded sample count (train only). The per-branch breakdown is read straight off
        # ``_segment_lengths`` -- the same single source the pool is actually built from -- because the
        # static per-image wording this used to print ("4 slices + 1 ratio + ... per image") described
        # the SWITCHES, not the layout: it claimed 4 slices per image even when slice_ratio selected
        # only a few images, and N/4 compose even when compose_ratio selected no group at all.
        if self.augment:
            _lens = self._segment_lengths()
            # Display names for the 7 segments, in layout order. NOTE the first one is printed as "sahi"
            # (Slicing Aided Hyper Inference) while the code keeps calling that segment "base" everywhere
            # -- ``_segment_bases().base``, ``_base_slot()``, ``base_len``. They are the same thing: the
            # tiles produced by the slice_transform pipeline. The log says "sahi" because "base" reads as
            # "baseline / un-augmented", which is exactly what this segment is NOT -- the un-augmented
            # whole frame lives in the "origin" segment instead.
            _names = ("sahi", "origin", "ratio", "blur", "compose", "weather", "occlusion")
            LOGGER.info(
                f"{self.prefix}Online augment: {sum(_lens)} training samples from {self.ni} images "
                f"(segment slots: " + ", ".join(f"{nm} {ln}" for nm, ln in zip(_names, _lens)) + ")"
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
                except Exception as e:  # noqa: BLE001 -- any corrupt/unreadable .npy falls back to a fresh imread
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
                    # would mix two index spaces -- e.g. slicing off + compose/blur/ratio/img_origin on.
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
        chosen each epoch (``round(x * ceil(N/4))`` groups for compose). Those images' slots in the
        branch's segment get the augmentation; the un-selected images simply own no slot in that
        segment (``_segment_lengths`` sizes it to the configured ratio), so ``len`` is constant and
        there is nothing to "fall back" to. A branch that is off, or whose ratio is >= 1, keeps the
        "all slots" sentinel (``None``). ``close_aug_epoch``: during the final N epochs every builder
        emits the plain original while the line-up keeps its shape.
        """
        if self._mp_epoch is not None:
            try:
                self._mp_epoch.value = int(epoch)
                self._mp_epochs.value = int(epochs) if epochs is not None else -1
            except Exception as e:  # noqa: BLE001 -- worker process: workers rebuild masks via _sync instead
                LOGGER.debug(f"set_epoch: shared-memory epoch update skipped in this process: {e}")
        self._log_raw_cache_stats()
        self._rebuild_epoch_masks(epoch, epochs)
        self._log_mask_summary(epoch, epochs)

    def _rebuild_epoch_masks(self, epoch: int, epochs: int | None = None) -> None:
        """(Re)draw this epoch's per-branch selection sets and reset OnlineSlice's counters.

        Deterministic per (dataset, epoch, epochs): the selections are drawn from an RNG seeded ONLY by
        ``(_mask_seed, epoch, epochs)`` -- NOT from the global ``random`` stream -- so the main process
        and every DataLoader worker rebuilding the same epoch derive IDENTICAL selections with no
        cross-process transport. A worker that missed epochs and rebuilds late therefore still produces
        exactly the selections of the current epoch. ``_mask_stamp`` is bumped first: the rebuild is
        idempotent for a given (epoch, epochs) pair.

        The DRAWN SET is the only thing that varies per epoch -- ``len(_sel_*)`` is
        ``_ratio_K(ratio_attr, count)``, which reads the configured ratio alone. That separation is what
        keeps the pool length constant for the whole run (see ``_segment_lengths``): a branch's segment
        holds exactly as many slots as it has selected images, so an un-selected image never needs a
        "fall back to the plain original" slot at all.

        Same images, same epoch: the draw is ``rng.sample(range(count), k)`` in the SAME branch order
        and under the SAME conditions as the boolean-mask version it replaces (draw only when
        ``0 < x < 1``; ``x >= 1`` or a negative ratio keeps the "all slots" sentinel). Upgrading to
        ratio-sized segments therefore changes the POOL SIZE without changing WHICH images a
        sub-1.0 ratio augments -- which is what makes "the only difference is the removed duplicates"
        a checkable claim.
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
        # --- close_aug_epoch: final N epochs disable all online augmentation (every slot -> original).
        # The SEGMENT SHAPE is deliberately left alone: ``_segment_lengths`` is ratio-derived and must
        # not move mid-run (mosaic buffer indices, nb, the sampler's units). Instead the selections are
        # the identity and every builder short-circuits on ``self._closing`` -- so those slots stay in
        # place and emit plain originals. That is the documented cost of keeping len() constant.
        close_epoch = int(getattr(self, "close_aug_epoch", 0))
        self._closing = bool(
            aug_on and close_epoch > 0 and epochs is not None and epoch >= epochs - close_epoch
        )
        # NOTE: the closing window deliberately does NOT skip the selection draw below.
        #
        # It used to publish ``_sel_* = list(range(K))`` here, reasoning that "no builder augments this
        # epoch, so only the LENGTH of the selection has to match the layout". The LENGTH half was right;
        # the rest was not. A slot in the closing window still resolves WHICH image it shows (every
        # builder emits the plain original OF ITS SLOT'S IMAGE), and the set of images an epoch can see
        # at all is the UNION of these per-branch selections. With every image-level branch taking the
        # same ``range(K)`` prefix, that union was exactly the first ``max(K_x)`` images by file order --
        # and the SAME prefix for every closing epoch, because ``range(K)`` ignores the rng entirely.
        # MEASURED at train.py's ratios (all 0.5) on the 8-image mini set: 4 of the 8 images present
        # (7/6/6/6 slots each), the other 4 absent from the whole window; see
        # ``_perf_review/closing_probe.py`` and 更新说明.md §19.
        #
        # The draw below already produces exactly a rotation: ``sorted(rng.sample(range(count), K))``,
        # seeded from ``f"{_mask_seed}:{epoch}:{epochs}"``, so it is reproducible in every worker and
        # varies per epoch. Over a window of >= N/K epochs it reaches every image.
        #
        # The LAYOUT is still pinned while this runs -- it comes from ``_ratio_K`` (the CONFIGURED
        # ratio), never from the draw, so ``len(dataset)`` cannot move. ``self._closing`` keeps doing its
        # real work in the builders (emit the original), in ``_apply_target_tile_schedule`` (no
        # scheduling) and in the base segment's slice gate.
        # --- 表驱动 : 六条增强分支共享同构的"选择集 = round(x*count) 张随机原图"逻辑。
        # 差异点只有: 开关谓词 / 比例属性 / 样本数 (compose 是组级 (n+3)//4, 其余原图级 n)。
        for attr, ratio_attr, on, count in self._mask_specs(n):
            if not on:
                setattr(self, f"_sel_{attr}", [])  # branch off -> its segment holds no slots
                continue
            if not aug_on:
                # Not augmenting (train-mode dataset without the online assembly): keep the sentinel so
                # the layout's slots are all "selected", which is what the mask=None version did.
                setattr(self, f"_sel_{attr}", None)
                continue
            x = float(getattr(self, ratio_attr, _online_default(ratio_attr)))
            self._warn_negative_ratio(attr, ratio_attr, x)
            if 0.0 <= x < 1.0:
                k = int(round(x * count))
                if x > 0:
                    # NOTE: sample() must be called even when k == 0. It advances the shared ``rng``, so
                    # skipping it would shift the selection of every LATER branch in this same rebuild --
                    # a silent change to branches that were working correctly. (sorted() only fixes the
                    # ORDER inside the segment; the selected set is exactly ``rng.sample``'s.)
                    sel = sorted(rng.sample(range(count), k))
                    if k == 0:
                        self._warn_ratio_selects_nothing(attr, ratio_attr, x, count)
                else:
                    sel = []
                setattr(self, f"_sel_{attr}", sel)
            else:
                # ratio >= 1 (or negative) -> every slot is augmented.
                setattr(self, f"_sel_{attr}", None)
        # Runs AFTER the table above because it REPLACES the slice branch's image-level draw with a
        # tile-level schedule (see _apply_target_tile_schedule). It draws from its own seed, so the
        # table's shared ``rng`` sequence -- and therefore every other branch's selection -- is
        # byte-identical whether or not this knob is on.
        self._apply_target_tile_schedule(epoch, n)

    def _warn_negative_ratio(self, attr: str, ratio_attr: str, x: float) -> None:
        """Warn ONCE per branch for a negative ``*_ratio``, which silently means "100%".

        ``x < 0`` is not a documented mode for these six knobs (``slice_background_ratio`` uses -1 as its
        own sentinel, so ``-1`` is an easy copy-paste), and the historical code let it fall out of the
        ``0 <= x < 1`` test straight into the "all slots" branch. Keep that behaviour so no run changes
        meaning under us, but stop it from being invisible.
        """
        if x >= 0 or attr in self._ratio_negative_warned:
            return
        self._ratio_negative_warned.add(attr)
        LOGGER.warning(
            f"{self.prefix}{ratio_attr}={x:g} is negative, which means 'augment EVERY slot' "
            f"(ratio >= 1) -- not 'off'. Use 0 to disable just this branch, or its *_keep switch."
        )

    def _warn_ratio_selects_nothing(self, attr: str, ratio_attr: str, x: float, count: int) -> None:
        """Warn ONCE per branch when a strictly positive ``*_ratio`` rounds down to zero slots.

        ``K = round(ratio * count)`` where ``count`` is the number of ORIGINAL images (or compose
        groups). A low ratio on a small dataset therefore silently selects NOTHING: the branch augments
        no sample at all, and (with ratio-sized segments) it allocates no slots either -- the pool
        simply has no blur / weather / occlusion / ratio / compose content in it. e.g. ``ratio=0.1``
        selects ``round(0.8) == 1`` on the shipped 8-image mini set but ``round(0.4) == 0`` on a 4-image
        set. A silent zero is the failure class this module keeps fighting, so say it out loud.
        """
        if attr in self._ratio_zero_warned:
            return
        self._ratio_zero_warned.add(attr)
        LOGGER.warning(
            f"{self.prefix}{ratio_attr}={x:g} selects round({x:g} x {count}) = 0 slots, so the '{attr}' "
            f"branch augments NOTHING for this dataset (its segment is empty and no sample of that kind "
            f"reaches training). Raise {ratio_attr} so that {ratio_attr} x {count} >= 0.5, or turn the "
            f"branch off via its *_keep switch if that is what you meant."
        )

    def _log_mask_summary(self, epoch: int, epochs: int | None) -> None:
        """Log how many slots each branch ACTUALLY augments this epoch (main process, once per epoch).

        This is the observability counterpart of the ``*_ratio`` semantics, and it exists because its
        absence let a real defect hide: ``ratio_pad_ratio`` / ``blur_ratio`` / ``compose_ratio`` were
        never mirrored onto the dataset, so their selection was always the "all" sentinel and those
        branches ran on 100% of images while the config said 10% -- with nothing in the log to
        contradict the config. The counts below are the ACTUAL selected images, so a mis-wired knob, a
        silently-zeroed ratio (flagged ``<-- NONE``) and the ratio/pool-length interaction are all
        visible at a glance:

            all/<count>  ratio >= 1 -> every slot augmented (this is the pre-fix behaviour of the
                         three mis-wired knobs)
            sel/<count>  ratio < 1  -> the branch's segment holds ``sel * multiplier`` slots and the
                         other ``count - sel`` images own no slot in it
        """
        if not self.augment:
            return
        n = len(self.labels)
        close_epoch = int(getattr(self, "close_aug_epoch", 0))
        closing = close_epoch > 0 and epochs is not None and epoch >= epochs - close_epoch
        parts = []
        for attr, ratio_attr, on, count in self._mask_specs(n):
            if not on:
                continue  # branch off -> allocates no slots at all
            x = float(getattr(self, ratio_attr, _online_default(ratio_attr)))
            sel = getattr(self, f"_sel_{attr}", None)
            if sel is None:
                parts.append(f"{attr} all/{count} (ratio {x:g})")
            else:
                part = f"{attr} {len(sel)}/{count} (ratio {x:g})"
                if not sel and not closing:  # during close_aug_epoch a zero is intentional
                    part += "  <-- NONE"
                parts.append(part)
        # With the target-tile schedule on, the "slice" line above counts (image, tile) UNITS rather than
        # images. Which pass/block this epoch is cannot be inferred from the config, and "why am I seeing
        # this tile again" is otherwise unanswerable from the log -- so print it.
        if self._tile_units is not None:
            queue = self._target_tile_queue()
            n_blocks = max(1, -(-len(queue) // max(1, len(self._tile_units))))
            pass_idx, block = divmod(int(epoch), n_blocks)
            parts.append(
                f"target tiles: {len(self._tile_units)} units of {len(queue)} "
                f"(block {block + 1}/{n_blocks}, pass {pass_idx + 1})"
            )
        if not parts:
            return
        head = f"augment masks @ epoch {epoch}"
        if closing:
            head += " [close_aug_epoch: every slot emits the original]"
        LOGGER.info(f"{self.prefix}{head}: " + " | ".join(parts))

    def _mask_specs(self, n: int) -> list[tuple[str, str, bool, int]]:
        """Table-driven branch definition -- the SINGLE source for both layout and selection.

        Each row: (segment/selection attr suffix, ratio attr, branch-on flag, slot base count).
        ``count`` is the number of selectable items: the ORIGINAL images (``N``) for every branch
        except compose, which is GROUP-level (``ceil(N/4)`` groups of four). The same table drives
        ``_segment_lengths`` (how many slots the branch owns), ``_rebuild_epoch_masks`` (which images get
        them) and ``_log_mask_summary``, so a new branch is one row and cannot be half-wired: that is
        precisely the failure the previous revision shipped -- ``ratio_pad_ratio`` / ``blur_ratio`` /
        ``compose_ratio`` were never mirrored by ``v8_transforms``, so ``x`` silently resolved to the 1.0
        fallback and those branches ran on EVERY image EVERY epoch (measured: 100% instead of the
        configured 10%, while slice/weather/occlusion honoured the ratio).

        Every ``ratio_attr`` listed here MUST be mirrored onto the dataset by ``v8_transforms``; the
        ``getattr(..., _online_default(...))`` fallback is only for datasets built WITHOUT
        v8_transforms (direct construction, tests). ``tests/test_ooo_branches.py`` asserts this
        mechanically.
        """
        return [
            ("slice", "slice_ratio", self._slice_on(), n),
            ("ratio", "ratio_pad_ratio", bool(getattr(self, "ratio_pad_keep", _online_default("ratio_pad_keep"))), n),
            ("blur", "blur_ratio", bool(getattr(self, "blur_keep", _online_default("blur_keep"))), n),
            ("compose", "compose_ratio", self._compose_on(), (n + 3) // 4),
            ("weather", "weather_ratio", self._weather_on(), n),
            ("occlusion", "occlusion_ratio", self._occlusion_on(), n),
        ]

    def _slice_on(self) -> bool:
        """True when the base segment runs the slicing pipeline (``slice_transform`` attached)."""
        return getattr(self, "slice_transform", None) is not None

    def _ratio_K(self, ratio_attr: str, count: int, on: bool = True) -> int:
        """Number of items the branch selects -- and therefore owns slots for -- from the ratio ALONE.

        This is the layout count used by ``_segment_lengths``, and it is deliberately computed from the
        CONFIGURED ratio rather than from ``len(_sel_*)``: the pool length must be knowable before the
        first epoch is ever published (the sampler builds its units at loader construction, and the
        trainer logs ``len(dataset)`` at setup), and it must not move when a later epoch draws a
        different selection. ``len(_sel_*)`` always equals this value once a rebuild has happened:
        ``_ensure_epoch_sel`` guarantees a rebuild is done before any selection is consulted whenever a
        ratio is in ``[0, 1)``, and outside that range the "all slots" sentinel makes the two agree.

        ``>= 1`` and negative both mean "every slot" (the historical ``mask = None`` semantics);
        ``0`` means the branch owns no slots at all.
        """
        if not on:
            return 0
        x = float(getattr(self, ratio_attr, _online_default(ratio_attr)))
        if x < 0.0 or x >= 1.0:
            return count
        return int(round(x * count))

    def _sel_indices(self, attr: str, count: int) -> Sequence[int]:
        """The image (compose: group) indices this branch augments this epoch, in segment order.

        ``None`` -- no rebuild yet, or a ratio >= 1 -- means "every slot is augmented", returned as
        ``range(count)`` so the no-mask path stays allocation-free. A branch that is OFF gets ``[]``,
        but its segment holds no slots and no index maps into it, so that state is never consulted.
        """
        sel = getattr(self, f"_sel_{attr}", None)
        return range(count) if sel is None else sel

    def _has_partial_ratio(self) -> bool:
        """True when at least one enabled branch has a ratio in ``[0, 1)`` -- i.e. it needs a draw."""
        # The target-tile schedule moves to a NEW block every epoch even when slice_ratio >= 1 (K == N),
        # so a dataset driven without a trainer still has to publish an epoch before its base slots are
        # meaningful. Without this the "all slots" sentinel would stand in for a schedule that is
        # supposed to advance, and every epoch would emit the same first block.
        if self._target_tile_schedule_on():
            return True
        n = len(self.labels)
        for _attr, ratio_attr, on, _count in self._mask_specs(n):
            if not on:
                continue
            x = float(getattr(self, ratio_attr, _online_default(ratio_attr)))
            if 0.0 <= x < 1.0:
                return True
        return False

    def _target_tile_schedule_on(self) -> bool:
        """True when this dataset should schedule BASE slots as target-bearing (image, tile) pairs.

        All three conditions are structural, not preferences: the knob has to be on, slicing has to be
        the owner of the base segment, and the base segment has to hold exactly ONE slot per item
        (``slice_all_tiles=False``). With ``all_tiles=True`` the segment is ``4*K_slice`` and every tile
        of every selected image is emitted, so there is no tile to choose -- ``v8_transforms`` warns
        once when the knob is set in that shape.
        """
        if not bool(getattr(self, "slice_target_tiles", _online_default("slice_target_tiles"))):
            return False
        return self._slice_on() and self._n_per() == 1

    def _target_tile_queue(self) -> list[tuple[int, int]]:
        """Every ``(image, tile)`` pair that keeps at least one target, in a fixed shuffled order.

        The scheduling UNIT is the tile, not the image, because with ``slice_all_tiles=False`` a base
        slot emits exactly one tile: queuing target-bearing tiles is what makes those slots positive,
        and it is also what makes "every target-bearing tile gets its turn" expressible at all. An
        image with several targets may contribute several units (and one with none contributes zero --
        a background image has no tile worth slicing, so it reaches training through the origin segment
        or whichever augmentation branch selects it).

        Built WITHOUT decoding any image: the tile filter needs only ``(h, w)`` and the boxes, both of
        which ``labels.cache`` already holds, and ``OnlineSlice.target_tiles`` runs the very same
        ``_geometry`` the emit path runs. One pass over the label cache, once per process, then cached
        for the whole run -- the order cannot change mid-run or the "no repeats" property would break.

        The shuffle is seeded from ``_mask_seed`` (the dataset's file list) for the same reason every
        other draw is: the main process and every worker must arrive at the identical order with no
        cross-process transport.
        """
        cached = self._target_tile_queue_cache
        if cached is not None:
            return cached
        st = getattr(self, "slice_transform", None)
        units: list[tuple[int, int]] = []
        if st is not None:
            for i, label in enumerate(self.labels):
                boxes = label.get("bboxes")
                if boxes is None or len(np.asarray(boxes)) == 0:
                    continue
                shape = label.get("shape") or label.get("ori_shape")
                if shape is None:
                    continue
                units.extend((i, k) for k in st.target_tiles(label, shape, key=i))
        random.Random(f"{self._mask_seed}:target_tiles").shuffle(units)
        self._target_tile_queue_cache = units
        return units

    def _target_tile_pass(self, pass_idx: int) -> list[tuple[int, int]]:
        """The queue reshuffled for pass ``pass_idx`` -- one full sweep of every target-bearing tile.

        Re-shuffling per pass (instead of reusing one fixed order for the whole run) is what makes the
        restart the user asked for a genuine new round: after the queue is exhausted the next pass
        visits the same units in a different order, so an image is not always paired with the same
        neighbours. Seeded by ``(mask_seed, pass_idx)`` so it stays recomputable in every process.
        """
        cached = self._target_tile_pass_cache
        if cached is not None and cached[0] == pass_idx:
            return cached[1]
        perm = list(self._target_tile_queue())
        random.Random(f"{self._mask_seed}:target_tiles:pass{int(pass_idx)}").shuffle(perm)
        self._target_tile_pass_cache = (int(pass_idx), perm)
        return perm

    def _apply_target_tile_schedule(self, epoch: int, n: int) -> None:
        """Publish this epoch's BASE slots as target-bearing tiles (see ``slice_target_tiles``).

        The layout is untouched: ``base = _ratio_K('slice_ratio', n)`` in every configuration, and this
        only decides WHICH (image, tile) pair each of those slots holds. That is why ``len(dataset)``
        still cannot move and the mosaic buffer / sampler units stay valid.

        Scheduling: the queue is cut into consecutive blocks of ``K`` and one block is consumed per
        epoch, so no unit repeats until every unit has been used -- which is the "do not re-pick last
        epoch's tile" requirement -- and when the queue runs out the next PASS starts from a freshly
        shuffled order ("restart"). Block index and pass index are pure functions of the epoch, so
        workers rebuilding epoch ``e`` derive exactly this schedule with no transport. No global RNG is
        touched, so enabling the knob does NOT shift any other branch's selection.

        When ``K`` does not divide the queue length the final block of a pass wraps around to units
        already seen earlier in that same pass (they are the least recently used, which is the best
        available choice), and when ``K`` exceeds the queue length a unit necessarily repeats inside one
        epoch -- warned once, because that silently lowers variety rather than merely sharing it.
        """
        on = self._target_tile_schedule_on() and not self._closing
        if not on:
            self._tile_units = None
            return
        k_slice = self._ratio_K("slice_ratio", n, self._slice_on())
        queue = self._target_tile_queue()
        if k_slice <= 0:
            self._tile_units = None
            return
        if not queue:
            # No image in this dataset keeps a box in any tile (all-background dataset, empty label
            # files, or a filter combination that drops every box). Fall back to the blind-random base
            # segment instead of publishing a schedule that cannot be filled, and say so once.
            self._tile_units = None
            self._warn_target_tiles_unusable(
                "no (image, tile) pair in this dataset keeps a target, so there is nothing to schedule"
            )
            return
        n_blocks = max(1, -(-len(queue) // k_slice))  # ceil, i.e. blocks needed for one full pass
        if k_slice > len(queue) and not self._target_tile_warned:
            self._target_tile_warned = True
            LOGGER.warning(
                f"{self.prefix}slice_target_tiles: slice_ratio selects {k_slice} slots but the dataset "
                f"has only {len(queue)} target-bearing tiles, so units REPEAT inside one epoch "
                f"({k_slice / len(queue):.2f}x per epoch). Lower slice_ratio to <= "
                f"{len(queue) / max(n, 1):.4g} to give every slot a distinct tile."
            )
        pass_idx, block = divmod(int(epoch), n_blocks)
        perm = self._target_tile_pass(pass_idx)
        base = (block * k_slice) % len(queue)
        self._tile_units = [perm[(base + j) % len(queue)] for j in range(k_slice)]
        # Keep `_sel_indices("slice")` truthful: it is the base segment's "which image owns slot i" map
        # (grouped_sample_units reads it that way), and the schedule REPLACES the image-level draw.
        self._sel_slice = [img for img, _k in self._tile_units]

    def _warn_target_tiles_unusable(self, detail: str) -> None:
        """Warn ONCE when ``slice_target_tiles`` is set but the schedule cannot be built."""
        if self._target_tile_empty_warned:
            return
        self._target_tile_empty_warned = True
        LOGGER.warning(
            f"{self.prefix}slice_target_tiles=True but {detail}; falling back to one blind random tile "
            "per selected image (the slice_all_tiles=False behaviour)."
        )

    def _ensure_epoch_sel(self) -> None:
        """Draw this epoch's selections if nothing has published an epoch yet (main process only).

        ``set_epoch`` normally does this, but a dataset used WITHOUT a trainer -- tests, the tools in
        ``tools/``, and the sampler building its units at loader-construction time -- would otherwise keep
        the "all slots" sentinel while ``_segment_lengths`` has already sized the segments to the
        configured ratio. The two would disagree: the segment would claim ``K`` slots but the mapping
        would hand out ``count`` of them, i.e. indices past the segment's end.

        Workers must NOT self-build: they would stamp epoch 0 without the total ``epochs`` (breaking
        ``close_aug_epoch``) and pin the shared channel, so they wait for the trainer's publish through
        ``_sync_epoch_masks`` instead. Same reasoning as the lazy build this replaces.
        """
        if self._mask_stamp >= 0 or multiprocessing.parent_process() is not None:
            return
        if not self._has_partial_ratio():
            return  # every branch is "all slots": the sentinel already agrees with the layout
        self._rebuild_epoch_masks(int(getattr(self, "epoch", 0)))

    def _base_slot(self, index: int) -> tuple[int, int | None, bool]:
        """Map a BASE-segment index to ``(original image index, tile index, takes the slice path)``.

        The base segment is ``n_per * K_slice`` slots: each of the ``K_slice`` images selected for slicing
        owns ``n_per`` CONSECUTIVE slots (so ``index // n_per`` is its position in the selection and
        ``index % n_per`` its tile, exactly the pre-refactor arithmetic). There is NO plain tail: an
        un-selected image owns no base slot at all -- it only appears via the IMG_ORIGIN segment (if on)
        or via whatever augmentation branch selected it. Slicing off gives ``K_slice == 0``, i.e. an
        empty base segment, so this is never reached for a valid index; ``slice_all_tiles=False`` does
        NOT empty it (that shape is ``K_slice`` slots wide, one per selected image).

        With the target-tile schedule on (``slice_target_tiles``), the ``n_per == 1`` case takes its
        (image, tile) pair from ``_tile_units`` instead: the tile is then already decided and the
        transform is handed it, rather than picking one at random.
        """
        n_per = self._n_per()
        sel = self._sel_indices("slice", len(self.labels))
        if n_per > 1:
            return sel[index // n_per], index % n_per, True
        units = self._tile_units
        if units is not None:
            # Target-tile schedule: the slot already knows its tile (see _apply_target_tile_schedule).
            # ``_sel_indices("slice")`` is kept in sync with these units, but the tile is only available
            # here -- which is the whole point of the schedule.
            img, k = units[index]
            return int(img), int(k), True
        return sel[index], None, True

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
        except Exception as e:  # noqa: BLE001 -- shared memory torn down mid-run; warn once and skip this poll
            self._warn_sync_failure(f"reading the shared epoch failed ({e})")
            return
        if epoch < 0 or epoch == self._mask_stamp:
            return
        ev = self._mp_epochs
        try:
            epochs = ev.value if ev is not None else -1
        except Exception as e:  # noqa: BLE001 -- same as above; fall back to "unknown total" (-1)
            self._warn_sync_failure(f"reading the shared total-epochs failed ({e})")
            epochs = -1
        self._rebuild_epoch_masks(epoch, epochs if epochs >= 0 else None)

    def _warn_sync_failure(self, detail: str) -> None:
        """Warn ONCE per process when the shared epoch publish cannot be read.

        Both reads used to swallow every exception silently. The per-epoch masks (slice_ratio /
        blur_ratio / compose_ratio / weather_ratio / occlusion_ratio and close_aug_epoch) change ONLY
        through this poll, so a silent failure pins every worker to whatever masks it last built: the
        augmentation ratios simply stop varying per epoch and close_aug_epoch never fires, with no
        symptom whatsoever. Warn once -- this runs on the per-sample path, so it must not spam.
        """
        if getattr(self, "_sync_fail_warned", False):
            return
        self._sync_fail_warned = True
        LOGGER.warning(
            f"{self.prefix}epoch-mask sync failed ({detail}); this process keeps using stale per-epoch "
            "masks, so slice_ratio / blur_ratio / compose_ratio / weather_ratio / occlusion_ratio and "
            "close_aug_epoch stop taking effect."
        )

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

    def _img_origin_on(self) -> bool:
        """True when the un-sliced ORIGIN segment allocates samples: the unified IMG_ORIGIN switch.

        Replaces the old per-slicing ``slice_keep_origin``. IMG_ORIGIN puts EVERY original image into the
        pool as one whole-frame slot, independent of slicing and of any augmentation branch. It is the
        single coverage knob:

        * with it ON, every image appears at least once per epoch (as its own whole frame);
        * with it OFF, an image that no augmentation branch selected is simply absent from the pool --
          which is exactly the "discard the un-selected" behaviour the ratio-sized layout enables.

        Unlike the old keep_origin it does NOT require slicing and is not limited to the sliced images:
        the segment is width ``N`` (one per original), not ``K_slice``. ``slice_keep_origin`` is still
        accepted as a deprecated alias by ``augment_setup.v8_transforms``.
        """
        return bool(getattr(self, "img_origin", _online_default("img_origin")))

    def _img_origin_len(self) -> int:
        """Number of slots in the origin segment: one per ORIGINAL image when img_origin is on (see _img_origin_on)."""
        if not self._img_origin_on():
            return 0
        return len(self.labels)

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

    def _begin_branch_label(self, index: int, img_index: int) -> dict[str, Any]:
        """Open a pooled branch sample: fresh label copy, ``im_file`` restored, mosaic buffer touched.

        Shared head of the online branches (blur / weather / occlusion). ``index`` is the EXPANDED
        mixed-pool index (for correct Mosaic buffer bookkeeping); ``img_index`` is the ORIGINAL image
        index this sample derives from. The branch fills in the image and any label edits, then calls
        ``_finish_branch`` / ``_save_branch``. This sequence used to be copy-pasted into every branch.
        """
        label = deepcopy(self.labels[img_index])
        label.pop("shape", None)  # shape is for rect, remove it
        label["im_file"] = self.im_files[img_index]
        # Keep the sample on the same Mosaic mix pool as every other sample (cache != 'ram')
        self._touch_buffer(index)
        return label

    def _save_branch(self, branch: str, label: dict[str, Any], img: np.ndarray, *, save_dir, save_tag: str,
                     key: tuple) -> None:
        """Shared save block of the online branches.

        ``save_dir`` is the branch's resolved directory, or a falsy value to skip saving entirely (the
        per-branch ``*_save_dir`` keys ARE the switches). Annotation follows the global
        ``slice_save_annotated``; the cap follows ``slice_save_max_<branch>`` (falling back to
        ``slice_save_max``). Saving deliberately does not depend on the slicing pipeline
        (``slice_transform`` may be None when ``slice_prob=0``).
        """
        if not save_dir:
            return
        self._save_annotated(
            branch, Path(save_dir), img,
            label.get("bboxes", np.empty((0, 4))), label.get("cls", np.empty((0, 1))),
            save_tag, key,
            bool(getattr(self, "slice_save_annotated", _online_default("slice_save_annotated"))),
        )

    def _finish_branch(self, label: dict[str, Any], img: np.ndarray) -> dict[str, Any]:
        """Close a pooled branch sample: attach the image and resize it to the training size."""
        label["img"] = np.ascontiguousarray(img)
        return self._finalize_label(label, img)

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

        Layout -- RATIO-SIZED segments (each optional branch is an independent, contiguous segment
        gated ONLY by its own switch; slicing lives entirely inside the base segment). With
        ``K_x = _ratio_K(<x>_ratio, count_x)`` the number of items branch ``x`` selects:
            [0, base_len)                 base:      n_per*K_slice slots -- TILES for the SELECTED only
            [base_len, +N)                origin:    one un-sliced whole frame per ORIGINAL image (img_origin)
            [base_len+N, +K_ratio)        ratio:     1 padded image per SELECTED image
            [.., +2*K_blur)               blur:      short + long blurred image per SELECTED image
            [.., +K_compose)              compose:   one 2x2 stitched image per SELECTED group of 4
            [.., +K_weather)              weather:   1 rain/haze/noise-degraded image per SELECTED image
            [.., +K_occlusion)            occlusion: 1 rect/stripe-occluded image per SELECTED image

        The base segment holds TILES for the ``K_slice`` selected images only -- ``n_per`` consecutive
        slots each (``4`` under ``slice_all_tiles``, ``1`` otherwise). There is deliberately NO plain tail
        for the un-selected images: an image that ``slice_ratio`` did not pick owns no base slot. Its only
        path back into the pool is the unified IMG_ORIGIN segment (when ``img_origin`` is on) or whichever
        augmentation branch selected it. That is the whole point -- "the un-selected half gets discarded"
        is a first-class, supported configuration, and the coverage knob for it is ``img_origin``, NOT a
        per-branch fallback. Nothing has to "fall back to the plain original", so un-augmented content
        cannot pile up.

        ``origin`` is the unified coverage segment: ``N`` whole-frame slots (one per ORIGINAL image) when
        ``img_origin`` is on, ``0`` otherwise. It replaces the old ``slice_keep_origin`` (which covered
        only the sliced images). With it on every image appears at least once per epoch; with it off an
        image no augmentation branch selected is simply absent from the pool.

        Two self-consistency boundaries fall out of the definition (both asserted by the tests):
            K == 0  (ratio 0 / branch off) -> base = 0: no image is sliced, so the pool is just the
                    IMG_ORIGIN segment (N whole frames) -- equivalent to "slicing disabled" for coverage.
            K == N  (ratio >= 1)           -> base = 4N (all images sliced) and every branch keeps its old
                    width; with img_origin on the layout is 4N + N + the branches, i.e. the pre-refactor
                    shape plus the whole-frame origin segment -- which is what makes the change adoptable
                    incrementally (see tools/ooo_layout_snapshot.py, and test_ooo_branches.py's
                    bit-identical check).

        ``_segment_bases`` derives BOTH the cumulative boundaries and ``total`` from this list, and
        uses ``tuple(...)`` of it as the ``_seg_cache`` version stamp. The layout is epoch-INDEPENDENT
        (``_ratio_K`` reads the configured ratio, never the drawn selection), so ``len(dataset)`` is
        knowable before the first epoch is published and never moves during a run -- which is what keeps
        the mosaic buffer's indices, ``nb``/``nw`` and the grouped sampler's units valid. The inputs are
        the branch table's ratios/switches plus ``_n_per``/``_img_origin_on``; adding a segment is a
        one-line change that invalidates the cache automatically (see ``_segment_bases``).
        """
        n = len(self.labels)
        n_per = self._n_per()
        k = {attr: self._ratio_K(ratio_attr, count, on)
             for attr, ratio_attr, on, count in self._mask_specs(n)}
        return [
            n_per * k["slice"],      # base: slicing tiles for the SELECTED images only (no plain fallback)
            self._img_origin_len(),  # origin: one whole frame per ORIGINAL image (unified coverage)
            k["ratio"],  # ratio
            2 * k["blur"],  # blur (short + long)
            k["compose"],  # compose (groups of 4)
            k["weather"],  # weather
            k["occlusion"],  # occlusion
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

        Cost (measured, 8-original all-branches ratio=1.0 pool): 6.0 us/call on the ratio-sized layout
        vs 2.5 us/call before it -- the difference is that the segment lengths now come from the branch
        table (ratio + switch per branch) instead of a bare switch test, which is what makes the ratios
        size their own segments. In absolute terms it is still noise: the stamp runs once per
        ``get_image_and_label``, so it costs ~0.15 s CPU per epoch at the ratio-sized 24k-sample pool of
        a 8520-image set (ratio 0.1) and ~0.5 s at the legacy 87k-sample ratio=1.0 shape -- against an
        epoch that decodes every sample at original resolution. Keep the stamp DERIVED from
        ``_segment_lengths``; never hand-maintain a second list of switches here -- that would move the
        drift, not remove it.
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

        The pool ALREADY clusters sub-samples of one image in the index dimension (``n_per`` consecutive
        slots per selected image in the base segment, contiguous runs in the other segments), but the
        trainer's global shuffle destroys that adjacency: with N images the siblings of a sub-sample end
        up ~``6.5 * N`` samples away, far beyond the raw LRU's capacity of 4, so every sub-sample
        re-decodes its original JPEG. This method rebuilds the adjacency in the ORDER dimension: it
        returns units where ``units[u][j]`` is the list of pool indices that decode one source image, so
        a sampler walking a unit round-robin keeps all of its source images resident.

        Layout rules -- these MUST mirror ``_segment_bases`` / ``get_image_and_label`` (the returned
        layout is validated against ``total`` below, and the guard turns a mismatch into a warning
        instead of silently dropping samples). Every segment fans its slots out through its OWN
        selection array, in the same order ``get_image_and_label`` walks them:
            base      ``n_per`` consecutive indices per SELECTED image (NO plain slot for un-selected)
            origin    one index per ORIGINAL image (img_origin coverage)
            ratio     one index per SELECTED image
            blur      two indices per SELECTED image (short, long)
            weather   one index per SELECTED image
            occlusion one index per SELECTED image
            compose   ONE index per SELECTED GROUP of four images; it reads all four, so it is emitted
                      first inside its unit and thereby primes the LRU for the whole unit

        Units are consecutive blocks of four images, which is also the compose group size, so a compose
        sample never straddles a unit boundary and every image owns exactly one block.

        The units are built from THIS epoch's selection. A later epoch draws a different selection, which
        permutes the slot -> image mapping inside each segment but never the SET of pool indices, so the
        units stay a valid permutation (no index is dropped or duplicated) and only the decode-locality
        hint goes slightly stale -- a reordering is a pure reordering either way.

        Returns ``None`` when grouping cannot pay off, in which case the caller keeps the plain
        shuffle:
            * the dataset is not augmenting, or the raw LRU is off -- there is nothing to reuse;
            * ``n_per == 1`` and no extended segment -- every image maps to exactly ONE pool index, so
              there is no repeated decode to absorb, and a block-ordered stream would only narrow
              Mosaic's recent-sample window for no gain;
            * no image owns more than one slot this epoch -- same reason, computed from the fan-out
              actually drawn below rather than from the switch list (a branch can be ON and still
              select nothing, e.g. a ratio that rounds down to zero).

        There is deliberately no OTHER disqualifying condition. In particular a cache larger than the
        busiest image's fan-out does NOT switch grouping off: see the note above the fan-out test.
        """
        if not self.augment or self._raw_cache_size <= 0:
            return None
        n = len(self.labels)
        n_per = self._n_per()
        if n < 1 or (n_per <= 1 and not self._extended_pool_on()):
            return None
        self._ensure_epoch_sel()  # the fan-out below reads the selections; they must be drawn
        segment_bases = self._segment_bases()
        per_image: list[list[int]] = [[] for _ in range(n)]
        # The branch gates are read from ``_mask_specs`` -- the SAME table ``_segment_lengths`` sizes the
        # segments from -- instead of being re-derived here. That duplication is exactly what let
        # compose fall out of this method's gate list while the other five branches kept theirs: a branch
        # that is OFF owns zero slots, but its selection is still the ``None`` sentinel until an epoch is
        # published, and ``_sel_indices`` turns ``None`` into ``range(count)``. That state is the NORMAL
        # one at loader-construction time (the trainer builds the data loaders before the first
        # ``on_train_epoch_start``), so the missing gate did not degrade grouping occasionally -- it
        # killed it for the whole run, with a warning that blamed a layout drift. One table, one gate.
        on = {attr: flag for attr, _ratio_attr, flag, _count in self._mask_specs(n)}
        on["origin"] = self._img_origin_on()  # the coverage segment has its own switch, not a ratio

        k_slice = self._ratio_K("slice_ratio", n, on["slice"])
        for pos, img in enumerate(self._sel_indices("slice", n)):
            if pos >= k_slice:
                break  # defensive: _sel_indices is range(n) for "all", which is exactly k_slice == n
            per_image[img].extend(range(pos * n_per, (pos + 1) * n_per))
        if on["origin"]:
            # Unified coverage: one whole-frame slot per ORIGINAL image (offset == image index).
            for img in range(n):
                per_image[img].append(segment_bases.origin + img)
        if on["ratio"]:
            for j, img in enumerate(self._sel_indices("ratio", n)):
                per_image[img].append(segment_bases.ratio + j)
        if on["blur"]:
            for j, img in enumerate(self._sel_indices("blur", n)):
                per_image[img].extend((segment_bases.blur + 2 * j, segment_bases.blur + 2 * j + 1))
        if on["weather"]:
            for j, img in enumerate(self._sel_indices("weather", n)):
                per_image[img].append(segment_bases.weather + j)
        if on["occlusion"]:
            for j, img in enumerate(self._sel_indices("occlusion", n)):
                per_image[img].append(segment_bases.occlusion + j)

        # --- the ONLY disqualifying condition: nothing to reuse ------------------------------------
        # An image with a single slot has no repeated decode to absorb, and grouping it would only
        # narrow Mosaic's recent-sample window for no gain.
        #
        # REMOVED GUARD: this used to be ``if fan_out < self._raw_cache_size: warn + return None``, i.e.
        # a cache LARGER than the busiest image's fan-out switched grouping OFF. That compared the wrong
        # two things -- "grouped sampler with a SMALL cache" against "plain shuffle with a BIG cache" --
        # and then recommended LOWERING the cache. Its evidence (24-image pool, 80.0% -> 60.0% hit at
        # cache 16 vs 4) came from a pool no bigger than the Mosaic window, where the entire dataset fits
        # in the cache and capacity cannot matter; it does not extrapolate. Measured at real scale (240
        # images, same test harness, real DataLoader): cache 16 grouped 41.0 items/s vs cache 16 shuffled
        # 34.4; cache 32 grouped 44.3 vs 37.7 -- grouping wins at the very cache size the old guard
        # switched it off at. Units hold <= 4 images and are walked round-robin, so every re-read of a
        # unit's image hits as soon as the LRU holds those <= 4 images, which the documented floor
        # (``slice_raw_cache_size >= 4``) guarantees. There is therefore no cache size at which turning
        # grouping off is the better choice, so the guard is deleted rather than inverted.
        fan_out = max((len(b) for b in per_image), default=0)
        if fan_out < 2:
            return None

        units: list[list[list[int]]] = []
        # Units are ALWAYS consecutive blocks of four images, so every image owns exactly one block
        # (no index can be emitted twice -- the check below enforces that).
        compose_base = segment_bases.compose
        # compose is gated HERE too (see the ``on`` table above): with compose OFF its segment is zero
        # slots wide, so an ungated lookup would hand out ``range(ceil(N/4))`` indices that land on / past
        # the segments that follow and fail the count check below.
        compose_slot = (
            {g: compose_base + j for j, g in enumerate(self._sel_indices("compose", (n + 3) // 4))}
            if on["compose"]
            else {}
        )
        for start in range(0, n, 4):
            blocks = [per_image[i] for i in range(start, min(start + 4, n))]
            slot = compose_slot.get(start // 4)
            if slot is not None:
                # The compose sample reads its whole group at once, so it is emitted FIRST inside the
                # unit that owns image ``start``: for a complete group its four decodes are exactly this
                # unit's images and prime the LRU for every block here. The tail group wraps around to
                # earlier images, which is harmless: it costs those decodes once per epoch and the unit
                # it belongs to only holds the images that are left.
                blocks[0] = [slot, *blocks[0]]
            units.append(blocks)

        flat = sorted(index for unit in units for block in unit for index in block)
        if flat != list(range(segment_bases.total)):
            # Falling back to the plain shuffle is always CORRECT (it still visits every index once),
            # it only loses the decode-locality win -- so degrade loudly instead of crashing a run. Name
            # the branch whose selection disagrees with its own segment width: "the rules drifted" is
            # true but tells the next reader nothing, and this failure is otherwise invisible (the run
            # just gets slower).
            drift = [
                f"'{attr}': {len(self._sel_indices(attr, cnt))} indices for {self._ratio_K(ratio_attr, cnt, flag)} slots"
                for attr, ratio_attr, flag, cnt in self._mask_specs(n)
                if len(self._sel_indices(attr, cnt)) != self._ratio_K(ratio_attr, cnt, flag)
            ]
            LOGGER.warning(
                f"{self.prefix}grouped_sample_units produced {len(flat)} indices but the pool holds "
                f"{segment_bases.total}: the grouping rules no longer mirror the pool layout"
                + (f" (segment width vs selection: {'; '.join(drift)})" if drift else "")
                + ". Falling back to the plain shuffle -- correct, but decoding gets slower. Fix the "
                "fan-out rules in grouped_sample_units / _mask_specs."
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
        except Exception:  # noqa: BLE001 -- shared memory already closed (e.g. loader torn down)
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
        except Exception:  # noqa: BLE001 -- shared counters already closed; losing one epoch's stats is harmless
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
        budget = f", budget {self._raw_cache_mb:.0f} MiB" if self._raw_cache_mb > 0 else ""
        LOGGER.info(
            f"{self.prefix}raw-image LRU (slice_raw_cache_size={self._raw_cache_size}{budget}, "
            f"{order} sampler): {hits / reads:.1%} hit on {reads} reads last epoch, "
            f"{self._raw_cache_bytes / (1 << 20):.0f} MiB resident"
        )

    def _build_origin_sample(self, index: int, img_index: int) -> dict[str, Any]:
        """Build one un-sliced ORIGINAL-resolution sample (the unified img_origin segment).

        The origin segment is laid out right after the base slicing segment (see _segment_bases) and
        behaves exactly like a plain full image in the mixed pool: LRU-backed read + training resize +
        ratio_pad, and it enters the Mosaic mix pool (dataset.buffer) like every other sample.
        Nothing is written to disk.

        It carries one slot per ORIGINAL image (offset into the segment == original image index) when
        ``img_origin`` is on -- the unified coverage knob that replaces the old per-slicing
        ``slice_keep_origin`` (which only covered the sliced images). ``compose``'s close_aug_epoch path
        also calls this, for the group's first image.

        ``index`` is the EXPANDED mixed-pool index (for correct Mosaic buffer bookkeeping);
        ``img_index`` is the ORIGINAL image index this sample derives from.
        """
        label = deepcopy(self.labels[img_index])
        label.pop("shape", None)  # shape is for rect, remove it
        # Read through the per-worker LRU instead of ``load_image``.
        #
        # ``load_image`` memoises NOTHING once slicing is on: its ``self.ims`` write sits behind
        # ``getattr(self, "slice_transform", None) is None``, so with the slicing pipeline attached it
        # skips the cache entirely (verified: ``ims_resident == 0`` in every slicing configuration).
        # This segment therefore re-decoded the original JPEG *and* re-ran the training resize on every
        # visit, even when the very same frame was already resident in the LRU for the image's other
        # sub-samples. Measured exclusive share of the whole epoch: 15.1% on the full pool, 51.7% on
        # the slice-one layout, 31.8% on 4000x3000 with slicing (0.517 reads/item x 119.7 ms).
        #
        # ``_load_image_cached`` returns the same ORIGINAL-resolution frame ``load_image`` decoded
        # (copy=True, so the caller owns it), and ``_finalize_label`` applies the identical
        # resize + imgsz clamp, so resized_shape / ratio_pad cannot drift from the load_image path.
        #
        # Trade-off, measured per read: on an LRU MISS this is SLOWER than load_image by one
        # full-frame copy (2.52 ms at 1280x720, 18.17 ms at 4000x3000). It breaks even at a 44% hit
        # rate at 1280x720 and 22% at 4000x3000 -- and the shipped grouped sampler delivers 0.90-0.95,
        # so the net is a win on both sizes. Keep `slice_grouped_sampler` on (or raise
        # `slice_raw_cache_mb`) if you turn grouping off, or this can invert on large frames.
        #
        # NOTE: this path no longer consults ``load_image``'s optional *.npy disk cache. That is safe
        # -- ``_load_image_cached`` reads the JPEG directly and the npy cache was measured to add
        # nothing (see its docstring) -- but it does mean a dataset pre-baked to *.npy gets no benefit
        # for this segment.
        im = self._load_image_cached(img_index)
        # ``self.batch`` is indexed by IMAGE index (0..ni-1) and yields a batch id, so the ORIGINAL
        # index is the correct key here -- passing the expanded mixed-pool index would read past the
        # end of ``self.batch`` (and pick the wrong batch) as soon as online augmentation and rect
        # were ever allowed to coexist.
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[img_index]]
        # Keep the original on the same Mosaic mix pool as every other sample (cache != 'ram')
        self._touch_buffer(index)
        return self._finalize_label(label, im)

    def _compose_on(self) -> bool:
        """True when the compose branch allocates samples (switch on AND >= 4 originals).

        With fewer than 4 originals a group would have to reuse the same image in 2+ quadrants,
        duplicating its targets and skewing the label distribution, so compose is switched off
        entirely in that case.
        """
        return bool(getattr(self, "compose_keep", _online_default("compose_keep"))) and len(self.labels) >= 4

    def _extended_pool_on(self) -> bool:
        """True when the pool allocates any sample beyond one plain original per image.

        ``load_image``'s self-managed buffer (pure-ultralytics path) is only safe when the buffer can
        only contain original-image indices. Once ANY segment writes EXPANDED indices into the same
        buffer, ``load_image`` must not pop/clear ``ims`` with them; buffer bookkeeping is then entirely
        owned by ``get_image_and_label`` (expanded indices).

        Defined as "the layout is wider than N", which is the property that actually matters and is
        checked against the single source of truth rather than against a second list of switches. The
        list this used to be (keep_origin / ratio / blur / weather / occlusion / compose) was correct
        only while every branch allocated a full-width segment; under ratio-sized segments a branch can
        be ON and still own zero slots (``ratio 0``), and a slicing pipeline alone expands the pool
        whenever ``K_slice > 0``.
        """
        return self._segment_bases().total != len(self.labels)

    def _load_image_cached(self, img_index: int, *, copy: bool = True) -> np.ndarray:
        """Load original-resolution image, with a tiny per-worker memory LRU (no .npy disk cache).

        Unified read path used by all online-augmentation branches (slice / blur / ratio /
        compose / weather / occlusion) AND by the origin segment -- which used to go through
        ``load_image`` and therefore bypassed this cache entirely once slicing was on, re-decoding
        its frame on every visit (see ``_build_origin_sample``). Reads the original JPEG directly via ``imread``; a small in-memory LRU
        (``slice_raw_cache_size`` frames, additionally capped by the ``slice_raw_cache_mb`` byte budget)
        absorbs repeated decoding of the same image across its sub-samples. The custom .npy disk cache
        was removed -- it measured no benefit.

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
            # Evict from the LRU end until BOTH limits hold; no key-list materialisation per miss.
            # ``>= 1`` keeps the just-decoded frame even if it alone exceeds the byte budget: a cache
            # that evicts what it is about to store would decode this image again on its next slot,
            # which is strictly worse than holding one oversized frame.
            budget = self._raw_cache_budget
            if budget and not self._raw_cache_eff_logged and im.nbytes > 0:
                self._raw_cache_eff_logged = True
                eff = max(1, min(size, int(budget // im.nbytes)))
                if eff < size:
                    LOGGER.info(
                        f"{self.prefix}raw-image LRU: {size} frames requested, but a frame is "
                        f"{im.nbytes / (1 << 20):.1f} MiB against a {budget / (1 << 20):.0f} MiB budget "
                        f"-> {eff} frame(s) can be resident ({eff * im.nbytes / (1 << 20):.0f} MiB/worker). "
                        f"slice_raw_cache_size cannot raise this; raise slice_raw_cache_mb (or lower it "
                        f"deliberately) if the decode reuse is not enough."
                    )
            while cache and (len(cache) >= size or (budget and self._raw_cache_bytes + im.nbytes > budget)):
                _old_key, old = cache.popitem(last=False)
                self._raw_cache_bytes -= old.nbytes
            cache[img_index] = im
            self._raw_cache_bytes += im.nbytes
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
            self._warn_cap_disabled("degrade_max_side")
            return 0.0
        # 640 = BaseDataset.__init__'s own default, so the fallback cannot disagree with the class.
        return v if v > 0 else 2.0 * float(getattr(self, "imgsz", 640))

    def _warn_cap_disabled(self, key: str) -> None:
        """Warn ONCE per key when a working-resolution cap is turned off with a negative value.

        ``< 0`` means "run the degradation at the ORIGINAL resolution" (legacy behaviour). That is a
        legitimate A/B switch, but the cost scales with the pixel-area ratio: measured against the auto
        cap, weather-noise peak allocation goes 18.4 MB -> 180 MB per sample at 4000x3000, and dense-PSF
        motion blur goes 52 ms -> 451 ms. With several workers that is an easy OOM or a sudden
        multi-x slowdown, so it must not be silent.
        """
        if key in self._cap_warned:
            return
        self._cap_warned.add(key)
        LOGGER.warning(
            f"{self.prefix}{key}<0 disables the working-resolution cap: online degradation now runs at "
            "ORIGINAL resolution. That costs roughly the square of the linear pixel ratio in memory and "
            "time (measured ~10x peak memory and ~5x time at 4000x3000). Use 0 (auto = 2*imgsz) or a "
            "positive pixel cap unless you are deliberately measuring the uncapped path."
        )

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
        also bit-identical to the general one and 5.3-5.5x faster over both tiers, measured at
        imgsz-capped 1280x960: 9.5 ms vs 52.4 ms, and 85 ms vs 451 ms uncapped at 4000x3000).
        Labels are UNCHANGED
        (blur does not move targets). The blurred image is resized to the training size like the other
        branches and enters the Mosaic mix pool (dataset.buffer). Nothing is written to disk (save via
        blur_save_dir).

        ``index`` is the EXPANDED mixed-pool index (for correct Mosaic buffer bookkeeping);
        ``img_index`` is the ORIGINAL image index this blurred sample derives from.
        """
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
        # blur_ratio: an un-selected image owns NO slot in this segment at all (the segment is sized to
        # the ratio), so there is no per-slot fallback left to test -- only close_aug_epoch, which keeps
        # the slots but routes every one of them back to the plain original.
        if self._closing:
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

        label = self._begin_branch_label(index, img_index)
        # Optional save for visual inspection (blur_save_dir set). Annotated per slice_save_annotated,
        # capped by slice_save_max_blur (per-branch override; falls back to slice_save_max when the
        # per-branch cap is not set), deduplicated per (image, tier) across epochs.
        self._save_branch(
            "blur", label, blur,
            save_dir=str(getattr(self, "blur_save_dir", _online_default("blur_save_dir")) or ""),
            save_tag=f"blur_{tier}_p{os.getpid()}_img{img_index}",
            key=("blur", tier, index),
        )

        # Resize to the training size (shared tail)
        return self._finish_branch(label, blur)

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
        # Degrade at the capped resolution; only the PIXEL-typed rain-line length scales with it
        # (haze_beta / noise_std are intensity quantities and therefore resolution-independent).
        im, scale = self._degrade_frame(img_index)
        # weather_ratio: un-selected images own no slot in this segment; close_aug_epoch routes them back
        # to the plain original (see _build_blur_sample).
        if self._closing:
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

        label = self._begin_branch_label(index, img_index)
        # Optional save for visual inspection (weather_save_dir set). Annotated per slice_save_annotated,
        # capped by slice_save_max_weather (falls back to slice_save_max), deduplicated per
        # (image, weather_type) across epochs.
        self._save_branch(
            "weather", label, out,
            save_dir=str(getattr(self, "weather_save_dir", _online_default("weather_save_dir")) or ""),
            save_tag=f"weather_{weather_type}_p{os.getpid()}_img{img_index}",
            key=("weather", weather_type, index),
        )

        # Resize to the training size (shared tail)
        return self._finish_branch(label, out)

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
        # occlusion_ratio: un-selected images own no slot in this segment; close_aug_epoch routes them
        # back to the plain original (see _build_blur_sample).
        if self._closing:
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

        label = self._begin_branch_label(index, img_index)
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

        # Optional save for visual inspection (occlusion_save_dir set). Annotated per
        # slice_save_annotated, capped by slice_save_max_occlusion (falls back to slice_save_max),
        # deduplicated per (image, occlusion_type) across epochs.
        _oc_tag = occlusion_type if occluder_boxes else "none"
        self._save_branch(
            "occlusion", label, out,
            save_dir=str(getattr(self, "occlusion_save_dir", _online_default("occlusion_save_dir")) or ""),
            save_tag=f"occlusion_{_oc_tag}_p{os.getpid()}_img{img_index}",
            key=("occlusion", _oc_tag, index),
        )

        # Resize to the training size (shared tail)
        return self._finish_branch(label, out)

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
        # ratio_pad_ratio: an un-selected image owns no slot in this segment (the segment is sized to the
        # ratio), so the only "skip the padding" case left is close_aug_epoch.
        skip_pad = self._closing
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
        self._save_branch(
            "ratio", label, big,
            save_dir=str(getattr(self, "ratio_pad_save_dir", _online_default("ratio_pad_save_dir")) or ""),
            save_tag=f"ratio_p{os.getpid()}_img{img_index}",
            key=("ratio", index),
        )

        # Resize to the training size (shared tail)
        return self._finish_branch(label, big)

    def _build_compose_sample(self, index: int, group: int) -> dict[str, Any]:
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
        # ``group`` is the ORIGINAL group of four images this slot was assigned by the compose
        # selection (get_image_and_label), not ``index - compose_base``: under ratio-sized segments the
        # slot offset and the group index differ as soon as compose_ratio < 1.
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
        # compose_ratio: an un-selected group owns no slot in this segment (the segment is sized to the
        # ratio); the only remaining "no stitching" case is close_aug_epoch, which keeps the slot and
        # emits the group's first image instead.
        if self._closing:
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
        elif max_side < 0:
            self._warn_cap_disabled("compose_max_side")
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
        # NOTE: compose is the one branch whose directory is DERIVED (slice_save_dir/compose) rather than
        # read straight from its own key, so the resolution stays here and _save_branch only consumes it.
        if getattr(self, "compose_save", False):
            comp_dir = str(getattr(self, "compose_save_dir", "") or "")
            st = getattr(self, "slice_transform", None)
            branch_dir = Path(comp_dir) if comp_dir else (
                st.save_dir / "compose" if st is not None and st.save_dir is not None else None
            )
            if branch_dir is None and not getattr(self, "_compose_warned", False):
                # args.yaml ships compose_save=True with an empty compose_save_dir and no
                # slice_save_dir -> nothing was ever written and nothing said why.
                self._compose_warned = True
                LOGGER.warning(
                    f"{self.prefix}compose_save=True but no output directory is configured: set "
                    f"compose_save_dir, or slice_save_dir (images then go to <slice_save_dir>/compose). "
                    f"Skipping compose save."
                )
            self._save_branch(
                "compose", label, big,
                save_dir=branch_dir,
                save_tag=f"compose_p{os.getpid()}_g{group}",
                key=("compose", group),
            )

        # Keep the composed image on the same Mosaic mix pool as every other sample (cache != 'ram')
        self._touch_buffer(index)

        # Resize to the training size (shared tail)
        return self._finish_branch(label, big)

    def get_image_and_label(self, index: int, count_slice: bool = True) -> dict[str, Any]:
        """Get and return label information from the dataset.

        Args:
            index (int): Index of the image to retrieve.
            count_slice (bool): Whether OnlineSlice updates its positive/background counters and saves tiles.
                Auxiliary "mix" samples requested by Mosaic/CutMix/MixUp pass ``False`` so they slice normally
                but do not inflate the ``neg_ratio`` quota or duplicate saved slices.
        """
        # --- non-augmenting build (mode="val"): no pool exists, so index -> image, exactly upstream ---
        # Upstream's YOLODataset.build_transforms returns a bare val transform when augment=False, so
        # v8_transforms never runs and NONE of slice_transform / img_origin / *_keep are mirrored onto
        # this object. Every segment of the layout therefore resolves to 0 and an index here is a plain
        # image index -- not a pool slot. Running the dispatch below anyway would misread it as a base
        # slot of an empty layout and index past the end of every (empty) selection array.
        # BaseDataset.get_image_and_label is the correct answer for val, including the rect
        # batch_shapes lookup val relies on, so delegate rather than re-implement.
        if not self.augment:
            return super().get_image_and_label(index)
        # pick up the trainer's latest set_epoch publish. No-op (one locked int read) in the
        # main process and whenever the epoch is unchanged; in a DataLoader worker with a stale
        # selection this deterministically rebuilds it for the published epoch.
        self._sync_epoch_masks()
        # Draw this epoch's selections if nothing has yet (no trainer / direct iteration). MUST run
        # before the dispatch below: every branch resolves its slot -> image mapping through those
        # arrays, and the lazy build used to sit after the dispatch (where the builders already needed
        # it). One int compare per sample once the stamp is set.
        self._ensure_epoch_sel()
        # Mixed-pool layout (see _segment_bases): a base segment holding the slicing pipeline's samples
        # for the SELECTED images only, followed by SIX INDEPENDENT segments gated only by their own
        # switches and sized by their ratio -- origin (one un-sliced whole frame per ORIGINAL image via
        # img_origin), ratio, blur (2 per selected image), compose (1 per selected group of 4), weather,
        # occlusion. None of them requires slicing, and each one's slot count is exactly the number of
        # images/GROUPS it selected. For the origin segment the offset into the segment IS the original
        # image index (it covers every original once), so `index - <segment base>` maps directly.
        emit_all = getattr(self, "slice_all_tiles", _online_default("slice_all_tiles"))
        origin_on = self._img_origin_on()
        ratio_on = bool(getattr(self, "ratio_pad_keep", _online_default("ratio_pad_keep")))
        blur_on = bool(getattr(self, "blur_keep", _online_default("blur_keep")))
        compose_on = self._compose_on()
        weather_on = self._weather_on()
        occlusion_on = self._occlusion_on()
        n = len(self.labels)
        # Named field access: keep the SegmentBases object and read ``segment_bases.origin`` / ``segment_bases.ratio`` ...
        # Unpacking it into seven positional locals (base_len, o_base, r_base, ...) would throw away
        # exactly the readability the NamedTuple was introduced for.
        segment_bases = self._segment_bases()

        if occlusion_on and index >= segment_bases.occlusion:
            return self._build_occlusion_sample(index, self._sel_indices("occlusion", n)[index - segment_bases.occlusion])
        if weather_on and index >= segment_bases.weather:
            return self._build_weather_sample(index, self._sel_indices("weather", n)[index - segment_bases.weather])
        if compose_on and index >= segment_bases.compose:
            # composed 2x2 sample from 4 original images; offset -> the GROUP it was selected for
            return self._build_compose_sample(index, self._sel_indices("compose", (n + 3) // 4)[index - segment_bases.compose])
        if blur_on and index >= segment_bases.blur:
            j = index - segment_bases.blur  # 0..2K-1: even -> short tier, odd -> long tier
            return self._build_blur_sample(index, self._sel_indices("blur", n)[j // 2], long=(j % 2 == 1))
        if ratio_on and index >= segment_bases.ratio:
            return self._build_ratio_sample(index, self._sel_indices("ratio", n)[index - segment_bases.ratio])
        if origin_on and index >= segment_bases.origin:
            # un-sliced whole frame for an ORIGINAL image: the offset IS the image index (img_origin
            # covers every original once, not just the sliced ones)
            return self._build_origin_sample(index, index - segment_bases.origin)

        # ---- base segment: n_per slots per SELECTED image only (no plain slot for un-selected images)
        img_index, k, sliced = self._base_slot(index)
        label = deepcopy(self.labels[img_index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        # Online slicing runs on the ORIGINAL-resolution image (before any training resize) so small
        # objects are genuinely enlarged when the sliced sub-image is resized to the training size.
        # slice_ratio: each epoch set_epoch() draws exactly round(slice_ratio*N) ORIGINAL images to
        # slice (original-level decision: with emit_all all 4 tiles share one fate). The base segment now
        # holds ONLY the tiles of those selected images (no plain slot for the un-selected), so every base
        # index here is a sliced tile (``sliced`` is True) -- which is the whole difference from the
        # pre-refactor layout. The independent origin segment (_build_origin_sample) never slices either
        # way and covers every original once when img_origin is on.
        slice_t = getattr(self, "slice_transform", None)
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
        if slice_t is not None and self.augment and sliced and not self._closing:
            # ``copy=False`` skips a full-resolution memcpy per slice call (measured 2.52 ms at
            # 1280x720 and 18.17 ms at 4000x3000, and ~80-90% of slice reads are LRU hits, so that is
            # most of them). It is sound ONLY because nothing in the slice pipeline writes into its
            # input: ``_emit`` hands out a VIEW of the ROI (not a contiguous copy -- see its comment)
            # and the transforms only read it; the resize that follows allocates its own output.
            #
            # Three paths DO return the object we passed in unchanged -- the ``p`` coin flip and the
            # degenerate-size guard in ``__call__``/``slice_at``, plus ``_emit``'s background-quota
            # Plan A fallback -- and those would leak the LRU's own buffer upward into the
            # affine/mosaic transforms, which the ``copy=True`` default exists to protect. A fourth
            # case is a view: when the tile's long side is already ``imgsz`` the shared resize tail
            # becomes a no-op and the view itself would flow on. Rather than thread a "was it shared?"
            # flag through four call sites, detect both by MEMORY SHARING: the copy is taken for a
            # pass-through (identity) and for an un-resized view, and skipped for every view that a
            # resize is about to replace anyway. With the LRU disabled nothing is cached, the array is
            # already exclusive, and the guard is skipped.
            shared = self._load_image_cached(img_index, copy=False)
            # A scheduled tile (slice_target_tiles) arrives exactly like an emit_all tile: the tile is
            # already decided, so slice_at is the right entry point and prefer_target lets it re-check
            # the choice against the LIVE grid when center_bias moved the seam. ``emit_all`` never sets
            # prefer_target -- there all four tiles are emitted, so there is nothing to correct.
            if emit_all or k is not None:
                tile_k = 0 if k is None else int(k)
                im, label = slice_t.slice_at(
                    shared, label, tile_k, src=(img_index, tile_k), count=count_slice,
                    key=img_index, prefer_target=(not emit_all),
                )
            else:
                im, label = slice_t(shared, label, src=img_index, count=count_slice, key=img_index)
            if self._raw_cache_size > 0 and (
                im is shared or (max(im.shape[:2]) == self.imgsz and np.shares_memory(im, shared))
            ):
                # Never let an LRU-backed array reach a writer. The identity test alone only covered
                # the first of the two leaks that exist now:
                #   * a PASS-THROUGH returns ``shared`` ITSELF -- the ``p`` coin flip, the
                #     degenerate-size guard, and ``_emit``'s background-quota Plan A fallback;
                #   * a TILE is a VIEW of ``shared`` (see OnlineSlice._emit), and this shared resize
                #     tail hands that view straight on whenever its own resize will NOT run -- i.e.
                #     when the tile's long side is already exactly ``imgsz``, because
                #     ``np.ascontiguousarray`` returns an already-contiguous view unchanged and a
                #     downstream in-place transform (affine / mosaic) would then write into the cache.
                # When the resize DOES run it is safe by construction: ``cv2.resize`` always allocates
                # its own output. That is exactly why the view case is copied only in the no-resize
                # branch -- copying every view would reinstate the full-resolution memcpy that the
                # view change exists to remove (measured 8.9 ms per tile at 4000x3000).
                im = im.copy()
            # reuse the shared tail instead of a third hand-rolled resize. The sliced sub-image
            # becomes the new "original" for downstream transforms; _finalize_label applies the exact same
            # resize + imgsz clamp as load_image and every other online branch, so resized_shape/ratio_pad
            # can no longer drift (the old inline resize omitted the clamp).
            return self._finalize_label(label, im)
        # load_image indexes the ORIGINAL image files, so always use img_index (== index in the non-sliced
        # case); in emit_all mode index is the expanded (4K + N-K) sample index and would overflow.
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

        Base segment (``n_per`` tiling slots per SELECTED image, NO plain slot for un-selected images)
        plus six independent, RATIO-SIZED segments: origin (img_origin -- one whole frame per ORIGINAL
        image), ratio, blur, compose, weather, occlusion -- each present only when its own switch is on,
        and each as wide as the number of images (groups) its ratio selected.

        Single source of truth (fix): the total comes from ``_segment_bases``, which derives
        it from the same per-segment lengths as every boundary. The accumulation used to be
        re-implemented here; adding a new branch and missing this copy would desynchronise
        ``len(dataset)`` from the decodable index range and silently drop samples.

        A non-augmenting build (``mode="val"``) has no pool and delegates to upstream. Validation is not
        a training surface: ``v8_transforms`` never runs there (see ``get_image_and_label``), so every
        segment is 0 and the pool answer would be 0 -- and 0 is not a harmless empty val set.
        ``build_dataloader`` starts with ``batch = min(batch, len(dataset))``, so a 0-length val dataset
        rewrites its own batch size to 0 and torch raises ``ValueError: batch_size should be a positive
        integer value, but got batch_size=0`` from inside ``_build_train_pipeline`` -- training dies
        before epoch 1, with a traceback that points nowhere near the pool and no mention of img_origin
        or slice_prob. That is exactly the trap ``img_origin=False`` (the default) sprang on every
        train run whose val split is non-empty. Delegating keeps val byte-for-byte upstream.
        """
        if not self.augment:
            return super().__len__()
        return self._segment_bases().total

