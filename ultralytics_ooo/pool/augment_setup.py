# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Online-augmentation assembly: OnlineSlice + the online-augment-aware v8_transforms.

Mirrored verbatim from the forked augment.py. The transform primitives it composes (Mosaic,
RandomPerspective, CopyPaste, ...) come from a stock Ultralytics; only OnlineSlice and the
dataset-property mirroring around them are new.
"""

from __future__ import annotations

import inspect
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


from ultralytics.utils import LOGGER, DEFAULT_CFG_DICT
from ultralytics.utils.instance import Instances
from ultralytics.data.augment import (
    BaseTransform,
    Mosaic,
    RandomPerspective,
    Compose,
    CopyPaste,
    MixUp,
    CutMix,
    Albumentations,
    RandomHSV,
    RandomFlip,
)
from ultralytics.utils import IterableSimpleNamespace
from ultralytics_ooo.core import (
    _WEATHER_TYPES,
    _OCCLUSION_TYPES,
    slice_geometry,
    compute_slice_bias,
    _ensure_dir,
    _imwrite,
)
from ultralytics_ooo.pool.constants import _online_default

_MISSING = object()


def _compat(cls, *args, **kwargs):
    """Instantiate ``cls`` on a pristine upstream Ultralytics, silently dropping kwargs the stock
    class does not accept (fork-only debug knobs like save_dir/save_max; they never affect the
    training math)."""
    sig = inspect.signature(cls.__init__)
    accepts_var = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    if not accepts_var:
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(*args, **kwargs)




class OnlineSlice(BaseTransform):
    """Online SAHI-style 2x2 overlap slicing on the ORIGINAL-resolution image.

    Ports the core algorithms of the offline SAHI equal-division slicing tool into an online
    per-sample transform: 2x2 equal division with overlap, the dual area filter, and background
    (empty-tile) retention. It runs in the dataset loading step on the raw (original-resolution)
    image — before the training resize — so the tile size is computed from the ORIGINAL size:
    e.g. with ``overlap_ratio=0.2`` an original 4000x3000 image yields 2400x1800 tiles. The sampled
    tile is then resized to the training size by the normal loader, which genuinely enlarges small
    objects (SAHI semantics). With probability ``p`` the original image is divided into a 2x2 grid of
    overlapping tiles (tile size = (1 + ``overlap_ratio``) * img / 2), one tile is sampled, and the
    instances intersecting it are kept after the dual area filter.

    When the sampled tile has no kept instances (a background tile), it is emitted as an empty-label
    background sample only while the emitted background count stays below ``emitted positive count *
    neg_ratio`` (the offline slicing tool's ratio rule); otherwise the original image is kept unchanged.
    ``neg_ratio < 0`` keeps every background tile.

    Sliced images can optionally be saved to ``save_dir`` (up to ``save_max`` images, annotated with
    boxes when ``save_annotated``) for visual inspection of the online slicing result.

    Attributes:
        p (float): Probability of applying the slicing.
        overlap_ratio (float): Overlap fraction in [0, 1); tile size = (1 + overlap_ratio) * img / 2.
        min_area_ratio (float): Dual-filter threshold relative to the tile area.
        min_retain_ratio (float): Dual-filter threshold relative to the original box area.
        neg_ratio (float): Background/positive tile count ratio (<0 keeps all backgrounds).
        save_dir (Path | None): Directory to save sliced images (None disables saving).
        save_max (int): Maximum number of sliced images to save (0 = unlimited).
        save_annotated (bool): Draw annotation boxes/classes on saved images.
        save_exist_ok (bool): Allow saving into an existing ``save_dir`` (False raises to avoid overwrite).

    Examples:
        >>> t = OnlineSlice(p=1.0, overlap_ratio=0.2, min_area_ratio=0.005, min_retain_ratio=0.4, neg_ratio=0.2)
        >>> sliced_img, label = t(img_orig, label)  # label: original dict (bboxes/segments/keypoints/cls)
    """

    # Class-level default so instances built via ``__new__`` stubs (tests) still answer `_grid_rng`.
    # Must stay in sync with the ``__init__`` assignment.
    _slice_epoch = 0

    def __init__(
        self,
        p: float = 0.0,
        overlap_ratio: float = 0.2,
        min_area_ratio: float = 0.005,
        min_retain_ratio: float = 0.4,
        neg_ratio: float = 0.2,
        save_dir: str | Path = "",
        save_max: int = 0,
        save_annotated: bool = True,
        exist_ok: bool = True,
        center_constraint: bool = False,
        min_center_ratio: float = 0.6,
        full_box_only: bool = False,
        center_bias: bool = False,
        bias_margin: float = 0.25,
        bias_jitter: float = 0.05,
    ):
        """Initialize OnlineSlice with slicing, filtering, background-ratio and save options.

        Args:
            p (float): Probability of applying the slicing.
            overlap_ratio (float): Overlap fraction in [0, 1).
            min_area_ratio (float): Dual-filter threshold relative to the tile area.
            min_retain_ratio (float): Dual-filter threshold relative to the original box area.
            neg_ratio (float): Background (empty-tile) retention ratio relative to emitted positive tiles:
                ``background_count < positive_count * neg_ratio`` gates emitting an empty tile;
                ``neg_ratio < 0`` keeps every background tile.
            save_dir (str | Path): Directory to save sliced images (empty disables saving).
            save_max (int): Maximum number of sliced images to save (0 = unlimited).
            save_annotated (bool): Draw annotation boxes/classes on saved images.
            exist_ok (bool): Allow saving into an already-existing ``save_dir``; when False, raise an error
                if the directory already exists to avoid overwriting previous sliced outputs.
            center_constraint (bool): When True, assign each box only to the tile containing its center
                (unique ownership), preventing a box from being split/repeated across tiles. ``min_center_ratio``
                still allows a box to also appear in an adjacent tile when it retains enough of its area there.
            min_center_ratio (float): In [0, 1]. With ``center_constraint=True``, a box whose center is outside
                a tile is kept in that tile only if its retained area ratio there is >= this value (compat for
                large objects). 1.0 = strictly unique (center-only).
            full_box_only (bool): Keep a box in a tile ONLY when the whole box lies fully inside that tile;
                a box that is cut by a tile boundary is filtered out (never kept as a partial/sliver box).
                This is the "keep every target unless the slice cut it" behavior. When True it takes precedence
                over ``center_constraint`` (which would otherwise drop targets from non-owning tiles), so
                every fully-contained target is preserved in every tile that fully contains it (duplicates in
                the overlap region are intentional).
            center_bias (bool): Target-aware seam shifting. When True, each sliced image computes the 2x2
                seam position from its own box-center distribution (projection onto each axis, seam placed at
                the sparsest candidate inside ``[bias_margin, 1-bias_margin]``), so fewer boxes get cut in
                half by a seam. Tile size/overlap/count are unchanged (only the seam moves). False = fixed
                centered grid (fully backward compatible).
            bias_margin (float): In (0, 0.5]. Seam search window edge: the seam position is restricted to
                ``[bias_margin, 1-bias_margin]`` of each axis so tiles never become too small.
            bias_jitter (float): Uniform seam perturbation (relative to the image extent) applied once per
                ``(epoch, image)``, so the same image does not get identical seams every epoch while all
                4 tiles of that image keep sharing one grid. 0 disables.
        """
        # Explicit ValueError rather than assert: every one of these is a user-supplied hyperparameter
        # from default.yaml, and ``python -O`` strips asserts -- the invalid value would then be
        # accepted silently and only show up as nonsense slicing geometry much later.
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"OnlineSlice: 'p' must be in [0, 1], got {p}.")
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(f"OnlineSlice: 'overlap_ratio' must be in [0, 1), got {overlap_ratio}.")
        if not 0.0 <= min_area_ratio <= 1.0:
            raise ValueError(f"OnlineSlice: 'min_area_ratio' must be in [0, 1], got {min_area_ratio}.")
        if not 0.0 <= min_retain_ratio <= 1.0:
            raise ValueError(f"OnlineSlice: 'min_retain_ratio' must be in [0, 1], got {min_retain_ratio}.")
        if not 0.0 <= min_center_ratio <= 1.0:
            raise ValueError(f"OnlineSlice: 'min_center_ratio' must be in [0, 1], got {min_center_ratio}.")
        self.p = p
        self.overlap_ratio = overlap_ratio
        self.min_area_ratio = min_area_ratio
        self.min_retain_ratio = min_retain_ratio
        self.center_constraint = center_constraint
        self.min_center_ratio = min_center_ratio
        self.full_box_only = full_box_only
        self.center_bias = center_bias
        self.bias_margin = bias_margin
        self.bias_jitter = bias_jitter
        self.neg_ratio = neg_ratio
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_exist_ok = exist_ok
        self.save_max = save_max
        self.save_annotated = save_annotated
        # Per-instance state: positive/background tile counters (per worker) and saved-image counter.
        self._pos_count = 0
        self._bg_count = 0
        self._saved = 0
        # Unique keys of tiles already saved (src = (img_index, k) or img_index), so each tile is saved only
        # once across epochs / mosaic mix visits instead of accumulating one file per epoch.
        self._saved_keys = set()
        # Current epoch, pushed by BaseDataset._rebuild_epoch_masks. It only feeds the deterministic seam
        # jitter (see _grid_rng): the seam must be stable inside one epoch and vary across epochs. 0 is a
        # valid deterministic default for standalone use (no trainer ever calling set_epoch).
        # Class-level default matters: tests build instances via __new__ stubs and never run __init__.
        self._slice_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Record the current epoch that seeds the deterministic seam jitter.

        Called from ``BaseDataset._rebuild_epoch_masks`` right next to ``reset_counters`` so the MAIN
        process and every DataLoader worker hold the same epoch -- the same "rebuild in every process,
        derive identically, transport nothing" contract the masks already rely on.
        """
        self._slice_epoch = int(epoch)

    def _grid_rng(self, key: Any) -> random.Random | None:
        """Deterministic seam-jitter RNG for ``key``, or ``None`` to use the global stream.

        ``key`` identifies the ORIGINAL image (not the tile), so all 4 tiles of one image in
        ``slice_all_tiles`` mode derive the SAME ``bias_x``/``bias_y`` and therefore one shared 2x2
        grid. Drawing per call instead (the old behavior) gave each tile its own grid: the union of
        the 4 tiles no longer covered the image once ``bias_jitter`` approached ``overlap_ratio/2``,
        and the "seam lands where targets are sparsest" guarantee became per-tile noise.

        ``key is None`` keeps the historical behavior (global ``random`` per call) for callers that
        have no stable image identity; nothing in the dataset pipeline takes that path.
        """
        if key is None:
            return None
        return random.Random(f"OnlineSlice:{int(self._slice_epoch)}:{key}")

    def reset_counters(self) -> None:
        """Reset the per-epoch positive/background tile counters (called from BaseDataset.set_epoch).

        Without a reset the counters accumulate monotonically across epochs, so the ``neg_ratio``
        background quota keeps tightening as training progresses and later epochs retain fewer
        background tiles (behavior drift over time). Called once per epoch so the background
        budget restarts every epoch. Multi-worker DataLoader processes keep independent counters
        (the quota stays a per-worker approximation, by design).
        """
        self._pos_count = 0
        self._bg_count = 0

    def _allow_background(self) -> bool:
        """Return whether a background (empty) tile may be emitted under the global ratio rule."""
        if self.neg_ratio < 0:
            return True
        return self._bg_count < self._pos_count * self.neg_ratio

    def _save_tile(self, tile: np.ndarray, boxes_px: np.ndarray, cls: np.ndarray, tag: str,
                   src: Any = None) -> None:
        """Save a sliced image (optionally with annotations), limited by ``save_max`` and deduplicated by ``src``."""
        if self.save_dir is None or (self.save_max > 0 and self._saved >= self.save_max):
            return
        if src is not None and src in self._saved_keys:
            return  # this tile was already saved (same image+tile across epochs / mosaic mix visits)
        if not self.save_exist_ok and self._saved == 0 and self.save_dir.exists():
            raise FileExistsError(
                f"OnlineSlice: save_dir '{self.save_dir}' already exists. Set slice_save_exist_ok=True to "
                f"overwrite previous sliced outputs, or use a new slice_save_dir."
            )
        cls = np.asarray(cls).reshape(-1)
        img = tile
        if self.save_annotated and len(boxes_px):
            img = tile.copy()
            if len(boxes_px) != len(cls):
                LOGGER.warning(
                    f"OnlineSlice._save_tile: {len(boxes_px)} boxes vs {len(cls)} cls for '{tag}' "
                    "-- drawing only the aligned prefix."
                )
            for b, c in zip(boxes_px, cls):
                x0, y0, x1, y1 = (int(round(float(v))) for v in b)
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(img, f"cls{int(c)}", (x0, max(0, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # _ensure_dir: mkdir once per process instead of a syscall per saved tile per worker.
        _ensure_dir(self.save_dir)
        if not _imwrite(str(self.save_dir / f"{tag}_{self._saved:05d}_n{len(boxes_px)}.jpg"), img):
            # never consume the save_max quota (nor silently pass) when the write actually failed.
            LOGGER.warning(
                f"OnlineSlice: tile save failed for '{self.save_dir}' (imwrite returned False) -- check "
                "path/permissions; save_max quota NOT consumed."
            )
            return
        self._saved += 1
        if src is not None:
            self._saved_keys.add(src)

    def _grid(self, w: int, h: int, xyxy: np.ndarray | None = None, key: Any = None) -> tuple[list, float, float]:
        """Return (tiles, bias_x, bias_y): the 4 (x0, y0, x1, y1) tiles of the 2x2 overlap grid.

        With ``center_bias`` the seam position is computed from the box centers (pixel ``xyxy``) via
        ``compute_slice_bias``; otherwise the centered grid is used (bias = 0, backward compatible).
        ``key`` is the ORIGINAL image index; it makes the seam jitter deterministic per
        ``(epoch, image)`` so all 4 tiles of one image share a single grid (see ``_grid_rng``).
        """
        bx, by = 0.0, 0.0
        if self.center_bias:
            bx, by = compute_slice_bias(
                w, h, xyxy, self.bias_margin, self.bias_jitter, jitter_rng=self._grid_rng(key)
            )
        return slice_geometry(w, h, self.overlap_ratio, bx, by), bx, by

    def _geometry(self, img: np.ndarray, label: dict[str, Any], key: Any = None) -> list:
        """Convert boxes to pixel xyxy and compute the 4 tile intersection results.

        Args:
            key (Any): Original-image identity forwarded to ``_grid`` for the deterministic seam jitter.

        Returns:
            (list): List of ``(x0, y0, x1, y1, keep_idx, tile_local_xyxy)`` for the 4 grid tiles.
        """
        h, w = img.shape[:2]
        bbox_format = label.get("bbox_format", "xywh")
        normalized = label.get("normalized", True)
        boxes = np.asarray(label["bboxes"], dtype=np.float64).copy()  # (N, 4) in `bbox_format`
        if normalized:
            if bbox_format == "xywh":
                cx, cy, bw, bh = boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h
                xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
            elif bbox_format == "ltwh":
                x0, y0, bw, bh = boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h
                xyxy = np.stack([x0, y0, x0 + bw, y0 + bh], axis=1)
            else:  # xyxy
                xyxy = boxes * np.array([w, h, w, h], dtype=np.float64)
        else:
            if bbox_format == "xywh":
                cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
            elif bbox_format == "ltwh":
                x0, y0, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                xyxy = np.stack([x0, y0, x0 + bw, y0 + bh], axis=1)
            else:  # xyxy
                xyxy = boxes

        n = len(xyxy)
        if n:
            ori_area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

        tiles, bx, by = self._grid(w, h, xyxy if n else None, key)
        tile_results = []  # (x0, y0, x1, y1, keep_idx, tile_local_xyxy)
        for x0, y0, x1, y1 in tiles:
            if n == 0:
                tile_results.append((x0, y0, x1, y1, np.array([], dtype=int), np.empty((0, 4), dtype=np.float32)))
                continue
            ix0 = np.maximum(xyxy[:, 0], x0)
            iy0 = np.maximum(xyxy[:, 1], y0)
            ix1 = np.minimum(xyxy[:, 2], x1)
            iy1 = np.minimum(xyxy[:, 3], y1)
            iw = ix1 - ix0
            ih = iy1 - iy0
            inter = (iw > 0) & (ih > 0)
            inter_area = iw * ih
            tile_area = (x1 - x0) * (y1 - y0)
            # Dual area filter (faithful to the offline slicing tool): drop only when BOTH conditions hold.
            drop = (inter_area < self.min_area_ratio * tile_area) & (inter_area < self.min_retain_ratio * ori_area)
            keep = inter & ~drop
            # Center constraint: unique ownership. With overlapping tiles a box center can lie inside more than one
            # tile, so ownership is decided by the ORIGINAL image midlines (non-overlapping equal division): each box
            # belongs to the cell (column/row) that contains its center -> exactly one tile. A box is also kept in an
            # adjacent tile when it retains >= min_center_ratio of its area there (large-object compat);
            # min_center_ratio=1.0 -> strictly unique. full_box_only takes precedence (see below).
            if self.center_constraint and not self.full_box_only:
                # Ownership midline follows the biased seam (w/2 + bias*w), NOT the image center: with
                # center_bias the non-overlap equal-division line is shifted, so ownership must shift too.
                mid_x = w / 2.0 + bx * w
                mid_y = h / 2.0 + by * h
                cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
                cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
                # Ownership column/row is decided by the TILE CENTER (not the tile origin): with overlapping
                # tiles the right/bottom tile origins lie left of / above the image midline.
                t_col = 1 if (x0 + x1) / 2.0 >= mid_x else 0
                t_row = 1 if (y0 + y1) / 2.0 >= mid_y else 0
                owner = ((cx >= mid_x).astype(int) == t_col) & ((cy >= mid_y).astype(int) == t_row)
                if self.min_center_ratio >= 1.0:
                    keep &= owner
                else:
                    retain = inter_area / np.maximum(ori_area, 1e-9)
                    keep &= owner | (retain >= self.min_center_ratio)
            # Full-box-only: keep a target only when the whole box is fully inside this tile; a box cut by a
            # tile boundary is dropped. This preserves every target that is not split by slicing (duplicates in
            # the overlap region are intentional) and never keeps a partial/sliver box.
            if self.full_box_only:
                full = (xyxy[:, 0] >= x0) & (xyxy[:, 1] >= y0) & (xyxy[:, 2] <= x1) & (xyxy[:, 3] <= y1)
                keep &= full
            idx = np.nonzero(keep)[0]
            if len(idx) == 0:
                tile_results.append((x0, y0, x1, y1, idx, np.empty((0, 4), dtype=np.float32)))
            else:
                local = np.stack(
                    [ix0[idx] - x0, iy0[idx] - y0, np.minimum(ix1[idx], x1) - x0, np.minimum(iy1[idx], y1) - y0], axis=1
                ).astype(np.float32)
                tile_results.append((x0, y0, x1, y1, idx, local))
        return tile_results

    def _empty_label(self, sub: np.ndarray, label: dict[str, Any]) -> dict[str, Any]:
        """Build a label dict for an empty (background) tile: same structure as the input, zero boxes/cls."""
        new_label = dict(label)
        new_label["img"] = sub
        new_label["bboxes"] = np.empty((0, 4), dtype=np.float32)
        new_label["bbox_format"] = "xywh"
        new_label["normalized"] = True
        # Keep the same cls shape as the input (YOLO labels store cls as (n,1)) so downstream
        # concatenations (e.g. Mosaic._cat_labels) see consistent dimensions for empty labels.
        new_label["cls"] = np.asarray(label["cls"])[np.array([], dtype=int)]
        new_label["segments"] = []
        if label.get("keypoints") is not None:
            new_label["keypoints"] = np.empty((0, 0, 3), dtype=np.float32)
        return new_label

    def _emit(self, img: np.ndarray, label: dict[str, Any], x0: int, y0: int, x1: int, y1: int, idx: np.ndarray,
              local: np.ndarray, src: Any = None, count: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
        """Build the (sub_img, updated label) for a selected tile; handles background and save.

        Args:
            count (bool): Whether to update the positive/background counters and save the tile. Auxiliary
                "mix" samples (mosaic/cutmix/mixup companions) pass ``count=False`` so they do not inflate
                the neg_ratio quota or duplicate saves.
        """
        h, w = img.shape[:2]
        tw, th = x1 - x0, y1 - y0
        sub = np.ascontiguousarray(img[y0:y1, x0:x1])
        if len(idx) == 0:  # background tile: emitted only while background_count < positive_count * neg_ratio
            if self._allow_background():
                if count:
                    self._bg_count += 1
                if count and self.save_dir is not None and (self.save_max == 0 or self._saved < self.save_max):
                    self._save_tile(sub, np.empty((0, 4), dtype=np.float32), np.empty(0), f"bg{os.getpid()}", src)
                return sub, self._empty_label(sub, label)
            # Background quota reached: keep the original image unchanged (mode A contract). In emit_all
            # mode (slice_at) this is exactly the Plan A fallback -- the ORIGINAL image is returned, never an
            # empty tile, so the returned image and its bbox coords always match.
            return img, label

        if count:
            self._pos_count += 1
        new_label = dict(label)
        new_label["img"] = sub
        # bboxes -> sub-image normalized xywh
        lx0, ly0 = local[:, 0] / tw, local[:, 1] / th
        lx1, ly1 = local[:, 2] / tw, local[:, 3] / th
        cx, cy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
        bw, bh = lx1 - lx0, ly1 - ly0
        new_label["bboxes"] = np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)
        new_label["bbox_format"] = "xywh"
        new_label["normalized"] = True
        # cls
        cls = np.asarray(label["cls"])[idx]
        new_label["cls"] = cls
        # segments (list of normalized polys) -> sub-image normalized
        segs = label.get("segments", [])
        if segs is not None and len(segs):
            new_segs = []
            for si in idx:
                s = np.asarray(segs[si], dtype=np.float64).copy()
                s[..., 0] = (s[..., 0] * w - x0) / tw
                s[..., 1] = (s[..., 1] * h - y0) / th
                new_segs.append(s.astype(np.float32))
            new_label["segments"] = new_segs
        else:
            new_label["segments"] = []
        # keypoints (normalized x, y + visibility) -> sub-image normalized
        kpts = label.get("keypoints", None)
        if kpts is not None:
            k = np.asarray(kpts, dtype=np.float64)[idx].copy()
            k[..., 0] = (k[..., 0] * w - x0) / tw
            k[..., 1] = (k[..., 1] * h - y0) / th
            new_label["keypoints"] = k.astype(np.float32)
        if count and self.save_dir is not None and (self.save_max == 0 or self._saved < self.save_max):
            self._save_tile(sub, local, cls, f"pos{os.getpid()}", src)
        return sub, new_label

    def __call__(self, img: np.ndarray, label: dict[str, Any], src: Any = None,
                 count: bool = True, key: Any = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Slice the ORIGINAL-resolution image and return a RANDOMLY sampled tile (mode A).

        The label dict uses the raw dataset format: ``bboxes`` (N, 4) in ``bbox_format``/``normalized``,
        ``cls`` (N,), optional ``segments`` (list of normalized polys) and ``keypoints`` (N, K, 3). The
        returned sub-image keeps its original resolution; downstream training resize enlarges small targets.

        Args:
            src (Any): Optional unique key (e.g. ``(img_index, k)`` or ``img_index``) used to save each
                tile only once across epochs / mosaic mix visits.
            count (bool): Whether to update counters/save (False for auxiliary mix samples).
            key (Any): Original image index. Pins the seam jitter to ``(epoch, key)`` so the grid is a
                stable property of the image within an epoch (``src`` cannot be reused for this: it also
                carries the tile index ``k`` in ``slice_all_tiles`` mode and keys the save-once set).
        """
        if random.uniform(0, 1) > self.p:
            return img, label
        if img.shape[1] < 2 or img.shape[0] < 2:
            return img, label
        tile_results = self._geometry(img, label, key)
        # Sample a random tile uniformly so that every tile is covered across epochs.
        x0, y0, x1, y1, idx, local = random.choice(tile_results)
        return self._emit(img, label, x0, y0, x1, y1, idx, local, src, count)

    def slice_at(self, img: np.ndarray, label: dict[str, Any], k: int,
                 src: Any = None, count: bool = True, key: Any = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Return the ``k``-th (0..3) tile so all 4 tiles participate in training (mode B / emit_all).

        Background-quota-exceeded tiles fall back to the ORIGINAL image (Plan A), never to an empty
        tile, so every sample in the 4N pool carries either a sliced tile or the full original.

        Args:
            src (Any): Optional unique key (e.g. ``(img_index, k)`` or ``img_index``) used to save each
                tile only once across epochs / mosaic mix visits.
            count (bool): Whether to update counters/save (False for auxiliary mix samples).
            key (Any): Original image index -- the SAME value for k=0..3 of one image. It is what makes
                the 4 tiles share one grid; do NOT pass ``(img_index, k)`` here.
        """
        if random.uniform(0, 1) > self.p:
            return img, label
        if img.shape[1] < 2 or img.shape[0] < 2:
            return img, label
        tile_results = self._geometry(img, label, key)
        x0, y0, x1, y1, idx, local = tile_results[k]
        sub, out_label = self._emit(img, label, x0, y0, x1, y1, idx, local, src, count)
        # Plan A fallback: when this tile is empty AND the background quota is reached, _emit returns
        # the ORIGINAL image unchanged (mode A contract). We keep that original image as-is -- bbox
        # coords match the returned image, len stays constant (4N), and the pure-negative empty tile
        # (up to ~67% of the pool in sparse small-object scenes) is replaced by a positive original.
        # Un-sliced originals enter the mosaic mix pool like every other sample (implicit oversampling
        # of the few positives, each copy independently augmented).
        return sub, out_label



def _hyp_get(hyp: Any, key: str, default: Any = _MISSING) -> Any:
    """Read one project hyperparameter, falling back to its ``default.yaml`` value.

    ``v8_transforms`` mirrors ~67 project keys (slicing / compose / ratio / blur / weather / occlusion /
    save caps) from ``hyp`` onto the dataset. Those reads used to be spelled ``getattr(hyp, "<key>")``
    with no default, which is exactly equivalent to ``hyp.<key>`` -- Ruff flags all 60 of them as B009,
    "not any safer than normal property access" -- and aborts the augmentation build with an
    ``AttributeError`` whenever ``hyp`` was not freshly derived from the current ``DEFAULT_CFG``: a
    third-party ``IterableSimpleNamespace``, a hand-built ``dict``, or the ``train_args`` restored from
    an older ``args.yaml`` / checkpoint that predates the key. Falling back the same way
    ``base._ONLINE_DEFAULTS`` already does for its per-call reads keeps one behaviour for the pipeline.

    Resolution order: attribute on ``hyp`` -> ``default`` when given -> ``DEFAULT_CFG_DICT[key]``. A key
    in neither place is a developer error (it is missing from ``ultralytics/cfg/default.yaml``), so it
    raises a ``ValueError`` naming the key instead of silently picking a built-in literal -- which makes
    the "register every new key in default.yaml" convention self-enforcing at build time.

    不要改回 `getattr(hyp, "<key>", <字面量>)`: 默认值只能有一个真源 (default.yaml), 两处各写一遍
    迟早漂移; 新增 cfg 键却忘了登记 default.yaml 时, 这里会立刻报错而不是静默用字面量兜底。

    Deliberately NOT used for the upstream YOLO keys (``hyp.mosaic``, ``hyp.mixup``, ...): a ``hyp``
    missing those is genuinely broken and upstream raises on them too.
    """
    if default is _MISSING:
        try:
            default = DEFAULT_CFG_DICT[key]
        except KeyError:
            # Online-augmentation key not present in the PRISTINE upstream default.yaml -> fall back to the
            # package's own default table instead of hard-failing (the upstream default.yaml is left untouched).
            default = _online_default(key)
    return getattr(hyp, key, default)



def v8_transforms(dataset, imgsz: int, hyp: IterableSimpleNamespace):
    """Apply a series of image transformations for training.

    This function creates a composition of image augmentation techniques to prepare images for YOLO training. It
    includes operations such as mosaic, copy-paste, random perspective, mixup, and various color adjustments.

    Args:
        dataset (Dataset): The dataset object containing image data and annotations.
        imgsz (int): The target image size for resizing.
        hyp (IterableSimpleNamespace): A namespace of hyperparameters controlling various aspects of the
            transformations. Project keys (``slice_*`` / ``compose_*`` / ``ratio_pad_*`` / ``blur_*`` /
            ``weather_*`` / ``occlusion_*`` / ``mosaic_save_*``) are read through ``_hyp_get``, so a hyp
            that predates a key -- an older ``args.yaml`` or checkpoint ``train_args`` -- falls back to
            ``default.yaml`` instead of aborting the build with an ``AttributeError``.

    Returns:
        (Compose): A composition of image transformations to be applied to the dataset.

    Examples:
        >>> from ultralytics.cfg import DEFAULT_CFG
        >>> from ultralytics.data.dataset import YOLODataset
        >>> from ultralytics.utils import IterableSimpleNamespace
        >>> dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, imgsz=640)
        >>> hyp = IterableSimpleNamespace(
        ...     **{
        ...         **vars(DEFAULT_CFG),
        ...         "mosaic": 1.0,
        ...         "copy_paste": 0.5,
        ...         "degrees": 10.0,
        ...         "translate": 0.2,
        ...         "scale": 0.9,
        ...     }
        ... )
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
        >>> augmented_data = transforms(dataset[0])

        >>> # With custom albumentations
        >>> import albumentations as A
        >>> augmentations = [A.Blur(p=0.01), A.CLAHE(p=0.01)]
        >>> hyp.augmentations = augmentations
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
    """
    mosaic = _compat(Mosaic,
        dataset,
        imgsz=imgsz,
        p=hyp.mosaic,
        save_dir=str(_hyp_get(hyp, "mosaic_save_dir") or ""),
        save_max=int(_hyp_get(hyp, "mosaic_save_max")),
        save_annotated=bool(_hyp_get(hyp, "mosaic_save_annotated")),
        exist_ok=bool(_hyp_get(hyp, "mosaic_save_exist_ok")),
    )
    affine = _compat(RandomPerspective,
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        size=(imgsz, imgsz),
        preserve_obb=getattr(dataset, "use_obb", False),
    )

    pre_transform = Compose([mosaic, affine])
    # Online augmentation master switch: every online branch (slice / compose / ratio / blur) is
    # disabled in rect and obb modes (same constraint as mosaic). After this shared gate, each
    # branch is controlled by ITS OWN independent switch -- compose_keep / ratio_pad_keep /
    # blur_keep no longer require slicing or keep_origin (slice_prob only gates the slicing
    # pipeline itself; see BaseDataset._segment_bases for the mixed-pool layout).
    online_aug_on = not getattr(dataset, "rect", False) and not getattr(dataset, "use_obb", False)

    # ---- 训练后期关闭在线增强 (close_aug_epoch, 与 close_mosaic 同构的时间维调度) ----
    # 修复附带: 此前该值从未被复制到 dataset 上, base.py 的 getattr(self, "close_aug_epoch", 0)
    # 恒为 0, 时间维调度从未生效; set_epoch/_rebuild_epoch_masks 靠它判断最后 N 个 epoch 全部关增强。
    dataset.close_aug_epoch = int(_hyp_get(hyp, "close_aug_epoch"))

    # mirror the LRU capacity onto the dataset so it is self-describing (base.py consumes the same
    # hyp key directly in __init__, since this function runs too late for an eager read there).
    dataset.slice_raw_cache_size = int(_hyp_get(hyp, "slice_raw_cache_size") or 0)
    # degradation resample kernel ("area" = antialiased/slower, "linear" = faster/slightly softer);
    # mirrored here because _degrade_frame reads it per-call from `self`.
    dataset.degrade_resample = str(_hyp_get(hyp, "degrade_resample") or "linear")

    # ---- 在线切片 (slice_prob 独立开关) ----
    slice_enabled = online_aug_on and _hyp_get(hyp, "slice_prob") > 0.0
    if slice_enabled:
        # tile cap honours slice_save_max_tile override (falls back to slice_save_max when None).
        # tile is the ONLY branch whose cap lives on the OnlineSlice instance itself (see
        # OnlineSlice._save_tile), so the override must be applied here at construction time --
        # unlike blur/ratio/compose whose caps base.py reads per-call from `self`.
        _tile_cap = _hyp_get(hyp, "slice_save_max_tile")
        if _tile_cap is None:
            _tile_cap = int(_hyp_get(hyp, "slice_save_max"))
        dataset.slice_transform = OnlineSlice(
            p=float(_hyp_get(hyp, "slice_prob")),
            overlap_ratio=float(_hyp_get(hyp, "slice_overlap_ratio")),
            min_area_ratio=float(_hyp_get(hyp, "slice_min_tile_area_ratio")),
            min_retain_ratio=float(_hyp_get(hyp, "slice_min_box_retain_ratio")),
            neg_ratio=float(_hyp_get(hyp, "slice_background_ratio")),
            save_dir=str(_hyp_get(hyp, "slice_save_dir") or ""),
            save_max=int(_tile_cap),
            save_annotated=bool(_hyp_get(hyp, "slice_save_annotated")),
            exist_ok=bool(_hyp_get(hyp, "slice_save_exist_ok")),
            center_constraint=bool(_hyp_get(hyp, "slice_center_constraint")),
            min_center_ratio=float(_hyp_get(hyp, "slice_min_center_retain_ratio")),
            full_box_only=bool(_hyp_get(hyp, "slice_full_box_only")),
            # 目标感知切缝 (方案1): 切缝按本图目标中心分布微移, 减少目标被劈碎
            center_bias=bool(_hyp_get(hyp, "slice_center_bias")),
            bias_margin=float(_hyp_get(hyp, "slice_bias_margin")),
            bias_jitter=float(_hyp_get(hyp, "slice_bias_jitter")),
        )
        dataset.slice_all_tiles = bool(_hyp_get(hyp, "slice_all_tiles"))
        dataset.slice_ratio = float(_hyp_get(hyp, "slice_ratio"))
    else:
        dataset.slice_transform = None
        dataset.slice_all_tiles = False
        dataset.slice_ratio = 1.0

    # ---- 独立增强开关 (slice_keep_origin / compose_keep / ratio_pad_keep / blur_keep 互不影响,
    # 不受 slice_prob 控制; keep_origin 无切片时由 _keep_origin_on() 自动抑制) ----
    dataset.slice_keep_origin = online_aug_on and bool(_hyp_get(hyp, "slice_keep_origin"))
    # ---- 独立增强开关 (compose_keep / ratio_pad_keep / blur_keep 互不影响, 不受 slice_prob 控制) ----
    # compose/ratio/blur 不需要切片或 keep_origin, 单独开启即生效(见 _segment_bases 区段布局)。
    dataset.compose_keep = online_aug_on and bool(_hyp_get(hyp, "compose_keep"))
    dataset.compose_save = online_aug_on and bool(_hyp_get(hyp, "compose_save"))
    dataset.compose_save_dir = str(_hyp_get(hyp, "compose_save_dir") or "")
    dataset.compose_max_side = int(_hyp_get(hyp, "compose_max_side") or 0)
    # Same working-resolution cap, but for the degradation branches (blur / weather / occlusion / ratio).
    # Mirrored onto the dataset exactly like compose_max_side: without this copy the key would be
    # registered in default.yaml yet never reach the dataset, and _degrade_max_side() would silently
    # stay on its "auto" default no matter what the user configured.
    dataset.degrade_max_side = int(_hyp_get(hyp, "degrade_max_side") or 0)
    dataset.ratio_pad_keep = online_aug_on and bool(_hyp_get(hyp, "ratio_pad_keep"))
    dataset.ratio_pad_target = str(_hyp_get(hyp, "ratio_pad_target") or "auto")
    dataset.ratio_pad_color = str(_hyp_get(hyp, "ratio_pad_color") or "black")
    dataset.ratio_pad_save_dir = str(_hyp_get(hyp, "ratio_pad_save_dir") or "")
    dataset.blur_keep = online_aug_on and bool(_hyp_get(hyp, "blur_keep"))
    dataset.blur_short_len_min = float(_hyp_get(hyp, "blur_short_len_min"))
    dataset.blur_short_len_max = float(_hyp_get(hyp, "blur_short_len_max"))
    dataset.blur_long_len_min = float(_hyp_get(hyp, "blur_long_len_min"))
    dataset.blur_long_len_max = float(_hyp_get(hyp, "blur_long_len_max"))
    dataset.blur_long_defocus_sigma = float(_hyp_get(hyp, "blur_long_defocus_sigma"))
    # 拖影方向是否限制为轴对齐 (水平/垂直)。与 blur_*_len_* 一样必须显式镜像到 dataset:
    # 否则键在 default.yaml 里注册了却到不了 dataset, _build_blur_sample 只会读到内置默认值。
    dataset.blur_axis_aligned = bool(_hyp_get(hyp, "blur_axis_aligned"))
    dataset.blur_save_dir = str(_hyp_get(hyp, "blur_save_dir") or "")
    # ---- 在线气象退化 (weather_*): 雨/雾/噪声, 标签不变, 独立开关 + epoch 级比例 (复用掩码机制) ----
    dataset.weather_keep = online_aug_on and bool(_hyp_get(hyp, "weather_keep"))
    dataset.weather_ratio = float(_hyp_get(hyp, "weather_ratio"))
    dataset.weather_types = str(_hyp_get(hyp, "weather_types") or "rain,haze,noise")
    dataset.weather_rain_density = float(_hyp_get(hyp, "weather_rain_density"))
    dataset.weather_rain_length = float(_hyp_get(hyp, "weather_rain_length"))
    dataset.weather_haze_beta = float(_hyp_get(hyp, "weather_haze_beta"))
    dataset.weather_noise_std = float(_hyp_get(hyp, "weather_noise_std"))
    dataset.weather_save_dir = str(_hyp_get(hyp, "weather_save_dir") or "")
    # ---- 在线遮挡模拟 (occlusion_*): rect/stripe 语义遮挡块, 标签不变(超阈值目标剔除), 独立开关 + epoch 比例 ----
    dataset.occlusion_keep = online_aug_on and bool(_hyp_get(hyp, "occlusion_keep"))
    dataset.occlusion_ratio = float(_hyp_get(hyp, "occlusion_ratio"))
    dataset.occlusion_types = str(_hyp_get(hyp, "occlusion_types") or "rect,stripe")
    dataset.occlusion_blocks = int(_hyp_get(hyp, "occlusion_blocks") or 1)
    dataset.occlusion_size_ratio = float(_hyp_get(hyp, "occlusion_size_ratio"))
    dataset.occlusion_color = str(_hyp_get(hyp, "occlusion_color") or "auto")
    dataset.occlusion_max_cover = float(_hyp_get(hyp, "occlusion_max_cover"))
    dataset.occlusion_save_dir = str(_hyp_get(hyp, "occlusion_save_dir") or "")
    # weather/occlusion 类型白名单校验 (拼错立即在构造期报错, 不再静默落入默认分支)。
    # 常量在模块顶层直接取自 online_degrade —— 与 _apply_weather / _apply_occlusion 的分派同源, 单一真源。
    # 不要改回 `from ultralytics.data.base import ...`: base.py 自己一次都不用这两个名字, 那样就是隐式
    # re-export, Ruff F401 一次 --fix (或 IDE 优化导入) 就会删掉 base 里那两行, 校验随之静默消失,
    # 拼错退化成运行时随机兜底 —— 类型白名单校验防的正是这个陷阱。
    # 空串回退默认类型 (与 base.py 运行时行为一致)。

    _w = [t.strip() for t in dataset.weather_types.split(",") if t.strip()]
    _bad = sorted(set(_w) - _WEATHER_TYPES)
    if _bad:
        raise ValueError(
            f"weather_types contains unknown type(s) {_bad}; valid types: {sorted(_WEATHER_TYPES)}."
        )
    dataset.weather_types = ",".join(_w) if _w else "haze"
    _o = [t.strip() for t in dataset.occlusion_types.split(",") if t.strip()]
    _bad = sorted(set(_o) - _OCCLUSION_TYPES)
    if _bad:
        raise ValueError(
            f"occlusion_types contains unknown type(s) {_bad}; valid types: {sorted(_OCCLUSION_TYPES)}."
        )
    dataset.occlusion_types = ",".join(_o) if _o else "rect"
    # per-branch save cap overrides (slice_save_max_{blur,ratio,compose,weather,occlusion}).
    # base.py's _save_cap() reads these from `self`; if not set here it falls back to slice_save_max.
    # tile is deliberately NOT listed: its cap lives on the OnlineSlice instance (save_max above).
    dataset.slice_save_max_blur = _hyp_get(hyp, "slice_save_max_blur")
    dataset.slice_save_max_ratio = _hyp_get(hyp, "slice_save_max_ratio")
    dataset.slice_save_max_compose = _hyp_get(hyp, "slice_save_max_compose")
    dataset.slice_save_max_weather = _hyp_get(hyp, "slice_save_max_weather")
    dataset.slice_save_max_occlusion = _hyp_get(hyp, "slice_save_max_occlusion")
    # 全局画框开关与切片解耦 —— 各在线分支 (blur/ratio/weather/occlusion/compose) 的保存块
    # 统一只读 dataset 属性, 不再依赖 slice_transform 实例 (slice_transform=None 时亦可保存)。
    dataset.slice_save_annotated = bool(_hyp_get(hyp, "slice_save_annotated"))

    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(dataset, p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if getattr(dataset, "use_keypoints", False):
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and (hyp.fliplr > 0.0 or hyp.flipud > 0.0):
            hyp.fliplr = hyp.flipud = 0.0  # both fliplr and flipud require flip_idx
            LOGGER.warning("No 'flip_idx' array defined in data.yaml, disabling 'fliplr' and 'flipud' augmentations.")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
            Albumentations(p=1.0, transforms=_hyp_get(hyp, "augmentations", None), flip_idx=flip_idx),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=flip_idx),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms


