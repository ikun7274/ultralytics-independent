"""Validation-side online slicing (SAHI eval): SliceValDataset + DetectionValidator patching.

Expands each val image into 2x2(+overlap) sub-tiles, infers every tile, remaps predictions back to
original-image coordinates and fuses duplicates with class-wise NMS before scoring against the whole-image
GT. Installed onto a stock Ultralytics by monkey-patching ``DetectionValidator``; upstream sources are
never edited. When ``val_slice_enable`` is False the patched validator behaves byte-for-byte upstream.
"""

from __future__ import annotations

import bisect
import random
import zlib
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.patches import imread
from ultralytics_ooo.core import slice_geometry


class SliceValDataset(torch.utils.data.Dataset):
    """Validation-side online slicing: expand each val image into 2x2 (+overlap) sub-tiles (SAHI eval).

    Wraps the native validation dataset and reuses its labels / transforms / collate_fn, so the
    validator pipeline stays untouched except for prediction remapping + NMS fusion. Every sub-tile is
    inferred (no target filtering -- val side evaluates slicing INFERENCE; overlapping duplicates are
    merged later by NMS).
    """

    def __init__(self, base, overlap_ratio: float = 0.2, all_tiles: bool = True, ratio: float = 1.0) -> None:
        self.base = base
        self.labels = base.labels
        self.n = len(self.labels)
        self.overlap_ratio = overlap_ratio
        self.all_tiles = all_tiles
        self.ratio = float(ratio)
        self.transforms = base.transforms
        self.collate_fn = base.collate_fn
        self._build()

    def _build(self) -> None:
        n = self.n
        mask = None
        if 0.0 <= self.ratio < 1.0:
            mask = np.zeros(n, dtype=bool)
            if self.ratio > 0:
                seed = zlib.crc32(
                    "\n".join(str(lb.get("im_file", "")) for lb in self.labels).encode("utf-8", "ignore")
                )
                mask[random.Random(seed).sample(range(n), int(round(self.ratio * n)))] = True
        self._mask = mask
        counts, metas = [], []
        self._shapes = []
        n_sliced = n_passthrough = n_unknown = 0
        for i, lb in enumerate(self.labels):
            raw = lb.get("shape")
            # ``shape`` is the ONLY source of the original H/W here, and it is what slice_geometry()
            # needs to compute tiles. Upstream's YOLODataset always carries it on ``self.labels`` (the
            # ``label.pop("shape")`` in ``BaseDataset.get_image_and_label`` acts on a deepcopy), and
            # the one path that DOES strip it -- ``BaseDataset.set_rectangle`` -- is neutralised by the
            # rect=False rebuild in ``DetectionValidator.get_dataloader``. If it ever goes missing the
            # whole dataset would silently degrade to "log in, log out, no slicing at all", so say so.
            h, w = (int(raw[0]), int(raw[1])) if raw else (0, 0)
            self._shapes.append((h, w))
            eligible = (mask is None or bool(mask[i])) and h >= 2 and w >= 2
            if eligible:
                tiles = slice_geometry(w, h, self.overlap_ratio)
                if self.all_tiles:
                    counts.append(len(tiles))
                    metas.append(tiles)
                else:
                    counts.append(1)
                    metas.append([random.choice(tiles)])
                n_sliced += 1
            else:
                counts.append(1)
                if raw:
                    metas.append([(0, 0, w, h)])
                else:
                    # Sentinel: width/height are unknown, so the loaded frame itself is authoritative
                    # (see __getitem__). Never store (0, 0, 0, 0) -- that slices to a 0x0 array and
                    # turns the failure into a downstream cv2.resize crash inside a DataLoader worker.
                    metas.append([None])
                    n_unknown += 1
                n_passthrough += 1
        if n_unknown:
            LOGGER.warning(
                f"SliceValDataset: {n_unknown}/{n} val labels carry no 'shape' key, so their geometry "
                "is unknown at construction -- those images are passed through unsliced (dims taken "
                "from the loaded frame). Slicing validation is therefore only partial; check that the "
                "val dataset is a plain YOLODataset (rect=False)."
            )
        self._counts = counts
        self._metas = metas
        self._cum = [0]
        for c in counts:
            self._cum.append(self._cum[-1] + c)

    def __len__(self) -> int:
        return self._cum[-1]

    def _read_orig(self, oi: int, im_file: str):
        """Read one ORIGINAL-resolution frame, preferring the base dataset's cached reader.

        ``copy=False`` is safe here because the caller only ever SLICES the returned array
        (``np.ascontiguousarray(im[y0:y1, x0:x1])`` copies the tile out and nothing writes into ``im``),
        and it skips one full-resolution memcpy per tile -- measured 13.6 ms at 4000x3000, i.e. ~54 ms
        per image with ``all_tiles=True``. A base whose reader predates the ``copy`` kwarg (plain
        YOLODataset, test doubles) degrades to the copying call; any other failure degrades to a direct
        ``imread``, which is what this did before.
        """
        reader = getattr(self.base, "_load_image_cached", None)
        if reader is not None:
            try:
                return reader(oi, copy=False)
            except TypeError:  # base reader has no `copy` kwarg
                try:
                    return reader(oi)
                except Exception as e:  # noqa: BLE001
                    LOGGER.warning(f"SliceValDataset: cached read of {im_file!r} failed ({e}); direct imread.")
            except Exception as e:  # noqa: BLE001
                LOGGER.warning(f"SliceValDataset: cached read of {im_file!r} failed ({e}); direct imread.")
        return imread(im_file)

    def _decode(self, index: int):
        i = bisect.bisect_right(self._cum, index) - 1
        return i, index - self._cum[i]

    def __getitem__(self, index: int) -> dict[str, Any]:
        oi, k = self._decode(index)
        meta = self._metas[oi][k]
        label = deepcopy(self.labels[oi])
        label.pop("shape", None)
        im = self._read_orig(oi, label["im_file"])
        if im is None:
            raise FileNotFoundError(f"SliceValDataset: failed to load image {label['im_file']!r}.")
        if meta is None:
            # Construction-time sentinel: the label had no 'shape', so the frame is the only authority
            # on H/W. Treat it as a whole-frame passthrough (never a 0x0 crop).
            h, w = int(im.shape[0]), int(im.shape[1])
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            x0, y0, x1, y1 = meta
            h, w = self._shapes[oi]
        sliced = (x0, y0, x1, y1) != (0, 0, w, h)
        tw, th = x1 - x0, y1 - y0
        sub = np.ascontiguousarray(im[y0:y1, x0:x1])
        if sliced:
            boxes = np.asarray(label["bboxes"], dtype=np.float64)
            if len(boxes):
                cx, cy, bw, bh = boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h
                xa, ya, xb, yb = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                lx0 = np.clip(xa - x0, 0.0, tw)
                ly0 = np.clip(ya - y0, 0.0, th)
                lx1 = np.clip(xb - x0, 0.0, tw)
                ly1 = np.clip(yb - y0, 0.0, th)
                nw2, nh2 = lx1 - lx0, ly1 - ly0
                keep = (nw2 > 1) & (nh2 > 1)
                if keep.any():
                    lx0, ly0, lx1, ly1 = lx0[keep], ly0[keep], lx1[keep], ly1[keep]
                    label["bboxes"] = np.stack(
                        [(lx0 + lx1) / 2 / tw, (ly0 + ly1) / 2 / th, (lx1 - lx0) / tw, (ly1 - ly0) / th], axis=1
                    ).astype(np.float32)
                    label["cls"] = np.asarray(label["cls"])[keep]
                else:
                    label["bboxes"] = np.empty((0, 4), dtype=np.float32)
                    label["cls"] = np.empty((0, 1), dtype=np.float32)
            else:
                label["bboxes"] = np.empty((0, 4), dtype=np.float32)
            label["bbox_format"] = "xywh"
            label["normalized"] = True
            label["segments"] = []
            if label.get("keypoints") is not None:
                label["keypoints"] = np.empty((0, 0, 3), dtype=np.float32)
        label["img"] = sub
        label = self.base.update_labels_info(label) if hasattr(self.base, "update_labels_info") else label
        label["ori_shape"] = (th, tw)
        label["resized_shape"] = sub.shape[:2]
        label["ratio_pad"] = (1.0, 1.0)
        label["val_slice_meta"] = {
            "orig_idx": oi,
            "k": k,
            "offset": (x0, y0),
            "tile_shape": (th, tw),
            "orig_shape": (h, w),
            "n_tiles": len(self._metas[oi]),
            "sliced": bool(sliced),
        }
        return self.transforms(label)


def _val_slice_active(self) -> bool:
    return bool(getattr(self.args, "val_slice_enable", False))


_SLICE_INERT_LOGGED = False


def _log_sliced_units() -> None:
    """Say ONCE that, in sliced mode, upstream's per-image figures are per SUB-TILE.

    ``BaseValidator.__call__`` divides the wall clock by ``len(self.dataloader.dataset)`` and the val
    loss by ``len(self.dataloader)``. In sliced mode those are the tile count and the tile-batch count,
    i.e. roughly K times the image count, so the printed "Speed: ... per image" is really per tile and
    ``val/box_loss`` is on a different denominator than ``whole_metrics/val/box_loss`` (measured on a
    3-image val set: 0.83794 vs 0.41349 for the same run). The two口径 must not be compared directly,
    and the number is not the per-image latency a deployment decision would use.
    """
    global _SLICE_UNITS_LOGGED
    if _SLICE_UNITS_LOGGED:
        return
    _SLICE_UNITS_LOGGED = True
    LOGGER.info(
        "val_slice: this validation runs on sub-tiles, so the 'Speed: ... per image' line divides by "
        "the TILE count and val/*_loss divides by the tile-batch count -- both are per sub-tile, not "
        "per original image, and they are NOT comparable with whole_metrics/val/*_loss. The mAP/精度 "
        "figures are unaffected: _finalize_sliced_orig maps every prediction back to ORIGINAL-image "
        "pixels and scores one entry per original image."
    )


_SLICE_UNITS_LOGGED = False


def _log_slicing_not_in_effect() -> None:
    """Escalate ONCE when ``val_slice_enable=True`` but the batch holds no sub-tile metadata.

    This is the hard signal for this package's worst failure mode: a switch that is ON and does
    nothing. It really happened. ``DetectionTrainer.get_validator`` builds the validator as
    ``DetectionValidator(self.test_loader, ...)`` -- a WHOLE-IMAGE loader -- and
    ``BaseValidator.__call__`` only opens ``self.dataloader = self.dataloader or
    self.get_dataloader(...)`` in its NON-training branch. During training it therefore never rebuilt
    the attribute, ``DetectionValidator.get_dataloader`` (the only site that constructs
    ``SliceValDataset``) was never reached, and the "sliced" primary metric -- the one driving
    fitness / best.pt / early stopping -- was silently the whole-image metric.

    ``ensure_sliced_val_loader`` repairs the wiring; this warning exists so that any FUTURE way of
    losing it (an upstream refactor of ``__call__``, a custom validator, a hand-built dataloader)
    fails loudly instead of silently degrading to whole-image scoring.
    """
    global _SLICE_INERT_LOGGED
    if _SLICE_INERT_LOGGED:
        return
    _SLICE_INERT_LOGGED = True
    LOGGER.error(
        "val_slice_enable=True but this validation batch carries no 'val_slice_meta': the SLICED "
        "metric is NOT being computed and this pass scores whole images instead. The validator is "
        "not running on sliced sub-tiles -- check that its dataloader is a SliceValDataset."
    )


def ensure_sliced_val_loader(trainer) -> bool:
    """Install the SLICED validation loader onto ``trainer.validator`` (idempotent per run).

    Returns ``True`` when a new sliced loader was built.

    Why the validator cannot do this for itself:

    * ``DetectionTrainer.get_validator`` (models/yolo/detect/train.py) constructs the validator as
      ``DetectionValidator(self.test_loader, ...)``, so ``validator.dataloader`` is already a
      truthy WHOLE-IMAGE loader.
    * ``BaseValidator.__call__`` reaches ``self.dataloader = self.dataloader or
      self.get_dataloader(...)`` only in its ``else`` (non-training) branch, i.e. during training it
      never rebuilds the attribute.
    * ``DetectionValidator.get_dataloader`` is the ONLY place a ``SliceValDataset`` is constructed.

    So without this helper ``val_slice_enable=True`` produced zero sliced batches during training
    (measured: ``[metric] update_metrics -> STOCK, batch_has_val_slice_meta=False`` on the primary
    pass, with the sliced metric only ever appearing for a standalone ``model.val()``).

    The rebuilt loader is cached on the validator, so a multi-epoch run pays one label scan, not one
    per epoch. It is also a no-op when slicing is off, when the task is not ``detect``, or when the
    validator already holds a sliced loader -- including the loader ``_run_whole`` restores after the
    second pass.
    """
    v = getattr(trainer, "validator", None)
    if v is None:
        return False
    vargs = getattr(v, "args", None)
    if not bool(getattr(vargs, "val_slice_enable", False)):
        return False  # slicing off -> never touch the loader
    if getattr(vargs, "task", "detect") != "detect":
        return False  # slicing is detect-only; get_dataloader would hand back a plain loader
    current = getattr(v, "dataloader", None)
    if isinstance(getattr(current, "dataset", None), SliceValDataset):
        return False  # already sliced (kept from a previous epoch / restored by _run_whole)
    data = getattr(trainer, "data", None) or getattr(v, "data", None)
    if not isinstance(data, dict) or not data.get("val"):
        return False  # nothing to build from
    # ``DetectionValidator.build_dataset`` reads ``self.data``, and during training the validator
    # only assigns it INSIDE BaseValidator.__call__ (``self.data = trainer.data``), i.e. after this
    # hook has run. Publish it first -- the value is the very same dict the validator would set.
    # (_run_whole gets away without this only because it runs after the primary pass, by which time
    # __call__ has already populated the attribute.)
    if getattr(v, "data", None) is None:
        v.data = data
    # The validator also needs ``device`` and ``stride`` for get_dataloader/build_dataset; during
    # training ``__call__`` has not set either yet, so publish the trainer's values first -- they are
    # exactly what ``__call__`` assigns (``self.device = trainer.device``). Leaving ``device`` as
    # None makes build_dataloader -> get_torch_device_backend("None") raise
    # "RuntimeError: Expected one of cpu, cuda, ...".
    if getattr(v, "device", None) is None:
        v.device = getattr(trainer, "device", torch.device("cpu"))
    if getattr(v, "stride", None) is None:
        v.stride = getattr(trainer, "stride", 32)
    v.dataloader = v.get_dataloader(data["val"], getattr(trainer, "batch_size", 1))
    built = isinstance(getattr(v.dataloader, "dataset", None), SliceValDataset)
    if not built:
        LOGGER.warning(
            "val_slice_enable=True but rebuilding the validator dataloader did not yield a "
            "SliceValDataset; sliced validation will not run."
        )
    return built


def _remap_boxes(boxes, meta, imgsz: int):
    """Map tile-local predicted boxes back to ORIGINAL-image pixel coordinates.

    A plain function (registered with ``staticmethod`` below), not a method: it needs no validator state.
    """
    th, tw = int(meta["tile_shape"][0]), int(meta["tile_shape"][1])
    x0, y0 = int(meta["offset"][0]), int(meta["offset"][1])
    r = min(1.0, imgsz / max(th, tw))
    left = round((imgsz - round(tw * r)) / 2.0 - 0.1)
    top = round((imgsz - round(th * r)) / 2.0 - 0.1)
    out = boxes.clone()
    if out.shape[0]:
        out[:, [0, 2]] = (out[:, [0, 2]] - left) / r + x0
        out[:, [1, 3]] = (out[:, [1, 3]] - top) / r + y0
    return out


def _class_wise_nms(boxes, conf, cls, iou_thres: float):
    import torchvision

    return torchvision.ops.batched_nms(boxes, conf, cls.long(), iou_thres)


def _gt_orig_pixels(self, lb: dict, h: int, w: int):
    key = lb["im_file"]
    hit = self._gt_cache.get(key)
    if hit is not None:
        return hit
    boxes = np.asarray(lb["bboxes"], dtype=np.float64)
    cls = np.asarray(lb["cls"]).reshape(-1)
    cls_t = torch.as_tensor(cls, dtype=torch.float32, device=self.device)
    if len(boxes):
        cx, cy, bw, bh = boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        boxes_t = torch.as_tensor(xyxy, dtype=torch.float32, device=self.device)
    else:
        boxes_t = torch.empty((0, 4), dtype=torch.float32, device=self.device)
    self._gt_cache[key] = (cls_t, boxes_t)
    return cls_t, boxes_t


def _update_metrics_sliced(self, preds, batch) -> None:
    if self._slice_base_labels is None:
        raise RuntimeError(
            "val_slice: sub-tile batches arrived but whole-image GT is unset. The dataloader must be built "
            "via DetectionValidator.get_dataloader."
        )
    _b, _c, h_img, w_img = batch["img"].shape
    if h_img != w_img:
        raise RuntimeError("val_slice assumes a square validation canvas; use a square imgsz.")
    imgsz = h_img
    for si, pred in enumerate(preds):
        meta = batch["val_slice_meta"][si]
        oi = int(meta["orig_idx"])
        acc = self._slice_acc.get(oi)
        if acc is None:
            acc = self._slice_acc[oi] = {
                "preds": [],
                "done": 0,
                "n_tiles": int(meta["n_tiles"]),
                "im_file": batch["im_file"][si],
            }
        if pred["cls"].shape[0]:
            boxes = self._remap_boxes(pred["bboxes"], meta, imgsz)
            acc["preds"].append({"bboxes": boxes, "conf": pred["conf"], "cls": pred["cls"], "extra": pred["extra"]})
        acc["done"] += 1
        if acc["done"] >= acc["n_tiles"]:
            _finalize_sliced_orig(self, oi, acc)


def _finalize_sliced_orig(self, oi: int, acc: dict) -> None:
    from pathlib import Path

    self.seen += 1
    if acc["preds"]:
        boxes = torch.cat([p["bboxes"] for p in acc["preds"]], 0)
        conf = torch.cat([p["conf"] for p in acc["preds"]], 0)
        cls = torch.cat([p["cls"] for p in acc["preds"]], 0)
        keep = _class_wise_nms(boxes, conf, cls, self.val_slice_nms_iou)
        boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
        extra = (
            torch.cat([p["extra"] for p in acc["preds"]], 0)[keep]
            if any(p["extra"] is not None for p in acc["preds"])
            else None
        )
    else:
        boxes = torch.empty((0, 4), device=self.device)
        conf = torch.empty(0, device=self.device)
        cls = torch.empty(0, device=self.device)
        extra = None
    lb = self._slice_base_labels[oi]
    h, w = int(lb["shape"][0]), int(lb["shape"][1])
    gt_cls, gt_boxes = _gt_orig_pixels(self, lb, h, w)
    pbatch = {"cls": gt_cls, "bboxes": gt_boxes}
    predn = {"bboxes": boxes, "conf": conf, "cls": cls, "extra": extra}
    if self.args.single_cls:
        predn["cls"] *= 0
    no_pred = predn["cls"].shape[0] == 0
    target_cls = gt_cls.cpu().numpy()
    # Confusion matrix. Upstream fills it inside update_metrics under `if self.args.plots`; the sliced
    # path skipped it entirely, so a sliced run with plots=True wrote an ALL-ZERO
    # confusion_matrix.png with nothing anywhere saying it was empty. predn/pbatch are exactly the
    # (detections, batch) pair upstream passes -- both in ORIGINAL-image pixel coordinates.
    if self.args.plots:
        self.confusion_matrix.process_batch(predn, pbatch, conf=self.confusion_matrix_conf)
    self.metrics.update_stats(
        {
            **self._process_batch(predn, pbatch),
            "target_cls": target_cls,
            "target_img": np.unique(target_cls),
            "conf": np.zeros(0) if no_pred else conf.cpu().numpy(),
            "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
            "im_name": Path(acc["im_file"]).name,
        }
    )
    del self._slice_acc[oi]


def patch_validator(validator_cls) -> None:
    """Install sliced-validation onto a stock DetectionValidator (idempotent)."""
    if getattr(validator_cls, "_ooo_valslice_patched", False):
        return

    _orig_init = validator_cls.__init__
    _orig_call = validator_cls.__call__
    _orig_update_metrics = validator_cls.update_metrics
    _orig_finalize_metrics = validator_cls.finalize_metrics
    _orig_gather_stats = validator_cls.gather_stats
    _orig_get_dataloader = validator_cls.get_dataloader

    def __call__(self, trainer=None, model=None):
        """Make sure the validator owns the SLICED loader before a training-mode pass.

        The hook lives here rather than in ``dual.py`` so that sliced validation works even when the
        dual-metric layer is not installed, and so the loader fix cannot be lost by a future change
        to the dual wrapper. ``trainer is None`` (standalone ``model.val()`` / ``final_eval``) takes
        upstream's own path, which already builds the sliced loader.
        """
        if trainer is not None:
            ensure_sliced_val_loader(trainer)
        return _orig_call(self, trainer=trainer, model=model)

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        _orig_init(self, dataloader, save_dir, args, _callbacks)

        def g(k, d):
            """Read a val_slice_* knob from either a dict or a namespace config."""
            return args.get(k, d) if isinstance(args, dict) else getattr(args, k, d)

        self.val_slice_overlap_ratio = float(g("val_slice_overlap_ratio", 0.2))
        self.val_slice_all_tiles = bool(g("val_slice_all_tiles", False))
        self.val_slice_ratio = float(g("val_slice_ratio", 1.0))
        self.val_slice_nms_iou = float(g("val_slice_nms_iou", 0.5))
        self._slice_acc = {}
        self._slice_base_labels = None
        self._gt_cache = {}

    def update_metrics(self, preds, batch):
        if _val_slice_active(self) and "val_slice_meta" in batch:
            _update_metrics_sliced(self, preds, batch)
            return
        if _val_slice_active(self):
            _log_slicing_not_in_effect()
        _orig_update_metrics(self, preds, batch)

    def finalize_metrics(self):
        if getattr(self, "_slice_acc", None):
            unfinished = len(self._slice_acc)
            LOGGER.warning(
                f"val_slice: {unfinished} original image(s) never completed all sub-tiles and were excluded "
                f"(scored originals: {self.seen})."
            )
            self._slice_acc.clear()
        if _val_slice_active(self):
            _log_sliced_units()
        _orig_finalize_metrics(self)

    def gather_stats(self):
        _orig_gather_stats(self)
        # Sliced mode already counts ORIGINAL images in _finalize_sliced_orig; don't overwrite with tile count.
        if not _val_slice_active(self):
            try:
                self.seen = len(self.dataloader.dataset)
            except Exception:  # noqa: BLE001
                pass

    def get_dataloader(self, dataset_path, batch_size=1):
        from ultralytics.data.build import build_dataloader

        sliced = _val_slice_active(self) and getattr(self.args, "task", "detect") == "detect"
        if not sliced:
            # Nothing to add when slicing is off: delegate to upstream. This used to re-implement the
            # whole stock tail (identical parameters, so no behaviour change) -- which is exactly the
            # kind of duplicate that silently drifts when upstream edits build_dataloader. Delegating
            # also removes the previously-dead ``_orig_get_dataloader`` capture.
            return _orig_get_dataloader(self, dataset_path, batch_size)
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        if self.args.rect:
            # ``set_rectangle`` reorders the labels and strips their "shape", which SliceValDataset
            # needs for slice_geometry (and which the rect batch_shapes would override anyway).
            self.args.rect = False
            dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        dataset = SliceValDataset(
            dataset,
            overlap_ratio=self.val_slice_overlap_ratio,
            all_tiles=self.val_slice_all_tiles,
            ratio=self.val_slice_ratio,
        )
        self._slice_base_labels = dataset.base.labels
        return build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=False,
            rank=-1,
            drop_last=False,  # == (self.args.compile and not sliced) with sliced=True
            pin_memory=self.training,
            device=self.device,
        )

    validator_cls.__init__ = __init__
    validator_cls.__call__ = __call__
    validator_cls.update_metrics = update_metrics
    validator_cls.finalize_metrics = finalize_metrics
    validator_cls.gather_stats = gather_stats
    validator_cls.get_dataloader = get_dataloader
    validator_cls.ensure_sliced_val_loader = staticmethod(ensure_sliced_val_loader)
    validator_cls._val_slice_active = _val_slice_active
    validator_cls._remap_boxes = staticmethod(_remap_boxes)
    validator_cls._class_wise_nms = staticmethod(_class_wise_nms)
    validator_cls._gt_orig_pixels = _gt_orig_pixels
    validator_cls._update_metrics_sliced = _update_metrics_sliced
    validator_cls._finalize_sliced_orig = _finalize_sliced_orig
    validator_cls._ooo_valslice_patched = True
