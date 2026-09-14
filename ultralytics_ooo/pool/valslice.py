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
        n_sliced = n_passthrough = 0
        for i, lb in enumerate(self.labels):
            h, w = lb.get("shape", (0, 0))[:2]
            self._shapes.append((int(h), int(w)))
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
                metas.append([(0, 0, w, h)])
                n_passthrough += 1
        self._counts = counts
        self._metas = metas
        self._cum = [0]
        for c in counts:
            self._cum.append(self._cum[-1] + c)

    def __len__(self) -> int:
        return self._cum[-1]

    def _decode(self, index: int):
        i = bisect.bisect_right(self._cum, index) - 1
        return i, index - self._cum[i]

    def __getitem__(self, index: int) -> dict[str, Any]:
        oi, k = self._decode(index)
        x0, y0, x1, y1 = self._metas[oi][k]
        h, w = self._shapes[oi]
        sliced = (x0, y0, x1, y1) != (0, 0, w, h)
        label = deepcopy(self.labels[oi])
        label.pop("shape", None)
        # Prefer the base dataset's cached reader; fall back to a direct imread on any failure.
        try:
            im = self.base._load_image_cached(oi)
        except AttributeError:
            im = imread(label["im_file"])
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(f"SliceValDataset: cached read of {label['im_file']!r} failed ({e}); direct imread.")
            im = imread(label["im_file"])
        if im is None:
            raise FileNotFoundError(f"SliceValDataset: failed to load image {label['im_file']!r}.")
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


def _remap_boxes(self, boxes, meta, imgsz: int):
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
    from pathlib import Path

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
    _orig_update_metrics = validator_cls.update_metrics
    _orig_finalize_metrics = validator_cls.finalize_metrics
    _orig_gather_stats = validator_cls.gather_stats
    _orig_get_dataloader = validator_cls.get_dataloader

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        _orig_init(self, dataloader, save_dir, args, _callbacks)
        g = (lambda k, d: args.get(k, d) if isinstance(args, dict) else getattr(args, k, d))
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
        _orig_update_metrics(self, preds, batch)

    def finalize_metrics(self):
        if getattr(self, "_slice_acc", None):
            unfinished = len(self._slice_acc)
            LOGGER.warning(
                f"val_slice: {unfinished} original image(s) never completed all sub-tiles and were excluded "
                f"(scored originals: {self.seen})."
            )
            self._slice_acc.clear()
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

        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        sliced = _val_slice_active(self) and getattr(self.args, "task", "detect") == "detect"
        if sliced:
            if self.args.rect:
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
            drop_last=(self.args.compile and not sliced),
            pin_memory=self.training,
            device=self.device,
        )

    validator_cls.__init__ = __init__
    validator_cls.update_metrics = update_metrics
    validator_cls.finalize_metrics = finalize_metrics
    validator_cls.gather_stats = gather_stats
    validator_cls.get_dataloader = get_dataloader
    validator_cls._val_slice_active = _val_slice_active
    validator_cls._remap_boxes = _remap_boxes
    validator_cls._class_wise_nms = staticmethod(_class_wise_nms)
    validator_cls._gt_orig_pixels = _gt_orig_pixels
    validator_cls._update_metrics_sliced = _update_metrics_sliced
    validator_cls._finalize_sliced_orig = _finalize_sliced_orig
    validator_cls._ooo_valslice_patched = True
