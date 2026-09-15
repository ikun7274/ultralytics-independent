"""Regression: the installed pool dataset must not touch the VAL (``augment=False``) build.

``InstalledYOLODataset`` swaps the mixed-pool ``__len__`` and ``get_image_and_label`` in for every
``build_yolo_dataset`` call, val included. On a val build (``mode="val"`` -> ``augment=False``)
upstream's ``YOLODataset.build_transforms`` returns a bare val transform, so ``v8_transforms`` never
runs and NONE of ``slice_transform`` / ``img_origin`` / ``*_keep`` are mirrored onto the object. Every
pool segment then resolves to 0, so the pool answer for ``len()`` is 0 -- and 0 is not a merely-empty
val set:

    build_dataloader opens with ``batch = min(batch, len(dataset))`` (ultralytics/data/build.py), so a
    0-length val dataset rewrites its own batch size to 0, and torch raises
    ``ValueError: batch_size should be a positive integer value, but got batch_size=0`` from inside
    ``_build_train_pipeline``'s ``self.test_loader = self.get_dataloader(...)``. Training dies before
    epoch 1, and the traceback never mentions the pool, img_origin or slice_prob.

Measured before the fix: len(val) == 0 with ``img_origin=False`` (the package default, and the value
``train.py`` ships). It appeared to work only when ``img_origin=True``, because that one knob is read
through ``_online_default`` even when unset, which accidentally made val ``len == N`` whole frames.

The contract these tests lock: a val build is byte-for-byte the stock dataset, and the pool answer
applies to training builds only.

Run: python -m pytest tests/test_ooo_val_dataset.py -q
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

S = 96  # image side
B = 24  # box side, pinned to a quadrant centre so it never straddles the central seam band
QUAD = [(24, 24), (72, 24), (24, 72), (72, 72)]

N_TRAIN = 8
N_VAL = 3


def _write_split(root: Path, split: str, n: int, first: int) -> None:
    """Write ``n`` images (one quadrant-pinned box each) into ``images/<split>`` + ``labels/<split>``."""
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(first, first + n):
        rng = np.random.default_rng(i)
        im = rng.integers(0, 255, size=(S, S, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img{i:03d}.png"), im)
        cx, cy = QUAD[i % 4]
        (lbl_dir / f"img{i:03d}.txt").write_text(
            f"0 {(cx + 0.5) / S:.6f} {(cy + 0.5) / S:.6f} {B / S:.6f} {B / S:.6f}\n", encoding="utf-8"
        )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Two-split dataset: 8 train images, 3 val images (the shape ``_mini_val_set`` has)."""
    _write_split(tmp_path, "train", N_TRAIN, first=0)
    _write_split(tmp_path, "val", N_VAL, first=100)
    return tmp_path


def _build(root: Path, mode: str, **overrides):
    """Drive the real factory exactly like ``DetectionTrainer.build_dataset`` (rect on for val)."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics_ooo import install

    install()
    cfg_in = dict(
        task="detect", mode="train", imgsz=64, fraction=1.0, workers=0,
        slice_prob=True, slice_all_tiles=False, slice_target_tiles=True, slice_ratio=0.5,
        img_origin=False, slice_background_ratio=-1,
    )
    cfg_in.update(overrides)
    cfg = get_cfg(overrides=cfg_in)
    data = {
        "path": str(root), "names": {0: "obj"}, "channels": 3, "nc": 1,
        "train": "images/train", "val": "images/val",
    }
    return build_yolo_dataset(
        cfg, str(root / "images" / mode), 4, data, mode=mode, rect=mode == "val", stride=32
    )


# --------------------------------------------------------------------------- the val build is upstream


@pytest.mark.parametrize("img_origin", [False, True])
def test_val_build_reports_one_slot_per_image(root, img_origin):
    """``len(val)`` is the image count for BOTH ``img_origin`` settings.

    ``img_origin=True`` masked this bug rather than causing it: ``_img_origin_on()`` reads the knob
    through ``_online_default`` even though ``v8_transforms`` never mirrored it onto the val build, so
    val accidentally got the right width (N) out of the origin segment. With the knob off, the very same
    build reported 0 -- which is what made the val batch size collapse to 0.
    """
    ds = _build(root, "val", img_origin=img_origin)
    assert ds.augment is False
    assert len(ds.labels) == N_VAL
    assert len(ds) == N_VAL


@pytest.mark.parametrize("img_origin", [False, True])
def test_val_never_allocates_slicing_tiles(root, img_origin):
    """Characterisation: a val build has no ``slice_transform``, so its base segment is structurally empty.

    This is why the delegation above is a correctness requirement rather than a fast path -- the pool
    layout simply does not describe a val build, whatever the other knobs say.
    """
    ds = _build(root, "val", img_origin=img_origin)
    assert getattr(ds, "slice_transform", None) is None
    assert ds._segment_lengths()[0] == 0


def test_val_items_are_whole_frames_with_their_ground_truth(root):
    """Val indices resolve to originals: full-size frames, targets intact, never a sliced tile.

    A pool slot would be a ``slice_at`` tile (~58px here) whose labels are re-normalised into the tile,
    so both ``ori_shape`` and the box coordinates discriminate. Coordinates are compared against the
    label FILE the item reports, which keeps the assertion independent of rect-mode sort order.
    """
    ds = _build(root, "val")
    for i in range(len(ds)):
        lab = ds.get_image_and_label(i)
        assert tuple(lab["ori_shape"]) == (S, S), f"index {i} is not a whole frame"
        boxes = np.asarray(lab["instances"].bboxes)
        assert boxes.shape[0] == 1
        row = (root / "labels" / "val" / f"{Path(lab['im_file']).stem}.txt").read_text(encoding="utf-8")
        _, cx, cy, w, h = (float(v) for v in row.split())
        # this upstream version keeps ``instances.bboxes`` in the cache's own NORMALISED xywh form
        assert boxes[0][0] == pytest.approx(cx, abs=1e-3)
        assert boxes[0][1] == pytest.approx(cy, abs=1e-3)
        assert boxes[0][2] == pytest.approx(w, abs=1e-3)
        assert boxes[0][3] == pytest.approx(h, abs=1e-3)


def test_the_val_dataloader_can_actually_be_built(root):
    """The exact failure: build_dataloader's ``batch = min(batch, len(dataset))`` must not become 0.

    This is what ``_build_train_pipeline`` does for ``test_loader`` (batch_size * 2 for detect), so a
    0-length val set is a hard crash before epoch 1 rather than a warning.
    """
    from ultralytics.data.build import build_dataloader

    ds = _build(root, "val")
    loader = build_dataloader(
        ds, batch=8, workers=0, shuffle=False, rank=-1, drop_last=False, pin_memory=False, device="cpu"
    )
    assert loader.batch_size == min(8, len(ds)) == N_VAL
    batch = next(iter(loader))
    assert batch["img"].shape[0] == N_VAL


# ----------------------------------------------------------------------- the train build still pools


def test_the_guard_does_not_leak_into_the_training_build(root):
    """``augment=True`` must keep reading the pool layout (base only, here: no *_keep passed)."""
    ds = _build(root, "train")
    assert ds.augment is True
    assert len(ds) == sum(ds._segment_lengths())
    # slice_ratio 0.5 over 8 images with slice_all_tiles=False -> one slot per selected image, each
    # bound to a target-bearing tile by the schedule
    ds.set_epoch(0, 2)
    assert len(ds) == len(ds._tile_units) == 4


def test_a_train_build_with_img_origin_still_adds_the_whole_frame_segment(root):
    """The knob that used to hide the bug still behaves: img_origin adds N whole frames."""
    assert len(_build(root, "val", img_origin=True)) == N_VAL  # val is unaffected either way
    assert len(_build(root, "train", img_origin=True)) == 4 + N_TRAIN
