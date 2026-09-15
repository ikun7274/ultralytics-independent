"""Integration: ``slice_target_tiles`` -- scheduling BASE slots as target-bearing (image, tile) pairs.

The knob exists because ``slice_all_tiles=False`` allocates ONE slot per selected image and then lets
``OnlineSlice.__call__`` pick the tile with ``random.randrange(4)``, i.e. blindly: measured on a
single-quadrant-box dataset only ~24% of those slots land on the target and the rest arrive as empty
tiles. With the schedule on, the slot is bound to a tile that really keeps a target, no unit repeats
until the whole queue has been consumed, and the queue then restarts in a fresh order.

These tests drive the real ``build_yolo_dataset`` factory on a synthetic set written to a temp dir
(no downloads, no weights) and assert the four properties that matter: positivity, no repeats inside a
pass, restart after a pass, and an untouched pool LENGTH.

Run: python -m pytest tests/test_ooo_target_tiles.py -q
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

S = 96  # image side; a 2x2 grid over it gives ~58 px tiles with the default 20% overlap
B = 24  # box side, pinned to a quadrant centre so it never straddles the central seam band
QUAD = [(24, 24), (72, 24), (24, 72), (72, 72)]

EPOCHS = 20


def _write_dataset(root, n, boxed=None):
    """Write ``n`` images; image ``i`` gets one quadrant-pinned box unless ``boxed`` excludes it.

    One box in one quadrant means exactly ONE of the four tiles keeps a target, so the scheduler's
    queue length is exactly the number of boxed images and every assertion below is exact.
    """
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    boxed = set(range(n)) if boxed is None else set(boxed)
    for i in range(n):
        rng = np.random.default_rng(i)
        im = rng.integers(0, 255, size=(S, S, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img{i:03d}.png"), im)
        line = ""
        if i in boxed:
            cx, cy = QUAD[i % 4]
            line = f"0 {(cx + 0.5) / S:.6f} {(cy + 0.5) / S:.6f} {B / S:.6f} {B / S:.6f}\n"
        (lbl_dir / f"img{i:03d}.txt").write_text(line, encoding="utf-8")
    return img_dir


def _build(root, n=12, boxed=None, **overrides):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics_ooo import install

    install()
    img_dir = _write_dataset(root, n, boxed)
    cfg_in = dict(task="detect", mode="train", imgsz=64, batch=2, fraction=1.0, workers=0,
                  slice_prob=True, slice_all_tiles=False, slice_ratio=0.25,
                  img_origin=False, slice_background_ratio=-1)
    cfg_in.update(overrides)
    cfg = get_cfg(overrides=cfg_in)
    data = {"path": str(root), "names": {0: "obj"}, "channels": 3, "nc": 1, "train": "images/train"}
    return build_yolo_dataset(cfg, str(img_dir), 2, data, mode="train")


def _classify_base(ds):
    """Count BASE slots holding a target tile / an empty tile / a whole frame."""
    shapes = {i: cv2.imread(f).shape[:2] for i, f in enumerate(ds.im_files)}
    stem2idx = {Path(f).stem: i for i, f in enumerate(ds.im_files)}
    base_len = ds._segment_lengths()[0]
    out = {"pos": 0, "empty": 0, "whole": 0, "base": base_len}
    for i in range(base_len):
        lab = ds.get_image_and_label(i)
        j = stem2idx.get(Path(lab["im_file"]).stem, 0)
        if tuple(lab["ori_shape"]) == shapes.get(j, (0, 0)):
            out["whole"] += 1
        elif len(lab["instances"].bboxes) == 0:
            out["empty"] += 1
        else:
            out["pos"] += 1
    return out


def _units(ds, epoch):
    ds.set_epoch(epoch, EPOCHS)
    return list(ds._tile_units)


# --------------------------------------------------------------------------------------- positivity


def test_the_schedule_replaces_blind_empty_tiles_with_target_tiles(tmp_path):
    """The whole point: same pool size, same slot count, but every slot carries a target.

    Off, the blind ``random.randrange(4)`` pick lands on the single target-bearing quadrant tile ~1/4
    of the time and the other three slots arrive as empty tiles (the pool has no whole frames here
    because ``img_origin=False`` and ``bg=-1``). On, every slot is a target tile.
    """
    off = _build(tmp_path / "off", slice_target_tiles=False)
    off.set_epoch(0, EPOCHS)
    r_off = _classify_base(off)

    on = _build(tmp_path / "on", slice_target_tiles=True)
    on.set_epoch(0, EPOCHS)
    r_on = _classify_base(on)

    assert r_on["base"] == r_off["base"] == 3  # round(0.25 * 12)
    assert r_on["pos"] == r_on["base"], r_on  # every slot positive
    assert r_on["empty"] == 0 and r_on["whole"] == 0, r_on
    assert r_off["pos"] < r_off["base"], "the blind pick must NOT be able to fill every slot"
    assert r_off["pos"] + r_off["empty"] + r_off["whole"] == r_off["base"]


def test_the_queue_holds_one_unit_per_target_bearing_tile(tmp_path):
    """A one-box-per-image set contributes exactly one unit per BOXED image, and none for the rest.

    The tile INDEX is deliberately not hardcoded: ``slice_geometry`` numbers the 2x2 grid in its own
    order (measured: top-left 0, bottom-left 1, top-right 2, bottom-right 3), so pinning "quadrant 1 ->
    tile 1" would test the convention rather than the property. What must hold is: one unit per boxed
    image, nothing for an unboxed one, and the four quadrants landing on four DISTINCT tiles.
    """
    ds = _build(tmp_path / "q", n=12, boxed=[0, 1, 2, 3, 4], slice_target_tiles=True)
    ds.set_epoch(0, EPOCHS)
    queue = ds._target_tile_queue()

    assert len(queue) == 5, queue  # 12 images, 5 of them carry a box
    assert {img for img, _tile in queue} == {0, 1, 2, 3, 4}, queue  # unboxed images contribute none
    # images 0..3 sit in the four different quadrants, so they must own four different tiles
    assert len({tile for img, tile in queue if img < 4}) == 4, queue
    assert len([1 for img, _t in queue if img == 4]) == 1, queue


# -------------------------------------------------------------------------------- no repeats / restart


def test_no_unit_repeats_until_the_whole_queue_is_consumed(tmp_path):
    """Consecutive epochs must take DISJOINT blocks; their union must be the entire queue.

    N=12, ratio=0.25 -> K=3 slots per epoch and a 12-unit queue -> one pass is exactly 4 epochs.
    """
    ds = _build(tmp_path / "pass1", slice_target_tiles=True)
    queue = set(ds._target_tile_queue())
    assert len(queue) == 12

    blocks = [set(_units(ds, e)) for e in range(4)]
    assert all(len(b) == 3 for b in blocks), blocks
    for i in range(3):
        assert not (blocks[i] & blocks[i + 1]), f"epoch {i} and {i + 1} overlap: {blocks[i] & blocks[i + 1]}"
    assert set().union(*blocks) == queue, "one pass must cover every target-bearing tile exactly once"


def test_the_pass_restarts_with_a_fresh_order_after_the_queue_is_exhausted(tmp_path):
    """Epoch 4 begins pass 2: the same queue, reshuffled, and again covering everything in 4 epochs."""
    ds = _build(tmp_path / "pass2", slice_target_tiles=True)
    queue = set(ds._target_tile_queue())

    pass1 = [set(_units(ds, e)) for e in range(4)]
    pass2 = [set(_units(ds, e)) for e in range(4, 8)]

    assert set().union(*pass2) == queue, "pass 2 must also cover the whole queue"
    for i in range(3):
        assert not (pass2[i] & pass2[i + 1]), "pass 2 must not repeat inside itself either"
    # the restart is a NEW order, not the same window replayed (this is what "重新开始选取" buys)
    assert ds._target_tile_pass(0) != ds._target_tile_pass(1)
    assert sorted(ds._target_tile_pass(0)) == sorted(ds._target_tile_pass(1)) == sorted(queue)
    assert any(pass1[i] != pass2[i] for i in range(4)), "reshuffling must change at least one block"


def test_the_schedule_is_deterministic_for_a_given_epoch(tmp_path):
    """Workers rebuild the schedule from the epoch alone -- no transport, so it must be reproducible."""
    ds = _build(tmp_path / "det", slice_target_tiles=True)
    first = [_units(ds, e) for e in range(4)]
    again = [_units(ds, e) for e in range(4)]
    assert first == again


# --------------------------------------------------------------------------------- layout invariance


@pytest.mark.parametrize(("ratio", "origin"), [(0.25, False), (0.5, True), (1.0, True)])
def test_the_schedule_never_changes_the_pool_length(tmp_path, ratio, origin):
    """The layout is derived from the ratio alone; the schedule only decides a slot's CONTENT."""
    off = _build(tmp_path / f"Loff{ratio}", slice_target_tiles=False, slice_ratio=ratio, img_origin=origin)
    on = _build(tmp_path / f"Lon{ratio}", slice_target_tiles=True, slice_ratio=ratio, img_origin=origin)
    off.set_epoch(0, EPOCHS)
    on.set_epoch(0, EPOCHS)
    assert len(off) == len(on)
    assert off._segment_lengths() == on._segment_lengths()


def test_the_schedule_leaves_the_other_branches_untouched(tmp_path):
    """It draws from its own seed, so every other branch selects the same images either way.

    Both datasets MUST live at the same path for this to mean anything: ``_mask_seed`` is derived from
    the image-file list, so comparing two datasets in different temp dirs would compare two different
    seeds and fail for a reason that has nothing to do with the schedule.
    """
    over = dict(slice_ratio=0.5, img_origin=True, ratio_pad_keep=True, ratio_pad_ratio=0.5,
                blur_keep=True, blur_ratio=0.5)
    root = tmp_path / "same_path"
    off = _build(root, slice_target_tiles=False, **over)
    on = _build(root, slice_target_tiles=True, **over)
    assert len(off.im_files) == len(on.im_files) > 0
    off.set_epoch(1, EPOCHS)
    on.set_epoch(1, EPOCHS)
    for attr in ("ratio", "blur"):
        assert list(off._sel_indices(attr, 12)) == list(on._sel_indices(attr, 12)), attr
    # ...and the slice branch itself is the one that DIFFERS (images vs scheduled units)
    assert len(on._tile_units) == len(off._sel_indices("slice", 12)) == 6


# ------------------------------------------------------------------------------------- degradations


def test_a_dataset_with_no_target_bearing_tile_warns_and_falls_back(tmp_path, monkeypatch, caplog):
    """An empty queue must not publish an unfillable schedule: warn once, then behave as before."""
    ds = _build(tmp_path / "empty", slice_target_tiles=True)
    monkeypatch.setattr(type(ds.slice_transform), "target_tiles", lambda self, *a, **k: [])
    ds._target_tile_queue_cache = None
    with caplog.at_level("WARNING"):
        ds.set_epoch(0, EPOCHS)
    assert ds._tile_units is None
    assert any("slice_target_tiles" in r.message and "nothing to schedule" in r.message for r in caplog.records)
    # the blind-random base segment still works, so the pool is still fully decodable
    assert _classify_base(ds)["base"] == 3


def test_selecting_more_slots_than_units_warns_about_repeats(tmp_path, caplog):
    """K > queue repeats units inside one epoch -- legal, but variety silently drops, so say so."""
    ds = _build(tmp_path / "repeat", n=8, boxed=[0, 1], slice_ratio=1.0, slice_target_tiles=True)
    queue = set(ds._target_tile_queue())
    with caplog.at_level("WARNING"):
        ds.set_epoch(0, EPOCHS)
    units = ds._tile_units
    assert len(queue) == 2, queue  # only 2 of the 8 images carry a box
    assert units is not None and len(units) == 8  # K = round(1.0 * 8) slots
    assert set(units) == queue, units  # the 8 slots wrap over exactly those 2 units
    assert any("REPEAT inside one epoch" in r.message for r in caplog.records), [
        r.message for r in caplog.records]


def test_the_schedule_is_inert_without_slicing(tmp_path):
    """Slicing off -> no base segment -> no schedule, and no crash on the way there."""
    ds = _build(tmp_path / "noslice", slice_prob=False, slice_target_tiles=True)
    ds.set_epoch(0, EPOCHS)
    assert ds._slice_on() is False
    assert ds._tile_units is None
    assert ds._segment_lengths()[0] == 0
