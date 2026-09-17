"""Integration: the mixed pool's SEVEN segments, built end-to-end from a real (tiny) dataset.

This is the layer the other test files never touched: the six ``_build_*_sample`` methods that produce
the blur / weather / occlusion / ratio / compose / origin samples. They share one skeleton
(``_begin_branch_label`` -> image -> ``_save_branch`` -> ``_finish_branch``) and a de-duplication
refactor of that skeleton is exactly the kind of change that silently alters the label dict, the
resize semantics or the save order -- none of which the unit tests would notice.

It builds the dataset through the real ``build_yolo_dataset`` factory (so the installed
``InstalledYOLODataset`` + the online-aware ``v8_transforms`` assembly are exercised too) on a
synthetic 4-image set written to a temp dir: no downloads, no weights.

Run: python -m pytest tests/test_ooo_branches.py -q
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

IMG_W, IMG_H = 64, 48  # non-square, so slicing geometry and ratio padding both do real work


def _write_dataset(root, n=4, size=(IMG_W, IMG_H)):
    """Write ``n`` tiny images + YOLO label files under ``root`` (images/train, labels/train).

    ``size`` is only overridden by the byte-budget test, which needs frames large enough to hit an
    integer-MiB budget: the raw LRU stores ORIGINAL-resolution frames.
    """
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    h, w = size[1], size[0]
    for i in range(n):
        im = (rng.random((h, w, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(img_dir / f"img{i}.jpg"), im)
        # two boxes: one central, one in the lower-right quadrant
        (lbl_dir / f"img{i}.txt").write_text("0 0.50 0.50 0.20 0.20\n0 0.80 0.80 0.15 0.15\n", encoding="utf-8")
    return img_dir


def _build(root, size=(IMG_W, IMG_H), **overrides):
    """Build the installed dataset through the stock factory with the online branches configured."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics_ooo import install

    install()  # idempotent: registers the project keys on DEFAULT_CFG
    img_dir = _write_dataset(root, size=size)
    cfg = get_cfg(overrides=dict(task="detect", mode="train", imgsz=64, batch=2, fraction=1.0, **overrides))
    data = {"path": str(root), "names": {0: "obj"}, "channels": 3, "nc": 1, "train": "images/train"}
    return build_yolo_dataset(cfg, str(img_dir), 2, data, mode="train")


ALL_ON = {
    "slice_prob": 1.0,
    "slice_all_tiles": True,
    "img_origin": True,
    "ratio_pad_keep": True,
    "blur_keep": True,
    "compose_keep": True,
    "weather_keep": True,
    "occlusion_keep": True,
    # ratios of 1.0 -> mask None -> every sample really goes through its transform (not the fallback)
    "slice_ratio": 1.0,
    "ratio_pad_ratio": 1.0,
    "blur_ratio": 1.0,
    "compose_ratio": 1.0,
    "weather_ratio": 1.0,
    "occlusion_ratio": 1.0,
    "workers": 0,
}

# segment layout for N=4 with every branch on (see _segment_lengths / train.py's own comment)
SEG = {"base": 0, "origin": 16, "ratio": 20, "blur": 24, "compose": 32, "weather": 33, "occlusion": 37}
TOTAL = 41
# _segment_lengths() order, for tests that index a segment count by name
SEG_ORDER = ["base", "origin", "ratio", "blur", "compose", "weather", "occlusion"]


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("ddd_branches"), **ALL_ON)


def _assert_common(lab, where):
    assert isinstance(lab, dict), where
    assert "img" in lab and isinstance(lab["img"], np.ndarray), where
    assert lab["img"].ndim == 3 and lab["img"].size > 0, f"{where}: bad image {lab['img'].shape}"
    for k in ("ori_shape", "resized_shape", "ratio_pad", "cls", "instances", "im_file"):
        assert k in lab, f"{where}: missing {k!r}"
    assert tuple(lab["resized_shape"]) == tuple(lab["img"].shape[:2]), where


def test_pool_length_matches_the_segment_layout(ds):
    """N=4 all-branches pool = 16 + 4 + 4 + 8 + 1 + 4 + 4 = 41 (the layout train.py documents)."""
    assert ds.ni == 4
    assert ds._segment_lengths() == [16, 4, 4, 8, 1, 4, 4]
    assert len(ds) == TOTAL


def test_every_segment_produces_a_usable_label(ds):
    cases = {
        "slice": SEG["base"],
        "origin": SEG["origin"],
        "ratio": SEG["ratio"],
        "blur-short": SEG["blur"],
        "blur-long": SEG["blur"] + 1,
        "compose": SEG["compose"],
        "weather": SEG["weather"],
        "occlusion": SEG["occlusion"],
    }
    for name, idx in cases.items():
        _assert_common(ds.get_image_and_label(idx), name)


def test_origin_segment_is_the_uncut_image(ds):
    lab = ds.get_image_and_label(SEG["origin"])
    assert tuple(lab["ori_shape"]) == (IMG_H, IMG_W)


def test_slice_segment_is_a_tile_not_the_whole_image(ds):
    """slice_all_tiles + overlap: a tile is bigger than half the image but its AREA must exceed the
    non-overlapping quadrant, proving slicing ran rather than falling back to the original."""
    areas = {tuple(ds.get_image_and_label(SEG["base"] + k)["ori_shape"]) for k in range(4)}
    assert all(h <= IMG_H and w <= IMG_W for h, w in areas), areas
    quadrant = IMG_H * IMG_W / 4
    assert all(h * w > quadrant for h, w in areas), f"tiles look un-sliced: {areas}"


def test_ratio_segment_is_padded_to_the_target_ratio(ds):
    lab = ds.get_image_and_label(SEG["ratio"])
    h, w = lab["ori_shape"]
    assert (h, w) != (IMG_H, IMG_W), "ratio branch returned the original, no padding applied"
    # 64x48 is exactly 4:3 -> auto targets 16:9 by widening, so height is preserved
    assert h == IMG_H and w > IMG_W
    assert abs(w / h - 16 / 9) < 0.05, (h, w)


def test_compose_segment_is_a_2x2_canvas(ds):
    lab = ds.get_image_and_label(SEG["compose"])
    h, w = lab["ori_shape"]
    # four sources are unified to the group max, then stitched 2x2: both sides roughly double
    assert h >= 2 * IMG_H * 0.9 and w >= 2 * IMG_W * 0.9, (h, w)


def test_blur_weather_occlusion_keep_the_original_geometry(ds):
    """Blur / weather / occlusion never resize or crop, so ori_shape stays the working frame."""
    for name, idx in (("blur", SEG["blur"]), ("weather", SEG["weather"]), ("occlusion", SEG["occlusion"])):
        lab = ds.get_image_and_label(idx)
        assert tuple(lab["ori_shape"]) == (IMG_H, IMG_W), f"{name} changed geometry: {lab['ori_shape']}"


def _src_of(ds, index):
    """The source image FILE a pool index resolves to (implementation-agnostic probe)."""
    from pathlib import Path

    return Path(ds.get_image_and_label(index)["im_file"]).name


def _legacy_seg_img(ds, index):
    """The pre-refactor index -> (segment, image index) mapping, re-derived from the layout.

    This is the mapping the boolean-mask version implemented (``index // n_per`` for the base segment,
    ``(index - base) // 2`` for the two-tier blur segment, ``index - base`` for the rest). It is
    reproduced here from the ORIGINAL formulas rather than from the current code so that the ratio=1.0
    test below is a genuine compatibility check and not a tautology.
    """
    n_per = ds._n_per()
    sb = ds._segment_bases()
    if index >= sb.occlusion:
        return "occlusion", index - sb.occlusion
    if index >= sb.weather:
        return "weather", index - sb.weather
    if index >= sb.compose:
        return "compose", index - sb.compose
    if index >= sb.blur:
        return "blur", (index - sb.blur) // 2  # even slot -> short tier, odd slot -> long tier
    if index >= sb.ratio:
        return "ratio", index - sb.ratio
    if index >= sb.origin:
        return "origin", index - sb.origin
    return "base", index // n_per


def test_ratio_1_layout_is_bit_identical_to_the_legacy_shape(tmp_path_factory):
    """COMPATIBILITY INVARIANT: with every ratio >= 1 the pool IS the pre-refactor pool.

    ratio-sized segments degenerate to the old full-width ones at ``K == count``, so this is the
    property that lets the change ship incrementally (and that keeps the legacy arm of an A/B
    comparable). Two halves, both checked against the ORIGINAL formulas (``_legacy_seg_img``) rather
    than against the current implementation -- a tautology here would be worthless:

    1. every one of the pool's indices resolves to the same source image the legacy mapping named;
    2. the base segment still routes slot ``i*n_per + k`` to tile ``k`` of image ``i``: for each ``k``
       the outcome is identical across all images, and the four ``k`` values are still mutually
       distinguishable (a collapsed or shifted tile mapping would merge them).
    """
    root = tmp_path_factory.mktemp("ddd_compat")
    # occlusion_max_cover > 1 disables the label-dropping path, whose outcome depends on where the
    # randomly placed occluder landed -- that would make this test flap for reasons unrelated to layout.
    ds = _build(root, **{**ALL_ON, "occlusion_max_cover": 2.0})

    assert ds._segment_lengths() == [16, 4, 4, 8, 1, 4, 4], ds._segment_lengths()
    assert len(ds) == TOTAL
    # every ratio >= 1 must leave the "all slots" sentinel in place (no draw, no list)
    for attr in ("slice", "ratio", "blur", "compose", "weather", "occlusion"):
        assert getattr(ds, f"_sel_{attr}") is None, f"{attr}: ratio 1.0 must mean 'all slots'"

    from tools.ooo_layout_snapshot import signature

    for index in range(len(ds)):
        seg, img = _legacy_seg_img(ds, index)
        legacy_src = ds.im_files[(img * 4) % len(ds.labels)] if seg == "compose" else ds.im_files[img]
        lab = ds.get_image_and_label(index)
        assert Path(lab["im_file"]).name == Path(legacy_src).name, f"index {index} ({seg}) -> wrong image"
        if seg == "origin":
            assert tuple(lab["ori_shape"]) == (IMG_H, IMG_W), f"index {index}: origin is not the whole frame"

    n_per = ds._n_per()
    assert n_per == 4
    per_tile: dict[int, set] = {k: set() for k in range(n_per)}
    for i in range(ds.ni):
        for k in range(n_per):
            sig = signature(ds.get_image_and_label(i * n_per + k))
            per_tile[k].add((sig["n_boxes"], sig["labels"]))
    assert all(len(v) == 1 for v in per_tile.values()), f"tile content varies within one tile index: {per_tile}"
    assert len({next(iter(v)) for v in per_tile.values()}) > 1, (
        f"the four tile slots became indistinguishable -- the tile mapping collapsed: {per_tile}"
    )


def test_ratio_below_1_sizes_the_segment_and_uses_img_origin_for_coverage(tmp_path_factory):
    """A ratio < 1 SHRINKS the base segment instead of padding it with duplicate originals.

    The base segment is now ``n_per * K_slice`` only -- the un-selected images own NO base slot. Their
    coverage comes from the unified IMG_ORIGIN segment (one whole frame per ORIGINAL image). So the pool
    is smaller AND every image still appears: the selected ones as their tiles, every image as its origin
    frame. Nothing falls back to a plain original inside the base segment.
    """
    root = tmp_path_factory.mktemp("ddd_sized")
    ds = _build(root, **{**ALL_ON, "ratio_pad_ratio": 0.5, "blur_ratio": 0.5, "compose_ratio": 0.9,
                         "weather_ratio": 0.5, "occlusion_ratio": 0.5, "slice_ratio": 0.5})
    ds.set_epoch(0, 10)

    # 4 x K_slice(2) = 8 tiles | origin N = 4 (img_origin) | ratio 2 | blur 2x2 | compose 1 | weather 2 | occ 2
    assert ds._segment_lengths() == [8, 4, 2, 4, 1, 2, 2], ds._segment_lengths()
    assert len(ds) == 23
    assert len(ds._sel_slice) == 2 and len(ds._sel_ratio) == 2 and len(ds._sel_blur) == 2
    assert len(ds._sel_compose) == 1

    # the ratio segment holds exactly the selected images, in selection order (offset -> selection)
    ratio_base = ds._segment_bases().ratio
    assert [Path(ds.get_image_and_label(ratio_base + j)["im_file"]).name for j in range(2)] == [
        Path(ds.im_files[i]).name for i in ds._sel_ratio
    ]

    # base holds ONLY the selected images, each as 4 tiles (no plain tail): histogram [4, 4], 2 distinct
    base_srcs = [Path(ds.get_image_and_label(i)["im_file"]).name for i in range(ds._segment_bases().origin)]
    counts = {s: base_srcs.count(s) for s in set(base_srcs)}
    assert sorted(counts.values()) == [4, 4], counts
    assert len(base_srcs) == 8, "base must be exactly n_per * K_slice, no fallback padding"
    assert len(counts) == 2, "base must contain only the selected images"

    # the unified img_origin segment covers EVERY original once (offset == image index)
    origin_base = ds._segment_bases().origin
    assert [Path(ds.get_image_and_label(origin_base + i)["im_file"]).name for i in range(4)] == [
        Path(f).name for f in ds.im_files
    ]
    for i in range(4):
        assert tuple(ds.get_image_and_label(origin_base + i)["ori_shape"]) == (IMG_H, IMG_W)


def test_compose_ratio_uses_banker_rounding_on_the_group_count(tmp_path_factory):
    """compose selects ``round(ratio * ceil(N/4))`` GROUPS, so a low ratio zeroes it first.

    ``round(0.5) == 0`` in Python (round-half-to-even), and compose has the smallest base count of all
    branches (1 group for N=4), so compose_ratio=0.5 -- which reads like "half the groups" -- actually
    disables the branch. Pinning the behaviour here because it is surprising and easy to misconfigure.
    """
    root = tmp_path_factory.mktemp("ddd_compose_round")
    ds = _build(root, **{**ALL_ON, "compose_ratio": 0.5})
    assert ds._segment_lengths()[4] == 0, "round(0.5 x 1) == 0 -> no compose slots"
    ds.set_epoch(0, 10)
    assert ds._sel_compose == []

    ds = _build(root, **{**ALL_ON, "compose_ratio": 0.75})
    assert ds._segment_lengths()[4] == 1, "round(0.75 x 1) == 1"


def test_pool_length_is_constant_across_epochs(tmp_path_factory):
    """The layout must not move when a later epoch draws a different selection.

    This is what makes ratio-sized segments safe: the mosaic buffer's indices, ``nb``/``nw`` and the
    grouped sampler's units are all fixed at training start, so a length that varied per epoch would
    silently invalidate all three. Only the SET of selected images may change.
    """
    root = tmp_path_factory.mktemp("ddd_const")
    ds = _build(root, **{**ALL_ON, "ratio_pad_ratio": 0.5, "blur_ratio": 0.5, "slice_ratio": 0.5})

    lengths_before = ds._segment_lengths()
    total_before = len(ds)  # read BEFORE the first set_epoch: this is what the trainer/sampler see
    seen = set()
    for epoch in range(6):
        ds.set_epoch(epoch, 10)
        assert ds._segment_lengths() == lengths_before, f"epoch {epoch} moved the layout"
        assert len(ds) == total_before, f"epoch {epoch} moved the pool length"
        seen.add(tuple(ds._sel_slice))
        assert len(ds._sel_slice) == 2 and len(ds._sel_ratio) == 2
    assert len(seen) > 1, "the selection never varied -- the ratio would be inert"


def test_ratio_0_is_equivalent_to_slicing_disabled(tmp_path_factory):
    """K == 0 boundary: base collapses to 0 slots and IMG_ORIGIN carries every image once.

    With slicing selecting nothing, the base segment is empty; the only samples are the IMG_ORIGIN
    whole frames (one per original). So the pool is one plain original per image, all of them present.
    """
    root = tmp_path_factory.mktemp("ddd_zero")
    ds = _build(root, **{**ALL_ON, "slice_ratio": 0.0, "ratio_pad_ratio": 0.0, "blur_ratio": 0.0,
                         "compose_ratio": 0.0, "weather_ratio": 0.0, "occlusion_ratio": 0.0})
    ds.set_epoch(0, 10)
    # base 0; origin N=4 (img_origin)
    assert ds._segment_lengths() == [0, 4, 0, 0, 0, 0, 0], ds._segment_lengths()
    assert len(ds) == 4
    for i in range(4):
        assert _src_of(ds, i) == Path(ds.im_files[i]).name  # one whole-frame slot per image, no tile
        assert tuple(ds.get_image_and_label(i)["ori_shape"]) == (IMG_H, IMG_W)


def test_img_origin_adds_every_original(tmp_path_factory):
    """``img_origin`` puts ONE whole-frame slot per ORIGINAL image into the pool (unified coverage).

    Unlike the old ``slice_keep_origin`` (which only covered the sliced images), img_origin covers every
    original image exactly once -- selected or not. With it on, no image can be absent from the pool.
    """
    root = tmp_path_factory.mktemp("ddd_origin")
    ds = _build(root, **{**ALL_ON, "slice_ratio": 0.5})
    ds.set_epoch(0, 10)

    assert ds._segment_lengths()[1] == 4, ds._segment_lengths()  # origin == N (all originals)
    origin_base = ds._segment_bases().origin
    assert [Path(ds.get_image_and_label(origin_base + i)["im_file"]).name for i in range(4)] == [
        Path(f).name for f in ds.im_files
    ]
    for i in range(4):
        assert tuple(ds.get_image_and_label(origin_base + i)["ori_shape"]) == (IMG_H, IMG_W)

    # selected images appear as tiles in the base segment; EVERY image appears as a whole frame in origin
    selected = {Path(ds.im_files[i]).name for i in ds._sel_slice}
    base_srcs = [Path(ds.get_image_and_label(i)["im_file"]).name
                 for i in range(ds._n_per() * len(ds._sel_slice))]
    origin_srcs = [Path(ds.get_image_and_label(origin_base + i)["im_file"]).name for i in range(4)]
    assert set(base_srcs) == selected
    assert sorted(origin_srcs) == sorted(Path(f).name for f in ds.im_files)

    # ratio >= 1 -> every image selected -> origin is still N (the legacy width)
    ds2 = _build(tmp_path_factory.mktemp("ddd_origin1"), **ALL_ON)
    assert ds2._segment_lengths()[1] == 4


def test_img_origin_off_and_only_slicing_selected_discards_unselected(tmp_path_factory):
    """The "discard the un-selected" mode: img_origin OFF + only slicing selects -> unselected are gone.

    With every other augmentation branch OFF and img_origin OFF, the pool is ONLY the slicing tiles of the
    selected images. An image not selected for slicing owns no slot anywhere -- it is genuinely absent,
    which is exactly what a user asking to "discard the un-selected half" gets. The user's proposal pairs
    this discard mode with ``img_origin=True`` to put ALL originals back in one explicit segment; turning
    img_origin off is the "truly discard" half of that design.
    """
    root = tmp_path_factory.mktemp("ddd_discard")
    ds = _build(root, **{**ALL_ON, "slice_ratio": 0.5,
                         "ratio_pad_keep": False, "blur_keep": False, "compose_keep": False,
                         "weather_keep": False, "occlusion_keep": False, "img_origin": False})
    ds.set_epoch(0, 10)
    # base = 4*2 = 8 tiles; everything else (including origin) is 0
    assert ds._segment_lengths() == [8, 0, 0, 0, 0, 0, 0], ds._segment_lengths()
    assert len(ds) == 8
    selected = {Path(ds.im_files[i]).name for i in ds._sel_slice}
    seen = {Path(ds.get_image_and_label(i)["im_file"]).name for i in range(8)}
    assert seen == selected, "only the selected images may appear"
    unselected = {Path(f).name for f in ds.im_files} - selected
    assert unselected, "sanity: there must be un-selected images"
    assert not (unselected & seen), f"un-selected images must be absent: {unselected & seen}"


def test_segment_lengths_always_match_the_selection_lengths(tmp_path_factory):
    """STRUCTURAL GUARD: layout and selections are two views of one table and must never disagree.

    ``_segment_lengths`` sizes each segment from the CONFIGURED ratio (so the length is knowable before
    the first epoch), while the mapping walks ``_sel_*``. If they drift, indices fall off the end of a
    segment or samples are dropped -- silently, because both numbers look plausible on their own. This
    walks a grid of ratios and asserts the equality on both the ratio>=1 and the ratio<1 side.
    """
    root = tmp_path_factory.mktemp("ddd_tie")
    for ratio in (1.0, 0.75, 0.5, 0.25):
        ds = _build(root, **{**ALL_ON, "ratio_pad_ratio": ratio, "blur_ratio": ratio,
                             "weather_ratio": ratio, "occlusion_ratio": ratio,
                             "compose_ratio": ratio, "slice_ratio": ratio})
        # before any rebuild: layout correct already, and every ratio>=1 branch is the "all" sentinel
        lens = ds._segment_lengths()
        ds.set_epoch(0, 10)
        assert ds._segment_lengths() == lens, f"ratio {ratio}: set_epoch changed the layout"
        n = len(ds.labels)
        multi = {"ratio": 1, "blur": 2, "compose": 1, "weather": 1, "occlusion": 1}
        for attr, mult in multi.items():
            cnt = (n + 3) // 4 if attr == "compose" else n
            sel = ds._sel_indices(attr, cnt)
            assert lens[SEG_ORDER.index(attr)] == mult * len(sel), (
                f"ratio {ratio} {attr}: segment holds {lens[SEG_ORDER.index(attr)]} slots for "
                f"{len(sel)} selected items x {mult}"
            )
        n_per = ds._n_per()
        assert lens[0] == n_per * len(ds._sel_indices("slice", n))
        # and the whole index range decodes
        assert len(ds) == sum(lens)
        for index in range(len(ds)):
            _assert_common(ds.get_image_and_label(index), f"ratio {ratio} index {index}")


def test_every_mask_ratio_knob_actually_reaches_the_dataset(tmp_path_factory):
    """STRUCTURAL GUARD for the S3 regression.

    ``_rebuild_epoch_masks`` reads each branch's ratio through
    ``getattr(self, ratio_attr, _online_default(ratio_attr))``. If ``v8_transforms`` forgets to mirror a
    ratio onto the dataset, that read silently returns the table fallback (1.0), the mask becomes None
    and the branch fires on every image every epoch regardless of what the user configured. That had
    happened for ratio_pad_ratio / blur_ratio / compose_ratio.

    So: every ratio attr named in _mask_specs must exist on a dataset that was assembled by
    v8_transforms, and a sub-1.0 configuration must actually produce a mask.
    """
    root = tmp_path_factory.mktemp("ddd_ratios")
    ratios = {"slice_ratio": 0.5, "ratio_pad_ratio": 0.5, "blur_ratio": 0.5,
              "compose_ratio": 0.5, "weather_ratio": 0.5, "occlusion_ratio": 0.5}
    ds = _build(root, **{**ALL_ON, **ratios})

    specs = ds._mask_specs(len(ds.labels))
    assert {s[0] for s in specs} == {"slice", "ratio", "blur", "compose", "weather", "occlusion"}
    missing = [ratio_attr for _attr, ratio_attr, _on, _cnt in specs if not hasattr(ds, ratio_attr)]
    assert not missing, f"v8_transforms never mirrored these ratio knobs onto the dataset: {missing}"

    ds.set_epoch(0, 10)
    for attr, ratio_attr, on, count in specs:
        assert on, f"{attr}: branch should be on for this configuration"
        sel = getattr(ds, f"_sel_{attr}")
        assert sel is not None, f"{attr}: {ratio_attr}=0.5 must draw a selection, not keep the sentinel"
        assert len(sel) == round(0.5 * count), f"{attr}: wrong number of selected images"
        assert len(set(sel)) == len(sel), f"{attr}: duplicated selection"


def test_occlusion_max_cover_drops_fully_covered_targets(tmp_path_factory):
    """occlusion_max_cover=0 removes every target that ANY occluder touches (labels stay aligned)."""
    root = tmp_path_factory.mktemp("ddd_occl")
    ds = _build(root, **{**ALL_ON, "occlusion_max_cover": 0.0, "occlusion_blocks": 3})
    assert ds._sel_occlusion is None  # occlusion_ratio == 1.0 -> "all slots", every sample occluded
    lab = ds.get_image_and_label(SEG["occlusion"])
    assert lab["cls"].shape[0] == 0, "every target intersects an occluder, so all must be dropped"
    assert lab["instances"].bboxes.shape[0] == 0


def test_close_aug_epoch_sends_every_segment_back_to_the_original(tmp_path_factory):
    """close_aug_epoch: within the final N epochs every slot yields the original.

    The LAYOUT deliberately does not move (``ratio >= 1`` here keeps it at the legacy width; the
    ratio-sized case is covered by test_close_aug_epoch_keeps_the_ratio_sized_layout).
    """
    root = tmp_path_factory.mktemp("ddd_close")
    ds = _build(root, **{**ALL_ON, "close_aug_epoch": 1})
    ds.set_epoch(9, 10)  # last epoch of 10
    assert ds._closing is True
    for name, idx in (("ratio", SEG["ratio"]), ("blur", SEG["blur"]), ("weather", SEG["weather"])):
        assert tuple(ds.get_image_and_label(idx)["ori_shape"]) == (IMG_H, IMG_W), name
    assert len(ds) == TOTAL


def test_close_aug_epoch_keeps_the_ratio_sized_layout(tmp_path_factory):
    """close_aug_epoch must NOT move the layout: only the content changes, never the segment widths.

    Same reasoning as a mid-run selection change -- ``len(dataset)``, ``nb`` and the sampler's units are
    fixed at training start. So the closing epochs keep every slot in place and route each one back to
    the plain original of the image its slot selects.

    What the selection still DOES decide is which images the window can reach at all, so the draw is not
    short-circuited -- see test_close_aug_epoch_rotates_the_covered_subset.
    """
    root = tmp_path_factory.mktemp("ddd_close_sized")
    ds = _build(root, **{**ALL_ON, "close_aug_epoch": 2, "ratio_pad_ratio": 0.5, "blur_ratio": 0.5,
                         "slice_ratio": 0.5})
    ds.set_epoch(0, 10)
    sizes = ds._segment_lengths()
    normal = len(ds)
    assert ds._closing is False

    ds.set_epoch(8, 10)  # first closing epoch (10 - 2)
    assert ds._closing is True
    assert ds._segment_lengths() == sizes, "closing must not resize a segment"
    assert len(ds) == normal
    # The selection keeps its ratio-sized LENGTH but is a real draw, not the fixed prefix range(K).
    # This used to be asserted as ``_sel_slice == [0, 1]`` -- i.e. images 0 and 1 and nothing else,
    # every closing epoch, which is exactly the coverage bug the next test guards.
    n = len(ds.labels)
    for attr in ("slice", "ratio"):
        sel = ds._sel_indices(attr, n)
        assert len(sel) == 2, f"{attr}: closing must keep the ratio-sized length, got {sel}"
        assert len(set(sel)) == 2 and all(0 <= i < n for i in sel), f"{attr}: bad selection {sel}"
    for idx in range(len(ds)):
        lab = ds.get_image_and_label(idx)
        _assert_common(lab, f"closing index {idx}")
    # the ratio segment yields unpadded frames, the base segment un-tiled ones
    assert tuple(ds.get_image_and_label(ds._segment_bases().ratio)["ori_shape"]) == (IMG_H, IMG_W)
    assert tuple(ds.get_image_and_label(0)["ori_shape"]) == (IMG_H, IMG_W)


def test_close_aug_epoch_rotates_the_covered_subset(tmp_path_factory):
    """REGRESSION: the closing window must rotate WHICH images it covers, not freeze on a prefix.

    The selection inside the closing window used to be ``list(range(K))`` -- the first K images by file
    order. Because the images an epoch can see at all are the UNION of the per-branch selections, and
    every image-level branch took that same prefix, the closing window reached exactly the first
    ``max(K_x)`` images and nothing else, in EVERY closing epoch (``range(K)`` ignores the rng, so not
    even the seed could move it). Measured at train.py's ratios (all 0.5) on the 8-image mini set: 4 of
    8 images present with 7/6/6/6 slots each, the other 4 absent from the entire window.

    Two properties separate the fix from the bug, and both are asserted here because either alone could
    pass by luck on a given seed:
      * the selection is not the same list every closing epoch;
      * some closing epoch reaches an image OUTSIDE the first max(K).

    ``close_aug_epoch=6`` with ``epochs=6`` makes every epoch a closing epoch, so the whole window is
    observable. ``_perf_review/closing_probe.py`` reproduces the end-to-end version on the real mini set.
    """
    root = tmp_path_factory.mktemp("ddd_close_cov")
    ratios = {"slice_ratio": 0.5, "ratio_pad_ratio": 0.5, "blur_ratio": 0.5, "compose_ratio": 0.5,
              "weather_ratio": 0.5, "occlusion_ratio": 0.5}
    ds = _build(root, **{**ALL_ON, **ratios, "close_aug_epoch": 6, "epochs": 6})
    n = len(ds.labels)
    k = n // 2
    assert (n, k) == (4, 2), f"fixture changed shape: N={n}, K={k} -- the numbers below assume 4 and 2"
    layout = ds._segment_lengths()

    image_level = ("slice", "ratio", "blur", "weather", "occlusion")
    seen_selections: set[tuple[int, ...]] = set()
    unions: list[set[int]] = []
    for epoch in range(6):
        ds.set_epoch(epoch, 6)
        assert ds._closing is True, f"epoch {epoch} should be inside the window"
        assert ds._segment_lengths() == layout, f"epoch {epoch}: the fix must not move the layout"
        covered: set[int] = set()
        for attr in image_level:
            sel = ds._sel_indices(attr, n)
            assert len(sel) == k, f"epoch {epoch} {attr}: closing resized the selection to {sel}"
            assert len(set(sel)) == k, f"epoch {epoch} {attr}: duplicate indices in {sel}"
            assert all(0 <= i < n for i in sel), f"epoch {epoch} {attr}: out of range: {sel}"
            covered |= set(sel)
            seen_selections.add(tuple(sel))
        unions.append(covered)

    assert len(seen_selections) > 1, (
        f"every closing epoch drew the SAME selection {seen_selections} -- that is the fixed-prefix bug"
    )
    assert any(c - set(range(k)) for c in unions), (
        f"no closing epoch reached past the first {k} images; per-epoch unions: {unions}"
    )


def _capture_messages(fn):
    """Run ``fn`` and return the messages logged to the ultralytics logger."""
    import logging

    records = []

    class _H(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())

    logger = logging.getLogger("ultralytics")
    logger.addHandler(_H())
    try:
        fn()
    finally:
        logger.handlers.pop()
    return records


def test_mask_summary_reports_the_actual_per_branch_counts(tmp_path_factory):
    """Observability guard: the log must show REAL selected counts, not the configured ratios.

    The three mis-wired ratio knobs were invisible precisely because nothing reported how many slots a
    branch actually augmented. The summary is built from the masks themselves, so it would have shown
    `all/16 (ratio 0.1)` -- the ratio and the actual behaviour disagreeing in one line.
    """
    root = tmp_path_factory.mktemp("ddd_summary")
    ds = _build(root, **{**ALL_ON, "ratio_pad_ratio": 0.5, "blur_ratio": 0.5})
    messages = _capture_messages(lambda: ds.set_epoch(0, 10))
    summary = next((m for m in messages if "augment masks @" in m), None)
    assert summary, f"set_epoch must emit a mask summary: {messages}"

    # ratio 0.5 of N=4 originals -> 2 selected slots. Masks are ORIGINAL-level (one bit per image),
    # so blur reports 4 slots too, even though the blur SEGMENT holds 2 samples per image.
    assert "ratio 2/4 (ratio 0.5)" in summary, summary
    assert "blur 2/4 (ratio 0.5)" in summary, summary
    # untouched branches stay at "all", with their configured ratio printed next to the real count so a
    # mis-wired knob (behaviour 100%, config 10%) is visible in this single line
    assert "slice all/4 (ratio 1)" in summary, summary
    assert "weather all/4 (ratio 1)" in summary, summary
    assert "occlusion all/4 (ratio 1)" in summary, summary


def test_mask_summary_labels_ratio_1_as_all_augmented(tmp_path_factory):
    """ratio >= 1 -> mask None -> every slot augmented, which is the pre-fix behaviour."""
    root = tmp_path_factory.mktemp("ddd_summary_all")
    ds = _build(root, **{**ALL_ON, "ratio_pad_ratio": 1.0, "blur_ratio": 1.0})
    messages = _capture_messages(lambda: ds.set_epoch(0, 10))
    summary = next(m for m in messages if "augment masks @" in m)
    assert "ratio all/4 (ratio 1)" in summary, summary
    assert "blur all/4 (ratio 1)" in summary, summary  # original-level mask: one bit per image
    assert "compose all/1 (ratio 1)" in summary, summary  # compose is group-level: ceil(4/4) = 1


def test_online_augment_header_names_the_slice_segment_sahi(tmp_path_factory):
    """The construction header must call the slicing segment ``sahi``, not ``base``.

    It used to print ``base``, which reads as "baseline / un-augmented" -- the exact opposite of what
    that segment holds (the ``slice_transform`` tiles). The un-augmented whole frame is a separate
    segment (``origin``), so the label invited the reader to think the sliced content and the plain
    content were the same thing. Only the LABEL changed; the code keeps naming that segment ``base``
    internally (``_segment_bases().base``, ``_base_slot()``), and the slot counts are untouched.
    """
    root = tmp_path_factory.mktemp("ddd_header")
    messages = _capture_messages(lambda: _build(root, **ALL_ON))
    header = next((m for m in messages if "Online augment:" in m), None)
    assert header, f"the dataset constructor must report the pool layout: {messages}"
    # N=4, all branches on: base 4*4, origin 4, ratio 4, blur 2*4, compose ceil(4/4), weather 4, occl 4
    assert (
        "(segment slots: sahi 16, origin 4, ratio 4, blur 8, compose 1, weather 4, occlusion 4)" in header
    ), header
    assert " base " not in header, f"the slice segment must be labelled 'sahi', not 'base': {header}"


def test_ratio_rounding_to_zero_is_reported(tmp_path_factory):
    """A positive ratio that rounds to 0 selected images must warn instead of silently doing nothing."""
    root = tmp_path_factory.mktemp("ddd_round")
    # N=4 and ratio=0.1 -> round(0.4) == 0 selected slots
    ds = _build(root, **{**ALL_ON, "blur_ratio": 0.1, "compose_ratio": 0.1})
    messages = _capture_messages(lambda: ds.set_epoch(0, 10))

    assert ds._sel_blur == []
    warned = [m for m in messages if "selects round" in m]
    assert any("blur_ratio" in m and "0 slots" in m for m in warned), f"missing warning: {messages}"
    summary = next(m for m in messages if "augment masks @" in m)
    assert "<-- NONE" in summary, summary


def test_ratio_one_selects_everything(tmp_path_factory):
    """ratio 1.0 -> the "all slots" sentinel -> all augmented. This is the documented escape hatch for
    reproducing experiments that ran before the ratio knobs actually took effect."""
    root = tmp_path_factory.mktemp("ddd_ratio1")
    ds = _build(root, **{**ALL_ON, "ratio_pad_ratio": 1.0, "blur_ratio": 1.0, "compose_ratio": 1.0})
    ds.set_epoch(0, 10)
    for attr in ("ratio", "blur", "compose"):
        assert getattr(ds, f"_sel_{attr}") is None, f"{attr}: ratio 1.0 must mean 'all slots'"


def test_no_switches_means_no_expansion_at_all(tmp_path_factory):
    """The zero-intrusion contract: with every online switch off the pool IS the plain dataset.

    ``install()`` must be completely inert unless a project switch is set -- same length, same index ->
    image mapping, no extra segment, and (via ``_extended_pool_on``) the stock buffer bookkeeping. This
    is the property that lets the package sit on a pristine Ultralytics, so it is asserted explicitly
    rather than left implicit in "the defaults are off".
    """
    root = tmp_path_factory.mktemp("ddd_off")
    ds = _build(root)  # no overrides at all
    n = len(ds.labels)

    # No slicing pipeline and no augmentation branch is on, BUT img_origin defaults to True for a
    # normal (non-rect/non-obb) train dataset, so every original lands in the ORIGIN segment (width N)
    # and the BASE segment holds zero slicing tiles. The pool is still exactly the N plain originals --
    # total == n and _extended_pool_on() is False (stock buffer bookkeeping) -- so the zero-intrusion
    # contract holds behaviourally; only the internal segment that carries the originals moved from the
    # base plain tail (old behaviour) to the dedicated origin segment.
    assert ds._segment_lengths() == [0, n, 0, 0, 0, 0, 0], ds._segment_lengths()
    assert len(ds) == n
    assert ds._n_per() == 1
    assert ds._extended_pool_on() is False, "no segment extends the pool -> stock buffer bookkeeping"
    assert [Path(ds.get_image_and_label(i)["im_file"]).name for i in range(n)] == [
        Path(f).name for f in ds.im_files
    ]
    for i in range(n):
        _assert_common(ds.get_image_and_label(i), f"index {i}")


def test_get_image_and_label_covers_the_whole_index_range(ds):
    """Every decodable index must produce a label: a layout/length drift silently drops samples."""
    for idx in range(len(ds)):
        _assert_common(ds.get_image_and_label(idx), f"index {idx}")


def _collect_logs():
    """Attach a capturing handler to the ``ultralytics`` logger; returns (records, detach)."""
    import logging

    records = []

    class _H(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())

    logger = logging.getLogger("ultralytics")
    handler = _H()
    logger.addHandler(handler)
    return records, lambda: logger.removeHandler(handler)


def test_grouping_is_kept_whatever_the_cache_size(tmp_path):
    """A cache bigger than the busiest image's fan-out must NOT switch grouping off.

    It used to: the guard was ``if fan_out < slice_raw_cache_size: warn + return None`` and told the
    user to LOWER the cache. Its evidence (24-image pool: 80.0% -> 60.0% hit at cache 16 vs 4) came from
    a pool no bigger than the Mosaic window, where the entire dataset fits in the cache and capacity
    cannot matter, so it did not extrapolate. Measured at real scale (240 images, real DataLoader):
    cache 16 grouped 41.0 items/s vs 16 shuffled 34.4, cache 32 grouped 44.3 vs 37.7 -- grouping wins
    at the very cache size the guard refused it at. Units hold <= 4 images and are walked round-robin,
    so the LRU only has to hold those <= 4 to absorb every re-read, which the floor of 4 guarantees.
    """
    records, detach = _collect_logs()
    try:
        for cap in (4, 16, 64):
            ds = _build(tmp_path / f"lru_{cap}", **{**ALL_ON, "slice_raw_cache_size": cap})
            ds.set_epoch(0, 10)
            assert ds._raw_cache_size == cap, ds._raw_cache_size
            units = ds.grouped_sample_units()
            assert units is not None, f"grouping refused at slice_raw_cache_size={cap}"
            flat = sorted(i for unit in units for blk in unit for i in blk)
            assert flat == list(range(len(ds))), f"cache {cap}: the units are not a permutation"
    finally:
        detach()
    blamed = [m for m in records if "fan-out" in m or "exceeds the busiest" in m]
    assert not blamed, blamed


def test_grouping_is_refused_when_no_image_owns_two_slots(tmp_path):
    """The one condition that still disqualifies grouping: there is no re-read to absorb.

    ``slice_all_tiles=False`` + ``slice_ratio=0.5`` + no ``img_origin`` gives 2 base slots for 4 images,
    i.e. 1 slot per image. The pool IS wider than N (so the extended-pool gate passes) -- only the
    fan-out test can refuse it, which is what this pins.
    """
    ds = _build(
        tmp_path / "no_reuse",
        slice_prob=1.0,
        slice_all_tiles=False,
        slice_ratio=0.5,
        img_origin=False,
        workers=0,
    )
    ds.set_epoch(0, 10)
    assert ds._segment_lengths() == [2, 0, 0, 0, 0, 0, 0], ds._segment_lengths()
    assert ds.grouped_sample_units() is None


def test_grouping_survives_the_trainers_build_order(tmp_path):
    """REGRESSION (silent, whole-run): the trainer builds the loaders BEFORE it publishes an epoch.

    ``trainer._build_train_pipeline`` calls ``build_dataloader`` -- which asks the dataset for its units
    via ``GroupedImageSampler.from_dataset`` -- while the only caller of ``set_epoch``,
    ``on_train_epoch_start``, has not run yet. Every selection is therefore still the "all slots"
    sentinel, and ``_sel_compose`` is ``None`` even when compose is OFF (its default). The old fan-out
    rules read that sentinel unconditionally and emitted ``ceil(N/4)`` compose indices into a zero-wide
    compose segment, so the permutation check failed and the whole run fell back to RandomSampler --
    with a warning that blamed a "layout drift". Every test and tool called ``set_epoch`` first, so the
    permutation check was the only thing that ever saw it.
    """
    records, detach = _collect_logs()
    try:
        # The shape that failed: all enabled ratios >= 1 (so _has_partial_ratio() is False and nothing
        # ever corrects the sentinel), slicing on, compose off at its default ratio of 1.0.
        ds = _build(
            tmp_path / "trainer_order",
            slice_prob=1.0,
            slice_all_tiles=True,
            slice_ratio=1.0,
            img_origin=True,
            workers=0,
        )
        assert ds._mask_stamp == -1, "the build published an epoch -- this test would be vacuous"
        units = ds.grouped_sample_units()
        assert units is not None, f"grouping dropped in the trainer's build order: {records}"
        flat = sorted(i for unit in units for blk in unit for i in blk)
        assert flat == list(range(len(ds))), "the units are not a permutation of the pool"

        # ...and the same has to hold for the loader the trainer actually builds, without set_epoch.
        from ultralytics.data.build import build_dataloader
        from ultralytics_ooo.pool.sampler import GroupedImageSampler

        loader = build_dataloader(ds, 2, 0, shuffle=True, rank=-1, pin_memory=False, device="cpu")
        assert isinstance(loader.sampler, GroupedImageSampler), (
            f"the real build_dataloader did not get the grouped sampler: {type(loader.sampler).__name__}"
        )
    finally:
        detach()
    drifted = [m for m in records if "drifted" in m or "grouping rules" in m or "Falling back" in m]
    assert not drifted, drifted


def test_the_raw_cache_default_covers_the_reuse_window(tmp_path):
    """The default capacity must be sized for the window that actually repeats.

    An image's slots are spread over the whole pool, so the only thing that re-reads is Mosaic's window
    (``max_buffer_length`` pool slots) -- ~13-25 distinct images at the shipped batch sizes -- plus the
    grouped sampler's unit (<= 4 images). The old default of 4 was set on a 24-image pool where the
    window holds the whole dataset and capacity is irrelevant; measured at 240 images it costs +85%.
    """
    ds = _build(tmp_path / "lru_default", **ALL_ON)
    ds.set_epoch(0, 10)
    assert ds._raw_cache_size == 16, f"default capacity should cover the reuse window, got {ds._raw_cache_size}"
    assert ds._raw_cache_mb == 256, ds._raw_cache_mb
    assert ds._raw_cache_budget == 256 * (1 << 20)
    # explicit overrides still work, and (0, 4) is still floored to 4
    assert _build(tmp_path / "lru_floor", **{**ALL_ON, "slice_raw_cache_size": 1})._raw_cache_size == 4
    assert _build(tmp_path / "lru_off", **{**ALL_ON, "slice_raw_cache_size": 0})._raw_cache_size == 0


def test_the_raw_cache_byte_budget_caps_resident_frames(tmp_path):
    """``slice_raw_cache_mb`` caps the frame count from above, so a large-frame dataset cannot be
    sized into an OOM by ``slice_raw_cache_size`` alone.

    The LRU holds ORIGINAL-resolution frames, whose size is unknown until the first decode, so the
    budget is enforced on every insert. 640x640x3 frames are 1.17 MiB; a 3 MiB budget fits exactly two
    of them, whatever ``slice_raw_cache_size`` says.
    """
    ds = _build(
        tmp_path / "budget",
        size=(640, 640),
        slice_prob=1.0,
        slice_all_tiles=True,
        slice_ratio=1.0,
        img_origin=True,
        slice_raw_cache_size=16,
        slice_raw_cache_mb=3,
        workers=0,
    )
    ds.set_epoch(0, 10)
    assert ds._raw_cache_budget == 3 * (1 << 20)
    for i in range(len(ds)):
        ds.get_image_and_label(i)
    assert len(ds._raw_cache) == 2, f"3 MiB budget / 1.17 MiB frames -> 2 frames, got {len(ds._raw_cache)}"
    assert ds._raw_cache_bytes == sum(v.nbytes for v in ds._raw_cache.values()), "the byte tally drifted"
    assert ds._raw_cache_bytes <= ds._raw_cache_budget


def test_slice_branch_never_hands_the_lru_buffer_downstream(tmp_path, monkeypatch):
    """Contract guard for the ``copy=False`` fast path in the slice branch.

    ``_load_image_cached(copy=False)`` removes one full-resolution memcpy per slice read (~20 ms at
    4000x3000, and ~80% of slice reads are LRU hits, so that is most of them). It is sound ONLY while
    the LRU's array is never exposed to a writer -- and three OnlineSlice paths DO return the input
    unchanged: the ``p`` coin flip, the degenerate-size guard, and ``_emit``'s background-quota Plan A
    fallback. The dataset detects that by IDENTITY and copies. This forces the pass-through path and
    asserts the array that reaches the rest of the pipeline is never an LRU entry.
    """
    from ultralytics_ooo.pool.augment_setup import OnlineSlice

    ds = _build(
        tmp_path / "lru_alias",
        slice_prob=0.5,
        slice_all_tiles=True,
        img_origin=False,
        slice_ratio=1.0,
        slice_grouped_sampler=False,
        slice_raw_cache_size=8,
        workers=0,
    )
    ds.set_epoch(0, 10)
    assert ds._raw_cache_size == 8, ds._raw_cache_size
    assert isinstance(ds.slice_transform, OnlineSlice)
    assert len(ds) > 0 and ds._segment_lengths()[0] > 0, ds._segment_lengths()
    # p == 0 -> `random.uniform(0, 1) > 0.0` is (almost surely) always true -> the transform returns
    # its INPUT object untouched, i.e. exactly the branch that used to leak the LRU buffer.
    monkeypatch.setattr(ds.slice_transform, "p", 0.0)

    handed = []
    orig_finalize = ds._finalize_label

    def _spy(label, im):
        handed.append(im)
        return orig_finalize(label, im)

    monkeypatch.setattr(ds, "_finalize_label", _spy)
    for i in range(len(ds)):
        ds.get_image_and_label(i)

    assert handed, "the slice branch never ran -- this test would be vacuous"
    cached_ids = {id(v) for v in ds._raw_cache.values()}
    aliased = sum(1 for arr in handed if id(arr) in cached_ids)
    assert aliased == 0, f"{aliased}/{len(handed)} samples were handed the LRU's own buffer"
