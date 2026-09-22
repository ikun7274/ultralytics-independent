"""Contract tests for the raw-cache decode shadow.

The shadow exists so that ``slice_raw_cache_size`` becomes a pure speed knob. Before it, the raw
cache capacity was a SEMANTIC input: the mosaic mix pool is fed by decode events
(``_touch_buffer_for_decode``), so a bigger cache appended less often, ``Mosaic.get_indexes``
sampled different partners, and the training data changed. The whole point here is that the
sequence of indices handed to ``dataset.buffer`` must NOT move.

So the tests pin, in order:

  * the auto rule engages exactly when the feed would otherwise change, and is OFF at the shipped
    configuration (so the default path is untouched and costs nothing);
  * a negative size is the documented A/B control, an explicit size is pinned, and a DISABLED raw
    cache always forces the shadow off -- otherwise turning ``slice_raw_cache_size`` off would stop
    meaning "the legacy/upstream-equivalent path";
  * the headline: growing the pixel cache with the shadow on leaves the feed sequence byte-for-byte
    identical AND the pool samples identical;
  * ...and the same growth WITHOUT the shadow really does move the feed. Without this test the
    headline one would pass just as happily on a fixture where nothing changed at all;
  * the shadow still READS every source frame -- it suppresses feeds, never reads;
  * it holds ints, not pixels, is FIFO-by-first-fill rather than LRU, and honours a byte budget
    without ever evicting the entry it is about to store.

Run: python -m pytest tests/test_ooo_raw_shadow.py -q
"""
from __future__ import annotations

import hashlib
import random
from collections import deque

import numpy as np
import pytest

from ultralytics_ooo.pool.constants import _legacy_ims_cap, _online_default

from .test_ooo_branches import _write_dataset

SHIPPED_RAW = int(_online_default("slice_raw_cache_size"))

# Production-shaped, NOT ALL_ON. The difference matters: with slice_all_tiles and no mosaic
# partners the pool does an almost single sweep over the images, so 16 and 19 frames miss
# identically and no capacity change can be observed at all. Mosaic partners are what create the
# reuse -- they are drawn from a buffer as deep as _legacy_ims_cap, so a cache smaller than that
# buffer re-decodes them.
PROD_LIKE = {
    "mosaic": 1.0, "close_mosaic": 0,
    "slice_prob": True, "slice_ratio": 0.5, "slice_target_tiles": True, "slice_all_tiles": False,
    "compose_keep": True, "compose_ratio": 0.5,
    "ratio_pad_keep": True, "ratio_pad_ratio": 0.5,
    "blur_keep": True, "blur_ratio": 0.5,
    "weather_keep": True, "weather_ratio": 0.5,
    "occlusion_keep": True, "occlusion_ratio": 0.5,
    "workers": 0,
}


# ---------------------------------------------------------------------------------------------
# Fixtures. The shared 4-image fixture cannot exercise the auto rule at all: _legacy_ims_cap is
# min(ni, batch*8, 1000)-1 = 3 there, and the real cache is clamped by the SAME bound, so
# _raw_cache_size can never exceed min(16, 3). A dataset with >=18 images at batch>=3 is the
# smallest shape where the shipped 16 is reachable as a true baseline.
# ---------------------------------------------------------------------------------------------
def _build(root, n=4, batch=2, **extra):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics_ooo import install

    install()
    img_dir = _write_dataset(root, n=n)
    cfg = get_cfg(overrides=dict(task="detect", mode="train", imgsz=64, batch=batch, fraction=1.0,
                                 **{**PROD_LIKE, **extra}))
    data = {"path": str(root), "names": {0: "obj"}, "channels": 3, "nc": 1, "train": "images/train"}
    ds = build_yolo_dataset(cfg, str(img_dir), batch, data, mode="train")
    ds.set_epoch(0, 10)
    return ds


def _big(root, **extra):
    """n=32, batch=8 -> _legacy_ims_cap = 31, i.e. the shipped 16 is a REAL bound and the mosaic
    buffer (31 deep) is deeper than the shipped cache -- which is what produces the reuse."""
    return _build(root, n=32, batch=8, **extra)


def _cap(ds) -> int:
    return _legacy_ims_cap(ds.ni, ds.batch_size)


def _reset(ds) -> None:
    """Fresh runtime state -- caches, counters, mosaic buffer, ims bookkeeping AND the shadow.

    Without this the second arm starts warm, fewer reads decode, the feed diverges for a reason
    that has nothing to do with the change under test, and the comparison measures the reset.
    """
    for name in ("_raw_cache", "_cap_cache", "_shadow"):
        obj = getattr(ds, name, None)
        if obj is not None:
            obj.clear()
    for name in ("_raw_cache_bytes", "_raw_hits", "_raw_misses", "_cap_cache_bytes", "_cap_hits",
                 "_cap_misses", "_shadow_bytes", "_shadow_hits", "_shadow_misses"):
        if hasattr(ds, name):
            setattr(ds, name, 0)
    if isinstance(ds.buffer, deque):
        ds.buffer.clear()
    for attr in ("ims", "im_hw0", "im_hw"):
        seq = getattr(ds, attr, None)
        if isinstance(seq, list):
            setattr(ds, attr, [None] * len(seq))
    ds._ims_keys.clear()


def _arm(ds, raw_size: int, shadow_size: int, seed: int = 20260918, monkeypatch=None):
    """One pass over the pool; returns the feed sequence, the real decode count and a pool digest."""
    from ultralytics_ooo.pool import dataset as _dsmod

    decodes = {"n": 0}
    _real_imread = _dsmod.imread

    def _counting_imread(*a, **kw):
        decodes["n"] += 1
        return _real_imread(*a, **kw)

    feed: list[int] = []
    original = ds._touch_buffer

    def _recording_touch(index, *a, **kw):
        feed.append(int(index))
        return original(index, *a, **kw)

    _reset(ds)
    ds._raw_cache_size = raw_size
    ds._shadow_size = shadow_size
    ds._touch_buffer = _recording_touch
    _dsmod.imread = _counting_imread
    try:
        random.seed(seed)
        np.random.seed(seed)
        h = hashlib.sha256()
        for i in range(len(ds)):
            lab = ds.get_image_and_label(i)
            h.update(np.ascontiguousarray(lab["img"]).tobytes())
            h.update(np.asarray(lab["ori_shape"]).tobytes())
            h.update(np.asarray(lab["resized_shape"]).tobytes())
            h.update(np.asarray(lab.get("bboxes", np.empty((0, 4))), dtype=np.float64).tobytes())
            h.update(np.asarray(lab.get("cls", np.empty((0, 1))), dtype=np.float64).tobytes())
    finally:
        ds._touch_buffer = original
        _dsmod.imread = _real_imread
    return {"feed": feed, "decodes": decodes["n"], "digest": h.hexdigest(), "slots": len(ds)}


# ---------------------------------------------------------------------------------------------
# 1. The auto rule.
# ---------------------------------------------------------------------------------------------
def test_the_shipped_configuration_leaves_the_shadow_off(tmp_path):
    """Auto must not engage at the default, or every existing baseline would move on upgrade."""
    ds = _big(tmp_path / "shadow_auto_off")
    assert ds._raw_cache_size == min(SHIPPED_RAW, _legacy_ims_cap(ds.ni, ds.batch_size))
    assert ds._shadow_size == 0, "the shadow engaged at the shipped default -- the feed would move"
    assert len(ds._shadow) == 0


def test_auto_engages_once_the_cache_outgrows_the_shipped_default(tmp_path):
    """The threshold is "did the real cache get bigger than the baseline", not "did anyone set a flag"."""
    cap = _cap(_big(tmp_path / "shadow_auto_probe"))
    assert cap > SHIPPED_RAW, f"fixture too small to test the auto rule (cap={cap})"
    ds = _big(tmp_path / "shadow_auto_on", slice_raw_cache_size=cap)
    assert ds._raw_cache_size == cap
    assert ds._shadow_size == SHIPPED_RAW, (ds._shadow_size, SHIPPED_RAW)

    # ...and one frame less than the threshold must NOT engage it (the boundary is exact).
    ds2 = _big(tmp_path / "shadow_auto_boundary", slice_raw_cache_size=SHIPPED_RAW)
    assert ds2._raw_cache_size == SHIPPED_RAW
    assert ds2._shadow_size == 0


def test_an_explicit_negative_size_reproducibly_disables_the_shadow(tmp_path):
    """The pre-shadow behaviour, kept as the A/B control that the third test below depends on."""
    ds = _big(tmp_path / "shadow_neg", slice_raw_cache_size=31, slice_decode_shadow_size=-1)
    assert ds._raw_cache_size == _cap(ds) > SHIPPED_RAW
    assert ds._shadow_size == 0


def test_an_explicit_size_pins_the_shadow(tmp_path):
    """Pinning to _legacy_ims_cap is how you reproduce upstream's OWN resident set instead of the 16."""
    ds = _big(tmp_path / "shadow_pin", slice_raw_cache_size=31, slice_decode_shadow_size=5)
    assert ds._shadow_size == 5
    probe = _big(tmp_path / "shadow_pin_probe")
    ds2 = _big(tmp_path / "shadow_pin_upstream", slice_raw_cache_size=31,
               slice_decode_shadow_size=_cap(probe))
    assert ds2._shadow_size == _cap(probe) > SHIPPED_RAW


def test_a_disabled_raw_cache_always_forces_the_shadow_off(tmp_path):
    """``slice_raw_cache_size=0`` must keep meaning "every read is a decode" (the upstream path)."""
    ds = _big(tmp_path / "shadow_raw_off", slice_raw_cache_size=0, slice_decode_shadow_size=9)
    assert ds._raw_cache_size == 0
    assert ds._shadow_size == 0
    assert len(ds._shadow) == 0


# ---------------------------------------------------------------------------------------------
# 2. The headline: the feed does not move when the cache grows.
# ---------------------------------------------------------------------------------------------
def test_the_shadow_keeps_the_mosaic_feed_when_the_cache_grows(tmp_path):
    """Growing the pixel cache 16 -> 19 (a stand-in for 16 -> 63) must not move one feed event."""
    ds = _big(tmp_path / "shadow_feed")
    cap = _cap(ds)
    assert cap > SHIPPED_RAW
    big = cap

    base = _arm(ds, SHIPPED_RAW, 0)
    fixed = _arm(ds, big, SHIPPED_RAW)

    assert base["feed"], "the pool never fed the mosaic buffer -- the test would be vacuous"
    assert fixed["feed"] == base["feed"], (
        "the shadow did not preserve the mosaic feed: the training data would have moved")
    assert fixed["digest"] == base["digest"], "the pool output changed"
    assert fixed["decodes"] < base["decodes"], (
        f"no decoding was saved ({fixed['decodes']} vs {base['decodes']}) -- "
        "so there is nothing to accept and the arms may not differ at all")


def test_without_the_shadow_the_same_growth_really_does_move_the_feed(tmp_path):
    """Non-vacuity. If this passes, the previous test could be believed for the wrong reason.

    With the shadow off, the identical capacity change must change the feed -- otherwise the two
    arms above were not actually different (e.g. the enlargement never took effect), and their
    agreement would prove nothing at all.
    """
    ds = _big(tmp_path / "shadow_feed_ctrl")
    cap = _cap(ds)
    assert cap > SHIPPED_RAW
    big = cap

    base = _arm(ds, SHIPPED_RAW, 0)
    reverted = _arm(ds, big, 0)

    assert base["feed"] != reverted["feed"], (
        "raising the capacity did not move the feed even without the shadow -- the enlargement "
        "never took effect, so test_the_shadow_keeps_the_mosaic_feed_when_the_cache_grows is vacuous")
    idx = next(i for i, (x, y) in enumerate(zip(base["feed"], reverted["feed"])) if x != y)
    assert idx > 0 and len(reverted["feed"]) < len(base["feed"])


# ---------------------------------------------------------------------------------------------
# 3. What the shadow does and does not touch.
# ---------------------------------------------------------------------------------------------
def test_the_shadow_suppresses_feeds_but_never_reads(tmp_path):
    """A memoisation must not shortcut the read: the raw cache is still consulted every time.

    If the shadow ever skipped a read it would stop being a feed-only structure and would silently
    become a second, unaudited cache -- with its own capacity error to get wrong.
    """
    ds = _big(tmp_path / "shadow_reads")
    calls = {"n": 0}
    original = ds._load_image_cached_ex

    def _counting(img_index, **kw):
        calls["n"] += 1
        return original(img_index, **kw)

    raw_sizes, shadow_sizes = (SHIPPED_RAW, 0), (_cap(ds), SHIPPED_RAW)
    observed = {}
    for tag, (raw, shadow) in (("base", raw_sizes), ("fixed", shadow_sizes)):
        _reset(ds)
        ds._raw_cache_size, ds._shadow_size = raw, shadow
        ds._load_image_cached_ex = _counting
        calls["n"] = 0
        try:
            random.seed(20260918)
            np.random.seed(20260918)
            for i in range(len(ds)):
                ds.get_image_and_label(i)
        finally:
            ds._load_image_cached_ex = original
        observed[tag] = calls["n"]

    assert observed["base"] > 0
    assert observed["fixed"] == observed["base"], (
        f"the shadow changed how often the source was read ({observed}) -- it must only gate feeds")


def test_the_flag_handed_to_the_feed_decouples_from_the_real_decode(tmp_path):
    """Directly pins the decoupling: same read, two different answers to two different questions."""
    ds = _big(tmp_path / "shadow_flag")
    ds._shadow_size = 3

    first = ds._decode_flag_for_feed(0, 4096, False)   # baseline would have decoded; we did not
    second = ds._decode_flag_for_feed(0, 4096, False)  # baseline would have hit too; we did not
    third = ds._decode_flag_for_feed(7, 4096, False)

    assert first is True, "the shadow did not report the baseline's first read as a decode"
    assert second is False, "the shadow kept reporting a resident key as a decode"
    assert third is True

    ds._shadow_size = 0
    assert ds._decode_flag_for_feed(0, 4096, False) is False, "the shadow was off but overrode the flag"
    assert ds._decode_flag_for_feed(0, 4096, True) is True


def test_the_shadow_holds_no_pixels_and_is_fifo_not_lru(tmp_path):
    """It must stay an int-keyed FIFO: LRU reordering is what breaks the resident-set agreement."""
    ds = _big(tmp_path / "shadow_fifo")
    ds._shadow_size = 3
    for i in range(3):
        assert ds._decode_flag_for_feed(i, 1024, True) is True
    assert list(ds._shadow) == [0, 1, 2]
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in ds._shadow.values())

    # Re-reading a resident key is a HIT and must NOT promote it: 0 stays the oldest.
    assert ds._decode_flag_for_feed(0, 1024, True) is False
    assert list(ds._shadow) == [0, 1, 2], "a hit reordered the shadow -- that is LRU, not FIFO"

    # So the next insert evicts 0, not 1.
    assert ds._decode_flag_for_feed(9, 1024, True) is True
    assert list(ds._shadow) == [1, 2, 9]
    assert ds._shadow_hits == 1 and ds._shadow_misses == 4


def test_the_shadow_byte_budget_bounds_residency_and_never_empties_it(tmp_path):
    ds = _big(tmp_path / "shadow_budget")
    ds._shadow_size = 100
    ds._shadow_budget = 3000  # room for exactly two 1024-byte frames

    for i in range(4):
        assert ds._decode_flag_for_feed(i, 1024, True) is True
    assert len(ds._shadow) == 2, f"the byte budget did not bind: {list(ds._shadow)}"
    assert list(ds._shadow) == [2, 3]

    # A single frame larger than the whole budget must still be kept -- evicting the entry under
    # construction would re-read it on the very next slot, which is strictly worse.
    ds._shadow_budget = 512
    assert ds._decode_flag_for_feed(100, 4096, True) is True
    assert list(ds._shadow)[-1] == 100


def test_the_shadow_does_not_change_the_decode_pattern_when_its_size_matches_the_cache(tmp_path):
    """With shadow size == cache size the two resident sets coincide, so nothing may differ.

    This is the property that makes the shipped default cheap: at the default the auto rule leaves
    the shadow off, and even if it were forced on with the baseline size the result is identical.
    """
    ds = _big(tmp_path / "shadow_equal")
    ds._shadow_budget = 0
    base = _arm(ds, SHIPPED_RAW, 0)
    forced = _arm(ds, SHIPPED_RAW, SHIPPED_RAW)
    assert forced["feed"] == base["feed"]
    assert forced["digest"] == base["digest"]
    assert forced["decodes"] == base["decodes"]
