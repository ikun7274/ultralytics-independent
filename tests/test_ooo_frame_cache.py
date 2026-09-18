"""Contract tests for the degraded-frame cache (the "F5" fix).

``_cap_frame_cached`` memoises what the working-resolution cap PRODUCED. That is a pure work
reduction, so the only things that can go wrong are (a) it changes WHAT the pool emits, or (b) it
hands out a buffer somebody else still owns. Every test here pins one of those two, never the
speed-up itself -- a timing assertion would be flaky on any machine:

  * the frame handed to a caller is OWNED by that caller, so it can never alias a cache entry; the
    cache is long-lived (one entry feeds dozens of later slots), so corruption would STICK;
  * writing through such a frame must therefore leave the cache byte-for-byte intact;
  * eviction is FIFO by FIRST FILL, not LRU, and the key separates both ``cap`` and ``kernel``;
  * the source frame is still read unconditionally, so the DECODE PATTERN -- and with it the mosaic
    mix pool -- is bit-identical with the cache on and off;
  * a read the cap did NOT fire on is never cached (that value is a raw-LRU frame; caching it would
    pin it and put a buffer the LRU still owns behind a second owner);
  * enabling the cache does not change one byte of any pool sample.

Run: python -m pytest tests/test_ooo_frame_cache.py -q
"""
from __future__ import annotations

import hashlib
import random
from collections import deque

import cv2
import numpy as np
import pytest

from ultralytics_ooo.core import _cap_long_side
from ultralytics_ooo.pool.constants import _legacy_ims_cap

from .test_ooo_branches import ALL_ON, _build

# The shared fixture writes 64x48 images at imgsz=64, where the AUTO caps (2*imgsz=128 for the
# degradation branches, 2*imgsz=128 -> half=64 for compose) can never fire -- and the cache, by
# design, only ever stores a frame a cap actually produced. Explicit small caps are what make the
# cache populate at all; giant fixtures would only slow the suite down.
FIRING = {**ALL_ON, "slice_prob": 0.0, "degrade_max_side": 32, "compose_max_side": 32}
LIN = cv2.INTER_LINEAR
AREA = cv2.INTER_AREA


def _new(root, **extra):
    ds = _build(root, **{**FIRING, **extra})
    ds.set_epoch(0, 10)
    return ds


def _frame(h: int = 64, w: int = 48) -> np.ndarray:
    """Deterministic, high-contrast frame -- so a wrong resample cannot look right by accident."""
    yy, xx = np.indices((h, w))
    base = ((yy * 7 + xx * 13) % 61).astype(np.uint8)
    return np.dstack([base, (base * 3) % 251, (255 - base)]).astype(np.uint8)


def _reset_runtime_state(ds) -> None:
    """Restore what a freshly built + ``set_epoch``'d dataset looks like.

    Needed by the A/B tests: without it the second pass starts with a warm raw LRU, so fewer reads
    decode, the mosaic buffer (fed by decode events) diverges, and the comparison would be measuring
    the reset instead of the cache. A SECOND DATASET is not an option either -- ``_mask_seed`` is
    derived from the image-file paths, so two roots pick different per-epoch selections.
    """
    ds._raw_cache.clear()
    ds._raw_cache_bytes = 0
    ds._raw_hits = ds._raw_misses = 0
    ds._cap_cache.clear()
    ds._cap_cache_bytes = 0
    ds._cap_hits = ds._cap_misses = 0
    if isinstance(ds.buffer, deque):
        ds.buffer.clear()
    for attr in ("ims", "im_hw0", "im_hw"):
        seq = getattr(ds, attr, None)
        if isinstance(seq, list):
            setattr(ds, attr, [None] * len(seq))
    ds._ims_keys.clear()


def _run_pool(ds) -> None:
    for i in range(len(ds)):
        ds.get_image_and_label(i)


def _digest(ds, seed: int = 20260918) -> str:
    """Hash every pool sample, so the two passes can be compared byte-for-byte.

    Seeds are reset first because the branch builders draw random parameters per call (blur angle,
    noise sigma, rain lines); with the same seeds and the same starting state the cache is the ONLY
    difference between the two passes -- which is the claim under test.
    """
    h = hashlib.sha256()
    random.seed(seed)
    np.random.seed(seed)
    for i in range(len(ds)):
        lab = ds.get_image_and_label(i)
        h.update(np.ascontiguousarray(lab["img"]).tobytes())
        h.update(np.asarray(lab["ori_shape"]).tobytes())
        h.update(np.asarray(lab["resized_shape"]).tobytes())
        h.update(np.asarray(lab.get("bboxes", np.empty((0, 4))), dtype=np.float64).tobytes())
        h.update(np.asarray(lab.get("cls", np.empty((0, 1))), dtype=np.float64).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# 1. Ownership: the returned frame belongs to the caller, never to the cache.
# ---------------------------------------------------------------------------------------------
def test_a_frame_handed_to_a_caller_is_never_the_cache_entry(tmp_path, monkeypatch):
    """Defence line 1: what ``_cap_frame_cached`` returns is owned by the caller.

    ``_finalize_label`` hands its input straight on whenever ``r == 1``, and Mosaic / affine then
    write ``label["img"]`` IN PLACE -- so a cache entry reaching that path would be corrupted with no
    error anywhere, and (unlike the raw LRU) the damage would stick, because one entry serves dozens
    of later slots. Hence the cache returns a copy unless the caller declares the frame read-only.

    Both a HIT and a MISS are covered: on a miss the frame is fresh, but the cache has just taken a
    reference to it, so it stops being exclusive the moment it is stored.
    """
    ds = _new(tmp_path / "cap_own")
    assert ds._cap_cache_size > 0 and ds._degrade_max_side() == 32, (ds._cap_cache_size, ds._degrade_max_side())

    handed: list[tuple[np.ndarray, bool]] = []
    orig = ds._cap_frame_cached

    def _spy(img_index, im, cap, interp, *, copy=True):
        out, scale = orig(img_index, im, cap, interp, copy=copy)
        handed.append((out, copy))
        return out, scale

    monkeypatch.setattr(ds, "_cap_frame_cached", _spy)
    _run_pool(ds)

    assert handed, "the cap path never ran -- this test would be vacuous"
    assert ds._cap_hits > 0, "no cache HIT happened -- the hit path would be untested"
    assert ds._cap_misses > 0, "no cache MISS happened -- the miss path would be untested"
    frames = [f for f, _ in ds._cap_cache.values()]
    assert frames, "the cache is empty -- this test would be vacuous"

    owned = [o for o, cp in handed if cp]
    assert owned, "no copy=True call happened -- this test would be vacuous"
    leaked = [o for o in owned if any(o is f or np.shares_memory(o, f) for f in frames)]
    assert not leaked, (
        f"{len(leaked)}/{len(owned)} handed frames are (or view) a live cache entry -- the cache "
        f"stopped copying on hand-out, so the next in-place write into label['img'] corrupts it"
    )


def test_writing_into_a_handed_frame_leaves_the_cache_byte_for_byte_intact(tmp_path, monkeypatch):
    """Defence line 1, adversarial form: the caller legitimately OWNS the array, so give it a writer.

    Pass 1 primes the cache and snapshots every entry. Pass 2 poisons every ``copy=True`` result --
    exactly what the pipeline is allowed to do -- and the snapshot must still match. Deleting the
    copy on hand-out makes this fail, which is what keeps the guard non-vacuous.
    """
    ds = _new(tmp_path / "cap_write")
    _run_pool(ds)
    assert ds._cap_hits > 0, "no hits on the priming pass -- this test would be vacuous"
    before = {k: v[0].copy() for k, v in ds._cap_cache.items()}
    assert before, "the cache is empty -- this test would be vacuous"

    poisoned: list[int] = []
    orig = ds._cap_frame_cached

    def _poison(img_index, im, cap, interp, *, copy=True):
        out, scale = orig(img_index, im, cap, interp, copy=copy)
        if copy and out.size:
            out[:] = 0  # the caller owns it; a legitimate in-place write
            poisoned.append(1)
        return out, scale

    monkeypatch.setattr(ds, "_cap_frame_cached", _poison)
    _run_pool(ds)

    assert poisoned, "no frame was poisoned -- this test would be vacuous"
    compared = 0
    for key, ref in before.items():
        got = ds._cap_cache.get(key)
        if got is None:
            continue  # evicted between the passes: nothing to compare
        compared += 1
        assert np.array_equal(got[0], ref), (
            f"cache entry {key} changed after a caller wrote into a frame it was handed -- the "
            f"hand-out copy is gone and the cache is being written through"
        )
    assert compared > 0, "no key survived both passes -- this test would be vacuous"


# ---------------------------------------------------------------------------------------------
# 2. Eviction order and key space.
# ---------------------------------------------------------------------------------------------
def test_eviction_is_fifo_by_first_fill_not_lru(tmp_path):
    """A hit must NOT refresh an entry's position, or the resident set follows the ACCESS order.

    Same rule (and same reason) as the raw-image LRU: upstream's resident set is "the last N images
    that really decoded", so an access-ordered cache would change which frames stay resident, hence
    which reads decode, hence the mosaic window -- silently breaking the "no switches -> byte-for-byte
    upstream" contract.
    """
    ds = _new(tmp_path / "cap_fifo")
    ds._cap_cache_size = 2
    ds._cap_cache.clear()
    ds._cap_cache_bytes = 0
    f = _frame()

    a, b, c = [(i, 32.0, int(LIN)) for i in range(3)]
    ds._cap_frame_cached(0, f, 32, LIN)
    ds._cap_frame_cached(1, f, 32, LIN)
    assert list(ds._cap_cache) == [a, b]

    ds._cap_frame_cached(0, f, 32, LIN)  # a HIT: must not reorder
    assert list(ds._cap_cache) == [a, b], "a hit refreshed the entry -- this is LRU, not FIFO"

    ds._cap_frame_cached(2, f, 32, LIN)  # insert C: the OLDEST BY FIRST FILL (A) is the one to go
    assert list(ds._cap_cache) == [b, c], (
        f"FIFO evicts A and keeps [B, C]; LRU would keep [A, C] instead. Got {list(ds._cap_cache)}"
    )


def test_the_key_separates_cap_and_kernel(tmp_path):
    """Key is ``(img_index, cap, kernel)`` -- both extra fields are load-bearing in real runs.

    ``cap``: the degradation branches cap at ``2*imgsz`` while compose caps its sources at ``imgsz``,
    so two cap values genuinely coexist (measured: both appear in one epoch). ``interp``:
    ``degrade_resample="area"`` makes the branches ask for INTER_AREA while compose always asks for
    INTER_LINEAR -- a kernel-blind key would hand compose a frame resampled with the wrong filter.
    """
    ds = _new(tmp_path / "cap_key")
    ds._cap_cache_size = 8
    ds._cap_cache.clear()
    ds._cap_cache_bytes = 0
    f = _frame()

    ds._cap_frame_cached(0, f, 32, LIN)     # image 0, cap 32, linear
    ds._cap_frame_cached(0, f, 16, LIN)     # same image, different cap
    ds._cap_frame_cached(0, f, 32, AREA)    # same image and cap, different kernel
    ds._cap_frame_cached(1, f, 32, LIN)     # different image

    keys = list(ds._cap_cache)
    assert len(keys) == 4, f"the key collapsed distinct resamplings into one entry: {keys}"
    assert [k[0] for k in keys] == [0, 0, 0, 1] and [k[1] for k in keys] == [32.0, 16.0, 32.0, 32.0], keys
    assert [k[2] for k in keys] == [int(LIN), int(LIN), int(AREA), int(LIN)], keys
    # The two same-cap entries are different COMPUTATIONS, not duplicates: the applied scale differs.
    assert ds._cap_cache[keys[0]][1] == 0.5 and ds._cap_cache[keys[1]][1] == 0.25, [
        ds._cap_cache[k][1] for k in keys
    ]


# ---------------------------------------------------------------------------------------------
# 3. Scope: what is NOT cached, and what the capacity is derived from.
# ---------------------------------------------------------------------------------------------
def test_a_cap_that_did_not_fire_is_never_cached(tmp_path):
    """Only a frame the cap PRODUCED is stored.

    When the long side already fits, ``_cap_long_side`` returns its INPUT -- a raw-LRU frame. Storing
    it would do two harmful things at once: pin a buffer the LRU believes it may evict, and hand the
    pool a frame the raw LRU still owns. There is also nothing to save, since no resample happened.
    """
    ds = _build(tmp_path / "cap_nofire", **{**ALL_ON, "slice_prob": 0.0})
    ds.set_epoch(0, 10)
    assert ds._degrade_max_side() == 2 * 64, ds._degrade_max_side()  # auto cap, above the 64x48 source
    lens = ds._segment_lengths()
    assert sum(lens[2:]) > 0, f"no degradation slots -- this test would be vacuous: {lens}"

    _run_pool(ds)
    assert not ds._cap_cache, f"a non-firing read was cached: {list(ds._cap_cache)}"
    assert (ds._cap_hits, ds._cap_misses) == (0, 0), (ds._cap_hits, ds._cap_misses)


def test_the_cache_never_holds_a_raw_lru_frame(tmp_path):
    """The cache and the raw LRU must not share a single buffer.

    This is the structural half of the test above: it catches ANY route that stores a passthrough
    value, not just the obvious one.
    """
    ds = _new(tmp_path / "cap_raw")
    _run_pool(ds)
    raw = list(ds._raw_cache.values())
    frames = [f for f, _ in ds._cap_cache.values()]
    assert raw and frames, f"vacuous: {len(raw)} raw frames, {len(frames)} cached frames"
    for f in frames:
        assert not any(f is r or np.shares_memory(f, r) for r in raw), (
            "a cached frame is (or views) a raw-LRU buffer -- the cap cache may only ever store "
            "frames its own resample allocated"
        )


def test_a_negative_size_disables_the_cache_but_not_the_cap(tmp_path):
    """``< 0`` is the A/B switch, and it must be exactly that: only the memoisation goes away."""
    ds = _new(tmp_path / "cap_off", degrade_frame_cache_size=-1)
    assert ds._cap_cache_size == 0, ds._cap_cache_size
    _run_pool(ds)
    assert not ds._cap_cache and (ds._cap_hits, ds._cap_misses) == (0, 0), ds.cap_cache_stats()

    # ... and the resample itself still happens, identically to calling _cap_long_side directly.
    f = _frame()
    out, scale = ds._cap_frame_cached(0, f, 32, LIN)
    ref, ref_scale = _cap_long_side(f, 32, interp=LIN)
    assert scale == ref_scale == 0.5, (scale, ref_scale)
    assert np.array_equal(out, ref), "the disabled path is no longer a plain _cap_long_side call"
    assert out.shape == (32, 24, 3), out.shape


def test_auto_capacity_tracks_batch_not_dataset_size(tmp_path):
    """Auto capacity is ``2 * _legacy_ims_cap(ni, batch)``, and once ni >= batch*8 it is batch-only.

    That is the whole justification for a fixed-size cache: the reuse it exploits comes from the
    mosaic mix pool, whose window upstream already bounds at ``_legacy_ims_cap``. Measured, the FIFO
    hit-rate curve is flat in dataset size over W=16..128 (400 vs 2000 images) and only diverges once
    the whole work set fits -- i.e. the right knob is ``batch``, not the dataset.
    """
    ds = _new(tmp_path / "cap_auto")
    assert ds._cap_cache_size == 2 * _legacy_ims_cap(ds.ni, ds.batch_size), ds._cap_cache_size
    # the batch-only half of the claim, stated on the bound itself
    assert _legacy_ims_cap(64, 8) == _legacy_ims_cap(2000, 8) == 63, (
        _legacy_ims_cap(64, 8),
        _legacy_ims_cap(2000, 8),
    )


def test_the_byte_budget_bounds_residency_and_never_empties_the_cache(tmp_path):
    """The budget is enforced per insert and still keeps the frame it just produced.

    A cache that evicted what it was about to store would re-resample that image on every visit --
    strictly worse than holding one oversized frame. Same ``>= 1`` rule as the raw LRU.
    """
    ds = _new(tmp_path / "cap_budget")
    ds._cap_cache_size = 8
    ds._cap_cache.clear()
    ds._cap_cache_bytes = 0
    ds._cap_cache_budget = 1  # far smaller than a single capped frame
    f = _frame()
    for i in range(4):
        ds._cap_frame_cached(i, f, 32, LIN)
        assert len(ds._cap_cache) == 1, f"byte budget ignored: {len(ds._cap_cache)} frames resident"
        assert ds._cap_cache_bytes == next(iter(ds._cap_cache.values()))[0].nbytes
    # The budget is a SEPARATE key from the raw LRU's, so tuning one cannot move the other.
    assert ds._cap_cache_budget == 1, ds._cap_cache_budget
    ds.slice_raw_cache_mb = 999.0
    assert ds._cap_cache_mb != 999.0, "the frame cache is reading the raw LRU's byte budget key"


# ---------------------------------------------------------------------------------------------
# 4. The contracts the rest of the package depends on.
# ---------------------------------------------------------------------------------------------
def test_the_cache_does_not_change_the_decode_pattern_or_the_mosaic_buffer(tmp_path, monkeypatch):
    """The cache sits strictly AFTER the source read, so decode events -- and the mosaic pool fed by
    them -- must be identical with it on and off."""
    ds = _new(tmp_path / "cap_decode")

    def run() -> tuple[list[bool], list[int]]:
        flags: list[bool] = []
        orig = ds._load_image_cached_ex

        def _load(img_index, *, copy=True):
            im, decoded = orig(img_index, copy=copy)
            flags.append(bool(decoded))
            return im, decoded

        monkeypatch.setattr(ds, "_load_image_cached_ex", _load)
        try:
            _run_pool(ds)
        finally:
            monkeypatch.setattr(ds, "_load_image_cached_ex", orig)
        return flags, list(ds.buffer)

    on_flags, on_buffer = run()
    assert ds._cap_hits > 0, "the cache served no hits -- the comparison would not exercise it"
    assert any(on_flags), "nothing decoded -- this test would be vacuous"

    _reset_runtime_state(ds)
    ds._cap_cache_size = 0
    off_flags, off_buffer = run()
    assert ds._cap_hits == 0, "the cache was disabled but still served"

    assert on_flags == off_flags, (
        f"the decode pattern changed with the cache on ({sum(on_flags)} decodes vs {sum(off_flags)}) "
        f"-- the cache is no longer downstream of the read"
    )
    assert on_buffer == off_buffer, "the mosaic mix pool changed -- upstream equivalence is broken"


def test_enabling_the_cache_changes_no_pool_sample(tmp_path):
    """Byte-for-byte equivalence of the whole pool, cache on vs off, same dataset and same order."""
    ds = _new(tmp_path / "cap_equiv")

    on = _digest(ds)
    assert ds._cap_hits > 0, "no hits on the cached pass -- this test would be vacuous"
    hits_on = ds._cap_hits

    _reset_runtime_state(ds)
    ds._cap_cache_size = 0
    off = _digest(ds)
    assert ds._cap_hits == 0, "the cache was disabled but still served"

    assert on == off, (
        f"the pool output changed when the frame cache was enabled ({hits_on} hits): {on} vs {off} "
        f"-- the cache is not transparent"
    )


@pytest.mark.parametrize("imgsz,fires", [(16, True), (64, False)])
def test_transparency_through_the_auto_cap(tmp_path, imgsz, fires):
    """The same equivalence, reached through the AUTO-sized cap instead of an explicit one.

    imgsz=16 makes the auto caps 32 (degradation branches) and 32 -> half=16 (compose) against 64x48
    sources, so both cap VALUES fire and the auto-sized cache really carries the read. imgsz=64 is
    the control: both auto caps land at 128/64, neither fires, and the cache must stay untouched.
    """
    # ``_build`` hardcodes imgsz=64 in its own overrides, so drive the auto cap by setting the
    # attribute afterwards: every consumer (_finalize_label, _degrade_max_side, the compose max-side
    # resolution) reads ``self.imgsz`` PER CALL, so this is the same state a built-at-16 dataset has.
    ds = _build(tmp_path / f"cap_auto_{imgsz}", **{**ALL_ON, "slice_prob": 0.0})
    ds.imgsz = imgsz
    ds.set_epoch(0, 10)
    assert ds._cap_cache_size > 0

    on = _digest(ds, seed=7)
    calls = ds._cap_hits + ds._cap_misses
    _reset_runtime_state(ds)
    ds._cap_cache_size = 0
    off = _digest(ds, seed=7)

    if fires:
        assert calls > 0, "no cap fired -- the cached pass would be vacuous"
    else:
        assert calls == 0, f"a cap fired unexpectedly at imgsz={imgsz}: {calls} calls"
    assert on == off, f"cache is not transparent at imgsz={imgsz}: {on} vs {off} ({calls} cap calls)"
