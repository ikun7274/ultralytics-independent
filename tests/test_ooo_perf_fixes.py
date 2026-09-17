"""Contract tests for the ninth-round performance fixes.

Each of these locks a guarantee that the fix DEPENDS on, rather than the speed-up itself (a timing
assertion would be flaky on any machine):

  * the origin segment must read through the raw LRU -- that is the entire point of the fix, and the
    old code silently bypassed it, which is also why the bug survived;
  * the slice tile is a VIEW now, so the shared resize tail must still hand downstream an exclusive
    buffer (the identity guard alone cannot see a view);
  * ``prefetch_factor`` must actually reach the loader, and must default to the stock value so an
    unconfigured run is unchanged;
  * the weather-noise kernel must keep the requested sigma -- it now computes in int16, and an
    "optimisation" that quietly halves the noise would be invisible in every other test.

Run: python -m pytest tests/test_ooo_perf_fixes.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from .test_ooo_branches import _build


def _origin_base(ds) -> int:
    """First index of the origin segment (SegmentBases fields are CUMULATIVE STARTS)."""
    return ds._segment_bases().origin


def _lru_reads(ds) -> tuple[int, int]:
    """(hits, misses) across BOTH counters: the local pair is zeroed every 16 samples on publish."""
    h, m = ds.raw_cache_stats()
    return h + ds._raw_hits, m + ds._raw_misses


def test_origin_segment_reads_through_the_raw_lru(tmp_path):
    """The origin segment must be served by the raw LRU, not by a bare ``load_image``.

    With slicing OFF the old code went through ``load_image``, which memoised into ``self.ims`` and
    never touched the raw LRU; with slicing ON it memoised nowhere at all. Either way the LRU counter
    stayed flat, which is how "origin re-decodes on every visit" hid for so long. So the assertion is
    on the LRU counter -- and on ``self.ims`` staying empty, so the hit cannot be credited to the other
    cache.
    """
    ds = _build(tmp_path / "origin_lru", slice_prob=0.0, img_origin=True, slice_raw_cache_size=8,
                slice_grouped_sampler=False)
    ds.set_epoch(0, 10)
    base = _origin_base(ds)
    assert len(ds) > 0 and ds._raw_cache_size == 8

    first = ds.get_image_and_label(base)
    hits_a, misses_a = _lru_reads(ds)
    assert misses_a >= 1, "a cold first read must be a miss, otherwise the check is vacuous"

    second = ds.get_image_and_label(base)
    hits_b, _ = _lru_reads(ds)
    assert hits_b > hits_a, (
        "the second origin read did not hit the raw LRU -- origin is bypassing the cache again"
    )
    assert not [v for v in ds.ims if v is not None], (
        "the origin read was served by self.ims, i.e. it went through load_image"
    )
    # The two reads must agree: the LRU serves the same pixels the decode did.
    assert first["img"].shape == second["img"].shape
    assert np.array_equal(first["img"], second["img"])


def test_origin_segment_reports_the_same_geometry_as_load_image(tmp_path):
    """The LRU-backed read must not drift from ``load_image`` on ori_shape / resized_shape / ratio_pad."""
    ds = _build(tmp_path / "origin_geom", slice_prob=1.0, slice_all_tiles=True, slice_ratio=1.0,
                img_origin=True, slice_grouped_sampler=False)
    ds.set_epoch(0, 10)
    base = _origin_base(ds)
    for i in range(len(ds.labels)):
        got = ds.get_image_and_label(base + i)
        ref_img, ori_shape, resized_shape = ds.load_image(i)
        assert tuple(got["ori_shape"]) == tuple(ori_shape), i
        assert tuple(got["resized_shape"]) == tuple(resized_shape), i
        assert got["img"].shape == ref_img.shape, i
        assert np.array_equal(got["img"], ref_img), i


def test_the_slice_tile_is_a_view_but_downstream_never_gets_the_lru_buffer(tmp_path, monkeypatch):
    """The tile is now a view, so the exclusive-buffer guarantee must come from the resize tail.

    Two assertions, and both are needed:
      * at least one handed array SHARES memory with an LRU frame without BEING it -- that is what
        proves the full-resolution ``ascontiguousarray`` copy is gone (if it came back, every handed
        array would be a fresh allocation and this would fail);
      * the OUTPUT of ``_finalize_label`` never shares memory with an LRU frame -- that is the
        guarantee the affine/mosaic transforms rely on when they write in place. The identity guard in
        ``get_image_and_label`` cannot cover this case on its own, because a view is not the frame.
    """
    ds = _build(tmp_path / "tile_view", slice_prob=1.0, slice_all_tiles=True, slice_ratio=1.0,
                img_origin=False, slice_raw_cache_size=8, slice_grouped_sampler=False)
    ds.set_epoch(0, 10)
    assert len(ds) > 0 and ds._segment_lengths()[0] > 0, "no slice slots -- the check would be vacuous"

    handed_in, handed_out = [], []
    orig = ds._finalize_label

    def _spy(label, im):
        handed_in.append(im)
        out = orig(label, im)
        handed_out.append(out["img"])
        return out

    monkeypatch.setattr(ds, "_finalize_label", _spy)
    for i in range(len(ds)):
        ds.get_image_and_label(i)

    cached = [v for v in ds._raw_cache.values()]
    assert handed_in, "the slice branch never ran"
    views = sum(
        1 for arr in handed_in
        if any(arr is not v and np.shares_memory(arr, v) for v in cached)
    )
    assert views > 0, (
        "no tile arrived as a view of the LRU frame -- the contiguous copy is back, which reinstates "
        "the full-resolution memcpy this fix removed"
    )
    leak_out = sum(1 for arr in handed_out if any(np.shares_memory(arr, v) for v in cached))
    assert leak_out == 0, f"{leak_out}/{len(handed_out)} resized outputs still alias the raw LRU"


def test_prefetch_factor_reaches_the_grouped_loader(tmp_path, monkeypatch):
    """A configured ``prefetch_factor`` must reach the loader the grouped path builds.

    ``InfiniteDataLoader`` is swapped for a capture stub so no worker process is spawned: the assertion
    is about the ARGUMENT the builder derives, which is exactly what a silent no-op would break.
    """
    import ultralytics.data.build as _b

    ds = _build(tmp_path / "prefetch", slice_prob=1.0, slice_all_tiles=True, slice_ratio=1.0,
                img_origin=False, prefetch_factor=2)
    ds.set_epoch(0, 10)
    assert ds.prefetch_factor == 2, "the value never left hyp -- the loader cannot see it"

    captured: dict = {}

    class _Capture:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(_b, "InfiniteDataLoader", _Capture)
    _b.build_dataloader(ds, batch=2, workers=2, shuffle=True, rank=-1)

    assert captured.get("num_workers") == 2, captured.get("num_workers")
    assert captured.get("prefetch_factor") == 2, captured.get("prefetch_factor")
    assert type(captured.get("sampler")).__name__ == "GroupedImageSampler", captured.get("sampler")


def test_prefetch_factor_defaults_to_the_stock_value(tmp_path):
    """Unconfigured runs must reproduce the stock hardcoded depth, so this key is opt-in only."""
    ds = _build(tmp_path / "prefetch_default", slice_prob=1.0, slice_all_tiles=True, slice_ratio=1.0,
                img_origin=False)
    assert ds.prefetch_factor == 4, ds.prefetch_factor


@pytest.mark.parametrize("sigma", [5.0, 15.0, 30.0])
def test_the_weather_noise_kernel_keeps_the_requested_sigma(sigma):
    """The int16 rewrite must not change the realised noise strength."""
    from ultralytics_ooo.core import _apply_weather

    flat = np.full((128, 128, 3), 128, np.uint8)
    out = _apply_weather(flat, "noise", noise_std=sigma).astype(np.float32) - 128.0
    measured = float(out.std())
    assert abs(measured - sigma) < 0.15 * sigma, f"sigma {sigma} -> realised {measured:.2f}"
    assert abs(float(out.mean())) < 0.15 * sigma, f"noise is biased: mean {out.mean():.2f}"
