"""Self-check for the extracted pure kernels in ultralytics_ooo.core.

Run with the conda env:
    python -m pytest tests/test_ooo_core.py -q

Note: the earlier "bit-identical to the in-tree refactored original" comparisons lived against the
deleted 8.4.137 fork's extra functions (slice_geometry / online_degrade / _cap_long_side), which do
not exist in pristine 8.4.126. That porting-correctness check already passed against the fork during
extraction; here we only self-test the kernels we ship.
"""
from __future__ import annotations

import numpy as np


def test_core_imports_without_ultralytics(monkeypatch):
    """core/ must import with ultralytics NOT in sys.modules."""
    import sys
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "ultralytics" or k.startswith("ultralytics.")}
    # Drop every ultralytics_ooo module too (that is the point: force a fresh import), but remember them:
    # leaving them deleted gives LATER tests a second, independent copy of the package, whose module-level
    # state (install()'s _INSTALLED flag) starts fresh. That used to make install() run twice in one
    # process and register the epoch callback twice -- install() is now idempotent by marker, and this
    # restore keeps the test from manufacturing the situation in the first place.
    dropped = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("ultralytics_ooo")}
    try:
        from ultralytics_ooo import core
        assert hasattr(core, "_apply_motion_blur")
        assert hasattr(core, "slice_geometry")
        assert hasattr(core, "_imwrite")
        # and no ultralytics leaked in
        assert not any(k.startswith("ultralytics.") for k in sys.modules if k != "ultralytics_ooo")
    finally:
        for mod in [k for k in list(sys.modules) if k.startswith("ultralytics_ooo")]:
            del sys.modules[mod]
        sys.modules.update(dropped)
        sys.modules.update(saved)


def test_degrade_operators_shapes_and_types():
    from ultralytics_ooo.core import _apply_motion_blur, _apply_occlusion, _apply_weather, _ratio_pad_params
    rng = np.random.default_rng(0)
    img = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    for length, ang in [(8, 0), (25, 90)]:
        out = _apply_motion_blur(img, length=length, angle=ang, axis_aligned=True)
        assert out.shape == img.shape and out.dtype == np.uint8
    for t in ["rain", "haze", "noise"]:
        out = _apply_weather(img, t)
        assert out.shape == img.shape and out.dtype == np.uint8
    for t in ["rect", "stripe"]:
        out, boxes = _apply_occlusion(img, t, blocks=2, size_ratio=0.1, color="auto")
        assert out.shape == img.shape and out.dtype == np.uint8
        assert len(boxes) == 2
    # ratio pad returns (new_w, new_h, pl, pt) or None
    assert _ratio_pad_params(320, 240, "auto", True) is not None
    assert _ratio_pad_params(320, 240, "4:3", False) is None  # already 4:3


def test_cap_long_side_self_check():
    from ultralytics_ooo.core import _cap_long_side
    rng = np.random.default_rng(2)
    # ``rng.integers(..., dtype=np.uint8)`` instead of ``(rng.random(...) * 255).astype(np.uint8)``:
    # the float64 intermediate for a 3000x4000x3 frame is 288 MiB, which turned this test into an
    # OOM on a small machine (the image itself is only 36 MiB). Same coverage, 8x less peak memory.
    try:
        img = rng.integers(0, 255, size=(3000, 4000, 3), dtype=np.uint8)
    except MemoryError as e:  # pragma: no cover - environment limit, not a code defect
        import pytest

        pytest.skip(f"needs ~36 MiB for a 3000x4000x3 frame and could not allocate it: {e}")
    out, scale = _cap_long_side(img, 640)
    # long side caps at 640, aspect preserved
    assert max(out.shape[:2]) == 640
    assert out.shape[2] == 3 and out.dtype == img.dtype
    # the scale is the long-side ratio the caller rescales boxes with (measured: 640 / 4000)
    assert abs(scale - 640 / 4000) < 1e-9, f"scale={scale!r} should be the long-side ratio"
    # cap <= 0 returns same object / scale 1.0
    same, s = _cap_long_side(img, 0)
    assert s == 1.0
    assert same is img, "cap <= 0 must pass the frame through instead of copying it"
