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
    try:
        # Force a fresh import of the core package
        for mod in [k for k in list(sys.modules) if k.startswith("ultralytics_ooo")]:
            del sys.modules[mod]
        import importlib
        import ultralytics_ooo.core as core
        assert hasattr(core, "_apply_motion_blur")
        assert hasattr(core, "slice_geometry")
        assert hasattr(core, "_imwrite")
        # and no ultralytics leaked in
        assert not any(k.startswith("ultralytics.") for k in sys.modules if k != "ultralytics_ooo")
    finally:
        sys.modules.update(saved)


def test_degrade_operators_shapes_and_types():
    from ultralytics_ooo.core import _apply_motion_blur, _apply_weather, _apply_occlusion, _ratio_pad_params
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
    img = (rng.random((3000, 4000, 3)) * 255).astype(np.uint8)
    out, scale = _cap_long_side(img, 640)
    # long side caps at 640, aspect preserved
    assert max(out.shape[:2]) == 640
    assert out.shape[2] == 3 and out.dtype == img.dtype
    # cap <= 0 returns same object / scale 1.0
    same, s = _cap_long_side(img, 0)
    assert s == 1.0
