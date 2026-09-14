"""Stage-1 self-check: the extracted core kernels are bit-identical to the in-tree originals.

Run with the conda env:
    python -m pytest tests/test_ooo_core.py -q
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


def test_slice_geometry_matches_original():
    from ultralytics_ooo.core import slice_geometry as new
    from ultralytics.data.augment import slice_geometry as old
    for w, h in [(4000, 3000), (1280, 720), (800, 800)]:
        for ov in [0.0, 0.2, 0.4]:
            assert new(w, h, ov) == old(w, h, ov)
    # bias paths
    assert new(4000, 3000, 0.2, 0.1, -0.05) == old(4000, 3000, 0.2, 0.1, -0.05)


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


def test_motion_blur_bit_identical_to_original():
    from ultralytics_ooo.core import _apply_motion_blur as new
    from ultralytics.data.online_degrade import _apply_motion_blur as old
    rng = np.random.default_rng(1)
    img = (rng.random((300, 400, 3)) * 255).astype(np.uint8)
    np.random.seed(42)
    a = new(img, length=22, angle=30, defocus_sigma=1.0, axis_aligned=False)
    np.random.seed(42)
    b = old(img, length=22, angle=30, defocus_sigma=1.0, axis_aligned=False)
    assert np.array_equal(a, b)
    np.random.seed(7)
    a2 = new(img, length=10, angle=0, axis_aligned=True)
    np.random.seed(7)
    b2 = old(img, length=10, angle=0, axis_aligned=True)
    assert np.array_equal(a2, b2)


def test_cap_long_side_bit_identical():
    from ultralytics_ooo.core import _cap_long_side as new
    from ultralytics.data.base import _cap_long_side as old
    rng = np.random.default_rng(2)
    img = (rng.random((3000, 4000, 3)) * 255).astype(np.uint8)
    a, sa = new(img, 640)
    b, sb = old(img, 640)
    assert np.array_equal(a, b) and sa == sb
    # cap <= 0 returns same object
    assert new(img, 0)[1] == 1.0
