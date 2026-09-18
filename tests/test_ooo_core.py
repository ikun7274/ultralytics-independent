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


def test_weather_noise_clamps_at_both_ends_instead_of_wrapping_or_absing(monkeypatch):
    """The noise tail is ONE saturating cast; this pins it to ``clip(0, 255)`` at both ends.

    It used to be ``np.clip(scratch, 0, 255, out=scratch)`` followed by ``scratch.astype(np.uint8)``
    (two full-frame numpy passes) and is now a single ``cv2.add(scratch, 0, dtype=cv2.CV_8U)``. That
    is only equivalent because cv2.add SATURATES into the destination type. The equivalent-looking
    shortcut ``cv2.convertScaleAbs`` would take the ABSOLUTE value of the negative excursions
    instead: no shape symptom, no dtype symptom, just different pixels everywhere a pixel undershoots.
    ``cv2.randn`` is replaced by a deterministic ramp spanning -300..+300 so both clamps are hit by
    every pixel and an ``abs()`` cannot masquerade as a clamp.
    """
    from ultralytics_ooo.core import _apply_weather, degrade

    img = np.full((20, 30, 3), 128, np.uint8)
    n = img.size

    def fake_randn(dst, mean, stddev):
        dst[...] = (((np.arange(dst.size, dtype=np.int64) * 97) % 601) - 300).reshape(dst.shape).astype(dst.dtype)

    monkeypatch.setattr(degrade.cv2, "randn", fake_randn)
    out = _apply_weather(img, "noise", noise_std=15.0)

    noise = (((np.arange(n, dtype=np.int64) * 97) % 601) - 300).reshape(img.shape)
    assert out.min() == 0, "the fixture stopped exercising the lower clamp"
    assert out.max() == 255, "the fixture stopped exercising the upper clamp"
    assert np.array_equal(out, np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)), (
        "the noise tail is no longer clip(0, 255) + uint8 cast"
    )


def test_weather_kernels_keep_the_shape_contract_at_every_channel_count():
    """``(H, W, 1)`` and ``(H, W)`` frames must come back with the SAME shape.

    OpenCV maps an ``(H, W, 1)`` array onto a 1-CHANNEL Mat, so ``convertScaleAbs`` / ``add`` /
    ``addWeighted`` all hand back ``(H, W)`` -- that is why every kernel here goes through
    ``_match_ndim``. The single-channel case is not exotic: it is an ``ndim == 3`` frame, so it takes
    the same int16 noise branch as a colour frame and loses its axis unless the result is re-wrapped.
    The rain ``dst=`` fast path is deliberately restricted to 3-channel frames for the same reason.
    """
    from ultralytics_ooo.core import _apply_weather

    for shape in [(24, 32, 3), (24, 32, 1), (24, 32)]:
        img = np.full(shape, 100, np.uint8)
        for wtype in ("noise", "rain", "haze"):
            out = _apply_weather(img, wtype, noise_std=20.0, rain_density=0.5, rain_length=5.0)
            assert out.shape == shape, f"{wtype} on {shape} came back as {out.shape}"
            assert out.dtype == np.uint8, f"{wtype} on {shape} came back as {out.dtype}"


def test_degradation_ops_never_write_through_to_their_input():
    """Every online degradation kernel must leave the frame it is handed BYTE-IDENTICAL.

    ``pool/dataset.py::_degrade_frame_ex`` now reads with ``copy=False``, so a kernel can be handed the
    raw LRU buffer itself on the un-capped path. Two independent defences keep that safe: the
    ``capped is im`` guard in the caller, and the fact that every kernel allocates its own output.
    Neither is load-bearing on its own -- which is exactly why each needs its own test. This pins the
    kernel half directly, without depending on the caller at all.

    Switching ``_apply_occlusion`` from ``out = img.copy()`` to ``out = img`` is the regression this
    catches (see _perf_review/inject_check.py). It is invisible from the outside: a caller that passes
    a private frame still gets the right picture, and the poisoned frame only shows up on a LATER
    sample -- through ``Mosaic`` / ``affine`` writing ``label["img"]`` in place, or through any other
    branch reading the same cached image again. So the check is on the INPUT, not on the output.
    """
    import random

    from ultralytics_ooo.core import _apply_motion_blur, _apply_occlusion, _apply_weather

    def seed():
        # The kernels draw from both global streams (numpy for the noise/rain geometry, ``random`` for
        # the occlusion boxes), so both have to be reset for a repeat call to be comparable.
        random.seed(3)
        np.random.seed(3)

    ops = [
        ("motion_blur", lambda im: _apply_motion_blur(im, length=9.0, angle=30.0)),
        ("motion_blur_axis", lambda im: _apply_motion_blur(im, length=9.0, angle=90.0, axis_aligned=True)),
        ("rain", lambda im: _apply_weather(im, "rain")),
        ("haze", lambda im: _apply_weather(im, "haze")),
        ("noise", lambda im: _apply_weather(im, "noise")),
        ("occlusion_rect", lambda im: _apply_occlusion(im, "rect", blocks=2, size_ratio=0.1, color="auto")[0]),
        ("occlusion_stripe", lambda im: _apply_occlusion(im, "stripe", blocks=2, size_ratio=0.1,
                                                         color="black")[0]),
    ]
    rng = np.random.default_rng(11)
    for shape in [(240, 320, 3), (240, 320, 1), (240, 320)]:
        for name, op in ops:
            img = rng.integers(0, 255, size=shape, dtype=np.uint8)
            before = img.copy()
            seed()
            first = op(img)
            assert np.array_equal(img, before), (
                f"{name} wrote through to the frame it was handed (shape {shape}) -- a degradation op "
                "may only READ its input; the caller's frame is not scratch space"
            )
            seed()
            second = op(img)
            assert np.array_equal(np.asarray(first), np.asarray(second)), (
                f"{name} is not deterministic across two calls on the same input (shape {shape})"
            )


def test_ratio_pad_matches_the_old_canvas_and_block_copy_on_every_shape():
    """``cv2.copyMakeBorder`` replaced "np.full canvas + block copy": same bytes, one allocation less.

    The trap is the argument order. ``copyMakeBorder`` wants ``(top, bottom, left, right)`` and the two
    FAR sides are REMAINDERS (``new - size - offset``); ``_ratio_pad_params`` returns the CENTRED
    offsets. Whenever ``new - size`` is odd the two sides differ by one pixel, so passing the offset
    for the far side shifts the whole frame by a pixel with no error -- which is why the fixture
    insists on covering an odd remainder.
    """
    import cv2

    from ultralytics_ooo.core import _RATIO_PAD_COLORS, _ratio_pad_params

    odd_remainders = 0
    checked = 0
    for w, h in [(64, 48), (48, 64), (100, 37), (37, 100), (96, 54), (54, 96), (160, 90), (90, 160),
                 (65, 49), (99, 101)]:
        for target in ("auto", "4:3", "16:9"):
            pad = _ratio_pad_params(w, h, target, auto=(target == "auto"))
            if pad is None:
                continue
            new_w, new_h, pad_left, pad_top = pad
            bottom, right = new_h - h - pad_top, new_w - w - pad_left
            odd_remainders += int((pad_top != bottom) or (pad_left != right))
            im = (np.arange(h * w * 3, dtype=np.uint32).reshape(h, w, 3) % 251).astype(np.uint8)
            for color in ("black", "gray", "white"):
                val = _RATIO_PAD_COLORS[color]
                old = np.full((new_h, new_w, 3), val, dtype=im.dtype)
                old[pad_top:pad_top + h, pad_left:pad_left + w] = im
                new = cv2.copyMakeBorder(im, pad_top, bottom, pad_left, right,
                                         cv2.BORDER_CONSTANT, value=val)
                assert new.shape == old.shape
                assert np.array_equal(old, new), f"{w}x{h} {target} {color}: pad differs"
                checked += 1
    assert checked >= 20, f"the fixture only reached {checked} pad cases"
    assert odd_remainders >= 1, (
        "no case produced an odd remainder, so passing the offset for the far side would still pass"
    )


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
