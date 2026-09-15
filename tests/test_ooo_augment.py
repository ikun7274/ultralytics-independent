"""Regression: the online-augmentation assembly layer (OnlineSlice / _hyp_get / v8_transforms).

These are pure unit checks -- no data download, no model weights. They pin the config-lookup
fallback chain and that the assembly module imports cleanly on top of the in-tree Ultralytics.

Run:
    python -m pytest tests/test_ooo_augment.py -q
"""
from __future__ import annotations

from types import SimpleNamespace


def test_augment_setup_imports():
    from ultralytics_ooo.pool import augment_setup as au

    assert callable(au.v8_transforms)
    assert callable(au._hyp_get)
    assert au.OnlineSlice is not None
    assert callable(au._compat)


def test_hyp_get_prefers_hyp_then_default_table():
    from ultralytics_ooo.pool.augment_setup import _hyp_get

    hyp = SimpleNamespace(slice_prob=0.7)
    # 1) hyp wins
    assert _hyp_get(hyp, "slice_prob") == 0.7
    # 2) hyp missing -> package default table
    v = _hyp_get(hyp, "blur_short_len_min")
    assert isinstance(v, (int, float)) and v > 0
    # 3) unknown key -> None (call sites' `or ""/0/False` take over)
    assert _hyp_get(hyp, "totally_made_up_xyz") is None


def test_hyp_get_explicit_default():
    from ultralytics_ooo.pool.augment_setup import _hyp_get

    hyp = SimpleNamespace()
    assert _hyp_get(hyp, "anything", "fallback") == "fallback"


def test_compat_filters_unknown_kwargs():
    """_compat must drop kwargs the stock class does not accept, keep valid ones."""
    from ultralytics_ooo.pool.augment_setup import _compat

    class Stock:
        def __init__(self, a, b=2):
            self.a = a
            self.b = b

    obj = _compat(Stock, 1, b=5, save_dir="x", save_max=3)
    assert obj.a == 1 and obj.b == 5  # valid kwargs kept; unknown silently dropped


def test_online_defaults_cover_fork_keys():
    """Every fork-only key the assembly reads through _hyp_get has a package fallback."""
    from ultralytics_ooo.pool.constants import _ONLINE_DEFAULTS

    for k in ("slice_prob", "blur_keep", "ratio_pad_keep", "compose_keep", "close_aug_epoch"):
        assert k in _ONLINE_DEFAULTS, k


def test_mosaic_save_knobs_warn_that_they_are_inert(monkeypatch):
    """``mosaic_save_*`` came from the fork's Mosaic, which the stock class cannot honour.

    Stock ``Mosaic.__init__`` is ``(dataset, imgsz, p, n)`` -- no save argument -- so ``_compat``
    strips all four keys at construction and they are silent no-ops. Setting one must therefore WARN
    (exactly once) instead of letting the user believe mosaics are being written to disk.
    """
    import inspect
    import logging

    from ultralytics.data.augment import Mosaic
    from ultralytics_ooo.pool import augment_setup as au

    # premise: the stock class really takes no save argument (this is what makes the keys inert)
    params = set(inspect.signature(Mosaic.__init__).parameters)
    assert not params & {"save_dir", "save_max", "save_annotated", "exist_ok"}, params

    records = []

    class _H(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())

    logger = logging.getLogger("ultralytics")
    monkeypatch.setattr(au, "_MOSAIC_SAVE_WARNED", False)
    logger.addHandler(_H())
    try:
        au._warn_mosaic_save_is_a_noop(SimpleNamespace())  # all defaults -> must stay silent
        assert not [m for m in records if "mosaic_save" in m], records
        au._warn_mosaic_save_is_a_noop(SimpleNamespace(mosaic_save_dir="/tmp/x"))
        au._warn_mosaic_save_is_a_noop(SimpleNamespace(mosaic_save_max=10))  # 2nd -> once per process
    finally:
        logger.handlers.pop()

    warned = [m for m in records if "mosaic_save" in m]
    assert len(warned) == 1, f"expected exactly one warning, got {len(warned)}: {records}"
    assert "mosaic_save_dir" in warned[0], warned[0]
    assert "slice_save_dir" in warned[0], warned[0]
