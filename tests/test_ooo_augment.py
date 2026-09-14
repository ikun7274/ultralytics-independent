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
