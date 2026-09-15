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


def test_slice_target_tiles_warns_when_it_cannot_choose_a_tile(monkeypatch):
    """``slice_target_tiles`` only exists for ``slice_all_tiles=False``; the other two shapes must say so.

    With ``all_tiles=True`` all four tiles are emitted anyway (the base segment is ``4*K_slice`` and the
    tile comes from ``index % 4``), and with slicing off there is no base segment. Both used to be the
    silent no-op class this package keeps hunting: a knob set to a non-default value that does nothing.
    """
    import logging

    from ultralytics_ooo.pool import augment_setup as au

    records = []

    class _H(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())

    logger = logging.getLogger("ultralytics")

    def ask(**kwargs):
        records.clear()
        monkeypatch.setattr(au, "_TARGET_TILES_WARNED", False)
        logger.addHandler(_H())
        try:
            au._warn_if_target_tiles_are_inert(**kwargs)
        finally:
            logger.handlers.pop()

    ask(slice_enabled=False, all_tiles=False, requested=True)  # slicing off -> inert
    assert len(records) == 1 and "has no effect" in records[0], records
    ask(slice_enabled=True, all_tiles=True, requested=True)  # all tiles already emitted -> inert
    assert len(records) == 1 and "slice_all_tiles=True" in records[0], records
    ask(slice_enabled=True, all_tiles=False, requested=True)  # the supported shape -> silent
    assert not records, records
    ask(slice_enabled=False, all_tiles=False, requested=False)  # knob off -> silent
    assert not records, records


def test_slice_at_prefer_target_substitutes_a_tile_that_keeps_the_target():
    """The scheduler's map is built on the CENTERED grid; the live grid can disagree (``center_bias``).

    ``slice_at(prefer_target=True)`` must therefore re-check its assigned tile against the live grid and
    substitute the first tile that really keeps a target -- otherwise a scheduled "target tile" would
    silently degrade into the empty-tile / Plan A fallback the scheduler exists to avoid.
    """
    import numpy as np

    from ultralytics_ooo.pool.augment_setup import OnlineSlice

    t = OnlineSlice(p=1.0, overlap_ratio=0.2, neg_ratio=-1)  # neg_ratio<0 -> every empty tile is emitted
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    # ONE box pinned inside the top-left quadrant, so only tile 0 can keep it
    label = {
        "bboxes": np.array([[24.5 / 96, 24.5 / 96, 24 / 96, 24 / 96]], dtype=np.float32),
        "bbox_format": "xywh", "normalized": True,
        "cls": np.array([[0]]), "segments": [],
    }
    assert t.target_tiles(label, (96, 96), key=0) == [0]
    assert t.target_tiles(label, (0, 0), key=0) == []  # degenerate shape -> no tile, no crash

    # blind mode honours the caller's k verbatim: tile 1 keeps nothing
    _sub, blind = t.slice_at(img, label, 1, key=0)
    assert len(blind["bboxes"]) == 0

    # prefer_target corrects it to the tile that does keep the target
    sub, corrected = t.slice_at(img, label, 1, key=0, prefer_target=True)
    assert len(corrected["bboxes"]) == 1, corrected["bboxes"]
    assert sub.shape[0] < 96 and sub.shape[1] < 96, sub.shape

    # ...and when NO tile keeps a target, k stands (the normal empty-tile rules still apply)
    lab_none = dict(label, bboxes=np.empty((0, 4), dtype=np.float32), cls=np.empty((0, 1)))
    _s, none_kept = t.slice_at(img, lab_none, 2, key=0, prefer_target=True)
    assert len(none_kept["bboxes"]) == 0
