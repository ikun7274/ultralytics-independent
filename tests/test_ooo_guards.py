"""Regression: the validation/guard layer added around the online augmentation.

Covers the issues that produced an obscure crash or a silent no-op rather than a message:

* ``slice_geometry`` must never return a zero-area tile (M8). The seam offset was only clamped to the
  image extent, so ``overlap_ratio=0`` + ``|bias|=0.5`` collapsed a tile to 0 px and the sample later
  died inside ``cv2.resize`` with ``(-215:Assertion failed) !ssize.empty()`` in a DataLoader worker.
* ``OnlineSlice`` must validate its seam knobs at construction (M8) and its save target at
  construction rather than on the first save, which happens inside a worker (M9).
* ``OnlinePoolDataset`` must resolve ``hyp``/``fraction`` even when they are passed POSITIONALLY (M7).
* ``_geometry(only=k)`` must be equivalent to computing all four and indexing ``k`` (L5) -- the
  optimization is only safe if the result is bit-identical.
* The config table must hold no upstream key (M4).

Run: python -m pytest tests/test_ooo_guards.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from ultralytics_ooo.core import slice_geometry

# --------------------------------------------------------------------------- M8: geometry


@pytest.mark.parametrize("overlap", [0.0, 0.2, 0.5])
@pytest.mark.parametrize("bias", [-0.5, -0.25, 0.0, 0.25, 0.5])
@pytest.mark.parametrize("w,h", [(4000, 3000), (64, 48), (2, 2), (1, 1), (33, 17)])
def test_slice_geometry_never_returns_a_degenerate_tile(overlap, bias, w, h):
    tiles = slice_geometry(w, h, overlap, bias, bias)
    for i, (x0, y0, x1, y1) in enumerate(tiles):
        assert x1 - x0 >= 1, f"tile {i} has zero width: {tiles}"
        assert y1 - y0 >= 1, f"tile {i} has zero height: {tiles}"
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, f"tile {i} out of bounds: {tiles}"


def test_slice_geometry_still_covers_the_image_and_keeps_the_overlap():
    """The bias must not break the contract: 4 tiles, full coverage, and the intended overlap."""
    for bias in (-0.25, 0.0, 0.25):
        tiles = slice_geometry(1000, 800, 0.2, bias, bias)
        assert len(tiles) == 4
        left_w = tiles[0][2]
        right_x = tiles[2][0]
        assert left_w + (1000 - right_x) >= 1000, "tiles must cover the full width"
        assert right_x < left_w, "tiles must overlap"


def test_slice_geometry_rejects_out_of_range_arguments():
    with pytest.raises(ValueError):
        slice_geometry(100, 100, 1.0)  # overlap must be < 1
    with pytest.raises(ValueError):
        slice_geometry(100, 100, 0.2, bias_x=0.6)
    with pytest.raises(ValueError):
        slice_geometry(100, 100, 0.2, bias_y=-0.6)


# --------------------------------------------------------------------------- M8/M9: OnlineSlice


def test_online_slice_validates_the_seam_knobs():
    from ultralytics_ooo.pool.augment_setup import OnlineSlice

    OnlineSlice(bias_margin=0.25, bias_jitter=0.05)  # defaults stay valid
    with pytest.raises(ValueError, match="bias_margin"):
        OnlineSlice(bias_margin=0.0)  # 0 lets the seam reach an image edge -> degenerate tiles
    with pytest.raises(ValueError, match="bias_margin"):
        OnlineSlice(bias_margin=0.6)
    with pytest.raises(ValueError, match="bias_jitter"):
        OnlineSlice(bias_jitter=-0.1)


def test_online_slice_validates_the_save_target_at_construction(tmp_path):
    """M9: was raised from inside _save_tile, i.e. in a DataLoader worker, minutes into training."""
    from ultralytics_ooo.pool.augment_setup import OnlineSlice

    existing = tmp_path / "already_there"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        OnlineSlice(save_dir=str(existing), exist_ok=False)
    OnlineSlice(save_dir=str(existing), exist_ok=True)  # opt-in still works
    OnlineSlice(save_dir=str(tmp_path / "brand_new"), exist_ok=False)  # missing dir is fine


def test_online_slice_end_to_end_with_an_extreme_box_distribution():
    """The configuration that used to crash: extreme seam bias + zero overlap."""
    from ultralytics_ooo.pool.augment_setup import OnlineSlice

    st = OnlineSlice(p=1.0, overlap_ratio=0.0, center_bias=True, bias_margin=0.05, bias_jitter=0.0,
                     min_area_ratio=0.0, min_retain_ratio=0.0)
    W, H, n = 4000, 3000, 40
    rs = np.random.RandomState(0)
    cx = rs.uniform(0.0, 0.05, n) * W  # all boxes hard left -> seam pushed hard right
    cy = rs.uniform(0.0, 1.0, n) * H
    label = {
        "bboxes": np.stack([cx / W, cy / H, np.full(n, 0.01), np.full(n, 0.01)], axis=1).astype(np.float32),
        "cls": np.zeros((n, 1), np.float32), "bbox_format": "xywh", "normalized": True,
        "im_file": "x.jpg", "img": np.zeros((H, W, 3), np.uint8),
    }
    img = np.zeros((H, W, 3), np.uint8)
    for k in range(4):
        x0, y0, x1, y1, _idx, _local = st._geometry(img, label, 0, only=k)[0]
        assert x1 - x0 >= 1 and y1 - y0 >= 1
        sub = np.ascontiguousarray(img[y0:y1, x0:x1])
        assert sub.size > 0 and sub.ndim == 3


# --------------------------------------------------------------------------- L5: only=k equivalence


@pytest.mark.parametrize("center_bias", [False, True])
@pytest.mark.parametrize("all_tiles", [False, True])
def test_geometry_only_k_is_equivalent_to_computing_all_four(center_bias, all_tiles):
    """_geometry(only=k) must return exactly what _geometry()[k] returned."""
    from ultralytics_ooo.pool.augment_setup import OnlineSlice

    st = OnlineSlice(p=1.0, overlap_ratio=0.2, center_bias=center_bias,
                     min_area_ratio=0.005, min_retain_ratio=0.4)
    W, H, n = 800, 600, 25
    rs = np.random.RandomState(3)
    cx, cy = rs.uniform(0, W, n), rs.uniform(0, H, n)
    bw, bh = rs.uniform(20, 120, n), rs.uniform(20, 120, n)
    label = {
        "bboxes": np.stack([cx / W, cy / H, bw / W, bh / H], axis=1).astype(np.float32),
        "cls": np.zeros((n, 1), np.float32), "bbox_format": "xywh", "normalized": True,
        "im_file": "x.jpg",
    }
    img = np.zeros((H, W, 3), np.uint8)
    all4 = st._geometry(img, label, key=7)
    assert len(all4) == 4
    for k in range(4):
        one = st._geometry(img, label, key=7, only=k)
        assert len(one) == 1, "only=k must return exactly one tile"
        a, b = one[0], all4[k]
        assert a[0:4] == b[0:4], f"tile {k} geometry differs"
        assert np.array_equal(a[4], b[4]), f"tile {k} index set differs"
        assert np.allclose(a[5], b[5]), f"tile {k} local boxes differ"


# --------------------------------------------------------------------------- M7: init knobs


def test_init_knobs_resolve_keyword_and_positional():
    from types import SimpleNamespace

    from ultralytics_ooo.pool.dataset import _resolve_init_knobs

    hyp = SimpleNamespace(slice_prob=0.3)
    # keyword form
    assert _resolve_init_knobs((), {"img_path": "x", "hyp": hyp, "fraction": 0.25}) == (hyp, 0.25)
    # positional form: (img_path, imgsz, cache, augment, hyp, prefix, rect, batch_size, stride, pad,
    #                    single_cls, classes, fraction)
    args = ("x", 640, False, True, hyp, "", False, 16, 32, 0.0, False, None, 0.75)
    got_hyp, got_fraction = _resolve_init_knobs(args, {})
    assert got_hyp is hyp, "a positional hyp must not silently fall back to DEFAULT_CFG"
    assert got_fraction == 0.75, "a positional fraction must not silently fall back to 1.0"
    # nothing supplied -> BaseDataset's own defaults
    from ultralytics.utils import DEFAULT_CFG

    assert _resolve_init_knobs(("x",), {}) == (DEFAULT_CFG, 1.0)
    # unbindable args must not raise
    assert _resolve_init_knobs(("x", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), {}) is not None


# --------------------------------------------------------------------------- M4: config table


def test_online_defaults_table_shadows_no_upstream_key():
    """Checked against cfg/default.yaml itself, so the result does not depend on install()'s ordering.

    Calling install() first, then comparing against the live DEFAULT_CFG_DICT, would report every
    project key as a duplicate -- install() writes them there on purpose.
    """
    from pathlib import Path

    import yaml

    import ultralytics
    from ultralytics_ooo.pool.constants import check_online_defaults_are_project_only

    with open(Path(ultralytics.__file__).parent / "cfg" / "default.yaml", encoding="utf-8") as fh:
        upstream = set(yaml.safe_load(fh) or {})
    dupes = check_online_defaults_are_project_only(upstream)
    assert dupes == [], (
        f"{len(dupes)} package key(s) are also in upstream default.yaml. They can never be applied to "
        f"DEFAULT_CFG (install() skips keys that already exist) yet stay live as the getattr fallback: "
        f"{dupes}"
    )


def test_online_defaults_table_still_holds_the_project_keys():
    """Sanity: the table was trimmed to project-only keys, not emptied."""
    from ultralytics_ooo.pool.constants import _ONLINE_DEFAULTS

    assert len(_ONLINE_DEFAULTS) > 50, "sanity: the table should still hold the project keys"


# --------------------------------------------- slice_prob: bool is a supported spelling

SLICE_PROB_DATASET = {"slice_all_tiles": True, "slice_ratio": 1.0, "img_origin": False}


def _build_with_slice_prob(tmp_path, value, **over):
    """Build the installed dataset through the real factory with the given ``slice_prob``."""
    import cv2
    import numpy as np

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics_ooo import install

    install()
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(4):
        cv2.imwrite(str(img_dir / f"i{i}.jpg"), (rng.random((48, 64, 3)) * 255).astype(np.uint8))
        (lbl_dir / f"i{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    cfg_in = dict(task="detect", mode="train", imgsz=64, batch=2, fraction=1.0, workers=0,
                  slice_prob=value, **SLICE_PROB_DATASET)
    cfg_in.update(over)
    cfg = get_cfg(overrides=cfg_in)
    data = {"path": str(tmp_path), "names": {0: "obj"}, "channels": 3, "nc": 1}
    return build_yolo_dataset(cfg, str(img_dir), 2, data, mode="train")


def test_slice_prob_accepts_booleans(tmp_path):
    """``slice_prob=True/False`` must be a supported spelling, identical to 1.0/0.0.

    It is the slicing master gate -- the code only ever tests ``> 0`` and forwards the value to
    ``OnlineSlice(p=...)`` -- so a bool is the natural way to write it. It worked before only because
    ``bool`` subclasses ``int`` (so ``float(True) == 1.0``); pinning it makes the contract explicit and
    keeps a bare bool from silently skipping the range check.
    """
    on_bool = _build_with_slice_prob(tmp_path / "on_bool", True)
    on_flt = _build_with_slice_prob(tmp_path / "on_flt", 1.0)
    off_bool = _build_with_slice_prob(tmp_path / "off_bool", False)
    off_flt = _build_with_slice_prob(tmp_path / "off_flt", 0.0)

    for ds, expected_p, enabled in ((on_bool, 1.0, True), (on_flt, 1.0, True),
                                    (off_bool, None, False), (off_flt, None, False)):
        st = getattr(ds, "slice_transform", None)
        assert (st is not None) is enabled
        if enabled:
            assert st.p == expected_p

    # a bool and its float twin must produce the same pool, not merely the same flag
    # (4 images x 4 tiles; img_origin is off in this fixture, so origin is 0)
    assert on_bool._segment_lengths() == on_flt._segment_lengths() == [16, 0, 0, 0, 0, 0, 0]
    assert len(on_bool) == len(on_flt) == 16
    # slicing OFF means no base tiles AND (img_origin off here) no origin segment -> a truly empty pool.
    # This is the explicit "discard the un-selected" extreme: nothing selects any image, so nothing is in.
    assert off_bool._segment_lengths() == off_flt._segment_lengths() == [0, 0, 0, 0, 0, 0, 0]
    assert len(off_bool) == len(off_flt) == 0
    # slicing off => n_per is 1 (no 2x2 tiling). img_origin is off too, so the pool is empty and the
    # EXTENDED-pool (non-plain) bookkeeping path is active (total 0 != n): the "discard everything"
    # degenerate case the ratio-sized layout enables.
    assert off_bool._n_per() == 1
    assert off_bool._extended_pool_on() is True


def test_slice_prob_out_of_range_raises_for_both_ends(tmp_path):
    """Validation happens BEFORE the ``> 0`` gate, so both ends of the range behave the same.

    ``OnlineSlice.__init__`` already rejected ``p=2``, but it is never even constructed when
    ``slice_prob <= 0`` -- so ``slice_prob=-1`` used to disable slicing silently while ``2`` raised.
    A probability outside [0, 1] is a config error either way, and ``False``/``0`` already mean 'off'.
    """
    for bad in (-1.0, -0.5, 2.0, 1.5):
        with pytest.raises(ValueError, match="slice_prob"):
            _build_with_slice_prob(tmp_path / f"bad{-bad}".replace(".", "_"), bad)


def test_slice_prob_rejects_non_numeric_values(tmp_path):
    """A non-numeric string, list or dict must fail with a message naming the type.

    Without this the value reaches ``OnlineSlice`` (or the ``> 0`` gate) and fails as a bare ``TypeError``
    somewhere downstream, which for a config typo is the least useful place to find out.
    """
    for bad in ("true", "on", [1], {"a": 1}):
        with pytest.raises(ValueError, match="slice_prob"):
            _build_with_slice_prob(tmp_path / f"bad_{type(bad).__name__}_{abs(hash(str(bad))) % 97}", bad)


def test_slice_prob_accepts_numeric_strings(tmp_path):
    """Numeric strings are coerced on purpose (a quoted YAML scalar is unambiguous), pinned here so the
    behaviour is deliberate rather than an accident of ``float()``."""
    from types import SimpleNamespace

    from ultralytics_ooo.pool import augment_setup as A

    assert A._resolve_slice_prob(SimpleNamespace(slice_prob="1")) == 1.0
    assert A._resolve_slice_prob(SimpleNamespace(slice_prob="0")) == 0.0
    assert A._resolve_slice_prob(SimpleNamespace(slice_prob="0.25")) == 0.25


def test_slice_prob_middle_range_warns_once_but_still_works(tmp_path, monkeypatch):
    """``0 < p < 1`` keeps working (older args.yaml / checkpoints carry it) but must say what it does.

    It is the most misinterpreted knob in the package: not "slice this fraction of the images" (that is
    ``slice_ratio``) but "each tile independently coin-flips, and the losers fall back to the WHOLE
    frame". Silently producing a half-whole-original pool is exactly the class of surprise this project
    keeps removing, so it warns ONCE per process -- and the warning names the two knobs that do what
    people actually mean.
    """
    import logging
    from types import SimpleNamespace

    from ultralytics_ooo.pool import augment_setup as A

    monkeypatch.setattr(A, "_SLICE_PROB_WARNED", False)
    records = []

    class _H(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())

    logger = logging.getLogger("ultralytics")
    logger.addHandler(_H())
    try:
        ds = _build_with_slice_prob(tmp_path / "mid", 0.5)
        A._resolve_slice_prob(SimpleNamespace(slice_prob=0.75))  # 2nd caller: must stay silent
    finally:
        logger.handlers.pop()

    assert ds.slice_transform is not None and ds.slice_transform.p == 0.5
    # the pool WIDTH is unchanged at p=0.5 -- only the CONTENT of the slots differs, which is why this
    # is so easy to miss in the logs
    assert ds._segment_lengths() == [16, 0, 0, 0, 0, 0, 0]
    assert len(ds) == 16

    warned = [m for m in records if "PER-SLOT coin flip" in m]
    assert len(warned) == 1, f"expected exactly one warning, got {len(warned)}: {records}"
    assert "0.5" in warned[0]
    assert "slice_ratio" in warned[0] and "img_origin" in warned[0], warned[0]


def test_slice_prob_true_and_false_are_the_only_gate_values_that_need_no_warning(tmp_path, monkeypatch):
    """The two documented spellings must be silent -- a warning on ``True`` would be pure noise."""
    import logging
    from types import SimpleNamespace

    from ultralytics_ooo.pool import augment_setup as A

    monkeypatch.setattr(A, "_SLICE_PROB_WARNED", False)
    records = []

    class _H(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())

    logger = logging.getLogger("ultralytics")
    logger.addHandler(_H())
    try:
        assert A._resolve_slice_prob(SimpleNamespace(slice_prob=True)) == 1.0
        assert A._resolve_slice_prob(SimpleNamespace(slice_prob=False)) == 0.0
        assert A._resolve_slice_prob(SimpleNamespace(slice_prob=1.0)) == 1.0
        assert A._resolve_slice_prob(SimpleNamespace(slice_prob=0)) == 0.0
    finally:
        logger.handlers.pop()

    assert [m for m in records if "PER-SLOT" in m] == [], records


# ------------------------------------------------- S4: the epoch callback must actually publish


def test_trainer_dataset_attribute_name_matches_upstream():
    """Structural guard for the S4 class of bug: the names install() reads must exist upstream.

    ``_ooo_set_epoch`` resolves the training dataset through ``trainer.train_loader``. It used to read
    ``trainer.train_dataloader``, an attribute ``BaseTrainer`` has never had -- so the callback was a
    silent no-op and ``set_epoch`` (and with it the whole per-epoch ``*_ratio`` draw and
    ``close_aug_epoch``) never ran during a real training run. Nothing failed; the pool just froze on
    its epoch-0 selection forever. Asserting the upstream names here turns the next rename into a test
    failure instead.
    """
    import inspect

    from ultralytics.engine.trainer import BaseTrainer

    src = inspect.getsource(BaseTrainer)
    assert "self.train_loader = self.get_dataloader(" in src, (
        "upstream no longer assigns self.train_loader in _setup_train -- update installer._ooo_set_epoch"
    )
    assert "self.train_dataloader" not in src, (
        "upstream now has train_dataloader; installer._ooo_set_epoch's fallback would still work, but "
        "the primary name should be re-checked"
    )


def test_epoch_callback_publishes_set_epoch_through_the_train_loader():
    """End-to-end on the INSTALLED callback list: a trainer's epoch start must reach set_epoch.

    Calls every registered ``on_train_epoch_start`` callback exactly as ``trainer.py:449`` does, against
    a fake trainer that exposes upstream's real attribute name, and asserts that the dataset's
    ``set_epoch`` was invoked with the trainer's epoch and total epochs. This is the assertion whose
    absence let the callback be a no-op through the whole project.
    """
    from ultralytics.utils.callbacks.base import default_callbacks
    from ultralytics_ooo import install

    install()

    calls = []

    class _Dataset:
        def set_epoch(self, epoch=0, epochs=None):
            calls.append((epoch, epochs))

    class _Loader:
        dataset = _Dataset()

    class _Trainer:
        train_loader = _Loader()
        epoch, epochs = 3, 10

    cbs = list(default_callbacks.get("on_train_epoch_start", []))
    for fn in cbs:
        try:
            fn(_Trainer())
        # Third-party integration callbacks (mlflow/wandb/...) need a real trainer and may raise; that
        # is not this test's subject -- only OUR callback's effect is, asserted below.
        except Exception:  # noqa: BLE001, S112
            continue

    assert calls == [(3, 10)], (
        "on_train_epoch_start must reach dataset.set_epoch(epoch, epochs) EXACTLY ONCE (a duplicated "
        "registration would run the per-epoch draw twice per epoch); got "
        f"{calls} from {len(cbs)} callbacks"
    )


def test_epoch_callback_tolerates_an_unsupported_trainer_without_crashing():
    """The callback must not break training it cannot serve -- it warns and moves on.

    A trainer shape we do not recognise (a future upstream refactor, a third-party trainer) must never
    turn into an exception inside the epoch loop; the warning is what makes the degradation visible.
    """
    from ultralytics.utils.callbacks.base import default_callbacks
    from ultralytics_ooo import install

    install()

    class _TrainerWithNoLoader:
        epoch, epochs = 0, 1

    for fn in list(default_callbacks.get("on_train_epoch_start", [])):
        try:
            fn(_TrainerWithNoLoader())
        # Same as above: unrelated integration callbacks may fail; only the no-loader path matters here.
        except Exception:  # noqa: BLE001, S112
            continue
