"""Regression: dual-metric sliced validation (pool/dual.py).

The two behaviours pinned here are exactly the ones the original implementation got wrong, and that
no other test covered -- the whole suite was green with both defects live.

1. ``_run_whole`` must hand ``v.dataloader`` back to the SLICED loader.
   The validator instance -- and hence its ``dataloader`` -- is reused for every epoch: the trainer
   builds it once, and ``engine/validator.py`` only builds a loader when the current one is falsy
   (``self.dataloader = self.dataloader or self.get_dataloader(...)``). Leaving the whole-image loader
   behind therefore made the PRIMARY (slice) pass run whole-image from epoch 2 onwards: the patched
   ``update_metrics`` finds no ``val_slice_meta`` in the batch and silently falls back to the stock
   whole-image path, so the dual metric collapsed into plain whole-image validation and ``best.pt``
   was selected on whole-image fitness while the run still reported itself as sliced + dual.

2. ``save_model`` must not clobber ``best_whole.pt`` on a TIE.
   ``best_whole_fitness`` is only updated on a strict improvement, but the old guard compared it to
   ``fitness_whole`` with ``==``. An epoch that merely matched the best therefore overwrote
   ``best_whole.pt`` with its own ``last.pt``, destroying the real best snapshot.

Run: python -m pytest tests/test_ooo_dual.py -q
"""
from __future__ import annotations

from types import SimpleNamespace


def _args(**over):
    """Trainer/validator args for a dual-metric detect run."""
    base = {
        "val_slice_enable": True,
        "val_slice_dual_metric": True,
        "task": "detect",
        "split": "val",
        "batch": 8,
        "compile": False,
    }
    base.update(over)
    return SimpleNamespace(**base)


class _FakeValidator:
    """Mirrors the two upstream contracts this fix depends on.

    * ``__call__`` reproduces ``engine/validator.py``: a loader is built only when the current one is falsy.
    * ``update_metrics`` branching reproduces the patched validator: the sliced path needs
      ``val_slice_meta`` in the batch, which only the sliced dataset produces.
    """

    def __init__(self, args):
        self.args = args
        self.data = {"val": "val.txt"}
        self.dataloader = None  # BaseValidator.__init__(dataloader=None)
        self.loader_builds = 0
        self.used = []  # [(loader_kind, metrics_mode), ...]
        self.slice_fitness = 0.50
        self.whole_fitness = 0.50
        self.fail_whole = False

    def get_dataloader(self, dataset_path, batch_size=1):
        task = getattr(self.args, "task", "detect")
        sliced = bool(getattr(self.args, "val_slice_enable", False)) and task == "detect"
        self.loader_builds += 1
        return "SLICED" if sliced else "WHOLE"

    def __call__(self, trainer=None, model=None):
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
        kind = self.dataloader
        if kind == "WHOLE" and self.fail_whole:
            raise RuntimeError("simulated whole-image pass failure")
        mode = "sliced-metrics" if (self.args.val_slice_enable and kind == "SLICED") else "whole-metrics"
        self.used.append((kind, mode))
        fit = self.slice_fitness if kind == "SLICED" else self.whole_fitness
        return {"fitness": fit, "metrics/mAP50(B)": fit}


class _FakeTrainer:
    """Minimal BaseTrainer stand-in: the patch needs validate/save_model/wdir/last/batch_size."""

    def __init__(self, validator, args, batch_size, wdir):
        self.validator = validator
        self.args = args
        self.batch_size = batch_size
        self.wdir = wdir
        self.last = wdir / "last.pt"
        self.last.write_bytes(b"initial")

    def validate(self):
        # Faithful to engine/trainer.py:871-877 -- the primary pass IS a call to self.validator(self)
        # (that call is what populates v.dataloader on the first epoch), and upstream pops 'fitness'
        # out of the returned metrics dict before it reaches the results table.
        m = self.validator(self)
        if m is None:
            return None, None
        return m, m.pop("fitness", None)

    def save_model(self):
        return None


def _build(tmp_path):
    from ultralytics_ooo.pool.dual import patch_dual_metric

    args = _args()
    v = _FakeValidator(args)
    trainer = _FakeTrainer(v, args, 8, tmp_path)
    patch_dual_metric(_FakeTrainer)
    return trainer, v


def test_primary_pass_stays_sliced_across_epochs(tmp_path):
    """The regression: epoch 1 primary was sliced, epochs 2+ ran on the whole-image loader."""
    trainer, v = _build(tmp_path)
    for _ in range(3):
        trainer.validate()

    primary = v.used[0::2]  # validate() runs the primary pass first, then the whole-image pass
    whole = v.used[1::2]
    assert len(primary) == len(whole) == 3
    assert [k for k, _ in primary] == ["SLICED"] * 3, f"primary loader degraded: {v.used}"
    assert [m for _, m in primary] == ["sliced-metrics"] * 3, f"primary metrics degraded: {v.used}"
    assert [k for k, _ in whole] == ["WHOLE"] * 3


def test_whole_loader_is_built_once_and_reused(tmp_path):
    """One sliced loader (primary) + one whole loader, no matter how many epochs run."""
    trainer, v = _build(tmp_path)
    for _ in range(4):
        trainer.validate()
    assert v.loader_builds == 2, f"expected 2 loader builds (sliced + whole), got {v.loader_builds}"


def test_slice_enable_is_restored(tmp_path):
    trainer, _ = _build(tmp_path)
    trainer.validate()
    assert trainer.args.val_slice_enable is True


def test_whole_pass_failure_restores_sliced_loader(tmp_path):
    """A failing second pass must still leave the sliced loader in place (and not crash)."""
    trainer, v = _build(tmp_path)
    v.fail_whole = True
    trainer.validate()  # must not raise
    assert trainer.validator.dataloader == "SLICED"
    assert trainer.args.val_slice_enable is True
    # the next epoch still validates through the sliced path
    v.fail_whole = False
    trainer.validate()
    assert v.used[-2][0] == "SLICED"


def test_whole_metrics_are_prefixed_and_fitness_popped(tmp_path):
    """The second pass reports under whole_metrics/*; its 'fitness' drives best_whole, not the table."""
    trainer, _ = _build(tmp_path)
    metrics, fitness = trainer.validate()
    assert "whole_metrics/fitness" not in metrics, "fitness must be popped, never prefixed"
    assert "whole_metrics/metrics/mAP50(B)" in metrics
    assert "fitness" not in metrics, "upstream pops 'fitness' out of the metrics dict"
    assert fitness == 0.5, "validate() must return the PRIMARY fitness"


def test_save_model_writes_on_strict_improvement(tmp_path):
    trainer, v = _build(tmp_path)
    dst = tmp_path / "best_whole.pt"

    v.whole_fitness = 0.80
    trainer.validate()
    trainer.last.write_bytes(b"weights-epoch1")
    trainer.save_model()
    assert dst.read_bytes() == b"weights-epoch1"


def test_save_model_tie_does_not_destroy_best_whole(tmp_path):
    """The M3 regression: a tie used to overwrite best_whole.pt with this epoch's last.pt."""
    trainer, v = _build(tmp_path)
    dst = tmp_path / "best_whole.pt"

    v.whole_fitness = 0.80
    trainer.validate()
    trainer.last.write_bytes(b"weights-epoch1")
    trainer.save_model()

    # epoch 2 merely ties the best score -> not an improvement -> must NOT overwrite
    v.whole_fitness = 0.80
    trainer.validate()
    trainer.last.write_bytes(b"weights-epoch2")
    trainer.save_model()
    assert dst.read_bytes() == b"weights-epoch1", "a tie must not overwrite the best snapshot"

    # epoch 3 beats it -> must overwrite
    v.whole_fitness = 0.85
    trainer.validate()
    trainer.last.write_bytes(b"weights-epoch3")
    trainer.save_model()
    assert dst.read_bytes() == b"weights-epoch3"


def test_stale_whole_fitness_is_cleared_when_second_pass_skipped(tmp_path):
    """A skipped second pass must not leave last epoch's score looking like this epoch's best."""
    trainer, v = _build(tmp_path)
    dst = tmp_path / "best_whole.pt"

    v.whole_fitness = 0.90
    trainer.validate()
    trainer.last.write_bytes(b"weights-epoch1")
    trainer.save_model()
    assert dst.read_bytes() == b"weights-epoch1"

    v.fail_whole = True
    trainer.validate()  # second pass fails -> fitness_whole must be reset to None
    assert trainer.fitness_whole is None
    assert trainer._whole_improved is False
    trainer.last.write_bytes(b"weights-epoch2")
    trainer.save_model()
    assert dst.read_bytes() == b"weights-epoch1"


def test_patch_dual_metric_is_idempotent(tmp_path):
    from ultralytics_ooo.pool.dual import patch_dual_metric

    _trainer, _ = _build(tmp_path)
    first = _FakeTrainer.validate
    patch_dual_metric(_FakeTrainer)
    assert _FakeTrainer.validate is first
    assert hasattr(_FakeTrainer, "_run_whole")
