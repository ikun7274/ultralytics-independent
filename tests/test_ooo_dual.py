"""Regression: dual-metric sliced validation (pool/dual.py).

The behaviours pinned here are the ones the implementation got wrong, plus the two checkpoint-contract
rules that came out of reading ``final_eval``. No other test covered any of them -- the whole suite was
green with every defect live.

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

3. The whole family must have the same SHAPE as the main family: ``last_whole.pt`` mirrors ``last.pt``
   every epoch, ``best_whole.pt`` mirrors ``best.pt`` on improvement, and neither exists unless dual
   mode is actually on.

4. ``final_eval`` must strip the whole-metric mirrors.
   Upstream strips only ``last.pt`` / ``best.pt``. The mirrors are copied from an UNSTRIPPED ``last.pt``
   before that point, so they used to survive as fp32 checkpoints still carrying the optimizer state --
   several times larger than the ``best.pt`` they exist to be compared against.

Run: python -m pytest tests/test_ooo_dual.py -q
"""
from __future__ import annotations

from pathlib import Path
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
    """Minimal BaseTrainer stand-in: the patch needs validate/save_model/final_eval/wdir/last/batch_size."""

    def __init__(self, validator, args, batch_size, wdir):
        self.validator = validator
        self.args = args
        self.batch_size = batch_size
        self.wdir = wdir
        self.last = wdir / "last.pt"
        self.last.write_bytes(b"initial")
        self.final_eval_calls = 0

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

    def final_eval(self):
        # Upstream strips last.pt / best.pt in here. The fake only counts the call: these tests are about
        # the wrapper's own work on the whole-metric mirrors, not about upstream's strip.
        self.final_eval_calls += 1

    def read_results_csv(self):
        return {"metrics/mAP50(B)": [0.5]}


def _build(tmp_path, **arg_over):
    from ultralytics_ooo.pool.dual import patch_dual_metric

    args = _args(**arg_over)
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


# --- the whole-checkpoint family: last_whole.pt + the strip on the way out --------------------------------------


def _write_real_ckpt(path, epoch=1):
    """Write a checkpoint shaped the way ``trainer.save_model`` serializes one: ``model=None``, ``ema=<Module>``.

    Deliberately real, so the REAL ``strip_optimizer`` can process it: it asserts ``"model" in x``, replaces
    that entry with ``x["ema"]``, then calls ``.half()`` / ``.parameters()`` on the result. So the ``ema``
    entry must be an ``nn.Module`` -- which is exactly what ``save_model`` stores (``unwrap_model(ema.ema)``
    after a ``deepcopy(...).half()``, NOT the ModelEMA wrapper, which has none of those methods).
    """
    import torch

    torch.save(
        {
            "epoch": epoch,
            "best_fitness": 0.5,
            "model": None,
            "ema": torch.nn.Linear(2, 2),
            "optimizer": {"state": {}, "param_groups": [{"lr": 0.01}]},
            "updates": epoch + 1,
            "train_args": {"epochs": 2},
            "train_results": {"metrics/mAP50(B)": [0.1]},
        },
        path,
    )


def _reload(path):
    from ultralytics.utils.patches import torch_load

    return torch_load(path, map_location="cpu")


def test_last_whole_pt_mirrors_last_pt_every_epoch(tmp_path):
    """last_whole.pt mirrors last.pt: always current, and NOT gated on the whole fitness improving."""
    trainer, v = _build(tmp_path)
    dst = tmp_path / "last_whole.pt"

    for i in (1, 2, 3):
        v.whole_fitness = 0.50  # first epoch IS an improvement (prev best is None), the rest tie
        trainer.validate()
        trainer.last.write_bytes(f"weights-epoch{i}".encode())
        trainer.save_model()
        assert dst.read_bytes() == f"weights-epoch{i}".encode(), f"epoch {i}: last_whole.pt not mirrored"


def test_last_whole_pt_is_written_even_when_the_whole_pass_fails(tmp_path):
    """A failed second pass still writes last_whole.pt -- see the reasoning in save_model.

    "last" promises the LATEST state; gating it on the pass would instead leave a stale file from an
    earlier epoch, which is worse for a file whose whole purpose is being current. The failure is already
    warned about, and best_whole.pt is correctly left alone.
    """
    trainer, v = _build(tmp_path)
    v.fail_whole = True
    trainer.validate()
    trainer.last.write_bytes(b"weights-after-failed-pass")
    trainer.save_model()
    assert (tmp_path / "last_whole.pt").read_bytes() == b"weights-after-failed-pass"
    assert not (tmp_path / "best_whole.pt").exists()


def test_no_whole_checkpoints_without_dual_mode(tmp_path):
    """With dual off the weights dir must look exactly like a stock run's."""
    trainer, _ = _build(tmp_path, val_slice_dual_metric=False)
    trainer.validate()
    trainer.last.write_bytes(b"weights")
    trainer.save_model()
    assert not (tmp_path / "last_whole.pt").exists()
    assert not (tmp_path / "best_whole.pt").exists()


def test_final_eval_strips_both_whole_mirrors(tmp_path):
    """The defect: the mirrors were copied from an UNSTRIPPED last.pt and survived as fp32 + optimizer."""
    trainer, v = _build(tmp_path)

    v.whole_fitness = 0.80
    trainer.validate()
    _write_real_ckpt(trainer.last, epoch=1)
    trainer.save_model()
    for name in ("last_whole.pt", "best_whole.pt"):
        assert (tmp_path / name).exists(), name

    assert _reload(tmp_path / "best_whole.pt")["epoch"] == 1, "precondition: unstripped before final_eval"
    assert _reload(tmp_path / "best_whole.pt")["optimizer"] is not None, "precondition: optimizer still present"

    trainer.final_eval()
    assert trainer.final_eval_calls == 1, "upstream's final_eval must still run exactly once"

    import torch

    for name in ("last_whole.pt", "best_whole.pt"):
        ck = _reload(tmp_path / name)
        assert ck["epoch"] == -1, f"{name}: epoch must be reset to -1"
        assert ck["optimizer"] is None, f"{name}: the optimizer state must be dropped"
        assert ck["ema"] is None, f"{name}: the heavy ema entry must be cleared"
        assert ck["model"].weight.dtype == torch.float16, f"{name}: weights must be fp16"
        assert not ck["model"].weight.requires_grad, f"{name}: requires_grad must be off"
        assert ck["train_results"], f"{name}: the full train_results curve must be carried over"


def test_final_eval_is_left_alone_without_dual_mode(tmp_path):
    """No whole mirrors -> nothing to strip, and the wrapper must not touch the weights dir."""
    trainer, _ = _build(tmp_path, val_slice_dual_metric=False)
    trainer.validate()
    _write_real_ckpt(trainer.last, epoch=1)
    trainer.save_model()
    trainer.final_eval()
    assert trainer.final_eval_calls == 1


def test_the_filtered_results_plot_still_lands_in_the_run_directory(tmp_path):
    """``results.png`` must survive the whole_*-stripping wrapper.

    Upstream ``plot_results`` globs every ``results*.csv`` next to ``file`` AND writes the figure to
    ``Path(file).parent``. The wrapper needs the glob to see ONLY the filtered CSV, so it plots a
    temporary one -- and the figure therefore went into the temporary directory and was deleted with it.
    Measured on a real dual-metric run: the run directory ended up with ``Box*_curve.png`` but no
    ``results.png``, silently.

    The assertion is on the file, deliberately: a return value or a "no exception" check passes on the
    broken version too.
    """
    import shutil

    from ultralytics_ooo.pool.dual import _patch_plot_results

    _patch_plot_results()
    import ultralytics.engine.trainer as _t

    assert getattr(_t, "_ooo_plot_patched", False), "the plot wrapper is not installed"

    run = tmp_path / "run"
    run.mkdir()
    # A realistic dual-metric results.csv: the column COUNT decides whether upstream's 2-row subplot
    # grid is even, so a hand-made CSV can be degenerate in a way that makes both the patched and the
    # stock call fail (and then the check would prove nothing).
    header = (
        "epoch,time,train/box_loss,train/cls_loss,train/l1_loss,metrics/precision(B),metrics/recall(B),"
        "metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/l1_loss,"
        "whole_metrics/metrics/precision(B),whole_metrics/metrics/recall(B),"
        "whole_metrics/metrics/mAP50(B),whole_metrics/metrics/mAP50-95(B),whole_metrics/val/box_loss,"
        "whole_metrics/val/cls_loss,whole_metrics/val/l1_loss\n"
    )
    (run / "results.csv").write_text(
        header + "1,1.0,0.9,0.8,0.7,0.5,1.0,0.5,0.2,0.8,0.7,0.6,0.4,1.0,0.3,0.1,0.7,0.6,0.5\n",
        encoding="utf-8",
    )
    assert not (run / "results.png").exists()

    plotted: list = []
    _t.plot_results(file=str(run / "results.csv"), on_plot=lambda name, data=None: plotted.append(Path(name)))

    assert (run / "results.png").exists(), (
        "the filtered plot wrote its figure into the temporary directory and deleted it"
    )
    assert (run / "results.png").stat().st_size > 0, "results.png is empty"
    assert plotted and plotted[0].parent == run, f"on_plot reported {plotted}, expected the run dir"
    assert (run / "results.csv").exists(), "the real results.csv must be left alone"

    # and the stock reader is untouched: the whole_ columns are still in the CSV it was given
    assert "whole_metrics/metrics/mAP50(B)" in (run / "results.csv").read_text(encoding="utf-8")

    # contrast: re-plotting the SAME csv without stripping whole_* must still work (that is why the
    # temporary copy exists at all), i.e. the wrapper is not simply bypassing the original path.
    import polars as pl

    from ultralytics.utils.plotting import plot_results as _stock

    bare = tmp_path / "bare"
    bare.mkdir()
    df = pl.read_csv(run / "results.csv", infer_schema_length=None)
    df.select([c for c in df.columns if not c.startswith("whole_")]).write_csv(bare / "results.csv")
    _stock(file=str(bare / "results.csv"))
    assert (bare / "results.png").exists(), "the stock path itself no longer produces results.png"
    shutil.rmtree(bare, ignore_errors=True)
