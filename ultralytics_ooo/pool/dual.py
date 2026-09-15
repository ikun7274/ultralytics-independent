"""Dual-metric sliced validation: slice fitness drives best.pt, whole-image fitness drives best_whole.pt.

When ``val_slice_dual_metric=True`` (and ``val_slice_enable=True``) every epoch runs validation TWICE:
the stock (already sliced) pass drives the primary ``fitness`` -> ``best.pt`` / ``last.pt`` path, and a
second pass with slicing turned off scores the whole-image set. Its metrics are prefixed
``whole_metrics/*`` in the results table and its fitness drives a separate ``best_whole.pt`` snapshot.
Resume / early-stopping / 续训 stay on the slice (primary) path; ``best_whole.pt`` is a reference
artifact for fair comparison against whole-image-only baselines. Implemented as monkey-patches on
``BaseTrainer`` only -- no upstream edits.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import LOGGER


def _patch_plot_results():
    """Wrapper stock plot_results so the extra ``whole_*`` columns don't break its even subplot layout.

    Stock plot_results builds a 2-row grid from the loss/metric column count; the dual-metric pass adds
    whole_* loss/metric columns, making the count odd and raising ``index out of bounds`` when it plots.
    We strip whole_* columns into a temp CSV and plot that; the whole metrics still live in results.csv.
    """
    import ultralytics.engine.trainer as _t
    if getattr(_t, "_ooo_plot_patched", False):
        return
    _orig = _t.plot_results

    def _safe(file: str = "", dir: str = "", on_plot=None):
        src = Path(file) if file else Path(dir) / "results.csv"
        try:
            if src.exists() and "whole_" in src.read_text(encoding="utf-8", errors="ignore"):
                import tempfile

                import polars as pl

                df = pl.read_csv(src, infer_schema_length=None)
                keep = [c for c in df.columns if not c.startswith("whole_")]
                # TemporaryDirectory removes itself; the previous mkdtemp() leaked one ooo_plot_* dir
                # (containing a results.csv) into the system temp dir on EVERY call, forever.
                with tempfile.TemporaryDirectory(prefix="ooo_plot_") as tmpdir:
                    tmp = Path(tmpdir) / "results.csv"
                    df.select(keep).write_csv(tmp)
                    return _orig(file=str(tmp), on_plot=on_plot)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(f"val_slice_dual_metric: plot wrapper fell back to stock ({e}).")
        return _orig(file=file, dir=dir, on_plot=on_plot)

    _t.plot_results = _safe
    _t._ooo_plot_patched = True


def _run_whole(self):
    """Second, whole-image validation pass: temporarily disable slicing and swap in a whole-image loader.

    The validator instance -- and therefore ``v.dataloader`` -- is REUSED for every epoch: the trainer
    builds the validator once (``engine/trainer.py``) and ``engine/validator.py`` only builds a loader
    when the current one is falsy (``self.dataloader = self.dataloader or self.get_dataloader(...)``).
    So this method MUST put ``v.dataloader`` back in ``finally``. Leaving the whole-image loader behind
    made every subsequent epoch's PRIMARY pass run on the whole-image loader; ``update_metrics`` then
    finds no ``val_slice_meta`` in the batch and silently falls back to the stock whole-image path, so
    the dual metric collapses into single-metric whole-image validation from epoch 2 on -- with the
    slice-primary fitness driving ``best.pt`` no longer being the sliced metric at all.
    """
    v = getattr(self, "validator", None)
    if v is None or getattr(v, "data", None) is None:
        return None
    vargs = v.args
    orig_enable = bool(getattr(vargs, "val_slice_enable", False))
    orig_loader = v.dataloader  # the sliced loader the primary pass just used
    try:
        vargs.val_slice_enable = False
        # Build the whole-image loader once and reuse it: rebuilding it every epoch re-runs
        # build_dataset (label scan / image verification) for nothing.
        whole = getattr(self, "_whole_val_loader", None)
        if whole is None:
            v.dataloader = None  # falsy -> the patched get_dataloader takes the whole-image branch
            whole = v.get_dataloader(v.data["val"], self.batch_size)
            self._whole_val_loader = whole
        v.dataloader = whole
        return v(self)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"val_slice_dual_metric: whole-image second pass failed ({e}); skipping best_whole.pt.")
        return None
    finally:
        vargs.val_slice_enable = orig_enable
        v.dataloader = orig_loader  # restore, or the next epoch's primary pass is whole-image (see docstring)


def patch_dual_metric(trainer_cls) -> None:
    """Install dual-metric whole-image second validation onto a stock BaseTrainer (idempotent)."""
    if getattr(trainer_cls, "_ooo_dual_patched", False):
        return

    _orig_validate = trainer_cls.validate
    _orig_save_model = trainer_cls.save_model

    def validate(self):
        metrics, fitness = _orig_validate(self)
        if metrics is None:
            return metrics, fitness
        dual = bool(getattr(self.args, "val_slice_dual_metric", False)) and bool(
            getattr(self.args, "val_slice_enable", False)
        )
        if not dual:
            return metrics, fitness

        # Reset BOTH per-epoch flags up front: a failed/skipped second pass must not leave the previous
        # epoch's fitness_whole behind, or save_model would treat a stale score as this epoch's best.
        self.fitness_whole = None
        self._whole_improved = False
        wm = _run_whole(self)
        if wm:
            wfit = wm.pop("fitness", None)
            for k, val in wm.items():
                metrics[f"whole_metrics/{k}"] = val
            self.fitness_whole = float(wfit) if wfit is not None else None
            prev_best = getattr(self, "best_whole_fitness", None)
            self._whole_improved = self.fitness_whole is not None and (
                prev_best is None or self.fitness_whole > prev_best
            )
            if self._whole_improved:
                self.best_whole_fitness = self.fitness_whole
        return metrics, fitness

    def save_model(self):
        # Stock save_model writes last.pt (and best.pt when primary fitness is best). We piggyback: when the
        # secondary whole-image fitness just hit its own best, copy the freshly written last.pt bytes to
        # best_whole.pt. Using the same bytes avoids a second serialization.
        #
        # Gate on the explicit ``_whole_improved`` flag set by validate() -- NOT on
        # ``best_whole_fitness == fitness_whole``. That float comparison was also true on a TIE (the
        # best value is only updated on a strict improvement), so an epoch that merely matched the best
        # overwrote best_whole.pt with its own last.pt and destroyed the real best snapshot.
        ret = _orig_save_model(self)
        if bool(getattr(self.args, "val_slice_dual_metric", False)) and getattr(self, "_whole_improved", False):
            try:
                dst = self.wdir / "best_whole.pt"
                dst.write_bytes(self.last.read_bytes())
            except Exception as e:  # noqa: BLE001
                LOGGER.warning(f"val_slice_dual_metric: failed to write best_whole.pt ({e}).")
        return ret

    trainer_cls.validate = validate
    trainer_cls.save_model = save_model
    trainer_cls._run_whole = _run_whole
    trainer_cls._ooo_dual_patched = True
    _patch_plot_results()
