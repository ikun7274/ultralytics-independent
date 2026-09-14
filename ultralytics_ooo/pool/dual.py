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
                tmpdir = Path(tempfile.mkdtemp(prefix="ooo_plot_"))
                tmp = tmpdir / "results.csv"
                df.select(keep).write_csv(tmp)
                return _orig(file=str(tmp), on_plot=on_plot)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(f"val_slice_dual_metric: plot wrapper fell back to stock ({e}).")
        return _orig(file=file, dir=dir, on_plot=on_plot)

    _t.plot_results = _safe
    _t._ooo_plot_patched = True


def _run_whole(self):
    """Second, whole-image validation pass: temporarily disable slicing and rebuild the val loader."""
    v = getattr(self, "validator", None)
    if v is None or getattr(v, "data", None) is None:
        return None
    vargs = v.args
    orig = bool(getattr(vargs, "val_slice_enable", False))
    try:
        vargs.val_slice_enable = False
        # Rebuild a whole-image val loader (patched get_dataloader takes the sliced branch only when active).
        v.dataloader = v.get_dataloader(v.data["val"], self.batch_size)
        return v(self)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"val_slice_dual_metric: whole-image second pass failed ({e}); skipping best_whole.pt.")
        return None
    finally:
        vargs.val_slice_enable = orig


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

        wm = _run_whole(self)
        if wm:
            wfit = wm.pop("fitness", None)
            for k, val in wm.items():
                metrics[f"whole_metrics/{k}"] = val
            self.fitness_whole = float(wfit) if wfit is not None else None
            if self.fitness_whole is not None and (
                getattr(self, "best_whole_fitness", None) is None
                or self.fitness_whole > self.best_whole_fitness
            ):
                self.best_whole_fitness = self.fitness_whole
        return metrics, fitness

    def save_model(self):
        # Stock save_model writes last.pt (and best.pt when primary fitness is best). We piggyback: when the
        # secondary whole-image fitness just hit its own best, copy the freshly written last.pt bytes to
        # best_whole.pt. Using the same bytes avoids a second serialization.
        ret = _orig_save_model(self)
        if (
            bool(getattr(self.args, "val_slice_dual_metric", False))
            and getattr(self, "fitness_whole", None) is not None
            and getattr(self, "best_whole_fitness", None) == self.fitness_whole
        ):
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
