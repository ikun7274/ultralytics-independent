"""Dual-metric sliced validation: the slice fitness and the whole-image fitness each pick their own snapshot.

When ``val_slice_dual_metric=True`` (and ``val_slice_enable=True``) every epoch runs validation TWICE:
the stock (already sliced) pass drives the primary ``fitness`` -> ``best.pt`` / ``last.pt`` path, and a
second pass with slicing turned off scores the whole-image set. Its metrics are prefixed
``whole_metrics/*`` in the results table, and its fitness drives a SECOND pair of checkpoints:

    best_whole.pt   written when the whole-image fitness hits its own best (mirrors best.pt)
    last_whole.pt   written every epoch (mirrors last.pt)

Resume / early-stopping / 续训 stay on the slice (primary) path; the whole family is a reference set for
fair comparison against whole-image-only baselines.

``last_whole.pt`` is a byte copy of ``last.pt`` at the moment it is written, and the two end up
CONTENT-equivalent: same weights (element-wise), same key set, same 716 zip entries with the same payload
bytes. They are NOT byte-identical files, and no two Ultralytics checkpoints ever are after stripping --
``strip_optimizer`` re-serializes each file with ``torch.save``, whose zip container embeds the archive
name (the file stem) in every entry prefix. Measured on a real 2-epoch run: ``last.pt`` 5,346,238 B with
prefix ``last/`` vs ``last_whole.pt`` 5,382,086 B with prefix ``last_whole/``, 716 equal entries and equal
payload totals, zero differing payloads. The same holds for upstream's own ``best.pt`` / ``last.pt``,
which ``save_model`` writes from ONE shared byte buffer yet which differ after their separate strips. So
compare these files by content (or by the metric in ``results.csv``), never by hash.

``final_eval`` is patched too, because upstream only strips ``last.pt`` / ``best.pt`` on its way out: the
whole-metric mirrors would otherwise survive as fp32 checkpoints still carrying the optimizer state,
several times larger than the ``best.pt`` they are meant to be compared against.

Implemented as monkey-patches on ``BaseTrainer`` only -- no upstream edits.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import LOGGER


def _patch_plot_results():
    """Wrapper stock plot_results so the extra ``whole_*`` columns don't break its even subplot layout.

    Stock plot_results builds a 2-row grid from the loss/metric column count; the dual-metric pass adds
    whole_* loss/metric columns, making the count odd and raising ``index out of bounds`` when it plots.
    We strip whole_* columns into a temp CSV and plot that; the whole metrics still live in results.csv.

    The temp CSV has to live in a directory of its own, because upstream plots EVERY ``results*.csv`` it
    globs next to ``file`` -- a sibling name such as ``results_no_whole.csv`` would be picked up twice.
    That directory is also where upstream writes the FIGURE (``save_dir = Path(file).parent``), so the
    PNG is moved back to the real run directory before the temp dir is removed. Without that the
    figure was created in the temp dir and deleted with it: measured, a patched call left the run
    directory with no ``results.png`` at all while the stock call produced one. ``on_plot`` is
    deliberately NOT forwarded to the inner call -- upstream would report the temp path, which no
    longer exists by the time the callback's consumers look at it -- and is invoked here with the final
    path instead.
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
                    _orig(file=str(tmp), on_plot=None)  # figure lands in tmpdir (see docstring)
                    produced = Path(tmpdir) / "results.png"
                    if not produced.exists():
                        # Nothing to rescue -- let upstream run its stock path so its own error/logging
                        # behaviour (including the "no results*.csv" assertion) still applies.
                        LOGGER.warning(
                            "val_slice_dual_metric: the filtered results plot produced no figure; "
                            "re-plotting the unfiltered CSV with the stock layout."
                        )
                    else:
                        final = src.parent / "results.png"
                        # write_bytes rather than move: shutil.move falls back to copy anyway, and this
                        # keeps the temp dir removable even when it sits on another volume.
                        final.write_bytes(produced.read_bytes())
                        if on_plot:
                            on_plot(final)
                        return
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(f"val_slice_dual_metric: plot wrapper fell back to stock ({e}).")
        return _orig(file=file, dir=dir, on_plot=on_plot)

    _t.plot_results = _safe
    _t._ooo_plot_patched = True


def _dual_enabled(args) -> bool:
    """True only when BOTH switches are on.

    Dual-metric is a MODE of sliced validation, not an alternative to it: with ``val_slice_enable`` off
    the whole-image pass would be the only pass, and there is nothing to be dual about (and no sliced
    loader for ``_run_whole`` to restore). Callers pass ``self.args``, which is None-ish on the stub
    trainers the tests use, hence the getattr.
    """
    return bool(getattr(args, "val_slice_dual_metric", False)) and bool(getattr(args, "val_slice_enable", False))


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
        LOGGER.warning(
            f"val_slice_dual_metric: whole-image second pass failed ({e}); this epoch contributes no "
            f"whole_metrics and no best_whole.pt update."
        )
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
        if not _dual_enabled(self.args):
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
        # Stock save_model writes last.pt (and best.pt when the PRIMARY fitness is best). We piggyback on
        # those bytes instead of serializing again: the checkpoint was built once at the top of that
        # method, so each mirror is a plain file copy.
        #
        #   last_whole.pt  mirrors last.pt -- written EVERY epoch. The weights do not depend on which
        #                  metric scored them, so the two are content-equivalent (same weights, same key
        #                  set, same zip payloads -- but NOT the same bytes; see the module docstring).
        #   best_whole.pt  mirrors best.pt -- written only when the SECONDARY fitness hits its own best.
        #
        # Gate the best snapshot on the explicit ``_whole_improved`` flag set by validate() -- NOT on
        # ``best_whole_fitness == fitness_whole``. That float comparison was also true on a TIE (the
        # best value is only updated on a strict improvement), so an epoch that merely matched the best
        # overwrote best_whole.pt with its own last.pt and destroyed the real best snapshot.
        ret = _orig_save_model(self)
        if not _dual_enabled(getattr(self, "args", None)):
            return ret
        try:
            written = self.last.read_bytes()
            (self.wdir / "last_whole.pt").write_bytes(written)
            if getattr(self, "_whole_improved", False):
                (self.wdir / "best_whole.pt").write_bytes(written)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(f"val_slice_dual_metric: failed to write the whole-metric checkpoint ({e}).")
        return ret

    def final_eval(self):
        # Upstream strips last.pt / best.pt here -- optimizer -> None, weights -> fp16 EMA, epoch -> -1 --
        # so what you release is an inference artifact. It knows nothing about our whole-metric mirrors,
        # which are copied from an UNSTRIPPED last.pt before this point and would therefore survive as
        # fp32 checkpoints still carrying the optimizer state, several times larger than the best.pt they
        # are meant to be compared against. Strip them here with the same train_results update best.pt
        # gets: a best_whole.pt taken at epoch k stores only the first k+1 rows of the curve.
        ret = _orig_final_eval(self)
        if not _dual_enabled(getattr(self, "args", None)):
            return ret
        try:
            from ultralytics.utils import RANK
            from ultralytics.utils.torch_utils import strip_optimizer
        except Exception as e:  # noqa: BLE001 -- upstream moved a symbol; the run itself is unaffected
            LOGGER.warning(
                f"val_slice_dual_metric: cannot reach the checkpoint-stripping helper ({e}); "
                f"best_whole.pt / last_whole.pt stay unstripped (fp32, with optimizer state)."
            )
            return ret
        if RANK not in {-1, 0}:  # only rank 0 owns the weights dir, same guard upstream uses
            return ret
        try:
            results = self.read_results_csv() if hasattr(self, "read_results_csv") else {}
            updates = {"train_results": results} if results else None
            for name in ("last_whole.pt", "best_whole.pt"):
                path = self.wdir / name
                if path.exists():
                    strip_optimizer(path, updates=updates)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(f"val_slice_dual_metric: failed to strip the whole-metric checkpoint ({e}).")
        return ret

    trainer_cls.validate = validate
    trainer_cls.save_model = save_model
    trainer_cls._run_whole = _run_whole
    # ``final_eval`` is what performs the strip, so it is patched as well -- but it IS optional at patch
    # time (the stub trainers in the test suite define only validate/save_model). Skipping it silently
    # would reintroduce exactly the defect the wrapper exists to fix, so it warns instead.
    _orig_final_eval = getattr(trainer_cls, "final_eval", None)
    if _orig_final_eval is None:
        LOGGER.warning(
            f"val_slice_dual_metric: {trainer_cls.__name__} has no final_eval(); best_whole.pt / "
            f"last_whole.pt will stay unstripped (fp32, with optimizer state). Re-check pool/dual.py "
            f"against this Ultralytics version."
        )
    else:
        # Marker on the WRAPPER, so the failure mode this project keeps hitting stays detectable: if a
        # future upstream rename makes the wrapper never land, the log warning above is the only trace.
        # tools/ooo_compat_check.py asserts this marker, which turns it into a hard failure instead.
        final_eval._ooo_final_eval_wrapper = True
        trainer_cls.final_eval = final_eval
    trainer_cls._ooo_dual_patched = True
    _patch_plot_results()
