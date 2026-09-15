"""Regression: resume_extend_epochs patching logic (no real training, no weights).

We exercise the patched resume_training closure against a fake trainer + fake checkpoint to pin:
(1) extend<=finished raises, (2) finished epoch is read back from ckpt train_args/args,
(3) a finished-marked ckpt (epoch<0) gets its index repaired, (4) epochs/patience are rewritten and
the LR scheduler rebuilt.

Run:
    python -m pytest tests/test_ooo_resume.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _make_trainer(ckpt, extend):
    """Build a fake trainer that mimics what resume_training touches."""
    trainer = SimpleNamespace(
        resume=True,
        ckpt=ckpt,
        args=SimpleNamespace(resume_extend_epochs=extend, patience=50, model="m.pt"),
        epochs=None,
    )
    trainer._scheduler_calls = 0

    def _setup_scheduler():
        trainer._scheduler_calls += 1

    trainer._setup_scheduler = _setup_scheduler
    return trainer


def _run_resume(trainer, ckpt):
    """Invoke the same closure patch_resume installs, against a no-op original."""
    from ultralytics_ooo.pool.resume import patch_resume

    called = {}

    class FakeTrainer:
        def resume_training(self, ckpt):
            called["orig"] = True

        def check_resume(self, overrides):
            called["check"] = True

    patch_resume(FakeTrainer)  # idempotent, installs closures on the class
    ft = FakeTrainer()
    ft.__dict__.update(trainer.__dict__)
    ft.resume_training(ckpt)
    return ft, called


def test_extend_le_finished_raises():
    ckpt = {"epoch": 9, "train_args": {"epochs": 10}}  # finished=10
    trainer = _make_trainer(ckpt, extend=10)  # extend==finished
    with pytest.raises(ValueError, match="必须大于"):
        _run_resume(trainer, ckpt)


def test_finished_read_from_train_args_and_rewrites():
    # ckpt marked finished (epoch=-1) with saved train_args.epochs=4
    ckpt = {"epoch": -1, "train_args": {"epochs": 4, "patience": 100}}
    trainer = _make_trainer(ckpt, extend=7)
    ft, called = _run_resume(trainer, ckpt)
    assert called["orig"] is True, "the patched resume_training must still delegate to the original"
    # index repaired from -1 back to finished-1 == 3
    assert ckpt["epoch"] == 3
    # ckpt metadata rewritten to the new total
    assert ckpt["train_args"]["epochs"] == 7
    assert ckpt["train_args"]["patience"] == 50  # from self.args.patience
    # trainer total + scheduler rebuilt
    assert ft.epochs == 7
    assert ft.args.epochs == 7
    assert trainer._scheduler_calls == 1  # closure mutates the source trainer


def test_check_resume_threads_extend_override():
    """check_resume must surface a user-passed resume_extend_epochs onto self.args."""
    from ultralytics_ooo.pool.resume import patch_resume

    class FakeTrainer:
        def resume_training(self, ckpt):
            pass

        def check_resume(self, overrides):
            self.args.epochs = 1  # rebuilt from ckpt (no extend key in ckpt)

    patch_resume(FakeTrainer)
    ft = FakeTrainer()
    ft.args = SimpleNamespace()
    ft.check_resume({"resume_extend_epochs": 12})
    assert ft.args.resume_extend_epochs == 12


def test_patch_is_idempotent():
    from ultralytics_ooo.pool.resume import patch_resume

    class T:
        def resume_training(self, ckpt):
            pass

        def check_resume(self, overrides):
            pass

    patch_resume(T)
    first = T.resume_training
    patch_resume(T)
    assert T.resume_training is first  # second call is a no-op
