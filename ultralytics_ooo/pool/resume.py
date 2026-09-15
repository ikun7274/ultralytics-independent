"""Patch resume training on a stock Ultralytics so ``resume_extend_epochs`` works.

Upstream ``check_resume`` rebuilds ``self.args`` from the checkpoint's saved train_args and only
re-applies a fixed whitelist of user overrides; ``resume_extend_epochs`` is not on that list. And
upstream ``resume_training`` asserts the checkpoint is not already finished. This module monkey-
patches both methods on ``BaseTrainer`` so that, when ``resume_extend_epochs > 0``, the checkpoint
metadata (epoch index / epochs / patience) is repaired and the LR schedule rebuilt to the new total,
mirroring the forked behaviour -- without editing any upstream source.
"""

from __future__ import annotations

from ultralytics.utils import LOGGER


def patch_resume(trainer_cls) -> None:
    """Install the resume_extend_epochs logic onto the given BaseTrainer class (idempotent)."""
    if getattr(trainer_cls, "_ooo_resume_patched", False):
        return

    _orig_resume = trainer_cls.resume_training
    _orig_check_resume = trainer_cls.check_resume

    def check_resume(self, overrides):
        _orig_check_resume(self, overrides)
        # Let the user's resume_extend_epochs survive the checkpoint-override rebuild (it is not on
        # upstream's resume whitelist).
        if "resume_extend_epochs" in overrides:
            self.args.resume_extend_epochs = overrides["resume_extend_epochs"]

    def resume_training(self, ckpt):
        if ckpt is None or not self.resume:
            return
        extend_epochs = int(getattr(self.args, "resume_extend_epochs", 0) or 0)
        if extend_epochs > 0:
            start_epoch = ckpt.get("epoch", -1) + 1
            finished_epoch = start_epoch if ckpt.get("epoch", -1) >= 0 else None
            if finished_epoch is None:
                for key in ("train_args", "args"):  # read original total from saved ckpt args
                    args = ckpt.get(key)
                    if args is not None:
                        finished_epoch = (
                            args.get("epochs") if isinstance(args, dict) else getattr(args, "epochs", None)
                        )
                        break
            if finished_epoch is None:
                raise ValueError("无法从 Checkpoint 中检测已完成的 Epoch 数, 无法续训延长。")
            if extend_epochs <= finished_epoch:
                raise ValueError(
                    f"resume_extend_epochs={extend_epochs} 必须大于 checkpoint 已完成轮数 {finished_epoch}, "
                    "否则续训无法继续/会倒退。"
                )
            # If the ckpt is marked finished (epoch<0, e.g. normal run / stripped), roll the epoch
            # index back to the last finished epoch so resume continues from finished+1.
            if ckpt.get("epoch", -1) < 0:
                ckpt["epoch"] = finished_epoch - 1
            # Repair ckpt metadata: epochs + patience, keep epoch index.
            for key in ("train_args", "args"):
                args = ckpt.get(key)
                if args is None:
                    continue
                if isinstance(args, dict):
                    args["epochs"] = extend_epochs
                    args["patience"] = self.args.patience
                else:
                    args.epochs = extend_epochs
                    args.patience = self.args.patience
            # New total + rebuild LR schedule against the new epoch count.
            self.epochs = self.args.epochs = extend_epochs
            self._setup_scheduler()
            LOGGER.info(
                f"[resume_extend] 自动修补 checkpoint 元数据: 已完成 {finished_epoch} 轮 -> "
                f"续训至 {extend_epochs} 轮 (patience={self.args.patience})"
            )
        _orig_resume(self, ckpt)

    trainer_cls.check_resume = check_resume
    trainer_cls.resume_training = resume_training
    trainer_cls._ooo_resume_patched = True
