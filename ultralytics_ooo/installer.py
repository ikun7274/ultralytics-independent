"""One-call installer: wire the extracted features onto a stock (pristine) Ultralytics.

After ``from ultralytics_ooo import install; install()``, a stock Ultralytics train gains the mixed
virtual-sample pool. It does NOT edit upstream source: it (1) swaps the dataset factory to a
multi-inheritance subclass, and (2) appends an epoch callback that publishes set_epoch to the train
dataset.

Note: on a stock Ultralytics the online slice/degrade augmentation is OFF by default
(``slice_prob=0``), so until the augment-assembly layer (OnlineSlice / project-level v8_transforms
overrides) is installed, behaviour is byte-for-byte upstream. That is the safe midpoint.

The dataset subclass is defined at MODULE TOP LEVEL (not as a closure inside install()). Windows
``spawn`` DataLoader workers unpickle it by re-importing this module and looking the name up there;
a class defined inside install() never exists in a worker, which only runs main() and imports the
package, so unpickling fails with "Can't get local object ...".
"""

from __future__ import annotations

_INSTALLED = False


def install() -> None:
    """Install the mixed-pool dataset factory and the set_epoch callback onto stock Ultralytics."""
    global _INSTALLED
    if _INSTALLED:
        return

    import ultralytics.data.build as _build
    from ultralytics_ooo.dataset_class import InstalledYOLODataset

    # Register the online hyperparameters onto the stock config namespace so model.train(slice_prob=...)
    # passes check_dict_alignment. get_cfg builds its base via cfg2dict(DEFAULT_CFG), so the keys must
    # live on the DEFAULT_CFG SimpleNamespace itself (not only DEFAULT_CFG_DICT). We never edit upstream
    # default.yaml; we only add missing attributes at runtime.
    from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT
    from ultralytics_ooo.pool.constants import _ONLINE_DEFAULTS

    for _k, _v in _ONLINE_DEFAULTS.items():
        if not hasattr(DEFAULT_CFG, _k):
            setattr(DEFAULT_CFG, _k, _v)
        DEFAULT_CFG_DICT.setdefault(_k, _v)

    # build_yolo_dataset references the module-level name `YOLODataset` at call time, so patching the
    # name in the build module redirects construction. (Depth/Semantic/MultiModal branches are untouched.)
    _build.YOLODataset = InstalledYOLODataset

    # YOLODataset.build_transforms calls the module-level name `v8_transforms` imported from augment;
    # swap it for the online-augment-aware version (installs OnlineSlice + mirrors the *_keep props).
    import ultralytics.data.dataset as _ds
    from ultralytics_ooo.pool.augment_setup import v8_transforms as _ooo_v8_transforms

    _ds.v8_transforms = _ooo_v8_transforms

    # Publish set_epoch at every epoch start. default_callbacks is module-level and deepcopy'ed into
    # each new trainer, so appending here reaches every future trainer instance.
    from ultralytics.utils.callbacks.base import default_callbacks

    def _ooo_set_epoch(trainer):
        dl = getattr(trainer, "train_dataloader", None)
        ds = getattr(dl, "dataset", None) if dl is not None else None
        # Unwrap DataLoader / InfiniteDataLoader / batch-sampler wrappers down to the real dataset.
        for _ in range(4):
            if ds is None:
                break
            inner = getattr(ds, "dataset", None)
            if inner is None:
                break
            ds = inner
        if ds is not None and hasattr(ds, "set_epoch"):
            ds.set_epoch(trainer.epoch, trainer.epochs)

    default_callbacks["on_train_epoch_start"].append(_ooo_set_epoch)

    # Patch resume_extend_epochs onto the trainer (修补续训): repair ckpt metadata + rebuild LR schedule
    # when resuming past the checkpoint's finished epoch count.
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics_ooo.pool.resume import patch_resume

    patch_resume(BaseTrainer)

    # Patch sliced validation (SAHI eval) onto DetectionValidator; off by default (val_slice_enable=False).
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics_ooo.pool.valslice import patch_validator

    patch_validator(DetectionValidator)

    # Route the training dataloader through GroupedImageSampler when the pool is worth grouping.
    from ultralytics_ooo.pool.sampler import patch_build_dataloader

    patch_build_dataloader()

    _INSTALLED = True
