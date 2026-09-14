"""One-call installer: wire the extracted features onto a stock (pristine) Ultralytics.

After ``from ultralytics_ooo import install; install()``, a stock Ultralytics train gains the mixed
virtual-sample pool. It does NOT edit upstream source: it (1) swaps the dataset factory to a
multi-inheritance subclass, and (2) appends an epoch callback that publishes set_epoch to the train
dataset.

Note: on a stock Ultralytics the online slice/degrade augmentation is OFF by default
(``slice_prob=0``), so until the augment-assembly layer (OnlineSlice / project-level v8_transforms
overrides) is installed, behaviour is byte-for-byte upstream. That is the safe midpoint.
"""

from __future__ import annotations

_INSTALLED = False


def install() -> None:
    """Install the mixed-pool dataset factory and the set_epoch callback onto stock Ultralytics."""
    global _INSTALLED
    if _INSTALLED:
        return

    import ultralytics.data.build as _build
    from ultralytics.data.dataset import YOLODataset
    from ultralytics_ooo.pool.dataset import OnlinePoolDataset

    # Cooperative subclass: OnlinePoolDataset's methods win the MRO; YOLODataset supplies get_labels
    # and the stock build_transforms.
    class InstalledYOLODataset(OnlinePoolDataset, YOLODataset):
        """YOLODataset + the mixed virtual-sample pool, with no upstream edits."""

    # build_yolo_dataset references the module-level name `YOLODataset` at call time, so patching the
    # name in the build module redirects construction. (Depth/Semantic/MultiModal branches are untouched.)
    _build.YOLODataset = InstalledYOLODataset

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
    _INSTALLED = True
