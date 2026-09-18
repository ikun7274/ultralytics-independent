"""One-call installer: wire the extracted features onto a stock (pristine) Ultralytics.

After ``from ultralytics_ooo import install; install()``, a stock Ultralytics train gains the mixed
virtual-sample pool. It does NOT edit upstream source: it (1) swaps the dataset factory to a
multi-inheritance subclass, and (2) appends an epoch callback that publishes set_epoch to the train
dataset.

Note: on a stock Ultralytics the online slice/degrade augmentation is OFF by default
(``slice_prob=0``), so until the augment-assembly layer (OnlineSlice / project-level v8_transforms
overrides) is installed, behaviour is byte-for-byte upstream. That is the safe midpoint.

Everything this module installs lives under ``pool/`` (they are NOT top-level submodules):
``pool.dataset`` (mixed pool + fraction guard), ``pool.augment_setup`` (OnlineSlice / v8_transforms),
``pool.sampler`` (grouped sampler + build_dataloader), ``pool.resume`` (resume_extend_epochs),
``pool.valslice`` (sliced validation) and ``pool.dual`` (dual-metric validation).

The dataset subclass is defined at MODULE TOP LEVEL (not as a closure inside install()). Windows
``spawn`` DataLoader workers unpickle it by re-importing this module and looking the name up there;
a class defined inside install() never exists in a worker, which only runs main() and imports the
package, so unpickling fails with "Can't get local object ...".
"""

from __future__ import annotations

_INSTALLED = False

# Cached "upstream-only" key set. See install() for why it must be cached AND filtered.
_PRISTINE_KEYS: set[str] | None = None


def install() -> None:
    """Install the mixed-pool dataset factory and the set_epoch callback onto stock Ultralytics."""
    global _INSTALLED
    if _INSTALLED:
        return

    import ultralytics.data.build as _build

    # Register the online hyperparameters onto the stock config namespace so model.train(slice_prob=...)
    # passes check_dict_alignment. get_cfg builds its base via cfg2dict(DEFAULT_CFG), so the keys must
    # live on the DEFAULT_CFG SimpleNamespace itself (not only DEFAULT_CFG_DICT). We never edit upstream
    # default.yaml; we only add missing attributes at runtime.
    from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER
    from ultralytics_ooo.dataset_class import InstalledYOLODataset
    from ultralytics_ooo.pool.constants import _ONLINE_DEFAULTS, check_online_defaults_are_project_only

    # Snapshot the upstream-only key set for the drift check below.
    #
    # ``set(DEFAULT_CFG_DICT)`` alone is NOT that set: the loop underneath writes our own keys into
    # the very same dict, so on any SECOND install (a re-imported copy of the package, or two copies
    # imported through different paths -- exactly the case the marker-based idempotency further down
    # is written to support) the snapshot can no longer tell our keys from upstream's. Measured: the
    # guard then reported all 80 project keys as upstream collisions and advised deleting the whole
    # fallback table, which would have left ``_online_default()`` returning None for every one of
    # them. Two changes make it order-independent: subtract the package's own keys, and cache the
    # result so the check always runs against one stable set.
    global _PRISTINE_KEYS
    if _PRISTINE_KEYS is None:
        _PRISTINE_KEYS = set(DEFAULT_CFG_DICT) - set(_ONLINE_DEFAULTS)
    _drift = check_online_defaults_are_project_only(_PRISTINE_KEYS)
    if _drift:
        LOGGER.warning(
            f"ultralytics_ooo: {len(_drift)} config key(s) in pool/constants.py::_ONLINE_DEFAULTS are "
            f"now ALSO defined by upstream default.yaml ({_drift}). Upstream has taken ownership of "
            f"them, so this package's fallback value is never applied (install() only writes a key "
            f"when it is absent) while still being live behind ~120 `getattr(self, key, "
            f"_online_default(key))` sites. Remove those rows from pool/constants.py::_ONLINE_DEFAULTS "
            f"so upstream stays the single source of truth -- do NOT remove the rest of the table."
        )

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
        # ``train_loader`` is the attribute upstream actually sets (trainer.py:286, and every use of it):
        # there is NO ``train_dataloader`` on BaseTrainer. Reading the wrong name made this callback a
        # silent no-op, which meant set_epoch -- and therefore the whole per-epoch ratio draw and
        # ``close_aug_epoch`` -- never ran in a real training. The fallback keeps a future upstream
        # rename from reverting to that same silent state, and the trailing warning makes it audible
        # instead of invisible.
        dl = getattr(trainer, "train_loader", None) or getattr(trainer, "train_dataloader", None)
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
            return True
        LOGGER.warning(
            f"ultralytics_ooo: could not find the training dataset on the trainer "
            f"({type(trainer).__name__}); the per-epoch *_ratio draw and close_aug_epoch stay frozen "
            f"at epoch 0. Attribute names to check: train_loader / train_dataloader."
        )
        return False

    # Idempotent by MARKER, not by identity: ``_INSTALLED`` only guards this process's first call, so a
    # second copy of the package (re-imported after a sys.modules purge, or imported twice via different
    # paths) would append a second callback and call set_epoch twice per epoch. The marker survives both.
    _ooo_set_epoch._ooo_epoch_callback = True
    if not any(getattr(f, "_ooo_epoch_callback", False) for f in default_callbacks["on_train_epoch_start"]):
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

    # Dual-metric whole-image second validation (best_whole.pt) on BaseTrainer; off by default.
    from ultralytics_ooo.pool.dual import patch_dual_metric

    patch_dual_metric(BaseTrainer)

    # Route the training dataloader through GroupedImageSampler when the pool is worth grouping.
    from ultralytics_ooo.pool.sampler import patch_build_dataloader

    patch_build_dataloader()

    # Survive fraction rounding to zero on a tiny dataset.
    from ultralytics_ooo.pool.fraction import patch_fraction_guard

    patch_fraction_guard()

    _INSTALLED = True
