"""Mixed-pool segment boundaries and the grouped image sampler."""

from __future__ import annotations

import math
import os
from typing import Iterator, NamedTuple

import torch
from torch.utils.data import Sampler

from .constants import _online_default


class SegmentBases(NamedTuple):
    """Mixed-pool segment boundaries: named fields instead of a bare 8-tuple.

    Field order equals the legacy tuple order, so position-unpacking call sites remain valid.
    """

    base: int
    origin: int
    ratio: int
    blur: int
    compose: int
    weather: int
    occlusion: int
    total: int


class GroupedImageSampler(Sampler[int]):
    """Shuffle the sample pool while keeping every sub-sample of one source image together.

    Consumes the "units" produced by the dataset's ``grouped_sample_units``: a unit is a group of at
    most four source images, holding each image's pool indices as its own block. Units are visited in
    random order and each unit is consumed round-robin over its blocks, so all of the unit's source
    images stay inside the per-worker raw LRU for the whole unit and each is decoded once.
    """

    def __init__(self, units: list[list[list[int]]], seed: int = 0):
        self.units = units
        self._total = sum(len(block) for unit in units for block in unit)
        self._seed = int(seed) % (1 << 63)
        self.epoch = 0
        self._pass = 0

    def set_epoch(self, epoch: int) -> None:
        """Accept the trainer's per-epoch call (the ``DistributedSampler`` API).

        ``trainer.py`` calls ``train_loader.sampler.set_epoch(epoch)`` whenever ``RANK != -1``, i.e. in
        DDP. The grouped loader only REPLACES the sampler on the single-process path (``rank == -1`` in
        ``patch_build_dataloader``), so this is unreachable today -- but a future DDP + grouped setup
        would otherwise die with ``AttributeError`` inside the epoch loop. It is not a no-op: the epoch
        feeds the order seed (see ``__iter__``), so a caller that does use it gets a reproducible
        order per epoch, which is exactly what the DDP call exists for.
        """
        self.epoch = int(epoch)

    @classmethod
    def from_dataset(cls, dataset, seed: int = 0) -> GroupedImageSampler | None:
        """Build a sampler for ``dataset``, or return ``None`` when grouping cannot pay off."""
        if not bool(getattr(dataset, "slice_grouped_sampler", _online_default("slice_grouped_sampler"))):
            return None
        build_units = getattr(dataset, "grouped_sample_units", None)
        units = build_units() if callable(build_units) else None
        return cls(units, seed=seed) if units else None

    def __len__(self) -> int:
        """Total pool size; always equal to ``len(dataset)``."""
        return self._total

    def __iter__(self) -> Iterator[int]:
        """Yield every pool index exactly once, grouped so the raw LRU can absorb the re-reads.

        The order is a pure function of ``(seed, epoch, pass)``: ``InfiniteDataLoader``'s
        ``_RepeatSampler`` re-enters this method once per epoch, and the pass counter folded into the
        seed keeps the visit order rotating for a caller that never calls ``set_epoch`` (the
        single-process path), while the epoch term keeps it reproducible for one that does.
        """
        self._pass += 1
        gen = torch.Generator()
        gen.manual_seed((self._seed + 1_000_003 * self.epoch + self._pass) % (1 << 63))
        for unit in (self.units[i] for i in torch.randperm(len(self.units), generator=gen).tolist()):
            blocks = unit
            if len(blocks) > 1:
                blocks = [blocks[j] for j in torch.randperm(len(blocks), generator=gen).tolist()]
            for k in range(max(len(block) for block in blocks)):
                for block in blocks:
                    if k < len(block):
                        yield block[k]


def patch_build_dataloader() -> None:
    """Redirect stock ``build_dataloader`` to use GroupedImageSampler on single-machine training.

    When the dataset can be grouped (``slice_grouped_sampler`` on and the pool is worth grouping), we
    build the loader ourselves with the grouped sampler in place of the built-in RandomSampler; every
    other path (val, multi-GPU, grouping off) falls straight through to the stock implementation.
    """
    import ultralytics.data.build as _b

    if getattr(_b, "_ooo_sampler_patched", False):
        return
    _orig = _b.build_dataloader

    def patched(dataset, batch, workers, shuffle=True, rank=-1, drop_last=False, pin_memory=True, device="cuda"):
        grouped = None
        if rank == -1 and shuffle:
            # EXACTLY the stock expression (build.py). It matters: seed_worker() seeds each worker's
            # numpy/random from torch.initial_seed(), which the LOADER generator built below supplies.
            # The old code omitted the `+ seed` term, so the generator seed was a run-independent
            # constant and args.seed no longer changed augmentation randomness at all.
            seed = torch.initial_seed() - int(_b.RANK) - 1
            grouped = GroupedImageSampler.from_dataset(dataset, seed=seed)
        if grouped is None:
            return _orig(dataset, batch, workers, shuffle, rank, drop_last, pin_memory, device)

        # Grouped sampler active: mirror the stock build_dataloader tail but pass sampler=grouped.
        # NOTE -- this tail deliberately duplicates stock build.py because InfiniteDataLoader wraps
        # batch_sampler and builds its iterator eagerly at construction, so the sampler cannot simply be
        # swapped in afterwards. Keep it in step with upstream: tests/test_ooo_sampler.py asserts every
        # derived value (nw / prefetch_factor / pin_memory / drop_last / generator seed / worker_init_fn)
        # against the stock formula, so upstream drift fails the suite instead of silently changing the
        # worker count, memory pinning or the augmentation seeding.
        dataset_len = len(dataset)
        batch = min(batch, dataset_len)
        samples = len(grouped)
        drop_last = drop_last and bool(batch) and dataset_len % batch != 0
        batches = (samples // batch if drop_last else math.ceil(samples / batch)) if batch else 0
        device_type = getattr(device, "type", str(device).split(":")[0])
        nd = _b.get_torch_device_backend(device).device_count() if device_type not in {"cpu", "mps"} else 0
        nw = min(os.cpu_count() // max(nd, 1), workers, 0 if batches <= 1 else batches)
        generator = torch.Generator()
        generator.manual_seed((6148914691236517205 + int(_b.RANK) + seed) % (1 << 64))
        pin_memory = nd > 0 and pin_memory
        # npu/xpu pinned-memory device selection: was silently dropped before, so the grouped path
        # ignored args.pin_memory_device-adjacent behaviour that the stock path applies.
        pin_memory_device = (
            device_type
            if pin_memory
            and device_type in {"npu", "xpu"}
            and getattr(_b, "TORCH_1_13", True)
            and not getattr(_b, "TORCH_2_7", False)
            else None
        )
        return _b.InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=False,  # sampler is supplied; stock requires shuffle=False when sampler is set
            num_workers=nw,
            sampler=grouped,
            prefetch_factor=4 if nw > 0 else None,
            pin_memory=pin_memory,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=_b.seed_worker,
            generator=generator,
            drop_last=drop_last,
            **({"pin_memory_device": pin_memory_device} if pin_memory_device else {}),
        )

    _b.build_dataloader = patched
    _b._ooo_sampler_patched = True
