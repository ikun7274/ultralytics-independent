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
        self._generator = torch.Generator()
        self._generator.manual_seed(int(seed) % (1 << 63))

    @classmethod
    def from_dataset(cls, dataset, seed: int = 0) -> "GroupedImageSampler | None":
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
        """Yield every pool index exactly once, grouped so the raw LRU can absorb the re-reads."""
        for unit in (self.units[i] for i in torch.randperm(len(self.units), generator=self._generator).tolist()):
            blocks = unit
            if len(blocks) > 1:
                blocks = [blocks[j] for j in torch.randperm(len(blocks), generator=self._generator).tolist()]
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
            grouped = GroupedImageSampler.from_dataset(dataset)
        if grouped is None:
            return _orig(dataset, batch, workers, shuffle, rank, drop_last, pin_memory, device)

        # Grouped sampler active: mirror the stock build_dataloader tail but pass sampler=grouped.
        dataset_len = len(dataset)
        batch = min(batch, dataset_len)
        samples = len(grouped)
        drop_last = drop_last and bool(batch) and dataset_len % batch != 0
        batches = (samples // batch if drop_last else math.ceil(samples / batch)) if batch else 0
        device_type = getattr(device, "type", str(device).split(":")[0])
        nd = _b.get_torch_device_backend(device).device_count() if device_type not in {"cpu", "mps"} else 0
        nw = min(os.cpu_count() // max(nd, 1), workers, 0 if batches <= 1 else batches)
        generator = torch.Generator()
        generator.manual_seed((6148914691236517205 + int(_b.RANK)) % (1 << 64))
        pin_memory = nd > 0 and pin_memory
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
        )

    _b.build_dataloader = patched
    _b._ooo_sampler_patched = True
