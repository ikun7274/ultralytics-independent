"""Mixed-pool segment boundaries and the grouped image sampler."""

from __future__ import annotations

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
