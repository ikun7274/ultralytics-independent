"""Regression: the grouped-sampler patch (pool/sampler.py).

Pins the two things the patch used to get wrong by copying stock ``build_dataloader``:

1. The loader generator seed must keep the stock ``+ seed`` term. ``seed_worker`` derives every
   worker's numpy/random seed from ``torch.initial_seed()``, which the loader generator supplies, so
   dropping the term froze the worker RNGs to a run-independent constant -- ``args.seed`` stopped
   changing the augmentation randomness entirely on the grouped path.
2. The rest of the copied loader tail (nw / prefetch_factor / pin_memory / drop_last / worker_init_fn)
   must stay equal to the stock formula. Asserting them here turns future upstream drift into a test
   failure instead of a silent change in worker count, memory pinning or seeding.

Run: python -m pytest tests/test_ooo_sampler.py -q
"""
from __future__ import annotations

import math
import os

import torch


class _GroupedDataset(torch.utils.data.Dataset):
    """Minimal stand-in: 4 units of 4 images, one pool index per image (total == __len__)."""

    def __init__(self, n_units: int = 4, groupable: bool = True):
        self._n_units = n_units
        self.slice_grouped_sampler = groupable
        self.collate_fn = None

    def __len__(self) -> int:
        return self._n_units * 4

    def grouped_sample_units(self):
        return [[[u * 4 + j] for j in range(4)] for u in range(self._n_units)]

    def __getitem__(self, i):
        return i


def _units(n_units=4):
    return [[[u * 4 + j] for j in range(4)] for u in range(n_units)]


def test_sampler_seed_changes_the_visit_order():
    from ultralytics_ooo.pool.sampler import GroupedImageSampler

    a = list(GroupedImageSampler(_units(), seed=1))
    b = list(GroupedImageSampler(_units(), seed=2))
    assert sorted(a) == sorted(b) == list(range(16)), "every index exactly once"
    assert a != b, "the sampler must actually honour its seed"


def test_sampler_len_matches_the_pool():
    from ultralytics_ooo.pool.sampler import GroupedImageSampler

    s = GroupedImageSampler(_units(3), seed=0)
    assert len(s) == 12


def test_grouped_loader_keeps_the_stock_generator_seed():
    """The M1 regression: the generator seed must depend on torch.initial_seed() (i.e. on args.seed)."""
    import ultralytics.data.build as _b
    from ultralytics_ooo.pool.sampler import patch_build_dataloader

    patch_build_dataloader()
    ds = _GroupedDataset()
    for run_seed in (1234, 9999):
        torch.manual_seed(run_seed)
        expected = (6148914691236517205 + int(_b.RANK) + (torch.initial_seed() - int(_b.RANK) - 1)) % (1 << 64)
        loader = _b.build_dataloader(ds, batch=2, workers=0, shuffle=True, rank=-1, device="cpu")
        assert loader.generator.initial_seed() == expected, f"seed {run_seed} did not reach the loader generator"


def test_grouped_loader_attributes_match_the_stock_formula():
    """Drift guard for the copied loader tail (M2)."""
    import ultralytics.data.build as _b
    from ultralytics_ooo.pool.sampler import patch_build_dataloader

    patch_build_dataloader()
    ds = _GroupedDataset()
    batch, workers = 2, 4
    loader = _b.build_dataloader(ds, batch=batch, workers=workers, shuffle=True, rank=-1,
                                 drop_last=False, pin_memory=True, device="cpu")

    dataset_len = len(ds)
    nd = 0  # cpu
    batches = math.ceil(dataset_len / batch)
    expected_nw = min(os.cpu_count() // max(nd, 1), workers, 0 if batches <= 1 else batches)
    assert loader.num_workers == expected_nw
    assert loader.prefetch_factor == (4 if expected_nw > 0 else None)
    assert loader.pin_memory == bool(nd > 0 and True)
    assert loader.drop_last is False
    assert loader.worker_init_fn is _b.seed_worker
    assert loader.batch_size == batch
    # an explicit sampler is used, so the stock path's RandomSampler must NOT be (DataLoader drops
    # `shuffle` from its namespace once a sampler is supplied, so it cannot be asserted directly)
    assert loader.sampler is not None
    assert not isinstance(loader.sampler, torch.utils.data.RandomSampler)
    assert len(loader.sampler) == dataset_len


def test_grouping_off_falls_through_to_the_stock_loader():
    import ultralytics.data.build as _b
    from ultralytics_ooo.pool.sampler import patch_build_dataloader

    patch_build_dataloader()
    ds = _GroupedDataset(groupable=False)
    loader = _b.build_dataloader(ds, batch=2, workers=0, shuffle=True, rank=-1, device="cpu")
    assert isinstance(loader.sampler, torch.utils.data.RandomSampler)


def test_val_loader_is_untouched():
    """shuffle=False (validation) must never take the grouped path."""
    import ultralytics.data.build as _b
    from ultralytics_ooo.pool.sampler import patch_build_dataloader

    patch_build_dataloader()
    ds = _GroupedDataset()
    loader = _b.build_dataloader(ds, batch=2, workers=0, shuffle=False, rank=-1, device="cpu")
    assert loader.sampler is not None and len(loader.sampler) == len(ds)


def test_patch_is_idempotent():
    import ultralytics.data.build as _b
    from ultralytics_ooo.pool.sampler import patch_build_dataloader

    patch_build_dataloader()
    first = _b.build_dataloader
    patch_build_dataloader()
    assert _b.build_dataloader is first
