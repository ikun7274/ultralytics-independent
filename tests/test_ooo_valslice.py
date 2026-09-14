"""Regression: SliceValDataset tile expansion + remap metadata (no real data/weights).

Run:
    python -m pytest tests/test_ooo_valslice.py -q
"""
from __future__ import annotations

import numpy as np


def _fake_base(n=4, w=800, h=600):
    """A minimal stand-in for the val YOLODataset: labels + a cached image reader."""
    rng = np.random.default_rng(0)
    labels = []
    for i in range(n):
        labels.append(
            {
                "im_file": f"img{i}.jpg",
                "shape": (h, w, 3),
                "bboxes": np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
                "cls": np.array([[0]], dtype=np.float32),
            }
        )
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)

    class Base:
        prefix = "val: "
        training = False
        device = "cpu"
        transforms = lambda self, x: x
        collate_fn = staticmethod(lambda b: b)

        def __init__(self, labels, img):
            self.labels = labels
            self._img = img

        def _load_image_cached(self, i):
            return self._img

        def update_labels_info(self, label):
            return label

    return Base(labels, img)


def test_slicedataset_expands_tiles():
    from ultralytics_ooo.pool.valslice import SliceValDataset

    base = _fake_base(n=4, w=800, h=600)
    ds = SliceValDataset(base, overlap_ratio=0.2, all_tiles=True, ratio=1.0)
    # 800x600 with 0.2 overlap -> 2x2 tiles (each image fully sliced)
    assert len(ds) > 4  # expanded beyond whole images
    assert ds._cum[-1] == len(ds)


def test_slicedataset_meta_and_passthrough():
    from ultralytics_ooo.pool.valslice import SliceValDataset

    base = _fake_base(n=3, w=400, h=300)
    ds = SliceValDataset(base, overlap_ratio=0.0, all_tiles=False, ratio=1.0)
    # all_tiles=False -> 1 random tile per image
    assert len(ds) == 3
    sample = ds[0]
    m = sample["val_slice_meta"]
    assert set(m) >= {"orig_idx", "offset", "tile_shape", "orig_shape", "n_tiles", "sliced"}
    assert m["orig_shape"] == (300, 400)


def test_ratio_subset_partial_passthrough():
    from ultralytics_ooo.pool.valslice import SliceValDataset

    base = _fake_base(n=10, w=800, h=600)
    ds = SliceValDataset(base, overlap_ratio=0.2, all_tiles=True, ratio=0.5)
    # ~5 images sliced (each -> several tiles), ~5 pass through as 1 whole image each
    assert len(ds) > 5


def test_patch_validator_idempotent():
    from ultralytics_ooo.pool.valslice import patch_validator

    class FakeV:
        def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
            self.args = args

        def update_metrics(self, preds, batch):
            pass

        def finalize_metrics(self):
            pass

        def gather_stats(self):
            pass

        def get_dataloader(self, dataset_path, batch_size=1):
            pass

    patch_validator(FakeV)
    first = FakeV.update_metrics
    patch_validator(FakeV)
    assert FakeV.update_metrics is first
    assert hasattr(FakeV, "_val_slice_active")
