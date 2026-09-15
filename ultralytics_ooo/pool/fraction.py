"""fraction guard: survive ``fraction`` rounding to zero on a tiny dataset.

Own module for the same reason ``resume.py`` / ``valslice.py`` / ``dual.py`` / ``sampler.py`` each own
their patch: every runtime patch gets one file, one installer, one ``_ooo_*`` idempotency flag. This
guard used to live at the bottom of ``pool/dataset.py``, mixing a monkey-patch installer in with the
dataset class.
"""

from __future__ import annotations


def patch_fraction_guard() -> None:
    """Survive fraction rounding to zero on a tiny dataset (idempotent).

    Stock ``get_img_files`` does ``im_files[:round(len * fraction)]``; on a tiny set that rounds to 0 and
    leaves an empty dataset, which only surfaces later as an ``IndexError`` deep in the loader. When the
    stock result is empty and ``fraction < 1``, re-run the lookup with fraction forced to 1.0 so at
    least the source image(s) are retained.
    """
    from ultralytics.data.base import BaseDataset
    from ultralytics.utils import LOGGER

    if getattr(BaseDataset, "_ooo_fraction_guard", False):
        return
    _orig = BaseDataset.get_img_files

    def get_img_files(self, img_path):
        files = _orig(self, img_path)
        if self.fraction < 1 and len(files) == 0:
            old = self.fraction
            self.fraction = 1.0
            try:
                files = _orig(self, img_path)
            finally:
                self.fraction = old
            LOGGER.warning(
                f"ooo fraction guard: fraction={old} selected 0 images (rounded to zero); "
                f"retained {len(files)} source image(s)."
            )
        return files

    BaseDataset.get_img_files = get_img_files
    BaseDataset._ooo_fraction_guard = True
