"""ultralytics_ooo — 在线数据增强 + 混合样本池 + 修补续训 的即插即用扩展包.

This package extracts three feature groups out of a forked Ultralytics tree and makes them
installable on top of an unmodified upstream Ultralytics:

* ``ultralytics_ooo.core``  — framework-free numeric kernels (numpy/cv2 only): slicing geometry,
  degradation operators (motion blur / weather / occlusion / aspect-ratio pad), unicode-safe savers.
* ``ultralytics_ooo.pool``  — everything that touches Ultralytics: the mixed virtual-sample pool that
  extends a YOLO dataset's index space, plus the runtime patches for augment assembly, grouped
  sampling, patch-resume, sliced validation and dual-metric validation.
* ``ultralytics_ooo.installer`` — one-call ``install()`` that wires all of the above onto a stock
  Ultralytics installation without editing upstream source.

The public entry point is :func:`ultralytics_ooo.installer.install`.
"""

__version__ = "0.1.0"

from .installer import install

__all__ = ["install"]
