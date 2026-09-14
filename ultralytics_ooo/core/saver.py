"""Save-path helpers shared by the online-augmentation branches.

Framework-free: directory / write handling with no Ultralytics dataset state. The original used
``ultralytics.utils.LOGGER`` and the patched ``imwrite``; this port uses stdlib ``logging`` and a
unicode-safe writer built on ``cv2.imencode`` + ``ndarray.tofile`` (OpenCV's own ``cv2.imwrite``
silently fails on non-ASCII paths).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger("ultralytics_ooo")

# Directories already created in this process.
_MKDIR_DONE: set[str] = set()


def _ensure_dir(path) -> None:
    """Create ``path`` (with parents) once per process; later calls are a no-op."""
    key = str(path)
    if key in _MKDIR_DONE:
        return
    Path(path).mkdir(parents=True, exist_ok=True)
    _MKDIR_DONE.add(key)


def _forget_dir(path) -> None:
    """Drop ``path`` from the per-process "already created" set so the next write retries ``mkdir``."""
    _MKDIR_DONE.discard(str(path))


def _imwrite(path, img) -> bool:
    """Write an image unicode-safely (``imencode`` + ``tofile``) and warn when it fails. Never ignore
    the return value."""
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        LOGGER.warning("%s save failed: imencode returned False for '%s'.", getattr(img, "shape", "?"), path)
        return False
    try:
        buf.astype(np.uint8).tofile(str(path))
    except OSError as e:
        LOGGER.warning("%s save failed: %s for '%s'.", getattr(img, "shape", "?"), e, path)
        _forget_dir(Path(path).parent)
        return False
    return True


def _save_cap(dataset, branch: str) -> int:
    """Return the per-branch save cap. ``slice_save_max_<branch>`` overrides the global ``slice_save_max``;
    a missing or ``None`` attribute falls back to ``slice_save_max``. ``0`` means unlimited."""
    val = getattr(dataset, f"slice_save_max_{branch}", None)
    if val is None:
        val = getattr(dataset, "slice_save_max", 0) or 0
    return int(val)
