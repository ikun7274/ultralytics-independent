"""Framework-free core kernels (numpy / cv2 only, zero Ultralytics dependency)."""

from .degrade import (
    _RATIO_PAD_COLORS,
    _WEATHER_TYPES,
    _OCCLUSION_TYPES,
    _apply_motion_blur,
    _apply_weather,
    _apply_occlusion,
    _union_area,
    _ratio_pad_params,
    _cap_long_side,
)
from .geometry import slice_geometry, compute_slice_bias
from .saver import _ensure_dir, _forget_dir, _imwrite

# NOTE: `_RATIO_PAD_COLORS` is part of the public surface -- pool/dataset.py imports it from
# `ultralytics_ooo.core` for the ratio-pad border colours -- so it is listed in __all__ too.
# `_save_cap` used to be re-exported from .saver; it now has a single home in pool/dataset.py (its only
# caller), because that version needs the package default table, which core/ must not depend on.
__all__ = [
    "_RATIO_PAD_COLORS",
    "_WEATHER_TYPES",
    "_OCCLUSION_TYPES",
    "_apply_motion_blur",
    "_apply_weather",
    "_apply_occlusion",
    "_union_area",
    "_ratio_pad_params",
    "_cap_long_side",
    "slice_geometry",
    "compute_slice_bias",
    "_ensure_dir",
    "_forget_dir",
    "_imwrite",
]
