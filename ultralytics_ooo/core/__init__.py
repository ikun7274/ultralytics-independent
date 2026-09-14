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
from .saver import _ensure_dir, _forget_dir, _imwrite, _save_cap

__all__ = [
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
    "_save_cap",
]
