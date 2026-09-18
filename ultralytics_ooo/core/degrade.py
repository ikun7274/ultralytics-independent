"""In-memory image-degradation primitives.

Framework-free numeric kernels: aspect-ratio padding, motion blur, weather degradation
(rain / haze / noise), semantic occlusion drawing, exact rectangle-union area, and a
long-side downscale helper. Every function is a pure transform over a NumPy image (and,
for geometry, over plain boxes) — no Ultralytics dataset, label or config object is touched.

These are the in-memory ports of the offline ``pytools/`` scripts. They run inside DataLoader
workers, so each kernel keeps its transient allocation low on purpose.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

# --- Aspect-ratio pad helpers -----------------------------------------------
_RATIO_REL_TOL = 0.02
_RATIO_43 = 4.0 / 3.0
_RATIO_169 = 16.0 / 9.0
_RATIO_PAD_COLORS = {"black": (0, 0, 0), "gray": (114, 114, 114), "white": (255, 255, 255)}

# Legal weather / occlusion types (validated up-front against the config by the caller).
_WEATHER_TYPES: frozenset[str] = frozenset({"rain", "haze", "noise"})
_OCCLUSION_TYPES: frozenset[str] = frozenset({"rect", "stripe"})


def _channels(img: np.ndarray) -> int:
    """Return the channel count of an image, treating 2-D arrays as single-channel."""
    return int(img.shape[2]) if img.ndim == 3 else 1


def _ratio_pad_params(w: int, h: int, target_ratio: str, auto: bool) -> tuple[int, int, int, int] | None:
    """Compute border-padding to reach a target aspect ratio.

    - ratio < target (portrait-ish): keep height, widen with left/right symmetric borders
    - ratio > target (landscape-ish): keep width, heighten with top/bottom symmetric borders
    - already close to the target: return None (no pad needed -> copy / use original)
    - auto: 4:3 <-> 16:9 bidirectional; any other ratio goes to its NEAREST of 4:3 or 16:9
    Returns (new_w, new_h, pad_left, pad_top) or None.
    """
    ratio = w / h
    if auto:
        if math.isclose(ratio, _RATIO_43, rel_tol=_RATIO_REL_TOL):
            target = _RATIO_169  # 4:3 -> 16:9
        elif math.isclose(ratio, _RATIO_169, rel_tol=_RATIO_REL_TOL):
            target = _RATIO_43  # 16:9 -> 4:3
        else:
            target = _RATIO_43 if abs(ratio - _RATIO_43) <= abs(ratio - _RATIO_169) else _RATIO_169
    else:
        target = _RATIO_43 if target_ratio == "4:3" else _RATIO_169
        if math.isclose(ratio, target, rel_tol=_RATIO_REL_TOL):
            return None
    if ratio < target:  # widen
        new_w = max(1, int(round(h * target)))
        new_h = h
        pad_left = (new_w - w) // 2
        pad_top = 0
    else:  # heighten
        new_w = w
        new_h = max(1, int(round(w / target)))
        pad_left = 0
        pad_top = (new_h - h) // 2
    return new_w, new_h, pad_left, pad_top


def _cap_long_side(im: np.ndarray, cap: float, interp: int = cv2.INTER_AREA) -> tuple[np.ndarray, float]:
    """Downscale ``im`` so its long side is at most ``cap`` pixels, with an explicit ``interp`` kernel.

    ``cap <= 0`` disables the cap. Returns ``(img, scale)`` where ``scale`` is the factor actually
    applied, so pixel-typed parameters (PSF length, defocus sigma, rain-line length) can be scaled
    with it. When no downscale is needed the SAME object is returned (``scale == 1.0``).
    """
    if cap <= 0:
        return im, 1.0
    h, w = im.shape[:2]
    longest = max(h, w)
    if longest <= cap:
        return im, 1.0
    s = cap / longest
    return cv2.resize(im, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp), s


# --- Motion blur -------------------------------------------------------------
def _psf_size(length: float) -> int:
    """Odd PSF side length that holds a motion-blur segment of ``length`` pixels."""
    return max(3, math.ceil(length) | 1)


def _motion_blur_kernel(length: float, angle: float) -> np.ndarray:
    """Build a line-segment PSF motion-blur kernel. Kernel is squared with odd size (>=3); the
    segment is anti-aliased and normalized to sum = 1 (brightness unchanged after convolution)."""
    rad = np.deg2rad(angle)
    size = _psf_size(length)
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    dx = np.cos(rad)
    dy = np.sin(rad)
    length_scaled = length / 2.0
    x1 = center - dx * length_scaled
    y1 = center - dy * length_scaled
    x2 = center + dx * length_scaled
    y2 = center + dy * length_scaled
    cv2.line(kernel, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
             1.0, thickness=1, lineType=cv2.LINE_AA)
    ksum = float(kernel.sum())
    if not ksum > 0:
        kernel[:] = 0.0
        kernel[size // 2, size // 2] = 1.0  # degenerate to an identity (no-op) kernel
    else:
        kernel /= ksum
    return kernel


def _crop_kernel(kernel: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Trim the all-zero border off a convolution kernel, returning ``(kernel, anchor)``. Bit-identical
    for a zero tap; the explicit ``anchor`` keeps the kernel centred on the original centre."""
    size = kernel.shape[0]
    c = size // 2
    nz = np.nonzero(kernel)
    y0, y1 = min(int(nz[0].min()), c), max(int(nz[0].max()), c)
    x0, x1 = min(int(nz[1].min()), c), max(int(nz[1].max()), c)
    cropped = np.ascontiguousarray(kernel[y0:y1 + 1, x0:x1 + 1])
    return cropped, (c - x0, c - y0)


def _match_ndim(src: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Re-attach the trailing channel axis OpenCV drops for single-channel input.

    ``cv2.blur`` / ``cv2.filter2D`` / ``cv2.convertScaleAbs`` all return a 2-D array when handed an
    ``(H, W, 1)`` array, so these kernels used to change the shape contract for grayscale frames
    (measured: motion blur, weather/rain and weather/haze turned ``(60, 80, 1)`` into ``(60, 80)``).
    The dataset path happens to recover -- ``_finalize_label`` does ``img[..., None]`` when ``ndim == 2``
    -- and the branch images are all produced by ``_load_image_cached_ex``, which guarantees at least
    3 dims. But a kernel that silently drops an axis is a trap for every other caller, so the contract
    "shape in == shape out" is restored here rather than left to the consumer.
    """
    if src.ndim == 3 and out.ndim == 2:
        return out[..., None]
    return out


def _apply_motion_blur(img: np.ndarray, length: float = 15.0, angle: float = 30.0,
                       defocus_sigma: float = 0.0, axis_aligned: bool = False) -> np.ndarray:
    """Apply motion blur (and optional defocus) to a BGR/grayscale image.

    ``axis_aligned=True`` restricts the smear to the image axes (angle 0=horizontal, 90=vertical),
    where the PSF is exactly a uniform box and ``cv2.blur`` (O(1) running sum) gives a bit-identical,
    faster result. Otherwise a dense anti-aliased line PSF is built and convolved via ``filter2D``.
    """
    if axis_aligned:
        n = _psf_size(length)
        blurred = cv2.blur(img, (1, n) if float(angle) % 180.0 >= 45.0 else (n, 1))
    else:
        kernel, anchor = _crop_kernel(_motion_blur_kernel(length, angle))
        blurred = cv2.filter2D(img, -1, kernel, anchor=anchor)
    if defocus_sigma > 0:
        ksize = int(6 * defocus_sigma) | 1  # odd kernel size
        blurred = cv2.GaussianBlur(blurred, (ksize, ksize), defocus_sigma)
    return _match_ndim(img, blurred)


# --- Weather degradation -----------------------------------------------------
def _apply_weather(img: np.ndarray, weather_type: str, rain_density: float = 0.15, rain_length: float = 15.0,
                   haze_beta: float = 0.4, noise_std: float = 15.0) -> np.ndarray:
    """Apply one weather degradation (rain / haze / Gaussian noise) to a BGR image. Labels are UNCHANGED.

    - rain: ~density*max(h,w) semi-transparent streaks at a fixed 20-degree slant; alpha 0.2-0.6.
    - haze: atmospheric-scattering model I = J*t + A*(1-t), t = 1-beta, A = gray atmosphere (200).
    - noise: additive Gaussian noise (RGB independent).
    """
    if weather_type == "rain":
        h, w = img.shape[:2]
        n = max(1, int(rain_density * max(h, w)))
        ang = math.radians(20.0)
        dx, dy = math.sin(ang), math.cos(ang)
        x0s = np.random.uniform(0.0, float(w), size=n)
        y0s = np.random.uniform(0.0, float(h), size=n)
        lens = rain_length * np.random.uniform(0.5, 1.0, size=n)
        x1s = np.clip(x0s - dx * lens, 0.0, float(w))
        y1s = np.clip(y0s - dy * lens, 0.0, float(h))
        pts = np.stack([np.stack([x0s, y0s], axis=1), np.stack([x1s, y1s], axis=1)], axis=1).astype(np.int32)
        thick2 = np.random.rand(n) < 0.5
        overlay = img.copy()
        for t in (1, 2):
            batch = pts[thick2] if t == 2 else pts[~thick2]
            if len(batch):
                cv2.polylines(overlay, [p for p in batch], isClosed=False, color=(205, 205, 225),
                              thickness=t, lineType=cv2.LINE_AA)
        alpha = np.random.uniform(0.2, 0.6)
        if overlay.ndim == 3 and overlay.shape[2] == 3:
            # Blend straight back into the overlay buffer: same pixels, one full-frame allocation less
            # (measured 5.77 -> 4.68 ms at 1280x960, np.array_equal True). Restricted to 3-channel
            # frames because OpenCV maps an (H, W, 1) input to a 1-CHANNEL Mat: there the blend comes
            # back 2-D and _match_ndim has to re-attach the axis, so handing it a dst= buffer is unsafe.
            cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0.0, dst=overlay)
            return overlay
        return _match_ndim(img, cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0))
    if weather_type == "haze":
        t = max(0.0, min(1.0, 1.0 - haze_beta))
        return _match_ndim(img, cv2.convertScaleAbs(img, alpha=t, beta=200.0 * (1.0 - t)))
    # noise: cv2.randn into ONE scratch buffer; handed a single-channel view so sigma is not scaled
    # by 1/sqrt(3); one integer pulled from the numpy stream seeds OpenCV's RNG to keep lockstep.
    #
    # The scratch is int16, not float32: the result is quantised straight back to uint8, so the
    # float32 frame only bought 2x the write traffic for the RNG and two extra full-frame
    # allocations (the float buffer plus the float->uint8 result). Measured on this project's own
    # hardware via _perf_review/noisebench.py, medians of 7: 1280x720 29.59 -> 25.31 ms (-14%),
    # 1280x960 36.43 -> 34.31 ms, and the realised sigma is unchanged (14.98 for both at sigma=15).
    # The same file rejected the obvious numpy alternatives: Generator.normal into int16 measured
    # 47.90 (single-channel broadcast) / 78.46 (per-channel) ms and into a float32 out 50.56 ms,
    # i.e. 1.6-2.7x SLOWER than the shipped kernel -- do not "simplify" this into numpy.
    sigma = max(0.0, float(noise_std))
    cv2.setRNGSeed(int(np.random.randint(0, 2**31 - 1)))
    if img.ndim != 3:  # single-channel: the int16 path below is only worth it for 3-channel frames
        out_f = np.empty(img.shape, dtype=np.float32)
        if sigma > 0:
            cv2.randn(out_f.reshape(out_f.shape[0], -1), 0.0, sigma)
        else:
            out_f.fill(0.0)
        out_f += img
        np.clip(out_f, 0, 255, out=out_f)
        return out_f.astype(np.uint8)
    scratch = np.empty(img.shape, dtype=np.int16)
    if sigma > 0:
        cv2.randn(scratch.reshape(scratch.shape[0], -1), 0.0, sigma)
        # ``dtype=cv2.CV_16S`` already converts the uint8 source to the int16 destination, so the
        # explicit ``img.astype(np.int16)`` only materialised a second full int16 frame (3.7 MB at
        # 1280x960) for the add to read back. Dropping it is bit-identical: measured 11.22 -> 7.52 ms
        # on this project's hardware, max|diff| = 0 (_perf_review/ooo6/verify6.py, part A).
        cv2.add(img, scratch, dst=scratch, dtype=cv2.CV_16S)
    else:
        scratch[...] = img
    # Saturation into CV_8U is exactly ``clip(0, 255)`` followed by ``astype(uint8)``, but as ONE pass
    # instead of two full-frame numpy traversals (measured 12.60 -> 3.64 ms, max|diff| = 0). This
    # relies on cv2.add saturating to the DST type: with an int16 input a negative excursion must
    # clamp to 0. Do NOT "simplify" this to cv2.convertScaleAbs -- that takes the ABSOLUTE value of
    # the negative excursions instead of clamping them, which is a silent pixel-level change.
    # ``_match_ndim`` is mandatory, not cosmetic: an (H, W, 1) frame reaches here too (ndim == 3, so it
    # takes the int16 branch above), OpenCV maps it to a 1-CHANNEL Mat and hands back (H, W) --
    # dropping the axis the old ``scratch.astype`` preserved. Same trap the other kernels document.
    return _match_ndim(img, cv2.add(scratch, 0, dtype=cv2.CV_8U))


# --- Occlusion ---------------------------------------------------------------
def _union_area(rects: list[tuple[int, int, int, int]]) -> float:
    """Exact area of the union of axis-aligned integer rectangles (x-sweep + y-interval merge)."""
    if not rects:
        return 0.0
    xs = sorted({x0 for x0, _, _, _ in rects} | {x1 for _, _, x1, _ in rects})
    total = 0.0
    for a, b in zip(xs, xs[1:]):
        if b <= a:
            continue
        ys = sorted((y0, y1) for x0, y0, x1, y1 in rects if x0 <= a and b <= x1)
        if not ys:
            continue
        cur0, cur1 = ys[0]
        span = 0.0
        for y0, y1 in ys:
            if y0 <= cur1:
                cur1 = max(cur1, y1)
            else:
                span += cur1 - cur0
                cur0, cur1 = y0, y1
        span += cur1 - cur0  # last merged interval
        total += span * (b - a)
    return total


def _apply_occlusion(
    img: np.ndarray,
    occlusion_type: str,
    blocks: int = 1,
    size_ratio: float = 0.1,
    color: str = "auto",
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Draw semantic occlusion blocks (rect / stripe) on a BGR image. Returns the occluded image plus
    the pixel ``(x0, y0, x1, y1)`` boxes of every block so the caller can compute per-target cover ratios."""
    h, w = img.shape[:2]
    out = img.copy()
    base = max(2.0, math.sqrt(max(1.0, size_ratio) * h * w))
    if color == "auto":
        # mean of the darkest ~25% pixels (per channel), sampled on a prime stride
        flat = img.reshape(-1, _channels(img))[::37]
        q = np.quantile(flat, 0.25, axis=0)
        oc_color = tuple(int(round(float(v))) for v in q)
    elif color == "black":
        oc_color = (0, 0, 0)
    else:  # gray
        oc_color = (128, 128, 128)
    boxes = []
    for _ in range(max(1, int(blocks))):
        if occlusion_type == "stripe":
            thick = max(2, int(0.02 * min(h, w)))
            length = int(max(h, w) * random.uniform(0.5, 1.0))
            vertical = random.random() < 0.5
            if vertical:
                cx = random.uniform(0.0, float(w))
                cy = random.uniform(0.0, float(h))
                x0, x1 = max(0, int(cx - thick // 2)), min(w, int(cx + thick // 2) + 1)
                y0, y1 = max(0, int(cy - length // 2)), min(h, int(cy + length // 2) + 1)
                cv2.rectangle(out, (x0, y0), (x1, y1), oc_color, -1)
                boxes.append((x0, y0, x1, y1))
            else:
                cx = random.uniform(0.0, float(w))
                cy = random.uniform(0.0, float(h))
                x0, x1 = max(0, int(cx - length // 2)), min(w, int(cx + length // 2) + 1)
                y0, y1 = max(0, int(cy - thick // 2)), min(h, int(cy + thick // 2) + 1)
                cv2.rectangle(out, (x0, y0), (x1, y1), oc_color, -1)
                boxes.append((x0, y0, x1, y1))
        else:  # rect
            side = base * random.uniform(0.5, 1.0)
            # max(1, ...) so a tiny image cannot produce a zero-area occluder: with base = max(2, ...)
            # at the floor, ``int(side * 0.7)`` is 0 for a 4x4 frame, which drew nothing yet was still
            # appended to ``boxes`` and then contributed 0 to every coverage ratio.
            bw = max(1, int(side * random.uniform(0.7, 1.3)))
            bh = max(1, int(side * random.uniform(0.7, 1.3)))
            x0 = random.uniform(0.0, max(1.0, float(w - bw)))
            y0 = random.uniform(0.0, max(1.0, float(h - bh)))
            x0i, y0i = int(x0), int(y0)
            x1i, y1i = min(w, x0i + bw), min(h, y0i + bh)
            cv2.rectangle(out, (x0i, y0i), (x1i, y1i), oc_color, -1)
            boxes.append((x0i, y0i, x1i, y1i))
    return np.ascontiguousarray(out), boxes
