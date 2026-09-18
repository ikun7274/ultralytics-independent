"""Pure slicing geometry: the 2x2 overlap tile grid and the target-aware seam bias.

Shared by the training-side slicing and the validation-side SAHI evaluation so both always agree.
These functions only need NumPy + stdlib ``random`` — no image I/O, no dataset object.
"""

from __future__ import annotations

import random

import numpy as np


def slice_geometry(
    w: int,
    h: int,
    overlap_ratio: float = 0.2,
    bias_x: float = 0.0,
    bias_y: float = 0.0,
) -> list[tuple[int, int, int, int]]:
    """Return the 4 ``(x0, y0, x1, y1)`` tiles of the 2x2 overlap grid for a ``w x h`` image.

    Slice size = half the image extent scaled by ``(1 + overlap_ratio)``. ``bias_x``/``bias_y``
    (in [-0.5, 0.5]) shift the cut seam away from the centre so the caller can place it in the
    sparsest object regions. The tile count, full-image coverage and total overlap are unchanged.

    Both tiles on each axis are guaranteed NON-DEGENERATE (width/height >= 1). The seam offset used to
    be clamped only to the image extent, which is not sufficient: with ``overlap_ratio == 0`` the slice
    is exactly ``w/2``, so a full ``|bias| == 0.5`` drove ``sw - dx`` to 0 and produced a zero-width
    tile that later died in ``cv2.resize`` with ``(-215:Assertion failed) !ssize.empty()`` inside a
    DataLoader worker. Reachable through ``slice_bias_margin=0`` before that key was validated.
    """
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError(f"slice_geometry: 'overlap_ratio' must be in [0, 1), got {overlap_ratio}.")
    if not -0.5 <= bias_x <= 0.5:
        raise ValueError(f"slice_geometry: 'bias_x' must be in [-0.5, 0.5], got {bias_x}.")
    if not -0.5 <= bias_y <= 0.5:
        raise ValueError(f"slice_geometry: 'bias_y' must be in [-0.5, 0.5], got {bias_y}.")
    # Tile extent: (1 + overlap) * half the image, floored at ceil(extent / 2).
    #
    # The ceil floor is a COVERAGE guarantee, not a nicety. With floor(extent / 2) an ODD extent and a
    # small overlap can leave ``2 * sw < w``: the two tiles then cannot reach each other and a 1px strip
    # is covered by NO tile. Measured at bias 0 -- 1279x719 with overlap 0 leaves 1997 px uncovered
    # (exactly one full row + one full column), 4001x3001 leaves 7001 px, 1281x721 leaves 2001 px; at
    # the shipped overlap 0.2 the gap is gone (``2 * floor(0.6 * extent) >= extent`` for extent >= 10),
    # so only ``slice_overlap_ratio`` near 0 was affected. ``ceil`` changes nothing for even extents
    # (floor == ceil there) and only moves the degenerate odd case -- the same "touch nothing except the
    # collapsing case" rule as the non-degeneracy clamp below.
    sw = min(w, max(1, (w + 1) // 2, int((1 + overlap_ratio) * w / 2)))
    sh = min(h, max(1, (h + 1) // 2, int((1 + overlap_ratio) * h / 2)))
    # Keep both tile extents within the image (the original rule) AND >= 1 px (the non-degeneracy
    # guarantee). Intersecting the two ranges means behaviour is untouched wherever the original clamp
    # was already non-degenerate, and only the collapsing cases move.
    dx = int(round(bias_x * w))
    dy = int(round(bias_y * h))
    dx = max(max(sw - w, 1 - sw), min(min(w - sw, sw - 1), dx))
    dy = max(max(sh - h, 1 - sh), min(min(h - sh, sh - 1), dy))
    tw1, tw2 = sw + dx, sw - dx
    th1, th2 = sh + dy, sh - dy
    return [
        (0, 0, min(tw1, w), min(th1, h)),
        (0, h - th2, min(tw1, w), h),
        (w - tw2, 0, w, min(th1, h)),
        (w - tw2, h - th2, w, h),
    ]


def compute_slice_bias(
    w: int,
    h: int,
    xyxy: np.ndarray | None,
    margin: float = 0.25,
    jitter: float = 0.05,
    jitter_rng: random.Random | None = None,
) -> tuple[float, float]:
    """Target-aware seam bias for ``slice_geometry``. Projects box centers onto the x/y axes and picks
    the seam position inside ``[margin, 1-margin]`` with the fewest centers in its window. Returns
    ``(bias_x, bias_y)`` in [-0.5, 0.5]. Empty boxes -> (0, 0).

    Pass a deterministic ``jitter_rng`` derived from ``(epoch, image index)`` when all 4 tiles of one
    original must share one stable grid; ``None`` uses the global ``random`` stream.
    """
    if xyxy is None or len(xyxy) == 0:
        return 0.0, 0.0
    b = np.asarray(xyxy, dtype=np.float64)
    cx = (b[:, 0] + b[:, 2]) / 2.0
    cy = (b[:, 1] + b[:, 3]) / 2.0
    win_x = max(np.median(b[:, 2] - b[:, 0]), w * 0.05)
    win_y = max(np.median(b[:, 3] - b[:, 1]), h * 0.05)
    cands = np.linspace(margin, 1.0 - margin, 32)
    score_x = (np.abs(cx[:, None] - (cands * w)[None, :]) < (win_x / 2)).sum(axis=0)
    score_y = (np.abs(cy[:, None] - (cands * h)[None, :]) < (win_y / 2)).sum(axis=0)
    bx = float(cands[int(np.argmin(score_x))])
    by = float(cands[int(np.argmin(score_y))])
    if jitter > 0:
        rng = random if jitter_rng is None else jitter_rng
        bx += rng.uniform(-jitter, jitter)
        by += rng.uniform(-jitter, jitter)
        bx = min(max(bx, margin), 1.0 - margin)
        by = min(max(by, margin), 1.0 - margin)
    return bx - 0.5, by - 0.5
