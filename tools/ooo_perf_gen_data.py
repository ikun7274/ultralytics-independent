"""Generate a photo-realistic synthetic detection dataset for performance work.

Random noise compresses terribly (a 1280x720 pure-noise JPEG is ~1.2 MB, nothing like a real
dataset), so IO numbers measured on it would be meaningless. These frames use a smooth gradient,
soft shapes and light grain, which lands at a realistic ~50-220 KB per 1280x720 JPEG.

    python tools/ooo_perf_gen_data.py <root> <n_train> <n_val> [--size WxH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# `python tools/<name>.py` puts `tools/` -- not the repo root -- on sys.path. Keep the root importable
# even when the package is not pip-installed (same convention as the other tools/ scripts).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _frame(rng: np.random.Generator, w: int, h: int):
    """One synthetic frame plus its YOLO-normalised xywh boxes."""
    gx = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    gy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    c0 = rng.uniform(40, 200, size=3).astype(np.float32)
    c1 = rng.uniform(40, 200, size=3).astype(np.float32)
    img = (c0[None, None, :] * (1 - gx) + c1[None, None, :] * gx) * (1 - 0.5 * gy) + 30 * gy
    img = np.clip(img, 0, 255).astype(np.uint8)

    boxes = []
    for _ in range(int(rng.integers(2, 7))):
        bw = float(rng.uniform(0.04, 0.20) * w)
        bh = float(rng.uniform(0.05, 0.25) * h)
        x0 = float(rng.uniform(0, max(1.0, w - bw)))
        y0 = float(rng.uniform(0, max(1.0, h - bh)))
        color = tuple(int(v) for v in rng.integers(0, 256, size=3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + bw), int(y0 + bh)), color, -1)
        else:
            cv2.ellipse(img, (int(x0 + bw / 2), int(y0 + bh / 2)),
                        (max(1, int(bw / 2)), max(1, int(bh / 2))), 0, 0, 360, color, -1)
        boxes.append(((x0 + bw / 2) / w, (y0 + bh / 2) / h, bw / w, bh / h, int(rng.integers(0, 3))))

    grain = rng.normal(0, 6.0, size=(h, w, 1)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    return img, boxes


def build(root: Path, n_train: int, n_val: int, w: int, h: int, quality: int = 88) -> dict:
    root = Path(root)
    total_bytes = 0
    for split, n in (("train", n_train), ("val", n_val)):
        idir, ldir = root / "images" / split, root / "labels" / split
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            rng = np.random.default_rng(1000 * (split == "val") + i)
            img, boxes = _frame(rng, w, h)
            p = idir / f"{split}_{i:05d}.jpg"
            cv2.imwrite(str(p), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            total_bytes += p.stat().st_size
            (ldir / f"{split}_{i:05d}.txt").write_text(
                "".join(f"{c} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n" for bx, by, bw, bh, c in boxes),
                encoding="utf-8",
            )
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: a\n  1: b\n  2: c\n",
        encoding="utf-8",
    )
    return {"root": str(root), "yaml": str(root / "data.yaml"), "n_train": n_train, "n_val": n_val,
            "size": [w, h], "total_bytes": total_bytes,
            "avg_kb": round(total_bytes / max(1, n_train + n_val) / 1024, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("n_train", type=int)
    ap.add_argument("n_val", type=int)
    ap.add_argument("--size", default="1280x720", help="WxH of every frame")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
    print(json.dumps(build(Path(a.root), a.n_train, a.n_val, w, h), indent=2))


if __name__ == "__main__":
    main()
