"""Measure the pool's shape: what each branch owns, and what the ratio costs/keeps.

Two things this reports, both from the LIVE dataset object rather than from arithmetic:

  * the real per-segment slot counts and how many SOURCE IMAGES each one covers;
  * the per-source slot histogram -- with ratio-sized segments every image appears exactly once as a
    plain original, and only the selected ones own extra slots, so the histogram says directly whether
    un-augmented duplicates are back;
  * the LEGACY layout for the same config, for comparison (the pre-refactor widths were ``n_per*N``,
    ``2N``, ``N``, ``ceil(N/4)``, ``N``, ``N`` regardless of the ratio).

Run from the repo root:  python tools/ooo_pool_measure.py --n 40 --ratio 0.1
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# `python tools/<name>.py` puts `tools/` -- not the repo root -- on sys.path, and this project is not
# necessarily pip-installed, so the lazy `from ultralytics.cfg import get_cfg` inside build() would fail
# on a clean checkout. Put the root first so the invocation in this docstring actually works.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SEG_ORDER = ["base", "origin", "ratio", "blur", "compose", "weather", "occlusion"]
RATIO_KEYS = ("slice_ratio", "ratio_pad_ratio", "blur_ratio", "compose_ratio", "weather_ratio",
              "occlusion_ratio")


def _synthetic(n: int, root: Path) -> Path:
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        cv2.imwrite(str(img_dir / f"i{i}.jpg"), (rng.random((96, 128, 3)) * 255).astype(np.uint8))
        (lbl_dir / f"i{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return img_dir


def build(n: int, ratios: dict, **extra):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics_ooo import install

    install()
    root = Path(tempfile.mkdtemp())
    img_dir = _synthetic(n, root)
    overrides = dict(
        task="detect", mode="train", imgsz=64, batch=4, fraction=1.0, workers=0,
        slice_prob=1.0, slice_all_tiles=True, img_origin=True,
        ratio_pad_keep=True, blur_keep=True, compose_keep=True,
        weather_keep=True, occlusion_keep=True,
        **ratios, **extra,
    )
    cfg = get_cfg(overrides=overrides)
    data = {"path": str(root), "names": {0: "obj"}, "channels": 3, "nc": 1}
    return build_yolo_dataset(cfg, str(img_dir), 4, data, mode="train")


def legacy_lengths(n: int, n_per: int = 4) -> list[int]:
    """The pre-refactor segment widths: full width for every enabled branch, ratio ignored."""
    return [n_per * n, n, n, 2 * n, (n + 3) // 4, n, n]


def per_source_histogram(ds) -> dict[str, int]:
    """How many pool slots resolve to each source image (probe: the label's ``im_file``)."""
    counts: dict[str, int] = {}
    for index in range(len(ds)):
        stem = Path(ds.get_image_and_label(index)["im_file"]).stem
        counts[stem] = counts.get(stem, 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40, help="number of synthetic images")
    ap.add_argument("--ratio", type=float, default=0.1, help="applied to all six *_ratio knobs")
    ap.add_argument("--no-histogram", action="store_true",
                    help="skip the per-source histogram (decodes every slot; slow on large N)")
    args = ap.parse_args()
    r = args.ratio
    n = args.n
    ratios = {k: r for k in RATIO_KEYS}
    ds = build(n, ratios)
    length_before = len(ds)  # before any epoch is published -- what the trainer/sampler see
    ds.set_epoch(0, 10)
    lens = ds._segment_lengths()
    n_per = ds._n_per()

    print("=" * 92)
    print(f"POOL SHAPE  (N={n} images, all six ratios = {r:g}, slice_all_tiles={bool(ds.slice_all_tiles)})")
    print("=" * 92)
    print(f"{'segment':<11}{'slots':>8}{'selection':>11}{'slots/item':>11}   {'note'}")
    spec = {attr: (ratio_attr, on, count) for attr, ratio_attr, on, count in ds._mask_specs(n)}
    total = 0
    for name in SEG_ORDER:
        slots = lens[SEG_ORDER.index(name)]
        total += slots
        if name == "origin":
            k_slice = len(ds._sel_indices("slice", n))
            print(f"{name:<11}{slots:>8}{f'{k_slice}/{n}':>11}{'1':>11}   "
                  f"keep_origin: one whole frame per SLICED image (un-selected ones already have one)")
            continue
        attr = "slice" if name == "base" else name
        _ratio_attr, on, count = spec[attr]
        sel = getattr(ds, f"_sel_{attr}", None)
        n_sel, mult = (count, n_per if attr == "slice" else 2 if attr == "blur" else 1) if sel is None \
            else (len(sel), n_per if attr == "slice" else 2 if attr == "blur" else 1)
        note = ""
        if attr == "slice":
            note = f"{n_sel} selected -> {n_per} tiles, {n - n_sel} un-selected -> 1 plain slot each"
        elif not on:
            note = "branch off"
        elif n_sel == 0:
            note = "ratio rounds to 0 selected -> the branch is effectively off"
        else:
            note = f"{n_sel} selected images x {mult}"
        print(f"{name:<11}{slots:>8}{f'{n_sel}/{count}':>11}{f'{mult}':>11}   {note}")
    print(f"{'TOTAL':<11}{total:>8}")
    print(f"  len(dataset) = {len(ds)}   (before set_epoch: {length_before} -- must be equal)")
    assert length_before == len(ds), "the layout moved when the epoch was published"

    legacy = legacy_lengths(n, n_per)
    legacy_total = sum(legacy)
    print()
    print(f"  ratio-sized layout : {lens}  total {total}")
    print(f"  pre-refactor layout: {legacy}  total {legacy_total}")
    print(f"  -> {legacy_total / total:.2f}x smaller; the ratio>=1 case reproduces the second line exactly")

    if not args.no_histogram:
        hist = per_source_histogram(ds)
        values = sorted(hist.values())
        once = sum(1 for v in values if v == 1)
        print()
        print(f"  slots per source image: min={values[0]} max={values[-1]} "
              f"avg={sum(values) / len(values):.2f} distinct={len(values)}/{n}")
        print(f"  sources owning exactly ONE slot (i.e. plain originals only): {once}/{n}")
        print("  (the pre-refactor layout gave every image n_per extra slots in the base segment even when")
        print("   slicing did not select it -- 4 byte-identical copies at K=0. Now an un-selected image owns")
        print("   one base slot and nothing else, so the only repetition left is genuine repetition: the")
        print("   selected images' tiles / tiers, which carry different pixels.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
