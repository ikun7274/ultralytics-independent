#!/usr/bin/env python
"""Report what the four slicing knobs actually put in the pool, on YOUR dataset.

The slicing knobs are a three-layer decision chain, and two of the layers are invisible in the config:

    slice_prob            the slicing MASTER GATE, so it takes booleans: True (= 1.0) / False (= 0.0).
                          Do NOT use 0 < p < 1 to "dial down" slicing: it coin-flips PER SLOT and the
                          losing slots fall back to the WHOLE image (the tool warns about it).
    slice_all_tiles       True  -> 4 slots per selected image (the base segment is 4K + N-K wide)
                          False -> 1 random slot per selected image (base is N wide)
    slice_ratio           per epoch, K = round(x * N) images take the slicing pipeline; the rest own
                          exactly ONE whole-frame slot each.
    slice_background_ratio  what an EMPTY tile becomes: kept as an empty tile while
                          ``empty <= x * positive`` (cumulative, per process, per epoch), otherwise
                          replaced by the WHOLE image. -1 keeps every empty tile.
    img_origin           adds one whole frame per ORIGINAL image (unified coverage knob).

Run this before tuning. It classifies every base/origin slot and prints the density, so you can see
whether "most of my tiles are empty" is even true -- which is the single fact that decides the
``slice_background_ratio`` setting.

    python tools/ooo_slice_knobs.py --data _mini_val_set/_mini_data.yaml
    python tools/ooo_slice_knobs.py --data my.yaml --ratio 0.25 --bg 0.1 --img-origin
    python tools/ooo_slice_knobs.py --data my.yaml --grid        # sweep bg over -1/0.2/0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

RATIO_KEYS = ("slice_ratio",)


def _resolve(data_yaml: Path) -> tuple[Path, dict]:
    with open(data_yaml, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    root = Path(data.get("path", "."))
    if not root.is_absolute():
        for cand in (Path.cwd() / root, data_yaml.parent / root, data_yaml.parent):
            if cand.exists():
                root = cand
                break
    names = data.get("names", {0: "obj"})
    if isinstance(names, list):
        data["names"] = dict(enumerate(names))
    data["nc"] = data.get("nc", len(data["names"]))
    data["path"] = str(root)
    return root / data.get("train", "images/train"), data


def build(train_dir: Path, data: dict, args, **over):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset

    from ultralytics_ooo import install

    install()
    cfg_in = dict(task="detect", mode="train", imgsz=args.imgsz, batch=args.batch, fraction=1.0,
                  workers=0, slice_prob=args.prob, slice_all_tiles=args.all_tiles,
                  slice_ratio=args.ratio, img_origin=args.img_origin,
                  slice_background_ratio=args.bg)
    cfg_in.update(over)
    cfg = get_cfg(overrides=cfg_in)
    return build_yolo_dataset(cfg, str(train_dir), args.batch, data, mode="train")


def classify(ds) -> dict:
    """Split the base + origin slots into positive tiles / kept empty tiles / whole frames."""
    ds.set_epoch(0, 10)
    lens = ds._segment_lengths()
    bounds = {}
    acc = 0
    for name, ln in zip(("base", "origin", "ratio"), lens):
        bounds[name] = (acc, acc + ln)
        acc += ln
    shapes = {i: cv2.imread(f).shape[:2] for i, f in enumerate(ds.im_files)}
    stem_to_index = {Path(f).stem: i for i, f in enumerate(ds.im_files)}

    out = {"pos": 0, "empty": 0, "whole": 0}
    for i in range(bounds["ratio"][0]):  # base + origin slots
        lab = ds.get_image_and_label(i)
        stem = Path(lab["im_file"]).stem
        idx = stem_to_index.get(stem)
        idx = 0 if idx is None else idx  # compose-derived sources keep their first source's name
        if tuple(lab["ori_shape"]) == shapes.get(idx, (0, 0)):
            out["whole"] += 1
        elif len(lab["instances"].bboxes) == 0:
            out["empty"] += 1
        else:
            out["pos"] += 1
    out.update(lens=lens, total=len(ds), k_slice=len(ds._sel_indices("slice", len(ds.labels))))
    return out


def report(tag: str, r: dict, n_images: int) -> None:
    slots = r["pos"] + r["empty"] + r["whole"]
    pos_share = r["pos"] / slots if slots else 0.0
    print(f"{tag:<26} pool={r['total']:>6}  base={r['lens'][0]:>6}  origin={r['lens'][1]:>5}  "
          f"K_slice={r['k_slice']:>4}/{n_images}")
    print(f"{'':<26} slots={slots:>6}  positive={r['pos']:>5} ({pos_share:5.1%})  "
          f"empty={r['empty']:>5}  whole={r['whole']:>5}")


def _probability(text: str) -> bool | float:
    """Parse ``--prob``: accepts the documented True/False spellings as well as numbers."""
    low = text.strip().lower()
    if low in {"true", "t", "yes", "on"}:
        return True
    if low in {"false", "f", "no", "off"}:
        return False
    return float(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="_mini_val_set/_mini_data.yaml")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--prob", type=_probability, default=True,
                    help="slice_prob: True/False (preferred) or a number in [0,1]")
    ap.add_argument("--ratio", type=float, default=1.0, help="slice_ratio")
    ap.add_argument("--bg", type=float, default=-1.0, help="slice_background_ratio")
    ap.add_argument("--all-tiles", dest="all_tiles", action="store_true", default=True)
    ap.add_argument("--one-tile", dest="all_tiles", action="store_false")
    ap.add_argument("--img-origin", dest="img_origin", action="store_true",
                    help="img_origin: put one whole-frame slot per ORIGINAL image into the pool (unified coverage)")
    ap.add_argument("--grid", action="store_true", help="also sweep bg over -1 / 0.2 / 0")
    args = ap.parse_args(argv)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"data yaml not found: {data_yaml}", file=sys.stderr)
        return 2
    train_dir, data = _resolve(data_yaml)
    n_images = len([p for p in train_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])

    print(f"dataset: {data_yaml}  ({train_dir})  N={n_images} images, imgsz={args.imgsz}\n")
    tag = (f"prob={args.prob:g} all_tiles={args.all_tiles} ratio={args.ratio:g} "
           f"bg={args.bg:g} img_origin={args.img_origin}")
    report(tag, classify(build(train_dir, data, args)), n_images)

    if args.grid:
        print("\nbg sweep at the same ratio / all_tiles (this is the knob's whole decision surface):")
        for bg in (-1.0, 0.2, 0.0):
            report(f"   bg={bg:g}", classify(build(train_dir, data, args, slice_background_ratio=bg)),
                   n_images)

    print("\nHow to read it:")
    print("  * positive share is the dataset's tile DENSITY. If it is low (< ~40%), slice_all_tiles=True")
    print("    is paying 4 slots per image for 1 useful tile, and the bg knob decides what the other 3 hold.")
    print("  * empty tiles are pure negatives (useful only if false positives are your problem).")
    print("  * 'whole' counts slots holding the ORIGINAL frame: those come from un-selected images (ratio<1),")
    print("    from img_origin, and from empty tiles the bg quota replaced. B+ removed the first kind only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
