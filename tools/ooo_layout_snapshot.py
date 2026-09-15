"""Snapshot the pool's index -> sample mapping, so a layout change can be diffed bit for bit.

The pool's compatibility contract is "with every ``*_ratio >= 1`` the layout AND the index-to-sample
mapping are identical to the pre-refactor shape". This tool makes that checkable: it walks the whole
index range and prints, per index, a DETERMINISTIC signature of the produced sample.

The signature is ``(source-file stem, ori_shape, sha1(boxes | cls))`` taken from the TRANSFORMED label
(``label['instances']``; ``view_transform`` pops ``bboxes`` into it) and deliberately NOT an image hash:
the blur / weather branches draw their parameters from the global ``random`` stream, so image bytes
differ between two runs of the SAME code while the geometry and labels do not.

Two configuration details exist to keep the signature stable run to run, and both are about the ONE
branch that can DROP labels:
  * ``occlusion_max_cover`` is pushed above 1.0, so no target is ever removed for being covered. With
    the shipped 0.95 the removal depends on where the randomly placed occluder landed, which would make
    the signature -- and therefore any diff -- flap for reasons unrelated to the layout.
  * the mapping itself (which image each index resolves to) is asserted independently and analytically
    by ``tests/test_ooo_branches.py::test_ratio_1_layout_is_bit_identical_to_the_legacy_shape``, because
    a "before" capture of the pre-refactor code needs a VCS checkout this project does not have.

Usage:
    python tools/ooo_layout_snapshot.py --ratio 1.0 --out baseline.json
    ...change the layout code...
    python tools/ooo_layout_snapshot.py --ratio 1.0 --compare baseline.json   # exit 1 on any diff

``--ratio`` applies to all six ratio knobs at once (1.0 = the legacy shape).
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
        # two boxes: one per quadrant, so slicing / padding / occlusion all move something
        (lbl_dir / f"i{i}.txt").write_text("0 0.30 0.30 0.20 0.20\n0 0.75 0.70 0.15 0.15\n", encoding="utf-8")
    return img_dir


def _dataset_root(n: int) -> Path:
    """A DETERMINISTIC dataset directory, rebuilt in place.

    The per-epoch selection is seeded from the dataset's image-file LIST
    (``zlib.crc32("\\n".join(im_files))`` -- see ``OnlinePoolDataset._mask_seed``), so a fresh
    ``mkdtemp()`` per run would draw a DIFFERENT selection every run and every ratio < 1 diff would show
    those different source images instead of a layout change. Rebuilding the same path with the same
    filenames keeps the seed -- and therefore the selection -- identical across runs, which is what a
    snapshot diff needs. (At ratio >= 1 the selection is the "all" sentinel, so this only ever bit the
    ratio < 1 cases.)
    """
    root = Path(tempfile.gettempdir()) / f"ooo_layout_snapshot_n{n}"
    return _synthetic(n, root).parent.parent


def build(n: int, ratio: float, **extra):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset

    from ultralytics_ooo import install

    install()
    img_dir = _dataset_root(n) / "images" / "train"
    root = img_dir.parent.parent
    overrides = dict(
        task="detect", mode="train", imgsz=64, batch=4, fraction=1.0, workers=0,
        slice_prob=1.0, slice_all_tiles=True, img_origin=True,
        ratio_pad_keep=True, blur_keep=True, compose_keep=True,
        weather_keep=True, occlusion_keep=True,
        occlusion_max_cover=2.0,  # > 1.0: never drop a target (see the module docstring)
        **{k: ratio for k in RATIO_KEYS}, **extra,
    )
    cfg = get_cfg(overrides=overrides)
    data = {"path": str(root), "names": {0: "obj"}, "channels": 3, "nc": 1}
    return build_yolo_dataset(cfg, str(img_dir), 4, data, mode="train")


def signature(label: dict) -> dict:
    """Deterministic per-sample signature: source stem, geometry and (transformed) label content.

    ``label['instances']`` holds the boxes the branch produced -- ``update_labels_info`` pops
    ``bboxes`` into it -- and ``cls`` stays a plain array next to it, so both are read from wherever
    they actually are instead of assuming the pre-transform dict shape.
    """
    inst = label.get("instances")
    if inst is not None:
        boxes = np.asarray(inst.bboxes, dtype=np.float32)
    else:
        boxes = np.asarray(label.get("bboxes", np.empty((0, 4))), dtype=np.float32)
    cls = np.asarray(label.get("cls", np.empty((0, 1))), dtype=np.float32)
    payload = np.concatenate([boxes.reshape(-1), cls.reshape(-1)]).tobytes()
    return {
        "src": Path(label["im_file"]).stem,  # compose/ratio report their first source
        "ori_shape": list(label["ori_shape"]),
        "n_boxes": int(len(boxes)),
        "labels": hashlib.sha1(payload).hexdigest()[:12],
    }


def snapshot(ds, n: int) -> dict:
    lens = ds._segment_lengths()
    bounds, acc = {}, 0
    for name, ln in zip(SEG_ORDER, lens):
        bounds[name] = [acc, acc + ln]
        acc += ln
    entries = {str(i): signature(ds.get_image_and_label(i)) for i in range(len(ds))}
    return {
        "n_images": n,
        "segment_lengths": lens,
        "total": len(ds),
        "segments": bounds,
        "entries": entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8, help="number of synthetic images")
    ap.add_argument("--ratio", type=float, default=1.0, help="applied to all six *_ratio knobs")
    ap.add_argument("--out", type=Path, default=None, help="write the snapshot JSON here")
    ap.add_argument("--compare", type=Path, default=None, help="diff against a previous snapshot")
    ap.add_argument("--quiet", action="store_true", help="only print the diff / summary")
    args = ap.parse_args()

    ds = build(args.n, args.ratio)
    if args.ratio < 1.0:
        ds.set_epoch(0, 10)  # ratio-gated: the selection set exists only after a rebuild
    snap = snapshot(ds, args.n)

    if args.out:
        args.out.write_text(json.dumps(snap, indent=1, sort_keys=True), encoding="utf-8")

    if not args.quiet:
        print(f"N={args.n} ratio={args.ratio:g}  total={snap['total']}")
        for name in SEG_ORDER:
            lo, hi = snap["segments"][name]
            print(f"  {name:<11} [{lo:>5}, {hi:>5})  len={hi - lo}")
        print(f"  segment_lengths = {snap['segment_lengths']}")
        counts: dict[str, int] = {}
        for e in snap["entries"].values():
            counts[e["src"]] = counts.get(e["src"], 0) + 1
        dup = {k: v for k, v in sorted(counts.items()) if v > 1}
        print(f"  slots per source: min={min(counts.values())} max={max(counts.values())} "
              f"distinct={len(counts)}")
        print(f"  sources appearing more than once: {len(dup)}/{len(counts)}")

    if args.compare:
        old = json.loads(args.compare.read_text(encoding="utf-8"))
        diffs = []
        if old["segment_lengths"] != snap["segment_lengths"]:
            diffs.append(f"segment_lengths: {old['segment_lengths']} -> {snap['segment_lengths']}")
        if old["total"] != snap["total"]:
            diffs.append(f"total: {old['total']} -> {snap['total']}")
        shared = sorted(set(old["entries"]) & set(snap["entries"]), key=int)
        changed = [k for k in shared if old["entries"][k] != snap["entries"][k]]
        if changed:
            diffs.append(f"{len(changed)}/{len(shared)} shared indices changed, first: "
                         + "; ".join(f"{k}: {old['entries'][k]} -> {snap['entries'][k]}"
                                     for k in changed[:3]))
        only_old = sorted(set(old["entries"]) - set(snap["entries"]), key=int)
        only_new = sorted(set(snap["entries"]) - set(old["entries"]), key=int)
        if only_old:
            diffs.append(f"{len(only_old)} indices removed (e.g. {only_old[:5]})")
        if only_new:
            diffs.append(f"{len(only_new)} indices added (e.g. {only_new[:5]})")
        if diffs:
            print("\nDIFF vs", args.compare)
            for d in diffs:
                print("  -", d)
            return 1
        print(f"\nIDENTICAL to {args.compare} ({len(shared)} indices, layout {snap['segment_lengths']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
