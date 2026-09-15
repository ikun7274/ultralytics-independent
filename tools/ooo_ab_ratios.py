#!/usr/bin/env python
"""A/B template for the per-epoch branch ratios: what changes when they are actually honoured.

WHY THIS EXISTS
---------------
Two things changed the meaning of the six ``*_ratio`` knobs, and both are visible from here:

1. ``ratio_pad_ratio`` / ``blur_ratio`` / ``compose_ratio`` never reached the dataset before the S3 fix,
   so those branches augmented **every** image every epoch no matter what the config said (measured:
   100% while the config asked for 10%).
2. The segments are now RATIO-SIZED: a branch owns exactly as many slots as it selected images, and an
   un-selected image keeps one plain slot instead of ``n_per`` / 2 / 1 duplicates. So lowering a ratio
   now SHRINKS the pool, where before it only swapped augmented slots for duplicate originals.

``ratio >= 1.0`` means "every slot augmented" and degenerates the segment widths back to the
pre-refactor ones, so the two arms are:

    legacy   ratios = 1.0    -> the pre-refactor pool, bit for bit (verified by
                                tests/test_ooo_branches.py::test_ratio_1_layout_is_bit_identical_...)
    intended ratios = 0.1    -> the documented semantics (10% of originals really augmented, and the
                                duplicates are gone: the pool is ~3-4x smaller)

Two modes
---------
    python tools/ooo_ab_ratios.py                 # DRY RUN (default): no training, prints what differs
    python tools/ooo_ab_ratios.py --train         # also runs both arms and diffs results.csv
    python tools/ooo_ab_ratios.py --train --epochs 2 --device cpu

The dry run is the important half: it prints the per-branch ACTUAL selected counts and the pool length
for each arm (the same line the trainer logs every epoch), which is what tells you whether a knob is
wired to anything at all. Run it from the repository root (the sample data yaml uses a relative
``path``).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

# `python tools/<name>.py` puts `tools/` -- not the repo root -- on sys.path, and this project is not
# necessarily pip-installed, so the lazy `from ultralytics.cfg import get_cfg` inside build() would fail
# on a clean checkout. Put the root first so the invocation in this docstring actually works.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RATIO_KEYS = ("ratio_pad_ratio", "blur_ratio", "compose_ratio")


def _resolve_dataset(data_yaml: Path) -> tuple[Path, dict]:
    """Return (train image dir, data dict) for a YOLO data.yaml, resolving a relative ``path``."""
    import yaml

    with open(data_yaml, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    root = Path(data.get("path", "."))
    if not root.is_absolute():
        candidates = [Path.cwd() / root, data_yaml.parent / root, data_yaml.parent]
        root = next((c for c in candidates if c.exists()), candidates[0])
    train = root / data.get("train", "images/train")
    names = data.get("names", {0: "obj"})
    if isinstance(names, list):
        names = dict(enumerate(names))
    return train, {
        "path": str(root),
        "names": names,
        "channels": data.get("channels", 3),
        "nc": data.get("nc", len(names)),
    }


def _build_dataset(data_yaml: Path, ratios: dict, args) -> object:
    """Build the training dataset through the stock factory with the online branches configured."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset

    from ultralytics_ooo import install

    install()
    img_dir, data = _resolve_dataset(data_yaml)
    overrides = dict(
        task="detect",
        mode="train",
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=1.0,
        workers=0,
        # every branch on; the three *_ratio values come from `ratios` (that is the only variable)
        slice_prob=1.0,
        slice_all_tiles=True,
        img_origin=True,
        slice_ratio=1.0,
        ratio_pad_keep=True,
        blur_keep=True,
        compose_keep=True,
        weather_keep=True,
        weather_ratio=1.0,
        occlusion_keep=True,
        occlusion_ratio=1.0,
        **ratios,
    )
    cfg = get_cfg(overrides=overrides)
    return build_yolo_dataset(cfg, str(img_dir), args.batch, data, mode="train")


def _capture(fn):
    """Run ``fn`` and return the messages logged to the ultralytics logger."""
    records = []

    class _H(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())

    logger = logging.getLogger("ultralytics")
    logger.addHandler(_H())
    try:
        fn()
    finally:
        logger.handlers.pop()
    return records


def dry_run(data_yaml: Path, arms: dict[str, dict], args) -> None:
    """Print the per-branch actual selected counts and pool length for each arm. No training."""
    print(f"dataset: {data_yaml}")
    print(f"(only `set_epoch` is called -- nothing trains; epochs={args.epochs} only feeds close_aug_epoch)\n")
    branch_of = {"ratio_pad_ratio": "ratio", "blur_ratio": "blur", "compose_ratio": "compose"}
    mult_of = {"ratio": 1, "blur": 2, "compose": 1}
    for label, ratios in arms.items():
        ds = _build_dataset(data_yaml, ratios, args)
        n_orig = len(ds.labels)
        length_before = len(ds)  # what the trainer and the sampler see, BEFORE any epoch is published
        msgs = _capture(lambda: ds.set_epoch(0, args.epochs))
        summary = next((m for m in msgs if "augment masks @" in m), "<no summary>")
        print(f"--- arm '{label}': {', '.join(f'{k}={ratios[k]:g}' for k in RATIO_KEYS)} (+ slice/weather/"
              f"occlusion ratios 1.0)")
        print(f"    pool length = {len(ds)} samples from {n_orig} images "
              f"(layout {ds._segment_lengths()}, len() before set_epoch = {length_before})")
        print(f"    {summary}")
        for k in RATIO_KEYS:
            branch = branch_of[k]
            count = next(c for a, _r, _on, c in ds._mask_specs(n_orig) if a == branch)
            sel = getattr(ds, f"_sel_{branch}", None)
            if sel is None:  # ratio >= 1 -> every slot augmented (the legacy shape)
                print(f"      {k:<18} -> {count}/{count} images augmented (all); "
                      f"segment holds {mult_of[branch] * count} slots")
            else:
                print(f"      {k:<18} -> {len(sel)} of {count} images augmented; "
                      f"segment holds {mult_of[branch] * len(sel)} slots (no fallback padding)")
        print()
    print("The two arms have DIFFERENT pool lengths: the ratio now sizes its own segment, so the")
    print("un-selected images stop occupying duplicates in it. To drop a branch entirely, turn off its")
    print("*_keep switch; to keep the branch but lower its strength, lower its ratio.")
    print("Beware the small-N rounding trap: selected = round(ratio x count) -- and note compose counts")
    print("GROUPS (ceil(N/4)), so compose_ratio=0.5 on a 4-image set rounds to 0 and does nothing.")
    print("The trainer warns about that and logs '<-- NONE'.")


def _run_arm(data_yaml: Path, label: str, ratios: dict, args, run_dir: Path) -> Path | None:
    """Train one arm. Returns its results.csv path (or None)."""
    from ultralytics import YOLO

    from ultralytics_ooo import install

    install()
    print(f"\n=== training arm '{label}': {', '.join(f'{k}={ratios[k]:g}' for k in RATIO_KEYS)} ===")
    model = YOLO(args.model)
    if args.weights and Path(args.weights).exists():
        model.load(args.weights)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        project=str(run_dir),
        name=label,
        exist_ok=True,
        val=True,
        val_slice_enable=True,
        val_slice_all_tiles=True,
        val_slice_dual_metric=True,
        # branch switches: every branch on, so the three ratios are the only variable
        slice_prob=1.0,
        slice_all_tiles=True,
        img_origin=True,
        ratio_pad_keep=True,
        blur_keep=True,
        compose_keep=True,
        weather_keep=True,
        occlusion_keep=True,
        **ratios,
    )
    csv_path = run_dir / label / "results.csv"
    return csv_path if csv_path.exists() else None


def _read_columns(csv_path: Path) -> tuple[list[str], list[dict]]:
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return [], []
    cols = [c.strip() for c in rows[0].keys()]
    parsed = [{c.strip(): (v or "").strip() for c, v in row.items()} for row in rows]
    return cols, parsed


def compare(csv_paths: dict[str, Path]) -> None:
    """Print a side-by-side 'best value over the run' comparison of the metric columns."""
    tables = {}
    for label, path in csv_paths.items():
        cols, rows = _read_columns(path)
        tables[label] = (cols, rows)
    labels = list(tables)
    if not labels or not tables[labels[0]][0]:
        print("no results.csv to compare")
        return
    metric_cols = [
        c for c in tables[labels[0]][0]
        if c.startswith("metrics/") or c.startswith("whole_metrics/") or c in {"train/box_loss", "train/cls_loss"}
    ]
    print("\n=== A/B metric comparison (best (max) over the run; losses shown as final (min-ish)) ===")
    header = f"{'column':<34}" + "".join(f"{lab:>16}" for lab in labels) + f"{'delta':>14}"
    print(header)
    print("-" * len(header))
    for col in metric_cols:
        vals = {}
        for lab in labels:
            _cols, rows = tables[lab]
            series = [float(r[col]) for r in rows if r.get(col) not in (None, "", "nan")]
            if not series:
                continue
            vals[lab] = min(series) if col.endswith("loss") else max(series)
        if len(vals) < 2:
            continue
        a, b = (vals.get(labels[0]), vals.get(labels[1]))
        delta = f"{b - a:+.4f}" if a is not None and b is not None else ""
        print(f"{col:<34}" + "".join(f"{vals.get(lab, float('nan')):>16.4f}" for lab in labels) + f"{delta:>14}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="_mini_val_set/_mini_data.yaml", help="YOLO data.yaml")
    ap.add_argument("--model", default="ultralytics/cfg/models/26/yolo26n.yaml", help="model YAML to build from")
    ap.add_argument("--weights", default="weights/yolo26n.pt", help="optional pretrained weights to load")
    ap.add_argument("--ratios", type=float, default=0.1, help="'intended' arm value for the three ratios")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0, help="kept identical across arms (seed now matters)")
    ap.add_argument("--project", default="runs/ooo_ab_ratios")
    ap.add_argument("--train", action="store_true", help="also run the two trainings and diff results.csv")
    args = ap.parse_args(argv)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"data yaml not found: {data_yaml}\nrun this script from the repository root", file=sys.stderr)
        return 2

    arms = {
        "legacy": {k: 1.0 for k in RATIO_KEYS},  # == the pre-fix behaviour
        "intended": {k: args.ratios for k in RATIO_KEYS},
    }
    dry_run(data_yaml, arms, args)

    if not args.train:
        print("\n(dry run only -- pass --train to also run both arms and compare results.csv)")
        return 0

    run_dir = Path(args.project)
    csv_paths = {}
    for label, ratios in arms.items():
        p = _run_arm(data_yaml, label, ratios, args, run_dir)
        if p:
            csv_paths[label] = p
    if len(csv_paths) == 2:
        compare(csv_paths)
        print(f"\nresults.csv: {', '.join(str(p) for p in csv_paths.values())}")
    else:
        print("\nWARNING: one or both arms produced no results.csv; cannot compare", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
