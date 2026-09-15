"""Primitive-level micro-benchmarks for the online-augmentation hot path.

Answers "what does one unit of work cost" for every operator the pool calls, at two frame sizes, so
an end-to-end number can be attributed to specific operators instead of guessed at.

Covers: JPEG decode, resize to imgsz, cvtColor/transpose, the Mosaic canvas allocation, the
RandomPerspective warpAffine, every blur tier, the three weather types, occlusion, the working
resolution cap, ratio-pad canvas allocation, label deepcopy, the per-sample segment-length stamp and
OnlineSlice geometry.

    python tools/ooo_perf_bench.py <dataset-root> [--imgsz 640]
"""
from __future__ import annotations

import argparse
import copy as _copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _median_ms(fn, n: int) -> float:
    ts = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - s)
    return float(np.median(ts) * 1e3)


def run(root: str, imgsz: int, big: tuple[int, int] | None) -> dict:
    from ultralytics.utils.patches import imread
    from ultralytics_ooo.core import (
        _apply_motion_blur,
        _apply_occlusion,
        _apply_weather,
        _cap_long_side,
        _ratio_pad_params,
    )
    from ultralytics_ooo.pool.augment_setup import OnlineSlice

    out: dict = {"imgsz": imgsz, "jpeg_kb": None, "big_res": None}
    files = sorted((Path(root) / "images" / "train").glob("*.jpg"))
    out["jpeg_kb"] = round(float(np.mean([f.stat().st_size for f in files])) / 1024, 1)
    im = imread(str(files[0]))
    h, w = im.shape[:2]
    out["frame"] = f"{w}x{h}"

    out["decode_ms"] = round(_median_ms(lambda: imread(str(files[0])), 8), 3)
    out[f"resize_{w}x{h}->{imgsz}_ms"] = round(
        _median_ms(lambda: cv2.resize(im, (imgsz, int(imgsz * h / w)), interpolation=cv2.INTER_LINEAR), 8), 3)
    out["cvtColor_ms"] = round(_median_ms(lambda: cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 8), 3)
    out[f"cvtColor+transpose_{imgsz}_ms"] = round(
        _median_ms(lambda: np.ascontiguousarray(
            cv2.cvtColor(cv2.resize(im, (imgsz, imgsz)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)), 8), 3)
    out[f"mosaic_canvas_alloc_{2 * imgsz}_ms"] = round(
        _median_ms(lambda: np.full((imgsz * 2, imgsz * 2, 3), 114, np.uint8), 8), 3)
    out[f"warpAffine_mosaic_{imgsz}_ms"] = round(_median_ms(
        lambda: cv2.warpAffine(np.full((imgsz * 2, imgsz * 2, 3), 114, np.uint8),
                               cv2.getRotationMatrix2D((imgsz, imgsz), 5.0, 1.0), (imgsz, imgsz),
                               borderValue=(114, 114, 114)), 8), 3)

    # label-side work, using a real dataset's label dict
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics_ooo import install

    install()
    import yaml as _yaml

    data = _yaml.safe_load((Path(root) / "data.yaml").read_text(encoding="utf-8"))
    data["path"] = root
    cfg = get_cfg(overrides={"task": "detect", "mode": "train", "imgsz": imgsz, "batch": 8, "workers": 0,
                                 "cache": False, "slice_prob": 1.0, "slice_all_tiles": True, "slice_ratio": 1.0})
    ds = build_yolo_dataset(cfg, str(Path(root) / "images" / "train"), 8, data,
                           mode="train", rect=False, stride=32)
    lab = ds.labels[0]
    out["boxes_per_image"] = len(np.asarray(lab["bboxes"]))
    out["deepcopy_label_ms"] = round(_median_ms(lambda: _copy.deepcopy(lab), 200), 4)
    out["segment_bases_stamp_us"] = round(_median_ms(lambda: ds._segment_bases(), 2000) * 1e3, 2)
    out["segment_lengths_us"] = round(_median_ms(lambda: ds._segment_lengths(), 2000) * 1e3, 2)

    sl = OnlineSlice(p=1.0, overlap_ratio=0.2, min_area_ratio=0.005, min_retain_ratio=0.4)
    lab_px = {"bboxes": np.asarray(lab["bboxes"], dtype=np.float64), "bbox_format": "xywh",
              "normalized": True, "cls": np.asarray(lab["cls"])}
    out["slice_geometry_all4_ms"] = round(_median_ms(lambda: sl._geometry(im, lab_px, 0, only=None), 100), 4)
    out["slice_geometry_only1_ms"] = round(_median_ms(lambda: sl._geometry(im, lab_px, 0, only=1), 100), 4)
    out["target_tiles_ms"] = round(_median_ms(lambda: sl.target_tiles(lab_px, (h, w), key=0), 100), 4)

    frames = [("native", im)]
    if big is not None and (big[0], big[1]) != (w, h):
        up = cv2.resize(im, big, interpolation=cv2.INTER_LINEAR)
        tmp = Path(root).parent / "_ooo_perf_big.jpg"
        cv2.imwrite(str(tmp), up, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        out["big_res"] = f"{big[0]}x{big[1]}"
        out["big_jpeg_kb"] = round(tmp.stat().st_size / 1024, 1)
        out["big_decode_ms"] = round(_median_ms(lambda: imread(str(tmp)), 5), 3)
        frames.append(("big", imread(str(tmp))))
        tmp.unlink(missing_ok=True)

    for tag, arr in frames:
        out[f"blur_short_axis_{tag}_ms"] = round(
            _median_ms(lambda arr=arr: _apply_motion_blur(arr, length=8.0, angle=0.0, axis_aligned=True), 5), 2)
        out[f"blur_short_arbitrary_{tag}_ms"] = round(
            _median_ms(lambda arr=arr: _apply_motion_blur(arr, length=8.0, angle=37.0, axis_aligned=False), 5), 2)
        out[f"blur_long_axis_defocus_{tag}_ms"] = round(_median_ms(
            lambda arr=arr: _apply_motion_blur(arr, length=27.0, angle=90.0, defocus_sigma=1.0, axis_aligned=True), 5), 2)
        out[f"blur_long_arbitrary_defocus_{tag}_ms"] = round(_median_ms(
            lambda arr=arr: _apply_motion_blur(arr, length=27.0, angle=37.0, defocus_sigma=1.0, axis_aligned=False), 5), 2)
        for wt in ("rain", "haze", "noise"):
            out[f"weather_{wt}_{tag}_ms"] = round(_median_ms(lambda arr=arr, wt=wt: _apply_weather(arr, wt), 5), 2)
        out[f"occlusion_rect_{tag}_ms"] = round(
            _median_ms(lambda arr=arr: _apply_occlusion(arr, "rect", blocks=1, size_ratio=0.1), 5), 2)
        out[f"cap_long_side_linear_{tag}_ms"] = round(
            _median_ms(lambda arr=arr: _cap_long_side(arr, 2 * imgsz, interp=cv2.INTER_LINEAR), 5), 2)
        out[f"cap_long_side_area_{tag}_ms"] = round(
            _median_ms(lambda arr=arr: _cap_long_side(arr, 2 * imgsz, interp=cv2.INTER_AREA), 5), 2)
        out[f"ratio_pad_params_{tag}_ms"] = round(
            _median_ms(lambda arr=arr: _ratio_pad_params(arr.shape[1], arr.shape[0], "auto", auto=True), 200), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--big", default="1920x1080", help="second frame size, or 'none'")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    big = None if a.big == "none" else tuple(int(v) for v in a.big.split("x"))  # type: ignore[assignment]
    txt = json.dumps(run(a.root, a.imgsz, big), indent=2, default=str)  # type: ignore[arg-type]
    print(txt, flush=True)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
