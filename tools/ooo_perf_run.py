"""End-to-end per-configuration measurement through the REAL dataloader.

For one configuration it reports:
  * pool geometry and every per-worker cache budget in MiB
  * the sampler the loader ACTUALLY ended up with (RandomSampler vs GroupedImageSampler) -- the
    grouped sampler is the package's decode-locality optimisation and it can be silently skipped
  * the Mosaic read multiplier: how many get_image_and_label calls one training item causes
  * dataset-side vs transform-side time per item
  * decode count and JPEG bytes actually read (imread is wrapped, not inferred)
  * the raw-LRU hit rate read from shared memory (the per-process counters are flushed every 16
    samples, so a per-process read reports ~0)
  * RSS delta

One configuration per process is the intended use: the pool objects plus torch are large, and this
path is run on datasets big enough that several datasets in one process is not worth the risk.

    python tools/ooo_perf_run.py <dataset-root> --key B_slice_all --imgsz 640
    python tools/ooo_perf_run.py <dataset-root> --list
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import psutil
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PROC = psutil.Process()
# Display names of the 7 segments, in layout order; "sahi" is the slice_transform tile segment that the
# code calls "base" internally. Positional: it is zipped against _segment_lengths().
SEG = ["sahi", "origin", "ratio", "blur", "compose", "weather", "occlusion"]

# Representative shapes of the pool. "A" is the inert baseline (every online switch off, so the pool
# is the plain dataset plus the img_origin coverage segment); the rest turn one lever at a time.
CONFIGS: dict[str, dict] = {
    "A_inert": {},
    "B_slice_all": {"slice_prob": 1.0, "slice_all_tiles": True, "slice_ratio": 1.0},
    "C_slice_one": {"slice_prob": 1.0, "slice_all_tiles": False, "slice_ratio": 0.5},
    "D_full": {"slice_prob": 1.0, "slice_all_tiles": True, "slice_ratio": 0.5, "ratio_pad_keep": True,
                   "blur_keep": True, "compose_keep": True, "weather_keep": True, "occlusion_keep": True,
                   "ratio_pad_ratio": 0.5, "blur_ratio": 0.5, "compose_ratio": 0.5, "weather_ratio": 0.5,
                   "occlusion_ratio": 0.5},
    "D_full_r1": {"slice_prob": 1.0, "slice_all_tiles": True, "slice_ratio": 1.0, "ratio_pad_keep": True,
                      "blur_keep": True, "compose_keep": True, "weather_keep": True, "occlusion_keep": True,
                      "ratio_pad_ratio": 1.0, "blur_ratio": 1.0, "compose_ratio": 1.0, "weather_ratio": 1.0,
                      "occlusion_ratio": 1.0},
    "E_full_uncapped": {"slice_prob": 1.0, "slice_all_tiles": True, "slice_ratio": 0.5, "ratio_pad_keep": True,
                            "blur_keep": True, "compose_keep": True, "weather_keep": True, "occlusion_keep": True,
                            "ratio_pad_ratio": 0.5, "blur_ratio": 0.5, "compose_ratio": 0.5, "weather_ratio": 0.5,
                            "occlusion_ratio": 0.5, "degrade_max_side": -1},
    "F_target_one": {"slice_prob": 1.0, "slice_all_tiles": False, "slice_ratio": 0.5, "slice_target_tiles": True},
    "G_slice_nolru": {"slice_prob": 1.0, "slice_all_tiles": True, "slice_ratio": 1.0, "slice_raw_cache_size": 0},
    "H_full_nomosaic": {"mosaic": 0.0, "slice_prob": 1.0, "slice_all_tiles": True, "slice_ratio": 0.5,
                            "ratio_pad_keep": True, "blur_keep": True, "compose_keep": True, "weather_keep": True,
                            "occlusion_keep": True, "ratio_pad_ratio": 0.5, "blur_ratio": 0.5, "compose_ratio": 0.5,
                            "weather_ratio": 0.5, "occlusion_ratio": 0.5},
    "J_degrade_only": {"blur_keep": True, "weather_keep": True, "occlusion_keep": True,
                           "blur_ratio": 1.0, "weather_ratio": 1.0, "occlusion_ratio": 1.0},
    "K_degrade_nocap": {"blur_keep": True, "weather_keep": True, "occlusion_keep": True, "blur_ratio": 1.0,
                            "weather_ratio": 1.0, "occlusion_ratio": 1.0, "degrade_max_side": -1},
    "L_slice_only_nomosaic": {"mosaic": 0.0, "slice_prob": 1.0, "slice_all_tiles": True, "slice_ratio": 1.0},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?")
    ap.add_argument("--key", default="A_inert")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cv-threads", type=int, default=0,
                    help="pin OpenCV's internal thread pool (0 = leave upstream default)")
    ap.add_argument("--prefetch", type=int, default=0,
                    help="DataLoader prefetch depth in batches/worker (0 = leave the configured default)")
    ap.add_argument("--out", default="")
    ap.add_argument("--list", action="store_true", help="print the known config keys and exit")
    a = ap.parse_args()
    if a.list or not a.root:
        print("\n".join(f"{k}" for k in CONFIGS))
        return
    if a.key not in CONFIGS:
        raise SystemExit(f"unknown --key {a.key!r}; known: {', '.join(CONFIGS)}")

    # OpenCV's thread pool runs INSIDE each DataLoader worker, so num_workers x 8 threads oversubscribe
    # a normal machine. This knob exists so the per-worker CPU/wall ratio can be measured deliberately.
    if a.cv_threads > 0:
        import cv2

        cv2.setNumThreads(a.cv_threads)

    # install() MUST come before the builder is bound. ``from ultralytics.data.build import
    # build_dataloader`` snapshots the function OBJECT, so doing it first silently kept the stock
    # loader -- this tool then reported ``RandomSampler`` for every configuration, including the ones
    # the project's GroupedImageSampler exists to accelerate, i.e. it could never measure the
    # package's own headline optimisation. Reading the attributes off the module after install()
    # picks up the patched function. Same for build_yolo_dataset: install() swaps the module-level
    # ``YOLODataset`` name it resolves at call time, so the order matters less there, but keeping
    # both uniform removes the trap.
    from ultralytics_ooo import install

    install()
    import ultralytics.data.build as _build
    from ultralytics.cfg import get_cfg

    build_dataloader = _build.build_dataloader
    build_yolo_dataset = _build.build_yolo_dataset
    import ultralytics_ooo.pool.dataset as D

    data = yaml.safe_load((Path(a.root) / "data.yaml").read_text(encoding="utf-8"))
    data["path"] = a.root
    extra = dict(CONFIGS[a.key])
    if a.prefetch > 0:
        extra["prefetch_factor"] = a.prefetch
    out: dict = {"config": a.key, "imgsz": a.imgsz, "extra": extra,
                 "cv_threads": a.cv_threads or None,
                 "rss_base_mb": round(PROC.memory_info().rss / (1 << 20), 1)}
    t0 = time.perf_counter()
    cfg = get_cfg(overrides=dict(task="detect", mode="train", imgsz=a.imgsz, batch=a.batch,
                                 workers=0, cache=False, **extra))
    ds = build_yolo_dataset(cfg, str(Path(a.root) / "images" / "train"), a.batch, data,
                            mode="train", rect=False, stride=32)
    out["build_s"] = round(time.perf_counter() - t0, 4)
    out["pool"] = len(ds)
    out["n_images"] = len(ds.labels)
    out["segments"] = dict(zip(SEG, ds._segment_lengths(), strict=True))
    out["pool_per_image"] = round(len(ds) / max(1, len(ds.labels)), 3)
    out["ims_cap"] = ds._ims_cap
    out["lru_cap"] = ds._raw_cache_size
    frame = ds.imgsz * ds.imgsz * ds.channels
    out["cache_mib"] = {"ims": round(ds._ims_cap * frame / (1 << 20), 1),
                        "raw_lru_entries": ds._raw_cache_size,
                        "mosaic_buffer_slots": getattr(ds.buffer, "maxlen", None)}
    t0 = time.perf_counter()
    ds.set_epoch(0, 100)
    out["set_epoch_s"] = round(time.perf_counter() - t0, 4)
    if ds._target_tile_schedule_on():
        t0 = time.perf_counter()
        out["target_queue_len"] = len(ds._target_tile_queue())
        out["target_queue_build_s"] = round(time.perf_counter() - t0, 4)

    # TRAINER ORDER: the loader is built in _build_train_pipeline, i.e. BEFORE the first set_epoch.
    dl = build_dataloader(ds, batch=a.batch, workers=0, shuffle=True, rank=-1)
    out["sampler"] = type(dl.sampler).__name__
    out["grouped_units"] = len(dl.sampler.units) if hasattr(dl.sampler, "units") else None
    # Resolved prefetch depth, read off the DATASET: the loader only carries one when
    # num_workers > 0 and this tool always builds at workers=0.
    out["prefetch_factor"] = getattr(ds, "prefetch_factor", None)

    orig_imread = D.imread
    dec = {"n": 0, "b": 0}

    def counting_imread(p, *args, **kwargs):
        dec["n"] += 1
        try:
            dec["b"] += Path(p).stat().st_size
        except OSError:
            pass
        return orig_imread(p, *args, **kwargs)

    D.imread = counting_imread
    real_gil = D.OnlinePoolDataset.get_image_and_label
    gil = {"n": 0, "t": 0.0}

    def counted_gil(self, index, count_slice=True):
        s = time.perf_counter()
        try:
            return real_gil(self, index, count_slice)
        finally:
            gil["n"] += 1
            gil["t"] += time.perf_counter() - s

    type(ds).get_image_and_label = counted_gil
    try:
        try:  # reset the shared LRU counters so the delta is this epoch's
            ds._mp_raw_hits.value = 0
            ds._mp_raw_misses.value = 0
        except Exception:  # noqa: BLE001,S110 -- shared memory unavailable; the delta is just unavailable
            pass
        gc.collect()
        rss0 = PROC.memory_info().rss
        c0 = PROC.cpu_times()
        gil.update(n=0, t=0.0)
        dec.update(n=0, b=0)
        nb = len(dl)
        t0 = time.perf_counter()
        it = iter(dl)
        for _ in range(nb):
            next(it)
        wall = time.perf_counter() - t0
        c1 = PROC.cpu_times()
        items = nb * a.batch
        h, m = ds.raw_cache_stats()
        out["batches"] = nb
        out["items_per_epoch"] = items
        out["epoch_s"] = round(wall, 3)
        out["item_ms"] = round(wall / items * 1e3, 2)
        out["item_cpu_ms"] = round(((c1.user - c0.user) + (c1.system - c0.system)) / items * 1e3, 2)
        out["cpu_per_wall"] = round(out["item_cpu_ms"] / out["item_ms"], 3)
        out["mosaic_multiplier"] = round(gil["n"] / items, 3)
        out["ds_side_ms"] = round(gil["t"] / items * 1e3, 2)
        out["transform_side_ms"] = round((wall - gil["t"]) / items * 1e3, 2)
        out["ds_side_share"] = round(gil["t"] / wall, 3) if wall else None
        out["decodes_per_item"] = round(dec["n"] / items, 3)
        out["jpeg_kb_per_item"] = round(dec["b"] / items / 1024, 1)
        out["decodes_per_epoch"] = dec["n"]
        out["lru"] = {"hits": h, "misses": m, "hit_rate": round(h / (h + m), 3) if h + m else None}
        out["ims_resident"] = sum(1 for v in ds.ims if v is not None)
        out["rss_peak_mb"] = round(PROC.memory_info().rss / (1 << 20), 1)
        out["rss_delta_mb"] = round((PROC.memory_info().rss - rss0) / (1 << 20), 1)
    finally:
        type(ds).get_image_and_label = real_gil
        D.imread = orig_imread

    txt = json.dumps(out, indent=2, default=str)
    print(txt, flush=True)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
