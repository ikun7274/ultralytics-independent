"""Migration probe: can ``ultralytics_ooo`` install and run on THIS Ultralytics build?

``install()`` is zero-intrusion by design -- it never edits upstream source -- which means it does all
of its work by REBINDING names that already exist upstream. That trade has one failure mode: rename or
relocate one upstream symbol and the patch silently stops taking effect (no ImportError, no traceback,
just a feature that no longer happens). The package has already been bitten by exactly this once: the
epoch callback read ``trainer.train_dataloader``, an attribute ``BaseTrainer`` never had, so
``set_epoch`` -- and with it the whole per-epoch ``*_ratio`` draw -- was frozen at epoch 0 for the
whole project with nothing printed.

So: run this FIRST, in the new environment, before training anything. It checks four layers.

    A. symbol presence   -- every module attribute install() rebinds, still resolvable?
    B. signature/contract -- the wrapped callables still take the arguments we forward?
    C. install() effects -- after install(), did each patch actually land? (markers + identity)
    D. end-to-end smoke  -- a real 4-image dataset through the real factory: pool geometry, sampler
                            class, and the epoch callback reaching ``set_epoch``.

A and B are cheap triage so a failure names the file to fix; C and D are the actual proof, because a
boolean "attribute exists" cannot tell you whether the rebind did anything.

Usage (from the repo root, in the new environment's Python):

    python tools/ooo_compat_check.py                 # human-readable report, exit 1 on any FAIL
    python tools/ooo_compat_check.py --out res.json  # also dump machine-readable results
    python tools/ooo_compat_check.py --quiet         # FAIL + WARN only

Offline, no weights, no downloads. One process; the smoke test uses ``workers=0`` on purpose so it
stays inside a small memory budget (spawn workers would add ~340 MiB each).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"

_results: list[dict] = []


def record(layer: str, name: str, status: str, detail: str = "", fix: str = "") -> None:
    """Append one check result. ``fix`` is the file to edit when it goes red."""
    _results.append({"layer": layer, "check": name, "status": status, "detail": detail, "fix": fix})


# --- A. upstream symbols install() rebinds ---------------------------------------------------------------------
# (module, attribute, what breaks without it, file to fix)
SYMBOLS: list[tuple[str, str, str, str]] = [
    ("ultralytics.data.build", "YOLODataset", "installer rebinds this name to InstalledYOLODataset", "installer.py"),
    ("ultralytics.data.build", "build_yolo_dataset", "the factory that reads the name above at call time", "-"),
    ("ultralytics.data.build", "build_dataloader", "wrapped by patch_build_dataloader", "pool/sampler.py"),
    ("ultralytics.data.build", "RANK", "grouped loader tail builds its seed from it", "pool/sampler.py"),
    ("ultralytics.data.build", "get_torch_device_backend", "grouped tail derives nw / pin_memory", "pool/sampler.py"),
    ("ultralytics.data.base", "BaseDataset", "superclass of OnlinePoolDataset; fraction guard host", "pool/dataset.py"),
    ("ultralytics.data.dataset", "YOLODataset", "other superclass of InstalledYOLODataset", "dataset_class.py"),
    ("ultralytics.data.dataset", "v8_transforms", "rebound to the online-augment-aware version", "installer.py"),
    ("ultralytics.data.augment", "Mosaic", "assembled by the online v8_transforms", "pool/augment_setup.py"),
    ("ultralytics.data.augment", "RandomPerspective", "assembled by the online v8_transforms", "pool/augment_setup.py"),
    ("ultralytics.data.augment", "RandomHSV", "assembled by the online v8_transforms", "pool/augment_setup.py"),
    ("ultralytics.data.augment", "RandomFlip", "assembled by the online v8_transforms", "pool/augment_setup.py"),
    ("ultralytics.data.augment", "Compose", "assembled by the online v8_transforms", "pool/augment_setup.py"),
    ("ultralytics.data.augment", "BaseTransform", "OnlineSlice subclasses it", "pool/augment_setup.py"),
    ("ultralytics.data.augment", "Albumentations", "assembled by the online v8_transforms", "pool/augment_setup.py"),
    ("ultralytics.data.augment", "CopyPaste", "assembled by the online v8_transforms", "pool/augment_setup.py"),
    ("ultralytics.data.augment", "MixUp", "assembled by the online v8_transforms", "pool/augment_setup.py"),
    ("ultralytics.data.augment", "CutMix", "assembled by the online v8_transforms", "pool/augment_setup.py"),
    ("ultralytics.utils", "DEFAULT_CFG", "online hyperparameters are registered onto it", "installer.py"),
    ("ultralytics.utils", "DEFAULT_CFG_DICT", "key-collision self-check reads it", "pool/constants.py"),
    ("ultralytics.utils", "IterableSimpleNamespace", "augment_setup builds a hyp namespace", "pool/augment_setup.py"),
    ("ultralytics.utils", "LOGGER", "all diagnostics go through it", "-"),
    ("ultralytics.utils.patches", "imread", "unicode-safe image read used by dataset + valslice", "pool/dataset.py"),
    ("ultralytics.utils", "RANK", "dual.py keeps the whole-metric writes on rank 0 only", "pool/dual.py"),
    ("ultralytics.utils.torch_utils", "strip_optimizer", "dual.py strips its whole-metric mirrors", "pool/dual.py"),
    ("ultralytics.utils.callbacks.base", "default_callbacks", "the set_epoch callback is appended", "installer.py"),
    ("ultralytics.engine.trainer", "BaseTrainer", "host of the resume + dual-metric patches", "pool/resume.py"),
    ("ultralytics.models.yolo.detect.val", "DetectionValidator", "host of the val-slice patch", "pool/valslice.py"),
]

# --- B. call-shape contracts ------------------------------------------------------------------------------------
# The wrapped function must still accept the arguments we forward, in this order. Only the leading
# names are asserted: later upstream additions are harmless, earlier renames or reorders are not.
SIGNATURES: list[tuple[str, str, list[str], str]] = [
    (
        "ultralytics.data.build",
        "build_dataloader",
        ["dataset", "batch", "workers", "shuffle", "rank", "drop_last", "pin_memory", "device"],
        "pool/sampler.py",
    ),
    ("ultralytics.data.base", "BaseDataset.get_img_files", ["self", "img_path"], "pool/fraction.py"),
    (
        "ultralytics.models.yolo.detect.val",
        "DetectionValidator.__init__",
        ["self", "dataloader", "save_dir", "args"],
        "pool/valslice.py",
    ),
    ("ultralytics.engine.trainer", "BaseTrainer.validate", ["self"], "pool/dual.py"),
    ("ultralytics.engine.trainer", "BaseTrainer.save_model", ["self"], "pool/dual.py"),
    ("ultralytics.engine.trainer", "BaseTrainer.resume_training", ["self"], "pool/resume.py"),
    ("ultralytics.engine.trainer", "BaseTrainer.final_eval", ["self"], "pool/dual.py"),
    ("ultralytics.engine.trainer", "BaseTrainer.check_resume", ["self", "overrides"], "pool/resume.py"),
    ("ultralytics.data.base", "BaseDataset.update_labels_info", ["self", "label"], "pool/dataset.py"),
    ("ultralytics.data.base", "BaseDataset.build_transforms", ["self"], "pool/augment_setup.py"),
]

# ``BaseDataset.__init__`` is checked by name, not position: ``_resolve_init_knobs`` binds *args against it
# and then reads .arguments["hyp"] / ["fraction"], so those two KEYWORDS must survive.
INIT_REQUIRED_KWARGS = ["hyp", "fraction"]


def _resolve(dotted: str):
    """Resolve ``pkg.module.attr`` / ``Class.method`` to the object, or raise ImportError/AttributeError."""
    # Walk left until an importable prefix is found, then getattr the rest (handles Class.method).
    parts = dotted.split(".")
    obj = None
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
            rest = parts[cut:]
            break
        except ImportError:
            continue
    if obj is None:
        raise ImportError(f"no importable prefix in {dotted!r}")
    for p in rest:
        obj = getattr(obj, p)
    return obj


def check_symbols() -> None:
    """A: every rebind target still resolves."""
    for mod, attr, why, fix in SYMBOLS:
        dotted = f"{mod}.{attr}"
        try:
            obj = _resolve(dotted)
            kind = type(obj).__name__ if not inspect.ismodule(obj) else "module"
            record("A", dotted, PASS, f"{kind}; {why}", fix)
        except Exception as e:  # noqa: BLE001
            record("A", dotted, FAIL, f"{type(e).__name__}: {e} ({why})", fix)


def check_signatures() -> None:
    """B: the wrapped callables accept the arguments we forward, in the same leading order."""
    for mod, attr, expected, fix in SIGNATURES:
        dotted = f"{mod}.{attr}"
        try:
            fn = _resolve(dotted)
            params = list(inspect.signature(fn).parameters)
            if params[: len(expected)] == expected:
                record("B", dotted, PASS, f"params start with {expected}", fix)
            else:
                record(
                    "B",
                    dotted,
                    FAIL,
                    f"expected leading params {expected}, got {params[: len(expected)]} (full: {params})",
                    fix,
                )
        except Exception as e:  # noqa: BLE001
            record("B", dotted, FAIL, f"{type(e).__name__}: {e}", fix)

    # BaseDataset.__init__ keyword names.
    try:
        from ultralytics.data.base import BaseDataset

        params = inspect.signature(BaseDataset.__init__).parameters
        missing = [k for k in INIT_REQUIRED_KWARGS if k not in params]
        if missing:
            record(
                "B",
                "BaseDataset.__init__ kwargs",
                FAIL,
                f"missing {missing}; _resolve_init_knobs binds *args against this signature",
                "pool/dataset.py",
            )
        else:
            record("B", "BaseDataset.__init__ kwargs", PASS, f"{INIT_REQUIRED_KWARGS} present", "pool/dataset.py")
    except Exception as e:  # noqa: BLE001
        record("B", "BaseDataset.__init__ kwargs", FAIL, f"{type(e).__name__}: {e}", "pool/dataset.py")

    # The epoch callback resolves the dataset through trainer.train_loader. Asserting the NAME in source --
    # not hasattr() -- because it is assigned inside _setup_train, so the attribute never exists on the class.
    try:
        from ultralytics.engine.trainer import BaseTrainer

        src = inspect.getsource(BaseTrainer)
        if "self.train_loader = " in src:
            record("B", "BaseTrainer.train_loader", PASS, "assigned in source; _ooo_set_epoch reads it", "installer.py")
        else:
            record(
                "B",
                "BaseTrainer.train_loader",
                FAIL,
                "not assigned anywhere in BaseTrainer; the epoch callback would be a silent no-op",
                "installer.py",
            )
        if "self.train_dataloader" in src:
            record(
                "B",
                "BaseTrainer.train_dataloader",
                WARN,
                "upstream now HAS train_dataloader; the callback's fallback still works, re-check the primary name",
                "installer.py",
            )
    except Exception as e:  # noqa: BLE001
        record("B", "BaseTrainer.train_loader", FAIL, f"{type(e).__name__}: {e}", "installer.py")

    # Mosaic is built through _compat(), which inspects its __init__ to drop unsupported kwargs. If upstream
    # ever adds **kwargs, _compat stops filtering and fork-only knobs get forwarded verbatim.
    try:
        from ultralytics.data.augment import Mosaic

        sig = inspect.signature(Mosaic.__init__)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            record(
                "B",
                "Mosaic.__init__ arity",
                WARN,
                "accepts **kwargs, so _compat cannot filter fork-only knobs; check they are still stripped",
                "pool/augment_setup.py",
            )
        else:
            record(
                "B",
                "Mosaic.__init__ arity",
                PASS,
                "no **kwargs, _compat filtering is effective",
                "pool/augment_setup.py",
            )
    except Exception as e:  # noqa: BLE001
        record("B", "Mosaic.__init__ arity", FAIL, f"{type(e).__name__}: {e}", "pool/augment_setup.py")


def check_install_effects() -> None:
    """C: after install(), each patch actually landed (marker flags + identity, not just 'no error')."""
    try:
        from ultralytics_ooo import install
        from ultralytics_ooo.dataset_class import InstalledYOLODataset
    except Exception as e:  # noqa: BLE001
        record("C", "import ultralytics_ooo", FAIL, f"{type(e).__name__}: {e}", "PYTHONPATH / package location")
        return
    record("C", "import ultralytics_ooo", PASS, "package importable", "-")

    try:
        install()
    except Exception as e:  # noqa: BLE001
        record("C", "install()", FAIL, f"{type(e).__name__}: {e}\n{traceback.format_exc()}", "installer.py")
        return
    record("C", "install()", PASS, "returned without raising", "-")
    record("C", "install() idempotent", PASS if install() is None else WARN, "second call is a no-op", "installer.py")

    # 1. dataset factory swap
    try:
        import ultralytics.data.build as b

        ok = b.YOLODataset is InstalledYOLODataset
        record("C", "build.YOLODataset rebind", PASS if ok else FAIL, f"{b.YOLODataset}", "installer.py")
    except Exception as e:  # noqa: BLE001
        record("C", "build.YOLODataset rebind", FAIL, f"{type(e).__name__}: {e}", "installer.py")

    # 2. v8_transforms swap
    try:
        import ultralytics.data.dataset as ds_mod
        from ultralytics_ooo.pool.augment_setup import v8_transforms as ours

        ok = ds_mod.v8_transforms is ours
        record("C", "dataset.v8_transforms rebind", PASS if ok else FAIL, f"{ds_mod.v8_transforms}", "installer.py")
    except Exception as e:  # noqa: BLE001
        record("C", "dataset.v8_transforms rebind", FAIL, f"{type(e).__name__}: {e}", "installer.py")

    # 3-6. the four wrapped-method patches + the fraction guard, each with its own idempotency marker
    markers = [
        ("ultralytics.data.build", "_ooo_sampler_patched", "pool/sampler.py"),
        ("ultralytics.data.base.BaseDataset", "_ooo_fraction_guard", "pool/fraction.py"),
        ("ultralytics.engine.trainer.BaseTrainer", "_ooo_resume_patched", "pool/resume.py"),
        ("ultralytics.engine.trainer.BaseTrainer", "_ooo_dual_patched", "pool/dual.py"),
        ("ultralytics.models.yolo.detect.val.DetectionValidator", "_ooo_valslice_patched", "pool/valslice.py"),
    ]
    for target, flag, fix in markers:
        try:
            obj = _resolve(target)
            ok = bool(getattr(obj, flag, False))
            record("C", f"patch marker {flag}", PASS if ok else FAIL, f"on {target}", fix)
        except Exception as e:  # noqa: BLE001
            record("C", f"patch marker {flag}", FAIL, f"{type(e).__name__}: {e}", fix)

    # 7. project keys registered on DEFAULT_CFG / DEFAULT_CFG_DICT
    try:
        from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT
        from ultralytics_ooo.pool.constants import _ONLINE_DEFAULTS

        miss_cfg = [k for k in _ONLINE_DEFAULTS if not hasattr(DEFAULT_CFG, k)]
        miss_dict = [k for k in _ONLINE_DEFAULTS if k not in DEFAULT_CFG_DICT]
        n = len(_ONLINE_DEFAULTS)
        if miss_cfg or miss_dict:
            record(
                "C",
                "online keys on DEFAULT_CFG",
                FAIL,
                f"{n} keys; missing on namespace {miss_cfg[:6]}, missing in dict {miss_dict[:6]}",
                "pool/constants.py",
            )
        else:
            record("C", "online keys on DEFAULT_CFG", PASS, f"all {n} project keys registered", "pool/constants.py")
    except Exception as e:  # noqa: BLE001
        record("C", "online keys on DEFAULT_CFG", FAIL, f"{type(e).__name__}: {e}", "pool/constants.py")

    # 8. the whole-metric strip wrapper must actually have landed on final_eval. It is optional at patch
    # time (stub trainers), so a silent skip would reintroduce the exact defect it exists to fix: the
    # mirrors surviving as fp32 checkpoints still holding the optimizer state.
    try:
        from ultralytics.engine.trainer import BaseTrainer

        ok = bool(getattr(BaseTrainer.final_eval, "_ooo_final_eval_wrapper", False))
        record(
            "C",
            "final_eval strip wrapper",
            PASS if ok else FAIL,
            "dual.py wrapped final_eval (the whole-metric mirrors get stripped)" if ok
            else "NOT wrapped; best_whole.pt / last_whole.pt would stay unstripped (fp32 + optimizer state)",
            "pool/dual.py",
        )
    except Exception as e:  # noqa: BLE001
        record("C", "final_eval strip wrapper", FAIL, f"{type(e).__name__}: {e}", "pool/dual.py")

    # 9. key-collision self-check: a project key that upstream now defines means our fallback is stale.
    #
    # Compare against the PRISTINE upstream key set, parsed from cfg/default.yaml -- NOT against the live
    # DEFAULT_CFG_DICT. install() has already written our 80 keys into that dict by this point, so the live
    # set makes every project key look like an upstream duplicate. (check_online_defaults_are_project_only's
    # own docstring warns about this; its no-arg form is only safe before install().)
    try:
        import yaml

        import ultralytics as _ul
        from ultralytics_ooo.pool.constants import check_online_defaults_are_project_only

        _default_yaml = Path(_ul.__file__).parent / "cfg" / "default.yaml"
        pristine = set(yaml.safe_load(_default_yaml.read_text(encoding="utf-8")) or {})
        drift = check_online_defaults_are_project_only(pristine)
        if drift:
            record(
                "C",
                "project-only key table",
                WARN,
                f"{len(drift)} project key(s) are ALSO defined by upstream now: {drift}. "
                f"Your package fallbacks for them are stale (install() skips them, the fallback stays live). "
                f"Delete them from _ONLINE_DEFAULTS so upstream is the single source of truth.",
                "pool/constants.py",
            )
        else:
            record("C", "project-only key table", PASS, "no overlap with upstream keys", "pool/constants.py")
    except Exception as e:  # noqa: BLE001
        record("C", "project-only key table", WARN, f"{type(e).__name__}: {e}", "pool/constants.py")

    # 10. epoch callback registered exactly once (a duplicate would run the per-epoch draw twice)
    try:
        from ultralytics.utils.callbacks.base import default_callbacks

        cbs = [f for f in default_callbacks.get("on_train_epoch_start", []) if getattr(f, "_ooo_epoch_callback", False)]
        if len(cbs) == 1:
            record("C", "epoch callback registered", PASS, "exactly one _ooo_epoch_callback", "installer.py")
        else:
            record(
                "C",
                "epoch callback registered",
                FAIL,
                f"found {len(cbs)} (expected exactly 1); 0 = the per-epoch draw is frozen at epoch 0",
                "installer.py",
            )
    except Exception as e:  # noqa: BLE001
        record("C", "epoch callback registered", FAIL, f"{type(e).__name__}: {e}", "installer.py")


# Segment lengths for N=4 with every branch on and slice_all_tiles=True. Hardcoded on purpose: this is the
# CONTRACT, and comparing it against the freshly built dataset is what proves _segment_lengths() still agrees
# with upstream's build_yolo_dataset call path (arg threading, YOLODataset.cache_labels, hyp propagation).
SMOKE_N = 4
SMOKE_IMG = (64, 48)
SMOKE_EXPECT = {"base": 16, "origin": 4, "ratio": 4, "blur": 8, "compose": 1, "weather": 4, "occlusion": 4}
SMOKE_BRANCHES = {
    "slice_prob": 1.0,
    "slice_all_tiles": True,
    "img_origin": True,
    "ratio_pad_keep": True,
    "blur_keep": True,
    "compose_keep": True,
    "weather_keep": True,
    "occlusion_keep": True,
    "slice_ratio": 1.0,
    "ratio_pad_ratio": 1.0,
    "blur_ratio": 1.0,
    "compose_ratio": 1.0,
    "weather_ratio": 1.0,
    "occlusion_ratio": 1.0,
}


def _write_dataset(root: Path, n: int = SMOKE_N) -> Path:
    """Write ``n`` tiny images + YOLO labels under ``root``; offline, no downloads."""
    import cv2
    import numpy as np

    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    w, h = SMOKE_IMG
    for i in range(n):
        cv2.imwrite(str(img_dir / f"img{i}.jpg"), (rng.random((h, w, 3)) * 255).astype(np.uint8))
        (lbl_dir / f"img{i}.txt").write_text("0 0.50 0.50 0.20 0.20\n0 0.80 0.80 0.15 0.15\n", encoding="utf-8")
    return img_dir


def check_smoke() -> None:
    """D: a real dataset through the real factory -- pool geometry, sampler class, set_epoch delivery."""
    tmp = Path(tempfile.mkdtemp(prefix="ooo_compat_"))
    try:
        # D1. build through the stock factory (so the rebound YOLODataset is what actually runs)
        try:
            from ultralytics.cfg import get_cfg
            from ultralytics.data.build import build_yolo_dataset
            from ultralytics_ooo.dataset_class import InstalledYOLODataset

            img_dir = _write_dataset(tmp)
            cfg = get_cfg(
                overrides=dict(task="detect", mode="train", imgsz=64, batch=2, fraction=1.0, **SMOKE_BRANCHES)
            )
            data = {"path": str(tmp), "names": {0: "obj"}, "channels": 3, "nc": 1, "train": "images/train"}
            ds = build_yolo_dataset(cfg, str(img_dir), 2, data, mode="train")
            record(
                "D",
                "build_yolo_dataset returns installed class",
                PASS if isinstance(ds, InstalledYOLODataset) else FAIL,
                f"{type(ds).__module__}.{type(ds).__name__}",
                "installer.py",
            )
        except Exception as e:  # noqa: BLE001
            record("D", "build_yolo_dataset returns installed class", FAIL, f"{type(e).__name__}: {e}", "installer.py")
            return

        # D2. pool geometry: the seven segment lengths, against the hardcoded contract
        try:
            lengths = ds._segment_lengths()
            got = dict(zip(["base", "origin", "ratio", "blur", "compose", "weather", "occlusion"], lengths))
            if got == SMOKE_EXPECT:
                record("D", "pool geometry (7 segments)", PASS, f"{got}, len(dataset)={len(ds)}", "pool/dataset.py")
            else:
                record(
                    "D",
                    "pool geometry (7 segments)",
                    FAIL,
                    f"expected {SMOKE_EXPECT}, got {got}. A wrong segment width changes what the model sees; "
                    f"check _segment_lengths vs upstream's arg threading",
                    "pool/dataset.py",
                )
        except Exception as e:  # noqa: BLE001
            record("D", "pool geometry (7 segments)", FAIL, f"{type(e).__name__}: {e}", "pool/dataset.py")

        # D3. POOL layer -- the contract is checked on get_image_and_label(), which is the pool's own output
        # (raw ndarray + the label keys the transforms consume). Sampling it via ds[i] instead would return
        # the POST-transform sample and report missing keys/tensors as failures.
        try:
            import numpy as np

            need = ("img", "ori_shape", "resized_shape", "ratio_pad", "cls", "instances", "im_file")
            problems = []
            n = len(ds)
            for idx in (0, SMOKE_EXPECT["base"] - 1, SMOKE_EXPECT["base"] + SMOKE_EXPECT["origin"] - 1, n - 1):
                lab = ds.get_image_and_label(idx)
                miss = [k for k in need if k not in lab]
                if miss:
                    problems.append(f"idx {idx}: missing {miss}")
                    continue
                if not isinstance(lab["img"], np.ndarray) or lab["img"].size == 0:
                    problems.append(f"idx {idx}: img {type(lab['img'])}")
            if problems:
                record("D", "pool label contract (get_image_and_label)", FAIL, "; ".join(problems), "pool/dataset.py")
            else:
                record(
                    "D",
                    "pool label contract (get_image_and_label)",
                    PASS,
                    f"4 spot indices of {n} ok",
                    "pool/dataset.py",
                )
        except Exception as e:  # noqa: BLE001
            record(
                "D", "pool label contract (get_image_and_label)", FAIL, f"{type(e).__name__}: {e}", "pool/dataset.py"
            )

        # D4. TRANSFORM layer -- ds[i] must come back as a ready-to-collate sample. This is what actually
        # exercises the rebound v8_transforms assembly (Mosaic / RandomPerspective / Format) end to end;
        # a broken assembly surfaces here rather than at the first training step.
        try:
            import torch

            sample = ds[0]
            img = sample.get("img")
            bad = (
                not torch.is_tensor(img)
                or img.ndim != 3
                or img.numel() == 0
                or "cls" not in sample
                or "bboxes" not in sample
            )
            if bad:
                record(
                    "D",
                    "transformed sample (ds[0])",
                    FAIL,
                    f"img={type(img).__name__}{getattr(img, 'shape', '')}, keys={sorted(sample)[:10]}; "
                    f"the online v8_transforms assembly may be incomplete",
                    "pool/augment_setup.py",
                )
            else:
                record(
                    "D",
                    "transformed sample (ds[0])",
                    PASS,
                    f"img {tuple(img.shape)} {img.dtype}",
                    "pool/augment_setup.py",
                )
        except Exception as e:  # noqa: BLE001
            record("D", "transformed sample (ds[0])", FAIL, f"{type(e).__name__}: {e}", "pool/augment_setup.py")

        # D5. the grouped sampler must actually be selected. Read build_dataloader from the MODULE after
        # install() -- `from ... import build_dataloader` binds the pre-patch function and would report
        # RandomSampler even when grouping works. That mistake made a project tool lie for months.
        try:
            import ultralytics.data.build as b

            dl = b.build_dataloader(ds, 2, 0, True, -1, False, True, "cpu")
            name = type(dl.sampler).__name__
            if name == "GroupedImageSampler":
                record(
                    "D", "grouped sampler selected", PASS, f"{name}, units={len(dl.sampler.units)}", "pool/sampler.py"
                )
            else:
                record(
                    "D",
                    "grouped sampler selected",
                    FAIL,
                    f"got {name}; grouping is silently off. Check GroupedImageSampler.from_dataset's guard "
                    f"(pool worth grouping? slice_grouped_sampler on?)",
                    "pool/sampler.py",
                )
        except Exception as e:  # noqa: BLE001
            record("D", "grouped sampler selected", FAIL, f"{type(e).__name__}: {e}", "pool/sampler.py")

        # D6. the epoch callback must reach set_epoch through trainer.train_loader
        try:
            from ultralytics.utils.callbacks.base import default_callbacks

            calls = []

            class _D:
                def set_epoch(self, epoch=0, epochs=None):
                    calls.append((epoch, epochs))

            class _L:
                dataset = _D()

            class _T:
                train_loader = _L()
                epoch, epochs = 3, 10

            cbs = [
                f for f in default_callbacks.get("on_train_epoch_start", []) if getattr(f, "_ooo_epoch_callback", False)
            ]
            for fn in cbs:
                fn(_T())
            if calls == [(3, 10)]:
                record("D", "epoch callback -> set_epoch", PASS, "fired exactly once with (3, 10)", "installer.py")
            else:
                record(
                    "D",
                    "epoch callback -> set_epoch",
                    FAIL,
                    f"expected [(3, 10)], got {calls}. This is the S4 failure class: set_epoch never runs, so the "
                    f"per-epoch *_ratio draw and close_aug_epoch stay frozen at epoch 0 with nothing printed",
                    "installer.py",
                )
        except Exception as e:  # noqa: BLE001
            record("D", "epoch callback -> set_epoch", FAIL, f"{type(e).__name__}: {e}", "installer.py")

        # D7. LRU bookkeeping: the origin segment now goes through the raw cache, so hits must be recorded
        try:
            ds2_hits = getattr(ds, "_raw_hits", 0)
            ds2_misses = getattr(ds, "_raw_misses", 0)
            if ds2_hits + ds2_misses > 0:
                record(
                    "D",
                    "raw-image LRU counters",
                    PASS,
                    f"hits={ds2_hits} misses={ds2_misses} (rate {ds2_hits / (ds2_hits + ds2_misses):.2f})",
                    "pool/dataset.py",
                )
            else:
                record(
                    "D",
                    "raw-image LRU counters",
                    WARN,
                    "no reads recorded yet; expected after D3 marked a slice/origin sample",
                    "pool/dataset.py",
                )
        except Exception as e:  # noqa: BLE001
            record("D", "raw-image LRU counters", WARN, f"{type(e).__name__}: {e}", "pool/dataset.py")
    finally:
        # Best-effort: the sandbox's delete path can be disabled, and a leftover temp dir is harmless.
        shutil.rmtree(tmp, ignore_errors=True)


def check_environment() -> dict:
    """Record what was actually tested -- the version string is the thing a future reader needs."""
    env: dict = {}
    try:
        import ultralytics

        env["ultralytics_version"] = ultralytics.__version__
        env["ultralytics_file"] = str(Path(ultralytics.__file__).resolve())
    except Exception as e:  # noqa: BLE001
        env["ultralytics_version"] = f"unimportable: {type(e).__name__}: {e}"
        record("A", "import ultralytics", FAIL, str(e), "install ultralytics in this environment")
        return env
    record("A", "import ultralytics", PASS, f"{env['ultralytics_version']} @ {env['ultralytics_file']}", "-")
    try:
        import platform

        import torch

        env["python"] = platform.python_version()
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = torch.version.cuda
        env["cpu_count"] = __import__("os").cpu_count()
        env["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:  # noqa: BLE001
        env["torch"] = f"unimportable: {type(e).__name__}: {e}"
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description="Check ultralytics_ooo against this Ultralytics build")
    ap.add_argument("--out", default="", help="write the full result as JSON to this path")
    ap.add_argument("--json", dest="to_stdout", action="store_true", help="print JSON to stdout instead of a report")
    ap.add_argument("--quiet", action="store_true", help="print FAIL and WARN lines only")
    a = ap.parse_args()

    env = check_environment()
    check_symbols()
    check_signatures()
    check_install_effects()
    check_smoke()

    n_fail = sum(1 for r in _results if r["status"] == FAIL)
    n_warn = sum(1 for r in _results if r["status"] == WARN)
    n_pass = sum(1 for r in _results if r["status"] == PASS)
    payload = {
        "environment": env,
        "summary": {"pass": n_pass, "fail": n_fail, "warn": n_warn, "checks": len(_results)},
        "results": _results,
    }

    if a.out:
        Path(a.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if a.to_stdout:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        layers = {
            "A": "A. 上游符号存在性",
            "B": "B. 签名 / 调用契约",
            "C": "C. install() 生效性",
            "D": "D. 端到端冒烟",
        }
        print("=" * 100)
        print("ultralytics_ooo 迁移兼容性自检")
        print("=" * 100)
        print(f"ultralytics : {env.get('ultralytics_version')}")
        print(f"location    : {env.get('ultralytics_file')}")
        print(f"python/torch: {env.get('python')} / {env.get('torch')}  cuda={env.get('cuda_version')} "
              f"avail={env.get('cuda_available')}  device={env.get('cuda_device')}  cpus={env.get('cpu_count')}")
        print()
        for key, title in layers.items():
            rows = [r for r in _results if r["layer"] == key]
            if not rows:
                continue
            print(f"--- {title} " + "-" * max(0, 92 - len(title)))
            for r in rows:
                if a.quiet and r["status"] == PASS:
                    continue
                mark = {PASS: "[ ok ]", FAIL: "[FAIL]", WARN: "[warn]", SKIP: "[skip]"}[r["status"]]
                print(f"  {mark} {r['check']}")
                if r["status"] != PASS and r["detail"]:
                    for line in str(r["detail"]).splitlines():
                        print(f"         {line}")
                    if r["fix"] and r["fix"] != "-":
                        print(f"         -> 需检查: {r['fix']}")
            print()
        print("=" * 100)
        print(f"合计 {len(_results)} 项: PASS {n_pass} / FAIL {n_fail} / WARN {n_warn}")
        if n_fail:
            print()
            print("有 FAIL 项：先按上面 -> 指出的文件修正，再重跑本脚本。")
            print("最常见的两个原因是 (1) 上游改了符号名或挪了模块，(2) 这份 ultralytics 不是本包已验证的版本。")
        elif n_warn:
            print()
            print("无 FAIL，但有 WARN：功能可用，请逐条确认 WARN 描述的语义是否仍然成立。")
        else:
            print()
            print("全部通过：本包可在此 Ultralytics 上安装并使用。")
        print("=" * 100)

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
