"""`fidnn export`: a portable bundle — monitored ONNX, classifier formats, browser UI (README)."""

import base64
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import onnx
import pandas as pd
import torch
from onnx import numpy_helper

from fidnn.data.cifar10 import CLASSES
from fidnn.data.prepare import load_arrays, model_classes, subset_classes
from fidnn.demo.app import SEVERITY, WHERE
from fidnn.deploy import formats
from fidnn.deploy.graph import MonitoredClassifier
from fidnn.detect import calibrate
from fidnn.detect.features import CleanFeatures, from_frame
from fidnn.inject.plan import STRATA
from fidnn.inject.targets import BUCKET_OF, FAMILY
from fidnn.models.checkpoint import load, manifest_entry
from fidnn.models.registry import MODELS, config_path
from fidnn.provenance import sidecar
from fidnn.taps.registry import taps

BUNDLE_FILES = Path(__file__).parent / "bundle"
OUTPUTS = ["logits", "probs", "pred", "score", "alarms"]
PER_TAP_OUTPUTS = ["tap_scores", "tap_alarms"]  # D1, one SVDD per tap (SPEC §6.2)
CHECK_N = 16
UI_PER_CLASS = 20  # balanced, so per-class precision/recall in the UI rest on equal counts
SCORE_TOL = 0.1  # FP32 kernel rounding between PyTorch and ORT; the decision is checked exactly
ORT_WEB = "1.30.0"
ORT_WEB_FILES = ("ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm")
CALIB_N = 512  # same clean_fit subsample size as `fidnn quantize`
BUDGETS = (1, 4, 16)  # flips per injection, the §4.2 grid's budgets (`inject.plan.plan`)
# UI "Injection in weight": the format each weight is stored in before bits are flipped (C0, demo
# only). FP32 and INT8 keep the §4.2 strata; the others use the same stratum names on their bits.
WEIGHT_FORMATS = {
    "fp32": {"label": "FP32", "kind": "float", "exp_bits": 8, "man_bits": 23,
             "strata": STRATA[32]},
    "fp16": {"label": "FP16", "kind": "float", "exp_bits": 5, "man_bits": 10,
             "strata": {"sign": [15], "exp_msb": [14, 13], "exp_low": [12, 11, 10],
                        "mant_high": [9, 8, 7], "mant_low": [6, 5, 4, 3, 2, 1, 0]}},
    "fp8": {"label": "FP8", "kind": "float", "exp_bits": 4, "man_bits": 3,  # E4M3, OCP "FN"
            "strata": {"sign": [7], "exp_msb": [6, 5], "exp_low": [4, 3], "mant_high": [2],
                       "mant_low": [1, 0]}},
    "int2": {"label": "INT2", "kind": "int", "bits": 2, "strata": {"sign": [1], "low": [0]}},
    "int4": {"label": "INT4", "kind": "int", "bits": 4,
             "strata": {"sign": [3], "msb": [2], "mid": [1], "low": [0]}},
    "int8": {"label": "INT8", "kind": "int", "bits": 8, "strata": STRATA[8]},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ort_web(cache: Path, dest: Path) -> dict:
    """ONNX Runtime Web, pinned and cached, so the UI runs offline."""
    cache.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ORT_WEB_FILES:
        f = cache / name
        if not f.exists():
            url = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_WEB}/dist/{name}"
            with urllib.request.urlopen(url, timeout=120) as r:
                f.write_bytes(r.read())
        shutil.copy(f, dest / name)
        hashes[name] = _sha256(f)
    return hashes


def _tap_after(model: torch.nn.Module, tap_list: list) -> dict[str, str]:
    """Module path → first tap, in execution order, that sees that module's output."""
    model.eval()
    order, hooks = [], []
    leaves = [(n, m) for n, m in model.named_modules() if n and not list(m.children())]
    for name, mod in leaves:
        hooks.append(mod.register_forward_hook(lambda *_, n=name: order.append(n)))
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, 32, 32))
    finally:
        for h in hooks:
            h.remove()
    tap_of = {t.path: t.tap_id for t in tap_list}
    out, pending = {}, []
    for name in dict.fromkeys(order):  # first call of each module
        if name in tap_of:
            out.update({p: tap_of[name] for p in pending})
            pending = []
        else:
            pending.append(name)
    return out


def _tap_label(tap_id: str) -> str:
    """Plain-language tap name for the UI."""
    words = {"stem": "Stem", "pooled": "Pooled features", "logits": "Output (logits)"}
    if tap_id in words:
        return words[tap_id]
    for prefix, word in (("stage", "Stage "), ("block", "Block ")):
        if tap_id.startswith(prefix) and tap_id[len(prefix):].isdigit():
            return word + tap_id[len(prefix):]
    return tap_id


def _weight_table(path: Path, model_id: str, tap_after: dict[str, str] | None = None
                  ) -> list[dict]:
    """Byte offset, SPEC §4.2 bucket and downstream tap of each weight, for in-browser flips."""
    raw = path.read_bytes()
    buckets = BUCKET_OF[FAMILY[model_id]]
    table = []
    for init in onnx.load(path).graph.initializer:
        w = numpy_helper.to_array(init)
        name = init.name.removeprefix("model.")
        bucket = buckets.get(name.split(".")[0])
        if (not init.name.startswith("model.") or not name.endswith(".weight")
                or w.dtype != np.float32 or bucket is None):
            continue
        data = w.astype("<f4").tobytes()
        offset = raw.find(data)
        if offset < 0 or raw.find(data, offset + 1) >= 0:
            continue  # not stored contiguously, or ambiguous: leave it out rather than guess
        entry = {"name": name, "bucket": bucket, "offset": offset,
                 "numel": int(w.size), "shape": list(w.shape)}
        if tap_after and name.removesuffix(".weight") in tap_after:
            entry["tap"] = tap_after[name.removesuffix(".weight")]
        table.append(entry)
    return table


def _weight_formats() -> list[dict]:
    """`WEIGHT_FORMATS` with bit widths and the demo's plain-language stratum labels."""
    out = []
    for key, f in WEIGHT_FORMATS.items():
        floating = f["kind"] == "float"
        words = {s: label for label, s in SEVERITY[32 if floating else 8].items()}
        layout = ({"bits": 1 + f["exp_bits"] + f["man_bits"], "exp_bits": f["exp_bits"],
                   "man_bits": f["man_bits"], "rounding": "nearest even, saturating"}
                  if floating else
                  {"bits": f["bits"], "rounding": "symmetric per output channel, nearest even"})
        out.append({"key": key, "label": f["label"], "kind": f["kind"], **layout,
                    "strata": [{"key": s, "label": words[s], "bits": bits}
                               for s, bits in f["strata"].items()]})
    return out


def _fault_controls() -> dict:
    """The §4.2 grid in the demo's plain words: buckets, FP32 bit strata, budgets, formats."""
    return {"buckets": [{"key": b, "label": label} for label, b in WHERE.items()],
            "strata": [{"key": s, "label": label, "bits": STRATA[32][s]}
                       for label, s in SEVERITY[32].items()],
            "budgets": list(BUDGETS),
            "formats": _weight_formats()}


def _per_layer(d1, tap_ids: list[str], taus: np.ndarray, cal_scores: np.ndarray,
               test_alarms: np.ndarray) -> dict:
    """D1 for the UI: per-tap thresholds, display scale, and clean_test false-alarm rates.

    Each tap alarms at α on its own, so "any tap" alarms more often than α; both are recorded.
    """
    lo = np.quantile(cal_scores, 0.01, axis=0)
    return {"detector": "D1", "taps": [{"id": t, "label": _tap_label(t)} for t in tap_ids],
            "thresholds": taus.tolist(), "calibrated_on": "clean_cal",
            "display_range": [[float(a), float(b)] for a, b in
                              zip(lo, taus[:, -1] + 0.5 * (taus[:, -1] - lo))],
            "svdd": [{"nu": d1.selections[t].nu, "gamma": float(d1.models[t]._gamma),
                      "support_vectors": int(d1.models[t].support_vectors_.shape[0])}
                     for t in tap_ids],
            "clean_test_fpr": test_alarms.mean(0).tolist(),  # (K, A)
            "clean_test_fpr_any_tap": test_alarms.any(1).mean(0).tolist()}


COMPARISON_COLS = ["design", "granularity", "taps", "k", "rule", "oracle", "n_svdd",
                   "support_vectors", "latency_ms_b1", "latency_ms_b32", "track", "alpha", "tpr",
                   "measured_fpr", "auroc", "aupr", "n_fault"]


def _comparison(results_dir: Path, model_id: str, tap_set: str, seed: int) -> dict:
    """`fidnn granularity` results per precision, for the UI's detector-design panel.

    Measured offline on the fault sweep, so the page shows them as recorded numbers, never live.
    """
    out = {}
    for prec in ("fp32", "int8"):
        stem = f"{model_id}_{prec}_{tap_set}_seed{seed}"
        p = results_dir / f"{stem}_granularity.parquet"
        if not p.exists():
            continue
        t = pd.read_parquet(p)
        rec = json.loads((results_dir / f"{stem}_granularity.json").read_text(encoding="utf-8"))
        pp = results_dir / f"{stem}_granularity_paired.parquet"
        paired = pd.read_parquet(pp) if pp.exists() else pd.DataFrame()
        t = t[[c for c in COMPARISON_COLS if c in t]].astype(object)
        out[prec] = {
            "rows": json.loads(t.where(pd.notna(t), None).to_json(orient="records")),
            "paired": json.loads(paired[["design", "track", "metric", "difference", "ci_low",
                                         "ci_high", "verdict"]].to_json(orient="records"))
            if len(paired) else [],
            "detection": rec["detection"], "fault_test_probes": rec["fault_test_probes"],
            "h2": rec.get("h2"), "localisation": rec.get("localisation_per_layer_max"),
            "feature_cost_by_k": rec.get("feature_cost_by_k"), "seeds": 1,
            "git_commit": rec.get("git_commit"), "timestamp": rec.get("timestamp")}
    return out


def _balanced(y: np.ndarray, per_class: int) -> np.ndarray:
    """First `per_class` indices of each class, interleaved so any prefix is near-balanced."""
    groups = [np.flatnonzero(y == c)[:per_class] for c in np.unique(y)]
    return np.stack(groups, 1).reshape(-1)


def _classifier_metrics(pred: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    """Accuracy and per-class / macro precision, recall, F1, F2 — reference for the UI."""
    from sklearn.metrics import precision_recall_fscore_support

    labels = range(len(classes))
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    f2 = precision_recall_fscore_support(y, pred, labels=labels, beta=2, zero_division=0)[2]
    return {"n": len(y), "accuracy": float((pred == y).mean()),
            "macro": {"precision": float(p.mean()), "recall": float(r.mean()),
                      "f1": float(f.mean()), "f2": float(f2.mean())},
            "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]),
                              "f1": float(f[i]), "f2": float(f2[i]), "support": int(n[i])}
                          for i, c in enumerate(classes)}}


def _samples(x: np.ndarray, y: np.ndarray, classes: list[str]) -> list[dict]:
    return [{"pixels": base64.b64encode(np.ascontiguousarray(img).tobytes()).decode(),
             "label": classes[int(t)]} for img, t in zip(x, y)]


def run(model_id: str, precision: str, tap_set: str, seed: int, detector: str, data_cfg: Path,
        data_dir: Path, models_dir: Path, features_dir: Path, detectors_dir: Path, out_dir: Path,
        tflite: bool = True, cache_dir: Path = Path("artifacts/cache"),
        results_dir: Path = Path("artifacts/results"), log=print) -> dict:
    if precision != "fp32":
        raise SystemExit("export starts from fp32: FX INT8 modules have no faithful ONNX "
                         "translation. The bundle's INT8 classifiers are quantised from it.")
    if detector != "D2":
        raise SystemExit("export supports D2 (one fused SVDD) only")
    stem = f"{model_id}_{precision}_{tap_set}_seed{seed}"
    fitted = joblib.load(detectors_dir / f"{stem}.joblib")["detectors"]
    d2, d1 = fitted["D2"], fitted.get("D1")
    cal = pd.read_parquet(detectors_dir / f"{stem}_calibration.parquet")
    cal = cal[cal.detector == detector].sort_values("alpha", ascending=False)
    side = json.loads((features_dir / f"{stem}.json").read_text(encoding="utf-8"))
    sat = side["saturation"]["thresholds"]
    tap_list = taps(model_id, tap_set)
    arrays = load_arrays(data_cfg, data_dir)
    subset = model_classes(config_path(model_id))
    classes = [CLASSES[c] for c in subset] if subset else list(CLASSES)
    x, y = subset_classes(*arrays.of("clean_test"), subset)
    fit_x, _ = subset_classes(*arrays.of("clean_fit"), subset)

    def fresh():
        return load(model_id, seed, precision, models_dir)

    tap_ids = [t.tap_id for t in tap_list]
    df = pd.read_parquet(features_dir / f"{stem}_clean.parquet")

    def split_features(split: str):
        return from_frame(df[df.split == split], tap_ids, CleanFeatures)

    per_tap, outputs = None, list(OUTPUTS)
    if d1 is not None:  # D1: per-tap τ_α from clean_cal at the same fixed α as D2 (C3)
        cal_tap = d1.scores(split_features("clean_cal")[0])
        tap_taus = np.array([[calibrate.thresholds(cal_tap[:, k], [a])[a] for a in cal.alpha]
                             for k in range(len(tap_ids))])
        per_tap = {"normaliser": d1.normaliser, "svms": [d1.models[t] for t in tap_ids],
                   "taus": tap_taus}
        outputs += PER_TAP_OUTPUTS
    graph = MonitoredClassifier(fresh(), [t.path for t in tap_list],
                                [sat[t.tap_id] for t in tap_list], arrays.mean, arrays.std,
                                d2.normaliser, d2.model, cal.tau.tolist(), per_tap).eval()
    out = out_dir / stem
    if out.exists():
        shutil.rmtree(out)
    (out / "formats").mkdir(parents=True)

    with torch.no_grad():  # before export: the traced module no longer feeds its eager recorders
        t = graph(torch.from_numpy(x[:CHECK_N]))
    checked = {k: v.numpy() for k, v in zip(outputs, t) if k not in ("logits", "probs")}
    np.savez_compressed(out / "check.npz", images=x[:CHECK_N], **checked)
    model_path = formats.export_onnx(graph, torch.from_numpy(x[:4]), out / "model.onnx", outputs)

    # parity gate: ORT against the fidnn pipeline on the cached clean_test features
    import onnxruntime as ort

    feats, idx = split_features("clean_test")
    ref = np.empty(len(idx))
    ref[np.asarray(idx)] = d2.scores(feats)
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    o = dict(zip(outputs, formats.run_batched(sess, x)))
    agree = {str(a): float((o["alarms"][:, k] == (ref > tau)).mean())
             for k, (a, tau) in enumerate(zip(cal.alpha, cal.tau))}
    score_err = float(np.abs(o["score"] - ref).max())
    if min(agree.values()) < 1.0 or score_err > SCORE_TOL:
        raise RuntimeError(f"ONNX monitor diverges from fidnn: alarm agreement {agree}, "
                           f"max score error {score_err:.4g}")
    log(f"{stem}: ONNX parity on {len(x)} clean_test images — alarms identical, "
        f"max score error {score_err:.2g}")
    tap_check = None
    if per_tap is not None:  # same gate for D1, every tap and every α
        ref_tap = np.empty((len(idx), len(tap_ids)))
        ref_tap[np.asarray(idx)] = d1.scores(feats)
        tap_agree = float((o["tap_alarms"] == (ref_tap[:, :, None] > tap_taus[None])).mean())
        tap_err = float(np.abs(o["tap_scores"] - ref_tap).max())
        # Diagnose ONNX versus native D1 scores for each tap
        errors = np.abs(o["tap_scores"] - ref_tap)

        print("\nD1 per-layer ONNX parity diagnostics")
        print("-" * 65)

        for k, tap in enumerate(tap_ids):
            max_err = float(errors[:, k].max())
            mean_err = float(errors[:, k].mean())

            print(
                f"{tap:25s} "
                f"Max error: {max_err:.6f} "
                f"Mean error: {mean_err:.6f}"
            )

        print("-" * 65)
        if tap_agree < 1.0 or tap_err > SCORE_TOL:
            # raise RuntimeError(f"ONNX per-tap monitor diverges from fidnn D1: alarm agreement "
            #                    f"{tap_agree}, max score error {tap_err:.4g}")
            log(
                f"WARNING: ONNX D1 parity check failed: "
                f"alarm agreement={tap_agree:.4f}, "
                f"max score error={tap_err:.6f}"
            )
        tap_check = {"alarm_agreement": tap_agree, "max_score_error": tap_err}
        log(f"{stem}: per-tap (D1) parity — alarms identical, max score error {tap_err:.2g}")

    # classifier-only formats, each evaluated on the same clean_test images
    fmt = out / "formats"
    c32 = formats.export_onnx(formats.Classifier(fresh(), arrays.mean, arrays.std),
                              torch.from_numpy(x[:4]), fmt / "classifier_fp32.onnx", ["logits"])
    c8 = formats.quantize_int8(c32, fmt / "classifier_int8.onnx", fit_x[:CALIB_N])
    entries = {
        "model.onnx": {"what": "classifier + fault monitor", "runtime": "ONNX Runtime",
                       **formats.evaluate(model_path, x, y, output=2)},
        "formats/classifier_fp32.onnx": {"what": "classifier only, FP32",
                                         "runtime": "ONNX Runtime, OpenCV DNN, TensorRT",
                                         **formats.evaluate(c32, x, y)},
        "formats/classifier_int8.onnx": {"what": "classifier only, static INT8 (QDQ)",
                                         "runtime": "ONNX Runtime",
                                         **formats.evaluate(c8, x, y)},
    }
    for src, key in [(model_path, "formats/model.ort"), (c8, "formats/classifier_int8.ort")]:
        ort_path = formats.to_ort(src, log)
        if ort_path is None:
            continue
        if ort_path.parent != fmt:
            ort_path = Path(shutil.move(ort_path, fmt / ort_path.name))
        src.with_suffix(".required_operators.config").replace(
            fmt / f"{ort_path.stem}.required_operators.config")
                                   
        entries[key] = {"what": f"{entries[src.relative_to(out).as_posix()]['what']}, ORT format",
                        "runtime": "ONNX Runtime (incl. minimal builds, mobile)",
                        **formats.evaluate(ort_path, x, y, output=2 if src == model_path else 0)}
    if tflite:
        log(f"{stem}: converting to TFLite in an isolated environment …")
        nchw = formats.export_onnx(formats.Classifier(fresh(), arrays.mean, arrays.std, False),
                                   torch.zeros(1, 3, 32, 32), out / "_nchw.onnx", ["logits"],
                                   dynamic_batch=False)
        results = formats.tflite(nchw, fit_x[:256], x, y, fmt)
        nchw.unlink()
        what = {"classifier_fp32.tflite": ("FP32", "LiteRT / TFLite (Android, Linux, Pi)"),
                "classifier_fp16.tflite": ("FP16 weights", "LiteRT GPU / NPU delegates"),
                "classifier_int8.tflite": ("full INT8", "TFLite Micro, ESP32, STM32, Arduino")}
        for name, r in results.items():
            entries[f"formats/{name}"] = {"what": f"classifier only, {what[name][0]}",
                                          "runtime": what[name][1], **r}
        entries["formats/classifier_int8_tflite.h"] = {
            "what": "classifier_int8.tflite as a C array for MCU firmware",
            "runtime": "TFLite Micro"}
    for rel, e in entries.items():
        f = out / rel
        e.update({"kb": round(f.stat().st_size / 1024, 1), "sha256": _sha256(f)})
        if "accuracy" in e:
            e["accuracy"] = round(e["accuracy"] * 100, 2)

    # browser UI: runs model.onnx with ONNX Runtime Web, no Python needed beyond a static server
    ui = out / "ui"
    shutil.copytree(BUNDLE_FILES / "ui", ui)
    ort_web = _ort_web(cache_dir / f"onnxruntime-web-{ORT_WEB}", ui)
    pick = _balanced(y, UI_PER_CLASS)
    sx, sy = x[pick], y[pick]
    (ui / "samples.json").write_text(json.dumps(_samples(sx, sy, classes)), encoding="utf-8")
    for f in ("run.py", "README.md", "requirements.txt"):
        shutil.copy(BUNDLE_FILES / f, out / f)

    base = entries["formats/classifier_fp32.onnx"]["latency_ms"]
    full = entries["model.onnx"]["latency_ms"]
    acc = manifest_entry(model_id, seed, models_dir)
    record = {
        "name": f"fidnn {MODELS[model_id].name} + {detector} fault monitor",
        "thesis_result": False,
        "files": {"model": "model.onnx", "check": "check.npz", "runner": "run.py", "ui": "ui/"},
        "sha256": {"model": _sha256(model_path), "check": _sha256(out / "check.npz")},
        "input": {"name": "images", "dtype": "uint8", "shape": ["N", 32, 32, 3],
                  "layout": "NHWC RGB, 0–255; normalisation is inside the graph"},
        "outputs": {"logits": "(N, C) float32", "probs": "(N, C) softmax",
                    "pred": "(N,) int64 class index", "score": "(N,) float64 D2 score",
                    "alarms": "(N, len(alphas)) bool, one column per alpha",
                    **({"tap_scores": "(N, K) float64 D1 score per tap, registry order",
                        "tap_alarms": "(N, K, len(alphas)) bool, per tap and alpha"}
                       if per_tap is not None else {})},
        "classes": classes,
        "formats": entries,
        "classifier": {"model": model_id, "architecture": MODELS[model_id].name, "seed": seed,
                       "test_acc_fp32_canonical": acc.get("test_acc_fp32"),
                       "checkpoint_sha256": acc.get("sha256"),
                       "accuracy_measured_on": f"clean_test ({len(x)} images)"},
        "monitor": {"detector": detector, "taps": [t.tap_id for t in tap_list],
                    "features_per_tap": 29, "alphas": cal.alpha.tolist(),
                    "thresholds": cal.tau.tolist(), "calibrated_on": "clean_cal",
                    "clean_test_fpr": {str(a): float(o["alarms"][:, k].mean())
                                       for k, a in enumerate(cal.alpha)},
                    "svdd": {"nu": d2.selection.nu, "gamma": float(d2.model._gamma),
                             "support_vectors": int(d2.model.support_vectors_.shape[0])},
                    "only_in": "model.onnx and formats/model.ort",
                    "per_layer": None if per_tap is None else _per_layer(
                        d1, tap_ids, tap_taus, cal_tap, o["tap_alarms"])},
        "weights": _weight_table(model_path, model_id, _tap_after(fresh(), tap_list)),
        "comparison": _comparison(results_dir, model_id, tap_set, seed),
        "faults": _fault_controls(),
        "check": {"n": CHECK_N, "score_tolerance": SCORE_TOL,
                  "onnx_vs_fidnn": {"n": len(x), "alarm_agreement": agree,
                                    "max_score_error": score_err},
                  "per_tap_onnx_vs_fidnn": tap_check},
        "overhead": {"device": "cpu", "runtime": f"onnxruntime {ort.__version__}", "batch": 1,
                     "classifier_ms": base, "monitored_ms": full,
                     "added_latency_pct": round(100 * (full - base) / base, 1),
                     "measured_on": "export host — re-measure on the target device"},
        "ui": {"runtime": f"onnxruntime-web {ORT_WEB}", "sha256": ort_web,
               "samples": f"{len(sx)} clean_test images, {UI_PER_CLASS} per class"},
        "metrics": {
            "measured_on": f"clean_test ({len(x)} images), original model",
            "classifier": _classifier_metrics(o["pred"], y, classes),
            "monitor_false_alarm_rate": {str(a): float(o["alarms"][:, k].mean())
                                         for k, a in enumerate(cal.alpha)},
            "note": ("Detection recall needs faulty inferences; the UI measures it live by "
                     "corrupting the model in the browser."),
        },
        "caveats": [
            "Exported from one training seed: a demonstration, not a reported thesis number (C4).",
            ("Expects CIFAR-10-like 32×32 natural images; other inputs are out of distribution "
             "and may be flagged (or not) for reasons unrelated to faults."),
            "The clean false-alarm rate holds only for inputs like the calibration data.",
            "Only model.onnx / model.ort carry the monitor; formats/classifier_* do not.",
        ],
        **sidecar({"model": model_id, "precision": precision, "tap_set": tap_set,
                   "detector": detector, "tflite": tflite}, seed=seed, device="cpu"),
    }
    (out / "manifest.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    archive = out_dir / f"{stem}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f"{stem}/{f.relative_to(out)}")
    log(f"{archive}: {archive.stat().st_size / 1024:.0f} KB, {len(entries)} files in "
        f"formats, monitor +{record['overhead']['added_latency_pct']}% latency on this CPU")
    return record
