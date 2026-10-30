"""Detector granularity: single-layer (D1), per-layer late fusion (D3), tap subsets, joint (D2).

SPEC §6.2 variants, the H2 test (§1) and §9.6(1)(2) in one comparison. Fit and calibration use
clean splits only (C2, C3); faults are scored, never fitted or used to choose a design.
"""

import itertools
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fidnn.detect import calibrate, svdd
from fidnn.detect.features import CleanFeatures, FaultFeatures, Features
from fidnn.detect.normalise import DELTA
from fidnn.detect.variants import D1, D2, D3
from fidnn.eval import metrics, splits
from fidnn.eval.ablations import greedy_tap_selection, subset
from fidnn.eval.localisation import tap_buckets
from fidnn.eval.stats import RESAMPLES
from fidnn.inject.targets import BUCKETS

GRANULARITIES = ("single", "per_layer", "subset", "joint")
COMBINERS = ("max", "mean", "fpr_weighted")
GREEDY_K = (2, 3, 4)
H2_ALPHA = 0.01  # SPEC §1: H2 is stated at TPR@1%FPR
LATENCY_REPS = {1: 300, 32: 100}


@dataclass
class Design:
    """One detector design. `oracle` marks exhaustive subsets: a map, never a method (C2)."""

    name: str
    granularity: str
    taps: list[str]
    rule: str
    oracle: bool = False
    detector: object = field(default=None, repr=False)
    tap_index: int | None = None  # single-layer designs score one column of the shared D1
    fit_s: float = float("nan")

    def scores(self, features: Features) -> np.ndarray:
        if self.granularity == "single":  # this tap's columns only: what a 1-tap monitor runs
            k, norm = self.tap_index, self.detector.normaliser
            z = (features.values[:, k, :] - norm.median[k]) / (norm.iqr[k] + DELTA)
            return svdd.score(self.detector.models[self.taps[0]], np.where(norm.keep[k], z, 0.0))
        if self.granularity in ("subset", "joint"):
            return self.detector.scores(subset(features, self.taps))
        return self.detector.scores(features)

    @property
    def n_svdd(self) -> int:
        return len(self.taps) if self.granularity == "per_layer" else 1

    @property
    def support_vectors(self) -> int:
        if self.granularity == "single":
            return int(self.detector.models[self.taps[0]].support_vectors_.shape[0])
        if self.granularity == "per_layer":
            return int(sum(m.support_vectors_.shape[0] for m in self.detector.d1.models.values()))
        return int(self.detector.model.support_vectors_.shape[0])


def _require_clean(*xs) -> None:
    if not all(isinstance(x, CleanFeatures) for x in xs):
        raise TypeError("designs are planned and fitted on clean features only (C2)")


def position_groups(taps: list[str]) -> dict[str, list[str]]:
    """§9.6(2) thirds by registry order, as `tap_position` folds them."""
    folded = tap_buckets(taps)
    return {b: [t for t, f in zip(taps, folded) if f == b] for b in BUCKETS}


def plan(clean_fit: CleanFeatures, clean_cal: CleanFeatures, seed: int = 0,
         alpha: float = H2_ALPHA, exhaustive: bool = False, log=print) -> list[Design]:
    """The fixed design list. Subsets come from structure or a clean criterion, never faults."""
    _require_clean(clean_fit, clean_cal)
    taps = list(clean_fit.taps)
    out = [Design(f"single:{t}", "single", [t], "per_tap", tap_index=k)
           for k, t in enumerate(taps)]
    out += [Design(f"per_layer:{c}", "per_layer", taps, c) for c in COMBINERS]
    named: dict[tuple, Design] = {}

    def add(name, tap_list, rule):
        key = tuple(sorted(tap_list, key=taps.index))
        if 1 < len(key) < len(taps) and key not in named:
            named[key] = Design(name, "subset", list(key), rule)

    for b, group in position_groups(taps).items():
        add(f"subset:{b}", group, "position")
    if "pooled" in taps and "logits" in taps:
        add("subset:backbone", [t for t in taps if t not in ("pooled", "logits")], "structure")
        add("subset:head", ["pooled", "logits"], "structure")
    for t in taps:
        add(f"subset:without_{t}", [x for x in taps if x != t], "leave_one_out")
    t0 = time.perf_counter()
    greedy = greedy_tap_selection(clean_fit, clean_cal, max(GREEDY_K), alpha, seed)
    log(f"greedy clean-FPR tap selection: {time.perf_counter() - t0:.0f} s")
    for k in GREEDY_K:
        row = greedy[greedy.k == k]
        if len(row):
            add(f"subset:greedy_k{k}", row.taps.iloc[0], "greedy_clean")
    out += list(named.values())
    if exhaustive:
        for k in range(2, len(taps)):
            for combo in itertools.combinations(taps, k):
                if combo not in named:
                    out.append(Design("subset:" + "+".join(combo), "subset", list(combo),
                                      "exhaustive", oracle=True))
    out.append(Design("joint:all", "joint", taps, "all"))
    return out


def fit(designs: list[Design], clean_fit: CleanFeatures, clean_cal: CleanFeatures,
        seed: int = 0, alpha: float = H2_ALPHA, log=print) -> list[Design]:
    """Fit every design on clean_fit, select ν/γ on clean_cal (§6.3). One D1 serves D1 and D3."""
    _require_clean(clean_fit, clean_cal)
    t0 = time.perf_counter()
    d1 = D1(seed=seed, alpha=alpha).fit(clean_fit, clean_cal)
    d1_s = time.perf_counter() - t0
    for i, d in enumerate(designs):
        t0 = time.perf_counter()
        if d.granularity == "single":
            d.detector, d.fit_s = d1, d1_s / len(clean_fit.taps)
            continue
        if d.granularity == "per_layer":
            d.detector = D3(d1, d.rule).fit(clean_cal)
            d.fit_s = d1_s + time.perf_counter() - t0
            continue
        d.detector = D2(seed=seed, alpha=alpha).fit(subset(clean_fit, d.taps),
                                                    subset(clean_cal, d.taps))
        d.fit_s = time.perf_counter() - t0
        if (i + 1) % 10 == 0:
            log(f"  fitted {i + 1}/{len(designs)}")
    return designs


def latency_ms(design: Design, features: Features, batch: int, reps: int, seed: int = 0) -> float:
    """Median per-image scoring time (normalise + SVDD + combine), one process, after warm-up."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(features) - batch, size=reps + 10)
    times = []
    for i, s in enumerate(starts):
        f = type(features)(features.values[s:s + batch], features.taps, features.names)
        t = time.perf_counter()
        design.scores(f)
        if i >= 10:
            times.append(time.perf_counter() - t)
    return float(np.median(times) * 1e3 / batch)


def auroc_contrib(fault_scores: np.ndarray, clean_scores: np.ndarray) -> np.ndarray:
    """Per-fault-sample share of clean scores below it (ties ½); its mean is the AUROC."""
    c = np.sort(clean_scores)
    below = np.searchsorted(c, fault_scores, side="left")
    ties = np.searchsorted(c, fault_scores, side="right") - below
    return (below + 0.5 * ties) / len(c)


def cluster_paired(a: np.ndarray, b: np.ndarray, clusters: np.ndarray,
                   resamples: int = RESAMPLES, confidence: float = 0.95, seed: int = 0) -> dict:
    """Paired bootstrap of mean(a − b) resampling injections, not probes.

    Probes of one injection share a corrupted model, so resampling probes would give CIs that are
    too narrow. The clean side is held fixed.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    ids, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(ids))
    counts = np.bincount(inv, minlength=len(ids)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(resamples, len(ids)))
    est = sums[draws].sum(1) / counts[draws].sum(1)
    lo, hi = np.quantile(est, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    detected = not (lo <= 0 <= hi)
    return {"n": len(d), "clusters": len(ids), "difference": float(d.mean()),
            "ci_low": float(lo), "ci_high": float(hi), "detected_difference": detected,
            "verdict": "difference" if detected else "no detected difference"}


def _first_alarm(design: Design, features: Features) -> np.ndarray:
    return design.detector.first_alarm_tap(features)


def evaluate(designs: list[Design], clean: dict[str, CleanFeatures],
             fault: FaultFeatures | None, labels: pd.DataFrame | None, alphas, seed: int = 0,
             log=print) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Rows per (design, track, α), paired differences against the joint SVDD, localisation."""
    joint = next(d for d in designs if d.granularity == "joint")
    held = None if labels is None else (labels.fault_split == "fault_test").to_numpy()
    cache, rows = {}, []
    for d in designs:
        cal = d.scores(clean["clean_cal"])
        taus = calibrate.thresholds(cal, alphas)
        test = d.scores(clean["clean_test"])
        meta = {"design": d.name, "granularity": d.granularity, "taps": "+".join(d.taps),
                "k": len(d.taps), "rule": d.rule, "oracle": d.oracle, "n_svdd": d.n_svdd,
                "support_vectors": d.support_vectors, "fit_s": d.fit_s,
                "latency_ms_b1": float("nan") if d.oracle else
                latency_ms(d, clean["clean_test"], 1, LATENCY_REPS[1], seed),
                "latency_ms_b32": float("nan") if d.oracle else
                latency_ms(d, clean["clean_test"], 32, LATENCY_REPS[32], seed)}
        if fault is None:
            for a, tau in taus.items():
                rows.append(meta | {"track": None, "alpha": a, "tau": tau,
                                    "measured_fpr": float((test > tau).mean())})
            continue
        fs = d.scores(fault)[held]
        cache[d.name] = (fs, test, taus)
        scored = labels[held].assign(score=fs)
        rows += [meta | r for r in metrics.by_track(scored, test, taus).to_dict("records")]
    table = pd.DataFrame(rows)
    if fault is None:
        return table, pd.DataFrame(), {}

    lab = labels[held]
    paired = []
    jf, jt, jtaus = cache[joint.name]
    for d in designs:
        if d is joint or d.oracle:
            continue
        fs, test, taus = cache[d.name]
        for track in metrics.TRACKS:
            m = (lab.track == track).to_numpy()
            cl = lab.injection_id.to_numpy()[m]
            res = cluster_paired(auroc_contrib(fs[m], test), auroc_contrib(jf[m], jt), cl,
                                 seed=seed)
            paired.append({"design": d.name, "versus": joint.name, "track": track,
                           "metric": "auroc", **res})
            for a in alphas:
                res = cluster_paired(fs[m] > taus[a], jf[m] > jtaus[a], cl, seed=seed)
                paired.append({"design": d.name, "versus": joint.name, "track": track,
                               "metric": f"tpr@{a:g}", **res})
    paired = pd.DataFrame(paired)

    loc = {}
    for d in designs:
        if d.granularity != "per_layer" or d.rule != "max":
            continue
        first = _first_alarm(d, fault)[held]
        folded = tap_buckets(d.taps)
        alarm = np.array([folded[i] if i >= 0 else "none" for i in first])
        for track in metrics.TRACKS:
            m = (lab.track == track).to_numpy() & (alarm != "none")
            hit = float((alarm[m] == lab.bucket.to_numpy()[m]).mean()) if m.any() else math.nan
            loc[track] = {"alarmed": int(m.sum()), "first_alarm_in_injected_region": hit,
                          "chance": 1 / len(BUCKETS),
                          "first_alarm_tap_share": {t: float((first[m] == k).mean())
                                                    for k, t in enumerate(d.taps)}}
    return table, paired, loc


def h2_verdict(table: pd.DataFrame, paired: pd.DataFrame, alpha: float = H2_ALPHA) -> list[dict]:
    """SPEC §1 H2: rejected if fused TPR@1% ≤ best single-layer TPR@1% within CI, per track.

    "Best single layer" is picked on the same fault data it is compared on, which favours the
    single-layer side; that makes this a conservative test of H2, never a design selection.
    """
    out = []
    for track in metrics.TRACKS:
        t = table[(table.track == track) & (table.alpha == alpha)]
        single = t[t.granularity == "single"]
        joint = t[t.granularity == "joint"]
        if single.empty or joint.empty:
            continue
        best = single.loc[single.tpr.idxmax()]
        p = paired[(paired.design == best.design) & (paired.track == track)
                   & (paired.metric == f"tpr@{alpha:g}")].iloc[0]
        # paired is single − joint: joint wins only if the whole CI is below zero
        holds = p.ci_high < 0
        out.append({"track": track, "best_single": best.design, "best_single_tpr": best.tpr,
                    "joint_tpr": float(joint.tpr.iloc[0]), "diff_single_minus_joint": p.difference,
                    "ci_low": p.ci_low, "ci_high": p.ci_high,
                    "h2": "supported" if holds else "rejected (fused not better within CI)"})
    return out


def overhead_by_k(overhead_dir: Path, model_id: str, seed: int) -> list[dict]:
    """Feature-extraction cost per K from the M-6 record: detection is read with cost (C5)."""
    p = overhead_dir / f"{model_id}_fp32_extended_seed{seed}_overhead.parquet"
    if not p.exists():
        return []
    o = pd.read_parquet(p)
    o = o[(o.device == "cpu") & (o.batch == 1)]
    base = o[o.stage == "model_only"].median_ms
    if base.empty:
        return []
    f = o[o.stage == "features"]
    return [{"k": int(r.k), "model_ms": float(base.iloc[0]), "features_added_ms": float(r.added_ms)}
            for r in f.itertuples()]


def run(model_id: str, precision: str, tap_set: str, seed: int, cfg_path: Path,
        features_dir: Path, faults_dir: Path, out_dir: Path, overhead_dir: Path,
        exhaustive: bool = False, log=print) -> dict:
    from fidnn.detect.run import _clean_splits, _fault_features
    from fidnn.provenance import sidecar
    from fidnn.taps.registry import taps

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    tap_ids = [t.tap_id for t in taps(model_id, tap_set)]
    stem = f"{model_id}_{precision}_{tap_set}_seed{seed}"
    clean = _clean_splits(features_dir, stem, tap_ids)
    faults = _fault_features(features_dir, faults_dir, model_id, precision, seed, tap_ids)
    fault, labels = (None, None) if faults is None else faults
    if labels is not None:
        labels = labels.assign(fault_split=splits.assign(labels, seed=seed),
                               is_gen=splits.gen_mask(labels))
    designs = plan(clean["clean_fit"], clean["clean_cal"], seed, cfg["alpha"], exhaustive, log)
    log(f"{stem}: {len(designs)} designs")
    fit(designs, clean["clean_fit"], clean["clean_cal"], seed, cfg["alpha"], log)
    table, paired, loc = evaluate(designs, clean, fault, labels, cfg["alphas"], seed, log)
    table = table.assign(model=model_id, precision=precision, tap_set=tap_set, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / f"{stem}_granularity.parquet", index=False)
    if len(paired):
        paired.assign(model=model_id, precision=precision, seed=seed).to_parquet(
            out_dir / f"{stem}_granularity_paired.parquet", index=False)
    record = {"model": model_id, "precision": precision, "tap_set": tap_set, "seed": seed,
              "designs": len(designs), "detection": fault is not None,
              "fault_test_probes": None if labels is None
              else int((labels.fault_split == "fault_test").sum()),
              "h2": h2_verdict(table, paired) if len(paired) else None,
              "localisation_per_layer_max": loc,
              "feature_cost_by_k": overhead_by_k(overhead_dir, model_id, seed),
              "thesis_result": False if model_id == "m1" else None,
              **sidecar(cfg, seed=seed, device="cpu")}
    (out_dir / f"{stem}_granularity.json").write_text(json.dumps(record, indent=2, default=str),
                                                     encoding="utf-8")
    log(f"{stem}: wrote {len(table)} rows" + ("" if fault is not None else
                                              " (no fault features: clean side only)"))
    return record


def _pct(v) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.1f} %"


def write_report(results_dir: Path, out: Path, model_id: str = "m2", tap_set: str = "default",
                 seed: int = 0) -> None:
    """docs/M5_granularity.md from the granularity parquet/JSON of every precision present."""
    from fidnn.tables import md

    parts = []
    for prec in ("fp32", "int8"):
        stem = f"{model_id}_{prec}_{tap_set}_seed{seed}"
        p = results_dir / f"{stem}_granularity.parquet"
        if not p.exists():
            continue
        t = pd.read_parquet(p)
        rec = json.loads((results_dir / f"{stem}_granularity.json").read_text(encoding="utf-8"))
        t = t[~t.oracle]
        parts.append(f"## {prec.upper()}\n")
        if not rec["detection"]:
            fpr = t.pivot_table(index=["granularity", "design"], columns="alpha",
                                values="measured_fpr").reset_index()
            parts.append("No fault features for this precision, so only the clean side is "
                         "measured: false-alarm rate on `clean_test` and scoring cost.\n\n"
                         + md(fpr.merge(t.drop_duplicates("design")[
                             ["design", "support_vectors", "latency_ms_b1"]], on="design"),
                             floatfmt=".4f") + "\n")
            continue
        for track in metrics.TRACKS:
            sel = t[(t.track == track)]
            wide = sel.pivot_table(index=["granularity", "design", "k"], columns="alpha",
                                   values="tpr").reset_index()
            wide.columns = [c if isinstance(c, str) else f"TPR@{c:g}" for c in wide.columns]
            extra = sel.drop_duplicates("design").set_index("design")[
                ["auroc", "aupr", "support_vectors", "latency_ms_b1"]]
            wide = wide.join(extra, on="design").sort_values(
                ["granularity", "auroc"], ascending=[True, False])
            n = int(sel.n_fault.iloc[0])
            parts.append(f"### Track {track} ({n:,} fault_test probes)\n\n"
                         + md(wide, floatfmt=".4f") + "\n")
        pp = results_dir / f"{stem}_granularity_paired.parquet"
        if pp.exists():
            pr = pd.read_parquet(pp)
            pr = pr[pr.metric.isin(["auroc", f"tpr@{H2_ALPHA:g}"])]
            parts.append("### Paired differences against the joint SVDD (design − joint)\n\n"
                         "Bootstrap over injection instances (probes of one injection share a "
                         "corrupted model); the clean side is held fixed. A CI that includes 0 "
                         "is no detected difference.\n\n"
                         + md(pr[["design", "track", "metric", "difference", "ci_low", "ci_high",
                                  "verdict"]], floatfmt=".4f") + "\n")
        if rec.get("h2"):
            parts.append("### H2 (SPEC §1)\n\n" + md(pd.DataFrame(rec["h2"]), floatfmt=".4f")
                         + "\n\nThe best single layer is picked on the evaluation data, which "
                         "favours the single-layer side, so this is a conservative test of H2.\n")
        loc = rec.get("localisation_per_layer_max") or {}
        if loc:
            rows = [{"track": k, "alarmed": v["alarmed"],
                     "first alarm in injected region": _pct(v["first_alarm_in_injected_region"]),
                     "chance": _pct(v["chance"])} for k, v in loc.items()]
            parts.append("### Localisation by the per-layer detector (first alarm, max)\n\n"
                         + md(pd.DataFrame(rows)) + "\n")
        cost = rec.get("feature_cost_by_k") or []
        if cost:
            parts.append("### Cost of observing K taps (M-6, FP32, CPU, batch 1)\n\n"
                         + md(pd.DataFrame(cost), floatfmt=".3f") + "\n")
    if not parts:
        raise FileNotFoundError(f"no {model_id}_*_granularity.parquet under {results_dir}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""# Detector granularity — single-layer vs per-layer vs subset vs joint SVDD

**Model:** {model_id.upper()} · **Tap set:** {tap_set} · **Seed:** {seed}

> **One seed: not reportable as a thesis number (C4).** Re-run with ≥ 5 seeds and aggregate.

| Granularity | SPEC | What it is |
|---|---|---|
| single | §6.2 D1 | one SVDD on one tap, judged on its own |
| per_layer | §6.2 D3 | one SVDD per tap, each score through its clean_cal CDF, then combined |
| subset | §9.6(1)(2) | one fused SVDD over a group of taps (position, leave-one-out, greedy) |
| joint | §6.2 D2 | one fused SVDD over all taps |

Every design is fitted on `clean_fit`, ν/γ chosen on `clean_cal`, thresholds at fixed α from
`clean_cal` (C2, C3). No design is chosen with fault data; exhaustive subsets, when run, are kept
out of these tables. Latency is normalise + SVDD + combine per image, batch 1, one process.

{chr(10).join(parts)}""", encoding="utf-8")
    print(f"[granularity] wrote {out}")
