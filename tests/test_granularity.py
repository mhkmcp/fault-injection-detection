"""fidnn.eval.granularity: designs come from clean data, metrics and bootstrap behave."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from fidnn.detect.features import BLOCK_E, CleanFeatures, FaultFeatures
from fidnn.eval import granularity as g
from fidnn.taps.features import FEATURE_NAMES

TAPS = ["stem", "stage1", "stage2", "stage3", "pooled", "logits"]
NAMES = FEATURE_NAMES + BLOCK_E


def _feats(kind, n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, len(TAPS), len(NAMES)))
    v[:, 3, :] += shift  # a fault that only stage3 sees
    return kind(v, TAPS, NAMES)


@pytest.fixture(scope="module")
def fitted():
    fit, cal = _feats(CleanFeatures, 300, seed=1), _feats(CleanFeatures, 200, seed=2)
    designs = g.plan(fit, cal, seed=0, alpha=0.05, log=lambda *_: None)
    return g.fit(designs, fit, cal, seed=0, alpha=0.05, log=lambda *_: None), fit, cal


def test_plan_covers_every_granularity_once_per_name(fitted):
    designs, *_ = fitted
    names = [d.name for d in designs]
    assert len(names) == len(set(names))
    assert {d.granularity for d in designs} == set(g.GRANULARITIES)
    assert [d.taps[0] for d in designs if d.granularity == "single"] == TAPS
    assert all(1 < len(d.taps) < len(TAPS) for d in designs if d.granularity == "subset")
    assert not any(d.oracle for d in designs)


def test_plan_and_fit_refuse_fault_features():
    clean, fault = _feats(CleanFeatures, 50), _feats(FaultFeatures, 50)
    with pytest.raises(TypeError):
        g.plan(fault, clean, log=lambda *_: None)
    with pytest.raises(TypeError):
        g.fit([], clean, fault)


def test_single_layer_design_is_the_d1_column(fitted):
    designs, fit, _ = fitted
    single = next(d for d in designs if d.name == "single:stage3")
    np.testing.assert_allclose(single.scores(fit), single.detector.scores(fit)[:, 3])


def test_auroc_contrib_mean_is_auroc():
    rng = np.random.default_rng(0)
    clean, fault = rng.normal(size=400), rng.normal(0.7, size=300).round(1)
    y = np.r_[np.ones(300), np.zeros(400)]
    assert g.auroc_contrib(fault, clean).mean() == pytest.approx(
        roc_auc_score(y, np.r_[fault, clean]))


def test_cluster_bootstrap_widens_with_clustering():
    rng = np.random.default_rng(0)
    per_cluster = rng.normal(size=40)
    clusters = np.repeat(np.arange(40), 50)
    a = per_cluster[clusters] + rng.normal(scale=0.1, size=len(clusters))
    b = np.zeros_like(a)
    clustered = g.cluster_paired(a, b, clusters, resamples=500)
    naive = g.cluster_paired(a, b, np.arange(len(a)), resamples=500)
    assert clustered["clusters"] == 40
    assert clustered["ci_high"] - clustered["ci_low"] > 3 * (naive["ci_high"] - naive["ci_low"])


def test_evaluate_and_h2(fitted):
    designs, fit, cal = fitted
    test = _feats(CleanFeatures, 200, seed=3)
    fault = _feats(FaultFeatures, 240, shift=3.0, seed=4)
    labels = pd.DataFrame({"track": np.where(np.arange(240) % 2, "S", "H"),
                           "injection_id": np.arange(240) // 8,
                           "bucket": "middle", "fault_split": "fault_test"})
    table, paired, loc = g.evaluate(designs, {"clean_cal": cal, "clean_test": test}, fault,
                                    labels, [0.05], seed=0, log=lambda *_: None)
    assert set(table.track) == {"S", "H"}
    best = table[(table.granularity == "single") & (table.track == "H")].sort_values("tpr")
    assert best.design.iloc[-1] == "single:stage3"
    assert set(paired.metric) == {"auroc", "tpr@0.05"}
    verdict = g.h2_verdict(table, paired, alpha=0.05)
    assert {v["track"] for v in verdict} == {"S", "H"}
    assert set(loc) == {"S", "H"}
