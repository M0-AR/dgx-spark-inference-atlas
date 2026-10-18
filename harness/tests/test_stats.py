"""Tests for the statistical machinery. Known-value checks are deliberate:
the harness must not judge model outputs with untested judges."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.bench import stats  # noqa: E402


def test_mcnemar_exact_known_value():
    r = stats.mcnemar_exact(10, 0)
    assert abs(r["p_value"] - 2 * (1 / 1024)) < 1e-12


def test_norm_ppf_known_value():
    assert abs(stats._norm_ppf(0.975) - 1.959964) < 1e-4


def test_chi2_sf_critical_values():
    # critical values at alpha=0.05 for df=1,2,4,5
    for df, crit in [(1, 3.841459), (2, 5.991465), (4, 9.487729), (5, 11.070498)]:
        p = stats._chi2_sf(crit, df)
        assert abs(p - 0.05) < 2e-3, (df, p)


def test_bootstrap_ci_covers_zero_for_symmetric_noise():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    r = stats.bootstrap_ci(x, n_boot=2000, seed=1)
    assert r["lo"] < 0 < r["hi"]


def test_paired_bootstrap_detects_shift():
    rng = np.random.default_rng(0)
    b = rng.normal(10, 1, 100)
    a = b + 0.5
    r = stats.paired_bootstrap_diff(a, b, n_boot=2000, seed=0)
    assert r["excludes_zero"] and r["lo"] > 0


def test_wilcoxon_perfect_shift_is_significant():
    x = list(range(20))
    y = [v - 1 for v in x]
    r = stats.wilcoxon_signed_rank(x, y)
    assert r["p_value"] < 0.001


def test_cliffs_delta_extremes():
    assert stats.cliffs_delta([3, 4, 5], [1, 2, 0])["delta"] == 1.0
    assert stats.cliffs_delta([1, 2, 0], [3, 4, 5])["delta"] == -1.0


def test_holm_bonferroni_ordering():
    out = stats.holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.001})
    assert out["c"]["p_adj"] <= out["a"]["p_adj"] <= out["b"]["p_adj"]
    assert out["b"]["p_adj"] <= 1.0


def test_tost_equivalence_identical_samples():
    x = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
    r = stats.tost_equivalence(x, x, margin=0.5, n_boot=1000, seed=0)
    assert r["equivalent"]


def test_changepoint_detects_known_shift():
    rng = np.random.default_rng(7)
    y = np.concatenate([rng.normal(10, 0.3, 20), rng.normal(5, 0.3, 20)])
    cps = stats.find_changepoints(y, alpha=0.01, n_perm=1000, seed=3,
                                  min_segment=3)
    assert any(abs(cp.index - 20) <= 2 for cp in cps)


def test_changepoint_no_false_positive_on_noise():
    rng = np.random.default_rng(11)
    y = rng.normal(0, 1, 60)
    cps = stats.find_changepoints(y, alpha=0.005, n_perm=2000, seed=5,
                                  min_segment=5)
    assert len(cps) == 0


def test_cv():
    assert abs(stats.coefficient_of_variation([1, 1, 1, 1]) - 0.0) < 1e-12
    assert math.isnan(stats.coefficient_of_variation([1.0]))
