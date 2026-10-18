"""Statistical tooling for the atlas study.

Implements the machinery required by docs/STATISTICS.md and the pre-registration:

  * Percentile bootstrap CIs (paired and unpaired).
  * McNemar exact / asymptotic tests for per-sample degradation detection
    (per arXiv:2602.10144, "When LLMs get significantly worse").
  * Wilcoxon signed-rank (normal approximation with tie correction).
  * Friedman + Nemenyi for multi-condition ranks (bpw ladder across models).
  * Holm-Bonferroni family-wise correction.
  * Cliff's delta effect size with Romano et al. thresholds.
  * TOST equivalence (bootstrap CI against a pre-declared margin).
  * Single-changepoint detection via permutation test + binary segmentation,
    for locating quantization "cliffs" (arXiv:2608.06564 margin shrinkage,
    arXiv:2606.08862 computation-collapse regimes).
  * Coefficient of variation for sequential-isolation protocol compliance.

All functions are deterministic given a seed. numpy is the only dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

def bootstrap_ci(
    x: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for a statistic of a single sample."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        raise ValueError("empty sample")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    stats = np.apply_along_axis(statistic, 1, arr[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(statistic(arr)),
        "lo": float(lo),
        "hi": float(hi),
        "alpha": alpha,
        "n_boot": n_boot,
        "n": int(arr.size),
    }


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap CI for mean(a - b) under pairing. a, b must be exchangeable pairs."""
    aa, bb = np.asarray(a, float), np.asarray(b, float)
    if aa.shape != bb.shape:
        raise ValueError("paired samples must have equal length")
    d = aa - bb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean_diff": float(d.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "alpha": alpha,
        "n_boot": n_boot,
        "n_pairs": int(d.size),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


# --------------------------------------------------------------------------- #
# Paired tests
# --------------------------------------------------------------------------- #

def mcnemar_exact(b: int, c: int) -> dict[str, float]:
    """Exact McNemar test on discordant pair counts.

    b: baseline correct, candidate wrong (degradation events)
    c: baseline wrong, candidate correct (repair events)
    Two-sided p via binomial(n=b+c, p=0.5).
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}
    k = min(b, c)
    # two-sided exact p = 2 * P(X <= k), capped at 1
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2.0 * tail)
    return {"b": b, "c": c, "n_discordant": n, "p_value": p}


def wilcoxon_signed_rank(
    x: Sequence[float], y: Sequence[float]
) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank test, normal approximation with tie
    correction. Suitable for >= 10 pairs; exact tables are not implemented.
    """
    xx, yy = np.asarray(x, float), np.asarray(y, float)
    if xx.shape != yy.shape:
        raise ValueError("paired samples must have equal length")
    d = xx - yy
    d = d[d != 0]
    if d.size == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_nonzero": 0}
    ranks = _average_ranks(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    n = d.size
    # tie correction
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = float(sum(t ** 3 - t for t in counts))
    mu = n * (n + 1) / 4
    sigma2 = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    if sigma2 <= 0:
        return {"statistic": w_plus, "p_value": 1.0, "n_nonzero": int(n)}
    z = (w_plus - mu) / math.sqrt(sigma2)
    p = 2 * (1 - _norm_cdf(abs(z)))
    return {"statistic": w_plus, "z": float(z), "p_value": float(p),
            "n_nonzero": int(n)}


def _average_ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty_like(v, dtype=float)
    sv = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and sv[j + 1] == sv[i]:
            j += 1
        avg = (i + j) / 2 + 1
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# --------------------------------------------------------------------------- #
# Multi-condition tests
# --------------------------------------------------------------------------- #

def friedman_test(samples: np.ndarray) -> dict[str, float]:
    """Friedman test. samples shape (n_blocks, k_conditions)."""
    m = np.asarray(samples, float)
    n, k = m.shape
    if n < 2 or k < 3:
        raise ValueError("need n>=2 blocks and k>=3 conditions")
    ranks = np.apply_along_axis(_average_ranks, 1, m)
    rj = ranks.sum(axis=0)
    # tie correction
    stat = (12.0 / (n * k * (k + 1))) * float(np.sum(rj ** 2)) - 3 * n * (k + 1)
    tie_sum = 0.0
    for i in range(n):
        _, counts = np.unique(m[i], return_counts=True)
        tie_sum += float(sum(t ** 3 - t for t in counts))
    denom = 1.0 - tie_sum / (n * (k ** 3 - k)) if (k ** 3 - k) else 1.0
    if denom > 0:
        stat /= denom
    # chi-square approximation with k-1 dof
    p = _chi2_sf(stat, k - 1)
    return {"chi2": float(stat), "df": k - 1, "p_value": float(p),
            "mean_ranks": rj.tolist()}


def _chi2_sf(x: float, k: int) -> float:
    """Survival function of chi-square with integer k via series (k small)."""
    if x <= 0:
        return 1.0
    if k % 2 == 0:
        # Q(x,k) = exp(-x/2) * sum_{j=0}^{k/2-1} (x/2)^j / j!
        half = k // 2
        s = sum((x / 2) ** j / math.factorial(j) for j in range(half))
        return float(min(1.0, math.exp(-x / 2) * s))
    # odd k: use erf-based recursion from k=1
    # Q(x,1) = 2*(1-Phi(sqrt(x)))
    p = 2 * (1 - _norm_cdf(math.sqrt(x)))
    j = 1
    # recursion: Q(x, j+2) = Q(x, j) + (x/2)^{j/2} e^{-x/2} / Gamma(j/2 + 1)
    while j < k:
        term = math.exp(-x / 2) * (x / 2) ** (j / 2) / math.gamma(j / 2 + 1)
        p += term
        j += 2
    return float(min(1.0, p))


# --------------------------------------------------------------------------- #
# Correction + effect sizes
# --------------------------------------------------------------------------- #

def holm_bonferroni(pvals: dict[str, float]) -> dict[str, dict[str, float]]:
    """Holm-Bonferroni step-down correction. Returns name -> {p, p_adj, rank}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, float]] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items, start=1):
        adj = min(1.0, max(running, (m - rank + 1) * p))
        running = adj
        out[name] = {"p": p, "p_adj": adj, "rank": rank}
    return out


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> dict[str, float | str]:
    """Cliff's delta (dominance) effect size, with Romano et al. magnitude."""
    xx, yy = np.asarray(x, float), np.asarray(y, float)
    gt = sum(1 for a in xx for b in yy if a > b)
    lt = sum(1 for a in xx for b in yy if a < b)
    n = xx.size * yy.size
    d = (gt - lt) / n if n else 0.0
    ad = abs(d)
    if ad < 0.147:
        mag = "negligible"
    elif ad < 0.33:
        mag = "small"
    elif ad < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return {"delta": float(d), "magnitude": mag}


def tost_equivalence(
    a: Sequence[float],
    b: Sequence[float],
    *,
    margin: float,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float | bool]:
    """Two one-sided tests via paired bootstrap CI inside +/- margin.

    Equivalent iff the (1-2*alpha) CI of mean(a-b) lies within [-m, +m].
    Pairing: a[i] and b[i] must be the same item under two conditions.
    """
    pb = paired_bootstrap_diff(a, b, n_boot=n_boot, alpha=alpha * 2, seed=seed)
    equivalent = (-margin < pb["lo"]) and (pb["hi"] < margin)
    return {"mean_diff": pb["mean_diff"], "lo": pb["lo"], "hi": pb["hi"],
            "margin": margin, "equivalent": bool(equivalent)}


# --------------------------------------------------------------------------- #
# Changepoints (quantization cliffs)
# --------------------------------------------------------------------------- #

@dataclass
class Changepoint:
    index: int
    level_left: float
    level_right: float
    stat: float
    p_value: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def single_changepoint(
    y: Sequence[float],
    *,
    n_perm: int = 2_000,
    seed: int = 0,
    min_segment: int = 2,
) -> Changepoint:
    """Permutation test for one mean-shift changepoint.

    Test statistic: max over candidate splits of |mean_left - mean_right|
    (pooled normalisation). p-value from a permutation null that shuffles y.
    """
    arr = np.asarray(y, float)
    n = arr.size
    if n < 2 * min_segment:
        raise ValueError("series too short for changepoint test")
    rng = np.random.default_rng(seed)

    def best_stat(v: np.ndarray) -> tuple[float, int, float, float]:
        best = (-1.0, -1, 0.0, 0.0)
        total = v.sum()
        for k in range(min_segment, n - min_segment + 1):
            left = v[:k]
            right = v[k:]
            ml, mr = left.mean(), right.mean()
            s = abs(ml - mr) * math.sqrt(k * (n - k) / n)
            if s > best[0]:
                best = (s, k, ml, mr)
        return best

    stat, k, ml, mr = best_stat(arr)
    # centred permutation null
    centred = arr - arr.mean()
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(centred) + arr.mean()
        s, _, _, _ = best_stat(perm)
        if s >= stat:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return Changepoint(index=int(k), level_left=float(ml),
                       level_right=float(mr), stat=float(stat), p_value=float(p))


def find_changepoints(
    y: Sequence[float],
    *,
    alpha: float = 0.01,
    n_perm: int = 2_000,
    seed: int = 0,
    min_segment: int = 3,
) -> list[Changepoint]:
    """Binary segmentation: recurse while a significant changepoint is found."""
    arr = np.asarray(y, float)
    found: list[Changepoint] = []

    def recurse(lo: int, hi: int, depth: int) -> None:
        seg = arr[lo:hi]
        if seg.size < 2 * min_segment:
            return
        cp = single_changepoint(seg, n_perm=n_perm, seed=seed + depth,
                                min_segment=min_segment)
        if cp.p_value <= alpha:
            found.append(Changepoint(index=lo + cp.index,
                                     level_left=cp.level_left,
                                     level_right=cp.level_right,
                                     stat=cp.stat, p_value=cp.p_value))
            recurse(lo, lo + cp.index, depth + 1)
            recurse(lo + cp.index, hi, depth + 1)

    recurse(0, arr.size, 0)
    return sorted(found, key=lambda c: c.index)


# --------------------------------------------------------------------------- #
# Protocol compliance
# --------------------------------------------------------------------------- #

def coefficient_of_variation(x: Sequence[float]) -> float:
    """CV = sd/mean. Sequential-isolation compliance target: <= 0.10 (see docs)."""
    arr = np.asarray(x, float)
    if arr.size < 2 or arr.mean() == 0:
        return float("nan")
    return float(arr.std(ddof=1) / arr.mean())


def minimum_detectable_degradation_mcnemar(
    n_items: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    p_discordant: float = 1.0,
) -> float:
    """Minimum detectable accuracy difference for McNemar on n paired items.

    Normal-approximation MDE in the difference of paired proportions:
    delta = (z_{1-alpha/2} + z_{power}) * sqrt(p_discordant / n_items),
    Connors (1987) / the standard power formula for the paired
    (McNemar) case, as implemented by e.g. Stata's power pairedproportions.
    'p_discordant' is the anticipated share of discordant pairs; the default
    1.0 gives the conservative worst case. Values are reported as accuracy
    differences (proportions), matching docs/STATISTICS.md section 6.
    """
    if n_items < 1:
        raise ValueError("n_items must be >= 1")
    if not 0.0 < p_discordant <= 1.0:
        raise ValueError("p_discordant must be in (0, 1]")
    z_a = _norm_ppf(1 - alpha / 2)
    z_b = _norm_ppf(power)
    return float((z_a + z_b) * math.sqrt(p_discordant / n_items))


def _norm_ppf(p: float) -> float:
    """Acklam's rational approximation for the standard normal quantile."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
