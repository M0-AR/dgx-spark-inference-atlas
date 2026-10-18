# Statistics — implementation, validation, and limitations

All procedures live in `harness/bench/stats.py` (numpy-only; no scipy
dependency, chosen so the harness runs on minimal containers and the Spark
kit). This document states exactly what each does, what was validated by
tests with known values (`harness/tests/test_stats.py`), and what is *not*
guaranteed.

**Rule of the atlas: the harness must not judge model outputs with untested
judges.** Every procedure used for a README claim either has a known-value
test in `test_stats.py` or is reported as descriptive only.

---

## 1. Interval estimation

| Function | Use | Method | Validation |
|----------|-----|--------|-----------|
| `bootstrap_ci` | CIs for C_outcome rate, tok/s, joules/token | percentile bootstrap, 10k resamples, default seed 0 (seed always recorded) | coverage sanity in tests; deterministic given seed |
| `paired_bootstrap_diff` | CI for paired condition differences | resample *episodes in pairs*, preserving the pairing | tested for symmetric zero-difference case |

Bootstrap CIs on rare-event proportions (C_outcome near 0 or 1) are known to
be optimistic; when the observed rate is 0 or 1 we additionally report the
exact rule-of-three bound and say so in the README.

## 2. Paired significance (agentic degradations)

| Function | Use | Method | Validation |
|----------|-----|--------|-----------|
| `mcnemar_exact(b, c)` | condition vs FP16 on the same episodes | exact binomial two-sided p over discordant pairs; `b`/`c` are the discordance counts | test suite checks the known value: b=1, c=9 → p=0.0215; symmetric b=c → p=1.0 |

McNemar exact is the pre-registered primary test (anchor: arXiv:2602.10144,
"McNemar's test for LLM evaluation"): it only sees disagreement and makes no
distributional assumption. Its power limit is documented in
`minimum_detectable_degradation_mcnemar` and reported per condition, so
non-significance is never mistaken for equivalence.

## 3. Equivalence (claiming "4 bpw preserves capability")

`tost_equivalence(diffs, margin)` — two one-sided tests on the bootstrap
distribution of the paired difference, pre-declared margin ±5 pp for
C_outcome (METHODOLOGY §5). We only ever write "preserves / equivalent" when
TOST passes; "no significant difference" is reported as exactly that
sentence, with the CI.

## 4. Rank consistency across the ladder

- `friedman_test` (implemented via the standard statistic with
  `_chi2_sf` survival function; validated against critical values
  χ²(0.05; df=1,2,4,5) = 3.841, 5.991, 9.488, 11.070 within 1e-6).
- Post-hoc ordering is reported descriptively with `cliffs_delta` effect
  sizes (Romano thresholds: negligible <0.147, small <0.33, medium <0.474,
  large otherwise), *not* as pairwise tests, to stay within the Holm family.

## 5. Changepoints on the bit-width axis

`single_changepoint` / `find_changepoints`: binary segmentation with a
permutation test (default 2000 permutations, seed 0) on the difference of
segment means. Used only for *where* series bend (KLD, C_outcome, tok/s,
joules/token), α=0.01, and results are phrased as pattern locations.
Known limitation: with 4–6 ladder points, a changepoint is a statement about
*this* ladder, not about the underlying continuum; the README says so.

## 6. Measurement-quality gates

- `coefficient_of_variation` flags any condition whose (sample) CV of decode
  tok/s exceeds 0.10 — the pre-registered isolation gate (H5). Flagged
  conditions still appear in tables, marked `†`, with a footnote; they are
  excluded from changepoint fitting.
- Percentiles use the linear-interpolation definition on sorted samples and
  are computed from committed per-request records only.

## 7. Energy accounting caveats (documented, not statistical)

- NVML `nvmlDeviceGetTotalEnergyConsumption` is a monotone counter; we
  difference it per condition and divide by generated tokens. Known
  caveats: counter granularity varies by driver; on some systems it is
  whole-device (not per-process); MIG is not supported here. We therefore
  (a) report idle-baseline-subtracted and raw values, (b) treat joule/token
  as an estimate with CI over repetitions, and (c) never convert it into
  cost claims. Anchor: NVIDIA's own measurement caveats + Watt Counts
  (arXiv:2604.09048).

## 8. Multiplicity policy

Family H1–H5: Holm–Bonferroni at α=0.05 (`holm_bonferroni`), applied to the
set of primary p-values (McNemar comparisons + changepoint permutation
p-values). Exploratory tables (latency percentiles, per-tool breakdowns) are
marked *descriptive* and carry no p-values, sidestepping
p-hacking-by-table-spelunking.

## 9. Reproducibility of the statistics themselves

- Every stochastic procedure takes an explicit seed; the default seed 0 is
  what the README tables were generated with; re-running `scripts/sync_readme.py`
  reproduces identical numbers (bit-for-bit) from the committed raw records.
- `test_stats.py` is run in CI (`make test`); 12 known-value tests cover the
  exact tests (McNemar), chi-square survival values, bootstrap determinism,
  changepoint detection on a synthetic step vs pure noise, CV, Cliff's delta
  extremes (±1), and TOST behaviour on identical samples.
