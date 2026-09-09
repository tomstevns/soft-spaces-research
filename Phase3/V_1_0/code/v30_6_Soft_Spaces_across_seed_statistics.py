#!/usr/bin/env python3
"""Soft Spaces Phase 3.1 v30.6 — across-seed REAL-vs-NULL correlation statistics.

Purpose
-------
v30.5 decomposed the ±E symmetry into REAL, NULL and REAL-minus-NULL components
for 12 frozen 11-term Hamiltonian seeds.

Observed v30.5 pattern:
* corr_REAL is near 1 for almost every seed.
* corr_NULL is also very high.
* delta_corr = corr_REAL - corr_NULL is positive in 11/12 seeds.
* seedwise permutation p-values are mostly not individually significant.

v30.6 asks whether the small REAL-over-NULL advantage is reproducible across
the frozen seed ensemble.

No physical model, Hamiltonian, perturbation, eigenspace or score calculation
is changed here.  This file is a pure statistical summary of the frozen v30.5
results.

Tests
-----
1. Exact sign test for delta_corr > 0.
2. Wilcoxon signed-rank test of delta_corr against zero.
3. Bootstrap confidence intervals for mean(delta_corr) and median(delta_corr).
4. Fisher-z transformed paired comparison between corr_REAL and corr_NULL.
5. Simple effect-size summaries.

Decision principle
------------------
A reproducible REAL-specific component requires consistent evidence across
multiple aggregate tests; no single p-value is treated as decisive.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


VERSION = "v30.6"

# Frozen values copied exactly from v30.5 output.
SEEDS = np.asarray([
    25042000, 25042001, 25042002, 25042003,
    25042004, 25042005, 25042006, 25042007,
    25042008, 25042009, 25042010, 25042011,
], dtype=int)

CORR_REAL = np.asarray([
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    0.981015,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    0.994540,
], dtype=float)

CORR_NULL = np.asarray([
    0.848158,
    0.990203,
    0.963691,
    0.826947,
    0.946301,
    0.991837,
    0.925965,
    0.972101,
    0.948696,
    0.984471,
    0.982882,
    0.988715,
], dtype=float)

DELTA = CORR_REAL - CORR_NULL

BOOTSTRAPS = 200_000
BOOTSTRAP_SEED = 30_006_001
EPS_CORR = 1.0e-12


def exact_sign_test_positive(values: np.ndarray) -> tuple[int, int, float, float]:
    """Return positive count, nonzero count, one-sided p, two-sided p."""
    nonzero = values[np.abs(values) > 0.0]
    n = nonzero.size
    k = int(np.sum(nonzero > 0.0))

    if n == 0:
        return 0, 0, float("nan"), float("nan")

    def binom_prob(i: int) -> float:
        return math.comb(n, i) / (2 ** n)

    p_one = sum(binom_prob(i) for i in range(k, n + 1))

    lower = min(k, n - k)
    p_two = min(
        1.0,
        2.0 * sum(binom_prob(i) for i in range(0, lower + 1))
    )

    return k, n, float(p_one), float(p_two)


def wilcoxon_signed_rank_positive(values: np.ndarray) -> tuple[float, float, float]:
    """Exact-ish Wilcoxon via sign enumeration for n<=20, handling no ties in abs ranks.

    Returns:
      W_plus, one-sided p (greater), two-sided p.
    """
    values = values[np.abs(values) > 0.0]
    n = values.size

    if n == 0:
        return float("nan"), float("nan"), float("nan")

    abs_values = np.abs(values)

    # Average ranks for ties.
    order = np.argsort(abs_values)
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i + 1
        while j < n and np.isclose(
            abs_values[order[j]],
            abs_values[order[i]],
            rtol=0.0,
            atol=1.0e-15,
        ):
            j += 1

        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    observed_w_plus = float(np.sum(ranks[values > 0.0]))

    # Enumerate all sign assignments exactly for n=12.
    totals = np.empty(1 << n, dtype=float)
    for mask in range(1 << n):
        plus = 0.0
        for idx in range(n):
            if (mask >> idx) & 1:
                plus += ranks[idx]
        totals[mask] = plus

    p_one = float(
        (np.sum(totals >= observed_w_plus) + 1.0)
        / (totals.size + 1.0)
    )

    center = float(np.sum(ranks) / 2.0)
    observed_distance = abs(observed_w_plus - center)

    p_two = float(
        (np.sum(np.abs(totals - center) >= observed_distance) + 1.0)
        / (totals.size + 1.0)
    )

    return observed_w_plus, p_one, min(1.0, p_two)


def bootstrap_ci(
    values: np.ndarray,
    statistic: str,
    bootstraps: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)

    n = values.size
    samples = rng.choice(
        values,
        size=(bootstraps, n),
        replace=True,
    )

    if statistic == "mean":
        stats = np.mean(samples, axis=1)
        observed = float(np.mean(values))
    elif statistic == "median":
        stats = np.median(samples, axis=1)
        observed = float(np.median(values))
    else:
        raise ValueError(statistic)

    low, high = np.quantile(
        stats,
        [0.025, 0.975],
    )

    return observed, float(low), float(high)


def fisher_z(corr: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        corr,
        -1.0 + EPS_CORR,
        1.0 - EPS_CORR,
    )
    return np.arctanh(clipped)


def paired_t_stat(values: np.ndarray) -> tuple[float, float]:
    """Return paired t-statistic and normal-approx two-sided p-value.

    With n=12 this is reported as a descriptive Fisher-z aggregate, not as the
    primary inferential test.  Exact sign/Wilcoxon are preferred.
    """
    n = values.size
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))

    if sd == 0.0:
        return float("inf"), 0.0

    t = mean / (sd / math.sqrt(n))

    # Normal approximation for compact standalone implementation.
    p_two = math.erfc(abs(t) / math.sqrt(2.0))
    return float(t), float(p_two)


def cohens_dz(values: np.ndarray) -> float:
    sd = float(np.std(values, ddof=1))
    if sd == 0.0:
        return float("inf")
    return float(np.mean(values) / sd)


def render(
    bootstraps: int,
) -> str:
    k_pos, n_nonzero, sign_p_one, sign_p_two = exact_sign_test_positive(
        DELTA
    )

    w_plus, wilcoxon_p_one, wilcoxon_p_two = wilcoxon_signed_rank_positive(
        DELTA
    )

    mean_delta, mean_low, mean_high = bootstrap_ci(
        DELTA,
        "mean",
        bootstraps,
        BOOTSTRAP_SEED,
    )

    median_delta, median_low, median_high = bootstrap_ci(
        DELTA,
        "median",
        bootstraps,
        BOOTSTRAP_SEED + 1,
    )

    z_real = fisher_z(CORR_REAL)
    z_null = fisher_z(CORR_NULL)
    z_delta = z_real - z_null

    z_t, z_p_two = paired_t_stat(
        z_delta
    )

    lines = [
        "=== Soft Spaces Phase 3.1 v30.6 ACROSS-SEED REAL-vs-NULL STATISTICS ===",
        f"Frozen seeds: {len(SEEDS)}",
        f"Bootstrap replicates: {bootstraps}",
        "",
        "seed | corr_REAL | corr_NULL | delta_corr",
    ]

    for seed, r_real, r_null, delta in zip(
        SEEDS,
        CORR_REAL,
        CORR_NULL,
        DELTA,
    ):
        lines.append(
            f"{seed} | "
            f"{r_real:+.6f} | "
            f"{r_null:+.6f} | "
            f"{delta:+.6f}"
        )

    lines.extend([
        "",
        "=== DESCRIPTIVE SUMMARY ===",
        f"Mean corr_REAL: {np.mean(CORR_REAL):+.6f}",
        f"Mean corr_NULL: {np.mean(CORR_NULL):+.6f}",
        f"Median corr_REAL: {np.median(CORR_REAL):+.6f}",
        f"Median corr_NULL: {np.median(CORR_NULL):+.6f}",
        f"Mean delta_corr: {np.mean(DELTA):+.6f}",
        f"Median delta_corr: {np.median(DELTA):+.6f}",
        f"Min delta_corr: {np.min(DELTA):+.6f}",
        f"Max delta_corr: {np.max(DELTA):+.6f}",
        f"Positive delta_corr seeds: {k_pos}/{n_nonzero}",
        f"Cohen dz on raw delta_corr: {cohens_dz(DELTA):+.6f}",
        "",
        "=== EXACT SIGN TEST ===",
        f"One-sided H1: median(delta_corr) > 0 : p = {sign_p_one:.8f}",
        f"Two-sided H1: median(delta_corr) != 0: p = {sign_p_two:.8f}",
        "",
        "=== WILCOXON SIGNED-RANK ===",
        f"W+ = {w_plus:.6f}",
        f"One-sided H1: delta_corr > 0 : p = {wilcoxon_p_one:.8f}",
        f"Two-sided H1: delta_corr != 0: p = {wilcoxon_p_two:.8f}",
        "",
        "=== BOOTSTRAP 95% CONFIDENCE INTERVALS ===",
        f"Mean delta_corr: {mean_delta:+.6f} "
        f"[{mean_low:+.6f}, {mean_high:+.6f}]",
        f"Median delta_corr: {median_delta:+.6f} "
        f"[{median_low:+.6f}, {median_high:+.6f}]",
        "",
        "=== FISHER-z PAIRED COMPARISON ===",
        f"Mean Fisher-z(REAL) - Fisher-z(NULL): {np.mean(z_delta):+.6f}",
        f"Median Fisher-z difference: {np.median(z_delta):+.6f}",
        f"Descriptive paired z-difference statistic: {z_t:+.6f}",
        f"Normal-approx two-sided p: {z_p_two:.8f}",
        "",
        "Decision guidance:",
        "  Primary aggregate evidence comes from the exact sign test, Wilcoxon",
        "  signed-rank test and bootstrap intervals.",
        "  Fisher-z is secondary because corr_REAL is numerically saturated near 1",
        "  for many seeds and therefore produces very large transformed values.",
        "",
        "Interpretation rule:",
        "  If sign/Wilcoxon support delta_corr > 0 and the bootstrap intervals",
        "  exclude zero, then the frozen seed ensemble supports a small but",
        "  reproducible REAL-specific enhancement of ±E correlation.",
        "  If those criteria fail, the ±E structure should be attributed mainly",
        "  to shared Hamiltonian/spectral symmetry rather than Soft Spaces.",
        "",
        "Scope:",
        "  v30.6 performs no new physics simulation.  It summarizes the 12 frozen",
        "  v30.5 seed results only.",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "v30_6_across_seed_statistics_output.txt"
        ),
    )

    parser.add_argument(
        "--bootstraps",
        type=int,
        default=BOOTSTRAPS,
        help=(
            "Bootstrap replicates "
            f"(default {BOOTSTRAPS})."
        ),
    )

    args = parser.parse_args()

    if args.bootstraps < 1000:
        raise ValueError(
            "--bootstraps must be at least 1000."
        )

    report = render(
        args.bootstraps
    )

    print(report, end="")
    args.output.write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
