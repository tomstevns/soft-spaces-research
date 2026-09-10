#!/usr/bin/env python3
"""Soft Spaces Phase 3.1 v30.8 — predeclared dimensional replication.

Purpose
-------
v30.7 independently confirmed at 8Q a small REAL-specific enhancement of
±E prominence correlation. v30.8 tests whether the same effect generalizes
out of dimension at 7Q and 9Q.

PREDECLARED BEFORE EXECUTION
----------------------------
Dimensions: 7Q and 9Q.
Independent Hamiltonian seeds: 25_044_000 ... 25_044_011.
Frozen physical/statistical kernel: v30.7, which itself uses the frozen v30.2
model namespace and the same 11-term Hamiltonian, complete-eigenspace
projector score, perturbation families, regularization, local prominence and
±E pairing definitions.

Primary endpoint per dimension:
    delta_corr = corr_REAL - corr_NULL

Confirmation gate, applied separately to 7Q and 9Q:
C1 median(delta_corr) > 0
C2 at least 9/12 seeds have delta_corr > 0
C3 one-sided exact sign-test p < 0.05
C4 one-sided Wilcoxon signed-rank p < 0.05
C5 95% bootstrap CI for median(delta_corr) lies entirely above 0

Overall dimensional replication is CONFIRMED only if BOTH dimensions pass all
five criteria. No post-hoc tuning is permitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import v31_7_Soft_Spaces_independent_replication as base

VERSION = "v30.8"
QUBIT_COUNTS = (7, 9)
HAMILTONIAN_SEEDS = tuple(range(25_044_000, 25_044_012))
BOOTSTRAPS = 200_000
MIN_POSITIVE_SEEDS = 9


def run_dimension(n_qubits: int) -> list[dict]:
    # Change dimension only; preserve all v30.7 physics/statistics logic.
    base.N_QUBITS = n_qubits
    base.DIM = 1 << n_qubits

    rows: list[dict] = []
    for seed in HAMILTONIAN_SEEDS:
        row = base.run_seed(seed)
        row = dict(row)
        row["q"] = n_qubits
        rows.append(row)
    return rows


def summarize(rows: list[dict], n_qubits: int, bootstraps: int) -> dict:
    delta = np.asarray([row["delta_corr"] for row in rows], dtype=float)

    pos_count, nonzero_count, sign_p_one, sign_p_two = (
        base.exact_sign_test_positive(delta)
    )
    w_plus, wilcoxon_p_one, wilcoxon_p_two = (
        base.wilcoxon_signed_rank_positive(delta)
    )

    mean_delta, mean_low, mean_high = base.bootstrap_ci(
        delta,
        "mean",
        bootstraps,
        30_008_001 + n_qubits * 10,
    )
    median_delta, median_low, median_high = base.bootstrap_ci(
        delta,
        "median",
        bootstraps,
        30_008_002 + n_qubits * 10,
    )

    criteria = {
        "C1": median_delta > 0.0,
        "C2": pos_count >= MIN_POSITIVE_SEEDS,
        "C3": sign_p_one < 0.05,
        "C4": wilcoxon_p_one < 0.05,
        "C5": median_low > 0.0,
    }

    return {
        "q": n_qubits,
        "positive": pos_count,
        "nonzero": nonzero_count,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "mean_low": mean_low,
        "mean_high": mean_high,
        "median_low": median_low,
        "median_high": median_high,
        "sign_p_one": sign_p_one,
        "sign_p_two": sign_p_two,
        "w_plus": w_plus,
        "wilcoxon_p_one": wilcoxon_p_one,
        "wilcoxon_p_two": wilcoxon_p_two,
        "dz": base.cohens_dz(delta),
        "criteria": criteria,
        "passed": all(criteria.values()),
        "mean_real": float(np.mean([row["corr_real"] for row in rows])),
        "mean_null": float(np.mean([row["corr_null"] for row in rows])),
        "median_real": float(np.median([row["corr_real"] for row in rows])),
        "median_null": float(np.median([row["corr_null"] for row in rows])),
    }


def render(all_rows: list[dict], summaries: list[dict], bootstraps: int) -> str:
    overall = all(summary["passed"] for summary in summaries)

    lines = [
        "=== Soft Spaces Phase 3.1 v30.8 PREDECLARED DIMENSIONAL REPLICATION ===",
        f"Dimensions: {list(QUBIT_COUNTS)}",
        f"Independent seeds per dimension: {len(HAMILTONIAN_SEEDS)}",
        f"Seeds: {HAMILTONIAN_SEEDS[0]} ... {HAMILTONIAN_SEEDS[-1]}",
        f"Frozen kernel: v30.7 / physical namespace {base.FROZEN_MODEL_VERSION}",
        f"Bootstrap replicates: {bootstraps}",
        "",
        "PREDECLARED PRIMARY ENDPOINT:",
        "  delta_corr = corr_REAL - corr_NULL",
        "",
        "PREDECLARED GATE PER DIMENSION:",
        "  C1 median(delta_corr) > 0",
        f"  C2 at least {MIN_POSITIVE_SEEDS}/12 seeds have delta_corr > 0",
        "  C3 one-sided exact sign-test p < 0.05",
        "  C4 one-sided Wilcoxon signed-rank p < 0.05",
        "  C5 95% bootstrap CI for median(delta_corr) entirely above 0",
        "",
        "seed | Q | groups | size min:max | ±E pairs | corr_REAL | corr_NULL | delta_corr",
    ]

    for row in all_rows:
        lines.append(
            f"{row['seed']} | {row['q']}Q | {row['groups']:6d} | "
            f"{row['min_size']:4d}:{row['max_size']:<4d} | {row['pairs']:8d} | "
            f"{row['corr_real']:+.6f} | {row['corr_null']:+.6f} | "
            f"{row['delta_corr']:+.6f}"
        )

    for s in summaries:
        q = s["q"]
        lines.extend([
            "",
            f"=== {q}Q AGGREGATE ===",
            f"Positive delta_corr seeds: {s['positive']}/{s['nonzero']}",
            f"Mean corr_REAL: {s['mean_real']:+.6f}",
            f"Mean corr_NULL: {s['mean_null']:+.6f}",
            f"Median corr_REAL: {s['median_real']:+.6f}",
            f"Median corr_NULL: {s['median_null']:+.6f}",
            f"Mean delta_corr: {s['mean_delta']:+.6f}",
            f"Median delta_corr: {s['median_delta']:+.6f}",
            f"Exact sign test one-sided p: {s['sign_p_one']:.8f}",
            f"Wilcoxon one-sided p: {s['wilcoxon_p_one']:.8f}",
            f"Bootstrap 95% CI mean: [{s['mean_low']:+.6f}, {s['mean_high']:+.6f}]",
            f"Bootstrap 95% CI median: [{s['median_low']:+.6f}, {s['median_high']:+.6f}]",
            f"Cohen dz: {s['dz']:+.6f}",
            "",
            f"{q}Q PREDECLARED GATE:",
            f"C1 median > 0: {'PASS' if s['criteria']['C1'] else 'FAIL'}",
            f"C2 >= {MIN_POSITIVE_SEEDS}/12 positive: {'PASS' if s['criteria']['C2'] else 'FAIL'}",
            f"C3 sign p < 0.05: {'PASS' if s['criteria']['C3'] else 'FAIL'}",
            f"C4 Wilcoxon p < 0.05: {'PASS' if s['criteria']['C4'] else 'FAIL'}",
            f"C5 bootstrap median CI > 0: {'PASS' if s['criteria']['C5'] else 'FAIL'}",
            f"{q}Q DECISION: {'CONFIRMED' if s['passed'] else 'NOT CONFIRMED'}",
        ])

    lines.extend([
        "",
        "=== FINAL PREDECLARED DIMENSIONAL DECISION ===",
        "CONFIRMED ACROSS DIMENSIONS" if overall else "NOT CONFIRMED ACROSS DIMENSIONS",
        "",
        "Interpretation:",
        "  CONFIRMED ACROSS DIMENSIONS means the independently replicated 8Q",
        "  REAL-specific ±E enhancement also survives the same frozen gate at",
        "  both 7Q and 9Q.",
        "",
        "  NOT CONFIRMED ACROSS DIMENSIONS means the broad dimensional claim",
        "  fails under the predeclared gate, regardless of suggestive secondary",
        "  values in either dimension.",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v31_8_dimensional_replication_output.txt"),
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=BOOTSTRAPS,
    )
    args = parser.parse_args()

    if args.bootstraps < 1000:
        raise ValueError("--bootstraps must be at least 1000")

    all_rows: list[dict] = []
    summaries: list[dict] = []

    for q in QUBIT_COUNTS:
        rows = run_dimension(q)
        all_rows.extend(rows)
        summaries.append(summarize(rows, q, args.bootstraps))

    report = render(all_rows, summaries, args.bootstraps)
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
