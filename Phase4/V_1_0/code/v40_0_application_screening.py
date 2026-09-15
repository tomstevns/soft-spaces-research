#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v40.0 — application screening: cross-energy ranked search.

Purpose
-------
Phase 3 established a small but reproducible REAL-specific enhancement of
±E organization under the frozen v30.7 / v30.2 physical model.

Phase 4 asks a different question:

    Can that structure be used operationally?

v40.0 tests a leakage-free ranked-search application. The response on the
negative-energy member of each exact ±E eigenspace pair is used as a prior
for where to search on the positive-energy side.

The target side is never used to build the ranking.

Frozen dependency
-----------------
Place this file in the same directory as:

    v31_7_Soft_Spaces_independent_replication.py

That file supplies the frozen 8Q / 11-term physical kernel, REAL/NULL
constructions, perturbations, exact eigenspace grouping and ±E pairing.

Primary endpoint
----------------
Precision@K for cross-energy ranking, REAL versus NULL. Default K = 8.

This is application screening, not a claim of quantum advantage and not yet
a gate-level hardware experiment.
"""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path

import numpy as np

VERSION = "v40.0"
BASE_MODULE = "v31_7_Soft_Spaces_independent_replication"
DEFAULT_K = 8
BOOTSTRAPS = 200_000
BOOTSTRAP_SEED = 40_000_001


def load_base():
    try:
        return importlib.import_module(BASE_MODULE)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Could not import {BASE_MODULE}.py\n"
            "Place v40_0_application_screening.py in the same directory as "
            f"{BASE_MODULE}.py."
        ) from exc


def build_prominence(base, seed: int, model: str):
    hamiltonian = base.dense_pauli_sum(
        base.N_QUBITS,
        base.random_hamiltonian_terms(seed),
    )
    eigenvalues, real_vectors = np.linalg.eigh(hamiltonian)
    groups = base.degenerate_groups(eigenvalues)

    if model == "REAL":
        vectors = real_vectors
    elif model == "NULL":
        vectors = base.haar_unitary(
            base.DIM,
            base.stable_hash_int(
                f"{base.FROZEN_MODEL_VERSION}|NULL|{seed}|terms{base.N_TERMS}"
            ),
        )
    else:
        raise ValueError(model)

    perturbations = {
        family: base.family_terms(
            base.stable_hash_int(
                f"{base.FROZEN_MODEL_VERSION}|{family}|PERT|{seed}"
            ),
            family,
        )
        for family in base.FAMILIES
    }

    by_family = {
        family: base.model_scores(
            eigenvalues,
            vectors,
            perturbations[family],
            groups,
        )
        for family in base.FAMILIES
    }

    robust = base.robust_family_min(by_family, len(groups))
    prominence = base.prominence_values(robust, len(groups))

    energies = np.asarray(
        [base.eigenspace_energy(eigenvalues, group) for group in groups],
        dtype=float,
    )
    pairs = base.opposite_energy_pairs(energies)

    oriented = []
    for i, j in pairs:
        if energies[i] <= energies[j]:
            neg, pos = i, j
        else:
            neg, pos = j, i

        if energies[neg] >= 0.0 or energies[pos] <= 0.0:
            continue
        if neg not in prominence or pos not in prominence:
            continue
        oriented.append((neg, pos))

    return prominence, oriented


def ranked_application(prominence: dict[int, float], pairs, k: int):
    if len(pairs) < k:
        raise ValueError(
            f"Need at least K={k} exact ±E pairs, found only {len(pairs)}."
        )

    records = [
        {
            "neg": neg,
            "pos": pos,
            "source": float(prominence[neg]),
            "target": float(prominence[pos]),
        }
        for neg, pos in pairs
    ]

    # Search order uses -E information only.
    predicted = sorted(records, key=lambda row: (-row["source"], row["pos"]))

    # +E target values are used only after ranking, for scoring.
    truth = sorted(records, key=lambda row: (-row["target"], row["pos"]))

    true_top = {row["pos"] for row in truth[:k]}
    predicted_top = {row["pos"] for row in predicted[:k]}
    precision = len(true_top & predicted_top) / float(k)

    positions = {
        row["pos"]: index + 1
        for index, row in enumerate(predicted)
    }
    evaluations = max(positions[pos] for pos in true_top)

    # Rank-weighted relevance: K ... 1 for the true Top-K, else 0.
    relevance = {
        row["pos"]: float(k - rank)
        for rank, row in enumerate(truth[:k])
    }

    def dcg(rows):
        total = 0.0
        for rank, row in enumerate(rows[:k], start=1):
            rel = relevance.get(row["pos"], 0.0)
            total += (2.0 ** rel - 1.0) / math.log2(rank + 1.0)
        return total

    ideal = dcg(truth)
    ndcg = dcg(predicted) / ideal if ideal > 0.0 else float("nan")

    return {
        "precision": float(precision),
        "evaluations": int(evaluations),
        "ndcg": float(ndcg),
        "pair_count": len(records),
    }


def bootstrap_paired_ci(values: np.ndarray, bootstraps: int, seed: int):
    rng = np.random.default_rng(seed)
    n = values.size
    samples = rng.choice(values, size=(bootstraps, n), replace=True)
    stats = np.mean(samples, axis=1)
    low, high = np.quantile(stats, [0.025, 0.975])
    return float(np.mean(values)), float(low), float(high)


def run_seed(base, seed: int, k: int):
    row = {"seed": seed}

    for model in ("REAL", "NULL"):
        prominence, pairs = build_prominence(base, seed, model)
        metrics = ranked_application(prominence, pairs, k)

        prefix = model.lower()
        row[f"{prefix}_p"] = metrics["precision"]
        row[f"{prefix}_eval"] = metrics["evaluations"]
        row[f"{prefix}_ndcg"] = metrics["ndcg"]
        row[f"{prefix}_pairs"] = metrics["pair_count"]

    row["delta_p"] = row["real_p"] - row["null_p"]
    row["delta_ndcg"] = row["real_ndcg"] - row["null_ndcg"]
    row["eval_saved"] = row["null_eval"] - row["real_eval"]
    return row


def render(base, results: list[dict], k: int, bootstraps: int) -> str:
    d_p = np.asarray([row["delta_p"] for row in results], dtype=float)
    d_n = np.asarray([row["delta_ndcg"] for row in results], dtype=float)
    d_e = np.asarray([row["eval_saved"] for row in results], dtype=float)

    p_mean, p_low, p_high = bootstrap_paired_ci(d_p, bootstraps, BOOTSTRAP_SEED)
    n_mean, n_low, n_high = bootstrap_paired_ci(d_n, bootstraps, BOOTSTRAP_SEED + 1)
    e_mean, e_low, e_high = bootstrap_paired_ci(d_e, bootstraps, BOOTSTRAP_SEED + 2)

    p_sign = base.exact_sign_test_positive(d_p)
    p_wilc = base.wilcoxon_signed_rank_positive(d_p)
    n_sign = base.exact_sign_test_positive(d_n)
    n_wilc = base.wilcoxon_signed_rank_positive(d_n)

    real_p = np.asarray([row["real_p"] for row in results])
    null_p = np.asarray([row["null_p"] for row in results])
    real_eval = np.asarray([row["real_eval"] for row in results])
    null_eval = np.asarray([row["null_eval"] for row in results])
    real_ndcg = np.asarray([row["real_ndcg"] for row in results])
    null_ndcg = np.asarray([row["null_ndcg"] for row in results])

    c1 = p_mean > 0.0
    c2 = p_low > 0.0
    c3 = p_sign[2] < 0.05
    c4 = p_wilc[1] < 0.05
    application_pass = all((c1, c2, c3, c4))

    lines = [
        "=== Soft Spaces Phase 4 v40.0 APPLICATION SCREENING ===",
        f"Frozen Phase-3 kernel: {base.VERSION} / namespace {base.FROZEN_MODEL_VERSION}",
        f"Qubits: {base.N_QUBITS}",
        f"Hamiltonian terms: {base.N_TERMS}",
        f"Independent seeds: {base.HAMILTONIAN_SEEDS[0]} ... {base.HAMILTONIAN_SEEDS[-1]}",
        "Application: -E prominence -> ranked search on +E",
        f"K: {k}",
        f"Bootstrap replicates: {bootstraps}",
        "",
        "Leakage rule:",
        "  +E target prominence is never used to construct the search order.",
        "  It is used only after ranking to score Precision@K, recovery effort and NDCG@K.",
        "",
        "seed | pairs | REAL P@K | NULL P@K | ΔP | REAL eval | NULL eval | saved | REAL NDCG | NULL NDCG | ΔNDCG",
    ]

    for row in results:
        lines.append(
            f"{row['seed']} | "
            f"{row['real_pairs']:5d} | "
            f"{row['real_p']:.3f} | "
            f"{row['null_p']:.3f} | "
            f"{row['delta_p']:+.3f} | "
            f"{row['real_eval']:9d} | "
            f"{row['null_eval']:9d} | "
            f"{row['eval_saved']:+5d} | "
            f"{row['real_ndcg']:.6f} | "
            f"{row['null_ndcg']:.6f} | "
            f"{row['delta_ndcg']:+.6f}"
        )

    lines.extend([
        "",
        "=== AGGREGATE APPLICATION RESULTS ===",
        f"Mean Precision@{k} REAL: {np.mean(real_p):.6f}",
        f"Mean Precision@{k} NULL: {np.mean(null_p):.6f}",
        f"Mean ΔPrecision@{k}: {p_mean:+.6f}",
        f"95% bootstrap CI ΔPrecision@{k}: [{p_low:+.6f}, {p_high:+.6f}]",
        f"Precision sign test positive: {p_sign[0]}/{p_sign[1]}; one-sided p={p_sign[2]:.8f}",
        f"Precision Wilcoxon W+: {p_wilc[0]:.6f}; one-sided p={p_wilc[1]:.8f}",
        "",
        f"Mean target evaluations REAL: {np.mean(real_eval):.6f}",
        f"Mean target evaluations NULL: {np.mean(null_eval):.6f}",
        f"Mean evaluations saved (NULL-REAL): {e_mean:+.6f}",
        f"95% bootstrap CI evaluations saved: [{e_low:+.6f}, {e_high:+.6f}]",
        "",
        f"Mean NDCG@{k} REAL: {np.mean(real_ndcg):.6f}",
        f"Mean NDCG@{k} NULL: {np.mean(null_ndcg):.6f}",
        f"Mean ΔNDCG@{k}: {n_mean:+.6f}",
        f"95% bootstrap CI ΔNDCG@{k}: [{n_low:+.6f}, {n_high:+.6f}]",
        f"NDCG sign test positive: {n_sign[0]}/{n_sign[1]}; one-sided p={n_sign[2]:.8f}",
        f"NDCG Wilcoxon W+: {n_wilc[0]:.6f}; one-sided p={n_wilc[1]:.8f}",
        "",
        "=== PREDECLARED APPLICATION GATE ===",
        f"C1 mean ΔPrecision@{k} > 0: {'PASS' if c1 else 'FAIL'}",
        f"C2 95% bootstrap CI lower bound > 0: {'PASS' if c2 else 'FAIL'}",
        f"C3 one-sided exact sign-test p < 0.05: {'PASS' if c3 else 'FAIL'}",
        f"C4 one-sided Wilcoxon p < 0.05: {'PASS' if c4 else 'FAIL'}",
        "",
        f"FINAL v40.0 APPLICATION DECISION: {'PASS' if application_pass else 'FAIL'}",
        "",
        "Interpretation:",
        "  PASS means the frozen REAL structure gives a reproducible ranked-search",
        "  advantage over the matched NULL control on the independent seed ensemble.",
        "",
        "  FAIL means this application should not be promoted as a REAL-specific",
        "  Soft-Spaces application, even if either model individually beats random search.",
        "",
        "Scope:",
        "  v40.0 is classical application screening on the frozen 8Q physical model.",
        "  It is not a Qiskit circuit simulation, hardware experiment, or claim of",
        "  quantum computational advantage.",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first frozen seed for a fast implementation check.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v40_0_application_screening_output.txt"),
    )
    args = parser.parse_args()

    if args.k < 1:
        raise ValueError("--k must be >= 1.")
    if args.bootstraps < 1000:
        raise ValueError("--bootstraps must be at least 1000.")

    base = load_base()
    seeds = base.HAMILTONIAN_SEEDS[:1] if args.smoke else base.HAMILTONIAN_SEEDS
    results = [run_seed(base, seed, args.k) for seed in seeds]
    report = render(base, results, args.k, args.bootstraps)
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
