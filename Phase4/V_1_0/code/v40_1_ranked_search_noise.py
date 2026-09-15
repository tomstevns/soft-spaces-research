#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v40.1 — ranked-search robustness under controlled score noise.

Purpose
-------
v40.0 established a reproducible REAL-specific operational advantage for
cross-energy ranked search under the frozen Phase-3 physical model.

v40.1 keeps the application endpoint fixed and asks:

    How robust is that ranked-search advantage when the -E source information
    is corrupted by controlled measurement/score noise?

This is deliberately NOT a gate-level or Qiskit noise model.  Noise is added
only to the source-side prominence values used to construct the search order:

    s_noisy = s + eta * std(s) * Normal(0, 1)

The +E target values remain untouched and are used only after ranking to score
the application.  Therefore the v40.0 leakage rule is preserved.

Frozen dependencies
-------------------
Place this file in the same directory as:

    v31_7_Soft_Spaces_independent_replication.py
    v40_0_application_screening.py

v40.1 reuses the exact v40.0 construction of the frozen 8Q / 11-term REAL/NULL
prominence data.  It changes only the source-side ranking information.

Primary robustness endpoint
---------------------------
Precision@K difference REAL - NULL as a function of relative score-noise eta.

Default:
    K = 8
    noise repetitions per seed/model/eta = 500

Predeclared noise grid
----------------------
0,
1e-4, 3e-4,
1e-3, 3e-3,
1e-2, 3e-2,
1e-1, 2e-1, 4e-1, 8e-1

Primary checkpoint:
    eta = 1e-3

A checkpoint PASS requires across-seed:
C1 mean delta Precision@K > 0
C2 95% bootstrap CI lower bound > 0
C3 one-sided exact sign-test p < 0.05
C4 one-sided Wilcoxon p < 0.05

Important scope statement
-------------------------
eta is dimensionless relative score noise.  eta = 1e-3 must NOT be interpreted
as a 1e-3 physical two-qubit gate-error rate.  Gate noise, decoherence,
readout error and finite-shot circuit effects belong to a later hardware-level
experiment.
"""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path

import numpy as np


VERSION = "v40.1"
BASE_MODULE = "v31_7_Soft_Spaces_independent_replication"
APP_MODULE = "v40_0_application_screening"

DEFAULT_K = 8
DEFAULT_REPS = 500
BOOTSTRAPS = 200_000
BOOTSTRAP_SEED = 40_100_001
NOISE_SEED = 40_100_101

NOISE_GRID = (
    0.0,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    2e-1,
    4e-1,
    8e-1,
)

PRIMARY_ETA = 1e-3


def load_modules():
    try:
        base = importlib.import_module(BASE_MODULE)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Could not import {BASE_MODULE}.py\n"
            "Place v40_1_ranked_search_noise.py in the same directory as "
            f"{BASE_MODULE}.py."
        ) from exc

    try:
        app0 = importlib.import_module(APP_MODULE)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Could not import {APP_MODULE}.py\n"
            "Place v40_1_ranked_search_noise.py in the same directory as "
            f"{APP_MODULE}.py."
        ) from exc

    return base, app0


def stable_noise_seed(seed: int, model: str, eta: float, rep: int) -> int:
    # Pure integer construction; independent of Python's randomized hash().
    model_code = 1 if model == "REAL" else 2
    eta_code = int(round(eta * 1_000_000_000))
    value = (
        NOISE_SEED
        + 1_000_003 * int(seed)
        + 10_007 * model_code
        + 97 * eta_code
        + 65_537 * int(rep)
    )
    return value % (2**63 - 1)


def ranked_application_with_noisy_source(
    prominence: dict[int, float],
    pairs,
    k: int,
    eta: float,
    rng: np.random.Generator,
):
    """
    Same v40.0 application, except source-side prominence is corrupted before
    ranking.  Target-side values are untouched and never used to build ranking.
    """
    if len(pairs) < k:
        raise ValueError(
            f"Need at least K={k} exact ±E pairs, found only {len(pairs)}."
        )

    source = np.asarray([float(prominence[neg]) for neg, _ in pairs], dtype=float)
    target = np.asarray([float(prominence[pos]) for _, pos in pairs], dtype=float)
    positions = np.asarray([int(pos) for _, pos in pairs], dtype=int)

    scale = float(np.std(source, ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    if eta == 0.0:
        noisy_source = source.copy()
    else:
        noisy_source = source + eta * scale * rng.normal(size=source.size)

    records = [
        {
            "pos": int(pos),
            "source": float(src),
            "target": float(tgt),
        }
        for pos, src, tgt in zip(positions, noisy_source, target)
    ]

    predicted = sorted(records, key=lambda row: (-row["source"], row["pos"]))
    truth = sorted(records, key=lambda row: (-row["target"], row["pos"]))

    true_top = {row["pos"] for row in truth[:k]}
    predicted_top = {row["pos"] for row in predicted[:k]}
    precision = len(true_top & predicted_top) / float(k)

    predicted_positions = {
        row["pos"]: index + 1
        for index, row in enumerate(predicted)
    }
    evaluations = max(predicted_positions[pos] for pos in true_top)

    relevance = {
        row["pos"]: float(k - rank)
        for rank, row in enumerate(truth[:k])
    }

    def dcg(rows):
        total = 0.0
        for rank, row in enumerate(rows[:k], start=1):
            rel = relevance.get(row["pos"], 0.0)
            total += (2.0**rel - 1.0) / math.log2(rank + 1.0)
        return total

    ideal = dcg(truth)
    ndcg = dcg(predicted) / ideal if ideal > 0.0 else float("nan")

    return float(precision), int(evaluations), float(ndcg)


def build_frozen_seed_data(app0, base, seeds):
    cache = {}
    for seed in seeds:
        cache[seed] = {}
        for model in ("REAL", "NULL"):
            prominence, pairs = app0.build_prominence(base, seed, model)
            cache[seed][model] = (prominence, pairs)
    return cache


def run_noise_level(cache, seeds, eta: float, k: int, reps: int):
    """
    Average repeated noisy rankings within each seed/model first.
    Across-seed inference is then performed on the paired seed-level means.
    """
    seed_rows = []

    for seed in seeds:
        row = {"seed": seed, "eta": eta}

        for model in ("REAL", "NULL"):
            prominence, pairs = cache[seed][model]

            p_vals = []
            e_vals = []
            n_vals = []

            local_reps = 1 if eta == 0.0 else reps

            for rep in range(local_reps):
                rng = np.random.default_rng(
                    stable_noise_seed(seed, model, eta, rep)
                )
                p, e, n = ranked_application_with_noisy_source(
                    prominence, pairs, k, eta, rng
                )
                p_vals.append(p)
                e_vals.append(e)
                n_vals.append(n)

            prefix = model.lower()
            row[f"{prefix}_p"] = float(np.mean(p_vals))
            row[f"{prefix}_eval"] = float(np.mean(e_vals))
            row[f"{prefix}_ndcg"] = float(np.mean(n_vals))
            row[f"{prefix}_pairs"] = len(pairs)

        row["delta_p"] = row["real_p"] - row["null_p"]
        row["delta_ndcg"] = row["real_ndcg"] - row["null_ndcg"]
        row["eval_saved"] = row["null_eval"] - row["real_eval"]
        seed_rows.append(row)

    return seed_rows


def bootstrap_paired_ci(values: np.ndarray, bootstraps: int, seed: int):
    rng = np.random.default_rng(seed)
    n = values.size
    samples = rng.choice(values, size=(bootstraps, n), replace=True)
    stats = np.mean(samples, axis=1)
    low, high = np.quantile(stats, [0.025, 0.975])
    return float(np.mean(values)), float(low), float(high)


def summarize_level(base, rows, eta: float, bootstraps: int):
    d_p = np.asarray([r["delta_p"] for r in rows], dtype=float)
    d_n = np.asarray([r["delta_ndcg"] for r in rows], dtype=float)
    d_e = np.asarray([r["eval_saved"] for r in rows], dtype=float)

    seed_shift = int(round(eta * 1e9))

    p_mean, p_low, p_high = bootstrap_paired_ci(
        d_p, bootstraps, BOOTSTRAP_SEED + seed_shift
    )
    n_mean, n_low, n_high = bootstrap_paired_ci(
        d_n, bootstraps, BOOTSTRAP_SEED + seed_shift + 1
    )
    e_mean, e_low, e_high = bootstrap_paired_ci(
        d_e, bootstraps, BOOTSTRAP_SEED + seed_shift + 2
    )

    p_sign = base.exact_sign_test_positive(d_p)
    p_wilc = base.wilcoxon_signed_rank_positive(d_p)
    n_sign = base.exact_sign_test_positive(d_n)
    n_wilc = base.wilcoxon_signed_rank_positive(d_n)

    real_p = np.asarray([r["real_p"] for r in rows], dtype=float)
    null_p = np.asarray([r["null_p"] for r in rows], dtype=float)
    real_eval = np.asarray([r["real_eval"] for r in rows], dtype=float)
    null_eval = np.asarray([r["null_eval"] for r in rows], dtype=float)
    real_ndcg = np.asarray([r["real_ndcg"] for r in rows], dtype=float)
    null_ndcg = np.asarray([r["null_ndcg"] for r in rows], dtype=float)

    c1 = p_mean > 0.0
    c2 = p_low > 0.0
    c3 = p_sign[2] < 0.05
    c4 = p_wilc[1] < 0.05
    passes = all((c1, c2, c3, c4))

    return {
        "eta": eta,
        "real_p": float(np.mean(real_p)),
        "null_p": float(np.mean(null_p)),
        "delta_p": p_mean,
        "delta_p_low": p_low,
        "delta_p_high": p_high,
        "p_sign": float(p_sign[2]),
        "p_wilc": float(p_wilc[1]),
        "real_eval": float(np.mean(real_eval)),
        "null_eval": float(np.mean(null_eval)),
        "eval_saved": e_mean,
        "eval_saved_low": e_low,
        "eval_saved_high": e_high,
        "real_ndcg": float(np.mean(real_ndcg)),
        "null_ndcg": float(np.mean(null_ndcg)),
        "delta_ndcg": n_mean,
        "delta_ndcg_low": n_low,
        "delta_ndcg_high": n_high,
        "ndcg_sign_p": float(n_sign[2]),
        "ndcg_wilc_p": float(n_wilc[1]),
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "c4": c4,
        "pass": passes,
    }


def fmt_eta(eta: float) -> str:
    if eta == 0.0:
        return "0"
    if eta < 0.01:
        return f"{eta:.0e}"
    return f"{eta:.3f}".rstrip("0").rstrip(".")


def render(base, seeds, all_rows, summaries, k: int, reps: int, bootstraps: int):
    lines = [
        "=== Soft Spaces Phase 4 v40.1 RANKED-SEARCH NOISE ROBUSTNESS ===",
        f"Frozen Phase-3 kernel: {base.VERSION} / namespace {base.FROZEN_MODEL_VERSION}",
        f"Qubits: {base.N_QUBITS}",
        f"Hamiltonian terms: {base.N_TERMS}",
        f"Independent seeds: {seeds[0]} ... {seeds[-1]}",
        "Application: -E prominence -> ranked search on +E",
        f"K: {k}",
        f"Noise repetitions per seed/model/eta: {reps}",
        f"Bootstrap replicates: {bootstraps}",
        "",
        "Noise definition:",
        "  source_noisy = source + eta * std(source) * Normal(0,1)",
        "  Noise is applied ONLY to the -E source information used for ranking.",
        "  +E target prominence remains untouched and is used only after ranking.",
        "",
        "Important:",
        "  eta is relative score noise, NOT a physical gate-error probability.",
        "",
        "=== NOISE-SWEEP SUMMARY ===",
        "eta | REAL P@K | NULL P@K | deltaP | 95% CI deltaP | sign p | Wilcoxon p | REAL eval | NULL eval | saved | REAL NDCG | NULL NDCG | decision",
    ]

    for s in summaries:
        lines.append(
            f"{fmt_eta(s['eta']):>6} | "
            f"{s['real_p']:.6f} | "
            f"{s['null_p']:.6f} | "
            f"{s['delta_p']:+.6f} | "
            f"[{s['delta_p_low']:+.6f},{s['delta_p_high']:+.6f}] | "
            f"{s['p_sign']:.8f} | "
            f"{s['p_wilc']:.8f} | "
            f"{s['real_eval']:.6f} | "
            f"{s['null_eval']:.6f} | "
            f"{s['eval_saved']:+.6f} | "
            f"{s['real_ndcg']:.6f} | "
            f"{s['null_ndcg']:.6f} | "
            f"{'PASS' if s['pass'] else 'FAIL'}"
        )

    # Exact primary checkpoint lookup.
    primary = min(summaries, key=lambda s: abs(s["eta"] - PRIMARY_ETA))

    passed_etas = [s["eta"] for s in summaries if s["pass"]]
    max_pass_eta = max(passed_etas) if passed_etas else None

    lines.extend([
        "",
        "=== PRIMARY PREDECLARED CHECKPOINT ===",
        f"eta = {PRIMARY_ETA:.0e}",
        f"Mean Precision@{k} REAL: {primary['real_p']:.6f}",
        f"Mean Precision@{k} NULL: {primary['null_p']:.6f}",
        f"Mean delta Precision@{k}: {primary['delta_p']:+.6f}",
        f"95% bootstrap CI: [{primary['delta_p_low']:+.6f}, {primary['delta_p_high']:+.6f}]",
        f"One-sided sign-test p: {primary['p_sign']:.8f}",
        f"One-sided Wilcoxon p: {primary['p_wilc']:.8f}",
        "",
        f"C1 mean delta Precision@{k} > 0: {'PASS' if primary['c1'] else 'FAIL'}",
        f"C2 95% bootstrap CI lower bound > 0: {'PASS' if primary['c2'] else 'FAIL'}",
        f"C3 exact sign-test p < 0.05: {'PASS' if primary['c3'] else 'FAIL'}",
        f"C4 Wilcoxon p < 0.05: {'PASS' if primary['c4'] else 'FAIL'}",
        "",
        f"FINAL v40.1 PRIMARY CHECKPOINT DECISION: {'PASS' if primary['pass'] else 'FAIL'}",
        "",
        "=== ROBUSTNESS ENVELOPE ===",
        (
            f"Largest tested eta satisfying all four Precision@{k} gates: "
            f"{fmt_eta(max_pass_eta)}"
            if max_pass_eta is not None
            else f"No tested eta satisfied all four Precision@{k} gates."
        ),
        "",
        "Interpretation:",
        "  PASS at the primary checkpoint means the REAL-specific ranked-search",
        "  advantage survives controlled corruption of the source-side score.",
        "",
        "  This experiment does NOT establish robustness to physical quantum-hardware",
        "  gate noise, decoherence, readout error or finite-shot circuit sampling.",
        "",
        "Scope:",
        "  v40.1 is a classical score-noise robustness experiment on the frozen",
        "  8Q application model. Hardware-level noise belongs to a later phase.",
    ])

    # Add compact seed table for the primary eta only.
    primary_rows = all_rows[PRIMARY_ETA]
    lines.extend([
        "",
        f"=== SEED-LEVEL RESULTS AT eta={PRIMARY_ETA:.0e} ===",
        "seed | REAL P@K | NULL P@K | deltaP | REAL eval | NULL eval | saved | REAL NDCG | NULL NDCG | deltaNDCG",
    ])
    for r in primary_rows:
        lines.append(
            f"{r['seed']} | "
            f"{r['real_p']:.6f} | "
            f"{r['null_p']:.6f} | "
            f"{r['delta_p']:+.6f} | "
            f"{r['real_eval']:.6f} | "
            f"{r['null_eval']:.6f} | "
            f"{r['eval_saved']:+.6f} | "
            f"{r['real_ndcg']:.6f} | "
            f"{r['null_ndcg']:.6f} | "
            f"{r['delta_ndcg']:+.6f}"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Top-K target set (default {DEFAULT_K}).",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPS,
        help=f"Noise repetitions per seed/model/eta (default {DEFAULT_REPS}).",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=BOOTSTRAPS,
        help=f"Bootstrap replicates (default {BOOTSTRAPS}).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast implementation check: first seed, 20 reps, primary eta only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v40_1_ranked_search_noise_output.txt"),
    )

    args = parser.parse_args()

    if args.k < 1:
        raise ValueError("--k must be >= 1.")
    if args.reps < 1:
        raise ValueError("--reps must be >= 1.")
    if args.bootstraps < 1000:
        raise ValueError("--bootstraps must be at least 1000.")

    base, app0 = load_modules()

    if args.smoke:
        seeds = base.HAMILTONIAN_SEEDS[:1]
        noise_grid = (PRIMARY_ETA,)
        reps = min(args.reps, 20)
        bootstraps = min(args.bootstraps, 5000)
    else:
        seeds = base.HAMILTONIAN_SEEDS
        noise_grid = NOISE_GRID
        reps = args.reps
        bootstraps = args.bootstraps

    cache = build_frozen_seed_data(app0, base, seeds)

    all_rows = {}
    summaries = []

    for eta in noise_grid:
        rows = run_noise_level(cache, seeds, eta, args.k, reps)
        all_rows[eta] = rows
        summaries.append(
            summarize_level(base, rows, eta, bootstraps)
        )

    report = render(
        base,
        seeds,
        all_rows,
        summaries,
        args.k,
        reps,
        bootstraps,
    )

    print(report, end="")
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
