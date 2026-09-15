#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v40.2 — hardware-proxy robustness for ranked search.

Purpose
-------
v40.0 established a REAL-specific operational advantage for cross-energy
ranked search under the frozen Phase-3 model.

v40.1 showed that this advantage survives controlled corruption of the
source-side score.

v40.2 introduces a deliberately simplified hardware proxy with:
* finite-shot sampling,
* symmetric readout error,
* depolarizing contrast loss,
* circuit-depth dependence through accumulated one- and two-qubit error.

This is NOT a Qiskit/Aer simulation and NOT a quantum-hardware experiment.
It is a hardware-oriented stress test that sits between score-noise (v40.1)
and a future explicit circuit implementation.

Frozen dependencies
-------------------
Place this file in the same directory as:

    v31_7_Soft_Spaces_independent_replication.py
    v40_0_application_screening.py

The physical kernel and application endpoint remain unchanged.

Hardware-proxy construction
---------------------------
For each seed/model, source-side prominence values are mapped monotonically
to probabilities in [0,1].  These probabilities are then degraded by:

1. Depolarizing contrast loss:
       p -> 0.5 + lambda * (p - 0.5)

   where
       lambda = (1-p_2q)^N_2q * (1-p_1q)^N_1q

2. Symmetric readout bit-flip error r:
       p -> r + (1-2r)*p

3. Finite-shot sampling:
       k ~ Binomial(shots, p)
       p_hat = k / shots

The estimated probabilities are used only to rank -E source candidates.
The +E target side remains untouched and is used only after ranking.

Default proxy parameters
------------------------
shots          = 4096
readout error  = 0.01
two-qubit error= 1.17e-3
one-qubit error= p_2q / 10

Depth sweep in equivalent two-qubit operations:
    0, 10, 20, 50, 100, 200, 400

One-qubit gate count is parameterized as:
    N_1q = oneq_per_twoq * N_2q
with default oneq_per_twoq = 2.

Primary checkpoint
------------------
N_2q = 20

The same four Precision@K gates used in v40.0/v40.1 are applied:
C1 mean delta Precision@K > 0
C2 95% bootstrap CI lower bound > 0
C3 one-sided exact sign-test p < 0.05
C4 one-sided Wilcoxon p < 0.05

Important scope statement
-------------------------
The numerical value p_2q is used as a proxy parameter only.  Passing v40.2
does NOT establish that the application works on IBM Heron or any other real
device.  A gate-level Qiskit/Aer implementation is required for that next step.
"""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path

import numpy as np


VERSION = "v40.2"

BASE_MODULE = "v31_7_Soft_Spaces_independent_replication"
APP_MODULE = "v40_0_application_screening"

DEFAULT_K = 8
DEFAULT_REPS = 500
DEFAULT_SHOTS = 4096
DEFAULT_READOUT = 0.01
DEFAULT_P2Q = 1.17e-3
DEFAULT_P1Q_RATIO = 0.1
DEFAULT_ONEQ_PER_TWOQ = 2.0

BOOTSTRAPS = 200_000
BOOTSTRAP_SEED = 40_200_001
PROXY_SEED = 40_200_101

DEPTH_GRID = (0, 10, 20, 50, 100, 200, 400)
PRIMARY_DEPTH = 20


def load_modules():
    try:
        base = importlib.import_module(BASE_MODULE)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Could not import {BASE_MODULE}.py\n"
            "Place v40_2_hardware_proxy.py in the same directory as "
            f"{BASE_MODULE}.py."
        ) from exc

    try:
        app0 = importlib.import_module(APP_MODULE)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Could not import {APP_MODULE}.py\n"
            "Place v40_2_hardware_proxy.py in the same directory as "
            f"{APP_MODULE}.py."
        ) from exc

    return base, app0


def stable_proxy_seed(
    seed: int,
    model: str,
    depth: int,
    rep: int,
    shots: int,
) -> int:
    model_code = 1 if model == "REAL" else 2
    value = (
        PROXY_SEED
        + 1_000_003 * int(seed)
        + 10_007 * model_code
        + 100_003 * int(depth)
        + 65_537 * int(rep)
        + 17 * int(shots)
    )
    return value % (2**63 - 1)


def monotone_probability_map(source: np.ndarray) -> np.ndarray:
    """
    Map source scores monotonically to [0.05, 0.95].

    The strict monotone map preserves noiseless ranking but avoids probabilities
    exactly at 0 or 1, where finite-shot noise would be artificially suppressed.
    """
    lo = float(np.min(source))
    hi = float(np.max(source))

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite source score encountered.")

    if hi <= lo:
        return np.full_like(source, 0.5, dtype=float)

    z = (source - lo) / (hi - lo)
    return 0.05 + 0.90 * z


def accumulated_contrast(
    depth_2q: int,
    p2q: float,
    p1q: float,
    oneq_per_twoq: float,
) -> float:
    n2 = max(0, int(depth_2q))
    n1 = max(0.0, float(oneq_per_twoq) * n2)

    # Independent-survival proxy.  This is intentionally simple and explicit.
    lam = ((1.0 - p2q) ** n2) * ((1.0 - p1q) ** n1)
    return float(np.clip(lam, 0.0, 1.0))


def hardware_proxy_measurement(
    source_scores: np.ndarray,
    depth_2q: int,
    shots: int,
    readout_error: float,
    p2q: float,
    p1q: float,
    oneq_per_twoq: float,
    rng: np.random.Generator,
) -> np.ndarray:
    probs = monotone_probability_map(source_scores)

    lam = accumulated_contrast(
        depth_2q,
        p2q,
        p1q,
        oneq_per_twoq,
    )

    # Depolarizing contraction toward maximally mixed binary marginal.
    degraded = 0.5 + lam * (probs - 0.5)

    # Symmetric readout bit flip.
    measured = readout_error + (1.0 - 2.0 * readout_error) * degraded
    measured = np.clip(measured, 0.0, 1.0)

    counts = rng.binomial(shots, measured)
    return counts.astype(float) / float(shots)


def ranked_application_from_proxy(
    prominence: dict[int, float],
    pairs,
    k: int,
    depth_2q: int,
    shots: int,
    readout_error: float,
    p2q: float,
    p1q: float,
    oneq_per_twoq: float,
    rng: np.random.Generator,
):
    if len(pairs) < k:
        raise ValueError(
            f"Need at least K={k} exact +/-E pairs, found only {len(pairs)}."
        )

    source = np.asarray(
        [float(prominence[neg]) for neg, _ in pairs],
        dtype=float,
    )
    target = np.asarray(
        [float(prominence[pos]) for _, pos in pairs],
        dtype=float,
    )
    positions = np.asarray([int(pos) for _, pos in pairs], dtype=int)

    proxy_source = hardware_proxy_measurement(
        source,
        depth_2q=depth_2q,
        shots=shots,
        readout_error=readout_error,
        p2q=p2q,
        p1q=p1q,
        oneq_per_twoq=oneq_per_twoq,
        rng=rng,
    )

    records = [
        {
            "pos": int(pos),
            "source": float(src),
            "target": float(tgt),
        }
        for pos, src, tgt in zip(positions, proxy_source, target)
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
            result = app0.build_prominence(base, seed, model)

            # Support both v40.0 variants:
            #   (prominence, pairs)
            #   (energies, prominence, pairs)
            if len(result) == 2:
                prominence, pairs = result
            elif len(result) == 3:
                _, prominence, pairs = result
            else:
                raise ValueError(
                    "Unexpected return shape from "
                    "v40_0_application_screening.build_prominence(): "
                    f"{len(result)} values"
                )

            cache[seed][model] = (prominence, pairs)

    return cache


def run_depth(
    cache,
    seeds,
    depth_2q: int,
    k: int,
    reps: int,
    shots: int,
    readout_error: float,
    p2q: float,
    p1q: float,
    oneq_per_twoq: float,
):
    rows = []

    for seed in seeds:
        row = {"seed": seed, "depth": depth_2q}

        for model in ("REAL", "NULL"):
            prominence, pairs = cache[seed][model]

            p_vals = []
            e_vals = []
            n_vals = []

            for rep in range(reps):
                rng = np.random.default_rng(
                    stable_proxy_seed(
                        seed,
                        model,
                        depth_2q,
                        rep,
                        shots,
                    )
                )

                p, e, n = ranked_application_from_proxy(
                    prominence,
                    pairs,
                    k,
                    depth_2q,
                    shots,
                    readout_error,
                    p2q,
                    p1q,
                    oneq_per_twoq,
                    rng,
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

        rows.append(row)

    return rows


def bootstrap_paired_ci(values: np.ndarray, bootstraps: int, seed: int):
    rng = np.random.default_rng(seed)
    n = values.size

    samples = rng.choice(
        values,
        size=(bootstraps, n),
        replace=True,
    )
    stats = np.mean(samples, axis=1)
    low, high = np.quantile(stats, [0.025, 0.975])

    return float(np.mean(values)), float(low), float(high)


def summarize_depth(
    base,
    rows,
    depth_2q: int,
    bootstraps: int,
):
    d_p = np.asarray([r["delta_p"] for r in rows], dtype=float)
    d_n = np.asarray([r["delta_ndcg"] for r in rows], dtype=float)
    d_e = np.asarray([r["eval_saved"] for r in rows], dtype=float)

    p_mean, p_low, p_high = bootstrap_paired_ci(
        d_p,
        bootstraps,
        BOOTSTRAP_SEED + int(depth_2q),
    )
    n_mean, n_low, n_high = bootstrap_paired_ci(
        d_n,
        bootstraps,
        BOOTSTRAP_SEED + int(depth_2q) + 1,
    )
    e_mean, e_low, e_high = bootstrap_paired_ci(
        d_e,
        bootstraps,
        BOOTSTRAP_SEED + int(depth_2q) + 2,
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
        "depth": int(depth_2q),
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


def render(
    base,
    seeds,
    all_rows,
    summaries,
    k: int,
    reps: int,
    shots: int,
    readout_error: float,
    p2q: float,
    p1q: float,
    oneq_per_twoq: float,
    bootstraps: int,
):
    lines = [
        "=== Soft Spaces Phase 4 v40.2 HARDWARE-PROXY ROBUSTNESS ===",
        f"Frozen Phase-3 kernel: {base.VERSION} / namespace {base.FROZEN_MODEL_VERSION}",
        f"Qubits: {base.N_QUBITS}",
        f"Hamiltonian terms: {base.N_TERMS}",
        f"Independent seeds: {seeds[0]} ... {seeds[-1]}",
        "Application: -E prominence -> ranked search on +E",
        f"K: {k}",
        f"Proxy repetitions per seed/model/depth: {reps}",
        f"Bootstrap replicates: {bootstraps}",
        "",
        "Proxy parameters:",
        f"  shots = {shots}",
        f"  symmetric readout error = {readout_error:.6g}",
        f"  two-qubit error proxy p2q = {p2q:.6g}",
        f"  one-qubit error proxy p1q = {p1q:.6g}",
        f"  oneq_per_twoq = {oneq_per_twoq:.6g}",
        "",
        "Proxy model:",
        "  source prominence -> monotone probability in [0.05,0.95]",
        "  accumulated contrast lambda = (1-p2q)^N2q * (1-p1q)^N1q",
        "  depolarizing contraction toward 0.5",
        "  symmetric readout error",
        "  finite-shot binomial sampling",
        "",
        "Important:",
        "  This is NOT a Qiskit/Aer gate-level simulation and NOT a hardware run.",
        "",
        "=== DEPTH-SWEEP SUMMARY ===",
        "N2q | REAL P@K | NULL P@K | deltaP | 95% CI deltaP | sign p | Wilcoxon p | REAL eval | NULL eval | saved | REAL NDCG | NULL NDCG | decision",
    ]

    for s in summaries:
        lines.append(
            f"{s['depth']:3d} | "
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

    primary = min(
        summaries,
        key=lambda s: abs(s["depth"] - PRIMARY_DEPTH),
    )

    passed_depths = [
        s["depth"]
        for s in summaries
        if s["pass"]
    ]
    max_pass_depth = max(passed_depths) if passed_depths else None

    lam_primary = accumulated_contrast(
        PRIMARY_DEPTH,
        p2q,
        p1q,
        oneq_per_twoq,
    )

    lines.extend([
        "",
        "=== PRIMARY PREDECLARED CHECKPOINT ===",
        f"Equivalent two-qubit depth N2q = {PRIMARY_DEPTH}",
        f"Accumulated contrast lambda = {lam_primary:.8f}",
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
        f"FINAL v40.2 PRIMARY CHECKPOINT DECISION: {'PASS' if primary['pass'] else 'FAIL'}",
        "",
        "=== HARDWARE-PROXY ROBUSTNESS ENVELOPE ===",
        (
            f"Largest tested N2q depth satisfying all four Precision@{k} gates: "
            f"{max_pass_depth}"
            if max_pass_depth is not None
            else f"No tested depth satisfied all four Precision@{k} gates."
        ),
        "",
        "Interpretation:",
        "  PASS means the REAL-specific ranked-search advantage survives this",
        "  finite-shot/readout/depolarizing hardware proxy at the checkpoint.",
        "",
        "  PASS does NOT mean that the application has been demonstrated on",
        "  IBM Heron or on any real quantum processor.",
        "",
        "Scope:",
        "  v40.2 is a classical hardware-oriented proxy.",
        "  The next scientific step is an explicit circuit-level Qiskit/Aer model.",
    ])

    primary_rows = all_rows[PRIMARY_DEPTH]

    lines.extend([
        "",
        f"=== SEED-LEVEL RESULTS AT N2q={PRIMARY_DEPTH} ===",
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
        help=f"Proxy repetitions per seed/model/depth (default {DEFAULT_REPS}).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help=f"Finite shots per proxy measurement (default {DEFAULT_SHOTS}).",
    )
    parser.add_argument(
        "--readout-error",
        type=float,
        default=DEFAULT_READOUT,
        help=f"Symmetric readout error (default {DEFAULT_READOUT}).",
    )
    parser.add_argument(
        "--p2q",
        type=float,
        default=DEFAULT_P2Q,
        help=f"Two-qubit depolarizing error proxy (default {DEFAULT_P2Q}).",
    )
    parser.add_argument(
        "--p1q-ratio",
        type=float,
        default=DEFAULT_P1Q_RATIO,
        help=f"p1q = p2q * ratio (default {DEFAULT_P1Q_RATIO}).",
    )
    parser.add_argument(
        "--oneq-per-twoq",
        type=float,
        default=DEFAULT_ONEQ_PER_TWOQ,
        help=f"Equivalent one-qubit gates per two-qubit gate (default {DEFAULT_ONEQ_PER_TWOQ}).",
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
        help="Fast implementation check: first seed, primary depth, 20 repetitions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v40_2_hardware_proxy_output.txt"),
    )

    args = parser.parse_args()

    if args.k < 1:
        raise ValueError("--k must be >= 1.")
    if args.reps < 1:
        raise ValueError("--reps must be >= 1.")
    if args.shots < 1:
        raise ValueError("--shots must be >= 1.")
    if not (0.0 <= args.readout_error < 0.5):
        raise ValueError("--readout-error must satisfy 0 <= r < 0.5.")
    if not (0.0 <= args.p2q < 1.0):
        raise ValueError("--p2q must satisfy 0 <= p2q < 1.")
    if args.p1q_ratio < 0.0:
        raise ValueError("--p1q-ratio must be >= 0.")
    if args.oneq_per_twoq < 0.0:
        raise ValueError("--oneq-per-twoq must be >= 0.")
    if args.bootstraps < 1000:
        raise ValueError("--bootstraps must be at least 1000.")

    p1q = args.p2q * args.p1q_ratio

    base, app0 = load_modules()

    if args.smoke:
        seeds = base.HAMILTONIAN_SEEDS[:1]
        depths = (PRIMARY_DEPTH,)
        reps = min(args.reps, 20)
        bootstraps = min(args.bootstraps, 5000)
    else:
        seeds = base.HAMILTONIAN_SEEDS
        depths = DEPTH_GRID
        reps = args.reps
        bootstraps = args.bootstraps

    cache = build_frozen_seed_data(
        app0,
        base,
        seeds,
    )

    all_rows = {}
    summaries = []

    for depth in depths:
        rows = run_depth(
            cache=cache,
            seeds=seeds,
            depth_2q=depth,
            k=args.k,
            reps=reps,
            shots=args.shots,
            readout_error=args.readout_error,
            p2q=args.p2q,
            p1q=p1q,
            oneq_per_twoq=args.oneq_per_twoq,
        )

        all_rows[depth] = rows
        summaries.append(
            summarize_depth(
                base,
                rows,
                depth,
                bootstraps,
            )
        )

    report = render(
        base=base,
        seeds=seeds,
        all_rows=all_rows,
        summaries=summaries,
        k=args.k,
        reps=reps,
        shots=args.shots,
        readout_error=args.readout_error,
        p2q=args.p2q,
        p1q=p1q,
        oneq_per_twoq=args.oneq_per_twoq,
        bootstraps=bootstraps,
    )

    print(report, end="")
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
