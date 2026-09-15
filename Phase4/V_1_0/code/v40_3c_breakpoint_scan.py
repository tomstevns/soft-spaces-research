#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v40.3c — fast breakpoint scan.

Rapid exploratory scan for the approximate CX-noise breakdown region of the
REAL-specific ranked-search advantage.

Reduced screening design:
    seeds   = first 4 frozen seeds
    reps    = 50
    shots   = 1024
    CX grid = 800, 1200, 2000, 4000

This is SCREENING ONLY, not a confirmatory PASS/FAIL experiment.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

BASE_MODULE = "v31_7_Soft_Spaces_independent_replication"
APP_MODULE = "v40_0_application_screening"
CIRCUIT_MODULE = "v40_3b_qiskit_aer_circuit"

DEFAULT_K = 8
DEFAULT_REPS = 50
DEFAULT_SHOTS = 1024
DEFAULT_READOUT = 0.01
DEFAULT_P2Q = 1.17e-3
DEFAULT_P1Q = 1.17e-4

BREAKPOINT_CX_GRID = (800, 1200, 2000, 4000)
SCREEN_SEED_COUNT = 4
BOOTSTRAPS = 50_000
BOOTSTRAP_SEED = 40_350_001


def require_python():
    if sys.version_info < (3, 10):
        raise SystemExit("v40.3c requires Python >= 3.10.")


def load_modules():
    base = importlib.import_module(BASE_MODULE)
    app0 = importlib.import_module(APP_MODULE)
    circuit = importlib.import_module(CIRCUIT_MODULE)
    return base, app0, circuit


def build_cache(app0, circuit, base, seeds):
    cache = {}
    for seed in seeds:
        cache[seed] = {}
        for model in ("REAL", "NULL"):
            prominence, pairs = circuit.extract_prominence_and_pairs(
                app0, base, seed, model
            )
            cache[seed][model] = (prominence, pairs)
    return cache


def bootstrap_ci(values: np.ndarray, bootstraps: int, seed: int):
    rng = np.random.default_rng(seed)
    n = values.size
    samples = rng.choice(values, size=(bootstraps, n), replace=True)
    means = np.mean(samples, axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(np.mean(values)), float(low), float(high)


def summarize_screen(rows, cx: int, bootstraps: int):
    delta_p = np.asarray([r["delta_p"] for r in rows], dtype=float)
    delta_eval = np.asarray([r["eval_saved"] for r in rows], dtype=float)
    delta_ndcg = np.asarray([r["delta_ndcg"] for r in rows], dtype=float)

    mean_p, low_p, high_p = bootstrap_ci(
        delta_p, bootstraps, BOOTSTRAP_SEED + cx
    )

    real_p = float(np.mean([r["real_p"] for r in rows]))
    null_p = float(np.mean([r["null_p"] for r in rows]))
    real_eval = float(np.mean([r["real_eval"] for r in rows]))
    null_eval = float(np.mean([r["null_eval"] for r in rows]))
    real_ndcg = float(np.mean([r["real_ndcg"] for r in rows]))
    null_ndcg = float(np.mean([r["null_ndcg"] for r in rows]))

    if mean_p >= 0.08 and low_p > 0.0:
        label = "CLEAR"
    elif mean_p >= 0.03 and high_p > 0.0:
        label = "WEAK"
    else:
        label = "NEAR-BREAKDOWN"

    return {
        "cx": cx,
        "real_p": real_p,
        "null_p": null_p,
        "delta_p": mean_p,
        "delta_p_low": low_p,
        "delta_p_high": high_p,
        "real_eval": real_eval,
        "null_eval": null_eval,
        "eval_saved": float(np.mean(delta_eval)),
        "real_ndcg": real_ndcg,
        "null_ndcg": null_ndcg,
        "delta_ndcg": float(np.mean(delta_ndcg)),
        "label": label,
    }


def main():
    require_python()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v40_3c_breakpoint_scan_output.txt"),
    )
    args = parser.parse_args()

    base, app0, circuit = load_modules()
    qk = circuit.load_qiskit()

    seeds = base.HAMILTONIAN_SEEDS[:SCREEN_SEED_COUNT]
    cache = build_cache(app0, circuit, base, seeds)

    summaries = []

    print(
        f"[v40.3c] Fast breakpoint scan: "
        f"{len(seeds)} seeds, reps={args.reps}, shots={args.shots}",
        flush=True,
    )

    for cx in BREAKPOINT_CX_GRID:
        print(
            f"[v40.3c] Starting CX={cx} across {len(seeds)} seeds...",
            flush=True,
        )

        rows = circuit.run_cx_level(
            qk=qk,
            base=base,
            cache=cache,
            seeds=seeds,
            total_cx=cx,
            k=DEFAULT_K,
            reps=args.reps,
            shots=args.shots,
            p1q=DEFAULT_P1Q,
            p2q=DEFAULT_P2Q,
            readout_error=DEFAULT_READOUT,
            optimization_level=0,
        )

        summary = summarize_screen(rows, cx, args.bootstraps)
        summaries.append(summary)

        print(
            f"[v40.3c] CX={cx}: REAL P@8={summary['real_p']:.4f}, "
            f"NULL P@8={summary['null_p']:.4f}, "
            f"delta={summary['delta_p']:+.4f}, "
            f"class={summary['label']}",
            flush=True,
        )

    lines = [
        "=== Soft Spaces Phase 4 v40.3c FAST BREAKPOINT SCAN ===",
        f"Seeds: {seeds[0]} ... {seeds[-1]} ({len(seeds)} total)",
        f"Repetitions per seed/model/CX: {args.reps}",
        f"Shots per candidate circuit: {args.shots}",
        f"CX grid: {BREAKPOINT_CX_GRID}",
        "",
        "NOTE:",
        "  Exploratory screening only; not a confirmatory PASS/FAIL experiment.",
        "",
        "CX | REAL P@8 | NULL P@8 | deltaP | 95% bootstrap CI | "
        "REAL eval | NULL eval | saved | REAL NDCG | NULL NDCG | class",
    ]

    for s in summaries:
        lines.append(
            f"{s['cx']:4d} | "
            f"{s['real_p']:.6f} | "
            f"{s['null_p']:.6f} | "
            f"{s['delta_p']:+.6f} | "
            f"[{s['delta_p_low']:+.6f},{s['delta_p_high']:+.6f}] | "
            f"{s['real_eval']:.6f} | "
            f"{s['null_eval']:.6f} | "
            f"{s['eval_saved']:+.6f} | "
            f"{s['real_ndcg']:.6f} | "
            f"{s['null_ndcg']:.6f} | "
            f"{s['label']}"
        )

    near = [s["cx"] for s in summaries if s["label"] == "NEAR-BREAKDOWN"]
    weak = [s["cx"] for s in summaries if s["label"] == "WEAK"]

    lines.extend(["", "=== SCREENING INTERPRETATION ==="])

    if near:
        first = min(near)
        lines.append(f"First tested near-breakdown level: CX={first}.")
        lines.append(
            "Recommended confirmatory zoom around this region."
        )
    elif weak:
        first = min(weak)
        lines.append(f"First weak-advantage level: CX={first}.")
        lines.append(
            "Recommended confirmatory zoom around this CX region."
        )
    else:
        lines.append("No breakdown region found on the tested grid.")
        lines.append(
            "Consider extending above CX=4000 before a full confirmatory run."
        )

    report = "\n".join(lines) + "\n"
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
