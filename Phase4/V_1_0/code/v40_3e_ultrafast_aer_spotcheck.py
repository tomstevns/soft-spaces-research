#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v40.3e — ultrafast explicit Aer spot-check.

Purpose
-------
Minimal explicit Qiskit/Aer confirmation of the breakpoint trend.

Default design:
    CX levels : 800, 1200
    seeds     : first 2 frozen seeds
    reps      : 10
    shots     : 512

This is an exploratory spot-check only.
It is NOT a confirmatory statistical experiment and does NOT replace v40.3b.

Interpretation:
- If CX=800 still shows a positive REAL advantage and CX=1200 is much weaker,
  that qualitatively supports the breakpoint locator.
- If not, increase resolution later.
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

DEFAULT_CX = (800, 1200)
DEFAULT_SEEDS = 2
DEFAULT_REPS = 10
DEFAULT_SHOTS = 512
DEFAULT_K = 8

P2Q = 1.17e-3
P1Q = 1.17e-4
READOUT = 0.01


def require_python():
    if sys.version_info < (3, 10):
        raise SystemExit("v40.3e requires Python >= 3.10.")


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


def summarize(rows):
    real_p = float(np.mean([r["real_p"] for r in rows]))
    null_p = float(np.mean([r["null_p"] for r in rows]))
    delta_p = real_p - null_p

    real_eval = float(np.mean([r["real_eval"] for r in rows]))
    null_eval = float(np.mean([r["null_eval"] for r in rows]))
    saved = null_eval - real_eval

    real_ndcg = float(np.mean([r["real_ndcg"] for r in rows]))
    null_ndcg = float(np.mean([r["null_ndcg"] for r in rows]))
    delta_ndcg = real_ndcg - null_ndcg

    return {
        "real_p": real_p,
        "null_p": null_p,
        "delta_p": delta_p,
        "real_eval": real_eval,
        "null_eval": null_eval,
        "saved": saved,
        "real_ndcg": real_ndcg,
        "null_ndcg": null_ndcg,
        "delta_ndcg": delta_ndcg,
    }


def main():
    require_python()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--cx",
        type=int,
        nargs="+",
        default=list(DEFAULT_CX),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v40_3e_ultrafast_aer_spotcheck_output.txt"),
    )
    args = parser.parse_args()

    base, app0, circuit = load_modules()
    qk = circuit.load_qiskit()

    seeds = list(base.HAMILTONIAN_SEEDS[:args.seeds])
    cache = build_cache(app0, circuit, base, seeds)

    lines = [
        "=== Soft Spaces Phase 4 v40.3e ULTRAFAST EXPLICIT AER SPOT-CHECK ===",
        f"Seeds: {seeds}",
        f"Repetitions per seed/model/CX: {args.reps}",
        f"Shots per candidate circuit: {args.shots}",
        f"CX levels: {tuple(args.cx)}",
        f"p2q={P2Q}, p1q={P1Q}, readout={READOUT}",
        "",
        "Exploratory spot-check only — no confirmatory PASS/FAIL inference.",
        "",
        "CX | REAL P@8 | NULL P@8 | deltaP | REAL eval | NULL eval | saved | "
        "REAL NDCG | NULL NDCG | deltaNDCG",
    ]

    for cx in args.cx:
        print(
            f"[v40.3e] Starting CX={cx} across {len(seeds)} seed(s) "
            f"(reps={args.reps}, shots={args.shots})...",
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
            p1q=P1Q,
            p2q=P2Q,
            readout_error=READOUT,
            optimization_level=0,
        )

        s = summarize(rows)

        line = (
            f"{cx:4d} | "
            f"{s['real_p']:.6f} | "
            f"{s['null_p']:.6f} | "
            f"{s['delta_p']:+.6f} | "
            f"{s['real_eval']:.3f} | "
            f"{s['null_eval']:.3f} | "
            f"{s['saved']:+.3f} | "
            f"{s['real_ndcg']:.6f} | "
            f"{s['null_ndcg']:.6f} | "
            f"{s['delta_ndcg']:+.6f}"
        )

        print("[v40.3e] " + line, flush=True)
        lines.append(line)

    lines += [
        "",
        "Qualitative expectation from v40.3c-fast:",
        "  CX=800  -> positive but weakened REAL advantage",
        "  CX=1200 -> near-breakdown / much smaller advantage",
        "",
        "If this pattern is reproduced, stop here for the current phase.",
        "Increase resolution only if the result is contradictory or ambiguous.",
    ]

    report = "\n".join(lines) + "\n"
    args.output.write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)


if __name__ == "__main__":
    main()
