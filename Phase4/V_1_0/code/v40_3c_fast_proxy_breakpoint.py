#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v40.3c-fast — rapid breakpoint locator.

Purpose
-------
Locate the approximate high-CX breakdown region in seconds/minutes before
spending hours on explicit Qiskit/Aer circuits.

IMPORTANT:
This is an exploratory hardware-proxy locator, NOT a replacement for v40.3b.
It uses the same finite-shot/readout/depolarizing logic as the Phase-4
hardware proxy, extended to high CX counts.

Once a likely breakdown interval is found, only 2-3 selected CX values should
be confirmed with the expensive explicit Qiskit/Aer circuit implementation.

Defaults
--------
Frozen seeds: first 4
Reps: 200
Shots: 1024
CX grid: 800, 1200, 2000, 3000, 4000, 6000

Noise:
p2q = 1.17e-3
p1q = 1.17e-4
readout = 0.01
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import numpy as np

BASE_MODULE = "v31_7_Soft_Spaces_independent_replication"
APP_MODULE = "v40_0_application_screening"

K = 8
SEED_COUNT = 4
DEFAULT_REPS = 200
DEFAULT_SHOTS = 1024
P2Q = 1.17e-3
P1Q = 1.17e-4
READOUT = 0.01
CX_GRID = (800, 1200, 2000, 3000, 4000, 6000)
RNG_SEED = 40_351_001


def load_modules():
    base = importlib.import_module(BASE_MODULE)
    app0 = importlib.import_module(APP_MODULE)
    return base, app0


def extract(app0, base, seed, model):
    result = app0.build_prominence(base, seed, model)
    if len(result) == 2:
        return result
    if len(result) == 3:
        _, prominence, pairs = result
        return prominence, pairs
    raise ValueError("Unexpected build_prominence return shape.")


def map_prob(x):
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        return np.full_like(x, 0.5)
    z = (x - lo) / (hi - lo)
    return 0.05 + 0.90 * z


def degraded_prob(p, cx):
    # Same explicit independent-survival approximation used as hardware proxy.
    # Approximate 2 one-qubit operations per CX in the high-level loading network.
    lam = ((1.0 - P2Q) ** cx) * ((1.0 - P1Q) ** (2 * cx))
    q = 0.5 + lam * (p - 0.5)
    q = READOUT + (1.0 - 2.0 * READOUT) * q
    return np.clip(q, 0.0, 1.0), lam


def metrics(measured_source, target, positions):
    pred_order = np.lexsort((positions, -measured_source))
    truth_order = np.lexsort((positions, -target))

    true_top = set(positions[truth_order[:K]])
    pred_top = set(positions[pred_order[:K]])
    precision = len(true_top & pred_top) / K

    loc = {int(positions[idx]): rank + 1 for rank, idx in enumerate(pred_order)}
    evaluations = max(loc[int(pos)] for pos in true_top)

    return precision, evaluations


def run_one(prominence, pairs, cx, reps, shots, rng):
    source = np.asarray([prominence[n] for n, _ in pairs], float)
    target = np.asarray([prominence[p] for _, p in pairs], float)
    positions = np.asarray([p for _, p in pairs], int)

    ideal_p = map_prob(source)
    q, lam = degraded_prob(ideal_p, cx)

    ps, es = [], []
    for _ in range(reps):
        measured = rng.binomial(shots, q) / shots
        p, e = metrics(measured, target, positions)
        ps.append(p)
        es.append(e)

    return float(np.mean(ps)), float(np.mean(es)), float(lam)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v40_3c_fast_proxy_breakpoint_output.txt"),
    )
    args = parser.parse_args()

    base, app0 = load_modules()
    seeds = base.HAMILTONIAN_SEEDS[:SEED_COUNT]
    rng = np.random.default_rng(RNG_SEED)

    cache = {}
    for seed in seeds:
        cache[seed] = {}
        for model in ("REAL", "NULL"):
            cache[seed][model] = extract(app0, base, seed, model)

    lines = [
        "=== Soft Spaces Phase 4 v40.3c-fast BREAKPOINT LOCATOR ===",
        f"Seeds: {seeds[0]} ... {seeds[-1]} ({len(seeds)} total)",
        f"Reps: {args.reps}",
        f"Shots: {args.shots}",
        f"p2q={P2Q}, p1q={P1Q}, readout={READOUT}",
        "",
        "SCREENING ONLY — hardware proxy, not explicit Aer confirmation.",
        "",
        "CX | contrast lambda | REAL P@8 | NULL P@8 | deltaP | REAL eval | NULL eval | saved | class",
    ]

    for cx in CX_GRID:
        rp, np_, re, ne, lams = [], [], [], [], []

        for seed in seeds:
            vals = {}
            for model in ("REAL", "NULL"):
                prominence, pairs = cache[seed][model]
                p, e, lam = run_one(
                    prominence, pairs, cx, args.reps, args.shots, rng
                )
                vals[model] = (p, e)
                lams.append(lam)

            rp.append(vals["REAL"][0])
            re.append(vals["REAL"][1])
            np_.append(vals["NULL"][0])
            ne.append(vals["NULL"][1])

        real_p = float(np.mean(rp))
        null_p = float(np.mean(np_))
        delta = real_p - null_p
        real_e = float(np.mean(re))
        null_e = float(np.mean(ne))
        saved = null_e - real_e
        lam = float(np.mean(lams))

        if delta >= 0.08:
            label = "CLEAR"
        elif delta >= 0.03:
            label = "WEAK"
        else:
            label = "NEAR-BREAKDOWN"

        line = (
            f"{cx:4d} | {lam:.6f} | {real_p:.6f} | {null_p:.6f} | "
            f"{delta:+.6f} | {real_e:.3f} | {null_e:.3f} | "
            f"{saved:+.3f} | {label}"
        )
        lines.append(line)
        print(line, flush=True)

    lines += [
        "",
        "Use the first WEAK/NEAR-BREAKDOWN interval only to choose",
        "2-3 expensive v40.3b-style Aer confirmation points."
    ]

    report = "\n".join(lines) + "\n"
    args.output.write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
