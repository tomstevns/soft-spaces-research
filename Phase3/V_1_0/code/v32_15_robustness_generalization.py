#!/usr/bin/env python3
"""Soft Spaces Phase 3.2 v32.15 — robustness/generalization test.

Purpose
-------
Test whether the degeneracy-closed, basis-invariant interference signal from
v32.14 survives moderate changes in Hamiltonian term count.

Primary model variations:
    N_TERMS = 7, 9, 11

For each term count, seed, and perturbation family, this program reuses the
v32.14 analysis and evaluates:
    A_group   normalized interference cross-term
    C_group   degeneracy-closed coherence factor
    cross_sum unnormalized interference cross-term

The 11-term case is the frozen reference. 7 and 9 terms are robustness probes.

Dependency
----------
Place this file next to:
    v32_14_degeneracy_closed.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path
import numpy as np

TERM_COUNTS = (7, 9, 11)
DEFAULT_SEEDS = tuple(range(25_042_000, 25_042_012))
FAMILIES = ("dephasing", "transverse")


def load_v3214():
    here = Path(__file__).resolve().parent
    path = here / "v32_14_degeneracy_closed.py"
    if not path.exists():
        raise FileNotFoundError(
            "v32_14_degeneracy_closed.py must be in the same directory as v32_15."
        )
    spec = importlib.util.spec_from_file_location("v3214_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sign_p(k: int, n: int) -> float:
    return sum(math.comb(n, j) for j in range(k, n + 1)) / (2 ** n)


def run_block(term_count: int, seed_start: int, seed_count: int, nperm: int):
    m = load_v3214()
    m.N_TERMS = int(term_count)

    selected = DEFAULT_SEEDS[seed_start:seed_start + seed_count]
    rows = []
    for i, seed in enumerate(selected, start=1):
        print(
            f"[terms={term_count}] [{i}/{len(selected)}] seed {seed}",
            flush=True,
        )
        for family in FAMILIES:
            family_rows = m.pair_metrics_for_family_v3214(
                seed, family, m.ENERGY_REG
            )
            for r in family_rows:
                r["term_count"] = term_count
            rows.extend(family_rows)

    stats = m.seed_stats_v3214(rows, nperm)
    for r in stats:
        r["term_count"] = term_count
    return rows, stats


def summarize(rows: list[dict], stats: list[dict]) -> str:
    lines = [
        "=== Soft Spaces Phase 3.2 v32.15 ROBUSTNESS / GENERALIZATION ===",
        "",
        "Question:",
        "  Does the degeneracy-closed interference/coherence signal survive",
        "  moderate changes in Hamiltonian term count?",
        "",
        "Reference: 11 terms",
        "Robustness probes: 7 and 9 terms",
        "",
    ]

    for nt in sorted(set(int(r["term_count"]) for r in rows)):
        lines.append(f"N_TERMS = {nt}")
        for family in FAMILIES:
            famrows = [
                r for r in rows
                if int(r["term_count"]) == nt and r["family"] == family
            ]
            y = np.asarray([float(r["prominence"]) for r in famrows], float)

            lines.append(f"  FAMILY: {family}")
            for metric in ("A_group", "C_group", "cross_sum"):
                ss = [
                    r for r in stats
                    if int(r["term_count"]) == nt
                    and r["family"] == family
                    and r["metric"] == metric
                ]
                vals = np.asarray([float(r["spearman"]) for r in ss], float)
                pos = int(np.sum(vals > 0))
                x = np.asarray([float(r[metric]) for r in famrows], float)
                pooled = m_corr(x, y)
                lines.append(
                    f"    {metric:10s} median seed rho={np.median(vals):+.4f}; "
                    f"positive={pos}/{len(vals)}; sign-p={sign_p(pos,len(vals)):.5f}; "
                    f"pooled rho={pooled:+.4f}"
                )
        lines.append("")

    lines += [
        "=== ROBUSTNESS GATE ===",
        "A robust structural signal should retain the same positive direction",
        "for A_group and/or C_group across 7, 9 and 11 terms, rather than",
        "appearing only in the frozen 11-term reference.",
        "",
        "Interpretation discipline:",
        "Changing term count changes the model ensemble. This test probes",
        "generalization; it does not alter or invalidate the frozen 11-term result.",
    ]
    return "\n".join(lines) + "\n"


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def m_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    sx, sy = np.std(rx), np.std(ry)
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--term-count", type=int, choices=TERM_COUNTS)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=12)
    parser.add_argument("--nperm", type=int, default=1000)
    parser.add_argument("--output-prefix", type=str, default="v32_15")
    args = parser.parse_args()

    counts = TERM_COUNTS if args.term_count is None else (args.term_count,)
    all_rows, all_stats = [], []
    for nt in counts:
        rows, stats = run_block(nt, args.seed_start, args.seed_count, args.nperm)
        all_rows.extend(rows)
        all_stats.extend(stats)

    prefix = Path(args.output_prefix)
    write_csv(prefix.with_name(prefix.name + "_rows.csv"), all_rows)
    write_csv(prefix.with_name(prefix.name + "_stats.csv"), all_stats)

    report = summarize(all_rows, all_stats)
    prefix.with_name(prefix.name + "_output.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
