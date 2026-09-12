#!/usr/bin/env python3
"""Soft Spaces Phase 3 Closure v31.9 — Hamiltonian robustness.

Predeclared purpose
-------------------
Test whether the frozen Phase-3 REAL>NULL ±E-correlation enhancement survives
small controlled changes in Hamiltonian term count within the same random-Pauli
Hamiltonian family.

Frozen items
------------
* 8Q model and exact-eigenspace projector score from v31.7/v30.7.
* Same 12 independent Hamiltonian seeds.
* Same perturbation families, regularization, degeneracy tolerance, ±E pairing,
  local prominence radius, and robust-family minimum.
* No parameter tuning after results are observed.

Controlled Hamiltonian variants
-------------------------------
9, 11, and 13 random Pauli terms. 11 terms is the reference model; 9 and 13
are symmetric nearby term-count perturbations.

Directional robustness gate per variant
----------------------------------------
A variant is directionally robust if BOTH hold:
1. median(delta_corr) > 0
2. at least 9/12 seeds have delta_corr > 0

Overall decision
----------------
PASS            : 9, 11 and 13 terms all pass.
PASS WITH SCOPE : 11 terms passes and exactly one neighbouring variant passes.
FAIL            : otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import v31_7_frozen_base as base

VERSION = "v31.9"
TERM_COUNTS = (9, 11, 13)
MIN_POSITIVE = 9


def summarize(rows: list[dict]) -> dict:
    delta = np.asarray([r["delta_corr"] for r in rows], dtype=float)
    positive = int(np.sum(delta > 0.0))
    median = float(np.median(delta))
    mean = float(np.mean(delta))
    return {
        "positive_seeds": positive,
        "n_seeds": int(delta.size),
        "mean_delta_corr": mean,
        "median_delta_corr": median,
        "passes_directional_gate": bool(median > 0.0 and positive >= MIN_POSITIVE),
    }


def run_variant(n_terms: int) -> tuple[list[dict], dict]:
    base.N_TERMS = int(n_terms)
    rows = [base.run_seed(seed) for seed in base.HAMILTONIAN_SEEDS]
    for row in rows:
        row["n_terms"] = n_terms
    return rows, summarize(rows)


def decide(summary: dict[int, dict]) -> str:
    passed = {k: bool(v["passes_directional_gate"]) for k, v in summary.items()}
    if all(passed.values()):
        return "PASS"
    if passed[11] and sum(int(passed[k]) for k in (9, 13)) == 1:
        return "PASS WITH SCOPE"
    return "FAIL"


def render(all_rows: dict[int, list[dict]], summary: dict[int, dict], decision: str) -> str:
    lines = [
        "=== Soft Spaces Phase 3 Closure v31.9 — HAMILTONIAN ROBUSTNESS ===",
        "Frozen reference: v31.7 / v30.7 physical model",
        f"Hamiltonian term-count variants: {TERM_COUNTS}",
        f"Directional gate: median(delta_corr)>0 and >= {MIN_POSITIVE}/12 positive seeds",
        "",
    ]
    for n_terms in TERM_COUNTS:
        lines.append(f"--- {n_terms}-TERM HAMILTONIAN ---")
        lines.append("seed | corr_REAL | corr_NULL | delta_corr")
        for row in all_rows[n_terms]:
            lines.append(
                f"{row['seed']} | {row['corr_real']:+.6f} | "
                f"{row['corr_null']:+.6f} | {row['delta_corr']:+.6f}"
            )
        s = summary[n_terms]
        lines += [
            f"Positive seeds: {s['positive_seeds']}/{s['n_seeds']}",
            f"Mean delta_corr: {s['mean_delta_corr']:+.6f}",
            f"Median delta_corr: {s['median_delta_corr']:+.6f}",
            f"Directional gate: {'PASS' if s['passes_directional_gate'] else 'FAIL'}",
            "",
        ]
    lines += [
        "=== v31.9 CLOSURE DECISION ===",
        decision,
        "",
        "Interpretation:",
        "  PASS means the REAL-specific ±E enhancement survives both nearby",
        "  Hamiltonian term-count variations without tuning.",
        "  PASS WITH SCOPE means the reference survives and robustness is",
        "  supported on only one side of the neighbouring term-count range.",
        "  FAIL means Hamiltonian robustness is not established by this test.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("v31_9_hamiltonian_robustness_output.txt"))
    parser.add_argument("--json", type=Path, default=Path("v31_9_hamiltonian_robustness.json"))
    args = parser.parse_args()

    all_rows: dict[int, list[dict]] = {}
    summary: dict[int, dict] = {}
    for n_terms in TERM_COUNTS:
        rows, s = run_variant(n_terms)
        all_rows[n_terms] = rows
        summary[n_terms] = s

    decision = decide(summary)
    report = render(all_rows, summary, decision)
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")
    args.json.write_text(json.dumps({
        "version": VERSION,
        "test": "hamiltonian_robustness",
        "term_counts": TERM_COUNTS,
        "summary": summary,
        "decision": decision,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
