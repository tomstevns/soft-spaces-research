#!/usr/bin/env python3
"""Soft Spaces Phase 3 Closure v31.11 — perturbation robustness.

Purpose
-------
Resolve whether the REAL-specific ±E organization is carried by both already
frozen perturbation families independently, rather than only by their combined
robust-min score.

No new perturbation family is introduced.

Families
--------
* dephasing: Z / ZZ
* transverse: X / XX
* robust_min: original minimum across both families

Directional gate per family
---------------------------
median(delta_corr)>0 and at least 9/12 positive seeds.

Overall decision
----------------
PASS            : dephasing and transverse both pass.
PASS WITH SCOPE : exactly one individual family passes and robust_min passes.
FAIL            : otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import v31_7_frozen_base as base

VERSION = "v31.11"
CHANNELS = ("dephasing", "transverse", "robust_min")
MIN_POSITIVE = 9


def run_seed(seed: int) -> dict[str, dict]:
    hamiltonian = base.dense_pauli_sum(base.N_QUBITS, base.random_hamiltonian_terms(seed))
    eigenvalues, real_vectors = np.linalg.eigh(hamiltonian)
    groups = base.degenerate_groups(eigenvalues)
    null_vectors = base.haar_unitary(
        base.DIM,
        base.stable_hash_int(f"{base.FROZEN_MODEL_VERSION}|NULL|{seed}|terms{base.N_TERMS}"),
    )
    perturbations = {
        family: base.family_terms(
            base.stable_hash_int(f"{base.FROZEN_MODEL_VERSION}|{family}|PERT|{seed}"), family
        )
        for family in base.FAMILIES
    }
    real_scores = {
        family: base.model_scores(eigenvalues, real_vectors, perturbations[family], groups)
        for family in base.FAMILIES
    }
    null_scores = {
        family: base.model_scores(eigenvalues, null_vectors, perturbations[family], groups)
        for family in base.FAMILIES
    }
    energies = np.asarray([base.eigenspace_energy(eigenvalues, g) for g in groups], dtype=float)
    pairs = base.opposite_energy_pairs(energies)

    output: dict[str, dict] = {}
    for family in base.FAMILIES:
        rp = base.prominence_values(real_scores[family], len(groups))
        np_ = base.prominence_values(null_scores[family], len(groups))
        rv = base.vector_from_dict(rp, len(groups))
        nv = base.vector_from_dict(np_, len(groups))
        cr = base.pearson_from_pairs(rv, pairs)
        cn = base.pearson_from_pairs(nv, pairs)
        output[family] = {"seed": seed, "corr_real": cr, "corr_null": cn, "delta_corr": cr-cn}

    rr = base.robust_family_min(real_scores, len(groups))
    nr = base.robust_family_min(null_scores, len(groups))
    rp = base.prominence_values(rr, len(groups))
    np_ = base.prominence_values(nr, len(groups))
    rv = base.vector_from_dict(rp, len(groups))
    nv = base.vector_from_dict(np_, len(groups))
    cr = base.pearson_from_pairs(rv, pairs)
    cn = base.pearson_from_pairs(nv, pairs)
    output["robust_min"] = {"seed": seed, "corr_real": cr, "corr_null": cn, "delta_corr": cr-cn}
    return output


def summarize(rows: list[dict]) -> dict:
    delta = np.asarray([r["delta_corr"] for r in rows], dtype=float)
    positive = int(np.sum(delta > 0.0))
    median = float(np.median(delta))
    return {
        "positive_seeds": positive,
        "n_seeds": int(delta.size),
        "mean_delta_corr": float(np.mean(delta)),
        "median_delta_corr": median,
        "passes_directional_gate": bool(median > 0.0 and positive >= MIN_POSITIVE),
    }


def decide(summary: dict[str, dict]) -> str:
    d = bool(summary["dephasing"]["passes_directional_gate"])
    t = bool(summary["transverse"]["passes_directional_gate"])
    r = bool(summary["robust_min"]["passes_directional_gate"])
    if d and t:
        return "PASS"
    if (d ^ t) and r:
        return "PASS WITH SCOPE"
    return "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("v31_11_perturbation_robustness_output.txt"))
    parser.add_argument("--json", type=Path, default=Path("v31_11_perturbation_robustness.json"))
    args = parser.parse_args()

    all_rows = {channel: [] for channel in CHANNELS}
    for seed in base.HAMILTONIAN_SEEDS:
        result = run_seed(seed)
        for channel in CHANNELS:
            all_rows[channel].append(result[channel])

    summary = {channel: summarize(all_rows[channel]) for channel in CHANNELS}
    decision = decide(summary)

    lines = [
        "=== Soft Spaces Phase 3 Closure v31.11 — PERTURBATION ROBUSTNESS ===",
        f"Directional gate: median(delta_corr)>0 and >= {MIN_POSITIVE}/12 positive seeds",
        "",
    ]
    for channel in CHANNELS:
        lines.append(f"--- CHANNEL: {channel} ---")
        lines.append("seed | corr_REAL | corr_NULL | delta_corr")
        for r in all_rows[channel]:
            lines.append(f"{r['seed']} | {r['corr_real']:+.6f} | {r['corr_null']:+.6f} | {r['delta_corr']:+.6f}")
        s = summary[channel]
        lines += [
            f"Positive seeds: {s['positive_seeds']}/{s['n_seeds']}",
            f"Mean delta_corr: {s['mean_delta_corr']:+.6f}",
            f"Median delta_corr: {s['median_delta_corr']:+.6f}",
            f"Directional gate: {'PASS' if s['passes_directional_gate'] else 'FAIL'}",
            "",
        ]
    lines += ["=== v31.11 CLOSURE DECISION ===", decision, ""]
    report = "\n".join(lines)
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")
    args.json.write_text(json.dumps({
        "version": VERSION,
        "test": "perturbation_robustness",
        "summary": summary,
        "decision": decision,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
