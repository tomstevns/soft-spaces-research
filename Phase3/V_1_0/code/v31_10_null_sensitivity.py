#!/usr/bin/env python3
"""Soft Spaces Phase 3 Closure v31.10 — NULL sensitivity.

Purpose
-------
Test whether the REAL>NULL ±E-correlation enhancement depends critically on
one particular NULL construction.

NULL variants
-------------
1. global_haar  : original frozen v31.7 NULL.
2. product_haar : tensor product of independently Haar-random single-qubit
                  bases; destroys the REAL eigenbasis while retaining a local
                  product structure.
3. permuted_real: the exact REAL eigenvectors with columns randomly permuted;
                  preserves the eigenvector set but destroys the association
                  between energy eigenspaces and eigenvectors.

Directional gate per NULL
-------------------------
median(delta_corr)>0 and at least 9/12 positive seeds.

Overall decision
----------------
PASS            : all three NULL constructions pass.
PASS WITH SCOPE : original global-Haar passes and at least one alternative passes.
FAIL            : otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np
import v31_7_frozen_base as base

VERSION = "v31.10"
NULL_KINDS = ("global_haar", "product_haar", "permuted_real")
MIN_POSITIVE = 9


def single_qubit_haar(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))) / math.sqrt(2.0)
    q, r = np.linalg.qr(raw)
    d = np.diag(r)
    phases = np.ones_like(d)
    nz = np.abs(d) > 0.0
    phases[nz] = d[nz] / np.abs(d[nz])
    return q * phases.conj()[None, :]


def product_haar_basis(seed: int) -> np.ndarray:
    out = np.asarray([[1.0 + 0.0j]])
    for q in range(base.N_QUBITS):
        sq_seed = base.stable_hash_int(f"{base.FROZEN_MODEL_VERSION}|PRODUCT_NULL|{seed}|q{q}")
        out = np.kron(out, single_qubit_haar(sq_seed))
    return out


def make_null(kind: str, seed: int, real_vectors: np.ndarray) -> np.ndarray:
    if kind == "global_haar":
        return base.haar_unitary(
            base.DIM,
            base.stable_hash_int(f"{base.FROZEN_MODEL_VERSION}|NULL|{seed}|terms{base.N_TERMS}"),
        )
    if kind == "product_haar":
        return product_haar_basis(seed)
    if kind == "permuted_real":
        rng = np.random.default_rng(
            base.stable_hash_int(f"{base.FROZEN_MODEL_VERSION}|PERMUTED_REAL_NULL|{seed}")
        )
        perm = rng.permutation(real_vectors.shape[1])
        return real_vectors[:, perm]
    raise ValueError(kind)


def run_seed(seed: int, null_kind: str) -> dict:
    hamiltonian = base.dense_pauli_sum(base.N_QUBITS, base.random_hamiltonian_terms(seed))
    eigenvalues, real_vectors = np.linalg.eigh(hamiltonian)
    groups = base.degenerate_groups(eigenvalues)
    null_vectors = make_null(null_kind, seed, real_vectors)

    perturbations = {
        family: base.family_terms(
            base.stable_hash_int(f"{base.FROZEN_MODEL_VERSION}|{family}|PERT|{seed}"), family
        )
        for family in base.FAMILIES
    }
    real_by_family = {
        family: base.model_scores(eigenvalues, real_vectors, perturbations[family], groups)
        for family in base.FAMILIES
    }
    null_by_family = {
        family: base.model_scores(eigenvalues, null_vectors, perturbations[family], groups)
        for family in base.FAMILIES
    }
    real_robust = base.robust_family_min(real_by_family, len(groups))
    null_robust = base.robust_family_min(null_by_family, len(groups))
    real_prom = base.prominence_values(real_robust, len(groups))
    null_prom = base.prominence_values(null_robust, len(groups))
    real_vec = base.vector_from_dict(real_prom, len(groups))
    null_vec = base.vector_from_dict(null_prom, len(groups))
    energies = np.asarray([base.eigenspace_energy(eigenvalues, g) for g in groups], dtype=float)
    pairs = base.opposite_energy_pairs(energies)
    corr_real = base.pearson_from_pairs(real_vec, pairs)
    corr_null = base.pearson_from_pairs(null_vec, pairs)
    return {
        "seed": seed,
        "null_kind": null_kind,
        "pairs": len(pairs),
        "corr_real": corr_real,
        "corr_null": corr_null,
        "delta_corr": corr_real - corr_null,
    }


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
    passed = {k: bool(v["passes_directional_gate"]) for k, v in summary.items()}
    if all(passed.values()):
        return "PASS"
    if passed["global_haar"] and (passed["product_haar"] or passed["permuted_real"]):
        return "PASS WITH SCOPE"
    return "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("v31_10_null_sensitivity_output.txt"))
    parser.add_argument("--json", type=Path, default=Path("v31_10_null_sensitivity.json"))
    args = parser.parse_args()

    all_rows: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    for kind in NULL_KINDS:
        rows = [run_seed(seed, kind) for seed in base.HAMILTONIAN_SEEDS]
        all_rows[kind] = rows
        summary[kind] = summarize(rows)

    decision = decide(summary)
    lines = [
        "=== Soft Spaces Phase 3 Closure v31.10 — NULL SENSITIVITY ===",
        f"Directional gate: median(delta_corr)>0 and >= {MIN_POSITIVE}/12 positive seeds",
        "",
    ]
    for kind in NULL_KINDS:
        lines.append(f"--- NULL: {kind} ---")
        lines.append("seed | corr_REAL | corr_NULL | delta_corr")
        for r in all_rows[kind]:
            lines.append(f"{r['seed']} | {r['corr_real']:+.6f} | {r['corr_null']:+.6f} | {r['delta_corr']:+.6f}")
        s = summary[kind]
        lines += [
            f"Positive seeds: {s['positive_seeds']}/{s['n_seeds']}",
            f"Mean delta_corr: {s['mean_delta_corr']:+.6f}",
            f"Median delta_corr: {s['median_delta_corr']:+.6f}",
            f"Directional gate: {'PASS' if s['passes_directional_gate'] else 'FAIL'}",
            "",
        ]
    lines += ["=== v31.10 CLOSURE DECISION ===", decision, ""]
    report = "\n".join(lines)
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")
    args.json.write_text(json.dumps({
        "version": VERSION,
        "test": "null_sensitivity",
        "summary": summary,
        "decision": decision,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
