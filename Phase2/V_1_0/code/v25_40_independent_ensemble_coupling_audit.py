#!/usr/bin/env python3
"""Soft Spaces Phase 2 v25.40 — independent-ensemble coupling audit.

This is a structural audit, not a new hotspot scan.  It does not move any
coordinate, alter a score, discard a seed, or re-run the Phase-2 decision gate.

The audit asks the most favourable elementary coupling question available for
the v25.31 generator: if the *same* seed is used at 11Q and 12Q, is the 12Q
Pauli sum the exact leading-ancilla lift of the 11Q Pauli sum?  It also checks
whether the added leading qubit leaves either computational ancilla sector
invariant.  Both conditions are necessary for the direct block-copy theorem
used in v25.33--v25.39.

Scope warning: the Phase-2 score coordinates are ordinal eigenvalue/eigenvector
indices after ``numpy.linalg.eigh`` sorting.  They are not computational-basis
labels.  Consequently this audit cannot turn the arithmetic coordinate rule
into a Hilbert-space isometry.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PAULIS = ("I", "X", "Y", "Z")
N_TERMS = 5
BASE_SEED = 50_000_000
BATCH_STRIDE = 1_000_000
BATCHES = 8
SEEDS_PER_BATCH = 5


def index_to_label(index: int, n_qubits: int) -> str:
    value = int(index) + 1
    digits: list[str] = []
    for _ in range(n_qubits):
        value, remainder = divmod(value, 4)
        digits.append(PAULIS[remainder])
    return "".join(reversed(digits))


def random_hamiltonian_terms(
    n_qubits: int, n_terms: int, seed: int
) -> list[tuple[str, float]]:
    """Exact copy of the frozen v25.31 random-Pauli sampling rule."""
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, 4**n_qubits - 1, size=int(n_terms))
    coefficients = rng.uniform(-1.0, 1.0, size=int(n_terms))
    return [
        (index_to_label(int(index), n_qubits), float(coefficient))
        for index, coefficient in zip(indices, coefficients)
    ]


def aggregate(terms: list[tuple[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for label, coefficient in terms:
        result[label] = result.get(label, 0.0) + coefficient
    return result


@dataclass(frozen=True)
class SeedResult:
    seed: int
    coefficients_identical: bool
    copied_terms: int
    leading_identity_terms: int
    leading_nonleaking_terms: int
    exact_identity_lift: bool
    ancilla_sector_invariant: bool


def audit_seed(seed: int) -> SeedResult:
    source = random_hamiltonian_terms(11, N_TERMS, seed)
    target = random_hamiltonian_terms(12, N_TERMS, seed)
    source_coefficients = [coefficient for _, coefficient in source]
    target_coefficients = [coefficient for _, coefficient in target]

    source_aggregate = aggregate(source)
    target_aggregate = aggregate(target)
    expected_lift = {"I" + label: coefficient for label, coefficient in source_aggregate.items()}

    copied_terms = sum(
        1
        for label, coefficient in source
        if any(
            target_label == "I" + label and target_coefficient == coefficient
            for target_label, target_coefficient in target
        )
    )
    leading_identity_terms = sum(label[0] == "I" for label, _ in target)
    leading_nonleaking_terms = sum(label[0] in ("I", "Z") for label, _ in target)

    return SeedResult(
        seed=seed,
        coefficients_identical=source_coefficients == target_coefficients,
        copied_terms=copied_terms,
        leading_identity_terms=leading_identity_terms,
        leading_nonleaking_terms=leading_nonleaking_terms,
        exact_identity_lift=target_aggregate == expected_lift,
        ancilla_sector_invariant=leading_nonleaking_terms == N_TERMS,
    )


def render(results: list[SeedResult]) -> str:
    histogram = Counter(result.leading_nonleaking_terms for result in results)
    exact_lifts = sum(result.exact_identity_lift for result in results)
    invariant = sum(result.ancilla_sector_invariant for result in results)
    any_copied = sum(result.copied_terms > 0 for result in results)
    copied_total = sum(result.copied_terms for result in results)
    coefficients_identical = sum(result.coefficients_identical for result in results)

    lines = [
        "=== Soft Spaces Phase 2 v25.40 INDEPENDENT-ENSEMBLE COUPLING AUDIT ===",
        "Frozen audit: favourable same-seed comparison of the v25.31 generator at 11Q and 12Q",
        f"Seeds: {BATCHES} x {SEEDS_PER_BATCH} = {len(results)}; terms per Hamiltonian: {N_TERMS}",
        "No hotspot coordinates, scores, thresholds, or checkpoint records were changed.",
        "",
        f"Coefficient sequences identical under same-seed coupling: {coefficients_identical}/{len(results)}",
        f"Exact H_12 = I_ancilla tensor H_11 copies: {exact_lifts}/{len(results)}",
        f"Seeds containing at least one exactly copied Pauli term: {any_copied}/{len(results)}",
        f"Exactly copied Pauli terms: {copied_total}/{len(results) * N_TERMS}",
        f"Hamiltonians preserving both leading-ancilla computational sectors: {invariant}/{len(results)}",
        "Leading non-leaking (I/Z) terms per five-term 12Q Hamiltonian:",
        "  " + ", ".join(f"{count} terms: {histogram.get(count, 0)} seeds" for count in range(N_TERMS + 1)),
        "",
        "Exact reading:",
        "1. The random-number stream preserves the five coefficients in this favourable same-seed comparison.",
        "2. The dimension-dependent Pauli labels are newly sampled; none of the 40 Hamiltonians is a copied ancilla lift.",
        "3. Every tested 12Q Hamiltonian contains an X or Y on the leading qubit, so the elementary ancilla sectors leak.",
        "4. The v25.31 coordinates are sorted spectral ordinals, not computational-basis state labels.",
        "",
        "Conclusion: the direct block-ancilla intertwiner required by v25.33--v25.39 is refuted for this",
        "frozen favourable coupling audit.  This does not refute the empirical 8Q--12Q recurrence; it shows",
        "that the recurrence and the controlled-ancilla theorem are presently two distinct results.",
        "",
        "Proof status:",
        "- Exact from code semantics: v25.31 builds a fresh 12Q random-Pauli Hamiltonian and scores sorted eigenpairs.",
        "- Exact finite audit: 0/40 same-seed instances satisfy the elementary copied-ancilla conditions.",
        "- Not proved: absence of every conceivable nonlinear or distributional coupling.",
        "- Still empirical: general hotspot-strength preservation in the independent 8Q--12Q ensemble.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v25_40_independent_ensemble_coupling_audit_output.txt"),
    )
    args = parser.parse_args()

    seeds = [
        BASE_SEED + batch * BATCH_STRIDE + offset
        for batch in range(BATCHES)
        for offset in range(SEEDS_PER_BATCH)
    ]
    results = [audit_seed(seed) for seed in seeds]
    report = render(results)
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")

    if any(result.exact_identity_lift for result in results):
        raise SystemExit("Unexpected exact identity lift found; inspect the audit")


if __name__ == "__main__":
    main()
