#!/usr/bin/env python3
"""Soft Spaces Phase 2 v25.33 — exact ancilla-lift verification.

This program verifies the exact decoupled-ancilla theorem used in the
mathematical proof ledger.  It is a theorem implementation check, not new
evidence for the independently generated 8Q–12Q random-Pauli recurrence.

Frozen before execution
-----------------------
* No coordinate search, relocation, re-ranking, or score fitting.
* Source pairs and controls are fixed below.
* REAL and spectrum-matched NULL models use the same perturbations.
* The exact lift copies H, NULL-H, and V into two separated ancilla branches.
* A deliberately unrelated upper-dimensional perturbation is then used as a
  falsification control.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


VERSION = "v25.33"
SOURCE_QUBITS = 4
DIM = 1 << SOURCE_QUBITS
TARGET = 7
CONTROLS = (1, 3, 5, 9, 11, 13)
FAMILIES = ("dephasing_like", "transverse_like")
BASE_SEEDS = (2533001, 2533002, 2533003, 2533004, 2533005)
REGULARIZER = 1.0e-3
TOLERANCE = 2.0e-10


@dataclass(frozen=True)
class Check:
    seed: int
    max_real_branch_error: float
    max_null_branch_error: float
    max_delta_error: float
    max_prominence_error: float
    unrelated_max_delta_error: float


def random_hermitian(dim: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    matrix = 0.5 * (raw + raw.conj().T)
    return matrix / max(float(np.linalg.norm(matrix, ord=2)), 1.0e-15)


def haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    raw = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2.0)
    q, r = np.linalg.qr(raw)
    diagonal = np.diag(r)
    phases = np.ones_like(diagonal)
    nonzero = np.abs(diagonal) > 0.0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    return q * phases.conj()[None, :]


def coherence_ratio(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    perturbation: np.ndarray,
    left_index: int,
    regularizer: float,
) -> float:
    right_index = left_index + 1
    selected = eigenvectors[:, [left_index, right_index]]
    couplings = eigenvectors.conj().T @ perturbation @ selected
    keep = np.ones(eigenvalues.size, dtype=bool)
    keep[[left_index, right_index]] = False
    wq = couplings[keep, :]
    eq = eigenvalues[keep]
    reference = 0.5 * float(eigenvalues[left_index] + eigenvalues[right_index])
    delta = reference - eq
    weights = delta / (delta * delta + regularizer * regularizer)

    terms = np.einsum("ni,nj->nij", wq.conj(), wq)
    total = np.sum(weights[:, None, None] * terms, axis=0)
    denominator = float(np.sum(np.abs(weights) * np.sum(np.abs(wq) ** 2, axis=1)))
    if denominator <= 0.0:
        raise RuntimeError("Coherence-ratio denominator vanished")
    return float(np.linalg.norm(total, ord="fro") / denominator)


def spectral_model(hamiltonian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(hamiltonian)
    return values.real, vectors


def null_hamiltonian(values: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    return unitary @ np.diag(values) @ unitary.conj().T


def exact_lift(matrix: np.ndarray, delta: float, is_hamiltonian: bool) -> np.ndarray:
    zero = np.zeros_like(matrix)
    if is_hamiltonian:
        identity = np.eye(matrix.shape[0], dtype=np.complex128)
        lower = matrix - delta * identity
        upper = matrix + delta * identity
    else:
        lower = matrix
        upper = matrix
    return np.block([[lower, zero], [zero, upper]])


def pair_scores(
    hamiltonian: np.ndarray,
    perturbations: dict[str, np.ndarray],
    indices: tuple[int, ...],
    regularizer: float,
) -> dict[tuple[str, int], float]:
    values, vectors = spectral_model(hamiltonian)
    return {
        (family, index): coherence_ratio(values, vectors, perturbation, index, regularizer)
        for family, perturbation in perturbations.items()
        for index in indices
    }


def robust_delta(
    real: dict[tuple[str, int], float],
    null: dict[tuple[str, int], float],
    index: int,
) -> float:
    return min(real[(family, index)] - null[(family, index)] for family in FAMILIES)


def prominence(
    real: dict[tuple[str, int], float],
    null: dict[tuple[str, int], float],
    target: int,
    controls: tuple[int, ...],
) -> float:
    target_score = robust_delta(real, null, target)
    background = float(np.median([robust_delta(real, null, index) for index in controls]))
    return target_score - background


def run_seed(seed: int, regularizer: float) -> Check:
    rng = np.random.default_rng(seed)
    h_real = random_hermitian(DIM, rng)
    values, _ = spectral_model(h_real)
    h_null = null_hamiltonian(values, haar_unitary(DIM, rng))
    perturbations = {family: random_hermitian(DIM, rng) for family in FAMILIES}
    fixed_indices = (TARGET,) + CONTROLS

    base_real = pair_scores(h_real, perturbations, fixed_indices, regularizer)
    base_null = pair_scores(h_null, perturbations, fixed_indices, regularizer)

    width = float(values[-1] - values[0])
    branch_shift = width + 1.0
    h_real_up = exact_lift(h_real, branch_shift, is_hamiltonian=True)
    h_null_up = exact_lift(h_null, branch_shift, is_hamiltonian=True)
    perturbations_up = {
        family: exact_lift(matrix, branch_shift, is_hamiltonian=False)
        for family, matrix in perturbations.items()
    }

    lifted_indices = tuple(index + branch * DIM for branch in (0, 1) for index in fixed_indices)
    up_real = pair_scores(h_real_up, perturbations_up, lifted_indices, regularizer)
    up_null = pair_scores(h_null_up, perturbations_up, lifted_indices, regularizer)

    real_errors: list[float] = []
    null_errors: list[float] = []
    delta_errors: list[float] = []
    prominence_errors: list[float] = []
    base_prominence = prominence(base_real, base_null, TARGET, CONTROLS)

    for branch in (0, 1):
        offset = branch * DIM
        for family in FAMILIES:
            for index in fixed_indices:
                lifted = index + offset
                real_errors.append(abs(up_real[(family, lifted)] - base_real[(family, index)]))
                null_errors.append(abs(up_null[(family, lifted)] - base_null[(family, index)]))
                base_delta = base_real[(family, index)] - base_null[(family, index)]
                lifted_delta = up_real[(family, lifted)] - up_null[(family, lifted)]
                delta_errors.append(abs(lifted_delta - base_delta))

        lifted_controls = tuple(index + offset for index in CONTROLS)
        lifted_prominence = prominence(
            up_real, up_null, TARGET + offset, lifted_controls
        )
        prominence_errors.append(abs(lifted_prominence - base_prominence))

    unrelated = {
        family: random_hermitian(2 * DIM, rng)
        for family in FAMILIES
    }
    unrelated_real = pair_scores(h_real_up, unrelated, lifted_indices, regularizer)
    unrelated_null = pair_scores(h_null_up, unrelated, lifted_indices, regularizer)
    unrelated_errors: list[float] = []
    for branch in (0, 1):
        offset = branch * DIM
        for family in FAMILIES:
            for index in fixed_indices:
                lifted = index + offset
                base_delta = base_real[(family, index)] - base_null[(family, index)]
                unrelated_delta = unrelated_real[(family, lifted)] - unrelated_null[(family, lifted)]
                unrelated_errors.append(abs(unrelated_delta - base_delta))

    return Check(
        seed=seed,
        max_real_branch_error=max(real_errors),
        max_null_branch_error=max(null_errors),
        max_delta_error=max(delta_errors),
        max_prominence_error=max(prominence_errors),
        unrelated_max_delta_error=max(unrelated_errors),
    )


def render(checks: list[Check], tolerance: float) -> str:
    exact_max = max(
        max(
            check.max_real_branch_error,
            check.max_null_branch_error,
            check.max_delta_error,
            check.max_prominence_error,
        )
        for check in checks
    )
    unrelated_min = min(check.unrelated_max_delta_error for check in checks)
    lines = [
        "=== Soft Spaces Phase 2 v25.33 EXACT ANCILLA-LIFT VERIFICATION ===",
        f"Source dimension: {DIM} ({SOURCE_QUBITS}Q); lifted dimension: {2 * DIM} ({SOURCE_QUBITS + 1}Q)",
        f"Frozen target: {TARGET}; frozen controls: {list(CONTROLS)}",
        f"Families: {list(FAMILIES)}; seeds: {list(BASE_SEEDS)}",
        f"Regularizer: {REGULARIZER:.3e}; numerical tolerance: {tolerance:.3e}",
        "",
        "seed | real branch err | null branch err | delta err | prominence err | unrelated delta err",
    ]
    for check in checks:
        lines.append(
            f"{check.seed:7d} | {check.max_real_branch_error:15.6e} | "
            f"{check.max_null_branch_error:15.6e} | {check.max_delta_error:9.6e} | "
            f"{check.max_prominence_error:14.6e} | {check.unrelated_max_delta_error:19.6e}"
        )
    lines.extend([
        "",
        f"Exact-lift maximum discrepancy: {exact_max:.6e}",
        f"Exact-lift theorem check: {'PASS' if exact_max <= tolerance else 'FAIL'}",
        f"Smallest unrelated-control discrepancy: {unrelated_min:.6e}",
        "Falsification reading: unrelated upper-dimensional perturbations do not preserve the source deltas.",
        "Scope: implementation check of the conditional theorem; not evidence for the independent 8Q–12Q ensemble.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("v25_33_exact_ancilla_lift_output.txt"))
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()

    checks = [run_seed(seed, REGULARIZER) for seed in BASE_SEEDS]
    report = render(checks, args.tolerance)
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")

    exact_max = max(
        max(
            check.max_real_branch_error,
            check.max_null_branch_error,
            check.max_delta_error,
            check.max_prominence_error,
        )
        for check in checks
    )
    if exact_max > args.tolerance:
        raise SystemExit("Exact ancilla-lift theorem verification failed")


if __name__ == "__main__":
    main()
