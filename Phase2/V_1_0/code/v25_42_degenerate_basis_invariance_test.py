#!/usr/bin/env python3
"""Soft Spaces Phase 2 v25.42 — degenerate-eigenbasis invariance test.

Predeclared before execution
----------------------------
* Dimension: 8Q, using the same five-term random-Pauli ensemble as v25.31.
* Frozen source coordinates: 31 and 95; two target branches separated by 128.
* Frozen controls: source coordinate +/-10, step 2.
* Families: dephasing Z/ZZ and transverse X/XX; one frozen perturbation per family.
* Hamiltonian seeds: 25_042_000 ... 25_042_011.
* Eight deterministic Haar rotations inside every numerically identified exact
  degenerate eigenspace.
* H, its eigenvalues, perturbations, coordinates, controls, score formula,
  eps_neighbor=0.05 and energy_reg=1e-3 remain unchanged.

The test rotates REAL and NULL eigenbases separately.  Since a unitary rotation
inside an exactly degenerate eigenspace leaves the Hamiltonian unchanged, any
change of the reported prominence proves that the statistic is not a function
of H and V alone.  A sign change is a direct counterexample to hotspot-sign
invariance.

This is a falsification test.  No seed is discarded because its candidate is
ineligible; ineligible seeds are reported explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


VERSION = "v25.42"
N_QUBITS = 8
SOURCE_QUBITS = 7
DIM = 1 << N_QUBITS
SOURCE_DIM = 1 << SOURCE_QUBITS
N_TERMS = 5
PERTURB_TERMS = 5
FROZEN = (31, 95)
LOCAL_RADIUS = 10
LOCAL_STEP = 2
EPS_NEIGHBOR = 0.05
ENERGY_REG = 1.0e-3
DEGENERACY_TOL = 1.0e-10
HAMILTONIAN_SEEDS = tuple(range(25_042_000, 25_042_012))
ROTATIONS = 8
FAMILIES = ("dephasing", "transverse")
PAULIS = ("I", "X", "Y", "Z")


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def index_to_label(index: int, n_qubits: int) -> str:
    value = int(index) + 1
    digits: list[str] = []
    for _ in range(n_qubits):
        value, remainder = divmod(value, 4)
        digits.append(PAULIS[remainder])
    return "".join(reversed(digits))


def pauli_permutation_phase(label: str) -> tuple[np.ndarray, np.ndarray]:
    basis = np.arange(1 << len(label), dtype=np.int64)
    flip_mask = 0
    phase = np.ones(basis.size, dtype=np.complex128)
    for position, symbol in enumerate(label):
        bit_number = len(label) - 1 - position
        bit = (basis >> bit_number) & 1
        if symbol in ("X", "Y"):
            flip_mask |= 1 << bit_number
        if symbol == "Z":
            phase *= 1.0 - 2.0 * bit
        elif symbol == "Y":
            phase *= 1j * (1.0 - 2.0 * bit)
    return basis ^ flip_mask, phase


def dense_pauli_sum(n_qubits: int, terms: Iterable[tuple[str, float]]) -> np.ndarray:
    dimension = 1 << n_qubits
    columns = np.arange(dimension, dtype=np.int64)
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for label, coefficient in terms:
        rows, phase = pauli_permutation_phase(label)
        matrix[rows, columns] += float(coefficient) * phase
    return matrix


def apply_pauli_sum(columns: np.ndarray, terms: Iterable[tuple[str, float]]) -> np.ndarray:
    result = np.zeros_like(columns, dtype=np.complex128)
    for label, coefficient in terms:
        rows, phase = pauli_permutation_phase(label)
        result[rows, :] += float(coefficient) * phase[:, None] * columns
    return result


def random_hamiltonian_terms(seed: int) -> list[tuple[str, float]]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 4**N_QUBITS - 1, size=N_TERMS)
    coefficients = rng.uniform(-1.0, 1.0, size=N_TERMS)
    return [
        (index_to_label(int(index), N_QUBITS), float(coefficient))
        for index, coefficient in zip(indices, coefficients)
    ]


def family_terms(seed: int, family: str) -> list[tuple[str, float]]:
    rng = np.random.default_rng(seed)
    symbol = "Z" if family == "dephasing" else "X"
    labels: list[str] = []
    for qubit in range(N_QUBITS):
        label = ["I"] * N_QUBITS
        label[qubit] = symbol
        labels.append("".join(label))
    for qubit in range(N_QUBITS - 1):
        label = ["I"] * N_QUBITS
        label[qubit] = symbol
        label[qubit + 1] = symbol
        labels.append("".join(label))
    selected = rng.choice(len(labels), size=PERTURB_TERMS, replace=False)
    coefficients = rng.uniform(-1.0, 1.0, size=PERTURB_TERMS)
    return [(labels[int(index)], float(coefficient)) for index, coefficient in zip(selected, coefficients)]


def haar_unitary(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = (rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))) / math.sqrt(2.0)
    q, r = np.linalg.qr(raw)
    diagonal = np.diag(r)
    phases = np.ones_like(diagonal)
    nonzero = np.abs(diagonal) > 0.0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    return q * phases.conj()[None, :]


def controls(center: int) -> list[int]:
    return [
        coordinate
        for coordinate in range(center - LOCAL_RADIUS, center + LOCAL_RADIUS + 1, LOCAL_STEP)
        if coordinate != center
    ]


ALL_SOURCE_COORDINATES = sorted({coordinate for center in FROZEN for coordinate in [center] + controls(center)})
SELECTED_INDICES = sorted({
    index
    for coordinate in ALL_SOURCE_COORDINATES
    for branch in range(2)
    for index in (coordinate + branch * SOURCE_DIM, coordinate + branch * SOURCE_DIM + 1)
})
COLUMN_POSITIONS = {index: position for position, index in enumerate(SELECTED_INDICES)}


def finite_median(values: Iterable[float]) -> float:
    array = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    return float(np.median(array)) if array.size else float("nan")


def cancel_from_selected_columns(
    eigenvalues: np.ndarray,
    b_columns: np.ndarray,
    left_index: int,
) -> float | None:
    right_index = left_index + 1
    if abs(float(eigenvalues[right_index] - eigenvalues[left_index])) >= EPS_NEIGHBOR:
        return None
    pair_columns = b_columns[:, [COLUMN_POSITIONS[left_index], COLUMN_POSITIONS[right_index]]]
    keep = np.ones(eigenvalues.size, dtype=bool)
    keep[[left_index, right_index]] = False
    wq = pair_columns[keep, :]
    eq = eigenvalues[keep]
    reference = 0.5 * float(eigenvalues[left_index] + eigenvalues[right_index])
    delta = reference - eq
    inverse = delta / (delta * delta + ENERGY_REG * ENERGY_REG)
    a, b = wq[:, 0], wq[:, 1]
    total = np.array([
        [np.sum(inverse * np.abs(a) ** 2), np.sum(inverse * np.conj(a) * b)],
        [np.sum(inverse * np.conj(b) * a), np.sum(inverse * np.abs(b) ** 2)],
    ], dtype=np.complex128)
    row_norms = np.sqrt(np.abs(a) ** 4 + np.abs(b) ** 4 + 2.0 * np.abs(np.conj(a) * b) ** 2)
    denominator = float(np.sum(np.abs(inverse) * row_norms))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return None
    return float(np.linalg.norm(total, ord="fro") / denominator)


def model_values(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    perturbation: list[tuple[str, float]],
) -> dict[int, float]:
    selected = eigenvectors[:, SELECTED_INDICES]
    acted = apply_pauli_sum(selected, perturbation)
    b_columns = eigenvectors.conj().T @ acted
    output: dict[int, float] = {}
    for coordinate in ALL_SOURCE_COORDINATES:
        branch_values: list[float] = []
        for branch in range(2):
            left = coordinate + branch * SOURCE_DIM
            value = cancel_from_selected_columns(eigenvalues, b_columns, left)
            if value is not None:
                branch_values.append(value)
        if branch_values:
            output[coordinate] = finite_median(branch_values)
    return output


def prominence(
    real_by_family: dict[str, dict[int, float]],
    null_by_family: dict[str, dict[int, float]],
    center: int,
) -> float | None:
    def robust(coordinate: int) -> float | None:
        values: list[float] = []
        for family in FAMILIES:
            if coordinate not in real_by_family[family] or coordinate not in null_by_family[family]:
                return None
            values.append(real_by_family[family][coordinate] - null_by_family[family][coordinate])
        return min(values)

    target = robust(center)
    if target is None:
        return None
    background_values = [value for coordinate in controls(center) if (value := robust(coordinate)) is not None]
    if not background_values:
        return None
    return float(target - np.median(background_values))


def degenerate_groups(eigenvalues: np.ndarray) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    start = 0
    for index in range(1, eigenvalues.size + 1):
        if index == eigenvalues.size or abs(float(eigenvalues[index] - eigenvalues[start])) > DEGENERACY_TOL:
            groups.append(np.arange(start, index, dtype=int))
            start = index
    return groups


def rotate_degenerate_basis(
    eigenvectors: np.ndarray,
    groups: list[np.ndarray],
    seed: int,
) -> np.ndarray:
    rotated = eigenvectors.copy()
    for group_number, group in enumerate(groups):
        if group.size <= 1:
            continue
        unitary = haar_unitary(
            int(group.size),
            stable_hash_int(f"{VERSION}|BLOCKROT|{seed}|group{group_number}"),
        )
        rotated[:, group] = eigenvectors[:, group] @ unitary
    return rotated


@dataclass(frozen=True)
class CandidateResult:
    h_seed: int
    center: int
    group_count: int
    min_group_size: int
    max_group_size: int
    real_h_reconstruction_error: float
    null_h_reconstruction_error: float
    baseline: float | None
    real_rotation_min: float | None
    real_rotation_max: float | None
    null_rotation_min: float | None
    null_rotation_max: float | None
    real_signs: tuple[int, int]
    null_signs: tuple[int, int]


def sign_counts(values: list[float]) -> tuple[int, int]:
    return sum(value > 0.0 for value in values), sum(value < 0.0 for value in values)


def run_hamiltonian(seed: int) -> list[CandidateResult]:
    hamiltonian = dense_pauli_sum(N_QUBITS, random_hamiltonian_terms(seed))
    eigenvalues, real_vectors = np.linalg.eigh(hamiltonian)
    null_vectors = haar_unitary(DIM, stable_hash_int(f"{VERSION}|NULL|{seed}"))
    null_hamiltonian = null_vectors @ np.diag(eigenvalues) @ null_vectors.conj().T
    groups = degenerate_groups(eigenvalues)
    perturbations = {
        family: family_terms(stable_hash_int(f"{VERSION}|{family}|PERT|{seed}"), family)
        for family in FAMILIES
    }

    real_base = {family: model_values(eigenvalues, real_vectors, perturbations[family]) for family in FAMILIES}
    null_base = {family: model_values(eigenvalues, null_vectors, perturbations[family]) for family in FAMILIES}
    baselines = {center: prominence(real_base, null_base, center) for center in FROZEN}
    real_rotated: dict[int, list[float]] = {center: [] for center in FROZEN}
    null_rotated: dict[int, list[float]] = {center: [] for center in FROZEN}
    max_real_reconstruction_error = 0.0
    max_null_reconstruction_error = 0.0

    for rotation in range(ROTATIONS):
        real_rotation = rotate_degenerate_basis(
            real_vectors, groups, stable_hash_int(f"{VERSION}|REALROT|{seed}|{rotation}")
        )
        null_rotation = rotate_degenerate_basis(
            null_vectors, groups, stable_hash_int(f"{VERSION}|NULLROT|{seed}|{rotation}")
        )
        reconstructed = real_rotation @ np.diag(eigenvalues) @ real_rotation.conj().T
        reconstructed_null = null_rotation @ np.diag(eigenvalues) @ null_rotation.conj().T
        max_real_reconstruction_error = max(
            max_real_reconstruction_error,
            float(np.linalg.norm(reconstructed - hamiltonian, ord=2)),
        )
        max_null_reconstruction_error = max(
            max_null_reconstruction_error,
            float(np.linalg.norm(reconstructed_null - null_hamiltonian, ord=2)),
        )
        real_values = {
            family: model_values(eigenvalues, real_rotation, perturbations[family])
            for family in FAMILIES
        }
        null_values = {
            family: model_values(eigenvalues, null_rotation, perturbations[family])
            for family in FAMILIES
        }
        for center in FROZEN:
            real_value = prominence(real_values, null_base, center)
            null_value = prominence(real_base, null_values, center)
            if real_value is not None:
                real_rotated[center].append(real_value)
            if null_value is not None:
                null_rotated[center].append(null_value)

    results: list[CandidateResult] = []
    sizes = [int(group.size) for group in groups]
    for center in FROZEN:
        rv = real_rotated[center]
        nv = null_rotated[center]
        results.append(CandidateResult(
            h_seed=seed,
            center=center,
            group_count=len(groups),
            min_group_size=min(sizes),
            max_group_size=max(sizes),
            real_h_reconstruction_error=max_real_reconstruction_error,
            null_h_reconstruction_error=max_null_reconstruction_error,
            baseline=baselines[center],
            real_rotation_min=min(rv) if rv else None,
            real_rotation_max=max(rv) if rv else None,
            null_rotation_min=min(nv) if nv else None,
            null_rotation_max=max(nv) if nv else None,
            real_signs=sign_counts(rv),
            null_signs=sign_counts(nv),
        ))
    return results


def fmt(value: float | None) -> str:
    return "INELIGIBLE" if value is None else f"{value:+.6f}"


def interval_crosses_zero(low: float | None, high: float | None) -> bool:
    return low is not None and high is not None and low < 0.0 < high


def render(results: list[CandidateResult]) -> str:
    eligible = [result for result in results if result.baseline is not None]
    real_cross = sum(interval_crosses_zero(result.real_rotation_min, result.real_rotation_max) for result in eligible)
    null_cross = sum(interval_crosses_zero(result.null_rotation_min, result.null_rotation_max) for result in eligible)
    baseline_sign_changed_real = sum(
        result.baseline is not None
        and result.real_rotation_min is not None
        and result.real_rotation_max is not None
        and ((result.baseline > 0 and result.real_rotation_min < 0) or (result.baseline < 0 and result.real_rotation_max > 0))
        for result in eligible
    )
    baseline_sign_changed_null = sum(
        result.baseline is not None
        and result.null_rotation_min is not None
        and result.null_rotation_max is not None
        and ((result.baseline > 0 and result.null_rotation_min < 0) or (result.baseline < 0 and result.null_rotation_max > 0))
        for result in eligible
    )
    max_real_h_error = max(result.real_h_reconstruction_error for result in results)
    max_null_h_error = max(result.null_h_reconstruction_error for result in results)

    lines = [
        "=== Soft Spaces Phase 2 v25.42 DEGENERATE-EIGENBASIS INVARIANCE TEST ===",
        f"8Q five-term random-Pauli Hamiltonians: {len(HAMILTONIAN_SEEDS)} frozen seeds",
        f"Frozen candidates: {list(FROZEN)}; rotations per degenerate basis: {ROTATIONS}",
        "REAL and NULL rotated separately; H, spectrum, V, coordinates, controls and score unchanged.",
        "",
        "seed | k | groups | group size min:max | baseline | REAL-rot range | signs +:- | NULL-rot range | signs +:-",
    ]
    for result in results:
        lines.append(
            f"{result.h_seed} | {result.center:2d} | {result.group_count:6d} | "
            f"{result.min_group_size:4d}:{result.max_group_size:<4d} | {fmt(result.baseline):>10s} | "
            f"[{fmt(result.real_rotation_min)},{fmt(result.real_rotation_max)}] | "
            f"{result.real_signs[0]}:{result.real_signs[1]} | "
            f"[{fmt(result.null_rotation_min)},{fmt(result.null_rotation_max)}] | "
            f"{result.null_signs[0]}:{result.null_signs[1]}"
        )
    lines.extend([
        "",
        f"Eligible seed-candidate instances: {len(eligible)}/{len(results)}",
        f"Maximum REAL ||U_rot diag(E) U_rot^dagger - H_REAL||_2: {max_real_h_error:.6e}",
        f"Maximum NULL ||U_rot diag(E) U_rot^dagger - H_NULL||_2: {max_null_h_error:.6e}",
        f"REAL-rotation ranges crossing zero: {real_cross}/{len(eligible)}",
        f"NULL-rotation ranges crossing zero: {null_cross}/{len(eligible)}",
        f"REAL rotations reversing the baseline sign at least once: {baseline_sign_changed_real}/{len(eligible)}",
        f"NULL rotations reversing the baseline sign at least once: {baseline_sign_changed_null}/{len(eligible)}",
        "",
        "Decision rule:",
        "  Any non-zero prominence range proves numerical non-invariance; any sign reversal is a direct",
        "  counterexample to hotspot-sign invariance for an unchanged Hamiltonian and perturbation.",
        "",
        "Scope:",
        "  The exact logical counterexample applies to the score definition wherever a selected 2D pair is",
        "  only an arbitrary slice of a larger degenerate eigenspace.  The finite run instantiates that issue",
        "  in the same five-term Pauli ensemble at 8Q; it does not re-run the expensive frozen 12Q matrices.",
    ])
    if baseline_sign_changed_real or baseline_sign_changed_null:
        lines.extend([
            "",
            "CONCLUSION: BASIS-INVARIANCE FALSIFIED.",
            "The hotspot sign can change under a legal eigenbasis rotation that leaves H and its spectrum",
            "unchanged.  The present pair score is therefore not a well-defined physical observable of the",
            "highly degenerate five-term Hamiltonian.  A valid replacement must score complete degenerate",
            "eigenspace projectors or otherwise specify and physically justify a basis-selection rule.",
        ])
    else:
        lines.extend([
            "",
            "CONCLUSION: NO SIGN COUNTEREXAMPLE IN THIS FINITE RUN; numerical invariance still requires proof.",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v25_42_degenerate_basis_invariance_test_output.txt"),
    )
    args = parser.parse_args()
    results = [result for seed in HAMILTONIAN_SEEDS for result in run_hamiltonian(seed)]
    report = render(results)
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
