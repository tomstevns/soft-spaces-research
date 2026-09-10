#!/usr/bin/env python3
"""Soft Spaces Phase 3.1 v30.0 — projector/subspace score invariance test.

Purpose
-------
This file is the first Phase 3.1 replacement for the basis-dependent 2D
eigenvector-pair score falsified by v25.42.

The frozen v25.42 test environment is retained as closely as possible:
* Dimension: 8Q.
* Same five-term random-Pauli Hamiltonian ensemble.
* Frozen source coordinates: 31 and 95; two target branches separated by 128.
* Frozen controls: source coordinate +/-10, step 2.
* Perturbation families: dephasing Z/ZZ and transverse X/XX.
* Hamiltonian seeds: 25_042_000 ... 25_042_011.
* Eight deterministic Haar rotations inside every numerically identified exact
  degenerate eigenspace.
* eps_neighbor=0.05 and energy_reg=1e-3.

Principal change
----------------
v25.42 selected two individual eigenvectors and formed a score from that
arbitrary 2D basis slice.  v30.0 replaces that slice by its COMPLETE
degeneracy-closed subspace P:

    P = direct sum of every exact degenerate eigenspace touched by the
        selected neighbouring pair.

For each exact Q eigenspace G outside P, define

    A_G = P V P_G V P,

where P_G is the projector onto G.  In any eigenbasis this can be evaluated as

    A_G = W_G^dagger W_G,

with W_G = <G|V|P>.

The regularised second-order effective operator is

    K = sum_G r_G A_G,

    r_G = (E_ref - E_G) / ((E_ref - E_G)^2 + energy_reg^2).

The cancellation score is

    C(P,V) = ||K||_F / sum_G |r_G| ||A_G||_F.

Both numerator and denominator are invariant under arbitrary unitary basis
rotations within P and within every exact degenerate Q eigenspace.

This is an invariance/regression test, not yet the final Phase 3 physical model.
The immediate gate is that legal degenerate-basis rotations must leave the
projector score unchanged up to numerical precision.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


VERSION = "v30.0"
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

# Numerical regression gate for basis invariance.
INVARIANCE_ATOL = 1.0e-10
INVARIANCE_RTOL = 1.0e-9


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


def dense_pauli_sum(
    n_qubits: int,
    terms: Iterable[tuple[str, float]],
) -> np.ndarray:
    dimension = 1 << n_qubits
    columns = np.arange(dimension, dtype=np.int64)
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for label, coefficient in terms:
        rows, phase = pauli_permutation_phase(label)
        matrix[rows, columns] += float(coefficient) * phase
    return matrix


def apply_pauli_sum(
    columns: np.ndarray,
    terms: Iterable[tuple[str, float]],
) -> np.ndarray:
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

    return [
        (labels[int(index)], float(coefficient))
        for index, coefficient in zip(selected, coefficients)
    ]


def haar_unitary(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = (
        rng.normal(size=(dimension, dimension))
        + 1j * rng.normal(size=(dimension, dimension))
    ) / math.sqrt(2.0)

    q, r = np.linalg.qr(raw)
    diagonal = np.diag(r)

    phases = np.ones_like(diagonal)
    nonzero = np.abs(diagonal) > 0.0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])

    return q * phases.conj()[None, :]


def controls(center: int) -> list[int]:
    return [
        coordinate
        for coordinate in range(
            center - LOCAL_RADIUS,
            center + LOCAL_RADIUS + 1,
            LOCAL_STEP,
        )
        if coordinate != center
    ]


ALL_SOURCE_COORDINATES = sorted(
    {
        coordinate
        for center in FROZEN
        for coordinate in [center] + controls(center)
    }
)


def finite_median(values: Iterable[float]) -> float:
    array = np.asarray(
        [float(value) for value in values if np.isfinite(value)],
        dtype=float,
    )
    return float(np.median(array)) if array.size else float("nan")


def degenerate_groups(eigenvalues: np.ndarray) -> list[np.ndarray]:
    """Partition sorted eigenvalues into exact-degeneracy groups."""
    groups: list[np.ndarray] = []
    start = 0

    for index in range(1, eigenvalues.size + 1):
        if (
            index == eigenvalues.size
            or abs(float(eigenvalues[index] - eigenvalues[start]))
            > DEGENERACY_TOL
        ):
            groups.append(np.arange(start, index, dtype=int))
            start = index

    return groups


def group_lookup(
    groups: list[np.ndarray],
    dimension: int,
) -> np.ndarray:
    """Map every eigenvector index to its exact-degeneracy group number."""
    lookup = np.empty(dimension, dtype=int)
    for group_number, group in enumerate(groups):
        lookup[group] = group_number
    return lookup


def degeneracy_closed_subspace(
    left_index: int,
    groups: list[np.ndarray],
    lookup: np.ndarray,
) -> np.ndarray:
    """Return the full degeneracy closure of the neighbouring pair i,i+1."""
    right_index = left_index + 1
    touched = sorted(
        {
            int(lookup[left_index]),
            int(lookup[right_index]),
        }
    )
    return np.concatenate([groups[group_number] for group_number in touched])


def projector_cancel_score(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    perturbation: list[tuple[str, float]],
    groups: list[np.ndarray],
    lookup: np.ndarray,
    left_index: int,
) -> float | None:
    """Basis-invariant cancellation score for the complete selected subspace.

    Eligibility deliberately retains the v25.42 neighbouring-pair gate.
    Once eligible, however, the selected P space is closed under exact
    degeneracy, so the score cannot depend on which basis was returned inside
    a degenerate eigenspace.
    """
    right_index = left_index + 1

    if right_index >= eigenvalues.size:
        return None

    if (
        abs(float(eigenvalues[right_index] - eigenvalues[left_index]))
        >= EPS_NEIGHBOR
    ):
        return None

    p_indices = degeneracy_closed_subspace(
        left_index,
        groups,
        lookup,
    )

    # P is a complete direct sum of exact eigenspaces.
    p_group_numbers = {
        int(lookup[int(index)])
        for index in p_indices
    }

    p_vectors = eigenvectors[:, p_indices]
    acted_p = apply_pauli_sum(p_vectors, perturbation)

    # B_QP = <Q|V|P>, evaluated only as needed group by group.
    b_to_p = eigenvectors.conj().T @ acted_p

    # Retain the v25.42 midpoint convention at the coordinate level.
    # This reference is independent of the basis inside the selected subspace.
    reference = 0.5 * float(
        eigenvalues[left_index] + eigenvalues[right_index]
    )

    effective = np.zeros(
        (p_indices.size, p_indices.size),
        dtype=np.complex128,
    )
    denominator = 0.0

    for group_number, group in enumerate(groups):
        if group_number in p_group_numbers:
            continue

        # Every state in an exact degenerate group has the same energy.
        e_group = float(np.mean(eigenvalues[group]))
        delta = reference - e_group
        inverse = delta / (
            delta * delta + ENERGY_REG * ENERGY_REG
        )

        w_group = b_to_p[group, :]
        a_group = w_group.conj().T @ w_group

        effective += inverse * a_group
        denominator += (
            abs(inverse)
            * float(np.linalg.norm(a_group, ord="fro"))
        )

    if denominator <= 0.0 or not np.isfinite(denominator):
        return None

    numerator = float(np.linalg.norm(effective, ord="fro"))

    if not np.isfinite(numerator):
        return None

    return numerator / denominator


def model_values(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    perturbation: list[tuple[str, float]],
    groups: list[np.ndarray],
    lookup: np.ndarray,
) -> dict[int, float]:
    output: dict[int, float] = {}

    for coordinate in ALL_SOURCE_COORDINATES:
        branch_values: list[float] = []

        for branch in range(2):
            left = coordinate + branch * SOURCE_DIM
            value = projector_cancel_score(
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                perturbation=perturbation,
                groups=groups,
                lookup=lookup,
                left_index=left,
            )
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
            if (
                coordinate not in real_by_family[family]
                or coordinate not in null_by_family[family]
            ):
                return None

            values.append(
                real_by_family[family][coordinate]
                - null_by_family[family][coordinate]
            )

        return min(values)

    target = robust(center)
    if target is None:
        return None

    background_values = [
        value
        for coordinate in controls(center)
        if (value := robust(coordinate)) is not None
    ]

    if not background_values:
        return None

    return float(target - np.median(background_values))


def rotate_degenerate_basis(
    eigenvectors: np.ndarray,
    groups: list[np.ndarray],
    seed: int,
) -> np.ndarray:
    """Apply independent deterministic Haar rotations inside exact eigenspaces."""
    rotated = eigenvectors.copy()

    for group_number, group in enumerate(groups):
        if group.size <= 1:
            continue

        unitary = haar_unitary(
            int(group.size),
            stable_hash_int(
                f"{VERSION}|BLOCKROT|{seed}|group{group_number}"
            ),
        )
        rotated[:, group] = eigenvectors[:, group] @ unitary

    return rotated


def reconstruction_error(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    hamiltonian: np.ndarray,
) -> float:
    reconstructed = (
        eigenvectors
        @ np.diag(eigenvalues)
        @ eigenvectors.conj().T
    )
    return float(
        np.linalg.norm(
            reconstructed - hamiltonian,
            ord=2,
        )
    )


def max_abs_difference(
    baseline: float | None,
    values: list[float],
) -> float | None:
    if baseline is None or not values:
        return None
    return max(abs(float(value) - baseline) for value in values)


def invariance_pass(
    baseline: float | None,
    max_difference: float | None,
) -> bool | None:
    if baseline is None or max_difference is None:
        return None

    tolerance = (
        INVARIANCE_ATOL
        + INVARIANCE_RTOL * abs(baseline)
    )
    return max_difference <= tolerance


@dataclass(frozen=True)
class CandidateResult:
    h_seed: int
    center: int
    group_count: int
    min_group_size: int
    max_group_size: int
    baseline: float | None
    real_rotation_min: float | None
    real_rotation_max: float | None
    null_rotation_min: float | None
    null_rotation_max: float | None
    real_max_abs_delta: float | None
    null_max_abs_delta: float | None
    real_invariance_pass: bool | None
    null_invariance_pass: bool | None
    max_real_h_reconstruction_error: float
    max_null_h_reconstruction_error: float


def run_hamiltonian(seed: int) -> list[CandidateResult]:
    hamiltonian = dense_pauli_sum(
        N_QUBITS,
        random_hamiltonian_terms(seed),
    )
    eigenvalues, real_vectors = np.linalg.eigh(hamiltonian)

    null_vectors = haar_unitary(
        DIM,
        stable_hash_int(f"{VERSION}|NULL|{seed}"),
    )
    null_hamiltonian = (
        null_vectors
        @ np.diag(eigenvalues)
        @ null_vectors.conj().T
    )

    groups = degenerate_groups(eigenvalues)
    lookup = group_lookup(groups, DIM)

    perturbations = {
        family: family_terms(
            stable_hash_int(
                f"{VERSION}|{family}|PERT|{seed}"
            ),
            family,
        )
        for family in FAMILIES
    }

    real_base = {
        family: model_values(
            eigenvalues,
            real_vectors,
            perturbations[family],
            groups,
            lookup,
        )
        for family in FAMILIES
    }

    null_base = {
        family: model_values(
            eigenvalues,
            null_vectors,
            perturbations[family],
            groups,
            lookup,
        )
        for family in FAMILIES
    }

    baselines = {
        center: prominence(
            real_base,
            null_base,
            center,
        )
        for center in FROZEN
    }

    real_rotated: dict[int, list[float]] = {
        center: []
        for center in FROZEN
    }
    null_rotated: dict[int, list[float]] = {
        center: []
        for center in FROZEN
    }

    max_real_reconstruction_error = 0.0
    max_null_reconstruction_error = 0.0

    for rotation in range(ROTATIONS):
        real_rotation = rotate_degenerate_basis(
            real_vectors,
            groups,
            stable_hash_int(
                f"{VERSION}|REALROT|{seed}|{rotation}"
            ),
        )

        null_rotation = rotate_degenerate_basis(
            null_vectors,
            groups,
            stable_hash_int(
                f"{VERSION}|NULLROT|{seed}|{rotation}"
            ),
        )

        max_real_reconstruction_error = max(
            max_real_reconstruction_error,
            reconstruction_error(
                real_rotation,
                eigenvalues,
                hamiltonian,
            ),
        )

        max_null_reconstruction_error = max(
            max_null_reconstruction_error,
            reconstruction_error(
                null_rotation,
                eigenvalues,
                null_hamiltonian,
            ),
        )

        real_values = {
            family: model_values(
                eigenvalues,
                real_rotation,
                perturbations[family],
                groups,
                lookup,
            )
            for family in FAMILIES
        }

        null_values = {
            family: model_values(
                eigenvalues,
                null_rotation,
                perturbations[family],
                groups,
                lookup,
            )
            for family in FAMILIES
        }

        for center in FROZEN:
            # Rotate REAL only; keep NULL fixed.
            real_value = prominence(
                real_values,
                null_base,
                center,
            )

            # Rotate NULL only; keep REAL fixed.
            null_value = prominence(
                real_base,
                null_values,
                center,
            )

            if real_value is not None:
                real_rotated[center].append(real_value)

            if null_value is not None:
                null_rotated[center].append(null_value)

    sizes = [int(group.size) for group in groups]
    results: list[CandidateResult] = []

    for center in FROZEN:
        rv = real_rotated[center]
        nv = null_rotated[center]
        baseline = baselines[center]

        real_delta = max_abs_difference(baseline, rv)
        null_delta = max_abs_difference(baseline, nv)

        results.append(
            CandidateResult(
                h_seed=seed,
                center=center,
                group_count=len(groups),
                min_group_size=min(sizes),
                max_group_size=max(sizes),
                baseline=baseline,
                real_rotation_min=min(rv) if rv else None,
                real_rotation_max=max(rv) if rv else None,
                null_rotation_min=min(nv) if nv else None,
                null_rotation_max=max(nv) if nv else None,
                real_max_abs_delta=real_delta,
                null_max_abs_delta=null_delta,
                real_invariance_pass=invariance_pass(
                    baseline,
                    real_delta,
                ),
                null_invariance_pass=invariance_pass(
                    baseline,
                    null_delta,
                ),
                max_real_h_reconstruction_error=(
                    max_real_reconstruction_error
                ),
                max_null_h_reconstruction_error=(
                    max_null_reconstruction_error
                ),
            )
        )

    return results


def fmt(value: float | None) -> str:
    return (
        "INELIGIBLE"
        if value is None
        else f"{value:+.9f}"
    )


def fmt_sci(value: float | None) -> str:
    return (
        "INELIGIBLE"
        if value is None
        else f"{value:.3e}"
    )


def fmt_pass(value: bool | None) -> str:
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"


def render(results: list[CandidateResult]) -> str:
    eligible = [
        result
        for result in results
        if result.baseline is not None
    ]

    real_passes = sum(
        result.real_invariance_pass is True
        for result in eligible
    )
    null_passes = sum(
        result.null_invariance_pass is True
        for result in eligible
    )

    real_failures = sum(
        result.real_invariance_pass is False
        for result in eligible
    )
    null_failures = sum(
        result.null_invariance_pass is False
        for result in eligible
    )

    max_real_delta = max(
        (
            result.real_max_abs_delta
            for result in eligible
            if result.real_max_abs_delta is not None
        ),
        default=float("nan"),
    )

    max_null_delta = max(
        (
            result.null_max_abs_delta
            for result in eligible
            if result.null_max_abs_delta is not None
        ),
        default=float("nan"),
    )

    max_real_h_error = max(
        result.max_real_h_reconstruction_error
        for result in results
    )

    max_null_h_error = max(
        result.max_null_h_reconstruction_error
        for result in results
    )

    seed_count = len({result.h_seed for result in results})

    lines = [
        "=== Soft Spaces Phase 3.1 v30.0 PROJECTOR-SCORE INVARIANCE TEST ===",
        f"8Q five-term random-Pauli Hamiltonians: "
        f"{seed_count} executed frozen seed(s)",
        f"Frozen candidates: {list(FROZEN)}; "
        f"rotations per degenerate basis: {ROTATIONS}",
        "REAL and NULL rotated separately; H, spectrum, V, coordinates, "
        "controls and score definition otherwise frozen.",
        "",
        "Replacement under test:",
        "  selected neighbouring pair -> complete degeneracy-closed P subspace",
        "  score uses exact-eigenspace blocks A_G = P V P_G V P",
        "  normalised Frobenius cancellation score is basis invariant by construction",
        "",
        "seed | k | groups | group size | baseline | "
        "REAL range | max|d| | gate | "
        "NULL range | max|d| | gate",
    ]

    for result in results:
        lines.append(
            f"{result.h_seed} | "
            f"{result.center:2d} | "
            f"{result.group_count:6d} | "
            f"{result.min_group_size:4d}:"
            f"{result.max_group_size:<4d} | "
            f"{fmt(result.baseline):>13s} | "
            f"[{fmt(result.real_rotation_min)},"
            f"{fmt(result.real_rotation_max)}] | "
            f"{fmt_sci(result.real_max_abs_delta):>10s} | "
            f"{fmt_pass(result.real_invariance_pass):4s} | "
            f"[{fmt(result.null_rotation_min)},"
            f"{fmt(result.null_rotation_max)}] | "
            f"{fmt_sci(result.null_max_abs_delta):>10s} | "
            f"{fmt_pass(result.null_invariance_pass):4s}"
        )

    lines.extend(
        [
            "",
            f"Eligible seed-candidate instances: "
            f"{len(eligible)}/{len(results)}",
            f"REAL invariance gate: "
            f"{real_passes} PASS, {real_failures} FAIL",
            f"NULL invariance gate: "
            f"{null_passes} PASS, {null_failures} FAIL",
            f"Maximum REAL prominence |delta|: "
            f"{max_real_delta:.6e}",
            f"Maximum NULL prominence |delta|: "
            f"{max_null_delta:.6e}",
            f"Maximum REAL ||U_rot diag(E) U_rot^dagger-H_REAL||_2: "
            f"{max_real_h_error:.6e}",
            f"Maximum NULL ||U_rot diag(E) U_rot^dagger-H_NULL||_2: "
            f"{max_null_h_error:.6e}",
            "",
            "Decision rule:",
            "  PASS requires every eligible REAL-only and NULL-only degenerate-basis",
            "  rotation to preserve prominence within",
            f"  atol={INVARIANCE_ATOL:.1e} + "
            f"rtol={INVARIANCE_RTOL:.1e}*|baseline|.",
            "",
            "Interpretation:",
            "  If all eligible cases PASS, the specific basis-dependence exposed by",
            "  v25.42 has been removed for this finite regression suite.",
            "  This does NOT yet prove that the projector score is the final physical",
            "  Soft Spaces observable; that is a separate Phase 3 model question.",
        ]
    )

    if real_failures == 0 and null_failures == 0 and eligible:
        lines.extend(
            [
                "",
                "CONCLUSION: PROJECTOR-SCORE BASIS-INVARIANCE TEST PASSED.",
                "The v25.42 degenerate-eigenbasis counterexample is not reproduced",
                "by the v30.0 degeneracy-closed projector score in this frozen suite.",
            ]
        )
    elif eligible:
        lines.extend(
            [
                "",
                "CONCLUSION: PROJECTOR-SCORE INVARIANCE TEST FAILED.",
                "At least one legal degenerate-basis rotation changes the reported",
                "prominence beyond the numerical regression tolerance.  Do not move",
                "to the larger Phase 3 Hamiltonian before resolving this.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "CONCLUSION: NO ELIGIBLE CASES; TEST INCONCLUSIVE.",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "v31_0_projector_score_invariance_output.txt"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run only the first frozen Hamiltonian seed. "
            "Default runs the full 12-seed frozen suite."
        ),
    )
    args = parser.parse_args()

    seeds = (
        HAMILTONIAN_SEEDS[:1]
        if args.smoke
        else HAMILTONIAN_SEEDS
    )

    results = [
        result
        for seed in seeds
        for result in run_hamiltonian(seed)
    ]

    report = render(results)
    print(report, end="")
    args.output.write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
