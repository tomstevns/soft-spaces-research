#!/usr/bin/env python3
"""Soft Spaces Phase 3.1 v30.1 — Hamiltonian degeneracy / hotspot scan.

Purpose
-------
v30.0 removed the degenerate-eigenbasis dependence by replacing the arbitrary
2D pair score with a degeneracy-closed projector/subspace score.

The first v30.0 smoke run then exposed a second issue:
the old 8Q five-term random-Pauli Hamiltonian had only four exact energy groups,
each of size 64, and the local hotspot prominence collapsed to ~0.

v30.1 therefore changes ONE structural ingredient only:
the number of random Pauli terms in H.

Frozen from v30.0 / v25.42
--------------------------
* Dimension: 8Q.
* Frozen source coordinates: 31 and 95; two branches separated by 128.
* Frozen controls: source coordinate +/-10, step 2.
* Perturbation families: dephasing Z/ZZ and transverse X/XX.
* eps_neighbor = 0.05.
* energy_reg = 1e-3.
* exact-degeneracy tolerance = 1e-10.
* same projector/subspace score as v30.0.
* same Hamiltonian seed family: 25_042_000 ... 25_042_011.

New scan
--------
Hamiltonian term counts:
    5, 7, 9, 11

Primary diagnostics
-------------------
For every seed and term count:
* number of exact energy groups,
* minimum and maximum exact-degeneracy group size,
* fraction of non-degenerate eigenvalues,
* projector-score hotspot prominence at k=31 and k=95.

Interpretation
--------------
This is NOT yet a declaration of the final Phase 3 Hamiltonian.

The purpose is to identify the smallest increase in Hamiltonian complexity that:
1. breaks the artificial massive degeneracy of the 5-term model, and
2. preserves or restores a non-trivial local projector-score prominence.

No term count is selected in advance.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np


VERSION = "v30.1"

N_QUBITS = 8
SOURCE_QUBITS = 7
DIM = 1 << N_QUBITS
SOURCE_DIM = 1 << SOURCE_QUBITS

TERM_COUNTS = (5, 7, 9, 11)
PERTURB_TERMS = 5

FROZEN = (31, 95)
LOCAL_RADIUS = 10
LOCAL_STEP = 2

EPS_NEIGHBOR = 0.05
ENERGY_REG = 1.0e-3
DEGENERACY_TOL = 1.0e-10

HAMILTONIAN_SEEDS = tuple(range(25_042_000, 25_042_012))
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


def random_hamiltonian_terms(
    seed: int,
    n_terms: int,
) -> list[tuple[str, float]]:
    """Generate H terms.

    For a fixed seed, the first five terms are identical to the 5-term model;
    7, 9 and 11 terms extend that same RNG stream.  Thus the scan adds terms
    rather than replacing the original five-term Hamiltonian.
    """
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 4**N_QUBITS - 1, size=n_terms)
    coefficients = rng.uniform(-1.0, 1.0, size=n_terms)

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

    selected = rng.choice(
        len(labels),
        size=PERTURB_TERMS,
        replace=False,
    )
    coefficients = rng.uniform(
        -1.0,
        1.0,
        size=PERTURB_TERMS,
    )

    return [
        (labels[int(index)], float(coefficient))
        for index, coefficient in zip(selected, coefficients)
    ]


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
    lookup = np.empty(dimension, dtype=int)

    for group_number, group in enumerate(groups):
        lookup[group] = group_number

    return lookup


def degeneracy_closed_subspace(
    left_index: int,
    groups: list[np.ndarray],
    lookup: np.ndarray,
) -> np.ndarray:
    right_index = left_index + 1

    touched = sorted(
        {
            int(lookup[left_index]),
            int(lookup[right_index]),
        }
    )

    return np.concatenate(
        [groups[group_number] for group_number in touched]
    )


def projector_cancel_score(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    perturbation: list[tuple[str, float]],
    groups: list[np.ndarray],
    lookup: np.ndarray,
    left_index: int,
) -> float | None:
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

    p_group_numbers = {
        int(lookup[int(index)])
        for index in p_indices
    }

    p_vectors = eigenvectors[:, p_indices]
    acted_p = apply_pauli_sum(
        p_vectors,
        perturbation,
    )

    b_to_p = eigenvectors.conj().T @ acted_p

    reference = 0.5 * float(
        eigenvalues[left_index]
        + eigenvalues[right_index]
    )

    effective = np.zeros(
        (p_indices.size, p_indices.size),
        dtype=np.complex128,
    )
    denominator = 0.0

    for group_number, group in enumerate(groups):
        if group_number in p_group_numbers:
            continue

        e_group = float(np.mean(eigenvalues[group]))
        delta = reference - e_group

        inverse = delta / (
            delta * delta
            + ENERGY_REG * ENERGY_REG
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

    numerator = float(
        np.linalg.norm(effective, ord="fro")
    )

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
            output[coordinate] = finite_median(
                branch_values
            )

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

    return float(
        target - np.median(background_values)
    )


def haar_unitary(
    dimension: int,
    seed: int,
) -> np.ndarray:
    """Deterministic Haar unitary used only for the NULL control."""
    rng = np.random.default_rng(seed)

    raw = (
        rng.normal(size=(dimension, dimension))
        + 1j * rng.normal(size=(dimension, dimension))
    ) / np.sqrt(2.0)

    q, r = np.linalg.qr(raw)
    diagonal = np.diag(r)

    phases = np.ones_like(diagonal)
    nonzero = np.abs(diagonal) > 0.0
    phases[nonzero] = (
        diagonal[nonzero]
        / np.abs(diagonal[nonzero])
    )

    return q * phases.conj()[None, :]


def fmt(value: float | None) -> str:
    return (
        "INELIGIBLE"
        if value is None
        else f"{value:+.9f}"
    )


def run_case(
    h_seed: int,
    n_terms: int,
) -> dict:
    hamiltonian = dense_pauli_sum(
        N_QUBITS,
        random_hamiltonian_terms(
            h_seed,
            n_terms,
        ),
    )

    eigenvalues, real_vectors = np.linalg.eigh(
        hamiltonian
    )

    groups = degenerate_groups(eigenvalues)
    lookup = group_lookup(groups, DIM)

    sizes = np.asarray(
        [group.size for group in groups],
        dtype=int,
    )

    nondegenerate_fraction = float(
        np.sum(sizes == 1) / DIM
    )

    null_vectors = haar_unitary(
        DIM,
        stable_hash_int(
            f"{VERSION}|NULL|{h_seed}|terms{n_terms}"
        ),
    )

    perturbations = {
        family: family_terms(
            stable_hash_int(
                f"{VERSION}|{family}|PERT|{h_seed}"
            ),
            family,
        )
        for family in FAMILIES
    }

    real_by_family = {
        family: model_values(
            eigenvalues,
            real_vectors,
            perturbations[family],
            groups,
            lookup,
        )
        for family in FAMILIES
    }

    null_by_family = {
        family: model_values(
            eigenvalues,
            null_vectors,
            perturbations[family],
            groups,
            lookup,
        )
        for family in FAMILIES
    }

    prominences = {
        center: prominence(
            real_by_family,
            null_by_family,
            center,
        )
        for center in FROZEN
    }

    return {
        "seed": h_seed,
        "terms": n_terms,
        "groups": len(groups),
        "min_group": int(np.min(sizes)),
        "max_group": int(np.max(sizes)),
        "nondeg_frac": nondegenerate_fraction,
        "p31": prominences[31],
        "p95": prominences[95],
    }


def median_or_nan(values: list[float]) -> float:
    finite = [
        value
        for value in values
        if np.isfinite(value)
    ]
    return (
        float(np.median(finite))
        if finite
        else float("nan")
    )


def render(results: list[dict]) -> str:
    lines = [
        "=== Soft Spaces Phase 3.1 v30.1 HAMILTONIAN DEGENERACY / HOTSPOT SCAN ===",
        f"8Q, term counts = {list(TERM_COUNTS)}",
        f"Frozen candidates = {list(FROZEN)}",
        "",
        "seed | terms | groups | group min:max | nondeg.frac | prominence k=31 | prominence k=95",
    ]

    for result in results:
        lines.append(
            f"{result['seed']} | "
            f"{result['terms']:5d} | "
            f"{result['groups']:6d} | "
            f"{result['min_group']:4d}:"
            f"{result['max_group']:<4d} | "
            f"{result['nondeg_frac']:11.6f} | "
            f"{fmt(result['p31']):>15s} | "
            f"{fmt(result['p95']):>15s}"
        )

    lines.extend(
        [
            "",
            "=== SUMMARY BY TERM COUNT ===",
            "terms | median groups | median max-group | median nondeg.frac | eligible | median |P31| | median |P95|",
        ]
    )

    for n_terms in TERM_COUNTS:
        subset = [
            row
            for row in results
            if row["terms"] == n_terms
        ]

        p31 = [
            abs(row["p31"])
            for row in subset
            if row["p31"] is not None
        ]
        p95 = [
            abs(row["p95"])
            for row in subset
            if row["p95"] is not None
        ]

        eligible = sum(
            row["p31"] is not None
            and row["p95"] is not None
            for row in subset
        )

        lines.append(
            f"{n_terms:5d} | "
            f"{median_or_nan([float(row['groups']) for row in subset]):13.1f} | "
            f"{median_or_nan([float(row['max_group']) for row in subset]):16.1f} | "
            f"{median_or_nan([row['nondeg_frac'] for row in subset]):18.6f} | "
            f"{eligible:8d}/{len(subset):<2d} | "
            f"{median_or_nan(p31):12.6e} | "
            f"{median_or_nan(p95):12.6e}"
        )

    lines.extend(
        [
            "",
            "Reading guide:",
            "  groups          : number of exact eigenspaces.",
            "  group min:max   : smallest/largest exact-degeneracy block.",
            "  nondeg.frac     : fraction of all 256 eigenvalues that are singleton groups.",
            "  prominence      : projector-score target minus local-control median.",
            "",
            "What we are looking for:",
            "  1. groups rising strongly above the 5-term baseline;",
            "  2. max-group size falling strongly below 64;",
            "  3. nondeg.frac increasing;",
            "  4. non-zero, reproducible prominence returning at k=31 and/or k=95.",
            "",
            "Important:",
            "  v30.1 is a diagnostic scan.  Do not choose the Phase 3 Hamiltonian",
            "  from one seed alone.  The full 12-seed run is the intended comparison.",
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
            "v30_1_hamiltonian_degeneracy_hotspot_scan_output.txt"
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run the first Hamiltonian seed only, "
            "but still scan all four term counts."
        ),
    )

    args = parser.parse_args()

    seeds = (
        HAMILTONIAN_SEEDS[:1]
        if args.smoke
        else HAMILTONIAN_SEEDS
    )

    results = [
        run_case(seed, n_terms)
        for seed in seeds
        for n_terms in TERM_COUNTS
    ]

    report = render(results)

    print(report, end="")
    args.output.write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
