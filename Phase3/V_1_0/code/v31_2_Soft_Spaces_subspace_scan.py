#!/usr/bin/env python3
"""Soft Spaces Phase 3.1 v30.2 — complete eigenspace hotspot scan.

Purpose
-------
v30.0 removed the degenerate-eigenbasis dependence by replacing arbitrary
2D eigenvector-pair scores with a degeneracy-closed projector score.

v30.1 then showed that increasing the random-Pauli Hamiltonian from 5 to
7/9/11 terms reduces the massive degeneracy, but the old neighbouring-pair
eligibility rule becomes inappropriate.

v30.2 therefore removes the old pair gate entirely and scans COMPLETE exact
eigenspaces directly.

Core question
-------------
For each exact eigenspace P_j of H, does the projector response to the frozen
perturbation families produce an unusually strong REAL-vs-NULL local prominence
relative to neighbouring eigenspaces in energy order?

Frozen
------
* 8Q.
* Hamiltonian seeds: 25_042_000 ... 25_042_011.
* Hamiltonian term counts: 5, 7, 9, 11.
* Perturbation families: dephasing Z/ZZ and transverse X/XX.
* Five perturbation terms per family.
* energy_reg = 1e-3.
* exact-degeneracy tolerance = 1e-10.
* projector/subspace score inherited from v30.0.

Primary working model
---------------------
11 Pauli terms are the primary Phase 3 working model because v30.1 reduced the
degeneracy to 64 exact eigenspaces of size 4 for the smoke seed.  The 5/7/9-term
models remain controls.

Hotspot definition
------------------
For exact eigenspace j:

    robust(j) = min_family( C_REAL(j,family) - C_NULL(j,family) )

Local prominence:

    prominence(j) = robust(j) - median(robust(neighbouring eigenspaces))

Neighbouring eigenspaces are defined by energy-order index, not individual
eigenvector index.

This is a discovery scan.  No hotspot coordinate is preselected.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


VERSION = "v30.2"

N_QUBITS = 8
DIM = 1 << N_QUBITS

TERM_COUNTS = (5, 7, 9, 11)
PRIMARY_TERMS = 11

PERTURB_TERMS = 5
ENERGY_REG = 1.0e-3
DEGENERACY_TOL = 1.0e-10

HAMILTONIAN_SEEDS = tuple(range(25_042_000, 25_042_012))
FAMILIES = ("dephasing", "transverse")
PAULIS = ("I", "X", "Y", "Z")

LOCAL_GROUP_RADIUS = 5
TOP_K = 10


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


def eigenspace_energy(
    eigenvalues: np.ndarray,
    group: np.ndarray,
) -> float:
    return float(np.mean(eigenvalues[group]))


def eigenspace_cancel_score(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    perturbation: list[tuple[str, float]],
    groups: list[np.ndarray],
    target_group_number: int,
) -> float | None:
    """Basis-invariant cancellation score for one complete exact eigenspace."""
    p_indices = groups[target_group_number]
    p_vectors = eigenvectors[:, p_indices]

    acted_p = apply_pauli_sum(
        p_vectors,
        perturbation,
    )

    b_to_p = eigenvectors.conj().T @ acted_p
    reference = eigenspace_energy(
        eigenvalues,
        p_indices,
    )

    effective = np.zeros(
        (p_indices.size, p_indices.size),
        dtype=np.complex128,
    )
    denominator = 0.0

    for group_number, group in enumerate(groups):
        if group_number == target_group_number:
            continue

        e_group = eigenspace_energy(
            eigenvalues,
            group,
        )

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

    numerator = float(
        np.linalg.norm(effective, ord="fro")
    )

    if not np.isfinite(numerator):
        return None

    return numerator / denominator


def model_scores(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    perturbation: list[tuple[str, float]],
    groups: list[np.ndarray],
) -> dict[int, float]:
    output: dict[int, float] = {}

    for group_number in range(len(groups)):
        value = eigenspace_cancel_score(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            perturbation=perturbation,
            groups=groups,
            target_group_number=group_number,
        )

        if value is not None:
            output[group_number] = value

    return output


def local_neighbours(
    group_number: int,
    group_count: int,
) -> list[int]:
    start = max(0, group_number - LOCAL_GROUP_RADIUS)
    stop = min(
        group_count,
        group_number + LOCAL_GROUP_RADIUS + 1,
    )

    return [
        index
        for index in range(start, stop)
        if index != group_number
    ]


def robust_values(
    real_by_family: dict[str, dict[int, float]],
    null_by_family: dict[str, dict[int, float]],
    group_count: int,
) -> dict[int, float]:
    output: dict[int, float] = {}

    for group_number in range(group_count):
        family_values: list[float] = []

        for family in FAMILIES:
            if (
                group_number not in real_by_family[family]
                or group_number not in null_by_family[family]
            ):
                family_values = []
                break

            family_values.append(
                real_by_family[family][group_number]
                - null_by_family[family][group_number]
            )

        if family_values:
            output[group_number] = min(family_values)

    return output


def prominence_values(
    robust: dict[int, float],
    group_count: int,
) -> dict[int, float]:
    output: dict[int, float] = {}

    for group_number, target in robust.items():
        neighbours = [
            robust[index]
            for index in local_neighbours(
                group_number,
                group_count,
            )
            if index in robust
        ]

        if not neighbours:
            continue

        output[group_number] = float(
            target - np.median(neighbours)
        )

    return output


@dataclass(frozen=True)
class Hotspot:
    seed: int
    terms: int
    group_number: int
    energy: float
    group_size: int
    first_eigen_index: int
    last_eigen_index: int
    robust_value: float
    prominence: float


def run_case(
    h_seed: int,
    n_terms: int,
) -> tuple[dict, list[Hotspot]]:
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

    groups = degenerate_groups(
        eigenvalues
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
        family: model_scores(
            eigenvalues,
            real_vectors,
            perturbations[family],
            groups,
        )
        for family in FAMILIES
    }

    null_by_family = {
        family: model_scores(
            eigenvalues,
            null_vectors,
            perturbations[family],
            groups,
        )
        for family in FAMILIES
    }

    robust = robust_values(
        real_by_family,
        null_by_family,
        len(groups),
    )

    prominence = prominence_values(
        robust,
        len(groups),
    )

    hotspots: list[Hotspot] = []

    for group_number, value in prominence.items():
        group = groups[group_number]

        hotspots.append(
            Hotspot(
                seed=h_seed,
                terms=n_terms,
                group_number=group_number,
                energy=eigenspace_energy(
                    eigenvalues,
                    group,
                ),
                group_size=int(group.size),
                first_eigen_index=int(group[0]),
                last_eigen_index=int(group[-1]),
                robust_value=float(
                    robust[group_number]
                ),
                prominence=float(value),
            )
        )

    hotspots.sort(
        key=lambda item: item.prominence,
        reverse=True,
    )

    sizes = [
        int(group.size)
        for group in groups
    ]

    summary = {
        "seed": h_seed,
        "terms": n_terms,
        "group_count": len(groups),
        "min_group_size": min(sizes),
        "max_group_size": max(sizes),
        "top_prominence": (
            hotspots[0].prominence
            if hotspots
            else float("nan")
        ),
        "positive_count": sum(
            item.prominence > 0.0
            for item in hotspots
        ),
        "scored_count": len(hotspots),
    }

    return summary, hotspots


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


def render(
    summaries: list[dict],
    hotspots_by_case: dict[tuple[int, int], list[Hotspot]],
) -> str:
    lines = [
        "=== Soft Spaces Phase 3.1 v30.2 COMPLETE EIGENSPACE HOTSPOT SCAN ===",
        f"8Q; term counts = {list(TERM_COUNTS)}",
        f"Primary working model = {PRIMARY_TERMS} terms",
        f"Local energy-order neighbourhood radius = {LOCAL_GROUP_RADIUS} eigenspaces",
        "",
        "=== CASE SUMMARY ===",
        "seed | terms | groups | group min:max | scored | positive | top prominence",
    ]

    for row in summaries:
        lines.append(
            f"{row['seed']} | "
            f"{row['terms']:5d} | "
            f"{row['group_count']:6d} | "
            f"{row['min_group_size']:4d}:"
            f"{row['max_group_size']:<4d} | "
            f"{row['scored_count']:6d} | "
            f"{row['positive_count']:8d} | "
            f"{row['top_prominence']:+.9f}"
        )

    lines.extend(
        [
            "",
            "=== SUMMARY BY TERM COUNT ===",
            "terms | median groups | median max-group | median scored | median positive | median top prominence",
        ]
    )

    for n_terms in TERM_COUNTS:
        subset = [
            row
            for row in summaries
            if row["terms"] == n_terms
        ]

        lines.append(
            f"{n_terms:5d} | "
            f"{median_or_nan([float(row['group_count']) for row in subset]):13.1f} | "
            f"{median_or_nan([float(row['max_group_size']) for row in subset]):16.1f} | "
            f"{median_or_nan([float(row['scored_count']) for row in subset]):13.1f} | "
            f"{median_or_nan([float(row['positive_count']) for row in subset]):15.1f} | "
            f"{median_or_nan([float(row['top_prominence']) for row in subset]):+.9f}"
        )

    lines.extend(
        [
            "",
            f"=== TOP {TOP_K} HOTSPOTS FOR PRIMARY {PRIMARY_TERMS}-TERM MODEL ===",
            "seed | rank | group | eig-index range | size | energy | robust | prominence",
        ]
    )

    primary_cases = sorted(
        [
            key
            for key in hotspots_by_case
            if key[1] == PRIMARY_TERMS
        ]
    )

    for seed, n_terms in primary_cases:
        hotspots = hotspots_by_case[(seed, n_terms)]

        for rank, hotspot in enumerate(
            hotspots[:TOP_K],
            start=1,
        ):
            lines.append(
                f"{seed} | "
                f"{rank:4d} | "
                f"{hotspot.group_number:5d} | "
                f"{hotspot.first_eigen_index:3d}:"
                f"{hotspot.last_eigen_index:<3d} | "
                f"{hotspot.group_size:4d} | "
                f"{hotspot.energy:+.9f} | "
                f"{hotspot.robust_value:+.9f} | "
                f"{hotspot.prominence:+.9f}"
            )

    lines.extend(
        [
            "",
            "Interpretation:",
            "  group            = exact eigenspace number in ascending energy order.",
            "  eig-index range  = eigenvector indices belonging to that complete eigenspace.",
            "  robust           = minimum REAL-minus-NULL score across dephasing/transverse.",
            "  prominence       = robust minus local median of neighbouring eigenspaces.",
            "",
            "Decision use:",
            "  v30.2 is a discovery scan, not a final claim.",
            "  We are looking for repeated non-zero prominence patterns in the 11-term model",
            "  and whether comparable structure appears across multiple Hamiltonian seeds.",
            "  Any candidate for Phase 3 must later survive dedicated invariance, NULL,",
            "  perturbation-family and seed-replication tests.",
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
            "v31_2_complete_eigenspace_hotspot_scan_output.txt"
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run first Hamiltonian seed only, "
            "while scanning all four term counts."
        ),
    )

    parser.add_argument(
        "--primary-only",
        action="store_true",
        help=(
            "Run only the 11-term primary working model."
        ),
    )

    args = parser.parse_args()

    seeds = (
        HAMILTONIAN_SEEDS[:1]
        if args.smoke
        else HAMILTONIAN_SEEDS
    )

    term_counts = (
        (PRIMARY_TERMS,)
        if args.primary_only
        else TERM_COUNTS
    )

    summaries: list[dict] = []
    hotspots_by_case: dict[
        tuple[int, int],
        list[Hotspot],
    ] = {}

    for seed in seeds:
        for n_terms in term_counts:
            summary, hotspots = run_case(
                seed,
                n_terms,
            )

            summaries.append(summary)
            hotspots_by_case[
                (seed, n_terms)
            ] = hotspots

    report = render(
        summaries,
        hotspots_by_case,
    )

    print(report, end="")
    args.output.write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
