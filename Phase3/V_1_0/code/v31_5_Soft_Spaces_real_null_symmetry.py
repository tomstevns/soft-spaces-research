#!/usr/bin/env python3
"""Soft Spaces Phase 3.1 v30.5 — REAL/NULL ±E symmetry decomposition test.

Purpose
-------
v30.4 established that the frozen 11-term model shows extremely strong
±E organization in the REAL-minus-NULL prominence pattern across all 12
Hamiltonian seeds.

v30.5 asks the key confound question:

    Is that ±E organization specific to REAL,
    or is it already present in NULL because of the Hamiltonian/spectrum
    and score construction?

Frozen physical model
---------------------
* 8Q.
* 11-term random-Pauli Hamiltonian.
* Hamiltonian seeds: 25_042_000 ... 25_042_011.
* NULL-basis and perturbation RNG namespaces frozen exactly to v30.2.
* Complete exact-eigenspace projector score.
* Perturbation families: dephasing Z/ZZ and transverse X/XX.
* Five perturbation terms per family.
* energy_reg = 1e-3.
* exact-degeneracy tolerance = 1e-10.
* local prominence radius = 5 eigenspaces.
* exact ±E pairing tolerance = 1e-8.

Primary outputs per seed
------------------------
1. corr_REAL:
   Pearson correlation of REAL prominence across exact ±E eigenspace partners.

2. corr_NULL:
   Pearson correlation of NULL prominence across exact ±E eigenspace partners.

3. corr_DIFF:
   Pearson correlation of (REAL-NULL) prominence across exact ±E partners.

4. delta_corr:
       corr_REAL - corr_NULL

5. permutation p-value for delta_corr:
   The labels REAL/NULL are randomly swapped independently within each
   eigenspace before recomputing corr_REAL - corr_NULL.

Interpretation
--------------
If corr_REAL and corr_NULL are both similarly high, the ±E organization is
likely driven mainly by Hamiltonian/spectral symmetry.

If corr_REAL materially exceeds corr_NULL across many frozen seeds, with low
permutation p-values for delta_corr, that is evidence that the ±E organization
contains a REAL-specific component.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


VERSION = "v30.5"
FROZEN_MODEL_VERSION = "v30.2"

N_QUBITS = 8
DIM = 1 << N_QUBITS
N_TERMS = 11
PERTURB_TERMS = 5

ENERGY_REG = 1.0e-3
DEGENERACY_TOL = 1.0e-10
PAIR_ENERGY_TOL = 1.0e-8
LOCAL_GROUP_RADIUS = 5

HAMILTONIAN_SEEDS = tuple(range(25_042_000, 25_042_012))
FAMILIES = ("dephasing", "transverse")
PAULIS = ("I", "X", "Y", "Z")

PERMUTATIONS = 2000


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
    indices = rng.integers(
        0,
        4**N_QUBITS - 1,
        size=N_TERMS,
    )
    coefficients = rng.uniform(
        -1.0,
        1.0,
        size=N_TERMS,
    )

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
            groups.append(
                np.arange(start, index, dtype=int)
            )
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


def robust_family_min(
    by_family: dict[str, dict[int, float]],
    group_count: int,
) -> dict[int, float]:
    output: dict[int, float] = {}

    for group_number in range(group_count):
        values: list[float] = []

        for family in FAMILIES:
            if group_number not in by_family[family]:
                values = []
                break

            values.append(
                by_family[family][group_number]
            )

        if values:
            output[group_number] = min(values)

    return output


def robust_difference(
    real_by_family: dict[str, dict[int, float]],
    null_by_family: dict[str, dict[int, float]],
    group_count: int,
) -> dict[int, float]:
    output: dict[int, float] = {}

    for group_number in range(group_count):
        values: list[float] = []

        for family in FAMILIES:
            if (
                group_number not in real_by_family[family]
                or group_number not in null_by_family[family]
            ):
                values = []
                break

            values.append(
                real_by_family[family][group_number]
                - null_by_family[family][group_number]
            )

        if values:
            output[group_number] = min(values)

    return output


def local_neighbours(
    group_number: int,
    group_count: int,
) -> list[int]:
    start = max(
        0,
        group_number - LOCAL_GROUP_RADIUS,
    )
    stop = min(
        group_count,
        group_number + LOCAL_GROUP_RADIUS + 1,
    )

    return [
        index
        for index in range(start, stop)
        if index != group_number
    ]


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


def opposite_energy_pairs(
    energies: np.ndarray,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()

    for i, energy in enumerate(energies):
        if i in used:
            continue

        candidates = [
            j
            for j in range(len(energies))
            if j != i and j not in used
        ]

        if not candidates:
            continue

        j = min(
            candidates,
            key=lambda idx: abs(
                energies[idx] + energy
            ),
        )

        if (
            abs(float(energies[j] + energy))
            <= PAIR_ENERGY_TOL
        ):
            a, b = sorted((i, j))
            pairs.append((a, b))
            used.add(a)
            used.add(b)

    return sorted(set(pairs))


def vector_from_dict(
    values: dict[int, float],
    group_count: int,
) -> np.ndarray:
    return np.asarray(
        [values[index] for index in range(group_count)],
        dtype=float,
    )


def pearson_from_pairs(
    values: np.ndarray,
    pairs: list[tuple[int, int]],
) -> float:
    if len(pairs) < 2:
        return float("nan")

    x = np.asarray(
        [values[i] for i, _ in pairs],
        dtype=float,
    )
    y = np.asarray(
        [values[j] for _, j in pairs],
        dtype=float,
    )

    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def empirical_p_upper(
    observed: float,
    null_values: np.ndarray,
) -> float:
    finite = null_values[np.isfinite(null_values)]

    if not np.isfinite(observed) or finite.size == 0:
        return float("nan")

    exceed = int(np.sum(finite >= observed))

    return (
        exceed + 1.0
    ) / (
        finite.size + 1.0
    )


@dataclass(frozen=True)
class SeedResult:
    seed: int
    groups: int
    group_size_min: int
    group_size_max: int
    pair_count: int
    corr_real: float
    corr_null: float
    corr_diff: float
    delta_corr: float
    delta_corr_p: float


def run_seed(
    seed: int,
    permutations: int,
) -> SeedResult:
    hamiltonian = dense_pauli_sum(
        N_QUBITS,
        random_hamiltonian_terms(seed),
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
            f"{FROZEN_MODEL_VERSION}|NULL|{seed}|terms{N_TERMS}"
        ),
    )

    perturbations = {
        family: family_terms(
            stable_hash_int(
                f"{FROZEN_MODEL_VERSION}|{family}|PERT|{seed}"
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

    real_robust = robust_family_min(
        real_by_family,
        len(groups),
    )

    null_robust = robust_family_min(
        null_by_family,
        len(groups),
    )

    diff_robust = robust_difference(
        real_by_family,
        null_by_family,
        len(groups),
    )

    real_prominence = prominence_values(
        real_robust,
        len(groups),
    )

    null_prominence = prominence_values(
        null_robust,
        len(groups),
    )

    diff_prominence = prominence_values(
        diff_robust,
        len(groups),
    )

    real_vec = vector_from_dict(
        real_prominence,
        len(groups),
    )

    null_vec = vector_from_dict(
        null_prominence,
        len(groups),
    )

    diff_vec = vector_from_dict(
        diff_prominence,
        len(groups),
    )

    energies = np.asarray(
        [
            eigenspace_energy(
                eigenvalues,
                group,
            )
            for group in groups
        ],
        dtype=float,
    )

    pairs = opposite_energy_pairs(
        energies
    )

    corr_real = pearson_from_pairs(
        real_vec,
        pairs,
    )

    corr_null = pearson_from_pairs(
        null_vec,
        pairs,
    )

    corr_diff = pearson_from_pairs(
        diff_vec,
        pairs,
    )

    delta_corr = corr_real - corr_null

    rng = np.random.default_rng(
        stable_hash_int(
            f"{VERSION}|DELTA_PERM|{seed}|{permutations}"
        )
    )

    permuted_delta = np.empty(
        permutations,
        dtype=float,
    )

    for iteration in range(permutations):
        swap_mask = rng.integers(
            0,
            2,
            size=len(groups),
            dtype=np.int8,
        ).astype(bool)

        perm_real = real_vec.copy()
        perm_null = null_vec.copy()

        temp = perm_real[swap_mask].copy()
        perm_real[swap_mask] = perm_null[swap_mask]
        perm_null[swap_mask] = temp

        perm_corr_real = pearson_from_pairs(
            perm_real,
            pairs,
        )

        perm_corr_null = pearson_from_pairs(
            perm_null,
            pairs,
        )

        permuted_delta[iteration] = (
            perm_corr_real
            - perm_corr_null
        )

    sizes = [
        int(group.size)
        for group in groups
    ]

    return SeedResult(
        seed=seed,
        groups=len(groups),
        group_size_min=min(sizes),
        group_size_max=max(sizes),
        pair_count=len(pairs),
        corr_real=corr_real,
        corr_null=corr_null,
        corr_diff=corr_diff,
        delta_corr=delta_corr,
        delta_corr_p=empirical_p_upper(
            delta_corr,
            permuted_delta,
        ),
    )


def median_or_nan(values: list[float]) -> float:
    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    return (
        float(np.median(array))
        if array.size
        else float("nan")
    )


def render(
    results: list[SeedResult],
    permutations: int,
) -> str:
    lines = [
        "=== Soft Spaces Phase 3.1 v30.5 REAL/NULL ±E SYMMETRY DECOMPOSITION ===",
        f"8Q; 11-term Hamiltonian; seeds = {len(results)}",
        f"Frozen physical model seed namespace = {FROZEN_MODEL_VERSION}",
        f"Permutation replicates per seed = {permutations}",
        "",
        "seed | groups | size min:max | ±E pairs | corr_REAL | corr_NULL | corr_DIFF | delta_corr | p_delta",
    ]

    for result in results:
        lines.append(
            f"{result.seed} | "
            f"{result.groups:6d} | "
            f"{result.group_size_min:4d}:"
            f"{result.group_size_max:<4d} | "
            f"{result.pair_count:8d} | "
            f"{result.corr_real:+.6f} | "
            f"{result.corr_null:+.6f} | "
            f"{result.corr_diff:+.6f} | "
            f"{result.delta_corr:+.6f} | "
            f"{result.delta_corr_p:.6f}"
        )

    corr_real = [
        result.corr_real
        for result in results
    ]
    corr_null = [
        result.corr_null
        for result in results
    ]
    corr_diff = [
        result.corr_diff
        for result in results
    ]
    delta_corr = [
        result.delta_corr
        for result in results
    ]
    p_delta = [
        result.delta_corr_p
        for result in results
    ]

    lines.extend(
        [
            "",
            "=== ACROSS-SEED SUMMARY ===",
            f"Median corr_REAL: {median_or_nan(corr_real):+.6f}",
            f"Median corr_NULL: {median_or_nan(corr_null):+.6f}",
            f"Median corr_DIFF: {median_or_nan(corr_diff):+.6f}",
            f"Median delta_corr = REAL-NULL: {median_or_nan(delta_corr):+.6f}",
            f"Median p_delta: {median_or_nan(p_delta):.6f}",
            f"Seeds with delta_corr > 0: {sum(x > 0 for x in delta_corr if np.isfinite(x))}/{len(results)}",
            f"Seeds with p_delta < 0.05: {sum(p < 0.05 for p in p_delta if np.isfinite(p))}/{len(results)}",
            "",
            "Interpretation:",
            "  corr_REAL : ±E correlation of REAL prominence alone.",
            "  corr_NULL : ±E correlation of NULL prominence alone.",
            "  corr_DIFF : ±E correlation of REAL-minus-NULL prominence.",
            "  delta_corr: corr_REAL - corr_NULL.",
            "  p_delta   : upper-tail permutation p-value for delta_corr under",
            "              independent REAL/NULL label swaps within eigenspaces.",
            "",
            "Decision logic:",
            "  If REAL and NULL correlations are similarly high, the ±E structure is",
            "  likely largely inherited from Hamiltonian/spectral symmetry.",
            "  If REAL consistently exceeds NULL with low p_delta across seeds, the",
            "  ±E organization contains a REAL-specific component.",
            "",
            "Important:",
            "  This test freezes the v30.2 physical model.  Only the statistical",
            "  decomposition/permutation layer is new in v30.5.",
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
            "v31_5_real_null_symmetry_decomposition_output.txt"
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run first frozen Hamiltonian seed only.",
    )

    parser.add_argument(
        "--permutations",
        type=int,
        default=PERMUTATIONS,
        help=(
            "Number of REAL/NULL label-swap permutations per seed "
            f"(default {PERMUTATIONS})."
        ),
    )

    args = parser.parse_args()

    if args.permutations < 100:
        raise ValueError(
            "--permutations must be at least 100."
        )

    seeds = (
        HAMILTONIAN_SEEDS[:1]
        if args.smoke
        else HAMILTONIAN_SEEDS
    )

    results = [
        run_seed(
            seed=seed,
            permutations=args.permutations,
        )
        for seed in seeds
    ]

    report = render(
        results,
        args.permutations,
    )

    print(report, end="")
    args.output.write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
