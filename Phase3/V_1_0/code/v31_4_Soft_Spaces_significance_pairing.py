#!/usr/bin/env python3
"""Soft Spaces Phase 3.1 v30.4 — frozen hotspot significance and ±E pairing test.

Purpose
-------
v30.2 showed robust local projector-score hotspots across 12 frozen 11-term
Hamiltonian seeds.  v30.4 asks whether those observations exceed what can be
explained by NULL/shuffled controls.

Frozen model
------------
* 8Q.
* 11-term random-Pauli Hamiltonian.
* Hamiltonian seeds: 25_042_000 ... 25_042_011.
* Exact eigenspaces scored as complete projectors.
* Perturbation families: dephasing Z/ZZ and transverse X/XX.
* Five perturbation terms per family.
* energy_reg = 1e-3.
* exact-degeneracy tolerance = 1e-10.
* local prominence radius = 5 eigenspaces.
* NULL-basis and perturbation RNG namespaces are frozen exactly to v30.2.

Tests
-----
For each seed:

A. Maximum-hotspot significance
   Compare observed REAL-vs-NULL maximum local prominence with a shuffle-null
   distribution obtained by randomly permuting the REAL robust values across
   eigenspaces before recomputing local prominence.

B. ±E pairing enrichment
   Identify energy-opposite eigenspace pairs and test whether strong positive
   prominences co-occur in opposite-energy partners more often than under
   shuffled prominence assignments.

C. ±E prominence correlation
   Compute Pearson correlation of prominence values across opposite-energy
   eigenspace pairs and compare with shuffled assignments.

This is a falsification/calibration test.  It does not alter the Hamiltonian.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


VERSION = "v30.4"
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

SHUFFLES = 2000
STRONG_QUANTILE = 0.75


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


def opposite_energy_pairs(
    energies: np.ndarray,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()

    for i, energy in enumerate(energies):
        if i in used:
            continue

        target = -energy
        candidates = [
            j for j in range(len(energies))
            if j != i and j not in used
        ]

        if not candidates:
            continue

        j = min(
            candidates,
            key=lambda idx: abs(energies[idx] - target),
        )

        if abs(float(energies[j] + energy)) <= PAIR_ENERGY_TOL:
            a, b = sorted((i, j))
            pairs.append((a, b))
            used.add(a)
            used.add(b)

    return sorted(set(pairs))


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

    return float(np.corrcoef(x, y)[0, 1])


def paired_strong_fraction(
    prominence: np.ndarray,
    pairs: list[tuple[int, int]],
) -> tuple[float, float]:
    if not pairs:
        return float("nan"), float("nan")

    threshold = float(
        np.quantile(prominence, STRONG_QUANTILE)
    )

    both = sum(
        prominence[i] >= threshold
        and prominence[j] >= threshold
        for i, j in pairs
    )

    return both / len(pairs), threshold


def empirical_p_upper(
    observed: float,
    null_values: np.ndarray,
) -> float:
    finite = null_values[np.isfinite(null_values)]

    if not np.isfinite(observed) or finite.size == 0:
        return float("nan")

    exceed = int(np.sum(finite >= observed))
    return (exceed + 1.0) / (finite.size + 1.0)


@dataclass(frozen=True)
class SeedResult:
    seed: int
    groups: int
    group_size_min: int
    group_size_max: int
    top_prominence: float
    top_p: float
    pair_count: int
    pair_corr: float
    pair_corr_p: float
    paired_strong_fraction: float
    paired_strong_p: float
    strong_threshold: float


def run_seed(
    seed: int,
    shuffles: int,
) -> SeedResult:
    hamiltonian = dense_pauli_sum(
        N_QUBITS,
        random_hamiltonian_terms(seed),
    )

    eigenvalues, real_vectors = np.linalg.eigh(
        hamiltonian
    )

    groups = degenerate_groups(eigenvalues)

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

    robust = robust_values(
        real_by_family,
        null_by_family,
        len(groups),
    )

    prominence_dict = prominence_values(
        robust,
        len(groups),
    )

    prominence = np.asarray(
        [
            prominence_dict[index]
            for index in range(len(groups))
        ],
        dtype=float,
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

    pairs = opposite_energy_pairs(energies)
    observed_top = float(np.max(prominence))
    observed_corr = pearson_from_pairs(
        prominence,
        pairs,
    )
    observed_pair_fraction, strong_threshold = (
        paired_strong_fraction(
            prominence,
            pairs,
        )
    )

    rng = np.random.default_rng(
        stable_hash_int(
            f"{VERSION}|SHUFFLES|{seed}|{shuffles}"
        )
    )

    null_top = np.empty(shuffles, dtype=float)
    null_corr = np.empty(shuffles, dtype=float)
    null_pair_fraction = np.empty(shuffles, dtype=float)

    robust_array = np.asarray(
        [
            robust[index]
            for index in range(len(groups))
        ],
        dtype=float,
    )

    for iteration in range(shuffles):
        shuffled_robust = rng.permutation(
            robust_array
        )

        shuffled_robust_dict = {
            index: float(shuffled_robust[index])
            for index in range(len(groups))
        }

        shuffled_prominence_dict = prominence_values(
            shuffled_robust_dict,
            len(groups),
        )

        shuffled_prominence = np.asarray(
            [
                shuffled_prominence_dict[index]
                for index in range(len(groups))
            ],
            dtype=float,
        )

        null_top[iteration] = float(
            np.max(shuffled_prominence)
        )

        null_corr[iteration] = pearson_from_pairs(
            shuffled_prominence,
            pairs,
        )

        null_pair_fraction[iteration], _ = (
            paired_strong_fraction(
                shuffled_prominence,
                pairs,
            )
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
        top_prominence=observed_top,
        top_p=empirical_p_upper(
            observed_top,
            null_top,
        ),
        pair_count=len(pairs),
        pair_corr=observed_corr,
        pair_corr_p=empirical_p_upper(
            observed_corr,
            null_corr,
        ),
        paired_strong_fraction=observed_pair_fraction,
        paired_strong_p=empirical_p_upper(
            observed_pair_fraction,
            null_pair_fraction,
        ),
        strong_threshold=strong_threshold,
    )


def median_or_nan(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    return (
        float(np.median(array))
        if array.size
        else float("nan")
    )


def fraction_below(
    values: list[float],
    threshold: float,
) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if array.size == 0:
        return float("nan")

    return float(
        np.mean(array < threshold)
    )


def render(
    results: list[SeedResult],
    shuffles: int,
) -> str:
    lines = [
        "=== Soft Spaces Phase 3.1 v30.4 FROZEN HOTSPOT SIGNIFICANCE / ±E PAIRING TEST ===",
        f"8Q; 11-term Hamiltonian; seeds = {len(results)}",
        f"Frozen physical model seed namespace = {FROZEN_MODEL_VERSION}",
        f"Shuffle replicates per seed = {shuffles}",
        f"Strong-hotspot threshold = seedwise top {int((1.0 - STRONG_QUANTILE) * 100)}%",
        "",
        "seed | groups | group min:max | top prominence | p_top | ±E pairs | corr | p_corr | paired-strong | p_pair",
    ]

    for result in results:
        lines.append(
            f"{result.seed} | "
            f"{result.groups:6d} | "
            f"{result.group_size_min:4d}:"
            f"{result.group_size_max:<4d} | "
            f"{result.top_prominence:+.9f} | "
            f"{result.top_p:.6f} | "
            f"{result.pair_count:8d} | "
            f"{result.pair_corr:+.6f} | "
            f"{result.pair_corr_p:.6f} | "
            f"{result.paired_strong_fraction:.6f} | "
            f"{result.paired_strong_p:.6f}"
        )

    top_ps = [result.top_p for result in results]
    corr_ps = [result.pair_corr_p for result in results]
    pair_ps = [result.paired_strong_p for result in results]
    corrs = [result.pair_corr for result in results]
    pair_fracs = [
        result.paired_strong_fraction
        for result in results
    ]

    lines.extend(
        [
            "",
            "=== ACROSS-SEED SUMMARY ===",
            f"Median top-hotspot p-value: {median_or_nan(top_ps):.6f}",
            f"Seeds with p_top < 0.05: {sum(p < 0.05 for p in top_ps if np.isfinite(p))}/{len(results)}",
            f"Median ±E prominence correlation: {median_or_nan(corrs):+.6f}",
            f"Median ±E correlation p-value: {median_or_nan(corr_ps):.6f}",
            f"Seeds with p_corr < 0.05: {sum(p < 0.05 for p in corr_ps if np.isfinite(p))}/{len(results)}",
            f"Median paired-strong fraction: {median_or_nan(pair_fracs):.6f}",
            f"Median paired-strong p-value: {median_or_nan(pair_ps):.6f}",
            f"Seeds with p_pair < 0.05: {sum(p < 0.05 for p in pair_ps if np.isfinite(p))}/{len(results)}",
            "",
            "Interpretation:",
            "  p_top  : empirical upper-tail probability for observing a maximum",
            "           prominence at least this large after shuffling robust values.",
            "  corr   : Pearson correlation of prominence across exact ±E partners.",
            "  p_corr : empirical upper-tail shuffle p-value for that correlation.",
            "  paired-strong:",
            "           fraction of exact ±E pairs where BOTH members lie in the",
            "           seedwise top quartile of prominence.",
            "  p_pair : empirical upper-tail shuffle p-value for paired-strong fraction.",
            "",
            "Decision principle:",
            "  Strong evidence requires repeated low p-values across independent frozen",
            "  Hamiltonian seeds.  A single low p-value is not sufficient.",
            "",
            "Important:",
            "  These are seedwise permutation p-values.  v30.3 does not yet perform a",
            "  formal combined-p test or multiple-testing correction; those belong in",
            "  the next confirmation/statistical-summary step if the signal survives.",
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
            "v31_4_hotspot_significance_pairing_output.txt"
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run first frozen Hamiltonian seed only.",
    )

    parser.add_argument(
        "--shuffles",
        type=int,
        default=SHUFFLES,
        help=(
            "Number of shuffle-null replicates per seed "
            f"(default {SHUFFLES})."
        ),
    )

    args = parser.parse_args()

    if args.shuffles < 100:
        raise ValueError("--shuffles must be at least 100.")

    seeds = (
        HAMILTONIAN_SEEDS[:1]
        if args.smoke
        else HAMILTONIAN_SEEDS
    )

    results = [
        run_seed(
            seed=seed,
            shuffles=args.shuffles,
        )
        for seed in seeds
    ]

    report = render(
        results,
        args.shuffles,
    )

    print(report, end="")
    args.output.write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
