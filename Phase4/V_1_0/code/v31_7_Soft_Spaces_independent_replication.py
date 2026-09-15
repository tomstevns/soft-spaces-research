#!/usr/bin/env python3
"""Soft Spaces Phase 3.1 v30.7 — independent frozen replication.

Purpose
-------
v30.6 found a small but reproducible REAL-specific enhancement of ±E
correlation across the original 12 frozen Hamiltonian seeds.

v30.7 is an OUT-OF-SAMPLE confirmation run.

No tuning is allowed after seeing the new results.

Frozen physical model
---------------------
* 8Q.
* 11-term random-Pauli Hamiltonian.
* Complete exact-eigenspace projector score.
* NULL-basis and perturbation RNG namespace frozen to v30.2 logic.
* Perturbation families: dephasing Z/ZZ and transverse X/XX.
* Five perturbation terms per family.
* energy_reg = 1e-3.
* exact-degeneracy tolerance = 1e-10.
* local prominence radius = 5 eigenspaces.
* exact ±E pairing tolerance = 1e-8.

Independent confirmation seeds
------------------------------
Hamiltonian seeds:
    25_043_000 ... 25_043_011

Primary endpoint
----------------
    delta_corr = corr_REAL - corr_NULL

Predeclared confirmation criteria
---------------------------------
The replication is considered CONFIRMED only if ALL of the following hold:

1. median(delta_corr) > 0
2. at least 9/12 seeds have delta_corr > 0
3. one-sided exact sign-test p < 0.05
4. one-sided Wilcoxon signed-rank p < 0.05
5. 95% bootstrap CI for median(delta_corr) excludes 0 on the positive side

Secondary outputs
-----------------
* mean(delta_corr)
* bootstrap CI for mean(delta_corr)
* Cohen dz
* Fisher-z descriptive comparison

Important
---------
These acceptance criteria are frozen before execution.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np


VERSION = "v30.7"
FROZEN_MODEL_VERSION = "v30.2"

N_QUBITS = 8
DIM = 1 << N_QUBITS
N_TERMS = 11
PERTURB_TERMS = 5

ENERGY_REG = 1.0e-3
DEGENERACY_TOL = 1.0e-10
PAIR_ENERGY_TOL = 1.0e-8
LOCAL_GROUP_RADIUS = 5

HAMILTONIAN_SEEDS = tuple(range(25_043_000, 25_043_012))
FAMILIES = ("dephasing", "transverse")
PAULIS = ("I", "X", "Y", "Z")

PERMUTATIONS = 2000
BOOTSTRAPS = 200_000
BOOTSTRAP_SEED = 30_007_001
EPS_CORR = 1.0e-12

MIN_POSITIVE_SEEDS = 9


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
        vals: list[float] = []

        for family in FAMILIES:
            if group_number not in by_family[family]:
                vals = []
                break

            vals.append(
                by_family[family][group_number]
            )

        if vals:
            output[group_number] = min(vals)

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
        idx
        for idx in range(start, stop)
        if idx != group_number
    ]


def prominence_values(
    robust: dict[int, float],
    group_count: int,
) -> dict[int, float]:
    output: dict[int, float] = {}

    for group_number, target in robust.items():
        neighbours = [
            robust[idx]
            for idx in local_neighbours(
                group_number,
                group_count,
            )
            if idx in robust
        ]

        if neighbours:
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
        [values[idx] for idx in range(group_count)],
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


def exact_sign_test_positive(
    values: np.ndarray,
) -> tuple[int, int, float, float]:
    nonzero = values[np.abs(values) > 0.0]
    n = nonzero.size
    k = int(np.sum(nonzero > 0.0))

    if n == 0:
        return 0, 0, float("nan"), float("nan")

    def prob(i: int) -> float:
        return math.comb(n, i) / (2 ** n)

    p_one = sum(
        prob(i)
        for i in range(k, n + 1)
    )

    lower = min(k, n - k)
    p_two = min(
        1.0,
        2.0 * sum(
            prob(i)
            for i in range(0, lower + 1)
        ),
    )

    return k, n, float(p_one), float(p_two)


def wilcoxon_signed_rank_positive(
    values: np.ndarray,
) -> tuple[float, float, float]:
    values = values[np.abs(values) > 0.0]
    n = values.size

    if n == 0:
        return float("nan"), float("nan"), float("nan")

    abs_values = np.abs(values)
    order = np.argsort(abs_values)
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i + 1

        while (
            j < n
            and np.isclose(
                abs_values[order[j]],
                abs_values[order[i]],
                rtol=0.0,
                atol=1.0e-15,
            )
        ):
            j += 1

        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    observed = float(
        np.sum(ranks[values > 0.0])
    )

    totals = np.empty(
        1 << n,
        dtype=float,
    )

    for mask in range(1 << n):
        total = 0.0

        for idx in range(n):
            if (mask >> idx) & 1:
                total += ranks[idx]

        totals[mask] = total

    p_one = float(
        (np.sum(totals >= observed) + 1.0)
        / (totals.size + 1.0)
    )

    center = float(
        np.sum(ranks) / 2.0
    )
    observed_distance = abs(
        observed - center
    )

    p_two = float(
        (
            np.sum(
                np.abs(totals - center)
                >= observed_distance
            )
            + 1.0
        )
        / (totals.size + 1.0)
    )

    return observed, p_one, min(1.0, p_two)


def bootstrap_ci(
    values: np.ndarray,
    statistic: str,
    bootstraps: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)

    n = values.size
    samples = rng.choice(
        values,
        size=(bootstraps, n),
        replace=True,
    )

    if statistic == "mean":
        stats = np.mean(samples, axis=1)
        observed = float(np.mean(values))
    elif statistic == "median":
        stats = np.median(samples, axis=1)
        observed = float(np.median(values))
    else:
        raise ValueError(statistic)

    low, high = np.quantile(
        stats,
        [0.025, 0.975],
    )

    return observed, float(low), float(high)


def fisher_z(corr: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        corr,
        -1.0 + EPS_CORR,
        1.0 - EPS_CORR,
    )
    return np.arctanh(clipped)


def cohens_dz(values: np.ndarray) -> float:
    sd = float(
        np.std(values, ddof=1)
    )

    if sd == 0.0:
        return float("inf")

    return float(
        np.mean(values) / sd
    )


def run_seed(seed: int) -> dict:
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

    real_prom = prominence_values(
        real_robust,
        len(groups),
    )

    null_prom = prominence_values(
        null_robust,
        len(groups),
    )

    real_vec = vector_from_dict(
        real_prom,
        len(groups),
    )

    null_vec = vector_from_dict(
        null_prom,
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

    sizes = [
        int(group.size)
        for group in groups
    ]

    return {
        "seed": seed,
        "groups": len(groups),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "pairs": len(pairs),
        "corr_real": corr_real,
        "corr_null": corr_null,
        "delta_corr": corr_real - corr_null,
    }


def render(
    results: list[dict],
    bootstraps: int,
) -> str:
    delta = np.asarray(
        [row["delta_corr"] for row in results],
        dtype=float,
    )

    corr_real = np.asarray(
        [row["corr_real"] for row in results],
        dtype=float,
    )

    corr_null = np.asarray(
        [row["corr_null"] for row in results],
        dtype=float,
    )

    pos_count, nonzero_count, sign_p_one, sign_p_two = (
        exact_sign_test_positive(delta)
    )

    w_plus, wilcoxon_p_one, wilcoxon_p_two = (
        wilcoxon_signed_rank_positive(delta)
    )

    mean_delta, mean_low, mean_high = bootstrap_ci(
        delta,
        "mean",
        bootstraps,
        BOOTSTRAP_SEED,
    )

    median_delta, median_low, median_high = bootstrap_ci(
        delta,
        "median",
        bootstraps,
        BOOTSTRAP_SEED + 1,
    )

    criterion_1 = median_delta > 0.0
    criterion_2 = pos_count >= MIN_POSITIVE_SEEDS
    criterion_3 = sign_p_one < 0.05
    criterion_4 = wilcoxon_p_one < 0.05
    criterion_5 = median_low > 0.0

    confirmed = all([
        criterion_1,
        criterion_2,
        criterion_3,
        criterion_4,
        criterion_5,
    ])

    z_delta = (
        fisher_z(corr_real)
        - fisher_z(corr_null)
    )

    lines = [
        "=== Soft Spaces Phase 3.1 v30.7 INDEPENDENT FROZEN REPLICATION ===",
        f"Independent Hamiltonian seeds: {HAMILTONIAN_SEEDS[0]} ... {HAMILTONIAN_SEEDS[-1]}",
        f"Frozen physical model namespace: {FROZEN_MODEL_VERSION}",
        f"Bootstrap replicates: {bootstraps}",
        "",
        "PREDECLARED PRIMARY ENDPOINT:",
        "  delta_corr = corr_REAL - corr_NULL",
        "",
        "PREDECLARED CONFIRMATION CRITERIA:",
        "  C1 median(delta_corr) > 0",
        f"  C2 at least {MIN_POSITIVE_SEEDS}/12 seeds have delta_corr > 0",
        "  C3 one-sided exact sign-test p < 0.05",
        "  C4 one-sided Wilcoxon signed-rank p < 0.05",
        "  C5 95% bootstrap CI for median(delta_corr) lies entirely above 0",
        "",
        "seed | groups | size min:max | ±E pairs | corr_REAL | corr_NULL | delta_corr",
    ]

    for row in results:
        lines.append(
            f"{row['seed']} | "
            f"{row['groups']:6d} | "
            f"{row['min_size']:4d}:"
            f"{row['max_size']:<4d} | "
            f"{row['pairs']:8d} | "
            f"{row['corr_real']:+.6f} | "
            f"{row['corr_null']:+.6f} | "
            f"{row['delta_corr']:+.6f}"
        )

    lines.extend([
        "",
        "=== PRIMARY AGGREGATE RESULTS ===",
        f"Positive delta_corr seeds: {pos_count}/{nonzero_count}",
        f"Mean delta_corr: {mean_delta:+.6f}",
        f"Median delta_corr: {median_delta:+.6f}",
        f"Exact sign test, one-sided p: {sign_p_one:.8f}",
        f"Exact sign test, two-sided p: {sign_p_two:.8f}",
        f"Wilcoxon W+: {w_plus:.6f}",
        f"Wilcoxon one-sided p: {wilcoxon_p_one:.8f}",
        f"Wilcoxon two-sided p: {wilcoxon_p_two:.8f}",
        f"Bootstrap 95% CI mean: [{mean_low:+.6f}, {mean_high:+.6f}]",
        f"Bootstrap 95% CI median: [{median_low:+.6f}, {median_high:+.6f}]",
        f"Cohen dz: {cohens_dz(delta):+.6f}",
        "",
        "=== PREDECLARED GATE ===",
        f"C1 median > 0: {'PASS' if criterion_1 else 'FAIL'}",
        f"C2 >= {MIN_POSITIVE_SEEDS}/12 positive seeds: {'PASS' if criterion_2 else 'FAIL'}",
        f"C3 sign-test one-sided p < 0.05: {'PASS' if criterion_3 else 'FAIL'}",
        f"C4 Wilcoxon one-sided p < 0.05: {'PASS' if criterion_4 else 'FAIL'}",
        f"C5 bootstrap median CI > 0: {'PASS' if criterion_5 else 'FAIL'}",
        "",
        f"FINAL PREDECLARED DECISION: {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}",
        "",
        "=== SECONDARY DESCRIPTIVE OUTPUTS ===",
        f"Mean corr_REAL: {np.mean(corr_real):+.6f}",
        f"Mean corr_NULL: {np.mean(corr_null):+.6f}",
        f"Median corr_REAL: {np.median(corr_real):+.6f}",
        f"Median corr_NULL: {np.median(corr_null):+.6f}",
        f"Mean Fisher-z difference: {np.mean(z_delta):+.6f}",
        f"Median Fisher-z difference: {np.median(z_delta):+.6f}",
        "",
        "Interpretation:",
        "  CONFIRMED means the independently predeclared seed ensemble reproduces",
        "  the small REAL-specific enhancement of ±E correlation under all five",
        "  frozen acceptance criteria.",
        "",
        "  NOT CONFIRMED means the original v30.6 effect should not be promoted as",
        "  a replicated Phase 3 result, regardless of any attractive individual",
        "  seeds or secondary statistics.",
        "",
        "Scope:",
        "  v30.7 is an out-of-sample replication.  No post-hoc tuning of seeds,",
        "  thresholds or acceptance criteria is permitted.",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "v31_7_independent_replication_output.txt"
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run only the first independent seed. "
            "The predeclared replication decision is only meaningful for all 12."
        ),
    )

    parser.add_argument(
        "--bootstraps",
        type=int,
        default=BOOTSTRAPS,
        help=f"Bootstrap replicates (default {BOOTSTRAPS}).",
    )

    args = parser.parse_args()

    if args.bootstraps < 1000:
        raise ValueError(
            "--bootstraps must be at least 1000."
        )

    seeds = (
        HAMILTONIAN_SEEDS[:1]
        if args.smoke
        else HAMILTONIAN_SEEDS
    )

    results = [
        run_seed(seed)
        for seed in seeds
    ]

    report = render(
        results,
        args.bootstraps,
    )

    print(report, end="")
    args.output.write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
