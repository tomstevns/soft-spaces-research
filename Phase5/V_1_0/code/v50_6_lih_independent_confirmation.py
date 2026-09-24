#!/usr/bin/env python3
"""
Soft Spaces Phase 5 — v50_6 independent LiH confirmation

Purpose
-------
Independent transfer test of the frozen molecular Soft-Spaces ranking on LiH.

The H4 results are NOT used to tune the LiH ranking.  The same ranking formula,
K values, and evaluation logic are carried over unchanged.

System
------
Molecule: LiH
Basis:    STO-3G
Charge:   0
Spin:     0  (N_alpha = N_beta)
Geometry: linear Li-H bond-distance scan

Primary question
----------------
Does the frozen REAL ranking retain useful determinant-selection performance
on a chemically different molecule, and how does it compare with EN2?

Frozen from H4
--------------
REAL:
    score_i = |H_ir| / (|H_ii - H_rr| + eps)

EN2:
    score_i = |H_ir|^2 / (|H_ii - H_rr| + eps)

Reference determinant:
    minimum diagonal Hamiltonian energy inside the fixed
    (N_alpha, N_beta) sector.

K:
    2, 4, 8, 12, 16

Seeds:
    25043000 ... 25043011

No exact ground-state amplitudes are used to construct any deterministic
ranking.  They are used only for evaluation.

Additional baselines
--------------------
CHEM_EXCITATION:
    reference first, then excitation rank, then diagonal energy.

RANDOM:
    random ranking of the same candidate determinant set, 12 fixed seeds.

Implementation
--------------
Qiskit Nature + PySCF constructs the second-quantized electronic Hamiltonian.
Jordan-Wigner maps it to qubits.  The sparse qubit matrix is restricted to the
physical fixed-(N_alpha,N_beta) determinant sector before dense diagonalization.

For LiH/STO-3G one normally obtains:
    6 spatial orbitals
    12 spin orbitals
    4 electrons
    N_alpha = 2
    N_beta  = 2
    sector dimension = C(6,2)^2 = 225

The script validates these values at runtime rather than assuming them silently.

Qubit/orbital convention
------------------------
The sector selector uses block spin ordering:
    first n spatial-orbital qubits -> alpha
    last  n spatial-orbital qubits -> beta

This is the same convention used in the corrected H4 Phase-5 scripts.

Outputs
-------
results/lih_confirmation/
    phase5_v50_6_seed_level_results.csv
    phase5_v50_6_aggregate_results.csv
    phase5_v50_6_summary.json
    phase5_v50_6_summary.sha256

Usage
-----
    python -X utf8 -u ./v50_6_lih_independent_confirmation.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_VERSION = "v50_6"
PROJECT = "Molecular Quantum Soft Spaces"
PHASE = "Phase 5"

# Independent LiH bond-length scan.
GEOMETRIES = (1.00, 1.20, 1.40, 1.60, 2.00, 2.50, 3.00, 4.00)

# Frozen from H4.  Do not retune for LiH.
K_VALUES = (2, 4, 8, 12, 16)
SEEDS = tuple(range(25043000, 25043012))

REFERENCE_IMPORTANCE_MASS = 0.95
ENERGY_DENOM_EPS = 1e-9
CHEMICAL_ACCURACY_HARTREE = 0.001593601437

EXPECTED_NUM_SPATIAL_ORBITALS = 6
EXPECTED_NUM_ALPHA = 2
EXPECTED_NUM_BETA = 2
EXPECTED_SECTOR_DIMENSION = (
    math.comb(EXPECTED_NUM_SPATIAL_ORBITALS, EXPECTED_NUM_ALPHA)
    * math.comb(EXPECTED_NUM_SPATIAL_ORBITALS, EXPECTED_NUM_BETA)
)


@dataclass(frozen=True)
class EvaluationRow:
    geometry_angstrom: float
    method: str
    seed: int | None
    k: int
    captured_probability_mass: float
    absolute_energy_error_hartree: float
    selected_subspace_energy_hartree: float
    exact_sector_energy_hartree: float
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    selection_efficiency: float
    reaches_chemical_accuracy: bool
    overlap_with_real: int | None


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def popcount(x: int) -> int:
    return bin(int(x)).count("1")


def alpha_beta_counts(
    basis_index: int,
    num_spatial_orbitals: int,
) -> tuple[int, int]:
    mask = (1 << num_spatial_orbitals) - 1

    alpha_bits = basis_index & mask
    beta_bits = (basis_index >> num_spatial_orbitals) & mask

    return popcount(alpha_bits), popcount(beta_bits)


def fixed_spin_sector_basis_indices(
    num_spatial_orbitals: int,
    num_alpha: int,
    num_beta: int,
) -> np.ndarray:
    """
    Full-qubit computational-basis indices in the fixed
    (N_alpha, N_beta) sector.
    """
    n_qubits = 2 * num_spatial_orbitals
    indices = []

    for basis_index in range(1 << n_qubits):
        na, nb = alpha_beta_counts(
            basis_index,
            num_spatial_orbitals,
        )

        if na == num_alpha and nb == num_beta:
            indices.append(basis_index)

    expected = (
        math.comb(num_spatial_orbitals, num_alpha)
        * math.comb(num_spatial_orbitals, num_beta)
    )

    if len(indices) != expected:
        raise RuntimeError(
            "Fixed-spin sector dimension mismatch: "
            f"expected {expected}, got {len(indices)}"
        )

    return np.asarray(indices, dtype=np.int64)


def build_lih_reference(spacing: float) -> dict[str, Any]:
    """
    Build the LiH/STO-3G electronic Hamiltonian and restrict it to
    the fixed-(N_alpha,N_beta) determinant sector.
    """
    try:
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        from qiskit_nature.units import DistanceUnit
    except Exception as exc:
        raise RuntimeError(
            "Qiskit Nature / PySCF imports failed. "
            "Run inside the Phase-5 WSL environment."
        ) from exc

    # Center the diatomic on z=0.
    half = spacing / 2.0

    atom = (
        f"Li 0.0 0.0 {-half:.12f}; "
        f"H 0.0 0.0 {half:.12f}"
    )

    driver = PySCFDriver(
        atom=atom,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    problem = driver.run()

    num_spatial = int(problem.num_spatial_orbitals)
    num_alpha = int(problem.num_alpha)
    num_beta = int(problem.num_beta)
    num_spin_orbitals = 2 * num_spatial

    if num_spatial != EXPECTED_NUM_SPATIAL_ORBITALS:
        raise RuntimeError(
            "Unexpected LiH/STO-3G spatial-orbital count: "
            f"expected {EXPECTED_NUM_SPATIAL_ORBITALS}, got {num_spatial}"
        )

    if (num_alpha, num_beta) != (
        EXPECTED_NUM_ALPHA,
        EXPECTED_NUM_BETA,
    ):
        raise RuntimeError(
            "Unexpected LiH electron partition: "
            f"expected ({EXPECTED_NUM_ALPHA},{EXPECTED_NUM_BETA}), "
            f"got ({num_alpha},{num_beta})"
        )

    second_q_op = problem.hamiltonian.second_q_op()

    mapper = JordanWignerMapper()
    qubit_op = mapper.map(second_q_op)

    if int(qubit_op.num_qubits) != num_spin_orbitals:
        raise RuntimeError(
            "Mapped qubit count mismatch: "
            f"expected {num_spin_orbitals}, got {qubit_op.num_qubits}"
        )

    sector_basis = fixed_spin_sector_basis_indices(
        num_spatial,
        num_alpha,
        num_beta,
    )

    if len(sector_basis) != EXPECTED_SECTOR_DIMENSION:
        raise RuntimeError(
            "Unexpected LiH corrected-sector dimension: "
            f"expected {EXPECTED_SECTOR_DIMENSION}, got {len(sector_basis)}"
        )

    # Important: keep the full 2^12 operator sparse, then extract only
    # the 225 x 225 physical sector before converting to dense.
    full_sparse = qubit_op.to_matrix(sparse=True)

    sector_sparse = full_sparse[
        sector_basis, :
    ][:, sector_basis]

    h = np.asarray(
        sector_sparse.toarray(),
        dtype=np.complex128,
    )

    hermiticity_error = float(
        np.max(np.abs(h - h.conjugate().T))
    )

    if hermiticity_error > 1e-10:
        raise RuntimeError(
            f"Hamiltonian is not Hermitian: {hermiticity_error:.3e}"
        )

    evals, evecs = np.linalg.eigh(h)

    exact_electronic_energy = float(np.real(evals[0]))
    ground_state = evecs[:, 0]
    probabilities = np.abs(ground_state) ** 2
    probabilities /= float(np.sum(probabilities))

    nuclear_repulsion = 0.0

    # Qiskit Nature stores nuclear repulsion as a Hamiltonian constant.
    constants = getattr(problem.hamiltonian, "constants", {})

    for key, value in constants.items():
        if "nuclear" in str(key).lower():
            nuclear_repulsion += float(np.real(value))

    total_energy = (
        exact_electronic_energy + nuclear_repulsion
    )

    return {
        "atom": atom,
        "num_spatial_orbitals": num_spatial,
        "num_spin_orbitals": num_spin_orbitals,
        "num_alpha": num_alpha,
        "num_beta": num_beta,
        "sector_basis_indices": sector_basis,
        "hamiltonian": h,
        "probabilities": probabilities,
        "exact_electronic_energy": exact_electronic_energy,
        "nuclear_repulsion_energy": nuclear_repulsion,
        "exact_total_energy": total_energy,
        "hermiticity_error": hermiticity_error,
    }


def choose_reference_determinant(h: np.ndarray) -> int:
    diagonal = np.real(np.diag(h))
    return int(np.argmin(diagonal))


def real_score(
    h: np.ndarray,
    reference_idx: int,
) -> np.ndarray:
    diagonal = np.real(np.diag(h))
    e_ref = float(diagonal[reference_idx])

    coupling = np.abs(h[:, reference_idx])
    denominator = np.abs(diagonal - e_ref) + ENERGY_DENOM_EPS

    score = coupling / denominator
    score = np.asarray(score, dtype=float)
    score[reference_idx] = np.inf

    return score


def en2_score(
    h: np.ndarray,
    reference_idx: int,
) -> np.ndarray:
    diagonal = np.real(np.diag(h))
    e_ref = float(diagonal[reference_idx])

    coupling_sq = np.abs(h[:, reference_idx]) ** 2
    denominator = np.abs(diagonal - e_ref) + ENERGY_DENOM_EPS

    score = coupling_sq / denominator
    score = np.asarray(score, dtype=float)
    score[reference_idx] = np.inf

    return score


def rank_descending(score: np.ndarray) -> np.ndarray:
    safe = np.nan_to_num(
        score,
        nan=-np.inf,
        posinf=np.finfo(float).max,
        neginf=-np.inf,
    )

    return np.argsort(
        -safe,
        kind="stable",
    )


def excitation_rank(
    basis_index: int,
    reference_basis_index: int,
) -> int:
    distance = popcount(
        basis_index ^ reference_basis_index
    )

    if distance % 2 != 0:
        raise RuntimeError(
            "Odd Hamming distance between equal-particle determinants."
        )

    return distance // 2


def chemistry_excitation_ranking(
    h: np.ndarray,
    sector_basis: np.ndarray,
    reference_idx: int,
) -> np.ndarray:
    diagonal = np.real(np.diag(h))
    ref_basis = int(sector_basis[reference_idx])

    records = []

    for local_idx, basis_index in enumerate(sector_basis):
        records.append(
            (
                excitation_rank(
                    int(basis_index),
                    ref_basis,
                ),
                float(diagonal[local_idx]),
                int(basis_index),
                int(local_idx),
            )
        )

    records.sort()

    ranking = np.asarray(
        [record[3] for record in records],
        dtype=np.int64,
    )

    if int(ranking[0]) != reference_idx:
        raise RuntimeError(
            "CHEM_EXCITATION did not place reference determinant first."
        )

    return ranking


def random_rankings(
    n_candidates: int,
) -> dict[int, np.ndarray]:
    out = {}

    for seed in SEEDS:
        rng = np.random.default_rng(
            seed + 10_000_000
        )
        out[seed] = rng.permutation(
            n_candidates
        )

    return out


def important_reference_set(
    probabilities: np.ndarray,
) -> set[int]:
    order = np.argsort(
        -probabilities,
        kind="stable",
    )

    cumulative = 0.0
    selected = []

    for idx in order:
        selected.append(int(idx))
        cumulative += float(probabilities[idx])

        if cumulative >= REFERENCE_IMPORTANCE_MASS:
            break

    return set(selected)


def dcg(relevances: np.ndarray) -> float:
    if len(relevances) == 0:
        return 0.0

    discounts = np.log2(
        np.arange(2, len(relevances) + 2)
    )

    gains = (2.0 ** relevances) - 1.0

    return float(
        np.sum(gains / discounts)
    )


def ndcg_at_k(
    ranking: np.ndarray,
    probabilities: np.ndarray,
    k: int,
) -> float:
    chosen = ranking[:k]
    rel = probabilities[chosen]

    ideal = np.argsort(
        -probabilities,
        kind="stable",
    )[:k]

    ideal_rel = probabilities[ideal]
    denom = dcg(ideal_rel)

    if denom <= 0.0:
        return 0.0

    return dcg(rel) / denom


def selected_subspace_energy(
    h: np.ndarray,
    selected: np.ndarray,
) -> float:
    sub_h = h[
        np.ix_(selected, selected)
    ]

    evals = np.linalg.eigvalsh(
        sub_h
    )

    return float(
        np.real(evals[0])
    )


def evaluate_ranking(
    spacing: float,
    method: str,
    seed: int | None,
    ranking: np.ndarray,
    real_ranking: np.ndarray,
    h: np.ndarray,
    probabilities: np.ndarray,
    exact_energy: float,
    important_set: set[int],
    k: int,
) -> EvaluationRow:
    selected = np.asarray(
        ranking[:k],
        dtype=int,
    )

    captured_mass = float(
        np.sum(probabilities[selected])
    )

    subspace_energy = selected_subspace_energy(
        h,
        selected,
    )

    error = abs(
        subspace_energy - exact_energy
    )

    selected_set = set(
        map(int, selected)
    )

    tp = len(
        selected_set & important_set
    )

    precision = tp / k

    recall = (
        tp / len(important_set)
        if important_set
        else 0.0
    )

    ndcg = ndcg_at_k(
        ranking,
        probabilities,
        k,
    )

    efficiency = (
        captured_mass / k
    )

    overlap = None

    if method != "REAL":
        overlap = len(
            selected_set
            & set(
                map(int, real_ranking[:k])
            )
        )

    return EvaluationRow(
        geometry_angstrom=float(spacing),
        method=method,
        seed=seed,
        k=int(k),
        captured_probability_mass=captured_mass,
        absolute_energy_error_hartree=float(error),
        selected_subspace_energy_hartree=float(subspace_energy),
        exact_sector_energy_hartree=float(exact_energy),
        precision_at_k=float(precision),
        recall_at_k=float(recall),
        ndcg_at_k=float(ndcg),
        selection_efficiency=float(efficiency),
        reaches_chemical_accuracy=bool(
            error <= CHEMICAL_ACCURACY_HARTREE
        ),
        overlap_with_real=overlap,
    )


def aggregate_rows(
    rows: list[EvaluationRow],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[float, str, int],
        list[EvaluationRow],
    ] = {}

    for row in rows:
        key = (
            row.geometry_angstrom,
            row.method,
            row.k,
        )

        groups.setdefault(
            key,
            [],
        ).append(row)

    metrics = (
        "captured_probability_mass",
        "absolute_energy_error_hartree",
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
        "selection_efficiency",
    )

    out = []

    for key in sorted(groups):
        geometry, method, k = key
        group = groups[key]

        record = {
            "geometry_angstrom": geometry,
            "method": method,
            "k": k,
            "n": len(group),
        }

        for metric in metrics:
            values = np.asarray(
                [
                    getattr(row, metric)
                    for row in group
                ],
                dtype=float,
            )

            record[
                f"{metric}_mean"
            ] = float(np.mean(values))

            record[
                f"{metric}_median"
            ] = float(np.median(values))

            record[
                f"{metric}_std"
            ] = float(np.std(values, ddof=0))

            record[
                f"{metric}_min"
            ] = float(np.min(values))

            record[
                f"{metric}_max"
            ] = float(np.max(values))

        record[
            "chemical_accuracy_fraction"
        ] = float(
            np.mean(
                [
                    row.reaches_chemical_accuracy
                    for row in group
                ]
            )
        )

        out.append(record)

    return out


def comparison_summary(
    aggregate: list[dict[str, Any]],
) -> dict[str, Any]:
    index = {
        (
            float(row["geometry_angstrom"]),
            str(row["method"]),
            int(row["k"]),
        ): row
        for row in aggregate
    }

    baselines = (
        "EN2",
        "CHEM_EXCITATION",
        "RANDOM",
    )

    out = {}

    for baseline in baselines:
        per_k = {}

        for k in K_VALUES:
            real_wins = 0
            baseline_wins = 0
            ties = 0
            mixed = 0

            mass_deltas = []
            error_improvements = []

            for spacing in GEOMETRIES:
                real = index[
                    (spacing, "REAL", k)
                ]

                other = index[
                    (spacing, baseline, k)
                ]

                real_mass = real[
                    "captured_probability_mass_median"
                ]

                other_mass = other[
                    "captured_probability_mass_median"
                ]

                real_error = real[
                    "absolute_energy_error_hartree_median"
                ]

                other_error = other[
                    "absolute_energy_error_hartree_median"
                ]

                dm = real_mass - other_mass
                de = other_error - real_error

                mass_deltas.append(dm)
                error_improvements.append(de)

                tol = 1e-12

                real_better_mass = dm > tol
                real_better_error = de > tol

                other_better_mass = dm < -tol
                other_better_error = de < -tol

                mass_tie = abs(dm) <= tol
                error_tie = abs(de) <= tol

                if (
                    real_better_mass
                    and real_better_error
                ):
                    real_wins += 1

                elif (
                    other_better_mass
                    and other_better_error
                ):
                    baseline_wins += 1

                elif (
                    mass_tie
                    and error_tie
                ):
                    ties += 1

                else:
                    mixed += 1

            per_k[str(k)] = {
                "REAL_wins_both_primary_metrics": real_wins,
                "baseline_wins_both_primary_metrics": baseline_wins,
                "ties": ties,
                "mixed": mixed,
                "median_REAL_minus_baseline_captured_mass": float(
                    np.median(mass_deltas)
                ),
                "median_baseline_minus_REAL_energy_error": float(
                    np.median(error_improvements)
                ),
            }

        out[baseline] = per_k

    return out


def save_rows_csv(
    rows: list[EvaluationRow],
    path: Path,
) -> None:
    fields = list(
        EvaluationRow.__dataclass_fields__.keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                asdict(row)
            )


def save_aggregate_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    if not rows:
        raise RuntimeError(
            "No aggregate rows."
        )

    fields = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5 v50_6 independent LiH confirmation."
        )
    )

    v1_root = (
        Path(__file__).resolve().parent.parent
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            v1_root
            / "results"
            / "lih_confirmation"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("SOFT SPACES — PHASE 5 v50_6 INDEPENDENT LiH CONFIRMATION")
    print("=" * 78)
    print(f"Version      : {SCRIPT_VERSION}")
    print("Molecule     : LiH")
    print("Basis        : STO-3G")
    print(f"Geometries   : {GEOMETRIES}")
    print(f"K values     : {K_VALUES}")
    print(f"Seeds        : {len(SEEDS)}")
    print(
        "Frozen REAL  : |H_ir| / (|H_ii - H_rr| + eps)"
    )
    print(
        "Frozen EN2   : |H_ir|^2 / (|H_ii - H_rr| + eps)"
    )
    print("-" * 78)

    all_rows: list[EvaluationRow] = []
    geometry_details: dict[str, Any] = {}

    for gi, spacing in enumerate(
        GEOMETRIES,
        start=1,
    ):
        print(
            f"[{gi}/{len(GEOMETRIES)}] "
            f"Li-H={spacing:.2f} Å",
            flush=True,
        )

        ref = build_lih_reference(
            spacing
        )

        h = ref["hamiltonian"]
        probabilities = ref["probabilities"]
        sector_basis = ref[
            "sector_basis_indices"
        ]
        exact_energy = ref[
            "exact_electronic_energy"
        ]

        reference_idx = (
            choose_reference_determinant(h)
        )

        reference_basis = int(
            sector_basis[reference_idx]
        )

        na, nb = alpha_beta_counts(
            reference_basis,
            ref["num_spatial_orbitals"],
        )

        real_ranking = rank_descending(
            real_score(
                h,
                reference_idx,
            )
        )

        en2_ranking = rank_descending(
            en2_score(
                h,
                reference_idx,
            )
        )

        excitation_ranking = (
            chemistry_excitation_ranking(
                h,
                sector_basis,
                reference_idx,
            )
        )

        random_map = random_rankings(
            len(sector_basis)
        )

        important_set = (
            important_reference_set(
                probabilities
            )
        )

        max_probability = float(
            np.max(probabilities)
        )

        participation_ratio = float(
            1.0
            / np.sum(
                probabilities ** 2
            )
        )

        geometry_details[str(spacing)] = {
            "atom": ref["atom"],
            "num_spatial_orbitals": ref[
                "num_spatial_orbitals"
            ],
            "num_spin_orbitals": ref[
                "num_spin_orbitals"
            ],
            "num_alpha": ref["num_alpha"],
            "num_beta": ref["num_beta"],
            "sector_dimension": int(
                len(sector_basis)
            ),
            "reference_local_index": int(
                reference_idx
            ),
            "reference_basis_index": (
                reference_basis
            ),
            "reference_n_alpha": na,
            "reference_n_beta": nb,
            "max_probability": max_probability,
            "participation_ratio": (
                participation_ratio
            ),
            "important_reference_set_size_95pct_mass": (
                len(important_set)
            ),
            "exact_electronic_energy_hartree": (
                ref[
                    "exact_electronic_energy"
                ]
            ),
            "nuclear_repulsion_energy_hartree": (
                ref[
                    "nuclear_repulsion_energy"
                ]
            ),
            "exact_total_energy_hartree": (
                ref["exact_total_energy"]
            ),
            "hermiticity_error": ref[
                "hermiticity_error"
            ],
        }

        for k in K_VALUES:
            real_row = evaluate_ranking(
                spacing,
                "REAL",
                None,
                real_ranking,
                real_ranking,
                h,
                probabilities,
                exact_energy,
                important_set,
                k,
            )

            en2_row = evaluate_ranking(
                spacing,
                "EN2",
                None,
                en2_ranking,
                real_ranking,
                h,
                probabilities,
                exact_energy,
                important_set,
                k,
            )

            excitation_row = evaluate_ranking(
                spacing,
                "CHEM_EXCITATION",
                None,
                excitation_ranking,
                real_ranking,
                h,
                probabilities,
                exact_energy,
                important_set,
                k,
            )

            all_rows.extend(
                [
                    real_row,
                    en2_row,
                    excitation_row,
                ]
            )

            random_mass = []
            random_error = []

            for seed in SEEDS:
                random_row = evaluate_ranking(
                    spacing,
                    "RANDOM",
                    seed,
                    random_map[seed],
                    real_ranking,
                    h,
                    probabilities,
                    exact_energy,
                    important_set,
                    k,
                )

                all_rows.append(
                    random_row
                )

                random_mass.append(
                    random_row.captured_probability_mass
                )

                random_error.append(
                    random_row.absolute_energy_error_hartree
                )

            print(
                f"    K={k:2d} | "
                f"REAL mass={real_row.captured_probability_mass:.6f}, "
                f"dE={real_row.absolute_energy_error_hartree:.6e} | "
                f"EN2 mass={en2_row.captured_probability_mass:.6f}, "
                f"dE={en2_row.absolute_energy_error_hartree:.6e} | "
                f"EXC mass={excitation_row.captured_probability_mass:.6f}, "
                f"dE={excitation_row.absolute_energy_error_hartree:.6e}"
            )

        print(
            f"    sector={len(sector_basis)} | "
            f"ref basis={reference_basis} | "
            f"maxP={max_probability:.6f} | "
            f"PR={participation_ratio:.3f}"
        )

    aggregate = aggregate_rows(
        all_rows
    )

    comparisons = comparison_summary(
        aggregate
    )

    seed_csv = (
        output_dir
        / "phase5_v50_6_seed_level_results.csv"
    )

    aggregate_csv = (
        output_dir
        / "phase5_v50_6_aggregate_results.csv"
    )

    summary_json = (
        output_dir
        / "phase5_v50_6_summary.json"
    )

    hash_file = (
        output_dir
        / "phase5_v50_6_summary.sha256"
    )

    save_rows_csv(
        all_rows,
        seed_csv,
    )

    save_aggregate_csv(
        aggregate,
        aggregate_csv,
    )

    summary = {
        "_metadata": {
            "generated_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "script_version": (
                SCRIPT_VERSION
            ),
            "python_version": (
                sys.version.split()[0]
            ),
        },
        "project": PROJECT,
        "phase": PHASE,
        "independent_confirmation": {
            "molecule": "LiH",
            "basis": "STO-3G",
            "geometries_angstrom": list(
                GEOMETRIES
            ),
            "transfer_without_retuning": True,
            "k_values_frozen_from_h4": list(
                K_VALUES
            ),
            "seeds_frozen_from_h4": list(
                SEEDS
            ),
        },
        "frozen_methods": {
            "REAL": (
                "|H_ir| / "
                "(|H_ii - H_rr| + eps)"
            ),
            "EN2": (
                "|H_ir|^2 / "
                "(|H_ii - H_rr| + eps)"
            ),
            "CHEM_EXCITATION": (
                "reference first, "
                "then excitation rank, "
                "then diagonal energy"
            ),
            "reference_rule": (
                "minimum diagonal energy "
                "inside fixed "
                "(N_alpha,N_beta) sector"
            ),
            "ground_state_outcome_used_for_ranking": False,
        },
        "geometry_details": (
            geometry_details
        ),
        "comparison_summary": (
            comparisons
        ),
        "aggregate_results": (
            aggregate
        ),
        "interpretation_boundary": {
            "purpose": (
                "Independent transfer test "
                "of the frozen H4 ranking "
                "construction on LiH."
            ),
            "no_h4_outcome_dependent_retuning": True,
            "quantum_advantage_claim": False,
            "industrial_deployment_claim": False,
        },
    }

    summary_json.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    digest = sha256_hex(
        summary
    )

    hash_file.write_text(
        digest + "\n",
        encoding="ascii",
    )

    print("-" * 78)
    print("v50_6 COMPLETE")
    print("-" * 78)
    print(
        f"Seed-level CSV : {seed_csv}"
    )
    print(
        f"Aggregate CSV  : {aggregate_csv}"
    )
    print(
        f"Summary JSON   : {summary_json}"
    )
    print(
        f"SHA-256        : {digest}"
    )
    print("-" * 78)

    for baseline in (
        "EN2",
        "CHEM_EXCITATION",
        "RANDOM",
    ):
        print(
            f"REAL vs {baseline}"
        )

        for k in K_VALUES:
            item = comparisons[
                baseline
            ][str(k)]

            print(
                f"    K={k:2d}: "
                f"REAL wins="
                f"{item['REAL_wins_both_primary_metrics']}/8 | "
                f"{baseline} wins="
                f"{item['baseline_wins_both_primary_metrics']}/8 | "
                f"ties={item['ties']}/8 | "
                f"mixed={item['mixed']}/8 | "
                f"median Δmass="
                f"{item['median_REAL_minus_baseline_captured_mass']:+.6f} | "
                f"median Δerror="
                f"{item['median_baseline_minus_REAL_energy_error']:+.6e}"
            )

    print("-" * 78)
    print(
        "LiH is an independent transfer test. "
        "No REAL/EN2 score parameter was retuned."
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
