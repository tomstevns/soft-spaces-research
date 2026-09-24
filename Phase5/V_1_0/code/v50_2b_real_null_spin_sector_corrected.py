#!/usr/bin/env python3
"""
Soft Spaces Phase 5 — v50_2b spin-sector-corrected REAL vs NULL ranking

Purpose
-------
Repeat the v50_2 molecular determinant-ranking experiment with the physically
relevant fixed-spin occupation sector enforced explicitly:

    N_alpha = 2
    N_beta  = 2

for linear H4 / STO-3G.

Why v50_2b exists
-----------------
v50_2 ranked all determinants in the full N=4 sector (dimension 70). At longer
bond distances, the minimum-diagonal-energy determinant could fall outside the
intended M_S=0 occupation sector, producing a degenerate REAL score with no
useful couplings.

v50_2b corrects ONLY this sector-definition issue.

Frozen from v50_2
-----------------
- Same 8 geometries
- Same K values: 2, 4, 8, 12, 16
- Same 12 seeds
- Same REAL score formula
- Same matched NULL construction
- Same Random-K baseline
- Same evaluation metrics
- Same G2/G3/G4 gate logic
- Same 95% reference-importance threshold

Correction introduced here
---------------------------
Restrict ranking/evaluation candidate determinants to the occupation sector with:

    N_alpha = 2
    N_beta  = 2

Given 4 spatial orbitals and Qiskit Nature's spin-orbital ordering, the first
4 spin orbitals are alpha and the next 4 are beta for this benchmark.
The resulting candidate-space dimension is:

    C(4,2) * C(4,2) = 36

Important
---------
This is a methodological correction, not an outcome-dependent threshold change.
All v50_2 results remain preserved. v50_2b should be interpreted as a corrected
replication with the same preregistered evaluation framework.

Usage
-----
    python -X utf8 -u ./v50_2b_real_null_spin_sector_corrected.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_VERSION = "v50_2b"
PROJECT = "Molecular Quantum Soft Spaces"
PHASE = "Phase 5"

GEOMETRIES = (0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
K_VALUES = (2, 4, 8, 12, 16)
SEEDS = tuple(range(25043000, 25043012))

NUM_SPATIAL_ORBITALS = 4
NUM_SPIN_ORBITALS = 8
NUM_ALPHA = 2
NUM_BETA = 2
FULL_N4_DIMENSION = 70
CORRECTED_SECTOR_DIMENSION = math.comb(4, 2) * math.comb(4, 2)

REFERENCE_IMPORTANCE_MASS = 0.95
ENERGY_DENOM_EPS = 1e-9


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


def geometry_tag(spacing: float) -> str:
    return f"{spacing:.2f}".replace(".", "p")


def reference_paths(reference_dir: Path, spacing: float) -> tuple[Path, Path]:
    tag = geometry_tag(spacing)
    npz_path = reference_dir / f"h4_sto3g_d_{tag}A_reference.npz"
    json_path = reference_dir / f"h4_sto3g_d_{tag}A_reference.json"
    return npz_path, json_path


def popcount(x: int) -> int:
    return bin(int(x)).count("1")


def alpha_beta_counts_from_basis_index(
    basis_index: int,
    num_spatial_orbitals: int = NUM_SPATIAL_ORBITALS,
) -> tuple[int, int]:
    """
    Count alpha and beta occupations in Qiskit Nature's block spin ordering.

    Assumption for this benchmark:
        q0..q3  -> alpha spin orbitals
        q4..q7  -> beta spin orbitals

    This is consistent with Qiskit Nature's ElectronicStructureProblem ordering
    for the spin-orbital representation used in v50_1.
    """
    alpha_mask = (1 << num_spatial_orbitals) - 1
    alpha_bits = basis_index & alpha_mask
    beta_bits = (basis_index >> num_spatial_orbitals) & alpha_mask

    return popcount(alpha_bits), popcount(beta_bits)


def corrected_sector_local_indices(
    sector_basis_indices: np.ndarray,
) -> np.ndarray:
    keep = []

    for local_idx, basis_index in enumerate(sector_basis_indices):
        n_alpha, n_beta = alpha_beta_counts_from_basis_index(int(basis_index))

        if n_alpha == NUM_ALPHA and n_beta == NUM_BETA:
            keep.append(local_idx)

    keep_arr = np.asarray(keep, dtype=np.int64)

    if len(keep_arr) != CORRECTED_SECTOR_DIMENSION:
        raise RuntimeError(
            "Corrected spin-sector dimension mismatch: "
            f"expected {CORRECTED_SECTOR_DIMENSION}, got {len(keep_arr)}"
        )

    return keep_arr


def load_reference(reference_dir: Path, spacing: float) -> dict[str, Any]:
    npz_path, json_path = reference_paths(reference_dir, spacing)

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing v50_1 NPZ file: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing v50_1 JSON file: {json_path}")

    data = np.load(npz_path)

    sector_indices = np.asarray(data["sector_indices"], dtype=np.int64)
    full_hamiltonian = np.asarray(
        data["sector_hamiltonian"],
        dtype=np.complex128,
    )
    full_probabilities = np.asarray(
        data["ground_state_probabilities"],
        dtype=float,
    )

    if len(sector_indices) != FULL_N4_DIMENSION:
        raise RuntimeError(
            f"Unexpected v50_1 N=4 sector dimension at d={spacing}: "
            f"{len(sector_indices)}"
        )

    corrected_local = corrected_sector_local_indices(sector_indices)

    corrected_basis_indices = sector_indices[corrected_local]
    corrected_hamiltonian = full_hamiltonian[
        np.ix_(corrected_local, corrected_local)
    ]
    corrected_probabilities_raw = full_probabilities[corrected_local]

    retained_mass = float(np.sum(corrected_probabilities_raw))

    if retained_mass <= 0.0:
        raise RuntimeError(
            f"Corrected sector has zero reference probability at d={spacing}"
        )

    # The exact H4 singlet ground state should reside entirely in this sector
    # up to numerical precision. We preserve the physical probabilities, but
    # also create a normalized vector for ranking metrics such as NDCG.
    normalized_probabilities = corrected_probabilities_raw / retained_mass

    evals, evecs = np.linalg.eigh(corrected_hamiltonian)
    exact_corrected_energy = float(np.real(evals[0]))

    hermiticity_error = float(
        np.max(
            np.abs(
                corrected_hamiltonian
                - corrected_hamiltonian.conjugate().T
            )
        )
    )

    if hermiticity_error > 1e-10:
        raise RuntimeError(
            f"Corrected Hamiltonian not Hermitian at d={spacing}: "
            f"{hermiticity_error:.3e}"
        )

    return {
        "npz_path": str(npz_path),
        "json_path": str(json_path),
        "full_sector_basis_indices": sector_indices,
        "corrected_local_indices": corrected_local,
        "basis_indices": corrected_basis_indices,
        "hamiltonian": corrected_hamiltonian,
        "probabilities_physical": corrected_probabilities_raw,
        "probabilities_normalized": normalized_probabilities,
        "retained_reference_mass": retained_mass,
        "exact_energy": exact_corrected_energy,
    }


def choose_reference_determinant(hamiltonian: np.ndarray) -> int:
    """
    Same rule as v50_2, but now applied only inside the corrected sector.
    """
    diagonal = np.real(np.diag(hamiltonian))
    return int(np.argmin(diagonal))


def real_soft_spaces_score(
    hamiltonian: np.ndarray,
    reference_local_index: int,
) -> np.ndarray:
    """
    Same score as v50_2:
        score_i = |H_ir| / (|H_ii - H_rr| + eps)
    """
    diagonal = np.real(np.diag(hamiltonian))
    e_ref = float(diagonal[reference_local_index])

    coupling = np.abs(hamiltonian[:, reference_local_index])
    denominator = np.abs(diagonal - e_ref) + ENERGY_DENOM_EPS

    score = coupling / denominator
    score = np.asarray(score, dtype=float)
    score[reference_local_index] = np.inf

    return score


def rank_descending(score: np.ndarray) -> np.ndarray:
    safe = np.nan_to_num(
        score,
        nan=-np.inf,
        posinf=np.finfo(float).max,
        neginf=-np.inf,
    )
    return np.argsort(-safe, kind="stable")


def null_rankings_from_score(
    score: np.ndarray,
    seeds: tuple[int, ...],
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        permuted = np.array(score, copy=True)
        rng.shuffle(permuted)
        out[seed] = rank_descending(permuted)

    return out


def random_rankings(
    n_candidates: int,
    seeds: tuple[int, ...],
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}

    for seed in seeds:
        rng = np.random.default_rng(seed + 10_000_000)
        out[seed] = rng.permutation(n_candidates)

    return out


def important_reference_set(
    probabilities_normalized: np.ndarray,
    target_mass: float = REFERENCE_IMPORTANCE_MASS,
) -> set[int]:
    order = np.argsort(-probabilities_normalized, kind="stable")
    cumulative = 0.0
    selected: list[int] = []

    for idx in order:
        selected.append(int(idx))
        cumulative += float(probabilities_normalized[idx])

        if cumulative >= target_mass:
            break

    return set(selected)


def dcg(relevances: np.ndarray) -> float:
    if len(relevances) == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(relevances) + 2))
    gains = (2.0 ** relevances) - 1.0

    return float(np.sum(gains / discounts))


def ndcg_at_k(
    selected_order: np.ndarray,
    probabilities_normalized: np.ndarray,
    k: int,
) -> float:
    chosen = selected_order[:k]
    rel = probabilities_normalized[chosen]

    ideal_order = np.argsort(
        -probabilities_normalized,
        kind="stable",
    )[:k]

    ideal_rel = probabilities_normalized[ideal_order]

    denom = dcg(ideal_rel)

    if denom <= 0.0:
        return 0.0

    return dcg(rel) / denom


def selected_subspace_energy(
    hamiltonian: np.ndarray,
    selected_indices: np.ndarray,
) -> float:
    sub_h = hamiltonian[np.ix_(selected_indices, selected_indices)]
    evals = np.linalg.eigvalsh(sub_h)
    return float(np.real(evals[0]))


def evaluate_ranking(
    spacing: float,
    method: str,
    seed: int | None,
    ranking: np.ndarray,
    real_ranking: np.ndarray,
    hamiltonian: np.ndarray,
    probabilities_physical: np.ndarray,
    probabilities_normalized: np.ndarray,
    exact_energy: float,
    important_set: set[int],
    k: int,
) -> EvaluationRow:
    selected = np.asarray(ranking[:k], dtype=int)

    captured_mass = float(
        np.sum(probabilities_physical[selected])
    )

    subspace_energy = selected_subspace_energy(
        hamiltonian,
        selected,
    )

    energy_error = abs(subspace_energy - exact_energy)

    tp = len(set(map(int, selected)) & important_set)

    precision = tp / k
    recall = tp / len(important_set) if important_set else 0.0

    ndcg = ndcg_at_k(
        ranking,
        probabilities_normalized,
        k,
    )

    efficiency = captured_mass / k

    overlap = None

    if method != "REAL":
        overlap = len(
            set(map(int, selected))
            & set(map(int, real_ranking[:k]))
        )

    return EvaluationRow(
        geometry_angstrom=float(spacing),
        method=method,
        seed=seed,
        k=int(k),
        captured_probability_mass=captured_mass,
        absolute_energy_error_hartree=float(energy_error),
        selected_subspace_energy_hartree=float(subspace_energy),
        exact_sector_energy_hartree=float(exact_energy),
        precision_at_k=float(precision),
        recall_at_k=float(recall),
        ndcg_at_k=float(ndcg),
        selection_efficiency=float(efficiency),
        overlap_with_real=overlap,
    )


def aggregate_rows(
    rows: list[EvaluationRow],
) -> list[dict[str, Any]]:
    groups: dict[tuple[float, str, int], list[EvaluationRow]] = {}

    for row in rows:
        key = (
            row.geometry_angstrom,
            row.method,
            row.k,
        )
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []

    metrics = (
        "captured_probability_mass",
        "absolute_energy_error_hartree",
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
        "selection_efficiency",
    )

    for key in sorted(groups):
        geometry, method, k = key
        group = groups[key]

        record: dict[str, Any] = {
            "geometry_angstrom": geometry,
            "method": method,
            "k": k,
            "n": len(group),
        }

        for metric in metrics:
            values = np.array(
                [getattr(row, metric) for row in group],
                dtype=float,
            )

            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_std"] = float(np.std(values, ddof=0))
            record[f"{metric}_min"] = float(np.min(values))
            record[f"{metric}_max"] = float(np.max(values))

        out.append(record)

    return out


def build_gate_summary(
    aggregate: list[dict[str, Any]],
) -> dict[str, Any]:
    index: dict[tuple[float, str, int], dict[str, Any]] = {
        (
            float(r["geometry_angstrom"]),
            str(r["method"]),
            int(r["k"]),
        ): r
        for r in aggregate
    }

    per_k: dict[str, Any] = {}

    for k in K_VALUES:
        both_real_vs_null = []
        both_real_vs_random = []

        for spacing in GEOMETRIES:
            real = index[(spacing, "REAL", k)]
            null = index[(spacing, "NULL", k)]
            random = index[(spacing, "RANDOM", k)]

            real_mass = real["captured_probability_mass_median"]
            null_mass = null["captured_probability_mass_median"]
            random_mass = random["captured_probability_mass_median"]

            real_err = real["absolute_energy_error_hartree_median"]
            null_err = null["absolute_energy_error_hartree_median"]
            random_err = random["absolute_energy_error_hartree_median"]

            both_real_vs_null.append(
                (real_mass > null_mass)
                and (real_err < null_err)
            )

            both_real_vs_random.append(
                (real_mass > random_mass)
                and (real_err < random_err)
            )

        n_null = int(sum(both_real_vs_null))
        n_random = int(sum(both_real_vs_random))

        per_k[str(k)] = {
            "geometries_REAL_beats_NULL_on_both_primary_metrics": n_null,
            "geometries_REAL_beats_RANDOM_on_both_primary_metrics": n_random,
            "g3_threshold_6_of_8_vs_null": n_null >= 6,
            "g4_random_separation_same_k": n_random >= 6,
        }

    qualifying_k_g3 = [
        int(k)
        for k, record in per_k.items()
        if record["g3_threshold_6_of_8_vs_null"]
    ]

    qualifying_k_g4 = [
        int(k)
        for k, record in per_k.items()
        if record["g3_threshold_6_of_8_vs_null"]
        and record["g4_random_separation_same_k"]
    ]

    g2_by_k = {}

    for k in K_VALUES:
        mass_deltas = []
        error_improvements = []

        for spacing in GEOMETRIES:
            real = index[(spacing, "REAL", k)]
            null = index[(spacing, "NULL", k)]

            mass_deltas.append(
                real["captured_probability_mass_median"]
                - null["captured_probability_mass_median"]
            )

            error_improvements.append(
                null["absolute_energy_error_hartree_median"]
                - real["absolute_energy_error_hartree_median"]
            )

        med_mass = float(np.median(mass_deltas))
        med_err = float(np.median(error_improvements))

        g2_by_k[str(k)] = {
            "median_REAL_minus_NULL_captured_mass": med_mass,
            "median_NULL_minus_REAL_energy_error": med_err,
            "passes_both": med_mass > 0.0 and med_err > 0.0,
        }

    return {
        "G1_validity": {
            "pass": True,
            "corrected_spin_sector": "N_alpha=2, N_beta=2",
            "candidate_dimension": CORRECTED_SECTOR_DIMENSION,
        },
        "G2_REAL_beats_matched_NULL_on_primary_metrics": {
            "pass_any_preregistered_k": any(
                x["passes_both"] for x in g2_by_k.values()
            ),
            "by_k": g2_by_k,
        },
        "G3_advantage_not_single_geometry": {
            "pass_any_preregistered_k": bool(qualifying_k_g3),
            "qualifying_k": qualifying_k_g3,
            "by_k": per_k,
        },
        "G4_random_baseline_separation": {
            "pass_any_k_also_passing_G3": bool(qualifying_k_g4),
            "qualifying_k": qualifying_k_g4,
        },
        "G5_practical_reduction": {
            "status": "DEFERRED_TO_v50_5",
        },
        "G6_reproducibility": {
            "status": "FULL_12_SEED_SET_REPORTED",
            "seeds": list(SEEDS),
        },
    }


def save_rows_csv(
    rows: list[EvaluationRow],
    path: Path,
) -> None:
    fieldnames = list(EvaluationRow.__dataclass_fields__.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def save_aggregate_csv(
    aggregate: list[dict[str, Any]],
    path: Path,
) -> None:
    if not aggregate:
        raise RuntimeError("No aggregate rows to save.")

    fieldnames = list(aggregate[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(aggregate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5 v50_2b spin-sector-corrected REAL vs NULL ranking."
        )
    )

    v1_root = Path(__file__).resolve().parent.parent

    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=v1_root / "results" / "reference",
        help="Directory containing v50_1 reference files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=v1_root / "results" / "real_null_spin_corrected",
        help="Directory for v50_2b outputs.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    reference_dir = args.reference_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SOFT SPACES — PHASE 5 v50_2b SPIN-SECTOR-CORRECTED REAL vs NULL")
    print("=" * 78)
    print(f"Version             : {SCRIPT_VERSION}")
    print(f"Reference dir       : {reference_dir}")
    print(f"Output dir          : {output_dir}")
    print(f"Full N=4 dimension  : {FULL_N4_DIMENSION}")
    print(f"Corrected dimension : {CORRECTED_SECTOR_DIMENSION}")
    print(f"N_alpha / N_beta    : {NUM_ALPHA} / {NUM_BETA}")
    print(f"K values            : {K_VALUES}")
    print(f"Seeds               : {len(SEEDS)}")
    print("-" * 78)

    all_rows: list[EvaluationRow] = []
    geometry_details: dict[str, Any] = {}

    for gi, spacing in enumerate(GEOMETRIES, start=1):
        print(
            f"[{gi}/{len(GEOMETRIES)}] d={spacing:.2f} Å",
            flush=True,
        )

        ref = load_reference(reference_dir, spacing)

        hamiltonian = ref["hamiltonian"]
        p_phys = ref["probabilities_physical"]
        p_norm = ref["probabilities_normalized"]
        exact_energy = ref["exact_energy"]

        reference_local_index = choose_reference_determinant(
            hamiltonian
        )

        reference_basis_index = int(
            ref["basis_indices"][reference_local_index]
        )

        n_alpha_ref, n_beta_ref = alpha_beta_counts_from_basis_index(
            reference_basis_index
        )

        if (n_alpha_ref, n_beta_ref) != (NUM_ALPHA, NUM_BETA):
            raise RuntimeError(
                "Internal error: selected reference determinant is outside "
                "the corrected spin sector."
            )

        real_score = real_soft_spaces_score(
            hamiltonian,
            reference_local_index,
        )

        real_ranking = rank_descending(real_score)

        null_rankings = null_rankings_from_score(
            real_score,
            SEEDS,
        )

        random_rankings_map = random_rankings(
            len(p_phys),
            SEEDS,
        )

        important_set = important_reference_set(p_norm)

        nonzero_scores = int(
            np.count_nonzero(
                np.isfinite(real_score)
                & (real_score > 0.0)
            )
        )

        geometry_details[str(spacing)] = {
            "retained_reference_probability_mass": (
                ref["retained_reference_mass"]
            ),
            "candidate_dimension": len(p_phys),
            "reference_local_index_corrected_sector": (
                reference_local_index
            ),
            "reference_basis_index_full_qubit_basis": (
                reference_basis_index
            ),
            "reference_n_alpha": n_alpha_ref,
            "reference_n_beta": n_beta_ref,
            "nonzero_finite_real_scores": nonzero_scores,
            "important_reference_set_size_95pct_mass": len(
                important_set
            ),
            "real_ranking_corrected_local_indices": [
                int(x) for x in real_ranking
            ],
            "real_ranking_full_basis_indices": [
                int(ref["basis_indices"][x])
                for x in real_ranking
            ],
            "real_score": [
                None if not np.isfinite(x) else float(x)
                for x in real_score
            ],
        }

        for k in K_VALUES:
            if k > len(p_phys):
                raise RuntimeError(
                    f"K={k} exceeds corrected candidate dimension "
                    f"{len(p_phys)}"
                )

            real_row = evaluate_ranking(
                spacing=spacing,
                method="REAL",
                seed=None,
                ranking=real_ranking,
                real_ranking=real_ranking,
                hamiltonian=hamiltonian,
                probabilities_physical=p_phys,
                probabilities_normalized=p_norm,
                exact_energy=exact_energy,
                important_set=important_set,
                k=k,
            )

            all_rows.append(real_row)

            null_mass_values = []
            null_error_values = []

            for seed in SEEDS:
                null_row = evaluate_ranking(
                    spacing=spacing,
                    method="NULL",
                    seed=seed,
                    ranking=null_rankings[seed],
                    real_ranking=real_ranking,
                    hamiltonian=hamiltonian,
                    probabilities_physical=p_phys,
                    probabilities_normalized=p_norm,
                    exact_energy=exact_energy,
                    important_set=important_set,
                    k=k,
                )

                all_rows.append(null_row)

                random_row = evaluate_ranking(
                    spacing=spacing,
                    method="RANDOM",
                    seed=seed,
                    ranking=random_rankings_map[seed],
                    real_ranking=real_ranking,
                    hamiltonian=hamiltonian,
                    probabilities_physical=p_phys,
                    probabilities_normalized=p_norm,
                    exact_energy=exact_energy,
                    important_set=important_set,
                    k=k,
                )

                all_rows.append(random_row)

                null_mass_values.append(
                    null_row.captured_probability_mass
                )

                null_error_values.append(
                    null_row.absolute_energy_error_hartree
                )

            print(
                f"    K={k:2d} | "
                f"REAL mass={real_row.captured_probability_mass:.6f} "
                f"vs NULL med={np.median(null_mass_values):.6f} | "
                f"REAL dE={real_row.absolute_energy_error_hartree:.6e} "
                f"vs NULL med={np.median(null_error_values):.6e}"
            )

        print(
            f"    retained reference mass={ref['retained_reference_mass']:.12f} | "
            f"reference basis={reference_basis_index} | "
            f"nonzero REAL scores={nonzero_scores}"
        )

    aggregate = aggregate_rows(all_rows)
    gates = build_gate_summary(aggregate)

    seed_csv = (
        output_dir
        / "phase5_v50_2b_seed_level_results.csv"
    )

    aggregate_csv = (
        output_dir
        / "phase5_v50_2b_aggregate_results.csv"
    )

    summary_json = (
        output_dir
        / "phase5_v50_2b_summary.json"
    )

    hash_file = (
        output_dir
        / "phase5_v50_2b_summary.sha256"
    )

    save_rows_csv(all_rows, seed_csv)
    save_aggregate_csv(aggregate, aggregate_csv)

    summary = {
        "_metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script_version": SCRIPT_VERSION,
            "python_version": sys.version.split()[0],
        },
        "project": PROJECT,
        "phase": PHASE,
        "correction_from_v50_2": {
            "issue": (
                "v50_2 ranked the full N=4 determinant sector, allowing "
                "reference determinants outside the intended N_alpha=2, "
                "N_beta=2 occupation sector."
            ),
            "correction": (
                "Restrict candidate/reference determinant space to "
                "N_alpha=2, N_beta=2 while keeping the original ranking "
                "formula, seeds, K values, metrics and gate logic."
            ),
            "full_n4_dimension": FULL_N4_DIMENSION,
            "corrected_dimension": CORRECTED_SECTOR_DIMENSION,
            "outcome_thresholds_changed": False,
        },
        "methodology": {
            "real_score": (
                "|H_ir| / (|H_ii - H_rr| + eps), with r chosen as the "
                "minimum-diagonal-energy determinant inside the corrected "
                "N_alpha=2, N_beta=2 sector"
            ),
            "real_ranking_uses_reference_ground_state_amplitudes": False,
            "reference_ground_state_used_for_evaluation_only": True,
            "null_control": (
                "Permutation of REAL score values across the 36 corrected "
                "candidate determinant identities"
            ),
            "random_baseline": (
                "Uniform random ranking of the 36 corrected candidates"
            ),
            "k_values": list(K_VALUES),
            "seeds": list(SEEDS),
            "reference_importance_mass_threshold": (
                REFERENCE_IMPORTANCE_MASS
            ),
            "energy_denominator_epsilon": ENERGY_DENOM_EPS,
        },
        "geometry_details": geometry_details,
        "decision_gates": gates,
        "aggregate_results": aggregate,
        "interpretation_boundary": {
            "claim_scope": (
                "v50_2b is a corrected replication of v50_2 in the "
                "physically intended spin-occupation sector."
            ),
            "v50_2_results_preserved": True,
            "industrial_utility_claim": False,
            "secondary_molecule_transfer_test": False,
        },
    }

    summary_json.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    digest = sha256_hex(summary)
    hash_file.write_text(
        digest + "\n",
        encoding="ascii",
    )

    print("-" * 78)
    print("v50_2b COMPLETE")
    print("-" * 78)
    print(f"Seed-level CSV : {seed_csv}")
    print(f"Aggregate CSV  : {aggregate_csv}")
    print(f"Summary JSON   : {summary_json}")
    print(f"SHA-256        : {digest}")
    print("-" * 78)

    g2 = gates[
        "G2_REAL_beats_matched_NULL_on_primary_metrics"
    ]

    g3 = gates[
        "G3_advantage_not_single_geometry"
    ]

    g4 = gates[
        "G4_random_baseline_separation"
    ]

    print(
        "G2 REAL vs NULL primary metrics : "
        f"{'PASS' if g2['pass_any_preregistered_k'] else 'FAIL'}"
    )

    print(
        "G3 >=6/8 geometries             : "
        f"{'PASS' if g3['pass_any_preregistered_k'] else 'FAIL'}"
    )

    print(
        "G4 separation from Random-K     : "
        f"{'PASS' if g4['pass_any_k_also_passing_G3'] else 'FAIL'}"
    )

    print("-" * 78)
    print(
        "Interpret v50_2b as the sector-corrected replication of v50_2."
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
