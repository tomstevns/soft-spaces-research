#!/usr/bin/env python3
"""
Soft Spaces Phase 5 — v50_2 REAL vs NULL determinant ranking

Purpose
-------
Run the first preregistered Phase 5 Soft-Spaces ranking experiment on the
H4/STO-3G reference data produced by v50_1.

This script does NOT recompute the molecular Hamiltonian. It consumes the
reference NPZ/JSON files written by v50_1 and evaluates whether a frozen
Soft-Spaces-inspired ranking can prioritise useful determinants better than
matched NULL/random controls at identical K budgets.

Important methodological boundary
---------------------------------
v50_2 is the FIRST outcome-facing ranking experiment in Phase 5.

The REAL score used here is deliberately outcome-blind with respect to the
reference ground-state amplitudes. The exact amplitudes/probabilities are used
ONLY for evaluation after the ranking has been produced.

The initial REAL ranking is based on Hamiltonian connectivity to a fixed
reference determinant:
    score_i = |H_ii - H_rr|^{-1} * |H_ir|   (regularised)

where r is the determinant with minimum diagonal Hamiltonian energy inside the
fixed N=4 sector. This is a simple perturbative/soft-coupling proxy: determinants
that are strongly coupled to a low-energy reference and not energetically far
away receive higher score.

This is intentionally conservative. It is a first molecular bridge, not a final
claim that this exact score is the definitive Soft-Spaces observable.

Matched NULL controls
---------------------
For each geometry and seed, the REAL score values are randomly permuted over the
same 70 determinants. This preserves:
- candidate count
- exact score distribution
- K
- molecular geometry
- sector
while destroying the mapping between Soft-Spaces score and determinant identity.

Mandatory baseline
------------------
Random-K is also evaluated explicitly.

Preregistered K values
----------------------
2, 4, 8, 12, 16

Preregistered seeds
-------------------
25043000 ... 25043011

Primary metrics
---------------
1. Captured probability mass
2. Absolute energy error after selected-subspace diagonalisation

Additional metrics
------------------
Precision@K
Recall@K
NDCG@K
Selection efficiency

Usage
-----
    python -X utf8 -u ./v50_2_real_null_ranking.py

Expected project layout
-----------------------
Phase5/V_1_0/
    code/
        v50_2_real_null_ranking.py
    results/
        reference/
            h4_sto3g_d_*.npz
            h4_sto3g_d_*_reference.json
        real_null/
            ... generated here ...
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


SCRIPT_VERSION = "v50_2"
PROJECT = "Molecular Quantum Soft Spaces"
PHASE = "Phase 5"

GEOMETRIES = (0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
K_VALUES = (2, 4, 8, 12, 16)
SEEDS = tuple(range(25043000, 25043012))

NUM_ELECTRONS = 4
NUM_SPIN_ORBITALS = 8
SECTOR_DIMENSION = 70

# Used only for Precision/Recall evaluation.
# "Important" determinants are frozen as the smallest reference-probability-ranked
# prefix whose cumulative probability reaches this threshold.
REFERENCE_IMPORTANCE_MASS = 0.95

# Regularisation for the REAL perturbative score denominator.
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


def load_reference(reference_dir: Path, spacing: float) -> dict[str, Any]:
    npz_path, json_path = reference_paths(reference_dir, spacing)

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing v50_1 NPZ file: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing v50_1 JSON file: {json_path}")

    data = np.load(npz_path)

    sector_indices = np.asarray(data["sector_indices"], dtype=np.int64)
    hamiltonian = np.asarray(data["sector_hamiltonian"], dtype=np.complex128)
    eigenvalues = np.asarray(data["eigenvalues_electronic_hartree"], dtype=float)
    amplitudes = np.asarray(data["ground_state_amplitudes"], dtype=np.complex128)
    probabilities = np.asarray(data["ground_state_probabilities"], dtype=float)

    payload = json.loads(json_path.read_text(encoding="utf-8"))

    if len(sector_indices) != SECTOR_DIMENSION:
        raise RuntimeError(
            f"Sector dimension mismatch at d={spacing}: "
            f"expected {SECTOR_DIMENSION}, got {len(sector_indices)}"
        )

    if hamiltonian.shape != (SECTOR_DIMENSION, SECTOR_DIMENSION):
        raise RuntimeError(
            f"Hamiltonian shape mismatch at d={spacing}: {hamiltonian.shape}"
        )

    if len(probabilities) != SECTOR_DIMENSION:
        raise RuntimeError(
            f"Probability length mismatch at d={spacing}"
        )

    if not math.isclose(
        float(np.sum(probabilities)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError(
            f"Reference probabilities do not sum to 1 at d={spacing}"
        )

    return {
        "npz_path": str(npz_path),
        "json_path": str(json_path),
        "sector_indices": sector_indices,
        "hamiltonian": hamiltonian,
        "eigenvalues": eigenvalues,
        "amplitudes": amplitudes,
        "probabilities": probabilities,
        "payload": payload,
    }


def choose_reference_determinant(hamiltonian: np.ndarray) -> int:
    """
    Outcome-blind reference determinant:
    choose the basis determinant with the minimum diagonal electronic energy.
    """
    diagonal = np.real(np.diag(hamiltonian))
    return int(np.argmin(diagonal))


def real_soft_spaces_score(
    hamiltonian: np.ndarray,
    reference_local_index: int,
) -> np.ndarray:
    """
    First molecular REAL score.

    score_i = |H_ir| / (|H_ii - H_rr| + eps)

    The reference determinant itself receives +inf so it is always included.
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
    """
    Stable descending rank. Stable ordering gives deterministic tie behaviour.
    """
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
    """
    Matched NULL:
    permute REAL score values across determinant identities, preserving the score
    distribution exactly while destroying score-to-state correspondence.
    """
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
    probabilities: np.ndarray,
    target_mass: float = REFERENCE_IMPORTANCE_MASS,
) -> set[int]:
    order = np.argsort(-probabilities, kind="stable")
    cumulative = 0.0
    selected: list[int] = []

    for idx in order:
        selected.append(int(idx))
        cumulative += float(probabilities[idx])
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
    probabilities: np.ndarray,
    k: int,
) -> float:
    chosen = selected_order[:k]
    rel = probabilities[chosen]

    ideal_order = np.argsort(-probabilities, kind="stable")[:k]
    ideal_rel = probabilities[ideal_order]

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
    probabilities: np.ndarray,
    exact_energy: float,
    important_set: set[int],
    k: int,
) -> EvaluationRow:
    selected = np.asarray(ranking[:k], dtype=int)

    captured_mass = float(np.sum(probabilities[selected]))
    subspace_energy = selected_subspace_energy(hamiltonian, selected)
    energy_error = abs(subspace_energy - exact_energy)

    tp = len(set(map(int, selected)) & important_set)
    precision = tp / k
    recall = tp / len(important_set) if important_set else 0.0
    ndcg = ndcg_at_k(ranking, probabilities, k)
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


def aggregate_rows(rows: list[EvaluationRow]) -> list[dict[str, Any]]:
    """
    Aggregate by geometry / method / K.

    REAL has one deterministic row.
    NULL and RANDOM have 12 seed rows.
    """
    groups: dict[tuple[float, str, int], list[EvaluationRow]] = {}

    for row in rows:
        key = (row.geometry_angstrom, row.method, row.k)
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
    """
    Evaluate the preregistered G2-G4 gates using aggregate results.

    G1 is handled as hard runtime validation.
    G5 belongs primarily to later efficiency analysis (v50_5).
    G6 is represented here by full seed-set reporting.
    """
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
        both_metrics_real_vs_null = []
        both_metrics_real_vs_random = []

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

            both_metrics_real_vs_null.append(
                (real_mass > null_mass) and (real_err < null_err)
            )

            both_metrics_real_vs_random.append(
                (real_mass > random_mass) and (real_err < random_err)
            )

        n_null_wins = int(sum(both_metrics_real_vs_null))
        n_random_wins = int(sum(both_metrics_real_vs_random))

        per_k[str(k)] = {
            "geometries_REAL_beats_NULL_on_both_primary_metrics": n_null_wins,
            "geometries_REAL_beats_RANDOM_on_both_primary_metrics": n_random_wins,
            "g3_threshold_6_of_8_vs_null": n_null_wins >= 6,
            "g4_random_separation_same_k": n_random_wins >= 6,
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

    # G2: positive median REAL advantage across all geometries for BOTH primary metrics.
    # We compute geometry-level REAL - NULL median deltas, then median across geometries.
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

        median_mass_delta = float(np.median(mass_deltas))
        median_error_improvement = float(np.median(error_improvements))

        g2_by_k[str(k)] = {
            "median_REAL_minus_NULL_captured_mass": median_mass_delta,
            "median_NULL_minus_REAL_energy_error": median_error_improvement,
            "passes_both": (
                median_mass_delta > 0.0
                and median_error_improvement > 0.0
            ),
        }

    g2_pass_any_k = any(
        item["passes_both"] for item in g2_by_k.values()
    )

    return {
        "G1_validity": {
            "pass": True,
            "basis_sector_and_probability_checks": "passed during load/evaluation",
        },
        "G2_REAL_beats_matched_NULL_on_primary_metrics": {
            "pass_any_preregistered_k": g2_pass_any_k,
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
            "reason": (
                "v50_2 records K-budget performance, but the final preregistered "
                "reference-quality threshold is frozen/evaluated in the later efficiency stage."
            ),
        },
        "G6_reproducibility": {
            "status": "FULL_12_SEED_SET_REPORTED",
            "seeds": list(SEEDS),
            "note": (
                "No selective seed removal is performed. Formal conclusion stability "
                "is evaluated from the complete stored seed-level table."
            ),
        },
    }


def save_rows_csv(rows: list[EvaluationRow], path: Path) -> None:
    fieldnames = list(EvaluationRow.__dataclass_fields__.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 v50_2 REAL vs NULL determinant ranking."
    )

    v1_root = Path(__file__).resolve().parent.parent

    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=v1_root / "results" / "reference",
        help="Directory containing v50_1 reference NPZ/JSON files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=v1_root / "results" / "real_null",
        help="Directory for v50_2 outputs.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    reference_dir = args.reference_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SOFT SPACES — PHASE 5 REAL vs NULL DETERMINANT RANKING")
    print("=" * 78)
    print(f"Version       : {SCRIPT_VERSION}")
    print(f"Reference dir : {reference_dir}")
    print(f"Output dir    : {output_dir}")
    print(f"Geometries    : {GEOMETRIES}")
    print(f"K values      : {K_VALUES}")
    print(f"Seeds         : {len(SEEDS)}")
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
        probabilities = ref["probabilities"]
        exact_energy = float(ref["eigenvalues"][0])

        reference_local_index = choose_reference_determinant(hamiltonian)
        real_score = real_soft_spaces_score(
            hamiltonian,
            reference_local_index,
        )
        real_ranking = rank_descending(real_score)

        null_rankings = null_rankings_from_score(real_score, SEEDS)
        random_rankings_map = random_rankings(
            len(probabilities),
            SEEDS,
        )

        important_set = important_reference_set(probabilities)

        geometry_details[str(spacing)] = {
            "reference_local_index": reference_local_index,
            "reference_basis_index": int(
                ref["sector_indices"][reference_local_index]
            ),
            "important_reference_set_size_95pct_mass": len(important_set),
            "real_ranking_local_indices": [
                int(x) for x in real_ranking
            ],
            "real_score": [
                None if not np.isfinite(x) else float(x)
                for x in real_score
            ],
        }

        for k in K_VALUES:
            real_row = evaluate_ranking(
                spacing=spacing,
                method="REAL",
                seed=None,
                ranking=real_ranking,
                real_ranking=real_ranking,
                hamiltonian=hamiltonian,
                probabilities=probabilities,
                exact_energy=exact_energy,
                important_set=important_set,
                k=k,
            )
            all_rows.append(real_row)

            null_mass_values = []
            null_error_values = []

            random_mass_values = []
            random_error_values = []

            for seed in SEEDS:
                null_row = evaluate_ranking(
                    spacing=spacing,
                    method="NULL",
                    seed=seed,
                    ranking=null_rankings[seed],
                    real_ranking=real_ranking,
                    hamiltonian=hamiltonian,
                    probabilities=probabilities,
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
                    probabilities=probabilities,
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

                random_mass_values.append(
                    random_row.captured_probability_mass
                )
                random_error_values.append(
                    random_row.absolute_energy_error_hartree
                )

            print(
                f"    K={k:2d} | "
                f"REAL mass={real_row.captured_probability_mass:.6f} "
                f"vs NULL med={np.median(null_mass_values):.6f} | "
                f"REAL dE={real_row.absolute_energy_error_hartree:.6e} "
                f"vs NULL med={np.median(null_error_values):.6e}"
            )

    aggregate = aggregate_rows(all_rows)
    gates = build_gate_summary(aggregate)

    seed_csv = output_dir / "phase5_v50_2_seed_level_results.csv"
    aggregate_csv = output_dir / "phase5_v50_2_aggregate_results.csv"
    summary_json = output_dir / "phase5_v50_2_summary.json"
    hash_file = output_dir / "phase5_v50_2_summary.sha256"

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
        "methodology": {
            "real_score": (
                "|H_ir| / (|H_ii - H_rr| + eps), with r chosen as the "
                "minimum-diagonal-energy determinant"
            ),
            "real_ranking_uses_reference_ground_state_amplitudes": False,
            "reference_ground_state_used_for_evaluation_only": True,
            "null_control": (
                "Permutation of the REAL score values across determinant identities"
            ),
            "random_baseline": "Uniform random ranking of all 70 determinants",
            "k_values": list(K_VALUES),
            "seeds": list(SEEDS),
            "reference_importance_mass_threshold": REFERENCE_IMPORTANCE_MASS,
            "energy_denominator_epsilon": ENERGY_DENOM_EPS,
        },
        "geometry_details": geometry_details,
        "decision_gates": gates,
        "aggregate_results": aggregate,
        "interpretation_boundary": {
            "claim_scope": (
                "v50_2 tests a first molecular REAL-vs-NULL ranking proxy. "
                "It does not yet establish industrial utility or general molecular transfer."
            ),
            "g5_final_efficiency_claim": False,
            "secondary_molecule_transfer_test": False,
        },
    }

    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    digest = sha256_hex(summary)
    hash_file.write_text(digest + "\n", encoding="ascii")

    print("-" * 78)
    print("v50_2 COMPLETE")
    print("-" * 78)
    print(f"Seed-level CSV : {seed_csv}")
    print(f"Aggregate CSV  : {aggregate_csv}")
    print(f"Summary JSON   : {summary_json}")
    print(f"SHA-256        : {digest}")
    print("-" * 78)

    g2 = gates["G2_REAL_beats_matched_NULL_on_primary_metrics"]
    g3 = gates["G3_advantage_not_single_geometry"]
    g4 = gates["G4_random_baseline_separation"]

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
        "Do not interpret PASS/FAIL beyond this preregistered first molecular "
        "ranking experiment."
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
