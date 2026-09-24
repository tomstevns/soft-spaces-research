#!/usr/bin/env python3
"""
Soft Spaces Phase 5 — v50_3 chemistry-informed baseline comparison

Purpose
-------
Compare the frozen v50_2b REAL ranking against stronger, outcome-blind,
chemistry-informed baselines in the physically corrected H4/STO-3G sector:

    N_alpha = 2
    N_beta  = 2
    candidate dimension = 36

This is the next Phase 5 step after v50_2b.

Frozen from v50_2b
------------------
- H4 / STO-3G
- 8 geometries
- corrected N_alpha=2, N_beta=2 sector
- K = 2, 4, 8, 12, 16
- seeds 25043000 ... 25043011 for stochastic baselines
- REAL score:
      |H_ir| / (|H_ii - H_rr| + eps)
- reference determinant:
      minimum diagonal Hamiltonian energy inside the corrected sector
- ground-state amplitudes/probabilities used ONLY for evaluation
- primary metrics:
      captured probability mass
      absolute energy error after selected-subspace diagonalisation
- additional metrics:
      Precision@K
      Recall@K
      NDCG@K
      selection efficiency

Chemistry-informed baselines
----------------------------
1. CHEM_CISD
   A deterministic excitation-structured ranking:
       a) reference determinant first
       b) lower excitation rank from the reference first
       c) lower diagonal Hamiltonian energy first
       d) basis index tie-breaker

   This is intended as a simple chemistry-informed determinant-selection
   heuristic. It does not inspect the exact ground-state amplitudes.

2. CHEM_EN2
   An Epstein-Nesbet-like second-order importance proxy:
       score_i = |H_ir|^2 / (|H_ii - H_rr| + eps)

   The reference determinant is forced first.

3. RANDOM
   Uniform random ranking of the same 36 candidates, retained for continuity.

Interpretation boundary
-----------------------
v50_3 does NOT modify the Phase 5 preregistered gates. It adds a stronger
baseline challenge. The script reports descriptive comparison counts:

- how often REAL beats CHEM_CISD on BOTH primary metrics
- how often REAL beats CHEM_EN2 on BOTH primary metrics
- same comparison against RANDOM median

No post-hoc tuning of K, seeds or score definitions is performed here.

Usage
-----
    python -X utf8 -u ./v50_3_chemistry_baselines.py
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


SCRIPT_VERSION = "v50_3"
PROJECT = "Molecular Quantum Soft Spaces"
PHASE = "Phase 5"

GEOMETRIES = (0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
K_VALUES = (2, 4, 8, 12, 16)
SEEDS = tuple(range(25043000, 25043012))

NUM_SPATIAL_ORBITALS = 4
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


def alpha_beta_counts(basis_index: int) -> tuple[int, int]:
    alpha_mask = (1 << NUM_SPATIAL_ORBITALS) - 1
    alpha_bits = basis_index & alpha_mask
    beta_bits = (basis_index >> NUM_SPATIAL_ORBITALS) & alpha_mask
    return popcount(alpha_bits), popcount(beta_bits)


def corrected_sector_local_indices(
    sector_basis_indices: np.ndarray,
) -> np.ndarray:
    keep = []

    for local_idx, basis_index in enumerate(sector_basis_indices):
        n_alpha, n_beta = alpha_beta_counts(int(basis_index))
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

    full_basis = np.asarray(data["sector_indices"], dtype=np.int64)
    full_h = np.asarray(data["sector_hamiltonian"], dtype=np.complex128)
    full_p = np.asarray(
        data["ground_state_probabilities"],
        dtype=float,
    )

    if len(full_basis) != FULL_N4_DIMENSION:
        raise RuntimeError(
            f"Unexpected N=4 dimension at d={spacing}: {len(full_basis)}"
        )

    keep = corrected_sector_local_indices(full_basis)

    basis = full_basis[keep]
    h = full_h[np.ix_(keep, keep)]
    p_phys = full_p[keep]

    retained_mass = float(np.sum(p_phys))

    if not math.isclose(
        retained_mass,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"Corrected sector does not retain unit reference mass at "
            f"d={spacing}: {retained_mass}"
        )

    p_norm = p_phys / retained_mass

    evals = np.linalg.eigvalsh(h)
    exact_energy = float(np.real(evals[0]))

    return {
        "basis_indices": basis,
        "hamiltonian": h,
        "probabilities_physical": p_phys,
        "probabilities_normalized": p_norm,
        "exact_energy": exact_energy,
        "retained_mass": retained_mass,
    }


def choose_reference_determinant(h: np.ndarray) -> int:
    diagonal = np.real(np.diag(h))
    return int(np.argmin(diagonal))


def rank_descending(score: np.ndarray) -> np.ndarray:
    safe = np.nan_to_num(
        score,
        nan=-np.inf,
        posinf=np.finfo(float).max,
        neginf=-np.inf,
    )
    return np.argsort(-safe, kind="stable")


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
    """
    Epstein-Nesbet-like second-order importance proxy.
    """
    diagonal = np.real(np.diag(h))
    e_ref = float(diagonal[reference_idx])

    coupling_sq = np.abs(h[:, reference_idx]) ** 2
    denominator = np.abs(diagonal - e_ref) + ENERGY_DENOM_EPS

    score = coupling_sq / denominator
    score = np.asarray(score, dtype=float)
    score[reference_idx] = np.inf

    return score


def excitation_rank(
    basis_index: int,
    reference_basis_index: int,
) -> int:
    """
    Number of occupied orbitals replaced relative to reference.

    Since both determinants have equal particle number, Hamming distance is even.
    excitation rank = Hamming distance / 2.
    """
    hamming_distance = popcount(basis_index ^ reference_basis_index)

    if hamming_distance % 2 != 0:
        raise RuntimeError(
            "Unexpected odd Hamming distance between equal-particle determinants."
        )

    return hamming_distance // 2


def chem_cisd_ranking(
    h: np.ndarray,
    basis_indices: np.ndarray,
    reference_idx: int,
) -> np.ndarray:
    """
    Deterministic chemistry-informed baseline.

    Ordering:
    1. excitation rank from reference
    2. diagonal Hamiltonian energy
    3. full basis index

    The reference determinant is naturally first because rank=0.
    """
    diagonal = np.real(np.diag(h))
    ref_basis = int(basis_indices[reference_idx])

    records = []

    for local_idx, basis_index in enumerate(basis_indices):
        ex_rank = excitation_rank(
            int(basis_index),
            ref_basis,
        )

        records.append(
            (
                ex_rank,
                float(diagonal[local_idx]),
                int(basis_index),
                int(local_idx),
            )
        )

    records.sort()

    ranking = np.array(
        [record[3] for record in records],
        dtype=np.int64,
    )

    if int(ranking[0]) != reference_idx:
        raise RuntimeError(
            "CHEM_CISD ranking failed to place reference determinant first."
        )

    return ranking


def random_rankings(
    n_candidates: int,
    seeds: tuple[int, ...],
) -> dict[int, np.ndarray]:
    out = {}

    for seed in seeds:
        rng = np.random.default_rng(seed + 10_000_000)
        out[seed] = rng.permutation(n_candidates)

    return out


def important_reference_set(
    probabilities_normalized: np.ndarray,
) -> set[int]:
    order = np.argsort(
        -probabilities_normalized,
        kind="stable",
    )

    selected = []
    cumulative = 0.0

    for idx in order:
        selected.append(int(idx))
        cumulative += float(probabilities_normalized[idx])

        if cumulative >= REFERENCE_IMPORTANCE_MASS:
            break

    return set(selected)


def dcg(relevances: np.ndarray) -> float:
    if len(relevances) == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(relevances) + 2))
    gains = (2.0 ** relevances) - 1.0

    return float(np.sum(gains / discounts))


def ndcg_at_k(
    ranking: np.ndarray,
    probabilities_normalized: np.ndarray,
    k: int,
) -> float:
    chosen = ranking[:k]
    rel = probabilities_normalized[chosen]

    ideal = np.argsort(
        -probabilities_normalized,
        kind="stable",
    )[:k]

    ideal_rel = probabilities_normalized[ideal]
    denom = dcg(ideal_rel)

    if denom <= 0.0:
        return 0.0

    return dcg(rel) / denom


def selected_subspace_energy(
    h: np.ndarray,
    selected: np.ndarray,
) -> float:
    sub_h = h[np.ix_(selected, selected)]
    evals = np.linalg.eigvalsh(sub_h)
    return float(np.real(evals[0]))


def evaluate_ranking(
    spacing: float,
    method: str,
    seed: int | None,
    ranking: np.ndarray,
    real_ranking: np.ndarray,
    h: np.ndarray,
    p_phys: np.ndarray,
    p_norm: np.ndarray,
    exact_energy: float,
    important_set: set[int],
    k: int,
) -> EvaluationRow:
    selected = np.asarray(ranking[:k], dtype=int)

    captured_mass = float(np.sum(p_phys[selected]))

    selected_energy = selected_subspace_energy(
        h,
        selected,
    )

    error = abs(selected_energy - exact_energy)

    selected_set = set(map(int, selected))
    tp = len(selected_set & important_set)

    precision = tp / k
    recall = tp / len(important_set) if important_set else 0.0

    ndcg = ndcg_at_k(
        ranking,
        p_norm,
        k,
    )

    efficiency = captured_mass / k

    overlap = None
    if method != "REAL":
        overlap = len(
            selected_set
            & set(map(int, real_ranking[:k]))
        )

    return EvaluationRow(
        geometry_angstrom=float(spacing),
        method=method,
        seed=seed,
        k=int(k),
        captured_probability_mass=captured_mass,
        absolute_energy_error_hartree=float(error),
        selected_subspace_energy_hartree=float(selected_energy),
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


def compare_methods(
    aggregate: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Descriptive comparison only; this does not redefine preregistered gates.
    """
    index = {
        (
            float(r["geometry_angstrom"]),
            str(r["method"]),
            int(r["k"]),
        ): r
        for r in aggregate
    }

    comparisons = {}

    baselines = (
        "CHEM_CISD",
        "CHEM_EN2",
        "RANDOM",
    )

    for baseline in baselines:
        per_k = {}

        for k in K_VALUES:
            wins = 0
            mass_deltas = []
            error_improvements = []

            for spacing in GEOMETRIES:
                real = index[(spacing, "REAL", k)]
                base = index[(spacing, baseline, k)]

                real_mass = real[
                    "captured_probability_mass_median"
                ]
                base_mass = base[
                    "captured_probability_mass_median"
                ]

                real_err = real[
                    "absolute_energy_error_hartree_median"
                ]
                base_err = base[
                    "absolute_energy_error_hartree_median"
                ]

                mass_delta = real_mass - base_mass
                error_improvement = base_err - real_err

                mass_deltas.append(mass_delta)
                error_improvements.append(error_improvement)

                if mass_delta > 0.0 and error_improvement > 0.0:
                    wins += 1

            per_k[str(k)] = {
                "geometries_REAL_beats_baseline_on_both_primary_metrics": wins,
                "median_REAL_minus_baseline_captured_mass": float(
                    np.median(mass_deltas)
                ),
                "median_baseline_minus_REAL_energy_error": float(
                    np.median(error_improvements)
                ),
            }

        comparisons[baseline] = per_k

    return comparisons


def save_rows_csv(
    rows: list[EvaluationRow],
    path: Path,
) -> None:
    fields = list(EvaluationRow.__dataclass_fields__.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def save_aggregate_csv(
    aggregate: list[dict[str, Any]],
    path: Path,
) -> None:
    if not aggregate:
        raise RuntimeError("No aggregate results.")

    fields = list(aggregate[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(aggregate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5 v50_3 chemistry-informed baseline comparison."
        )
    )

    v1_root = Path(__file__).resolve().parent.parent

    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=v1_root / "results" / "reference",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=v1_root / "results" / "chemistry_baselines",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    reference_dir = args.reference_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SOFT SPACES — PHASE 5 v50_3 CHEMISTRY-INFORMED BASELINES")
    print("=" * 78)
    print(f"Version             : {SCRIPT_VERSION}")
    print(f"Corrected dimension : {CORRECTED_SECTOR_DIMENSION}")
    print(f"N_alpha / N_beta    : {NUM_ALPHA} / {NUM_BETA}")
    print(f"K values            : {K_VALUES}")
    print(f"Random seeds        : {len(SEEDS)}")
    print("-" * 78)

    all_rows: list[EvaluationRow] = []
    geometry_details: dict[str, Any] = {}

    for gi, spacing in enumerate(GEOMETRIES, start=1):
        print(
            f"[{gi}/{len(GEOMETRIES)}] d={spacing:.2f} Å",
            flush=True,
        )

        ref = load_reference(reference_dir, spacing)

        basis = ref["basis_indices"]
        h = ref["hamiltonian"]
        p_phys = ref["probabilities_physical"]
        p_norm = ref["probabilities_normalized"]
        exact_energy = ref["exact_energy"]

        reference_idx = choose_reference_determinant(h)
        reference_basis = int(basis[reference_idx])

        n_alpha, n_beta = alpha_beta_counts(reference_basis)

        if (n_alpha, n_beta) != (NUM_ALPHA, NUM_BETA):
            raise RuntimeError(
                "Reference determinant outside corrected sector."
            )

        real_scores = real_score(
            h,
            reference_idx,
        )
        real_ranking = rank_descending(real_scores)

        en2_scores = en2_score(
            h,
            reference_idx,
        )
        en2_ranking = rank_descending(en2_scores)

        cisd_ranking = chem_cisd_ranking(
            h,
            basis,
            reference_idx,
        )

        random_map = random_rankings(
            len(basis),
            SEEDS,
        )

        important_set = important_reference_set(p_norm)

        excitation_counts = {}
        for basis_index in basis:
            rank = excitation_rank(
                int(basis_index),
                reference_basis,
            )
            excitation_counts[str(rank)] = (
                excitation_counts.get(str(rank), 0) + 1
            )

        geometry_details[str(spacing)] = {
            "reference_local_index": reference_idx,
            "reference_basis_index": reference_basis,
            "reference_n_alpha": n_alpha,
            "reference_n_beta": n_beta,
            "retained_reference_probability_mass": (
                ref["retained_mass"]
            ),
            "important_reference_set_size_95pct_mass": len(
                important_set
            ),
            "excitation_rank_counts": excitation_counts,
            "real_ranking_full_basis_indices": [
                int(basis[x]) for x in real_ranking
            ],
            "chem_cisd_ranking_full_basis_indices": [
                int(basis[x]) for x in cisd_ranking
            ],
            "chem_en2_ranking_full_basis_indices": [
                int(basis[x]) for x in en2_ranking
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
                p_phys,
                p_norm,
                exact_energy,
                important_set,
                k,
            )
            all_rows.append(real_row)

            cisd_row = evaluate_ranking(
                spacing,
                "CHEM_CISD",
                None,
                cisd_ranking,
                real_ranking,
                h,
                p_phys,
                p_norm,
                exact_energy,
                important_set,
                k,
            )
            all_rows.append(cisd_row)

            en2_row = evaluate_ranking(
                spacing,
                "CHEM_EN2",
                None,
                en2_ranking,
                real_ranking,
                h,
                p_phys,
                p_norm,
                exact_energy,
                important_set,
                k,
            )
            all_rows.append(en2_row)

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
                    p_phys,
                    p_norm,
                    exact_energy,
                    important_set,
                    k,
                )

                all_rows.append(random_row)
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
                f"CISD mass={cisd_row.captured_probability_mass:.6f}, "
                f"dE={cisd_row.absolute_energy_error_hartree:.6e} | "
                f"EN2 mass={en2_row.captured_probability_mass:.6f}, "
                f"dE={en2_row.absolute_energy_error_hartree:.6e}"
            )

    aggregate = aggregate_rows(all_rows)
    comparisons = compare_methods(aggregate)

    seed_csv = (
        output_dir
        / "phase5_v50_3_seed_level_results.csv"
    )

    aggregate_csv = (
        output_dir
        / "phase5_v50_3_aggregate_results.csv"
    )

    summary_json = (
        output_dir
        / "phase5_v50_3_summary.json"
    )

    hash_file = (
        output_dir
        / "phase5_v50_3_summary.sha256"
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
        "frozen_from_v50_2b": {
            "corrected_sector": "N_alpha=2, N_beta=2",
            "candidate_dimension": CORRECTED_SECTOR_DIMENSION,
            "k_values": list(K_VALUES),
            "seeds": list(SEEDS),
            "real_score": (
                "|H_ir| / (|H_ii - H_rr| + eps)"
            ),
            "reference_rule": (
                "minimum diagonal Hamiltonian energy inside corrected sector"
            ),
            "outcome_used_for_ranking": False,
        },
        "baselines": {
            "CHEM_CISD": (
                "Reference first, then excitation rank, then diagonal energy, "
                "then basis-index tie-breaker."
            ),
            "CHEM_EN2": (
                "|H_ir|^2 / (|H_ii - H_rr| + eps), reference first."
            ),
            "RANDOM": (
                "Uniform random ranking over same 36 candidates."
            ),
        },
        "geometry_details": geometry_details,
        "descriptive_baseline_comparison": comparisons,
        "aggregate_results": aggregate,
        "interpretation_boundary": {
            "preregistered_gate_logic_modified": False,
            "new_claim": (
                "v50_3 evaluates whether frozen REAL remains competitive "
                "against chemistry-informed outcome-blind baselines."
            ),
            "industrial_utility_claim": False,
            "transfer_to_other_molecules_claim": False,
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
    print("v50_3 COMPLETE")
    print("-" * 78)
    print(f"Seed-level CSV : {seed_csv}")
    print(f"Aggregate CSV  : {aggregate_csv}")
    print(f"Summary JSON   : {summary_json}")
    print(f"SHA-256        : {digest}")
    print("-" * 78)

    for baseline in ("CHEM_CISD", "CHEM_EN2", "RANDOM"):
        print(f"REAL vs {baseline}")

        for k in K_VALUES:
            item = comparisons[baseline][str(k)]

            print(
                f"    K={k:2d}: "
                f"{item['geometries_REAL_beats_baseline_on_both_primary_metrics']}"
                f"/8 geometries | "
                f"median Δmass="
                f"{item['median_REAL_minus_baseline_captured_mass']:+.6f} | "
                f"median Δerror improvement="
                f"{item['median_baseline_minus_REAL_energy_error']:+.6e}"
            )

    print("-" * 78)
    print(
        "No preregistered Phase 5 gate was changed in v50_3."
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
