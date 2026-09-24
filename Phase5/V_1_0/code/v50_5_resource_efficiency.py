#!/usr/bin/env python3
"""
Soft Spaces Phase 5 — v50_5 resource / efficiency analysis

Purpose
-------
Evaluate whether the frozen Soft-Spaces REAL ranking offers any practical
resource advantage relative to the EN2-like perturbative baseline.

This script DOES NOT alter the ranking formulas.

It reuses the same corrected H4/STO-3G sector and asks:

    If REAL and EN2 achieve similar ranking quality, do they require
    different computational resources to construct and use?

Frozen system
-------------
H4 / STO-3G
N_alpha = 2
N_beta  = 2
candidate dimension = 36
geometries = 0.75 ... 3.00 Å
K = 2, 4, 8, 12, 16

REAL score:
    |H_ir| / (|H_ii - H_rr| + eps)

EN2 score:
    |H_ir|^2 / (|H_ii - H_rr| + eps)

Reference:
    minimum diagonal Hamiltonian energy inside corrected sector

Primary efficiency questions
----------------------------
1. How many Hamiltonian elements are required to construct REAL?
2. How many Hamiltonian elements are required to construct EN2?
3. How many candidate-level arithmetic evaluations are required?
4. At each K, how much selected-subspace work is avoided relative to using
   the full 36-determinant corrected sector?
5. What accuracy is achieved at that reduced K?
6. Does either method reach a chemical-accuracy-like energy threshold
   with a smaller K?

Important limitation
--------------------
REAL and EN2, as currently defined, both depend on:
    - all diagonal H_ii terms
    - all couplings H_ir to one reference determinant

Therefore their raw Hamiltonian-access cost is expected to be very similar.
v50_5 measures this explicitly rather than assuming a resource advantage.

Energy-quality threshold
------------------------
The script uses:
    1 kcal/mol = 0.001593601437 Hartree

This is used only as an engineering-quality threshold in the efficiency
summary. It does not retroactively modify earlier Phase 5 gates.

Usage
-----
    python -X utf8 -u ./v50_5_resource_efficiency.py
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


SCRIPT_VERSION = "v50_5"
PROJECT = "Molecular Quantum Soft Spaces"
PHASE = "Phase 5"

GEOMETRIES = (0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
K_VALUES = (2, 4, 8, 12, 16)

NUM_SPATIAL_ORBITALS = 4
NUM_ALPHA = 2
NUM_BETA = 2
FULL_N4_DIMENSION = 70
CORRECTED_SECTOR_DIMENSION = math.comb(4, 2) * math.comb(4, 2)

ENERGY_DENOM_EPS = 1e-9
CHEMICAL_ACCURACY_HARTREE = 0.001593601437


@dataclass(frozen=True)
class ResourceProfile:
    method: str
    candidate_dimension: int
    diagonal_elements_required: int
    reference_couplings_required: int
    total_hamiltonian_elements_required: int
    score_multiplications_per_candidate: int
    score_divisions_per_candidate: int
    score_absolute_values_per_candidate: int
    score_power_operations_per_candidate: int
    total_candidate_score_evaluations: int


@dataclass(frozen=True)
class EfficiencyRow:
    geometry_angstrom: float
    method: str
    k: int
    captured_probability_mass: float
    absolute_energy_error_hartree: float
    reaches_chemical_accuracy: bool
    selected_fraction_of_sector: float
    determinant_reduction_fraction: float
    dense_subspace_dimension: int
    dense_subspace_elements: int
    full_sector_dense_elements: int
    dense_element_reduction_fraction: float


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
            "Corrected sector dimension mismatch: "
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
    full_p = np.asarray(data["ground_state_probabilities"], dtype=float)

    if len(full_basis) != FULL_N4_DIMENSION:
        raise RuntimeError(
            f"Unexpected full N=4 dimension at d={spacing}: {len(full_basis)}"
        )

    keep = corrected_sector_local_indices(full_basis)

    basis = full_basis[keep]
    h = full_h[np.ix_(keep, keep)]
    p = full_p[keep]

    retained = float(np.sum(p))

    if not math.isclose(
        retained,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"Corrected sector probability mass != 1 at d={spacing}: {retained}"
        )

    p = p / retained

    evals = np.linalg.eigvalsh(h)
    exact_energy = float(np.real(evals[0]))

    return {
        "basis_indices": basis,
        "hamiltonian": h,
        "probabilities": p,
        "exact_energy": exact_energy,
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
    return np.argsort(-safe, kind="stable")


def selected_subspace_energy(
    h: np.ndarray,
    selected: np.ndarray,
) -> float:
    sub_h = h[np.ix_(selected, selected)]
    evals = np.linalg.eigvalsh(sub_h)
    return float(np.real(evals[0]))


def resource_profiles() -> list[ResourceProfile]:
    n = CORRECTED_SECTOR_DIMENSION

    # Both methods need the same Hamiltonian access pattern:
    # all H_ii plus all H_ir to one reference determinant.
    # H_rr is counted in both sets conceptually but stored once in total.
    total_unique = n + (n - 1)

    real = ResourceProfile(
        method="REAL",
        candidate_dimension=n,
        diagonal_elements_required=n,
        reference_couplings_required=n - 1,
        total_hamiltonian_elements_required=total_unique,
        score_multiplications_per_candidate=0,
        score_divisions_per_candidate=1,
        score_absolute_values_per_candidate=3,
        score_power_operations_per_candidate=0,
        total_candidate_score_evaluations=n,
    )

    en2 = ResourceProfile(
        method="EN2",
        candidate_dimension=n,
        diagonal_elements_required=n,
        reference_couplings_required=n - 1,
        total_hamiltonian_elements_required=total_unique,
        score_multiplications_per_candidate=1,
        score_divisions_per_candidate=1,
        score_absolute_values_per_candidate=3,
        score_power_operations_per_candidate=1,
        total_candidate_score_evaluations=n,
    )

    return [real, en2]


def evaluate_method(
    spacing: float,
    method: str,
    ranking: np.ndarray,
    h: np.ndarray,
    probabilities: np.ndarray,
    exact_energy: float,
) -> list[EfficiencyRow]:
    rows = []

    full_elements = CORRECTED_SECTOR_DIMENSION ** 2

    for k in K_VALUES:
        selected = np.asarray(ranking[:k], dtype=int)

        captured_mass = float(
            np.sum(probabilities[selected])
        )

        subspace_energy = selected_subspace_energy(
            h,
            selected,
        )

        error = abs(subspace_energy - exact_energy)

        selected_fraction = k / CORRECTED_SECTOR_DIMENSION
        determinant_reduction = 1.0 - selected_fraction

        dense_elements = k * k
        dense_reduction = 1.0 - (dense_elements / full_elements)

        rows.append(
            EfficiencyRow(
                geometry_angstrom=float(spacing),
                method=method,
                k=int(k),
                captured_probability_mass=captured_mass,
                absolute_energy_error_hartree=float(error),
                reaches_chemical_accuracy=(
                    error <= CHEMICAL_ACCURACY_HARTREE
                ),
                selected_fraction_of_sector=float(selected_fraction),
                determinant_reduction_fraction=float(
                    determinant_reduction
                ),
                dense_subspace_dimension=int(k),
                dense_subspace_elements=int(dense_elements),
                full_sector_dense_elements=int(full_elements),
                dense_element_reduction_fraction=float(
                    dense_reduction
                ),
            )
        )

    return rows


def minimum_k_reaching_threshold(
    rows: list[EfficiencyRow],
    method: str,
    geometry: float,
) -> int | None:
    candidates = [
        row.k
        for row in rows
        if row.method == method
        and row.geometry_angstrom == geometry
        and row.reaches_chemical_accuracy
    ]

    if not candidates:
        return None

    return min(candidates)


def summarize_thresholds(
    rows: list[EfficiencyRow],
) -> dict[str, Any]:
    summary = {}

    for method in ("REAL", "EN2"):
        per_geometry = {}
        reached = 0

        for spacing in GEOMETRIES:
            k = minimum_k_reaching_threshold(
                rows,
                method,
                spacing,
            )

            per_geometry[str(spacing)] = k

            if k is not None:
                reached += 1

        valid_k = [
            k
            for k in per_geometry.values()
            if k is not None
        ]

        summary[method] = {
            "geometries_reaching_threshold": reached,
            "minimum_k_by_geometry": per_geometry,
            "median_minimum_k_where_reached": (
                float(np.median(valid_k))
                if valid_k else None
            ),
        }

    return summary


def save_dataclass_csv(
    rows: list[Any],
    path: Path,
) -> None:
    if not rows:
        raise RuntimeError(f"No rows to save for {path}")

    fields = list(rows[0].__dataclass_fields__.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 v50_5 resource / efficiency analysis."
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
        default=v1_root / "results" / "resource_efficiency",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    reference_dir = args.reference_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SOFT SPACES — PHASE 5 v50_5 RESOURCE / EFFICIENCY ANALYSIS")
    print("=" * 78)
    print(f"Version                  : {SCRIPT_VERSION}")
    print(f"Corrected dimension      : {CORRECTED_SECTOR_DIMENSION}")
    print(f"K values                 : {K_VALUES}")
    print(
        f"Chemical accuracy target : "
        f"{CHEMICAL_ACCURACY_HARTREE:.12f} Ha"
    )
    print("-" * 78)

    profiles = resource_profiles()

    for profile in profiles:
        print(
            f"{profile.method:4s} ranking cost: "
            f"H-elements={profile.total_hamiltonian_elements_required}, "
            f"candidate evaluations={profile.total_candidate_score_evaluations}"
        )

    print("-" * 78)

    efficiency_rows: list[EfficiencyRow] = []
    geometry_details: dict[str, Any] = {}

    for gi, spacing in enumerate(GEOMETRIES, start=1):
        print(
            f"[{gi}/{len(GEOMETRIES)}] d={spacing:.2f} Å",
            flush=True,
        )

        ref = load_reference(reference_dir, spacing)

        h = ref["hamiltonian"]
        probabilities = ref["probabilities"]
        exact_energy = ref["exact_energy"]

        reference_idx = choose_reference_determinant(h)

        real_order = rank_descending(
            real_score(h, reference_idx)
        )

        en2_order = rank_descending(
            en2_score(h, reference_idx)
        )

        real_rows = evaluate_method(
            spacing,
            "REAL",
            real_order,
            h,
            probabilities,
            exact_energy,
        )

        en2_rows = evaluate_method(
            spacing,
            "EN2",
            en2_order,
            h,
            probabilities,
            exact_energy,
        )

        efficiency_rows.extend(real_rows)
        efficiency_rows.extend(en2_rows)

        per_k = {}

        for k in K_VALUES:
            real_row = next(
                row for row in real_rows if row.k == k
            )
            en2_row = next(
                row for row in en2_rows if row.k == k
            )

            per_k[str(k)] = {
                "REAL": asdict(real_row),
                "EN2": asdict(en2_row),
            }

            print(
                f"    K={k:2d} | "
                f"REAL dE={real_row.absolute_energy_error_hartree:.6e}, "
                f"mass={real_row.captured_probability_mass:.6f}, "
                f"chem={'YES' if real_row.reaches_chemical_accuracy else 'NO '} | "
                f"EN2 dE={en2_row.absolute_energy_error_hartree:.6e}, "
                f"mass={en2_row.captured_probability_mass:.6f}, "
                f"chem={'YES' if en2_row.reaches_chemical_accuracy else 'NO '}"
            )

        geometry_details[str(spacing)] = {
            "reference_local_index": int(reference_idx),
            "per_k": per_k,
        }

    threshold_summary = summarize_thresholds(
        efficiency_rows
    )

    # Structural resource comparison
    real_profile = next(
        p for p in profiles if p.method == "REAL"
    )
    en2_profile = next(
        p for p in profiles if p.method == "EN2"
    )

    ranking_cost_comparison = {
        "same_hamiltonian_access_count": (
            real_profile.total_hamiltonian_elements_required
            == en2_profile.total_hamiltonian_elements_required
        ),
        "REAL_hamiltonian_elements": (
            real_profile.total_hamiltonian_elements_required
        ),
        "EN2_hamiltonian_elements": (
            en2_profile.total_hamiltonian_elements_required
        ),
        "REAL_candidate_evaluations": (
            real_profile.total_candidate_score_evaluations
        ),
        "EN2_candidate_evaluations": (
            en2_profile.total_candidate_score_evaluations
        ),
        "interpretation": (
            "As currently defined, REAL and EN2 require the same Hamiltonian "
            "access pattern and the same number of candidate score evaluations. "
            "REAL uses slightly simpler arithmetic, but v50_5 does not treat "
            "that as a meaningful asymptotic resource advantage."
        ),
    }

    efficiency_csv = (
        output_dir
        / "phase5_v50_5_efficiency_results.csv"
    )

    profiles_json = (
        output_dir
        / "phase5_v50_5_resource_profiles.json"
    )

    summary_json = (
        output_dir
        / "phase5_v50_5_summary.json"
    )

    hash_file = (
        output_dir
        / "phase5_v50_5_summary.sha256"
    )

    save_dataclass_csv(
        efficiency_rows,
        efficiency_csv,
    )

    profiles_json.write_text(
        json.dumps(
            [asdict(p) for p in profiles],
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    summary = {
        "_metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script_version": SCRIPT_VERSION,
            "python_version": sys.version.split()[0],
        },
        "project": PROJECT,
        "phase": PHASE,
        "frozen_methodology": {
            "candidate_dimension": CORRECTED_SECTOR_DIMENSION,
            "geometries": list(GEOMETRIES),
            "k_values": list(K_VALUES),
            "real_score": (
                "|H_ir| / (|H_ii - H_rr| + eps)"
            ),
            "en2_score": (
                "|H_ir|^2 / (|H_ii - H_rr| + eps)"
            ),
            "chemical_accuracy_hartree": (
                CHEMICAL_ACCURACY_HARTREE
            ),
            "ranking_formulas_changed": False,
        },
        "resource_profiles": [
            asdict(profile) for profile in profiles
        ],
        "ranking_cost_comparison": ranking_cost_comparison,
        "chemical_accuracy_summary": threshold_summary,
        "geometry_details": geometry_details,
        "interpretation_boundary": {
            "claim": (
                "v50_5 tests whether REAL has a practical resource advantage "
                "over EN2 under the current implementation."
            ),
            "important_limit": (
                "Both current scores require the same diagonal and reference-"
                "coupling Hamiltonian data. Any future resource advantage "
                "would require a different sparse/local estimator or search "
                "strategy, which is not introduced here."
            ),
            "industrial_utility_claim": False,
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
    print("v50_5 COMPLETE")
    print("-" * 78)
    print(f"Efficiency CSV : {efficiency_csv}")
    print(f"Profiles JSON  : {profiles_json}")
    print(f"Summary JSON   : {summary_json}")
    print(f"SHA-256        : {digest}")
    print("-" * 78)

    print("Resource comparison")
    print(
        f"    REAL Hamiltonian elements : "
        f"{real_profile.total_hamiltonian_elements_required}"
    )
    print(
        f"    EN2  Hamiltonian elements : "
        f"{en2_profile.total_hamiltonian_elements_required}"
    )
    print(
        f"    Same Hamiltonian access   : "
        f"{ranking_cost_comparison['same_hamiltonian_access_count']}"
    )

    print("-" * 78)
    print("Chemical-accuracy threshold summary")

    for method in ("REAL", "EN2"):
        item = threshold_summary[method]

        print(
            f"    {method:4s}: "
            f"{item['geometries_reaching_threshold']}/8 geometries "
            f"reach <= {CHEMICAL_ACCURACY_HARTREE:.6e} Ha"
        )

        print(
            f"          minimum K by geometry: "
            f"{item['minimum_k_by_geometry']}"
        )

    print("-" * 78)
    print(
        "v50_5 changes no ranking formula. It measures efficiency only."
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
