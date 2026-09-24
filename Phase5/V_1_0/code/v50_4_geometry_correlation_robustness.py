#!/usr/bin/env python3
"""
Soft Spaces Phase 5 — v50_4 geometry and correlation-regime robustness

Purpose
-------
Diagnose how the frozen Soft-Spaces REAL ranking behaves across the H4/STO-3G
geometry scan as the molecular ground state changes from weakly correlated to
more strongly multiconfigurational.

This script specifically addresses the key question raised after v50_3:

    Is REAL merely reproducing the same ranking as an EN2-like perturbative
    baseline, or do the two methods diverge in identifiable correlation regimes?

No ranking rule is tuned in v50_4.

Frozen definitions
------------------
System:
    H4 / STO-3G
    N_alpha = 2
    N_beta  = 2
    candidate dimension = 36

Geometries:
    0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00 Å

K:
    2, 4, 8, 12, 16

REAL:
    score_i = |H_ir| / (|H_ii - H_rr| + eps)

EN2:
    score_i = |H_ir|^2 / (|H_ii - H_rr| + eps)

Reference determinant:
    minimum diagonal Hamiltonian energy inside the corrected sector.

Ground-state amplitudes/probabilities are used ONLY for evaluation and for
describing the correlation regime.

Main outputs
------------
For every geometry:
- max reference determinant probability
- participation ratio
- Shannon entropy
- effective support at 95% probability mass
- REAL vs EN2 Spearman rank correlation
- REAL vs EN2 Kendall tau-a
- top-K Jaccard overlap
- top-K exact-order agreement
- differences in captured probability mass
- differences in selected-subspace energy error
- which method is better on each primary metric
- whether disagreement grows with stronger correlation

Interpretation
--------------
v50_4 is diagnostic. It does not introduce a new PASS/FAIL gate and does not
change any preregistered Phase 5 threshold.

Usage
-----
    python -X utf8 -u ./v50_4_geometry_correlation_robustness.py
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


SCRIPT_VERSION = "v50_4"
PROJECT = "Molecular Quantum Soft Spaces"
PHASE = "Phase 5"

GEOMETRIES = (0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
K_VALUES = (2, 4, 8, 12, 16)

NUM_SPATIAL_ORBITALS = 4
NUM_ALPHA = 2
NUM_BETA = 2
FULL_N4_DIMENSION = 70
CORRECTED_SECTOR_DIMENSION = math.comb(4, 2) * math.comb(4, 2)

REFERENCE_IMPORTANCE_MASS = 0.95
ENERGY_DENOM_EPS = 1e-9


@dataclass(frozen=True)
class GeometryRegimeRow:
    geometry_angstrom: float
    max_probability: float
    participation_ratio: float
    shannon_entropy_nats: float
    support_size_95pct_mass: int
    reference_basis_index: int
    real_en2_spearman: float
    real_en2_kendall_tau_a: float


@dataclass(frozen=True)
class TopKComparisonRow:
    geometry_angstrom: float
    k: int
    topk_jaccard: float
    topk_overlap_count: int
    exact_order_match_count: int
    exact_order_match_fraction: float
    real_captured_mass: float
    en2_captured_mass: float
    real_minus_en2_mass: float
    real_energy_error_hartree: float
    en2_energy_error_hartree: float
    en2_minus_real_error: float
    mass_winner: str
    energy_winner: str
    both_primary_winner: str


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

    retained_mass = float(np.sum(p))

    if not math.isclose(
        retained_mass,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"Corrected sector probability mass != 1 at d={spacing}: "
            f"{retained_mass}"
        )

    p = p / retained_mass

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


def rank_positions(order: np.ndarray) -> np.ndarray:
    positions = np.empty(len(order), dtype=int)
    for pos, idx in enumerate(order):
        positions[int(idx)] = pos
    return positions


def spearman_from_orders(
    order_a: np.ndarray,
    order_b: np.ndarray,
) -> float:
    """
    Spearman correlation over complete deterministic rankings.
    """
    ra = rank_positions(order_a).astype(float)
    rb = rank_positions(order_b).astype(float)

    ra -= np.mean(ra)
    rb -= np.mean(rb)

    denom = math.sqrt(
        float(np.sum(ra * ra)) * float(np.sum(rb * rb))
    )

    if denom == 0.0:
        return 1.0

    return float(np.sum(ra * rb) / denom)


def kendall_tau_a_from_orders(
    order_a: np.ndarray,
    order_b: np.ndarray,
) -> float:
    """
    Kendall tau-a for two complete orderings without tie correction.
    """
    ra = rank_positions(order_a)
    rb = rank_positions(order_b)

    concordant = 0
    discordant = 0

    n = len(order_a)

    for i in range(n):
        for j in range(i + 1, n):
            da = ra[i] - ra[j]
            db = rb[i] - rb[j]

            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1

    total = concordant + discordant

    if total == 0:
        return 1.0

    return (concordant - discordant) / total


def participation_ratio(probabilities: np.ndarray) -> float:
    denom = float(np.sum(probabilities ** 2))

    if denom <= 0.0:
        return float("nan")

    return 1.0 / denom


def shannon_entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]

    if len(positive) == 0:
        return 0.0

    return float(-np.sum(positive * np.log(positive)))


def support_size_for_mass(
    probabilities: np.ndarray,
    target_mass: float = REFERENCE_IMPORTANCE_MASS,
) -> int:
    order = np.argsort(-probabilities, kind="stable")
    cumulative = 0.0

    for count, idx in enumerate(order, start=1):
        cumulative += float(probabilities[idx])
        if cumulative >= target_mass:
            return count

    return len(probabilities)


def selected_subspace_energy(
    h: np.ndarray,
    selected: np.ndarray,
) -> float:
    sub_h = h[np.ix_(selected, selected)]
    evals = np.linalg.eigvalsh(sub_h)
    return float(np.real(evals[0]))


def topk_metrics(
    real_order: np.ndarray,
    en2_order: np.ndarray,
    probabilities: np.ndarray,
    h: np.ndarray,
    exact_energy: float,
    k: int,
) -> TopKComparisonRow:
    real_top = np.asarray(real_order[:k], dtype=int)
    en2_top = np.asarray(en2_order[:k], dtype=int)

    real_set = set(map(int, real_top))
    en2_set = set(map(int, en2_top))

    overlap_count = len(real_set & en2_set)
    union_count = len(real_set | en2_set)
    jaccard = overlap_count / union_count if union_count else 1.0

    order_matches = int(np.sum(real_top == en2_top))
    order_match_fraction = order_matches / k

    real_mass = float(np.sum(probabilities[real_top]))
    en2_mass = float(np.sum(probabilities[en2_top]))

    real_energy = selected_subspace_energy(h, real_top)
    en2_energy = selected_subspace_energy(h, en2_top)

    real_error = abs(real_energy - exact_energy)
    en2_error = abs(en2_energy - exact_energy)

    mass_delta = real_mass - en2_mass
    error_improvement = en2_error - real_error

    tol = 1e-12

    if mass_delta > tol:
        mass_winner = "REAL"
    elif mass_delta < -tol:
        mass_winner = "EN2"
    else:
        mass_winner = "TIE"

    if error_improvement > tol:
        energy_winner = "REAL"
    elif error_improvement < -tol:
        energy_winner = "EN2"
    else:
        energy_winner = "TIE"

    if mass_winner == "REAL" and energy_winner == "REAL":
        both = "REAL"
    elif mass_winner == "EN2" and energy_winner == "EN2":
        both = "EN2"
    elif mass_winner == "TIE" and energy_winner == "TIE":
        both = "TIE"
    else:
        both = "MIXED"

    return TopKComparisonRow(
        geometry_angstrom=0.0,  # overwritten by caller
        k=int(k),
        topk_jaccard=float(jaccard),
        topk_overlap_count=int(overlap_count),
        exact_order_match_count=int(order_matches),
        exact_order_match_fraction=float(order_match_fraction),
        real_captured_mass=real_mass,
        en2_captured_mass=en2_mass,
        real_minus_en2_mass=float(mass_delta),
        real_energy_error_hartree=float(real_error),
        en2_energy_error_hartree=float(en2_error),
        en2_minus_real_error=float(error_improvement),
        mass_winner=mass_winner,
        energy_winner=energy_winner,
        both_primary_winner=both,
    )


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y) or len(x) < 2:
        return float("nan")

    xc = x - np.mean(x)
    yc = y - np.mean(y)

    denom = math.sqrt(
        float(np.sum(xc * xc)) * float(np.sum(yc * yc))
    )

    if denom == 0.0:
        return float("nan")

    return float(np.sum(xc * yc) / denom)


def save_dataclass_csv(rows: list[Any], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"No rows to save for {path}")

    fieldnames = list(rows[0].__dataclass_fields__.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5 v50_4 geometry/correlation robustness diagnostics."
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
        default=v1_root / "results" / "geometry_robustness",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    reference_dir = args.reference_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SOFT SPACES — PHASE 5 v50_4 GEOMETRY / CORRELATION ROBUSTNESS")
    print("=" * 78)
    print(f"Version             : {SCRIPT_VERSION}")
    print(f"Corrected dimension : {CORRECTED_SECTOR_DIMENSION}")
    print(f"N_alpha / N_beta    : {NUM_ALPHA} / {NUM_BETA}")
    print(f"K values            : {K_VALUES}")
    print("-" * 78)

    regime_rows: list[GeometryRegimeRow] = []
    topk_rows: list[TopKComparisonRow] = []
    geometry_details: dict[str, Any] = {}

    for gi, spacing in enumerate(GEOMETRIES, start=1):
        print(
            f"[{gi}/{len(GEOMETRIES)}] d={spacing:.2f} Å",
            flush=True,
        )

        ref = load_reference(reference_dir, spacing)

        basis = ref["basis_indices"]
        h = ref["hamiltonian"]
        probabilities = ref["probabilities"]
        exact_energy = ref["exact_energy"]

        reference_idx = choose_reference_determinant(h)
        reference_basis = int(basis[reference_idx])

        n_alpha, n_beta = alpha_beta_counts(reference_basis)

        if (n_alpha, n_beta) != (NUM_ALPHA, NUM_BETA):
            raise RuntimeError(
                f"Reference outside corrected sector at d={spacing}"
            )

        real_scores = real_score(h, reference_idx)
        en2_scores = en2_score(h, reference_idx)

        real_order = rank_descending(real_scores)
        en2_order = rank_descending(en2_scores)

        max_p = float(np.max(probabilities))
        pr = participation_ratio(probabilities)
        entropy = shannon_entropy(probabilities)
        support95 = support_size_for_mass(probabilities)

        rho = spearman_from_orders(real_order, en2_order)
        tau = kendall_tau_a_from_orders(real_order, en2_order)

        regime_rows.append(
            GeometryRegimeRow(
                geometry_angstrom=float(spacing),
                max_probability=max_p,
                participation_ratio=float(pr),
                shannon_entropy_nats=float(entropy),
                support_size_95pct_mass=int(support95),
                reference_basis_index=reference_basis,
                real_en2_spearman=float(rho),
                real_en2_kendall_tau_a=float(tau),
            )
        )

        per_k = {}

        for k in K_VALUES:
            row = topk_metrics(
                real_order,
                en2_order,
                probabilities,
                h,
                exact_energy,
                k,
            )

            row = TopKComparisonRow(
                geometry_angstrom=float(spacing),
                k=row.k,
                topk_jaccard=row.topk_jaccard,
                topk_overlap_count=row.topk_overlap_count,
                exact_order_match_count=row.exact_order_match_count,
                exact_order_match_fraction=row.exact_order_match_fraction,
                real_captured_mass=row.real_captured_mass,
                en2_captured_mass=row.en2_captured_mass,
                real_minus_en2_mass=row.real_minus_en2_mass,
                real_energy_error_hartree=row.real_energy_error_hartree,
                en2_energy_error_hartree=row.en2_energy_error_hartree,
                en2_minus_real_error=row.en2_minus_real_error,
                mass_winner=row.mass_winner,
                energy_winner=row.energy_winner,
                both_primary_winner=row.both_primary_winner,
            )

            topk_rows.append(row)

            per_k[str(k)] = asdict(row)

            print(
                f"    K={k:2d} | "
                f"Jaccard={row.topk_jaccard:.3f} | "
                f"Δmass REAL-EN2={row.real_minus_en2_mass:+.6f} | "
                f"Δerror EN2-REAL={row.en2_minus_real_error:+.6e} | "
                f"winner={row.both_primary_winner}"
            )

        print(
            f"    regime: maxP={max_p:.6f} | PR={pr:.3f} | "
            f"H={entropy:.3f} | support95={support95} | "
            f"rho={rho:.4f} | tau={tau:.4f}"
        )

        geometry_details[str(spacing)] = {
            "reference_basis_index": reference_basis,
            "reference_n_alpha": n_alpha,
            "reference_n_beta": n_beta,
            "max_probability": max_p,
            "participation_ratio": pr,
            "shannon_entropy_nats": entropy,
            "support_size_95pct_mass": support95,
            "real_en2_spearman": rho,
            "real_en2_kendall_tau_a": tau,
            "real_ranking_full_basis_indices": [
                int(basis[x]) for x in real_order
            ],
            "en2_ranking_full_basis_indices": [
                int(basis[x]) for x in en2_order
            ],
            "topk": per_k,
        }

    # Cross-geometry diagnostics:
    # Does stronger correlation associate with more REAL/EN2 divergence?
    pr_values = np.array(
        [r.participation_ratio for r in regime_rows],
        dtype=float,
    )

    entropy_values = np.array(
        [r.shannon_entropy_nats for r in regime_rows],
        dtype=float,
    )

    rho_values = np.array(
        [r.real_en2_spearman for r in regime_rows],
        dtype=float,
    )

    disagreement = 1.0 - rho_values

    cross_geometry = {
        "pearson_participation_ratio_vs_rank_disagreement": (
            pearson(pr_values, disagreement)
        ),
        "pearson_entropy_vs_rank_disagreement": (
            pearson(entropy_values, disagreement)
        ),
    }

    per_k_summary = {}

    for k in K_VALUES:
        rows_k = [row for row in topk_rows if row.k == k]

        real_wins = sum(
            1 for row in rows_k
            if row.both_primary_winner == "REAL"
        )

        en2_wins = sum(
            1 for row in rows_k
            if row.both_primary_winner == "EN2"
        )

        ties = sum(
            1 for row in rows_k
            if row.both_primary_winner == "TIE"
        )

        mixed = sum(
            1 for row in rows_k
            if row.both_primary_winner == "MIXED"
        )

        per_k_summary[str(k)] = {
            "REAL_both_primary_wins": real_wins,
            "EN2_both_primary_wins": en2_wins,
            "ties": ties,
            "mixed": mixed,
            "median_topk_jaccard": float(
                np.median([row.topk_jaccard for row in rows_k])
            ),
            "median_REAL_minus_EN2_captured_mass": float(
                np.median(
                    [row.real_minus_en2_mass for row in rows_k]
                )
            ),
            "median_EN2_minus_REAL_energy_error": float(
                np.median(
                    [row.en2_minus_real_error for row in rows_k]
                )
            ),
        }

    regime_csv = (
        output_dir
        / "phase5_v50_4_geometry_regimes.csv"
    )

    topk_csv = (
        output_dir
        / "phase5_v50_4_real_vs_en2_topk.csv"
    )

    summary_json = (
        output_dir
        / "phase5_v50_4_summary.json"
    )

    hash_file = (
        output_dir
        / "phase5_v50_4_summary.sha256"
    )

    save_dataclass_csv(regime_rows, regime_csv)
    save_dataclass_csv(topk_rows, topk_csv)

    summary = {
        "_metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script_version": SCRIPT_VERSION,
            "python_version": sys.version.split()[0],
        },
        "project": PROJECT,
        "phase": PHASE,
        "frozen_methodology": {
            "corrected_sector": "N_alpha=2, N_beta=2",
            "candidate_dimension": CORRECTED_SECTOR_DIMENSION,
            "geometries": list(GEOMETRIES),
            "k_values": list(K_VALUES),
            "real_score": (
                "|H_ir| / (|H_ii - H_rr| + eps)"
            ),
            "en2_score": (
                "|H_ir|^2 / (|H_ii - H_rr| + eps)"
            ),
            "reference_rule": (
                "minimum diagonal Hamiltonian energy inside corrected sector"
            ),
            "outcome_used_for_ranking": False,
        },
        "geometry_details": geometry_details,
        "per_k_summary": per_k_summary,
        "cross_geometry_diagnostics": cross_geometry,
        "interpretation_boundary": {
            "new_gate_added": False,
            "score_tuned": False,
            "purpose": (
                "Determine whether REAL and EN2 remain equivalent across "
                "weak-to-stronger correlation regimes, and localize any "
                "systematic divergence."
            ),
            "industrial_utility_claim": False,
            "transfer_claim": False,
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
    print("v50_4 COMPLETE")
    print("-" * 78)
    print(f"Geometry CSV : {regime_csv}")
    print(f"Top-K CSV    : {topk_csv}")
    print(f"Summary JSON : {summary_json}")
    print(f"SHA-256      : {digest}")
    print("-" * 78)

    print("REAL vs EN2 summary")

    for k in K_VALUES:
        item = per_k_summary[str(k)]

        print(
            f"    K={k:2d}: "
            f"REAL wins={item['REAL_both_primary_wins']}/8 | "
            f"EN2 wins={item['EN2_both_primary_wins']}/8 | "
            f"ties={item['ties']}/8 | "
            f"mixed={item['mixed']}/8 | "
            f"median Jaccard={item['median_topk_jaccard']:.3f}"
        )

    print("-" * 78)
    print(
        "Correlation-regime diagnostic: "
        f"PR vs disagreement r="
        f"{cross_geometry['pearson_participation_ratio_vs_rank_disagreement']:+.4f}"
    )
    print(
        "Correlation-regime diagnostic: "
        f"entropy vs disagreement r="
        f"{cross_geometry['pearson_entropy_vs_rank_disagreement']:+.4f}"
    )
    print("-" * 78)
    print(
        "v50_4 is diagnostic only. No Phase 5 gate or ranking formula was changed."
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
