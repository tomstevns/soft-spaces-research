#!/usr/bin/env python3
"""
Soft Spaces Phase 5 — v50_8 EN2 predictability of Soft-Spaces hotspots

Purpose
-------
Test whether conventional EN2 determinant information can explain or predict
the DUAL-PASS Soft-Spaces eigenstate hotspots found by v50_7c.

Important object mismatch
-------------------------
Soft Spaces v50_7c labels ADJACENT EIGENSTATE PAIRS.

EN2 naturally ranks DETERMINANTS relative to a reference determinant.

Therefore this script does NOT pretend that EN2 has a canonical "hotspot
index" in eigenstate space.  Instead it uses two explicit, auditable bridge
metrics from EN2 determinant space to an eigenstate pair:

1) EN2_TOPK_SUPPORT(K)
   Fraction of the two eigenstates' total determinant probability that lies
   inside the EN2 top-K determinant set.

2) EN2_WEIGHTED_SCORE
   Probability-weighted average EN2 determinant score across the two
   eigenstates.

Question
--------
Do v50_7c DUAL-PASS eigenstate pairs receive systematically higher EN2 bridge
scores than non-PASS eligible pairs?

If YES:
    EN2 determinant structure substantially predicts the Soft-Spaces hotspots.

If NO:
    the original Soft-Spaces hotspot signal contains structure not captured
    by this EN2 determinant bridge.

This is a comparator/audit, not a claim that these bridge metrics are the only
or canonical way to map EN2 into eigenstate space.

Inputs
------
Default:
    ../results/phase23_method_bridge/phase5_v50_7c_summary.json

The script reconstructs the same H4/STO-3G fixed
(N_alpha,N_beta)=(2,2) Hamiltonians.

Frozen Phase-5 EN2 determinant score
------------------------------------
For determinant i and reference determinant r:

    EN2_i = |H_ir|^2 / (|H_ii - H_rr| + eps)

Reference determinant:
    determinant with minimum diagonal H_ii

The reference determinant itself is assigned +inf only for ranking so it is
always included as anchor; bridge score calculations use its finite score 0.

Outputs
-------
- per-pair CSV
- per-geometry summary CSV
- JSON summary + SHA256

Usage
-----
    python -X utf8 -u ./v50_8_en2_hotspot_predictability.py
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


SCRIPT_VERSION = "v50_8"

GEOMETRIES = (0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
K_VALUES = (2, 4, 8, 12, 16)
EN2_EPS = 1e-12


@dataclass(frozen=True)
class PairRow:
    geometry_angstrom: float
    eigenpair_index: int
    softspaces_dual_pass: bool
    softspaces_prominence_rate: float
    softspaces_robust_rate: float
    gap_hartree: float
    en2_weighted_score: float
    en2_top2_support: float
    en2_top4_support: float
    en2_top8_support: float
    en2_top12_support: float
    en2_top16_support: float


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


def alpha_beta_counts(index: int, n_spatial: int) -> tuple[int, int]:
    mask = (1 << n_spatial) - 1
    alpha = index & mask
    beta = (index >> n_spatial) & mask
    return popcount(alpha), popcount(beta)


def fixed_sector_basis(
    n_spatial: int,
    n_alpha: int,
    n_beta: int,
) -> np.ndarray:
    out = []
    for idx in range(1 << (2 * n_spatial)):
        na, nb = alpha_beta_counts(idx, n_spatial)
        if na == n_alpha and nb == n_beta:
            out.append(idx)

    expected = math.comb(n_spatial, n_alpha) * math.comb(n_spatial, n_beta)
    if len(out) != expected:
        raise RuntimeError(
            f"Sector dimension mismatch: expected {expected}, got {len(out)}"
        )

    return np.asarray(out, dtype=np.int64)


def build_h4(spacing: float) -> dict[str, Any]:
    try:
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        from qiskit_nature.units import DistanceUnit
    except Exception as exc:
        raise RuntimeError(
            "Qiskit Nature / PySCF imports failed. "
            "Run in the Phase-5 WSL environment."
        ) from exc

    zs = (-1.5 * spacing, -0.5 * spacing, 0.5 * spacing, 1.5 * spacing)
    atom = "; ".join(f"H 0.0 0.0 {z:.12f}" for z in zs)

    driver = PySCFDriver(
        atom=atom,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()

    n_spatial = int(problem.num_spatial_orbitals)
    n_alpha = int(problem.num_alpha)
    n_beta = int(problem.num_beta)

    if (n_spatial, n_alpha, n_beta) != (4, 2, 2):
        raise RuntimeError(
            f"Unexpected H4/STO-3G metadata: "
            f"n_spatial={n_spatial}, Nalpha={n_alpha}, Nbeta={n_beta}"
        )

    basis = fixed_sector_basis(n_spatial, n_alpha, n_beta)

    mapper = JordanWignerMapper()
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())

    full_sparse = qubit_op.to_matrix(sparse=True)
    sector_sparse = full_sparse[basis, :][:, basis]
    h = np.asarray(sector_sparse.toarray(), dtype=np.complex128)

    herm_err = float(np.max(np.abs(h - h.conjugate().T)))
    if herm_err > 1e-10:
        raise RuntimeError(f"Hamiltonian not Hermitian: {herm_err:.3e}")

    evals, evecs = np.linalg.eigh(h)

    return {
        "basis": basis,
        "hamiltonian": h,
        "evals": np.asarray(evals, dtype=float),
        "evecs": np.asarray(evecs, dtype=np.complex128),
    }


def en2_scores(h: np.ndarray) -> tuple[np.ndarray, int]:
    diag = np.real(np.diag(h)).astype(float)
    r = int(np.argmin(diag))
    hrr = float(diag[r])

    score = np.zeros(len(diag), dtype=float)

    for i in range(len(diag)):
        if i == r:
            score[i] = 0.0
            continue

        numerator = float(abs(h[i, r]) ** 2)
        denominator = abs(float(diag[i] - hrr)) + EN2_EPS
        score[i] = numerator / denominator

    return score, r


def topk_indices(scores: np.ndarray, reference: int, k: int) -> np.ndarray:
    ranking_score = scores.copy()
    ranking_score[reference] = np.inf
    order = np.argsort(-ranking_score, kind="stable")
    return np.asarray(order[:k], dtype=int)


def pair_probabilities(
    evecs: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Mean determinant probability distribution across eigenstates k and k+1.
    Sum = 1.
    """
    p1 = np.abs(evecs[:, k]) ** 2
    p2 = np.abs(evecs[:, k + 1]) ** 2
    pair = 0.5 * (p1 + p2)
    norm = float(np.sum(pair))
    if norm <= 0.0:
        raise RuntimeError("Invalid pair probability norm.")
    return pair / norm


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """
    Mann-Whitney / ROC-AUC without sklearn.
    Ties receive average ranks.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)

    if pos == 0 or neg == 0:
        return None

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]

    ranks = np.empty(len(scores), dtype=float)

    i = 0
    while i < len(scores):
        j = i + 1
        while (
            j < len(scores)
            and sorted_scores[j] == sorted_scores[i]
        ):
            j += 1

        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    u = rank_sum_pos - pos * (pos + 1) / 2.0
    return float(u / (pos * neg))


def median_or_none(x: list[float]) -> float | None:
    if not x:
        return None
    return float(np.median(np.asarray(x, dtype=float)))


def load_softspaces_labels(
    path: Path,
) -> dict[float, dict[int, dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    version = data.get("_metadata", {}).get("script_version")
    if version != "v50_7c":
        raise RuntimeError(
            f"Expected v50_7c JSON, got script_version={version!r}"
        )

    out: dict[float, dict[int, dict[str, Any]]] = {}

    geometry_details = data.get("geometry_details", {})

    for geometry_str, details in geometry_details.items():
        g = float(geometry_str)
        rows = {}

        # top_candidates does not contain every eligible candidate, but v50_7c
        # global output stores only PASS rows.  Therefore reconstruct PASS labels
        # from passing_candidates and mark all other adjacent pairs as non-PASS.
        passing = details.get("passing_candidates", [])

        for row in passing:
            k = int(row["eigenpair_index"])
            rows[k] = {
                "pass": True,
                "prominence_rate": float(
                    row["prominence_positive_rate"]
                ),
                "robust_rate": float(
                    row["robust_positive_rate"]
                ),
            }

        out[g] = rows

    return out


def save_csv(rows: list[PairRow], path: Path) -> None:
    fields = list(PairRow.__dataclass_fields__.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare v50_7c Soft-Spaces hotspots with EN2 determinant structure."
    )

    v1_root = Path(__file__).resolve().parent.parent

    parser.add_argument(
        "--softspaces-json",
        type=Path,
        default=(
            v1_root
            / "results"
            / "phase23_method_bridge"
            / "phase5_v50_7c_summary.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            v1_root
            / "results"
            / "en2_hotspot_predictability"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ss_path = args.softspaces_json.resolve()
    if not ss_path.exists():
        raise FileNotFoundError(
            f"Cannot find v50_7c summary JSON: {ss_path}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_by_geometry = load_softspaces_labels(ss_path)

    print("=" * 78)
    print("SOFT SPACES — PHASE 5 v50_8 EN2 HOTSPOT PREDICTABILITY")
    print("=" * 78)
    print(f"Soft-Spaces labels : {ss_path}")
    print("Molecule           : H4 / STO-3G")
    print("Sector             : (N_alpha,N_beta)=(2,2)")
    print(f"Geometries         : {GEOMETRIES}")
    print(f"EN2 top-K          : {K_VALUES}")
    print("-" * 78)
    print(
        "Comparator bridge: determinant EN2 -> eigenpair support. "
        "No claim of canonical EN2 eigenstate-hotspot definition."
    )
    print("-" * 78)

    all_rows: list[PairRow] = []
    geometry_summary = {}

    for gi, g in enumerate(GEOMETRIES, start=1):
        print(
            f"[{gi}/{len(GEOMETRIES)}] H4 geometry={g:.2f} Å",
            flush=True,
        )

        p = build_h4(g)
        h = p["hamiltonian"]
        evals = p["evals"]
        evecs = p["evecs"]

        scores, reference = en2_scores(h)

        topk = {
            k: topk_indices(scores, reference, k)
            for k in K_VALUES
        }

        ss_pass = labels_by_geometry.get(float(g), {})

        rows_g = []

        for k in range(len(evals) - 1):
            prob = pair_probabilities(evecs, k)

            weighted = float(np.sum(prob * scores))

            supports = {
                kk: float(np.sum(prob[topk[kk]]))
                for kk in K_VALUES
            }

            label = ss_pass.get(
                k,
                {
                    "pass": False,
                    "prominence_rate": 0.0,
                    "robust_rate": 0.0,
                },
            )

            row = PairRow(
                geometry_angstrom=float(g),
                eigenpair_index=int(k),
                softspaces_dual_pass=bool(label["pass"]),
                softspaces_prominence_rate=float(label["prominence_rate"]),
                softspaces_robust_rate=float(label["robust_rate"]),
                gap_hartree=float(evals[k + 1] - evals[k]),
                en2_weighted_score=weighted,
                en2_top2_support=supports[2],
                en2_top4_support=supports[4],
                en2_top8_support=supports[8],
                en2_top12_support=supports[12],
                en2_top16_support=supports[16],
            )

            rows_g.append(row)
            all_rows.append(row)

        labels = np.asarray(
            [1 if r.softspaces_dual_pass else 0 for r in rows_g],
            dtype=int,
        )

        metric_vectors = {
            "en2_weighted_score": np.asarray(
                [r.en2_weighted_score for r in rows_g],
                dtype=float,
            ),
            "en2_top2_support": np.asarray(
                [r.en2_top2_support for r in rows_g],
                dtype=float,
            ),
            "en2_top4_support": np.asarray(
                [r.en2_top4_support for r in rows_g],
                dtype=float,
            ),
            "en2_top8_support": np.asarray(
                [r.en2_top8_support for r in rows_g],
                dtype=float,
            ),
            "en2_top12_support": np.asarray(
                [r.en2_top12_support for r in rows_g],
                dtype=float,
            ),
            "en2_top16_support": np.asarray(
                [r.en2_top16_support for r in rows_g],
                dtype=float,
            ),
        }

        metric_summary = {}

        for name, vec in metric_vectors.items():
            pass_vals = vec[labels == 1]
            fail_vals = vec[labels == 0]

            metric_summary[name] = {
                "auc": rank_auc(labels, vec),
                "median_pass": (
                    float(np.median(pass_vals))
                    if len(pass_vals)
                    else None
                ),
                "median_fail": (
                    float(np.median(fail_vals))
                    if len(fail_vals)
                    else None
                ),
                "median_difference_pass_minus_fail": (
                    float(np.median(pass_vals) - np.median(fail_vals))
                    if len(pass_vals) and len(fail_vals)
                    else None
                ),
            }

        geometry_summary[str(g)] = {
            "sector_dimension": int(len(evals)),
            "reference_determinant_local_index": int(reference),
            "softspaces_dual_pass_pairs": int(np.sum(labels)),
            "nonpass_pairs": int(np.sum(labels == 0)),
            "metrics": metric_summary,
        }

        auc8 = metric_summary["en2_top8_support"]["auc"]
        aucw = metric_summary["en2_weighted_score"]["auc"]

        print(
            f"    SS dual-pass={int(np.sum(labels))} | "
            f"AUC EN2-weighted={aucw if aucw is not None else float('nan'):.3f} | "
            f"AUC top8-support={auc8 if auc8 is not None else float('nan'):.3f}"
        )

    # Global pooled analysis.
    labels = np.asarray(
        [1 if r.softspaces_dual_pass else 0 for r in all_rows],
        dtype=int,
    )

    global_metrics = {}

    metric_names = (
        "en2_weighted_score",
        "en2_top2_support",
        "en2_top4_support",
        "en2_top8_support",
        "en2_top12_support",
        "en2_top16_support",
    )

    for name in metric_names:
        vec = np.asarray(
            [float(getattr(r, name)) for r in all_rows],
            dtype=float,
        )

        pass_vals = vec[labels == 1]
        fail_vals = vec[labels == 0]

        global_metrics[name] = {
            "auc": rank_auc(labels, vec),
            "median_pass": (
                float(np.median(pass_vals))
                if len(pass_vals)
                else None
            ),
            "median_fail": (
                float(np.median(fail_vals))
                if len(fail_vals)
                else None
            ),
            "median_difference_pass_minus_fail": (
                float(np.median(pass_vals) - np.median(fail_vals))
                if len(pass_vals) and len(fail_vals)
                else None
            ),
        }

    pair_csv = output_dir / "phase5_v50_8_pair_metrics.csv"
    summary_json = output_dir / "phase5_v50_8_summary.json"
    hash_file = output_dir / "phase5_v50_8_summary.sha256"

    save_csv(all_rows, pair_csv)

    summary = {
        "_metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script_version": SCRIPT_VERSION,
            "python_version": sys.version.split()[0],
        },
        "input_softspaces_json": str(ss_path),
        "benchmark": {
            "molecule": "H4",
            "basis": "STO-3G",
            "geometries": list(GEOMETRIES),
            "fixed_sector": "(N_alpha,N_beta)=(2,2)",
        },
        "en2_definition": {
            "score": "|H_ir|^2 / (|H_ii-H_rr| + eps)",
            "eps": EN2_EPS,
            "reference": "minimum diagonal H_ii determinant",
            "top_k": list(K_VALUES),
        },
        "bridge_definition": {
            "en2_topk_support": (
                "mean determinant probability of eigenstates (k,k+1) "
                "contained in EN2 top-K determinant set"
            ),
            "en2_weighted_score": (
                "pair determinant probability weighted mean EN2 score"
            ),
            "canonical_en2_eigenstate_hotspot_claim": False,
        },
        "geometry_summary": geometry_summary,
        "global_summary": {
            "total_pairs": len(all_rows),
            "softspaces_dual_pass_pairs": int(np.sum(labels)),
            "nonpass_pairs": int(np.sum(labels == 0)),
            "metrics": global_metrics,
        },
        "interpretation_guide": {
            "auc_0_5": "no rank discrimination",
            "auc_above_0_5": "higher EN2 bridge metric tends to occur on SS PASS pairs",
            "auc_near_1": "EN2 bridge strongly predicts SS PASS pairs",
            "important_limit": (
                "EN2 is determinant-based while Soft Spaces v50_7c is "
                "eigenstate-pair based; this script tests an explicit bridge, "
                "not identity of the two methods."
            ),
        },
    }

    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    digest = sha256_hex(summary)
    hash_file.write_text(digest + "\n", encoding="ascii")

    print("-" * 78)
    print("v50_8 COMPLETE")
    print("-" * 78)
    print(f"Pair CSV     : {pair_csv}")
    print(f"Summary JSON : {summary_json}")
    print(f"SHA-256      : {digest}")
    print("-" * 78)

    for name in metric_names:
        m = global_metrics[name]
        auc = m["auc"]
        diff = m["median_difference_pass_minus_fail"]

        print(
            f"{name:22s} | "
            f"AUC={auc if auc is not None else float('nan'):.3f} | "
            f"median PASS-FAIL={diff if diff is not None else float('nan'):+.6e}"
        )

    print("-" * 78)
    print(
        "Interpretation: AUC near 0.5 means EN2 bridge does not distinguish "
        "Soft-Spaces PASS from non-PASS pairs; high AUC means it does."
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
