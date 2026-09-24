#!/usr/bin/env python3
"""
Soft Spaces Phase 5 — v50_9 matched-budget subspace utility test

Tests whether v50_7c DUAL-PASS Soft-Spaces hotspots can be converted into a
determinant selection with lower ground-state energy error than EN2 at the
same determinant budget.

IMPORTANT:
The Soft-Spaces selector uses exact eigenvectors to translate validated
eigenpair hotspots into determinant-support scores. Therefore this is an
ORACLE / INFORMATION-CONTENT test, not yet a computational-speedup test.

Soft-Spaces determinant score:
    SS_i = sum_hotspots c_h * 0.5*(|V[i,k]|^2 + |V[i,k+1]|^2)
where
    c_h = min(prominence_positive_rate, robust_positive_rate)

EN2 determinant score:
    EN2_i = |H_ir|^2 / (|H_ii-H_rr| + eps)

Both methods include the same reference determinant r, defined as the
determinant with minimum diagonal H_ii.

Budgets: K = 2,4,8,12,16.
Primary metric: absolute selected-subspace ground-state energy error.
Negative (SS error - EN2 error) means Soft Spaces is better.
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

SCRIPT_VERSION = "v50_9"
GEOMETRIES = (0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
K_VALUES = (2, 4, 8, 12, 16)
EN2_EPS = 1e-12


@dataclass(frozen=True)
class ResultRow:
    geometry_angstrom: float
    k_budget: int
    n_softspaces_hotspots: int
    exact_ground_energy: float
    softspaces_subspace_energy: float
    en2_subspace_energy: float
    softspaces_abs_energy_error: float
    en2_abs_energy_error: float
    error_difference_ss_minus_en2: float
    softspaces_captured_ground_mass: float
    en2_captured_ground_mass: float
    captured_mass_difference_ss_minus_en2: float
    softspaces_better_energy: bool
    en2_better_energy: bool
    energy_tie: bool


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def popcount(x: int) -> int:
    return bin(int(x)).count("1")


def alpha_beta_counts(index: int, n_spatial: int) -> tuple[int, int]:
    mask = (1 << n_spatial) - 1
    return popcount(index & mask), popcount((index >> n_spatial) & mask)


def fixed_sector_basis(n_spatial: int, n_alpha: int, n_beta: int) -> np.ndarray:
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
            "Qiskit Nature / PySCF imports failed. Run in the Phase-5 WSL environment."
        ) from exc

    zs = (-1.5 * spacing, -0.5 * spacing, 0.5 * spacing, 1.5 * spacing)
    atom = "; ".join(f"H 0.0 0.0 {z:.12f}" for z in zs)

    problem = PySCFDriver(
        atom=atom,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    ).run()

    n_spatial = int(problem.num_spatial_orbitals)
    n_alpha = int(problem.num_alpha)
    n_beta = int(problem.num_beta)

    if (n_spatial, n_alpha, n_beta) != (4, 2, 2):
        raise RuntimeError(
            f"Unexpected H4 metadata: {n_spatial=}, {n_alpha=}, {n_beta=}"
        )

    basis = fixed_sector_basis(n_spatial, n_alpha, n_beta)
    qubit_op = JordanWignerMapper().map(problem.hamiltonian.second_q_op())
    full = qubit_op.to_matrix(sparse=True)
    h = np.asarray(full[basis, :][:, basis].toarray(), dtype=np.complex128)

    err = float(np.max(np.abs(h - h.conjugate().T)))
    if err > 1e-10:
        raise RuntimeError(f"Hamiltonian non-Hermitian: {err:.3e}")

    evals, evecs = np.linalg.eigh(h)
    return {
        "hamiltonian": h,
        "evals": np.asarray(evals, dtype=float),
        "evecs": np.asarray(evecs, dtype=np.complex128),
    }


def load_hotspots(path: Path) -> dict[float, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("_metadata", {}).get("script_version") != "v50_7c":
        raise RuntimeError("Expected v50_7c summary JSON.")

    out = {}
    for g_str, block in data.get("geometry_details", {}).items():
        out[float(g_str)] = list(block.get("passing_candidates", []))
    return out


def en2_scores(h: np.ndarray) -> tuple[np.ndarray, int]:
    diag = np.real(np.diag(h)).astype(float)
    r = int(np.argmin(diag))
    hrr = float(diag[r])
    s = np.zeros(len(diag), dtype=float)

    for i in range(len(diag)):
        if i == r:
            continue
        s[i] = float(abs(h[i, r]) ** 2) / (
            abs(float(diag[i] - hrr)) + EN2_EPS
        )

    return s, r


def softspaces_scores(
    evecs: np.ndarray, hotspot_rows: list[dict[str, Any]]
) -> np.ndarray:
    score = np.zeros(evecs.shape[0], dtype=float)

    for row in hotspot_rows:
        k = int(row["eigenpair_index"])
        c = min(
            float(row["prominence_positive_rate"]),
            float(row["robust_positive_rate"]),
        )

        pair_support = 0.5 * (
            np.abs(evecs[:, k]) ** 2
            + np.abs(evecs[:, k + 1]) ** 2
        )

        score += c * pair_support

    return score


def topk_with_reference(scores: np.ndarray, reference: int, k: int) -> np.ndarray:
    ranking = np.asarray(scores, dtype=float).copy()
    ranking[reference] = np.inf
    return np.asarray(np.argsort(-ranking, kind="stable")[:k], dtype=int)


def selected_ground_energy(h: np.ndarray, selected: np.ndarray) -> float:
    sub = h[np.ix_(selected, selected)]
    return float(np.min(np.linalg.eigvalsh(sub)).real)


def captured_mass(ground: np.ndarray, selected: np.ndarray) -> float:
    return float(np.sum(np.abs(ground[selected]) ** 2))


def save_csv(rows: list[ResultRow], path: Path) -> None:
    fields = list(ResultRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        default=v1_root / "results" / "matched_budget_utility",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ss_path = args.softspaces_json.resolve()
    if not ss_path.exists():
        raise FileNotFoundError(f"Missing v50_7c JSON: {ss_path}")

    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    hotspots = load_hotspots(ss_path)

    print("=" * 78)
    print("SOFT SPACES — PHASE 5 v50_9 MATCHED-BUDGET SUBSPACE UTILITY")
    print("=" * 78)
    print("Molecule       : H4 / STO-3G")
    print("Sector         : (N_alpha,N_beta)=(2,2)")
    print(f"Geometries     : {GEOMETRIES}")
    print(f"Budgets K      : {K_VALUES}")
    print("Primary metric : absolute ground-state energy error at equal K")
    print(
        "LIMIT          : Soft-Spaces determinant ranking uses exact eigenvectors "
        "(oracle bridge)."
    )
    print("-" * 78)

    rows: list[ResultRow] = []
    geometry_details = {}

    for gi, g in enumerate(GEOMETRIES, start=1):
        print(f"[{gi}/8] H4 geometry={g:.2f} Å", flush=True)

        p = build_h4(g)
        h = p["hamiltonian"]
        evals = p["evals"]
        evecs = p["evecs"]

        exact_e = float(evals[0])
        ground = evecs[:, 0]

        en2, ref = en2_scores(h)
        hs = hotspots.get(float(g), [])
        ss = softspaces_scores(evecs, hs)

        geom_rows = []

        for k in K_VALUES:
            ss_sel = topk_with_reference(ss, ref, k)
            en2_sel = topk_with_reference(en2, ref, k)

            e_ss = selected_ground_energy(h, ss_sel)
            e_en2 = selected_ground_energy(h, en2_sel)

            err_ss = abs(e_ss - exact_e)
            err_en2 = abs(e_en2 - exact_e)

            m_ss = captured_mass(ground, ss_sel)
            m_en2 = captured_mass(ground, en2_sel)

            tol = 1e-12

            row = ResultRow(
                geometry_angstrom=float(g),
                k_budget=int(k),
                n_softspaces_hotspots=len(hs),
                exact_ground_energy=exact_e,
                softspaces_subspace_energy=e_ss,
                en2_subspace_energy=e_en2,
                softspaces_abs_energy_error=err_ss,
                en2_abs_energy_error=err_en2,
                error_difference_ss_minus_en2=err_ss - err_en2,
                softspaces_captured_ground_mass=m_ss,
                en2_captured_ground_mass=m_en2,
                captured_mass_difference_ss_minus_en2=m_ss - m_en2,
                softspaces_better_energy=bool(err_ss < err_en2 - tol),
                en2_better_energy=bool(err_en2 < err_ss - tol),
                energy_tie=bool(abs(err_ss - err_en2) <= tol),
            )

            rows.append(row)
            geom_rows.append(asdict(row))

            if row.softspaces_better_energy:
                winner = "SS"
            elif row.en2_better_energy:
                winner = "EN2"
            else:
                winner = "TIE"

            print(
                f"    K={k:2d} | SS err={err_ss:.8f} | "
                f"EN2 err={err_en2:.8f} | "
                f"Δ={err_ss-err_en2:+.8f} | "
                f"mass SS={m_ss:.4f} EN2={m_en2:.4f} | {winner}"
            )

        geometry_details[str(g)] = {
            "n_softspaces_hotspots": len(hs),
            "reference_determinant_local_index": int(ref),
            "results": geom_rows,
        }

    by_k = {}

    for k in K_VALUES:
        subset = [r for r in rows if r.k_budget == k]
        d = np.asarray(
            [r.error_difference_ss_minus_en2 for r in subset],
            dtype=float,
        )
        md = np.asarray(
            [r.captured_mass_difference_ss_minus_en2 for r in subset],
            dtype=float,
        )

        by_k[str(k)] = {
            "softspaces_energy_wins": int(sum(r.softspaces_better_energy for r in subset)),
            "en2_energy_wins": int(sum(r.en2_better_energy for r in subset)),
            "energy_ties": int(sum(r.energy_tie for r in subset)),
            "median_error_difference_ss_minus_en2": float(np.median(d)),
            "mean_error_difference_ss_minus_en2": float(np.mean(d)),
            "median_captured_mass_difference_ss_minus_en2": float(np.median(md)),
        }

    diffs = np.asarray(
        [r.error_difference_ss_minus_en2 for r in rows],
        dtype=float,
    )
    massdiffs = np.asarray(
        [r.captured_mass_difference_ss_minus_en2 for r in rows],
        dtype=float,
    )

    ss_wins = int(sum(r.softspaces_better_energy for r in rows))
    en2_wins = int(sum(r.en2_better_energy for r in rows))
    ties = int(sum(r.energy_tie for r in rows))

    csv_path = outdir / "phase5_v50_9_matched_budget_results.csv"
    json_path = outdir / "phase5_v50_9_summary.json"
    hash_path = outdir / "phase5_v50_9_summary.sha256"

    save_csv(rows, csv_path)

    summary = {
        "_metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script_version": SCRIPT_VERSION,
            "python_version": sys.version.split()[0],
        },
        "benchmark": {
            "molecule": "H4",
            "basis": "STO-3G",
            "geometries": list(GEOMETRIES),
            "fixed_sector": "(N_alpha,N_beta)=(2,2)",
            "budgets": list(K_VALUES),
        },
        "softspaces_selector": {
            "source": "v50_7c DUAL-PASS eigenpair hotspots",
            "determinant_score": (
                "sum_hotspots min(prominence_rate,robust_rate) * "
                "0.5*(|V_i,k|^2+|V_i,k+1|^2)"
            ),
            "oracle_bridge": True,
            "reference_forced": True,
        },
        "en2_selector": {
            "score": "|H_ir|^2/(|H_ii-H_rr|+eps)",
            "eps": EN2_EPS,
            "reference": "minimum diagonal H_ii determinant",
            "reference_forced": True,
        },
        "geometry_details": geometry_details,
        "aggregate": {
            "total_matched_budget_comparisons": len(rows),
            "softspaces_energy_wins": ss_wins,
            "en2_energy_wins": en2_wins,
            "energy_ties": ties,
            "median_error_difference_ss_minus_en2": float(np.median(diffs)),
            "mean_error_difference_ss_minus_en2": float(np.mean(diffs)),
            "median_captured_mass_difference_ss_minus_en2": float(np.median(massdiffs)),
            "by_k": by_k,
        },
        "interpretation_boundary": {
            "positive_result_means": (
                "Soft-Spaces hotspots contain downstream determinant-selection "
                "information that can beat EN2 at matched selected-subspace size."
            ),
            "not_yet_proven": (
                "lower total computational cost or Hamiltonian-access cost"
            ),
            "reason": (
                "v50_9 uses exact eigenvectors as an oracle bridge from "
                "eigenpair hotspots to determinant scores."
            ),
        },
    }

    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    digest = sha256_hex(summary)
    hash_path.write_text(digest + "\n", encoding="ascii")

    print("-" * 78)
    print("v50_9 COMPLETE")
    print("-" * 78)
    print(f"Results CSV  : {csv_path}")
    print(f"Summary JSON : {json_path}")
    print(f"SHA-256      : {digest}")
    print("-" * 78)
    print(f"Matched comparisons : {len(rows)}")
    print(f"Soft-Spaces wins    : {ss_wins}")
    print(f"EN2 wins            : {en2_wins}")
    print(f"Ties                : {ties}")
    print(f"Median Δerror SS-EN2: {np.median(diffs):+.8e}")
    print(f"Median Δmass SS-EN2 : {np.median(massdiffs):+.8e}")
    print("-" * 78)

    for k in K_VALUES:
        s = by_k[str(k)]
        print(
            f"K={k:2d} | SS wins={s['softspaces_energy_wins']} | "
            f"EN2 wins={s['en2_energy_wins']} | ties={s['energy_ties']} | "
            f"median Δerr={s['median_error_difference_ss_minus_en2']:+.8e}"
        )

    print("-" * 78)
    print("Negative Δerror => Soft Spaces has lower energy error than EN2.")
    print("This is an information-content test, not yet a speedup test.")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
