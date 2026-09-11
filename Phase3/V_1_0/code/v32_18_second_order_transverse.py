#!/usr/bin/env python3
"""
Soft Spaces Phase 3.2 — v32.18 Second-Order Transverse Response
===============================================================

Purpose
-------
Test the next perturbative order suggested by v32.17.1.

v32.17.1 showed:

    * dephasing: a clean first-order frozen-P/Q response with the expected
      O(delta^2) central-difference convergence;
    * transverse: the analytic first derivative is numerically extremely
      small, so first-order rank correlations are not physically trustworthy.

The natural hypothesis is therefore:

    transverse hotspot structure may be controlled by the leading
    non-vanishing SECOND-order response.

Frozen-P/Q model
----------------
As in v32.17:

    H(lambda) = H0 + lambda V

with P and Q frozen at lambda=0,

    A(lambda) = E_P(lambda) I_Q - Q H(lambda) Q,

    B = Q V P,

and

    Sigma(lambda) = B^dagger g_eta(A(lambda)) B,

where

    g_eta(A) = A (A^2 + eta^2 I)^(-1).

No fresh eigendecomposition is used for Sigma(lambda).

Primary second-order candidates
-------------------------------
    R2_E = || d^2 Sigma_E / dlambda^2 ||_F

    D2_E = | d^2 ||Sigma_E||_F^2 / dlambda^2 |

A 5-point central stencil is used:

    f''(0) ~= [-f(2h) + 16f(h) - 30f(0)
               + 16f(-h) - f(-2h)] / (12 h^2)

which has O(h^4) truncation error before roundoff dominates.

Scientific discipline
---------------------
This script is intentionally narrow:

    * 8Q
    * frozen 11-term Hamiltonian
    * seed 25042000 by default
    * transverse family by default
    * dephasing can be included only as a control
    * predeclared delta scan
    * no feature fitting
    * no optimization over delta

The delta scan is for numerical stability. The scientific quantities
are R2_E and D2_E.

Dependencies
------------
Must be beside this file:

    v32_14_degeneracy_closed.py
    v32_17_perturbative_response.py

Outputs
-------
    v32_18_seed25042000_rows.csv
    v32_18_seed25042000_stats.csv
    v32_18_delta_scan.csv
    v32_18_seed25042000_summary.txt

Usage
-----
    python .\\v32_18_second_order_transverse.py

Optional dephasing control:
    python .\\v32_18_second_order_transverse.py --families transverse dephasing
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


VERSION = "v32.18"
DEFAULT_SEED = 25_042_000
DEFAULT_DELTAS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4)
DEFAULT_FAMILIES = ("transverse",)


def load_module(filename: str, module_name: str):
    here = Path(__file__).resolve().parent
    path = here / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} must be beside this file.")

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fro_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord="fro"))


def spearman(base, x: Iterable[float], y: Iterable[float]) -> float:
    return float(
        base.spearman_np(
            np.asarray(list(x), dtype=float),
            np.asarray(list(y), dtype=float),
        )
    )


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def direct_regularized_matrix_function(A: np.ndarray, eta: float) -> np.ndarray:
    A = 0.5 * (A + A.conj().T)
    n = A.shape[0]
    M = A @ A + (eta * eta) * np.eye(n, dtype=np.complex128)
    X = np.linalg.solve(M, A)
    return 0.5 * (X + X.conj().T)


def sigma_direct(prepared: dict, lam: float, eta: float) -> np.ndarray:
    q_energies = np.asarray(prepared["q_energies"], dtype=float)
    v_qq = np.asarray(prepared["v_qq"], dtype=np.complex128)
    b_qp = np.asarray(prepared["b_qp"], dtype=np.complex128)

    z = (
        float(prepared["energy"])
        + float(lam) * float(prepared["target_slope"])
    )

    A = (
        np.diag(z - q_energies).astype(np.complex128)
        - float(lam) * v_qq
    )
    A = 0.5 * (A + A.conj().T)

    gA = direct_regularized_matrix_function(A, eta)
    sigma = b_qp.conj().T @ gA @ b_qp
    return 0.5 * (sigma + sigma.conj().T)


def second_derivative_5point(prepared: dict, h: float, eta: float) -> dict:
    s0 = sigma_direct(prepared, 0.0, eta)
    sp1 = sigma_direct(prepared, +h, eta)
    sm1 = sigma_direct(prepared, -h, eta)
    sp2 = sigma_direct(prepared, +2.0 * h, eta)
    sm2 = sigma_direct(prepared, -2.0 * h, eta)

    d2_sigma = (
        -sp2 + 16.0 * sp1 - 30.0 * s0 + 16.0 * sm1 - sm2
    ) / (12.0 * h * h)

    n0 = fro_norm(s0) ** 2
    np1 = fro_norm(sp1) ** 2
    nm1 = fro_norm(sm1) ** 2
    np2 = fro_norm(sp2) ** 2
    nm2 = fro_norm(sm2) ** 2

    d2_norm2 = (
        -np2 + 16.0 * np1 - 30.0 * n0 + 16.0 * nm1 - nm2
    ) / (12.0 * h * h)

    return {
        "R2_E": fro_norm(d2_sigma),
        "D2_E": abs(float(d2_norm2)),
        "D2_E_signed": float(d2_norm2),
    }


def second_derivative_3point(prepared: dict, h: float, eta: float) -> dict:
    s0 = sigma_direct(prepared, 0.0, eta)
    sp = sigma_direct(prepared, +h, eta)
    sm = sigma_direct(prepared, -h, eta)

    d2_sigma = (sp - 2.0 * s0 + sm) / (h * h)

    n0 = fro_norm(s0) ** 2
    np1 = fro_norm(sp) ** 2
    nm1 = fro_norm(sm) ** 2
    d2_norm2 = (np1 - 2.0 * n0 + nm1) / (h * h)

    return {
        "R2_E": fro_norm(d2_sigma),
        "D2_E": abs(float(d2_norm2)),
    }


def pair_records(ref: dict) -> List[dict]:
    out = []
    for a, b in ref["pairs"]:
        gm, gp = (
            (a, b)
            if ref["energies"][a] <= ref["energies"][b]
            else (b, a)
        )

        prominence = max(
            ref["prominence"].get(gm, float("-inf")),
            ref["prominence"].get(gp, float("-inf")),
        )
        if not np.isfinite(prominence):
            continue

        out.append(
            {
                "g_minus": int(gm),
                "g_plus": int(gp),
                "E_abs": float(abs(ref["energies"][gp])),
                "prominence": float(prominence),
            }
        )
    return out


def static_group_descriptors(base, ref: dict, gi: int, perturbation, eta: float):
    terms = base.degeneracy_closed_terms(
        eigenvalues=ref["evals"],
        eigenvectors=ref["evecs"],
        groups=ref["groups"],
        target_group_number=gi,
        perturbation=perturbation,
        eta=eta,
    )
    return base.interference_descriptors_group_closed(terms)


def run_pilot(base, v3217, seed: int, deltas: List[float], families: List[str]):
    eta = float(base.ENERGY_REG)
    ref = v3217.build_reference(base, seed)
    pairs = pair_records(ref)
    if not pairs:
        raise RuntimeError("No usable +/-E pairs found.")

    rows: List[dict] = []
    scan: List[dict] = []

    for family in families:
        perturbation = ref["perturbations"][family]
        needed_groups = sorted(
            set(p["g_minus"] for p in pairs)
            | set(p["g_plus"] for p in pairs)
        )

        prepared: Dict[int, dict] = {}
        static_desc: Dict[int, dict] = {}

        for gi in needed_groups:
            prepared[gi] = v3217.prepare_target(base, ref, gi, perturbation)
            static_desc[gi] = static_group_descriptors(
                base, ref, gi, perturbation, eta
            )

        for h in deltas:
            pair_rows = []

            for pair in pairs:
                gm = pair["g_minus"]
                gp = pair["g_plus"]

                m5 = second_derivative_5point(prepared[gm], h, eta)
                p5 = second_derivative_5point(prepared[gp], h, eta)
                m3 = second_derivative_3point(prepared[gm], h, eta)
                p3 = second_derivative_3point(prepared[gp], h, eta)

                sm = static_desc[gm]
                sp = static_desc[gp]

                r2_5 = 0.5 * (float(m5["R2_E"]) + float(p5["R2_E"]))
                d2_5 = 0.5 * (float(m5["D2_E"]) + float(p5["D2_E"]))
                r2_3 = 0.5 * (float(m3["R2_E"]) + float(p3["R2_E"]))
                d2_3 = 0.5 * (float(m3["D2_E"]) + float(p3["D2_E"]))

                row = {
                    "version": VERSION,
                    "seed": int(seed),
                    "family": family,
                    "delta": float(h),
                    "g_minus": int(gm),
                    "g_plus": int(gp),
                    "E_abs": float(pair["E_abs"]),
                    "prominence": float(pair["prominence"]),
                    "dim_minus": int(ref["groups"][gm].size),
                    "dim_plus": int(ref["groups"][gp].size),
                    "A_group": 0.5 * (
                        float(sm["A_group"]) + float(sp["A_group"])
                    ),
                    "C_group": 0.5 * (
                        float(sm["C_group"]) + float(sp["C_group"])
                    ),
                    "R2_E": r2_5,
                    "D2_E": d2_5,
                    "D2_E_signed_pairmean": 0.5 * (
                        float(m5["D2_E_signed"]) + float(p5["D2_E_signed"])
                    ),
                    "R2_3point": r2_3,
                    "D2_3point": d2_3,
                    "R2_3v5_rel_diff": abs(r2_3 - r2_5) / max(1e-30, r2_5),
                    "D2_3v5_rel_diff": abs(d2_3 - d2_5) / max(1e-30, d2_5),
                }

                pair_rows.append(row)
                rows.append(row)

            y = [r["prominence"] for r in pair_rows]
            for metric in ("A_group", "C_group", "R2_E", "D2_E"):
                rho = spearman(base, [r[metric] for r in pair_rows], y)
                scan.append(
                    {
                        "version": VERSION,
                        "seed": int(seed),
                        "family": family,
                        "delta": float(h),
                        "metric": metric,
                        "n_pairs": int(len(pair_rows)),
                        "spearman_rho_vs_prominence": float(rho),
                        "previous_delta": "",
                    }
                )

    for family in families:
        family_rows = [r for r in rows if r["family"] == family]
        by_delta = {}
        for h in deltas:
            sub = [
                r for r in family_rows
                if math.isclose(
                    float(r["delta"]), float(h),
                    rel_tol=0.0, abs_tol=1e-18
                )
            ]
            sub.sort(key=lambda r: (r["g_minus"], r["g_plus"]))
            by_delta[float(h)] = sub

        ordered = sorted([float(h) for h in deltas], reverse=True)

        for metric in ("R2_E", "D2_E"):
            previous = None
            previous_h = None

            for h in ordered:
                sub = by_delta[h]
                vals = [float(r[metric]) for r in sub]

                stability = float("nan")
                if previous is not None:
                    stability = spearman(base, previous, vals)

                scan.append(
                    {
                        "version": VERSION,
                        "seed": int(seed),
                        "family": family,
                        "delta": float(h),
                        "metric": metric + "_rank_stability",
                        "n_pairs": int(len(sub)),
                        "spearman_rho_vs_prominence": stability,
                        "previous_delta": (
                            "" if previous_h is None else float(previous_h)
                        ),
                    }
                )

                previous = vals
                previous_h = h

    stats = [
        r for r in scan
        if not str(r["metric"]).endswith("_rank_stability")
    ]

    group_sizes = [int(g.size) for g in ref["groups"]]

    lines = [
        f"Soft Spaces Phase 3.2 {VERSION}",
        "Second-order frozen-P/Q response pilot",
        "",
        f"seed={seed}",
        f"qubits={base.N_QUBITS}",
        f"dimension={base.DIM}",
        f"hamiltonian_terms={base.N_TERMS}",
        f"groups={len(ref['groups'])}",
        f"group_size_range={min(group_sizes)}:{max(group_sizes)}",
        f"exact_pmE_pairs={len(ref['pairs'])}",
        f"usable_pairs={len(pairs)}",
        f"eta={eta:.6g}",
        "families=" + ",".join(families),
        "deltas=" + ",".join(f"{d:.6g}" for d in deltas),
        "",
        "PRIMARY HYPOTHESIS",
        "------------------",
        "If transverse first-order susceptibility is symmetry-suppressed,",
        "the leading non-vanishing local response may occur at second order:",
        "",
        "  R2_E = || d2 Sigma_E / d lambda2 ||_F",
        "  D2_E = | d2 ||Sigma_E||_F^2 / d lambda2 |",
        "",
        "RESULTS BY DELTA",
        "----------------",
    ]

    for family in families:
        lines.append("")
        lines.append(f"[{family}]")

        for h in deltas:
            lines.append(f"  delta={h:.6g}")

            for metric in ("A_group", "C_group", "R2_E", "D2_E"):
                matches = [
                    s for s in stats
                    if s["family"] == family
                    and s["metric"] == metric
                    and math.isclose(
                        float(s["delta"]), float(h),
                        rel_tol=0.0, abs_tol=1e-18
                    )
                ]
                if matches:
                    rho = float(matches[0]["spearman_rho_vs_prominence"])
                    lines.append(f"    {metric:10s} rho={rho:+.4f}")

            sub = [
                r for r in rows
                if r["family"] == family
                and math.isclose(
                    float(r["delta"]), float(h),
                    rel_tol=0.0, abs_tol=1e-18
                )
            ]
            if sub:
                med_r = float(
                    np.median([r["R2_3v5_rel_diff"] for r in sub])
                )
                med_d = float(
                    np.median([r["D2_3v5_rel_diff"] for r in sub])
                )
                lines.append(f"    median 3v5 rel diff R2={med_r:.3e}")
                lines.append(f"    median 3v5 rel diff D2={med_d:.3e}")

    lines += [
        "",
        "DECISION RULE",
        "-------------",
        "Do not select the delta that maximizes rho.",
        "A credible second-order law should show:",
        "  1) stable R2/D2 rankings across a finite delta window;",
        "  2) reasonable 3-point/5-point agreement before roundoff;",
        "  3) a systematic positive relation to hotspot prominence;",
        "  4) preferably stronger/more stable transverse behavior than the",
        "     numerically suppressed first-order response.",
        "",
        "No cross-seed or full-spectral claim is made by this pilot.",
    ]

    return rows, stats, scan, "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Soft Spaces Phase 3.2 v32.18 — second-order transverse "
            "frozen-P/Q response pilot."
        )
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed (default {DEFAULT_SEED}).",
    )
    p.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=list(DEFAULT_DELTAS),
        help="Predeclared second-derivative step sizes.",
    )
    p.add_argument(
        "--families",
        nargs="+",
        default=list(DEFAULT_FAMILIES),
        choices=("dephasing", "transverse"),
        help="Families to run. Default: transverse.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (default current directory).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    base = load_module(
        "v32_14_degeneracy_closed.py",
        "v3214_for_v3218",
    )
    v3217 = load_module(
        "v32_17_perturbative_response.py",
        "v3217_for_v3218",
    )

    if int(base.N_QUBITS) != 8:
        raise RuntimeError(
            f"v32.18 pilot is frozen for 8Q; base reports {base.N_QUBITS}Q."
        )

    if int(base.N_TERMS) != 11:
        raise RuntimeError(
            f"v32.18 expects 11 Hamiltonian terms; "
            f"base reports {base.N_TERMS}."
        )

    deltas = sorted(
        set(float(d) for d in args.deltas),
        reverse=True,
    )
    if not deltas or any(
        (not np.isfinite(d) or d <= 0.0) for d in deltas
    ):
        raise ValueError("All delta values must be finite and > 0.")

    families = list(dict.fromkeys(args.families))

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"{VERSION}: 8Q second-order frozen-P/Q pilot, "
        f"seed {args.seed}, families={families}"
    )

    rows, stats, scan, summary = run_pilot(
        base=base,
        v3217=v3217,
        seed=int(args.seed),
        deltas=deltas,
        families=families,
    )

    rows_path = outdir / f"v32_18_seed{args.seed}_rows.csv"
    stats_path = outdir / f"v32_18_seed{args.seed}_stats.csv"
    scan_path = outdir / "v32_18_delta_scan.csv"
    summary_path = outdir / f"v32_18_seed{args.seed}_summary.txt"

    write_csv(rows_path, rows)
    write_csv(stats_path, stats)
    write_csv(scan_path, scan)
    summary_path.write_text(summary, encoding="utf-8")

    print(summary)
    print("WROTE")
    print(f"  {rows_path}")
    print(f"  {stats_path}")
    print(f"  {scan_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
