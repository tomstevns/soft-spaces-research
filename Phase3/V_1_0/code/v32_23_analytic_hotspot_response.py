#!/usr/bin/env python3
"""
Soft Spaces Phase 3.2 — v32.23 Analytic Hotspot Response Law
============================================================

Purpose
-------
Validate the analytic bridge

    virtual pathways -> coherent self-energy -> effective P-space response

for the frozen 8Q Phase-3 model.

The degeneracy-closed second-order pathway operators are

    T_G = g_eta(E - E_G) P V P_G V P,

with

    g_eta(x) = x / (x^2 + eta^2),

and

    Sigma_E = sum_G T_G.

Define

    L1_E = sum_G ||T_G||_F,

    C_E = ||Sigma_E||_F / L1_E.

Then the exact norm law is

    ||Sigma_E||_F = C_E * L1_E.

Therefore, for a scaled perturbation lambda V, the leading regularized
second-order Feshbach response is

    Delta H_eff^(2)(lambda) = lambda^2 Sigma_E,

so that

    ||Delta H_eff^(2)(lambda)||_F
        = lambda^2 C_E L1_E.

This is the analytic response law tested here.

A stronger finite-lambda check is also performed.  In the H0 eigenbasis,
for a fixed target energy E,

    F_exact(lambda)
      = lambda^2 Re[
            B^dagger
            (E I - Q H0 Q - lambda Q V Q + i eta I)^(-1)
            B
        ],

where B = Q V P.

Expanding the resolvent gives

    F_exact(lambda)
      = lambda^2 Sigma_E + O(lambda^3).

Hence:

    absolute error  ~ O(lambda^3)
    relative error  ~ O(lambda)

for sufficiently small lambda.

Scientific role
---------------
v32.21 established exactly

    C_E^2 = S_E + A_E,

with pathway concentration S_E and interference A_E.

v32.23 now gives C_E a direct dynamical/effective-Hamiltonian meaning:
at fixed total virtual-pathway budget L1_E, C_E is exactly the factor that
amplifies or suppresses the leading second-order effective response.

This does NOT yet prove that hotspot prominence is exactly proportional to
C_E.  It provides the analytic mechanism connecting coherent virtual paths
to enhanced local effective response.

Frozen pilot
------------
Default:
    8 qubits
    11 Hamiltonian terms
    seed 25042000
    both perturbation families
    all exact +/-E pairs
    eta = frozen ENERGY_REG
    lambdas = 1e-2, 3e-3, 1e-3, 3e-4

Dependencies
------------
Must be beside this file:

    v32_14_degeneracy_closed.py
    v32_17_perturbative_response.py

Outputs
-------
    v32_23_rows.csv
    v32_23_lambda_checks.csv
    v32_23_summary.txt

Usage
-----
    python .\v32_23_analytic_hotspot_response.py

Optional:
    python .\v32_23_analytic_hotspot_response.py --seed 25042001
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


VERSION = "v32.23"
DENOM_EPS = 1.0e-30


# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def frob(a: np.ndarray) -> float:
    return float(np.linalg.norm(a, ord="fro"))


def log_slope(xs: List[float], ys: List[float]) -> float:
    pts = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x > 0.0 and y > 0.0 and np.isfinite(y)
    ]
    if len(pts) < 2:
        return float("nan")

    lx = np.log([p[0] for p in pts])
    ly = np.log([p[1] for p in pts])
    return float(np.polyfit(lx, ly, 1)[0])


def pair_records(ref: dict) -> List[dict]:
    out = []
    for a, b in ref["pairs"]:
        gm, gp = (
            (a, b)
            if ref["energies"][a] <= ref["energies"][b]
            else (b, a)
        )

        prom = max(
            ref["prominence"].get(gm, float("-inf")),
            ref["prominence"].get(gp, float("-inf")),
        )

        if np.isfinite(prom):
            out.append(
                {
                    "g_minus": int(gm),
                    "g_plus": int(gp),
                    "E_abs": float(abs(ref["energies"][gp])),
                    "prominence": float(prom),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Grouped pathway descriptors
# ---------------------------------------------------------------------------

def grouped_pathway_data(base, ref: dict, gi: int, perturbation) -> dict:
    terms = base.degeneracy_closed_terms(
        eigenvalues=ref["evals"],
        eigenvectors=ref["evecs"],
        groups=ref["groups"],
        target_group_number=gi,
        perturbation=perturbation,
        eta=base.ENERGY_REG,
    )

    if not terms:
        z = np.zeros(
            (len(ref["groups"][gi]), len(ref["groups"][gi])),
            dtype=np.complex128,
        )
        return {
            "Sigma": z,
            "L1": 0.0,
            "C": 0.0,
            "S": 0.0,
            "A": 0.0,
            "identity_residual": 0.0,
            "norm_law_residual": 0.0,
        }

    mats = [np.asarray(t, dtype=np.complex128) for t in terms]
    norms = np.asarray([frob(t) for t in mats], dtype=float)

    sigma = np.sum(mats, axis=0)
    sigma_norm = frob(sigma)

    L1 = float(np.sum(norms))
    self_sum = float(np.sum(norms * norms))
    sigma2 = sigma_norm * sigma_norm

    if L1 <= DENOM_EPS:
        C = S = A = 0.0
    else:
        denom = L1 * L1
        C = float(sigma_norm / L1)
        S = float(self_sum / denom)
        A = float((sigma2 - self_sum) / denom)

    return {
        "Sigma": sigma,
        "L1": L1,
        "C": C,
        "S": S,
        "A": A,
        "identity_residual": float(C * C - (S + A)),
        "norm_law_residual": float(sigma_norm - C * L1),
    }


# ---------------------------------------------------------------------------
# Direct regularized Feshbach response
# ---------------------------------------------------------------------------

def feshbach_objects(
    base,
    ref: dict,
    family: str,
) -> dict:
    """
    Build V in the H0 eigenbasis once for one perturbation family.
    """
    perturbation = ref["perturbations"][family]
    V = base.dense_pauli_sum(base.N_QUBITS, perturbation)

    U = np.asarray(ref["evecs"], dtype=np.complex128)
    V_eig = U.conj().T @ V @ U

    return {
        "V_eig": V_eig,
    }


def exact_regularized_response(
    ref: dict,
    V_eig: np.ndarray,
    gi: int,
    lam: float,
    eta: float,
) -> np.ndarray:
    """
    lambda^2 * Re[B^dagger (E - QH0Q - lambda QVQ + i eta)^(-1) B]
    in the P-space eigenbasis.
    """
    pidx = np.asarray(ref["groups"][gi], dtype=int)
    all_idx = np.arange(len(ref["evals"]), dtype=int)

    mask = np.ones(len(all_idx), dtype=bool)
    mask[pidx] = False
    qidx = all_idx[mask]

    E = float(np.mean(ref["evals"][pidx]))
    Eq = np.asarray(ref["evals"][qidx], dtype=float)

    B = V_eig[np.ix_(qidx, pidx)]
    QVQ = V_eig[np.ix_(qidx, qidx)]

    A = (
        (E - Eq + 1j * eta)[:, None]
        * np.eye(len(qidx), dtype=np.complex128)
        - lam * QVQ
    )

    # Solve instead of explicitly forming the inverse.
    X = np.linalg.solve(A, B)
    F_complex = B.conj().T @ X

    # Real/Hermitian part of the regularized self-energy.
    F_real = 0.5 * (F_complex + F_complex.conj().T)

    return (lam * lam) * F_real


def direct_sigma0_from_complex_resolvent(
    ref: dict,
    V_eig: np.ndarray,
    gi: int,
    eta: float,
) -> np.ndarray:
    pidx = np.asarray(ref["groups"][gi], dtype=int)
    all_idx = np.arange(len(ref["evals"]), dtype=int)

    mask = np.ones(len(all_idx), dtype=bool)
    mask[pidx] = False
    qidx = all_idx[mask]

    E = float(np.mean(ref["evals"][pidx]))
    Eq = np.asarray(ref["evals"][qidx], dtype=float)

    B = V_eig[np.ix_(qidx, pidx)]

    diag = E - Eq + 1j * eta
    X = B / diag[:, None]
    F_complex = B.conj().T @ X
    return 0.5 * (F_complex + F_complex.conj().T)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_family(base, ref: dict, family: str, lambdas: List[float]):
    perturbation = ref["perturbations"][family]
    fobj = feshbach_objects(base, ref, family)
    V_eig = fobj["V_eig"]

    pairs = pair_records(ref)
    needed = sorted(
        set(p["g_minus"] for p in pairs)
        | set(p["g_plus"] for p in pairs)
    )

    group_data: Dict[int, dict] = {}
    lambda_rows: List[dict] = []

    max_sigma_equiv = 0.0
    max_identity = 0.0
    max_norm_law = 0.0

    for gi in needed:
        gd = grouped_pathway_data(base, ref, gi, perturbation)
        sigma_group = gd["Sigma"]

        sigma_direct = direct_sigma0_from_complex_resolvent(
            ref=ref,
            V_eig=V_eig,
            gi=gi,
            eta=float(base.ENERGY_REG),
        )

        sigma_equiv_resid = frob(sigma_direct - sigma_group)

        gd["sigma_equiv_residual"] = sigma_equiv_resid
        group_data[gi] = gd

        max_sigma_equiv = max(max_sigma_equiv, sigma_equiv_resid)
        max_identity = max(
            max_identity, abs(float(gd["identity_residual"]))
        )
        max_norm_law = max(
            max_norm_law, abs(float(gd["norm_law_residual"]))
        )

        abs_errors = []
        rel_errors = []

        for lam in lambdas:
            exact = exact_regularized_response(
                ref=ref,
                V_eig=V_eig,
                gi=gi,
                lam=float(lam),
                eta=float(base.ENERGY_REG),
            )

            leading = (lam * lam) * sigma_group
            abs_err = frob(exact - leading)
            lead_norm = frob(leading)
            rel_err = abs_err / max(lead_norm, DENOM_EPS)

            abs_errors.append(abs_err)
            rel_errors.append(rel_err)

            lambda_rows.append(
                {
                    "version": VERSION,
                    "seed": int(ref["seed"]),
                    "family": family,
                    "group": int(gi),
                    "lambda": float(lam),
                    "leading_norm": float(lead_norm),
                    "exact_norm": float(frob(exact)),
                    "abs_error": float(abs_err),
                    "rel_error": float(rel_err),
                }
            )

        gd["abs_error_slope"] = log_slope(lambdas, abs_errors)
        gd["rel_error_slope"] = log_slope(lambdas, rel_errors)

    rows: List[dict] = []

    for p in pairs:
        gm = p["g_minus"]
        gp = p["g_plus"]
        dm = group_data[gm]
        dp = group_data[gp]

        rows.append(
            {
                "version": VERSION,
                "seed": int(ref["seed"]),
                "family": family,
                "g_minus": int(gm),
                "g_plus": int(gp),
                "E_abs": float(p["E_abs"]),
                "prominence": float(p["prominence"]),
                "L1": 0.5 * (dm["L1"] + dp["L1"]),
                "C_group": 0.5 * (dm["C"] + dp["C"]),
                "S_group": 0.5 * (dm["S"] + dp["S"]),
                "A_group": 0.5 * (dm["A"] + dp["A"]),
                "response_norm_coeff": 0.5 * (
                    frob(dm["Sigma"]) + frob(dp["Sigma"])
                ),
                "C_times_L1": 0.5 * (
                    dm["C"] * dm["L1"] + dp["C"] * dp["L1"]
                ),
                "abs_error_slope": 0.5 * (
                    dm["abs_error_slope"] + dp["abs_error_slope"]
                ),
                "rel_error_slope": 0.5 * (
                    dm["rel_error_slope"] + dp["rel_error_slope"]
                ),
            }
        )

    diagnostics = {
        "max_sigma_equiv_residual": max_sigma_equiv,
        "max_identity_residual": max_identity,
        "max_norm_law_residual": max_norm_law,
        "median_abs_error_slope": float(
            np.nanmedian([g["abs_error_slope"] for g in group_data.values()])
        ),
        "median_rel_error_slope": float(
            np.nanmedian([g["rel_error_slope"] for g in group_data.values()])
        ),
        "median_smallest_lambda_rel_error": float(
            np.median(
                [
                    r["rel_error"]
                    for r in lambda_rows
                    if math.isclose(
                        r["lambda"],
                        min(lambdas),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                ]
            )
        ),
    }

    return rows, lambda_rows, diagnostics


def parse_args():
    p = argparse.ArgumentParser(
        description="Soft Spaces Phase 3.2 v32.23 analytic response law."
    )
    p.add_argument(
        "--seed",
        type=int,
        default=25_042_000,
        help="Frozen 8Q seed. Default 25042000.",
    )
    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[1e-2, 3e-3, 1e-3, 3e-4],
        help="Positive finite-lambda values.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=".",
    )
    return p.parse_args()


def main():
    args = parse_args()

    base = load_module(
        "v32_14_degeneracy_closed.py",
        "v3214_for_v3223",
    )
    v3217 = load_module(
        "v32_17_perturbative_response.py",
        "v3217_for_v3223",
    )

    if int(base.N_QUBITS) != 8:
        raise RuntimeError(
            f"v32.23 pilot is frozen for 8Q; base reports {base.N_QUBITS}Q."
        )
    if int(base.N_TERMS) != 11:
        raise RuntimeError(
            f"v32.23 expects 11 Hamiltonian terms; base reports {base.N_TERMS}."
        )

    lambdas = sorted(
        [float(x) for x in args.lambdas if float(x) > 0.0],
        reverse=True,
    )
    if len(lambdas) < 2:
        raise ValueError("Need at least two positive lambda values.")

    ref = v3217.build_reference(base, int(args.seed))
    ref["seed"] = int(args.seed)

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    all_lambda_rows: List[dict] = []
    all_diag: Dict[str, dict] = {}

    print(f"{VERSION}: Analytic Hotspot Response Law")
    print(f"seed={args.seed}, qubits={base.N_QUBITS}, terms={base.N_TERMS}")
    print("lambdas=" + ",".join(f"{x:g}" for x in lambdas))
    print()
    print("Leading law:")
    print("  Delta H_eff^(2)(lambda) = lambda^2 Sigma_E")
    print("  ||Delta H_eff^(2)||_F = lambda^2 C_E L1_E")
    print()

    for family in base.FAMILIES:
        print(f"[{family}] ...", flush=True)

        rows, lambda_rows, diag = analyze_family(
            base=base,
            ref=ref,
            family=family,
            lambdas=lambdas,
        )

        all_rows.extend(rows)
        all_lambda_rows.extend(lambda_rows)
        all_diag[family] = diag

        print(
            "  max grouped-vs-resolvent Sigma residual = "
            f"{diag['max_sigma_equiv_residual']:.6e}"
        )
        print(
            "  max C^2-(S+A) residual              = "
            f"{diag['max_identity_residual']:.6e}"
        )
        print(
            "  max ||Sigma||-C*L1 residual         = "
            f"{diag['max_norm_law_residual']:.6e}"
        )
        print(
            "  median absolute-error slope         = "
            f"{diag['median_abs_error_slope']:+.4f} "
            "(expected ~3)"
        )
        print(
            "  median relative-error slope         = "
            f"{diag['median_rel_error_slope']:+.4f} "
            "(expected ~1)"
        )
        print(
            "  median relative error @ smallest λ  = "
            f"{diag['median_smallest_lambda_rel_error']:.6e}"
        )
        print()

    rows_path = outdir / "v32_23_rows.csv"
    lambda_path = outdir / "v32_23_lambda_checks.csv"
    summary_path = outdir / "v32_23_summary.txt"

    write_csv(rows_path, all_rows)
    write_csv(lambda_path, all_lambda_rows)

    lines = [
        f"Soft Spaces Phase 3.2 {VERSION}",
        "Analytic Hotspot Response Law",
        "",
        f"seed={args.seed}",
        f"qubits={base.N_QUBITS}",
        f"dimension={base.DIM}",
        f"hamiltonian_terms={base.N_TERMS}",
        "lambdas=" + ",".join(f"{x:g}" for x in lambdas),
        "",
        "ANALYTIC LAW",
        "------------",
        "Sigma_E = sum_G T_G",
        "L1_E    = sum_G ||T_G||_F",
        "C_E     = ||Sigma_E||_F / L1_E",
        "",
        "Therefore exactly:",
        "||Sigma_E||_F = C_E * L1_E",
        "",
        "For scaled perturbation lambda V:",
        "Delta H_eff^(2)(lambda) = lambda^2 Sigma_E",
        "||Delta H_eff^(2)(lambda)||_F = lambda^2 C_E L1_E",
        "",
        "Finite-lambda regularized Feshbach expansion:",
        "F_exact(lambda) = lambda^2 Sigma_E + O(lambda^3)",
        "",
        "VALIDATION",
        "----------",
    ]

    for family in base.FAMILIES:
        d = all_diag[family]
        lines += [
            f"[{family}]",
            (
                "  max grouped-vs-resolvent Sigma residual = "
                f"{d['max_sigma_equiv_residual']:.6e}"
            ),
            (
                "  max C^2-(S+A) residual                  = "
                f"{d['max_identity_residual']:.6e}"
            ),
            (
                "  max ||Sigma||-C*L1 residual             = "
                f"{d['max_norm_law_residual']:.6e}"
            ),
            (
                "  median absolute-error slope             = "
                f"{d['median_abs_error_slope']:+.4f} "
                "(target ~3)"
            ),
            (
                "  median relative-error slope             = "
                f"{d['median_rel_error_slope']:+.4f} "
                "(target ~1)"
            ),
            (
                "  median relative error @ smallest lambda  = "
                f"{d['median_smallest_lambda_rel_error']:.6e}"
            ),
            "",
        ]

    lines += [
        "INTERPRETATION",
        "--------------",
        "At fixed virtual-pathway budget L1_E, C_E is the exact multiplicative",
        "coherence factor controlling the norm of the leading second-order",
        "effective P-space response.",
        "",
        "This supplies the analytic mechanism:",
        "coherent virtual pathways -> larger ||Sigma_E|| -> larger effective",
        "second-order local response.",
        "",
        "It does not by itself prove an exact proportionality between hotspot",
        "prominence and C_E; that remains an empirical/statistical relation.",
    ]

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("WROTE")
    print(f"  {rows_path}")
    print(f"  {lambda_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
