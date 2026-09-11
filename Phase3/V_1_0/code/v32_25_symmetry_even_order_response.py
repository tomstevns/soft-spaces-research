#!/usr/bin/env python3
"""
Soft Spaces Phase 3.2 — v32.25 Symmetry and Even-Order Response
===============================================================

Purpose
-------
Test the symmetry structure behind the v32.24 observation that the transverse
regularized Feshbach response appears to contain only even powers of lambda.

Frozen reference symmetry (seed 25042000)
-----------------------------------------
The Phase-3 symmetry operator is

    Gamma = XIXXIIXX,

with

    Gamma H0 Gamma = -H0.

For the two frozen perturbation families,

    Gamma V Gamma = s V,

where

    s = -1  dephasing
    s = +1  transverse.

For an exact +/-E pair, Gamma maps

    P_E  -> P_-E,
    Q_E  -> Q_-E.

Define

    B_E = Q_E V P_E,
    W_E = Q_E V Q_E,

and the regularized response

    F_E(lambda)
      = lambda^2 Herm[
            B_E^dagger
            (E I - Q_E H0 Q_E - lambda W_E + i eta I)^(-1)
            B_E
        ].

Because g_eta(x)=x/(x^2+eta^2) is odd, the Gamma symmetry predicts the
paired response identity

    Gamma_P^dagger F_-E(lambda) Gamma_P
      = - F_E(-s lambda).

Expanding

    F_E(lambda) = sum_{m>=2} lambda^m K_m(E)

gives the coefficient relation

    Gamma_P^dagger K_m(-E) Gamma_P
      = -(-s)^m K_m(E).

For transverse (s=+1):

    even m : K_m(-E) = -Gamma K_m(E) Gamma
    odd  m : K_m(-E) = +Gamma K_m(E) Gamma.

For dephasing (s=-1):

    all m : K_m(-E) = -Gamma K_m(E) Gamma.

Important logical distinction
-----------------------------
The paired Gamma identity alone does NOT force K3(E)=0 at a single target E.
Therefore v32.25 tests BOTH:

  A. the exact +/-E Gamma identities, and
  B. direct same-target odd-order suppression (K3 and K5).

If transverse K3 and K5 are both at numerical zero and

    F_E(lambda) ~= F_E(-lambda)

for every tested target, then the numerical evidence supports an even-order
single-target response law

    F_T(lambda)
      = lambda^2 K2 + lambda^4 K4 + lambda^6 K6 + ...

The script deliberately distinguishes:
  * exact symmetry-pairing evidence, from
  * the stronger same-target evenness statement.

No feature fitting is performed.

Frozen pilot
------------
    * 8 qubits
    * 11 Hamiltonian terms
    * seed 25042000
    * Gamma = XIXXIIXX
    * both perturbation families
    * all exact +/-E pairs
    * K2 ... K6
    * lambdas = 1e-2, 3e-3, 1e-3, 3e-4

Dependencies
------------
Must be beside this file:

    v32_14_degeneracy_closed.py
    v32_17_perturbative_response.py

Outputs
-------
    v32_25_group_orders.csv
    v32_25_pair_symmetry.csv
    v32_25_lambda_parity.csv
    v32_25_summary.txt

Usage
-----
    python .\\v32_25_symmetry_even_order_response.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


VERSION = "v32.25"
DEFAULT_GAMMA = "XIXXIIXX"
DENOM_EPS = 1.0e-30


# ---------------------------------------------------------------------------
# Imports
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
# Generic helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def frob(a: np.ndarray) -> float:
    return float(np.linalg.norm(a, ord="fro"))


def herm(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def rel_residual(a: np.ndarray, b: np.ndarray) -> float:
    return frob(a - b) / max(frob(a), frob(b), DENOM_EPS)


def pauli_string_matrix(label: str) -> np.ndarray:
    mats = {
        "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }

    out = np.array([[1.0 + 0.0j]])
    for ch in label:
        if ch not in mats:
            raise ValueError(f"Invalid Pauli character {ch!r} in {label!r}")
        out = np.kron(out, mats[ch])
    return out


def pair_records(ref: dict) -> List[dict]:
    out = []

    for a, b in ref["pairs"]:
        ea = float(ref["energies"][a])
        eb = float(ref["energies"][b])

        gm, gp = (a, b) if ea <= eb else (b, a)

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


# ---------------------------------------------------------------------------
# Blocks and perturbative coefficients
# ---------------------------------------------------------------------------

def family_operator_in_eigenbasis(base, ref: dict, family: str):
    perturbation = ref["perturbations"][family]
    V = base.dense_pauli_sum(base.N_QUBITS, perturbation)
    U = np.asarray(ref["evecs"], dtype=np.complex128)
    V_eig = U.conj().T @ V @ U
    return V, V_eig


def target_blocks(
    ref: dict,
    V_eig: np.ndarray,
    gi: int,
    eta: float,
):
    pidx = np.asarray(ref["groups"][gi], dtype=int)
    all_idx = np.arange(len(ref["evals"]), dtype=int)

    mask = np.ones(len(all_idx), dtype=bool)
    mask[pidx] = False
    qidx = all_idx[mask]

    E = float(np.mean(ref["evals"][pidx]))
    Eq = np.asarray(ref["evals"][qidx], dtype=float)

    B = V_eig[np.ix_(qidx, pidx)]
    W = V_eig[np.ix_(qidx, qidx)]
    rdiag = 1.0 / (E - Eq + 1j * eta)

    return pidx, qidx, E, Eq, B, W, rdiag


def apply_R0(rdiag: np.ndarray, x: np.ndarray) -> np.ndarray:
    return rdiag[:, None] * x


def coefficients_K2_to_K6(
    ref: dict,
    V_eig: np.ndarray,
    gi: int,
    eta: float,
) -> Dict[int, np.ndarray]:
    """
    K_m = Herm[B^dagger R0 (W R0)^(m-2) B], m=2,...,6.
    """
    _, _, _, _, B, W, rdiag = target_blocks(
        ref, V_eig, gi, eta
    )

    x = apply_R0(rdiag, B)
    coeffs = {2: herm(B.conj().T @ x)}

    for m in range(3, 7):
        x = apply_R0(rdiag, W @ x)
        coeffs[m] = herm(B.conj().T @ x)

    return coeffs


def exact_response(
    ref: dict,
    V_eig: np.ndarray,
    gi: int,
    eta: float,
    lam: float,
) -> np.ndarray:
    _, _, E, Eq, B, W, _ = target_blocks(
        ref, V_eig, gi, eta
    )

    D = np.diag(E - Eq + 1j * eta)
    X = np.linalg.solve(D - lam * W, B)

    return (lam * lam) * herm(B.conj().T @ X)


# ---------------------------------------------------------------------------
# Gamma maps between +/-E subspaces
# ---------------------------------------------------------------------------

def subspace_gamma_map(
    ref: dict,
    Gamma: np.ndarray,
    g_from: int,
    g_to: int,
) -> np.ndarray:
    """
    Coordinates of Gamma mapping P_from -> P_to:
        M = U_to^dagger Gamma U_from
    """
    U = np.asarray(ref["evecs"], dtype=np.complex128)
    idx_from = np.asarray(ref["groups"][g_from], dtype=int)
    idx_to = np.asarray(ref["groups"][g_to], dtype=int)

    U_from = U[:, idx_from]
    U_to = U[:, idx_to]

    return U_to.conj().T @ Gamma @ U_from


def projector_from_group(ref: dict, gi: int) -> np.ndarray:
    U = np.asarray(ref["evecs"], dtype=np.complex128)
    idx = np.asarray(ref["groups"][gi], dtype=int)
    Ug = U[:, idx]
    return Ug @ Ug.conj().T


# ---------------------------------------------------------------------------
# Main family analysis
# ---------------------------------------------------------------------------

def analyze_family(
    base,
    ref: dict,
    H: np.ndarray,
    Gamma: np.ndarray,
    family: str,
    lambdas: List[float],
):
    s = -1 if family == "dephasing" else +1

    V, V_eig = family_operator_in_eigenbasis(base, ref, family)

    gamma_V_resid = rel_residual(
        Gamma @ V @ Gamma.conj().T,
        s * V,
    )

    eta = float(base.ENERGY_REG)
    pairs = pair_records(ref)

    needed_groups = sorted(
        set(p["g_minus"] for p in pairs)
        | set(p["g_plus"] for p in pairs)
    )

    coeff_by_group: Dict[int, Dict[int, np.ndarray]] = {}
    group_rows: List[dict] = []

    for gi in needed_groups:
        coeffs = coefficients_K2_to_K6(
            ref=ref,
            V_eig=V_eig,
            gi=gi,
            eta=eta,
        )
        coeff_by_group[gi] = coeffs

        n2 = frob(coeffs[2])

        group_rows.append(
            {
                "version": VERSION,
                "seed": int(ref["seed"]),
                "family": family,
                "group": int(gi),
                "energy": float(
                    np.mean(ref["evals"][ref["groups"][gi]])
                ),
                "group_dim": int(len(ref["groups"][gi])),
                "K2_norm": float(n2),
                "K3_norm": float(frob(coeffs[3])),
                "K4_norm": float(frob(coeffs[4])),
                "K5_norm": float(frob(coeffs[5])),
                "K6_norm": float(frob(coeffs[6])),
                "K3_over_K2": float(
                    frob(coeffs[3]) / max(n2, DENOM_EPS)
                ),
                "K5_over_K2": float(
                    frob(coeffs[5]) / max(n2, DENOM_EPS)
                ),
            }
        )

    pair_rows: List[dict] = []
    lambda_rows: List[dict] = []

    for p in pairs:
        gm = p["g_minus"]
        gp = p["g_plus"]

        Gp = subspace_gamma_map(
            ref=ref,
            Gamma=Gamma,
            g_from=gp,
            g_to=gm,
        )

        dim_p = Gp.shape[1]
        I_p = np.eye(dim_p, dtype=np.complex128)
        gamma_map_unitarity = frob(
            Gp.conj().T @ Gp - I_p
        )

        Pp = projector_from_group(ref, gp)
        Pm = projector_from_group(ref, gm)
        projector_map_resid = frob(
            Pm - Gamma @ Pp @ Gamma.conj().T
        )

        row = {
            "version": VERSION,
            "seed": int(ref["seed"]),
            "family": family,
            "g_minus": int(gm),
            "g_plus": int(gp),
            "E_abs": float(p["E_abs"]),
            "prominence": float(p["prominence"]),
            "gamma_map_unitarity_residual": float(gamma_map_unitarity),
            "projector_map_residual": float(projector_map_resid),
        }

        # Coefficient symmetry:
        # Gp^dag K_m(-E) Gp = -(-s)^m K_m(+E)
        for m in range(2, 7):
            km_minus_in_plus = (
                Gp.conj().T
                @ coeff_by_group[gm][m]
                @ Gp
            )

            predicted = -((-s) ** m) * coeff_by_group[gp][m]

            row[f"K{m}_pair_symmetry_residual"] = rel_residual(
                km_minus_in_plus,
                predicted,
            )

        # Same-target odd suppression, pair averaged.
        k2mean = 0.5 * (
            frob(coeff_by_group[gm][2])
            + frob(coeff_by_group[gp][2])
        )
        k3mean = 0.5 * (
            frob(coeff_by_group[gm][3])
            + frob(coeff_by_group[gp][3])
        )
        k5mean = 0.5 * (
            frob(coeff_by_group[gm][5])
            + frob(coeff_by_group[gp][5])
        )

        row["K3_over_K2_pair"] = float(
            k3mean / max(k2mean, DENOM_EPS)
        )
        row["K5_over_K2_pair"] = float(
            k5mean / max(k2mean, DENOM_EPS)
        )

        pair_rows.append(row)

        # Exact finite-lambda paired identity and same-target lambda parity.
        for lam in lambdas:
            Fplus_pos = exact_response(
                ref, V_eig, gp, eta, +lam
            )
            Fplus_neg = exact_response(
                ref, V_eig, gp, eta, -lam
            )
            Fminus_pos = exact_response(
                ref, V_eig, gm, eta, +lam
            )

            mapped_minus = (
                Gp.conj().T @ Fminus_pos @ Gp
            )

            # Gamma prediction:
            # mapped F_-E(lambda) = -F_+E(-s lambda)
            rhs = (
                -Fplus_neg
                if s == +1
                else -Fplus_pos
            )

            pair_exact_resid = rel_residual(
                mapped_minus,
                rhs,
            )

            # Stronger same-target evenness test.
            parity_resid = rel_residual(
                Fplus_pos,
                Fplus_neg,
            )

            lambda_rows.append(
                {
                    "version": VERSION,
                    "seed": int(ref["seed"]),
                    "family": family,
                    "g_minus": int(gm),
                    "g_plus": int(gp),
                    "E_abs": float(p["E_abs"]),
                    "lambda": float(lam),
                    "paired_gamma_response_residual": float(
                        pair_exact_resid
                    ),
                    "same_target_evenness_residual": float(
                        parity_resid
                    ),
                }
            )

    summary = {
        "gamma_V_residual": float(gamma_V_resid),

        "max_projector_map_residual": float(
            max(r["projector_map_residual"] for r in pair_rows)
        ),
        "max_gamma_map_unitarity_residual": float(
            max(r["gamma_map_unitarity_residual"] for r in pair_rows)
        ),

        "max_K2_pair_symmetry_residual": float(
            max(r["K2_pair_symmetry_residual"] for r in pair_rows)
        ),
        "max_K3_pair_symmetry_residual": float(
            max(r["K3_pair_symmetry_residual"] for r in pair_rows)
        ),
        "max_K4_pair_symmetry_residual": float(
            max(r["K4_pair_symmetry_residual"] for r in pair_rows)
        ),
        "max_K5_pair_symmetry_residual": float(
            max(r["K5_pair_symmetry_residual"] for r in pair_rows)
        ),
        "max_K6_pair_symmetry_residual": float(
            max(r["K6_pair_symmetry_residual"] for r in pair_rows)
        ),

        "median_K3_over_K2": float(
            np.median([r["K3_over_K2"] for r in group_rows])
        ),
        "max_K3_over_K2": float(
            np.max([r["K3_over_K2"] for r in group_rows])
        ),
        "median_K5_over_K2": float(
            np.median([r["K5_over_K2"] for r in group_rows])
        ),
        "max_K5_over_K2": float(
            np.max([r["K5_over_K2"] for r in group_rows])
        ),

        "max_paired_gamma_response_residual": float(
            max(
                r["paired_gamma_response_residual"]
                for r in lambda_rows
            )
        ),
        "median_same_target_evenness_residual": float(
            np.median(
                [
                    r["same_target_evenness_residual"]
                    for r in lambda_rows
                ]
            )
        ),
        "max_same_target_evenness_residual": float(
            max(
                r["same_target_evenness_residual"]
                for r in lambda_rows
            )
        ),
    }

    return group_rows, pair_rows, lambda_rows, summary


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Soft Spaces Phase 3.2 v32.25 — Gamma symmetry and "
            "same-target even-order response."
        )
    )

    p.add_argument(
        "--seed",
        type=int,
        default=25_042_000,
        help="Frozen symmetry seed. Default 25042000.",
    )

    p.add_argument(
        "--gamma",
        type=str,
        default=DEFAULT_GAMMA,
        help=f"Pauli-string Gamma. Default {DEFAULT_GAMMA}.",
    )

    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[1e-2, 3e-3, 1e-3, 3e-4],
        help="Positive lambda values.",
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
        "v3214_for_v3225",
    )
    v3217 = load_module(
        "v32_17_perturbative_response.py",
        "v3217_for_v3225",
    )

    if int(base.N_QUBITS) != 8:
        raise RuntimeError(
            f"v32.25 is frozen for 8Q; base reports {base.N_QUBITS}Q."
        )

    if int(base.N_TERMS) != 11:
        raise RuntimeError(
            f"v32.25 expects 11 Hamiltonian terms; base reports "
            f"{base.N_TERMS}."
        )

    if int(args.seed) != 25_042_000:
        print(
            "WARNING: Gamma=XIXXIIXX was established for seed 25042000. "
            "A different seed is exploratory unless Gamma was independently "
            "verified for that seed.",
            flush=True,
        )

    gamma_label = args.gamma.strip().upper()
    if len(gamma_label) != int(base.N_QUBITS):
        raise ValueError(
            f"Gamma string has length {len(gamma_label)}, "
            f"expected {base.N_QUBITS}."
        )

    lambdas = sorted(
        [float(x) for x in args.lambdas if float(x) > 0.0],
        reverse=True,
    )

    if not lambdas:
        raise ValueError("Need at least one positive lambda.")

    ref = v3217.build_reference(base, int(args.seed))
    ref["seed"] = int(args.seed)

    U = np.asarray(ref["evecs"], dtype=np.complex128)
    H = U @ np.diag(np.asarray(ref["evals"], dtype=float)) @ U.conj().T

    Gamma = pauli_string_matrix(gamma_label)

    gamma_unitarity = frob(
        Gamma.conj().T @ Gamma
        - np.eye(Gamma.shape[0], dtype=np.complex128)
    )
    gamma_square = frob(
        Gamma @ Gamma
        - np.eye(Gamma.shape[0], dtype=np.complex128)
    )
    gamma_H_resid = rel_residual(
        Gamma @ H @ Gamma.conj().T,
        -H,
    )

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"{VERSION}: Symmetry and Even-Order Response")
    print(
        f"seed={args.seed}, qubits={base.N_QUBITS}, "
        f"terms={base.N_TERMS}"
    )
    print(f"Gamma={gamma_label}")
    print(
        "Gamma H Gamma = -H residual = "
        f"{gamma_H_resid:.6e}"
    )
    print()

    all_group_rows = []
    all_pair_rows = []
    all_lambda_rows = []
    summaries = {}

    for family in base.FAMILIES:
        print(f"[{family}] ...", flush=True)

        grows, prows, lrows, summary = analyze_family(
            base=base,
            ref=ref,
            H=H,
            Gamma=Gamma,
            family=family,
            lambdas=lambdas,
        )

        all_group_rows.extend(grows)
        all_pair_rows.extend(prows)
        all_lambda_rows.extend(lrows)
        summaries[family] = summary

        print(
            f"  Gamma V Gamma parity residual       = "
            f"{summary['gamma_V_residual']:.6e}"
        )
        print(
            f"  median ||K3||/||K2||                = "
            f"{summary['median_K3_over_K2']:.6e}"
        )
        print(
            f"  median ||K5||/||K2||                = "
            f"{summary['median_K5_over_K2']:.6e}"
        )
        print(
            f"  max paired F symmetry residual      = "
            f"{summary['max_paired_gamma_response_residual']:.6e}"
        )
        print(
            f"  median same-target evenness residual= "
            f"{summary['median_same_target_evenness_residual']:.6e}"
        )
        print()

    group_path = outdir / "v32_25_group_orders.csv"
    pair_path = outdir / "v32_25_pair_symmetry.csv"
    lambda_path = outdir / "v32_25_lambda_parity.csv"
    summary_path = outdir / "v32_25_summary.txt"

    write_csv(group_path, all_group_rows)
    write_csv(pair_path, all_pair_rows)
    write_csv(lambda_path, all_lambda_rows)

    lines = [
        f"Soft Spaces Phase 3.2 {VERSION}",
        "Symmetry and Even-Order Response",
        "",
        f"seed={args.seed}",
        f"qubits={base.N_QUBITS}",
        f"dimension={base.DIM}",
        f"hamiltonian_terms={base.N_TERMS}",
        f"Gamma={gamma_label}",
        "lambdas=" + ",".join(f"{x:g}" for x in lambdas),
        "",
        "GLOBAL GAMMA CHECK",
        "------------------",
        f"||Gamma^dag Gamma-I||_F = {gamma_unitarity:.6e}",
        f"||Gamma^2-I||_F         = {gamma_square:.6e}",
        f"relres(Gamma H Gamma,-H)= {gamma_H_resid:.6e}",
        "",
        "PAIRED RESPONSE IDENTITY",
        "------------------------",
        "For Gamma V Gamma = s V:",
        "",
        "Gamma_P^dag F_-E(lambda) Gamma_P = -F_E(-s lambda)",
        "",
        "and therefore",
        "",
        "Gamma_P^dag K_m(-E) Gamma_P = -(-s)^m K_m(E).",
        "",
        "RESULTS",
        "-------",
    ]

    for family in base.FAMILIES:
        s = summaries[family]
        parity_name = "-1" if family == "dephasing" else "+1"

        lines += [
            f"[{family}]",
            f"  expected Gamma-V parity s                 = {parity_name}",
            (
                "  relres(Gamma V Gamma, s V)               = "
                f"{s['gamma_V_residual']:.6e}"
            ),
            (
                "  max projector-map residual               = "
                f"{s['max_projector_map_residual']:.6e}"
            ),
            (
                "  max Gamma-P map unitarity residual       = "
                f"{s['max_gamma_map_unitarity_residual']:.6e}"
            ),
            "",
            (
                "  max K2 paired-symmetry residual          = "
                f"{s['max_K2_pair_symmetry_residual']:.6e}"
            ),
            (
                "  max K3 paired-symmetry residual          = "
                f"{s['max_K3_pair_symmetry_residual']:.6e}"
            ),
            (
                "  max K4 paired-symmetry residual          = "
                f"{s['max_K4_pair_symmetry_residual']:.6e}"
            ),
            (
                "  max K5 paired-symmetry residual          = "
                f"{s['max_K5_pair_symmetry_residual']:.6e}"
            ),
            (
                "  max K6 paired-symmetry residual          = "
                f"{s['max_K6_pair_symmetry_residual']:.6e}"
            ),
            "",
            (
                "  median ||K3||/||K2||                     = "
                f"{s['median_K3_over_K2']:.6e}"
            ),
            (
                "  max    ||K3||/||K2||                     = "
                f"{s['max_K3_over_K2']:.6e}"
            ),
            (
                "  median ||K5||/||K2||                     = "
                f"{s['median_K5_over_K2']:.6e}"
            ),
            (
                "  max    ||K5||/||K2||                     = "
                f"{s['max_K5_over_K2']:.6e}"
            ),
            "",
            (
                "  max paired finite-lambda symmetry resid  = "
                f"{s['max_paired_gamma_response_residual']:.6e}"
            ),
            (
                "  median same-target evenness residual     = "
                f"{s['median_same_target_evenness_residual']:.6e}"
            ),
            (
                "  max same-target evenness residual        = "
                f"{s['max_same_target_evenness_residual']:.6e}"
            ),
            "",
        ]

    lines += [
        "DECISION LOGIC",
        "--------------",
        "1. Machine-precision paired residuals establish the Gamma-based",
        "   +/-E response identity.",
        "",
        "2. That paired identity alone does NOT prove K3(E)=0.",
        "",
        "3. A stronger transverse even-order law is numerically supported only",
        "   if K3 and K5 are both near zero AND F_E(lambda)=F_E(-lambda)",
        "   holds at each individual target.",
        "",
        "4. If those same-target tests pass at numerical precision, the next",
        "   task is an analytic derivation of the additional symmetry/cancellation",
        "   that eliminates the odd orders.",
    ]

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("WROTE")
    print(f"  {group_path}")
    print(f"  {pair_path}")
    print(f"  {lambda_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
