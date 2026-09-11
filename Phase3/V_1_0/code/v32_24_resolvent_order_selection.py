#!/usr/bin/env python3
"""
Soft Spaces Phase 3.2 — v32.24 Resolvent Order Selection
========================================================

Purpose
-------
Test whether the finite-lambda Feshbach response obeys a family-dependent
perturbative order-selection rule.

For a fixed target eigenspace P of H0, let Q = I-P and define, in the H0
eigenbasis,

    B = Q V P,
    W = Q V Q,

    R0 = (E I_Q - Q H0 Q + i eta I_Q)^(-1).

The regularized effective second-order response at finite lambda is

    F(lambda)
      = lambda^2 Herm[
            B^dagger
            (E I_Q - Q H0 Q - lambda W + i eta I_Q)^(-1)
            B
        ],

where Herm[X] = (X + X^dagger)/2.

Using the Neumann expansion,

    (D - lambda W)^(-1)
      = R0
        + lambda R0 W R0
        + lambda^2 R0 W R0 W R0
        + ...

we obtain

    F(lambda)
      = lambda^2 K2
        + lambda^3 K3
        + lambda^4 K4
        + O(lambda^5),

with

    K2 = Herm[B^dagger R0 B],

    K3 = Herm[B^dagger R0 W R0 B],

    K4 = Herm[B^dagger R0 W R0 W R0 B].

v32.23 suggested that the transverse family may suppress K3, because its
K2-only absolute truncation error scaled approximately as lambda^4 instead of
the generic lambda^3.

v32.24 directly tests that operator statement.

Primary questions
-----------------
1. Is ||K3|| strongly suppressed for transverse perturbations?
2. Is K3 exactly/near-zero at machine precision, or only numerically small?
3. Does the K2-only finite-lambda truncation scale with the first non-negligible
   omitted coefficient?
4. Does adding K3 and K4 restore the expected perturbative convergence order?

This is an operator test, not feature fitting.

Frozen pilot
------------
    * 8 qubits
    * 11 Hamiltonian terms
    * default seed 25042000
    * both perturbation families
    * all exact +/-E pairs
    * eta = frozen ENERGY_REG
    * lambdas = 1e-2, 3e-3, 1e-3, 3e-4

Dependencies
------------
Must be beside this file:

    v32_14_degeneracy_closed.py
    v32_17_perturbative_response.py

Outputs
-------
    v32_24_group_orders.csv
    v32_24_pair_orders.csv
    v32_24_lambda_checks.csv
    v32_24_summary.txt

Usage
-----
    python .\\v32_24_resolvent_order_selection.py

Optional:
    python .\\v32_24_resolvent_order_selection.py --seed 25042001
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


VERSION = "v32.24"
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
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def frob(a: np.ndarray) -> float:
    return float(np.linalg.norm(a, ord="fro"))


def herm(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def log_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    pts = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if float(x) > 0.0 and float(y) > 0.0 and np.isfinite(y)
    ]

    if len(pts) < 2:
        return float("nan")

    lx = np.log([x for x, _ in pts])
    ly = np.log([y for _, y in pts])
    return float(np.polyfit(lx, ly, 1)[0])


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


def spearman(base, x, y) -> float:
    return float(
        base.spearman_np(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )
    )


# ---------------------------------------------------------------------------
# Resolvent objects and coefficients
# ---------------------------------------------------------------------------

def family_operator_in_eigenbasis(base, ref: dict, family: str) -> np.ndarray:
    perturbation = ref["perturbations"][family]
    V = base.dense_pauli_sum(base.N_QUBITS, perturbation)
    U = np.asarray(ref["evecs"], dtype=np.complex128)
    return U.conj().T @ V @ U


def target_blocks(ref: dict, V_eig: np.ndarray, gi: int, eta: float):
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

    return E, Eq, B, W, rdiag


def apply_R0_left(rdiag: np.ndarray, x: np.ndarray) -> np.ndarray:
    return rdiag[:, None] * x


def perturbative_coefficients(
    ref: dict,
    V_eig: np.ndarray,
    gi: int,
    eta: float,
):
    """
    Compute K2, K3, K4 without constructing dense diagonal R0.
    """
    _, _, B, W, rdiag = target_blocks(ref, V_eig, gi, eta)

    R0B = apply_R0_left(rdiag, B)
    K2c = B.conj().T @ R0B

    WR0B = W @ R0B
    R0WR0B = apply_R0_left(rdiag, WR0B)
    K3c = B.conj().T @ R0WR0B

    WR0WR0B = W @ R0WR0B
    R0WR0WR0B = apply_R0_left(rdiag, WR0WR0B)
    K4c = B.conj().T @ R0WR0WR0B

    return herm(K2c), herm(K3c), herm(K4c)


def exact_response(
    ref: dict,
    V_eig: np.ndarray,
    gi: int,
    eta: float,
    lam: float,
) -> np.ndarray:
    """
    Exact finite-lambda regularized Feshbach response:
      lambda^2 Herm[B^dagger (D - lambda W)^(-1) B].
    """
    E, Eq, B, W, _ = target_blocks(ref, V_eig, gi, eta)

    D = np.diag(E - Eq + 1j * eta)
    A = D - lam * W

    X = np.linalg.solve(A, B)
    return (lam * lam) * herm(B.conj().T @ X)


# ---------------------------------------------------------------------------
# Group analysis
# ---------------------------------------------------------------------------

def analyze_group(
    ref: dict,
    V_eig: np.ndarray,
    gi: int,
    eta: float,
    lambdas: List[float],
):
    K2, K3, K4 = perturbative_coefficients(
        ref=ref,
        V_eig=V_eig,
        gi=gi,
        eta=eta,
    )

    n2 = frob(K2)
    n3 = frob(K3)
    n4 = frob(K4)

    ratio32 = n3 / max(n2, DENOM_EPS)
    ratio42 = n4 / max(n2, DENOM_EPS)

    errs_k2 = []
    errs_k23 = []
    errs_k234 = []
    rels_k2 = []
    rels_k23 = []
    rels_k234 = []
    lambda_rows = []

    for lam in lambdas:
        exact = exact_response(
            ref=ref,
            V_eig=V_eig,
            gi=gi,
            eta=eta,
            lam=lam,
        )

        approx2 = (lam ** 2) * K2
        approx23 = approx2 + (lam ** 3) * K3
        approx234 = approx23 + (lam ** 4) * K4

        exact_norm = frob(exact)

        e2 = frob(exact - approx2)
        e23 = frob(exact - approx23)
        e234 = frob(exact - approx234)

        r2 = e2 / max(exact_norm, DENOM_EPS)
        r23 = e23 / max(exact_norm, DENOM_EPS)
        r234 = e234 / max(exact_norm, DENOM_EPS)

        errs_k2.append(e2)
        errs_k23.append(e23)
        errs_k234.append(e234)

        rels_k2.append(r2)
        rels_k23.append(r23)
        rels_k234.append(r234)

        lambda_rows.append(
            {
                "version": VERSION,
                "seed": int(ref["seed"]),
                "family": "",
                "group": int(gi),
                "lambda": float(lam),
                "exact_norm": float(exact_norm),
                "k2_term_norm": float((lam ** 2) * n2),
                "k3_term_norm": float((lam ** 3) * n3),
                "k4_term_norm": float((lam ** 4) * n4),
                "abs_err_K2": float(e2),
                "abs_err_K2K3": float(e23),
                "abs_err_K2K3K4": float(e234),
                "rel_err_K2": float(r2),
                "rel_err_K2K3": float(r23),
                "rel_err_K2K3K4": float(r234),
            }
        )

    out = {
        "K2_norm": float(n2),
        "K3_norm": float(n3),
        "K4_norm": float(n4),
        "K3_over_K2": float(ratio32),
        "K4_over_K2": float(ratio42),

        "slope_abs_K2": log_slope(lambdas, errs_k2),
        "slope_abs_K2K3": log_slope(lambdas, errs_k23),
        "slope_abs_K2K3K4": log_slope(lambdas, errs_k234),

        "slope_rel_K2": log_slope(lambdas, rels_k2),
        "slope_rel_K2K3": log_slope(lambdas, rels_k23),
        "slope_rel_K2K3K4": log_slope(lambdas, rels_k234),

        "smallest_lambda_rel_K2": float(rels_k2[-1]),
        "smallest_lambda_rel_K2K3": float(rels_k23[-1]),
        "smallest_lambda_rel_K2K3K4": float(rels_k234[-1]),
    }

    return out, lambda_rows


# ---------------------------------------------------------------------------
# Family / pair analysis
# ---------------------------------------------------------------------------

def analyze_family(base, ref: dict, family: str, lambdas: List[float]):
    V_eig = family_operator_in_eigenbasis(base, ref, family)
    eta = float(base.ENERGY_REG)

    pairs = pair_records(ref)

    needed_groups = sorted(
        set(p["g_minus"] for p in pairs)
        | set(p["g_plus"] for p in pairs)
    )

    group_rows = []
    lambda_rows = []
    group_data: Dict[int, dict] = {}

    for gi in needed_groups:
        gd, lrows = analyze_group(
            ref=ref,
            V_eig=V_eig,
            gi=gi,
            eta=eta,
            lambdas=lambdas,
        )

        gd["group"] = int(gi)
        group_data[gi] = gd

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
                **{k: v for k, v in gd.items() if k != "group"},
            }
        )

        for r in lrows:
            r["family"] = family
            lambda_rows.append(r)

    pair_rows = []

    for p in pairs:
        gm = p["g_minus"]
        gp = p["g_plus"]

        dm = group_data[gm]
        dp = group_data[gp]

        pair_rows.append(
            {
                "version": VERSION,
                "seed": int(ref["seed"]),
                "family": family,
                "g_minus": int(gm),
                "g_plus": int(gp),
                "E_abs": float(p["E_abs"]),
                "prominence": float(p["prominence"]),

                "K2_norm": 0.5 * (
                    dm["K2_norm"] + dp["K2_norm"]
                ),
                "K3_norm": 0.5 * (
                    dm["K3_norm"] + dp["K3_norm"]
                ),
                "K4_norm": 0.5 * (
                    dm["K4_norm"] + dp["K4_norm"]
                ),
                "K3_over_K2": 0.5 * (
                    dm["K3_over_K2"] + dp["K3_over_K2"]
                ),
                "K4_over_K2": 0.5 * (
                    dm["K4_over_K2"] + dp["K4_over_K2"]
                ),

                "slope_abs_K2": 0.5 * (
                    dm["slope_abs_K2"] + dp["slope_abs_K2"]
                ),
                "slope_abs_K2K3": 0.5 * (
                    dm["slope_abs_K2K3"] + dp["slope_abs_K2K3"]
                ),
                "slope_abs_K2K3K4": 0.5 * (
                    dm["slope_abs_K2K3K4"] + dp["slope_abs_K2K3K4"]
                ),
            }
        )

    summary = {
        "n_groups": len(needed_groups),
        "n_pairs": len(pair_rows),

        "median_K2_norm": float(
            np.median([r["K2_norm"] for r in group_rows])
        ),
        "median_K3_norm": float(
            np.median([r["K3_norm"] for r in group_rows])
        ),
        "median_K4_norm": float(
            np.median([r["K4_norm"] for r in group_rows])
        ),

        "median_K3_over_K2": float(
            np.median([r["K3_over_K2"] for r in group_rows])
        ),
        "max_K3_over_K2": float(
            np.max([r["K3_over_K2"] for r in group_rows])
        ),
        "median_K4_over_K2": float(
            np.median([r["K4_over_K2"] for r in group_rows])
        ),

        "median_slope_abs_K2": float(
            np.nanmedian([r["slope_abs_K2"] for r in group_rows])
        ),
        "median_slope_abs_K2K3": float(
            np.nanmedian([r["slope_abs_K2K3"] for r in group_rows])
        ),
        "median_slope_abs_K2K3K4": float(
            np.nanmedian([r["slope_abs_K2K3K4"] for r in group_rows])
        ),

        "median_slope_rel_K2": float(
            np.nanmedian([r["slope_rel_K2"] for r in group_rows])
        ),
        "median_slope_rel_K2K3": float(
            np.nanmedian([r["slope_rel_K2K3"] for r in group_rows])
        ),
        "median_slope_rel_K2K3K4": float(
            np.nanmedian([r["slope_rel_K2K3K4"] for r in group_rows])
        ),

        "rho_prominence_K3_over_K2": spearman(
            base,
            [r["K3_over_K2"] for r in pair_rows],
            [r["prominence"] for r in pair_rows],
        ),
        "rho_prominence_K4_over_K2": spearman(
            base,
            [r["K4_over_K2"] for r in pair_rows],
            [r["prominence"] for r in pair_rows],
        ),
    }

    return group_rows, pair_rows, lambda_rows, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Soft Spaces Phase 3.2 v32.24 — direct resolvent order selection."
        )
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
        help="Output directory. Default current directory.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    base = load_module(
        "v32_14_degeneracy_closed.py",
        "v3214_for_v3224",
    )
    v3217 = load_module(
        "v32_17_perturbative_response.py",
        "v3217_for_v3224",
    )

    if int(base.N_QUBITS) != 8:
        raise RuntimeError(
            f"v32.24 pilot is frozen for 8Q; base reports {base.N_QUBITS}Q."
        )

    if int(base.N_TERMS) != 11:
        raise RuntimeError(
            f"v32.24 expects 11 Hamiltonian terms; base reports {base.N_TERMS}."
        )

    lambdas = sorted(
        [float(x) for x in args.lambdas if float(x) > 0.0],
        reverse=True,
    )

    if len(lambdas) < 3:
        raise ValueError(
            "Use at least three positive lambda values for slope estimation."
        )

    ref = v3217.build_reference(base, int(args.seed))
    ref["seed"] = int(args.seed)

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    all_group_rows = []
    all_pair_rows = []
    all_lambda_rows = []
    summaries = {}

    print(f"{VERSION}: Resolvent Order Selection")
    print(
        f"seed={args.seed}, qubits={base.N_QUBITS}, "
        f"terms={base.N_TERMS}"
    )
    print("lambdas=" + ",".join(f"{x:g}" for x in lambdas))
    print()
    print("Expansion:")
    print("  F(lambda) = lambda^2 K2 + lambda^3 K3 + lambda^4 K4 + ...")
    print()

    for family in base.FAMILIES:
        print(f"[{family}] ...", flush=True)

        grows, prows, lrows, summary = analyze_family(
            base=base,
            ref=ref,
            family=family,
            lambdas=lambdas,
        )

        all_group_rows.extend(grows)
        all_pair_rows.extend(prows)
        all_lambda_rows.extend(lrows)
        summaries[family] = summary

        print(
            f"  median ||K3||/||K2||      = "
            f"{summary['median_K3_over_K2']:.6e}"
        )
        print(
            f"  max    ||K3||/||K2||      = "
            f"{summary['max_K3_over_K2']:.6e}"
        )
        print(
            f"  median ||K4||/||K2||      = "
            f"{summary['median_K4_over_K2']:.6e}"
        )
        print(
            f"  median abs slope, K2 only = "
            f"{summary['median_slope_abs_K2']:+.4f}"
        )
        print(
            f"  median abs slope, K2+K3   = "
            f"{summary['median_slope_abs_K2K3']:+.4f}"
        )
        print(
            f"  median abs slope, +K4     = "
            f"{summary['median_slope_abs_K2K3K4']:+.4f}"
        )
        print()

    group_path = outdir / "v32_24_group_orders.csv"
    pair_path = outdir / "v32_24_pair_orders.csv"
    lambda_path = outdir / "v32_24_lambda_checks.csv"
    summary_path = outdir / "v32_24_summary.txt"

    write_csv(group_path, all_group_rows)
    write_csv(pair_path, all_pair_rows)
    write_csv(lambda_path, all_lambda_rows)

    lines = [
        f"Soft Spaces Phase 3.2 {VERSION}",
        "Resolvent Order Selection",
        "",
        f"seed={args.seed}",
        f"qubits={base.N_QUBITS}",
        f"dimension={base.DIM}",
        f"hamiltonian_terms={base.N_TERMS}",
        "lambdas=" + ",".join(f"{x:g}" for x in lambdas),
        "",
        "EXPANSION",
        "---------",
        "F(lambda) = lambda^2 K2 + lambda^3 K3 + lambda^4 K4 + O(lambda^5)",
        "",
        "K2 = Herm[B^dagger R0 B]",
        "K3 = Herm[B^dagger R0 W R0 B]",
        "K4 = Herm[B^dagger R0 W R0 W R0 B]",
        "",
        "PRIMARY QUESTION",
        "----------------",
        "Is K3 structurally suppressed, especially for transverse perturbations?",
        "",
        "RESULTS",
        "-------",
    ]

    for family in base.FAMILIES:
        s = summaries[family]

        lines += [
            f"[{family}]",
            f"  n_groups={s['n_groups']}",
            f"  n_pairs={s['n_pairs']}",
            f"  median ||K2||             = {s['median_K2_norm']:.6e}",
            f"  median ||K3||             = {s['median_K3_norm']:.6e}",
            f"  median ||K4||             = {s['median_K4_norm']:.6e}",
            f"  median ||K3||/||K2||      = {s['median_K3_over_K2']:.6e}",
            f"  max    ||K3||/||K2||      = {s['max_K3_over_K2']:.6e}",
            f"  median ||K4||/||K2||      = {s['median_K4_over_K2']:.6e}",
            "",
            f"  median abs-error slope K2       = {s['median_slope_abs_K2']:+.4f}",
            f"  median abs-error slope K2+K3    = {s['median_slope_abs_K2K3']:+.4f}",
            f"  median abs-error slope K2+K3+K4 = {s['median_slope_abs_K2K3K4']:+.4f}",
            "",
            f"  median rel-error slope K2       = {s['median_slope_rel_K2']:+.4f}",
            f"  median rel-error slope K2+K3    = {s['median_slope_rel_K2K3']:+.4f}",
            f"  median rel-error slope K2+K3+K4 = {s['median_slope_rel_K2K3K4']:+.4f}",
            "",
            (
                "  rho(prominence, ||K3||/||K2||) = "
                f"{s['rho_prominence_K3_over_K2']:+.4f}"
            ),
            (
                "  rho(prominence, ||K4||/||K2||) = "
                f"{s['rho_prominence_K4_over_K2']:+.4f}"
            ),
            "",
        ]

    lines += [
        "DECISION RULE",
        "-------------",
        "A genuine transverse order-selection rule is supported if:",
        "",
        "  1. transverse ||K3||/||K2|| is near numerical zero or is",
        "     parametrically much smaller than the dephasing value;",
        "",
        "  2. transverse K2-only truncation shows ~lambda^4 absolute error;",
        "",
        "  3. adding K4 removes that leading error and raises the convergence",
        "     order again.",
        "",
        "If K3 is merely small but not structurally suppressed, the v32.23",
        "lambda^4 behavior should be interpreted as a numerical/crossover",
        "effect rather than an exact selection rule.",
    ]

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("WROTE")
    print(f"  {group_path}")
    print(f"  {pair_path}")
    print(f"  {lambda_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
