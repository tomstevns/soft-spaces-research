#!/usr/bin/env python3
"""
Soft Spaces Phase 3.2 — v32.17.1 Numerical Validation
=====================================================

Purpose
-------
Numerically validate the analytic frozen-P/Q derivative used in v32.17:

    dSigma_E/dlambda

without introducing fresh eigendecompositions of QH(lambda)Q.

The validation uses the same frozen P/Q model as v32.17,

    H(lambda) = H0 + lambda V,

with P and Q frozen at lambda = 0, and

    A(lambda) = E_P(lambda) I_Q - Q H(lambda) Q.

The regularized propagator is evaluated directly as

    g_eta(A) = A (A^2 + eta^2 I)^(-1),

so that

    Sigma(lambda) = B^dagger g_eta(A(lambda)) B,
    B = Q V P.

This avoids eigenvector gauge/noise from repeatedly diagonalizing QH(lambda)Q
and gives a cleaner central-difference check of the analytic Frechet derivative.

Primary validation quantities
-----------------------------
For representative +/-E pairs and both perturbation families, compare

    analytic dSigma/dlambda

with

    [Sigma(+delta) - Sigma(-delta)] / (2 delta).

Report:

    * relative Frobenius error in dSigma/dlambda
    * absolute Frobenius error
    * analytic vs finite-difference R_E
    * analytic vs finite-difference D_E_signed
    * convergence slope across delta

Expected behavior
-----------------
For a correct analytic derivative and a numerically stable central difference,
the absolute error should scale approximately as O(delta^2) until floating-point
roundoff becomes dominant.

Dependency
----------
Requires beside this file:

    v32_14_degeneracy_closed.py
    v32_17_perturbative_response.py

Usage
-----
    python .\\v32_17_1_validation.py

or

    python .\\v32_17_1_validation.py --seed 25042000

Optional:
    python .\\v32_17_1_validation.py --deltas 1e-2 3e-3 1e-3 3e-4 1e-4

Outputs
-------
    v32_17_1_validation_seed25042000.csv
    v32_17_1_validation_seed25042000_summary.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


VERSION = "v32.17.1"
DEFAULT_SEED = 25_042_000
DEFAULT_DELTAS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
DEFAULT_VALIDATION_PAIRS = 3


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


def direct_regularized_matrix_function(A: np.ndarray, eta: float) -> np.ndarray:
    """
    Compute

        g_eta(A) = A (A^2 + eta^2 I)^(-1)

    directly, with no eigendecomposition.

    We solve the linear system instead of explicitly forming an inverse.
    For Hermitian A, A commutes with A^2 + eta^2 I, so this is equivalent
    to the scalar regularization g_eta(x) = x/(x^2+eta^2).
    """
    A = 0.5 * (A + A.conj().T)
    n = A.shape[0]
    M = A @ A + (eta * eta) * np.eye(n, dtype=np.complex128)

    # Solve M X = A, hence X = M^{-1} A = A M^{-1}.
    X = np.linalg.solve(M, A)
    X = 0.5 * (X + X.conj().T)
    return X


def sigma_direct(prepared: dict, lam: float, eta: float) -> np.ndarray:
    """
    Frozen-P/Q Sigma(lambda), evaluated without diagonalizing QH(lambda)Q.
    """
    q_energies = np.asarray(prepared["q_energies"], dtype=float)
    v_qq = np.asarray(prepared["v_qq"], dtype=np.complex128)
    b_qp = np.asarray(prepared["b_qp"], dtype=np.complex128)

    z = (
        float(prepared["energy"])
        + float(lam) * float(prepared["target_slope"])
    )

    # A(lambda) = E_P(lambda) I - [diag(E_q) + lambda V_QQ]
    A = (
        np.diag(z - q_energies).astype(np.complex128)
        - float(lam) * v_qq
    )
    A = 0.5 * (A + A.conj().T)

    gA = direct_regularized_matrix_function(A, eta)
    sigma = b_qp.conj().T @ gA @ b_qp
    sigma = 0.5 * (sigma + sigma.conj().T)
    return sigma


def fro_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord="fro"))


def central_difference(prepared: dict, delta: float, eta: float):
    sp = sigma_direct(prepared, +delta, eta)
    sm = sigma_direct(prepared, -delta, eta)
    fd = (sp - sm) / (2.0 * delta)

    fd_D_signed = float(
        (
            fro_norm(sp) ** 2
            - fro_norm(sm) ** 2
        )
        / (2.0 * delta)
    )

    return sp, sm, fd, fd_D_signed


def choose_validation_pairs(ref: dict, count: int) -> List[dict]:
    pairs = []

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
        if np.isfinite(prominence):
            pairs.append(
                {
                    "g_minus": int(gm),
                    "g_plus": int(gp),
                    "prominence": float(prominence),
                    "E_abs": float(abs(ref["energies"][gp])),
                }
            )

    pairs.sort(key=lambda r: r["prominence"])

    if not pairs:
        return []

    if count <= 1:
        return [pairs[len(pairs) // 2]]

    idxs = np.linspace(0, len(pairs) - 1, count)
    selected = []
    seen = set()

    for idx in idxs:
        j = int(round(float(idx)))
        j = max(0, min(len(pairs) - 1, j))
        key = (pairs[j]["g_minus"], pairs[j]["g_plus"])
        if key not in seen:
            selected.append(pairs[j])
            seen.add(key)

    return selected


def convergence_slope(deltas: List[float], errors: List[float]) -> float:
    """
    Log-log slope of absolute error vs delta.

    Ideal central-difference truncation gives slope ~2 before roundoff.
    Only finite positive errors are used.
    """
    xs = []
    ys = []

    for d, e in zip(deltas, errors):
        if d > 0 and e > 0 and np.isfinite(d) and np.isfinite(e):
            xs.append(math.log(d))
            ys.append(math.log(e))

    if len(xs) < 2:
        return float("nan")

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)

    denom = float(np.dot(x0, x0))
    if denom == 0.0:
        return float("nan")

    return float(np.dot(x0, y0) / denom)


def write_csv(path: Path, rows: List[dict]) -> None:
    import csv

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Soft Spaces Phase 3.2 v32.17.1 — direct-matrix-function "
            "validation of the analytic frozen-P/Q derivative."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed to validate (default {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=list(DEFAULT_DELTAS),
        help="Central-difference validation steps.",
    )
    parser.add_argument(
        "--validation-pairs",
        type=int,
        default=DEFAULT_VALIDATION_PAIRS,
        help=(
            "Number of deterministic low/mid/high-prominence +/-E pairs "
            f"(default {DEFAULT_VALIDATION_PAIRS})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (default current directory).",
    )
    args = parser.parse_args()

    deltas = sorted(
        set(float(d) for d in args.deltas),
        reverse=True,
    )
    if not deltas or any((not np.isfinite(d) or d <= 0.0) for d in deltas):
        raise ValueError("All delta values must be finite and > 0.")

    base = load_module(
        "v32_14_degeneracy_closed.py",
        "v3214_base_validation",
    )
    v3217 = load_module(
        "v32_17_perturbative_response.py",
        "v3217_base_validation",
    )

    if int(base.N_QUBITS) != 8:
        raise RuntimeError(
            f"v32.17.1 is frozen for 8Q; base reports {base.N_QUBITS}Q."
        )
    if int(base.N_TERMS) != 11:
        raise RuntimeError(
            f"v32.17.1 expects 11 Hamiltonian terms; "
            f"base reports {base.N_TERMS}."
        )

    eta = float(base.ENERGY_REG)

    print(
        f"{VERSION}: building frozen 8Q reference for seed {args.seed} ..."
    )

    ref = v3217.build_reference(base, int(args.seed))
    selected_pairs = choose_validation_pairs(
        ref, int(args.validation_pairs)
    )

    if not selected_pairs:
        raise RuntimeError("No usable +/-E pairs found.")

    rows: List[dict] = []
    prepared_cache: Dict[tuple, dict] = {}

    for family in base.FAMILIES:
        perturbation = ref["perturbations"][family]

        for pair_rank, pair in enumerate(selected_pairs, start=1):
            for sign, gi in (
                ("minus", pair["g_minus"]),
                ("plus", pair["g_plus"]),
            ):
                cache_key = (family, gi)
                if cache_key not in prepared_cache:
                    prepared_cache[cache_key] = v3217.prepare_target(
                        base,
                        ref,
                        gi,
                        perturbation,
                    )

                prepared = prepared_cache[cache_key]
                analytic = prepared["d_sigma"]
                analytic_R = fro_norm(analytic)
                analytic_D_signed = float(prepared["D_E_signed"])

                abs_errors = []

                for delta in deltas:
                    sp, sm, fd, fd_D_signed = central_difference(
                        prepared, delta, eta
                    )

                    abs_err = fro_norm(fd - analytic)
                    rel_err = (
                        abs_err / analytic_R
                        if analytic_R > 1e-30
                        else float("nan")
                    )

                    abs_errors.append(abs_err)

                    rows.append(
                        {
                            "version": VERSION,
                            "seed": int(args.seed),
                            "family": family,
                            "validation_pair_rank": int(pair_rank),
                            "pair_prominence": float(pair["prominence"]),
                            "E_abs": float(pair["E_abs"]),
                            "sign": sign,
                            "group": int(gi),
                            "delta": float(delta),
                            "analytic_R_E": analytic_R,
                            "fd_R_E": fro_norm(fd),
                            "abs_fro_error_dSigma": abs_err,
                            "rel_fro_error_dSigma": rel_err,
                            "analytic_D_signed": analytic_D_signed,
                            "fd_D_signed": fd_D_signed,
                            "abs_error_D_signed": abs(
                                analytic_D_signed - fd_D_signed
                            ),
                            "sigma_plus_fro": fro_norm(sp),
                            "sigma_minus_fro": fro_norm(sm),
                        }
                    )

                slope = convergence_slope(deltas, abs_errors)
                for row in rows:
                    if (
                        row["family"] == family
                        and row["validation_pair_rank"] == pair_rank
                        and row["sign"] == sign
                        and row["group"] == gi
                    ):
                        row["loglog_error_slope"] = slope

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"v32_17_1_validation_seed{args.seed}.csv"
    txt_path = outdir / f"v32_17_1_validation_seed{args.seed}_summary.txt"

    write_csv(csv_path, rows)

    # Aggregate diagnostics.
    smallest_delta = min(deltas)
    smallest = [
        r for r in rows
        if math.isclose(
            float(r["delta"]),
            smallest_delta,
            rel_tol=0.0,
            abs_tol=1e-18,
        )
    ]

    finite_rel = [
        float(r["rel_fro_error_dSigma"])
        for r in smallest
        if np.isfinite(float(r["rel_fro_error_dSigma"]))
    ]

    finite_abs = [
        float(r["abs_fro_error_dSigma"])
        for r in smallest
        if np.isfinite(float(r["abs_fro_error_dSigma"]))
    ]

    slopes = sorted(
        {
            (
                r["family"],
                r["validation_pair_rank"],
                r["sign"],
                r["group"],
                float(r["loglog_error_slope"]),
            )
            for r in rows
        }
    )

    lines = [
        f"Soft Spaces Phase 3.2 {VERSION}",
        "Numerical validation of analytic frozen-P/Q derivative",
        "",
        f"seed={args.seed}",
        f"qubits={base.N_QUBITS}",
        f"dimension={base.DIM}",
        f"hamiltonian_terms={base.N_TERMS}",
        f"groups={len(ref['groups'])}",
        f"exact_pmE_pairs={len(ref['pairs'])}",
        f"validation_pairs={len(selected_pairs)}",
        f"eta={eta:.6g}",
        "deltas=" + ",".join(f"{d:.6g}" for d in deltas),
        "",
        "METHOD",
        "------",
        "Sigma(lambda) evaluated directly with",
        "  g_eta(A) = A (A^2 + eta^2 I)^(-1)",
        "using a linear solve; no eigendecomposition is used in the",
        "finite-difference validation.",
        "",
        "VALIDATION PAIRS",
        "----------------",
    ]

    for rank, pair in enumerate(selected_pairs, start=1):
        lines.append(
            f"{rank}: groups ({pair['g_minus']},{pair['g_plus']}), "
            f"|E|={pair['E_abs']:.8g}, "
            f"prominence={pair['prominence']:+.8g}"
        )

    lines += [
        "",
        "CONVERGENCE SLOPES",
        "------------------",
        "Central difference ideally gives slope about +2 before roundoff.",
    ]

    for family, pair_rank, sign, group, slope in slopes:
        lines.append(
            f"{family:10s} pair={pair_rank} {sign:5s} "
            f"group={group:3d} slope={slope:+.4f}"
        )

    lines += [
        "",
        "SMALLEST-DELTA SUMMARY",
        "----------------------",
        f"delta={smallest_delta:.6g}",
        (
            "median relative Frobenius error="
            f"{np.median(finite_rel):.6e}"
            if finite_rel
            else "median relative Frobenius error=nan"
        ),
        (
            "max relative Frobenius error="
            f"{np.max(finite_rel):.6e}"
            if finite_rel
            else "max relative Frobenius error=nan"
        ),
        (
            "median absolute Frobenius error="
            f"{np.median(finite_abs):.6e}"
            if finite_abs
            else "median absolute Frobenius error=nan"
        ),
        (
            "max absolute Frobenius error="
            f"{np.max(finite_abs):.6e}"
            if finite_abs
            else "max absolute Frobenius error=nan"
        ),
        "",
        "DECISION RULE",
        "-------------",
        "Accept the analytic v32.17 derivative if the absolute error decreases",
        "systematically over a stable delta range and the observed convergence",
        "is consistent with second-order central differencing before roundoff.",
        "",
        "Do not reject the derivative solely because a relative error becomes",
        "large when the true analytic derivative is itself near machine zero;",
        "in that regime inspect absolute error and convergence slope instead.",
    ]

    summary = "\n".join(lines) + "\n"
    txt_path.write_text(summary, encoding="utf-8")

    print(summary)
    print("WROTE")
    print(f"  {csv_path}")
    print(f"  {txt_path}")


if __name__ == "__main__":
    main()
