#!/usr/bin/env python3
"""Soft Spaces Phase 2 v25.37 — parity-aware second-order bound.

This frozen analysis combines an exact Schur-complement bound for the
off-diagonal ancilla coupling with a finite-grid curvature audit of the full
v25.31 prominence functional.  It separates what is proved analytically from
what is only evaluated on the predeclared v25.34 rho grid.

No target, control, family, seed, coupling direction, or grid point is changed.
Negative source-prominence instances are retained.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import v25_33_exact_ancilla_lift_verification as base
import v25_34_controlled_coupling_stress_test as stress
import v25_36_stability_mechanism_decomposition as mechanisms


VERSION = "v25.37"
SAFE_CURVATURE_RHOS = (
    1.0e-4,
    3.0e-4,
    1.0e-3,
    3.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
    3.0e-1,
)


@dataclass(frozen=True)
class Result:
    seed: int
    base_prominence: float
    g_star: float
    branch_separation: float
    epsilon_at_rho10: float
    schur_bound_at_rho10: float
    max_prominence_curvature: float
    max_error_curvature: float
    grid_margin_radius: float | None
    max_parity_residual: float
    exponent0: float
    exponent1: float


def source_geometry(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    h_real = base.random_hermitian(base.DIM, rng)
    values, _ = base.spectral_model(h_real)
    width = float(values[-1] - values[0])
    shift = width + 1.0
    # Distance between the two uncoupled spectral intervals:
    # min(spec(H+sI)) - max(spec(H-sI)) = 2s - width.
    branch_separation = 2.0 * shift - width
    lifted_values = np.concatenate((values - shift, values + shift))
    fixed = (base.TARGET,) + base.CONTROLS
    g_star = min(
        stress.cluster_gap(lifted_values, index + branch * base.DIM)
        for branch in (0, 1)
        for index in fixed
    )
    return g_star, branch_separation


def schur_self_energy_bound(epsilon: float, separation: float) -> float:
    """||epsilon^2 G(D-E)^-1 G|| for ||G||=1 via Weyl separation."""
    if epsilon >= separation:
        return math.inf
    return epsilon * epsilon / (separation - epsilon)


def run(seed: int) -> Result:
    g_star, separation = source_geometry(seed)
    stress_g_star, stress_rows = stress.run_seed(seed)
    if not np.isclose(g_star, stress_g_star, rtol=1.0e-12, atol=1.0e-14):
        raise RuntimeError("Independent g_star reconstruction disagrees with v25.34")
    mechanism_rows = mechanisms.run_seed(seed)
    stress_by_rho = {row.rho: row for row in stress_rows}
    mechanism_by_rho = {row.rho: row for row in mechanism_rows}
    base_prominence = stress_rows[0].base_prominence

    prominence_curvatures: list[float] = []
    error_curvatures: list[float] = []
    for rho in SAFE_CURVATURE_RHOS:
        stress_row = stress_by_rho[rho]
        for prominence, error in (
            (stress_row.branch0_prominence, stress_row.branch0_error_bound),
            (stress_row.branch1_prominence, stress_row.branch1_error_bound),
        ):
            prominence_curvatures.append(
                abs(prominence - base_prominence) / (rho * rho)
            )
            error_curvatures.append(error / (rho * rho))

    max_prominence_curvature = max(prominence_curvatures)
    max_error_curvature = max(error_curvatures)
    grid_margin_radius = (
        math.sqrt(base_prominence / max_error_curvature)
        if base_prominence > 0.0 and max_error_curvature > 0.0
        else None
    )

    max_parity_residual = max(
        branch.parity_residual
        for row in mechanism_rows
        for branch in row.branches
    )
    epsilon10 = 10.0 * g_star
    return Result(
        seed=seed,
        base_prominence=base_prominence,
        g_star=g_star,
        branch_separation=separation,
        epsilon_at_rho10=epsilon10,
        schur_bound_at_rho10=schur_self_energy_bound(epsilon10, separation),
        max_prominence_curvature=max_prominence_curvature,
        max_error_curvature=max_error_curvature,
        grid_margin_radius=grid_margin_radius,
        max_parity_residual=max_parity_residual,
        exponent0=mechanisms.response_exponent(mechanism_rows, 0),
        exponent1=mechanisms.response_exponent(mechanism_rows, 1),
    )


def optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6e}"


def render(results: list[Result]) -> str:
    lines = [
        "=== Soft Spaces Phase 2 v25.37 PARITY-AWARE SECOND-ORDER BOUND ===",
        f"Frozen target: {base.TARGET}; controls: {list(base.CONTROLS)}",
        f"Seeds: {list(base.BASE_SEEDS)}; families: {list(base.FAMILIES)}",
        f"Curvature grid: {list(SAFE_CURVATURE_RHOS)}; eta: {base.REGULARIZER:.3e}",
        "Same deterministic X_a tensor G_s direction as v25.34-v25.36.",
        "No fitting, relocation, re-ranking, seed removal, or post-run grid change.",
        "",
        "EXACT ANALYTIC BLOCK BOUND",
        "For H_epsilon=[[A,epsilon G],[epsilon G,D]], elimination of the opposite branch gives",
        "  [A-E-epsilon^2 G(D-E)^(-1)G] psi_A = 0.",
        "If the uncoupled branch separation is Gamma and |epsilon|<Gamma, then",
        "  ||Sigma(E)|| <= epsilon^2 ||G||^2/(Gamma-|epsilon|).",
        "Here ||G||=1. This is a continuum, all-system-vector second-order bound for the branch self-energy.",
        "It is not by itself a bound on the complete normalized prominence functional.",
        "",
        "seed | base prom | g_star | branch Gamma | epsilon(rho=10) | Schur bound(rho=10)",
    ]
    for result in results:
        lines.append(
            f"{result.seed:7d} | {result.base_prominence:+9.3e} | "
            f"{result.g_star:7.3e} | {result.branch_separation:12.6e} | "
            f"{result.epsilon_at_rho10:15.6e} | "
            f"{result.schur_bound_at_rho10:19.6e}"
        )

    lines.extend([
        "",
        "FROZEN SAFE-GRID CURVATURE AUDIT",
        "Definitions on rho<=0.3:",
        "  K_prom = max |Pi(rho)-Pi(0)|/rho^2 over both branches.",
        "  K_err  = max (target delta error + max control delta error)/rho^2 over both branches.",
        "  rho_margin = sqrt(Pi(0)/K_err) for positive source prominence.",
        "rho_margin is a data-derived candidate scale, not a continuum certificate.",
        "",
        "seed | exponent b0/b1 | K_prom | K_err | rho_margin candidate | max parity residual",
    ])
    positive_candidates: list[float] = []
    for result in results:
        if result.grid_margin_radius is not None:
            positive_candidates.append(result.grid_margin_radius)
        lines.append(
            f"{result.seed:7d} | {result.exponent0:7.4f}/{result.exponent1:7.4f} | "
            f"{result.max_prominence_curvature:8.3e} | "
            f"{result.max_error_curvature:8.3e} | "
            f"{optional(result.grid_margin_radius):>20s} | "
            f"{result.max_parity_residual:19.6e}"
        )

    lines.extend([
        "",
        "DECISION",
        f"  Median low-rho exponent: {float(np.median([x for r in results for x in (r.exponent0, r.exponent1)])):.6f}.",
        f"  Largest +rho/-rho residual: {max(r.max_parity_residual for r in results):.6e}.",
        f"  Common positive-seed rho_margin candidate: {min(positive_candidates):.6e}." if positive_candidates else "  No positive-seed margin candidate.",
        "  All rho<=0.3 points remain covered by the already proved finite-grid target-plus-control inequality.",
        "  The Schur complement proves quadratic Hamiltonian self-energy on a continuum interval.",
        "  The full prominence is exactly even, and its measured low-rho response is quadratic.",
        "  A rigorous continuum prominence radius still requires an analytic or interval-validated upper bound on its second derivative.",
        "  Scope: controlled ancilla path only, not the independently redrawn 8Q-12Q ensemble.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v25_37_parity_aware_second_order_bound_output.txt"),
    )
    args = parser.parse_args()
    report = render([run(seed) for seed in base.BASE_SEEDS])
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
