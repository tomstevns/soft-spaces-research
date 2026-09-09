#!/usr/bin/env python3
"""Soft Spaces Phase 2 v25.34 — controlled ancilla-coupling stress test.

This is a frozen finite-grid stress test around the exact v25.33 ancilla lift.
It does not fit a score, relocate a target, discard negative base prominence,
or claim a continuum/universal coupling threshold.

The dimensionless coupling rho is defined by

    ||delta H||_2 = rho * g_star,

where g_star is the smallest source pair-to-Q spectral gap over the frozen
target, frozen controls, and both ancilla branches.  The coupling direction is
a deterministic off-diagonal ancilla-system operator X_a tensor G_s with unit
operator norm.

For positive source prominence m, the already-proved prominence theorem gives
a finite-instance certificate at a sampled rho whenever

    target_delta_error + maximum_control_delta_error < m

in both branches.  This is a certificate for the evaluated instance/grid
point, not an analytic guarantee for every coupling direction or every rho
between grid points.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import v25_33_exact_ancilla_lift_verification as base


VERSION = "v25.34"
RHO_GRID = (
    0.0,
    1.0e-4,
    3.0e-4,
    1.0e-3,
    3.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
    3.0e-1,
    1.0,
    3.0,
    10.0,
)


@dataclass(frozen=True)
class GridRow:
    seed: int
    rho: float
    epsilon: float
    base_prominence: float
    branch0_prominence: float
    branch1_prominence: float
    branch0_error_bound: float
    branch1_error_bound: float
    branch0_certified_positive: bool
    branch1_certified_positive: bool
    branch0_sign_preserved: bool
    branch1_sign_preserved: bool


def cluster_gap(eigenvalues: np.ndarray, left: int) -> float:
    """Distance of the adjacent two-level cluster to the rest of the spectrum."""
    candidates: list[float] = []
    if left > 0:
        candidates.append(float(eigenvalues[left] - eigenvalues[left - 1]))
    if left + 2 < eigenvalues.size:
        candidates.append(float(eigenvalues[left + 2] - eigenvalues[left + 1]))
    if not candidates:
        raise ValueError("Pair has no surrounding Q spectrum")
    gap = min(candidates)
    if gap <= 0.0:
        raise RuntimeError("Selected pair is not an isolated spectral cluster")
    return gap


def off_diagonal_coupling(system_matrix: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(system_matrix)
    coupling = np.block([[zero, system_matrix], [system_matrix, zero]])
    norm = float(np.linalg.norm(coupling, ord=2))
    if norm <= 0.0:
        raise RuntimeError("Coupling norm vanished")
    return coupling / norm


def family_delta(
    real: dict[tuple[str, int], float],
    null: dict[tuple[str, int], float],
    family: str,
    index: int,
) -> float:
    return real[(family, index)] - null[(family, index)]


def delta_error_bounds(
    base_real: dict[tuple[str, int], float],
    base_null: dict[tuple[str, int], float],
    lifted_real: dict[tuple[str, int], float],
    lifted_null: dict[tuple[str, int], float],
    branch: int,
) -> tuple[float, float]:
    offset = branch * base.DIM
    target_error = max(
        abs(
            family_delta(lifted_real, lifted_null, family, base.TARGET + offset)
            - family_delta(base_real, base_null, family, base.TARGET)
        )
        for family in base.FAMILIES
    )
    control_error = max(
        abs(
            family_delta(lifted_real, lifted_null, family, control + offset)
            - family_delta(base_real, base_null, family, control)
        )
        for family in base.FAMILIES
        for control in base.CONTROLS
    )
    return target_error, control_error


def same_strict_sign(value: float, reference: float) -> bool:
    if reference > 0.0:
        return value > 0.0
    if reference < 0.0:
        return value < 0.0
    return value == 0.0


def run_seed(seed: int) -> tuple[float, list[GridRow]]:
    rng = np.random.default_rng(seed)
    h_real = base.random_hermitian(base.DIM, rng)
    values, _ = base.spectral_model(h_real)
    h_null = base.null_hamiltonian(values, base.haar_unitary(base.DIM, rng))
    perturbations = {
        family: base.random_hermitian(base.DIM, rng)
        for family in base.FAMILIES
    }
    fixed_indices = (base.TARGET,) + base.CONTROLS
    base_real = base.pair_scores(
        h_real, perturbations, fixed_indices, base.REGULARIZER
    )
    base_null = base.pair_scores(
        h_null, perturbations, fixed_indices, base.REGULARIZER
    )
    base_prom = base.prominence(
        base_real, base_null, base.TARGET, base.CONTROLS
    )

    width = float(values[-1] - values[0])
    branch_shift = width + 1.0
    h_real_up = base.exact_lift(h_real, branch_shift, is_hamiltonian=True)
    h_null_up = base.exact_lift(h_null, branch_shift, is_hamiltonian=True)
    perturbations_up = {
        family: base.exact_lift(matrix, branch_shift, is_hamiltonian=False)
        for family, matrix in perturbations.items()
    }
    lifted_indices = tuple(
        index + branch * base.DIM
        for branch in (0, 1)
        for index in fixed_indices
    )

    lifted_values, _ = base.spectral_model(h_real_up)
    g_star = min(
        cluster_gap(lifted_values, index + branch * base.DIM)
        for branch in (0, 1)
        for index in fixed_indices
    )

    system_coupling = base.random_hermitian(base.DIM, rng)
    coupling = off_diagonal_coupling(system_coupling)

    rows: list[GridRow] = []
    for rho in RHO_GRID:
        epsilon = float(rho * g_star)
        real_coupled = h_real_up + epsilon * coupling
        null_coupled = h_null_up + epsilon * coupling
        coupled_real_scores = base.pair_scores(
            real_coupled, perturbations_up, lifted_indices, base.REGULARIZER
        )
        coupled_null_scores = base.pair_scores(
            null_coupled, perturbations_up, lifted_indices, base.REGULARIZER
        )

        prominences: list[float] = []
        bounds: list[float] = []
        certified: list[bool] = []
        signs: list[bool] = []
        for branch in (0, 1):
            offset = branch * base.DIM
            controls = tuple(control + offset for control in base.CONTROLS)
            value = base.prominence(
                coupled_real_scores,
                coupled_null_scores,
                base.TARGET + offset,
                controls,
            )
            target_error, control_error = delta_error_bounds(
                base_real,
                base_null,
                coupled_real_scores,
                coupled_null_scores,
                branch,
            )
            error_bound = target_error + control_error
            prominences.append(value)
            bounds.append(error_bound)
            certified.append(base_prom > 0.0 and error_bound < base_prom)
            signs.append(same_strict_sign(value, base_prom))

        rows.append(GridRow(
            seed=seed,
            rho=rho,
            epsilon=epsilon,
            base_prominence=base_prom,
            branch0_prominence=prominences[0],
            branch1_prominence=prominences[1],
            branch0_error_bound=bounds[0],
            branch1_error_bound=bounds[1],
            branch0_certified_positive=certified[0],
            branch1_certified_positive=certified[1],
            branch0_sign_preserved=signs[0],
            branch1_sign_preserved=signs[1],
        ))
    return g_star, rows


def largest_contiguous_rho(rows: list[GridRow], field0: str, field1: str) -> float | None:
    accepted: float | None = None
    for row in rows:
        if bool(getattr(row, field0)) and bool(getattr(row, field1)):
            accepted = row.rho
        else:
            break
    return accepted


def format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3e}"


def render(seed_results: list[tuple[float, list[GridRow]]]) -> str:
    lines = [
        "=== Soft Spaces Phase 2 v25.34 CONTROLLED ANCILLA-COUPLING STRESS TEST ===",
        f"Frozen target: {base.TARGET}; controls: {list(base.CONTROLS)}",
        f"Seeds: {list(base.BASE_SEEDS)}",
        f"Dimensionless rho grid: {list(RHO_GRID)}",
        "Definition: ||delta H||_2 = rho * g_star",
        "No coordinate search, relocation, re-ranking, score fitting, or failed-seed removal.",
        "",
        "seed | g_star | base prominence | largest certified-positive rho | largest sign-preserved rho",
    ]
    positive_seed_certificates: list[float] = []
    for g_star, rows in seed_results:
        base_prom = rows[0].base_prominence
        certified = largest_contiguous_rho(
            rows, "branch0_certified_positive", "branch1_certified_positive"
        ) if base_prom > 0.0 else None
        sign_preserved = largest_contiguous_rho(
            rows, "branch0_sign_preserved", "branch1_sign_preserved"
        )
        if certified is not None:
            positive_seed_certificates.append(certified)
        lines.append(
            f"{rows[0].seed:7d} | {g_star:7.3e} | {base_prom:+15.6e} | "
            f"{format_optional(certified):>30s} | {format_optional(sign_preserved):>26s}"
        )

    lines.extend([
        "",
        "Detailed frozen grid:",
        "seed | rho | epsilon | prom b0 | prom b1 | theorem err b0 | theorem err b1 | cert b0/b1 | sign b0/b1",
    ])
    for _, rows in seed_results:
        for row in rows:
            cert = f"{'Y' if row.branch0_certified_positive else 'N'}/{'Y' if row.branch1_certified_positive else 'N'}"
            sign = f"{'Y' if row.branch0_sign_preserved else 'N'}/{'Y' if row.branch1_sign_preserved else 'N'}"
            lines.append(
                f"{row.seed:7d} | {row.rho:7.1e} | {row.epsilon:8.2e} | "
                f"{row.branch0_prominence:+8.3e} | {row.branch1_prominence:+8.3e} | "
                f"{row.branch0_error_bound:14.3e} | {row.branch1_error_bound:14.3e} | "
                f"{cert:^10s} | {sign:^10s}"
            )

    lines.extend(["", "Decision and scope:"])
    positive_count = sum(rows[0].base_prominence > 0.0 for _, rows in seed_results)
    weyl_safe_rows = [rho for rho in RHO_GRID if rho < 0.5]
    largest_weyl_safe_grid_rho = max(weyl_safe_rows)
    safe_grid_certified = all(
        next(row for row in rows if row.rho == largest_weyl_safe_grid_rho).branch0_certified_positive
        and next(row for row in rows if row.rho == largest_weyl_safe_grid_rho).branch1_certified_positive
        for _, rows in seed_results
        if rows[0].base_prominence > 0.0
    )
    lines.append(f"  Positive source-prominence seeds: {positive_count}/{len(seed_results)} (reported without filtering).")
    if positive_seed_certificates:
        lines.append(
            "  Common finite-grid certified rho across all positive source seeds: "
            f"{min(positive_seed_certificates):.3e}."
        )
    else:
        lines.append("  No positive source seed had a non-empty finite-grid certificate.")
    lines.extend([
        f"  Weyl pair-isolation condition: rho < 0.5; largest evaluated safe-grid rho: {largest_weyl_safe_grid_rho:.1e}.",
        f"  All positive source seeds certified at that safe-grid point: {'YES' if safe_grid_certified else 'NO'}.",
        "  A certificate means the proved target-plus-control error inequality holds at that evaluated grid point in both branches.",
        "  Sign preservation without the inequality is observation, not a theorem certificate.",
        "  Certificates at rho >= 0.5 concern the fixed energy-index score; pair identity is not guaranteed there by the global gap bound.",
        "  This run does not establish a continuum interval or a universal ensemble threshold.",
        "  Without a uniform prominence margin, cluster gap, and denominator floor, the universal guaranteed coupling radius is zero.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v25_34_controlled_coupling_stress_test_output.txt"),
    )
    args = parser.parse_args()
    seed_results = [run_seed(seed) for seed in base.BASE_SEEDS]
    report = render(seed_results)
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
