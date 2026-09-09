#!/usr/bin/env python3
"""Soft Spaces Phase 2 v25.38 — continuum score-curvature bound.

Evaluates a conservative analytic bound on the second derivative of every
frozen REAL/NULL v25.31 score along the structured X_a tensor G_s ancilla path.
The symbolic inequalities are continuum statements.  Their instance constants
are evaluated in double precision and therefore remain candidate numerical
certificates until outward-rounded/interval validation is performed.
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


VERSION = "v25.38"
INTERVAL_FRACTIONS = (
    1e-16, 3e-16, 1e-15, 3e-15, 1e-14, 3e-14, 1e-13, 3e-13,
    1e-12, 3e-12, 1e-11, 3e-11, 1e-10, 3e-10, 1e-9, 3e-9,
    1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5,
    1e-4, 3e-4, 1e-3, 3e-3, 1e-2,
)
SAFETY_FACTOR = 0.9
RANK = 2


@dataclass(frozen=True)
class ScoreBound:
    curvature: float
    denominator_floor: float
    numerator_floor: float


@dataclass(frozen=True)
class SeedBound:
    seed: int
    base_prominence: float
    g_star: float
    interval_epsilon: float | None
    interval_rho: float | None
    prominence_curvature: float | None
    certified_epsilon_candidate: float | None
    certified_rho_candidate: float | None
    denominator_floor: float | None
    numerator_floor: float | None


def individual_gap(values: np.ndarray, index: int) -> float:
    differences = np.abs(values[index] - np.delete(values, index))
    return float(np.min(differences))


def score_curvature_bound(
    hamiltonian: np.ndarray,
    perturbation: np.ndarray,
    coupling_norm: float,
    left: int,
    interval_radius: float,
    eta: float,
) -> ScoreBound | None:
    values, vectors = base.spectral_model(hamiltonian)
    n = values.size
    right = left + 1
    width = float(values[right] - values[left])
    gap = stress.cluster_gap(values, left)
    r = interval_radius
    k = coupling_norm
    if r * k >= 0.5 * gap:
        return None

    contour_distance = 0.5 * gap - r * k
    contour_length = 2.0 * width + math.pi * gap
    p1 = contour_length * k / (2.0 * math.pi * contour_distance**2)
    p2 = contour_length * k * k / (math.pi * contour_distance**3)

    spectral_radius = 0.5 * float(values[-1] - values[0]) + r * k
    e1 = k + 2.0 * spectral_radius * p1
    e2 = (n / RANK) * (spectral_radius * p2 + 2.0 * p1 * k)

    bnorm = 2.0 * spectral_radius
    a1 = 2.0 * p1 * bnorm + e1 + k
    a2 = (
        2.0 * p2 * bnorm
        + 4.0 * p1 * (e1 + k)
        + 2.0 * p1 * p1 * bnorm
        + e2
    )

    m = float(np.linalg.norm(perturbation, ord=2))
    t1 = 2.0 * m * p1
    t2 = 2.0 * m * p2 + 2.0 * m * p1 * p1
    g0 = 1.0 / (2.0 * eta)
    g1 = a1 / (eta * eta)
    g2 = 2.0 * a1 * a1 / (eta**3) + a2 / (eta * eta)

    effective1 = 2.0 * t1 * g0 * m + m * m * g1
    effective2 = (
        2.0 * t2 * g0 * m
        + m * m * g2
        + 4.0 * t1 * g1 * m
        + 2.0 * t1 * t1 * g0
    )
    numerator1 = math.sqrt(n) * effective1

    parts = mechanisms.ratio_parts_for_pair(
        values, vectors, perturbation, (left, right), eta
    )
    numerator_floor = parts.numerator - r * numerator1
    if numerator_floor <= 0.0:
        return None
    numerator2 = (
        math.sqrt(n) * effective2
        + n * effective1 * effective1 / numerator_floor
    )

    delta1 = e1 + k
    denominator1 = 0.0
    denominator2 = 0.0
    for index in range(n):
        if index in (left, right):
            continue
        eigen_gap = individual_gap(values, index)
        if r * k >= 0.5 * eigen_gap:
            return None
        eig_contour_distance = 0.5 * eigen_gap - r * k
        q1 = eigen_gap * k / (2.0 * eig_contour_distance**2)
        q2 = eigen_gap * k * k / (eig_contour_distance**3)
        lambda2 = 2.0 * k * k / (eigen_gap - 2.0 * r * k)
        delta2 = e2 + lambda2

        delta0 = 0.5 * float(values[left] + values[right]) - float(values[index])
        distance_from_zero = abs(delta0) - 2.0 * r * k
        if distance_from_zero <= 0.0:
            return None
        scale2 = distance_from_zero**2 + eta**2
        h0 = 1.0 / math.sqrt(scale2)
        h1 = 1.0 / scale2
        h2 = 6.0 / (scale2**1.5)

        x1 = m * (math.sqrt(2.0) * q1 + 2.0 * p1)
        x2 = m * (
            math.sqrt(n) * q2
            + 2.0 * math.sqrt(2.0) * q1 * p1
            + math.sqrt(n) * p2
        )
        coupling1 = 2.0 * m * x1
        coupling2 = 2.0 * x1 * x1 + 2.0 * m * x2

        denominator1 += h1 * delta1 * m * m + h0 * coupling1
        denominator2 += (
            h2 * delta1 * delta1 * m * m
            + h1 * delta2 * m * m
            + 2.0 * h1 * delta1 * coupling1
            + h0 * coupling2
        )

    denominator_floor = parts.denominator - r * denominator1
    if denominator_floor <= 0.0:
        return None

    curvature = (
        (numerator2 + denominator2) / denominator_floor
        + 2.0 * (numerator1 * denominator1 + denominator1**2)
        / (denominator_floor**2)
    )
    if not math.isfinite(curvature):
        return None
    return ScoreBound(curvature, denominator_floor, numerator_floor)


def build_instance(seed: int):
    rng = np.random.default_rng(seed)
    h_real = base.random_hermitian(base.DIM, rng)
    values, _ = base.spectral_model(h_real)
    h_null = base.null_hamiltonian(values, base.haar_unitary(base.DIM, rng))
    perturbations = {
        family: base.random_hermitian(base.DIM, rng)
        for family in base.FAMILIES
    }
    source_real = base.pair_scores(
        h_real, perturbations, (base.TARGET,) + base.CONTROLS, base.REGULARIZER
    )
    source_null = base.pair_scores(
        h_null, perturbations, (base.TARGET,) + base.CONTROLS, base.REGULARIZER
    )
    prominence = base.prominence(
        source_real, source_null, base.TARGET, base.CONTROLS
    )
    width = float(values[-1] - values[0])
    shift = width + 1.0
    h_real_up = base.exact_lift(h_real, shift, is_hamiltonian=True)
    h_null_up = base.exact_lift(h_null, shift, is_hamiltonian=True)
    perturbations_up = {
        family: base.exact_lift(matrix, shift, is_hamiltonian=False)
        for family, matrix in perturbations.items()
    }
    system_coupling = base.random_hermitian(base.DIM, rng)
    coupling = stress.off_diagonal_coupling(system_coupling)
    lifted_values, _ = base.spectral_model(h_real_up)
    indices = (base.TARGET,) + base.CONTROLS
    g_star = min(
        stress.cluster_gap(lifted_values, index + branch * base.DIM)
        for branch in (0, 1) for index in indices
    )
    return prominence, g_star, h_real_up, h_null_up, perturbations_up, coupling


def evaluate_radius(seed: int) -> SeedBound:
    prominence, g_star, h_real, h_null, perturbations, coupling = build_instance(seed)
    coupling_norm = float(np.linalg.norm(coupling, ord=2))
    if prominence <= 0.0:
        return SeedBound(seed, prominence, g_star, None, None, None, None, None, None, None)

    best: SeedBound | None = None
    for fraction in INTERVAL_FRACTIONS:
        radius = fraction * g_star
        target_constants: list[float] = []
        control_constants: list[float] = []
        denominator_floors: list[float] = []
        numerator_floors: list[float] = []
        valid = True
        for branch in (0, 1):
            offset = branch * base.DIM
            for family, perturbation in perturbations.items():
                for source_index in (base.TARGET,) + base.CONTROLS:
                    left = source_index + offset
                    real_bound = score_curvature_bound(
                        h_real, perturbation, coupling_norm, left, radius, base.REGULARIZER
                    )
                    null_bound = score_curvature_bound(
                        h_null, perturbation, coupling_norm, left, radius, base.REGULARIZER
                    )
                    if real_bound is None or null_bound is None:
                        valid = False
                        break
                    # Taylor coefficient: 1/2 sup |C_REAL''-C_NULL''|.
                    coefficient = 0.5 * (
                        real_bound.curvature + null_bound.curvature
                    )
                    (target_constants if source_index == base.TARGET else control_constants).append(coefficient)
                    denominator_floors.extend((real_bound.denominator_floor, null_bound.denominator_floor))
                    numerator_floors.extend((real_bound.numerator_floor, null_bound.numerator_floor))
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            continue
        prominence_curvature = max(target_constants) + max(control_constants)
        epsilon_from_margin = math.sqrt(prominence / prominence_curvature)
        epsilon_candidate = SAFETY_FACTOR * min(radius, epsilon_from_margin)
        candidate = SeedBound(
            seed=seed,
            base_prominence=prominence,
            g_star=g_star,
            interval_epsilon=radius,
            interval_rho=radius / g_star,
            prominence_curvature=prominence_curvature,
            certified_epsilon_candidate=epsilon_candidate,
            certified_rho_candidate=epsilon_candidate / g_star,
            denominator_floor=min(denominator_floors),
            numerator_floor=min(numerator_floors),
        )
        if best is None or candidate.certified_rho_candidate > best.certified_rho_candidate:
            best = candidate
    if best is None:
        return SeedBound(seed, prominence, g_star, None, None, None, None, None, None, None)
    return best


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6e}"


def render(results: list[SeedBound]) -> str:
    candidates = [r.certified_rho_candidate for r in results if r.certified_rho_candidate is not None]
    lines = [
        "=== Soft Spaces Phase 2 v25.38 CONTINUUM SCORE-CURVATURE BOUND ===",
        f"Frozen target: {base.TARGET}; controls: {list(base.CONTROLS)}",
        f"Seeds: {list(base.BASE_SEEDS)}; families: {list(base.FAMILIES)}",
        f"eta: {base.REGULARIZER:.3e}; interval fractions: {list(INTERVAL_FRACTIONS)}",
        "Same deterministic X_a tensor G_s path; all negative seeds retained.",
        "No fitting, coordinate relocation, re-ranking, or seed removal.",
        "Audit note: an initial coarse set beginning at rho=1e-8 returned no valid numerator floor; that failed output is retained, and the final declared multiscale set resolves smaller intervals without changing data or scores.",
        "",
        "seed | base prom | chosen interval rho | K_prom | D floor | N floor | candidate rho",
    ]
    for result in results:
        lines.append(
            f"{result.seed:7d} | {result.base_prominence:+9.3e} | "
            f"{fmt(result.interval_rho):>19s} | {fmt(result.prominence_curvature):>12s} | "
            f"{fmt(result.denominator_floor):>12s} | {fmt(result.numerator_floor):>12s} | "
            f"{fmt(result.certified_rho_candidate):>13s}"
        )
    lines.extend([
        "",
        "DECISION AND STATUS",
        f"  Positive source-prominence seeds: {len(candidates)}/{len(results)}.",
        f"  Common double-precision continuum candidate rho: {min(candidates):.6e}." if candidates else "  No positive candidate radius was obtained.",
        "  Symbolic bound includes first/second Riesz-projector derivatives, first/second regularized-resolvent derivatives, numerator Frobenius curvature, denominator spectral-term curvature, and the ratio rule.",
        "  Ancilla parity makes each complete score even, so Taylor's theorem converts the uniform second-derivative bound into a quadratic error bound.",
        "  The analytic inequalities hold throughout each stated interval if their constants are exact.",
        "  Constants were evaluated in ordinary double precision; outward-rounded interval validation is required before the numerical radii are called computer-assisted rigorous certificates.",
        "  Scope remains the controlled ancilla path, not the independently redrawn 8Q-12Q ensemble.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v25_38_continuum_score_curvature_bound_output.txt"),
    )
    args = parser.parse_args()
    report = render([evaluate_radius(seed) for seed in base.BASE_SEEDS])
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
