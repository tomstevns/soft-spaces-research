#!/usr/bin/env python3
"""Soft Spaces Phase 2 v25.35 — uniform analytic prominence certificate.

For each frozen positive-prominence source instance from v25.33, this program
computes a conservative coupling radius that certifies positive prominence for
EVERY Hermitian Hamiltonian perturbation direction with

    ||delta H||_2 <= rho * g_star.

The certificate combines:
* a Riesz-contour bound for the moving rank-2 spectral projector,
* a regularized-resolvent bound,
* a basis-independent trace representation of the v25.31 denominator,
* the proved target-plus-control prominence inequality.

The perturbation operators V are held fixed.  Negative source-prominence seeds
are retained and reported as ineligible, not removed or reclassified.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import v25_33_exact_ancilla_lift_verification as base
import v25_34_controlled_coupling_stress_test as stress


VERSION = "v25.35"
RANK = 2
RHO_UPPER = 0.499999
BISECTION_STEPS = 100
ENDPOINT_SAFETY_FACTOR = 0.9


@dataclass(frozen=True)
class Certificate:
    seed: int
    base_prominence: float
    g_star: float
    denominator_floor: float
    perturbation_norm: float
    spectral_radius: float
    certified_rho: float | None
    certified_epsilon: float | None
    projector_bound: float | None
    compressed_operator_bound: float | None
    denominator_change_bound: float | None
    single_model_score_bound: float | None
    prominence_change_bound: float | None


def ratio_parts(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    perturbation: np.ndarray,
    left_index: int,
    regularizer: float,
) -> tuple[float, float]:
    right_index = left_index + 1
    selected = eigenvectors[:, [left_index, right_index]]
    couplings = eigenvectors.conj().T @ perturbation @ selected
    keep = np.ones(eigenvalues.size, dtype=bool)
    keep[[left_index, right_index]] = False
    wq = couplings[keep, :]
    eq = eigenvalues[keep]
    reference = 0.5 * float(eigenvalues[left_index] + eigenvalues[right_index])
    delta = reference - eq
    weights = delta / (delta * delta + regularizer * regularizer)
    terms = np.einsum("ni,nj->nij", wq.conj(), wq)
    total = np.sum(weights[:, None, None] * terms, axis=0)
    numerator = float(np.linalg.norm(total, ord="fro"))
    denominator = float(np.sum(np.abs(weights) * np.sum(np.abs(wq) ** 2, axis=1)))
    if denominator <= 0.0:
        raise RuntimeError("Base denominator vanished")
    return numerator, denominator


def riesz_projector_bound(
    delta_h: float,
    pair_widths: tuple[float, ...],
    cluster_gaps: tuple[float, ...],
) -> float:
    if delta_h == 0.0:
        return 0.0
    bounds: list[float] = []
    for width, gap in zip(pair_widths, cluster_gaps):
        if delta_h >= 0.5 * gap:
            return 1.0
        contour_length = 2.0 * width + math.pi * gap
        bound = (
            contour_length
            * delta_h
            / (math.pi * gap * (0.5 * gap - delta_h))
        )
        bounds.append(bound)
    return min(1.0, max(bounds))


def uniform_score_bound(
    rho: float,
    g_star: float,
    pair_widths: tuple[float, ...],
    cluster_gaps: tuple[float, ...],
    spectral_radius: float,
    perturbation_norm: float,
    denominator_floor: float,
    regularizer: float,
) -> tuple[bool, dict[str, float]]:
    delta_h = rho * g_star
    p = riesz_projector_bound(delta_h, pair_widths, cluster_gaps)

    # Rank-r cluster midpoint E=Tr(PH)/r.  Since Tr(P'-P)=0, centering H
    # gives |E'-E| <= ||delta H|| + 2 ||P'-P|| R.
    delta_e = delta_h + 2.0 * p * spectral_radius

    # A=Q(E-H)Q, including zero on P.  This bound includes both projector
    # motion and motion of E-H.
    delta_a = (
        p * (4.0 * spectral_radius + delta_e + delta_h)
        + delta_e
        + delta_h
    )

    eta = regularizer
    m = perturbation_norm
    t_change = 2.0 * m * p  # ||P'VQ' - PVQ||, with V fixed.

    delta_n = math.sqrt(2.0 * RANK) * (
        m * t_change / eta + m * m * delta_a / (eta * eta)
    )

    # S=|g_eta(A)|=(g_eta(A)^2)^(1/2).  The square-root Holder bound gives
    # ||S'-S|| <= sqrt(delta_a / eta^3).
    delta_s = math.sqrt(max(delta_a, 0.0) / (eta ** 3))
    delta_d = (
        RANK * math.sqrt(2.0) * m * t_change / eta
        + RANK * m * m * delta_s
    )

    remaining_denominator = denominator_floor - delta_d
    if remaining_denominator <= 0.0:
        return False, {
            "delta_h": delta_h,
            "projector": p,
            "delta_a": delta_a,
            "delta_n": delta_n,
            "delta_d": delta_d,
            "delta_c": math.inf,
            "prominence": math.inf,
        }

    delta_c = (delta_n + delta_d) / remaining_denominator

    # REAL and NULL each move by delta_c: family delta moves by 2 delta_c.
    # Target robust score and control median each move by 2 delta_c, so the
    # full prominence moves by at most 4 delta_c.
    prominence_bound = 4.0 * delta_c
    return True, {
        "delta_h": delta_h,
        "projector": p,
        "delta_a": delta_a,
        "delta_n": delta_n,
        "delta_d": delta_d,
        "delta_c": delta_c,
        "prominence": prominence_bound,
    }


def build_certificate(seed: int) -> Certificate:
    rng = np.random.default_rng(seed)
    h_real = base.random_hermitian(base.DIM, rng)
    values, vectors_real = base.spectral_model(h_real)
    h_null = base.null_hamiltonian(values, base.haar_unitary(base.DIM, rng))
    _, vectors_null = base.spectral_model(h_null)
    perturbations = {
        family: base.random_hermitian(base.DIM, rng)
        for family in base.FAMILIES
    }
    fixed_indices = (base.TARGET,) + base.CONTROLS

    real_scores = base.pair_scores(
        h_real, perturbations, fixed_indices, base.REGULARIZER
    )
    null_scores = base.pair_scores(
        h_null, perturbations, fixed_indices, base.REGULARIZER
    )
    base_prom = base.prominence(
        real_scores, null_scores, base.TARGET, base.CONTROLS
    )

    denominators: list[float] = []
    for perturbation in perturbations.values():
        for index in fixed_indices:
            denominators.append(
                ratio_parts(values, vectors_real, perturbation, index, base.REGULARIZER)[1]
            )
            denominators.append(
                ratio_parts(values, vectors_null, perturbation, index, base.REGULARIZER)[1]
            )
    denominator_floor = min(denominators)
    perturbation_norm = max(
        float(np.linalg.norm(matrix, ord=2))
        for matrix in perturbations.values()
    )

    pair_widths = tuple(
        float(values[index + 1] - values[index])
        for index in fixed_indices
    )
    cluster_gaps = tuple(
        stress.cluster_gap(values, index)
        for index in fixed_indices
    )
    g_star = min(cluster_gaps)

    width = float(values[-1] - values[0])
    branch_shift = width + 1.0
    lifted_min = float(values[0] - branch_shift)
    lifted_max = float(values[-1] + branch_shift)
    spectral_radius = 0.5 * (lifted_max - lifted_min)

    if base_prom <= 0.0:
        return Certificate(
            seed=seed,
            base_prominence=base_prom,
            g_star=g_star,
            denominator_floor=denominator_floor,
            perturbation_norm=perturbation_norm,
            spectral_radius=spectral_radius,
            certified_rho=None,
            certified_epsilon=None,
            projector_bound=None,
            compressed_operator_bound=None,
            denominator_change_bound=None,
            single_model_score_bound=None,
            prominence_change_bound=None,
        )

    def passes(rho: float) -> tuple[bool, dict[str, float]]:
        denominator_ok, details = uniform_score_bound(
            rho=rho,
            g_star=g_star,
            pair_widths=pair_widths,
            cluster_gaps=cluster_gaps,
            spectral_radius=spectral_radius,
            perturbation_norm=perturbation_norm,
            denominator_floor=denominator_floor,
            regularizer=base.REGULARIZER,
        )
        return denominator_ok and details["prominence"] < base_prom, details

    low = 0.0
    high = RHO_UPPER
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (low + high)
        ok, _ = passes(mid)
        if ok:
            low = mid
        else:
            high = mid

    # Stay strictly inside the numerically located boundary.  This is a
    # floating-point evaluation of an analytic sufficient condition, not an
    # interval-arithmetic or formally verified certificate.
    low *= ENDPOINT_SAFETY_FACTOR
    ok, details = passes(low)
    if not ok:
        raise RuntimeError("Bisection failed to retain its certified endpoint")

    return Certificate(
        seed=seed,
        base_prominence=base_prom,
        g_star=g_star,
        denominator_floor=denominator_floor,
        perturbation_norm=perturbation_norm,
        spectral_radius=spectral_radius,
        certified_rho=low,
        certified_epsilon=details["delta_h"],
        projector_bound=details["projector"],
        compressed_operator_bound=details["delta_a"],
        denominator_change_bound=details["delta_d"],
        single_model_score_bound=details["delta_c"],
        prominence_change_bound=details["prominence"],
    )


def fmt(value: float | None, width: int = 12) -> str:
    return f"{'n/a':>{width}s}" if value is None else f"{value:{width}.5e}"


def render(certificates: list[Certificate]) -> str:
    lines = [
        "=== Soft Spaces Phase 2 v25.35 UNIFORM ANALYTIC PROMINENCE BOUND ===",
        "Scope: every Hermitian delta-H direction with fixed perturbation operators V",
        f"Frozen target: {base.TARGET}; controls: {list(base.CONTROLS)}",
        f"Seeds: {list(base.BASE_SEEDS)} (negative source prominence retained)",
        f"Regularizer eta: {base.REGULARIZER:.3e}",
        "Definition: ||delta H||_2 <= rho * g_star; structural domain rho < 0.5",
        "No fitted score, coordinate relocation, seed removal, or numerical direction sampling.",
        "",
        "seed | base prom | g_star | D_floor | candidate rho | candidate ||dH|| | prom bound",
    ]
    positive_rhos: list[float] = []
    for result in certificates:
        if result.certified_rho is not None:
            positive_rhos.append(result.certified_rho)
        lines.append(
            f"{result.seed:7d} | {result.base_prominence:+9.3e} | "
            f"{result.g_star:7.3e} | {result.denominator_floor:7.3e} | "
            f"{fmt(result.certified_rho)} | {fmt(result.certified_epsilon, 16)} | "
            f"{fmt(result.prominence_change_bound)}"
        )

    lines.extend(["", "Certificate details for positive source instances:"])
    lines.append(
        "seed | ||V|| | spectral radius | P drift bound | A change bound | D change bound | one-model C bound"
    )
    for result in certificates:
        if result.certified_rho is None:
            continue
        lines.append(
            f"{result.seed:7d} | {result.perturbation_norm:7.3e} | "
            f"{result.spectral_radius:15.6e} | {fmt(result.projector_bound)} | "
            f"{fmt(result.compressed_operator_bound)} | "
            f"{fmt(result.denominator_change_bound)} | "
            f"{fmt(result.single_model_score_bound)}"
        )

    lines.extend(["", "Decision and interpretation:"])
    lines.append(
        f"  Positive source-prominence instances: {len(positive_rhos)}/{len(certificates)}."
    )
    if positive_rhos:
        lines.append(
            "  Common conservative all-directions candidate: rho <= "
            f"{min(positive_rhos):.5e}."
        )
    lines.extend([
        "  The analytic inequalities imply strict positivity inside each stated radius if the displayed constants are exact.",
        "  The constants here were evaluated in double precision; interval or higher-precision validation is still required before calling the numerical radii rigorous certificates.",
        "  The bound is sufficient and deliberately conservative; it is not an observed failure threshold.",
        "  The very small radius is expected from worst-case 1/eta^2 and 1/eta^(3/2) factors at eta=0.001.",
        "  Scope remains the coupled ancilla baseline, not the independently redrawn 8Q–12Q ensemble.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v25_35_uniform_analytic_prominence_certificate_output.txt"),
    )
    args = parser.parse_args()
    certificates = [build_certificate(seed) for seed in base.BASE_SEEDS]
    report = render(certificates)
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
