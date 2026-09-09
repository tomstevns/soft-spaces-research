#!/usr/bin/env python3
"""Soft Spaces Phase 2 v25.36 — stability-mechanism decomposition.

Frozen follow-up to v25.34.  It measures five overlapping mechanisms along
the exact same deterministic ancilla-coupling path:

1. branch symmetry,
2. REAL–NULL common-motion cancellation,
3. numerator/denominator normalization cancellation,
4. target–control common-mode cancellation,
5. sorted-index versus overlap-tracked spectral pairs.

The diagnostics are not additive causal percentages.  No target, control,
family, seed, coupling direction, or rho point is selected after seeing the
results.  Negative source-prominence seeds remain in the report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import v25_33_exact_ancilla_lift_verification as base
import v25_34_controlled_coupling_stress_test as stress


VERSION = "v25.36"
KEY_RHOS = (0.3, 1.0, 3.0, 10.0)
LOW_RHOS = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1)


@dataclass(frozen=True)
class Parts:
    score: float
    numerator: float
    denominator: float


@dataclass(frozen=True)
class BranchDiagnostic:
    seed: int
    rho: float
    branch: int
    base_prominence: float
    prominence: float
    rn_cancellation: float | None
    normalization_cancellation: float | None
    target_control_cancellation: float | None
    tracked_prominence: float
    tracking_discrepancy: float
    minimum_tracking_fidelity: float
    tracking_sign_disagreement: bool
    parity_residual: float


@dataclass(frozen=True)
class SeedRhoDiagnostic:
    seed: int
    rho: float
    base_prominence: float
    branch_symmetry: float | None
    branch_split: float
    branches: tuple[BranchDiagnostic, BranchDiagnostic]


def ratio_parts_for_pair(
    values: np.ndarray,
    vectors: np.ndarray,
    perturbation: np.ndarray,
    pair: tuple[int, int],
    eta: float,
) -> Parts:
    selected_indices = np.asarray(pair, dtype=int)
    selected = vectors[:, selected_indices]
    couplings = vectors.conj().T @ perturbation @ selected
    keep = np.ones(values.size, dtype=bool)
    keep[selected_indices] = False
    wq = couplings[keep, :]
    eq = values[keep]
    reference = 0.5 * float(np.sum(values[selected_indices]))
    delta = reference - eq
    weights = delta / (delta * delta + eta * eta)
    terms = np.einsum("ni,nj->nij", wq.conj(), wq)
    total = np.sum(weights[:, None, None] * terms, axis=0)
    numerator = float(np.linalg.norm(total, ord="fro"))
    denominator = float(
        np.sum(np.abs(weights) * np.sum(np.abs(wq) ** 2, axis=1))
    )
    if denominator <= 0.0:
        raise RuntimeError("Coherence-ratio denominator vanished")
    return Parts(numerator / denominator, numerator, denominator)


def cancellation_fraction(changed_a: float, changed_b: float) -> float | None:
    """Fraction of two absolute motions removed by taking their difference."""
    total = abs(changed_a) + abs(changed_b)
    if total <= 1.0e-15:
        return None
    value = 1.0 - abs(changed_a - changed_b) / total
    return float(np.clip(value, 0.0, 1.0))


def log_normalization_cancellation(base_parts: Parts, new_parts: Parts) -> float | None:
    tiny = np.finfo(float).tiny
    if min(
        base_parts.score,
        new_parts.score,
        base_parts.numerator,
        new_parts.numerator,
        base_parts.denominator,
        new_parts.denominator,
    ) <= tiny:
        return None
    dlog_n = float(np.log(new_parts.numerator / base_parts.numerator))
    dlog_d = float(np.log(new_parts.denominator / base_parts.denominator))
    total = abs(dlog_n) + abs(dlog_d)
    if total <= 1.0e-15:
        return None
    dlog_c = float(np.log(new_parts.score / base_parts.score))
    value = 1.0 - abs(dlog_c) / total
    return float(np.clip(value, 0.0, 1.0))


def strict_sign(value: float) -> int:
    return 1 if value > 0.0 else (-1 if value < 0.0 else 0)


def robust_score(scores: dict[tuple[str, int], float], index: int) -> float:
    return min(scores[(family, index)] for family in base.FAMILIES)


def prominence_from_deltas(
    deltas: dict[tuple[str, int], float],
    target: int,
    controls: tuple[int, ...],
) -> tuple[float, float, float]:
    target_score = robust_score(deltas, target)
    control_median = float(
        np.median([robust_score(deltas, index) for index in controls])
    )
    return target_score - control_median, target_score, control_median


def best_overlap_pair(
    reference_vectors: np.ndarray,
    reference_pair: tuple[int, int],
    current_vectors: np.ndarray,
) -> tuple[tuple[int, int], float]:
    reference = reference_vectors[:, np.asarray(reference_pair, dtype=int)]
    weights = np.sum(np.abs(reference.conj().T @ current_vectors) ** 2, axis=0)
    chosen = np.argsort(-weights, kind="stable")[:2]
    chosen = np.sort(chosen)
    fidelity = float(np.sum(weights[chosen]) / 2.0)
    return (int(chosen[0]), int(chosen[1])), fidelity


def make_parts_map(
    values: np.ndarray,
    vectors: np.ndarray,
    perturbations: dict[str, np.ndarray],
    pairs: dict[int, tuple[int, int]],
) -> dict[tuple[str, int], Parts]:
    return {
        (family, label): ratio_parts_for_pair(
            values, vectors, perturbation, pair, base.REGULARIZER
        )
        for family, perturbation in perturbations.items()
        for label, pair in pairs.items()
    }


def score_map(parts: dict[tuple[str, int], Parts]) -> dict[tuple[str, int], float]:
    return {key: value.score for key, value in parts.items()}


def delta_map(
    real: dict[tuple[str, int], Parts],
    null: dict[tuple[str, int], Parts],
) -> dict[tuple[str, int], float]:
    return {key: real[key].score - null[key].score for key in real}


def run_seed(seed: int) -> list[SeedRhoDiagnostic]:
    rng = np.random.default_rng(seed)
    h_real = base.random_hermitian(base.DIM, rng)
    source_values, _ = base.spectral_model(h_real)
    h_null = base.null_hamiltonian(
        source_values, base.haar_unitary(base.DIM, rng)
    )
    perturbations = {
        family: base.random_hermitian(base.DIM, rng)
        for family in base.FAMILIES
    }
    labels = (base.TARGET,) + base.CONTROLS
    source_pairs = {index: (index, index + 1) for index in labels}
    source_real_values, source_real_vectors = base.spectral_model(h_real)
    source_null_values, source_null_vectors = base.spectral_model(h_null)
    source_real_parts = make_parts_map(
        source_real_values, source_real_vectors, perturbations, source_pairs
    )
    source_null_parts = make_parts_map(
        source_null_values, source_null_vectors, perturbations, source_pairs
    )
    source_deltas = delta_map(source_real_parts, source_null_parts)
    base_prominence, base_target, base_control = prominence_from_deltas(
        source_deltas, base.TARGET, base.CONTROLS
    )

    width = float(source_values[-1] - source_values[0])
    shift = width + 1.0
    h_real_up = base.exact_lift(h_real, shift, is_hamiltonian=True)
    h_null_up = base.exact_lift(h_null, shift, is_hamiltonian=True)
    perturbations_up = {
        family: base.exact_lift(matrix, shift, is_hamiltonian=False)
        for family, matrix in perturbations.items()
    }
    real0_values, real0_vectors = base.spectral_model(h_real_up)
    null0_values, null0_vectors = base.spectral_model(h_null_up)
    lifted_pairs = {
        index + branch * base.DIM: (
            index + branch * base.DIM,
            index + branch * base.DIM + 1,
        )
        for branch in (0, 1)
        for index in labels
    }
    g_star = min(
        stress.cluster_gap(real0_values, index + branch * base.DIM)
        for branch in (0, 1)
        for index in labels
    )
    system_coupling = base.random_hermitian(base.DIM, rng)
    coupling = stress.off_diagonal_coupling(system_coupling)

    results: list[SeedRhoDiagnostic] = []
    for rho in stress.RHO_GRID:
        epsilon = rho * g_star
        real_values, real_vectors = base.spectral_model(h_real_up + epsilon * coupling)
        null_values, null_vectors = base.spectral_model(h_null_up + epsilon * coupling)
        sorted_real = make_parts_map(
            real_values, real_vectors, perturbations_up, lifted_pairs
        )
        sorted_null = make_parts_map(
            null_values, null_vectors, perturbations_up, lifted_pairs
        )
        negative_real_values, negative_real_vectors = base.spectral_model(
            h_real_up - epsilon * coupling
        )
        negative_null_values, negative_null_vectors = base.spectral_model(
            h_null_up - epsilon * coupling
        )
        negative_real = make_parts_map(
            negative_real_values,
            negative_real_vectors,
            perturbations_up,
            lifted_pairs,
        )
        negative_null = make_parts_map(
            negative_null_values,
            negative_null_vectors,
            perturbations_up,
            lifted_pairs,
        )
        negative_deltas = delta_map(negative_real, negative_null)

        branch_results: list[BranchDiagnostic] = []
        for branch in (0, 1):
            offset = branch * base.DIM
            branch_labels = tuple(index + offset for index in labels)
            target_label = base.TARGET + offset
            control_labels = tuple(index + offset for index in base.CONTROLS)

            base_real_branch = {
                (family, index + offset): source_real_parts[(family, index)]
                for family in base.FAMILIES for index in labels
            }
            base_null_branch = {
                (family, index + offset): source_null_parts[(family, index)]
                for family in base.FAMILIES for index in labels
            }

            new_deltas = delta_map(sorted_real, sorted_null)
            branch_subset = {
                (family, label): new_deltas[(family, label)]
                for family in base.FAMILIES for label in branch_labels
            }
            prominence, target_score, control_median = prominence_from_deltas(
                branch_subset, target_label, control_labels
            )
            negative_subset = {
                (family, label): negative_deltas[(family, label)]
                for family in base.FAMILIES for label in branch_labels
            }
            negative_prominence, _, _ = prominence_from_deltas(
                negative_subset, target_label, control_labels
            )

            rn_values: list[float] = []
            norm_values: list[float] = []
            for family in base.FAMILIES:
                for index in labels:
                    label = index + offset
                    dr = sorted_real[(family, label)].score - source_real_parts[(family, index)].score
                    dn = sorted_null[(family, label)].score - source_null_parts[(family, index)].score
                    rn = None if rho == 0.0 else cancellation_fraction(dr, dn)
                    if rn is not None:
                        rn_values.append(rn)
                    for old, new in (
                        (source_real_parts[(family, index)], sorted_real[(family, label)]),
                        (source_null_parts[(family, index)], sorted_null[(family, label)]),
                    ):
                        nc = (
                            None if rho == 0.0
                            else log_normalization_cancellation(old, new)
                        )
                        if nc is not None:
                            norm_values.append(nc)

            tc = (
                None if rho == 0.0 else cancellation_fraction(
                    target_score - base_target,
                    control_median - base_control,
                )
            )

            tracked_real_pairs: dict[int, tuple[int, int]] = {}
            tracked_null_pairs: dict[int, tuple[int, int]] = {}
            fidelities: list[float] = []
            for index in labels:
                label = index + offset
                real_pair, real_fidelity = best_overlap_pair(
                    real0_vectors, lifted_pairs[label], real_vectors
                )
                null_pair, null_fidelity = best_overlap_pair(
                    null0_vectors, lifted_pairs[label], null_vectors
                )
                tracked_real_pairs[label] = real_pair
                tracked_null_pairs[label] = null_pair
                fidelities.extend((real_fidelity, null_fidelity))

            tracked_real = make_parts_map(
                real_values, real_vectors, perturbations_up, tracked_real_pairs
            )
            tracked_null = make_parts_map(
                null_values, null_vectors, perturbations_up, tracked_null_pairs
            )
            tracked_deltas = delta_map(tracked_real, tracked_null)
            tracked_prominence, _, _ = prominence_from_deltas(
                tracked_deltas, target_label, control_labels
            )

            branch_results.append(BranchDiagnostic(
                seed=seed,
                rho=rho,
                branch=branch,
                base_prominence=base_prominence,
                prominence=prominence,
                rn_cancellation=(float(np.median(rn_values)) if rn_values else None),
                normalization_cancellation=(float(np.median(norm_values)) if norm_values else None),
                target_control_cancellation=tc,
                tracked_prominence=tracked_prominence,
                tracking_discrepancy=abs(prominence - tracked_prominence),
                minimum_tracking_fidelity=min(fidelities),
                tracking_sign_disagreement=(
                    strict_sign(prominence) != strict_sign(tracked_prominence)
                ),
                parity_residual=abs(prominence - negative_prominence),
            ))

        change0 = branch_results[0].prominence - base_prominence
        change1 = branch_results[1].prominence - base_prominence
        symmetry = (
            None if rho == 0.0 else cancellation_fraction(change0, change1)
        )
        results.append(SeedRhoDiagnostic(
            seed=seed,
            rho=rho,
            base_prominence=base_prominence,
            branch_symmetry=symmetry,
            branch_split=abs(
                branch_results[0].prominence - branch_results[1].prominence
            ),
            branches=(branch_results[0], branch_results[1]),
        ))
    return results


def median_optional(values: list[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    return float(np.median(kept)) if kept else None


def fmt_fraction(value: float | None) -> str:
    return " n/a " if value is None else f"{100.0 * value:5.1f}%"


def response_exponent(
    seed_results: list[SeedRhoDiagnostic], branch: int
) -> float:
    """Log-log slope of |prominence(rho)-prominence(0)| on frozen low-rho grid."""
    base_prominence = seed_results[0].base_prominence
    xs: list[float] = []
    ys: list[float] = []
    for row in seed_results:
        if row.rho not in LOW_RHOS:
            continue
        displacement = abs(row.branches[branch].prominence - base_prominence)
        if displacement > 1.0e-14:
            xs.append(float(np.log(row.rho)))
            ys.append(float(np.log(displacement)))
    if len(xs) < 3:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def render(all_results: list[list[SeedRhoDiagnostic]]) -> str:
    flat = [item for seed_results in all_results for item in seed_results]
    lines = [
        "=== Soft Spaces Phase 2 v25.36 STABILITY-MECHANISM DECOMPOSITION ===",
        f"Frozen target: {base.TARGET}; controls: {list(base.CONTROLS)}",
        f"Seeds: {list(base.BASE_SEEDS)}; families: {list(base.FAMILIES)}",
        f"rho grid: {list(stress.RHO_GRID)}; eta: {base.REGULARIZER:.3e}",
        "Same deterministic X_a tensor G_s direction as v25.34.",
        "No fitting, relocation, re-ranking, seed removal, or post-run grid change.",
        "Cancellation diagnostics overlap and must not be added as causal percentages.",
        "",
        "AGGREGATE BY RHO (all five seeds and both branches retained)",
        "rho | branch symmetry | REAL-NULL cancel | N/D normalize | target-control cancel | max branch split | min track fidelity | max sorted-track gap | sign disagreements",
    ]
    for rho in stress.RHO_GRID:
        rows = [row for row in flat if row.rho == rho]
        branches = [branch for row in rows for branch in row.branches]
        symmetry = median_optional([row.branch_symmetry for row in rows])
        rn = median_optional([branch.rn_cancellation for branch in branches])
        norm = median_optional(
            [branch.normalization_cancellation for branch in branches]
        )
        tc = median_optional(
            [branch.target_control_cancellation for branch in branches]
        )
        sign_disagreements = sum(
            branch.tracking_sign_disagreement for branch in branches
        )
        lines.append(
            f"{rho:7.1e} | {fmt_fraction(symmetry):>15s} | {fmt_fraction(rn):>16s} | "
            f"{fmt_fraction(norm):>13s} | {fmt_fraction(tc):>21s} | "
            f"{max(row.branch_split for row in rows):16.3e} | "
            f"{min(branch.minimum_tracking_fidelity for branch in branches):18.6f} | "
            f"{max(branch.tracking_discrepancy for branch in branches):20.3e} | "
            f"{sign_disagreements:18d}"
        )

    lines.extend([
        "",
        "KEY-RHO DETAIL (negative source seeds included)",
        "seed | rho | base prom | prom b0/b1 | branch symmetry | RN cancel b0/b1 | N/D cancel b0/b1 | TC cancel b0/b1 | min fidelity | max track gap | track sign diff",
    ])
    for row in flat:
        if row.rho not in KEY_RHOS:
            continue
        b0, b1 = row.branches
        lines.append(
            f"{row.seed:7d} | {row.rho:4.1f} | {row.base_prominence:+9.3e} | "
            f"{b0.prominence:+8.3e}/{b1.prominence:+8.3e} | "
            f"{fmt_fraction(row.branch_symmetry):>15s} | "
            f"{fmt_fraction(b0.rn_cancellation)}/{fmt_fraction(b1.rn_cancellation)} | "
            f"{fmt_fraction(b0.normalization_cancellation)}/{fmt_fraction(b1.normalization_cancellation)} | "
            f"{fmt_fraction(b0.target_control_cancellation)}/{fmt_fraction(b1.target_control_cancellation)} | "
            f"{min(b0.minimum_tracking_fidelity, b1.minimum_tracking_fidelity):12.6f} | "
            f"{max(b0.tracking_discrepancy, b1.tracking_discrepancy):13.3e} | "
            f"{'YES' if b0.tracking_sign_disagreement or b1.tracking_sign_disagreement else 'NO'}"
        )

    nonzero = [row for row in flat if row.rho > 0.0]
    nonzero_branches = [branch for row in nonzero for branch in row.branches]
    lines.extend([
        "",
        "LOW-COUPLING RESPONSE EXPONENT",
        "seed | exponent branch 0 | exponent branch 1",
    ])
    exponents: list[float] = []
    for seed_results in all_results:
        exponent0 = response_exponent(seed_results, 0)
        exponent1 = response_exponent(seed_results, 1)
        exponents.extend((exponent0, exponent1))
        lines.append(
            f"{seed_results[0].seed:7d} | {exponent0:17.6f} | {exponent1:17.6f}"
        )

    lines.extend([
        "",
        "GLOBAL DIAGNOSTIC SUMMARY (all nonzero rho points)",
        f"  Median low-coupling response exponent: {float(np.nanmedian(exponents)):.6f} (quadratic response predicts 2).",
        f"  Maximum +rho versus -rho prominence residual: {max(b.parity_residual for b in nonzero_branches):.6e}.",
        f"  Median branch-symmetry cancellation: {fmt_fraction(median_optional([row.branch_symmetry for row in nonzero]))}.",
        f"  Median REAL-NULL common-motion cancellation: {fmt_fraction(median_optional([b.rn_cancellation for b in nonzero_branches]))}.",
        f"  Median numerator/denominator log-cancellation: {fmt_fraction(median_optional([b.normalization_cancellation for b in nonzero_branches]))}.",
        f"  Median target-control common-mode cancellation: {fmt_fraction(median_optional([b.target_control_cancellation for b in nonzero_branches]))}.",
        f"  Minimum overlap-tracking fidelity: {min(b.minimum_tracking_fidelity for b in nonzero_branches):.6f}.",
        f"  Maximum sorted-versus-tracked prominence gap: {max(b.tracking_discrepancy for b in nonzero_branches):.6e}.",
        f"  Sorted/tracked sign disagreements: {sum(b.tracking_sign_disagreement for b in nonzero_branches)}/{len(nonzero_branches)} branch-points.",
        "",
        "Interpretation rules:",
        "  100% cancellation means the two compared motions are equal and disappear in their difference; 0% means no common-mode cancellation.",
        "  Tracking fidelity is Tr(P_initial P_tracked)/2; values near 1 indicate the same two-dimensional subspace.",
        "  These diagnostics describe the frozen coupling path only. They are not an ensemble-wide proof and do not sum to 100%.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v25_36_stability_mechanism_decomposition_output.txt"),
    )
    args = parser.parse_args()
    results = [run_seed(seed) for seed in base.BASE_SEEDS]
    report = render(results)
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
