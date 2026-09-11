#!/usr/bin/env python3
"""Soft Spaces Phase 3.2 v32.3 — local hotspot mechanism probe.

Primary question
----------------
Can the REAL-specific ±E organization observed in Phase 3.1 be associated
with a specific coupling structure between the degeneracy-closed P subspace
and its complement Q?

Frozen from Phase 3.1
---------------------
* 8 qubits.
* 11-term random-Pauli Hamiltonian.
* Hamiltonian seeds 25_042_000 ... 25_042_011.
* Exact degeneracy tolerance 1e-10.
* Exact ±E pairing tolerance 1e-8.
* NULL basis = frozen Haar-unitary namespace from Phase 3.1.
* Perturbation families = dephasing Z/ZZ and transverse X/XX.
* Five perturbation terms per family.
* energy_reg = 1e-3.
* Local prominence radius = 5 eigenspaces.
* Discovery/ranking score is the frozen REAL-minus-NULL robust prominence.

New in v32.0
------------
For selected hotspot ±E pairs, and separately for REAL and NULL, measure

    ||P V P||_F
    ||P V Q||_F
    ||P V Q||_F^2
    Delta_PQ = min relevant |E_P - E_Q|
    Sigma_P(E) = P V Q G_Q(E) Q V P
    ||Sigma_P(E)||_F

where the regularized resolvent uses

    g_eta(x) = x / (x^2 + eta^2).

For REAL, QHQ is diagonal in the Hamiltonian eigenbasis.  For NULL, the
same operator expression is evaluated in the random P/Q decomposition:
QHQ is explicitly constructed and diagonalized before applying g_eta.

This is a focused control experiment, not a significance test and not a
new hotspot scan.  It runs the first frozen seed (25042000) and evaluates
all 32 exact ±E pairs, preserving the frozen Phase-3.1 prominence only as
a ranking variable.  It also compares TOP, MIDDLE and BOTTOM prominence
tertiles to test whether ±E symmetry is hotspot-specific or global.

Outputs
-------
1. Human-readable text report.
2. CSV with one row per seed / pair / sign / family / basis (REAL or NULL).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np


VERSION = "v32.3"
FROZEN_MODEL_VERSION = "v30.2"

N_QUBITS = 8
DIM = 1 << N_QUBITS
N_TERMS = 11
PERTURB_TERMS = 5
ENERGY_REG = 1.0e-3
DEGENERACY_TOL = 1.0e-10
PAIR_ENERGY_TOL = 1.0e-8
LOCAL_GROUP_RADIUS = 5

HAMILTONIAN_SEEDS = tuple(range(25_042_000, 25_042_012))
FAMILIES = ("dephasing", "transverse")
PAULIS = ("I", "X", "Y", "Z")


# ---------------------------------------------------------------------------
# Frozen Phase-3.1 model kernel
# ---------------------------------------------------------------------------

def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def index_to_label(index: int, n_qubits: int) -> str:
    value = int(index) + 1
    digits: list[str] = []
    for _ in range(n_qubits):
        value, remainder = divmod(value, 4)
        digits.append(PAULIS[remainder])
    return "".join(reversed(digits))


def pauli_permutation_phase(label: str) -> tuple[np.ndarray, np.ndarray]:
    basis = np.arange(1 << len(label), dtype=np.int64)
    flip_mask = 0
    phase = np.ones(basis.size, dtype=np.complex128)

    for position, symbol in enumerate(label):
        bit_number = len(label) - 1 - position
        bit = (basis >> bit_number) & 1

        if symbol in ("X", "Y"):
            flip_mask |= 1 << bit_number

        if symbol == "Z":
            phase *= 1.0 - 2.0 * bit
        elif symbol == "Y":
            phase *= 1j * (1.0 - 2.0 * bit)

    return basis ^ flip_mask, phase


def dense_pauli_sum(
    n_qubits: int,
    terms: Iterable[tuple[str, float]],
) -> np.ndarray:
    dimension = 1 << n_qubits
    columns = np.arange(dimension, dtype=np.int64)
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)

    for label, coefficient in terms:
        rows, phase = pauli_permutation_phase(label)
        matrix[rows, columns] += float(coefficient) * phase

    # Numerical hygiene: all Pauli sums here are Hermitian by construction.
    return 0.5 * (matrix + matrix.conj().T)


def apply_pauli_sum(
    columns: np.ndarray,
    terms: Iterable[tuple[str, float]],
) -> np.ndarray:
    result = np.zeros_like(columns, dtype=np.complex128)

    for label, coefficient in terms:
        rows, phase = pauli_permutation_phase(label)
        result[rows, :] += float(coefficient) * phase[:, None] * columns

    return result


def random_hamiltonian_terms(seed: int) -> list[tuple[str, float]]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        4**N_QUBITS - 1,
        size=N_TERMS,
    )
    coefficients = rng.uniform(
        -1.0,
        1.0,
        size=N_TERMS,
    )

    return [
        (index_to_label(int(index), N_QUBITS), float(coefficient))
        for index, coefficient in zip(indices, coefficients)
    ]


def family_terms(seed: int, family: str) -> list[tuple[str, float]]:
    rng = np.random.default_rng(seed)
    symbol = "Z" if family == "dephasing" else "X"

    labels: list[str] = []

    for qubit in range(N_QUBITS):
        label = ["I"] * N_QUBITS
        label[qubit] = symbol
        labels.append("".join(label))

    for qubit in range(N_QUBITS - 1):
        label = ["I"] * N_QUBITS
        label[qubit] = symbol
        label[qubit + 1] = symbol
        labels.append("".join(label))

    selected = rng.choice(
        len(labels),
        size=PERTURB_TERMS,
        replace=False,
    )
    coefficients = rng.uniform(
        -1.0,
        1.0,
        size=PERTURB_TERMS,
    )

    return [
        (labels[int(index)], float(coefficient))
        for index, coefficient in zip(selected, coefficients)
    ]


def haar_unitary(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = (
        rng.normal(size=(dimension, dimension))
        + 1j * rng.normal(size=(dimension, dimension))
    ) / math.sqrt(2.0)

    q, r = np.linalg.qr(raw)
    diagonal = np.diag(r)
    phases = np.ones_like(diagonal)

    nonzero = np.abs(diagonal) > 0.0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])

    return q * phases.conj()[None, :]


def degenerate_groups(eigenvalues: np.ndarray) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    start = 0

    for index in range(1, eigenvalues.size + 1):
        if (
            index == eigenvalues.size
            or abs(float(eigenvalues[index] - eigenvalues[start]))
            > DEGENERACY_TOL
        ):
            groups.append(np.arange(start, index, dtype=int))
            start = index

    return groups


def eigenspace_energy(
    eigenvalues: np.ndarray,
    group: np.ndarray,
) -> float:
    return float(np.mean(eigenvalues[group]))


def eigenspace_cancel_score(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    perturbation: list[tuple[str, float]],
    groups: list[np.ndarray],
    target_group_number: int,
) -> float | None:
    """Frozen Phase-3.1 projector cancellation score."""
    p_indices = groups[target_group_number]
    p_vectors = eigenvectors[:, p_indices]

    acted_p = apply_pauli_sum(p_vectors, perturbation)
    b_to_p = eigenvectors.conj().T @ acted_p

    reference = eigenspace_energy(eigenvalues, p_indices)
    effective = np.zeros(
        (p_indices.size, p_indices.size),
        dtype=np.complex128,
    )
    denominator = 0.0

    for group_number, group in enumerate(groups):
        if group_number == target_group_number:
            continue

        e_group = eigenspace_energy(eigenvalues, group)
        delta = reference - e_group
        inverse = delta / (
            delta * delta + ENERGY_REG * ENERGY_REG
        )

        w_group = b_to_p[group, :]
        a_group = w_group.conj().T @ w_group

        effective += inverse * a_group
        denominator += (
            abs(inverse) * float(np.linalg.norm(a_group, ord="fro"))
        )

    if denominator <= 0.0 or not np.isfinite(denominator):
        return None

    numerator = float(np.linalg.norm(effective, ord="fro"))
    if not np.isfinite(numerator):
        return None

    return numerator / denominator


def model_scores(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    perturbation: list[tuple[str, float]],
    groups: list[np.ndarray],
) -> dict[int, float]:
    output: dict[int, float] = {}

    for group_number in range(len(groups)):
        value = eigenspace_cancel_score(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            perturbation=perturbation,
            groups=groups,
            target_group_number=group_number,
        )
        if value is not None:
            output[group_number] = value

    return output


def robust_difference(
    real_by_family: dict[str, dict[int, float]],
    null_by_family: dict[str, dict[int, float]],
    group_count: int,
) -> dict[int, float]:
    output: dict[int, float] = {}

    for group_number in range(group_count):
        values: list[float] = []

        for family in FAMILIES:
            if (
                group_number not in real_by_family[family]
                or group_number not in null_by_family[family]
            ):
                values = []
                break

            values.append(
                real_by_family[family][group_number]
                - null_by_family[family][group_number]
            )

        if values:
            output[group_number] = min(values)

    return output


def local_neighbours(
    group_number: int,
    group_count: int,
) -> list[int]:
    start = max(0, group_number - LOCAL_GROUP_RADIUS)
    stop = min(group_count, group_number + LOCAL_GROUP_RADIUS + 1)

    return [
        index
        for index in range(start, stop)
        if index != group_number
    ]


def prominence_values(
    robust: dict[int, float],
    group_count: int,
) -> dict[int, float]:
    output: dict[int, float] = {}

    for group_number, target in robust.items():
        neighbours = [
            robust[index]
            for index in local_neighbours(group_number, group_count)
            if index in robust
        ]

        if not neighbours:
            continue

        output[group_number] = float(
            target - np.median(neighbours)
        )

    return output


def opposite_energy_pairs(
    energies: np.ndarray,
) -> list[tuple[int, int]]:
    """Frozen greedy exact ±E pairing rule from Phase 3.1."""
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()

    for i, energy in enumerate(energies):
        if i in used:
            continue

        candidates = [
            j
            for j in range(len(energies))
            if j != i and j not in used
        ]

        if not candidates:
            continue

        j = min(
            candidates,
            key=lambda idx: abs(energies[idx] + energy),
        )

        if abs(float(energies[j] + energy)) <= PAIR_ENERGY_TOL:
            a, b = sorted((i, j))
            pairs.append((a, b))
            used.add(a)
            used.add(b)

    return sorted(set(pairs))


# ---------------------------------------------------------------------------
# Phase-3.2 mechanism diagnostics
# ---------------------------------------------------------------------------

def complement_indices(
    dimension: int,
    p_indices: np.ndarray,
) -> np.ndarray:
    mask = np.ones(dimension, dtype=bool)
    mask[p_indices] = False
    return np.flatnonzero(mask)


def regularized_inverse_scalar(delta: np.ndarray | float, eta: float):
    return delta / (delta * delta + eta * eta)


@dataclass(frozen=True)
class DiagnosticRow:
    seed: int
    pair_rank: int
    group: int
    partner_group: int
    sign: str
    energy: float
    partner_energy: float
    dim_p: int
    family: str
    basis: str
    pvp_fro: float
    pvq_fro: float
    pvq_fro_sq: float
    gap_pq: float
    sigma_fro: float
    sigma_trace_abs: float
    discovery_prominence: float


def pq_diagnostics(
    *,
    hamiltonian: np.ndarray,
    eigenvalues: np.ndarray,
    basis_vectors: np.ndarray,
    groups: list[np.ndarray],
    target_group_number: int,
    perturbation: list[tuple[str, float]],
    eta: float,
    basis_name: str,
) -> tuple[float, float, float, float, float, float]:
    """
    Evaluate PVP, PVQ and the regularized second-order self-energy.

    The reference energy E is always the frozen spectral energy assigned to the
    target exact eigenspace.  REAL uses the Hamiltonian eigenbasis.  NULL uses
    the frozen Haar basis, while QHQ is computed as an operator in that random
    Q subspace.
    """
    p_indices = groups[target_group_number]
    q_indices = complement_indices(DIM, p_indices)

    p_vectors = basis_vectors[:, p_indices]
    q_vectors = basis_vectors[:, q_indices]

    acted_p = apply_pauli_sum(p_vectors, perturbation)

    # Q V P and P V P.  Since V is Hermitian, PVQ = (QVP)†.
    pvp = p_vectors.conj().T @ acted_p
    qvp = q_vectors.conj().T @ acted_p
    pvq = qvp.conj().T

    pvp_fro = float(np.linalg.norm(pvp, ord="fro"))
    pvq_fro = float(np.linalg.norm(pvq, ord="fro"))
    pvq_fro_sq = pvq_fro * pvq_fro

    reference = eigenspace_energy(
        eigenvalues,
        groups[target_group_number],
    )

    if basis_name == "REAL":
        # QHQ is diagonal because q_vectors are H eigenvectors.
        q_evals = eigenvalues[q_indices]
        gap_pq = float(
            np.min(np.abs(reference - q_evals))
        )
        delta = reference - q_evals
        g = regularized_inverse_scalar(delta, eta)
        sigma = (pvq * g[None, :]) @ qvp

    elif basis_name == "NULL":
        # In a random Q decomposition, QHQ is not diagonal.  Diagonalize QHQ
        # and use its actual spectrum for BOTH the gap and the regularized
        # second-order operator.
        hq = q_vectors.conj().T @ (hamiltonian @ q_vectors)
        hq = 0.5 * (hq + hq.conj().T)

        q_evals, q_rot = np.linalg.eigh(hq)
        gap_pq = float(
            np.min(np.abs(reference - q_evals))
        )
        g_diag = regularized_inverse_scalar(reference - q_evals, eta)

        # Rotate QVP into the QHQ eigenbasis; no explicit inverse is formed.
        qvp_rot = q_rot.conj().T @ qvp
        sigma = (qvp_rot.conj().T * g_diag[None, :]) @ qvp_rot

    else:
        raise ValueError(f"Unknown basis_name: {basis_name}")

    sigma = 0.5 * (sigma + sigma.conj().T)
    sigma_fro = float(np.linalg.norm(sigma, ord="fro"))
    sigma_trace_abs = float(abs(np.trace(sigma)))

    return (
        pvp_fro,
        pvq_fro,
        pvq_fro_sq,
        gap_pq,
        sigma_fro,
        sigma_trace_abs,
    )


def discovery_prominence_for_seed(
    *,
    seed: int,
    eigenvalues: np.ndarray,
    real_vectors: np.ndarray,
    null_vectors: np.ndarray,
    groups: list[np.ndarray],
    perturbations: dict[str, list[tuple[str, float]]],
) -> dict[int, float]:
    """Recompute the frozen v30.2/v30.5 REAL-minus-NULL prominence."""
    real_by_family = {
        family: model_scores(
            eigenvalues,
            real_vectors,
            perturbations[family],
            groups,
        )
        for family in FAMILIES
    }

    null_by_family = {
        family: model_scores(
            eigenvalues,
            null_vectors,
            perturbations[family],
            groups,
        )
        for family in FAMILIES
    }

    diff_robust = robust_difference(
        real_by_family,
        null_by_family,
        len(groups),
    )

    return prominence_values(
        diff_robust,
        len(groups),
    )


def choose_hotspot_pairs(
    *,
    energies: np.ndarray,
    prominence: dict[int, float],
    top_pairs: int,
) -> list[tuple[int, int, float]]:
    """
    Rank ALL exact ±E pairs by the stronger member's frozen discovery prominence.

    v32.0.1 deliberately evaluates the complete ±E pair set for seed 25042000.
    The prominence is retained only for ordering and TOP/MIDDLE/BOTTOM grouping.
    """
    pairs = opposite_energy_pairs(energies)

    ranked: list[tuple[int, int, float]] = []
    for a, b in pairs:
        pa = prominence.get(a, float("-inf"))
        pb = prominence.get(b, float("-inf"))
        score = max(pa, pb)

        if np.isfinite(score):
            ranked.append((a, b, float(score)))

    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked


def run_seed(
    *,
    seed: int,
    top_pairs: int,
    eta: float,
) -> tuple[list[DiagnosticRow], list[str]]:
    h_terms = random_hamiltonian_terms(seed)
    hamiltonian = dense_pauli_sum(N_QUBITS, h_terms)

    eigenvalues, real_vectors = np.linalg.eigh(hamiltonian)
    groups = degenerate_groups(eigenvalues)

    null_vectors = haar_unitary(
        DIM,
        stable_hash_int(
            f"{FROZEN_MODEL_VERSION}|NULL|{seed}|terms{N_TERMS}"
        ),
    )

    perturbations = {
        family: family_terms(
            stable_hash_int(
                f"{FROZEN_MODEL_VERSION}|{family}|PERT|{seed}"
            ),
            family,
        )
        for family in FAMILIES
    }

    prominence = discovery_prominence_for_seed(
        seed=seed,
        eigenvalues=eigenvalues,
        real_vectors=real_vectors,
        null_vectors=null_vectors,
        groups=groups,
        perturbations=perturbations,
    )

    energies = np.asarray(
        [
            eigenspace_energy(eigenvalues, group)
            for group in groups
        ],
        dtype=float,
    )

    selected = choose_hotspot_pairs(
        energies=energies,
        prominence=prominence,
        top_pairs=top_pairs,
    )

    rows: list[DiagnosticRow] = []
    report: list[str] = []

    sizes = [int(group.size) for group in groups]
    report.extend(
        [
            f"SEED {seed}",
            f"  eigenspaces={len(groups)}; size={min(sizes)}:{max(sizes)}",
            f"  exact ±E pairs={len(opposite_energy_pairs(energies))}",
            f"  selected diagnostic pairs={len(selected)}",
            "",
        ]
    )

    for rank, (ga, gb, pair_score) in enumerate(selected, start=1):
        # Put negative/less-positive energy first for readable ± ordering.
        if energies[ga] <= energies[gb]:
            g_minus, g_plus = ga, gb
        else:
            g_minus, g_plus = gb, ga

        report.append(
            f"PAIR {rank}: groups {g_minus} / {g_plus}; "
            f"E={energies[g_minus]:+.9f} / {energies[g_plus]:+.9f}; "
            f"selection prominence={pair_score:+.9f}"
        )

        pair_rows: dict[tuple[str, str, str], DiagnosticRow] = {}

        for sign, group_number, partner_number in (
            ("-", g_minus, g_plus),
            ("+", g_plus, g_minus),
        ):
            for family in FAMILIES:
                for basis_name, basis_vectors in (
                    ("REAL", real_vectors),
                    ("NULL", null_vectors),
                ):
                    (
                        pvp_fro,
                        pvq_fro,
                        pvq_fro_sq,
                        gap_pq,
                        sigma_fro,
                        sigma_trace_abs,
                    ) = pq_diagnostics(
                        hamiltonian=hamiltonian,
                        eigenvalues=eigenvalues,
                        basis_vectors=basis_vectors,
                        groups=groups,
                        target_group_number=group_number,
                        perturbation=perturbations[family],
                        eta=eta,
                        basis_name=basis_name,
                    )

                    row = DiagnosticRow(
                        seed=seed,
                        pair_rank=rank,
                        group=group_number,
                        partner_group=partner_number,
                        sign=sign,
                        energy=float(energies[group_number]),
                        partner_energy=float(energies[partner_number]),
                        dim_p=int(groups[group_number].size),
                        family=family,
                        basis=basis_name,
                        pvp_fro=pvp_fro,
                        pvq_fro=pvq_fro,
                        pvq_fro_sq=pvq_fro_sq,
                        gap_pq=gap_pq,
                        sigma_fro=sigma_fro,
                        sigma_trace_abs=sigma_trace_abs,
                        discovery_prominence=float(
                            prominence.get(group_number, float("nan"))
                        ),
                    )

                    rows.append(row)
                    pair_rows[(sign, family, basis_name)] = row

        # Compact pair-level display.
        for family in FAMILIES:
            report.append(f"  {family}:")
            for sign in ("-", "+"):
                r = pair_rows[(sign, family, "REAL")]
                n = pair_rows[(sign, family, "NULL")]

                report.append(
                    f"    {sign}E REAL "
                    f"PVP={r.pvp_fro:.6e}  PVQ²={r.pvq_fro_sq:.6e}  "
                    f"gap={r.gap_pq:.6e}  Sigma={r.sigma_fro:.6e}"
                )
                report.append(
                    f"       NULL "
                    f"PVP={n.pvp_fro:.6e}  PVQ²={n.pvq_fro_sq:.6e}  "
                    f"gap={n.gap_pq:.6e}  Sigma={n.sigma_fro:.6e}"
                )
                report.append(
                    f"       Δ(R-N) "
                    f"PVP={r.pvp_fro-n.pvp_fro:+.6e}  "
                    f"PVQ²={r.pvq_fro_sq-n.pvq_fro_sq:+.6e}  "
                    f"Sigma={r.sigma_fro-n.sigma_fro:+.6e}"
                )

            # ±E organization diagnostic for Sigma and PVQ².
            rm = pair_rows[("-", family, "REAL")]
            rp = pair_rows[("+", family, "REAL")]
            nm = pair_rows[("-", family, "NULL")]
            np_ = pair_rows[("+", family, "NULL")]

            asym_sigma_real = abs(rp.sigma_fro - rm.sigma_fro)
            asym_sigma_null = abs(np_.sigma_fro - nm.sigma_fro)
            asym_pvq_real = abs(rp.pvq_fro_sq - rm.pvq_fro_sq)
            asym_pvq_null = abs(np_.pvq_fro_sq - nm.pvq_fro_sq)

            report.append(
                f"    ±E asymmetry: "
                f"|ΔSigma| REAL={asym_sigma_real:.6e}, NULL={asym_sigma_null:.6e}; "
                f"|ΔPVQ²| REAL={asym_pvq_real:.6e}, NULL={asym_pvq_null:.6e}"
            )

        report.append("")

    return rows, report


def tertile_label(rank_index: int, pair_count: int) -> str:
    """TOP / MIDDLE / BOTTOM by frozen pair prominence rank."""
    if pair_count <= 0:
        return "NA"
    frac = rank_index / pair_count
    if frac <= 1.0 / 3.0:
        return "TOP"
    if frac <= 2.0 / 3.0:
        return "MIDDLE"
    return "BOTTOM"


def pair_level_summary(rows: list[DiagnosticRow]) -> list[str]:
    """
    Summarize ±E asymmetry by prominence tertile.

    One observation = one pair/family/basis.  We report median and maximum
    asymmetry in Sigma and PVQ², plus median PVP for context.
    """
    if not rows:
        return ["No rows available for global summary."]

    pair_ranks = sorted(set(r.pair_rank for r in rows))
    pair_count = len(pair_ranks)

    lines = [
        "",
        "=== GLOBAL ±E SYMMETRY SUMMARY ===",
        f"Exact ±E pairs analysed={pair_count}",
        "Grouping: TOP / MIDDLE / BOTTOM by frozen Phase-3.1 pair prominence.",
        "",
    ]

    records = []
    for rank in pair_ranks:
        for family in FAMILIES:
            for basis in ("REAL", "NULL"):
                subset = [
                    r for r in rows
                    if r.pair_rank == rank
                    and r.family == family
                    and r.basis == basis
                ]
                by_sign = {r.sign: r for r in subset}
                if "-" not in by_sign or "+" not in by_sign:
                    continue

                rm = by_sign["-"]
                rp = by_sign["+"]

                records.append({
                    "rank": rank,
                    "tertile": tertile_label(rank, pair_count),
                    "family": family,
                    "basis": basis,
                    "sigma_asym": abs(rp.sigma_fro - rm.sigma_fro),
                    "pvq2_asym": abs(rp.pvq_fro_sq - rm.pvq_fro_sq),
                    "pvp_mean": 0.5 * (rp.pvp_fro + rm.pvp_fro),
                })

    for tertile in ("TOP", "MIDDLE", "BOTTOM"):
        lines.append(f"{tertile}:")
        for family in FAMILIES:
            for basis in ("REAL", "NULL"):
                vals = [
                    rec for rec in records
                    if rec["tertile"] == tertile
                    and rec["family"] == family
                    and rec["basis"] == basis
                ]
                if not vals:
                    continue

                sigma = np.asarray([v["sigma_asym"] for v in vals], dtype=float)
                pvq2 = np.asarray([v["pvq2_asym"] for v in vals], dtype=float)
                pvp = np.asarray([v["pvp_mean"] for v in vals], dtype=float)

                lines.append(
                    f"  {family:10s} {basis:4s}  "
                    f"|ΔSigma| median={np.median(sigma):.6e} max={np.max(sigma):.6e}; "
                    f"|ΔPVQ²| median={np.median(pvq2):.6e} max={np.max(pvq2):.6e}; "
                    f"PVP median={np.median(pvp):.6e}"
                )
        lines.append("")

    # Global REAL/NULL comparison independent of tertile.
    lines.append("ALL PAIRS:")
    for family in FAMILIES:
        for basis in ("REAL", "NULL"):
            vals = [
                rec for rec in records
                if rec["family"] == family and rec["basis"] == basis
            ]
            sigma = np.asarray([v["sigma_asym"] for v in vals], dtype=float)
            pvq2 = np.asarray([v["pvq2_asym"] for v in vals], dtype=float)

            lines.append(
                f"  {family:10s} {basis:4s}  n={len(vals):2d}; "
                f"|ΔSigma| median={np.median(sigma):.6e}, max={np.max(sigma):.6e}; "
                f"|ΔPVQ²| median={np.median(pvq2):.6e}, max={np.max(pvq2):.6e}"
            )

    # Simple classification statement; deliberately deterministic, not inferential.
    real_sigma = np.asarray(
        [v["sigma_asym"] for v in records if v["basis"] == "REAL"],
        dtype=float,
    )
    null_sigma = np.asarray(
        [v["sigma_asym"] for v in records if v["basis"] == "NULL"],
        dtype=float,
    )

    lines.extend(["", "CONTROL QUESTION:"])
    if real_sigma.size and null_sigma.size:
        real_max = float(np.max(real_sigma))
        null_min = float(np.min(null_sigma))
        if real_max < 1.0e-9 and null_min > 1.0e-6:
            lines.append(
                "  RESULT: REAL ±E symmetry is global across the analysed pair set "
                "to numerical precision, while NULL is not."
            )
            lines.append(
                "  INTERPRETATION: the symmetry is therefore not hotspot-specific "
                "for this seed; hotspots sit on top of a broader algebraic structure."
            )
        else:
            lines.append(
                "  RESULT: REAL ±E symmetry is not uniformly machine-precision across "
                "the complete pair set, so hotspot-specific structure remains plausible."
            )

    lines.append("")
    lines.append(
        "No causality claim is made here. This control only distinguishes "
        "global algebraic symmetry from hotspot-local symmetry."
    )
    return lines

# ---------------------------------------------------------------------------
# v32.3 local hotspot-mechanism analysis
# ---------------------------------------------------------------------------

def rankdata_simple(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties; scipy-free Spearman support."""
    values=np.asarray(values, dtype=float)
    order=np.argsort(values, kind="mergesort")
    ranks=np.empty(len(values), dtype=float)
    i=0
    while i < len(values):
        j=i+1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg=0.5*((i+1)+j)
        ranks[order[i:j]]=avg
        i=j
    return ranks

def corr_pair(x: np.ndarray, y: np.ndarray) -> tuple[float,float]:
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    mask=np.isfinite(x)&np.isfinite(y)
    x=x[mask]; y=y[mask]
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:
        return float('nan'), float('nan')
    pear=float(np.corrcoef(x,y)[0,1])
    rx=rankdata_simple(x); ry=rankdata_simple(y)
    spear=float(np.corrcoef(rx,ry)[0,1])
    return pear,spear

def local_contrast(values: np.ndarray) -> np.ndarray:
    """Value minus median of immediate neighbours in energy-ordered pair list."""
    out=np.full(len(values), np.nan, dtype=float)
    for i in range(len(values)):
        neigh=[]
        if i>0: neigh.append(values[i-1])
        if i+1<len(values): neigh.append(values[i+1])
        if neigh: out[i]=values[i]-float(np.median(neigh))
    return out

def pair_metrics_for_family(seed:int, family:str, eta:float):
    h_terms=random_hamiltonian_terms(seed)
    h=dense_pauli_sum(N_QUBITS,h_terms)
    evals, real_vecs=np.linalg.eigh(h)
    groups=degenerate_groups(evals)
    null_vecs=haar_unitary(DIM, stable_hash_int(f"{FROZEN_MODEL_VERSION}|NULL|{seed}|terms{N_TERMS}"))
    pert=family_terms(stable_hash_int(f"{FROZEN_MODEL_VERSION}|{family}|PERT|{seed}"), family)
    real_by_family={f:model_scores(evals,real_vecs, family_terms(stable_hash_int(f"{FROZEN_MODEL_VERSION}|{f}|PERT|{seed}"),f),groups) for f in FAMILIES}
    null_by_family={f:model_scores(evals,null_vecs, family_terms(stable_hash_int(f"{FROZEN_MODEL_VERSION}|{f}|PERT|{seed}"),f),groups) for f in FAMILIES}
    prom=prominence_values(robust_difference(real_by_family,null_by_family,len(groups)),len(groups))
    energies=np.asarray([eigenspace_energy(evals,g) for g in groups],dtype=float)
    pairs=opposite_energy_pairs(energies)
    rows=[]
    for a,b in pairs:
        gm,gp=(a,b) if energies[a] <= energies[b] else (b,a)
        pair_prom=max(prom.get(gm,float('-inf')),prom.get(gp,float('-inf')))
        if not np.isfinite(pair_prom):
            continue
        vals={}
        for basis_name,basis_vecs in (("REAL",real_vecs),("NULL",null_vecs)):
            signvals=[]
            for g in (gm,gp):
                signvals.append(pq_diagnostics(hamiltonian=h,eigenvalues=evals,basis_vectors=basis_vecs,groups=groups,target_group_number=g,perturbation=pert,eta=eta,basis_name=basis_name))
            arr=np.asarray(signvals,dtype=float)
            # columns: pvp, pvq, pvq2, gap, sigma, sigma_trace_abs
            vals[basis_name]={
                'pvp':float(np.mean(arr[:,0])),
                'pvq2':float(np.mean(arr[:,2])),
                'gap':float(np.mean(arr[:,3])),
                'sigma':float(np.mean(arr[:,4])),
                'pvq2_asym':float(abs(arr[1,2]-arr[0,2])),
                'sigma_asym':float(abs(arr[1,4]-arr[0,4])),
            }
        rr=vals['REAL']; nn=vals['NULL']
        eps=1e-15
        rows.append({
            'seed':seed,'family':family,'g_minus':gm,'g_plus':gp,
            'E_abs':float(abs(energies[gp])),'prominence':float(pair_prom),
            'real_pvp':rr['pvp'],'null_pvp':nn['pvp'],'delta_pvp':rr['pvp']-nn['pvp'],
            'real_pvq2':rr['pvq2'],'null_pvq2':nn['pvq2'],'delta_pvq2':rr['pvq2']-nn['pvq2'],
            'real_sigma':rr['sigma'],'null_sigma':nn['sigma'],'delta_sigma':rr['sigma']-nn['sigma'],
            'real_gap':rr['gap'],'null_gap':nn['gap'],'delta_gap':rr['gap']-nn['gap'],
            'real_pvq2_over_gap':rr['pvq2']/max(rr['gap'],eps),
            'null_pvq2_over_gap':nn['pvq2']/max(nn['gap'],eps),
            'delta_pvq2_over_gap':rr['pvq2']/max(rr['gap'],eps)-nn['pvq2']/max(nn['gap'],eps),
            'real_sigma_over_gap':rr['sigma']/max(rr['gap'],eps),
            'null_sigma_over_gap':nn['sigma']/max(nn['gap'],eps),
            'delta_sigma_over_gap':rr['sigma']/max(rr['gap'],eps)-nn['sigma']/max(nn['gap'],eps),
            'real_pvq2_asym':rr['pvq2_asym'],'null_pvq2_asym':nn['pvq2_asym'],
            'real_sigma_asym':rr['sigma_asym'],'null_sigma_asym':nn['sigma_asym'],
        })
    rows.sort(key=lambda r:r['E_abs'])
    for key in ('real_pvq2','real_sigma','real_gap','delta_pvq2','delta_sigma','delta_gap'):
        cont=local_contrast(np.asarray([r[key] for r in rows],dtype=float))
        for r,v in zip(rows,cont): r['local_'+key]=float(v)
    return rows

def write_local_csv(path:Path, rows:list[dict]):
    if not rows: return
    import csv
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)




# ---------------------------------------------------------------------------
# v32.6 Q-spectrum weighted descriptors
# ---------------------------------------------------------------------------

def q_spectrum_descriptors(
    *,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    group: np.ndarray,
    perturbation: list[tuple[str, float]],
    eta: float,
) -> dict:
    """
    Characterize the Q spectrum seen from one P eigenspace.

    We use the frozen regularized second-order kernel
        g_eta(delta) = delta / (delta^2 + eta^2)
    and the actual coupling weights
        w_q = || <q|V|P> ||_F^2.

    New descriptors:
      q_gap1              nearest |E-E_q|
      q_gap3_mean         mean of three nearest gaps
      q_gap5_mean         mean of five nearest gaps
      q_density_eta       sum 1/(delta^2+eta^2)
      q_weight_near       coupling weight in five nearest Q levels
      q_weighted_abs_g    sum w_q * |g_eta(delta)|
      q_weighted_g2       sum w_q * g_eta(delta)^2
      q_weighted_invabs   sum w_q / sqrt(delta^2+eta^2)
      q_effective_n       participation-like effective number of weighted Q levels
    """
    p_vec = eigenvectors[:, group]
    q_idx = complement_indices(DIM, group)
    q_vec = eigenvectors[:, q_idx]

    energy = eigenspace_energy(eigenvalues, group)
    delta = energy - eigenvalues[q_idx]
    abs_delta = np.abs(delta)

    acted_p = apply_pauli_sum(p_vec, perturbation)
    qvp = q_vec.conj().T @ acted_p

    # Coupling weight per Q eigenstate, summed over P columns.
    w = np.sum(np.abs(qvp) ** 2, axis=1).real

    order = np.argsort(abs_delta)
    nearest = abs_delta[order]
    k3 = min(3, nearest.size)
    k5 = min(5, nearest.size)

    g = regularized_inverse_scalar(delta, eta)
    abs_g = np.abs(g)

    total_w = float(np.sum(w))
    w_norm = w / total_w if total_w > 0 else np.zeros_like(w)
    eff_n = 1.0 / float(np.sum(w_norm**2)) if np.sum(w_norm**2) > 0 else float("nan")

    return {
        "q_gap1": float(nearest[0]),
        "q_gap3_mean": float(np.mean(nearest[:k3])),
        "q_gap5_mean": float(np.mean(nearest[:k5])),
        "q_density_eta": float(np.sum(1.0 / (delta * delta + eta * eta))),
        "q_weight_near": float(np.sum(w[order[:k5]])),
        "q_weighted_abs_g": float(np.sum(w * abs_g)),
        "q_weighted_g2": float(np.sum(w * (g * g))),
        "q_weighted_invabs": float(np.sum(w / np.sqrt(delta * delta + eta * eta))),
        "q_effective_n": float(eff_n),
    }


def pair_metrics_for_family_v326(seed:int, family:str, eta:float) -> list[dict]:
    """
    Reuse the frozen v32.3 pair metrics and append Q-spectrum descriptors.
    Pair descriptors are averaged over the two ±E members because v32.2
    established exact REAL ±E covariance for the physical model.
    """
    rows = pair_metrics_for_family(seed, family, eta)

    h_terms = random_hamiltonian_terms(seed)
    h = dense_pauli_sum(N_QUBITS, h_terms)
    evals, evecs = np.linalg.eigh(h)
    groups = degenerate_groups(evals)
    energies = np.asarray([eigenspace_energy(evals,g) for g in groups], dtype=float)
    perturbation = family_terms(
        stable_hash_int(f"{FROZEN_MODEL_VERSION}|{family}|PERT|{seed}"),
        family,
    )

    # pair_metrics_for_family returns rows sorted by exact ±E pairs and carries pair_rank/group ids.
    by_group = {}
    for gi, group in enumerate(groups):
        by_group[gi] = q_spectrum_descriptors(
            eigenvalues=evals,
            eigenvectors=evecs,
            group=group,
            perturbation=perturbation,
            eta=eta,
        )

    for row in rows:
        gm = int(row["g_minus"])
        gp = int(row["g_plus"])
        dm = by_group[gm]
        dp = by_group[gp]
        for key in dm:
            row[key] = 0.5 * (float(dm[key]) + float(dp[key]))
    return rows


FEATURES_V326 = [
    "real_sigma",
    "delta_sigma",
    "real_gap",
    "local_real_sigma",
    "q_gap3_mean",
    "q_gap5_mean",
    "q_density_eta",
    "q_weight_near",
    "q_weighted_abs_g",
    "q_weighted_g2",
    "q_weighted_invabs",
    "q_effective_n",
]


def build_xy_v326(rows: list[dict], family: str):
    sub = [r for r in rows if r["family"] == family]
    X = np.asarray([[float(r[f]) for f in FEATURES_V326] for r in sub], dtype=float)
    y = np.asarray([float(r["prominence"]) for r in sub], dtype=float)
    seeds = np.asarray([int(r["seed"]) for r in sub], dtype=int)
    return sub, X, y, seeds


def loso_family_v326(rows: list[dict], family: str, alpha: float):
    sub, X, y, seeds = build_xy_v326(rows, family)
    unique_seeds = sorted(set(int(s) for s in seeds))
    folds=[]; all_true=[]; all_pred=[]
    for held in unique_seeds:
        train=seeds!=held; test=seeds==held
        mu,sd=standardize_fit(X[train])
        Xtr=standardize_apply(X[train],mu,sd)
        Xte=standardize_apply(X[test],mu,sd)
        beta=fit_ridge(Xtr,y[train],alpha)
        pred=predict_ridge(Xte,beta)
        yt=y[test]
        all_true.extend(yt.tolist()); all_pred.extend(pred.tolist())
        folds.append({
            "family":family,"held_seed":held,"n_test":int(np.sum(test)),
            "r2":r2_score_np(yt,pred),"rmse":rmse_np(yt,pred),
            "spearman":spearman_np(yt,pred),
        })
    return folds,np.asarray(all_true),np.asarray(all_pred)


def full_fit_coefficients_v326(rows: list[dict], family: str, alpha: float):
    _,X,y,_=build_xy_v326(rows,family)
    mu,sd=standardize_fit(X)
    Xs=standardize_apply(X,mu,sd)
    beta=fit_ridge(Xs,y,alpha)
    coeffs=list(zip(FEATURES_V326,beta[1:]))
    coeffs.sort(key=lambda x:abs(x[1]),reverse=True)
    return beta[0],coeffs


def univariate_q_correlations(rows:list[dict], family:str):
    keys=[
        "q_gap3_mean","q_gap5_mean","q_density_eta","q_weight_near",
        "q_weighted_abs_g","q_weighted_g2","q_weighted_invabs","q_effective_n"
    ]
    out=[]
    sub=[r for r in rows if r["family"]==family]
    y=np.asarray([r["prominence"] for r in sub],float)
    for key in keys:
        x=np.asarray([r[key] for r in sub],float)
        p,s=corr_pair(x,y)
        out.append((key,p,s))
    out.sort(key=lambda t:abs(t[2]),reverse=True)
    return out


def summary_v326(rows:list[dict], alpha:float):
    lines=[
        "=== Soft Spaces Phase 3.2 v32.6 Q-SPECTRUM WEIGHTED HOTSPOT MECHANISM ===",
        f"Seeds={len(set(r['seed'] for r in rows))}",
        f"Ridge alpha={alpha}",
        "",
        "Question:",
        "  Does the missing cross-seed hotspot information reside in the local",
        "  Q-spectrum and in how V-weighted Q levels enter the regularized resolvent?",
        "",
        "New descriptors:",
    ]
    for f in FEATURES_V326[4:]:
        lines.append(f"  - {f}")
    lines.append("")

    all_folds=[]
    for fam in FAMILIES:
        folds,yt,yp=loso_family_v326(rows,fam,alpha)
        all_folds.extend(folds)
        rho=spearman_np(yt,yp); r2=r2_score_np(yt,yp); rm=rmse_np(yt,yp)
        overlap=top_bottom_classification(yt,yp)
        fold_rhos=[f["spearman"] for f in folds if np.isfinite(f["spearman"])]
        pos=sum(v>0 for v in fold_rhos)
        _,coeffs=full_fit_coefficients_v326(rows,fam,alpha)

        lines += [
            f"FAMILY: {fam}",
            f"  LOSO pooled: Spearman rho={rho:+.4f}; R2={r2:+.4f}; "
            f"RMSE={rm:.6e}; top-third overlap={overlap:.3f}",
            f"  Fold Spearman median={np.median(fold_rhos):+.4f}; "
            f"positive={pos}/{len(fold_rhos)}; "
            f"range=[{np.min(fold_rhos):+.4f},{np.max(fold_rhos):+.4f}]",
            "  Strongest standardized coefficients:",
        ]
        for name,val in coeffs[:8]:
            lines.append(f"    {name:24s} {val:+.6e}")

        lines.append("  Pooled univariate Q-descriptor correlations:")
        for name,p,s in univariate_q_correlations(rows,fam)[:6]:
            lines.append(f"    {name:24s} Pearson={p:+.4f}  Spearman={s:+.4f}")
        lines.append("")

    lines += [
        "=== INTERPRETATION GATE ===",
        "Compare v32.6 against v32.5, especially transverse:",
        "  v32.5 pooled LOSO Spearman = +0.2991",
        "  v32.5 pooled LOSO R2       = +0.0693",
        "  v32.5 fold Spearman median = +0.4011",
        "",
        "A material improvement would indicate that the missing hotspot variable",
        "is not just a minimum gap, but the weighted local Q-spectrum entering Sigma.",
        "No universal closed-form law is claimed by this diagnostic alone.",
    ]
    return "\n".join(lines)+"\n", all_folds


def write_rows_v326(path:Path, rows:list[dict]) -> None:
    if not rows: return
    keys=list(rows[0].keys())
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------------------
# v32.7 Sigma matrix-structure descriptors
# ---------------------------------------------------------------------------

def sigma_small_matrix(
    *,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    group: np.ndarray,
    perturbation: list[tuple[str, float]],
    eta: float,
) -> np.ndarray:
    p_vec = eigenvectors[:, group]
    q_idx = complement_indices(DIM, group)
    q_vec = eigenvectors[:, q_idx]

    energy = eigenspace_energy(eigenvalues, group)
    delta = energy - eigenvalues[q_idx]
    g = regularized_inverse_scalar(delta, eta)

    acted_p = apply_pauli_sum(p_vec, perturbation)
    qvp = q_vec.conj().T @ acted_p
    sigma = (qvp.conj().T * g[None, :]) @ qvp
    sigma = 0.5 * (sigma + sigma.conj().T)
    return sigma


def sigma_structure_descriptors(sigma: np.ndarray) -> dict:
    """
    Gauge-invariant descriptors of the Hermitian effective operator Sigma
    within the degeneracy-closed P subspace.
    """
    evals = np.linalg.eigvalsh(sigma).real
    svals = np.abs(evals)

    fro = float(np.linalg.norm(sigma, ord="fro"))
    trace = float(np.trace(sigma).real)
    trace_abs = abs(trace)

    max_abs = float(np.max(svals)) if svals.size else 0.0
    min_abs = float(np.min(svals)) if svals.size else 0.0

    # Remove isotropic component to measure shape/anistropy only.
    d = sigma.shape[0]
    iso = (trace / d) * np.eye(d, dtype=np.complex128)
    traceless = sigma - iso
    traceless_fro = float(np.linalg.norm(traceless, ord="fro"))

    anisotropy = traceless_fro / fro if fro > 0 else 0.0
    trace_ratio = trace_abs / (np.sqrt(d) * fro) if fro > 0 else 0.0

    # Spectral spread and sign cancellation.
    ev_mean = float(np.mean(evals)) if evals.size else 0.0
    ev_std = float(np.std(evals)) if evals.size else 0.0
    ev_range = float(np.max(evals) - np.min(evals)) if evals.size else 0.0

    l1 = float(np.sum(svals))
    l2sq = float(np.sum(svals**2))
    eff_rank = (l1*l1/l2sq) if l2sq > 0 else 0.0

    pos = float(np.sum(evals[evals > 0.0]))
    neg = float(-np.sum(evals[evals < 0.0]))
    cancellation = abs(pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0

    # Spectral dominance: 1 means one mode dominates.
    dominance = max_abs / fro if fro > 0 else 0.0

    return {
        "sigma_trace": trace,
        "sigma_trace_abs": trace_abs,
        "sigma_trace_ratio": trace_ratio,
        "sigma_traceless_fro": traceless_fro,
        "sigma_anisotropy": anisotropy,
        "sigma_eval_std": ev_std,
        "sigma_eval_range": ev_range,
        "sigma_max_abs_eval": max_abs,
        "sigma_min_abs_eval": min_abs,
        "sigma_effective_rank": eff_rank,
        "sigma_sign_cancellation": cancellation,
        "sigma_dominance": dominance,
    }


def pair_metrics_for_family_v327(seed:int, family:str, eta:float) -> list[dict]:
    rows = pair_metrics_for_family(seed, family, eta)

    h_terms = random_hamiltonian_terms(seed)
    h = dense_pauli_sum(N_QUBITS, h_terms)
    evals, evecs = np.linalg.eigh(h)
    groups = degenerate_groups(evals)
    perturbation = family_terms(
        stable_hash_int(f"{FROZEN_MODEL_VERSION}|{family}|PERT|{seed}"),
        family,
    )

    desc = {}
    for gi, group in enumerate(groups):
        sigma = sigma_small_matrix(
            eigenvalues=evals,
            eigenvectors=evecs,
            group=group,
            perturbation=perturbation,
            eta=eta,
        )
        desc[gi] = sigma_structure_descriptors(sigma)

    for row in rows:
        gm = int(row["g_minus"])
        gp = int(row["g_plus"])
        dm = desc[gm]
        dp = desc[gp]
        # Symmetry-established pair average.
        for key in dm:
            row[key] = 0.5 * (float(dm[key]) + float(dp[key]))
    return rows


FEATURES_V327 = [
    "real_sigma",
    "delta_sigma",
    "real_gap",
    "local_real_sigma",
    "sigma_trace_ratio",
    "sigma_traceless_fro",
    "sigma_anisotropy",
    "sigma_eval_std",
    "sigma_eval_range",
    "sigma_max_abs_eval",
    "sigma_effective_rank",
    "sigma_sign_cancellation",
    "sigma_dominance",
]


def build_xy_v327(rows:list[dict], family:str):
    sub=[r for r in rows if r["family"]==family]
    X=np.asarray([[float(r[f]) for f in FEATURES_V327] for r in sub],dtype=float)
    y=np.asarray([float(r["prominence"]) for r in sub],dtype=float)
    seeds=np.asarray([int(r["seed"]) for r in sub],dtype=int)
    return sub,X,y,seeds


def loso_family_v327(rows:list[dict], family:str, alpha:float):
    sub,X,y,seeds=build_xy_v327(rows,family)
    uniq=sorted(set(int(s) for s in seeds))
    folds=[]; all_true=[]; all_pred=[]
    for held in uniq:
        tr=seeds!=held; te=seeds==held
        mu,sd=standardize_fit(X[tr])
        Xtr=standardize_apply(X[tr],mu,sd)
        Xte=standardize_apply(X[te],mu,sd)
        beta=fit_ridge(Xtr,y[tr],alpha)
        pred=predict_ridge(Xte,beta)
        yt=y[te]
        all_true.extend(yt.tolist()); all_pred.extend(pred.tolist())
        folds.append({
            "family":family,"held_seed":held,"n_test":int(np.sum(te)),
            "r2":r2_score_np(yt,pred),"rmse":rmse_np(yt,pred),
            "spearman":spearman_np(yt,pred),
        })
    return folds,np.asarray(all_true),np.asarray(all_pred)


def full_fit_coefficients_v327(rows:list[dict], family:str, alpha:float):
    _,X,y,_=build_xy_v327(rows,family)
    mu,sd=standardize_fit(X)
    Xs=standardize_apply(X,mu,sd)
    beta=fit_ridge(Xs,y,alpha)
    coeffs=list(zip(FEATURES_V327,beta[1:]))
    coeffs.sort(key=lambda x:abs(x[1]),reverse=True)
    return coeffs


def univariate_sigma_structure(rows:list[dict], family:str):
    keys=[
        "sigma_trace_ratio","sigma_traceless_fro","sigma_anisotropy",
        "sigma_eval_std","sigma_eval_range","sigma_max_abs_eval",
        "sigma_effective_rank","sigma_sign_cancellation","sigma_dominance"
    ]
    sub=[r for r in rows if r["family"]==family]
    y=np.asarray([r["prominence"] for r in sub],float)
    out=[]
    for key in keys:
        x=np.asarray([r[key] for r in sub],float)
        p,s=corr_pair(x,y)
        out.append((key,p,s))
    out.sort(key=lambda t:abs(t[2]),reverse=True)
    return out


def summary_v327(rows:list[dict], alpha:float):
    lines=[
        "=== Soft Spaces Phase 3.2 v32.7 SIGMA MATRIX-STRUCTURE HOTSPOT MECHANISM ===",
        f"Seeds={len(set(r['seed'] for r in rows))}",
        f"Ridge alpha={alpha}",
        "",
        "Question:",
        "  Is hotspot prominence encoded in the INTERNAL MATRIX SHAPE of Sigma,",
        "  rather than in ||Sigma||_F or scalar Q-spectrum descriptors alone?",
        "",
        "New matrix descriptors:",
    ]
    for f in FEATURES_V327[4:]:
        lines.append(f"  - {f}")
    lines.append("")

    all_folds=[]
    for fam in FAMILIES:
        folds,yt,yp=loso_family_v327(rows,fam,alpha)
        all_folds.extend(folds)
        rho=spearman_np(yt,yp); r2=r2_score_np(yt,yp); rm=rmse_np(yt,yp)
        overlap=top_bottom_classification(yt,yp)
        fr=[f["spearman"] for f in folds if np.isfinite(f["spearman"])]
        pos=sum(v>0 for v in fr)
        coeffs=full_fit_coefficients_v327(rows,fam,alpha)

        lines += [
            f"FAMILY: {fam}",
            f"  LOSO pooled: Spearman rho={rho:+.4f}; R2={r2:+.4f}; "
            f"RMSE={rm:.6e}; top-third overlap={overlap:.3f}",
            f"  Fold Spearman median={np.median(fr):+.4f}; positive={pos}/{len(fr)}; "
            f"range=[{np.min(fr):+.4f},{np.max(fr):+.4f}]",
            "  Strongest standardized coefficients:",
        ]
        for name,val in coeffs[:8]:
            lines.append(f"    {name:24s} {val:+.6e}")

        lines.append("  Pooled univariate Sigma-structure correlations:")
        for name,p,s in univariate_sigma_structure(rows,fam)[:7]:
            lines.append(f"    {name:24s} Pearson={p:+.4f}  Spearman={s:+.4f}")
        lines.append("")

    lines += [
        "=== INTERPRETATION GATE ===",
        "Reference results:",
        "  v32.5 transverse pooled LOSO Spearman = +0.2991; R2=+0.0693",
        "  v32.6 transverse pooled LOSO Spearman = +0.2571; R2=-0.0578",
        "",
        "A material v32.7 improvement would support the hypothesis that hotspot",
        "strength depends on matrix anisotropy/eigenmode organization inside Sigma.",
        "If it does not improve, the next step should move from static descriptors",
        "to interference/cancellation contributions term-by-term.",
    ]
    return "\n".join(lines)+"\n", all_folds


def write_rows_v327(path:Path, rows:list[dict]) -> None:
    if not rows: return
    keys=list(rows[0].keys())
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------------------
# v32.8 termwise Sigma interference / cancellation descriptors
# ---------------------------------------------------------------------------

def sigma_termwise_descriptors(
    *,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    group: np.ndarray,
    perturbation: list[tuple[str, float]],
    eta: float,
) -> dict:
    """
    Decompose Sigma_E into Q-state contributions:
        Sigma_E = sum_q T_q
    with
        T_q = g_eta(E-E_q) (P V |q>)(<q| V P).

    Measures how strongly the T_q contributions cancel or align.
    """
    p_vec = eigenvectors[:, group]
    q_idx = complement_indices(DIM, group)
    q_vec = eigenvectors[:, q_idx]

    energy = eigenspace_energy(eigenvalues, group)
    delta = energy - eigenvalues[q_idx]
    g = regularized_inverse_scalar(delta, eta)

    acted_p = apply_pauli_sum(p_vec, perturbation)
    qvp = q_vec.conj().T @ acted_p  # shape (nQ, dimP)

    term_norms = []
    signed_scalar = []
    pos_norm_sum = 0.0
    neg_norm_sum = 0.0
    sigma = np.zeros((p_vec.shape[1], p_vec.shape[1]), dtype=np.complex128)

    for i in range(qvp.shape[0]):
        vrow = qvp[i, :][None, :]
        aq = vrow.conj().T @ vrow
        tq = g[i] * aq
        sigma += tq

        nrm = float(np.linalg.norm(tq, ord="fro"))
        term_norms.append(nrm)
        signed_scalar.append(float(g[i] * np.trace(aq).real))

        if g[i] >= 0:
            pos_norm_sum += nrm
        else:
            neg_norm_sum += nrm

    sigma = 0.5 * (sigma + sigma.conj().T)
    sigma_norm = float(np.linalg.norm(sigma, ord="fro"))
    sum_term_norms = float(np.sum(term_norms))
    rss_term_norms = float(np.sqrt(np.sum(np.square(term_norms))))

    # 1 = perfect alignment/no cancellation; 0 = strong cancellation.
    coherence_l1 = sigma_norm / sum_term_norms if sum_term_norms > 0 else 0.0
    coherence_l2 = sigma_norm / rss_term_norms if rss_term_norms > 0 else 0.0

    # Cancellation fractions.
    cancellation_l1 = 1.0 - coherence_l1
    signed_total = abs(float(np.sum(signed_scalar)))
    signed_abs_sum = float(np.sum(np.abs(signed_scalar)))
    scalar_cancellation = 1.0 - (signed_total / signed_abs_sum if signed_abs_sum > 0 else 0.0)

    # Balance of positive- and negative-resolvent sectors.
    pn_balance = (
        abs(pos_norm_sum - neg_norm_sum) / (pos_norm_sum + neg_norm_sum)
        if (pos_norm_sum + neg_norm_sum) > 0 else 0.0
    )

    # Concentration of contribution magnitudes.
    tn = np.asarray(term_norms, dtype=float)
    if np.sum(tn) > 0:
        p = tn / np.sum(tn)
        eff_terms = 1.0 / float(np.sum(p*p))
        max_share = float(np.max(p))
    else:
        eff_terms = 0.0
        max_share = 0.0

    return {
        "sigma_term_sum_norms": sum_term_norms,
        "sigma_term_rss_norms": rss_term_norms,
        "sigma_coherence_l1": coherence_l1,
        "sigma_coherence_l2": coherence_l2,
        "sigma_cancellation_l1": cancellation_l1,
        "sigma_scalar_cancellation": scalar_cancellation,
        "sigma_pn_balance": pn_balance,
        "sigma_effective_terms": eff_terms,
        "sigma_max_term_share": max_share,
    }


def pair_metrics_for_family_v328(seed:int, family:str, eta:float) -> list[dict]:
    rows = pair_metrics_for_family(seed, family, eta)

    h_terms = random_hamiltonian_terms(seed)
    h = dense_pauli_sum(N_QUBITS, h_terms)
    evals, evecs = np.linalg.eigh(h)
    groups = degenerate_groups(evals)
    perturbation = family_terms(
        stable_hash_int(f"{FROZEN_MODEL_VERSION}|{family}|PERT|{seed}"),
        family,
    )

    desc = {}
    for gi, group in enumerate(groups):
        desc[gi] = sigma_termwise_descriptors(
            eigenvalues=evals,
            eigenvectors=evecs,
            group=group,
            perturbation=perturbation,
            eta=eta,
        )

    for row in rows:
        gm = int(row["g_minus"])
        gp = int(row["g_plus"])
        dm = desc[gm]
        dp = desc[gp]
        for key in dm:
            row[key] = 0.5 * (float(dm[key]) + float(dp[key]))
    return rows


FEATURES_V328 = [
    "real_sigma",
    "delta_sigma",
    "real_gap",
    "local_real_sigma",
    "sigma_coherence_l1",
    "sigma_coherence_l2",
    "sigma_cancellation_l1",
    "sigma_scalar_cancellation",
    "sigma_pn_balance",
    "sigma_effective_terms",
    "sigma_max_term_share",
]


def build_xy_v328(rows:list[dict], family:str):
    sub=[r for r in rows if r["family"]==family]
    X=np.asarray([[float(r[f]) for f in FEATURES_V328] for r in sub],dtype=float)
    y=np.asarray([float(r["prominence"]) for r in sub],dtype=float)
    seeds=np.asarray([int(r["seed"]) for r in sub],dtype=int)
    return sub,X,y,seeds


def loso_family_v328(rows:list[dict], family:str, alpha:float):
    sub,X,y,seeds=build_xy_v328(rows,family)
    uniq=sorted(set(int(s) for s in seeds))
    folds=[]; all_true=[]; all_pred=[]
    for held in uniq:
        tr=seeds!=held; te=seeds==held
        mu,sd=standardize_fit(X[tr])
        Xtr=standardize_apply(X[tr],mu,sd)
        Xte=standardize_apply(X[te],mu,sd)
        beta=fit_ridge(Xtr,y[tr],alpha)
        pred=predict_ridge(Xte,beta)
        yt=y[te]
        all_true.extend(yt.tolist()); all_pred.extend(pred.tolist())
        folds.append({
            "family":family,"held_seed":held,"n_test":int(np.sum(te)),
            "r2":r2_score_np(yt,pred),"rmse":rmse_np(yt,pred),
            "spearman":spearman_np(yt,pred),
        })
    return folds,np.asarray(all_true),np.asarray(all_pred)


def full_fit_coefficients_v328(rows:list[dict], family:str, alpha:float):
    _,X,y,_=build_xy_v328(rows,family)
    mu,sd=standardize_fit(X)
    Xs=standardize_apply(X,mu,sd)
    beta=fit_ridge(Xs,y,alpha)
    coeffs=list(zip(FEATURES_V328,beta[1:]))
    coeffs.sort(key=lambda x:abs(x[1]),reverse=True)
    return coeffs


def univariate_termwise(rows:list[dict], family:str):
    keys=[
        "sigma_coherence_l1","sigma_coherence_l2","sigma_cancellation_l1",
        "sigma_scalar_cancellation","sigma_pn_balance","sigma_effective_terms",
        "sigma_max_term_share","sigma_term_sum_norms","sigma_term_rss_norms"
    ]
    sub=[r for r in rows if r["family"]==family]
    y=np.asarray([r["prominence"] for r in sub],float)
    out=[]
    for key in keys:
        x=np.asarray([r[key] for r in sub],float)
        p,s=corr_pair(x,y)
        out.append((key,p,s))
    out.sort(key=lambda t:abs(t[2]),reverse=True)
    return out


def summary_v328(rows:list[dict], alpha:float):
    lines=[
        "=== Soft Spaces Phase 3.2 v32.8 TERMWISE SIGMA INTERFERENCE/CANCELLATION ===",
        f"Seeds={len(set(r['seed'] for r in rows))}",
        f"Ridge alpha={alpha}",
        "",
        "Question:",
        "  Are hotspots controlled by constructive/destructive interference among",
        "  the individual Q-state contributions T_q that sum to Sigma_E?",
        "",
        "New descriptors:",
    ]
    for f in FEATURES_V328[4:]:
        lines.append(f"  - {f}")
    lines.append("")

    all_folds=[]
    for fam in FAMILIES:
        folds,yt,yp=loso_family_v328(rows,fam,alpha)
        all_folds.extend(folds)
        rho=spearman_np(yt,yp); r2=r2_score_np(yt,yp); rm=rmse_np(yt,yp)
        overlap=top_bottom_classification(yt,yp)
        fr=[f["spearman"] for f in folds if np.isfinite(f["spearman"])]
        pos=sum(v>0 for v in fr)
        coeffs=full_fit_coefficients_v328(rows,fam,alpha)

        lines += [
            f"FAMILY: {fam}",
            f"  LOSO pooled: Spearman rho={rho:+.4f}; R2={r2:+.4f}; "
            f"RMSE={rm:.6e}; top-third overlap={overlap:.3f}",
            f"  Fold Spearman median={np.median(fr):+.4f}; positive={pos}/{len(fr)}; "
            f"range=[{np.min(fr):+.4f},{np.max(fr):+.4f}]",
            "  Strongest standardized coefficients:",
        ]
        for name,val in coeffs[:8]:
            lines.append(f"    {name:26s} {val:+.6e}")

        lines.append("  Pooled univariate termwise correlations:")
        for name,p,s in univariate_termwise(rows,fam)[:7]:
            lines.append(f"    {name:26s} Pearson={p:+.4f}  Spearman={s:+.4f}")
        lines.append("")

    lines += [
        "=== INTERPRETATION GATE ===",
        "Reference transverse results:",
        "  v32.5 pooled LOSO Spearman = +0.2991; R2=+0.0693",
        "  v32.6 pooled LOSO Spearman = +0.2571; R2=-0.0578",
        "  v32.7 pooled LOSO Spearman = +0.2846; R2=-0.2352",
        "",
        "A clear v32.8 improvement would support interference/cancellation as the",
        "missing local hotspot mechanism. If not, the next step should move toward",
        "operator-direction alignment between neighboring eigenspaces rather than",
        "more scalar summaries.",
    ]
    return "\n".join(lines)+"\n", all_folds


def write_rows_v328(path:Path, rows:list[dict]) -> None:
    if not rows: return
    keys=list(rows[0].keys())
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------------------
# v32.9 focused coherence-factor confirmation
# ---------------------------------------------------------------------------

def pair_metrics_for_family_v329(seed:int, family:str, eta:float) -> list[dict]:
    rows = pair_metrics_for_family(seed, family, eta)

    h_terms = random_hamiltonian_terms(seed)
    h = dense_pauli_sum(N_QUBITS, h_terms)
    evals, evecs = np.linalg.eigh(h)
    groups = degenerate_groups(evals)
    perturbation = family_terms(
        stable_hash_int(f"{FROZEN_MODEL_VERSION}|{family}|PERT|{seed}"),
        family,
    )

    desc = {}
    for gi, group in enumerate(groups):
        desc[gi] = sigma_termwise_descriptors(
            eigenvalues=evals,
            eigenvectors=evecs,
            group=group,
            perturbation=perturbation,
            eta=eta,
        )

    for row in rows:
        gm = int(row["g_minus"])
        gp = int(row["g_plus"])
        dm = desc[gm]
        dp = desc[gp]
        for key in dm:
            row[key] = 0.5 * (float(dm[key]) + float(dp[key]))
    return rows


def seedwise_coherence_stats(rows:list[dict]) -> list[dict]:
    out=[]
    for seed in sorted(set(r["seed"] for r in rows)):
        for fam in FAMILIES:
            sub=[r for r in rows if r["seed"]==seed and r["family"]==fam]
            y=np.asarray([r["prominence"] for r in sub],float)
            for key in [
                "sigma_coherence_l1",
                "sigma_coherence_l2",
                "sigma_cancellation_l1",
                "sigma_scalar_cancellation",
                "sigma_pn_balance",
            ]:
                x=np.asarray([r[key] for r in sub],float)
                p,s=corr_pair(x,y)
                out.append({
                    "seed":seed,
                    "family":fam,
                    "metric":key,
                    "pearson":p,
                    "spearman":s,
                    "n_pairs":len(sub),
                })
    return out


def write_stats_csv(path:Path, rows:list[dict]) -> None:
    if not rows: return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def focused_summary_v329(rows:list[dict], stats:list[dict]) -> str:
    lines=[
        "=== Soft Spaces Phase 3.2 v32.9 FOCUSED COHERENCE-FACTOR CONFIRMATION ===",
        f"Seeds={len(set(r['seed'] for r in rows))}",
        "",
        "Primary candidate:",
        "  C_E = ||sum_q T_q||_F / sum_q ||T_q||_F",
        "  implemented as sigma_coherence_l1.",
        "",
        "Question:",
        "  Does C_E alone provide a reproducible cross-seed ranking signal for",
        "  hotspot prominence, without a multivariate feature model?",
        "",
    ]

    for fam in FAMILIES:
        lines.append(f"FAMILY: {fam}")
        sub=[r for r in stats if r["family"]==fam]
        for key in [
            "sigma_coherence_l1",
            "sigma_coherence_l2",
            "sigma_cancellation_l1",
            "sigma_scalar_cancellation",
            "sigma_pn_balance",
        ]:
            vals=[r["spearman"] for r in sub if r["metric"]==key and np.isfinite(r["spearman"])]
            if not vals: continue
            pos=sum(v>0 for v in vals)
            neg=sum(v<0 for v in vals)
            lines.append(
                f"  {key:28s} median rho={np.median(vals):+.4f}; "
                f"mean={np.mean(vals):+.4f}; "
                f"positive={pos}/{len(vals)}; negative={neg}/{len(vals)}; "
                f"range=[{np.min(vals):+.4f},{np.max(vals):+.4f}]"
            )

        # pooled direct correlation
        allrows=[r for r in rows if r["family"]==fam]
        y=np.asarray([r["prominence"] for r in allrows],float)
        for key in ["sigma_coherence_l1","sigma_coherence_l2"]:
            x=np.asarray([r[key] for r in allrows],float)
            p,s=corr_pair(x,y)
            lines.append(
                f"  pooled {key:21s} Pearson={p:+.4f}; Spearman={s:+.4f}"
            )

        # Within-seed normalized tertiles for C_E
        trip=[]
        monotonic=0
        for seed in sorted(set(r["seed"] for r in allrows)):
            sr=sorted(
                [r for r in allrows if r["seed"]==seed],
                key=lambda r:r["prominence"], reverse=True
            )
            n=len(sr); c1=(n+2)//3; c2=(2*n+2)//3
            meds=[
                float(np.median([r["sigma_coherence_l1"] for r in sr[:c1]])),
                float(np.median([r["sigma_coherence_l1"] for r in sr[c1:c2]])),
                float(np.median([r["sigma_coherence_l1"] for r in sr[c2:]])),
            ]
            scale=max(float(np.median([r["sigma_coherence_l1"] for r in sr])),1e-15)
            trip.append([m/scale for m in meds])
            monotonic += int(meds[0] > meds[1] > meds[2])

        arr=np.asarray(trip,float)
        med=np.median(arr,axis=0)
        lines.append(
            f"  C_E normalized TOP/MID/BOT medians={med[0]:.3f}/{med[1]:.3f}/{med[2]:.3f}; "
            f"strict TOP>MID>BOTTOM in {monotonic}/{len(arr)} seeds"
        )
        lines.append("")

    lines += [
        "=== DECISION GUIDE ===",
        "The coherence factor is supported as a serious local-law candidate if:",
        "  1) its seedwise Spearman sign is consistently positive,",
        "  2) median seedwise rho is materially above zero,",
        "  3) TOP/MID/BOTTOM ordering is recurrent,",
        "  4) transverse and dephasing show the same physical direction.",
        "",
        "This test intentionally avoids multivariate fitting.",
    ]
    return "\n".join(lines)+"\n"


def write_rows_v329(path:Path, rows:list[dict]) -> None:
    if not rows: return
    keys=list(rows[0].keys())
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader(); w.writerows(rows)

# ---------------------------------------------------------------------------
# v32.5 multivariate cross-seed hotspot model
# ---------------------------------------------------------------------------

FEATURES = [
    "real_sigma",
    "delta_sigma",
    "real_gap",
    "real_pvq2",
    "local_real_sigma",
]

RIDGE_ALPHA = 1.0


def build_xy(rows: list[dict], family: str):
    sub = [r for r in rows if r["family"] == family]
    X = np.asarray([[float(r[f]) for f in FEATURES] for r in sub], dtype=float)
    y = np.asarray([float(r["prominence"]) for r in sub], dtype=float)
    seeds = np.asarray([int(r["seed"]) for r in sub], dtype=int)
    return sub, X, y, seeds


def standardize_fit(X: np.ndarray):
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd = np.where(sd < 1e-15, 1.0, sd)
    return mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float):
    """
    Ridge with unpenalized intercept.
    X is assumed standardized from training data.
    """
    X1 = np.column_stack([np.ones(X.shape[0]), X])
    penalty = np.eye(X1.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(X1.T @ X1 + alpha * penalty, X1.T @ y)
    return beta


def predict_ridge(X: np.ndarray, beta: np.ndarray):
    X1 = np.column_stack([np.ones(X.shape[0]), X])
    return X1 @ beta


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    return corr_pair(np.asarray(x, float), np.asarray(y, float))[1]


def loso_family(rows: list[dict], family: str, alpha: float):
    sub, X, y, seeds = build_xy(rows, family)
    unique_seeds = sorted(set(int(s) for s in seeds))

    fold_rows = []
    all_true = []
    all_pred = []

    for held in unique_seeds:
        train = seeds != held
        test = seeds == held

        mu, sd = standardize_fit(X[train])
        Xtr = standardize_apply(X[train], mu, sd)
        Xte = standardize_apply(X[test], mu, sd)

        beta = fit_ridge(Xtr, y[train], alpha)
        pred = predict_ridge(Xte, beta)

        yt = y[test]
        all_true.extend(yt.tolist())
        all_pred.extend(pred.tolist())

        fold_rows.append({
            "family": family,
            "held_seed": held,
            "n_test": int(np.sum(test)),
            "r2": r2_score_np(yt, pred),
            "rmse": rmse_np(yt, pred),
            "spearman": spearman_np(yt, pred),
        })

    return fold_rows, np.asarray(all_true), np.asarray(all_pred)


def full_fit_coefficients(rows: list[dict], family: str, alpha: float):
    _, X, y, _ = build_xy(rows, family)
    mu, sd = standardize_fit(X)
    Xs = standardize_apply(X, mu, sd)
    beta = fit_ridge(Xs, y, alpha)

    # beta[1:] are directly comparable because features were standardized.
    coeffs = list(zip(FEATURES, beta[1:]))
    coeffs.sort(key=lambda x: abs(x[1]), reverse=True)
    return beta[0], coeffs


def top_bottom_classification(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Simple ranking diagnostic:
    fraction of actual top-third points recovered in predicted top-third.
    """
    n = len(y_true)
    if n < 3:
        return float("nan")
    k = max(1, n // 3)
    actual = set(np.argsort(y_true)[-k:])
    pred = set(np.argsort(y_pred)[-k:])
    return len(actual & pred) / k


def write_fold_csv(path: Path, folds: list[dict]) -> None:
    if not folds:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(folds[0].keys()))
        w.writeheader()
        w.writerows(folds)


def multivariate_summary(rows: list[dict], alpha: float) -> tuple[str, list[dict]]:
    lines = [
        "=== Soft Spaces Phase 3.2 v32.5 MULTIVARIATE CROSS-SEED HOTSPOT MODEL ===",
        f"Seeds={len(set(r['seed'] for r in rows))}",
        f"Ridge alpha={alpha}",
        "",
        "Goal:",
        "  Test whether a SMALL combination of physical descriptors generalizes",
        "  better than any single scalar hotspot descriptor.",
        "",
        "Features:",
    ]
    for f in FEATURES:
        lines.append(f"  - {f}")

    lines += [
        "",
        "Validation:",
        "  Leave-one-seed-out (LOSO). Each held-out seed is never seen during fit.",
        "  Separate models are fit for dephasing and transverse.",
        "",
    ]

    all_folds = []

    for family in FAMILIES:
        folds, y_true, y_pred = loso_family(rows, family, alpha)
        all_folds.extend(folds)

        rho = spearman_np(y_true, y_pred)
        r2 = r2_score_np(y_true, y_pred)
        rmse = rmse_np(y_true, y_pred)
        top_recall = top_bottom_classification(y_true, y_pred)

        intercept, coeffs = full_fit_coefficients(rows, family, alpha)

        lines.append(f"FAMILY: {family}")
        lines.append(
            f"  LOSO pooled: Spearman rho={rho:+.4f}; R2={r2:+.4f}; "
            f"RMSE={rmse:.6e}; top-third overlap={top_recall:.3f}"
        )

        fold_rhos = [f["spearman"] for f in folds if np.isfinite(f["spearman"])]
        fold_r2s = [f["r2"] for f in folds if np.isfinite(f["r2"])]
        positive = sum(v > 0 for v in fold_rhos)

        if fold_rhos:
            lines.append(
                f"  Fold Spearman: median={np.median(fold_rhos):+.4f}; "
                f"positive={positive}/{len(fold_rhos)}; "
                f"range=[{np.min(fold_rhos):+.4f},{np.max(fold_rhos):+.4f}]"
            )
        if fold_r2s:
            lines.append(
                f"  Fold R2: median={np.median(fold_r2s):+.4f}; "
                f"range=[{np.min(fold_r2s):+.4f},{np.max(fold_r2s):+.4f}]"
            )

        lines.append("  Standardized full-fit coefficients:")
        for name, value in coeffs:
            lines.append(f"    {name:22s} {value:+.6e}")
        lines.append("")

    lines += [
        "=== DECISION GUIDE ===",
        "A useful cross-seed local law should ideally show:",
        "  1) positive pooled LOSO Spearman correlation,",
        "  2) positive rank correlation in most held-out seeds,",
        "  3) non-catastrophic held-out R2,",
        "  4) coefficients with interpretable physical structure.",
        "",
        "Interpretation discipline:",
        "  This is a low-dimensional phenomenological test, not black-box ML.",
        "  Even a successful result is a candidate local law, not yet a derivation.",
    ]

    return "\n".join(lines) + "\n", all_folds

def write_csv(path: Path, rows: list[DiagnosticRow]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        "seed", "pair_rank", "group", "partner_group", "sign",
        "energy", "partner_energy", "dim_p", "family", "basis",
        "pvp_fro", "pvq_fro", "pvq_fro_sq", "gap_pq",
        "sigma_fro", "sigma_trace_abs", "discovery_prominence",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def render_header(
    *,
    seeds: tuple[int, ...],
    top_pairs: int,
    eta: float,
) -> list[str]:
    return [
        "=== Soft Spaces Phase 3.2 v32.0.1 GLOBAL ±E SYMMETRY CONTROL ===",
        f"{N_QUBITS}Q; {N_TERMS}-term Hamiltonian; seeds={len(seeds)}",
        f"Frozen physical-model namespace={FROZEN_MODEL_VERSION}",
        f"Families={','.join(FAMILIES)}; perturbation terms/family={PERTURB_TERMS}",
        f"Degeneracy tolerance={DEGENERACY_TOL:.1e}",
        f"±E pairing tolerance={PAIR_ENERGY_TOL:.1e}",
        f"Regularized resolvent eta={eta:.3e}",
        f"Exact ±E pairs requested=ALL (ranking retained; top_pairs arg ignored in v32.0.1)",
        "",
        "Primary hypothesis:",
        "  REAL-specific ±E organization may be associated with structured",
        "  coupling between the degeneracy-closed P subspace and Q.",
        "",
        "Measured per REAL/NULL, family and sign:",
        "  ||PVP||_F, ||PVQ||_F, ||PVQ||_F^2, gap_PQ,",
        "  ||PVQ g_eta(E-QHQ) QVP||_F.",
        "",
        "Important:",
        "  This is a diagnostic mechanism probe. It does not establish causality",
        "  and does not introduce a new hotspot-discovery statistic.",
        "",
    ]


def render_footer(rows: list[DiagnosticRow]) -> list[str]:
    if not rows:
        return ["No diagnostic rows produced."]

    lines = [
        "=== COMPACT INTERPRETATION GUIDE ===",
        "Look first for three linked patterns:",
        "  1) REAL-NULL separation in PVQ² and/or Sigma at hotspot pairs.",
        "  2) smaller ±E asymmetry in REAL than NULL for the same quantity.",
        "  3) recurrence of that pattern across both perturbation families.",
        "",
        "A positive result here is mechanistic evidence, not yet a causal proof.",
        "If REAL symmetry is global, the next task is to identify the algebraic",
        "symmetry operator before attempting a causal P<->Q intervention.",
    ]
    return lines








def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eta",type=float,default=ENERGY_REG)
    parser.add_argument("--output",type=Path,default=Path("v32_9_coherence_factor_output.txt"))
    parser.add_argument("--csv",type=Path,default=Path("v32_9_coherence_factor.csv"))
    parser.add_argument("--stats-csv",type=Path,default=Path("v32_9_coherence_factor_stats.csv"))
    args=parser.parse_args()

    if args.eta<=0:
        raise ValueError("--eta must be > 0")

    rows=[]
    for i,seed in enumerate(HAMILTONIAN_SEEDS,start=1):
        print(f"[{i}/{len(HAMILTONIAN_SEEDS)}] seed {seed}",flush=True)
        for fam in FAMILIES:
            rows.extend(pair_metrics_for_family_v329(seed,fam,args.eta))

    stats=seedwise_coherence_stats(rows)
    report=focused_summary_v329(rows,stats)
    print(report,end="")

    args.output.write_text(report,encoding="utf-8")
    write_rows_v329(args.csv,rows)
    write_stats_csv(args.stats_csv,stats)


if __name__ == "__main__":
    main()
