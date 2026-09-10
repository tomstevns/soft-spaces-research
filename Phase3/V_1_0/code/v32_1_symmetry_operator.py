#!/usr/bin/env python3
"""Soft Spaces Phase 3.2 v32.1 — symmetry-operator probe.

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


VERSION = "v32.1"
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
# v32.1 symmetry-operator diagnostics
# ---------------------------------------------------------------------------

def pauli_matrix_from_label(label: str) -> np.ndarray:
    """Dense Pauli-string matrix using the frozen convention."""
    return dense_pauli_sum(len(label), [(label, 1.0)])


def anticommutation_residual(
    gamma: np.ndarray,
    hamiltonian: np.ndarray,
) -> float:
    """
    Relative Frobenius residual for Gamma H Gamma† = -H.

    Returns ||Gamma H Gamma† + H||_F / ||H||_F.
    """
    lhs = gamma @ hamiltonian @ gamma.conj().T + hamiltonian
    denom = float(np.linalg.norm(hamiltonian, ord="fro"))
    return float(np.linalg.norm(lhs, ord="fro") / denom)


def commutation_residual(
    gamma: np.ndarray,
    operator: np.ndarray,
    sign: int,
) -> float:
    """
    Relative residual for Gamma O Gamma† = sign * O, sign in {+1,-1}.
    """
    lhs = gamma @ operator @ gamma.conj().T - sign * operator
    denom = float(np.linalg.norm(operator, ord="fro"))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(lhs, ord="fro") / denom)


def pauli_symmetry_candidates(
    h_terms: list[tuple[str, float]],
) -> list[str]:
    """
    Search all non-identity Pauli strings and retain those that anticommute
    with every Hamiltonian Pauli term individually.

    For a Pauli-string Hamiltonian H=sum c_j P_j, this is a strong structural
    criterion for Gamma H Gamma† = -H.
    """
    candidates: list[str] = []

    # Generate all 4^N Pauli strings in the same label convention.
    for idx in range(4**N_QUBITS - 1):
        gamma_label = index_to_label(idx, N_QUBITS)
        if set(gamma_label) == {"I"}:
            continue

        good = True
        for h_label, _ in h_terms:
            if not pauli_strings_anticommute(gamma_label, h_label):
                good = False
                break

        if good:
            candidates.append(gamma_label)

    return candidates


def pauli_strings_anticommute(a: str, b: str) -> bool:
    """
    Two Pauli strings anticommute iff the number of sites with distinct
    non-identity Paulis is odd.
    """
    count = 0
    for sa, sb in zip(a, b):
        if sa == "I" or sb == "I" or sa == sb:
            continue
        count += 1
    return (count % 2) == 1


def pauli_strings_commute(a: str, b: str) -> bool:
    return not pauli_strings_anticommute(a, b)


def operator_sign_under_pauli(
    gamma_label: str,
    operator_terms: list[tuple[str, float]],
) -> int | None:
    """
    If every Pauli term in O transforms with the same sign under Gamma,
    return +1 or -1. Otherwise return None.
    """
    signs = set()
    for label, _ in operator_terms:
        signs.add(-1 if pauli_strings_anticommute(gamma_label, label) else +1)
    if len(signs) == 1:
        return signs.pop()
    return None


def eigenspace_mapping_residual(
    gamma: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    groups: list[np.ndarray],
) -> tuple[float, float]:
    """
    Check whether Gamma maps each +E eigenspace onto the matching -E eigenspace.

    Returns median and maximum projector mismatch:
        || Gamma P_E Gamma† - P_-E ||_F
    over exact ±E pairs.
    """
    energies = np.asarray(
        [eigenspace_energy(eigenvalues, g) for g in groups],
        dtype=float,
    )
    pairs = opposite_energy_pairs(energies)

    residuals = []

    for a, b in pairs:
        pa = eigenvectors[:, groups[a]]
        pb = eigenvectors[:, groups[b]]

        proj_a = pa @ pa.conj().T
        proj_b = pb @ pb.conj().T

        mapped = gamma @ proj_a @ gamma.conj().T
        residuals.append(
            float(np.linalg.norm(mapped - proj_b, ord="fro"))
        )

    if not residuals:
        return float("nan"), float("nan")

    arr = np.asarray(residuals, dtype=float)
    return float(np.median(arr)), float(np.max(arr))


def format_sign(sign: int | None) -> str:
    if sign == +1:
        return "+V"
    if sign == -1:
        return "-V"
    return "mixed"


def run_symmetry_probe(seed: int) -> str:
    h_terms = random_hamiltonian_terms(seed)
    hamiltonian = dense_pauli_sum(N_QUBITS, h_terms)

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    groups = degenerate_groups(eigenvalues)

    perturbations = {
        family: family_terms(
            stable_hash_int(
                f"{FROZEN_MODEL_VERSION}|{family}|PERT|{seed}"
            ),
            family,
        )
        for family in FAMILIES
    }

    candidates = pauli_symmetry_candidates(h_terms)

    lines = [
        "=== Soft Spaces Phase 3.2 v32.1 SYMMETRY-OPERATOR PROBE ===",
        f"Seed={seed}",
        f"{N_QUBITS}Q; Hamiltonian terms={N_TERMS}",
        f"Eigenspaces={len(groups)}",
        f"Exact ±E pairs={len(opposite_energy_pairs(np.asarray([eigenspace_energy(eigenvalues,g) for g in groups])))}",
        "",
        "Primary question:",
        "  Is there a concrete Pauli-string Gamma satisfying",
        "      Gamma H Gamma† = -H ?",
        "",
        "Hamiltonian terms:",
    ]

    for label, coeff in h_terms:
        lines.append(f"  {coeff:+.9f}  {label}")

    lines.extend(["", f"Pauli candidates that anticommute term-by-term with H: {len(candidates)}"])

    if not candidates:
        lines.extend([
            "",
            "RESULT:",
            "  No Pauli-string Gamma was found that anticommutes with every H term.",
            "  The ±E symmetry may therefore arise from a more general unitary or",
            "  antiunitary symmetry, or from a structured block form not generated",
            "  by a single Pauli string.",
        ])
        return "\n".join(lines) + "\n"

    scored = []
    for gamma_label in candidates:
        gamma = pauli_matrix_from_label(gamma_label)
        h_res = anticommutation_residual(gamma, hamiltonian)

        mapping_median, mapping_max = eigenspace_mapping_residual(
            gamma,
            eigenvalues,
            eigenvectors,
            groups,
        )

        family_info = {}
        for family in FAMILIES:
            sign_rule = operator_sign_under_pauli(
                gamma_label,
                perturbations[family],
            )

            op = dense_pauli_sum(
                N_QUBITS,
                perturbations[family],
            )

            plus_res = commutation_residual(gamma, op, +1)
            minus_res = commutation_residual(gamma, op, -1)

            family_info[family] = (
                sign_rule,
                plus_res,
                minus_res,
            )

        scored.append(
            (
                h_res,
                mapping_max,
                gamma_label,
                mapping_median,
                family_info,
            )
        )

    scored.sort(key=lambda item: (item[0], item[1], item[2]))

    lines.extend(["", "Candidate diagnostics:"])

    for rank, item in enumerate(scored[:20], start=1):
        h_res, mapping_max, gamma_label, mapping_median, family_info = item
        lines.append(
            f"{rank:2d}. Gamma={gamma_label}  "
            f"H-anticomm residual={h_res:.3e}  "
            f"projector-map median={mapping_median:.3e} max={mapping_max:.3e}"
        )

        for family in FAMILIES:
            sign_rule, plus_res, minus_res = family_info[family]
            lines.append(
                f"      {family:10s}: rule={format_sign(sign_rule):5s}; "
                f"res(+V)={plus_res:.3e}; res(-V)={minus_res:.3e}"
            )

    best = scored[0]
    h_res, mapping_max, gamma_label, mapping_median, family_info = best

    lines.extend([
        "",
        "=== BEST CANDIDATE ===",
        f"Gamma = {gamma_label}",
        f"Relative ||Gamma H Gamma† + H||_F / ||H||_F = {h_res:.6e}",
        f"±E projector-map residual: median={mapping_median:.6e}, max={mapping_max:.6e}",
        "",
        "Perturbation transformation:",
    ])

    for family in FAMILIES:
        sign_rule, plus_res, minus_res = family_info[family]
        lines.append(
            f"  {family}: algebraic rule={format_sign(sign_rule)}, "
            f"res(+V)={plus_res:.6e}, res(-V)={minus_res:.6e}"
        )

    lines.extend([
        "",
        "INTERPRETATION:",
    ])

    if h_res < 1.0e-12 and mapping_max < 1.0e-10:
        lines.append(
            "  STRONG: a concrete Pauli-string symmetry generator has been found."
        )
        lines.append(
            "  It anticommutes with H to numerical precision and maps E eigenspaces"
        )
        lines.append(
            "  onto their -E partners."
        )
    elif h_res < 1.0e-9:
        lines.append(
            "  PARTIAL: Gamma anticommutes with H numerically, but eigenspace mapping"
        )
        lines.append(
            "  is not yet exact enough to claim the full projector relation."
        )
    else:
        lines.append(
            "  WEAK: candidate search found only approximate Pauli symmetries."
        )

    lines.extend([
        "",
        "Next theoretical relation to test if STRONG:",
        "  P_-E = Gamma P_E Gamma†",
        "  Q_-E = Gamma Q_E Gamma†",
        "  Gamma V Gamma† = ±V  (or family-specific rule)",
        "",
        "These imply equality of unitary-invariant norms such as",
        "  ||P_E V Q_E||_F = ||P_-E V Q_-E||_F,",
        "and strongly constrain the second-order effective operator Sigma_P(E).",
    ])

    return "\n".join(lines) + "\n"

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=HAMILTONIAN_SEEDS[0],
        help=f"Frozen Hamiltonian seed (default {HAMILTONIAN_SEEDS[0]}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v32_1_symmetry_operator_output.txt"),
        help="Text report path.",
    )

    args = parser.parse_args()

    if args.seed not in HAMILTONIAN_SEEDS:
        raise ValueError(
            f"--seed must be one of {HAMILTONIAN_SEEDS[0]}..{HAMILTONIAN_SEEDS[-1]}"
        )

    report = run_symmetry_probe(args.seed)
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
