#!/usr/bin/env python3
"""Soft Spaces Phase 3.2 v32.0 — P/Q mechanism probe.

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

This is a diagnostic mechanism probe, not a significance test and not a
new hotspot scan.  By default it runs the first frozen seed and reports
the strongest five exact ±E pairs.  Use --all-seeds for all 12 frozen seeds.

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


VERSION = "v32.0"
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

    # Frozen spectral gap label, comparable between REAL and NULL.
    gap_pq = float(
        np.min(np.abs(reference - eigenvalues[q_indices]))
    )

    if basis_name == "REAL":
        # QHQ is diagonal because q_vectors are H eigenvectors.
        delta = reference - eigenvalues[q_indices]
        g = regularized_inverse_scalar(delta, eta)
        sigma = (pvq * g[None, :]) @ qvp

    elif basis_name == "NULL":
        # In a random Q decomposition, QHQ is not diagonal.  Diagonalize QHQ
        # and apply the same regularized spectral function g_eta(E - QHQ).
        hq = q_vectors.conj().T @ (hamiltonian @ q_vectors)
        hq = 0.5 * (hq + hq.conj().T)

        q_evals, q_rot = np.linalg.eigh(hq)
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
    Rank exact ±E pairs by the stronger member's frozen discovery prominence.

    This does NOT create a new score.  It only selects a small diagnostic set
    from the existing Phase-3.1 prominence landscape.
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
    return ranked[:top_pairs]


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
        "=== Soft Spaces Phase 3.2 v32.0 P/Q MECHANISM PROBE ===",
        f"{N_QUBITS}Q; {N_TERMS}-term Hamiltonian; seeds={len(seeds)}",
        f"Frozen physical-model namespace={FROZEN_MODEL_VERSION}",
        f"Families={','.join(FAMILIES)}; perturbation terms/family={PERTURB_TERMS}",
        f"Degeneracy tolerance={DEGENERACY_TOL:.1e}",
        f"±E pairing tolerance={PAIR_ENERGY_TOL:.1e}",
        f"Regularized resolvent eta={eta:.3e}",
        f"Top exact ±E pairs per seed={top_pairs}",
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
        "The natural follow-up would be v32.1: controlled attenuation or",
        "scrambling of P<->Q coupling while holding the rest of the model fixed.",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v32_0_mechanism_probe_output.txt"),
        help="Text report path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("v32_0_mechanism_probe.csv"),
        help="CSV diagnostic table path.",
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Run all 12 frozen Phase-3.1 Hamiltonian seeds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run one explicit frozen Hamiltonian seed.",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=5,
        help="Number of strongest exact ±E hotspot pairs per seed (default 5).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=ENERGY_REG,
        help=f"Regularized resolvent eta (default {ENERGY_REG}).",
    )

    args = parser.parse_args()

    if args.top_pairs < 1:
        raise ValueError("--top-pairs must be >= 1.")
    if args.eta <= 0.0:
        raise ValueError("--eta must be > 0.")

    if args.seed is not None:
        if args.seed not in HAMILTONIAN_SEEDS:
            raise ValueError(
                f"--seed must be one of {HAMILTONIAN_SEEDS[0]}..{HAMILTONIAN_SEEDS[-1]}"
            )
        seeds = (args.seed,)
    elif args.all_seeds:
        seeds = HAMILTONIAN_SEEDS
    else:
        # Deliberately small diagnostic default.
        seeds = HAMILTONIAN_SEEDS[:1]

    all_rows: list[DiagnosticRow] = []
    report_lines = render_header(
        seeds=seeds,
        top_pairs=args.top_pairs,
        eta=args.eta,
    )

    for seed in seeds:
        rows, seed_report = run_seed(
            seed=seed,
            top_pairs=args.top_pairs,
            eta=args.eta,
        )
        all_rows.extend(rows)
        report_lines.extend(seed_report)

    report_lines.extend(render_footer(all_rows))
    report = "\n".join(report_lines) + "\n"

    print(report, end="")
    args.output.write_text(report, encoding="utf-8")
    write_csv(args.csv, all_rows)


if __name__ == "__main__":
    main()
