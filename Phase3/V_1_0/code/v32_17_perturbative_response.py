#!/usr/bin/env python3
"""
Soft Spaces Phase 3.2 — v32.17 Perturbative Response Law
========================================================

Purpose
-------
Test whether the frozen Phase-3 hotspot prominence is associated with the
perturbative susceptibility of the degeneracy-closed effective P-space
feedback operator

    Sigma_E = P V Q g_eta(E - Q H Q) Q V P,

where

    g_eta(x) = x / (x^2 + eta^2).

This is the first, deliberately narrow v32.17 pilot:

    * 8 qubits
    * frozen 11-term Hamiltonian model
    * one seed by default: 25042000
    * dephasing and transverse perturbation families
    * frozen P/Q projectors
    * exact analytic first derivative at lambda = 0
    * small finite-difference convergence check on representative pairs
    * direct comparison with the already frozen hotspot prominence
    * no feature search and no cross-qubit loop

Why this implementation is slightly sharper than the planning sketch
----------------------------------------------------------------------
A naive frozen-projector replacement V -> lambda V inside

    P V P_G V P

would force Sigma(lambda) ~ lambda^2 Sigma and make A_group/C_group
largely scale-trivial.

Instead, this code uses

    H(lambda) = H0 + lambda V

while P, Q, and the probe couplings P V Q are frozen at lambda=0.

The Q-space propagator therefore responds through

    Q H(lambda) Q = Q H0 Q + lambda Q V Q.

This gives a non-trivial susceptibility even when the first-order
centroid shifts of degenerate groups vanish.

The derivative is evaluated analytically with the Frechet derivative of
the matrix function g_eta.  If

    A(lambda) = E_P(lambda) I_Q - Q H(lambda) Q,

then

    Sigma(lambda) = B^dagger g_eta(A(lambda)) B,
    B = Q V P.

At lambda=0, A is diagonal in the unperturbed Hamiltonian eigenbasis.
The Frechet derivative is therefore

    [L_g(A, A')]_ij = g^[1](a_i, a_j) A'_ij,

where g^[1] is the first divided difference, with g'(a_i) on degenerate
diagonal blocks.

Primary candidates
------------------
    R_E = || dSigma_E / dlambda ||_F

    D_E = | d ||Sigma_E||_F^2 / dlambda |
        = | 2 Re Tr(Sigma_E^dagger dSigma_E/dlambda) |

The response is additionally decomposed into:

    D_Q_within   : contribution from matrix elements staying inside the
                   same frozen degeneracy-closed Q group.

    D_Q_crossmix : contribution from matrix elements coupling different
                   frozen Q groups.

This decomposition is exact at first order:

    D_signed = D_Q_within_signed + D_Q_crossmix_signed.

Static A_group and C_group are retained only as the frozen v32.14
baseline.  Their derivatives are not promoted to primary observables in
this implementation, because once Q-space mixing is included there is no
unique basis-independent attribution of the derivative to the original
individual T_G channels.

Dependency
----------
The frozen model kernel is imported from:

    v32_14_degeneracy_closed.py

which must be beside this script.

Outputs
-------
    v32_17_seed25042000_rows.csv
    v32_17_seed25042000_stats.csv
    v32_17_delta_scan.csv
    v32_17_seed25042000_summary.txt

Usage
-----
    python .\\v32_17_perturbative_response.py

or

    python .\\v32_17_perturbative_response.py --seed 25042000

Optional finite-difference validation scan:

    python .\\v32_17_perturbative_response.py ^
        --deltas 1e-4 3e-4 1e-3 3e-3

The delta scan validates the analytic derivative.  It is not used to
select or optimize the scientific result.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


VERSION = "v32.17"
DEFAULT_SEED = 25_042_000
DEFAULT_DELTAS = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3)
DEFAULT_VALIDATION_PAIRS = 3


# ---------------------------------------------------------------------------
# Frozen base model
# ---------------------------------------------------------------------------

def load_base():
    here = Path(__file__).resolve().parent
    path = here / "v32_14_degeneracy_closed.py"

    if not path.exists():
        raise FileNotFoundError(
            "v32_14_degeneracy_closed.py must be beside this file."
        )

    spec = importlib.util.spec_from_file_location("v3214_base", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load v32_14_degeneracy_closed.py")

    module = importlib.util.module_from_spec(spec)

    # Needed by dataclasses/type machinery during dynamic import.
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# IO and statistics helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def spearman(base, x: Iterable[float], y: Iterable[float]) -> float:
    return float(
        base.spearman_np(
            np.asarray(list(x), dtype=float),
            np.asarray(list(y), dtype=float),
        )
    )


def rel_fro_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference, ord="fro"))
    err = float(np.linalg.norm(estimate - reference, ord="fro"))
    if denom <= 1.0e-30:
        return err
    return err / denom


# ---------------------------------------------------------------------------
# Regularized kernel and Frechet derivative
# ---------------------------------------------------------------------------

def g_eta(x: np.ndarray, eta: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / (x * x + eta * eta)


def g_eta_prime(x: np.ndarray, eta: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    den = x * x + eta * eta
    return (eta * eta - x * x) / (den * den)


def divided_difference_matrix(
    a: np.ndarray,
    eta: float,
    equality_tol: float = 1.0e-12,
) -> np.ndarray:
    """
    First divided-difference matrix for g_eta:

        F_ij = (g(a_i)-g(a_j))/(a_i-a_j),  a_i != a_j
             = g'(a_i),                    a_i == a_j

    Using g'(a) on exactly/near-degenerate blocks is what makes the
    Frechet derivative well-defined without choosing basis vectors
    inside the degenerate subspace.
    """
    a = np.asarray(a, dtype=float)
    ai = a[:, None]
    aj = a[None, :]
    diff = ai - aj

    ga = g_eta(a, eta)
    numer = ga[:, None] - ga[None, :]

    out = np.empty_like(diff, dtype=float)
    mask = np.abs(diff) > equality_tol
    out[mask] = numer[mask] / diff[mask]

    amid = 0.5 * (ai + aj)
    out[~mask] = g_eta_prime(amid[~mask], eta)
    return out


# ---------------------------------------------------------------------------
# Frozen reference model
# ---------------------------------------------------------------------------

def build_reference(base, seed: int):
    h_terms = base.random_hamiltonian_terms(seed)
    h = base.dense_pauli_sum(base.N_QUBITS, h_terms)
    evals, evecs = np.linalg.eigh(h)
    groups = base.degenerate_groups(evals)

    null_vectors = base.haar_unitary(
        base.DIM,
        base.stable_hash_int(
            f"{base.FROZEN_MODEL_VERSION}|NULL|{seed}|terms{base.N_TERMS}"
        ),
    )

    perturbations = {
        family: base.family_terms(
            base.stable_hash_int(
                f"{base.FROZEN_MODEL_VERSION}|{family}|PERT|{seed}"
            ),
            family,
        )
        for family in base.FAMILIES
    }

    prominence = base.discovery_prominence_for_seed(
        seed=seed,
        eigenvalues=evals,
        real_vectors=evecs,
        null_vectors=null_vectors,
        groups=groups,
        perturbations=perturbations,
    )

    energies = np.asarray(
        [base.eigenspace_energy(evals, g) for g in groups],
        dtype=float,
    )
    pairs = base.opposite_energy_pairs(energies)

    state_to_group = np.empty(evals.size, dtype=int)
    for gi, group in enumerate(groups):
        state_to_group[group] = gi

    return {
        "hamiltonian": h,
        "evals": evals,
        "evecs": evecs,
        "groups": groups,
        "energies": energies,
        "perturbations": perturbations,
        "prominence": prominence,
        "pairs": pairs,
        "state_to_group": state_to_group,
    }


def static_group_descriptors(
    base,
    ref: dict,
    gi: int,
    perturbation: List[Tuple[str, float]],
    eta: float,
) -> dict:
    terms = base.degeneracy_closed_terms(
        eigenvalues=ref["evals"],
        eigenvectors=ref["evecs"],
        groups=ref["groups"],
        target_group_number=gi,
        perturbation=perturbation,
        eta=eta,
    )
    return base.interference_descriptors_group_closed(terms)


# ---------------------------------------------------------------------------
# Analytic frozen-P/Q susceptibility
# ---------------------------------------------------------------------------

def prepare_target(
    base,
    ref: dict,
    gi: int,
    perturbation: List[Tuple[str, float]],
) -> dict:
    """
    Prepare one frozen target P.

    The unperturbed Hamiltonian eigenbasis is used only as a convenient
    representation.  All final operators are subspace/projector objects.
    """
    evals = ref["evals"]
    evecs = ref["evecs"]
    groups = ref["groups"]
    state_to_group = ref["state_to_group"]

    p_idx = groups[gi]
    p_vec = evecs[:, p_idx]
    e_p = float(ref["energies"][gi])

    all_idx = np.arange(evals.size, dtype=int)
    q_idx = np.setdiff1d(all_idx, p_idx, assume_unique=True)
    q_vec = evecs[:, q_idx]

    # B = Q V P
    acted_p = base.apply_pauli_sum(p_vec, perturbation)
    b_qp = q_vec.conj().T @ acted_p

    # Q V Q
    acted_q = base.apply_pauli_sum(q_vec, perturbation)
    v_qq = q_vec.conj().T @ acted_q
    v_qq = 0.5 * (v_qq + v_qq.conj().T)

    # First-order centroid shift of the frozen target P.
    v_pp = p_vec.conj().T @ acted_p
    target_slope = float(np.trace(v_pp).real / int(p_idx.size))

    # At lambda=0, Q H0 Q is diagonal in this basis.
    q_energies = evals[q_idx].astype(float)
    a = e_p - q_energies

    # Exact static Sigma in the frozen P/Q formulation.
    weights = g_eta(a, float(base.ENERGY_REG))
    sigma = (b_qp.conj().T * weights[None, :]) @ b_qp
    sigma = 0.5 * (sigma + sigma.conj().T)

    # A'(0) = E'_P I - Q V Q
    a_prime = (
        target_slope * np.eye(q_idx.size, dtype=np.complex128)
        - v_qq
    )

    # Frechet derivative of g_eta(A).
    dd = divided_difference_matrix(
        a,
        float(base.ENERGY_REG),
        equality_tol=max(1.0e-12, float(base.DEGENERACY_TOL)),
    )
    dg = dd * a_prime

    # dSigma = B^dagger L_g(A,A') B
    d_sigma = b_qp.conj().T @ dg @ b_qp
    d_sigma = 0.5 * (d_sigma + d_sigma.conj().T)

    # Mechanism split: matrix elements within the same frozen,
    # degeneracy-closed Q group vs elements connecting different groups.
    q_group_labels = state_to_group[q_idx]
    same_group = q_group_labels[:, None] == q_group_labels[None, :]

    dg_within = dg * same_group
    dg_cross = dg * (~same_group)

    d_sigma_within = b_qp.conj().T @ dg_within @ b_qp
    d_sigma_cross = b_qp.conj().T @ dg_cross @ b_qp

    d_sigma_within = 0.5 * (
        d_sigma_within + d_sigma_within.conj().T
    )
    d_sigma_cross = 0.5 * (
        d_sigma_cross + d_sigma_cross.conj().T
    )

    r_e = float(np.linalg.norm(d_sigma, ord="fro"))

    d_signed = float(
        2.0 * np.real(np.trace(sigma.conj().T @ d_sigma))
    )
    d_within_signed = float(
        2.0 * np.real(np.trace(sigma.conj().T @ d_sigma_within))
    )
    d_cross_signed = float(
        2.0 * np.real(np.trace(sigma.conj().T @ d_sigma_cross))
    )

    split_residual = float(
        d_signed - (d_within_signed + d_cross_signed)
    )

    return {
        "group": int(gi),
        "dim_p": int(p_idx.size),
        "energy": e_p,
        "target_slope": target_slope,
        "q_idx": q_idx,
        "q_vec": q_vec,
        "q_energies": q_energies,
        "b_qp": b_qp,
        "v_qq": v_qq,
        "sigma": sigma,
        "d_sigma": d_sigma,
        "R_E": r_e,
        "D_E": abs(d_signed),
        "D_E_signed": d_signed,
        "D_Q_within": abs(d_within_signed),
        "D_Q_within_signed": d_within_signed,
        "D_Q_crossmix": abs(d_cross_signed),
        "D_Q_crossmix_signed": d_cross_signed,
        "split_residual": split_residual,
    }


# ---------------------------------------------------------------------------
# Exact finite-difference validation of the analytic derivative
# ---------------------------------------------------------------------------

def sigma_frozen_exact(
    prepared: dict,
    lam: float,
    eta: float,
) -> np.ndarray:
    """
    Exact frozen-P/Q Sigma(lambda) for validation:

        H_Q(lambda) = diag(E_q) + lambda V_QQ
        z(lambda)   = E_P + lambda E'_P

        Sigma(lambda) = B^dagger g_eta(z I - H_Q(lambda)) B.

    P and Q remain frozen; only the Q-space propagator changes.
    """
    hq = np.diag(prepared["q_energies"]).astype(np.complex128)
    hq = hq + float(lam) * prepared["v_qq"]
    hq = 0.5 * (hq + hq.conj().T)

    qevals, qrot = np.linalg.eigh(hq)

    z = (
        float(prepared["energy"])
        + float(lam) * float(prepared["target_slope"])
    )
    weights = g_eta(z - qevals, eta)

    b_rot = qrot.conj().T @ prepared["b_qp"]
    sigma = (b_rot.conj().T * weights[None, :]) @ b_rot
    return 0.5 * (sigma + sigma.conj().T)


def validate_target(
    prepared: dict,
    deltas: List[float],
    eta: float,
) -> List[dict]:
    rows = []
    analytic = prepared["d_sigma"]
    sigma0 = prepared["sigma"]
    analytic_dnorm2 = float(prepared["D_E_signed"])

    for delta in deltas:
        sp = sigma_frozen_exact(prepared, +delta, eta)
        sm = sigma_frozen_exact(prepared, -delta, eta)
        fd = (sp - sm) / (2.0 * delta)

        fd_dnorm2 = float(
            (
                np.linalg.norm(sp, ord="fro") ** 2
                - np.linalg.norm(sm, ord="fro") ** 2
            )
            / (2.0 * delta)
        )

        rows.append(
            {
                "delta": float(delta),
                "rel_fro_error_dSigma": rel_fro_error(analytic, fd),
                "analytic_R_E": float(np.linalg.norm(analytic, ord="fro")),
                "fd_R_E": float(np.linalg.norm(fd, ord="fro")),
                "analytic_D_signed": analytic_dnorm2,
                "fd_D_signed": fd_dnorm2,
                "abs_error_D_signed": abs(
                    analytic_dnorm2 - fd_dnorm2
                ),
                "sigma0_fro": float(np.linalg.norm(sigma0, ord="fro")),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Pair-level analysis
# ---------------------------------------------------------------------------

def pair_order(ref: dict, a: int, b: int) -> Tuple[int, int]:
    energies = ref["energies"]
    return (a, b) if energies[a] <= energies[b] else (b, a)


def valid_pair_records(ref: dict) -> List[dict]:
    out = []
    for a, b in ref["pairs"]:
        gm, gp = pair_order(ref, a, b)
        p = max(
            ref["prominence"].get(gm, float("-inf")),
            ref["prominence"].get(gp, float("-inf")),
        )
        if np.isfinite(p):
            out.append(
                {
                    "g_minus": int(gm),
                    "g_plus": int(gp),
                    "prominence": float(p),
                    "E_abs": float(abs(ref["energies"][gp])),
                }
            )
    return out


def representative_validation_pairs(
    pairs: List[dict],
    count: int,
) -> List[dict]:
    """
    Deterministic convergence sample spanning the prominence range.

    For count=3: bottom, median, top pair.
    This is for numerical validation only, never model selection.
    """
    if not pairs:
        return []

    ordered = sorted(pairs, key=lambda r: r["prominence"])
    if count <= 1:
        return [ordered[len(ordered) // 2]]

    positions = np.linspace(0, len(ordered) - 1, count)
    chosen = []
    used = set()

    for pos in positions:
        idx = int(round(float(pos)))
        idx = max(0, min(len(ordered) - 1, idx))
        key = (ordered[idx]["g_minus"], ordered[idx]["g_plus"])
        if key not in used:
            chosen.append(ordered[idx])
            used.add(key)

    return chosen


def run_pilot(
    base,
    seed: int,
    deltas: List[float],
    validation_pair_count: int,
) -> Tuple[List[dict], List[dict], List[dict], str]:
    ref = build_reference(base, seed)
    eta = float(base.ENERGY_REG)

    pairs = valid_pair_records(ref)
    rows: List[dict] = []
    stats: List[dict] = []
    delta_scan: List[dict] = []

    prepared: Dict[Tuple[str, int], dict] = {}
    static_desc: Dict[Tuple[str, int], dict] = {}

    needed_groups = sorted(
        set(r["g_minus"] for r in pairs)
        | set(r["g_plus"] for r in pairs)
    )

    for family in base.FAMILIES:
        perturbation = ref["perturbations"][family]

        for gi in needed_groups:
            prepared[(family, gi)] = prepare_target(
                base, ref, gi, perturbation
            )
            static_desc[(family, gi)] = static_group_descriptors(
                base, ref, gi, perturbation, eta
            )

        for pair in pairs:
            gm = pair["g_minus"]
            gp = pair["g_plus"]

            pm = prepared[(family, gm)]
            pp = prepared[(family, gp)]
            sm = static_desc[(family, gm)]
            sp = static_desc[(family, gp)]

            row = {
                "version": VERSION,
                "seed": int(seed),
                "family": family,
                "g_minus": int(gm),
                "g_plus": int(gp),
                "E_abs": float(pair["E_abs"]),
                "prominence": float(pair["prominence"]),
                "dim_minus": int(ref["groups"][gm].size),
                "dim_plus": int(ref["groups"][gp].size),

                # Frozen static baseline.
                "A_group": 0.5 * (
                    float(sm["A_group"]) + float(sp["A_group"])
                ),
                "C_group": 0.5 * (
                    float(sm["C_group"]) + float(sp["C_group"])
                ),
                "cross_sum": 0.5 * (
                    float(sm["cross_sum"]) + float(sp["cross_sum"])
                ),

                # Primary analytic susceptibility.
                "R_E": 0.5 * (
                    float(pm["R_E"]) + float(pp["R_E"])
                ),
                "D_E": 0.5 * (
                    float(pm["D_E"]) + float(pp["D_E"])
                ),

                # Mechanism split.
                "D_Q_within": 0.5 * (
                    float(pm["D_Q_within"])
                    + float(pp["D_Q_within"])
                ),
                "D_Q_crossmix": 0.5 * (
                    float(pm["D_Q_crossmix"])
                    + float(pp["D_Q_crossmix"])
                ),

                # Signed pair diagnostics.
                "D_E_signed_pairmean": 0.5 * (
                    float(pm["D_E_signed"])
                    + float(pp["D_E_signed"])
                ),
                "D_Q_within_signed_pairmean": 0.5 * (
                    float(pm["D_Q_within_signed"])
                    + float(pp["D_Q_within_signed"])
                ),
                "D_Q_crossmix_signed_pairmean": 0.5 * (
                    float(pm["D_Q_crossmix_signed"])
                    + float(pp["D_Q_crossmix_signed"])
                ),
                "split_residual_maxabs": max(
                    abs(float(pm["split_residual"])),
                    abs(float(pp["split_residual"])),
                ),
            }
            rows.append(row)

    metrics = [
        "A_group",
        "C_group",
        "cross_sum",
        "R_E",
        "D_E",
        "D_Q_within",
        "D_Q_crossmix",
    ]

    for family in base.FAMILIES:
        sub = [r for r in rows if r["family"] == family]
        y = [r["prominence"] for r in sub]

        for metric in metrics:
            rho = spearman(base, [r[metric] for r in sub], y)
            stats.append(
                {
                    "version": VERSION,
                    "seed": int(seed),
                    "family": family,
                    "metric": metric,
                    "n_pairs": int(len(sub)),
                    "spearman_rho_vs_prominence": float(rho),
                }
            )

    # Numerical convergence check on a small deterministic set of pairs.
    validation_pairs = representative_validation_pairs(
        pairs, max(1, int(validation_pair_count))
    )

    for family in base.FAMILIES:
        for rank, pair in enumerate(validation_pairs, start=1):
            for sign, gi in (
                ("minus", pair["g_minus"]),
                ("plus", pair["g_plus"]),
            ):
                check_rows = validate_target(
                    prepared[(family, gi)], deltas, eta
                )
                for cr in check_rows:
                    delta_scan.append(
                        {
                            "version": VERSION,
                            "seed": int(seed),
                            "family": family,
                            "validation_pair_rank": int(rank),
                            "pair_prominence": float(pair["prominence"]),
                            "sign": sign,
                            "group": int(gi),
                            **cr,
                        }
                    )

    group_sizes = [int(g.size) for g in ref["groups"]]
    max_split_resid = max(
        abs(float(r["split_residual_maxabs"])) for r in rows
    ) if rows else float("nan")

    lines = [
        f"Soft Spaces Phase 3.2 {VERSION}",
        "Perturbative Response Law — analytic frozen-P/Q pilot",
        "",
        f"seed={seed}",
        f"qubits={base.N_QUBITS}",
        f"dimension={base.DIM}",
        f"hamiltonian_terms={base.N_TERMS}",
        f"groups={len(ref['groups'])}",
        f"group_size_range={min(group_sizes)}:{max(group_sizes)}",
        f"exact_pmE_pairs={len(ref['pairs'])}",
        f"usable_pairs={len(pairs)}",
        f"eta={eta:.6g}",
        "",
        "MODEL",
        "-----",
        "H(lambda)=H0+lambda V",
        "P and Q frozen at lambda=0",
        "B=QVP frozen",
        "QH(lambda)Q = QH0Q + lambda QVQ",
        "dSigma/dlambda computed analytically by the Frechet derivative",
        "of g_eta(E_P I - QHQ).",
        "",
        "PRIMARY RESULTS",
        "---------------",
    ]

    for family in base.FAMILIES:
        lines.append("")
        lines.append(f"[{family}]")
        for metric in metrics:
            match = [
                s for s in stats
                if s["family"] == family and s["metric"] == metric
            ][0]
            lines.append(
                f"  {metric:15s} "
                f"rho={float(match['spearman_rho_vs_prominence']):+.4f}"
            )

    lines += [
        "",
        "EXACT FIRST-ORDER SPLIT CHECK",
        "-----------------------------",
        (
            "max |D_signed - (D_Q_within_signed + "
            f"D_Q_crossmix_signed)| = {max_split_resid:.6e}"
        ),
        "",
        "FINITE-DIFFERENCE VALIDATION",
        "----------------------------",
        (
            f"Representative +/-E pairs checked: "
            f"{len(validation_pairs)}"
        ),
        "deltas=" + ",".join(f"{d:.6g}" for d in deltas),
        "The delta scan validates the analytic derivative only.",
        "It is not used to choose a scientific result.",
        "",
        "INTERPRETATION",
        "--------------",
        "Compare R_E and D_E directly with the frozen static A_group/C_group",
        "baseline.  If D_Q_crossmix carries the signal, Q-space mixing between",
        "different degeneracy-closed virtual sectors is implicated.  If",
        "D_Q_within dominates, the response is primarily internal to the",
        "individual frozen Q sectors.",
        "",
        "No full spectral response and no cross-seed claim is made here.",
    ]

    return rows, stats, delta_scan, "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Soft Spaces Phase 3.2 v32.17 — analytic frozen-P/Q "
            "perturbative response pilot."
        )
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Hamiltonian seed (default {DEFAULT_SEED}).",
    )
    p.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=list(DEFAULT_DELTAS),
        help=(
            "Finite-difference validation steps. Default: "
            + " ".join(str(x) for x in DEFAULT_DELTAS)
        ),
    )
    p.add_argument(
        "--validation-pairs",
        type=int,
        default=DEFAULT_VALIDATION_PAIRS,
        help=(
            "Number of deterministic bottom/middle/top prominence pairs "
            f"used only for finite-difference validation "
            f"(default {DEFAULT_VALIDATION_PAIRS})."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (default current directory).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = load_base()

    if int(base.N_QUBITS) != 8:
        raise RuntimeError(
            f"v32.17 pilot is frozen for 8Q; base reports {base.N_QUBITS}Q."
        )
    if int(base.N_TERMS) != 11:
        raise RuntimeError(
            f"v32.17 pilot expects 11 Hamiltonian terms; "
            f"base reports {base.N_TERMS}."
        )

    deltas = sorted(set(float(d) for d in args.deltas))
    if not deltas or any(d <= 0.0 or not np.isfinite(d) for d in deltas):
        raise ValueError("All delta values must be finite and > 0.")

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"{VERSION}: analytic 8Q frozen-P/Q pilot, "
        f"seed {args.seed} ..."
    )

    rows, stats, delta_scan, summary = run_pilot(
        base=base,
        seed=int(args.seed),
        deltas=deltas,
        validation_pair_count=int(args.validation_pairs),
    )

    rows_path = outdir / f"v32_17_seed{args.seed}_rows.csv"
    stats_path = outdir / f"v32_17_seed{args.seed}_stats.csv"
    scan_path = outdir / "v32_17_delta_scan.csv"
    summary_path = outdir / f"v32_17_seed{args.seed}_summary.txt"

    write_csv(rows_path, rows)
    write_csv(stats_path, stats)
    write_csv(scan_path, delta_scan)
    summary_path.write_text(summary, encoding="utf-8")

    print(summary)
    print("WROTE")
    print(f"  {rows_path}")
    print(f"  {stats_path}")
    print(f"  {scan_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
