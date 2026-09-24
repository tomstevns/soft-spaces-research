#!/usr/bin/env python3
"""
Soft Spaces Phase 6 — v60_1 Quantum Sensing / Metrology Gauntlet

One-file, self-contained Phase-6 Point-1 test.

PURPOSE
-------
Test whether the original Phase-2/3 Soft-Spaces signal has predictive value
for quantum-sensing-like local spectral sensitivity, and whether it adds
information beyond an EN2 determinant baseline.

The script runs BOTH:
    H4 / STO-3G
    LiH / STO-3G

It reconstructs the molecular Hamiltonians, rebuilds the v50_7c-style
Soft-Spaces scores from scratch, generates NEW held-out perturbations, and
tests three sensing-relevant response quantities.

NO intermediate file from Phase 5 is required.

---------------------------------------------------------------------------
SENSING TESTS
---------------------------------------------------------------------------

S1. SUBSPACE LEAKAGE SUSCEPTIBILITY
    For adjacent eigenstate pair P={k,k+1}, measure perturbative coupling from
    P to the complement Q:

        chi_leak(P,V)
          = sum_{a in P, b in Q}
              |V_ba|^2 / ((E_a-E_b)^2 + reg^2)

    This is closely related to fidelity/projector susceptibility: a large
    value means the local 2D eigensubspace is highly sensitive to a weak
    perturbation.

S2. FIRST-ORDER GAP SENSITIVITY
    For the same pair:

        g1(P,V)
          = | <k+1|V|k+1> - <k|V|k> |

    This measures first-order sensitivity of the local level spacing.

S3. SECOND-ORDER GAP CURVATURE
    Using regularized second-order energy corrections:

        dE_n^(2)
          = sum_{m != n}
              |V_mn|^2 * (E_n-E_m)
              / ((E_n-E_m)^2 + reg^2)

        g2(P,V)
          = | dE_(k+1)^(2) - dE_k^(2) |

    This measures curvature/nonlinear spectral sensitivity.

For each test the response is averaged over held-out random perturbations
from three families:
    dephasing
    transverse
    mixed

The final response for each sensing test is the median across those three
family averages. This rule is frozen in the code.

---------------------------------------------------------------------------
SOFT-SPACES PREDICTOR
---------------------------------------------------------------------------

Rebuilds the v50_7c mechanism:

    DeltaC_f = C_REAL,f - C_NULL,f

    robust_seed_score
        = min(DeltaC_dephasing, DeltaC_transverse)

    prominence_seed_score
        = robust(k,seed) - median(local robust controls at same seed)

The continuous Soft-Spaces predictor used in Phase 6 is:

    SS_score
        = min(
            prominence_positive_rate,
            robust_positive_rate
          )

DUAL PASS remains:
    prominence_positive_rate >= 0.75
    robust_positive_rate     >= 0.75

No sensing outcomes are used to tune this score.

---------------------------------------------------------------------------
EN2 COMPARATOR
---------------------------------------------------------------------------

Determinant EN2 score:

    EN2_i
      = |H_ir|^2 / (|H_ii-H_rr| + eps)

where r is the determinant with minimum diagonal H_ii.

EN2 is mapped to an eigenpair using pair determinant support:

    EN2_pair(P)
      = sum_i p_i(P) * EN2_i

    p_i(P)
      = 0.5*(|V_i,k|^2 + |V_i,k+1|^2)

This is the same explicit bridge used in Phase 5.

---------------------------------------------------------------------------
PREDECLARED SUCCESS RULE
---------------------------------------------------------------------------

For EACH sensing test S1-S3, calculate AUC for identifying the top quartile
of truly sensitive eigenpairs.

A sensing test is a CLEAR PASS only if ALL hold:

    pooled Soft-Spaces AUC >= 0.65
    pooled (SS AUC - EN2 AUC) >= 0.10
    H4: SS AUC > EN2 AUC
    LiH: SS AUC > EN2 AUC
    median geometry-level (SS AUC - EN2 AUC) > 0
    at least 10 of 16 molecule/geometries have positive delta AUC

OVERALL POINT-1 SUCCESS:
    at least 2 of the 3 sensing tests are CLEAR PASS.

If not:
    Point 1 is NOT demonstrated under these frozen rules.

This is intentionally conservative and avoids post-hoc rescue.

---------------------------------------------------------------------------
COMPUTATIONAL NOTE
---------------------------------------------------------------------------

Held-out sensing responses use perturbation theory directly; they do NOT
re-diagonalize the Hamiltonian for every held-out seed. This keeps the run
substantially faster than brute-force finite-difference diagonalization.

Usage
-----
Place this file in:
    Phase6/V_1_0/code/sensing/

Then run from that directory:

    python -X utf8 -u ./v60_1_sensing_gauntlet.py

Outputs:
    Phase6/V_1_0/results/sensing/
        phase6_v60_1_sensing_summary.json
        phase6_v60_1_sensing_summary.sha256
        phase6_v60_1_candidate_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_VERSION = "v60_1"

# Independent molecule / geometry panels.
MOLECULES = {
    "H4": {
        "geometries": (0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00),
    },
    "LiH": {
        "geometries": (1.00, 1.20, 1.40, 1.60, 2.00, 2.50, 3.00, 4.00),
    },
}

# Frozen Soft-Spaces confirmation seeds.
CONFIRM_SEEDS = tuple(range(25043000, 25043012))

# New Phase-6 held-out sensing seeds.
HELDOUT_SEEDS = tuple(range(45043000, 45043032))

LOCAL_RADIUS = 10
LOCAL_STEP = 2
EPS_NEIGHBOR = 0.05
ENERGY_REG = 1e-3

PROM_THRESHOLD = 0.75
ROBUST_THRESHOLD = 0.75

EN2_EPS = 1e-12
SENSING_REG = 1e-3

CLEAR_SS_AUC = 0.65
CLEAR_DELTA_AUC = 0.10
CLEAR_POSITIVE_GEOMETRIES = 10
OVERALL_REQUIRED_CLEAR_TESTS = 2


@dataclass(frozen=True)
class CandidateMetric:
    molecule: str
    geometry: float
    eigenpair_index: int
    gap_hartree: float
    prominence_positive_rate: float
    robust_positive_rate: float
    softspaces_score: float
    dual_pass: bool
    en2_pair_score: float
    leakage_response: float
    first_order_gap_response: float
    second_order_gap_response: float


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def popcount(x: int) -> int:
    return bin(int(x)).count("1")


def fixed_sector_basis(
    n_spatial: int,
    n_alpha: int,
    n_beta: int,
) -> np.ndarray:
    mask = (1 << n_spatial) - 1
    out = []

    for idx in range(1 << (2 * n_spatial)):
        na = popcount(idx & mask)
        nb = popcount((idx >> n_spatial) & mask)

        if na == n_alpha and nb == n_beta:
            out.append(idx)

    expected = math.comb(n_spatial, n_alpha) * math.comb(n_spatial, n_beta)

    if len(out) != expected:
        raise RuntimeError(
            f"Sector dimension mismatch: expected {expected}, got {len(out)}"
        )

    return np.asarray(out, dtype=np.int64)


def build_problem(
    molecule: str,
    geometry: float,
) -> dict[str, Any]:
    try:
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        from qiskit_nature.units import DistanceUnit
    except Exception as exc:
        raise RuntimeError(
            "Qiskit Nature / PySCF imports failed. "
            "Run this script in the Phase-5/6 WSL environment."
        ) from exc

    if molecule == "H4":
        zs = (
            -1.5 * geometry,
            -0.5 * geometry,
            +0.5 * geometry,
            +1.5 * geometry,
        )

        atom = "; ".join(
            f"H 0.0 0.0 {z:.12f}"
            for z in zs
        )

    elif molecule == "LiH":
        atom = (
            "Li 0.0 0.0 0.0; "
            f"H 0.0 0.0 {geometry:.12f}"
        )

    else:
        raise ValueError(
            f"Unsupported molecule: {molecule}"
        )

    problem = PySCFDriver(
        atom=atom,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    ).run()

    n_spatial = int(problem.num_spatial_orbitals)
    n_alpha = int(problem.num_alpha)
    n_beta = int(problem.num_beta)

    if (n_alpha, n_beta) != (2, 2):
        raise RuntimeError(
            f"Unexpected electron sector for {molecule}: "
            f"Nalpha={n_alpha}, Nbeta={n_beta}"
        )

    basis = fixed_sector_basis(
        n_spatial,
        n_alpha,
        n_beta,
    )

    qubit_op = JordanWignerMapper().map(
        problem.hamiltonian.second_q_op()
    )

    full = qubit_op.to_matrix(
        sparse=True
    )

    h = np.asarray(
        full[basis, :][:, basis].toarray(),
        dtype=np.complex128,
    )

    herm_error = float(
        np.max(
            np.abs(
                h - h.conjugate().T
            )
        )
    )

    if herm_error > 1e-10:
        raise RuntimeError(
            f"Hamiltonian non-Hermitian: {herm_error:.3e}"
        )

    evals, evecs = np.linalg.eigh(
        h
    )

    return {
        "molecule": molecule,
        "geometry": float(geometry),
        "basis": basis,
        "n_spatial": n_spatial,
        "n_qubits": 2 * n_spatial,
        "h": h,
        "evals": np.asarray(
            evals,
            dtype=float,
        ),
        "evecs": np.asarray(
            evecs,
            dtype=np.complex128,
        ),
    }


def haar_unitary(
    d: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(
        seed
    )

    z = (
        rng.normal(
            size=(d, d)
        )
        + 1j
        * rng.normal(
            size=(d, d)
        )
    ) / math.sqrt(2.0)

    q, r = np.linalg.qr(
        z
    )

    diag = np.diag(
        r
    )

    phases = np.ones_like(
        diag,
        dtype=np.complex128,
    )

    nz = np.abs(
        diag
    ) > 0

    phases[nz] = (
        diag[nz]
        / np.abs(
            diag[nz]
        )
    )

    return np.asarray(
        q * phases.conjugate(),
        dtype=np.complex128,
    )


def projected_z(
    basis: np.ndarray,
    q: int,
) -> np.ndarray:
    diag = np.asarray(
        [
            -1.0
            if (
                (int(x) >> q)
                & 1
            )
            else 1.0
            for x in basis
        ],
        dtype=float,
    )

    return np.diag(
        diag
    ).astype(
        np.complex128
    )


def projected_zz(
    basis: np.ndarray,
    q1: int,
    q2: int,
) -> np.ndarray:
    vals = []

    for x in basis:
        z1 = (
            -1.0
            if (
                (int(x) >> q1)
                & 1
            )
            else 1.0
        )

        z2 = (
            -1.0
            if (
                (int(x) >> q2)
                & 1
            )
            else 1.0
        )

        vals.append(
            z1 * z2
        )

    return np.diag(
        np.asarray(
            vals,
            dtype=float,
        )
    ).astype(
        np.complex128
    )


def projected_flip(
    basis: np.ndarray,
    flip_mask: int,
) -> np.ndarray:
    lookup = {
        int(full): local
        for local, full
        in enumerate(basis)
    }

    d = len(
        basis
    )

    out = np.zeros(
        (d, d),
        dtype=np.complex128,
    )

    for col, full in enumerate(
        basis
    ):
        target = (
            int(full)
            ^ int(flip_mask)
        )

        row = lookup.get(
            target
        )

        if row is not None:
            out[row, col] = 1.0

    return out


def perturbation_pools(
    basis: np.ndarray,
    n_qubits: int,
) -> dict[str, list[np.ndarray]]:
    dephasing = []
    transverse = []

    for q in range(
        n_qubits
    ):
        dephasing.append(
            projected_z(
                basis,
                q,
            )
        )

        transverse.append(
            projected_flip(
                basis,
                1 << q,
            )
        )

    for q in range(
        n_qubits - 1
    ):
        dephasing.append(
            projected_zz(
                basis,
                q,
                q + 1,
            )
        )

        transverse.append(
            projected_flip(
                basis,
                (1 << q)
                | (1 << (q + 1)),
            )
        )

    return {
        "dephasing": dephasing,
        "transverse": transverse,
    }


def nonzero_pool(
    pool: list[np.ndarray],
) -> list[np.ndarray]:
    return [
        a
        for a in pool
        if float(
            np.linalg.norm(
                a,
                ord="fro",
            )
        ) > 0.0
    ]


def random_combo(
    pool: list[np.ndarray],
    seed: int,
) -> np.ndarray:
    if not pool:
        raise RuntimeError(
            "Empty perturbation pool."
        )

    rng = np.random.default_rng(
        seed
    )

    coeffs = rng.normal(
        size=len(pool)
    )

    out = np.zeros_like(
        pool[0],
        dtype=np.complex128,
    )

    for coeff, term in zip(
        coeffs,
        pool,
    ):
        out += (
            float(coeff)
            * term
        )

    norm = float(
        np.linalg.norm(
            out,
            ord="fro",
        )
    )

    if norm > 0.0:
        out /= norm

    return 0.5 * (
        out
        + out.conjugate().T
    )


def c_matrix(
    evals: np.ndarray,
    bmat: np.ndarray,
    i: int,
) -> float | None:
    d = len(
        evals
    )

    j = i + 1

    if j >= d:
        return None

    gap = abs(
        float(
            evals[j]
            - evals[i]
        )
    )

    if gap >= EPS_NEIGHBOR:
        return None

    qmask = np.ones(
        d,
        dtype=bool,
    )

    qmask[
        [i, j]
    ] = False

    wq = np.asarray(
        bmat[
            qmask,
            :
        ][
            :,
            [i, j]
        ],
        dtype=np.complex128,
    )

    eq = np.asarray(
        evals[
            qmask
        ],
        dtype=float,
    )

    eref = 0.5 * (
        float(
            evals[i]
        )
        + float(
            evals[j]
        )
    )

    de = (
        eref
        - eq
    )

    reg = max(
        float(
            ENERGY_REG
        ),
        1e-15,
    )

    inv = (
        de
        / (
            de * de
            + reg * reg
        )
    )

    total = np.zeros(
        (2, 2),
        dtype=np.complex128,
    )

    norm_sum = 0.0

    for r in range(
        wq.shape[0]
    ):
        v = np.asarray(
            wq[
                r,
                :
            ],
            dtype=np.complex128,
        ).reshape(
            2,
            1,
        )

        a = (
            float(
                inv[r]
            )
            * (
                v.conjugate()
                @ v.T
            )
        )

        a = 0.5 * (
            a
            + a.conjugate().T
        )

        total += a

        norm_sum += float(
            np.linalg.norm(
                a,
                ord="fro",
            )
        )

    if (
        norm_sum <= 0.0
        or not np.isfinite(
            norm_sum
        )
    ):
        return None

    return float(
        np.linalg.norm(
            total,
            ord="fro",
        )
        / norm_sum
    )


def build_softspaces_candidates(
    system: dict[str, Any],
) -> list[dict[str, Any]]:
    evals = system[
        "evals"
    ]

    v_real = system[
        "evecs"
    ]

    d = len(
        evals
    )

    pools = perturbation_pools(
        system["basis"],
        system["n_qubits"],
    )

    raw: dict[
        tuple[int, str, int],
        float,
    ] = {}

    for seed in CONFIRM_SEEDS:
        v_null = haar_unitary(
            d,
            seed
            + 99_000_000,
        )

        for family_index, family in enumerate(
            (
                "dephasing",
                "transverse",
            )
        ):
            dh = random_combo(
                pools[family],
                seed
                + 1_000_000
                * (
                    family_index
                    + 1
                ),
            )

            b_real = (
                v_real.conjugate().T
                @ dh
                @ v_real
            )

            b_null = (
                v_null.conjugate().T
                @ dh
                @ v_null
            )

            for k in range(
                d - 1
            ):
                cr = c_matrix(
                    evals,
                    b_real,
                    k,
                )

                cn = c_matrix(
                    evals,
                    b_null,
                    k,
                )

                if (
                    cr is None
                    or cn is None
                ):
                    continue

                raw[
                    (
                        seed,
                        family,
                        k,
                    )
                ] = float(
                    cr - cn
                )

    robust: dict[
        tuple[int, int],
        float,
    ] = {}

    for seed in CONFIRM_SEEDS:
        for k in range(
            d - 1
        ):
            a = raw.get(
                (
                    seed,
                    "dephasing",
                    k,
                )
            )

            t = raw.get(
                (
                    seed,
                    "transverse",
                    k,
                )
            )

            if (
                a is not None
                and t is not None
                and np.isfinite(a)
                and np.isfinite(t)
            ):
                robust[
                    (
                        seed,
                        k,
                    )
                ] = min(
                    float(a),
                    float(t),
                )

    candidates = []

    for k in range(
        d - 1
    ):
        prominence_values = []
        robust_values = []

        for seed in CONFIRM_SEEDS:
            current = robust.get(
                (
                    seed,
                    k,
                )
            )

            if current is None:
                continue

            valid_k = [
                kk
                for kk
                in range(
                    d - 1
                )
                if (
                    seed,
                    kk,
                )
                in robust
            ]

            controls = [
                kk
                for kk
                in valid_k
                if (
                    kk != k
                    and abs(
                        kk - k
                    )
                    <= LOCAL_RADIUS
                    and abs(
                        kk - k
                    )
                    % LOCAL_STEP
                    == 0
                )
            ]

            if not controls:
                continue

            background = float(
                np.median(
                    [
                        robust[
                            (
                                seed,
                                kk,
                            )
                        ]
                        for kk
                        in controls
                    ]
                )
            )

            prominence_values.append(
                float(
                    current
                    - background
                )
            )

            robust_values.append(
                float(
                    current
                )
            )

        if not prominence_values:
            continue

        prom_rate = float(
            np.mean(
                np.asarray(
                    prominence_values
                )
                > 0.0
            )
        )

        robust_rate = float(
            np.mean(
                np.asarray(
                    robust_values
                )
                > 0.0
            )
        )

        score = min(
            prom_rate,
            robust_rate,
        )

        candidates.append(
            {
                "k": int(k),
                "gap": float(
                    evals[
                        k + 1
                    ]
                    - evals[k]
                ),
                "prominence_positive_rate": prom_rate,
                "robust_positive_rate": robust_rate,
                "softspaces_score": score,
                "dual_pass": bool(
                    prom_rate
                    >= PROM_THRESHOLD
                    and robust_rate
                    >= ROBUST_THRESHOLD
                ),
            }
        )

    return candidates


def en2_determinant_scores(
    h: np.ndarray,
) -> np.ndarray:
    diag = np.real(
        np.diag(
            h
        )
    ).astype(
        float
    )

    reference = int(
        np.argmin(
            diag
        )
    )

    hrr = float(
        diag[
            reference
        ]
    )

    scores = np.zeros(
        len(
            diag
        ),
        dtype=float,
    )

    for i in range(
        len(
            diag
        )
    ):
        if i == reference:
            continue

        scores[i] = (
            float(
                abs(
                    h[
                        i,
                        reference,
                    ]
                )
                ** 2
            )
            / (
                abs(
                    float(
                        diag[i]
                        - hrr
                    )
                )
                + EN2_EPS
            )
        )

    return scores


def pair_probability(
    evecs: np.ndarray,
    k: int,
) -> np.ndarray:
    p = 0.5 * (
        np.abs(
            evecs[
                :,
                k,
            ]
        )
        ** 2
        + np.abs(
            evecs[
                :,
                k + 1,
            ]
        )
        ** 2
    )

    total = float(
        np.sum(
            p
        )
    )

    if total <= 0.0:
        raise RuntimeError(
            "Invalid pair probability."
        )

    return p / total


def en2_pair_score(
    det_scores: np.ndarray,
    evecs: np.ndarray,
    k: int,
) -> float:
    return float(
        np.sum(
            pair_probability(
                evecs,
                k,
            )
            * det_scores
        )
    )


def sensing_single_response(
    evals: np.ndarray,
    bmat: np.ndarray,
    k: int,
) -> tuple[float, float, float]:
    """
    Returns:
        leakage susceptibility,
        first-order gap sensitivity,
        second-order gap curvature.
    """
    d = len(
        evals
    )

    i = k
    j = k + 1

    if j >= d:
        raise ValueError(
            "Invalid eigenpair index."
        )

    leakage = 0.0

    for a in (
        i,
        j,
    ):
        for b in range(
            d
        ):
            if b in (
                i,
                j,
            ):
                continue

            de = float(
                evals[a]
                - evals[b]
            )

            denom = (
                de * de
                + SENSING_REG
                * SENSING_REG
            )

            leakage += (
                float(
                    abs(
                        bmat[
                            b,
                            a,
                        ]
                    )
                    ** 2
                )
                / denom
            )

    first_order_gap = abs(
        float(
            np.real(
                bmat[
                    j,
                    j,
                ]
                - bmat[
                    i,
                    i,
                ]
            )
        )
    )

    second = []

    for a in (
        i,
        j,
    ):
        corr = 0.0

        for b in range(
            d
        ):
            if b == a:
                continue

            de = float(
                evals[a]
                - evals[b]
            )

            weight = (
                de
                / (
                    de * de
                    + SENSING_REG
                    * SENSING_REG
                )
            )

            corr += (
                float(
                    abs(
                        bmat[
                            b,
                            a,
                        ]
                    )
                    ** 2
                )
                * weight
            )

        second.append(
            corr
        )

    second_order_gap = abs(
        float(
            second[1]
            - second[0]
        )
    )

    return (
        float(
            leakage
        ),
        float(
            first_order_gap
        ),
        float(
            second_order_gap
        ),
    )


def heldout_sensing_responses(
    system: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    evals = system[
        "evals"
    ]

    evecs = system[
        "evecs"
    ]

    pools_raw = perturbation_pools(
        system["basis"],
        system["n_qubits"],
    )

    pools = {
        family: nonzero_pool(
            terms
        )
        for family, terms
        in pools_raw.items()
    }

    n = len(
        candidates
    )

    family_results = {
        "dephasing": {
            "leakage": np.zeros(
                n,
                dtype=float,
            ),
            "first_order_gap": np.zeros(
                n,
                dtype=float,
            ),
            "second_order_gap": np.zeros(
                n,
                dtype=float,
            ),
        },
        "transverse": {
            "leakage": np.zeros(
                n,
                dtype=float,
            ),
            "first_order_gap": np.zeros(
                n,
                dtype=float,
            ),
            "second_order_gap": np.zeros(
                n,
                dtype=float,
            ),
        },
        "mixed": {
            "leakage": np.zeros(
                n,
                dtype=float,
            ),
            "first_order_gap": np.zeros(
                n,
                dtype=float,
            ),
            "second_order_gap": np.zeros(
                n,
                dtype=float,
            ),
        },
    }

    for seed in HELDOUT_SEEDS:
        deph = random_combo(
            pools["dephasing"],
            seed + 101,
        )

        trans = random_combo(
            pools["transverse"],
            seed + 202,
        )

        mixed = (
            deph
            + trans
        )

        mixed_norm = float(
            np.linalg.norm(
                mixed,
                ord="fro",
            )
        )

        if mixed_norm > 0.0:
            mixed /= mixed_norm

        perturbations = {
            "dephasing": deph,
            "transverse": trans,
            "mixed": mixed,
        }

        for family, dh in perturbations.items():
            bmat = (
                evecs.conjugate().T
                @ dh
                @ evecs
            )

            for ci, c in enumerate(
                candidates
            ):
                leak, first, second = sensing_single_response(
                    evals,
                    bmat,
                    c["k"],
                )

                family_results[
                    family
                ][
                    "leakage"
                ][
                    ci
                ] += leak

                family_results[
                    family
                ][
                    "first_order_gap"
                ][
                    ci
                ] += first

                family_results[
                    family
                ][
                    "second_order_gap"
                ][
                    ci
                ] += second

    divisor = float(
        len(
            HELDOUT_SEEDS
        )
    )

    for family in family_results:
        for metric in family_results[
            family
        ]:
            family_results[
                family
            ][
                metric
            ] /= divisor

    # Frozen robust sensing response:
    # median across dephasing / transverse / mixed family averages.
    output = {}

    for metric in (
        "leakage",
        "first_order_gap",
        "second_order_gap",
    ):
        stacked = np.vstack(
            [
                family_results[
                    "dephasing"
                ][
                    metric
                ],
                family_results[
                    "transverse"
                ][
                    metric
                ],
                family_results[
                    "mixed"
                ][
                    metric
                ],
            ]
        )

        output[
            metric
        ] = np.median(
            stacked,
            axis=0,
        )

    return output


def average_ranks(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    order = np.argsort(
        values,
        kind="stable",
    )

    sorted_values = values[
        order
    ]

    ranks = np.empty(
        len(
            values
        ),
        dtype=float,
    )

    i = 0

    while i < len(
        values
    ):
        j = i + 1

        while (
            j < len(
                values
            )
            and sorted_values[j]
            == sorted_values[i]
        ):
            j += 1

        avg_rank = 0.5 * (
            (
                i + 1
            )
            + j
        )

        ranks[
            order[
                i:j
            ]
        ] = avg_rank

        i = j

    return ranks


def auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float | None:
    labels = np.asarray(
        labels,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    n_pos = int(
        np.sum(
            labels
            == 1
        )
    )

    n_neg = int(
        np.sum(
            labels
            == 0
        )
    )

    if (
        n_pos == 0
        or n_neg == 0
    ):
        return None

    ranks = average_ranks(
        scores
    )

    rank_sum_pos = float(
        np.sum(
            ranks[
                labels
                == 1
            ]
        )
    )

    u = (
        rank_sum_pos
        - n_pos
        * (
            n_pos
            + 1
        )
        / 2.0
    )

    return float(
        u
        / (
            n_pos
            * n_neg
        )
    )


def top_quartile_labels(
    response: np.ndarray,
) -> np.ndarray:
    response = np.asarray(
        response,
        dtype=float,
    )

    threshold = float(
        np.quantile(
            response,
            0.75,
        )
    )

    labels = (
        response
        >= threshold
    ).astype(
        int
    )

    if np.all(
        labels
        == 1
    ):
        order = np.argsort(
            response
        )

        labels[:] = 0

        labels[
            order[
                -max(
                    1,
                    len(response)
                    // 4,
                ):
            ]
        ] = 1

    return labels


def evaluate_metric(
    metric_name: str,
    rows: list[CandidateMetric],
) -> dict[str, Any]:
    response_attr = {
        "S1_leakage": "leakage_response",
        "S2_first_order_gap": "first_order_gap_response",
        "S3_second_order_gap": "second_order_gap_response",
    }[
        metric_name
    ]

    pooled_labels = []
    pooled_ss = []
    pooled_en2 = []

    molecule_pools = {
        "H4": {
            "labels": [],
            "ss": [],
            "en2": [],
        },
        "LiH": {
            "labels": [],
            "ss": [],
            "en2": [],
        },
    }

    geometry_results = []

    for molecule, config in MOLECULES.items():
        for geometry in config[
            "geometries"
        ]:
            subset = [
                r
                for r in rows
                if (
                    r.molecule
                    == molecule
                    and abs(
                        r.geometry
                        - geometry
                    )
                    < 1e-12
                )
            ]

            if len(
                subset
            ) < 4:
                continue

            response = np.asarray(
                [
                    float(
                        getattr(
                            r,
                            response_attr,
                        )
                    )
                    for r
                    in subset
                ],
                dtype=float,
            )

            labels = top_quartile_labels(
                response
            )

            ss = np.asarray(
                [
                    r.softspaces_score
                    for r
                    in subset
                ],
                dtype=float,
            )

            en2 = np.asarray(
                [
                    r.en2_pair_score
                    for r
                    in subset
                ],
                dtype=float,
            )

            a_ss = auc(
                labels,
                ss,
            )

            a_en2 = auc(
                labels,
                en2,
            )

            if (
                a_ss is None
                or a_en2 is None
            ):
                continue

            delta = float(
                a_ss
                - a_en2
            )

            geometry_results.append(
                {
                    "molecule": molecule,
                    "geometry": geometry,
                    "n_candidates": len(
                        subset
                    ),
                    "softspaces_auc": float(
                        a_ss
                    ),
                    "en2_auc": float(
                        a_en2
                    ),
                    "delta_auc": delta,
                }
            )

            pooled_labels.extend(
                labels.tolist()
            )

            pooled_ss.extend(
                ss.tolist()
            )

            pooled_en2.extend(
                en2.tolist()
            )

            molecule_pools[
                molecule
            ][
                "labels"
            ].extend(
                labels.tolist()
            )

            molecule_pools[
                molecule
            ][
                "ss"
            ].extend(
                ss.tolist()
            )

            molecule_pools[
                molecule
            ][
                "en2"
            ].extend(
                en2.tolist()
            )

    pooled_ss_auc = auc(
        np.asarray(
            pooled_labels
        ),
        np.asarray(
            pooled_ss
        ),
    )

    pooled_en2_auc = auc(
        np.asarray(
            pooled_labels
        ),
        np.asarray(
            pooled_en2
        ),
    )

    molecule_results = {}

    for molecule, pool in molecule_pools.items():
        m_ss = auc(
            np.asarray(
                pool[
                    "labels"
                ]
            ),
            np.asarray(
                pool[
                    "ss"
                ]
            ),
        )

        m_en2 = auc(
            np.asarray(
                pool[
                    "labels"
                ]
            ),
            np.asarray(
                pool[
                    "en2"
                ]
            ),
        )

        molecule_results[
            molecule
        ] = {
            "softspaces_auc": m_ss,
            "en2_auc": m_en2,
            "delta_auc": (
                None
                if (
                    m_ss is None
                    or m_en2 is None
                )
                else float(
                    m_ss
                    - m_en2
                )
            ),
        }

    if (
        pooled_ss_auc is None
        or pooled_en2_auc is None
    ):
        pooled_delta = None
        median_geometry_delta = None
        positive_geometries = 0
        clear_pass = False

    else:
        pooled_delta = float(
            pooled_ss_auc
            - pooled_en2_auc
        )

        deltas = [
            r[
                "delta_auc"
            ]
            for r
            in geometry_results
        ]

        median_geometry_delta = (
            float(
                np.median(
                    deltas
                )
            )
            if deltas
            else None
        )

        positive_geometries = int(
            sum(
                d > 0.0
                for d
                in deltas
            )
        )

        h4_delta = molecule_results[
            "H4"
        ][
            "delta_auc"
        ]

        lih_delta = molecule_results[
            "LiH"
        ][
            "delta_auc"
        ]

        clear_pass = bool(
            pooled_ss_auc
            >= CLEAR_SS_AUC
            and pooled_delta
            >= CLEAR_DELTA_AUC
            and h4_delta is not None
            and h4_delta > 0.0
            and lih_delta is not None
            and lih_delta > 0.0
            and median_geometry_delta is not None
            and median_geometry_delta > 0.0
            and positive_geometries
            >= CLEAR_POSITIVE_GEOMETRIES
        )

    return {
        "metric": metric_name,
        "pooled_softspaces_auc": pooled_ss_auc,
        "pooled_en2_auc": pooled_en2_auc,
        "pooled_delta_auc": pooled_delta,
        "molecule_results": molecule_results,
        "median_geometry_delta_auc": median_geometry_delta,
        "positive_geometry_count": positive_geometries,
        "total_geometry_count": len(
            geometry_results
        ),
        "geometry_results": geometry_results,
        "clear_pass": clear_pass,
    }


def save_candidate_csv(
    rows: list[CandidateMetric],
    path: Path,
) -> None:
    fields = list(
        CandidateMetric.__dataclass_fields__.keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                asdict(
                    row
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6 Point 1: Soft-Spaces quantum sensing / metrology gauntlet."
        )
    )

    # Script should live in Phase6/V_1_0/code/sensing/.
    v1_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            v1_root
            / "results"
            / "sensing"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 84)
    print("SOFT SPACES — PHASE 6 v60_1 QUANTUM SENSING / METROLOGY GAUNTLET")
    print("=" * 84)
    print("Molecules       : H4 + LiH")
    print(f"Confirm seeds   : {CONFIRM_SEEDS[0]}..{CONFIRM_SEEDS[-1]}")
    print(f"Held-out seeds  : {HELDOUT_SEEDS[0]}..{HELDOUT_SEEDS[-1]}")
    print(
        "CLEAR PASS/test : pooled SS AUC>=0.65, ΔAUC>=0.10, "
        "SS>EN2 on both molecules, median geom ΔAUC>0, >=10/16 positive"
    )
    print(
        f"OVERALL SUCCESS : at least {OVERALL_REQUIRED_CLEAR_TESTS}/3 sensing tests CLEAR PASS"
    )
    print("-" * 84)

    rows: list[
        CandidateMetric
    ] = []

    build_summary = {}

    for molecule, config in MOLECULES.items():
        build_summary[
            molecule
        ] = {}

        for gi, geometry in enumerate(
            config[
                "geometries"
            ],
            start=1,
        ):
            print(
                f"[{molecule} {gi}/8] geometry={geometry:.2f} Å",
                flush=True,
            )

            system = build_problem(
                molecule,
                geometry,
            )

            candidates = build_softspaces_candidates(
                system
            )

            en2_det = en2_determinant_scores(
                system[
                    "h"
                ]
            )

            responses = heldout_sensing_responses(
                system,
                candidates,
            )

            n_pass = int(
                sum(
                    c[
                        "dual_pass"
                    ]
                    for c
                    in candidates
                )
            )

            build_summary[
                molecule
            ][
                str(
                    geometry
                )
            ] = {
                "sector_dimension": len(
                    system[
                        "evals"
                    ]
                ),
                "eligible_candidates": len(
                    candidates
                ),
                "dual_pass_candidates": n_pass,
                "zero_projected_transverse_terms": int(
                    sum(
                        float(
                            np.linalg.norm(
                                a,
                                ord="fro",
                            )
                        )
                        == 0.0
                        for a
                        in perturbation_pools(
                            system[
                                "basis"
                            ],
                            system[
                                "n_qubits"
                            ],
                        )[
                            "transverse"
                        ]
                    )
                ),
            }

            for ci, c in enumerate(
                candidates
            ):
                k = c[
                    "k"
                ]

                rows.append(
                    CandidateMetric(
                        molecule=molecule,
                        geometry=float(
                            geometry
                        ),
                        eigenpair_index=int(
                            k
                        ),
                        gap_hartree=float(
                            c[
                                "gap"
                            ]
                        ),
                        prominence_positive_rate=float(
                            c[
                                "prominence_positive_rate"
                            ]
                        ),
                        robust_positive_rate=float(
                            c[
                                "robust_positive_rate"
                            ]
                        ),
                        softspaces_score=float(
                            c[
                                "softspaces_score"
                            ]
                        ),
                        dual_pass=bool(
                            c[
                                "dual_pass"
                            ]
                        ),
                        en2_pair_score=float(
                            en2_pair_score(
                                en2_det,
                                system[
                                    "evecs"
                                ],
                                k,
                            )
                        ),
                        leakage_response=float(
                            responses[
                                "leakage"
                            ][
                                ci
                            ]
                        ),
                        first_order_gap_response=float(
                            responses[
                                "first_order_gap"
                            ][
                                ci
                            ]
                        ),
                        second_order_gap_response=float(
                            responses[
                                "second_order_gap"
                            ][
                                ci
                            ]
                        ),
                    )
                )

            print(
                f"    sector={len(system['evals'])} | "
                f"eligible={len(candidates)} | "
                f"DUAL PASS={n_pass}"
            )

    print("-" * 84)
    print("Evaluating sensing tests...")

    results = []

    for metric_name in (
        "S1_leakage",
        "S2_first_order_gap",
        "S3_second_order_gap",
    ):
        result = evaluate_metric(
            metric_name,
            rows,
        )

        results.append(
            result
        )

        print("-" * 84)
        print(
            metric_name
        )

        print(
            f"  pooled SS AUC  = "
            f"{result['pooled_softspaces_auc']:.3f}"
        )

        print(
            f"  pooled EN2 AUC = "
            f"{result['pooled_en2_auc']:.3f}"
        )

        print(
            f"  ΔAUC SS-EN2    = "
            f"{result['pooled_delta_auc']:+.3f}"
        )

        for molecule in (
            "H4",
            "LiH",
        ):
            m = result[
                "molecule_results"
            ][
                molecule
            ]

            print(
                f"  {molecule:3s}: "
                f"SS={m['softspaces_auc']:.3f} | "
                f"EN2={m['en2_auc']:.3f} | "
                f"Δ={m['delta_auc']:+.3f}"
            )

        print(
            f"  + geometries   = "
            f"{result['positive_geometry_count']}/"
            f"{result['total_geometry_count']}"
        )

        print(
            "  RESULT         = "
            + (
                "CLEAR PASS"
                if result[
                    "clear_pass"
                ]
                else "FAIL"
            )
        )

    clear_tests = [
        r[
            "metric"
        ]
        for r
        in results
        if r[
            "clear_pass"
        ]
    ]

    overall_success = (
        len(
            clear_tests
        )
        >= OVERALL_REQUIRED_CLEAR_TESTS
    )

    candidate_csv = (
        output_dir
        / "phase6_v60_1_candidate_metrics.csv"
    )

    summary_json = (
        output_dir
        / "phase6_v60_1_sensing_summary.json"
    )

    hash_path = (
        output_dir
        / "phase6_v60_1_sensing_summary.sha256"
    )

    save_candidate_csv(
        rows,
        candidate_csv,
    )

    summary = {
        "_metadata": {
            "generated_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "script_version": SCRIPT_VERSION,
            "python_version": sys.version.split()[0],
        },
        "purpose": (
            "Test Soft-Spaces predictive utility for quantum-sensing-like "
            "local spectral susceptibility against EN2 on H4 and LiH."
        ),
        "frozen_design": {
            "molecules": {
                key: {
                    "geometries": list(
                        value[
                            "geometries"
                        ]
                    )
                }
                for key, value
                in MOLECULES.items()
            },
            "confirm_seeds": list(
                CONFIRM_SEEDS
            ),
            "heldout_seeds": list(
                HELDOUT_SEEDS
            ),
            "local_radius": LOCAL_RADIUS,
            "local_step": LOCAL_STEP,
            "eps_neighbor": EPS_NEIGHBOR,
            "energy_reg": ENERGY_REG,
            "sensing_reg": SENSING_REG,
            "prominence_threshold": PROM_THRESHOLD,
            "robust_threshold": ROBUST_THRESHOLD,
            "clear_ss_auc": CLEAR_SS_AUC,
            "clear_delta_auc": CLEAR_DELTA_AUC,
            "clear_positive_geometries": CLEAR_POSITIVE_GEOMETRIES,
            "overall_required_clear_tests": OVERALL_REQUIRED_CLEAR_TESTS,
        },
        "sensing_metrics": {
            "S1_leakage": (
                "regularized perturbative coupling from adjacent eigenpair "
                "subspace to its complement"
            ),
            "S2_first_order_gap": (
                "absolute first-order perturbative change in adjacent-state gap"
            ),
            "S3_second_order_gap": (
                "absolute difference of regularized second-order energy "
                "corrections for the adjacent states"
            ),
        },
        "build_summary": build_summary,
        "results": results,
        "clear_tests": clear_tests,
        "overall_point1_success": overall_success,
        "interpretation": (
            "POINT 1 SUPPORTED under the preregistered Phase-6 rule."
            if overall_success
            else
            "POINT 1 NOT DEMONSTRATED under the preregistered Phase-6 rule. "
            "Do not retune these thresholds post hoc."
        ),
        "claim_boundary": {
            "tests_predictive_information": True,
            "tests_computational_speedup": False,
            "tests_hardware_sensing_advantage": False,
            "reason": (
                "The benchmark uses exact molecular Hamiltonians/eigensystems "
                "and perturbation-theory response metrics."
            ),
        },
    }

    summary_json.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    digest = sha256_obj(
        summary
    )

    hash_path.write_text(
        digest + "\n",
        encoding="ascii",
    )

    print("=" * 84)
    print("v60_1 SENSING GAUNTLET COMPLETE")
    print("=" * 84)
    print(f"Candidate CSV : {candidate_csv}")
    print(f"Summary JSON  : {summary_json}")
    print(f"SHA-256       : {digest}")
    print(
        "CLEAR TESTS   : "
        + (
            ", ".join(
                clear_tests
            )
            if clear_tests
            else "NONE"
        )
    )
    print(
        "POINT 1       : "
        + (
            "SUPPORTED"
            if overall_success
            else "NOT DEMONSTRATED"
        )
    )
    print("=" * 84)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
