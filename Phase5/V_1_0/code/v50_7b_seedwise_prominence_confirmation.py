#!/usr/bin/env python3
"""
Soft Spaces Phase 5 — v50_7b seed-wise prominence confirmation

Purpose
-------
Correct the v50_7 method bridge so that the original Phase-2/3 hotspot
decision rule is applied at the proper recurrence-unit level.

Frozen Phase-2/3 law
--------------------
For each independent seed and each candidate adjacent-eigenstate pair k:

    DeltaC_f(k, seed)
        = C_REAL,f(k, seed) - C_NULL,f(k, seed)

for f in {dephasing, transverse}.

Then, seed by seed:

    S(k, seed)
        = min(
            DeltaC_dephasing(k, seed),
            DeltaC_transverse(k, seed)
          )

    prominence(k, seed)
        = S(k, seed)
          - median(
                S(k', seed)
                for local controls k'
            )

The confirmation statistic is then

    prominence_positive_rate(k)
        = fraction of valid independent seeds
          with prominence(k, seed) > 0

Frozen gate
-----------
    PASS if prominence_positive_rate >= 0.75

With 12 valid seeds this corresponds to at least 9/12 positive seed units.

Scientific role
---------------
This is NOT the EN2-like Phase-5 determinant proxy.

It tests whether the original Soft-Spaces cancellation/prominence mechanism
recurs inside a molecular Hamiltonian using the original style of
REAL-vs-spectrum-matched-Haar-NULL comparison.

Default benchmark
-----------------
H4 / STO-3G
fixed (N_alpha,N_beta)=(2,2) sector
eight Phase-5 geometries

Perturbation adaptation
-----------------------
The original qubit perturbation families are projected into the physical
fixed-particle determinant sector:

dephasing:
    local Z + nearest-neighbor ZZ

transverse:
    local X + nearest-neighbor XX

Single-X projections vanish in a fixed-particle sector.  Some XX projections
remain nonzero.  The number of zero transverse terms is reported.

Frozen settings
---------------
seeds          = 25043000 ... 25043011
local radius   = +/-10 eigenpair indices
local step     = 2
eps_neighbor   = 0.05 Hartree
energy_reg     = 1e-3 Hartree
PASS threshold = 0.75

No outcome-dependent retuning is performed.

Usage
-----
    python -X utf8 -u ./v50_7b_seedwise_prominence_confirmation.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_VERSION = "v50_7b"
PROJECT = "Molecular Quantum Soft Spaces"
PHASE = "Phase 5"

SEEDS = tuple(range(25043000, 25043012))

GEOMETRIES = (
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00,
)

LOCAL_RADIUS = 10
LOCAL_STEP = 2

EPS_NEIGHBOR = 0.05
ENERGY_REG = 1e-3

PROMINENCE_THRESHOLD = 0.75


@dataclass(frozen=True)
class SeedCandidateRow:
    geometry_angstrom: float
    seed: int
    eigenpair_index: int
    energy_low: float
    energy_high: float
    gap_hartree: float
    delta_c_dephasing: float
    delta_c_transverse: float
    robust_delta_c: float
    local_background: float
    prominence: float
    positive_prominence: bool


@dataclass(frozen=True)
class CandidateSummaryRow:
    geometry_angstrom: float
    eigenpair_index: int
    energy_low: float
    energy_high: float
    gap_hartree: float
    valid_seed_units: int
    positive_prominence_units: int
    prominence_positive_rate: float
    median_prominence: float
    min_prominence: float
    max_prominence: float
    median_robust_delta_c: float
    dephasing_positive_rate: float
    transverse_positive_rate: float
    robust_positive_rate: float
    status: str


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(
        canonical_json_bytes(obj)
    ).hexdigest()


def popcount(x: int) -> int:
    return bin(int(x)).count("1")


def alpha_beta_counts(
    basis_index: int,
    n_spatial: int,
) -> tuple[int, int]:
    mask = (1 << n_spatial) - 1
    alpha_bits = basis_index & mask
    beta_bits = (
        basis_index >> n_spatial
    ) & mask

    return (
        popcount(alpha_bits),
        popcount(beta_bits),
    )


def fixed_sector_basis(
    n_spatial: int,
    n_alpha: int,
    n_beta: int,
) -> np.ndarray:
    n_qubits = 2 * n_spatial
    out = []

    for basis_index in range(
        1 << n_qubits
    ):
        na, nb = alpha_beta_counts(
            basis_index,
            n_spatial,
        )

        if (
            na == n_alpha
            and nb == n_beta
        ):
            out.append(
                basis_index
            )

    expected = (
        math.comb(
            n_spatial,
            n_alpha,
        )
        * math.comb(
            n_spatial,
            n_beta,
        )
    )

    if len(out) != expected:
        raise RuntimeError(
            "Fixed-sector dimension mismatch: "
            f"expected {expected}, got {len(out)}"
        )

    return np.asarray(
        out,
        dtype=np.int64,
    )


def build_h4(
    spacing: float,
) -> dict[str, Any]:
    try:
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        from qiskit_nature.units import DistanceUnit
    except Exception as exc:
        raise RuntimeError(
            "Qiskit Nature / PySCF imports failed. "
            "Run in the Phase-5 WSL environment."
        ) from exc

    z_positions = (
        -1.5 * spacing,
        -0.5 * spacing,
        +0.5 * spacing,
        +1.5 * spacing,
    )

    atom = "; ".join(
        f"H 0.0 0.0 {z:.12f}"
        for z in z_positions
    )

    driver = PySCFDriver(
        atom=atom,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    problem = driver.run()

    n_spatial = int(
        problem.num_spatial_orbitals
    )

    n_alpha = int(
        problem.num_alpha
    )

    n_beta = int(
        problem.num_beta
    )

    n_qubits = 2 * n_spatial

    if (
        n_spatial != 4
        or n_alpha != 2
        or n_beta != 2
    ):
        raise RuntimeError(
            "Unexpected H4/STO-3G metadata: "
            f"n_spatial={n_spatial}, "
            f"Nalpha={n_alpha}, "
            f"Nbeta={n_beta}"
        )

    mapper = JordanWignerMapper()

    qubit_op = mapper.map(
        problem.hamiltonian.second_q_op()
    )

    basis = fixed_sector_basis(
        n_spatial,
        n_alpha,
        n_beta,
    )

    full_sparse = qubit_op.to_matrix(
        sparse=True
    )

    sector_sparse = full_sparse[
        basis, :
    ][:, basis]

    h = np.asarray(
        sector_sparse.toarray(),
        dtype=np.complex128,
    )

    herm_error = float(
        np.max(
            np.abs(
                h
                - h.conjugate().T
            )
        )
    )

    if herm_error > 1e-10:
        raise RuntimeError(
            f"Hamiltonian is not Hermitian: {herm_error:.3e}"
        )

    evals, evecs = np.linalg.eigh(
        h
    )

    return {
        "atom": atom,
        "n_spatial": n_spatial,
        "n_qubits": n_qubits,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "basis": basis,
        "hamiltonian": h,
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
    dimension: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(
        seed
    )

    z = (
        rng.normal(
            size=(
                dimension,
                dimension,
            )
        )
        + 1j
        * rng.normal(
            size=(
                dimension,
                dimension,
            )
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
    ) > 0.0

    phases[nz] = (
        diag[nz]
        / np.abs(
            diag[nz]
        )
    )

    q = (
        q
        * phases.conjugate()
    )

    return np.asarray(
        q,
        dtype=np.complex128,
    )


def projected_z(
    basis: np.ndarray,
    qubit: int,
) -> np.ndarray:
    diag = np.asarray(
        [
            -1.0
            if (
                (
                    int(x)
                    >> qubit
                )
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
    diag = []

    for x in basis:
        z1 = (
            -1.0
            if (
                (
                    int(x)
                    >> q1
                )
                & 1
            )
            else 1.0
        )

        z2 = (
            -1.0
            if (
                (
                    int(x)
                    >> q2
                )
                & 1
            )
            else 1.0
        )

        diag.append(
            z1 * z2
        )

    return np.diag(
        np.asarray(
            diag,
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
        int(full_index): local_index
        for (
            local_index,
            full_index,
        ) in enumerate(
            basis
        )
    }

    n = len(
        basis
    )

    out = np.zeros(
        (n, n),
        dtype=np.complex128,
    )

    for (
        col,
        full_index,
    ) in enumerate(
        basis
    ):
        target = (
            int(full_index)
            ^ int(flip_mask)
        )

        row = lookup.get(
            target
        )

        if row is not None:
            out[
                row,
                col,
            ] = 1.0

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
                (
                    (1 << q)
                    | (
                        1
                        << (
                            q + 1
                        )
                    )
                ),
            )
        )

    return {
        "dephasing": (
            dephasing
        ),
        "transverse": (
            transverse
        ),
    }


def random_family_perturbation(
    pool: list[np.ndarray],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(
        seed
    )

    coeffs = rng.normal(
        size=len(
            pool
        )
    )

    d = pool[0].shape[0]

    out = np.zeros(
        (d, d),
        dtype=np.complex128,
    )

    for (
        coeff,
        term,
    ) in zip(
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
    """
    Frozen Phase-2 matrix-cancellation quantity.
    """
    d = int(
        len(
            evals
        )
    )

    j = i + 1

    if (
        i < 0
        or j >= d
    ):
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
            [i, j],
        ],
        dtype=np.complex128,
    )

    eq = np.asarray(
        evals[qmask],
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

    inv1 = (
        de
        / (
            de * de
            + reg * reg
        )
    )

    sum_a = np.zeros(
        (2, 2),
        dtype=np.complex128,
    )

    sum_norm_a = 0.0

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

        a = float(
            inv1[r]
        ) * (
            v.conjugate()
            @ v.T
        )

        a = 0.5 * (
            a
            + a.conjugate().T
        )

        sum_a += a

        sum_norm_a += float(
            np.linalg.norm(
                a,
                ord="fro",
            )
        )

    if (
        sum_norm_a <= 0.0
        or not np.isfinite(
            sum_norm_a
        )
    ):
        return None

    return float(
        np.linalg.norm(
            sum_a,
            ord="fro",
        )
        / sum_norm_a
    )


def finite_median(
    values: list[float],
) -> float:
    finite = np.asarray(
        [
            x
            for x in values
            if np.isfinite(
                x
            )
        ],
        dtype=float,
    )

    if len(
        finite
    ) == 0:
        return float(
            "nan"
        )

    return float(
        np.median(
            finite
        )
    )


def positive_rate(
    values: list[float],
) -> float:
    finite = np.asarray(
        [
            x
            for x in values
            if np.isfinite(
                x
            )
        ],
        dtype=float,
    )

    if len(
        finite
    ) == 0:
        return float(
            "nan"
        )

    return float(
        np.mean(
            finite > 0.0
        )
    )


def seedwise_geometry(
    spacing: float,
) -> tuple[
    list[SeedCandidateRow],
    list[CandidateSummaryRow],
    dict[str, Any],
]:
    p = build_h4(
        spacing
    )

    evals = p[
        "evals"
    ]

    v_real = p[
        "evecs"
    ]

    basis = p[
        "basis"
    ]

    d = len(
        evals
    )

    pools = perturbation_pools(
        basis,
        p[
            "n_qubits"
        ],
    )

    zero_transverse = sum(
        np.linalg.norm(
            term,
            ord="fro",
        )
        == 0.0
        for term in pools[
            "transverse"
        ]
    )

    # raw[(seed, family, k)] = DeltaC
    raw: dict[
        tuple[
            int,
            str,
            int,
        ],
        float,
    ] = {}

    for seed in SEEDS:
        v_null = haar_unitary(
            d,
            seed
            + 99_000_000,
        )

        for (
            family_index,
            family,
        ) in enumerate(
            (
                "dephasing",
                "transverse",
            )
        ):
            dh = (
                random_family_perturbation(
                    pools[
                        family
                    ],
                    seed
                    + 1_000_000
                    * (
                        family_index
                        + 1
                    ),
                )
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

    # robust[(seed,k)] = min(dephasing, transverse)
    robust: dict[
        tuple[
            int,
            int,
        ],
        float,
    ] = {}

    for seed in SEEDS:
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
                and np.isfinite(
                    a
                )
                and np.isfinite(
                    t
                )
            ):
                robust[
                    (
                        seed,
                        k,
                    )
                ] = min(
                    float(
                        a
                    ),
                    float(
                        t
                    ),
                )

    seed_rows: list[
        SeedCandidateRow
    ] = []

    by_candidate: dict[
        int,
        list[
            SeedCandidateRow
        ],
    ] = {
        k: []
        for k in range(
            d - 1
        )
    }

    for seed in SEEDS:
        valid_k = [
            k
            for k in range(
                d - 1
            )
            if (
                seed,
                k,
            )
            in robust
        ]

        for k in valid_k:
            local_controls = [
                kk
                for kk in valid_k
                if kk != k
                and abs(
                    kk - k
                )
                <= LOCAL_RADIUS
                and (
                    abs(
                        kk - k
                    )
                    % LOCAL_STEP
                    == 0
                )
            ]

            background = finite_median(
                [
                    robust[
                        (
                            seed,
                            kk,
                        )
                    ]
                    for kk in local_controls
                ]
            )

            if not np.isfinite(
                background
            ):
                continue

            prominence = (
                robust[
                    (
                        seed,
                        k,
                    )
                ]
                - background
            )

            row = SeedCandidateRow(
                geometry_angstrom=float(
                    spacing
                ),
                seed=int(
                    seed
                ),
                eigenpair_index=int(
                    k
                ),
                energy_low=float(
                    evals[
                        k
                    ]
                ),
                energy_high=float(
                    evals[
                        k + 1
                    ]
                ),
                gap_hartree=float(
                    evals[
                        k + 1
                    ]
                    - evals[
                        k
                    ]
                ),
                delta_c_dephasing=float(
                    raw[
                        (
                            seed,
                            "dephasing",
                            k,
                        )
                    ]
                ),
                delta_c_transverse=float(
                    raw[
                        (
                            seed,
                            "transverse",
                            k,
                        )
                    ]
                ),
                robust_delta_c=float(
                    robust[
                        (
                            seed,
                            k,
                        )
                    ]
                ),
                local_background=float(
                    background
                ),
                prominence=float(
                    prominence
                ),
                positive_prominence=bool(
                    prominence > 0.0
                ),
            )

            seed_rows.append(
                row
            )

            by_candidate[
                k
            ].append(
                row
            )

    summaries: list[
        CandidateSummaryRow
    ] = []

    for k in range(
        d - 1
    ):
        rows = by_candidate[
            k
        ]

        if not rows:
            continue

        prominence_values = [
            row.prominence
            for row in rows
        ]

        robust_values = [
            row.robust_delta_c
            for row in rows
        ]

        deph_values = [
            row.delta_c_dephasing
            for row in rows
        ]

        trans_values = [
            row.delta_c_transverse
            for row in rows
        ]

        n_valid = len(
            rows
        )

        n_positive = sum(
            row.positive_prominence
            for row in rows
        )

        prate = (
            n_positive
            / n_valid
        )

        status = (
            "PASS"
            if prate
            >= PROMINENCE_THRESHOLD
            else "FAIL"
        )

        summaries.append(
            CandidateSummaryRow(
                geometry_angstrom=float(
                    spacing
                ),
                eigenpair_index=int(
                    k
                ),
                energy_low=float(
                    evals[
                        k
                    ]
                ),
                energy_high=float(
                    evals[
                        k + 1
                    ]
                ),
                gap_hartree=float(
                    evals[
                        k + 1
                    ]
                    - evals[
                        k
                    ]
                ),
                valid_seed_units=int(
                    n_valid
                ),
                positive_prominence_units=int(
                    n_positive
                ),
                prominence_positive_rate=float(
                    prate
                ),
                median_prominence=float(
                    finite_median(
                        prominence_values
                    )
                ),
                min_prominence=float(
                    np.min(
                        prominence_values
                    )
                ),
                max_prominence=float(
                    np.max(
                        prominence_values
                    )
                ),
                median_robust_delta_c=float(
                    finite_median(
                        robust_values
                    )
                ),
                dephasing_positive_rate=float(
                    positive_rate(
                        deph_values
                    )
                ),
                transverse_positive_rate=float(
                    positive_rate(
                        trans_values
                    )
                ),
                robust_positive_rate=float(
                    positive_rate(
                        robust_values
                    )
                ),
                status=status,
            )
        )

    summaries.sort(
        key=lambda x: (
            x.prominence_positive_rate,
            x.median_prominence,
            x.median_robust_delta_c,
        ),
        reverse=True,
    )

    meta = {
        "geometry_angstrom": float(
            spacing
        ),
        "sector_dimension": int(
            d
        ),
        "total_adjacent_pairs": int(
            d - 1
        ),
        "seed_count": len(
            SEEDS
        ),
        "zero_projected_transverse_terms": int(
            zero_transverse
        ),
        "total_transverse_terms": int(
            len(
                pools[
                    "transverse"
                ]
            )
        ),
        "passing_candidates": [
            asdict(
                row
            )
            for row in summaries
            if row.status
            == "PASS"
        ],
        "top_candidates": [
            asdict(
                row
            )
            for row in summaries[
                :10
            ]
        ],
        "ground_pair": (
            next(
                (
                    asdict(
                        row
                    )
                    for row in summaries
                    if row.eigenpair_index
                    == 0
                ),
                None,
            )
        ),
    }

    return (
        seed_rows,
        summaries,
        meta,
    )


def save_dataclass_csv(
    rows: list[Any],
    path: Path,
) -> None:
    if not rows:
        raise RuntimeError(
            f"No rows to save: {path}"
        )

    fields = list(
        rows[0].__dataclass_fields__.keys()
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
            "Phase 5 v50_7b seed-wise "
            "Soft-Spaces prominence confirmation."
        )
    )

    v1_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            v1_root
            / "results"
            / "phase23_method_bridge"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("SOFT SPACES — PHASE 5 v50_7b SEED-WISE PROMINENCE CONFIRMATION")
    print("=" * 78)
    print("Molecule       : H4")
    print("Basis          : STO-3G")
    print(f"Geometries     : {GEOMETRIES}")
    print(f"Seeds          : {SEEDS}")
    print(f"Gate           : prominence positive-rate >= {PROMINENCE_THRESHOLD:.2f}")
    print(f"eps_neighbor   : {EPS_NEIGHBOR}")
    print(f"energy_reg     : {ENERGY_REG}")
    print(f"local radius   : +/-{LOCAL_RADIUS}")
    print(f"local step     : {LOCAL_STEP}")
    print("-" * 78)
    print(
        "Each independent seed is now one recurrence unit."
    )
    print("-" * 78)

    all_seed_rows: list[
        SeedCandidateRow
    ] = []

    all_summary_rows: list[
        CandidateSummaryRow
    ] = []

    geometry_meta: dict[
        str,
        Any,
    ] = {}

    for (
        gi,
        spacing,
    ) in enumerate(
        GEOMETRIES,
        start=1,
    ):
        print(
            f"[{gi}/{len(GEOMETRIES)}] "
            f"H4 geometry={spacing:.2f} Å",
            flush=True,
        )

        (
            seed_rows,
            summaries,
            meta,
        ) = seedwise_geometry(
            spacing
        )

        all_seed_rows.extend(
            seed_rows
        )

        all_summary_rows.extend(
            summaries
        )

        geometry_meta[
            str(
                spacing
            )
        ] = meta

        passing = [
            row
            for row in summaries
            if row.status
            == "PASS"
        ]

        print(
            f"    candidates with finite seed-wise prominence: "
            f"{len(summaries)}"
        )

        print(
            f"    PASS candidates (>=75%): "
            f"{len(passing)}"
        )

        if summaries:
            top = summaries[
                0
            ]

            print(
                f"    top k={top.eigenpair_index} | "
                f"positive={top.positive_prominence_units}/"
                f"{top.valid_seed_units} | "
                f"rate={top.prominence_positive_rate:.3f} | "
                f"median prominence={top.median_prominence:+.6f} | "
                f"status={top.status}"
            )

        gp = meta[
            "ground_pair"
        ]

        if gp is not None:
            print(
                f"    ground pair k=0 | "
                f"positive={gp['positive_prominence_units']}/"
                f"{gp['valid_seed_units']} | "
                f"rate={gp['prominence_positive_rate']:.3f} | "
                f"median prominence={gp['median_prominence']:+.6f} | "
                f"status={gp['status']}"
            )
        else:
            print(
                "    ground pair k=0 | no valid seed-wise prominence units"
            )

    seed_csv = (
        output_dir
        / "phase5_v50_7b_seedwise_prominence.csv"
    )

    candidate_csv = (
        output_dir
        / "phase5_v50_7b_candidate_summary.csv"
    )

    summary_json = (
        output_dir
        / "phase5_v50_7b_summary.json"
    )

    hash_file = (
        output_dir
        / "phase5_v50_7b_summary.sha256"
    )

    save_dataclass_csv(
        all_seed_rows,
        seed_csv,
    )

    save_dataclass_csv(
        all_summary_rows,
        candidate_csv,
    )

    passing_rows = [
        row
        for row in all_summary_rows
        if row.status
        == "PASS"
    ]

    geometries_with_pass = sorted(
        {
            row.geometry_angstrom
            for row in passing_rows
        }
    )

    full_12_seed_passes = [
        row
        for row in passing_rows
        if row.valid_seed_units
        == len(
            SEEDS
        )
    ]

    summary = {
        "_metadata": {
            "generated_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "script_version": (
                SCRIPT_VERSION
            ),
            "python_version": (
                sys.version.split()[0]
            ),
        },
        "project": PROJECT,
        "phase": PHASE,
        "benchmark": {
            "molecule": "H4",
            "basis": "STO-3G",
            "geometries": list(
                GEOMETRIES
            ),
            "fixed_sector": (
                "(N_alpha,N_beta)=(2,2)"
            ),
        },
        "frozen_phase23_rule": {
            "delta_c": (
                "C_REAL - C_NULL"
            ),
            "robust_seed_score": (
                "min(DeltaC_dephasing, DeltaC_transverse)"
            ),
            "prominence_seed_score": (
                "robust(k,seed) - median(local robust controls at same seed)"
            ),
            "confirmation_gate": (
                "fraction of valid independent seeds "
                "with prominence > 0 must be >= 0.75"
            ),
            "threshold": (
                PROMINENCE_THRESHOLD
            ),
            "seeds": list(
                SEEDS
            ),
        },
        "geometry_details": (
            geometry_meta
        ),
        "global_confirmation": {
            "candidate_geometry_rows": len(
                all_summary_rows
            ),
            "passing_candidate_geometry_rows": len(
                passing_rows
            ),
            "geometries_with_at_least_one_pass": (
                geometries_with_pass
            ),
            "n_geometries_with_at_least_one_pass": len(
                geometries_with_pass
            ),
            "full_12_seed_pass_count": len(
                full_12_seed_passes
            ),
            "passing_rows": [
                asdict(
                    row
                )
                for row in passing_rows
            ],
        },
        "interpretation_boundary": {
            "phase5_proxy_used": False,
            "en2_used": False,
            "determinant_selection_claim": False,
            "purpose": (
                "Confirm or falsify transfer of the original "
                "Phase-2/3 seed-wise Soft-Spaces prominence mechanism "
                "to H4 molecular eigenstate subspaces."
            ),
            "important_adaptation": (
                "The original qubit Pauli perturbations are projected "
                "into the fixed-particle determinant sector."
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

    digest = sha256_hex(
        summary
    )

    hash_file.write_text(
        digest + "\n",
        encoding="ascii",
    )

    print("-" * 78)
    print("v50_7b COMPLETE")
    print("-" * 78)
    print(f"Seed CSV      : {seed_csv}")
    print(f"Candidate CSV : {candidate_csv}")
    print(f"Summary JSON  : {summary_json}")
    print(f"SHA-256       : {digest}")
    print("-" * 78)
    print(
        f"PASS candidate-geometry rows : {len(passing_rows)}"
    )
    print(
        f"Geometries with >=1 PASS     : "
        f"{len(geometries_with_pass)}/{len(GEOMETRIES)}"
    )
    print(
        f"PASS rows with all 12 seeds  : "
        f"{len(full_12_seed_passes)}"
    )
    print("-" * 78)
    print(
        "PASS means seed-wise prominence positive-rate >= 0.75."
    )
    print(
        "This is the corrected Phase-2/3 confirmation rule; "
        "EN2 is not part of this test."
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
