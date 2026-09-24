#!/usr/bin/env python3
"""
Soft Spaces Phase 5 — v50_7 frozen Phase-2/3 method bridge

Purpose
-------
Test the ACTUAL frozen Soft-Spaces perturbative mechanism from Phase 2/3
on molecular Hamiltonians, rather than the v50_2-v50_6 molecular proxy.

This file deliberately does NOT use the Phase-5 proxy ranking

    |H_ir| / (|H_ii - H_rr| + eps)

as its Soft-Spaces definition.

Instead it transfers the frozen Phase-2/3 mechanism:

    C_matrix = ||sum_m A_m||_F / sum_m ||A_m||_F

    DeltaC_f(k) = C_REAL,f(k) - C_NULL,f(k)

    S(k) = min(
        median_seed DeltaC_dephasing(k),
        median_seed DeltaC_transverse(k)
    )

    prominence(k) = S(k) - median(S(local controls))

where each candidate k denotes a two-dimensional adjacent-eigenstate subspace
P = span{|k>, |k+1>}.

Scientific role
---------------
v50_7 is a METHOD BRIDGE / MECHANISM TRANSFER test.

It asks:

    Does the original Phase-2/3 Soft-Spaces cancellation/prominence law
    occur in molecular electronic Hamiltonians?

It does NOT yet claim to be a determinant-selection algorithm.
That distinction is essential: the original mechanism acts on neighboring
eigenstate subspaces, whereas v50_2-v50_6 ranked determinants.

REAL / NULL
-----------
REAL:
    the molecular electronic Hamiltonian in the physical
    fixed-(N_alpha,N_beta) determinant sector.

NULL:
    exactly the same REAL eigenvalues, but a seeded Haar-random eigenbasis.

Thus the NULL preserves the spectrum but destroys the molecular eigenvector
structure, matching the central Phase-2/3 REAL-vs-NULL logic.

Perturbation families
---------------------
The original Phase-2/3 families were qubit Z/ZZ ("dephasing") and X/XX
("transverse").

For a fixed-particle molecular sector we project those same computational-basis
Pauli actions into the physical determinant sector:

dephasing:
    projected local Z and nearest-neighbor ZZ terms.

transverse:
    projected local X and nearest-neighbor XX terms.

Important:
    projected single-X terms vanish in a fixed-particle sector.
    Nearest-neighbor XX terms can remain non-zero when the two-bit flip maps
    one allowed determinant to another.  This is reported explicitly.

Frozen transfer settings
------------------------
- 12 seeds: 25043000 ... 25043011
- local radius: +/-10 eigenpair indices
- local step: 2
- neighbor-gap threshold eps_neighbor = 0.05 Hartree
- resolvent regularizer = 1e-3 Hartree
- prominence gate = positive prominence (descriptive in v50_7)

These values are not tuned to molecular outcomes.

Default benchmark
-----------------
H4 / STO-3G at the same eight Phase-5 geometries.

Optional:
    --molecule lih

LiH is supported, but the H4 run is the clean first bridge because its
physical sector is much smaller.

Usage
-----
    python -X utf8 -u ./v50_7_frozen_phase23_method_bridge.py

Optional LiH:
    python -X utf8 -u ./v50_7_frozen_phase23_method_bridge.py --molecule lih
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


SCRIPT_VERSION = "v50_7"
PROJECT = "Molecular Quantum Soft Spaces"
PHASE = "Phase 5"

SEEDS = tuple(range(25043000, 25043012))

LOCAL_RADIUS = 10
LOCAL_STEP = 2
EPS_NEIGHBOR = 0.05
ENERGY_REG = 1e-3

H4_GEOMETRIES = (0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
LIH_GEOMETRIES = (1.00, 1.20, 1.40, 1.60, 2.00, 2.50, 3.00, 4.00)


@dataclass(frozen=True)
class CandidateRow:
    molecule: str
    geometry_angstrom: float
    eigenpair_index: int
    energy_low: float
    energy_high: float
    gap_hartree: float
    eligible: bool
    dephasing_delta_c_median: float
    transverse_delta_c_median: float
    robust_delta_c: float
    local_background: float
    prominence: float
    dephasing_positive_rate: float
    transverse_positive_rate: float
    robust_positive_rate: float


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def popcount(x: int) -> int:
    return bin(int(x)).count("1")


def alpha_beta_counts(
    basis_index: int,
    n_spatial: int,
) -> tuple[int, int]:
    mask = (1 << n_spatial) - 1
    alpha = basis_index & mask
    beta = (basis_index >> n_spatial) & mask
    return popcount(alpha), popcount(beta)


def fixed_sector_basis(
    n_spatial: int,
    n_alpha: int,
    n_beta: int,
) -> np.ndarray:
    n_qubits = 2 * n_spatial
    out = []

    for basis_index in range(1 << n_qubits):
        na, nb = alpha_beta_counts(
            basis_index,
            n_spatial,
        )
        if na == n_alpha and nb == n_beta:
            out.append(basis_index)

    expected = (
        math.comb(n_spatial, n_alpha)
        * math.comb(n_spatial, n_beta)
    )

    if len(out) != expected:
        raise RuntimeError(
            f"Sector dimension mismatch: expected {expected}, got {len(out)}"
        )

    return np.asarray(out, dtype=np.int64)


def build_problem(
    molecule: str,
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

    if molecule == "h4":
        positions = (
            -1.5 * spacing,
            -0.5 * spacing,
            +0.5 * spacing,
            +1.5 * spacing,
        )
        atom = "; ".join(
            f"H 0.0 0.0 {z:.12f}"
            for z in positions
        )
    elif molecule == "lih":
        half = spacing / 2.0
        atom = (
            f"Li 0.0 0.0 {-half:.12f}; "
            f"H 0.0 0.0 {half:.12f}"
        )
    else:
        raise ValueError(
            f"Unsupported molecule: {molecule}"
        )

    driver = PySCFDriver(
        atom=atom,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    problem = driver.run()

    n_spatial = int(problem.num_spatial_orbitals)
    n_alpha = int(problem.num_alpha)
    n_beta = int(problem.num_beta)
    n_qubits = 2 * n_spatial

    mapper = JordanWignerMapper()
    qubit_op = mapper.map(
        problem.hamiltonian.second_q_op()
    )

    if int(qubit_op.num_qubits) != n_qubits:
        raise RuntimeError(
            "Mapped qubit count mismatch."
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

    herm_err = float(
        np.max(
            np.abs(
                h - h.conjugate().T
            )
        )
    )

    if herm_err > 1e-10:
        raise RuntimeError(
            f"Hamiltonian not Hermitian: {herm_err:.3e}"
        )

    evals, evecs = np.linalg.eigh(h)

    return {
        "atom": atom,
        "n_spatial": n_spatial,
        "n_qubits": n_qubits,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "basis": basis,
        "hamiltonian": h,
        "evals": np.asarray(evals, dtype=float),
        "evecs": np.asarray(evecs, dtype=np.complex128),
        "hermiticity_error": herm_err,
    }


def haar_unitary(
    dimension: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    z = (
        rng.normal(size=(dimension, dimension))
        + 1j * rng.normal(size=(dimension, dimension))
    ) / math.sqrt(2.0)

    q, r = np.linalg.qr(z)

    diagonal = np.diag(r)
    phases = np.ones_like(diagonal, dtype=np.complex128)

    nz = np.abs(diagonal) > 0
    phases[nz] = diagonal[nz] / np.abs(diagonal[nz])

    q = q * phases.conjugate()

    return np.asarray(
        q,
        dtype=np.complex128,
    )


def projected_pauli_z(
    basis: np.ndarray,
    qubit: int,
) -> np.ndarray:
    signs = np.array(
        [
            -1.0 if ((int(x) >> qubit) & 1) else 1.0
            for x in basis
        ],
        dtype=float,
    )
    return np.diag(signs).astype(np.complex128)


def projected_pauli_zz(
    basis: np.ndarray,
    q1: int,
    q2: int,
) -> np.ndarray:
    signs = np.array(
        [
            (
                -1.0
                if ((int(x) >> q1) & 1)
                else 1.0
            )
            * (
                -1.0
                if ((int(x) >> q2) & 1)
                else 1.0
            )
            for x in basis
        ],
        dtype=float,
    )
    return np.diag(signs).astype(np.complex128)


def projected_bitflip_operator(
    basis: np.ndarray,
    flip_mask: int,
) -> np.ndarray:
    """
    Projection of computational-basis bit flips into the fixed sector.

    For X_q: flip_mask = 1<<q
    For X_q X_r: flip_mask = (1<<q) | (1<<r)
    """
    n = len(basis)
    lookup = {
        int(full_idx): local_idx
        for local_idx, full_idx in enumerate(basis)
    }

    out = np.zeros(
        (n, n),
        dtype=np.complex128,
    )

    for col, full_idx in enumerate(basis):
        target = int(full_idx) ^ int(flip_mask)
        row = lookup.get(target)

        if row is not None:
            out[row, col] = 1.0

    return out


def perturbation_term_pools(
    basis: np.ndarray,
    n_qubits: int,
) -> dict[str, list[np.ndarray]]:
    dephasing = []
    transverse = []

    # Exact projected Z and X local terms.
    for q in range(n_qubits):
        dephasing.append(
            projected_pauli_z(
                basis,
                q,
            )
        )

        transverse.append(
            projected_bitflip_operator(
                basis,
                1 << q,
            )
        )

    # Exact projected nearest-neighbor ZZ and XX terms.
    for q in range(n_qubits - 1):
        dephasing.append(
            projected_pauli_zz(
                basis,
                q,
                q + 1,
            )
        )

        transverse.append(
            projected_bitflip_operator(
                basis,
                (1 << q) | (1 << (q + 1)),
            )
        )

    return {
        "dephasing": dephasing,
        "transverse": transverse,
    }


def random_family_perturbation(
    term_pool: list[np.ndarray],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    coeffs = rng.normal(
        loc=0.0,
        scale=1.0,
        size=len(term_pool),
    )

    d = term_pool[0].shape[0]

    out = np.zeros(
        (d, d),
        dtype=np.complex128,
    )

    for coeff, term in zip(
        coeffs,
        term_pool,
    ):
        out += float(coeff) * term

    norm = float(
        np.linalg.norm(
            out,
            ord="fro",
        )
    )

    if norm > 0.0:
        out /= norm

    return 0.5 * (
        out + out.conjugate().T
    )


def c_matrix(
    evals: np.ndarray,
    bmat: np.ndarray,
    i: int,
) -> float | None:
    """
    Frozen v25.18-v25.30 matrix-cancellation quantity.

    P = adjacent eigenstates i and i+1.
    Q = all remaining eigenstates.

    A_m = g_m * v_m^* v_m^T
    C_matrix = ||sum_m A_m||_F / sum_m ||A_m||_F
    """
    d = len(evals)
    j = i + 1

    if i < 0 or j >= d:
        return None

    gap = abs(
        float(evals[j])
        - float(evals[i])
    )

    if gap >= EPS_NEIGHBOR:
        return None

    qmask = np.ones(
        d,
        dtype=bool,
    )
    qmask[[i, j]] = False

    wq = np.asarray(
        bmat[qmask, :][:, [i, j]],
        dtype=np.complex128,
    )

    eq = np.asarray(
        evals[qmask],
        dtype=float,
    )

    eref = 0.5 * (
        float(evals[i])
        + float(evals[j])
    )

    de = eref - eq

    reg = max(
        ENERGY_REG,
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
            wq[r, :],
            dtype=np.complex128,
        ).reshape(2, 1)

        a = float(inv1[r]) * (
            v.conjugate()
            @ v.T
        )

        a = 0.5 * (
            a + a.conjugate().T
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


def positive_rate(
    values: list[float],
) -> float:
    finite = np.asarray(
        [
            x
            for x in values
            if np.isfinite(x)
        ],
        dtype=float,
    )

    if len(finite) == 0:
        return float("nan")

    return float(
        np.mean(
            finite > 0.0
        )
    )


def finite_median(
    values: list[float],
) -> float:
    finite = np.asarray(
        [
            x
            for x in values
            if np.isfinite(x)
        ],
        dtype=float,
    )

    if len(finite) == 0:
        return float("nan")

    return float(
        np.median(finite)
    )


def run_geometry(
    molecule: str,
    spacing: float,
) -> tuple[list[CandidateRow], dict[str, Any]]:
    p = build_problem(
        molecule,
        spacing,
    )

    evals = p["evals"]
    v_real = p["evecs"]
    basis = p["basis"]
    d = len(evals)

    pools = perturbation_term_pools(
        basis,
        p["n_qubits"],
    )

    zero_transverse_terms = sum(
        1
        for term in pools["transverse"]
        if np.linalg.norm(term, ord="fro") == 0.0
    )

    raw: dict[
        tuple[str, int],
        list[float],
    ] = {}

    for seed in SEEDS:
        # Spectrum-matched Haar NULL.
        v_null = haar_unitary(
            d,
            seed + 99_000_000,
        )

        for family_idx, family in enumerate(
            ("dephasing", "transverse")
        ):
            d_h = random_family_perturbation(
                pools[family],
                seed
                + 1_000_000 * (family_idx + 1),
            )

            # Same physical perturbation operator for REAL and NULL.
            b_real = (
                v_real.conjugate().T
                @ d_h
                @ v_real
            )

            b_null = (
                v_null.conjugate().T
                @ d_h
                @ v_null
            )

            for i in range(d - 1):
                cr = c_matrix(
                    evals,
                    b_real,
                    i,
                )

                cn = c_matrix(
                    evals,
                    b_null,
                    i,
                )

                if cr is None or cn is None:
                    continue

                delta = float(
                    cr - cn
                )

                raw.setdefault(
                    (family, i),
                    [],
                ).append(delta)

    med: dict[
        tuple[str, int],
        float,
    ] = {}

    pos: dict[
        tuple[str, int],
        float,
    ] = {}

    robust: dict[
        int,
        float,
    ] = {}

    robust_seedwise: dict[
        int,
        list[float],
    ] = {}

    for family in (
        "dephasing",
        "transverse",
    ):
        for i in range(d - 1):
            vals = raw.get(
                (family, i),
                [],
            )

            med[
                (family, i)
            ] = finite_median(vals)

            pos[
                (family, i)
            ] = positive_rate(vals)

    for i in range(d - 1):
        a = med[
            ("dephasing", i)
        ]

        t = med[
            ("transverse", i)
        ]

        if (
            np.isfinite(a)
            and np.isfinite(t)
        ):
            robust[i] = min(
                a,
                t,
            )
        else:
            robust[i] = float("nan")

        da = raw.get(
            ("dephasing", i),
            [],
        )

        dt = raw.get(
            ("transverse", i),
            [],
        )

        n = min(
            len(da),
            len(dt),
        )

        robust_seedwise[i] = [
            min(
                float(da[s]),
                float(dt[s]),
            )
            for s in range(n)
        ]

    rows = []

    eligible_indices = [
        i
        for i in range(d - 1)
        if np.isfinite(
            robust.get(
                i,
                float("nan"),
            )
        )
    ]

    for i in range(d - 1):
        gap = float(
            evals[i + 1]
            - evals[i]
        )

        eligible = i in eligible_indices

        if eligible:
            controls = [
                j
                for j in eligible_indices
                if j != i
                and abs(j - i) <= LOCAL_RADIUS
                and abs(j - i) % LOCAL_STEP == 0
            ]

            background = finite_median(
                [
                    robust[j]
                    for j in controls
                ]
            )

            prominence = (
                robust[i] - background
                if np.isfinite(background)
                else float("nan")
            )
        else:
            background = float("nan")
            prominence = float("nan")

        rows.append(
            CandidateRow(
                molecule=molecule.upper(),
                geometry_angstrom=float(spacing),
                eigenpair_index=int(i),
                energy_low=float(evals[i]),
                energy_high=float(evals[i + 1]),
                gap_hartree=gap,
                eligible=bool(eligible),
                dephasing_delta_c_median=float(
                    med[
                        ("dephasing", i)
                    ]
                ),
                transverse_delta_c_median=float(
                    med[
                        ("transverse", i)
                    ]
                ),
                robust_delta_c=float(
                    robust[i]
                ),
                local_background=float(
                    background
                ),
                prominence=float(
                    prominence
                ),
                dephasing_positive_rate=float(
                    pos[
                        ("dephasing", i)
                    ]
                ),
                transverse_positive_rate=float(
                    pos[
                        ("transverse", i)
                    ]
                ),
                robust_positive_rate=float(
                    positive_rate(
                        robust_seedwise[i]
                    )
                ),
            )
        )

    ranked = sorted(
        [
            row
            for row in rows
            if row.eligible
            and np.isfinite(
                row.prominence
            )
        ],
        key=lambda row: (
            row.prominence,
            row.robust_delta_c,
            row.robust_positive_rate,
        ),
        reverse=True,
    )

    meta = {
        "molecule": molecule.upper(),
        "geometry_angstrom": float(spacing),
        "num_spatial_orbitals": int(
            p["n_spatial"]
        ),
        "num_qubits": int(
            p["n_qubits"]
        ),
        "num_alpha": int(
            p["n_alpha"]
        ),
        "num_beta": int(
            p["n_beta"]
        ),
        "sector_dimension": int(d),
        "eligible_adjacent_pairs": int(
            len(eligible_indices)
        ),
        "total_adjacent_pairs": int(
            d - 1
        ),
        "zero_projected_transverse_terms": int(
            zero_transverse_terms
        ),
        "total_transverse_terms": int(
            len(
                pools["transverse"]
            )
        ),
        "top_candidates": [
            asdict(row)
            for row in ranked[:10]
        ],
        "ground_adjacent_pair": (
            asdict(rows[0])
            if rows
            else None
        ),
    }

    return rows, meta


def save_rows_csv(
    rows: list[CandidateRow],
    path: Path,
) -> None:
    fields = list(
        CandidateRow.__dataclass_fields__.keys()
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
                asdict(row)
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5 v50_7 frozen Phase-2/3 molecular method bridge."
        )
    )

    parser.add_argument(
        "--molecule",
        choices=("h4", "lih"),
        default="h4",
    )

    v1_root = (
        Path(__file__).resolve().parent.parent
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

    molecule = args.molecule.lower()

    geometries = (
        H4_GEOMETRIES
        if molecule == "h4"
        else LIH_GEOMETRIES
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("SOFT SPACES — PHASE 5 v50_7 FROZEN PHASE-2/3 METHOD BRIDGE")
    print("=" * 78)
    print(f"Molecule       : {molecule.upper()}")
    print("Basis          : STO-3G")
    print(f"Geometries     : {geometries}")
    print(f"Seeds          : {len(SEEDS)}")
    print(f"eps_neighbor   : {EPS_NEIGHBOR}")
    print(f"energy_reg     : {ENERGY_REG}")
    print(f"local radius   : +/-{LOCAL_RADIUS}")
    print(f"local step     : {LOCAL_STEP}")
    print("-" * 78)
    print(
        "Frozen law: DeltaC_f = C_REAL - C_NULL; "
        "S=min(dephasing,transverse); prominence=S-local median."
    )
    print("-" * 78)

    all_rows: list[CandidateRow] = []
    geometry_meta: dict[str, Any] = {}

    for gi, spacing in enumerate(
        geometries,
        start=1,
    ):
        print(
            f"[{gi}/{len(geometries)}] "
            f"{molecule.upper()} geometry={spacing:.2f} Å",
            flush=True,
        )

        rows, meta = run_geometry(
            molecule,
            spacing,
        )

        all_rows.extend(rows)
        geometry_meta[str(spacing)] = meta

        ranked = meta[
            "top_candidates"
        ]

        print(
            f"    sector={meta['sector_dimension']} | "
            f"eligible pairs={meta['eligible_adjacent_pairs']}/"
            f"{meta['total_adjacent_pairs']} | "
            f"zero transverse terms="
            f"{meta['zero_projected_transverse_terms']}/"
            f"{meta['total_transverse_terms']}"
        )

        if ranked:
            top = ranked[0]

            print(
                f"    top pair k={top['eigenpair_index']} | "
                f"prominence={top['prominence']:+.6f} | "
                f"robust ΔC={top['robust_delta_c']:+.6f} | "
                f"robust+={top['robust_positive_rate']:.3f}"
            )
        else:
            print(
                "    no finite local-prominence candidate "
                "under frozen eps_neighbor."
            )

        gp = meta[
            "ground_adjacent_pair"
        ]

        if gp is not None:
            print(
                f"    ground pair k=0 | "
                f"eligible={gp['eligible']} | "
                f"gap={gp['gap_hartree']:.6e} | "
                f"prominence={gp['prominence']:+.6f}"
            )

    eligible_rows = [
        row
        for row in all_rows
        if row.eligible
        and np.isfinite(
            row.prominence
        )
    ]

    positive_prominence = [
        row
        for row in eligible_rows
        if row.prominence > 0.0
    ]

    geometry_positive = {}

    for spacing in geometries:
        rows_g = [
            row
            for row in eligible_rows
            if row.geometry_angstrom
            == float(spacing)
        ]

        geometry_positive[
            str(spacing)
        ] = {
            "finite_prominence_candidates": len(
                rows_g
            ),
            "positive_prominence_candidates": sum(
                row.prominence > 0.0
                for row in rows_g
            ),
            "max_prominence": (
                max(
                    (
                        row.prominence
                        for row in rows_g
                    ),
                    default=float("nan"),
                )
            ),
            "median_prominence": (
                finite_median(
                    [
                        row.prominence
                        for row in rows_g
                    ]
                )
            ),
        }

    stem = (
        f"phase5_v50_7_{molecule}_phase23_bridge"
    )

    csv_path = (
        output_dir
        / f"{stem}_candidates.csv"
    )

    json_path = (
        output_dir
        / f"{stem}_summary.json"
    )

    hash_path = (
        output_dir
        / f"{stem}_summary.sha256"
    )

    save_rows_csv(
        all_rows,
        csv_path,
    )

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
        "molecule": molecule.upper(),
        "method_bridge": {
            "phase5_proxy_used_as_softspaces_definition": False,
            "frozen_phase23_c_matrix": (
                "||sum_m A_m||_F / sum_m ||A_m||_F"
            ),
            "delta_c": (
                "C_REAL - C_NULL"
            ),
            "robust_cross_family_score": (
                "min(median DeltaC_dephasing, "
                "median DeltaC_transverse)"
            ),
            "prominence": (
                "S(k) - median nearby S(k')"
            ),
            "null": (
                "same molecular spectrum, seeded Haar-random eigenbasis"
            ),
            "candidate_object": (
                "adjacent molecular eigenstate pair, not determinant"
            ),
        },
        "frozen_settings": {
            "seeds": list(SEEDS),
            "eps_neighbor": EPS_NEIGHBOR,
            "energy_reg": ENERGY_REG,
            "local_radius": LOCAL_RADIUS,
            "local_step": LOCAL_STEP,
        },
        "geometry_details": (
            geometry_meta
        ),
        "geometry_positive_summary": (
            geometry_positive
        ),
        "global_summary": {
            "finite_prominence_candidates": len(
                eligible_rows
            ),
            "positive_prominence_candidates": len(
                positive_prominence
            ),
            "positive_fraction": (
                len(positive_prominence)
                / len(eligible_rows)
                if eligible_rows
                else None
            ),
        },
        "interpretation_boundary": {
            "determinant_selection_claim": False,
            "en2_superiority_claim": False,
            "purpose": (
                "Test whether the original Phase-2/3 "
                "matrix-cancellation/prominence mechanism "
                "transfers to molecular eigenstate subspaces."
            ),
            "important_adaptation": (
                "Pauli Z/ZZ and X/XX perturbations are projected into "
                "the fixed-particle determinant sector. Single-X terms "
                "therefore vanish; this is reported and must be considered "
                "when interpreting transverse-family transfer."
            ),
        },
    }

    json_path.write_text(
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

    hash_path.write_text(
        digest + "\n",
        encoding="ascii",
    )

    print("-" * 78)
    print("v50_7 COMPLETE")
    print("-" * 78)
    print(f"Candidates CSV : {csv_path}")
    print(f"Summary JSON   : {json_path}")
    print(f"SHA-256        : {digest}")
    print("-" * 78)
    print(
        f"Finite prominence candidates  : {len(eligible_rows)}"
    )
    print(
        f"Positive prominence candidates: {len(positive_prominence)}"
    )

    if eligible_rows:
        print(
            f"Positive fraction              : "
            f"{len(positive_prominence)/len(eligible_rows):.3f}"
        )

    print("-" * 78)
    print(
        "This is the Phase-2/3 mechanism bridge. "
        "It is NOT the v50_2-v50_6 EN2-like determinant proxy."
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
