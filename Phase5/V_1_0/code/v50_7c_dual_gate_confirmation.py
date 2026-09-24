#!/usr/bin/env python3
"""
Soft Spaces Phase 5 — v50_7c dual-gate confirmation

Purpose
-------
Apply a stricter molecular confirmation rule to the original Phase-2/3
Soft-Spaces mechanism.

A candidate adjacent-eigenstate pair k is accepted only if BOTH are true:

1) Local prominence is positive in at least 75% of valid independent seeds.

2) Robust REAL-vs-NULL cancellation advantage is positive in at least 75%
   of valid independent seeds, where

       robust_delta_c(k, seed)
           = min(
               DeltaC_dephasing(k, seed),
               DeltaC_transverse(k, seed)
             )

This prevents a candidate from passing merely because it is better than its
local neighbors while still failing to beat NULL in absolute terms.

Frozen Phase-2/3 quantities
---------------------------
C_matrix
    = ||sum_m A_m||_F / sum_m ||A_m||_F

DeltaC_f
    = C_REAL,f - C_NULL,f

robust_delta_c
    = min(DeltaC_dephasing, DeltaC_transverse)

prominence
    = robust_delta_c(k)
      - median(local robust controls)

Dual PASS gate
--------------
prominence_positive_rate >= 0.75
AND
robust_positive_rate     >= 0.75

With 12 seeds, this normally means at least 9/12 positive units for BOTH.

Default benchmark
-----------------
H4 / STO-3G
fixed (N_alpha,N_beta)=(2,2) sector
geometries:
    0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00 Å

No EN2 is used in this script.
No Phase-5 determinant proxy is used.
No outcome-dependent retuning is performed.

Usage
-----
    python -X utf8 -u ./v50_7c_dual_gate_confirmation.py
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


SCRIPT_VERSION = "v50_7c"

SEEDS = tuple(range(25043000, 25043012))
GEOMETRIES = (0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)

LOCAL_RADIUS = 10
LOCAL_STEP = 2
EPS_NEIGHBOR = 0.05
ENERGY_REG = 1e-3

PROMINENCE_THRESHOLD = 0.75
ROBUST_THRESHOLD = 0.75


@dataclass(frozen=True)
class SeedRow:
    geometry_angstrom: float
    seed: int
    eigenpair_index: int
    gap_hartree: float
    delta_c_dephasing: float
    delta_c_transverse: float
    robust_delta_c: float
    local_background: float
    prominence: float
    positive_robust: bool
    positive_prominence: bool


@dataclass(frozen=True)
class CandidateSummary:
    geometry_angstrom: float
    eigenpair_index: int
    gap_hartree: float
    valid_seed_units: int
    robust_positive_units: int
    robust_positive_rate: float
    prominence_positive_units: int
    prominence_positive_rate: float
    median_robust_delta_c: float
    median_prominence: float
    min_prominence: float
    max_prominence: float
    dephasing_positive_rate: float
    transverse_positive_rate: float
    status: str


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


def alpha_beta_counts(index: int, n_spatial: int) -> tuple[int, int]:
    mask = (1 << n_spatial) - 1
    a = index & mask
    b = (index >> n_spatial) & mask
    return popcount(a), popcount(b)


def fixed_sector_basis(
    n_spatial: int,
    n_alpha: int,
    n_beta: int,
) -> np.ndarray:
    n_qubits = 2 * n_spatial
    out = []

    for idx in range(1 << n_qubits):
        na, nb = alpha_beta_counts(idx, n_spatial)
        if na == n_alpha and nb == n_beta:
            out.append(idx)

    expected = math.comb(n_spatial, n_alpha) * math.comb(n_spatial, n_beta)

    if len(out) != expected:
        raise RuntimeError(
            f"Sector dimension mismatch: expected {expected}, got {len(out)}"
        )

    return np.asarray(out, dtype=np.int64)


def build_h4(spacing: float) -> dict[str, Any]:
    try:
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        from qiskit_nature.units import DistanceUnit
    except Exception as exc:
        raise RuntimeError(
            "Qiskit Nature / PySCF imports failed. "
            "Run in the Phase-5 WSL environment."
        ) from exc

    zs = (-1.5 * spacing, -0.5 * spacing, 0.5 * spacing, 1.5 * spacing)
    atom = "; ".join(f"H 0.0 0.0 {z:.12f}" for z in zs)

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

    if (n_spatial, n_alpha, n_beta) != (4, 2, 2):
        raise RuntimeError(
            f"Unexpected H4 metadata: spatial={n_spatial}, "
            f"Nalpha={n_alpha}, Nbeta={n_beta}"
        )

    basis = fixed_sector_basis(n_spatial, n_alpha, n_beta)

    mapper = JordanWignerMapper()
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())

    full_sparse = qubit_op.to_matrix(sparse=True)
    sector_sparse = full_sparse[basis, :][:, basis]
    h = np.asarray(sector_sparse.toarray(), dtype=np.complex128)

    herm_error = float(np.max(np.abs(h - h.conjugate().T)))
    if herm_error > 1e-10:
        raise RuntimeError(f"Hamiltonian non-Hermitian: {herm_error:.3e}")

    evals, evecs = np.linalg.eigh(h)

    return {
        "basis": basis,
        "n_qubits": n_qubits,
        "evals": np.asarray(evals, dtype=float),
        "evecs": np.asarray(evecs, dtype=np.complex128),
    }


def haar_unitary(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = (
        rng.normal(size=(d, d))
        + 1j * rng.normal(size=(d, d))
    ) / math.sqrt(2.0)

    q, r = np.linalg.qr(z)
    diag = np.diag(r)
    phases = np.ones_like(diag, dtype=np.complex128)

    nz = np.abs(diag) > 0
    phases[nz] = diag[nz] / np.abs(diag[nz])

    return np.asarray(q * phases.conjugate(), dtype=np.complex128)


def projected_z(basis: np.ndarray, q: int) -> np.ndarray:
    diag = np.array(
        [-1.0 if ((int(x) >> q) & 1) else 1.0 for x in basis],
        dtype=float,
    )
    return np.diag(diag).astype(np.complex128)


def projected_zz(
    basis: np.ndarray,
    q1: int,
    q2: int,
) -> np.ndarray:
    vals = []
    for x in basis:
        z1 = -1.0 if ((int(x) >> q1) & 1) else 1.0
        z2 = -1.0 if ((int(x) >> q2) & 1) else 1.0
        vals.append(z1 * z2)
    return np.diag(np.asarray(vals, dtype=float)).astype(np.complex128)


def projected_flip(
    basis: np.ndarray,
    flip_mask: int,
) -> np.ndarray:
    lookup = {int(full): local for local, full in enumerate(basis)}
    n = len(basis)
    out = np.zeros((n, n), dtype=np.complex128)

    for col, full in enumerate(basis):
        target = int(full) ^ int(flip_mask)
        row = lookup.get(target)
        if row is not None:
            out[row, col] = 1.0

    return out


def perturbation_pools(
    basis: np.ndarray,
    n_qubits: int,
) -> dict[str, list[np.ndarray]]:
    dephasing = []
    transverse = []

    for q in range(n_qubits):
        dephasing.append(projected_z(basis, q))
        transverse.append(projected_flip(basis, 1 << q))

    for q in range(n_qubits - 1):
        dephasing.append(projected_zz(basis, q, q + 1))
        transverse.append(
            projected_flip(
                basis,
                (1 << q) | (1 << (q + 1)),
            )
        )

    return {
        "dephasing": dephasing,
        "transverse": transverse,
    }


def random_family_perturbation(
    pool: list[np.ndarray],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=len(pool))
    d = pool[0].shape[0]

    out = np.zeros((d, d), dtype=np.complex128)

    for coeff, term in zip(coeffs, pool):
        out += float(coeff) * term

    norm = float(np.linalg.norm(out, ord="fro"))
    if norm > 0.0:
        out /= norm

    return 0.5 * (out + out.conjugate().T)


def c_matrix(
    evals: np.ndarray,
    bmat: np.ndarray,
    i: int,
) -> float | None:
    d = len(evals)
    j = i + 1

    if j >= d:
        return None

    gap = abs(float(evals[j] - evals[i]))
    if gap >= EPS_NEIGHBOR:
        return None

    qmask = np.ones(d, dtype=bool)
    qmask[[i, j]] = False

    wq = np.asarray(
        bmat[qmask, :][:, [i, j]],
        dtype=np.complex128,
    )

    eq = np.asarray(evals[qmask], dtype=float)
    eref = 0.5 * (float(evals[i]) + float(evals[j]))
    de = eref - eq

    reg = max(float(ENERGY_REG), 1e-15)
    inv1 = de / (de * de + reg * reg)

    sum_a = np.zeros((2, 2), dtype=np.complex128)
    sum_norm_a = 0.0

    for r in range(wq.shape[0]):
        v = np.asarray(wq[r, :], dtype=np.complex128).reshape(2, 1)
        a = float(inv1[r]) * (v.conjugate() @ v.T)
        a = 0.5 * (a + a.conjugate().T)

        sum_a += a
        sum_norm_a += float(np.linalg.norm(a, ord="fro"))

    if sum_norm_a <= 0.0 or not np.isfinite(sum_norm_a):
        return None

    return float(
        np.linalg.norm(sum_a, ord="fro")
        / sum_norm_a
    )


def finite_median(values: list[float]) -> float:
    x = np.asarray(
        [v for v in values if np.isfinite(v)],
        dtype=float,
    )
    if len(x) == 0:
        return float("nan")
    return float(np.median(x))


def positive_rate(values: list[float]) -> float:
    x = np.asarray(
        [v for v in values if np.isfinite(v)],
        dtype=float,
    )
    if len(x) == 0:
        return float("nan")
    return float(np.mean(x > 0.0))


def run_geometry(
    spacing: float,
) -> tuple[list[SeedRow], list[CandidateSummary], dict[str, Any]]:
    p = build_h4(spacing)

    evals = p["evals"]
    v_real = p["evecs"]
    basis = p["basis"]
    d = len(evals)

    pools = perturbation_pools(basis, p["n_qubits"])

    zero_transverse_terms = sum(
        np.linalg.norm(term, ord="fro") == 0.0
        for term in pools["transverse"]
    )

    raw: dict[tuple[int, str, int], float] = {}

    for seed in SEEDS:
        v_null = haar_unitary(d, seed + 99_000_000)

        for family_idx, family in enumerate(("dephasing", "transverse")):
            dh = random_family_perturbation(
                pools[family],
                seed + 1_000_000 * (family_idx + 1),
            )

            b_real = v_real.conjugate().T @ dh @ v_real
            b_null = v_null.conjugate().T @ dh @ v_null

            for k in range(d - 1):
                cr = c_matrix(evals, b_real, k)
                cn = c_matrix(evals, b_null, k)

                if cr is None or cn is None:
                    continue

                raw[(seed, family, k)] = float(cr - cn)

    robust: dict[tuple[int, int], float] = {}

    for seed in SEEDS:
        for k in range(d - 1):
            a = raw.get((seed, "dephasing", k))
            t = raw.get((seed, "transverse", k))

            if (
                a is not None
                and t is not None
                and np.isfinite(a)
                and np.isfinite(t)
            ):
                robust[(seed, k)] = min(float(a), float(t))

    seed_rows: list[SeedRow] = []
    by_candidate: dict[int, list[SeedRow]] = {
        k: [] for k in range(d - 1)
    }

    for seed in SEEDS:
        valid_k = [
            k
            for k in range(d - 1)
            if (seed, k) in robust
        ]

        for k in valid_k:
            local_controls = [
                kk
                for kk in valid_k
                if kk != k
                and abs(kk - k) <= LOCAL_RADIUS
                and abs(kk - k) % LOCAL_STEP == 0
            ]

            background = finite_median(
                [robust[(seed, kk)] for kk in local_controls]
            )

            if not np.isfinite(background):
                continue

            rdc = robust[(seed, k)]
            prominence = rdc - background

            row = SeedRow(
                geometry_angstrom=float(spacing),
                seed=int(seed),
                eigenpair_index=int(k),
                gap_hartree=float(evals[k + 1] - evals[k]),
                delta_c_dephasing=float(raw[(seed, "dephasing", k)]),
                delta_c_transverse=float(raw[(seed, "transverse", k)]),
                robust_delta_c=float(rdc),
                local_background=float(background),
                prominence=float(prominence),
                positive_robust=bool(rdc > 0.0),
                positive_prominence=bool(prominence > 0.0),
            )

            seed_rows.append(row)
            by_candidate[k].append(row)

    summaries: list[CandidateSummary] = []

    for k in range(d - 1):
        rows = by_candidate[k]

        if not rows:
            continue

        robust_vals = [r.robust_delta_c for r in rows]
        prominence_vals = [r.prominence for r in rows]
        deph_vals = [r.delta_c_dephasing for r in rows]
        trans_vals = [r.delta_c_transverse for r in rows]

        n_valid = len(rows)
        robust_pos = sum(r.positive_robust for r in rows)
        prominence_pos = sum(r.positive_prominence for r in rows)

        robust_rate = robust_pos / n_valid
        prominence_rate = prominence_pos / n_valid

        status = (
            "PASS"
            if (
                robust_rate >= ROBUST_THRESHOLD
                and prominence_rate >= PROMINENCE_THRESHOLD
            )
            else "FAIL"
        )

        summaries.append(
            CandidateSummary(
                geometry_angstrom=float(spacing),
                eigenpair_index=int(k),
                gap_hartree=float(evals[k + 1] - evals[k]),
                valid_seed_units=int(n_valid),
                robust_positive_units=int(robust_pos),
                robust_positive_rate=float(robust_rate),
                prominence_positive_units=int(prominence_pos),
                prominence_positive_rate=float(prominence_rate),
                median_robust_delta_c=float(finite_median(robust_vals)),
                median_prominence=float(finite_median(prominence_vals)),
                min_prominence=float(np.min(prominence_vals)),
                max_prominence=float(np.max(prominence_vals)),
                dephasing_positive_rate=float(positive_rate(deph_vals)),
                transverse_positive_rate=float(positive_rate(trans_vals)),
                status=status,
            )
        )

    summaries.sort(
        key=lambda x: (
            x.status == "PASS",
            min(
                x.robust_positive_rate,
                x.prominence_positive_rate,
            ),
            x.median_prominence,
            x.median_robust_delta_c,
        ),
        reverse=True,
    )

    meta = {
        "geometry_angstrom": float(spacing),
        "sector_dimension": int(d),
        "seed_count": len(SEEDS),
        "zero_projected_transverse_terms": int(zero_transverse_terms),
        "total_transverse_terms": int(len(pools["transverse"])),
        "passing_candidates": [
            asdict(r)
            for r in summaries
            if r.status == "PASS"
        ],
        "top_candidates": [
            asdict(r)
            for r in summaries[:10]
        ],
    }

    return seed_rows, summaries, meta


def save_dataclass_csv(rows: list[Any], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"No rows to save: {path}")

    fields = list(rows[0].__dataclass_fields__.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 v50_7c dual-gate confirmation."
    )

    v1_root = Path(__file__).resolve().parent.parent

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=v1_root / "results" / "phase23_method_bridge",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SOFT SPACES — PHASE 5 v50_7c DUAL-GATE CONFIRMATION")
    print("=" * 78)
    print("Molecule       : H4")
    print("Basis          : STO-3G")
    print(f"Geometries     : {GEOMETRIES}")
    print(f"Seeds          : {SEEDS}")
    print(
        f"Gate 1         : prominence positive-rate >= "
        f"{PROMINENCE_THRESHOLD:.2f}"
    )
    print(
        f"Gate 2         : robust ΔC positive-rate >= "
        f"{ROBUST_THRESHOLD:.2f}"
    )
    print(f"eps_neighbor   : {EPS_NEIGHBOR}")
    print(f"energy_reg     : {ENERGY_REG}")
    print(f"local radius   : +/-{LOCAL_RADIUS}")
    print(f"local step     : {LOCAL_STEP}")
    print("-" * 78)

    all_seed_rows = []
    all_summary_rows = []
    geometry_meta = {}

    for gi, spacing in enumerate(GEOMETRIES, start=1):
        print(
            f"[{gi}/{len(GEOMETRIES)}] H4 geometry={spacing:.2f} Å",
            flush=True,
        )

        seed_rows, summaries, meta = run_geometry(spacing)

        all_seed_rows.extend(seed_rows)
        all_summary_rows.extend(summaries)
        geometry_meta[str(spacing)] = meta

        passing = [r for r in summaries if r.status == "PASS"]

        print(
            f"    finite candidate rows : {len(summaries)}"
        )
        print(
            f"    DUAL PASS candidates  : {len(passing)}"
        )

        if summaries:
            top = summaries[0]
            print(
                f"    top k={top.eigenpair_index} | "
                f"prom={top.prominence_positive_units}/"
                f"{top.valid_seed_units} "
                f"({top.prominence_positive_rate:.3f}) | "
                f"robust={top.robust_positive_units}/"
                f"{top.valid_seed_units} "
                f"({top.robust_positive_rate:.3f}) | "
                f"medProm={top.median_prominence:+.6f} | "
                f"medΔC={top.median_robust_delta_c:+.6f} | "
                f"{top.status}"
            )

    seed_csv = (
        output_dir
        / "phase5_v50_7c_seedwise_dual_gate.csv"
    )

    candidate_csv = (
        output_dir
        / "phase5_v50_7c_candidate_summary.csv"
    )

    summary_json = (
        output_dir
        / "phase5_v50_7c_summary.json"
    )

    hash_file = (
        output_dir
        / "phase5_v50_7c_summary.sha256"
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
        r
        for r in all_summary_rows
        if r.status == "PASS"
    ]

    geometries_with_pass = sorted(
        {
            r.geometry_angstrom
            for r in passing_rows
        }
    )

    summary = {
        "_metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script_version": SCRIPT_VERSION,
            "python_version": sys.version.split()[0],
        },
        "benchmark": {
            "molecule": "H4",
            "basis": "STO-3G",
            "geometries": list(GEOMETRIES),
            "fixed_sector": "(N_alpha,N_beta)=(2,2)",
        },
        "dual_gate": {
            "prominence_positive_rate_threshold": PROMINENCE_THRESHOLD,
            "robust_positive_rate_threshold": ROBUST_THRESHOLD,
            "rule": (
                "PASS iff prominence positive-rate >=0.75 "
                "AND robust DeltaC positive-rate >=0.75"
            ),
            "seeds": list(SEEDS),
        },
        "geometry_details": geometry_meta,
        "global_confirmation": {
            "candidate_geometry_rows": len(all_summary_rows),
            "passing_candidate_geometry_rows": len(passing_rows),
            "geometries_with_at_least_one_dual_pass": geometries_with_pass,
            "n_geometries_with_at_least_one_dual_pass": len(geometries_with_pass),
            "dual_pass_rows": [
                asdict(r)
                for r in passing_rows
            ],
        },
        "interpretation_boundary": {
            "en2_used": False,
            "phase5_proxy_used": False,
            "determinant_selection_claim": False,
            "purpose": (
                "Require both reproducible local prominence and reproducible "
                "absolute REAL>NULL robust cancellation advantage."
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

    digest = sha256_hex(summary)
    hash_file.write_text(digest + "\n", encoding="ascii")

    print("-" * 78)
    print("v50_7c COMPLETE")
    print("-" * 78)
    print(f"Seed CSV      : {seed_csv}")
    print(f"Candidate CSV : {candidate_csv}")
    print(f"Summary JSON  : {summary_json}")
    print(f"SHA-256       : {digest}")
    print("-" * 78)
    print(
        f"DUAL PASS candidate-geometry rows : {len(passing_rows)}"
    )
    print(
        f"Geometries with >=1 DUAL PASS     : "
        f"{len(geometries_with_pass)}/{len(GEOMETRIES)}"
    )
    print("-" * 78)
    print(
        "A DUAL PASS requires both reproducible local prominence "
        "and reproducible positive robust REAL-vs-NULL DeltaC."
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
