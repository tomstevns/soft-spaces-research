#!/usr/bin/env python3
"""
Soft Spaces Phase 5 — v50_11 LiH independent replication

Replicates H4 v50_10 T1–T4 on LiH/STO-3G with the same frozen thresholds.
No retuning.

CLEAR PASS requires all:
- pooled Soft-Spaces AUC >= 0.65
- pooled (SS AUC - EN2 AUC) >= 0.10
- median per-geometry AUC difference > 0
- at least 5 geometries with positive AUC difference

LiH geometries:
1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0, 4.0 Å

Fixed sector:
(N_alpha,N_beta)=(2,2)

The script:
1) reconstructs the v50_7c-style DUAL-PASS Soft-Spaces candidates on LiH,
2) runs held-out dephasing, transverse, mixed and geometry-gap sensitivity,
3) compares the frozen Soft-Spaces score against the same EN2 eigenpair bridge.

If LiH fails, do not retune post hoc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_VERSION = "v50_11"

GEOMETRIES = (1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0, 4.0)
CONFIRM_SEEDS = tuple(range(25043000, 25043012))
HELDOUT_SEEDS = tuple(range(35043000, 35043024))

LOCAL_RADIUS = 10
LOCAL_STEP = 2
EPS_NEIGHBOR = 0.05
ENERGY_REG = 1e-3

PROM_THRESHOLD = 0.75
ROBUST_THRESHOLD = 0.75

PERTURBATION_STRENGTH = 0.01
GEOMETRY_DELTA = 0.025
EN2_EPS = 1e-12

CLEAR_SS_AUC = 0.65
CLEAR_DELTA_AUC = 0.10
CLEAR_POSITIVE_GEOMETRIES = 5


def sha256_obj(obj: Any) -> str:
    raw = json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def popcount(x: int) -> int:
    return bin(int(x)).count("1")


def fixed_sector_basis(n_spatial: int, n_alpha: int, n_beta: int) -> np.ndarray:
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


def build_lih(distance: float) -> dict[str, Any]:
    try:
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        from qiskit_nature.units import DistanceUnit
    except Exception as exc:
        raise RuntimeError(
            "Qiskit Nature / PySCF unavailable. Run in Phase-5 WSL environment."
        ) from exc

    atom = f"Li 0.0 0.0 0.0; H 0.0 0.0 {distance:.12f}"

    problem = PySCFDriver(
        atom=atom,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    ).run()

    ns = int(problem.num_spatial_orbitals)
    na = int(problem.num_alpha)
    nb = int(problem.num_beta)

    if (na, nb) != (2, 2):
        raise RuntimeError(
            f"Unexpected LiH sector: Nalpha={na}, Nbeta={nb}"
        )

    basis = fixed_sector_basis(ns, na, nb)

    op = JordanWignerMapper().map(problem.hamiltonian.second_q_op())
    full = op.to_matrix(sparse=True)
    h = np.asarray(full[basis, :][:, basis].toarray(), dtype=np.complex128)

    if float(np.max(np.abs(h - h.conjugate().T))) > 1e-10:
        raise RuntimeError("Hamiltonian is not Hermitian.")

    evals, evecs = np.linalg.eigh(h)

    return {
        "basis": basis,
        "n_qubits": 2 * ns,
        "h": h,
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
    diag = np.asarray(
        [-1.0 if ((int(x) >> q) & 1) else 1.0 for x in basis],
        dtype=float,
    )
    return np.diag(diag).astype(np.complex128)


def projected_zz(basis: np.ndarray, q1: int, q2: int) -> np.ndarray:
    vals = []
    for x in basis:
        z1 = -1.0 if ((int(x) >> q1) & 1) else 1.0
        z2 = -1.0 if ((int(x) >> q2) & 1) else 1.0
        vals.append(z1 * z2)
    return np.diag(np.asarray(vals, dtype=float)).astype(np.complex128)


def projected_flip(basis: np.ndarray, flip_mask: int) -> np.ndarray:
    lookup = {int(full): local for local, full in enumerate(basis)}
    d = len(basis)
    out = np.zeros((d, d), dtype=np.complex128)

    for col, full in enumerate(basis):
        row = lookup.get(int(full) ^ int(flip_mask))
        if row is not None:
            out[row, col] = 1.0

    return out


def perturbation_pools(basis: np.ndarray, n_qubits: int) -> dict[str, list[np.ndarray]]:
    deph = []
    trans = []

    for q in range(n_qubits):
        deph.append(projected_z(basis, q))
        trans.append(projected_flip(basis, 1 << q))

    for q in range(n_qubits - 1):
        deph.append(projected_zz(basis, q, q + 1))
        trans.append(projected_flip(
            basis,
            (1 << q) | (1 << (q + 1)),
        ))

    return {"dephasing": deph, "transverse": trans}


def random_combo(pool: list[np.ndarray], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=len(pool))
    out = np.zeros_like(pool[0], dtype=np.complex128)

    for c, a in zip(coeffs, pool):
        out += float(c) * a

    norm = float(np.linalg.norm(out, ord="fro"))
    if norm > 0:
        out /= norm

    return 0.5 * (out + out.conjugate().T)


def c_matrix(evals: np.ndarray, bmat: np.ndarray, i: int) -> float | None:
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
    inv = de / (de * de + reg * reg)

    total = np.zeros((2, 2), dtype=np.complex128)
    norm_sum = 0.0

    for r in range(wq.shape[0]):
        v = np.asarray(wq[r, :], dtype=np.complex128).reshape(2, 1)
        a = float(inv[r]) * (v.conjugate() @ v.T)
        a = 0.5 * (a + a.conjugate().T)
        total += a
        norm_sum += float(np.linalg.norm(a, ord="fro"))

    if norm_sum <= 0.0 or not np.isfinite(norm_sum):
        return None

    return float(np.linalg.norm(total, ord="fro") / norm_sum)


def build_candidates(system: dict[str, Any]) -> list[dict[str, Any]]:
    evals = system["evals"]
    v_real = system["evecs"]
    d = len(evals)

    pools = perturbation_pools(system["basis"], system["n_qubits"])
    raw: dict[tuple[int, str, int], float] = {}

    for seed in CONFIRM_SEEDS:
        v_null = haar_unitary(d, seed + 99_000_000)

        for fi, family in enumerate(("dephasing", "transverse")):
            dh = random_combo(
                pools[family],
                seed + 1_000_000 * (fi + 1),
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

    for seed in CONFIRM_SEEDS:
        for k in range(d - 1):
            a = raw.get((seed, "dephasing", k))
            t = raw.get((seed, "transverse", k))

            if a is not None and t is not None:
                robust[(seed, k)] = min(float(a), float(t))

    candidates = []

    for k in range(d - 1):
        prom_vals = []
        robust_vals = []

        for seed in CONFIRM_SEEDS:
            current = robust.get((seed, k))
            if current is None:
                continue

            valid = [
                kk for kk in range(d - 1)
                if (seed, kk) in robust
            ]

            controls = [
                kk for kk in valid
                if kk != k
                and abs(kk - k) <= LOCAL_RADIUS
                and abs(kk - k) % LOCAL_STEP == 0
            ]

            if not controls:
                continue

            background = float(np.median(
                [robust[(seed, kk)] for kk in controls]
            ))

            prom_vals.append(float(current - background))
            robust_vals.append(float(current))

        if not prom_vals:
            continue

        prom_rate = float(np.mean(np.asarray(prom_vals) > 0.0))
        robust_rate = float(np.mean(np.asarray(robust_vals) > 0.0))

        candidates.append({
            "k": int(k),
            "gap": float(evals[k + 1] - evals[k]),
            "prominence_rate": prom_rate,
            "robust_rate": robust_rate,
            "ss_score": min(prom_rate, robust_rate),
            "status": (
                "PASS"
                if (
                    prom_rate >= PROM_THRESHOLD
                    and robust_rate >= ROBUST_THRESHOLD
                )
                else "FAIL"
            ),
        })

    return candidates


def en2_scores(h: np.ndarray) -> np.ndarray:
    diag = np.real(np.diag(h)).astype(float)
    r = int(np.argmin(diag))
    hrr = float(diag[r])

    score = np.zeros(len(diag), dtype=float)

    for i in range(len(diag)):
        if i == r:
            continue
        score[i] = float(abs(h[i, r]) ** 2) / (
            abs(float(diag[i] - hrr)) + EN2_EPS
        )

    return score


def pair_prob(evecs: np.ndarray, k: int) -> np.ndarray:
    p = 0.5 * (
        np.abs(evecs[:, k]) ** 2
        + np.abs(evecs[:, k + 1]) ** 2
    )
    return p / float(np.sum(p))


def en2_pair_score(det_scores: np.ndarray, evecs: np.ndarray, k: int) -> float:
    return float(np.sum(pair_prob(evecs, k) * det_scores))


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg
        i = j

    return ranks


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))

    if n_pos == 0 or n_neg == 0:
        return None

    ranks = average_ranks(scores)
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0

    return float(u / (n_pos * n_neg))


def top_quartile_labels(response: np.ndarray) -> np.ndarray:
    response = np.asarray(response, dtype=float)
    threshold = float(np.quantile(response, 0.75))
    labels = (response >= threshold).astype(int)

    if np.all(labels == 1):
        order = np.argsort(response)
        labels[:] = 0
        labels[order[-max(1, len(response) // 4):]] = 1

    return labels


def pair_projector(evecs: np.ndarray, k: int) -> np.ndarray:
    u = evecs[:, [k, k + 1]]
    return u @ u.conjugate().T


def heldout_response(
    system: dict[str, Any],
    candidates: list[dict[str, Any]],
    family: str,
) -> np.ndarray:
    pools = perturbation_pools(system["basis"], system["n_qubits"])

    # Held-out generation excludes zero-projected terms.
    for fam in pools:
        pools[fam] = [
            a for a in pools[fam]
            if np.linalg.norm(a, ord="fro") > 0.0
        ]

    out = np.zeros(len(candidates), dtype=float)

    for seed in HELDOUT_SEEDS:
        if family == "mixed":
            d1 = random_combo(pools["dephasing"], seed + 101)
            d2 = random_combo(pools["transverse"], seed + 202)
            dh = d1 + d2
            nrm = float(np.linalg.norm(dh, ord="fro"))
            if nrm > 0:
                dh /= nrm
        else:
            offset = 101 if family == "dephasing" else 202
            dh = random_combo(pools[family], seed + offset)

        _, pert_vecs = np.linalg.eigh(
            system["h"] + PERTURBATION_STRENGTH * dh
        )

        for ci, c in enumerate(candidates):
            p0 = pair_projector(system["evecs"], c["k"])
            p1 = pair_projector(pert_vecs, c["k"])

            overlap = float(np.real(np.trace(p0 @ p1)) / 2.0)
            out[ci] += max(0.0, 1.0 - overlap)

    out /= len(HELDOUT_SEEDS)
    return out


def geometry_response(
    g: float,
    candidates: list[dict[str, Any]],
) -> np.ndarray:
    minus = build_lih(g - GEOMETRY_DELTA)
    plus = build_lih(g + GEOMETRY_DELTA)

    gaps_minus = np.diff(minus["evals"])
    gaps_plus = np.diff(plus["evals"])

    return np.asarray(
        [
            abs(float(gaps_plus[c["k"]] - gaps_minus[c["k"]]))
            for c in candidates
        ],
        dtype=float,
    )


def evaluate_stage(
    name: str,
    responses: dict[float, np.ndarray],
    candidates_by_geometry: dict[float, list[dict[str, Any]]],
    systems: dict[float, dict[str, Any]],
) -> dict[str, Any]:
    pooled_labels = []
    pooled_ss = []
    pooled_en2 = []
    per_geometry = []

    for g in GEOMETRIES:
        candidates = candidates_by_geometry[g]
        if len(candidates) < 4:
            continue

        labels = top_quartile_labels(responses[g])

        ss = np.asarray(
            [c["ss_score"] for c in candidates],
            dtype=float,
        )

        det_en2 = en2_scores(systems[g]["h"])

        en2 = np.asarray(
            [
                en2_pair_score(
                    det_en2,
                    systems[g]["evecs"],
                    c["k"],
                )
                for c in candidates
            ],
            dtype=float,
        )

        a_ss = auc(labels, ss)
        a_en2 = auc(labels, en2)

        if a_ss is None or a_en2 is None:
            continue

        delta = float(a_ss - a_en2)

        per_geometry.append({
            "geometry": g,
            "n_candidates": len(candidates),
            "softspaces_auc": float(a_ss),
            "en2_auc": float(a_en2),
            "delta_auc": delta,
        })

        pooled_labels.extend(labels.tolist())
        pooled_ss.extend(ss.tolist())
        pooled_en2.extend(en2.tolist())

    p_ss = auc(np.asarray(pooled_labels), np.asarray(pooled_ss))
    p_en2 = auc(np.asarray(pooled_labels), np.asarray(pooled_en2))

    if p_ss is None or p_en2 is None:
        delta_pooled = None
        median_delta = None
        positive_geometries = 0
        clear_pass = False
    else:
        delta_pooled = float(p_ss - p_en2)
        deltas = [r["delta_auc"] for r in per_geometry]
        median_delta = float(np.median(deltas)) if deltas else None
        positive_geometries = int(sum(d > 0.0 for d in deltas))

        clear_pass = bool(
            p_ss >= CLEAR_SS_AUC
            and delta_pooled >= CLEAR_DELTA_AUC
            and median_delta is not None
            and median_delta > 0.0
            and positive_geometries >= CLEAR_POSITIVE_GEOMETRIES
        )

    return {
        "name": name,
        "pooled_softspaces_auc": p_ss,
        "pooled_en2_auc": p_en2,
        "pooled_delta_auc": delta_pooled,
        "median_geometry_delta_auc": median_delta,
        "positive_delta_geometries": positive_geometries,
        "clear_pass": clear_pass,
        "per_geometry": per_geometry,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "v50_11_lih_replication",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SOFT SPACES PHASE 5 — v50_11 LiH INDEPENDENT REPLICATION")
    print("=" * 80)
    print("Molecule       : LiH / STO-3G")
    print(f"Geometries     : {GEOMETRIES}")
    print(f"Confirm seeds  : {CONFIRM_SEEDS[0]}..{CONFIRM_SEEDS[-1]}")
    print(f"Held-out seeds : {HELDOUT_SEEDS[0]}..{HELDOUT_SEEDS[-1]}")
    print(
        "CLEAR PASS     : SS AUC>=0.65, ΔAUC>=0.10, "
        "median geom ΔAUC>0, >=5 positive geometries"
    )
    print("-" * 80)

    systems = {}
    candidates_by_geometry = {}

    for i, g in enumerate(GEOMETRIES, start=1):
        print(f"[build {i}/8] LiH R={g:.2f} Å", flush=True)

        system = build_lih(g)
        systems[g] = system

        candidates = build_candidates(system)
        candidates_by_geometry[g] = candidates

        n_pass = int(sum(c["status"] == "PASS" for c in candidates))

        print(
            f"    sector dim={len(system['evals'])} | "
            f"eligible={len(candidates)} | DUAL PASS={n_pass}"
        )

    results = []

    for name, family in (
        ("T1 held-out dephasing projector sensitivity", "dephasing"),
        ("T2 held-out transverse projector sensitivity", "transverse"),
        ("T3 held-out mixed projector sensitivity", "mixed"),
    ):
        print("-" * 80)
        print(name)

        responses = {
            g: heldout_response(
                systems[g],
                candidates_by_geometry[g],
                family,
            )
            for g in GEOMETRIES
        }

        stage = evaluate_stage(
            name,
            responses,
            candidates_by_geometry,
            systems,
        )
        results.append(stage)

        print(f"  pooled SS AUC  = {stage['pooled_softspaces_auc']:.3f}")
        print(f"  pooled EN2 AUC = {stage['pooled_en2_auc']:.3f}")
        print(f"  ΔAUC SS-EN2    = {stage['pooled_delta_auc']:+.3f}")
        print(f"  + geometries   = {stage['positive_delta_geometries']}")
        print(
            "  RESULT         = "
            + ("CLEAR PASS" if stage["clear_pass"] else "FAIL")
        )

    print("-" * 80)
    print("T4 geometry-displacement gap sensitivity")

    geo_responses = {
        g: geometry_response(
            g,
            candidates_by_geometry[g],
        )
        for g in GEOMETRIES
    }

    stage = evaluate_stage(
        "T4 geometry-displacement gap sensitivity",
        geo_responses,
        candidates_by_geometry,
        systems,
    )
    results.append(stage)

    print(f"  pooled SS AUC  = {stage['pooled_softspaces_auc']:.3f}")
    print(f"  pooled EN2 AUC = {stage['pooled_en2_auc']:.3f}")
    print(f"  ΔAUC SS-EN2    = {stage['pooled_delta_auc']:+.3f}")
    print(f"  + geometries   = {stage['positive_delta_geometries']}")
    print(
        "  RESULT         = "
        + ("CLEAR PASS" if stage["clear_pass"] else "FAIL")
    )

    clear_stages = [r["name"] for r in results if r["clear_pass"]]

    summary = {
        "_metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script_version": SCRIPT_VERSION,
            "python_version": sys.version.split()[0],
        },
        "benchmark": {
            "molecule": "LiH",
            "basis": "STO-3G",
            "geometries": list(GEOMETRIES),
            "fixed_sector": "(N_alpha,N_beta)=(2,2)",
        },
        "frozen_rules": {
            "prominence_threshold": PROM_THRESHOLD,
            "robust_threshold": ROBUST_THRESHOLD,
            "clear_ss_auc": CLEAR_SS_AUC,
            "clear_delta_auc": CLEAR_DELTA_AUC,
            "positive_geometry_min": CLEAR_POSITIVE_GEOMETRIES,
            "perturbation_strength": PERTURBATION_STRENGTH,
            "geometry_delta": GEOMETRY_DELTA,
            "confirm_seeds": list(CONFIRM_SEEDS),
            "heldout_seeds": list(HELDOUT_SEEDS),
        },
        "candidate_summary": {
            str(g): {
                "eligible_candidates": len(candidates_by_geometry[g]),
                "dual_pass_candidates": int(
                    sum(
                        c["status"] == "PASS"
                        for c in candidates_by_geometry[g]
                    )
                ),
            }
            for g in GEOMETRIES
        },
        "results": results,
        "clear_pass_stages": clear_stages,
        "all_stages_failed": len(clear_stages) == 0,
        "interpretation_if_all_fail": (
            "H4 predictive tendency did not independently replicate on LiH "
            "under unchanged rules. Do not retune thresholds post hoc."
        ),
        "interpretation_if_any_pass": (
            "At least one preregistered predictive Soft-Spaces advantage over "
            "EN2 independently replicated on LiH under unchanged criteria."
        ),
    }

    json_path = outdir / "phase5_v50_11_lih_replication_summary.json"
    hash_path = outdir / "phase5_v50_11_lih_replication_summary.sha256"

    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    digest = sha256_obj(summary)
    hash_path.write_text(digest + "\n", encoding="ascii")

    print("=" * 80)
    print("v50_11 LiH REPLICATION COMPLETE")
    print("=" * 80)
    print(f"Summary JSON : {json_path}")
    print(f"SHA-256      : {digest}")
    print(
        "CLEAR PASS stages: "
        + (", ".join(clear_stages) if clear_stages else "NONE")
    )
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
