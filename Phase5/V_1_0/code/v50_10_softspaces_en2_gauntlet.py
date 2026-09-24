#!/usr/bin/env python3
"""
Soft Spaces Phase 5 — v50_10 preregistered EN2 gauntlet

Goal
----
Run several DIFFERENT downstream tests in a fixed order. Stop early only if
Soft Spaces shows a predeclared, material advantage over EN2. If all tests
fail, record the negative result rather than inventing additional post-hoc
tasks.

This script is deliberately conservative.

Input
-----
phase5_v50_7c_candidate_summary.csv

Soft-Spaces predictor
---------------------
For every v50_7c eligible adjacent eigenpair:

    SS_score = min(prominence_positive_rate, robust_positive_rate)

No weights are tuned after seeing these gauntlet outcomes.

EN2 predictor
-------------
EN2 is determinant-based. For an eigenpair (k,k+1), the bridge predictor is

    EN2_pair =
        sum_i p_i(pair) * EN2_i

where
    p_i(pair) = 0.5*(|V[i,k]|^2 + |V[i,k+1]|^2)
    EN2_i = |H_ir|^2 / (|H_ii-H_rr| + eps)

r is the determinant with minimum diagonal H_ii.

Sequential tests
----------------
T1  Held-out DEPHASING projector sensitivity
T2  Held-out TRANSVERSE projector sensitivity
T3  Held-out MIXED perturbation projector sensitivity
T4  Geometry-displacement gap sensitivity
T5  Matched-budget ground-state determinant selection

T1-T4 compare Soft Spaces with EN2 as predictors of a held-out response.
T5 reproduces the direct equal-K downstream energy comparison.

Early-stop success rule for T1-T4
---------------------------------
A stage is a CLEAR PASS only if ALL hold:

    pooled SS AUC >= 0.65
    pooled (SS AUC - EN2 AUC) >= 0.10
    median per-geometry AUC difference > 0
    at least 5 geometries have a positive AUC difference

The threshold was frozen before this gauntlet run.

T5 clear-pass rule
------------------
At least 24/40 matched-budget energy comparisons are SS wins AND
median(SS error - EN2 error) < 0.

If any stage CLEAR PASSes, the gauntlet stops and writes results.

If all five fail
----------------
The intended Phase-5 interpretation is:

    Molecular Soft-Spaces structure may be reproducible, but this preregistered
    H4 gauntlet found no demonstrated incremental utility over EN2 on the
    tested perturbative/spectral/subspace tasks.

That should trigger closure or a scientifically new hypothesis, not an
unbounded sequence of post-hoc rescue tests.

IMPORTANT LIMIT
---------------
T1-T4 use exact eigenvectors to evaluate spectral response and to bridge EN2
into eigenpair space. They test information/predictive value, not total
computational cost.

T5 uses exact Soft-Spaces hotspot eigenvectors as an oracle bridge to
determinant selection, so it also tests information content, not speedup.
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


SCRIPT_VERSION = "v50_10"

GEOMETRIES = (0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
K_VALUES = (2, 4, 8, 12, 16)

EN2_EPS = 1e-12
PERTURBATION_STRENGTH = 0.01
HELDOUT_SEEDS = tuple(range(35043000, 35043024))
GEOMETRY_DELTA = 0.025

CLEAR_SS_AUC = 0.65
CLEAR_DELTA_AUC = 0.10
CLEAR_POSITIVE_GEOMETRIES = 5
T5_MIN_SS_WINS = 24


@dataclass(frozen=True)
class Candidate:
    geometry: float
    k: int
    ss_score: float
    prominence_rate: float
    robust_rate: float
    status: str


def sha256_obj(obj: Any) -> str:
    raw = json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def popcount(x: int) -> int:
    return bin(int(x)).count("1")


def fixed_sector_basis(
    n_spatial: int, n_alpha: int, n_beta: int
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


def build_h4(spacing: float) -> dict[str, Any]:
    try:
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        from qiskit_nature.units import DistanceUnit
    except Exception as exc:
        raise RuntimeError(
            "Qiskit Nature / PySCF unavailable. Run in Phase-5 WSL env."
        ) from exc

    zs = (-1.5 * spacing, -0.5 * spacing, 0.5 * spacing, 1.5 * spacing)
    atom = "; ".join(f"H 0.0 0.0 {z:.12f}" for z in zs)

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

    if (ns, na, nb) != (4, 2, 2):
        raise RuntimeError(f"Unexpected H4 metadata: {(ns, na, nb)}")

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


def load_candidates(path: Path) -> dict[float, list[Candidate]]:
    out: dict[float, list[Candidate]] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "geometry_angstrom",
            "eigenpair_index",
            "prominence_positive_rate",
            "robust_positive_rate",
            "status",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"v50_7c candidate CSV missing columns: {sorted(missing)}"
            )

        for row in reader:
            g = float(row["geometry_angstrom"])
            pr = float(row["prominence_positive_rate"])
            rr = float(row["robust_positive_rate"])

            c = Candidate(
                geometry=g,
                k=int(row["eigenpair_index"]),
                ss_score=min(pr, rr),
                prominence_rate=pr,
                robust_rate=rr,
                status=str(row["status"]),
            )
            out.setdefault(g, []).append(c)

    return out


def en2_scores(h: np.ndarray) -> tuple[np.ndarray, int]:
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
    return score, r


def pair_prob(evecs: np.ndarray, k: int) -> np.ndarray:
    p = 0.5 * (
        np.abs(evecs[:, k]) ** 2
        + np.abs(evecs[:, k + 1]) ** 2
    )
    return p / float(np.sum(p))


def en2_pair_score(
    determinant_scores: np.ndarray, evecs: np.ndarray, k: int
) -> float:
    return float(np.sum(pair_prob(evecs, k) * determinant_scores))


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

    # Avoid pathological all-one labels due ties.
    if np.all(labels == 1):
        order = np.argsort(response)
        labels[:] = 0
        labels[order[-max(1, len(response) // 4):]] = 1

    return labels


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


def projected_flip(basis: np.ndarray, mask: int) -> np.ndarray:
    lookup = {int(full): local for local, full in enumerate(basis)}
    d = len(basis)
    out = np.zeros((d, d), dtype=np.complex128)

    for col, full in enumerate(basis):
        row = lookup.get(int(full) ^ int(mask))
        if row is not None:
            out[row, col] = 1.0

    return out


def perturbation_pools(
    basis: np.ndarray, n_qubits: int
) -> dict[str, list[np.ndarray]]:
    deph = []
    trans = []

    for q in range(n_qubits):
        deph.append(projected_z(basis, q))
        trans.append(projected_flip(basis, 1 << q))

    for q in range(n_qubits - 1):
        deph.append(projected_zz(basis, q, q + 1))
        trans.append(
            projected_flip(basis, (1 << q) | (1 << (q + 1)))
        )

    # Remove exactly-zero terms for held-out response generation.
    deph = [a for a in deph if np.linalg.norm(a, "fro") > 0.0]
    trans = [a for a in trans if np.linalg.norm(a, "fro") > 0.0]

    return {"dephasing": deph, "transverse": trans}


def random_combo(
    pool: list[np.ndarray], seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=len(pool))
    out = np.zeros_like(pool[0], dtype=np.complex128)

    for c, a in zip(coeffs, pool):
        out += float(c) * a

    norm = float(np.linalg.norm(out, "fro"))
    if norm > 0:
        out /= norm

    return 0.5 * (out + out.conjugate().T)


def pair_projector(evecs: np.ndarray, k: int) -> np.ndarray:
    u = evecs[:, [k, k + 1]]
    return u @ u.conjugate().T


def projector_response(
    h: np.ndarray,
    evecs: np.ndarray,
    candidates: list[Candidate],
    family: str,
    pools: dict[str, list[np.ndarray]],
) -> np.ndarray:
    out = np.zeros(len(candidates), dtype=float)

    for si, seed in enumerate(HELDOUT_SEEDS):
        if family == "mixed":
            d1 = random_combo(pools["dephasing"], seed + 101)
            d2 = random_combo(pools["transverse"], seed + 202)
            dh = d1 + d2
            norm = float(np.linalg.norm(dh, "fro"))
            if norm > 0:
                dh /= norm
        else:
            offset = 101 if family == "dephasing" else 202
            dh = random_combo(pools[family], seed + offset)

        _, pert_vecs = np.linalg.eigh(
            h + PERTURBATION_STRENGTH * dh
        )

        for ci, c in enumerate(candidates):
            p0 = pair_projector(evecs, c.k)
            p1 = pair_projector(pert_vecs, c.k)
            overlap = float(
                np.real(np.trace(p0 @ p1)) / 2.0
            )
            response = max(0.0, 1.0 - overlap)
            out[ci] += response

    out /= len(HELDOUT_SEEDS)
    return out


def evaluate_predictive_stage(
    name: str,
    response_by_geometry: dict[float, np.ndarray],
    candidates_by_geometry: dict[float, list[Candidate]],
    systems: dict[float, dict[str, Any]],
) -> dict[str, Any]:
    pooled_labels = []
    pooled_ss = []
    pooled_en2 = []
    per_geometry = []

    for g in GEOMETRIES:
        candidates = candidates_by_geometry.get(g, [])
        if len(candidates) < 4:
            continue

        system = systems[g]
        en2_det, _ = en2_scores(system["h"])

        response = np.asarray(response_by_geometry[g], dtype=float)
        labels = top_quartile_labels(response)

        ss_scores = np.asarray(
            [c.ss_score for c in candidates],
            dtype=float,
        )

        en2_scores_pair = np.asarray(
            [
                en2_pair_score(en2_det, system["evecs"], c.k)
                for c in candidates
            ],
            dtype=float,
        )

        a_ss = auc(labels, ss_scores)
        a_en2 = auc(labels, en2_scores_pair)

        if a_ss is None or a_en2 is None:
            continue

        delta = float(a_ss - a_en2)

        per_geometry.append(
            {
                "geometry": g,
                "n_candidates": len(candidates),
                "ss_auc": float(a_ss),
                "en2_auc": float(a_en2),
                "delta_auc": delta,
            }
        )

        pooled_labels.extend(labels.tolist())
        pooled_ss.extend(ss_scores.tolist())
        pooled_en2.extend(en2_scores_pair.tolist())

    pooled_ss_auc = auc(
        np.asarray(pooled_labels), np.asarray(pooled_ss)
    )
    pooled_en2_auc = auc(
        np.asarray(pooled_labels), np.asarray(pooled_en2)
    )

    if pooled_ss_auc is None or pooled_en2_auc is None:
        clear_pass = False
        delta_pooled = None
        median_delta = None
        positive_geometries = 0
    else:
        delta_pooled = float(pooled_ss_auc - pooled_en2_auc)
        deltas = [r["delta_auc"] for r in per_geometry]
        median_delta = (
            float(np.median(deltas)) if deltas else None
        )
        positive_geometries = int(
            sum(d > 0.0 for d in deltas)
        )

        clear_pass = bool(
            pooled_ss_auc >= CLEAR_SS_AUC
            and delta_pooled >= CLEAR_DELTA_AUC
            and median_delta is not None
            and median_delta > 0.0
            and positive_geometries >= CLEAR_POSITIVE_GEOMETRIES
        )

    return {
        "name": name,
        "pooled_ss_auc": pooled_ss_auc,
        "pooled_en2_auc": pooled_en2_auc,
        "pooled_delta_auc": delta_pooled,
        "median_geometry_delta_auc": median_delta,
        "positive_delta_geometries": positive_geometries,
        "per_geometry": per_geometry,
        "clear_pass": clear_pass,
    }


def geometry_gap_response(
    systems: dict[float, dict[str, Any]],
    candidates_by_geometry: dict[float, list[Candidate]],
) -> dict[float, np.ndarray]:
    out = {}

    for g in GEOMETRIES:
        minus = build_h4(g - GEOMETRY_DELTA)
        plus = build_h4(g + GEOMETRY_DELTA)

        gaps_minus = np.diff(minus["evals"])
        gaps_plus = np.diff(plus["evals"])

        vals = []
        for c in candidates_by_geometry.get(g, []):
            vals.append(
                abs(
                    float(gaps_plus[c.k])
                    - float(gaps_minus[c.k])
                )
            )

        out[g] = np.asarray(vals, dtype=float)

    return out


def softspaces_oracle_scores(
    evecs: np.ndarray, candidates: list[Candidate]
) -> np.ndarray:
    score = np.zeros(evecs.shape[0], dtype=float)

    for c in candidates:
        if c.status != "PASS":
            continue

        support = pair_prob(evecs, c.k)
        score += c.ss_score * support

    return score


def topk_with_reference(
    scores: np.ndarray, reference: int, k: int
) -> np.ndarray:
    s = np.asarray(scores, dtype=float).copy()
    s[reference] = np.inf
    return np.asarray(
        np.argsort(-s, kind="stable")[:k],
        dtype=int,
    )


def t5_matched_budget(
    systems: dict[float, dict[str, Any]],
    candidates_by_geometry: dict[float, list[Candidate]],
) -> dict[str, Any]:
    rows = []

    for g in GEOMETRIES:
        system = systems[g]
        h = system["h"]
        evals = system["evals"]
        evecs = system["evecs"]

        exact_e = float(evals[0])

        en2_det, reference = en2_scores(h)
        ss_det = softspaces_oracle_scores(
            evecs,
            candidates_by_geometry.get(g, []),
        )

        for k in K_VALUES:
            ss_sel = topk_with_reference(ss_det, reference, k)
            en2_sel = topk_with_reference(en2_det, reference, k)

            e_ss = float(
                np.min(
                    np.linalg.eigvalsh(
                        h[np.ix_(ss_sel, ss_sel)]
                    )
                ).real
            )
            e_en2 = float(
                np.min(
                    np.linalg.eigvalsh(
                        h[np.ix_(en2_sel, en2_sel)]
                    )
                ).real
            )

            err_ss = abs(e_ss - exact_e)
            err_en2 = abs(e_en2 - exact_e)

            rows.append(
                {
                    "geometry": g,
                    "k": k,
                    "ss_error": err_ss,
                    "en2_error": err_en2,
                    "delta_error": err_ss - err_en2,
                    "ss_win": bool(err_ss < err_en2 - 1e-12),
                }
            )

    wins = int(sum(r["ss_win"] for r in rows))
    median_delta = float(
        np.median([r["delta_error"] for r in rows])
    )

    clear_pass = bool(
        wins >= T5_MIN_SS_WINS
        and median_delta < 0.0
    )

    return {
        "name": "T5 matched-budget ground-state subspace",
        "softspaces_wins": wins,
        "total_comparisons": len(rows),
        "median_delta_error_ss_minus_en2": median_delta,
        "clear_pass": clear_pass,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    root = Path(__file__).resolve().parent.parent

    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=(
            root
            / "results"
            / "phase23_method_bridge"
            / "phase5_v50_7c_candidate_summary.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "v50_10_gauntlet",
    )

    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Run all five stages even if an earlier stage CLEAR PASSes.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    candidate_csv = args.candidate_csv.resolve()
    if not candidate_csv.exists():
        raise FileNotFoundError(
            f"Cannot find v50_7c candidate summary: {candidate_csv}"
        )

    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(candidate_csv)

    print("=" * 80)
    print("SOFT SPACES PHASE 5 — v50_10 PREREGISTERED EN2 GAUNTLET")
    print("=" * 80)
    print(f"Candidate input : {candidate_csv}")
    print(f"Held-out seeds : {HELDOUT_SEEDS[0]}..{HELDOUT_SEEDS[-1]}")
    print(f"Perturbation λ : {PERTURBATION_STRENGTH}")
    print(f"Geometry delta : +/-{GEOMETRY_DELTA} Å")
    print(
        "CLEAR PASS T1-T4: SS AUC>=0.65; ΔAUC>=0.10; "
        "median geom ΔAUC>0; >=5 positive geometries"
    )
    print(
        f"CLEAR PASS T5   : >={T5_MIN_SS_WINS}/40 SS energy wins "
        "and median Δerror<0"
    )
    print(
        "Early stop      : "
        + ("disabled" if args.no_early_stop else "enabled")
    )
    print("-" * 80)

    # Cache baseline systems once.
    systems = {}
    for i, g in enumerate(GEOMETRIES, start=1):
        print(f"Building baseline H4 {i}/8: {g:.2f} Å", flush=True)
        systems[g] = build_h4(g)

    results = []
    winner = None

    # T1-T3: held-out projector response.
    for stage_name, family in (
        ("T1 held-out dephasing projector sensitivity", "dephasing"),
        ("T2 held-out transverse projector sensitivity", "transverse"),
        ("T3 held-out mixed projector sensitivity", "mixed"),
    ):
        print("-" * 80)
        print(stage_name)

        response = {}

        for g in GEOMETRIES:
            cs = candidates.get(g, [])
            pools = perturbation_pools(
                systems[g]["basis"],
                systems[g]["n_qubits"],
            )

            response[g] = projector_response(
                systems[g]["h"],
                systems[g]["evecs"],
                cs,
                family,
                pools,
            )

        stage = evaluate_predictive_stage(
            stage_name,
            response,
            candidates,
            systems,
        )
        results.append(stage)

        print(
            f"  pooled SS AUC  = {stage['pooled_ss_auc']:.3f}"
            if stage["pooled_ss_auc"] is not None
            else "  pooled SS AUC  = nan"
        )
        print(
            f"  pooled EN2 AUC = {stage['pooled_en2_auc']:.3f}"
            if stage["pooled_en2_auc"] is not None
            else "  pooled EN2 AUC = nan"
        )
        print(
            f"  ΔAUC SS-EN2    = {stage['pooled_delta_auc']:+.3f}"
            if stage["pooled_delta_auc"] is not None
            else "  ΔAUC SS-EN2    = nan"
        )
        print(
            f"  + geometries   = {stage['positive_delta_geometries']}"
        )
        print(
            "  RESULT         = "
            + ("CLEAR PASS" if stage["clear_pass"] else "FAIL")
        )

        if stage["clear_pass"] and not args.no_early_stop:
            winner = stage_name
            break

    # T4 if no earlier clear pass.
    if winner is None:
        print("-" * 80)
        print("T4 geometry-displacement gap sensitivity")

        response = geometry_gap_response(
            systems,
            candidates,
        )

        stage = evaluate_predictive_stage(
            "T4 geometry-displacement gap sensitivity",
            response,
            candidates,
            systems,
        )
        results.append(stage)

        print(
            f"  pooled SS AUC  = {stage['pooled_ss_auc']:.3f}"
            if stage["pooled_ss_auc"] is not None
            else "  pooled SS AUC  = nan"
        )
        print(
            f"  pooled EN2 AUC = {stage['pooled_en2_auc']:.3f}"
            if stage["pooled_en2_auc"] is not None
            else "  pooled EN2 AUC = nan"
        )
        print(
            f"  ΔAUC SS-EN2    = {stage['pooled_delta_auc']:+.3f}"
            if stage["pooled_delta_auc"] is not None
            else "  ΔAUC SS-EN2    = nan"
        )
        print(
            f"  + geometries   = {stage['positive_delta_geometries']}"
        )
        print(
            "  RESULT         = "
            + ("CLEAR PASS" if stage["clear_pass"] else "FAIL")
        )

        if stage["clear_pass"] and not args.no_early_stop:
            winner = stage["name"]

    # T5 if no earlier clear pass.
    if winner is None:
        print("-" * 80)
        print("T5 matched-budget ground-state determinant selection")

        stage = t5_matched_budget(
            systems,
            candidates,
        )
        results.append(stage)

        print(
            f"  SS wins        = {stage['softspaces_wins']}/"
            f"{stage['total_comparisons']}"
        )
        print(
            f"  median Δerror  = "
            f"{stage['median_delta_error_ss_minus_en2']:+.8e}"
        )
        print(
            "  RESULT         = "
            + ("CLEAR PASS" if stage["clear_pass"] else "FAIL")
        )

        if stage["clear_pass"]:
            winner = stage["name"]

    all_failed = winner is None

    summary = {
        "_metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script_version": SCRIPT_VERSION,
            "python_version": sys.version.split()[0],
        },
        "preregistered_rules": {
            "t1_t4": {
                "ss_auc_min": CLEAR_SS_AUC,
                "delta_auc_min": CLEAR_DELTA_AUC,
                "median_geometry_delta_auc_must_be_positive": True,
                "positive_delta_geometries_min": CLEAR_POSITIVE_GEOMETRIES,
            },
            "t5": {
                "softspaces_wins_min": T5_MIN_SS_WINS,
                "total_comparisons": 40,
                "median_delta_error_must_be_negative": True,
            },
            "early_stop_enabled": not args.no_early_stop,
        },
        "results": results,
        "clear_pass_stage": winner,
        "all_stages_failed": all_failed,
        "closure_interpretation_if_all_fail": (
            "Reproducible molecular Soft-Spaces structure was observed previously, "
            "but this preregistered H4 gauntlet found no demonstrated incremental "
            "utility over EN2 on held-out perturbative sensitivity, geometry "
            "sensitivity, or matched-budget ground-state subspace selection. "
            "Do not continue with indefinite post-hoc rescue tests; close this "
            "application branch or formulate a genuinely new hypothesis."
        ),
    }

    json_path = outdir / "phase5_v50_10_gauntlet_summary.json"
    hash_path = outdir / "phase5_v50_10_gauntlet_summary.sha256"

    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    digest = sha256_obj(summary)
    hash_path.write_text(digest + "\n", encoding="ascii")

    print("=" * 80)
    print("v50_10 GAUNTLET COMPLETE")
    print("=" * 80)
    print(f"Summary JSON : {json_path}")
    print(f"SHA-256      : {digest}")

    if winner is not None:
        print(f"CLEAR PASS    : {winner}")
        print("Gauntlet stopped: a preregistered material SS advantage was found.")
    else:
        print("CLEAR PASS    : NONE")
        print(
            "All preregistered stages failed. Recommended action: close this "
            "H4 application branch rather than add post-hoc rescue tests."
        )

    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
