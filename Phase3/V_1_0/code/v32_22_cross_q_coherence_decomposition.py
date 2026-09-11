#!/usr/bin/env python3
"""
Soft Spaces Phase 3.2 — v32.22 Cross-Q Coherence Decomposition
==============================================================

Purpose
-------
Extend the exact v32.21 coherence decomposition from 8Q across the finite-size
range 7Q ... 12Q.

For each degeneracy-closed target eigenspace P and perturbation family,

    T_G = g_eta(E - E_G) P V P_G V P,

    Sigma_E = sum_G T_G,

and

    C_group^2 = S_group + A_group,

with

    S_group =
        sum_G ||T_G||_F^2
        -------------------------
        (sum_G ||T_G||_F)^2

and

    A_group =
        2 sum_{G<K} Re Tr(T_G^dagger T_K)
        -----------------------------------
        (sum_G ||T_G||_F)^2.

Scientific question
-------------------
Does the decomposition retain the same qualitative hotspot organization from
7Q through 12Q?

For each qubit count, family, and seed, the script computes Spearman rho between
hotspot prominence and:

    S_group      pathway concentration
    A_group      normalized interference
    C2_group     exact sum S_group + A_group
    C_group      coherence factor

The primary cross-Q criterion is finite-size robustness, not monotonic scaling:
the median seedwise rho should remain positive over the investigated Q range.

Protocol
--------
    * qubits 7 ... 12
    * 11 Hamiltonian terms
    * seeds 25042000 ... 25042011
    * dephasing and transverse perturbation families
    * frozen REAL/NULL construction from v32.14
    * complete degeneracy-closed Q groups
    * no feature fitting or hyperparameter tuning

Important runtime note
----------------------
12Q uses dense 4096 x 4096 matrices and is much more expensive than 7Q-11Q.
For safety, the script supports running one qubit count at a time.  Example:

    python .\\v32_22_cross_q_coherence_decomposition.py --qubits 12

Dependencies
------------
Must be beside this file:

    v32_14_degeneracy_closed.py

Outputs
-------
For each requested Q:

    v32_22_q7_rows.csv
    v32_22_q7_seed_stats.csv
    v32_22_q7_summary.csv
    ...
    v32_22_q12_*.csv

And for the complete invocation:

    v32_22_cross_q_summary.csv
    v32_22_cross_q_summary.txt

Usage
-----
Recommended staged execution:

    python .\\v32_22_cross_q_coherence_decomposition.py --qubits 7
    python .\\v32_22_cross_q_coherence_decomposition.py --qubits 8
    python .\\v32_22_cross_q_coherence_decomposition.py --qubits 9
    python .\\v32_22_cross_q_coherence_decomposition.py --qubits 10
    python .\\v32_22_cross_q_coherence_decomposition.py --qubits 11
    python .\\v32_22_cross_q_coherence_decomposition.py --qubits 12

Or several at once:

    python .\\v32_22_cross_q_coherence_decomposition.py --qubits 7 8 9
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


VERSION = "v32.22"
ALLOWED_QUBITS = (7, 8, 9, 10, 11, 12)
DEFAULT_SEEDS = tuple(range(25_042_000, 25_042_012))
FAMILIES = ("dephasing", "transverse")
DENOM_EPS = 1.0e-30


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

def load_base():
    here = Path(__file__).resolve().parent
    path = here / "v32_14_degeneracy_closed.py"

    if not path.exists():
        raise FileNotFoundError(
            "v32_14_degeneracy_closed.py must be beside this file."
        )

    spec = importlib.util.spec_from_file_location("v3214_for_v3222", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load v32_14_degeneracy_closed.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
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


def exact_one_sided_sign_p(k_positive: int, n: int) -> float:
    if n <= 0:
        return float("nan")

    return float(
        sum(math.comb(n, k) for k in range(k_positive, n + 1))
        / (2 ** n)
    )


# ---------------------------------------------------------------------------
# Exact decomposition
# ---------------------------------------------------------------------------

def decomposition_from_terms(terms: List[np.ndarray]) -> dict:
    if not terms:
        return {
            "S_group": 0.0,
            "A_group": 0.0,
            "C2_group": 0.0,
            "C_group": 0.0,
            "identity_residual": 0.0,
            "n_channels": 0,
        }

    mats = [np.asarray(t, dtype=np.complex128) for t in terms]

    norms = np.asarray(
        [float(np.linalg.norm(t, ord="fro")) for t in mats],
        dtype=float,
    )

    l1 = float(np.sum(norms))
    self_sum = float(np.sum(norms * norms))

    sigma = np.sum(mats, axis=0)
    sigma2 = float(np.linalg.norm(sigma, ord="fro") ** 2)

    if l1 <= DENOM_EPS:
        return {
            "S_group": 0.0,
            "A_group": 0.0,
            "C2_group": 0.0,
            "C_group": 0.0,
            "identity_residual": 0.0,
            "n_channels": int(len(mats)),
        }

    denom = l1 * l1
    cross_sum = float(sigma2 - self_sum)

    S_group = float(self_sum / denom)
    A_group = float(cross_sum / denom)
    C2_group = float(sigma2 / denom)
    C_group = float(np.sqrt(max(C2_group, 0.0)))

    return {
        "S_group": S_group,
        "A_group": A_group,
        "C2_group": C2_group,
        "C_group": C_group,
        "identity_residual": float(C2_group - (S_group + A_group)),
        "n_channels": int(len(mats)),
    }


# ---------------------------------------------------------------------------
# One seed / one Q
# ---------------------------------------------------------------------------

def build_seed_reference(base, seed: int, n_qubits: int):
    h_terms = base.random_hamiltonian_terms(seed)
    h = base.dense_pauli_sum(n_qubits, h_terms)

    evals, real_vecs = np.linalg.eigh(h)
    groups = base.degenerate_groups(evals)

    null_vecs = base.haar_unitary(
        1 << n_qubits,
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
        for family in FAMILIES
    }

    real_by_family = {
        family: base.model_scores(
            evals, real_vecs, perturbations[family], groups
        )
        for family in FAMILIES
    }

    null_by_family = {
        family: base.model_scores(
            evals, null_vecs, perturbations[family], groups
        )
        for family in FAMILIES
    }

    prominence = base.prominence_values(
        base.robust_difference(
            real_by_family,
            null_by_family,
            len(groups),
        ),
        len(groups),
    )

    energies = np.asarray(
        [base.eigenspace_energy(evals, g) for g in groups],
        dtype=float,
    )

    pairs = base.opposite_energy_pairs(energies)

    # H and NULL are no longer needed after hotspot prominence is frozen.
    del h
    del null_vecs
    del real_by_family
    del null_by_family
    gc.collect()

    return evals, real_vecs, groups, energies, perturbations, prominence, pairs


def analyze_seed(base, seed: int, n_qubits: int):
    (
        evals,
        real_vecs,
        groups,
        energies,
        perturbations,
        prominence,
        pairs,
    ) = build_seed_reference(base, seed, n_qubits)

    pair_records = []

    for a, b in pairs:
        gm, gp = (a, b) if energies[a] <= energies[b] else (b, a)

        pair_prom = max(
            prominence.get(gm, float("-inf")),
            prominence.get(gp, float("-inf")),
        )

        if np.isfinite(pair_prom):
            pair_records.append(
                {
                    "g_minus": int(gm),
                    "g_plus": int(gp),
                    "E_abs": float(abs(energies[gp])),
                    "prominence": float(pair_prom),
                }
            )

    if not pair_records:
        raise RuntimeError(
            f"{n_qubits}Q seed {seed}: no usable exact +/-E pairs."
        )

    needed_groups = sorted(
        set(p["g_minus"] for p in pair_records)
        | set(p["g_plus"] for p in pair_records)
    )

    rows: List[dict] = []
    seed_stats: List[dict] = []

    for family in FAMILIES:
        perturbation = perturbations[family]
        desc_by_group: Dict[int, dict] = {}

        for gi in needed_groups:
            terms = base.degeneracy_closed_terms(
                eigenvalues=evals,
                eigenvectors=real_vecs,
                groups=groups,
                target_group_number=gi,
                perturbation=perturbation,
                eta=base.ENERGY_REG,
            )

            desc_by_group[gi] = decomposition_from_terms(terms)

        for pair in pair_records:
            gm = pair["g_minus"]
            gp = pair["g_plus"]

            dm = desc_by_group[gm]
            dp = desc_by_group[gp]

            rows.append(
                {
                    "version": VERSION,
                    "n_qubits": int(n_qubits),
                    "dimension": int(1 << n_qubits),
                    "seed": int(seed),
                    "family": family,
                    "g_minus": int(gm),
                    "g_plus": int(gp),
                    "E_abs": float(pair["E_abs"]),
                    "prominence": float(pair["prominence"]),
                    "S_group": 0.5 * (
                        float(dm["S_group"]) + float(dp["S_group"])
                    ),
                    "A_group": 0.5 * (
                        float(dm["A_group"]) + float(dp["A_group"])
                    ),
                    "C2_group": 0.5 * (
                        float(dm["C2_group"]) + float(dp["C2_group"])
                    ),
                    "C_group": 0.5 * (
                        float(dm["C_group"]) + float(dp["C_group"])
                    ),
                    "n_channels_mean": 0.5 * (
                        float(dm["n_channels"]) + float(dp["n_channels"])
                    ),
                    "identity_residual_maxabs": max(
                        abs(float(dm["identity_residual"])),
                        abs(float(dp["identity_residual"])),
                    ),
                    "n_groups": int(len(groups)),
                    "n_pairs": int(len(pair_records)),
                }
            )

        sub = [
            r
            for r in rows
            if r["family"] == family
        ]

        y = [float(r["prominence"]) for r in sub]

        for metric in ("S_group", "A_group", "C2_group", "C_group"):
            rho = spearman(
                base,
                [float(r[metric]) for r in sub],
                y,
            )

            seed_stats.append(
                {
                    "version": VERSION,
                    "n_qubits": int(n_qubits),
                    "dimension": int(1 << n_qubits),
                    "seed": int(seed),
                    "family": family,
                    "metric": metric,
                    "spearman_rho_vs_prominence": float(rho),
                    "n_pairs": int(len(sub)),
                    "n_groups": int(len(groups)),
                }
            )

    max_identity_residual = max(
        abs(float(r["identity_residual_maxabs"]))
        for r in rows
    )

    # Release the large eigensystem before the next seed.
    del evals
    del real_vecs
    del groups
    gc.collect()

    return rows, seed_stats, max_identity_residual


# ---------------------------------------------------------------------------
# Per-Q and cross-Q summaries
# ---------------------------------------------------------------------------

def summarize_q(seed_stats: List[dict], n_qubits: int) -> List[dict]:
    out = []

    for family in FAMILIES:
        for metric in ("S_group", "A_group", "C2_group", "C_group"):
            vals = [
                float(r["spearman_rho_vs_prominence"])
                for r in seed_stats
                if int(r["n_qubits"]) == int(n_qubits)
                and r["family"] == family
                and r["metric"] == metric
            ]

            positive = sum(v > 0.0 for v in vals)
            negative = sum(v < 0.0 for v in vals)
            zero = sum(v == 0.0 for v in vals)

            out.append(
                {
                    "version": VERSION,
                    "n_qubits": int(n_qubits),
                    "dimension": int(1 << n_qubits),
                    "family": family,
                    "metric": metric,
                    "n_seeds": int(len(vals)),
                    "median_rho": float(np.median(vals)),
                    "mean_rho": float(np.mean(vals)),
                    "min_rho": float(np.min(vals)),
                    "max_rho": float(np.max(vals)),
                    "positive_seeds": int(positive),
                    "negative_seeds": int(negative),
                    "zero_seeds": int(zero),
                    "sign_test_p_one_sided": exact_one_sided_sign_p(
                        positive, len(vals)
                    ),
                }
            )

    return out


def summarize_cross_q(all_q_summary: List[dict]) -> List[dict]:
    out = []

    for family in FAMILIES:
        for metric in ("S_group", "A_group", "C2_group", "C_group"):
            sub = [
                r
                for r in all_q_summary
                if r["family"] == family and r["metric"] == metric
            ]

            q_medians = [float(r["median_rho"]) for r in sub]

            out.append(
                {
                    "version": VERSION,
                    "family": family,
                    "metric": metric,
                    "n_qubit_levels": int(len(sub)),
                    "median_of_q_medians": float(np.median(q_medians)),
                    "min_q_median": float(np.min(q_medians)),
                    "max_q_median": float(np.max(q_medians)),
                    "positive_q_levels": int(sum(v > 0.0 for v in q_medians)),
                }
            )

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Soft Spaces Phase 3.2 v32.22 — cross-Q coherence "
            "decomposition robustness test."
        )
    )

    p.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        choices=ALLOWED_QUBITS,
        required=True,
        help="One or more qubit counts from 7 through 12.",
    )

    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Seed list. Default: 25042000 ... 25042011.",
    )

    p.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (default current directory).",
    )

    return p.parse_args()


def main():
    args = parse_args()
    base = load_base()

    base.N_TERMS = 11

    qubits = list(dict.fromkeys(int(q) for q in args.qubits))
    seeds = [int(s) for s in args.seeds]

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    all_q_summary: List[dict] = []
    max_identity_global = 0.0

    print(
        f"{VERSION}: cross-Q coherence decomposition"
    )
    print(
        "  C_group^2 = S_group + A_group"
    )
    print(
        "  qubits=" + ",".join(str(q) for q in qubits)
    )
    print(
        f"  seeds={len(seeds)}"
    )

    for q_index, n_qubits in enumerate(qubits, start=1):
        base.N_QUBITS = int(n_qubits)
        base.DIM = int(1 << n_qubits)

        print(
            f"\n=== [{q_index}/{len(qubits)}] {n_qubits}Q "
            f"(dim={1 << n_qubits}) ===",
            flush=True,
        )

        q_rows: List[dict] = []
        q_seed_stats: List[dict] = []
        q_max_identity = 0.0

        for seed_index, seed in enumerate(seeds, start=1):
            print(
                f"[{n_qubits}Q] [{seed_index}/{len(seeds)}] "
                f"seed {seed}",
                flush=True,
            )

            rows, stats, resid = analyze_seed(
                base=base,
                seed=seed,
                n_qubits=n_qubits,
            )

            q_rows.extend(rows)
            q_seed_stats.extend(stats)

            q_max_identity = max(q_max_identity, float(resid))
            max_identity_global = max(max_identity_global, float(resid))

            # Compact progress summary.
            for family in FAMILIES:
                rel = [
                    r
                    for r in stats
                    if r["family"] == family
                ]
                bits = [
                    f"{r['metric']}="
                    f"{r['spearman_rho_vs_prominence']:+.3f}"
                    for r in rel
                ]
                print(
                    f"  {family:10s}: " + ", ".join(bits),
                    flush=True,
                )

        q_summary = summarize_q(q_seed_stats, n_qubits)
        all_q_summary.extend(q_summary)

        write_csv(
            outdir / f"v32_22_q{n_qubits}_rows.csv",
            q_rows,
        )
        write_csv(
            outdir / f"v32_22_q{n_qubits}_seed_stats.csv",
            q_seed_stats,
        )
        write_csv(
            outdir / f"v32_22_q{n_qubits}_summary.csv",
            q_summary,
        )

        print(
            f"\n{n_qubits}Q SUMMARY",
            flush=True,
        )

        for family in FAMILIES:
            print(f"[{family}]")
            for r in q_summary:
                if r["family"] != family:
                    continue
                print(
                    f"  {r['metric']:8s} "
                    f"median_rho={r['median_rho']:+.4f} "
                    f"positive={r['positive_seeds']}/{r['n_seeds']} "
                    f"p={r['sign_test_p_one_sided']:.6f}"
                )

        print(
            f"max identity residual ({n_qubits}Q) = "
            f"{q_max_identity:.6e}"
        )

        # Release Q-specific rows before next system size where possible.
        del q_rows
        del q_seed_stats
        gc.collect()

    cross_q = summarize_cross_q(all_q_summary)

    write_csv(
        outdir / "v32_22_cross_q_summary.csv",
        all_q_summary,
    )

    # Separate compact family/metric cross-Q overview.
    write_csv(
        outdir / "v32_22_cross_q_overview.csv",
        cross_q,
    )

    lines = [
        f"Soft Spaces Phase 3.2 {VERSION}",
        "Cross-Q coherence decomposition robustness",
        "",
        "EXACT IDENTITY",
        "--------------",
        "C_group^2 = S_group + A_group",
        "",
        "QUBITS",
        "------",
        ",".join(str(q) for q in qubits),
        "",
        "PER-Q RESULTS",
        "-------------",
    ]

    for n_qubits in qubits:
        lines.append("")
        lines.append(f"[{n_qubits}Q]")

        for family in FAMILIES:
            lines.append(f"  {family}")

            for r in all_q_summary:
                if (
                    int(r["n_qubits"]) == int(n_qubits)
                    and r["family"] == family
                ):
                    lines.append(
                        f"    {r['metric']:8s} "
                        f"median_rho={r['median_rho']:+.4f} "
                        f"positive={r['positive_seeds']}/{r['n_seeds']} "
                        f"sign_p={r['sign_test_p_one_sided']:.6f}"
                    )

    lines += [
        "",
        "CROSS-Q OVERVIEW",
        "----------------",
    ]

    for family in FAMILIES:
        lines.append("")
        lines.append(f"[{family}]")

        for r in cross_q:
            if r["family"] != family:
                continue

            lines.append(
                f"  {r['metric']:8s} "
                f"median_of_Q_medians={r['median_of_q_medians']:+.4f} "
                f"positive_Q={r['positive_q_levels']}/{r['n_qubit_levels']} "
                f"range=[{r['min_q_median']:+.4f},"
                f"{r['max_q_median']:+.4f}]"
            )

    lines += [
        "",
        "ALGEBRAIC CONSISTENCY",
        "---------------------",
        f"max |C2-(S+A)| across invocation = {max_identity_global:.6e}",
        "",
        "DECISION RULE",
        "-------------",
        "The mechanism is considered finite-size robust if C_group/C2_group",
        "retain positive median seedwise association throughout the tested",
        "7Q-12Q range, while S_group and A_group reveal how concentration",
        "and interference share the contribution.",
        "",
        "No asymptotic theorem is claimed.",
    ]

    summary_text = "\n".join(lines) + "\n"
    summary_path = outdir / "v32_22_cross_q_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    print("\n" + summary_text)
    print("WROTE")
    print(f"  {outdir / 'v32_22_cross_q_summary.csv'}")
    print(f"  {outdir / 'v32_22_cross_q_overview.csv'}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
