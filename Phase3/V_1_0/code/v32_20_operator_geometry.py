#!/usr/bin/env python3
"""
Soft Spaces Phase 3.2 — v32.20 Operator-Geometry Law
====================================================

Purpose
-------
Test whether hotspot prominence is controlled by the geometry of the
degeneracy-closed virtual pathway operators

    T_G = g_eta(E - E_G) P V P_G V P,

rather than by a simple perturbative susceptibility.

The core object is the Hilbert-Schmidt Gram matrix of the nonzero T_G:

    K_GK = Re Tr(T_G^dagger T_K).

To isolate orientation from channel amplitude, define the normalized
operator-geometry matrix

    M_GK =
        Re Tr(T_G^dagger T_K)
        ---------------------------------
        ||T_G||_F ||T_K||_F .

This is a real symmetric positive-semidefinite Gram matrix for the
operators regarded as vectors in Hilbert-Schmidt space.

Predeclared geometry descriptors
--------------------------------
No feature search is performed.  Four interpretable descriptors are
tested:

1. mean_cosine
       Mean off-diagonal element of M.
       Positive values indicate net constructive alignment.

2. positive_fraction
       Fraction of off-diagonal M_GK values > 0.

3. mode_excess
       (lambda_max(M) - 1)/(n_channels - 1).

       For mutually orthogonal channels lambda_max ~ 1, so mode_excess ~ 0.
       A large value indicates a dominant collective aligned mode.

4. weighted_mode_fraction
       lambda_max(K) / Tr(K).

       This retains both geometry and channel strengths.

Static baselines
----------------
The established degeneracy-closed descriptors are retained:

    A_group
    C_group

The purpose is not to fit a larger model, but to ask whether a compact
operator-geometric quantity rivals or improves on those static baselines.

Frozen protocol
---------------
    * 8 qubits
    * 11 Hamiltonian terms
    * seeds 25042000 ... 25042011
    * dephasing and transverse families
    * same frozen REAL/NULL hotspot prominence as v32.14/v32.17
    * complete degeneracy-closed Q groups
    * no hyperparameter scan
    * no feature combination / regression

Pair-level convention
---------------------
For each exact +/-E pair, hotspot prominence is the frozen pair prominence
used in Phase 3.  Geometry descriptors are averaged over the -E and +E
members before rank correlation with prominence.

Dependencies
------------
Must be beside this file:

    v32_14_degeneracy_closed.py
    v32_17_perturbative_response.py

Outputs
-------
    v32_20_rows.csv
    v32_20_seed_stats.csv
    v32_20_summary.csv
    v32_20_summary.txt

Usage
-----
    python .\\v32_20_operator_geometry.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


VERSION = "v32.20"
DEFAULT_SEEDS = tuple(range(25_042_000, 25_042_012))
NORM_EPS = 1.0e-14


# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

def load_module(filename: str, module_name: str):
    here = Path(__file__).resolve().parent
    path = here / filename

    if not path.exists():
        raise FileNotFoundError(f"{filename} must be beside this file.")

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Helpers
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


def exact_one_sided_sign_p(k_positive: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    total = sum(math.comb(n, k) for k in range(k_positive, n + 1))
    return float(total / (2 ** n))


def pair_records(ref: dict) -> List[dict]:
    out = []

    for a, b in ref["pairs"]:
        gm, gp = (
            (a, b)
            if ref["energies"][a] <= ref["energies"][b]
            else (b, a)
        )

        prominence = max(
            ref["prominence"].get(gm, float("-inf")),
            ref["prominence"].get(gp, float("-inf")),
        )

        if not np.isfinite(prominence):
            continue

        out.append(
            {
                "g_minus": int(gm),
                "g_plus": int(gp),
                "E_abs": float(abs(ref["energies"][gp])),
                "prominence": float(prominence),
            }
        )

    return out


# ---------------------------------------------------------------------------
# Operator geometry
# ---------------------------------------------------------------------------

def operator_geometry_descriptors(
    terms: List[np.ndarray],
    norm_eps: float = NORM_EPS,
) -> dict:
    """
    Compute basis-invariant geometry descriptors from degeneracy-closed T_G.

    Each T_G is itself basis-invariant with respect to rotations inside the
    complete degenerate Q eigenspace.  Hilbert-Schmidt inner products are
    additionally invariant under basis changes in the target P subspace.
    """
    kept = []
    norms = []

    for t in terms:
        nrm = float(np.linalg.norm(t, ord="fro"))
        if np.isfinite(nrm) and nrm > norm_eps:
            kept.append(np.asarray(t, dtype=np.complex128))
            norms.append(nrm)

    n = len(kept)

    if n == 0:
        return {
            "n_channels": 0,
            "mean_cosine": 0.0,
            "positive_fraction": 0.0,
            "mode_excess": 0.0,
            "weighted_mode_fraction": 0.0,
            "gram_trace": 0.0,
        }

    # Raw Hilbert-Schmidt Gram matrix.
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = float(np.real(np.trace(kept[i].conj().T @ kept[j])))
            K[i, j] = val
            K[j, i] = val

    K = 0.5 * (K + K.T)

    norm_arr = np.asarray(norms, dtype=float)
    denom = np.outer(norm_arr, norm_arr)
    M = K / denom
    M = 0.5 * (M + M.T)

    # Numerical cleanup: diagonal should be exactly 1.
    np.fill_diagonal(M, 1.0)

    if n == 1:
        mean_cosine = 0.0
        positive_fraction = 0.0
        mode_excess = 0.0
    else:
        mask = ~np.eye(n, dtype=bool)
        off = M[mask]
        mean_cosine = float(np.mean(off))
        positive_fraction = float(np.mean(off > 0.0))

        evals_M = np.linalg.eigvalsh(M)
        lambda_max_M = float(np.max(evals_M))
        mode_excess = float((lambda_max_M - 1.0) / (n - 1.0))

    evals_K = np.linalg.eigvalsh(K)
    # Tiny negative eigenvalues can occur from roundoff.
    evals_K = np.maximum(evals_K, 0.0)
    trace_K = float(np.sum(evals_K))

    if trace_K > 0.0:
        weighted_mode_fraction = float(np.max(evals_K) / trace_K)
    else:
        weighted_mode_fraction = 0.0

    return {
        "n_channels": int(n),
        "mean_cosine": mean_cosine,
        "positive_fraction": positive_fraction,
        "mode_excess": mode_excess,
        "weighted_mode_fraction": weighted_mode_fraction,
        "gram_trace": trace_K,
    }


def group_descriptors(base, ref: dict, gi: int, perturbation, eta: float):
    terms = base.degeneracy_closed_terms(
        eigenvalues=ref["evals"],
        eigenvectors=ref["evecs"],
        groups=ref["groups"],
        target_group_number=gi,
        perturbation=perturbation,
        eta=eta,
    )

    static_desc = base.interference_descriptors_group_closed(terms)
    geom_desc = operator_geometry_descriptors(terms)

    return static_desc, geom_desc


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------

def analyze_seed(base, v3217, seed: int):
    eta = float(base.ENERGY_REG)
    ref = v3217.build_reference(base, seed)
    pairs = pair_records(ref)

    if not pairs:
        raise RuntimeError(f"Seed {seed}: no usable +/-E pairs.")

    rows: List[dict] = []
    seed_stats: List[dict] = []

    needed_groups = sorted(
        set(p["g_minus"] for p in pairs)
        | set(p["g_plus"] for p in pairs)
    )

    for family in base.FAMILIES:
        perturbation = ref["perturbations"][family]

        static_by_group: Dict[int, dict] = {}
        geom_by_group: Dict[int, dict] = {}

        for gi in needed_groups:
            static_desc, geom_desc = group_descriptors(
                base, ref, gi, perturbation, eta
            )
            static_by_group[gi] = static_desc
            geom_by_group[gi] = geom_desc

        for pair in pairs:
            gm = pair["g_minus"]
            gp = pair["g_plus"]

            sm = static_by_group[gm]
            sp = static_by_group[gp]
            gm_desc = geom_by_group[gm]
            gp_desc = geom_by_group[gp]

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

                # Established static baselines.
                "A_group": 0.5 * (
                    float(sm["A_group"]) + float(sp["A_group"])
                ),
                "C_group": 0.5 * (
                    float(sm["C_group"]) + float(sp["C_group"])
                ),

                # Predeclared operator-geometry descriptors.
                "mean_cosine": 0.5 * (
                    float(gm_desc["mean_cosine"])
                    + float(gp_desc["mean_cosine"])
                ),
                "positive_fraction": 0.5 * (
                    float(gm_desc["positive_fraction"])
                    + float(gp_desc["positive_fraction"])
                ),
                "mode_excess": 0.5 * (
                    float(gm_desc["mode_excess"])
                    + float(gp_desc["mode_excess"])
                ),
                "weighted_mode_fraction": 0.5 * (
                    float(gm_desc["weighted_mode_fraction"])
                    + float(gp_desc["weighted_mode_fraction"])
                ),

                # Diagnostics only.
                "n_channels_mean": 0.5 * (
                    float(gm_desc["n_channels"])
                    + float(gp_desc["n_channels"])
                ),
                "gram_trace_mean": 0.5 * (
                    float(gm_desc["gram_trace"])
                    + float(gp_desc["gram_trace"])
                ),
            }
            rows.append(row)

        sub = [r for r in rows if r["family"] == family]
        y = [float(r["prominence"]) for r in sub]

        metrics = (
            "A_group",
            "C_group",
            "mean_cosine",
            "positive_fraction",
            "mode_excess",
            "weighted_mode_fraction",
        )

        for metric in metrics:
            rho = spearman(base, [float(r[metric]) for r in sub], y)
            seed_stats.append(
                {
                    "version": VERSION,
                    "seed": int(seed),
                    "family": family,
                    "metric": metric,
                    "n_pairs": int(len(sub)),
                    "spearman_rho_vs_prominence": float(rho),
                }
            )

    return rows, seed_stats, len(ref["groups"]), len(pairs)


# ---------------------------------------------------------------------------
# Across-seed summary
# ---------------------------------------------------------------------------

def summarize_across_seeds(seed_stats: List[dict]) -> List[dict]:
    metrics = (
        "A_group",
        "C_group",
        "mean_cosine",
        "positive_fraction",
        "mode_excess",
        "weighted_mode_fraction",
    )

    out = []

    for family in ("dephasing", "transverse"):
        for metric in metrics:
            vals = [
                float(r["spearman_rho_vs_prominence"])
                for r in seed_stats
                if r["family"] == family and r["metric"] == metric
            ]

            positive = sum(v > 0.0 for v in vals)
            negative = sum(v < 0.0 for v in vals)
            zero = sum(v == 0.0 for v in vals)

            out.append(
                {
                    "version": VERSION,
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


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Soft Spaces Phase 3.2 v32.20 — cross-seed "
            "degeneracy-closed operator-geometry test."
        )
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

    base = load_module(
        "v32_14_degeneracy_closed.py",
        "v3214_for_v3220",
    )
    v3217 = load_module(
        "v32_17_perturbative_response.py",
        "v3217_for_v3220",
    )

    if int(base.N_QUBITS) != 8:
        raise RuntimeError(
            f"v32.20 is frozen for 8Q; base reports {base.N_QUBITS}Q."
        )
    if int(base.N_TERMS) != 11:
        raise RuntimeError(
            f"v32.20 expects 11 Hamiltonian terms; "
            f"base reports {base.N_TERMS}."
        )

    seeds = [int(s) for s in args.seeds]

    all_rows: List[dict] = []
    all_seed_stats: List[dict] = []
    run_meta = []

    print(
        f"{VERSION}: operator-geometry law test across "
        f"{len(seeds)} frozen 8Q seeds"
    )

    for idx, seed in enumerate(seeds, start=1):
        print(f"\n[{idx}/{len(seeds)}] seed {seed} ...", flush=True)

        rows, stats, n_groups, n_pairs = analyze_seed(
            base=base,
            v3217=v3217,
            seed=seed,
        )

        all_rows.extend(rows)
        all_seed_stats.extend(stats)
        run_meta.append(
            {
                "seed": int(seed),
                "groups": int(n_groups),
                "usable_pairs": int(n_pairs),
            }
        )

        for family in ("dephasing", "transverse"):
            relevant = [
                r for r in stats if r["family"] == family
            ]
            bits = [
                f"{r['metric']}={r['spearman_rho_vs_prominence']:+.4f}"
                for r in relevant
            ]
            print(f"  {family:10s}: " + ", ".join(bits), flush=True)

    summary = summarize_across_seeds(all_seed_stats)

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows_path = outdir / "v32_20_rows.csv"
    seed_stats_path = outdir / "v32_20_seed_stats.csv"
    summary_csv_path = outdir / "v32_20_summary.csv"
    summary_txt_path = outdir / "v32_20_summary.txt"

    write_csv(rows_path, all_rows)
    write_csv(seed_stats_path, all_seed_stats)
    write_csv(summary_csv_path, summary)

    lines = [
        f"Soft Spaces Phase 3.2 {VERSION}",
        "Degeneracy-closed operator-geometry law",
        "",
        f"qubits={base.N_QUBITS}",
        f"dimension={base.DIM}",
        f"hamiltonian_terms={base.N_TERMS}",
        "seeds=" + ",".join(str(s) for s in seeds),
        "",
        "PREDECLARED GEOMETRY",
        "--------------------",
        "M_GK = Re Tr(T_G^dagger T_K) / (||T_G||_F ||T_K||_F)",
        "",
        "mean_cosine            : mean off-diagonal M_GK",
        "positive_fraction      : fraction of off-diagonal M_GK > 0",
        "mode_excess            : (lambda_max(M)-1)/(n_channels-1)",
        "weighted_mode_fraction : lambda_max(K)/Tr(K),",
        "                         K_GK=Re Tr(T_G^dagger T_K)",
        "",
        "ACROSS-SEED RESULTS",
        "-------------------",
    ]

    for family in ("dephasing", "transverse"):
        lines.append("")
        lines.append(f"[{family}]")

        for r in summary:
            if r["family"] != family:
                continue

            lines.append(
                f"  {r['metric']:22s} "
                f"median_rho={r['median_rho']:+.4f}  "
                f"positive={r['positive_seeds']}/{r['n_seeds']}  "
                f"sign_p={r['sign_test_p_one_sided']:.6f}"
            )

    lines += [
        "",
        "DECISION RULE",
        "-------------",
        "The operator-geometry hypothesis is supported if one or more",
        "predeclared geometric descriptors show a systematic positive",
        "cross-seed association with hotspot prominence and remain",
        "competitive with the established A_group/C_group baselines.",
        "",
        "No descriptor combination, regression, or seed-specific tuning is",
        "performed in this test.",
        "",
        "INTERPRETATION",
        "--------------",
        "A positive mode_excess or weighted_mode_fraction result would support",
        "the idea that hotspots correspond to a dominant collective alignment",
        "mode among degeneracy-closed virtual Q-space pathways.",
        "",
        "A positive mean_cosine result would support net constructive pathway",
        "alignment directly.  positive_fraction tests whether the sign pattern",
        "alone carries useful organization.",
        "",
        "RUN META",
        "--------",
    ]

    for m in run_meta:
        lines.append(
            f"seed={m['seed']} groups={m['groups']} "
            f"usable_pairs={m['usable_pairs']}"
        )

    summary_txt = "\n".join(lines) + "\n"
    summary_txt_path.write_text(summary_txt, encoding="utf-8")

    print("\n" + summary_txt)
    print("WROTE")
    print(f"  {rows_path}")
    print(f"  {seed_stats_path}")
    print(f"  {summary_csv_path}")
    print(f"  {summary_txt_path}")


if __name__ == "__main__":
    main()
