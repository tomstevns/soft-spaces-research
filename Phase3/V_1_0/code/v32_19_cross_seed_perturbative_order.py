#!/usr/bin/env python3
"""
Soft Spaces Phase 3.2 — v32.19 Cross-Seed Perturbative-Order Test
================================================================

Purpose
-------
Test the symmetry-selected perturbative-order hypothesis across the same
12 frozen 8Q Hamiltonian seeds used throughout Phase 3:

    dephasing  -> first-order frozen-P/Q response
    transverse -> second-order frozen-P/Q response

The script compares those response laws with the static degeneracy-closed
baseline descriptors A_group and C_group.

Primary quantities
------------------
Dephasing:
    R1_E = || d Sigma_E / d lambda ||_F
    D1_E = | d ||Sigma_E||_F^2 / d lambda |

Transverse:
    R2_E = || d^2 Sigma_E / d lambda^2 ||_F
    D2_E = | d^2 ||Sigma_E||_F^2 / d lambda^2 |

Static baselines:
    A_group
    C_group

Frozen model
------------
    * 8 qubits
    * 11 Hamiltonian terms
    * seeds 25042000 ... 25042011
    * same frozen REAL/NULL hotspot prominence from v32.14/v32.17
    * P and Q frozen at lambda=0
    * no feature fitting
    * no delta optimization

The transverse second derivative is evaluated with the validated five-point
central stencil at a PREDECLARED step h = 1e-3 by default:

    f''(0) ~= [-f(2h)+16f(h)-30f(0)+16f(-h)-f(-2h)] / (12 h^2)

v32.18 showed stable transverse rankings and excellent 3-point/5-point
agreement over a broad delta window, so this script freezes h rather than
choosing a seed-specific value.

Dependencies
------------
Must be beside this file:

    v32_14_degeneracy_closed.py
    v32_17_perturbative_response.py
    v32_18_second_order_transverse.py

Outputs
-------
    v32_19_seed_stats.csv
    v32_19_rows.csv
    v32_19_summary.csv
    v32_19_summary.txt

Usage
-----
    python .\\v32_19_cross_seed_perturbative_order.py

Optional:
    python .\\v32_19_cross_seed_perturbative_order.py --h2 0.001
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


VERSION = "v32.19"
DEFAULT_SEEDS = tuple(range(25_042_000, 25_042_012))
DEFAULT_H2 = 1.0e-3


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
    """
    One-sided exact sign-test p-value under p=0.5:
        P[X >= k_positive], X ~ Binomial(n, 0.5)
    """
    if n <= 0:
        return float("nan")

    total = 0.0
    for k in range(k_positive, n + 1):
        total += math.comb(n, k)
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


def static_group_descriptors(base, ref, gi, perturbation, eta):
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
# One seed
# ---------------------------------------------------------------------------

def analyze_seed(base, v3217, v3218, seed: int, h2: float):
    eta = float(base.ENERGY_REG)
    ref = v3217.build_reference(base, seed)
    pairs = pair_records(ref)

    if not pairs:
        raise RuntimeError(f"Seed {seed}: no usable +/-E pairs.")

    rows: List[dict] = []

    # ------------------------------
    # Dephasing: first order
    # ------------------------------
    family = "dephasing"
    perturbation = ref["perturbations"][family]

    needed_groups = sorted(
        set(p["g_minus"] for p in pairs)
        | set(p["g_plus"] for p in pairs)
    )

    prepared_d: Dict[int, dict] = {}
    static_d: Dict[int, dict] = {}

    for gi in needed_groups:
        prepared_d[gi] = v3217.prepare_target(
            base, ref, gi, perturbation
        )
        static_d[gi] = static_group_descriptors(
            base, ref, gi, perturbation, eta
        )

    for pair in pairs:
        gm = pair["g_minus"]
        gp = pair["g_plus"]

        pm = prepared_d[gm]
        pp = prepared_d[gp]
        sm = static_d[gm]
        sp = static_d[gp]

        rows.append(
            {
                "version": VERSION,
                "seed": int(seed),
                "family": family,
                "order": 1,
                "g_minus": int(gm),
                "g_plus": int(gp),
                "E_abs": float(pair["E_abs"]),
                "prominence": float(pair["prominence"]),
                "A_group": 0.5 * (
                    float(sm["A_group"]) + float(sp["A_group"])
                ),
                "C_group": 0.5 * (
                    float(sm["C_group"]) + float(sp["C_group"])
                ),
                "R_response": 0.5 * (
                    float(pm["R_E"]) + float(pp["R_E"])
                ),
                "D_response": 0.5 * (
                    float(pm["D_E"]) + float(pp["D_E"])
                ),
                "response_label_R": "R1_E",
                "response_label_D": "D1_E",
                "h2": "",
            }
        )

    # ------------------------------
    # Transverse: second order
    # ------------------------------
    family = "transverse"
    perturbation = ref["perturbations"][family]

    prepared_t: Dict[int, dict] = {}
    static_t: Dict[int, dict] = {}

    for gi in needed_groups:
        prepared_t[gi] = v3217.prepare_target(
            base, ref, gi, perturbation
        )
        static_t[gi] = static_group_descriptors(
            base, ref, gi, perturbation, eta
        )

    for pair in pairs:
        gm = pair["g_minus"]
        gp = pair["g_plus"]

        pm = v3218.second_derivative_5point(
            prepared_t[gm], h2, eta
        )
        pp = v3218.second_derivative_5point(
            prepared_t[gp], h2, eta
        )

        sm = static_t[gm]
        sp = static_t[gp]

        rows.append(
            {
                "version": VERSION,
                "seed": int(seed),
                "family": family,
                "order": 2,
                "g_minus": int(gm),
                "g_plus": int(gp),
                "E_abs": float(pair["E_abs"]),
                "prominence": float(pair["prominence"]),
                "A_group": 0.5 * (
                    float(sm["A_group"]) + float(sp["A_group"])
                ),
                "C_group": 0.5 * (
                    float(sm["C_group"]) + float(sp["C_group"])
                ),
                "R_response": 0.5 * (
                    float(pm["R2_E"]) + float(pp["R2_E"])
                ),
                "D_response": 0.5 * (
                    float(pm["D2_E"]) + float(pp["D2_E"])
                ),
                "response_label_R": "R2_E",
                "response_label_D": "D2_E",
                "h2": float(h2),
            }
        )

    # Per-seed statistics
    seed_stats: List[dict] = []

    for family in ("dephasing", "transverse"):
        sub = [r for r in rows if r["family"] == family]
        y = [float(r["prominence"]) for r in sub]

        for metric in ("A_group", "C_group", "R_response", "D_response"):
            rho = spearman(base, [float(r[metric]) for r in sub], y)

            label = metric
            if metric == "R_response":
                label = "R1_E" if family == "dephasing" else "R2_E"
            elif metric == "D_response":
                label = "D1_E" if family == "dephasing" else "D2_E"

            seed_stats.append(
                {
                    "version": VERSION,
                    "seed": int(seed),
                    "family": family,
                    "metric": label,
                    "order": 1 if family == "dephasing" else 2,
                    "n_pairs": int(len(sub)),
                    "spearman_rho_vs_prominence": float(rho),
                    "h2": "" if family == "dephasing" else float(h2),
                }
            )

    return rows, seed_stats, len(ref["groups"]), len(pairs)


# ---------------------------------------------------------------------------
# Across-seed summary
# ---------------------------------------------------------------------------

def summarize_across_seeds(seed_stats: List[dict]) -> List[dict]:
    out = []

    for family in ("dephasing", "transverse"):
        metrics = (
            ("A_group", "C_group", "R1_E", "D1_E")
            if family == "dephasing"
            else ("A_group", "C_group", "R2_E", "D2_E")
        )

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
                    "order": 1 if family == "dephasing" else 2,
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
            "Soft Spaces Phase 3.2 v32.19 — 12-seed symmetry-selected "
            "perturbative-order test."
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
        "--h2",
        type=float,
        default=DEFAULT_H2,
        help=(
            "Frozen transverse 5-point second-derivative step "
            f"(default {DEFAULT_H2})."
        ),
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

    if not np.isfinite(args.h2) or args.h2 <= 0.0:
        raise ValueError("--h2 must be finite and > 0.")

    base = load_module(
        "v32_14_degeneracy_closed.py",
        "v3214_for_v3219",
    )
    v3217 = load_module(
        "v32_17_perturbative_response.py",
        "v3217_for_v3219",
    )
    v3218 = load_module(
        "v32_18_second_order_transverse.py",
        "v3218_for_v3219",
    )

    if int(base.N_QUBITS) != 8:
        raise RuntimeError(
            f"v32.19 is frozen for 8Q; base reports {base.N_QUBITS}Q."
        )
    if int(base.N_TERMS) != 11:
        raise RuntimeError(
            f"v32.19 expects 11 Hamiltonian terms; "
            f"base reports {base.N_TERMS}."
        )

    seeds = [int(s) for s in args.seeds]

    all_rows: List[dict] = []
    all_seed_stats: List[dict] = []
    run_meta = []

    print(
        f"{VERSION}: cross-seed perturbative-order test "
        f"for {len(seeds)} seeds"
    )
    print(
        "  dephasing  -> first order (R1_E, D1_E)\n"
        "  transverse -> second order (R2_E, D2_E)\n"
        f"  transverse h2={args.h2:g}"
    )

    for idx, seed in enumerate(seeds, start=1):
        print(f"\n[{idx}/{len(seeds)}] seed {seed} ...", flush=True)

        rows, stats, n_groups, n_pairs = analyze_seed(
            base=base,
            v3217=v3217,
            v3218=v3218,
            seed=seed,
            h2=float(args.h2),
        )

        all_rows.extend(rows)
        all_seed_stats.extend(stats)
        run_meta.append(
            {
                "seed": seed,
                "groups": n_groups,
                "usable_pairs": n_pairs,
            }
        )

        # Compact progress report
        for family in ("dephasing", "transverse"):
            relevant = [
                r for r in stats if r["family"] == family
            ]
            bits = []
            for r in relevant:
                bits.append(
                    f"{r['metric']}={r['spearman_rho_vs_prominence']:+.4f}"
                )
            print(f"  {family:10s}: " + ", ".join(bits), flush=True)

    summary = summarize_across_seeds(all_seed_stats)

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows_path = outdir / "v32_19_rows.csv"
    seed_stats_path = outdir / "v32_19_seed_stats.csv"
    summary_csv_path = outdir / "v32_19_summary.csv"
    summary_txt_path = outdir / "v32_19_summary.txt"

    write_csv(rows_path, all_rows)
    write_csv(seed_stats_path, all_seed_stats)
    write_csv(summary_csv_path, summary)

    lines = [
        f"Soft Spaces Phase 3.2 {VERSION}",
        "Cross-seed symmetry-selected perturbative-order test",
        "",
        f"qubits={base.N_QUBITS}",
        f"dimension={base.DIM}",
        f"hamiltonian_terms={base.N_TERMS}",
        "seeds=" + ",".join(str(s) for s in seeds),
        f"transverse_h2={args.h2:g}",
        "",
        "FROZEN HYPOTHESIS",
        "-----------------",
        "dephasing  : first-order local response",
        "transverse : second-order local response",
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
                f"  {r['metric']:8s} "
                f"median_rho={r['median_rho']:+.4f}  "
                f"positive={r['positive_seeds']}/{r['n_seeds']}  "
                f"sign_p={r['sign_test_p_one_sided']:.6f}"
            )

    lines += [
        "",
        "DECISION RULE",
        "-------------",
        "The perturbative-order hypothesis is supported if:",
        "  * dephasing first-order R1_E/D1_E is systematically positive",
        "    across seeds, and",
        "  * transverse second-order R2_E/D2_E is systematically positive",
        "    across seeds,",
        "while numerical validation from v32.17.1/v32.18 remains satisfied.",
        "",
        "Static A_group and C_group are retained as baselines, not fitted",
        "ingredients of the response law.",
        "",
        "This test does not claim asymptotic scaling and does not yet include",
        "full spectral/projector motion.",
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
