#!/usr/bin/env python3
"""
Soft Spaces Phase 3.2 — v32.21 Coherence Decomposition Law
==========================================================

Purpose
-------
Test the exact algebraic decomposition behind the robust degeneracy-closed
coherence descriptor C_group.

For degeneracy-closed virtual pathway operators

    T_G = g_eta(E - E_G) P V P_G V P,

define

    Sigma_E = sum_G T_G

and

    C_group =
        ||Sigma_E||_F
        ----------------
        sum_G ||T_G||_F .

Then exactly,

    C_group^2 = S_group + A_group,

where

    S_group =
        sum_G ||T_G||_F^2
        -------------------------
        (sum_G ||T_G||_F)^2

is a pathway-concentration / participation term, while

    A_group =
        2 sum_{G<K} Re Tr(T_G^dagger T_K)
        -----------------------------------
        (sum_G ||T_G||_F)^2

is the normalized interference term.

Scientific question
-------------------
v32.14-v32.20 showed that C_group is the most robust static descriptor across
seeds and system sizes.  v32.21 asks why.

The predeclared test compares hotspot prominence with:

    S_group              pathway concentration
    A_group              interference
    C2_group             S_group + A_group = C_group^2
    C_group              original coherence baseline

No feature fitting, regression, or parameter tuning is performed.

Interpretation
--------------
If S_group is strongly positive across seeds, C_group may outperform A_group
because hotspots require both concentration of virtual pathways and constructive
interference.

If S_group is weak while C_group remains strong, then the nonlinear combination
S_group + A_group is itself essential.

Frozen protocol
---------------
    * 8 qubits
    * 11 Hamiltonian terms
    * seeds 25042000 ... 25042011
    * dephasing and transverse perturbation families
    * same frozen REAL/NULL hotspot prominence used in prior Phase-3 tests
    * complete degeneracy-closed Q groups
    * no hyperparameter scan

Dependencies
------------
Must be beside this file:

    v32_14_degeneracy_closed.py
    v32_17_perturbative_response.py

Outputs
-------
    v32_21_rows.csv
    v32_21_seed_stats.csv
    v32_21_summary.csv
    v32_21_summary.txt

Usage
-----
    python .\\v32_21_coherence_decomposition.py
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


VERSION = "v32.21"
DEFAULT_SEEDS = tuple(range(25_042_000, 25_042_012))
DENOM_EPS = 1.0e-30


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
# Generic helpers
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
    return float(
        sum(math.comb(n, k) for k in range(k_positive, n + 1))
        / (2 ** n)
    )


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
# Exact coherence decomposition
# ---------------------------------------------------------------------------

def decomposition_from_terms(terms: List[np.ndarray]) -> dict:
    """
    Compute S_group, A_group, C_group and an explicit identity residual.

    The calculation is performed directly from the degeneracy-closed T_G
    operators rather than inferred from existing descriptors, so v32.21 also
    acts as an algebraic consistency check.
    """
    if not terms:
        return {
            "n_channels": 0,
            "term_l1": 0.0,
            "self_sum": 0.0,
            "cross_sum": 0.0,
            "S_group": 0.0,
            "A_group_direct": 0.0,
            "C_group_direct": 0.0,
            "C2_group": 0.0,
            "identity_residual": 0.0,
        }

    mats = [np.asarray(t, dtype=np.complex128) for t in terms]
    norms = np.asarray(
        [float(np.linalg.norm(t, ord="fro")) for t in mats],
        dtype=float,
    )

    term_l1 = float(np.sum(norms))
    self_sum = float(np.sum(norms * norms))

    sigma = np.sum(mats, axis=0)
    sigma2 = float(np.linalg.norm(sigma, ord="fro") ** 2)

    # By identity: ||sum T_G||^2 = self_sum + cross_sum
    cross_sum = float(sigma2 - self_sum)

    if term_l1 <= DENOM_EPS:
        return {
            "n_channels": int(len(mats)),
            "term_l1": term_l1,
            "self_sum": self_sum,
            "cross_sum": cross_sum,
            "S_group": 0.0,
            "A_group_direct": 0.0,
            "C_group_direct": 0.0,
            "C2_group": 0.0,
            "identity_residual": 0.0,
        }

    denom = term_l1 * term_l1

    S_group = float(self_sum / denom)
    A_group_direct = float(cross_sum / denom)
    C2_group = float(sigma2 / denom)
    C_group_direct = float(np.sqrt(max(C2_group, 0.0)))

    identity_residual = float(C2_group - (S_group + A_group_direct))

    return {
        "n_channels": int(len(mats)),
        "term_l1": term_l1,
        "self_sum": self_sum,
        "cross_sum": cross_sum,
        "S_group": S_group,
        "A_group_direct": A_group_direct,
        "C_group_direct": C_group_direct,
        "C2_group": C2_group,
        "identity_residual": identity_residual,
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

    frozen = base.interference_descriptors_group_closed(terms)
    dec = decomposition_from_terms(terms)

    # Consistency checks against the already-established v32.14 descriptors.
    dec["A_vs_base_residual"] = float(
        dec["A_group_direct"] - float(frozen["A_group"])
    )
    dec["C_vs_base_residual"] = float(
        dec["C_group_direct"] - float(frozen["C_group"])
    )

    return frozen, dec


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

        frozen_by_group: Dict[int, dict] = {}
        dec_by_group: Dict[int, dict] = {}

        for gi in needed_groups:
            frozen, dec = group_descriptors(
                base, ref, gi, perturbation, eta
            )
            frozen_by_group[gi] = frozen
            dec_by_group[gi] = dec

        for pair in pairs:
            gm = pair["g_minus"]
            gp = pair["g_plus"]

            fm = frozen_by_group[gm]
            fp = frozen_by_group[gp]
            dm = dec_by_group[gm]
            dp = dec_by_group[gp]

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

                # Established baselines from v32.14.
                "A_group": 0.5 * (
                    float(fm["A_group"]) + float(fp["A_group"])
                ),
                "C_group": 0.5 * (
                    float(fm["C_group"]) + float(fp["C_group"])
                ),

                # Exact decomposition components.
                "S_group": 0.5 * (
                    float(dm["S_group"]) + float(dp["S_group"])
                ),
                "C2_group": 0.5 * (
                    float(dm["C2_group"]) + float(dp["C2_group"])
                ),
                "A_group_direct": 0.5 * (
                    float(dm["A_group_direct"])
                    + float(dp["A_group_direct"])
                ),
                "C_group_direct": 0.5 * (
                    float(dm["C_group_direct"])
                    + float(dp["C_group_direct"])
                ),

                # Diagnostics only.
                "n_channels_mean": 0.5 * (
                    float(dm["n_channels"]) + float(dp["n_channels"])
                ),
                "identity_residual_maxabs": max(
                    abs(float(dm["identity_residual"])),
                    abs(float(dp["identity_residual"])),
                ),
                "A_vs_base_residual_maxabs": max(
                    abs(float(dm["A_vs_base_residual"])),
                    abs(float(dp["A_vs_base_residual"])),
                ),
                "C_vs_base_residual_maxabs": max(
                    abs(float(dm["C_vs_base_residual"])),
                    abs(float(dp["C_vs_base_residual"])),
                ),
            }
            rows.append(row)

        sub = [r for r in rows if r["family"] == family]
        y = [float(r["prominence"]) for r in sub]

        for metric in (
            "S_group",
            "A_group",
            "C2_group",
            "C_group",
        ):
            rho = spearman(
                base,
                [float(r[metric]) for r in sub],
                y,
            )

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
    out = []

    for family in ("dephasing", "transverse"):
        for metric in ("S_group", "A_group", "C2_group", "C_group"):
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
            "Soft Spaces Phase 3.2 v32.21 — exact coherence "
            "decomposition across 12 frozen 8Q seeds."
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
        "v3214_for_v3221",
    )
    v3217 = load_module(
        "v32_17_perturbative_response.py",
        "v3217_for_v3221",
    )

    if int(base.N_QUBITS) != 8:
        raise RuntimeError(
            f"v32.21 is frozen for 8Q; base reports {base.N_QUBITS}Q."
        )

    if int(base.N_TERMS) != 11:
        raise RuntimeError(
            f"v32.21 expects 11 Hamiltonian terms; "
            f"base reports {base.N_TERMS}."
        )

    seeds = [int(s) for s in args.seeds]

    all_rows: List[dict] = []
    all_seed_stats: List[dict] = []
    run_meta = []

    print(
        f"{VERSION}: coherence-decomposition test across "
        f"{len(seeds)} frozen 8Q seeds"
    )
    print("  C_group^2 = S_group + A_group")

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

    rows_path = outdir / "v32_21_rows.csv"
    seed_stats_path = outdir / "v32_21_seed_stats.csv"
    summary_csv_path = outdir / "v32_21_summary.csv"
    summary_txt_path = outdir / "v32_21_summary.txt"

    write_csv(rows_path, all_rows)
    write_csv(seed_stats_path, all_seed_stats)
    write_csv(summary_csv_path, summary)

    max_identity_residual = max(
        abs(float(r["identity_residual_maxabs"]))
        for r in all_rows
    ) if all_rows else float("nan")

    max_A_residual = max(
        abs(float(r["A_vs_base_residual_maxabs"]))
        for r in all_rows
    ) if all_rows else float("nan")

    max_C_residual = max(
        abs(float(r["C_vs_base_residual_maxabs"]))
        for r in all_rows
    ) if all_rows else float("nan")

    lines = [
        f"Soft Spaces Phase 3.2 {VERSION}",
        "Exact coherence decomposition law",
        "",
        f"qubits={base.N_QUBITS}",
        f"dimension={base.DIM}",
        f"hamiltonian_terms={base.N_TERMS}",
        "seeds=" + ",".join(str(s) for s in seeds),
        "",
        "EXACT IDENTITY",
        "--------------",
        "C_group^2 = S_group + A_group",
        "",
        "S_group = sum_G ||T_G||_F^2 / (sum_G ||T_G||_F)^2",
        "A_group = 2 sum_{G<K} Re Tr(T_G^dagger T_K)",
        "          / (sum_G ||T_G||_F)^2",
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
                f"  {r['metric']:9s} "
                f"median_rho={r['median_rho']:+.4f}  "
                f"positive={r['positive_seeds']}/{r['n_seeds']}  "
                f"sign_p={r['sign_test_p_one_sided']:.6f}"
            )

    lines += [
        "",
        "ALGEBRAIC CONSISTENCY",
        "---------------------",
        f"max |C2-(S+A)|              = {max_identity_residual:.6e}",
        f"max |A_direct-A_base|       = {max_A_residual:.6e}",
        f"max |C_direct-C_base|       = {max_C_residual:.6e}",
        "",
        "DECISION RULE",
        "-------------",
        "If S_group is systematically positive and competitive with A_group,",
        "then the robustness of C_group is naturally explained by a combination",
        "of pathway concentration and constructive interference.",
        "",
        "If S_group is weak while C_group remains strong, the exact nonlinear",
        "combination S_group + A_group is itself the important object.",
        "",
        "No regression, feature combination, or parameter tuning is used.",
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
