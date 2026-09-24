#!/usr/bin/env python3
"""
Soft Spaces / DLBCL Phase 4
v41.12 — Feature Stability / Stability Selection

Purpose:
Test whether the Soft-Spaces-derived 16-gene panel is stable under repeated
patient resampling of the DEVELOPMENT cohort.

IMPORTANT:
This script is a reproducibility scaffold. Two project-specific functions must
be connected to the already-frozen v41 pipeline:
    1) load_frozen_development_data()
    2) select_softspaces_panel()

Do NOT change preprocessing, feature universe, or selection logic here.
Do NOT use GSE87371 labels for feature discovery.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = "v41.12"
PANEL_SIZE = 16
DEFAULT_RESAMPLES = 500
DEFAULT_SAMPLE_FRACTION = 0.80
DEFAULT_RANDOM_SEED = 4112001


def load_frozen_development_data():
    """
    Return:
        X : pandas.DataFrame, rows=patients, cols=features/genes
        y : array-like, ABC/GCB labels
        metadata : dict

    Connect this to the corrected GSE10846 development-data loader used after
    the v41.7 preprocessing correction.
    """
    raise NotImplementedError(
        "Connect load_frozen_development_data() to the frozen GSE10846 loader."
    )


def select_softspaces_panel(X, y, panel_size=PANEL_SIZE):
    """
    Return:
        selected_genes : list[str] in rank order
        scores : pandas.Series or dict, higher = better

    Connect this to the exact frozen Soft-Spaces selector that produced the
    16-gene panel used before v41.11.
    """
    raise NotImplementedError(
        "Connect select_softspaces_panel() to the frozen Soft-Spaces selector."
    )


def select_null_panel(X, panel_size, rng):
    cols = np.asarray(X.columns, dtype=object)
    idx = rng.choice(len(cols), size=panel_size, replace=False)
    return [str(cols[i]) for i in idx]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def pairwise_jaccards(panels):
    vals = []
    for i in range(len(panels)):
        for j in range(i + 1, len(panels)):
            vals.append(jaccard(panels[i], panels[j]))
    return np.asarray(vals, float)


def bootstrap_mean_ci(values, rng, n_boot=20000):
    values = np.asarray(values, float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def run(args):
    X, y, metadata = load_frozen_development_data()
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas.DataFrame.")

    y = np.asarray(y)
    rng = np.random.default_rng(args.seed)

    frozen_panel, _ = select_softspaces_panel(X, y, args.panel_size)
    frozen_panel = [str(g) for g in frozen_panel]
    frozen_set = set(frozen_panel)

    selection_counts = Counter()
    rank_lists = {}
    real_panels = []
    null_panels = []
    frozen_overlap = []

    classes = np.unique(y)

    for r in range(args.n_resamples):
        chosen = []

        if len(classes) == 2:
            for cls in classes:
                idx_cls = np.where(y == cls)[0]
                n_cls = max(1, int(round(len(idx_cls) * args.sample_fraction)))
                chosen.extend(rng.choice(idx_cls, size=n_cls, replace=True))
            idx = np.asarray(chosen, int)
            rng.shuffle(idx)
        else:
            n = max(2, int(round(len(X) * args.sample_fraction)))
            idx = rng.choice(len(X), size=n, replace=True)

        Xr = X.iloc[idx].copy()
        yr = y[idx]

        selected, scores = select_softspaces_panel(Xr, yr, args.panel_size)
        selected = [str(g) for g in selected]

        if len(selected) != args.panel_size:
            raise ValueError(
                f"Selector returned {len(selected)} genes; expected {args.panel_size}."
            )

        real_panels.append(selected)
        frozen_overlap.append(len(set(selected) & frozen_set) / args.panel_size)

        for g in selected:
            selection_counts[g] += 1

        if isinstance(scores, dict):
            scores = pd.Series(scores, dtype=float)
        elif not isinstance(scores, pd.Series):
            raise TypeError("scores must be dict or pandas.Series")

        scores = scores.reindex(X.columns, fill_value=-np.inf)
        ordered = scores.sort_values(ascending=False, kind="mergesort").index
        ranks = {str(g): i + 1 for i, g in enumerate(ordered)}

        for g, rank in ranks.items():
            rank_lists.setdefault(g, []).append(rank)

        null_panels.append(select_null_panel(Xr, args.panel_size, rng))

        if (r + 1) % max(1, args.n_resamples // 20) == 0:
            print(f"[v41.12] {r+1}/{args.n_resamples} resamples", flush=True)

    rows = []
    for g in X.columns:
        gs = str(g)
        ranks = np.asarray(rank_lists.get(gs, []), float)
        rows.append({
            "gene": gs,
            "selection_count": int(selection_counts[gs]),
            "selection_frequency": selection_counts[gs] / args.n_resamples,
            "mean_rank": float(np.mean(ranks)) if len(ranks) else np.nan,
            "median_rank": float(np.median(ranks)) if len(ranks) else np.nan,
            "in_frozen_16": gs in frozen_set,
        })

    table = pd.DataFrame(rows).sort_values(
        ["selection_frequency", "mean_rank"],
        ascending=[False, True],
        kind="mergesort"
    )

    real_j = pairwise_jaccards(real_panels)
    null_j = pairwise_jaccards(null_panels)

    ci_rng = np.random.default_rng(args.seed + 99)
    real_mean, real_lo, real_hi = bootstrap_mean_ci(real_j, ci_rng)
    null_mean, null_lo, null_hi = bootstrap_mean_ci(null_j, ci_rng)
    ov_mean, ov_lo, ov_hi = bootstrap_mean_ci(
        np.asarray(frozen_overlap, float), ci_rng
    )

    stable80 = table.loc[
        table["selection_frequency"] >= 0.80, "gene"
    ].tolist()
    stable60 = table.loc[
        table["selection_frequency"] >= 0.60, "gene"
    ].tolist()
    stable50 = table.loc[
        table["selection_frequency"] >= 0.50, "gene"
    ].tolist()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "v41_12_gene_selection_frequencies.csv"
    txt_path = outdir / "v41_12_feature_stability_summary.txt"

    table.to_csv(csv_path, index=False)

    lines = [
        f"=== Soft Spaces / DLBCL {VERSION} FEATURE STABILITY ===",
        f"Development samples: {len(X)}",
        f"Feature universe: {X.shape[1]}",
        f"Resamples: {args.n_resamples}",
        f"Sample fraction: {args.sample_fraction}",
        f"Panel size: {args.panel_size}",
        "",
        "Frozen full-development 16-gene panel:",
    ]
    lines += [f"  {i:2d}. {g}" for i, g in enumerate(frozen_panel, 1)]
    lines += [
        "",
        f"REAL mean pairwise Jaccard: {real_mean:.6f}",
        f"REAL 95% bootstrap CI: [{real_lo:.6f}, {real_hi:.6f}]",
        f"NULL mean pairwise Jaccard: {null_mean:.6f}",
        f"NULL 95% bootstrap CI: [{null_lo:.6f}, {null_hi:.6f}]",
        "",
        f"Mean overlap with frozen panel: {ov_mean:.6f}",
        f"95% bootstrap CI: [{ov_lo:.6f}, {ov_hi:.6f}]",
        "",
        f"Stable core >=80% ({len(stable80)} genes):",
        "  " + ", ".join(stable80),
        "",
        f"Stable core >=60% ({len(stable60)} genes):",
        "  " + ", ".join(stable60),
        "",
        f"Stable core >=50% ({len(stable50)} genes):",
        "  " + ", ".join(stable50),
        "",
        "Interpretation:",
        "  >=0.80 selection frequency: very stable",
        "  0.60-0.80: moderately stable",
        "  0.50-0.60: recurrent but weaker",
        "  <0.50: unstable/context-sensitive",
        "",
        "Caution:",
        "  This measures feature-selection stability, not causal biology and",
        "  not external classification performance.",
    ]

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("\nWrote:")
    print(" ", csv_path)
    print(" ", txt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument(
        "--sample-fraction", type=float, default=DEFAULT_SAMPLE_FRACTION
    )
    parser.add_argument("--panel-size", type=int, default=PANEL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    if not (0.5 <= args.sample_fraction <= 1.0):
        raise SystemExit("--sample-fraction must be between 0.5 and 1.0.")

    print(f"=== {VERSION} FEATURE STABILITY ===")
    print(f"Resamples: {args.n_resamples}")
    print(f"Sample fraction: {args.sample_fraction}")
    print(f"Panel size: {args.panel_size}")
    run(args)


if __name__ == "__main__":
    main()
