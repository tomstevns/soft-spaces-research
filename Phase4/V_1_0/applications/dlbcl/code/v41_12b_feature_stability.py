#!/usr/bin/env python3
"""
Soft Spaces / DLBCL Phase 4
v41.12b — Feature Stability / Stability Selection (memory-safe)

This is a corrected version of v41.12.

Scientific design is unchanged:
- GSE10846 development cohort only
- corrected v41.7 preprocessing
- v41.3 Soft-Spaces operator/scoring/ranking
- stratified bootstrap resampling
- 16-gene panel
- matched random NULL from the same resample-specific variance pool

Correction
----------
The original v41.12 successfully completed 500 resamples but could exhaust RAM
when bootstrapping all pairwise Jaccard values at once.

v41.12b keeps the same stability experiment but replaces the memory-heavy
bootstrap with a compact per-resample mean-Jaccard summary.

For each panel i:
    mean_jaccard_i = mean_j Jaccard(panel_i, panel_j), j != i

This produces only N values for N resamples, so bootstrap confidence intervals
remain memory-safe even for 500+ resamples.

Outputs
-------
v41_12b_gene_selection_frequencies.csv
v41_12b_resample_panels.csv
v41_12b_feature_stability_summary.txt
v41_12b_feature_stability_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import v41_7_corrected_preprocessing_audit as v7
import v41_3_softspaces_mapping_probe as v3


VERSION = "v41.12b"
PANEL_SIZE = 16
DEFAULT_RESAMPLES = 500
DEFAULT_SAMPLE_FRACTION = 0.80
DEFAULT_SEED = 4112001
DEFAULT_BOOTSTRAP_REPS = 20000


def project_dir_from_script():
    return Path(__file__).resolve().parent.parent


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_locked_inputs(project_dir):
    X, y, genes, folds, splits_path = v7.load_inputs(project_dir)

    if X.shape != (350, 22154):
        raise RuntimeError(f"Unexpected locked input shape: {X.shape}")

    counts = Counter(y.tolist())
    if counts != Counter({0: 183, 1: 167}):
        raise RuntimeError(f"Unexpected class counts: {counts}")

    return X, y, genes, folds, splits_path


def softspaces_panel_from_raw(X_raw, y, genes, panel_size):
    # Corrected preprocessing:
    # raw variance -> Top-256 -> scaling
    Xtr, _, pool_genes, chosen_global, _ = v7.preprocess_corrected(
        X_raw,
        X_raw,
        genes,
    )

    H = v3.class_contrast_operator(Xtr, y)
    _, coupling, diagnostics = v3.softspaces_decompose_and_score(H)
    rank = v3.stable_rank(coupling, pool_genes)

    top = rank[:panel_size]

    return (
        [str(g) for g in pool_genes[top]],
        [float(coupling[i]) for i in top],
        np.asarray(pool_genes, dtype=object),
        np.asarray(chosen_global, dtype=int),
        diagnostics,
    )


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def per_panel_mean_jaccard(panels):
    """
    Memory-safe stability vector of length N.

    Entry i is the mean Jaccard overlap between panel i and all other panels.
    """
    n = len(panels)

    if n == 0:
        return np.asarray([], dtype=float)

    if n == 1:
        return np.asarray([1.0], dtype=float)

    sums = np.zeros(n, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            v = jaccard(panels[i], panels[j])
            sums[i] += v
            sums[j] += v

    return sums / (n - 1)


def bootstrap_mean_ci(values, seed, reps=DEFAULT_BOOTSTRAP_REPS, chunk=1000):
    """
    Memory-safe bootstrap of the mean.

    Bootstraps in chunks rather than creating a huge reps x N array.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    if len(values) == 1:
        v = float(values[0])
        return v, v, v

    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)

    pos = 0
    while pos < reps:
        m = min(chunk, reps - pos)
        idx = rng.integers(
            0,
            len(values),
            size=(m, len(values)),
        )
        means[pos:pos + m] = values[idx].mean(axis=1)
        pos += m

    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def stratified_bootstrap_indices(y, fraction, rng):
    out = []

    for cls in sorted(np.unique(y)):
        idx = np.where(y == cls)[0]
        n_take = max(1, int(round(len(idx) * fraction)))
        sampled = rng.choice(idx, size=n_take, replace=True)
        out.extend(sampled.tolist())

    out = np.asarray(out, dtype=int)
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--n-resamples",
        type=int,
        default=DEFAULT_RESAMPLES,
    )
    ap.add_argument(
        "--sample-fraction",
        type=float,
        default=DEFAULT_SAMPLE_FRACTION,
    )
    ap.add_argument(
        "--panel-size",
        type=int,
        default=PANEL_SIZE,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    ap.add_argument(
        "--bootstrap-reps",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPS,
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    args = ap.parse_args()

    if not 0.5 <= args.sample_fraction <= 1.0:
        raise SystemExit("--sample-fraction must be between 0.5 and 1.0")

    project_dir = project_dir_from_script()
    outdir = args.output_dir or (project_dir / "results" / "stability")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== {VERSION} FEATURE STABILITY ===")
    print("Project:", project_dir)
    print("Resamples:", args.n_resamples)
    print("Sample fraction:", args.sample_fraction)
    print("Panel size:", args.panel_size)
    print("Bootstrap reps:", args.bootstrap_reps)

    X, y, genes, folds, splits_path = load_locked_inputs(project_dir)

    print("Locked matrix:", X.shape)
    print(
        "Class counts: GCB=",
        int(np.sum(y == 0)),
        "ABC=",
        int(np.sum(y == 1)),
    )

    (
        frozen_panel,
        frozen_scores,
        frozen_pool,
        _,
        frozen_diag,
    ) = softspaces_panel_from_raw(
        X,
        y,
        genes,
        args.panel_size,
    )

    print("\nFull-development 16-gene panel:")
    for i, (g, s) in enumerate(zip(frozen_panel, frozen_scores), 1):
        print(f"{i:2d}. {g:<20} score={s:.8f}")

    rng = np.random.default_rng(args.seed)

    selection_counts = Counter()
    variance_pool_counts = Counter()

    real_panels = []
    null_panels = []
    overlap_frozen = []
    panel_rows = []

    for r in range(args.n_resamples):
        idx = stratified_bootstrap_indices(
            y,
            args.sample_fraction,
            rng,
        )

        Xr = X[idx]
        yr = y[idx]

        (
            panel,
            scores,
            pool_genes,
            _,
            diag,
        ) = softspaces_panel_from_raw(
            Xr,
            yr,
            genes,
            args.panel_size,
        )

        real_panels.append(panel)

        for g in panel:
            selection_counts[g] += 1

        for g in pool_genes:
            variance_pool_counts[str(g)] += 1

        null_idx = rng.choice(
            len(pool_genes),
            size=args.panel_size,
            replace=False,
        )
        null_panel = [str(pool_genes[i]) for i in null_idx]
        null_panels.append(null_panel)

        overlap = len(set(panel) & set(frozen_panel)) / args.panel_size
        overlap_frozen.append(overlap)

        panel_rows.append(
            {
                "resample": r,
                "n_samples": int(len(idx)),
                "n_gcb": int(np.sum(yr == 0)),
                "n_abc": int(np.sum(yr == 1)),
                "overlap_fraction_with_frozen16": float(overlap),
                "panel": "|".join(panel),
                "null_panel": "|".join(null_panel),
                "coupling_mean": diag.get("coupling_mean"),
                "coupling_median": diag.get("coupling_median"),
                "coupling_max": diag.get("coupling_max"),
                "topP_abs_fraction": diag.get("topP_abs_fraction"),
            }
        )

        if (r + 1) % max(1, args.n_resamples // 20) == 0:
            print(
                f"[v41.12b] {r + 1}/{args.n_resamples}",
                flush=True,
            )

    # Gene-level table
    frozen_set = set(frozen_panel)
    gene_rows = []

    for g in genes.astype(str):
        gene_rows.append(
            {
                "gene": g,
                "selection_count": int(selection_counts[g]),
                "selection_frequency": float(
                    selection_counts[g] / args.n_resamples
                ),
                "variance_pool_count": int(variance_pool_counts[g]),
                "variance_pool_frequency": float(
                    variance_pool_counts[g] / args.n_resamples
                ),
                "in_frozen_16": g in frozen_set,
            }
        )

    gene_df = pd.DataFrame(gene_rows).sort_values(
        [
            "selection_frequency",
            "variance_pool_frequency",
            "gene",
        ],
        ascending=[False, False, True],
        kind="mergesort",
    )

    panel_df = pd.DataFrame(panel_rows)

    # Memory-safe stability summaries
    print("\n[v41.12b] Computing memory-safe REAL stability summary...")
    real_stability = per_panel_mean_jaccard(real_panels)

    print("[v41.12b] Computing memory-safe NULL stability summary...")
    null_stability = per_panel_mean_jaccard(null_panels)

    print("[v41.12b] Bootstrapping compact stability vectors...")
    real_mean, real_lo, real_hi = bootstrap_mean_ci(
        real_stability,
        args.seed + 100,
        reps=args.bootstrap_reps,
    )
    null_mean, null_lo, null_hi = bootstrap_mean_ci(
        null_stability,
        args.seed + 101,
        reps=args.bootstrap_reps,
    )
    ov_mean, ov_lo, ov_hi = bootstrap_mean_ci(
        np.asarray(overlap_frozen, dtype=float),
        args.seed + 102,
        reps=args.bootstrap_reps,
    )

    stable80 = gene_df.loc[
        gene_df.selection_frequency >= 0.80,
        "gene",
    ].tolist()

    stable60 = gene_df.loc[
        gene_df.selection_frequency >= 0.60,
        "gene",
    ].tolist()

    stable50 = gene_df.loc[
        gene_df.selection_frequency >= 0.50,
        "gene",
    ].tolist()

    frozen_stats = gene_df[
        gene_df.in_frozen_16
    ].sort_values(
        "selection_frequency",
        ascending=False,
    )

    # Outputs
    gene_csv = outdir / "v41_12b_gene_selection_frequencies.csv"
    panel_csv = outdir / "v41_12b_resample_panels.csv"
    txt = outdir / "v41_12b_feature_stability_summary.txt"
    manifest = outdir / "v41_12b_feature_stability_manifest.json"

    gene_df.to_csv(gene_csv, index=False)
    panel_df.to_csv(panel_csv, index=False)

    manifest.write_text(
        json.dumps(
            {
                "version": VERSION,
                "development_dataset": "GSE10846",
                "locked_input_shape": list(X.shape),
                "class_counts": {
                    "GCB": int(np.sum(y == 0)),
                    "ABC": int(np.sum(y == 1)),
                },
                "n_resamples": args.n_resamples,
                "sample_fraction": args.sample_fraction,
                "panel_size": args.panel_size,
                "random_seed": args.seed,
                "bootstrap_reps": args.bootstrap_reps,
                "variance_pool": int(
                    getattr(v7, "VARIANCE_POOL", len(frozen_pool))
                ),
                "softspaces_P_DIM": int(
                    getattr(v3, "P_DIM", -1)
                ),
                "frozen_full_development_panel": frozen_panel,
                "locked_cv_splits": str(splits_path),
                "locked_cv_splits_sha256": sha256_file(splits_path),
                "preprocessing_source":
                    "v41_7_corrected_preprocessing_audit.py",
                "ranking_source":
                    "v41_3_softspaces_mapping_probe.py",
                "null_definition":
                    "random 16 from same resample-specific variance pool",
                "stability_ci_method":
                    "bootstrap of per-panel mean Jaccard vectors",
                "external_labels_used": False,
                "GSE87371_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        f"=== {VERSION} FEATURE STABILITY ===",
        "",
        f"Samples: {X.shape[0]}",
        f"Genes: {X.shape[1]}",
        f"Resamples: {args.n_resamples}",
        f"Sample fraction: {args.sample_fraction}",
        f"Panel size: {args.panel_size}",
        f"VARIANCE_POOL: {getattr(v7, 'VARIANCE_POOL', 'unknown')}",
        f"P_DIM: {getattr(v3, 'P_DIM', 'unknown')}",
        "",
        "Full-development 16-gene panel:",
    ]

    lines += [
        f"{i:2d}. {g:<20} score={s:.8f}"
        for i, (g, s) in enumerate(
            zip(frozen_panel, frozen_scores),
            1,
        )
    ]

    lines += [
        "",
        "PANEL STABILITY",
        "---------------",
        f"REAL mean per-panel Jaccard: {real_mean:.6f}",
        f"REAL 95% bootstrap CI: [{real_lo:.6f}, {real_hi:.6f}]",
        f"NULL mean per-panel Jaccard: {null_mean:.6f}",
        f"NULL 95% bootstrap CI: [{null_lo:.6f}, {null_hi:.6f}]",
        "",
        "OVERLAP WITH FROZEN 16",
        "----------------------",
        f"Mean overlap fraction: {ov_mean:.6f}",
        f"95% bootstrap CI: [{ov_lo:.6f}, {ov_hi:.6f}]",
        "",
        f"Stable core >=80% ({len(stable80)}): "
        + (", ".join(stable80) if stable80 else "(none)"),
        f"Stable core >=60% ({len(stable60)}): "
        + (", ".join(stable60) if stable60 else "(none)"),
        f"Stable core >=50% ({len(stable50)}): "
        + (", ".join(stable50) if stable50 else "(none)"),
        "",
        "Frozen 16 selection frequencies:",
    ]

    for _, row in frozen_stats.iterrows():
        lines.append(
            f"{row.gene:<20} "
            f"selection={row.selection_frequency:.3f} "
            f"pool={row.variance_pool_frequency:.3f}"
        )

    lines += [
        "",
        "NOTE",
        "----",
        "v41.12b uses the same resampling experiment as v41.12.",
        "Only the confidence-interval implementation was changed to avoid",
        "the RAM explosion caused by bootstrapping all pairwise Jaccards.",
        "",
        "This measures feature-selection stability only.",
        "It does not establish causal biology, external performance,",
        "clinical utility, or quantum advantage.",
    ]

    txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "\n".join(lines))

    print("\nWrote:")
    for p in (
        gene_csv,
        panel_csv,
        txt,
        manifest,
    ):
        print(" ", p)


if __name__ == "__main__":
    main()
