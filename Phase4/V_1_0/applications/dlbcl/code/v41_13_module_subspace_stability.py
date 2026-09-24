#!/usr/bin/env python3
"""
Soft Spaces / DLBCL Phase 4
v41.13 — Module / Expression-Subspace Stability

Purpose
-------
v41.12b showed that exact 16-gene identity is only moderately recurrent under
patient resampling, while REAL panels are still more stable than matched NULL.

v41.13 asks a deeper question:

    Do different REAL gene panels preserve the same underlying expression
    geometry / correlated gene module even when exact gene identity changes?

This script DOES NOT rerun the 500 Soft-Spaces bootstrap selections.
It reuses the frozen v41.12b outputs.

Primary metrics
---------------
1) Frozen-subspace capture
   Each 16-gene panel is represented by the subspace spanned by its standardized
   gene-expression vectors across the full GSE10846 development cohort.

   If Q_F is an orthonormal basis for the frozen 16-gene reference subspace and
   Q_P is the basis for a resampled panel, then:

       capture = || Q_F^T Q_P ||_F^2 / rank(Q_F)

   This equals the mean cos^2 principal-angle overlap relative to the frozen
   subspace, and is bounded approximately in [0,1].

2) Correlation-neighborhood capture
   For each frozen reference gene, find the largest absolute Pearson
   correlation with any gene in the candidate panel. Average these 16 maxima.

   Exact recurrence counts as correlation 1.0; highly correlated substitutes
   can also preserve module structure.

3) Exact overlap
   Fraction of the frozen 16 genes present exactly in the candidate panel.

REAL is compared against the matched NULL panel saved for the same v41.12b
resample.

No external cohort labels are used.
No GSE87371 labels are used.
This is a development-cohort structure/stability analysis, not a classifier
performance test and not a causal-biological analysis.

Expected input
--------------
results/stability/v41_12b_resample_panels.csv
results/stability/v41_12b_feature_stability_manifest.json
data/processed/GSE10846_ABC_GCB_gene_expression.csv.gz

Outputs
-------
results/stability/v41_13_per_resample_metrics.csv
results/stability/v41_13_gene_replacement_map.csv
results/stability/v41_13_module_subspace_stability_summary.txt
results/stability/v41_13_module_subspace_stability_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v41.13"
DEFAULT_BOOTSTRAP_REPS = 20000
DEFAULT_SEED = 4113001
EPS = 1e-12


def project_dir_from_script() -> Path:
    # .../applications/dlbcl/code/v41_13_module_subspace_stability.py
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_panel(value: str):
    if pd.isna(value):
        return []
    return [x for x in str(value).split("|") if x]


def standardize_columns(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd < EPS)] = 1.0
    Z = (X - mu) / sd
    return np.nan_to_num(Z, copy=False)


def orthonormal_basis(X: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Orthonormal basis for the column space of X via SVD.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.size == 0:
        return np.zeros((X.shape[0], 0), dtype=np.float64)

    U, s, _ = np.linalg.svd(X, full_matrices=False)
    if len(s) == 0:
        return np.zeros((X.shape[0], 0), dtype=np.float64)

    threshold = max(X.shape) * np.max(s) * tol
    r = int(np.sum(s > threshold))
    return U[:, :r]


def frozen_subspace_capture(Q_frozen: np.ndarray, X_panel: np.ndarray) -> float:
    """
    Fraction of the frozen reference subspace captured by panel expression span.
    """
    Q_panel = orthonormal_basis(X_panel)

    if Q_frozen.shape[1] == 0 or Q_panel.shape[1] == 0:
        return 0.0

    cross = Q_frozen.T @ Q_panel
    val = np.sum(cross * cross) / Q_frozen.shape[1]

    # numerical guard
    return float(np.clip(val, 0.0, 1.0))


def correlation_neighborhood_capture(
    corr: np.ndarray,
    frozen_idx: np.ndarray,
    panel_idx: np.ndarray,
) -> float:
    """
    Mean, over frozen genes, of the largest |correlation| to any panel gene.
    """
    if len(panel_idx) == 0:
        return 0.0

    block = np.abs(corr[np.ix_(frozen_idx, panel_idx)])
    maxima = np.max(block, axis=1)
    return float(np.mean(maxima))


def exact_overlap_fraction(frozen_genes, panel_genes) -> float:
    frozen = set(frozen_genes)
    panel = set(panel_genes)
    return len(frozen & panel) / len(frozen) if frozen else 0.0


def bootstrap_mean_ci(values, seed, reps=DEFAULT_BOOTSTRAP_REPS, chunk=1000):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    if len(values) == 1:
        v = float(values[0])
        return v, v, v

    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)

    pos = 0
    while pos < reps:
        m = min(chunk, reps - pos)
        idx = rng.integers(0, len(values), size=(m, len(values)))
        means[pos:pos + m] = values[idx].mean(axis=1)
        pos += m

    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def paired_summary(real, null, seed, reps):
    real = np.asarray(real, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)

    if len(real) != len(null):
        raise RuntimeError("REAL/NULL length mismatch.")

    delta = real - null

    real_m, real_lo, real_hi = bootstrap_mean_ci(real, seed, reps)
    null_m, null_lo, null_hi = bootstrap_mean_ci(null, seed + 1, reps)
    d_m, d_lo, d_hi = bootstrap_mean_ci(delta, seed + 2, reps)

    return {
        "real_mean": real_m,
        "real_ci_low": real_lo,
        "real_ci_high": real_hi,
        "null_mean": null_m,
        "null_ci_low": null_lo,
        "null_ci_high": null_hi,
        "delta_mean": d_m,
        "delta_ci_low": d_lo,
        "delta_ci_high": d_hi,
        "positive_delta_fraction": float(np.mean(delta > 0)),
        "tie_fraction": float(np.mean(delta == 0)),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--bootstrap-reps",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPS,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    args = ap.parse_args()

    project_dir = project_dir_from_script()
    stability_dir = project_dir / "results" / "stability"
    outdir = args.output_dir or stability_dir
    outdir.mkdir(parents=True, exist_ok=True)

    panels_path = stability_dir / "v41_12b_resample_panels.csv"
    manifest12_path = stability_dir / "v41_12b_feature_stability_manifest.json"
    expr_path = (
        project_dir
        / "data"
        / "processed"
        / "GSE10846_ABC_GCB_gene_expression.csv.gz"
    )

    for p in (panels_path, manifest12_path, expr_path):
        if not p.exists():
            raise FileNotFoundError(p)

    manifest12 = json.loads(manifest12_path.read_text(encoding="utf-8"))
    frozen_genes = list(manifest12["frozen_full_development_panel"])

    panels_df = pd.read_csv(panels_path)

    if len(panels_df) != int(manifest12["n_resamples"]):
        raise RuntimeError(
            f"Resample count mismatch: CSV={len(panels_df)} "
            f"manifest={manifest12['n_resamples']}"
        )

    real_panels = [split_panel(x) for x in panels_df["panel"]]
    null_panels = [split_panel(x) for x in panels_df["null_panel"]]

    expected_panel_size = int(manifest12["panel_size"])

    for i, p in enumerate(real_panels):
        if len(p) != expected_panel_size:
            raise RuntimeError(
                f"REAL panel {i} has {len(p)} genes, expected {expected_panel_size}"
            )

    for i, p in enumerate(null_panels):
        if len(p) != expected_panel_size:
            raise RuntimeError(
                f"NULL panel {i} has {len(p)} genes, expected {expected_panel_size}"
            )

    # Only load genes that are actually needed.
    needed_genes = set(frozen_genes)

    for p in real_panels:
        needed_genes.update(p)

    for p in null_panels:
        needed_genes.update(p)

    # Read header first so we can validate and then load only needed columns.
    header = pd.read_csv(expr_path, nrows=0)
    all_columns = list(header.columns)

    # First CSV column is the sample index column.
    sample_col = all_columns[0]
    available_genes = set(all_columns[1:])

    missing = sorted(needed_genes - available_genes)
    if missing:
        raise RuntimeError(
            "Genes used in v41.12b are missing from expression matrix: "
            + ", ".join(missing[:20])
        )

    usecols = [sample_col] + sorted(needed_genes)

    expr = pd.read_csv(
        expr_path,
        usecols=usecols,
        index_col=sample_col,
    )

    # Preserve a deterministic gene order for geometry/correlation lookup.
    gene_order = sorted(needed_genes)
    X = expr.loc[:, gene_order].to_numpy(dtype=np.float64)
    Z = standardize_columns(X)

    gene_to_idx = {g: i for i, g in enumerate(gene_order)}

    # Gene-gene Pearson correlation after column standardization.
    # np.corrcoef is safe here because we only loaded the union of genes used.
    corr = np.corrcoef(Z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    frozen_idx = np.asarray(
        [gene_to_idx[g] for g in frozen_genes],
        dtype=int,
    )

    X_frozen = Z[:, frozen_idx]
    Q_frozen = orthonormal_basis(X_frozen)

    if Q_frozen.shape[1] == 0:
        raise RuntimeError("Frozen reference subspace has rank 0.")

    print(f"=== {VERSION} MODULE / SUBSPACE STABILITY ===")
    print("Project:", project_dir)
    print("v41.12b resamples:", len(real_panels))
    print("Frozen panel size:", len(frozen_genes))
    print("Genes needed for geometry:", len(gene_order))
    print("Frozen expression-subspace rank:", Q_frozen.shape[1])
    print("Bootstrap reps:", args.bootstrap_reps)
    print()

    rows = []

    # For replacement-map diagnostics.
    replacement_counts = {
        g: Counter()
        for g in frozen_genes
    }
    replacement_corr_sums = {
        g: defaultdict(float)
        for g in frozen_genes
    }

    for r, (real_panel, null_panel) in enumerate(
        zip(real_panels, null_panels)
    ):
        real_idx = np.asarray(
            [gene_to_idx[g] for g in real_panel],
            dtype=int,
        )
        null_idx = np.asarray(
            [gene_to_idx[g] for g in null_panel],
            dtype=int,
        )

        real_sub = frozen_subspace_capture(
            Q_frozen,
            Z[:, real_idx],
        )
        null_sub = frozen_subspace_capture(
            Q_frozen,
            Z[:, null_idx],
        )

        real_corr = correlation_neighborhood_capture(
            corr,
            frozen_idx,
            real_idx,
        )
        null_corr = correlation_neighborhood_capture(
            corr,
            frozen_idx,
            null_idx,
        )

        real_exact = exact_overlap_fraction(
            frozen_genes,
            real_panel,
        )
        null_exact = exact_overlap_fraction(
            frozen_genes,
            null_panel,
        )

        rows.append(
            {
                "resample": int(r),
                "real_subspace_capture": real_sub,
                "null_subspace_capture": null_sub,
                "delta_subspace_capture": real_sub - null_sub,
                "real_corr_neighborhood": real_corr,
                "null_corr_neighborhood": null_corr,
                "delta_corr_neighborhood": real_corr - null_corr,
                "real_exact_overlap": real_exact,
                "null_exact_overlap": null_exact,
                "delta_exact_overlap": real_exact - null_exact,
            }
        )

        # For each frozen gene, identify the closest correlated gene
        # in the REAL panel for this resample.
        for fg in frozen_genes:
            fi = gene_to_idx[fg]
            vals = np.abs(corr[fi, real_idx])
            best_pos = int(np.argmax(vals))
            best_gene = real_panel[best_pos]
            best_corr = float(vals[best_pos])

            replacement_counts[fg][best_gene] += 1
            replacement_corr_sums[fg][best_gene] += best_corr

        if (r + 1) % max(1, len(real_panels) // 20) == 0:
            print(
                f"[v41.13] {r + 1}/{len(real_panels)}",
                flush=True,
            )

    metrics_df = pd.DataFrame(rows)

    sub_summary = paired_summary(
        metrics_df["real_subspace_capture"],
        metrics_df["null_subspace_capture"],
        args.seed + 100,
        args.bootstrap_reps,
    )

    corr_summary = paired_summary(
        metrics_df["real_corr_neighborhood"],
        metrics_df["null_corr_neighborhood"],
        args.seed + 200,
        args.bootstrap_reps,
    )

    exact_summary = paired_summary(
        metrics_df["real_exact_overlap"],
        metrics_df["null_exact_overlap"],
        args.seed + 300,
        args.bootstrap_reps,
    )

    # Build compact replacement map.
    replacement_rows = []

    for fg in frozen_genes:
        total = sum(replacement_counts[fg].values())

        if total == 0:
            continue

        top = replacement_counts[fg].most_common(10)

        for rank, (replacement_gene, count) in enumerate(top, 1):
            mean_corr = (
                replacement_corr_sums[fg][replacement_gene] / count
            )

            replacement_rows.append(
                {
                    "frozen_gene": fg,
                    "replacement_rank": rank,
                    "replacement_gene": replacement_gene,
                    "count": int(count),
                    "frequency": float(count / total),
                    "mean_abs_correlation_when_best": float(mean_corr),
                    "exact_same_gene": replacement_gene == fg,
                }
            )

    replacement_df = pd.DataFrame(replacement_rows)

    metrics_path = outdir / "v41_13_per_resample_metrics.csv"
    replacement_path = outdir / "v41_13_gene_replacement_map.csv"
    summary_path = outdir / "v41_13_module_subspace_stability_summary.txt"
    manifest_path = outdir / "v41_13_module_subspace_stability_manifest.json"

    metrics_df.to_csv(metrics_path, index=False)
    replacement_df.to_csv(replacement_path, index=False)

    def fmt_summary(title, s):
        return [
            title,
            "-" * len(title),
            f"REAL mean: {s['real_mean']:.6f}",
            f"REAL 95% CI: [{s['real_ci_low']:.6f}, {s['real_ci_high']:.6f}]",
            f"NULL mean: {s['null_mean']:.6f}",
            f"NULL 95% CI: [{s['null_ci_low']:.6f}, {s['null_ci_high']:.6f}]",
            f"Delta REAL-NULL: {s['delta_mean']:+.6f}",
            f"Delta 95% CI: [{s['delta_ci_low']:+.6f}, {s['delta_ci_high']:+.6f}]",
            f"Positive paired delta fraction: {s['positive_delta_fraction']:.6f}",
            f"Tie fraction: {s['tie_fraction']:.6f}",
            "",
        ]

    lines = [
        f"=== {VERSION} MODULE / EXPRESSION-SUBSPACE STABILITY ===",
        "",
        "INPUT",
        "-----",
        f"Development cohort: {manifest12['development_dataset']}",
        f"Resamples reused from v41.12b: {len(real_panels)}",
        f"Frozen panel size: {len(frozen_genes)}",
        f"Candidate gene union size: {len(gene_order)}",
        f"Frozen expression-subspace rank: {Q_frozen.shape[1]}",
        "",
        "INTERPRETATION",
        "--------------",
        "Subspace capture tests whether a resampled panel spans the same",
        "sample-expression geometry as the frozen 16-gene panel.",
        "",
        "Correlation-neighborhood capture tests whether exact genes are",
        "replaced by strongly correlated alternatives.",
        "",
    ]

    lines += fmt_summary(
        "FROZEN-SUBSPACE CAPTURE",
        sub_summary,
    )

    lines += fmt_summary(
        "CORRELATION-NEIGHBORHOOD CAPTURE",
        corr_summary,
    )

    lines += fmt_summary(
        "EXACT 16-GENE OVERLAP",
        exact_summary,
    )

    lines += [
        "READING RULE",
        "------------",
        "If REAL exceeds NULL for subspace/correlation capture much more",
        "clearly than for exact overlap, that supports the hypothesis that",
        "Soft Spaces is recovering a stable distributed module/subspace",
        "with interchangeable correlated genes rather than a unique fixed",
        "16-gene biomarker list.",
        "",
        "If REAL does not exceed NULL on these geometry-aware metrics, then",
        "the low exact-gene stability should not be rescued by a module",
        "interpretation.",
        "",
        "CAUTION",
        "-------",
        "This analysis is descriptive/stability-oriented within GSE10846.",
        "It does not prove causal gene modules, clinical utility, external",
        "generalization, or quantum advantage.",
    ]

    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest13 = {
        "version": VERSION,
        "source_version": manifest12.get("version"),
        "development_dataset": manifest12.get("development_dataset"),
        "n_resamples": len(real_panels),
        "panel_size": expected_panel_size,
        "frozen_panel": frozen_genes,
        "candidate_gene_union_size": len(gene_order),
        "frozen_expression_subspace_rank": int(Q_frozen.shape[1]),
        "metrics": {
            "subspace_capture":
                "||Q_frozen^T Q_panel||_F^2 / rank(Q_frozen)",
            "correlation_neighborhood":
                "mean frozen-gene max absolute Pearson correlation "
                "to candidate panel",
            "exact_overlap":
                "fraction of frozen 16 genes exactly present",
        },
        "real_null_pairing": "matched by v41.12b resample row",
        "bootstrap_reps": args.bootstrap_reps,
        "random_seed": args.seed,
        "input_files": {
            "panels": str(panels_path),
            "panels_sha256": sha256_file(panels_path),
            "v41_12b_manifest": str(manifest12_path),
            "v41_12b_manifest_sha256": sha256_file(manifest12_path),
            "expression": str(expr_path),
            "expression_sha256": sha256_file(expr_path),
        },
        "external_labels_used": False,
        "GSE87371_used": False,
        "summary": {
            "subspace_capture": sub_summary,
            "correlation_neighborhood": corr_summary,
            "exact_overlap": exact_summary,
        },
    }

    manifest_path.write_text(
        json.dumps(manifest13, indent=2),
        encoding="utf-8",
    )

    print()
    print("\n".join(lines))

    print("\nWrote:")
    for p in (
        metrics_path,
        replacement_path,
        summary_path,
        manifest_path,
    ):
        print(" ", p)


if __name__ == "__main__":
    main()
