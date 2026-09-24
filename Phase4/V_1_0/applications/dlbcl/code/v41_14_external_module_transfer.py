#!/usr/bin/env python3
"""
Soft Spaces / DLBCL Phase 4
v41.14 — External Gene-Space / Module Transfer

Scientific question
-------------------
v41.13 showed that resampled REAL panels preserve more of the frozen GSE10846
module/subspace geometry than matched NULL panels internally.

v41.14 asks whether the FROZEN 16-gene structure itself is preserved in
independent external expression cohorts.

Primary external cohort:
    GSE87371
    (prospective cohort from v41.11; no labels are used here)

Secondary external cohort:
    GSE31312
    (historically post-hoc because it was already observed in v41.6)

Important geometry point
------------------------
v41.13 used sample-expression column spaces within one cohort. Across cohorts
the number and identity of patients differ, so those sample-space bases cannot
be compared directly.

v41.14 therefore compares structure in GENE COORDINATE SPACE using the same
frozen 16 genes across cohorts.

Metrics
-------
For a 16-gene panel in each cohort:
1. Pairwise-correlation structure similarity
   Pearson correlation between upper triangles of the 16x16 gene correlation
   matrices.

2. Top-r gene-eigensubspace overlap
       S_r = || U_train^T U_external ||_F^2 / r
   for r=4 and r=8, where U are leading eigenvectors of the gene correlation
   matrix. Values are in [0,1].

3. Eigenvalue-profile similarity
   Cosine similarity between descending eigenvalue vectors.

NULL controls
-------------
Random 16-gene panels are sampled from the GSE10846 raw-variance Top-256,
restricted to genes measurable in BOTH external cohorts.

This is a new but frozen, label-free NULL definition for v41.14.
External disease labels are never read.

Inputs
------
- results/stability/v41_12b_feature_stability_manifest.json
- data/processed/GSE10846_ABC_GCB_gene_expression.csv.gz
- data/processed/GSE10846_probe_to_gene_mapping.csv.gz
- data/external/GSE87371/raw/GSE87371_series_matrix.txt.gz
- data/external/GSE31312/raw/GSE31312_series_matrix.txt.gz

Outputs
-------
- results/stability/v41_14_external_module_transfer_summary.txt
- results/stability/v41_14_external_module_transfer_manifest.json
- results/stability/v41_14_null_metrics.csv
- results/stability/v41_14_real_metrics.csv

No external COO/GEP labels are used.
No classifier is fitted.
No quantum advantage is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v41_11_prospective_GSE87371 as v11


VERSION = "v41.14"
DEFAULT_N_NULL = 1000
DEFAULT_SEED = 4114001
VARIANCE_POOL = 256
PANEL_SIZE = 16
RANKS = (4, 8)
EPS = 1e-12


def project_dir_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cohort_z(X: np.ndarray) -> np.ndarray:
    """
    Cohort-wise z-score, label-free, gene by gene.
    Input: samples x genes.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd < EPS)] = 1.0
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def gene_corr(X: np.ndarray) -> np.ndarray:
    """
    Gene-gene Pearson correlation from samples x genes.
    """
    Z = cohort_z(X)
    C = np.corrcoef(Z, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C


def upper_triangle(C: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(C.shape[0], k=1)
    return C[i, j]


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if len(a) != len(b):
        raise RuntimeError("Vector length mismatch.")

    ac = a - a.mean()
    bc = b - b.mean()
    den = np.linalg.norm(ac) * np.linalg.norm(bc)

    if den < EPS:
        return 0.0

    return float(np.dot(ac, bc) / den)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = np.linalg.norm(a) * np.linalg.norm(b)

    if den < EPS:
        return 0.0

    return float(np.dot(a, b) / den)


def eigensystem(C: np.ndarray):
    e, U = np.linalg.eigh(C)
    order = np.argsort(e)[::-1]
    return e[order], U[:, order]


def subspace_overlap(Ua: np.ndarray, Ub: np.ndarray, r: int) -> float:
    r = min(r, Ua.shape[1], Ub.shape[1])
    if r <= 0:
        return 0.0
    A = Ua[:, :r]
    B = Ub[:, :r]
    cross = A.T @ B
    val = np.sum(cross * cross) / r
    return float(np.clip(val, 0.0, 1.0))


def panel_metrics(Xa: np.ndarray, Xb: np.ndarray) -> dict:
    Ca = gene_corr(Xa)
    Cb = gene_corr(Xb)

    ea, Ua = eigensystem(Ca)
    eb, Ub = eigensystem(Cb)

    out = {
        "corr_structure_similarity": safe_pearson(
            upper_triangle(Ca),
            upper_triangle(Cb),
        ),
        "eigenvalue_profile_cosine": cosine(ea, eb),
    }

    for r in RANKS:
        out[f"top{r}_eigensubspace_overlap"] = subspace_overlap(Ua, Ub, r)

    return out


def empirical_p(real_value: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=np.float64)
    return float(
        (1 + np.sum(null_values >= real_value)) / (len(null_values) + 1)
    )


def summarize_null(values: np.ndarray) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "sd": float(np.std(a, ddof=1)),
        "median": float(np.median(a)),
        "q025": float(np.quantile(a, 0.025)),
        "q975": float(np.quantile(a, 0.975)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def load_external_gene_matrix(raw_path: Path, mapping_path: Path) -> pd.DataFrame:
    """
    Reuse the exact v41.11 GEO parser / probe->gene aggregation.
    Returns samples x genes.
    """
    probe = v11.read_expression(raw_path)
    gene = v11.aggregate(probe, v11.mapping(mapping_path))
    return gene.T


def subset_matrix(df: pd.DataFrame, genes) -> np.ndarray:
    missing = [g for g in genes if g not in df.columns]
    if missing:
        raise RuntimeError(
            "Panel genes missing from cohort: " + ", ".join(missing)
        )

    X = df.loc[:, list(genes)].to_numpy(dtype=np.float64)

    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite values in panel expression matrix.")

    return X


def compare_three(
    panel,
    tr_df,
    g87371_df,
    g31312_df,
):
    Xtr = subset_matrix(tr_df, panel)
    X87 = subset_matrix(g87371_df, panel)
    X31 = subset_matrix(g31312_df, panel)

    result = {}

    for label, A, B in (
        ("GSE10846_to_GSE87371", Xtr, X87),
        ("GSE10846_to_GSE31312", Xtr, X31),
        ("GSE87371_to_GSE31312", X87, X31),
    ):
        result[label] = panel_metrics(A, B)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=DEFAULT_N_NULL)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    project = project_dir_from_script()
    stability = project / "results" / "stability"
    outdir = args.output_dir or stability
    outdir.mkdir(parents=True, exist_ok=True)

    manifest12_path = (
        stability / "v41_12b_feature_stability_manifest.json"
    )
    train_path = (
        project
        / "data"
        / "processed"
        / "GSE10846_ABC_GCB_gene_expression.csv.gz"
    )
    mapping_path = (
        project
        / "data"
        / "processed"
        / "GSE10846_probe_to_gene_mapping.csv.gz"
    )
    g87371_path = (
        project
        / "data"
        / "external"
        / "GSE87371"
        / "raw"
        / "GSE87371_series_matrix.txt.gz"
    )
    g31312_path = (
        project
        / "data"
        / "external"
        / "GSE31312"
        / "raw"
        / "GSE31312_series_matrix.txt.gz"
    )

    for p in (
        manifest12_path,
        train_path,
        mapping_path,
        g87371_path,
        g31312_path,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    manifest12 = json.loads(
        manifest12_path.read_text(encoding="utf-8")
    )
    frozen = list(manifest12["frozen_full_development_panel"])

    if len(frozen) != PANEL_SIZE:
        raise RuntimeError(
            f"Frozen panel has {len(frozen)} genes, expected {PANEL_SIZE}"
        )

    print(f"=== {VERSION} EXTERNAL MODULE TRANSFER ===")
    print("Primary external cohort:   GSE87371")
    print("Secondary external cohort: GSE31312")
    print("External labels used:      NO")
    print("NULL panels:", args.n_null)
    print()

    print("[1/6] Loading GSE10846 expression...")
    tr_df = pd.read_csv(train_path, index_col=0)

    print("[2/6] Mapping GSE87371 probes -> genes...")
    g87371_df = load_external_gene_matrix(
        g87371_path,
        mapping_path,
    )
    print("      shape:", g87371_df.shape)

    print("[3/6] Mapping GSE31312 probes -> genes...")
    g31312_df = load_external_gene_matrix(
        g31312_path,
        mapping_path,
    )
    print("      shape:", g31312_df.shape)

    common = [
        g
        for g in tr_df.columns
        if g in g87371_df.columns and g in g31312_df.columns
    ]
    common_set = set(common)

    missing_frozen = [g for g in frozen if g not in common_set]
    if missing_frozen:
        raise RuntimeError(
            "Frozen 16 cannot be tested unchanged because these genes "
            "are not measurable in all three cohorts:\n  "
            + "\n  ".join(missing_frozen)
        )

    print("[4/6] Frozen 16 present unchanged in all cohorts: YES")
    print("      common genes:", len(common))

    # Freeze NULL universe from training raw variance only.
    Xcommon = tr_df.loc[:, common].to_numpy(dtype=np.float64)
    raw_var = np.var(Xcommon, axis=0, ddof=1)
    genes_common = np.asarray(common, dtype=object)
    order = np.lexsort((genes_common.astype(str), -raw_var))

    if len(order) < VARIANCE_POOL:
        raise RuntimeError(
            f"Only {len(order)} common genes; need {VARIANCE_POOL}."
        )

    null_pool = genes_common[order[:VARIANCE_POOL]].astype(str)

    frozen_not_in_pool = [g for g in frozen if g not in set(null_pool)]
    if frozen_not_in_pool:
        print(
            "NOTE: frozen panel contains genes outside the common Top-256 "
            "NULL universe:",
            ", ".join(frozen_not_in_pool),
        )

    print("[5/6] Computing frozen REAL geometry...")
    real = compare_three(
        frozen,
        tr_df,
        g87371_df,
        g31312_df,
    )

    print("[6/6] Running label-free matched NULL geometry controls...")
    rng = np.random.default_rng(args.seed)
    null_rows = []

    pair_names = [
        "GSE10846_to_GSE87371",
        "GSE10846_to_GSE31312",
        "GSE87371_to_GSE31312",
    ]
    metric_names = [
        "corr_structure_similarity",
        "top4_eigensubspace_overlap",
        "top8_eigensubspace_overlap",
        "eigenvalue_profile_cosine",
    ]

    for j in range(args.n_null):
        panel = rng.choice(
            null_pool,
            size=PANEL_SIZE,
            replace=False,
        ).tolist()

        res = compare_three(
            panel,
            tr_df,
            g87371_df,
            g31312_df,
        )

        row = {
            "null_rep": j + 1,
            "panel": "|".join(panel),
        }

        for pair in pair_names:
            for metric in metric_names:
                row[f"{pair}__{metric}"] = res[pair][metric]

        null_rows.append(row)

        if (j + 1) % max(1, args.n_null // 20) == 0:
            print(
                f"      NULL {j + 1}/{args.n_null}",
                flush=True,
            )

    null_df = pd.DataFrame(null_rows)

    real_rows = []
    summary = {}

    for pair in pair_names:
        summary[pair] = {}

        for metric in metric_names:
            rv = float(real[pair][metric])
            col = f"{pair}__{metric}"
            nv = null_df[col].to_numpy(dtype=float)
            ns = summarize_null(nv)
            p = empirical_p(rv, nv)

            summary[pair][metric] = {
                "real": rv,
                "null": ns,
                "real_minus_null_mean": float(rv - ns["mean"]),
                "empirical_one_sided_p": p,
            }

            real_rows.append(
                {
                    "comparison": pair,
                    "metric": metric,
                    "real": rv,
                    "null_mean": ns["mean"],
                    "null_sd": ns["sd"],
                    "null_q025": ns["q025"],
                    "null_q975": ns["q975"],
                    "real_minus_null_mean": rv - ns["mean"],
                    "empirical_one_sided_p": p,
                }
            )

    real_df = pd.DataFrame(real_rows)

    null_csv = outdir / "v41_14_null_metrics.csv"
    real_csv = outdir / "v41_14_real_metrics.csv"
    txt = outdir / "v41_14_external_module_transfer_summary.txt"
    manifest_path = (
        outdir / "v41_14_external_module_transfer_manifest.json"
    )

    null_df.to_csv(null_csv, index=False)
    real_df.to_csv(real_csv, index=False)

    lines = [
        f"=== {VERSION} EXTERNAL GENE-SPACE / MODULE TRANSFER ===",
        "",
        "DESIGN",
        "------",
        "Frozen reference panel: v41.12b full-development 16 genes",
        "Primary external cohort: GSE87371",
        "Secondary external cohort: GSE31312",
        "External disease labels used: NO",
        f"NULL panels: {args.n_null}",
        "NULL universe: GSE10846 raw-variance Top-256 restricted to",
        "genes measurable in both external cohorts.",
        "",
        "IMPORTANT",
        "---------",
        "Across cohorts, patient/sample spaces differ. Therefore this test",
        "compares geometry in 16-dimensional GENE COORDINATE SPACE, not",
        "the within-cohort sample-space basis used in v41.13.",
        "",
    ]

    display_names = {
        "GSE10846_to_GSE87371":
            "PRIMARY: GSE10846 -> GSE87371",
        "GSE10846_to_GSE31312":
            "SECONDARY: GSE10846 -> GSE31312",
        "GSE87371_to_GSE31312":
            "EXTERNAL <-> EXTERNAL: GSE87371 -> GSE31312",
    }

    metric_labels = {
        "corr_structure_similarity":
            "Pairwise correlation-structure similarity",
        "top4_eigensubspace_overlap":
            "Top-4 gene-eigensubspace overlap",
        "top8_eigensubspace_overlap":
            "Top-8 gene-eigensubspace overlap",
        "eigenvalue_profile_cosine":
            "Eigenvalue-profile cosine similarity",
    }

    for pair in pair_names:
        title = display_names[pair]
        lines += [
            title,
            "-" * len(title),
        ]

        for metric in metric_names:
            s = summary[pair][metric]
            n = s["null"]

            lines += [
                metric_labels[metric],
                f"  REAL:            {s['real']:.6f}",
                f"  NULL mean:       {n['mean']:.6f}",
                f"  NULL 95% range:  [{n['q025']:.6f}, {n['q975']:.6f}]",
                f"  REAL-NULL mean:  {s['real_minus_null_mean']:+.6f}",
                f"  empirical p:     {s['empirical_one_sided_p']:.6g}",
                "",
            ]

    lines += [
        "READING RULE",
        "------------",
        "Strongest support for external module transfer would be obtained",
        "if the frozen REAL panel exceeds the label-free NULL distribution",
        "on multiple geometry metrics in GSE87371, with qualitatively",
        "consistent behavior also in GSE31312.",
        "",
        "Failure on these metrics would argue against claiming that the",
        "v41.13 internal module/subspace geometry transfers externally.",
        "",
        "LIMITS",
        "------",
        "This test is label-free and geometric.",
        "It does not test causal biology or clinical utility.",
        "GSE31312 is secondary/post-hoc because it had been seen earlier.",
        "GSE87371 remains the more important independent external cohort.",
        "No quantum advantage is claimed.",
    ]

    txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "version": VERSION,
        "source_panel_version": manifest12.get("version"),
        "frozen_panel": frozen,
        "panel_size": PANEL_SIZE,
        "primary_external_cohort": "GSE87371",
        "secondary_external_cohort": "GSE31312",
        "external_labels_used": False,
        "cohort_alignment":
            "cohort-wise gene standardization for correlation geometry",
        "geometry_space":
            "gene coordinate space; not sample space",
        "null": {
            "n": args.n_null,
            "seed": args.seed,
            "definition":
                "random 16 from GSE10846 raw-variance Top-256 among genes "
                "measurable in GSE10846, GSE87371, and GSE31312",
        },
        "common_gene_count": len(common),
        "null_pool_size": len(null_pool),
        "metrics": summary,
        "input_sha256": {
            "v41_12b_manifest": sha256_file(manifest12_path),
            "GSE10846_expression": sha256_file(train_path),
            "probe_to_gene_mapping": sha256_file(mapping_path),
            "GSE87371_series_matrix": sha256_file(g87371_path),
            "GSE31312_series_matrix": sha256_file(g31312_path),
        },
        "historical_status": {
            "GSE87371":
                "primary independent external cohort; labels not used here",
            "GSE31312":
                "secondary/post-hoc cohort; previously observed in v41.6",
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("\n".join(lines))

    print("\nWrote:")
    for p in (
        real_csv,
        null_csv,
        txt,
        manifest_path,
    ):
        print(" ", p)


if __name__ == "__main__":
    main()
