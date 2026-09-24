#!/usr/bin/env python3
"""
Soft Spaces / DLBCL Phase 4
v41.15x — Exploratory reduced-panel external replication on GSE117556

STATUS
------
EXPLORATORY ONLY.

The preregistered v41.15 confirmatory test was TECHNICALLY NON-EVALUABLE
because 4 of the 16 frozen coordinates could not be represented on GPL14951.

This script therefore performs a clearly separate exploratory analysis using
only the 12 reproducibly measurable frozen coordinates from the v41.15a3
mapping audit.

It MUST NOT be reported as the preregistered v41.15 confirmatory result.

Primary exploratory endpoint
----------------------------
Top-4 gene-eigensubspace overlap between:
    GSE10846 reference geometry
and
    GSE117556 external geometry

Secondary exploratory endpoint
------------------------------
Pairwise gene-correlation structure similarity.

Additional exploratory metrics
------------------------------
- Top-8 gene-eigensubspace overlap
- Eigenvalue-profile cosine similarity

NULL
----
10,000 random panels of the SAME reduced panel size, drawn from the GSE10846
raw-variance Top-256 among genes measurable in BOTH GSE10846 and GSE117556.

No external disease/outcome labels are used.
No classifier is fitted.
No causal or clinical claim is made.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v41.15x"
DEFAULT_N_NULL = 10000
DEFAULT_SEED = 4115001
VARIANCE_POOL = 256
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
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd < EPS)] = 1.0
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def gene_corr(X: np.ndarray) -> np.ndarray:
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
    A = Ua[:, :r]
    B = Ub[:, :r]
    cross = A.T @ B
    return float(np.clip(np.sum(cross * cross) / r, 0.0, 1.0))


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
        (1 + np.sum(null_values >= real_value)) / (1 + len(null_values))
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


def load_mapping_audit(path: Path):
    df = pd.read_csv(path)

    usable = df.loc[df["confirmatory_usable"] == True].copy()

    if usable.empty:
        raise RuntimeError("No usable coordinates in mapping audit.")

    return df, usable


def parse_series_expression(path: Path) -> pd.DataFrame:
    """
    Parse GEO series matrix into samples x probe dataframe.
    """
    sample_ids = None
    data = []
    probes = []
    inside = False

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                inside = True
                continue

            if line.startswith("!series_matrix_table_end"):
                break

            if not inside:
                continue

            if line.startswith('"ID_REF"') or line.startswith("ID_REF"):
                parts = [x.strip().strip('"') for x in line.rstrip("\n\r").split("\t")]
                sample_ids = parts[1:]
                continue

            if not line.strip():
                continue

            parts = line.rstrip("\n\r").split("\t")
            probe = parts[0].strip().strip('"')
            vals = [float(x.strip().strip('"')) for x in parts[1:]]

            probes.append(probe)
            data.append(vals)

    if sample_ids is None:
        raise RuntimeError("Could not find sample header in series matrix.")

    M = np.asarray(data, dtype=np.float64)

    if M.shape[1] != len(sample_ids):
        raise RuntimeError("Series matrix sample dimension mismatch.")

    return pd.DataFrame(
        M.T,
        index=sample_ids,
        columns=probes,
    )


def parse_gpl14951_mapping(path: Path):
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    try:
        begin = next(
            i for i, line in enumerate(lines)
            if line.startswith("!platform_table_begin")
        )
        end = next(
            i for i, line in enumerate(lines)
            if line.startswith("!platform_table_end")
        )
    except StopIteration:
        raise RuntimeError("GPL14951 table markers not found.")

    table = lines[begin + 1:end]
    header = table[0].split("\t")

    lut = {x.strip().lower(): x for x in header}
    id_col = lut.get("id")
    symbol_col = lut.get("symbol")

    if not id_col or not symbol_col:
        raise RuntimeError(
            f"Expected ID/Symbol columns, got: {header}"
        )

    probe_to_symbol = {}

    reader = csv.DictReader(
        table[1:],
        fieldnames=header,
        delimiter="\t",
    )

    for row in reader:
        probe = str(row.get(id_col, "")).strip()
        sym = str(row.get(symbol_col, "")).strip()

        if probe and sym and sym not in {"---", "NA", "N/A"}:
            probe_to_symbol[probe] = sym

    return probe_to_symbol


def aggregate_probe_to_gene(
    probe_df: pd.DataFrame,
    probe_to_symbol: dict,
    genes_needed: set[str] | None = None,
) -> pd.DataFrame:
    symbol_to_probes = defaultdict(list)

    for probe in probe_df.columns:
        sym = probe_to_symbol.get(probe)

        if not sym:
            continue

        if genes_needed is not None and sym not in genes_needed:
            continue

        symbol_to_probes[sym].append(probe)

    out = {}

    for sym, probes in symbol_to_probes.items():
        out[sym] = probe_df.loc[:, probes].mean(axis=1)

    if not out:
        raise RuntimeError("No genes could be aggregated from probe data.")

    return pd.DataFrame(out, index=probe_df.index)


def subset_matrix(df: pd.DataFrame, genes) -> np.ndarray:
    missing = [g for g in genes if g not in df.columns]
    if missing:
        raise RuntimeError(
            "Missing genes in matrix: " + ", ".join(missing)
        )
    return df.loc[:, list(genes)].to_numpy(dtype=np.float64)


def resolve_usable_symbols(usable_df: pd.DataFrame):
    """
    For each usable frozen coordinate, use the mapping-audit resolved symbol.
    For RESOLVED-COMPOUND there is exactly one resolved symbol.
    """
    genes = []

    for _, row in usable_df.sort_values("frozen_rank").iterrows():
        resolved = str(row["resolved_symbols"]).strip()

        if not resolved:
            raise RuntimeError(
                f"Usable coordinate has no resolved symbol: "
                f"{row['frozen_coordinate']}"
            )

        parts = [x for x in resolved.split("///") if x]

        if len(parts) != 1:
            raise RuntimeError(
                "Exploratory reduced-panel analysis requires exactly one "
                "resolved gene symbol per usable frozen coordinate."
            )

        genes.append(parts[0])

    return genes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=DEFAULT_N_NULL)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    project = project_dir_from_script()
    results = project / "results" / "stability"
    outdir = args.output_dir or results
    outdir.mkdir(parents=True, exist_ok=True)

    audit_path = results / "v41_15_mapping_audit.csv"
    prep_manifest_path = results / "v41_15a3_prepare_manifest.json"

    train_path = (
        project
        / "data"
        / "processed"
        / "GSE10846_ABC_GCB_gene_expression.csv.gz"
    )

    series_path = (
        project
        / "data"
        / "external"
        / "GSE117556"
        / "raw"
        / "GSE117556_series_matrix.txt.gz"
    )

    gpl_path = (
        project
        / "data"
        / "external"
        / "GSE117556"
        / "platform"
        / "GPL14951_full_geo_table.txt"
    )

    for p in (
        audit_path,
        prep_manifest_path,
        train_path,
        series_path,
        gpl_path,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    prep_manifest = json.loads(
        prep_manifest_path.read_text(encoding="utf-8")
    )

    if prep_manifest.get("eligibility_gate") != "TECHNICALLY_NON_EVALUABLE":
        print(
            "NOTE: v41.15x is intended for the technically non-evaluable "
            "confirmatory case."
        )

    audit_df, usable_df = load_mapping_audit(audit_path)

    reduced_genes = resolve_usable_symbols(usable_df)
    panel_size = len(reduced_genes)

    print(f"=== {VERSION} EXPLORATORY REDUCED-PANEL REPLICATION ===")
    print("Confirmatory v41.15 status: TECHNICALLY NON-EVALUABLE")
    print("Exploratory reduced panel size:", panel_size)
    print("Reduced genes:")
    for i, g in enumerate(reduced_genes, 1):
        print(f"  {i:2d}. {g}")
    print("External labels used: NO")
    print("NULL panels:", args.n_null)
    print()

    print("[1/5] Loading GSE10846 gene expression...")
    train_df = pd.read_csv(train_path, index_col=0)

    print("[2/5] Loading GSE117556 probe expression...")
    probe_df = parse_series_expression(series_path)
    print("      samples x probes:", probe_df.shape)

    print("[3/5] Parsing GPL14951 probe->gene mapping...")
    probe_to_symbol = parse_gpl14951_mapping(gpl_path)

    print("[4/5] Aggregating GSE117556 to gene level...")
    external_gene_df = aggregate_probe_to_gene(
        probe_df,
        probe_to_symbol,
        genes_needed=None,
    )
    print("      samples x genes:", external_gene_df.shape)

    common = [
        g for g in train_df.columns
        if g in external_gene_df.columns
    ]
    common_set = set(common)

    missing_reduced = [
        g for g in reduced_genes
        if g not in common_set
    ]

    if missing_reduced:
        raise RuntimeError(
            "Reduced panel genes unexpectedly missing after aggregation: "
            + ", ".join(missing_reduced)
        )

    print("[5/5] Building training-only Top-256 NULL universe...")

    Xcommon = train_df.loc[:, common].to_numpy(dtype=np.float64)
    raw_var = np.var(Xcommon, axis=0, ddof=1)
    genes_common = np.asarray(common, dtype=object)

    order = np.lexsort(
        (genes_common.astype(str), -raw_var)
    )

    if len(order) < VARIANCE_POOL:
        raise RuntimeError(
            f"Only {len(order)} common genes; need {VARIANCE_POOL}."
        )

    null_pool = genes_common[order[:VARIANCE_POOL]].astype(str)

    # REAL exploratory metrics
    Xtr_real = subset_matrix(train_df, reduced_genes)
    Xext_real = subset_matrix(external_gene_df, reduced_genes)

    real = panel_metrics(Xtr_real, Xext_real)

    # NULL distribution
    rng = np.random.default_rng(args.seed)
    null_rows = []

    metric_names = [
        "corr_structure_similarity",
        "top4_eigensubspace_overlap",
        "top8_eigensubspace_overlap",
        "eigenvalue_profile_cosine",
    ]

    for i in range(args.n_null):
        panel = rng.choice(
            null_pool,
            size=panel_size,
            replace=False,
        ).tolist()

        Xtr = subset_matrix(train_df, panel)
        Xext = subset_matrix(external_gene_df, panel)

        m = panel_metrics(Xtr, Xext)

        row = {
            "null_rep": i + 1,
            "panel": "|".join(panel),
        }
        row.update(m)
        null_rows.append(row)

        if (i + 1) % max(1, args.n_null // 20) == 0:
            print(
                f"[v41.15x] NULL {i + 1}/{args.n_null}",
                flush=True,
            )

    null_df = pd.DataFrame(null_rows)

    summary = {}

    for metric in metric_names:
        nv = null_df[metric].to_numpy(dtype=np.float64)
        ns = summarize_null(nv)
        rv = float(real[metric])

        summary[metric] = {
            "real": rv,
            "null": ns,
            "real_minus_null_mean": float(rv - ns["mean"]),
            "empirical_one_sided_p": empirical_p(rv, nv),
        }

    # Write outputs
    real_csv = outdir / "v41_15x_real_metrics.csv"
    null_csv = outdir / "v41_15x_null_metrics.csv"
    txt = outdir / "v41_15x_exploratory_summary.txt"
    manifest_path = outdir / "v41_15x_exploratory_manifest.json"

    pd.DataFrame(
        [
            {
                "metric": metric,
                "real": summary[metric]["real"],
                "null_mean": summary[metric]["null"]["mean"],
                "null_sd": summary[metric]["null"]["sd"],
                "null_q025": summary[metric]["null"]["q025"],
                "null_q975": summary[metric]["null"]["q975"],
                "real_minus_null_mean":
                    summary[metric]["real_minus_null_mean"],
                "empirical_one_sided_p":
                    summary[metric]["empirical_one_sided_p"],
            }
            for metric in metric_names
        ]
    ).to_csv(real_csv, index=False)

    null_df.to_csv(null_csv, index=False)

    lines = [
        f"=== {VERSION} EXPLORATORY REDUCED-PANEL REPLICATION ===",
        "",
        "STATUS",
        "------",
        "The preregistered v41.15 confirmatory test was technically",
        "non-evaluable because 4/16 frozen coordinates were not measurable",
        "on GPL14951.",
        "",
        "This analysis is EXPLORATORY ONLY and must not be reported as",
        "the preregistered v41.15 confirmatory result.",
        "",
        f"Reduced measurable panel size: {panel_size}",
        f"NULL panels: {args.n_null}",
        "External labels used: NO",
        "",
        "REDUCED PANEL",
        "-------------",
    ]

    lines += [
        f"{i:2d}. {g}"
        for i, g in enumerate(reduced_genes, 1)
    ]

    labels = {
        "top4_eigensubspace_overlap":
            "PRIMARY EXPLORATORY: Top-4 gene-eigensubspace overlap",
        "corr_structure_similarity":
            "SECONDARY EXPLORATORY: Correlation-structure similarity",
        "top8_eigensubspace_overlap":
            "Exploratory: Top-8 gene-eigensubspace overlap",
        "eigenvalue_profile_cosine":
            "Exploratory: Eigenvalue-profile cosine similarity",
    }

    for metric in (
        "top4_eigensubspace_overlap",
        "corr_structure_similarity",
        "top8_eigensubspace_overlap",
        "eigenvalue_profile_cosine",
    ):
        s = summary[metric]
        n = s["null"]

        title = labels[metric]

        lines += [
            "",
            title,
            "-" * len(title),
            f"REAL:           {s['real']:.6f}",
            f"NULL mean:      {n['mean']:.6f}",
            f"NULL 95% range: [{n['q025']:.6f}, {n['q975']:.6f}]",
            f"REAL-NULL mean: {s['real_minus_null_mean']:+.6f}",
            f"empirical p:    {s['empirical_one_sided_p']:.6g}",
        ]

    lines += [
        "",
        "INTERPRETATION LIMIT",
        "--------------------",
        "A strong result here may motivate a future newly preregistered",
        "platform-compatible replication, but it cannot retroactively turn",
        "v41.15 into a confirmatory PASS.",
        "",
        "No causal biology, clinical utility, or quantum advantage is claimed.",
    ]

    txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "version": VERSION,
        "status": "EXPLORATORY_ONLY",
        "confirmatory_v41_15_status": "TECHNICALLY_NON_EVALUABLE",
        "source_mapping_audit": str(audit_path),
        "source_mapping_audit_sha256": sha256_file(audit_path),
        "source_prepare_manifest": str(prep_manifest_path),
        "source_prepare_manifest_sha256": sha256_file(prep_manifest_path),
        "development_cohort": "GSE10846",
        "external_cohort": "GSE117556",
        "external_platform": "GPL14951",
        "panel_size": panel_size,
        "reduced_panel_genes": reduced_genes,
        "n_null": args.n_null,
        "seed": args.seed,
        "null_definition":
            "random panels of same reduced size drawn from GSE10846 "
            "raw-variance Top-256 among genes measurable in both cohorts",
        "external_labels_used": False,
        "metrics": summary,
        "input_sha256": {
            "GSE10846_expression": sha256_file(train_path),
            "GSE117556_series_matrix": sha256_file(series_path),
            "GPL14951_platform_table": sha256_file(gpl_path),
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
