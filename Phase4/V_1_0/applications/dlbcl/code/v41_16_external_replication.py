#!/usr/bin/env python3
"""
Soft Spaces / DLBCL Phase 4
v41.16 — Preregistered external replication on GSE159472

CONFIRMATORY STATUS
-------------------
This script implements the preregistered v41.16 replication exactly:

Frozen panel:
    DNER, SPIC, TCL1B, MMP20, CKAP2, RNF183, MYOCD,
    UMODL1, CTAG2, HRK, RGS13, TCL1A

Primary endpoint:
    Top-4 gene-eigensubspace overlap

Primary NULL:
    10,000 random 12-gene panels sampled without replacement from the
    GSE10846 raw-variance Top-256 among genes measurable in BOTH cohorts.

Primary PASS rule:
    REAL S4 > NULL mean
    AND empirical one-sided p <= 0.05

Secondary endpoint:
    Pairwise gene-correlation structure similarity

Exploratory:
    Top-8 eigensubspace overlap
    Eigenvalue-profile cosine similarity

No ABC/GCB, survival, treatment, response, or other outcome labels are used.
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


VERSION = "v41.16"

EXPECTED_PREREG_SHA256 = (
    "00f27acd8c4b3bc8fcdd58368235f1bb8538588aa8e9c40720eb885d9b555505"
)

FROZEN_PANEL = [
    "DNER",
    "SPIC",
    "TCL1B",
    "MMP20",
    "CKAP2",
    "RNF183",
    "MYOCD",
    "UMODL1",
    "CTAG2",
    "HRK",
    "RGS13",
    "TCL1A",
]

DEFAULT_N_NULL = 10000
DEFAULT_SEED = 4116001
VARIANCE_POOL = 256
PRIMARY_RANK = 4
EXPLORATORY_RANK = 8
EPS = 1e-12


def project_dir_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_preregistration(project: Path) -> Path:
    candidates = [
        project / "docs" / "preregistration" / "v41_16_PREREGISTRATION.txt",
        project / "docs" / "preregistration" / "v41_16" / "v41_16_PREREGISTRATION.txt",
        project / "code" / "v41_16_PREREGISTRATION.txt",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError("v41_16_PREREGISTRATION.txt not found")


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

    if r <= 0:
        return 0.0

    A = Ua[:, :r]
    B = Ub[:, :r]
    cross = A.T @ B

    return float(
        np.clip(
            np.sum(cross * cross) / r,
            0.0,
            1.0,
        )
    )


def panel_metrics(Xa: np.ndarray, Xb: np.ndarray) -> dict:
    Ca = gene_corr(Xa)
    Cb = gene_corr(Xb)

    ea, Ua = eigensystem(Ca)
    eb, Ub = eigensystem(Cb)

    return {
        "corr_structure_similarity": safe_pearson(
            upper_triangle(Ca),
            upper_triangle(Cb),
        ),
        "top4_eigensubspace_overlap": subspace_overlap(
            Ua,
            Ub,
            PRIMARY_RANK,
        ),
        "top8_eigensubspace_overlap": subspace_overlap(
            Ua,
            Ub,
            EXPLORATORY_RANK,
        ),
        "eigenvalue_profile_cosine": cosine(ea, eb),
    }


def empirical_p(real_value: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=np.float64)

    return float(
        (1 + np.sum(null_values >= real_value))
        / (1 + len(null_values))
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


def parse_series_expression(path: Path) -> pd.DataFrame:
    sample_ids = None
    probes = []
    rows = []
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
                parts = [
                    x.strip().strip('"')
                    for x in line.rstrip("\r\n").split("\t")
                ]
                sample_ids = parts[1:]
                continue

            if not line.strip():
                continue

            parts = line.rstrip("\r\n").split("\t")
            probe = parts[0].strip().strip('"')
            vals = [
                float(x.strip().strip('"'))
                for x in parts[1:]
            ]

            probes.append(probe)
            rows.append(vals)

    if sample_ids is None:
        raise RuntimeError("Series matrix header not found.")

    M = np.asarray(rows, dtype=np.float64)

    if M.shape[1] != len(sample_ids):
        raise RuntimeError("Series matrix sample dimension mismatch.")

    return pd.DataFrame(
        M.T,
        index=sample_ids,
        columns=probes,
    )


def split_symbols(value: str):
    value = str(value or "").strip()

    if not value:
        return []

    # GPL570 often uses " /// " between multiple symbols.
    parts = [
        x.strip()
        for x in value.replace("///", "|").split("|")
    ]

    return [
        x
        for x in parts
        if x and x.upper() not in {"---", "NA", "N/A", "NULL"}
    ]


def parse_gpl570_mapping(path: Path):
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    begin = next(
        i for i, line in enumerate(lines)
        if line.startswith("!platform_table_begin")
    )

    end = next(
        i for i, line in enumerate(lines)
        if line.startswith("!platform_table_end")
    )

    table = lines[begin + 1:end]
    header = table[0].split("\t")
    lut = {x.strip().lower(): x for x in header}

    id_col = lut.get("id")
    symbol_col = (
        lut.get("gene symbol")
        or lut.get("gene_symbol")
        or lut.get("symbol")
    )

    if not id_col or not symbol_col:
        raise RuntimeError(
            f"Could not identify GPL570 mapping columns: {header}"
        )

    probe_to_symbols = {}

    reader = csv.DictReader(
        table[1:],
        fieldnames=header,
        delimiter="\t",
    )

    for row in reader:
        probe = str(row.get(id_col, "")).strip()

        if not probe:
            continue

        syms = split_symbols(row.get(symbol_col, ""))

        if syms:
            probe_to_symbols[probe] = syms

    return probe_to_symbols


def aggregate_probe_to_gene(
    probe_df: pd.DataFrame,
    probe_to_symbols: dict,
) -> pd.DataFrame:
    """
    Deterministic arithmetic mean over all measured probes mapping to a gene.
    """
    gene_to_probes = defaultdict(list)

    for probe in probe_df.columns:
        syms = probe_to_symbols.get(probe, [])

        for sym in syms:
            gene_to_probes[sym].append(probe)

    out = {}

    for gene, probes in gene_to_probes.items():
        out[gene] = probe_df.loc[:, probes].mean(axis=1)

    if not out:
        raise RuntimeError("No genes aggregated from GPL570 probe matrix.")

    return pd.DataFrame(out, index=probe_df.index)


def subset_matrix(df: pd.DataFrame, genes) -> np.ndarray:
    missing = [
        g
        for g in genes
        if g not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing genes: " + ", ".join(missing)
        )

    X = df.loc[:, list(genes)].to_numpy(dtype=np.float64)

    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite expression values detected.")

    return X


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--n-null",
        type=int,
        default=DEFAULT_N_NULL,
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

    project = project_dir_from_script()
    results = project / "results" / "stability"
    outdir = args.output_dir or results
    outdir.mkdir(parents=True, exist_ok=True)

    prereg_path = find_preregistration(project)

    audit_path = results / "v41_16_mapping_audit.csv"
    prep_manifest_path = results / "v41_16a_prepare_manifest.json"

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
        / "GSE159472"
        / "raw"
        / "GSE159472_series_matrix.txt.gz"
    )

    gpl_path = (
        project
        / "data"
        / "external"
        / "GSE159472"
        / "platform"
        / "GPL570_full_geo_table.txt"
    )

    for p in (
        prereg_path,
        audit_path,
        prep_manifest_path,
        train_path,
        series_path,
        gpl_path,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    prereg_sha = sha256_file(prereg_path)

    print(f"=== {VERSION} PREREGISTERED EXTERNAL REPLICATION ===")
    print("Expected prereg SHA:", EXPECTED_PREREG_SHA256)
    print("Actual prereg SHA:  ", prereg_sha)

    if prereg_sha != EXPECTED_PREREG_SHA256:
        raise SystemExit("FAIL: preregistration SHA mismatch.")

    prep_manifest = json.loads(
        prep_manifest_path.read_text(encoding="utf-8")
    )

    if prep_manifest.get("eligibility_gate") != "PASS":
        raise SystemExit(
            "STOP: v41.16 mapping eligibility gate is not PASS."
        )

    audit_df = pd.read_csv(audit_path)

    if len(audit_df) != len(FROZEN_PANEL):
        raise RuntimeError("Unexpected mapping-audit row count.")

    if not audit_df["confirmatory_usable"].astype(bool).all():
        raise RuntimeError("Not all 12 frozen genes are mapping-usable.")

    print("PASS: preregistration integrity verified.")
    print("PASS: mapping eligibility verified.")
    print("External labels used: NO")
    print("NULL panels:", args.n_null)
    print()

    print("[1/5] Loading frozen GSE10846 gene-level expression...")
    train_df = pd.read_csv(
        train_path,
        index_col=0,
    )

    print("[2/5] Loading GSE159472 probe-level expression...")
    external_probe_df = parse_series_expression(series_path)
    print("      samples x probes:", external_probe_df.shape)

    print("[3/5] Parsing GPL570 probe->gene map...")
    probe_to_symbols = parse_gpl570_mapping(gpl_path)

    print("[4/5] Aggregating GSE159472 probes to genes...")
    external_gene_df = aggregate_probe_to_gene(
        external_probe_df,
        probe_to_symbols,
    )
    print("      samples x genes:", external_gene_df.shape)

    # Confirm all 12 again at execution.
    for gene in FROZEN_PANEL:
        if gene not in train_df.columns:
            raise RuntimeError(
                f"Frozen gene missing from GSE10846: {gene}"
            )

        if gene not in external_gene_df.columns:
            raise RuntimeError(
                f"Frozen gene missing from GSE159472: {gene}"
            )

    print("[5/5] Building preregistered training-only Top-256 NULL universe...")

    common = [
        g
        for g in train_df.columns
        if g in external_gene_df.columns
    ]

    genes_common = np.asarray(common, dtype=object)

    Xcommon = train_df.loc[:, common].to_numpy(dtype=np.float64)

    raw_var = np.var(
        Xcommon,
        axis=0,
        ddof=1,
    )

    order = np.lexsort(
        (
            genes_common.astype(str),
            -raw_var,
        )
    )

    if len(order) < VARIANCE_POOL:
        raise RuntimeError(
            f"Only {len(order)} common genes; need {VARIANCE_POOL}."
        )

    null_pool = genes_common[
        order[:VARIANCE_POOL]
    ].astype(str)

    # REAL frozen-panel metrics
    Xtrain_real = subset_matrix(
        train_df,
        FROZEN_PANEL,
    )

    Xext_real = subset_matrix(
        external_gene_df,
        FROZEN_PANEL,
    )

    real = panel_metrics(
        Xtrain_real,
        Xext_real,
    )

    # NULL
    rng = np.random.default_rng(args.seed)
    null_rows = []

    metric_names = [
        "top4_eigensubspace_overlap",
        "corr_structure_similarity",
        "top8_eigensubspace_overlap",
        "eigenvalue_profile_cosine",
    ]

    for i in range(args.n_null):
        panel = rng.choice(
            null_pool,
            size=len(FROZEN_PANEL),
            replace=False,
        ).tolist()

        Xtr = subset_matrix(
            train_df,
            panel,
        )

        Xext = subset_matrix(
            external_gene_df,
            panel,
        )

        metrics = panel_metrics(
            Xtr,
            Xext,
        )

        row = {
            "null_rep": i + 1,
            "panel": "|".join(panel),
        }

        row.update(metrics)
        null_rows.append(row)

        if (i + 1) % max(
            1,
            args.n_null // 20,
        ) == 0:
            print(
                f"[v41.16] NULL {i + 1}/{args.n_null}",
                flush=True,
            )

    null_df = pd.DataFrame(null_rows)

    summary = {}

    for metric in metric_names:
        nv = null_df[metric].to_numpy(dtype=np.float64)
        ns = summarize_null(nv)
        rv = float(real[metric])
        p = empirical_p(rv, nv)

        summary[metric] = {
            "real": rv,
            "null": ns,
            "real_minus_null_mean": float(rv - ns["mean"]),
            "empirical_one_sided_p": p,
        }

    primary = summary["top4_eigensubspace_overlap"]

    primary_pass = (
        primary["real"] > primary["null"]["mean"]
        and primary["empirical_one_sided_p"] <= 0.05
    )

    confirmatory_status = (
        "PASS"
        if primary_pass
        else "FAIL"
    )

    secondary = summary["corr_structure_similarity"]

    secondary_support = (
        secondary["real"] > secondary["null"]["mean"]
        and secondary["empirical_one_sided_p"] <= 0.05
    )

    # Outputs
    real_csv = outdir / "v41_16_real_metrics.csv"
    null_csv = outdir / "v41_16_null_metrics.csv"
    summary_txt = outdir / "v41_16_external_replication_summary.txt"
    manifest_path = outdir / "v41_16_external_replication_manifest.json"

    real_rows = []

    for metric in metric_names:
        s = summary[metric]
        n = s["null"]

        real_rows.append(
            {
                "metric": metric,
                "real": s["real"],
                "null_mean": n["mean"],
                "null_sd": n["sd"],
                "null_q025": n["q025"],
                "null_q975": n["q975"],
                "real_minus_null_mean": s["real_minus_null_mean"],
                "empirical_one_sided_p": s["empirical_one_sided_p"],
            }
        )

    pd.DataFrame(real_rows).to_csv(
        real_csv,
        index=False,
    )

    null_df.to_csv(
        null_csv,
        index=False,
    )

    labels = {
        "top4_eigensubspace_overlap":
            "PRIMARY CONFIRMATORY: Top-4 gene-eigensubspace overlap",
        "corr_structure_similarity":
            "SECONDARY: Correlation-structure similarity",
        "top8_eigensubspace_overlap":
            "EXPLORATORY: Top-8 gene-eigensubspace overlap",
        "eigenvalue_profile_cosine":
            "EXPLORATORY: Eigenvalue-profile cosine similarity",
    }

    lines = [
        f"=== {VERSION} PREREGISTERED EXTERNAL REPLICATION ===",
        "",
        "DESIGN",
        "------",
        "Development reference: GSE10846",
        "External replication: GSE159472",
        "Platform: GPL570",
        "Frozen panel size: 12",
        f"NULL panels: {args.n_null}",
        "External disease/outcome labels used: NO",
        "",
        "FROZEN PANEL",
        "------------",
    ]

    lines += [
        f"{i:2d}. {g}"
        for i, g in enumerate(FROZEN_PANEL, 1)
    ]

    for metric in metric_names:
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
        "CONFIRMATORY DECISION",
        "---------------------",
        f"PRIMARY PASS RULE SATISFIED: {primary_pass}",
        f"v41.16 CONFIRMATORY STATUS: {confirmatory_status}",
        "",
        "SECONDARY SUPPORT",
        "-----------------",
        f"Secondary support criterion satisfied: {secondary_support}",
        "",
        "INTERPRETATION LIMIT",
        "--------------------",
        "A PASS supports only the preregistered external top-4",
        "gene-space geometry-transfer claim.",
        "",
        "It does not establish causal biology, clinical utility,",
        "diagnostic approval, biomarker superiority, or quantum advantage.",
    ]

    summary_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    script_path = Path(__file__).resolve()

    manifest = {
        "version": VERSION,
        "status": confirmatory_status,
        "primary_pass": primary_pass,
        "secondary_support": secondary_support,
        "development_cohort": "GSE10846",
        "external_cohort": "GSE159472",
        "platform": "GPL570",
        "panel_size": len(FROZEN_PANEL),
        "frozen_panel": FROZEN_PANEL,
        "primary_endpoint": "top4_gene_eigensubspace_overlap",
        "n_null": args.n_null,
        "seed": args.seed,
        "primary_decision_rule": {
            "real_gt_null_mean": True,
            "empirical_one_sided_p_lte": 0.05,
        },
        "external_labels_used": False,
        "metrics": summary,
        "sha256": {
            "preregistration": sha256_file(prereg_path),
            "mapping_audit": sha256_file(audit_path),
            "prepare_manifest": sha256_file(prep_manifest_path),
            "GSE10846_expression": sha256_file(train_path),
            "GSE159472_series_matrix": sha256_file(series_path),
            "GPL570_platform_table": sha256_file(gpl_path),
            "execution_script": sha256_file(script_path),
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
        summary_txt,
        manifest_path,
    ):
        print(" ", p)


if __name__ == "__main__":
    main()
