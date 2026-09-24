#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v41.1a — GSE10846 preprocessing for DLBCL benchmark.

Purpose
-------
Create a clean, reproducible ABC-vs-GCB gene-expression matrix from the
locally downloaded GSE10846 Series Matrix and GPL570 annotation.

Inputs expected from v41.0/v41.0c
---------------------------------
applications/dlbcl/
    data/raw/GSE10846_series_matrix.txt.gz
    data/raw/GPL570.annot.gz
    data/metadata/GSE10846_COO_resolved_v41_0c.csv

Outputs
-------
data/processed/
    GSE10846_ABC_GCB_gene_expression.csv.gz
    GSE10846_ABC_GCB_sample_labels.csv
    GSE10846_probe_to_gene_mapping.csv.gz
    v41_1_preprocessing_manifest.json

Scientific choices
------------------
1. Use only samples with verified COO label ABC or GCB.
2. Do NOT force the 64 Unclassified samples into either binary class.
3. Do NOT use the 6 samples with missing COO metadata.
4. Preserve the processed GEO expression values as supplied in the Series Matrix.
5. Map GPL570 probes to gene symbols conservatively.
6. Exclude probes without a usable gene symbol.
7. If one probe maps to multiple gene symbols, expand it to those symbols.
8. If multiple probes map to the same gene, aggregate per sample by the median.
9. Do NOT standardize/z-score globally here. Scaling belongs inside training
   folds in v41.2 to avoid information leakage.
10. No modelling, REAL/NULL construction, Aer, or QPU execution in v41.1.

Run
---
From:
    Phase4\\V_1_0\\applications\\dlbcl\\code

Command:
    python -X utf8 -u .\\v41_1_preprocess.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v41.1a-dlbcl-preprocess-gplfix-1"
ACCESSION = "GSE10846"
PLATFORM = "GPL570"
ALLOWED_LABELS = {"ABC", "GCB"}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_coo_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"sample_id", "coo_label", "status"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"COO file missing columns: {sorted(missing)}")

    df = df[df["status"].eq("RESOLVED")].copy()
    df = df[df["coo_label"].isin(ALLOWED_LABELS)].copy()

    if df["sample_id"].duplicated().any():
        dup = df.loc[df["sample_id"].duplicated(), "sample_id"].tolist()[:10]
        raise RuntimeError(f"Duplicate sample IDs in COO table, e.g. {dup}")

    counts = df["coo_label"].value_counts().to_dict()
    expected = {"GCB": 183, "ABC": 167}
    if counts != expected:
        raise RuntimeError(
            f"ABC/GCB counts do not match locked v41.0c ground truth. "
            f"Observed={counts}, expected={expected}"
        )

    return df[["sample_id", "title", "coo_label"]].copy()


def read_series_matrix(path: Path):
    """
    Read only the expression table from GEO Series Matrix.
    Returns DataFrame indexed by probe ID, columns = GSM sample IDs.
    """
    rows = []
    header = None
    in_table = False

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")

            if line == "!series_matrix_table_begin":
                in_table = True
                continue
            if line == "!series_matrix_table_end":
                break
            if not in_table:
                continue

            parts = next(csv.reader([line], delimiter="\t", quotechar='"'))

            if header is None:
                header = parts
                continue

            rows.append(parts)

    if header is None:
        raise RuntimeError("Series Matrix expression table header not found.")

    # GEO normally uses ID_REF as first column.
    probe_col = header[0]
    sample_ids = header[1:]

    # Convert rows robustly.
    probe_ids = []
    data = np.empty((len(rows), len(sample_ids)), dtype=np.float32)

    for i, parts in enumerate(rows):
        if len(parts) != len(header):
            raise RuntimeError(
                f"Expression row {i+1} has {len(parts)} columns; expected {len(header)}"
            )
        probe_ids.append(parts[0])

        vals = []
        for x in parts[1:]:
            x = x.strip()
            if x in {"", "NA", "NaN", "nan", "null", "NULL"}:
                vals.append(np.nan)
            else:
                vals.append(float(x))
        data[i, :] = vals

    df = pd.DataFrame(data, index=pd.Index(probe_ids, name="probe_id"), columns=sample_ids)
    return probe_col, df


def detect_annotation_columns(columns):
    """
    Find probe-ID and gene-symbol columns in GPL570 annotation robustly.
    """
    normalized = {c.strip().lower(): c for c in columns}

    probe_candidates = [
        "id",
        "id_ref",
        "probe set id",
        "probe_set_id",
        "probe id",
    ]
    gene_candidates = [
        "gene symbol",
        "gene_symbol",
        "gene symbol ///",
        "symbol",
        "gene assignment",
    ]

    probe_col = None
    gene_col = None

    for cand in probe_candidates:
        if cand in normalized:
            probe_col = normalized[cand]
            break

    # First prefer an explicit gene-symbol field.
    for cand in gene_candidates:
        if cand in normalized:
            gene_col = normalized[cand]
            break

    if probe_col is None:
        # GPL570 annot typically uses ID.
        for c in columns:
            if c.strip().upper() == "ID":
                probe_col = c
                break

    if gene_col is None:
        for c in columns:
            cl = c.strip().lower()
            if "gene symbol" in cl:
                gene_col = c
                break

    if probe_col is None or gene_col is None:
        raise RuntimeError(
            "Could not identify GPL570 probe/gene columns. "
            f"Columns seen: {list(columns)[:40]}"
        )

    return probe_col, gene_col


def split_gene_symbols(value: str):
    """
    Conservative GPL annotation parsing.

    Common separators include:
      "GENE1 /// GENE2"
      "GENE1 // GENE2"
      "GENE1; GENE2"
    """
    if value is None:
        return []

    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "---", "null"}:
        return []

    # Some annotation columns may carry "symbol // description".
    # Prefer ' /// ' as the canonical multi-gene separator.
    parts = []
    if " /// " in text:
        parts = text.split(" /// ")
    elif ";" in text:
        parts = text.split(";")
    else:
        parts = [text]

    cleaned = []
    for p in parts:
        p = p.strip()

        # If a value is of the form SYMBOL // something, take leading token.
        if " // " in p:
            p = p.split(" // ", 1)[0].strip()

        # Gene symbols should not contain whitespace-rich descriptions.
        # Keep plausible gene symbols only.
        if not p or p in {"---", "NA", "na"}:
            continue
        if len(p) > 80:
            continue

        cleaned.append(p)

    # de-duplicate while preserving order
    seen = set()
    out = []
    for g in cleaned:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def read_gpl_annotation(path: Path):
    """
    Read GPL570.annot.gz robustly.

    GEO annotation files can contain a metadata preamble with lines that do not
    have the same number of tab-separated fields as the actual annotation table.
    Therefore we do NOT hand the whole gzip file directly to pandas.

    Instead:
      1. scan until the real annotation header beginning with "ID",
      2. collect only table rows,
      3. stop at GEO table-end markers if present,
      4. build the DataFrame from the clean table only.
    """
    header = None
    table_rows = []

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if not line:
                continue

            # Ignore GEO metadata/preamble until the true tabular header.
            if header is None:
                parts = next(csv.reader([line], delimiter="\t", quotechar='"'))
                if parts and parts[0].strip().upper() == "ID" and len(parts) > 1:
                    header = [p.strip() for p in parts]
                continue

            # Stop at possible GEO end markers.
            if line.startswith("!platform_table_end"):
                break

            # Ignore any residual metadata lines after the header.
            if line.startswith(("!", "^", "#")):
                continue

            parts = next(csv.reader([line], delimiter="\t", quotechar='"'))

            # Keep only rows matching the annotation table width.
            if len(parts) != len(header):
                continue

            table_rows.append(parts)

    if header is None:
        raise RuntimeError(
            "Could not locate GPL570 annotation table header beginning with 'ID'."
        )

    if not table_rows:
        raise RuntimeError(
            "GPL570 annotation table header was found, but no table rows were read."
        )

    df = pd.DataFrame(table_rows, columns=header)

    probe_col, gene_col = detect_annotation_columns(df.columns)

    rows = []
    for _, r in df[[probe_col, gene_col]].fillna("").iterrows():
        probe = str(r[probe_col]).strip()
        if not probe:
            continue
        genes = split_gene_symbols(r[gene_col])
        for gene in genes:
            rows.append((probe, gene))

    mapping = pd.DataFrame(
        rows, columns=["probe_id", "gene_symbol"]
    ).drop_duplicates()

    if mapping.empty:
        raise RuntimeError(
            f"GPL570 annotation was read, but no probe-to-gene mappings were produced. "
            f"Probe column={probe_col!r}, gene column={gene_col!r}"
        )

    return mapping, probe_col, gene_col


def expression_qc(df: pd.DataFrame):
    arr = df.to_numpy(dtype=np.float64)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        raise RuntimeError("Expression matrix contains no finite values.")

    q = np.quantile(finite, [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])

    return {
        "n_probes": int(df.shape[0]),
        "n_samples": int(df.shape[1]),
        "finite_fraction": float(np.isfinite(arr).mean()),
        "min": float(q[0]),
        "q01": float(q[1]),
        "q25": float(q[2]),
        "median": float(q[3]),
        "q75": float(q[4]),
        "q99": float(q[5]),
        "max": float(q[6]),
        "sample_mean_range": [
            float(np.nanmin(np.nanmean(arr, axis=0))),
            float(np.nanmax(np.nanmean(arr, axis=0))),
        ],
        "sample_sd_range": [
            float(np.nanmin(np.nanstd(arr, axis=0, ddof=1))),
            float(np.nanmax(np.nanstd(arr, axis=0, ddof=1))),
        ],
    }


def aggregate_probe_to_gene(expr_probe: pd.DataFrame, mapping: pd.DataFrame):
    """
    Aggregate probes to genes with median expression across probes.

    Returns:
      gene x sample DataFrame
      mapping QC dict
    """
    common = mapping[mapping["probe_id"].isin(expr_probe.index)].copy()

    if common.empty:
        raise RuntimeError("No GPL570 probe IDs matched expression matrix probe IDs.")

    # Expand expression rows by mapping then median over gene.
    probe_to_row = expr_probe.loc[common["probe_id"].unique()]

    # Join by explicit merge in long-ish index form without exploding samples to full long table.
    grouped = defaultdict(list)
    for probe, gene in common[["probe_id", "gene_symbol"]].itertuples(index=False):
        grouped[gene].append(probe)

    gene_names = sorted(grouped)
    out = np.empty((len(gene_names), expr_probe.shape[1]), dtype=np.float32)

    probes_per_gene = []
    for i, gene in enumerate(gene_names):
        probes = list(dict.fromkeys(grouped[gene]))
        vals = expr_probe.loc[probes].to_numpy(dtype=np.float32)
        out[i, :] = np.nanmedian(vals, axis=0)
        probes_per_gene.append(len(probes))

    gene_df = pd.DataFrame(
        out,
        index=pd.Index(gene_names, name="gene_symbol"),
        columns=expr_probe.columns,
    )

    qc = {
        "mapping_rows_probe_gene": int(len(common)),
        "unique_mapped_probes": int(common["probe_id"].nunique()),
        "unique_genes": int(common["gene_symbol"].nunique()),
        "probes_per_gene_min": int(min(probes_per_gene)),
        "probes_per_gene_median": float(statistics.median(probes_per_gene)),
        "probes_per_gene_max": int(max(probes_per_gene)),
    }

    return gene_df, qc, common


def main() -> int:
    code_dir = Path(__file__).resolve().parent
    project_dir = code_dir.parent

    raw_dir = project_dir / "data" / "raw"
    meta_dir = project_dir / "data" / "metadata"
    processed_dir = project_dir / "data" / "processed"

    matrix_path = raw_dir / "GSE10846_series_matrix.txt.gz"
    annot_path = raw_dir / "GPL570.annot.gz"
    labels_path = meta_dir / "GSE10846_COO_resolved_v41_0c.csv"

    out_expr = processed_dir / "GSE10846_ABC_GCB_gene_expression.csv.gz"
    out_labels = processed_dir / "GSE10846_ABC_GCB_sample_labels.csv"
    out_mapping = processed_dir / "GSE10846_probe_to_gene_mapping.csv.gz"
    out_manifest = processed_dir / "v41_1_preprocessing_manifest.json"

    for p in [matrix_path, annot_path, labels_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=== Soft Spaces Phase 4 v41.1a — DLBCL preprocessing ===")
    print("Locked benchmark: ABC vs GCB")
    print()

    print("[1/6] Reading verified COO labels...")
    labels = read_coo_labels(labels_path)
    label_counts = labels["coo_label"].value_counts().to_dict()
    selected_samples = labels["sample_id"].tolist()
    print(f"      samples: {len(labels)}  counts: {label_counts}")

    print("[2/6] Reading GSE10846 expression matrix...")
    probe_col_name, expr_all = read_series_matrix(matrix_path)
    qc_all = expression_qc(expr_all)
    print(f"      probes: {expr_all.shape[0]}")
    print(f"      GEO samples: {expr_all.shape[1]}")

    missing_samples = sorted(set(selected_samples) - set(expr_all.columns))
    if missing_samples:
        raise RuntimeError(
            f"{len(missing_samples)} labelled samples absent from expression matrix, "
            f"e.g. {missing_samples[:10]}"
        )

    expr = expr_all.loc[:, selected_samples].copy()
    print(f"      selected ABC/GCB samples: {expr.shape[1]}")

    print("[3/6] Reading GPL570 annotation...")
    mapping, gpl_probe_col, gpl_gene_col = read_gpl_annotation(annot_path)
    print(f"      probe-gene mapping rows: {len(mapping)}")
    print(f"      unique annotated probes: {mapping['probe_id'].nunique()}")
    print(f"      unique annotated genes: {mapping['gene_symbol'].nunique()}")

    print("[4/6] Aggregating probes -> gene symbols by median...")
    gene_expr, mapping_qc, mapping_used = aggregate_probe_to_gene(expr, mapping)
    qc_gene = expression_qc(gene_expr)
    print(f"      resulting genes: {gene_expr.shape[0]}")
    print(f"      samples: {gene_expr.shape[1]}")

    # Drop genes with no finite values; should normally be none.
    finite_per_gene = np.isfinite(gene_expr.to_numpy()).any(axis=1)
    dropped_all_missing = int((~finite_per_gene).sum())
    if dropped_all_missing:
        gene_expr = gene_expr.loc[finite_per_gene].copy()

    # Preserve sample order matching labels CSV.
    assert gene_expr.columns.tolist() == selected_samples

    print("[5/6] Writing processed benchmark files...")

    # Expression orientation: samples x genes is convenient for ML.
    sample_by_gene = gene_expr.T
    sample_by_gene.index.name = "sample_id"
    sample_by_gene.to_csv(out_expr, compression="gzip")

    labels.to_csv(out_labels, index=False)

    mapping_used.sort_values(["gene_symbol", "probe_id"]).to_csv(
        out_mapping,
        index=False,
        compression="gzip",
    )

    print("[6/6] Writing reproducibility manifest...")

    manifest = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "accession": ACCESSION,
            "platform": PLATFORM,
            "task": "ABC vs GCB binary benchmark",
            "n_samples": int(len(labels)),
            "class_counts": label_counts,
            "excluded_from_binary_benchmark": {
                "Unclassified": 64,
                "missing_or_unresolved_COO": 6,
            },
        },
        "inputs": {
            "series_matrix": {
                "path": str(matrix_path.relative_to(project_dir)),
                "sha256": sha256_file(matrix_path),
            },
            "platform_annotation": {
                "path": str(annot_path.relative_to(project_dir)),
                "sha256": sha256_file(annot_path),
            },
            "verified_labels": {
                "path": str(labels_path.relative_to(project_dir)),
                "sha256": sha256_file(labels_path),
            },
        },
        "annotation": {
            "series_matrix_probe_column": probe_col_name,
            "gpl_probe_column": gpl_probe_col,
            "gpl_gene_column": gpl_gene_col,
            "aggregation": "median across mapped probes per gene, per sample",
            **mapping_qc,
        },
        "expression_qc_before_sample_filter": qc_all,
        "expression_qc_gene_level_ABC_GCB": qc_gene,
        "all_missing_genes_dropped": dropped_all_missing,
        "preprocessing_policy": {
            "input_expression": "GEO processed Series Matrix values preserved",
            "extra_log_transform": False,
            "global_zscore": False,
            "global_feature_selection": False,
            "reason": (
                "Any scaling/feature selection must occur inside training folds "
                "in v41.2 to prevent information leakage."
            ),
        },
        "outputs": {
            "sample_by_gene_expression": str(out_expr.relative_to(project_dir)),
            "sample_labels": str(out_labels.relative_to(project_dir)),
            "probe_gene_mapping_used": str(out_mapping.relative_to(project_dir)),
        },
        "scope": (
            "Preprocessing only. No model fitting, benchmark scoring, "
            "REAL/NULL construction, Aer simulation, or QPU execution."
        ),
    }

    out_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=== v41.1 SUMMARY ===")
    print(f"ABC samples: {label_counts.get('ABC', 0)}")
    print(f"GCB samples: {label_counts.get('GCB', 0)}")
    print(f"Total benchmark samples: {len(labels)}")
    print(f"Gene features: {sample_by_gene.shape[1]}")
    print(
        "Expression range "
        f"{qc_gene['min']:.3f} .. {qc_gene['max']:.3f}; "
        f"median={qc_gene['median']:.3f}"
    )
    print(f"Output expression: {out_expr}")
    print(f"Output labels:     {out_labels}")
    print(f"Output mapping:    {out_mapping}")
    print(f"Manifest:          {out_manifest}")
    print()
    print("v41.1a COMPLETE.")
    print("No modelling, REAL/NULL, Aer simulation, or QPU execution performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
