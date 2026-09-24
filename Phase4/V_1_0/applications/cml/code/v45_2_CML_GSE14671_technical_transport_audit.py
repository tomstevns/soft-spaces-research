#!/usr/bin/env python3
"""
Soft Spaces / CML
v45.2 — GSE14671 technical transport audit

Purpose
-------
Test technical transport of the already-frozen v44.5a Top-256 coordinates
to GSE14671 / GPL570 BEFORE any predictive outcome scoring.

This script:
1. verifies the frozen v45.1 preregistration SHA;
2. reuses the existing GSE14671 family SOFT file;
3. parses GPL570 probe -> gene-symbol annotation from the SOFT platform table;
4. parses all 59 sample expression tables;
5. maps frozen Top-256 genes to GPL570 probes;
6. aggregates multiple probes for one frozen gene by arithmetic mean
   within each sample (outcome-blind, fixed here before scoring);
7. checks 256/256 gene transport;
8. checks non-zero across-sample SD for all frozen genes;
9. writes a frozen gene-level external matrix for later v45.3 scoring.

NO response labels are used.
NO frozen classifier score is calculated.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v45.2"
GSE = "GSE14671"
EXPECTED_N = 59
TOP_K = 256


def project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_symbol_token(x: str) -> str:
    return re.sub(r"\s+", "", x.strip())


def split_gene_symbols(text: str):
    """
    GPL570 annotations can contain separators such as:
      ///  ;  ,  //
    We conservatively split common multi-symbol delimiters.
    """
    if text is None:
        return []

    text = text.strip()

    if not text or text in {"---", "NA", "nan"}:
        return []

    parts = re.split(r"\s*///\s*|\s*//\s*|\s*;\s*|\s*,\s*", text)

    out = []

    for p in parts:
        p = clean_symbol_token(p)
        if p and p not in {"---", "NA"}:
            out.append(p)

    return out


def parse_family_soft(soft_path: Path):
    """
    Parse:
      - GPL570 platform annotation table
      - all SAMPLE expression tables

    Returns:
      probe_to_symbols: dict probe -> list[str]
      sample_values: dict gsm -> dict probe -> float
    """
    probe_to_symbols = {}
    sample_values = {}

    current_entity = None
    current_id = None
    in_platform_table = False
    in_sample_table = False

    platform_header = None
    sample_header = None

    # We will identify the gene-symbol annotation column by header name.
    symbol_col_idx = None
    probe_col_idx = None
    sample_probe_idx = None
    sample_value_idx = None

    with gzip.open(soft_path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if line.startswith("^PLATFORM"):
                current_entity = "PLATFORM"
                current_id = line.split("=", 1)[1].strip()
                in_platform_table = False
                in_sample_table = False
                platform_header = None
                symbol_col_idx = None
                probe_col_idx = None
                continue

            if line.startswith("^SAMPLE"):
                current_entity = "SAMPLE"
                current_id = line.split("=", 1)[1].strip()
                in_platform_table = False
                in_sample_table = False
                sample_header = None
                sample_probe_idx = None
                sample_value_idx = None
                sample_values[current_id] = {}
                continue

            if current_entity == "PLATFORM" and current_id == "GPL570":
                if line == "!platform_table_begin":
                    in_platform_table = True
                    platform_header = None
                    continue

                if line == "!platform_table_end":
                    in_platform_table = False
                    continue

                if in_platform_table:
                    fields = line.split("\t")

                    if platform_header is None:
                        platform_header = fields
                        norm = [x.strip().lower() for x in fields]

                        # Probe ID column
                        for candidate in ("id", "id_ref", "probe set id", "probe_set_id"):
                            if candidate in norm:
                                probe_col_idx = norm.index(candidate)
                                break

                        if probe_col_idx is None:
                            probe_col_idx = 0

                        # Gene symbol column: GPL570 commonly uses "Gene Symbol"
                        symbol_candidates = [
                            "gene symbol",
                            "gene_symbol",
                            "symbol",
                        ]

                        for candidate in symbol_candidates:
                            if candidate in norm:
                                symbol_col_idx = norm.index(candidate)
                                break

                        if symbol_col_idx is None:
                            # fallback: header containing both "gene" and "symbol"
                            for i, h in enumerate(norm):
                                if "gene" in h and "symbol" in h:
                                    symbol_col_idx = i
                                    break

                        if symbol_col_idx is None:
                            raise RuntimeError(
                                "Could not identify GPL570 Gene Symbol column. "
                                f"Platform header: {platform_header}"
                            )

                        continue

                    max_idx = max(probe_col_idx, symbol_col_idx)

                    if len(fields) <= max_idx:
                        continue

                    probe = fields[probe_col_idx].strip()
                    symbols = split_gene_symbols(fields[symbol_col_idx])

                    if probe and symbols:
                        probe_to_symbols[probe] = symbols

                    continue

            if current_entity == "SAMPLE":
                if line == "!sample_table_begin":
                    in_sample_table = True
                    sample_header = None
                    continue

                if line == "!sample_table_end":
                    in_sample_table = False
                    continue

                if in_sample_table:
                    fields = line.split("\t")

                    if sample_header is None:
                        sample_header = fields
                        norm = [x.strip().lower() for x in fields]

                        if "id_ref" not in norm:
                            raise RuntimeError(
                                f"{current_id}: sample table lacks ID_REF."
                            )

                        if "value" not in norm:
                            raise RuntimeError(
                                f"{current_id}: sample table lacks VALUE."
                            )

                        sample_probe_idx = norm.index("id_ref")
                        sample_value_idx = norm.index("value")
                        continue

                    max_idx = max(sample_probe_idx, sample_value_idx)

                    if len(fields) <= max_idx:
                        continue

                    probe = fields[sample_probe_idx].strip()
                    value_text = fields[sample_value_idx].strip()

                    try:
                        value = float(value_text)
                    except ValueError:
                        continue

                    sample_values[current_id][probe] = value

    return probe_to_symbols, sample_values


def main():
    project = project_dir()

    docs = project / "docs"
    ds = project / "results" / "direct_subspace"
    metadata = project / "data" / "external" / GSE / "metadata"

    prereg = docs / "v45_1_CML_GSE14671_CROSS_ENDPOINT_PREREGISTRATION.txt"
    prereg_manifest = docs / "v45_1_CML_GSE14671_CROSS_ENDPOINT_PREREGISTRATION_manifest.json"
    frozen_model = ds / "v44_5a_frozen_development_model.npz"
    soft_path = metadata / "GSE14671_family.soft.gz"

    for p in [prereg, prereg_manifest, frozen_model, soft_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    pm = json.loads(prereg_manifest.read_text(encoding="utf-8"))

    expected_sha = pm["protocol_sha256"]
    actual_sha = sha256_file(prereg)

    print("=== v45.2 GSE14671 TECHNICAL TRANSPORT AUDIT ===")
    print("Expected v45.1 protocol SHA:", expected_sha)
    print("Actual v45.1 protocol SHA:  ", actual_sha)

    if expected_sha != actual_sha:
        raise SystemExit("FAIL: v45.1 protocol SHA mismatch.")

    print("PASS: v45.1 preregistration verified.")
    print("Outcome labels used: NO")
    print("Predictive scoring performed: NO")
    print()

    model = np.load(frozen_model, allow_pickle=False)

    if "genes" not in model.files:
        raise RuntimeError(
            f"{frozen_model.name} does not contain frozen 'genes'. "
            f"Available arrays: {model.files}"
        )

    genes = [str(x) for x in model["genes"].tolist()]

    if len(genes) != TOP_K:
        raise RuntimeError(
            f"Expected {TOP_K} frozen genes, found {len(genes)}."
        )

    print("Parsing GEO family SOFT...")
    probe_to_symbols, sample_values = parse_family_soft(soft_path)

    print("GPL570 annotated probes parsed:", len(probe_to_symbols))
    print("Sample expression tables parsed:", len(sample_values))

    if len(sample_values) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} sample expression tables, "
            f"found {len(sample_values)}."
        )

    # Reverse map symbol -> probes.
    symbol_to_probes = defaultdict(list)

    for probe, symbols in probe_to_symbols.items():
        for symbol in symbols:
            symbol_to_probes[symbol].append(probe)

    mapping_rows = []
    missing = []

    for gene in genes:
        probes = sorted(set(symbol_to_probes.get(gene, [])))

        mapping_rows.append({
            "gene": gene,
            "probe_count": len(probes),
            "probes": " | ".join(probes),
        })

        if not probes:
            missing.append(gene)

    mapping_df = pd.DataFrame(mapping_rows)

    mapping_out = ds / "v45_2_GSE14671_frozen_gene_probe_mapping.tsv"
    mapping_df.to_csv(mapping_out, sep="\t", index=False)

    mapped_n = TOP_K - len(missing)

    print()
    print(f"Frozen genes mapped: {mapped_n}/{TOP_K}")

    if missing:
        summary_out = ds / "v45_2_GSE14671_technical_transport_summary.txt"

        lines = [
            "=== Soft Spaces / CML v45.2 GSE14671 TECHNICAL TRANSPORT AUDIT ===",
            "",
            f"Frozen Top-K:                    {TOP_K}",
            f"Mapped to GPL570:                {mapped_n}",
            f"Missing frozen genes:            {len(missing)}",
            "",
            "STATUS: TECHNICALLY NON-EVALUABLE",
            "",
            "Missing:",
        ]

        lines.extend(f"  {g}" for g in missing)

        lines += [
            "",
            "No response labels were used.",
            "No predictive score was calculated.",
            "No gene substitution is permitted.",
        ]

        summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

        print(summary_out.read_text(encoding="utf-8"))
        raise SystemExit(
            "TECHNICALLY NON-EVALUABLE: one or more frozen genes "
            "cannot be mapped to GPL570."
        )

    # Build gene-level matrix.
    gsms = sorted(sample_values.keys())
    X = np.empty((len(gsms), TOP_K), dtype=float)

    incomplete = []

    for j, gene in enumerate(genes):
        probes = sorted(set(symbol_to_probes[gene]))

        for i, gsm in enumerate(gsms):
            vals = [
                sample_values[gsm][p]
                for p in probes
                if p in sample_values[gsm]
            ]

            if not vals:
                incomplete.append((gsm, gene))
                X[i, j] = np.nan
            else:
                # Frozen, outcome-blind multi-probe aggregation rule.
                X[i, j] = float(np.mean(vals))

    if incomplete:
        first = incomplete[:20]
        raise RuntimeError(
            "TECHNICALLY NON-EVALUABLE: missing sample/gene expression "
            f"values. First entries: {first}"
        )

    if not np.all(np.isfinite(X)):
        raise RuntimeError(
            "TECHNICALLY NON-EVALUABLE: non-finite external expression."
        )

    sd = np.std(X, axis=0, ddof=0)
    zero_idx = np.where(sd == 0)[0]
    zero_genes = [genes[i] for i in zero_idx]

    matrix_out = ds / "v45_2_GSE14671_frozen_top256_expression.tsv"

    matrix_df = pd.DataFrame(
        X,
        index=gsms,
        columns=genes,
    )
    matrix_df.index.name = "gsm"
    matrix_df.to_csv(matrix_out, sep="\t")

    stats_out = ds / "v45_2_GSE14671_frozen_gene_variation.tsv"

    stats_df = pd.DataFrame({
        "gene": genes,
        "mean_expression": np.mean(X, axis=0),
        "sd_expression_ddof0": sd,
        "zero_sd": sd == 0,
    })
    stats_df.to_csv(stats_out, sep="\t", index=False)

    summary_out = ds / "v45_2_GSE14671_technical_transport_summary.txt"

    status = (
        "TECHNICALLY EVALUABLE"
        if len(zero_genes) == 0
        else "TECHNICALLY NON-EVALUABLE"
    )

    multi_probe_n = int((mapping_df["probe_count"] > 1).sum())
    single_probe_n = int((mapping_df["probe_count"] == 1).sum())

    lines = [
        "=== Soft Spaces / CML v45.2 GSE14671 TECHNICAL TRANSPORT AUDIT ===",
        "",
        "PREREGISTRATION",
        "---------------",
        f"v45.1 SHA verified:              {actual_sha}",
        "",
        "NO OUTCOME SCORING",
        "------------------",
        "Response labels used:            NO",
        "Frozen classifier score:         NOT CALCULATED",
        "External model refit:            NO",
        "Outcome tuning:                  NO",
        "",
        "PLATFORM TRANSPORT",
        "------------------",
        f"External cohort:                 {GSE}",
        f"Platform:                        GPL570",
        f"Samples parsed:                  {len(gsms)}",
        f"Frozen Top-K:                    {TOP_K}",
        f"Mapped frozen genes:             {mapped_n}",
        f"Single-probe genes:              {single_probe_n}",
        f"Multi-probe genes:               {multi_probe_n}",
        "",
        "FROZEN PROBE AGGREGATION",
        "------------------------",
        "Multiple GPL570 probes for the same frozen gene are aggregated",
        "within each sample by arithmetic mean.",
        "This rule is outcome-blind and fixed before predictive scoring.",
        "",
        "VARIATION GATE",
        "--------------",
        f"Zero-SD frozen genes:            {len(zero_genes)}",
    ]

    if zero_genes:
        lines.extend(f"  {g}" for g in zero_genes)

    lines += [
        "",
        f"v45.2 STATUS: {status}",
        "",
    ]

    if status == "TECHNICALLY EVALUABLE":
        lines += [
            "All 256 frozen genes are represented on GPL570 and have",
            "non-zero variation across the 59 GSE14671 samples.",
            "",
            "NEXT:",
            "Freeze/use this exact gene-level matrix for v45.3 outcome scoring.",
        ]
    else:
        lines += [
            "The preregistered technical gate failed.",
            "No predictive scoring should be performed.",
            "No gene deletion or substitution is permitted.",
        ]

    summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_out = ds / "v45_2_manifest.json"

    manifest = {
        "version": VERSION,
        "dataset": GSE,
        "platform": "GPL570",
        "sample_n": len(gsms),
        "frozen_top_k": TOP_K,
        "mapped_gene_n": mapped_n,
        "single_probe_gene_n": single_probe_n,
        "multi_probe_gene_n": multi_probe_n,
        "zero_sd_gene_n": len(zero_genes),
        "zero_sd_genes": zero_genes,
        "status": status,
        "response_labels_used": False,
        "predictive_scoring_performed": False,
        "external_model_refit": False,
        "outcome_tuning": False,
        "probe_aggregation": "arithmetic mean across all GPL570 probes mapping to frozen gene",
        "sha256": {
            "v45_1_protocol": actual_sha,
            "v44_5a_frozen_model": sha256_file(frozen_model),
            "gse14671_family_soft": sha256_file(soft_path),
            "mapping_tsv": sha256_file(mapping_out),
            "expression_matrix_tsv": sha256_file(matrix_out),
            "variation_tsv": sha256_file(stats_out),
            "execution_script": sha256_file(Path(__file__).resolve()),
        },
    }

    manifest_out.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print(summary_out.read_text(encoding="utf-8"))

    print("Wrote:")
    for p in [
        mapping_out,
        matrix_out,
        stats_out,
        summary_out,
        manifest_out,
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
