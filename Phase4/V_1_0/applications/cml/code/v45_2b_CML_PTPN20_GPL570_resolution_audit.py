#!/usr/bin/env python3
"""
Soft Spaces / CML
v45.2b — PTPN20 resolution audit on GPL570

Purpose
-------
Investigate the single unresolved v45.2 transport coordinate PTPN20
WITHOUT using outcome labels or predictive scores.

Allowed evidence:
- GPL570 platform annotation in the already-downloaded GSE14671 family SOFT
- exact Gene Symbol fields
- Entrez Gene identifiers
- gene title / gene assignment / synonym-like annotation text

Forbidden:
- outcome labels
- classifier scores
- deleting PTPN20
- substituting a different biological gene
- selecting an alternative probe based on predictive performance

Decision:
1. If a unique, annotation-supported GPL570 probe mapping to PTPN20 can be
   resolved, write a corrected mapping override for use in v45.2c.
2. Otherwise, retain TECHNICALLY NON-EVALUABLE.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


VERSION = "v45.2b"
TARGET = "PTPN20"
GSE = "GSE14671"


def project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_gpl570_table(soft_path: Path) -> pd.DataFrame:
    rows = []
    in_platform = False
    in_table = False
    header = None

    with gzip.open(soft_path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if line.startswith("^PLATFORM"):
                pid = line.split("=", 1)[1].strip()
                in_platform = (pid == "GPL570")
                in_table = False
                header = None
                continue

            if in_platform and line == "!platform_table_begin":
                in_table = True
                header = None
                continue

            if in_platform and line == "!platform_table_end":
                break

            if in_platform and in_table:
                fields = line.split("\t")
                if header is None:
                    header = fields
                    continue

                if len(fields) < len(header):
                    fields += [""] * (len(header) - len(fields))

                rows.append(dict(zip(header, fields[:len(header)])))

    if not rows:
        raise RuntimeError("Could not parse GPL570 platform table.")

    return pd.DataFrame(rows)


def find_column(df, candidates):
    norm = {c.lower().strip(): c for c in df.columns}

    for cand in candidates:
        if cand in norm:
            return norm[cand]

    for c in df.columns:
        cl = c.lower().strip()
        for cand in candidates:
            if cand in cl:
                return c

    return None


def token_match(text: str, token: str) -> bool:
    if not isinstance(text, str):
        return False
    return re.search(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", text.upper()) is not None


def main():
    project = project_dir()
    ds = project / "results" / "direct_subspace"
    metadata = project / "data" / "external" / GSE / "metadata"
    docs = project / "docs"

    soft_path = metadata / "GSE14671_family.soft.gz"
    v452_manifest = ds / "v45_2_manifest.json"
    v451_protocol = docs / "v45_1_CML_GSE14671_CROSS_ENDPOINT_PREREGISTRATION.txt"

    for p in [soft_path, v451_protocol]:
        if not p.exists():
            raise FileNotFoundError(p)

    # v45.2 may have stopped before writing its manifest.
    # If present, verify it; otherwise continue from the printed/known v45.2 state.
    if v452_manifest.exists():
        m452 = json.loads(v452_manifest.read_text(encoding="utf-8"))
        if m452.get("mapped_gene_n") not in (255, None):
            raise RuntimeError("Unexpected v45.2 mapped-gene count.")

    print("=== v45.2b PTPN20 GPL570 RESOLUTION AUDIT ===")
    print("Outcome labels used: NO")
    print("Predictive scores used: NO")
    print("Target unresolved gene:", TARGET)
    print()

    df = parse_gpl570_table(soft_path)

    id_col = find_column(df, ["id", "id_ref", "probe set id"])
    symbol_col = find_column(df, ["gene symbol", "gene_symbol", "symbol"])
    entrez_col = find_column(df, ["entrez gene", "entrez_gene", "gene id", "gene_id"])
    title_col = find_column(df, ["gene title", "gene_title"])
    assignment_col = find_column(df, ["gene assignment", "gene_assignment"])

    print("Detected columns:")
    print("  probe:", id_col)
    print("  symbol:", symbol_col)
    print("  entrez:", entrez_col)
    print("  title:", title_col)
    print("  assignment:", assignment_col)
    print()

    if id_col is None:
        raise RuntimeError("GPL570 probe ID column not found.")

    # Search every annotation column for exact token PTPN20.
    hit_mask = pd.Series(False, index=df.index)

    searched_cols = []

    for c in df.columns:
        if df[c].dtype == object:
            searched_cols.append(c)
            hit_mask |= df[c].fillna("").astype(str).map(
                lambda x: token_match(x, TARGET)
            )

    hits = df.loc[hit_mask].copy()

    candidates_out = ds / "v45_2b_PTPN20_GPL570_annotation_hits.tsv"
    hits.to_csv(candidates_out, sep="\t", index=False)

    # Derive unique probes.
    probe_hits = []

    if not hits.empty:
        for _, row in hits.iterrows():
            probe = str(row[id_col]).strip()
            symbol = str(row[symbol_col]).strip() if symbol_col else ""
            entrez = str(row[entrez_col]).strip() if entrez_col else ""
            title = str(row[title_col]).strip() if title_col else ""
            assign = str(row[assignment_col]).strip() if assignment_col else ""

            exact_symbol = token_match(symbol, TARGET)
            any_annotation = any(
                token_match(x, TARGET)
                for x in [symbol, entrez, title, assign]
            )

            probe_hits.append({
                "probe": probe,
                "gene_symbol": symbol,
                "entrez_gene": entrez,
                "gene_title": title,
                "gene_assignment": assign,
                "exact_symbol_token": exact_symbol,
                "annotation_support": any_annotation,
            })

    ph = pd.DataFrame(probe_hits)

    unique_supported_probes = []

    if not ph.empty:
        unique_supported_probes = sorted(
            set(
                ph.loc[
                    ph["annotation_support"] == True,
                    "probe"
                ].astype(str)
            )
        )

    # Conservative resolution rule:
    # exactly one unique GPL570 probe with annotation text containing exact token PTPN20.
    if len(unique_supported_probes) == 1:
        status = "RESOLVED"
        resolved_probe = unique_supported_probes[0]
    elif len(unique_supported_probes) == 0:
        status = "UNRESOLVED"
        resolved_probe = None
    else:
        status = "AMBIGUOUS"
        resolved_probe = None

    summary_out = ds / "v45_2b_PTPN20_resolution_summary.txt"

    lines = [
        "=== Soft Spaces / CML v45.2b PTPN20 RESOLUTION AUDIT ===",
        "",
        "SCOPE",
        "-----",
        "Outcome labels used: NO",
        "Predictive scores used: NO",
        "Model refit: NO",
        "",
        f"Target gene:                     {TARGET}",
        f"GPL570 annotation rows:          {len(df)}",
        f"Annotation rows containing PTPN20 token: {len(hits)}",
        f"Unique supported probe hits:     {len(unique_supported_probes)}",
    ]

    for p in unique_supported_probes:
        lines.append(f"  {p}")

    lines += [
        "",
        f"v45.2b STATUS: {status}",
    ]

    if status == "RESOLVED":
        lines += [
            f"Resolved probe:                  {resolved_probe}",
            "",
            "The resolution is based only on GPL570 annotation.",
            "No outcome data or model score contributed.",
            "",
            "NEXT:",
            "Run v45.2c corrected technical transport audit using this",
            "single frozen annotation-supported override.",
        ]
    elif status == "AMBIGUOUS":
        lines += [
            "",
            "More than one annotation-supported GPL570 probe remains.",
            "Do not choose among them using outcome or predictive performance.",
            "",
            "v45 remains TECHNICALLY NON-EVALUABLE unless an external",
            "annotation authority resolves the ambiguity outcome-blindly.",
        ]
    else:
        lines += [
            "",
            "No annotation-supported GPL570 probe for PTPN20 was found.",
            "",
            "v45 remains TECHNICALLY NON-EVALUABLE.",
            "Do not delete or replace PTPN20.",
        ]

    summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    override_out = ds / "v45_2b_PTPN20_mapping_override.json"

    override = {
        "version": VERSION,
        "target_gene": TARGET,
        "status": status,
        "resolved_probe": resolved_probe,
        "unique_supported_probes": unique_supported_probes,
        "outcome_labels_used": False,
        "predictive_scores_used": False,
        "model_refit": False,
        "sha256": {
            "gse14671_family_soft": sha256_file(soft_path),
            "v45_1_protocol": sha256_file(v451_protocol),
            "annotation_hits_tsv": sha256_file(candidates_out),
            "execution_script": sha256_file(Path(__file__).resolve()),
        },
    }

    override_out.write_text(
        json.dumps(override, indent=2),
        encoding="utf-8",
    )

    print(summary_out.read_text(encoding="utf-8"))
    print("Wrote:")
    print(" ", candidates_out)
    print(" ", summary_out)
    print(" ", override_out)


if __name__ == "__main__":
    main()
