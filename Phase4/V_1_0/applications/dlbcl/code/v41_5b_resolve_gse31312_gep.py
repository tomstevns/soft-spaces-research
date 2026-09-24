
#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v41.5b — GSE31312 GEP-only COO resolution.

Purpose
-------
Resolve GSE31312 cell-of-origin labels using ONLY the GEO metadata field:

    "gene expression profiling subgroup: ..."

and explicitly ignore:

    "immunohistochemistry subgroup: ..."

for COO ground truth.

Why
---
v41.5 found two biologically related but non-identical subtype fields, which
created artificial conflicts when both were treated as equivalent labels.

This script:
  1. reads the already downloaded GSE31312 Series Matrix,
  2. identifies the exact GEP subgroup metadata row,
  3. resolves GCB / ABC from that row only,
  4. records IHC separately for audit,
  5. reports GEP-vs-IHC concordance/discordance,
  6. writes the 47 non-GEP-labelled samples separately,
  7. DOES NOT force the dataset down to any published downstream sample count.

No Soft Spaces fit, no parameter tuning, no Aer, and no QPU execution.

Run
---
From:
    Phase4\\V_1_0\\applications\\dlbcl\\code

Command:
    python -X utf8 -u .\\v41_5b_resolve_gse31312_gep.py
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VERSION = "v41.5b-gse31312-gep-only-resolution-1"

ACCESSION = "GSE31312"

EXPECTED_SERIES_TOTAL = 498
EXPECTED_GEO_GEP = {
    "GCB": 237,
    "ABC": 214,
}
EXPECTED_GEO_GEP_TOTAL = 451


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip('"'))


def parse_sample_rows(matrix_path: Path):
    rows = []
    occurrences = Counter()

    with gzip.open(matrix_path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if line == "!series_matrix_table_begin":
                break

            if not line.startswith("!Sample_"):
                continue

            parts = next(csv.reader([line], delimiter="\t", quotechar='"'))
            key = parts[0]
            vals = [clean(v) for v in parts[1:]]

            occurrences[key] += 1
            rows.append({
                "key": key,
                "occurrence": occurrences[key],
                "values": vals,
            })

    sample_row = next(
        (r for r in rows if r["key"] == "!Sample_geo_accession"),
        None,
    )
    if sample_row is None:
        raise RuntimeError("No !Sample_geo_accession row found.")

    sample_ids = sample_row["values"]
    return rows, sample_ids


def find_exact_subgroup_row(rows, sample_count, prefix: str):
    """
    Find one aligned metadata row where non-empty values consistently begin
    with the requested prefix.
    """
    matches = []

    for r in rows:
        if len(r["values"]) != sample_count:
            continue

        nonempty = [v for v in r["values"] if v]
        if not nonempty:
            continue

        n_prefix = sum(v.lower().startswith(prefix.lower()) for v in nonempty)

        # Exact row if nearly all non-empty values use this field label.
        if n_prefix == len(nonempty):
            matches.append(r)

    if len(matches) != 1:
        details = [
            f'{r["key"]}#{r["occurrence"]}'
            for r in matches
        ]
        raise RuntimeError(
            f"Expected exactly one row for prefix {prefix!r}; "
            f"found {len(matches)}: {details}"
        )

    return matches[0]


def parse_subgroup_value(value: str, prefix: str):
    if not value:
        return ""

    t = clean(value)

    m = re.match(
        rf"^{re.escape(prefix)}\s*:\s*(.+?)\s*$",
        t,
        flags=re.I,
    )
    if not m:
        return ""

    v = m.group(1).strip().upper()

    if v == "GCB":
        return "GCB"
    if v == "ABC":
        return "ABC"

    if v in {"UNCLASSIFIED", "UNCLASSIFIABLE", "UC", "TYPE III", "TYPE 3"}:
        return "Unclassified"

    if v in {"NA", "N/A", "NONE", "UNKNOWN", ""}:
        return ""

    return f"OTHER:{m.group(1).strip()}"


def get_title_row(rows, sample_count):
    rows2 = [
        r for r in rows
        if r["key"] == "!Sample_title"
        and len(r["values"]) == sample_count
    ]
    if len(rows2) != 1:
        return None
    return rows2[0]


def main() -> int:
    code_dir = Path(__file__).resolve().parent
    project_dir = code_dir.parent

    ext_root = project_dir / "data" / "external" / ACCESSION
    raw_dir = ext_root / "raw"
    meta_dir = ext_root / "metadata"

    matrix_path = raw_dir / "GSE31312_series_matrix.txt.gz"

    out_all = meta_dir / "GSE31312_COO_GEP_only.csv"
    out_binary = meta_dir / "GSE31312_ABC_GCB_GEP_only.csv"
    out_unresolved = meta_dir / "GSE31312_nonbinary_or_missing_GEP.csv"
    out_discordance = meta_dir / "GSE31312_GEP_IHC_discordance.csv"
    out_audit = meta_dir / "v41_5b_GEP_only_audit.json"

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Missing {matrix_path}\n"
            "Run v41_5_external_prepare.py first."
        )

    meta_dir.mkdir(parents=True, exist_ok=True)

    print("=== Soft Spaces Phase 4 v41.5b — GSE31312 GEP-only COO ===")
    print()

    rows, sample_ids = parse_sample_rows(matrix_path)
    n = len(sample_ids)

    print(f"Series samples: {n}")
    if n != EXPECTED_SERIES_TOTAL:
        print(f"WARNING: expected {EXPECTED_SERIES_TOTAL}")

    gep_prefix = "gene expression profiling subgroup"
    ihc_prefix = "immunohistochemistry subgroup"

    gep_row = find_exact_subgroup_row(rows, n, gep_prefix)
    ihc_row = find_exact_subgroup_row(rows, n, ihc_prefix)
    title_row = get_title_row(rows, n)

    print(
        f"GEP source: {gep_row['key']}#{gep_row['occurrence']}"
    )
    print(
        f"IHC source: {ihc_row['key']}#{ihc_row['occurrence']}"
    )
    print()

    records = []
    for i, gsm in enumerate(sample_ids):
        gep_raw = gep_row["values"][i]
        ihc_raw = ihc_row["values"][i]

        gep = parse_subgroup_value(gep_raw, gep_prefix)
        ihc = parse_subgroup_value(ihc_raw, ihc_prefix)

        title = title_row["values"][i] if title_row else ""

        if gep in {"ABC", "GCB"}:
            gep_status = "BINARY_RESOLVED"
        elif gep == "Unclassified":
            gep_status = "UNCLASSIFIED"
        elif gep.startswith("OTHER:"):
            gep_status = "OTHER"
        else:
            gep_status = "MISSING"

        if gep in {"ABC", "GCB"} and ihc in {"ABC", "GCB"}:
            concordance = "CONCORDANT" if gep == ihc else "DISCORDANT"
        else:
            concordance = "NOT_COMPARABLE"

        records.append({
            "sample_id": gsm,
            "title": title,
            "gep_coo": gep,
            "gep_status": gep_status,
            "gep_raw": gep_raw,
            "ihc_subgroup": ihc,
            "ihc_raw": ihc_raw,
            "gep_ihc_relation": concordance,
        })

    df = __import__("pandas").DataFrame(records)

    gep_counts = Counter(
        x for x in df["gep_coo"]
        if x
    )
    status_counts = Counter(df["gep_status"])
    relation_counts = Counter(df["gep_ihc_relation"])

    binary_df = df[df["gep_coo"].isin(["ABC", "GCB"])].copy()
    nonbinary_df = df[~df["gep_coo"].isin(["ABC", "GCB"])].copy()
    discord_df = df[df["gep_ihc_relation"] == "DISCORDANT"].copy()

    # Exact GEO header count lock, not a downstream-paper count.
    geo_count_match = (
        gep_counts.get("GCB", 0) == EXPECTED_GEO_GEP["GCB"]
        and gep_counts.get("ABC", 0) == EXPECTED_GEO_GEP["ABC"]
    )

    df.to_csv(out_all, index=False)
    binary_df.to_csv(out_binary, index=False)
    nonbinary_df.to_csv(out_unresolved, index=False)
    discord_df.to_csv(out_discordance, index=False)

    audit = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "accession": ACCESSION,
            "series_samples": n,
        },
        "ground_truth_rule": (
            "Use only the GEO 'gene expression profiling subgroup' field "
            "for COO ground truth. IHC subgroup is audit-only."
        ),
        "sources": {
            "gep": f'{gep_row["key"]}#{gep_row["occurrence"]}',
            "ihc": f'{ihc_row["key"]}#{ihc_row["occurrence"]}',
        },
        "gep_counts": dict(gep_counts),
        "gep_status_counts": dict(status_counts),
        "gep_binary_total": int(len(binary_df)),
        "expected_geo_gep_counts": EXPECTED_GEO_GEP,
        "expected_geo_gep_total": EXPECTED_GEO_GEP_TOTAL,
        "exact_geo_gep_count_match": geo_count_match,
        "gep_ihc_relation_counts": dict(relation_counts),
        "discordant_GEP_vs_IHC_count": int(len(discord_df)),
        "nonbinary_or_missing_GEP_count": int(len(nonbinary_df)),
        "downstream_subset_policy": (
            "No attempt is made in v41.5b to force the 451 GEO GEP-labelled "
            "cases to a smaller published analysis subset. Any later exclusion "
            "must be justified sample-by-sample from a verified source."
        ),
        "frozen_softspaces_parameters_for_later_external_test": {
            "variance_pool": 256,
            "p_dim": 16,
            "feature_budget_k": 16,
            "parameter_tuning_allowed": False,
        },
        "outputs": {
            "all_samples": str(out_all.relative_to(project_dir)),
            "binary_gep_samples": str(out_binary.relative_to(project_dir)),
            "nonbinary_or_missing_gep": str(out_unresolved.relative_to(project_dir)),
            "gep_ihc_discordance": str(out_discordance.relative_to(project_dir)),
        },
        "scope": (
            "COO resolution and audit only. "
            "No Soft Spaces fitting, no model evaluation, no Aer, no QPU."
        ),
    }

    out_audit.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== GEP-ONLY SUMMARY ===")
    print(f"GCB:          {gep_counts.get('GCB', 0)}")
    print(f"ABC:          {gep_counts.get('ABC', 0)}")
    print(f"Unclassified: {gep_counts.get('Unclassified', 0)}")
    print(f"Other:        {status_counts.get('OTHER', 0)}")
    print(f"Missing:      {status_counts.get('MISSING', 0)}")
    print(f"Binary total: {len(binary_df)}")
    print()

    if geo_count_match:
        print("GEO GEP COUNT CROSS-CHECK: PASS")
    else:
        print("GEO GEP COUNT CROSS-CHECK: NOT YET PASS")

    print()
    print("GEP vs IHC:")
    print(f"  concordant:     {relation_counts.get('CONCORDANT', 0)}")
    print(f"  discordant:     {relation_counts.get('DISCORDANT', 0)}")
    print(f"  not comparable: {relation_counts.get('NOT_COMPARABLE', 0)}")
    print()
    print(f"All samples:      {out_all}")
    print(f"Binary GEP set:   {out_binary}")
    print(f"Nonbinary/missing:{out_unresolved}")
    print(f"Discordant set:   {out_discordance}")
    print(f"Audit JSON:       {out_audit}")
    print()
    print("v41.5b COMPLETE.")
    print("No Soft Spaces fit, no Aer, and no QPU execution performed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
