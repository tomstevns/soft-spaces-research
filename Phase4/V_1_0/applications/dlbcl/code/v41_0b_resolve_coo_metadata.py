#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v41.0b — Resolve GSE10846 DLBCL cell-of-origin labels.

Why this script exists
----------------------
v41.0 downloaded GSE10846 correctly, but its first conservative metadata parser
reported all 420 samples as UNRESOLVED. The reason can be structural: GEO Series
Matrix files may contain multiple repeated !Sample_characteristics_ch1 rows.
A parser that concatenates those rows loses their per-sample alignment.

v41.0b therefore:
  1. Reads the existing local GSE10846 Series Matrix header.
  2. Preserves EVERY repeated sample-metadata row separately.
  3. Reconstructs all metadata fields sample-by-sample.
  4. Searches conservatively for published COO labels:
       ABC, GCB, Unclassified / Type III.
  5. Records the exact metadata text that produced each label.
  6. Cross-checks the resulting counts against published GSE10846 totals.
  7. Writes a verified/auditable COO table and an audit JSON.

No expression values are modified. No modelling, REAL/NULL construction, Aer,
or QPU execution is performed.

Expected published benchmark counts
-----------------------------------
For the 414 DLBCL cases used in the published COO analyses:
  ABC          167
  GCB          183
  Unclassified  64
Total          414

The GEO Series Matrix contains 420 samples, so up to six samples may remain
outside that three-class DLBCL benchmark. The script does NOT force those six
into a COO class.

Run from:
    Phase4\\V_1_0\\applications\\dlbcl\\code

Command:
    python -X utf8 -u .\\v41_0b_resolve_coo_metadata.py
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


VERSION = "v41.0b-dlbcl-resolve-coo-1"
ACCESSION = "GSE10846"

EXPECTED_PUBLISHED = {
    "ABC": 167,
    "GCB": 183,
    "Unclassified": 64,
}
EXPECTED_PUBLISHED_TOTAL = sum(EXPECTED_PUBLISHED.values())  # 414


def clean(s: str) -> str:
    s = s.strip().strip('"')
    return re.sub(r"\s+", " ", s)


def parse_header_rows(matrix_path: Path):
    """
    Preserve repeated GEO header rows as separate row occurrences.

    Returns
    -------
    rows : list[dict]
        Each item:
          {
            "key": "!Sample_characteristics_ch1",
            "occurrence": 3,
            "values": [... one value per sample ...]
          }
    sample_ids : list[str]
    """
    rows = []
    occurrence_counter = Counter()

    with gzip.open(matrix_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line == "!series_matrix_table_begin":
                break
            if not line.startswith("!Sample_"):
                continue

            parts = next(csv.reader([line], delimiter="\t", quotechar='"'))
            key = parts[0]
            values = [clean(v) for v in parts[1:]]
            occurrence_counter[key] += 1
            rows.append(
                {
                    "key": key,
                    "occurrence": occurrence_counter[key],
                    "values": values,
                }
            )

    sample_rows = [r for r in rows if r["key"] == "!Sample_geo_accession"]
    if not sample_rows:
        raise RuntimeError("No !Sample_geo_accession row found.")
    if len(sample_rows) != 1:
        raise RuntimeError(
            f"Expected one !Sample_geo_accession row, found {len(sample_rows)}."
        )

    sample_ids = sample_rows[0]["values"]
    return rows, sample_ids


def classify_text(text: str):
    """
    Conservative COO classification from a single metadata string.

    Returns (label, rule) or ("", "").

    Rules deliberately require explicit COO-like wording or an isolated class
    value. The function avoids guessing from unrelated gene names.
    """
    original = clean(text)
    t = original.lower()

    # Strong explicit phrases first.
    explicit_rules = [
        (
            "ABC",
            [
                r"\bactivated b[- ]?cell(?:[- ]like)?\b",
                r"\bactivated b cell[- ]like\b",
                r"\bcell[ -]?of[ -]?origin\s*[:=]\s*abc\b",
                r"\bcoo\s*[:=]\s*abc\b",
                r"\bsubtype\s*[:=]\s*abc\b",
                r"\bmolecular subtype\s*[:=]\s*abc\b",
            ],
        ),
        (
            "GCB",
            [
                r"\bgerminal cent(?:er|re) b[- ]?cell(?:[- ]like)?\b",
                r"\bcell[ -]?of[ -]?origin\s*[:=]\s*gcb\b",
                r"\bcoo\s*[:=]\s*gcb\b",
                r"\bsubtype\s*[:=]\s*gcb\b",
                r"\bmolecular subtype\s*[:=]\s*gcb\b",
            ],
        ),
        (
            "Unclassified",
            [
                r"\bcell[ -]?of[ -]?origin\s*[:=]\s*(?:unclassified|type ?iii|uc)\b",
                r"\bcoo\s*[:=]\s*(?:unclassified|type ?iii|uc)\b",
                r"\bsubtype\s*[:=]\s*(?:unclassified|type ?iii|uc)\b",
                r"\bmolecular subtype\s*[:=]\s*(?:unclassified|type ?iii|uc)\b",
                r"\btype ?iii\b",
            ],
        ),
    ]

    for label, patterns in explicit_rules:
        for p in patterns:
            if re.search(p, t, flags=re.I):
                return label, p

    # GEO sometimes stores a characteristic simply as "ABC", "GCB", etc.
    stripped = re.sub(r"^[^:]{1,60}:\s*", "", t).strip()
    if stripped in {"abc", "abc-like", "abc like"}:
        return "ABC", "isolated-class-value"
    if stripped in {"gcb", "gcb-like", "gcb like"}:
        return "GCB", "isolated-class-value"
    if stripped in {
        "unclassified",
        "unclassifiable",
        "type iii",
        "type-iii",
        "type 3",
        "uc",
    }:
        return "Unclassified", "isolated-class-value"

    return "", ""


def reconstruct_samples(rows, sample_ids):
    n = len(sample_ids)

    # Validate row cardinalities and keep only sample-aligned rows.
    aligned_rows = []
    bad_rows = []
    for r in rows:
        if len(r["values"]) == n:
            aligned_rows.append(r)
        else:
            bad_rows.append(
                {
                    "key": r["key"],
                    "occurrence": r["occurrence"],
                    "value_count": len(r["values"]),
                }
            )

    records = []
    for i, gsm in enumerate(sample_ids):
        evidence = []
        all_metadata = []

        for r in aligned_rows:
            if r["key"] == "!Sample_geo_accession":
                continue
            value = r["values"][i]
            if not value:
                continue

            source = f'{r["key"]}#{r["occurrence"]}'
            all_metadata.append(f"{source}: {value}")

            label, rule = classify_text(value)
            if label:
                evidence.append(
                    {
                        "label": label,
                        "source": source,
                        "text": value,
                        "rule": rule,
                    }
                )

        labels = sorted({e["label"] for e in evidence})

        if len(labels) == 1:
            resolved = labels[0]
            status = "RESOLVED"
        elif len(labels) == 0:
            resolved = ""
            status = "UNRESOLVED"
        else:
            resolved = ""
            status = "CONFLICT"

        title = ""
        for r in aligned_rows:
            if r["key"] == "!Sample_title" and r["values"][i]:
                title = r["values"][i]
                break

        records.append(
            {
                "sample_id": gsm,
                "title": title,
                "coo_label": resolved,
                "status": status,
                "evidence": evidence,
                "metadata_text": all_metadata,
            }
        )

    return records, bad_rows, aligned_rows


def candidate_field_summary(aligned_rows, sample_ids):
    """
    Produce an audit view of repeated metadata rows that look potentially
    relevant to COO classification even when strict rules do not resolve them.
    """
    hints = re.compile(
        r"\b(abc|gcb|germinal|activated|origin|subtype|classification|type iii|unclass)\b",
        flags=re.I,
    )

    out = []
    n = len(sample_ids)

    for r in aligned_rows:
        matches = [v for v in r["values"] if v and hints.search(v)]
        if not matches:
            continue

        counts = Counter(matches)
        out.append(
            {
                "source": f'{r["key"]}#{r["occurrence"]}',
                "matched_samples": len(matches),
                "unique_matched_values": len(counts),
                "examples": [
                    {"value": value, "count": count}
                    for value, count in counts.most_common(12)
                ],
            }
        )

    return out


def write_csv(records, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "title",
                "coo_label",
                "status",
                "evidence_source",
                "evidence_text",
            ],
        )
        writer.writeheader()

        for r in records:
            sources = " | ".join(e["source"] for e in r["evidence"])
            texts = " | ".join(e["text"] for e in r["evidence"])
            writer.writerow(
                {
                    "sample_id": r["sample_id"],
                    "title": r["title"],
                    "coo_label": r["coo_label"],
                    "status": r["status"],
                    "evidence_source": sources,
                    "evidence_text": texts,
                }
            )


def main() -> int:
    code_dir = Path(__file__).resolve().parent
    project_dir = code_dir.parent
    matrix_path = project_dir / "data" / "raw" / "GSE10846_series_matrix.txt.gz"
    metadata_dir = project_dir / "data" / "metadata"

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Missing {matrix_path}\n"
            "Run v41_0_download_prepare.py first."
        )

    out_csv = metadata_dir / "GSE10846_COO_resolved.csv"
    audit_json = metadata_dir / "v41_0b_coo_resolution_audit.json"

    print("=== Soft Spaces Phase 4 v41.0b — Resolve GSE10846 COO ===")
    print(f"Input: {matrix_path}")
    print()

    rows, sample_ids = parse_header_rows(matrix_path)
    records, bad_rows, aligned_rows = reconstruct_samples(rows, sample_ids)

    status_counts = Counter(r["status"] for r in records)
    label_counts = Counter(
        r["coo_label"] for r in records if r["coo_label"]
    )
    candidate_fields = candidate_field_summary(aligned_rows, sample_ids)

    write_csv(records, out_csv)

    published_match = all(
        label_counts.get(label, 0) == expected
        for label, expected in EXPECTED_PUBLISHED.items()
    )
    resolved_three_class_total = sum(
        label_counts.get(label, 0)
        for label in EXPECTED_PUBLISHED
    )

    audit = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "accession": ACCESSION,
        "series_matrix_samples": len(sample_ids),
        "status_counts": dict(status_counts),
        "resolved_label_counts": dict(label_counts),
        "published_reference_counts": EXPECTED_PUBLISHED,
        "published_reference_total": EXPECTED_PUBLISHED_TOTAL,
        "resolved_three_class_total": resolved_three_class_total,
        "published_count_match": published_match,
        "bad_cardinality_rows": bad_rows,
        "candidate_metadata_fields": candidate_fields,
        "output_csv": str(out_csv.relative_to(project_dir)),
        "scope": (
            "Metadata label resolution only. No expression preprocessing, "
            "model fitting, REAL/NULL construction, Aer, or QPU execution."
        ),
    }

    metadata_dir.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== COO RESOLUTION SUMMARY ===")
    print(f"Series Matrix samples: {len(sample_ids)}")
    print(f"Status counts: {dict(status_counts)}")
    print(f"Resolved COO counts: {dict(label_counts)}")
    print()
    print("Published GSE10846 benchmark reference:")
    for label, expected in EXPECTED_PUBLISHED.items():
        print(
            f"  {label:13s} expected={expected:3d} "
            f"resolved={label_counts.get(label, 0):3d}"
        )
    print(f"  total         expected={EXPECTED_PUBLISHED_TOTAL:3d} "
          f"resolved={resolved_three_class_total:3d}")
    print()

    if published_match:
        print("COUNT CROSS-CHECK: PASS")
        print(
            "The locally resolved ABC/GCB/Unclassified counts match the "
            "published 414-case GSE10846 COO benchmark exactly."
        )
    else:
        print("COUNT CROSS-CHECK: NOT YET PASS")
        print(
            "Do NOT force labels. Inspect the audit JSON candidate fields "
            "and unresolved samples before v41.1."
        )

    print()
    if candidate_fields:
        print("Candidate metadata fields containing COO-like text:")
        for field in candidate_fields:
            print(
                f"  {field['source']}: "
                f"{field['matched_samples']} matched samples; "
                f"{field['unique_matched_values']} unique values"
            )
            for ex in field["examples"][:5]:
                print(f"      {ex['count']:3d} x {ex['value']}")
    else:
        print("No COO-like metadata fields were found by the audit scan.")

    print()
    print(f"Resolved table: {out_csv}")
    print(f"Audit JSON:     {audit_json}")
    print()
    print("v41.0b COMPLETE.")
    print("No preprocessing, modelling, Aer simulation, or QPU execution performed.")

    return 0 if published_match else 2


if __name__ == "__main__":
    raise SystemExit(main())
