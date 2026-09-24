#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v41.0c — Resolve GSE10846 COO labels from GEO metadata.

This fixes the v41.0b parser by explicitly recognizing the GEO metadata form:
    "Clinical info: Final microarray diagnosis: GCB DLBCL"
    "Clinical info: Final microarray diagnosis: ABC DLBCL"

It also inspects the remaining unresolved samples for a published
Unclassified / Type III label without forcing any assignment.

No expression preprocessing, modelling, Aer, or QPU execution is performed.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VERSION = "v41.0c-dlbcl-resolve-coo-2"
EXPECTED = {"ABC": 167, "GCB": 183, "Unclassified": 64}

def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip('"'))

def parse_header_rows(matrix_path: Path):
    rows = []
    counts = Counter()
    with gzip.open(matrix_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line == "!series_matrix_table_begin":
                break
            if not line.startswith("!Sample_"):
                continue
            parts = next(csv.reader([line], delimiter="\t", quotechar='"'))
            key = parts[0]
            vals = [clean(x) for x in parts[1:]]
            counts[key] += 1
            rows.append({"key": key, "occ": counts[key], "values": vals})

    sample_row = next(r for r in rows if r["key"] == "!Sample_geo_accession")
    return rows, sample_row["values"]

def classify(value: str):
    t = clean(value).lower()

    # Exact GSE10846 GEO wording.
    m = re.search(r"final microarray diagnosis\s*:\s*(abc|gcb)\s+dlbcl\b", t)
    if m:
        return m.group(1).upper(), "final-microarray-diagnosis"

    # Conservative explicit unclassified / Type III forms.
    if re.search(r"final microarray diagnosis\s*:\s*(unclassified|type\s*iii|type\s*3)\b", t):
        return "Unclassified", "final-microarray-diagnosis"

    # Other explicit COO wording.
    if re.search(r"\bactivated b[- ]?cell(?:[- ]like)?\b", t):
        return "ABC", "explicit-abc"
    if re.search(r"\bgerminal cent(?:er|re) b[- ]?cell(?:[- ]like)?\b", t):
        return "GCB", "explicit-gcb"
    if re.search(r"\b(unclassified|unclassifiable|type\s*iii)\b", t):
        return "Unclassified", "explicit-unclassified"

    return "", ""

def main():
    code_dir = Path(__file__).resolve().parent
    project = code_dir.parent
    matrix = project / "data" / "raw" / "GSE10846_series_matrix.txt.gz"
    meta_dir = project / "data" / "metadata"
    out_csv = meta_dir / "GSE10846_COO_resolved_v41_0c.csv"
    audit_json = meta_dir / "v41_0c_coo_resolution_audit.json"

    if not matrix.exists():
        raise FileNotFoundError(f"Missing {matrix}")

    rows, sample_ids = parse_header_rows(matrix)
    n = len(sample_ids)
    aligned = [r for r in rows if len(r["values"]) == n]

    records = []
    for i, gsm in enumerate(sample_ids):
        evidence = []
        title = ""

        for r in aligned:
            val = r["values"][i]
            if not val:
                continue
            if r["key"] == "!Sample_title":
                title = val

            label, rule = classify(val)
            if label:
                evidence.append({
                    "label": label,
                    "source": f'{r["key"]}#{r["occ"]}',
                    "text": val,
                    "rule": rule,
                })

        labels = sorted(set(e["label"] for e in evidence))
        if len(labels) == 1:
            status, label = "RESOLVED", labels[0]
        elif len(labels) == 0:
            status, label = "UNRESOLVED", ""
        else:
            status, label = "CONFLICT", ""

        records.append({
            "sample_id": gsm,
            "title": title,
            "coo_label": label,
            "status": status,
            "evidence": evidence,
        })

    counts = Counter(r["coo_label"] for r in records if r["coo_label"])
    status_counts = Counter(r["status"] for r in records)

    # Collect unresolved metadata examples for safe follow-up inspection.
    unresolved_examples = []
    unresolved_ids = {r["sample_id"] for r in records if r["status"] == "UNRESOLVED"}
    gsm_to_i = {gsm: i for i, gsm in enumerate(sample_ids)}

    for gsm in list(unresolved_ids)[:20]:
        i = gsm_to_i[gsm]
        vals = []
        for r in aligned:
            v = r["values"][i]
            if v and r["key"].startswith("!Sample_characteristics_ch1"):
                vals.append(f'{r["key"]}#{r["occ"]}: {v}')
        unresolved_examples.append({"sample_id": gsm, "characteristics": vals})

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["sample_id","title","coo_label","status","evidence_source","evidence_text"]
        )
        w.writeheader()
        for r in records:
            w.writerow({
                "sample_id": r["sample_id"],
                "title": r["title"],
                "coo_label": r["coo_label"],
                "status": r["status"],
                "evidence_source": " | ".join(e["source"] for e in r["evidence"]),
                "evidence_text": " | ".join(e["text"] for e in r["evidence"]),
            })

    audit = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": n,
        "status_counts": dict(status_counts),
        "resolved_counts": dict(counts),
        "expected_published_counts": EXPECTED,
        "exact_abc_gcb_match": (
            counts.get("ABC",0) == EXPECTED["ABC"]
            and counts.get("GCB",0) == EXPECTED["GCB"]
        ),
        "full_three_class_match": all(counts.get(k,0) == v for k,v in EXPECTED.items()),
        "unresolved_count": status_counts.get("UNRESOLVED",0),
        "unresolved_examples_first_20": unresolved_examples,
        "note": (
            "No labels were forced. If 64 Unclassified are not explicitly present "
            "in GEO metadata, they must be sourced from a verified publication/supplement "
            "before inclusion as ground truth."
        )
    }
    audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Soft Spaces Phase 4 v41.0c — COO resolution ===")
    print(f"Samples: {n}")
    print(f"Status counts: {dict(status_counts)}")
    print(f"Resolved COO counts: {dict(counts)}")
    print()
    print(f"ABC expected/resolved: {EXPECTED['ABC']} / {counts.get('ABC',0)}")
    print(f"GCB expected/resolved: {EXPECTED['GCB']} / {counts.get('GCB',0)}")
    print(f"Unclassified expected/resolved: {EXPECTED['Unclassified']} / {counts.get('Unclassified',0)}")
    print()

    if audit["exact_abc_gcb_match"]:
        print("ABC/GCB CROSS-CHECK: PASS")
    else:
        print("ABC/GCB CROSS-CHECK: NOT YET PASS")

    if audit["full_three_class_match"]:
        print("FULL THREE-CLASS CROSS-CHECK: PASS")
    else:
        print("FULL THREE-CLASS CROSS-CHECK: NOT YET PASS")
        print("No labels were forced. Remaining samples require verified source resolution.")

    print()
    print(f"Resolved CSV: {out_csv}")
    print(f"Audit JSON:   {audit_json}")
    print("v41.0c COMPLETE.")

if __name__ == "__main__":
    main()
