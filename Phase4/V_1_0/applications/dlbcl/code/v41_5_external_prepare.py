#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v41.5 — GSE31312 external-validation prepare.

Purpose
-------
Prepare an INDEPENDENT external DLBCL cohort before running the frozen
Soft Spaces mapping from v41.3/v41.4.

External cohort
---------------
GSE31312
Platform: GPL570
498 de-novo adult DLBCL cases

Published binary COO subset used in prior analyses:
    GCB = 227
    ABC = 199
    Total ABC/GCB = 426
The remaining cases are unclassified and/or otherwise outside the binary task.

Scientific rule
---------------
This script DOES NOT fit Soft Spaces, tune K, choose P dimension, or inspect
performance. It only:
  1. downloads the GSE31312 Series Matrix,
  2. parses sample metadata safely,
  3. attempts conservative COO label resolution,
  4. writes an auditable GSM -> COO table,
  5. cross-checks counts if labels are explicitly available.

No labels are forced from expected totals.

Frozen mapping to be tested later
---------------------------------
The external test must preserve the already chosen mapping:
    variance pool = 256
    P dimension   = 16
    feature K     = 16
    H             = <xx^T>_ABC - <xx^T>_GCB
    C_g           = ||P V_g Q||_F^2 = p_g(1-p_g)

IMPORTANT:
Those parameters are NOT re-optimized in this script.

Run
---
From:
    Phase4\\V_1_0\\applications\\dlbcl\\code

Command:
    python -X utf8 -u .\\v41_5_external_prepare.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import shutil
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


VERSION = "v41.5-gse31312-external-prepare-1"

ACCESSION = "GSE31312"
PLATFORM = "GPL570"

SERIES_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE31nnn/"
    "GSE31312/matrix/GSE31312_series_matrix.txt.gz"
)

SUPPLEMENT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE31nnn/"
    "GSE31312/suppl/"
    "GSE31312_Microarray_and_clinical_data_DLBCL_475_cases_PMID_22437443.pdf.gz"
)

EXPECTED = {
    "GCB": 227,
    "ABC": 199,
}
EXPECTED_BINARY_TOTAL = 426
EXPECTED_SERIES_TOTAL = 498


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(url: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_size > 0:
        print(f"      exists: {out.name} ({out.stat().st_size/1024/1024:.1f} MiB)")
        return

    print(f"      downloading: {url}")
    with urllib.request.urlopen(url, timeout=120) as response, out.open("wb") as f:
        shutil.copyfileobj(response, f)

    print(f"      saved: {out.name} ({out.stat().st_size/1024/1024:.1f} MiB)")


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip('"'))


def parse_sample_header_rows(matrix_path: Path):
    """
    Preserve all repeated !Sample_* rows separately.
    """
    rows = []
    occurrence = Counter()

    with gzip.open(matrix_path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if line == "!series_matrix_table_begin":
                break

            if not line.startswith("!Sample_"):
                continue

            parts = next(csv.reader([line], delimiter="\t", quotechar='"'))
            key = parts[0]
            values = [clean(v) for v in parts[1:]]

            occurrence[key] += 1
            rows.append({
                "key": key,
                "occurrence": occurrence[key],
                "values": values,
            })

    sample_rows = [r for r in rows if r["key"] == "!Sample_geo_accession"]
    if len(sample_rows) != 1:
        raise RuntimeError(
            f"Expected exactly one !Sample_geo_accession row, found {len(sample_rows)}"
        )

    sample_ids = sample_rows[0]["values"]
    return rows, sample_ids


def classify_coo(text: str):
    """
    Conservative explicit COO resolver.
    Returns (label, rule) or ("", "").
    """
    original = clean(text)
    t = original.lower()

    patterns = [
        ("ABC", [
            r"\bfinal microarray diagnosis\s*:\s*abc(?:\s+dlbcl)?\b",
            r"\bcell[ -]?of[ -]?origin\s*[:=]\s*abc\b",
            r"\bcoo\s*[:=]\s*abc\b",
            r"\bsubtype\s*[:=]\s*abc\b",
            r"\bactivated b[- ]?cell(?:[- ]like)?\b",
        ]),
        ("GCB", [
            r"\bfinal microarray diagnosis\s*:\s*gcb(?:\s+dlbcl)?\b",
            r"\bcell[ -]?of[ -]?origin\s*[:=]\s*gcb\b",
            r"\bcoo\s*[:=]\s*gcb\b",
            r"\bsubtype\s*[:=]\s*gcb\b",
            r"\bgerminal cent(?:er|re) b[- ]?cell(?:[- ]like)?\b",
        ]),
        ("Unclassified", [
            r"\bfinal microarray diagnosis\s*:\s*(?:unclassified|unclassifiable|uc|type\s*iii)\b",
            r"\bcell[ -]?of[ -]?origin\s*[:=]\s*(?:unclassified|unclassifiable|uc|type\s*iii)\b",
            r"\bcoo\s*[:=]\s*(?:unclassified|unclassifiable|uc|type\s*iii)\b",
            r"\bsubtype\s*[:=]\s*(?:unclassified|unclassifiable|uc|type\s*iii)\b",
            r"\bunclass(?:ified|ifiable)\b",
            r"\btype\s*iii\b",
        ]),
    ]

    for label, pats in patterns:
        for p in pats:
            if re.search(p, t, flags=re.I):
                return label, p

    # Strict isolated-value forms.
    stripped = re.sub(r"^[^:]{1,80}:\s*", "", t).strip()
    if stripped in {"abc", "abc dlbcl", "abc-like", "abc like"}:
        return "ABC", "isolated-value"
    if stripped in {"gcb", "gcb dlbcl", "gcb-like", "gcb like"}:
        return "GCB", "isolated-value"
    if stripped in {
        "unclassified", "unclassifiable", "uc",
        "type iii", "type 3", "unclassified dlbcl"
    }:
        return "Unclassified", "isolated-value"

    return "", ""


def reconstruct_samples(rows, sample_ids):
    n = len(sample_ids)

    aligned = []
    bad_rows = []

    for r in rows:
        if len(r["values"]) == n:
            aligned.append(r)
        else:
            bad_rows.append({
                "key": r["key"],
                "occurrence": r["occurrence"],
                "value_count": len(r["values"]),
            })

    records = []

    for i, gsm in enumerate(sample_ids):
        title = ""
        evidence = []
        metadata = []

        for r in aligned:
            v = r["values"][i]
            if not v:
                continue

            source = f'{r["key"]}#{r["occurrence"]}'
            metadata.append(f"{source}: {v}")

            if r["key"] == "!Sample_title":
                title = v

            label, rule = classify_coo(v)
            if label:
                evidence.append({
                    "label": label,
                    "source": source,
                    "text": v,
                    "rule": rule,
                })

        labels = sorted({e["label"] for e in evidence})

        if len(labels) == 1:
            status = "RESOLVED"
            label = labels[0]
        elif len(labels) == 0:
            status = "UNRESOLVED"
            label = ""
        else:
            status = "CONFLICT"
            label = ""

        records.append({
            "sample_id": gsm,
            "title": title,
            "coo_label": label,
            "status": status,
            "evidence": evidence,
            "metadata": metadata,
        })

    return records, aligned, bad_rows


def candidate_field_summary(aligned):
    hints = re.compile(
        r"\b(abc|gcb|germinal|activated|origin|subtype|classification|"
        r"unclass|type iii|cell of origin)\b",
        flags=re.I,
    )

    out = []

    for r in aligned:
        matches = [v for v in r["values"] if v and hints.search(v)]
        if not matches:
            continue

        counts = Counter(matches)

        out.append({
            "source": f'{r["key"]}#{r["occurrence"]}',
            "matched_samples": len(matches),
            "unique_matched_values": len(counts),
            "examples": [
                {"value": value, "count": count}
                for value, count in counts.most_common(20)
            ],
        })

    return out


def write_label_csv(records, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
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
        w.writeheader()

        for r in records:
            w.writerow({
                "sample_id": r["sample_id"],
                "title": r["title"],
                "coo_label": r["coo_label"],
                "status": r["status"],
                "evidence_source": " | ".join(
                    e["source"] for e in r["evidence"]
                ),
                "evidence_text": " | ".join(
                    e["text"] for e in r["evidence"]
                ),
            })


def main() -> int:
    code_dir = Path(__file__).resolve().parent
    project_dir = code_dir.parent

    ext_root = project_dir / "data" / "external" / ACCESSION
    raw_dir = ext_root / "raw"
    metadata_dir = ext_root / "metadata"

    matrix_path = raw_dir / "GSE31312_series_matrix.txt.gz"
    supplement_path = (
        raw_dir /
        "GSE31312_Microarray_and_clinical_data_DLBCL_475_cases_PMID_22437443.pdf.gz"
    )

    labels_csv = metadata_dir / "GSE31312_COO_inspection.csv"
    audit_json = metadata_dir / "v41_5_external_prepare_audit.json"

    print("=== Soft Spaces Phase 4 v41.5 — GSE31312 external prepare ===")
    print()
    print("Frozen external-validation target:")
    print("  variance pool = 256")
    print("  P dimension   = 16")
    print("  feature K     = 16")
    print("  NO parameter tuning in this stage")
    print()

    print("[1/4] Downloading independent GSE31312 data...")
    download(SERIES_URL, matrix_path)

    # Supplement is useful for later audit/clinical cross-reference.
    try:
        download(SUPPLEMENT_URL, supplement_path)
        supplement_ok = True
    except Exception as e:
        supplement_ok = False
        print(f"      supplement download warning: {e}")

    print()
    print("[2/4] Parsing GEO sample metadata...")
    rows, sample_ids = parse_sample_header_rows(matrix_path)

    if len(sample_ids) != EXPECTED_SERIES_TOTAL:
        print(
            f"      WARNING: expected {EXPECTED_SERIES_TOTAL} samples, "
            f"found {len(sample_ids)}"
        )
    else:
        print(f"      samples found: {len(sample_ids)}")

    records, aligned, bad_rows = reconstruct_samples(rows, sample_ids)

    print("[3/4] Resolving explicit COO labels...")
    status_counts = Counter(r["status"] for r in records)
    label_counts = Counter(
        r["coo_label"] for r in records if r["coo_label"]
    )

    print(f"      status: {dict(status_counts)}")
    print(f"      COO:    {dict(label_counts)}")

    candidate_fields = candidate_field_summary(aligned)

    abc_gcb_match = (
        label_counts.get("ABC", 0) == EXPECTED["ABC"]
        and label_counts.get("GCB", 0) == EXPECTED["GCB"]
    )

    binary_total = (
        label_counts.get("ABC", 0) +
        label_counts.get("GCB", 0)
    )

    write_label_csv(records, labels_csv)

    print("[4/4] Writing audit manifest...")

    audit = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "external_validation_status": "PREPARE_ONLY",
        "dataset": {
            "accession": ACCESSION,
            "platform": PLATFORM,
            "series_samples_found": len(sample_ids),
            "expected_series_samples": EXPECTED_SERIES_TOTAL,
        },
        "frozen_softspaces_mapping_for_later_test": {
            "variance_pool": 256,
            "p_dim": 16,
            "feature_budget_k": 16,
            "operator": "H = mean(xx^T|ABC) - mean(xx^T|GCB)",
            "coupling_score": "C_g = ||P V_g Q||_F^2 = p_g(1-p_g)",
            "parameter_tuning_allowed": False,
        },
        "published_binary_reference_counts": {
            "GCB": EXPECTED["GCB"],
            "ABC": EXPECTED["ABC"],
            "ABC_GCB_total": EXPECTED_BINARY_TOTAL,
        },
        "resolution": {
            "status_counts": dict(status_counts),
            "resolved_label_counts": dict(label_counts),
            "resolved_ABC_GCB_total": binary_total,
            "exact_ABC_GCB_count_match": abc_gcb_match,
        },
        "candidate_metadata_fields": candidate_fields,
        "bad_cardinality_rows": bad_rows,
        "inputs": {
            "series_matrix": {
                "url": SERIES_URL,
                "path": str(matrix_path.relative_to(project_dir)),
                "sha256": sha256_file(matrix_path),
            },
            "supplement": {
                "url": SUPPLEMENT_URL,
                "downloaded": supplement_ok,
                "path": (
                    str(supplement_path.relative_to(project_dir))
                    if supplement_ok else None
                ),
                "sha256": (
                    sha256_file(supplement_path)
                    if supplement_ok else None
                ),
            },
        },
        "outputs": {
            "coo_inspection_csv": str(labels_csv.relative_to(project_dir)),
        },
        "scientific_rule": (
            "No labels forced from published counts. "
            "No Soft Spaces fitting or performance evaluation performed."
        ),
    }

    metadata_dir.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=== v41.5 SUMMARY ===")
    print(f"Series samples: {len(sample_ids)}")
    print(f"Resolved ABC:   {label_counts.get('ABC', 0)}")
    print(f"Resolved GCB:   {label_counts.get('GCB', 0)}")
    print(f"Resolved UC:    {label_counts.get('Unclassified', 0)}")
    print(f"Unresolved:     {status_counts.get('UNRESOLVED', 0)}")
    print()

    if abc_gcb_match:
        print("ABC/GCB EXTERNAL COUNT CROSS-CHECK: PASS")
        print(
            f"Resolved binary total = {binary_total} "
            f"(expected {EXPECTED_BINARY_TOTAL})"
        )
    else:
        print("ABC/GCB EXTERNAL COUNT CROSS-CHECK: NOT YET PASS")
        print("Do NOT force labels.")
        print("Inspect candidate metadata fields / supplementary clinical data next.")

    if candidate_fields:
        print()
        print("Candidate COO metadata fields:")
        for field in candidate_fields[:10]:
            print(
                f"  {field['source']}: "
                f"{field['matched_samples']} matches, "
                f"{field['unique_matched_values']} unique values"
            )
            for ex in field["examples"][:5]:
                print(f"      {ex['count']:3d} x {ex['value']}")

    print()
    print(f"COO inspection: {labels_csv}")
    print(f"Audit JSON:     {audit_json}")
    print()
    print("v41.5 COMPLETE.")
    print("No Soft Spaces fit, no Aer, and no QPU execution performed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
