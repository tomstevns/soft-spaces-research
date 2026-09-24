#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v41.0 — DLBCL benchmark data acquisition / inspection.

Purpose
-------
Prepare the public GEO dataset GSE10846 for the first DLBCL application track.

This script:
  1. Downloads the processed GSE10846 Series Matrix from NCBI GEO.
  2. Downloads the GPL570 probe annotation file.
  3. Verifies files with SHA-256 hashes.
  4. Extracts sample-level metadata from the Series Matrix header.
  5. Performs a conservative text-based inspection for COO labels
     (ABC, GCB, Unclassified/UC) without altering the source data.
  6. Writes a reproducibility manifest.

It intentionally does NOT:
  * normalize or filter expression values,
  * map probes to genes,
  * train a model,
  * construct REAL/NULL,
  * use any QPU resources.

Those steps belong to v41.1+.

Expected directory layout
-------------------------
applications/dlbcl/
    code/v41_0_download_prepare.py
    data/raw/
    data/metadata/
    data/processed/
    results/
    paper/

Run
---
From applications/dlbcl/code:

    python -X utf8 -u .\\v41_0_download_prepare.py

Optional:
    python -X utf8 -u .\\v41_0_download_prepare.py --force
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


VERSION = "v41.0-dlbcl-download-prepare-1"
ACCESSION = "GSE10846"
PLATFORM = "GPL570"

SERIES_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE10nnn/"
    "GSE10846/matrix/GSE10846_series_matrix.txt.gz"
)
GPL570_ANNOT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/"
    "GPL570/annot/GPL570.annot.gz"
)

USER_AGENT = "SoftSpaces-DLBCL-v41.0/1.0"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(url: str, target: Path, force: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        print(f"[exists] {target.name}")
        return

    tmp = target.with_suffix(target.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    print(f"[download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as out:
        total = response.headers.get("Content-Length")
        total = int(total) if total else None
        downloaded = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)

            if total:
                pct = 100.0 * downloaded / total
                print(
                    f"\r  {downloaded / 1024**2:8.1f} MiB / "
                    f"{total / 1024**2:8.1f} MiB ({pct:5.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\r  {downloaded / 1024**2:8.1f} MiB",
                    end="",
                    flush=True,
                )

    print()
    tmp.replace(target)
    print(f"[saved] {target}")


def parse_series_matrix_metadata(path: Path):
    """
    Read only the metadata/header part of a GEO Series Matrix.
    Returns:
        metadata_rows: dict[str, list[str]]
        sample_ids: list[str]
    """
    rows: dict[str, list[str]] = {}

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line == "!series_matrix_table_begin":
                break
            if not line.startswith("!Sample_"):
                continue

            parts = next(csv.reader([line], delimiter="\t", quotechar='"'))
            key = parts[0]
            values = parts[1:]
            rows.setdefault(key, []).extend(values)

    sample_ids = rows.get("!Sample_geo_accession", [])
    return rows, sample_ids


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def infer_coo_label(text: str) -> str:
    """
    Conservative metadata text inspection only.
    It does not create scientific labels where metadata is ambiguous.
    """
    t = text.lower()

    # More specific phrases first.
    if re.search(r"\bactivated b[- ]?cell\b", t) or re.search(r"\babc\b", t):
        return "ABC"
    if re.search(r"\bgerminal cent(er|re) b[- ]?cell\b", t) or re.search(r"\bgcb\b", t):
        return "GCB"
    if re.search(r"\bunclassified\b", t) or re.search(r"\buc\b", t):
        return "Unclassified"

    return ""


def write_metadata_csv(rows: dict[str, list[str]], sample_ids: list[str], out_path: Path):
    sample_count = len(sample_ids)

    # Keep sample-wise fields whose cardinality matches sample count.
    compatible = {
        key: vals
        for key, vals in rows.items()
        if len(vals) == sample_count
    }

    # Preserve all characteristics by concatenating them per sample.
    characteristic_keys = sorted(
        k for k in compatible if k.startswith("!Sample_characteristics_ch1")
    )

    records = []
    for i, gsm in enumerate(sample_ids):
        title = compatible.get("!Sample_title", [""] * sample_count)[i]
        source = compatible.get("!Sample_source_name_ch1", [""] * sample_count)[i]

        characteristic_values = []
        for key in characteristic_keys:
            v = compatible[key][i]
            if v:
                characteristic_values.append(normalize_text(v))

        combined = " | ".join([title, source] + characteristic_values)
        coo = infer_coo_label(combined)

        records.append(
            {
                "sample_id": gsm,
                "title": normalize_text(title),
                "source_name": normalize_text(source),
                "characteristics": " | ".join(characteristic_values),
                "coo_label_text_inspection": coo,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "title",
                "source_name",
                "characteristics",
                "coo_label_text_inspection",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    counts = {}
    for r in records:
        label = r["coo_label_text_inspection"] or "UNRESOLVED"
        counts[label] = counts.get(label, 0) + 1

    return records, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even when they already exist.",
    )
    args = parser.parse_args()

    code_dir = Path(__file__).resolve().parent
    dlbcl_dir = code_dir.parent
    raw_dir = dlbcl_dir / "data" / "raw"
    metadata_dir = dlbcl_dir / "data" / "metadata"

    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = raw_dir / "GSE10846_series_matrix.txt.gz"
    annot_path = raw_dir / "GPL570.annot.gz"

    metadata_csv = metadata_dir / "GSE10846_sample_metadata_inspection.csv"
    manifest_path = metadata_dir / "v41_0_download_manifest.json"

    print("=== Soft Spaces Phase 4 v41.0 — DLBCL data preparation ===")
    print(f"Dataset: {ACCESSION}")
    print(f"Platform: {PLATFORM}")
    print(f"Project root: {dlbcl_dir}")
    print()

    download(SERIES_MATRIX_URL, matrix_path, force=args.force)
    download(GPL570_ANNOT_URL, annot_path, force=args.force)

    print()
    print("[inspect] Reading Series Matrix metadata...")
    rows, sample_ids = parse_series_matrix_metadata(matrix_path)

    if not sample_ids:
        raise RuntimeError(
            "No !Sample_geo_accession values found in Series Matrix header."
        )

    records, coo_counts = write_metadata_csv(rows, sample_ids, metadata_csv)

    matrix_hash = sha256_file(matrix_path)
    annot_hash = sha256_file(annot_path)

    manifest = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "accession": ACCESSION,
            "platform": PLATFORM,
            "series_matrix_url": SERIES_MATRIX_URL,
            "platform_annotation_url": GPL570_ANNOT_URL,
            "sample_count": len(sample_ids),
        },
        "files": {
            "series_matrix": {
                "path": str(matrix_path.relative_to(dlbcl_dir)),
                "sha256": matrix_hash,
                "size_bytes": matrix_path.stat().st_size,
            },
            "platform_annotation": {
                "path": str(annot_path.relative_to(dlbcl_dir)),
                "sha256": annot_hash,
                "size_bytes": annot_path.stat().st_size,
            },
            "sample_metadata_inspection": {
                "path": str(metadata_csv.relative_to(dlbcl_dir)),
                "rows": len(records),
            },
        },
        "coo_text_inspection_counts": coo_counts,
        "scope": (
            "Acquisition and metadata inspection only. "
            "No expression preprocessing, probe-to-gene mapping, "
            "REAL/NULL construction, model fitting, or QPU execution."
        ),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=== DATA PREPARATION SUMMARY ===")
    print(f"Samples found: {len(sample_ids)}")
    print(f"COO metadata text inspection: {coo_counts}")
    print(f"Series Matrix SHA-256: {matrix_hash}")
    print(f"GPL570 annotation SHA-256: {annot_hash}")
    print(f"Metadata CSV: {metadata_csv}")
    print(f"Manifest: {manifest_path}")
    print()
    print("v41.0 COMPLETE.")
    print("No preprocessing, modelling, Aer simulation, or QPU execution performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
