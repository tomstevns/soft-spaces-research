#!/usr/bin/env python3
"""
Soft Spaces / DLBCL Phase 4
v41.16a — Prepare GSE159472 + frozen 12-gene mapping audit

Purpose
-------
First execution step after the v41.16 preregistration integrity lock.

This script:
1. verifies the frozen preregistration SHA256;
2. downloads GSE159472 series matrix if needed;
3. downloads the GPL570 full GEO platform table;
4. maps GPL570 probe IDs deterministically to gene symbols;
5. audits the frozen 12-gene panel;
6. writes an eligibility manifest.

It does NOT:
- calculate the confirmatory top-4 result;
- use ABC/GCB or any clinical/outcome labels;
- alter the frozen 12-gene panel.

Eligibility rule
----------------
All 12 frozen genes must be reproducibly measurable.
Otherwise v41.16 is TECHNICALLY_NON_EVALUABLE.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path


VERSION = "v41.16a"

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

GSE = "GSE159472"
GPL = "GPL570"

SERIES_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE159nnn/"
    "GSE159472/matrix/GSE159472_series_matrix.txt.gz"
)

GPL_TEXT_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    "?acc=GPL570&targ=self&form=text&view=full"
)


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
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("v41_16_PREREGISTRATION.txt not found")


def download_binary(url: str, path: Path):
    if path.exists() and path.stat().st_size > 0:
        print("[download] reuse:", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".part")

    print("[download]", url)
    print("        ->", path)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 SoftSpaces-v41.16a"},
    )

    with urllib.request.urlopen(req, timeout=180) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)

    tmp.replace(path)
    print("[download] done:", f"{path.stat().st_size:,}", "bytes")


def download_text(url: str, path: Path):
    if path.exists() and path.stat().st_size > 0:
        print("[download] reuse:", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".part")

    print("[download]", url)
    print("        ->", path)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 SoftSpaces-v41.16a",
            "Accept": "text/plain,text/*;q=0.9,*/*;q=0.1",
        },
    )

    with urllib.request.urlopen(req, timeout=180) as r:
        payload = r.read()

    if b"!platform_table_begin" not in payload:
        raise RuntimeError(
            "GEO response did not contain a GPL570 platform table."
        )

    tmp.write_bytes(payload)
    tmp.replace(path)

    print("[download] done:", f"{path.stat().st_size:,}", "bytes")


def parse_series_probe_ids(path: Path):
    ids = set()
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

            if (
                line.startswith('"ID_REF"')
                or line.startswith("ID_REF")
                or not line.strip()
            ):
                continue

            probe = line.split("\t", 1)[0].strip().strip('"')
            if probe:
                ids.add(probe)

    if not ids:
        raise RuntimeError("No probe IDs found in GSE159472 series matrix.")

    return ids


def split_symbols(value: str):
    value = str(value or "").strip()

    if not value:
        return []

    parts = re.split(r"\s*///\s*|\s*//\s*|[;,|]", value)

    out = []
    for x in parts:
        x = x.strip()
        if not x:
            continue
        if x.upper() in {"---", "NA", "N/A", "NULL"}:
            continue
        out.append(x)

    return out


def parse_gpl570_geo_text(path: Path):
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
        raise RuntimeError("GPL570 platform-table markers not found.")

    table = lines[begin + 1:end]
    if len(table) < 2:
        raise RuntimeError("GPL570 platform table is empty.")

    header = table[0].split("\t")
    lut = {x.strip().lower(): x for x in header}

    id_col = lut.get("id")

    symbol_candidates = [
        "gene symbol",
        "gene_symbol",
        "symbol",
    ]
    symbol_col = next(
        (lut[x] for x in symbol_candidates if x in lut),
        None,
    )

    if id_col is None or symbol_col is None:
        raise RuntimeError(
            "Could not identify GPL570 ID / Gene Symbol columns. "
            f"Columns were: {header}"
        )

    probe_to_symbols = defaultdict(set)

    reader = csv.DictReader(
        table[1:],
        fieldnames=header,
        delimiter="\t",
    )

    for row in reader:
        probe = str(row.get(id_col, "")).strip()
        if not probe:
            continue

        for sym in split_symbols(row.get(symbol_col, "")):
            probe_to_symbols[probe].add(sym)

    if not probe_to_symbols:
        raise RuntimeError("No GPL570 probe->gene mappings parsed.")

    return dict(probe_to_symbols), {
        "probe_id_column": id_col,
        "gene_symbol_column": symbol_col,
        "all_columns": header,
    }


def build_symbol_to_measured_probes(probe_to_symbols, measured):
    out = defaultdict(set)

    for probe, symbols in probe_to_symbols.items():
        if probe not in measured:
            continue

        for sym in symbols:
            out[sym.upper()].add(probe)

    return dict(out)


def main():
    project = project_dir_from_script()
    prereg = find_preregistration(project)
    actual_sha = sha256_file(prereg)

    print(f"=== {VERSION} GSE159472 / GPL570 MAPPING AUDIT ===")
    print("Expected prereg SHA:", EXPECTED_PREREG_SHA256)
    print("Actual prereg SHA:  ", actual_sha)

    if actual_sha != EXPECTED_PREREG_SHA256:
        raise SystemExit(
            "FAIL: preregistration SHA mismatch. Stop before data analysis."
        )

    print("PASS: preregistration integrity verified.\n")

    root = project / "data" / "external" / GSE
    raw_dir = root / "raw"
    platform_dir = root / "platform"
    results_dir = project / "results" / "stability"

    for d in (raw_dir, platform_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    series_path = raw_dir / "GSE159472_series_matrix.txt.gz"
    gpl_path = platform_dir / "GPL570_full_geo_table.txt"

    download_binary(SERIES_URL, series_path)
    download_text(GPL_TEXT_URL, gpl_path)

    print("\n[mapping] reading measured probe IDs...")
    measured = parse_series_probe_ids(series_path)
    print("[mapping] measured probes:", f"{len(measured):,}")

    print("\n[mapping] parsing GPL570...")
    probe_to_symbols, mapping_info = parse_gpl570_geo_text(gpl_path)
    print("[mapping] annotated probes:", f"{len(probe_to_symbols):,}")
    print("[mapping] ID column:", mapping_info["probe_id_column"])
    print("[mapping] Symbol column:", mapping_info["gene_symbol_column"])

    overlap = set(probe_to_symbols).intersection(measured)
    print("[mapping] measured probes found in GPL570:", f"{len(overlap):,}")

    if len(overlap) < 10000:
        raise RuntimeError(
            "Unexpectedly low overlap between GSE159472 probes and GPL570."
        )

    symbol_to_probes = build_symbol_to_measured_probes(
        probe_to_symbols,
        measured,
    )

    print("[mapping] measurable gene symbols:", f"{len(symbol_to_probes):,}")

    rows = []

    for rank, gene in enumerate(FROZEN_PANEL, 1):
        probes = sorted(symbol_to_probes.get(gene.upper(), set()))

        if probes:
            status = "EXACT"
            usable = True
            note = "Frozen gene directly measurable on GPL570."
        else:
            status = "MISSING"
            usable = False
            note = "Frozen gene not measurable in GSE159472/GPL570."

        rows.append(
            {
                "frozen_rank": rank,
                "frozen_gene": gene,
                "status": status,
                "probe_ids": "|".join(probes),
                "n_probes": len(probes),
                "confirmatory_usable": usable,
                "note": note,
            }
        )

    audit_path = results_dir / "v41_16_mapping_audit.csv"

    with audit_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "frozen_rank",
            "frozen_gene",
            "status",
            "probe_ids",
            "n_probes",
            "confirmatory_usable",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nFROZEN 12 MAPPING AUDIT")
    print("-----------------------")

    for row in rows:
        print(
            f"{row['frozen_rank']:2d}. "
            f"{row['frozen_gene']:<12} "
            f"{row['status']:<8} "
            f"probes={row['n_probes']}"
        )

    usable = all(row["confirmatory_usable"] for row in rows)

    gate = (
        "PASS"
        if usable
        else "TECHNICALLY_NON_EVALUABLE"
    )

    print("\nELIGIBILITY GATE")
    print("----------------")
    print(gate)

    if usable:
        print(
            "All 12 frozen genes are reproducibly measurable. "
            "Confirmatory v41.16 may proceed."
        )
    else:
        print(
            "At least one frozen gene is missing. "
            "Do NOT alter or reduce the confirmatory panel."
        )

    manifest = {
        "version": VERSION,
        "gse": GSE,
        "platform": GPL,
        "preregistration_sha256": actual_sha,
        "series_matrix": str(series_path),
        "series_matrix_sha256": sha256_file(series_path),
        "platform_table": str(gpl_path),
        "platform_table_sha256": sha256_file(gpl_path),
        "mapping_info": mapping_info,
        "n_measured_probes": len(measured),
        "n_annotated_probes": len(probe_to_symbols),
        "n_measured_annotated_probe_overlap": len(overlap),
        "n_measurable_symbols": len(symbol_to_probes),
        "panel_size": len(FROZEN_PANEL),
        "eligibility_gate": gate,
        "expression_values_analyzed": False,
        "external_labels_used": False,
        "mapping_audit": str(audit_path),
        "mapping_audit_sha256": sha256_file(audit_path),
    }

    manifest_path = results_dir / "v41_16a_prepare_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nWrote:")
    print(" ", audit_path)
    print(" ", manifest_path)


if __name__ == "__main__":
    main()
