#!/usr/bin/env python3
"""
Soft Spaces / DLBCL Phase 4
v41.15a3 — Correct GSE117556 GPL14951 mapping audit

This version fixes the mapping source.

GSE117556 expression uses GPL14951 probe IDs (ILMN_*).
Therefore the mapping audit now uses the FULL GEO text table for GPL14951 itself,
whose table contains:
    ID      = Illumina probe identifier
    Symbol  = gene symbol

No expression geometry is calculated.
No external disease/outcome labels are read.
No frozen panel gene is changed.
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

VERSION = "v41.15a3"

EXPECTED_PREREG_SHA256 = (
    "979f7f4880fa3936edcdcd1055641bf5922abcc00caa3917e0fed8c8e464b56b"
)

FROZEN_PANEL = [
    "DNER",
    "SPIC",
    "TCL1B",
    "NPIPA5///NPIPB6///NPIPB8///NPIPB3",
    "EML6",
    "MMP20",
    "KCCAT333",
    "IGLJ3///IGLV1-44///CKAP2///IGLV@///IGLC1",
    "RNF183",
    "MYOCD",
    "UMODL1",
    "CTAG2",
    "IGK///IGKC",
    "LOC283454///HRK",
    "RGS13",
    "TCL1A",
]

GSE = "GSE117556"
GPL = "GPL14951"

SERIES_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE117nnn/"
    "GSE117556/matrix/GSE117556_series_matrix.txt.gz"
)

GPL_TEXT_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    "?acc=GPL14951&targ=self&form=text&view=full"
)


def project_dir_from_script():
    return Path(__file__).resolve().parent.parent


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_preregistration(project):
    candidates = [
        project / "docs" / "preregistration" / "v41_15_PREREGISTRATION.txt",
        project / "docs" / "preregistration" / "v41_15" / "v41_15_PREREGISTRATION.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("v41_15_PREREGISTRATION.txt not found")


def download_binary(url, path):
    if path.exists() and path.stat().st_size > 0:
        print("[download] reuse:", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".part")

    print("[download]", url)
    print("        ->", path)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 SoftSpaces-v41.15a3"},
    )

    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)

    tmp.replace(path)
    print("[download] done:", f"{path.stat().st_size:,}", "bytes")


def download_text(url, path):
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
            "User-Agent": "Mozilla/5.0 SoftSpaces-v41.15a3",
            "Accept": "text/plain,text/*;q=0.9,*/*;q=0.1",
        },
    )

    with urllib.request.urlopen(req, timeout=180) as r:
        payload = r.read()

    if b"!platform_table_begin" not in payload:
        raise RuntimeError(
            "GEO response did not contain a platform table. "
            "Delete any partial file and retry."
        )

    tmp.write_bytes(payload)
    tmp.replace(path)

    print("[download] done:", f"{path.stat().st_size:,}", "bytes")


def parse_series_probe_ids(path):
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
        raise RuntimeError("No probe IDs found in GSE117556 series matrix")

    return ids


def split_symbols(value):
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


def parse_gpl14951_geo_text(path):
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
        raise RuntimeError("GPL14951 table markers not found")

    table = lines[begin + 1:end]

    if len(table) < 2:
        raise RuntimeError("GPL14951 table empty")

    header = table[0].split("\t")
    lut = {x.strip().lower(): x for x in header}

    id_col = lut.get("id")
    symbol_col = lut.get("symbol")

    if id_col is None or symbol_col is None:
        raise RuntimeError(
            "Expected GPL14951 columns ID and Symbol. "
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
        raise RuntimeError("No GPL14951 probe->symbol mappings parsed")

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


def resolve_coordinate(coord, symbol_to_probes):
    components = [
        x.strip()
        for x in coord.split("///")
        if x.strip()
    ]

    matches = []

    for comp in components:
        probes = sorted(
            symbol_to_probes.get(comp.upper(), set())
        )
        if probes:
            matches.append((comp, probes))

    if not matches:
        return {
            "status": "MISSING",
            "resolved_symbols": "",
            "n_resolved_symbols": 0,
            "probe_ids": "",
            "n_probes": 0,
            "confirmatory_usable": False,
            "note": "No frozen component measurable on GPL14951.",
        }

    resolved = [c for c, _ in matches]
    probes = sorted(
        {p for _, ps in matches for p in ps}
    )

    if len(components) == 1:
        status = "EXACT"
        usable = True
        note = "Frozen symbol directly measurable."

    elif len(matches) == 1:
        status = "RESOLVED-COMPOUND"
        usable = True
        note = "Exactly one frozen compound component measurable."

    else:
        status = "AMBIGUOUS"
        usable = False
        note = (
            "Multiple frozen compound components measurable; "
            "no post-hoc choice permitted."
        )

    return {
        "status": status,
        "resolved_symbols": "///".join(resolved),
        "n_resolved_symbols": len(resolved),
        "probe_ids": "|".join(probes),
        "n_probes": len(probes),
        "confirmatory_usable": usable,
        "note": note,
    }


def main():
    project = project_dir_from_script()

    prereg = find_preregistration(project)
    actual_sha = sha256_file(prereg)

    print(f"=== {VERSION} GPL14951 MAPPING AUDIT ===")
    print("Expected prereg SHA:", EXPECTED_PREREG_SHA256)
    print("Actual prereg SHA:  ", actual_sha)

    if actual_sha != EXPECTED_PREREG_SHA256:
        raise SystemExit("FAIL: preregistration SHA mismatch")

    print("PASS: preregistration integrity verified.\n")

    root = project / "data" / "external" / GSE
    raw_dir = root / "raw"
    platform_dir = root / "platform"
    results_dir = project / "results" / "stability"

    raw_dir.mkdir(parents=True, exist_ok=True)
    platform_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    series_path = raw_dir / "GSE117556_series_matrix.txt.gz"
    gpl_path = platform_dir / "GPL14951_full_geo_table.txt"

    download_binary(SERIES_URL, series_path)
    download_text(GPL_TEXT_URL, gpl_path)

    print("\n[mapping] reading measured probe IDs...")
    measured = parse_series_probe_ids(series_path)
    print("[mapping] measured probes:", f"{len(measured):,}")

    print("\n[mapping] parsing GPL14951...")
    probe_to_symbols, mapping_info = parse_gpl14951_geo_text(gpl_path)
    print("[mapping] annotated probes:", f"{len(probe_to_symbols):,}")
    print("[mapping] ID column:", mapping_info["probe_id_column"])
    print("[mapping] Symbol column:", mapping_info["gene_symbol_column"])

    overlap = set(probe_to_symbols).intersection(measured)
    print("[mapping] measured probes found in GPL14951:", f"{len(overlap):,}")

    if len(overlap) < 10000:
        raise RuntimeError(
            "Unexpectedly low overlap between GSE117556 probes "
            "and GPL14951 annotation."
        )

    symbol_to_probes = build_symbol_to_measured_probes(
        probe_to_symbols,
        measured,
    )

    print(
        "[mapping] measurable gene symbols:",
        f"{len(symbol_to_probes):,}",
    )

    rows = []

    for rank, coord in enumerate(FROZEN_PANEL, 1):
        res = resolve_coordinate(
            coord,
            symbol_to_probes,
        )

        rows.append(
            {
                "frozen_rank": rank,
                "frozen_coordinate": coord,
                **res,
            }
        )

    audit_path = results_dir / "v41_15_mapping_audit.csv"

    with audit_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "frozen_rank",
            "frozen_coordinate",
            "status",
            "resolved_symbols",
            "n_resolved_symbols",
            "probe_ids",
            "n_probes",
            "confirmatory_usable",
            "note",
        ]

        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        w.writeheader()
        w.writerows(rows)

    print("\nFROZEN 16 MAPPING AUDIT")
    print("-----------------------")

    counts = defaultdict(int)

    for row in rows:
        counts[row["status"]] += 1

        print(
            f"{row['frozen_rank']:2d}. "
            f"{row['frozen_coordinate']:<45} "
            f"{row['status']:<18} "
            f"probes={row['n_probes']}"
        )

    usable = all(
        bool(row["confirmatory_usable"])
        for row in rows
    )

    gate = (
        "PASS"
        if usable
        else "TECHNICALLY_NON_EVALUABLE"
    )

    print("\nSTATUS COUNTS")
    print("-------------")

    for key in (
        "EXACT",
        "RESOLVED-COMPOUND",
        "AMBIGUOUS",
        "MISSING",
    ):
        print(f"{key:<20}: {counts[key]}")

    print("\nELIGIBILITY GATE")
    print("----------------")
    print(gate)

    if usable:
        print(
            "All 16 frozen coordinates are reproducibly representable. "
            "The confirmatory v41.15 test may proceed."
        )
    else:
        print(
            "At least one frozen coordinate is missing or ambiguous. "
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
        "status_counts": dict(counts),
        "eligibility_gate": gate,
        "expression_values_analyzed": False,
        "external_labels_used": False,
        "mapping_audit": str(audit_path),
        "mapping_audit_sha256": sha256_file(audit_path),
    }

    manifest_path = (
        results_dir / "v41_15a3_prepare_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nWrote:")
    print(" ", audit_path)
    print(" ", manifest_path)


if __name__ == "__main__":
    main()
