#!/usr/bin/env python3
"""
Soft Spaces / DLBCL Phase 4
v41.15a2 — Corrected GSE117556 preparation + frozen-panel mapping audit

Correction relative to v41.15a
------------------------------
The GSE117556 series matrix URL was valid.
The attempted GPL14951 /annot/GPL14951.annot.gz URL does not exist.

This corrected version uses GPL18281, GEO's official "gene symbol version" of
the same Illumina HumanHT-12 WG-DASL V4.0 R2 expression beadchip, solely as
the probe-ID -> gene-symbol annotation source.

No expression geometry is calculated.
No disease/outcome labels are read.
No frozen panel member is changed.

The preregistration SHA256 is verified before proceeding.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import shutil
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


VERSION = "v41.15a2"

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
EXPRESSION_PLATFORM = "GPL14951"
ANNOTATION_PLATFORM = "GPL18281"

SERIES_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE117nnn/"
    "GSE117556/matrix/GSE117556_series_matrix.txt.gz"
)

# GEO text endpoint: full platform table, not the 2.3 GB family SOFT archive.
GPL_TEXT_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    "?acc=GPL18281&targ=self&form=text&view=full"
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
        project / "docs" / "preregistration" / "v41_15_PREREGISTRATION.txt",
        project / "docs" / "preregistration" / "v41_15" / "v41_15_PREREGISTRATION.txt",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError("v41_15_PREREGISTRATION.txt not found.")


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
        headers={"User-Agent": "Mozilla/5.0 SoftSpaces-v41.15a2"},
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
            "User-Agent": "Mozilla/5.0 SoftSpaces-v41.15a2",
            "Accept": "text/plain,text/*;q=0.9,*/*;q=0.1",
        },
    )

    with urllib.request.urlopen(req, timeout=180) as r:
        payload = r.read()

    # Guard against HTML error/challenge pages.
    head = payload[:1000].lower()

    if b"<html" in head and b"!platform_table_begin" not in payload.lower():
        raise RuntimeError(
            "GEO returned HTML instead of the platform text table. "
            "Open GPL18281 in a browser once and rerun, or download the "
            "full text platform table manually."
        )

    tmp.write_bytes(payload)
    tmp.replace(path)

    print("[download] done:", f"{path.stat().st_size:,}", "bytes")


def parse_series_probe_ids(path: Path):
    """
    Read probe identifiers only. Numeric expression values are deliberately
    not parsed or analyzed.
    """
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
        raise RuntimeError("No probe IDs found in GSE117556 series matrix.")

    return ids


def split_symbols(value: str):
    value = str(value or "").strip()

    if not value:
        return []

    parts = re.split(
        r"\s*///\s*|\s*//\s*|[;,|]",
        value,
    )

    out = []

    for x in parts:
        x = x.strip()

        if not x or x.upper() in {"---", "NA", "N/A", "NULL"}:
            continue

        out.append(x)

    return out


def parse_geo_platform_text(path: Path):
    """
    Parse !platform_table_begin ... !platform_table_end from GEO text output.

    Returns probe -> set(gene symbols).
    """
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
        raise RuntimeError(
            "Could not find GEO platform table markers in GPL18281 text."
        )

    table = lines[begin + 1:end]

    if len(table) < 2:
        raise RuntimeError("GPL18281 platform table is empty.")

    header = next(
        (line for line in table if line.strip()),
        None,
    )

    if header is None:
        raise RuntimeError("GPL18281 platform header is missing.")

    fields = header.rstrip("\r\n").split("\t")
    lut = {x.strip().lower(): x for x in fields}

    id_candidates = [
        "id",
        "id_ref",
        "probe_id",
        "probe id",
        "ilmnid",
        "array_address_id",
    ]

    symbol_candidates = [
        "gene symbol",
        "gene_symbol",
        "symbol",
        "gene symbols",
    ]

    id_col = next(
        (lut[x] for x in id_candidates if x in lut),
        None,
    )

    symbol_col = next(
        (lut[x] for x in symbol_candidates if x in lut),
        None,
    )

    # GPL18281 is specifically a "gene symbol version". In some GEO custom
    # platforms the probe identifier itself can be the gene symbol while the
    # original Illumina ID appears in another annotation column.
    original_probe_candidates = [
        "illumina_id",
        "illumina id",
        "ilmnid",
        "probe_id",
        "probe id",
        "id_ref",
    ]

    original_probe_col = next(
        (lut[x] for x in original_probe_candidates if x in lut),
        None,
    )

    reader = csv.DictReader(
        table[1:],
        fieldnames=fields,
        delimiter="\t",
    )

    rows = list(reader)

    if not rows:
        raise RuntimeError("No rows parsed from GPL18281.")

    # Strategy A:
    # conventional platform table: ID = Illumina probe, Gene Symbol column.
    probe_to_symbols = defaultdict(set)

    if id_col and symbol_col:
        for row in rows:
            probe = str(row.get(id_col, "")).strip()

            if not probe:
                continue

            for sym in split_symbols(row.get(symbol_col, "")):
                probe_to_symbols[probe].add(sym)

        if probe_to_symbols:
            return dict(probe_to_symbols), {
                "strategy": "ID + gene-symbol column",
                "probe_id_column": id_col,
                "gene_symbol_column": symbol_col,
                "columns": fields,
            }

    # Strategy B:
    # gene-symbol-version platform where ID itself is gene symbol and a
    # separate column carries the original ILMN probe.
    if id_col and original_probe_col and id_col != original_probe_col:
        for row in rows:
            sym = str(row.get(id_col, "")).strip()
            probe = str(row.get(original_probe_col, "")).strip()

            if probe and sym:
                probe_to_symbols[probe].add(sym)

        if probe_to_symbols:
            return dict(probe_to_symbols), {
                "strategy": "gene-symbol ID + original probe column",
                "probe_id_column": original_probe_col,
                "gene_symbol_column": id_col,
                "columns": fields,
            }

    # Diagnostic failure with columns exposed.
    raise RuntimeError(
        "Could not deterministically identify ILMN probe and gene-symbol "
        "columns in GPL18281. Columns were:\n" + "\n".join(fields)
    )


def build_symbol_to_measured_probes(
    probe_to_symbols,
    measured_probe_ids,
):
    out = defaultdict(set)

    for probe, symbols in probe_to_symbols.items():
        if probe not in measured_probe_ids:
            continue

        for sym in symbols:
            out[str(sym).upper()].add(probe)

    return dict(out)


def resolve_coordinate(coord: str, symbol_to_probes):
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
            "note": "No frozen component measurable.",
        }

    resolved = [x for x, _ in matches]
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
        note = (
            "Exactly one frozen compound component is measurable."
        )

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

    print(f"=== {VERSION} GSE117556 MAPPING AUDIT ===")
    print("Expected prereg SHA:", EXPECTED_PREREG_SHA256)
    print("Actual prereg SHA:  ", actual_sha)

    if actual_sha != EXPECTED_PREREG_SHA256:
        raise SystemExit(
            "FAIL: preregistration SHA mismatch. Stop."
        )

    print("PASS: preregistration integrity verified.\n")

    root = project / "data" / "external" / GSE
    raw = root / "raw"
    platform = root / "platform"
    results = project / "results" / "stability"

    raw.mkdir(parents=True, exist_ok=True)
    platform.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    series_path = raw / "GSE117556_series_matrix.txt.gz"
    gpl_path = platform / "GPL18281_gene_symbol_version.txt"

    # Reuses the series matrix already downloaded by v41.15a.
    download_binary(SERIES_URL, series_path)

    # Corrected annotation source.
    download_text(GPL_TEXT_URL, gpl_path)

    print("\n[mapping] reading measured GSE117556 probe IDs only...")
    measured = parse_series_probe_ids(series_path)
    print("[mapping] measured probes:", f"{len(measured):,}")

    print("\n[mapping] parsing GPL18281 gene-symbol platform...")
    probe_to_symbols, mapping_info = parse_geo_platform_text(gpl_path)
    print("[mapping] mapping strategy:", mapping_info["strategy"])
    print("[mapping] annotated probe mappings:", f"{len(probe_to_symbols):,}")

    symbol_to_probes = build_symbol_to_measured_probes(
        probe_to_symbols,
        measured,
    )
    print(
        "[mapping] measurable mapped symbols:",
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

    audit_path = results / "v41_15_mapping_audit.csv"

    with audit_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "frozen_rank",
                "frozen_coordinate",
                "status",
                "resolved_symbols",
                "n_resolved_symbols",
                "probe_ids",
                "n_probes",
                "confirmatory_usable",
                "note",
            ],
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

    for k in (
        "EXACT",
        "RESOLVED-COMPOUND",
        "AMBIGUOUS",
        "MISSING",
    ):
        print(f"{k:<20} {counts[k]}")

    print("\nELIGIBILITY GATE")
    print("----------------")
    print(gate)

    if gate == "PASS":
        print(
            "All 16 frozen coordinates are reproducibly "
            "representable. Confirmatory v41.15 may proceed."
        )
    else:
        print(
            "At least one coordinate is missing or ambiguous. "
            "Do not modify/reduce the confirmatory panel."
        )

    manifest = {
        "version": VERSION,
        "gse": GSE,
        "expression_platform": EXPRESSION_PLATFORM,
        "annotation_source_platform": ANNOTATION_PLATFORM,
        "annotation_source_note":
            "GPL18281 is GEO's gene-symbol version of the same "
            "Illumina HumanHT-12 WG-DASL V4.0 R2 platform; used "
            "only for deterministic probe-to-symbol mapping.",
        "preregistration_sha256": actual_sha,
        "series_matrix": str(series_path),
        "series_matrix_sha256": sha256_file(series_path),
        "annotation_file": str(gpl_path),
        "annotation_file_sha256": sha256_file(gpl_path),
        "mapping_info": mapping_info,
        "n_measured_probes": len(measured),
        "n_mapped_symbols": len(symbol_to_probes),
        "status_counts": dict(counts),
        "eligibility_gate": gate,
        "expression_values_analyzed": False,
        "external_labels_used": False,
        "mapping_audit": str(audit_path),
        "mapping_audit_sha256": sha256_file(audit_path),
    }

    manifest_path = results / "v41_15a2_prepare_manifest.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nWrote:")
    print(" ", audit_path)
    print(" ", manifest_path)


if __name__ == "__main__":
    main()
