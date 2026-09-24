#!/usr/bin/env python3
"""
Soft Spaces / CML
v44.1b — Full transport-aware gene-universe rebuild

Purpose
-------
Build the FULL development gene universe that is simultaneously:
1. represented in GSE130404/GPL10558,
2. resolvable on GPL570 by exact symbol or stable Entrez Gene ID,
3. present in ALL 9 GSE236233 Lin-CD34+ RNA feature tables.

This is entirely outcome-blind with respect to external cohorts.

External response outcomes are NOT used.

Inputs
------
- v44.0 preregistration
- GPL10558 full GEO platform table
- GPL570 full GEO platform table
- already downloaded/extracted GSE236233 processed files
- v43.4d technical manifest to identify the 9 primary patient samples

Outputs
-------
- full transport-aware eligible gene list
- mapping/audit table
- summary
- manifest

Decision
--------
If eligible universe >= 256:
    PASS TO v44.2
else:
    TECHNICALLY INSUFFICIENT
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


VERSION = "v44.1b"
MIN_GENES = 256


def project_dir():
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_header(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", x.strip().lower())


def split_multi(raw):
    s = str(raw or "").strip()
    for sep in ("///", "//", ";", ",", "|"):
        s = s.replace(sep, "|")
    return [
        x.strip()
        for x in s.split("|")
        if x.strip()
        and x.strip().upper() not in {"---", "NA", "N/A", "NULL", "NONE"}
    ]


def parse_entrez(raw):
    out = set()
    for token in split_multi(raw):
        out.update(re.findall(r"(?<!\d)(\d+)(?!\d)", token))
    out.discard("0")
    return out


def parse_platform(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    b = next(i for i, x in enumerate(lines) if x.startswith("!platform_table_begin"))
    e = next(i for i, x in enumerate(lines) if x.startswith("!platform_table_end"))

    table = lines[b + 1:e]
    header = table[0].split("\t")
    lut = {normalize_header(x): x for x in header}

    id_col = next((lut[x] for x in ("id", "idref", "probeid", "probesetid") if x in lut), None)
    sym_col = next((lut[x] for x in ("symbol", "genesymbol", "genesymbols", "officialgenesymbol") if x in lut), None)
    entrez_col = next((lut[x] for x in ("entrezgene", "entrezgeneid", "entrezid", "geneid", "ncbigeneid") if x in lut), None)

    if entrez_col is None:
        entrez_col = next((orig for n, orig in lut.items() if "entrez" in n), None)

    if id_col is None or sym_col is None:
        raise RuntimeError(f"Cannot identify probe/symbol columns in {path.name}: {header}")

    symbol_to_entrez = defaultdict(set)
    symbol_to_probes = defaultdict(set)
    entrez_to_symbols = defaultdict(set)
    entrez_to_probes = defaultdict(set)

    reader = csv.DictReader(table[1:], fieldnames=header, delimiter="\t")

    for row in reader:
        probe = str(row.get(id_col, "")).strip()
        if not probe:
            continue

        symbols = set(split_multi(row.get(sym_col, "")))
        entrez_ids = parse_entrez(row.get(entrez_col, "")) if entrez_col else set()

        for sym in symbols:
            symbol_to_probes[sym].add(probe)
            for eid in entrez_ids:
                symbol_to_entrez[sym].add(eid)
                entrez_to_symbols[eid].add(sym)

        for eid in entrez_ids:
            entrez_to_probes[eid].add(probe)

    return {
        "symbol_to_entrez": dict(symbol_to_entrez),
        "symbol_to_probes": dict(symbol_to_probes),
        "entrez_to_symbols": dict(entrez_to_symbols),
        "entrez_to_probes": dict(entrez_to_probes),
        "symbol_col": sym_col,
        "entrez_col": entrez_col,
    }


def read_feature_symbols(path: Path):
    name = path.name.lower()
    compression = "gzip" if name.endswith(".gz") else None
    sep = "," if ".csv" in name else "\t"

    df = pd.read_csv(
        path,
        sep=sep,
        header=None,
        dtype=str,
        compression=compression,
        engine="python",
    )

    if df.shape[1] >= 2:
        col = df.iloc[:, 1]
    else:
        col = df.iloc[:, 0]

    return {
        str(x).strip()
        for x in col.fillna("").astype(str)
        if str(x).strip()
        and str(x).strip().upper() not in {"NA", "N/A", "NULL", "---"}
    }


def find_feature_file(sample_extract_dir: Path):
    files = [p for p in sample_extract_dir.rglob("*") if p.is_file()]

    candidates = []
    for p in files:
        n = p.name.lower()

        if not n.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz", ".csv", ".csv.gz")):
            continue

        score = 0

        if "feature" in n:
            score += 100
        if "gene" in n:
            score += 50
        if "barcode" in n:
            score -= 100
        if "cell" in n or "meta" in n or "adt" in n or "protein" in n:
            score -= 40

        if score >= 0:
            candidates.append((score, p))

    if not candidates:
        raise RuntimeError(f"No feature file found under {sample_extract_dir}")

    candidates.sort(key=lambda x: (-x[0], str(x[1])))

    return candidates[0][1]


def main():
    project = project_dir()
    docs = project / "docs"
    results = project / "results" / "direct_subspace"

    protocol = docs / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION.txt"
    protocol_manifest = docs / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION_manifest.json"

    for p in [protocol, protocol_manifest]:
        if not p.exists():
            raise FileNotFoundError(p)

    pm = json.loads(protocol_manifest.read_text(encoding="utf-8"))
    expected_sha = pm["protocol_sha256"]
    actual_sha = sha256_file(protocol)

    print(f"=== {VERSION} FULL TRANSPORT-AWARE GENE UNIVERSE REBUILD ===")
    print("Expected protocol SHA:", expected_sha)
    print("Actual protocol SHA:  ", actual_sha)

    if expected_sha != actual_sha:
        raise SystemExit("FAIL: v44.0 protocol SHA mismatch.")

    print("PASS: v44.0 protocol verified.")
    print("External response outcomes used: NO")
    print()

    dev_platform = (
        project
        / "data"
        / "external"
        / "GSE130404"
        / "platform"
        / "GPL10558_full_geo_table.txt"
    )

    ext_platform = (
        project
        / "data"
        / "external"
        / "GSE44589"
        / "platform"
        / "GPL570_full_geo_table.txt"
    )

    technical_manifest = (
        results
        / "v43_4d_GSE236233_sample_technical_manifest.json"
    )

    for p in [dev_platform, ext_platform, technical_manifest]:
        if not p.exists():
            raise FileNotFoundError(p)

    dev = parse_platform(dev_platform)
    ext = parse_platform(ext_platform)

    tech = json.loads(technical_manifest.read_text(encoding="utf-8"))

    # Build ALL-9 GSE236233 feature intersection from previously extracted files.
    patient_feature_sets = []

    for rec in tech:
        extract_dir = Path(rec["matrix_path"]).parent

        feature_path = Path(rec["feature_path"])
        if not feature_path.exists():
            feature_path = find_feature_file(extract_dir)

        symbols = read_feature_symbols(feature_path)
        patient_feature_sets.append(symbols)

        print(
            rec["patient_id"],
            "features:",
            len(symbols),
            "|",
            feature_path.name,
        )

    if len(patient_feature_sets) != 9:
        raise RuntimeError(
            f"Expected 9 primary Lin-CD34+ patient feature sets, found {len(patient_feature_sets)}"
        )

    rna_all9 = set.intersection(*patient_feature_sets)

    print()
    print("GSE236233 genes present in all 9 feature tables:", len(rna_all9))

    dev_symbols = set(dev["symbol_to_probes"].keys())
    ext_symbols = set(ext["symbol_to_probes"].keys())

    rows = []
    eligible = []

    for gene in sorted(dev_symbols):
        dev_eids = set(dev["symbol_to_entrez"].get(gene, set()))

        exact_gpl570 = gene in ext_symbols

        matched_eids = {
            eid
            for eid in dev_eids
            if ext["entrez_to_probes"].get(eid)
        }

        if exact_gpl570:
            gpl570_status = "EXACT_SYMBOL"
            gpl570_ok = True
        elif len(matched_eids) == 1:
            gpl570_status = "ENTREZ_ID_MATCH"
            gpl570_ok = True
        elif len(matched_eids) > 1:
            gpl570_status = "AMBIGUOUS_ENTREZ"
            gpl570_ok = False
        else:
            gpl570_status = "UNRESOLVED"
            gpl570_ok = False

        rna_ok = gene in rna_all9

        ok = gpl570_ok and rna_ok

        if ok:
            eligible.append(gene)

        rows.append({
            "development_gene": gene,
            "development_entrez_ids": "|".join(sorted(dev_eids)),
            "gpl570_status": gpl570_status,
            "matched_entrez_ids": "|".join(sorted(matched_eids)),
            "gse236233_present_all9": int(rna_ok),
            "eligible_v44": int(ok),
        })

    audit_df = pd.DataFrame(rows)

    n_dev = len(dev_symbols)
    n_gpl = int(
        audit_df["gpl570_status"].isin(
            ["EXACT_SYMBOL", "ENTREZ_ID_MATCH"]
        ).sum()
    )
    n_rna = int(audit_df["gse236233_present_all9"].sum())
    n_eligible = len(eligible)

    status = (
        "PASS TO v44.2"
        if n_eligible >= MIN_GENES
        else "TECHNICALLY INSUFFICIENT"
    )

    audit_path = results / "v44_1b_full_transport_universe_audit.csv"
    eligible_path = results / "v44_1b_eligible_genes.txt"
    summary_path = results / "v44_1b_full_transport_universe_summary.txt"
    manifest_path = results / "v44_1b_manifest.json"

    audit_df.to_csv(audit_path, index=False)
    eligible_path.write_text(
        "\n".join(sorted(eligible)) + "\n",
        encoding="utf-8",
    )

    summary = [
        "=== Soft Spaces / CML v44.1b FULL TRANSPORT-AWARE GENE UNIVERSE ===",
        "",
        "OUTCOME BLIND",
        "-------------",
        "External response outcomes used: NO",
        "",
        "UNIVERSE COUNTS",
        "---------------",
        f"Development GPL10558 symbols:              {n_dev}",
        f"Resolvable on GPL570:                     {n_gpl}",
        f"Present in all 9 GSE236233 feature sets:  {n_rna}",
        f"Final cross-platform eligible universe:   {n_eligible}",
        "",
        f"Minimum required for v44 Top-256 design:  {MIN_GENES}",
        "",
        f"v44.1b STATUS: {status}",
        "",
        "RULE",
        "----",
        "Eligible means BOTH:",
        "1. resolvable on GPL570 by exact symbol or one stable Entrez ID;",
        "2. present in all 9 GSE236233 Lin-CD34+ feature tables.",
        "",
        "No external response outcome was used.",
    ]

    summary_path.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    manifest_path.write_text(
        json.dumps(
            {
                "version": VERSION,
                "status": status,
                "development_symbol_n": n_dev,
                "gpl570_resolvable_n": n_gpl,
                "gse236233_all9_n": n_rna,
                "eligible_universe_n": n_eligible,
                "minimum_required": MIN_GENES,
                "external_outcomes_used": False,
                "sha256": {
                    "v44_0_protocol": actual_sha,
                    "development_platform": sha256_file(dev_platform),
                    "gpl570_platform": sha256_file(ext_platform),
                    "gse236233_technical_manifest": sha256_file(technical_manifest),
                    "eligible_genes": sha256_file(eligible_path),
                    "execution_script": sha256_file(Path(__file__).resolve()),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(summary_path.read_text(encoding="utf-8"))
    print("Wrote:")
    for p in [audit_path, eligible_path, summary_path, manifest_path]:
        print(" ", p)


if __name__ == "__main__":
    main()
