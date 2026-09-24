#!/usr/bin/env python3
"""
Soft Spaces / CML
v43.4b — Outcome-blind identifier harmonization audit

Allowed matching:
1) exact gene symbol
2) same stable Entrez Gene ID across GPL10558 and GPL570

No GSE44589 response labels are used.
Ambiguous/unresolved mappings are not rescued.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "v43.4b"
EXPECTED_V43_4_SHA256 = "361f3c57ccc5687bedc829080f86412fdc7f297a61fd814120502f5bc2b836b9"
TOP_K = 256
FROZEN_R = 16


def project_dir():
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_file(candidates, label):
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(label + " not found:\n" + "\n".join(map(str, candidates)))


def norm_header(s):
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())


def split_multi(raw):
    s = str(raw or "").strip()
    for sep in ("///", "//", ";", ",", "|"):
        s = s.replace(sep, "|")
    return [
        x.strip() for x in s.split("|")
        if x.strip() and x.strip().upper() not in {"---", "NA", "N/A", "NULL", "NONE"}
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
    lut = {norm_header(x): x for x in header}

    id_col = next((lut[x] for x in ("id", "idref", "probeid", "probesetid") if x in lut), None)
    sym_col = next((lut[x] for x in ("symbol", "genesymbol", "genesymbols", "officialgenesymbol") if x in lut), None)
    entrez_col = next((lut[x] for x in ("entrezgene", "entrezgeneid", "entrezid", "geneid", "ncbigeneid") if x in lut), None)

    if entrez_col is None:
        entrez_col = next((orig for n, orig in lut.items() if "entrez" in n), None)

    if id_col is None or sym_col is None or entrez_col is None:
        raise RuntimeError(
            f"Could not identify ID/symbol/Entrez columns in {path.name}\nColumns: {header}"
        )

    symbol_to_entrez = defaultdict(set)
    entrez_to_symbols = defaultdict(set)
    entrez_to_probes = defaultdict(set)
    symbol_to_probes = defaultdict(set)

    reader = csv.DictReader(table[1:], fieldnames=header, delimiter="\t")
    for row in reader:
        probe = str(row.get(id_col, "")).strip()
        if not probe:
            continue
        syms = set(split_multi(row.get(sym_col, "")))
        eids = parse_entrez(row.get(entrez_col, ""))

        for s in syms:
            symbol_to_probes[s].add(probe)
            for eid in eids:
                symbol_to_entrez[s].add(eid)
                entrez_to_symbols[eid].add(s)

        for eid in eids:
            entrez_to_probes[eid].add(probe)

    return {
        "id_col": id_col,
        "symbol_col": sym_col,
        "entrez_col": entrez_col,
        "symbol_to_entrez": dict(symbol_to_entrez),
        "entrez_to_symbols": dict(entrez_to_symbols),
        "entrez_to_probes": dict(entrez_to_probes),
        "symbol_to_probes": dict(symbol_to_probes),
    }


def main():
    project = project_dir()
    result_dir = project / "results" / "direct_subspace"

    prereg = find_file([
        project / "docs" / "v43_4_CML_EXTERNAL_GENERALIZATION_PREREGISTRATION.txt",
        project / "code" / "v43_4_CML_EXTERNAL_GENERALIZATION_PREREGISTRATION.txt",
    ], "v43.4 preregistration")

    actual_sha = sha256_file(prereg)
    print(f"=== {VERSION} CML IDENTIFIER HARMONIZATION ===")
    print("Expected v43.4 SHA:", EXPECTED_V43_4_SHA256)
    print("Actual v43.4 SHA:  ", actual_sha)
    if actual_sha != EXPECTED_V43_4_SHA256:
        raise SystemExit("FAIL: v43.4 preregistration SHA mismatch.")
    print("PASS: v43.4 preregistration verified.")
    print("External response labels used: NO\n")

    m43a = result_dir / "v43_4a_manifest.json"
    audit43a = result_dir / "v43_4a_transport_gene_audit.csv"
    basis = result_dir / "v43_4a_frozen_development_basis.npz"

    for p in (m43a, audit43a, basis):
        if not p.exists():
            raise FileNotFoundError(p)

    prior = json.loads(m43a.read_text(encoding="utf-8"))
    if prior.get("status") != "TECHNICALLY NON-EVALUABLE":
        raise RuntimeError("Expected v43.4a TECHNICALLY NON-EVALUABLE.")

    dev_platform = find_file([
        project / "data" / "external" / "GSE130404" / "platform" / "GPL10558_full_geo_table.txt",
    ], "GPL10558")

    ext_platform = find_file([
        project / "data" / "external" / "GSE44589" / "platform" / "GPL570_full_geo_table.txt",
        project / "data" / "external" / "GSE44589" / "platform" / "GPL570_platform.txt",
    ], "GPL570")

    dev = parse_platform(dev_platform)
    ext = parse_platform(ext_platform)

    print("Development symbol / Entrez columns:", dev["symbol_col"], "/", dev["entrez_col"])
    print("External symbol / Entrez columns:   ", ext["symbol_col"], "/", ext["entrez_col"])
    print()

    frozen = np.load(basis, allow_pickle=False)
    genes = [str(x) for x in frozen["genes"].tolist()]
    U16 = np.asarray(frozen["U16"], dtype=float)

    if len(genes) != TOP_K or U16.shape != (TOP_K, FROZEN_R):
        raise RuntimeError("Frozen basis dimensions do not match v43.4.")

    coord_mass = np.sum(U16 * U16, axis=1)
    ext_symbols = set(ext["symbol_to_probes"].keys())

    rows = []
    for i, gene in enumerate(genes):
        dev_eids = set(dev["symbol_to_entrez"].get(gene, set()))
        exact = gene in ext_symbols

        matched_eids = set()
        candidate_ext_symbols = set()

        if not exact:
            for eid in dev_eids:
                if ext["entrez_to_probes"].get(eid):
                    matched_eids.add(eid)
                    candidate_ext_symbols.update(ext["entrez_to_symbols"].get(eid, set()))

        if exact:
            status = "EXACT_SYMBOL"
        elif len(matched_eids) == 1:
            status = "ENTREZ_ID_MATCH"
        elif len(matched_eids) > 1:
            status = "AMBIGUOUS_ENTREZ"
        else:
            status = "UNRESOLVED"

        rows.append({
            "coordinate_index": i,
            "development_gene": gene,
            "projector_diagonal_mass": float(coord_mass[i]),
            "development_entrez_ids": "|".join(sorted(dev_eids)),
            "match_status": status,
            "matched_entrez_ids": "|".join(sorted(matched_eids)),
            "external_symbols_for_match": "|".join(sorted(candidate_ext_symbols)),
        })

    df = pd.DataFrame(rows)
    accepted = df["match_status"].isin(["EXACT_SYMBOL", "ENTREZ_ID_MATCH"])

    exact_n = int((df["match_status"] == "EXACT_SYMBOL").sum())
    entrez_n = int((df["match_status"] == "ENTREZ_ID_MATCH").sum())
    ambiguous_n = int((df["match_status"] == "AMBIGUOUS_ENTREZ").sum())
    unresolved_n = int((df["match_status"] == "UNRESOLVED").sum())
    accepted_n = int(accepted.sum())
    remaining_n = TOP_K - accepted_n

    total_mass = float(df["projector_diagonal_mass"].sum())
    retained_mass = float(df.loc[accepted, "projector_diagonal_mass"].sum())
    retained_fraction = retained_mass / total_mass

    status = "TECHNICALLY EVALUABLE" if remaining_n == 0 else "TECHNICALLY NON-EVALUABLE"

    audit_out = result_dir / "v43_4b_identifier_harmonization.csv"
    summary_out = result_dir / "v43_4b_identifier_harmonization_summary.txt"
    manifest_out = result_dir / "v43_4b_manifest.json"

    df.to_csv(audit_out, index=False)

    unresolved = df.loc[~accepted]

    lines = [
        "=== Soft Spaces / CML v43.4b OUTCOME-BLIND IDENTIFIER HARMONIZATION ===",
        "",
        "ALLOWED MATCHING",
        "----------------",
        "1. Exact gene symbol",
        "2. Same stable Entrez Gene ID across GPL10558 and GPL570",
        "",
        "RESULT",
        "------",
        f"Exact-symbol coordinates:       {exact_n}",
        f"Recovered by Entrez ID:         {entrez_n}",
        f"Ambiguous Entrez mappings:      {ambiguous_n}",
        f"Unresolved coordinates:         {unresolved_n}",
        f"Total accepted coordinates:     {accepted_n}",
        f"Still non-transportable:        {remaining_n}",
        f"Coordinate coverage:            {accepted_n / TOP_K:.6f}",
        "",
        "PROJECTOR MASS",
        "--------------",
        f"Total projector mass:           {total_mass:.6f}",
        f"Accepted-coordinate mass:       {retained_mass:.6f}",
        f"Retained projector fraction:    {retained_fraction:.6f}",
        "",
        f"v43.4b STATUS: {status}",
        "",
        "UNRESOLVED / AMBIGUOUS",
        "----------------------",
    ]

    if unresolved.empty:
        lines.append("NONE")
    else:
        for _, r in unresolved.iterrows():
            lines.append(
                f"{r['development_gene']} | dev Entrez={r['development_entrez_ids'] or '-'} | "
                f"status={r['match_status']} | matched Entrez={r['matched_entrez_ids'] or '-'} | "
                f"external symbols={r['external_symbols_for_match'] or '-'} | "
                f"mass={r['projector_diagonal_mass']:.6f}"
            )

    lines += [
        "",
        "INTERPRETATION",
        "--------------",
        "No GSE44589 response outcomes were used.",
        "Only exact symbol identity or stable Entrez identity is accepted.",
        "Ambiguous or unresolved coordinates are not rescued.",
        "",
        "Under the strict frozen rule, v43.5 may proceed only if every",
        "frozen coordinate is transportable without outcome-guided substitution.",
    ]

    summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_out.write_text(json.dumps({
        "version": VERSION,
        "status": status,
        "development_platform": "GPL10558",
        "external_platform": "GPL570",
        "frozen_r": FROZEN_R,
        "frozen_coordinates": TOP_K,
        "exact_symbol_n": exact_n,
        "entrez_recovered_n": entrez_n,
        "ambiguous_entrez_n": ambiguous_n,
        "unresolved_n": unresolved_n,
        "accepted_n": accepted_n,
        "nontransportable_n": remaining_n,
        "coordinate_coverage": accepted_n / TOP_K,
        "retained_projector_mass_fraction": retained_fraction,
        "external_response_labels_used": False,
        "sha256": {
            "v43_4_preregistration": actual_sha,
            "v43_4a_manifest": sha256_file(m43a),
            "frozen_basis": sha256_file(basis),
            "development_platform": sha256_file(dev_platform),
            "external_platform": sha256_file(ext_platform),
            "execution_script": sha256_file(Path(__file__).resolve()),
        }
    }, indent=2), encoding="utf-8")

    print(summary_out.read_text(encoding="utf-8"))
    print("Wrote:")
    for p in (audit_out, summary_out, manifest_out):
        print(" ", p)


if __name__ == "__main__":
    main()
