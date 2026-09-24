#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import pandas as pd

EXPECTED_PROTOCOL_SHA256 = "856f7a161d8d76c114a32d98ca812f36b52b018ef8108c4065a924f68b265e7c"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    code_dir = Path(__file__).resolve().parent
    project = code_dir.parent
    docs = project / "docs"
    results = project / "results" / "direct_subspace"

    protocol = docs / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION.txt"
    actual = sha256_file(protocol)

    print("=== v44.1 TRANSPORT-AWARE UNIVERSE PRECHECK ===")
    print("Expected protocol SHA:", EXPECTED_PROTOCOL_SHA256)
    print("Actual protocol SHA:  ", actual)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise SystemExit("FAIL: v44.0 protocol SHA mismatch.")
    print("PASS: v44.0 protocol verified.")
    print("External response outcomes used: NO")
    print()

    gpl_path = results / "v43_4b_identifier_harmonization.csv"
    rna_path = results / "v43_4d_GSE236233_frozen_gene_coverage.csv"
    for p in [gpl_path, rna_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    gpl = pd.read_csv(gpl_path)
    rna = pd.read_csv(rna_path)

    gpl_ok = set(gpl.loc[
        gpl["match_status"].isin(["EXACT_SYMBOL", "ENTREZ_ID_MATCH"]),
        "development_gene"
    ].astype(str))

    rna_ok = set(rna.loc[
        rna["present_all_9"] == 1,
        "gene"
    ].astype(str))

    prior_intersection = sorted(gpl_ok & rna_ok)

    summary_path = results / "v44_1_transport_universe_precheck_summary.txt"
    list_path = results / "v44_1_prior256_crossplatform_intersection.txt"
    manifest_path = results / "v44_1_manifest.json"

    summary = [
        "=== Soft Spaces / CML v44.1 TRANSPORT-AWARE UNIVERSE PRECHECK ===",
        "",
        "External response outcomes used: NO",
        "",
        "PRIOR v43 TOP-256 DIAGNOSTIC",
        "---------------------------",
        "GPL570 transportable: " + str(len(gpl_ok)),
        "GSE236233 all-9 present: " + str(len(rna_ok)),
        "Cross-platform intersection: " + str(len(prior_intersection)),
        "",
        "STATUS",
        "------",
        "FULL-UNIVERSE REBUILD REQUIRED",
        "",
        "The old audit tables contain only the prior v43 Top-256.",
        "They are not sufficient to define the full v44 eligible universe.",
        "",
        "NEXT",
        "----",
        "Use v44.1b to rebuild the full development universe outcome-blind",
        "from GPL10558, GPL570 and the downloaded GSE236233 feature tables."
    ]

    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    list_path.write_text("\n".join(prior_intersection) + "\n", encoding="utf-8")

    manifest_path.write_text(json.dumps({
        "version": "v44.1",
        "status": "FULL-UNIVERSE REBUILD REQUIRED",
        "prior256_gpl570_transportable": len(gpl_ok),
        "prior256_gse236233_all9_present": len(rna_ok),
        "prior256_crossplatform_intersection": len(prior_intersection),
        "external_outcomes_used": False,
        "sha256": {
            "v44_0_protocol": actual,
            "v43_4b": sha256_file(gpl_path),
            "v43_4d": sha256_file(rna_path)
        }
    }, indent=2), encoding="utf-8")

    print(summary_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
