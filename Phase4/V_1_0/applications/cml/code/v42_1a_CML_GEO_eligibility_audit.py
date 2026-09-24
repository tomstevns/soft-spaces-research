#!/usr/bin/env python3
"""
Soft Spaces / CML
v42.1a — GEO metadata + eligibility audit for GSE130404 and GSE44589

NO MODEL FITTING.
NO SOFT-SPACES SCORING.
NO CLASSIFIER TRAINING.

This script downloads GEO series matrices and inspects only:
- sample count
- platform IDs
- titles / source names
- characteristics metadata
- matrix dimensions
- timepoint / response metadata availability

Purpose:
Determine whether GSE130404 and GSE44589 are suitable for the frozen v42.0
CML protocol before any modeling is performed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "v42.1a"

SERIES = {
    "GSE130404": {
        "ftp": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/"
            "GSE130404/matrix/GSE130404_series_matrix.txt.gz"
        ),
        "expected_platform": "GPL10558",
        "expected_samples": 96,
        "provisional_role": "preferred_development_candidate",
    },
    "GSE44589": {
        "ftp": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE44nnn/"
            "GSE44589/matrix/GSE44589_series_matrix.txt.gz"
        ),
        "expected_platform": "GPL570",
        "expected_samples": 198,
        "provisional_role": "alternative_development_candidate",
    },
}


def project_dir_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path):
    if path.exists() and path.stat().st_size > 0:
        print("[download] reuse:", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".part")

    print("[download]", url)
    print("        ->", path)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 SoftSpaces-v42.1a"},
    )

    with urllib.request.urlopen(req, timeout=180) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)

    tmp.replace(path)
    print("[download] done:", f"{path.stat().st_size:,}", "bytes")


def parse_quoted_tab_line(line: str):
    parts = line.rstrip("\r\n").split("\t")
    return [x.strip().strip('"') for x in parts]


def parse_series_matrix(path: Path):
    meta = defaultdict(list)
    matrix_header = None
    matrix_rows = 0
    n_values_first_row = None
    inside = False

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_"):
                key, *vals = parse_quoted_tab_line(line)
                meta[key].append(vals)
                continue

            if line.startswith("!series_matrix_table_begin"):
                inside = True
                continue

            if line.startswith("!series_matrix_table_end"):
                inside = False
                break

            if inside:
                if matrix_header is None:
                    matrix_header = parse_quoted_tab_line(line)
                    continue

                if line.strip():
                    matrix_rows += 1
                    if n_values_first_row is None:
                        n_values_first_row = len(line.rstrip("\r\n").split("\t")) - 1

    if matrix_header is None:
        raise RuntimeError(f"No matrix header found in {path}")

    sample_ids = matrix_header[1:]

    return {
        "sample_ids": sample_ids,
        "n_samples": len(sample_ids),
        "n_matrix_rows": matrix_rows,
        "n_values_first_row": n_values_first_row,
        "metadata": dict(meta),
    }


def flatten_meta(meta, key):
    """
    GEO can repeat Sample_characteristics_ch1 lines.
    Return list of lists, one row per repeated metadata line.
    """
    return meta.get(key, [])


def summarize_vector(values):
    c = Counter(values)
    return [
        {"value": k, "count": v}
        for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def metadata_field_summary(meta):
    out = {}

    for key, rows in meta.items():
        if not rows:
            continue

        # Each repeated GEO metadata row should have one value per sample.
        row_summaries = []
        for idx, vals in enumerate(rows, 1):
            row_summaries.append(
                {
                    "occurrence": idx,
                    "n_values": len(vals),
                    "unique_values": len(set(vals)),
                    "value_counts": summarize_vector(vals)[:40],
                }
            )

        out[key] = row_summaries

    return out


def find_metadata_terms(meta, terms):
    hits = []

    for key, rows in meta.items():
        for occ, vals in enumerate(rows, 1):
            joined = " || ".join(vals)
            low = joined.lower()

            found = [
                term
                for term in terms
                if term.lower() in low or term.lower() in key.lower()
            ]

            if found:
                hits.append(
                    {
                        "key": key,
                        "occurrence": occ,
                        "matched_terms": found,
                        "example_values": vals[:10],
                        "n_unique": len(set(vals)),
                    }
                )

    return hits


def extract_platform_ids(meta):
    rows = meta.get("!Sample_platform_id", [])
    vals = []
    for row in rows:
        vals.extend(row)
    return sorted(set(vals))


def inspect_accession(accession: str, spec: dict, root: Path):
    raw_dir = root / accession / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{accession}_series_matrix.txt.gz"
    download(spec["ftp"], path)

    parsed = parse_series_matrix(path)
    meta = parsed["metadata"]
    platforms = extract_platform_ids(meta)

    terms = [
        "response",
        "responder",
        "non-responder",
        "molecular response",
        "early molecular response",
        "EMR",
        "BCR-ABL",
        "imatinib",
        "treatment",
        "therapy",
        "week",
        "month",
        "diagnosis",
        "chronic phase",
        "cytogenetic",
    ]

    relevant_hits = find_metadata_terms(meta, terms)

    result = {
        "accession": accession,
        "role": spec["provisional_role"],
        "matrix_path": str(path),
        "matrix_sha256": sha256_file(path),
        "expected_sample_count": spec["expected_samples"],
        "observed_sample_count": parsed["n_samples"],
        "sample_count_matches_expectation": (
            parsed["n_samples"] == spec["expected_samples"]
        ),
        "expected_platform": spec["expected_platform"],
        "observed_platform_ids": platforms,
        "platform_matches_expectation": (
            spec["expected_platform"] in platforms
        ),
        "n_expression_rows": parsed["n_matrix_rows"],
        "n_values_first_expression_row": parsed["n_values_first_row"],
        "matrix_shape_consistent": (
            parsed["n_values_first_row"] == parsed["n_samples"]
        ),
        "metadata_relevant_hits": relevant_hits,
        "metadata_summary": metadata_field_summary(meta),
    }

    # Technical eligibility only. Biological phenotype clarity is reviewed
    # from the written audit output, not automatically inferred.
    result["technical_matrix_eligibility"] = bool(
        result["sample_count_matches_expectation"]
        and result["platform_matches_expectation"]
        and result["matrix_shape_consistent"]
        and result["n_expression_rows"] > 1000
    )

    return result


def human_readable_audit(results):
    lines = [
        f"=== Soft Spaces / CML {VERSION} GEO ELIGIBILITY AUDIT ===",
        "",
        "NO MODELING WAS PERFORMED.",
        "",
    ]

    for r in results:
        lines += [
            r["accession"],
            "-" * len(r["accession"]),
            f"Provisional role: {r['role']}",
            f"Expected samples: {r['expected_sample_count']}",
            f"Observed samples: {r['observed_sample_count']}",
            f"Sample count match: {r['sample_count_matches_expectation']}",
            f"Expected platform: {r['expected_platform']}",
            f"Observed platform IDs: {', '.join(r['observed_platform_ids'])}",
            f"Platform match: {r['platform_matches_expectation']}",
            f"Expression rows: {r['n_expression_rows']}",
            f"Matrix shape consistent: {r['matrix_shape_consistent']}",
            f"Technical matrix eligibility: {r['technical_matrix_eligibility']}",
            "",
            "Relevant metadata hits:",
        ]

        if not r["metadata_relevant_hits"]:
            lines.append("  NONE FOUND IN SERIES-MATRIX SAMPLE METADATA")
        else:
            for h in r["metadata_relevant_hits"]:
                lines.append(
                    f"  {h['key']} occurrence {h['occurrence']} | "
                    f"unique={h['n_unique']} | terms={','.join(h['matched_terms'])}"
                )
                for x in h["example_values"][:5]:
                    lines.append(f"      {x}")

        lines.append("")

    lines += [
        "DECISION RULE FOR v42.1b",
        "------------------------",
        "Do NOT choose the development/external pair merely because the matrix",
        "downloads successfully.",
        "",
        "Before v42.2, manually confirm:",
        "- primary phenotype definition;",
        "- response class counts;",
        "- baseline/pre-treatment sample subset;",
        "- unique-patient status;",
        "- independence of external cohort;",
        "- compatible gene mapping;",
        "- no outcome leakage.",
        "",
        "No Soft-Spaces or classifier result may be generated in v42.1a.",
    ]

    return "\n".join(lines) + "\n"


def main():
    project = project_dir_from_script()

    data_root = project / "data" / "external"
    results_dir = project / "results" / "dataset_audit"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"=== {VERSION} CML GEO METADATA AUDIT ===")
    print("NO MODELING")
    print()

    for accession, spec in SERIES.items():
        print(f"\n[{accession}]")
        results.append(
            inspect_accession(
                accession,
                spec,
                data_root,
            )
        )

    txt_path = results_dir / "v42_1a_CML_GEO_eligibility_audit.txt"
    json_path = results_dir / "v42_1a_CML_GEO_eligibility_audit.json"

    txt_path.write_text(
        human_readable_audit(results),
        encoding="utf-8",
    )

    manifest = {
        "version": VERSION,
        "modeling_performed": False,
        "softspaces_scoring_performed": False,
        "classifier_training_performed": False,
        "datasets": results,
    }

    json_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print(txt_path.read_text(encoding="utf-8"))
    print("Wrote:")
    print(" ", txt_path)
    print(" ", json_path)


if __name__ == "__main__":
    main()
