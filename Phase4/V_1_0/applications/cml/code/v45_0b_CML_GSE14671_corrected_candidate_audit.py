#!/usr/bin/env python3
"""
Soft Spaces / CML
v45.0b — Corrected GSE14671 candidate audit

Correction
----------
v45.0 failed to recognize response classes because GSE14671 encodes
the 12-month imatinib outcome directly in Sample_source_name_ch1:

- complete cytogenetic response (CCyR) after 12 months
- >65% Ph-positive metaphases after 12 months

This script corrects ONLY the metadata parser.

It does NOT:
- load the frozen v44 model;
- score any sample;
- select genes;
- tune any model;
- alter any v44 result.

The script also audits SERIES-level text separately for evidence that the
expression material was collected before treatment. It does not infer
pretreatment status merely from the 12-month outcome wording in source_name.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


VERSION = "v45.0b"
GSE = "GSE14671"

SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE14nnn/GSE14671/soft/GSE14671_family.soft.gz"
)


def project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_if_missing(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        print("Using existing:", path)
        return

    print("Downloading:", url)
    print(" ->", path)
    urllib.request.urlretrieve(url, path)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def parse_soft(path: Path):
    samples = []
    current = None
    series = defaultdict(list)

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if line.startswith("^SERIES"):
                current = None
                continue

            if line.startswith("^SAMPLE"):
                gsm = line.split("=", 1)[1].strip()
                current = {
                    "gsm": gsm,
                    "title": "",
                    "source_name": "",
                    "platform_id": "",
                    "characteristics": [],
                    "description": [],
                    "treatment_protocol": [],
                }
                samples.append(current)
                continue

            if not line.startswith("!") or "=" not in line:
                continue

            key, value = line[1:].split("=", 1)
            key = key.strip()
            value = normalize_ws(value)

            if current is None:
                if key.startswith("Series_"):
                    series[key].append(value)
                continue

            if key == "Sample_title":
                current["title"] = value
            elif key == "Sample_source_name_ch1":
                current["source_name"] = value
            elif key == "Sample_platform_id":
                current["platform_id"] = value
            elif key == "Sample_characteristics_ch1":
                current["characteristics"].append(value)
            elif key == "Sample_description":
                current["description"].append(value)
            elif key == "Sample_treatment_protocol_ch1":
                current["treatment_protocol"].append(value)

    return samples, dict(series)


def classify_response_from_source(source_name: str):
    t = normalize_ws(source_name).lower()

    if (
        "complete cytogenetic response" in t
        or "ccyr" in t
    ):
        return "RESPONDER_CCYR"

    if (
        "65% ph-positive" in t
        or "65% ph positive" in t
        or ">65% ph" in t
    ):
        return "NON_RESPONDER_GT65_PH_POS"

    return "UNKNOWN"


def audit_pretreatment_series_design(series):
    """
    Inspect SERIES-level text only.

    Strong evidence:
      explicit pretreatment / pre-treatment / before treatment /
      before imatinib / prior to imatinib wording.

    We keep this separate from sample response labels.
    """
    relevant_keys = [
        "Series_title",
        "Series_summary",
        "Series_overall_design",
    ]

    text_parts = []

    for key in relevant_keys:
        text_parts.extend(series.get(key, []))

    text = " || ".join(text_parts)
    tl = text.lower()

    patterns = [
        r"\bpretreat",
        r"\bpre[- ]treat",
        r"\bbefore\s+(?:imatinib|treatment|therapy)",
        r"\bprior\s+to\s+(?:imatinib|treatment|therapy)",
        r"\bat\s+diagnosis\b",
        r"\bnewly diagnosed\b",
    ]

    hits = [
        p for p in patterns
        if re.search(p, tl)
    ]

    return {
        "series_text": text,
        "pretreatment_evidence": bool(hits),
        "matched_patterns": hits,
    }


def main():
    project = project_dir()

    metadata_dir = (
        project
        / "data"
        / "external"
        / GSE
        / "metadata"
    )

    results_dir = (
        project
        / "results"
        / "direct_subspace"
    )

    metadata_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    soft_path = metadata_dir / f"{GSE}_family.soft.gz"
    download_if_missing(SOFT_URL, soft_path)

    samples, series = parse_soft(soft_path)

    if len(samples) != 59:
        raise RuntimeError(
            f"Expected 59 GEO SAMPLE records, found {len(samples)}."
        )

    platform_counts = Counter(
        x["platform_id"] for x in samples
    )

    source_counts = Counter(
        x["source_name"] for x in samples
    )

    response_counts = Counter()

    rows = []

    for rec in samples:
        response = classify_response_from_source(
            rec["source_name"]
        )

        response_counts[response] += 1

        rows.append({
            "gsm": rec["gsm"],
            "title": rec["title"],
            "source_name": rec["source_name"],
            "platform_id": rec["platform_id"],
            "response_class_frozen_candidate": response,
            "characteristics": " || ".join(rec["characteristics"]),
            "sample_description": " || ".join(rec["description"]),
            "sample_treatment_protocol": " || ".join(rec["treatment_protocol"]),
        })

    design = audit_pretreatment_series_design(series)

    responder_n = response_counts.get(
        "RESPONDER_CCYR",
        0,
    )

    nonresponder_n = response_counts.get(
        "NON_RESPONDER_GT65_PH_POS",
        0,
    )

    unknown_n = response_counts.get(
        "UNKNOWN",
        0,
    )

    exact_response_structure = (
        responder_n == 41
        and nonresponder_n == 18
        and unknown_n == 0
    )

    single_gpl570 = (
        platform_counts == Counter({"GPL570": 59})
    )

    if (
        single_gpl570
        and exact_response_structure
    ):
        status = (
            "CANDIDATE ELIGIBLE FOR v45.1 PREREGISTRATION"
        )
    else:
        status = (
            "CANDIDATE REQUIRES FURTHER METADATA RESOLUTION"
        )

    samples_out = (
        results_dir
        / "v45_0b_GSE14671_corrected_samples.tsv"
    )

    cols = [
        "gsm",
        "title",
        "source_name",
        "platform_id",
        "response_class_frozen_candidate",
        "characteristics",
        "sample_description",
        "sample_treatment_protocol",
    ]

    with samples_out.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        f.write("\t".join(cols) + "\n")

        for row in rows:
            f.write(
                "\t".join(
                    str(row[c]).replace(
                        "\t",
                        " ",
                    )
                    for c in cols
                )
                + "\n"
            )

    series_out = (
        results_dir
        / "v45_0b_GSE14671_series_design_audit.txt"
    )

    series_lines = [
        "=== GSE14671 SERIES-LEVEL DESIGN AUDIT ===",
        "",
        "Series title:",
    ]

    series_lines.extend(
        series.get(
            "Series_title",
            ["<missing>"],
        )
    )

    series_lines += [
        "",
        "Series summary:",
    ]

    series_lines.extend(
        series.get(
            "Series_summary",
            ["<missing>"],
        )
    )

    series_lines += [
        "",
        "Overall design:",
    ]

    series_lines.extend(
        series.get(
            "Series_overall_design",
            ["<missing>"],
        )
    )

    series_lines += [
        "",
        "Pretreatment wording detected in series-level text: "
        + (
            "YES"
            if design["pretreatment_evidence"]
            else "NO"
        ),
        "",
        "Important:",
        "Pretreatment status is audited separately from the 12-month",
        "response wording contained in Sample_source_name_ch1.",
    ]

    series_out.write_text(
        "\n".join(series_lines) + "\n",
        encoding="utf-8",
    )

    summary_out = (
        results_dir
        / "v45_0b_GSE14671_corrected_candidate_audit_summary.txt"
    )

    summary = [
        "=== Soft Spaces / CML v45.0b GSE14671 CORRECTED CANDIDATE AUDIT ===",
        "",
        "CORRECTION",
        "----------",
        "v45.0 did not recognize the response labels because GSE14671",
        "encodes them directly in Sample_source_name_ch1.",
        "",
        "This correction changes metadata parsing only.",
        "Frozen v44 model loaded: NO",
        "Predictive scoring performed: NO",
        "External outcome tuning performed: NO",
        "",
        "DATASET",
        "-------",
        "GEO accession: GSE14671",
        f"GEO sample records: {len(samples)}",
        "",
        "PLATFORM",
        "--------",
    ]

    for platform, count in platform_counts.most_common():
        summary.append(
            f"{platform or '<missing>'}: {count}"
        )

    summary += [
        "",
        "FROZEN-CANDIDATE RESPONSE DEFINITION",
        "------------------------------------",
        "Responder candidate:",
        "    complete cytogenetic response (CCyR) after 12 months",
        "",
        "Non-responder candidate:",
        "    >65% Ph-positive metaphases after 12 months of imatinib",
        "",
        "RESPONSE COUNTS",
        "---------------",
        f"Responder / CCyR:                 {responder_n}",
        f"Non-responder / >65% Ph+:        {nonresponder_n}",
        f"Unknown:                          {unknown_n}",
        "",
        "SERIES-DESIGN TIMING AUDIT",
        "--------------------------",
        "Pretreatment evidence in series-level text: "
        + (
            "YES"
            if design["pretreatment_evidence"]
            else "NO / NOT EXPLICITLY DETECTED"
        ),
        "",
        "The response labels describe outcome after 12 months;",
        "they are not being interpreted as the time at which expression",
        "was measured.",
        "",
        "TECHNICAL CANDIDACY",
        "-------------------",
        f"59/59 on GPL570:                  {'YES' if single_gpl570 else 'NO'}",
        f"Exact 41/18 response split:       {'YES' if exact_response_structure else 'NO'}",
        "",
        f"v45.0b STATUS: {status}",
        "",
        "NEXT",
        "----",
        "If this audit is accepted, v45.1 should freeze the exact",
        "cross-endpoint test BEFORE the frozen v44 model is scored.",
        "",
        "No predictive result has yet been observed for GSE14671.",
    ]

    summary_out.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    manifest_out = (
        results_dir
        / "v45_0b_manifest.json"
    )

    manifest_out.write_text(
        json.dumps(
            {
                "version": VERSION,
                "dataset": GSE,
                "sample_n": len(samples),
                "platform_counts": dict(platform_counts),
                "response_definition": {
                    "responder": (
                        "complete cytogenetic response (CCyR) "
                        "after 12 months of imatinib"
                    ),
                    "non_responder": (
                        ">65% Ph-positive metaphases after "
                        "12 months of imatinib"
                    ),
                },
                "responder_n": responder_n,
                "non_responder_n": nonresponder_n,
                "unknown_n": unknown_n,
                "exact_41_18_split": exact_response_structure,
                "pretreatment_evidence_in_series_text": design[
                    "pretreatment_evidence"
                ],
                "status": status,
                "frozen_v44_model_loaded": False,
                "predictive_scoring_performed": False,
                "external_outcome_tuning_performed": False,
                "sha256": {
                    "family_soft": sha256_file(
                        soft_path
                    ),
                    "samples_tsv": sha256_file(
                        samples_out
                    ),
                    "series_design_audit": sha256_file(
                        series_out
                    ),
                    "execution_script": sha256_file(
                        Path(__file__).resolve()
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        summary_out.read_text(
            encoding="utf-8"
        )
    )

    print("Wrote:")

    for p in [
        samples_out,
        series_out,
        summary_out,
        manifest_out,
    ]:
        print(
            " ",
            p,
        )


if __name__ == "__main__":
    main()
