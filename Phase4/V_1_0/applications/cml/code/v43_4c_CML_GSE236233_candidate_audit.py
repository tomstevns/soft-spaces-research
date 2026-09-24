#!/usr/bin/env python3
"""
Soft Spaces / CML
v43.4c — New external candidate audit: GSE236233

Purpose
-------
Outcome-blind eligibility audit of GSE236233 as a replacement external cohort
after GSE44589 proved technically non-evaluable for the frozen v43 subspace.

Known design target
-------------------
GSE236233:
- Homo sapiens
- chronic-phase CML
- diagnosis bone marrow
- CITE-seq / single-cell RNA-seq
- Lin-CD34+ and Lin-CD34+CD38- populations
- nine CML patients
- patients stratified by molecular response after 12 months of TKI

Critical rule
-------------
The statistical unit MUST be the patient, not the cell.

Cells may be aggregated within patient/population to construct patient-level
expression summaries, but individual cells must never be treated as independent
external response observations.

This script:
- downloads/parses GEO family SOFT metadata;
- audits samples, patient IDs and cell populations;
- records processed supplementary files;
- does NOT calculate external predictive performance;
- does NOT use cell-level outcome labels for model scoring;
- does NOT modify the frozen v43 representation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


VERSION = "v43.4c"
GSE = "GSE236233"

SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE236nnn/GSE236233/soft/GSE236233_family.soft.gz"
)


def project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print("Using existing:", path)
        return
    print("Downloading:", url)
    urllib.request.urlretrieve(url, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_soft(path: Path):
    """
    Minimal GEO SOFT parser sufficient for sample-level metadata and
    supplementary file inventory.
    """
    samples = []
    current = None
    series = defaultdict(list)

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if line.startswith("^SERIES"):
                current = ("SERIES", None)
                continue

            if line.startswith("^SAMPLE"):
                if "=" not in line:
                    continue
                gsm = line.split("=", 1)[1].strip()
                rec = {
                    "gsm": gsm,
                    "title": "",
                    "source": "",
                    "organism": "",
                    "characteristics": [],
                    "supplementary": [],
                    "platform": "",
                    "library_strategy": "",
                    "instrument": "",
                }
                samples.append(rec)
                current = ("SAMPLE", rec)
                continue

            if not line.startswith("!"):
                continue

            if current and current[0] == "SERIES":
                if "=" in line:
                    k, v = line[1:].split("=", 1)
                    series[k.strip()].append(v.strip())
                continue

            if current and current[0] == "SAMPLE":
                rec = current[1]
                if "=" not in line:
                    continue
                k, v = line[1:].split("=", 1)
                k = k.strip()
                v = v.strip()

                if k == "Sample_title":
                    rec["title"] = v
                elif k == "Sample_source_name_ch1":
                    rec["source"] = v
                elif k == "Sample_organism_ch1":
                    rec["organism"] = v
                elif k == "Sample_characteristics_ch1":
                    rec["characteristics"].append(v)
                elif k.startswith("Sample_supplementary_file"):
                    rec["supplementary"].append(v)
                elif k == "Sample_platform_id":
                    rec["platform"] = v
                elif k == "Sample_library_strategy":
                    rec["library_strategy"] = v
                elif k == "Sample_instrument_model":
                    rec["instrument"] = v

    return dict(series), samples


def patient_id_from_text(text: str):
    """
    Titles are expected to contain tokens such as [CML5_34p].
    We conservatively extract CML<number>.
    """
    m = re.search(r"\b(CML\d+)\b", text, flags=re.I)
    return m.group(1).upper() if m else None


def classify_population(text: str):
    t = text.lower().replace(" ", "")
    if "cd34+cd38-" in t or "cd34+38-" in t or "38n" in t:
        return "Lin-CD34+CD38-"
    if "cd34+" in t or "34p" in t:
        return "Lin-CD34+"
    return "OTHER/UNKNOWN"


def main():
    project = project_dir()

    rawdir = (
        project
        / "data"
        / "external"
        / GSE
        / "metadata"
    )
    outdir = (
        project
        / "results"
        / "direct_subspace"
    )

    rawdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    soft_path = rawdir / f"{GSE}_family.soft.gz"
    download(SOFT_URL, soft_path)

    series, samples = parse_soft(soft_path)

    print(f"=== {VERSION} NEW EXTERNAL CANDIDATE AUDIT ===")
    print("Candidate:", GSE)
    print("External predictive performance evaluated: NO")
    print("Cell-level observations treated as independent patients: NO")
    print()

    patient_to_samples = defaultdict(list)
    population_counts = Counter()
    platform_counts = Counter()
    organism_counts = Counter()
    strategy_counts = Counter()
    instrument_counts = Counter()

    no_patient_id = []

    for rec in samples:
        combined = " ".join(
            [
                rec["title"],
                rec["source"],
                *rec["characteristics"],
            ]
        )

        pid = patient_id_from_text(combined)
        pop = classify_population(combined)

        rec["patient_id"] = pid
        rec["population"] = pop

        if pid:
            patient_to_samples[pid].append(rec)
        else:
            no_patient_id.append(rec["gsm"])

        population_counts[pop] += 1
        platform_counts[rec["platform"]] += 1
        organism_counts[rec["organism"]] += 1
        strategy_counts[rec["library_strategy"]] += 1
        instrument_counts[rec["instrument"]] += 1

    patient_ids = sorted(patient_to_samples)
    n_patients = len(patient_ids)

    supplementary = []
    for rec in samples:
        for url in rec["supplementary"]:
            supplementary.append(
                {
                    "gsm": rec["gsm"],
                    "patient_id": rec["patient_id"],
                    "population": rec["population"],
                    "url": url,
                }
            )

    # Conservative eligibility conditions.
    all_human = (
        len(organism_counts) == 1
        and next(iter(organism_counts), "") == "Homo sapiens"
    )

    patient_level_identifiable = (
        n_patients >= 2
        and len(no_patient_id) == 0
    )

    has_processed_files = len(supplementary) > 0

    # This audit deliberately does not label it externally evaluable yet:
    # response-group mapping and frozen gene-coordinate coverage remain to audit.
    status = (
        "CANDIDATE ELIGIBLE FOR TECHNICAL FOLLOW-UP"
        if all_human and patient_level_identifiable and has_processed_files
        else "CANDIDATE NOT YET ELIGIBLE"
    )

    summary_path = (
        outdir
        / "v43_4c_GSE236233_candidate_audit_summary.txt"
    )
    manifest_path = (
        outdir
        / "v43_4c_GSE236233_candidate_audit_manifest.json"
    )
    sample_path = (
        outdir
        / "v43_4c_GSE236233_samples.tsv"
    )

    with sample_path.open("w", encoding="utf-8") as f:
        f.write(
            "gsm\tpatient_id\tpopulation\ttitle\tplatform\t"
            "library_strategy\tinstrument\tsupplementary_files\n"
        )
        for rec in samples:
            f.write(
                "\t".join(
                    [
                        rec["gsm"],
                        rec["patient_id"] or "",
                        rec["population"],
                        rec["title"].replace("\t", " "),
                        rec["platform"],
                        rec["library_strategy"],
                        rec["instrument"],
                        "|".join(rec["supplementary"]),
                    ]
                )
                + "\n"
            )

    lines = [
        "=== Soft Spaces / CML v43.4c GSE236233 CANDIDATE AUDIT ===",
        "",
        "PURPOSE",
        "-------",
        "Outcome-blind audit of a replacement external cohort after GSE44589",
        "was technically non-evaluable for the frozen v43 representation.",
        "",
        "DATASET",
        "-------",
        f"GEO accession: {GSE}",
        f"GEO sample records: {len(samples)}",
        f"Unique patient IDs detected: {n_patients}",
        f"Patient IDs: {', '.join(patient_ids) if patient_ids else 'NONE'}",
        "",
        "SAMPLE STRUCTURE",
        "----------------",
    ]

    for k, v in sorted(population_counts.items()):
        lines.append(f"{k}: {v}")

    lines += [
        "",
        "PLATFORM",
        "--------",
    ]
    for k, v in sorted(platform_counts.items()):
        lines.append(f"{k or 'UNKNOWN'}: {v}")

    lines += [
        "",
        "LIBRARY STRATEGY",
        "----------------",
    ]
    for k, v in sorted(strategy_counts.items()):
        lines.append(f"{k or 'UNKNOWN'}: {v}")

    lines += [
        "",
        "INSTRUMENT",
        "----------",
    ]
    for k, v in sorted(instrument_counts.items()):
        lines.append(f"{k or 'UNKNOWN'}: {v}")

    lines += [
        "",
        "PROCESSED DATA",
        "--------------",
        f"Supplementary processed-file references: {len(supplementary)}",
        "",
        "CRITICAL STATISTICAL RULE",
        "-------------------------",
        "The patient is the statistical unit.",
        "Single cells must NEVER be counted as independent response samples.",
        "Any later external expression representation must first aggregate",
        "cells within patient and a preregistered cell population.",
        "",
        "OUTCOME STATUS",
        "--------------",
        "No external predictive performance was calculated.",
        "No classifier was fitted to GSE236233.",
        "",
        f"v43.4c STATUS: {status}",
        "",
        "NEXT REQUIRED AUDIT",
        "-------------------",
        "Before any v43.5-style external test:",
        "1. freeze which cell population is used;",
        "2. freeze patient-level pseudobulk aggregation;",
        "3. verify the 256 frozen development coordinates in GSE236233;",
        "4. verify patient-level 12-month response mapping;",
        "5. preregister the exact external analysis before scoring outcomes.",
    ]

    if no_patient_id:
        lines += [
            "",
            "SAMPLES WITHOUT PARSABLE PATIENT ID",
            "-----------------------------------",
            *no_patient_id,
        ]

    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "version": VERSION,
        "candidate": GSE,
        "status": status,
        "geo_sample_records": len(samples),
        "detected_patient_n": n_patients,
        "patient_ids": patient_ids,
        "population_counts": dict(population_counts),
        "platform_counts": dict(platform_counts),
        "organism_counts": dict(organism_counts),
        "library_strategy_counts": dict(strategy_counts),
        "instrument_counts": dict(instrument_counts),
        "supplementary_file_references": len(supplementary),
        "external_predictive_performance_evaluated": False,
        "cell_level_independence_used": False,
        "statistical_unit": "patient",
        "sha256": {
            "geo_family_soft": sha256_file(soft_path),
            "execution_script": sha256_file(Path(__file__).resolve()),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(summary_path.read_text(encoding="utf-8"))
    print("Wrote:")
    for p in [sample_path, summary_path, manifest_path]:
        print(" ", p)


if __name__ == "__main__":
    main()
