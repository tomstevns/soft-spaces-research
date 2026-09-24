#!/usr/bin/env python3
"""
Soft Spaces / CML
v43.4c1 — Corrected external candidate audit for GSE236233

Correction:
v43.4c failed to parse patient IDs because Python regex word-boundary
semantics treat underscore as a word character in labels such as CML1_34p.
v43.4c1 changes only the parser from \\b(CML\\d+)\\b to (CML\\d+).

No scientific criterion or dataset-selection rule is changed.
No external predictive performance is calculated.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "v43.4c1"
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
    samples = []
    current = None

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if line.startswith("^SAMPLE"):
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
                current = rec
                continue

            if current is None or not line.startswith("!") or "=" not in line:
                continue

            k, v = line[1:].split("=", 1)
            k = k.strip()
            v = v.strip()

            if k == "Sample_title":
                current["title"] = v
            elif k == "Sample_source_name_ch1":
                current["source"] = v
            elif k == "Sample_organism_ch1":
                current["organism"] = v
            elif k == "Sample_characteristics_ch1":
                current["characteristics"].append(v)
            elif k.startswith("Sample_supplementary_file"):
                current["supplementary"].append(v)
            elif k == "Sample_platform_id":
                current["platform"] = v
            elif k == "Sample_library_strategy":
                current["library_strategy"] = v
            elif k == "Sample_instrument_model":
                current["instrument"] = v

    return samples


def patient_id_from_text(text: str):
    # Corrected: underscore after CML1 is a word char, so \b was wrong.
    m = re.search(r"(CML\d+)", text, flags=re.I)
    return m.group(1).upper() if m else None


def classify_population(text: str):
    t = text.lower().replace(" ", "")
    if "cd34+cd38-" in t or "38n" in t:
        return "Lin-CD34+CD38-"
    if "cd34+" in t or "34p" in t:
        return "Lin-CD34+"
    return "OTHER/UNKNOWN"


def main():
    project = project_dir()
    rawdir = project / "data" / "external" / GSE / "metadata"
    outdir = project / "results" / "direct_subspace"
    rawdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    soft_path = rawdir / f"{GSE}_family.soft.gz"
    download(SOFT_URL, soft_path)

    samples = parse_soft(soft_path)

    patient_to_samples = defaultdict(list)
    population_counts = Counter()
    platform_counts = Counter()
    organism_counts = Counter()
    strategy_counts = Counter()
    instrument_counts = Counter()
    no_patient_id = []

    for rec in samples:
        combined = " ".join(
            [rec["title"], rec["source"], *rec["characteristics"]]
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

    patient_ids = sorted(
        patient_to_samples,
        key=lambda x: int(re.search(r"\d+", x).group())
    )
    n_patients = len(patient_ids)

    patient_population_table = {}
    for pid in patient_ids:
        patient_population_table[pid] = sorted(
            {rec["population"] for rec in patient_to_samples[pid]}
        )

    supplementary_n = sum(len(rec["supplementary"]) for rec in samples)

    all_human = (
        len(organism_counts) == 1
        and next(iter(organism_counts), "") == "Homo sapiens"
    )
    patient_level_identifiable = (
        n_patients == 9
        and len(no_patient_id) == 0
    )
    has_processed = supplementary_n > 0

    status = (
        "CANDIDATE ELIGIBLE FOR TECHNICAL FOLLOW-UP"
        if all_human and patient_level_identifiable and has_processed
        else "CANDIDATE NOT YET ELIGIBLE"
    )

    sample_path = outdir / "v43_4c1_GSE236233_samples.tsv"
    summary_path = outdir / "v43_4c1_GSE236233_candidate_audit_summary.txt"
    manifest_path = outdir / "v43_4c1_GSE236233_candidate_audit_manifest.json"

    with sample_path.open("w", encoding="utf-8") as f:
        f.write(
            "gsm\tpatient_id\tpopulation\ttitle\tplatform\t"
            "library_strategy\tinstrument\tsupplementary_files\n"
        )
        for rec in samples:
            f.write(
                "\t".join([
                    rec["gsm"],
                    rec["patient_id"] or "",
                    rec["population"],
                    rec["title"].replace("\t", " "),
                    rec["platform"],
                    rec["library_strategy"],
                    rec["instrument"],
                    "|".join(rec["supplementary"]),
                ]) + "\n"
            )

    lines = [
        "=== Soft Spaces / CML v43.4c1 GSE236233 CORRECTED CANDIDATE AUDIT ===",
        "",
        "CORRECTION",
        "----------",
        "v43.4c failed to parse patient IDs because underscore is a Python",
        "word character in labels such as CML1_34p. The parser is corrected",
        "to extract CML<number> without a trailing word-boundary requirement.",
        "No scientific rule is changed.",
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
        "PATIENT / POPULATION COVERAGE",
        "-----------------------------",
    ]
    for pid in patient_ids:
        lines.append(f"{pid}: {', '.join(patient_population_table[pid])}")

    lines += [
        "",
        "PLATFORM / TECHNOLOGY",
        "---------------------",
    ]
    for k, v in sorted(platform_counts.items()):
        lines.append(f"Platform {k or 'UNKNOWN'}: {v}")
    for k, v in sorted(strategy_counts.items()):
        lines.append(f"Library strategy {k or 'UNKNOWN'}: {v}")
    for k, v in sorted(instrument_counts.items()):
        lines.append(f"Instrument {k or 'UNKNOWN'}: {v}")

    lines += [
        "",
        "PROCESSED DATA",
        "--------------",
        f"Supplementary processed-file references: {supplementary_n}",
        "",
        "CRITICAL STATISTICAL RULE",
        "-------------------------",
        "The patient is the statistical unit.",
        "Cells must not be treated as independent response observations.",
        "",
        "OUTCOME STATUS",
        "--------------",
        "External predictive performance evaluated: NO",
        "Classifier fitted to GSE236233: NO",
        "",
        f"v43.4c1 STATUS: {status}",
        "",
        "NEXT",
        "----",
        "Proceed only to an outcome-blind pseudobulk/gene-coverage transport audit.",
        "Freeze one cell population and patient-level aggregation before scoring",
        "any external response outcome.",
    ]

    if no_patient_id:
        lines += [
            "",
            "SAMPLES WITHOUT PARSABLE PATIENT ID",
            "-----------------------------------",
            *no_patient_id,
        ]

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "version": VERSION,
        "candidate": GSE,
        "status": status,
        "geo_sample_records": len(samples),
        "detected_patient_n": n_patients,
        "patient_ids": patient_ids,
        "patient_population_table": patient_population_table,
        "population_counts": dict(population_counts),
        "platform_counts": dict(platform_counts),
        "organism_counts": dict(organism_counts),
        "library_strategy_counts": dict(strategy_counts),
        "instrument_counts": dict(instrument_counts),
        "supplementary_file_references": supplementary_n,
        "external_predictive_performance_evaluated": False,
        "statistical_unit": "patient",
        "correction_from_v43_4c": (
            r"patient regex changed from \b(CML\d+)\b to (CML\d+)"
        ),
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
    for p in (sample_path, summary_path, manifest_path):
        print(" ", p)


if __name__ == "__main__":
    main()
