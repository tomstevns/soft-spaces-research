#!/usr/bin/env python3
"""
Soft Spaces / CML
v42.1c — Corrected sample-level timepoint / response audit

Corrects v42.1b timepoint classification for GSE44589.

IMPORTANT
---------
No expression modeling.
No Soft-Spaces scoring.
No classifier training.

The correction is methodological:
v42.1b searched broad metadata text, including global protocol descriptions.
Those protocol strings contain "pretreatment" for every sample and therefore
can falsely classify post-treatment samples as baseline.

v42.1c uses ONLY sample-specific fields for timepoint assignment:
- !Sample_source_name_ch1
- characteristic "treatment"

Response labels are taken only from sample-specific characteristics.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "v42.1c"
ACCESSIONS = ["GSE130404", "GSE44589"]


def project_dir_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_tab_line(line: str):
    return [x.strip().strip('"') for x in line.rstrip("\r\n").split("\t")]


def parse_sample_metadata(path: Path):
    rows = defaultdict(list)

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_"):
                parts = parse_tab_line(line)
                rows[parts[0]].append(parts[1:])

    gsm_rows = rows.get("!Sample_geo_accession", [])
    if not gsm_rows:
        raise RuntimeError(f"No sample IDs in {path}")

    ids = gsm_rows[0]
    n = len(ids)
    samples = [{"geo_accession": x} for x in ids]

    for key, occurrences in rows.items():
        for occ, vals in enumerate(occurrences, 1):
            if len(vals) != n:
                continue
            field = key if len(occurrences) == 1 else f"{key}__{occ}"
            for i, v in enumerate(vals):
                samples[i][field] = v

    for s in samples:
        chars = {}
        for k, v in s.items():
            if k.startswith("!Sample_characteristics_ch1") and ":" in v:
                name, val = v.split(":", 1)
                chars[name.strip().lower()] = val.strip()
        s["_characteristics"] = chars

    return samples


def find_char_key(samples, candidates):
    keys = Counter()
    for s in samples:
        keys.update(s["_characteristics"].keys())

    for candidate in candidates:
        c = candidate.lower()
        if c in keys:
            return c

    for candidate in candidates:
        c = candidate.lower()
        hits = sorted(k for k in keys if c in k)
        if hits:
            return hits[0]

    return None


def classify_timepoint_sample_specific(sample):
    source = sample.get("!Sample_source_name_ch1", "").strip().lower()
    treatment = sample["_characteristics"].get("treatment", "").strip().lower()

    # Highest-specificity rules first.
    if "after 6 weeks" in source or "6 weeks of treatment" in source:
        return "POST_6W"

    if treatment == "imatinib":
        return "POST_6W_OR_ON_TREATMENT"

    if "prior to treatment" in source:
        return "BASELINE_PRETREATMENT"

    if treatment == "prior to treatment":
        return "BASELINE_PRETREATMENT"

    # GSE130404 baseline at diagnosis.
    stage = sample["_characteristics"].get("disease stage", "").lower()
    if "diagnostic chronic phase" in stage:
        return "BASELINE_DIAGNOSIS"

    return "UNRESOLVED"


def audit(accession: str, project: Path):
    path = (
        project / "data" / "external" / accession / "raw"
        / f"{accession}_series_matrix.txt.gz"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    samples = parse_sample_metadata(path)

    response_key = find_char_key(
        samples,
        ["bcr-abl1 at 3 month", "molecular response", "response"],
    )

    time_counts = Counter()
    response_counts = Counter()
    cross = Counter()
    rows = []

    for s in samples:
        tp = classify_timepoint_sample_specific(s)
        response = (
            s["_characteristics"].get(response_key, "")
            if response_key else ""
        )

        time_counts[tp] += 1
        if response:
            response_counts[response] += 1
        cross[(tp, response or "<MISSING>")] += 1

        rows.append({
            "geo_accession": s["geo_accession"],
            "source_name": s.get("!Sample_source_name_ch1", ""),
            "treatment": s["_characteristics"].get("treatment", ""),
            "timepoint_class": tp,
            "response_key": response_key or "",
            "response_value": response,
        })

    return {
        "accession": accession,
        "n_samples": len(samples),
        "response_key": response_key,
        "timepoint_counts": dict(time_counts),
        "response_counts": dict(response_counts),
        "timepoint_by_response": [
            {"timepoint": tp, "response": resp, "count": n}
            for (tp, resp), n in sorted(cross.items())
        ],
        "rows": rows,
    }


def write_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_report(audits):
    lines = [
        f"=== Soft Spaces / CML {VERSION} CORRECTED TIMEPOINT AUDIT ===",
        "",
        "NO EXPRESSION MODELING WAS PERFORMED.",
        "",
        "Correction:",
        "Timepoint classification uses only sample-specific source_name and",
        "sample-specific treatment characteristics. Global protocol strings",
        "are deliberately ignored.",
        "",
    ]

    for a in audits:
        lines += [
            a["accession"],
            "-" * len(a["accession"]),
            f"Samples: {a['n_samples']}",
            f"Response key: {a['response_key']}",
            "",
            "Timepoint counts:",
        ]
        for k, v in sorted(a["timepoint_counts"].items()):
            lines.append(f"  {k}: {v}")

        lines += ["", "Response counts:"]
        for k, v in sorted(a["response_counts"].items()):
            lines.append(f"  {k}: {v}")

        lines += ["", "Timepoint x response:"]
        for r in a["timepoint_by_response"]:
            lines.append(
                f"  {r['timepoint']} | {r['response']} | n={r['count']}"
            )
        lines.append("")

    lines += [
        "INTERPRETATION RULE",
        "-------------------",
        "For baseline-response prediction, ONLY BASELINE_DIAGNOSIS or",
        "BASELINE_PRETREATMENT samples may enter the development matrix.",
        "",
        "Post-treatment samples must not enter a baseline predictor.",
        "",
        "The response endpoint must be measured later than the baseline",
        "expression measurement.",
    ]

    return "\n".join(lines) + "\n"


def main():
    project = project_dir_from_script()
    outdir = project / "results" / "dataset_audit"
    outdir.mkdir(parents=True, exist_ok=True)

    audits = [audit(acc, project) for acc in ACCESSIONS]

    for a in audits:
        write_csv(
            a["rows"],
            outdir / f"{VERSION}_{a['accession']}_sample_metadata_audit.csv",
        )

    txt = outdir / "v42_1c_CML_corrected_timepoint_audit.txt"
    js = outdir / "v42_1c_CML_corrected_timepoint_audit.json"

    txt.write_text(make_report(audits), encoding="utf-8")
    js.write_text(
        json.dumps({
            "version": VERSION,
            "expression_values_used": False,
            "modeling_performed": False,
            "correction_of": "v42.1b timepoint classification",
            "audits": audits,
        }, indent=2),
        encoding="utf-8",
    )

    print(txt.read_text(encoding="utf-8"))
    print("Wrote:")
    print(" ", txt)
    print(" ", js)


if __name__ == "__main__":
    main()
