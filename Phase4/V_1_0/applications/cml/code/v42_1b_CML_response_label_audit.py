#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "v42.1b"
ACCESSIONS = ["GSE130404", "GSE44589"]

def project_dir_from_script() -> Path:
    return Path(__file__).resolve().parent.parent

def parse_tab_line(line: str):
    return [x.strip().strip('"') for x in line.rstrip("\r\n").split("\t")]

def parse_sample_metadata(series_path: Path):
    rows = defaultdict(list)
    with gzip.open(series_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_"):
                parts = parse_tab_line(line)
                rows[parts[0]].append(parts[1:])

    gsm_rows = rows.get("!Sample_geo_accession", [])
    if not gsm_rows:
        raise RuntimeError(f"No !Sample_geo_accession in {series_path}")

    sample_ids = gsm_rows[0]
    n = len(sample_ids)
    samples = [{"geo_accession": gsm} for gsm in sample_ids]

    for key, occurrences in rows.items():
        for occ_idx, vals in enumerate(occurrences, 1):
            if len(vals) != n:
                continue
            field = key if len(occurrences) == 1 else f"{key}__{occ_idx}"
            for i, v in enumerate(vals):
                samples[i][field] = v

    for s in samples:
        chars = {}
        for k, v in s.items():
            if k.startswith("!Sample_characteristics_ch1") and ":" in v:
                name, value = v.split(":", 1)
                chars[name.strip().lower()] = value.strip()
        s["_characteristics"] = chars

    return samples

def find_char_key(samples, candidates):
    keys = Counter()
    for s in samples:
        keys.update(s["_characteristics"].keys())

    for c in candidates:
        c = c.lower()
        if c in keys:
            return c
    for c in candidates:
        c = c.lower()
        hits = sorted(k for k in keys if c in k)
        if hits:
            return hits[0]
    return None

def classify_timepoint(sample):
    parts = []
    for k, v in sample.items():
        if k.startswith("!Sample_") and isinstance(v, str):
            parts.append(v.lower())
    parts += [str(v).lower() for v in sample["_characteristics"].values()]
    blob = " || ".join(parts)

    if "prior to treatment" in blob or "pretreatment" in blob or "pre-treatment" in blob:
        return "BASELINE_PRETREATMENT"
    if "after 6 weeks" in blob or "6 weeks of treatment" in blob:
        return "POST_6W"
    if "diagnostic chronic phase" in blob or "at diagnosis" in blob:
        return "BASELINE_DIAGNOSIS"
    if "treatment: imatinib" in blob:
        return "ON_TREATMENT"
    return "UNRESOLVED"

def audit_accession(accession, project):
    series_path = (
        project / "data" / "external" / accession / "raw"
        / f"{accession}_series_matrix.txt.gz"
    )
    if not series_path.exists():
        raise FileNotFoundError(series_path)

    samples = parse_sample_metadata(series_path)
    response_key = find_char_key(
        samples,
        ["molecular response", "response", "bcr-abl1 at 3 month", "bcr-abl at 3 month"],
    )
    stage_key = find_char_key(samples, ["disease stage", "stage"])

    response_counts = Counter()
    stage_counts = Counter()
    timepoint_counts = Counter()
    cross = Counter()
    rows = []

    for s in samples:
        chars = s["_characteristics"]
        response = chars.get(response_key, "") if response_key else ""
        stage = chars.get(stage_key, "") if stage_key else ""
        tp = classify_timepoint(s)

        if response:
            response_counts[response] += 1
        if stage:
            stage_counts[stage] += 1
        timepoint_counts[tp] += 1
        cross[(tp, response or "<MISSING>")] += 1

        rows.append({
            "geo_accession": s["geo_accession"],
            "timepoint_class": tp,
            "response_key": response_key or "",
            "response_value": response,
            "stage_key": stage_key or "",
            "stage_value": stage,
        })

    return {
        "accession": accession,
        "n_samples": len(samples),
        "response_key": response_key,
        "stage_key": stage_key,
        "response_counts": dict(response_counts),
        "stage_counts": dict(stage_counts),
        "timepoint_counts": dict(timepoint_counts),
        "timepoint_by_response": [
            {"timepoint": tp, "response": resp, "count": n}
            for (tp, resp), n in sorted(cross.items())
        ],
        "sample_rows": rows,
    }

def write_csv(rows, path):
    import csv
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def report(audits):
    lines = [
        f"=== Soft Spaces / CML {VERSION} RESPONSE-LABEL AUDIT ===",
        "",
        "NO EXPRESSION MODELING WAS PERFORMED.",
        "",
    ]
    for a in audits:
        lines += [
            a["accession"],
            "-" * len(a["accession"]),
            f"Samples: {a['n_samples']}",
            f"Detected response key: {a['response_key']}",
            f"Detected stage key: {a['stage_key']}",
            "",
            "Response counts:",
        ]
        for k, v in sorted(a["response_counts"].items()):
            lines.append(f"  {k}: {v}")
        if not a["response_counts"]:
            lines.append("  NONE")

        lines += ["", "Timepoint counts:"]
        for k, v in sorted(a["timepoint_counts"].items()):
            lines.append(f"  {k}: {v}")

        lines += ["", "Timepoint x response:"]
        for r in a["timepoint_by_response"]:
            lines.append(f"  {r['timepoint']} | {r['response']} | n={r['count']}")

        if a["stage_counts"]:
            lines += ["", "Stage counts:"]
            for k, v in sorted(a["stage_counts"].items()):
                lines.append(f"  {k}: {v}")
        lines.append("")

    lines += [
        "DECISION CHECKLIST",
        "------------------",
        "- baseline subset unambiguous",
        "- response endpoint unambiguous",
        "- class counts adequate",
        "- response measured after baseline expression",
        "- post-treatment repeats excluded from baseline prediction",
        "- no external outcome information used for feature tuning",
    ]
    return "\n".join(lines) + "\n"

def main():
    project = project_dir_from_script()
    outdir = project / "results" / "dataset_audit"
    outdir.mkdir(parents=True, exist_ok=True)

    audits = [audit_accession(acc, project) for acc in ACCESSIONS]

    for a in audits:
        write_csv(
            a["sample_rows"],
            outdir / f"{VERSION}_{a['accession']}_sample_metadata_audit.csv"
        )

    txt = outdir / "v42_1b_CML_response_label_audit.txt"
    js = outdir / "v42_1b_CML_response_label_audit.json"

    txt.write_text(report(audits), encoding="utf-8")
    js.write_text(json.dumps({
        "version": VERSION,
        "expression_values_used": False,
        "modeling_performed": False,
        "audits": audits,
    }, indent=2), encoding="utf-8")

    print(txt.read_text(encoding="utf-8"))
    print("Wrote:")
    print(" ", txt)
    print(" ", js)

if __name__ == "__main__":
    main()
