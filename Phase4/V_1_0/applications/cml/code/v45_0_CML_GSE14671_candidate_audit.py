#!/usr/bin/env python3
"""
Soft Spaces / CML
v45.0 — GSE14671 outcome-blind candidate audit

Purpose
-------
Audit GSE14671 BEFORE any v45 model scoring.

This script automatically:
1. downloads the GEO family SOFT file if needed;
2. parses all SAMPLE records;
3. audits sample count, platform, source/tissue, treatment timing,
   and sample-specific response fields;
4. reports the exact response labels found in GEO metadata;
5. writes a sample-level TSV, summary, and manifest.

It DOES NOT:
- load the frozen v44 model;
- score any patient;
- select genes;
- tune any model;
- decide PASS/FAIL predictive performance.

This is a candidate-definition / metadata stage only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


VERSION = "v45.0"
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


def normalize_key(x: str) -> str:
    return re.sub(r"\s+", " ", x.strip().lower())


def parse_soft(path: Path):
    samples = []
    current = None

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")

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
                    "extract_protocol": [],
                }
                samples.append(current)
                continue

            if current is None or not line.startswith("!") or "=" not in line:
                continue

            key, value = line[1:].split("=", 1)
            key = key.strip()
            value = value.strip()

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
            elif key == "Sample_extract_protocol_ch1":
                current["extract_protocol"].append(value)

    return samples


def parse_characteristics(items):
    out = defaultdict(list)

    for item in items:
        if ":" in item:
            k, v = item.split(":", 1)
            out[normalize_key(k)].append(v.strip())
        else:
            out["_unkeyed"].append(item.strip())

    return dict(out)


def flatten_values(d):
    vals = []

    for key, arr in d.items():
        for x in arr:
            vals.append(f"{key}: {x}")

    return vals


def detect_response_from_characteristics(chars):
    """
    Only inspect SAMPLE-SPECIFIC characteristic values.

    Return:
      RESPONDER
      NON_RESPONDER
      UNKNOWN

    We intentionally do not use shared study description or protocol text.
    """
    response_like = []

    for key, vals in chars.items():
        kl = key.lower()

        if any(token in kl for token in (
            "response",
            "responder",
            "cytogenetic",
            "molecular",
            "outcome",
            "imatinib",
        )):
            response_like.extend(vals)

    # If no obviously named field, search all sample-specific characteristic values.
    if not response_like:
        response_like = [
            x
            for vals in chars.values()
            for x in vals
        ]

    text = " || ".join(response_like).lower()

    # More specific negative patterns first.
    negative_patterns = [
        r"\bnon[- ]?responder\b",
        r"\bnonresponse\b",
        r"\bnon[- ]?response\b",
        r"\bresistant\b",
        r"\bno\s+response\b",
        r"\bwithout\s+response\b",
        r"\bno\s+mcy?r\b",
    ]

    if any(re.search(p, text) for p in negative_patterns):
        return "NON_RESPONDER"

    positive_patterns = [
        r"\bresponder\b",
        r"\bresponse\b",
        r"\bsensitive\b",
        r"\bmcy?r\b",
        r"\bmajor cytogenetic response\b",
    ]

    if any(re.search(p, text) for p in positive_patterns):
        return "RESPONDER"

    return "UNKNOWN"


def detect_pretreatment(chars, source_name, title):
    """
    Conservative, sample-specific timing audit.

    Return:
      PRETREATMENT
      POST_TREATMENT
      UNKNOWN
    """
    items = []

    for key, vals in chars.items():
        kl = key.lower()

        if any(token in kl for token in (
            "time",
            "treatment",
            "therapy",
            "sample",
            "status",
        )):
            items.extend(vals)

    items.extend([source_name, title])

    text = " || ".join(items).lower()

    post_patterns = [
        r"\bpost[- ]?treat",
        r"\bafter\s+(?:imatinib|treatment|therapy)",
        r"\bon\s+imatinib\b",
        r"\bduring\s+treatment\b",
    ]

    if any(re.search(p, text) for p in post_patterns):
        return "POST_TREATMENT"

    pre_patterns = [
        r"\bpretreat",
        r"\bpre[- ]?treat",
        r"\bbefore\s+(?:imatinib|treatment|therapy)",
        r"\bprior\s+to\s+(?:imatinib|treatment|therapy)",
        r"\bat\s+diagnosis\b",
        r"\bdiagnostic\b",
        r"\bbaseline\b",
    ]

    if any(re.search(p, text) for p in pre_patterns):
        return "PRETREATMENT"

    return "UNKNOWN"


def main():
    project = project_dir()

    data_dir = (
        project
        / "data"
        / "external"
        / GSE
        / "metadata"
    )

    results = (
        project
        / "results"
        / "direct_subspace"
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    soft_path = data_dir / f"{GSE}_family.soft.gz"
    download_if_missing(SOFT_URL, soft_path)

    samples = parse_soft(soft_path)

    if not samples:
        raise RuntimeError("No SAMPLE records parsed from GSE14671 SOFT.")

    rows = []

    characteristic_keys = Counter()
    characteristic_values = defaultdict(Counter)

    for rec in samples:
        chars = parse_characteristics(rec["characteristics"])

        for key, vals in chars.items():
            characteristic_keys[key] += 1
            for val in vals:
                characteristic_values[key][val] += 1

        response = detect_response_from_characteristics(chars)

        timing = detect_pretreatment(
            chars,
            rec["source_name"],
            rec["title"],
        )

        rows.append({
            "gsm": rec["gsm"],
            "title": rec["title"],
            "source_name": rec["source_name"],
            "platform_id": rec["platform_id"],
            "timing_class": timing,
            "response_class_candidate": response,
            "sample_characteristics": " || ".join(flatten_values(chars)),
        })

    platform_counts = Counter(r["platform_id"] for r in rows)
    source_counts = Counter(r["source_name"] for r in rows)
    timing_counts = Counter(r["timing_class"] for r in rows)
    response_counts = Counter(r["response_class_candidate"] for r in rows)

    # Candidate eligibility here is deliberately technical only.
    # We require:
    # - at least 30 samples
    # - one dominant platform
    # - at least two response classes recognized OR metadata sufficient for
    #   manual/follow-up label audit.
    dominant_platform, dominant_n = platform_counts.most_common(1)[0]

    technical_candidate = (
        len(rows) >= 30
        and dominant_n == len(rows)
    )

    if (
        response_counts.get("RESPONDER", 0) > 0
        and response_counts.get("NON_RESPONDER", 0) > 0
    ):
        label_status = "RESPONSE CLASSES DETECTED"
    else:
        label_status = "RESPONSE LABEL FOLLOW-UP REQUIRED"

    sample_tsv = (
        results
        / "v45_0_GSE14671_candidate_samples.tsv"
    )

    with sample_tsv.open("w", encoding="utf-8", newline="") as f:
        columns = [
            "gsm",
            "title",
            "source_name",
            "platform_id",
            "timing_class",
            "response_class_candidate",
            "sample_characteristics",
        ]

        f.write("\t".join(columns) + "\n")

        for row in rows:
            f.write(
                "\t".join(
                    str(row[c]).replace("\t", " ")
                    for c in columns
                )
                + "\n"
            )

    characteristic_path = (
        results
        / "v45_0_GSE14671_characteristic_inventory.txt"
    )

    inv = [
        "=== GSE14671 SAMPLE-SPECIFIC CHARACTERISTIC INVENTORY ===",
        "",
    ]

    for key in sorted(characteristic_keys):
        inv.append(
            f"[{key}] present in {characteristic_keys[key]}/{len(rows)} samples"
        )

        for val, count in characteristic_values[key].most_common():
            inv.append(f"  {count:3d}  {val}")

        inv.append("")

    characteristic_path.write_text(
        "\n".join(inv) + "\n",
        encoding="utf-8",
    )

    summary_path = (
        results
        / "v45_0_GSE14671_candidate_audit_summary.txt"
    )

    summary = [
        "=== Soft Spaces / CML v45.0 GSE14671 CANDIDATE AUDIT ===",
        "",
        "AUDIT TYPE",
        "----------",
        "Metadata / candidate audit only",
        "Frozen v44 model loaded: NO",
        "Predictive scoring performed: NO",
        "External outcome tuning performed: NO",
        "",
        "DATASET",
        "-------",
        f"GEO accession: {GSE}",
        f"GEO SAMPLE records: {len(rows)}",
        "",
        "PLATFORM",
        "--------",
    ]

    for platform, count in platform_counts.most_common():
        summary.append(f"{platform or '<missing>'}: {count}")

    summary += [
        "",
        "SOURCE NAME",
        "-----------",
    ]

    for source, count in source_counts.most_common():
        summary.append(f"{source or '<missing>'}: {count}")

    summary += [
        "",
        "SAMPLE TIMING AUDIT",
        "-------------------",
    ]

    for label in ["PRETREATMENT", "POST_TREATMENT", "UNKNOWN"]:
        summary.append(
            f"{label}: {timing_counts.get(label, 0)}"
        )

    summary += [
        "",
        "RESPONSE LABEL AUDIT",
        "--------------------",
    ]

    for label in ["RESPONDER", "NON_RESPONDER", "UNKNOWN"]:
        summary.append(
            f"{label}: {response_counts.get(label, 0)}"
        )

    summary += [
        "",
        f"Label status: {label_status}",
        "",
        "TECHNICAL CANDIDACY",
        "-------------------",
        f"Single-platform >=30-sample candidate: {'YES' if technical_candidate else 'NO'}",
        "",
        "IMPORTANT",
        "---------",
        "No assumption is made here about the final v45 response definition.",
        "The exact GEO sample-specific characteristics must determine the",
        "outcome mapping before a predictive protocol is frozen.",
        "",
        "NEXT",
        "----",
        "If metadata confirms a usable pretreatment responder/non-responder",
        "cohort, freeze v45.1 cross-endpoint external protocol BEFORE",
        "applying the already-frozen v44 model.",
    ]

    summary_path.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    manifest_path = (
        results
        / "v45_0_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            {
                "version": VERSION,
                "dataset": GSE,
                "sample_n": len(rows),
                "platform_counts": dict(platform_counts),
                "source_counts": dict(source_counts),
                "timing_counts": dict(timing_counts),
                "response_candidate_counts": dict(response_counts),
                "technical_candidate": technical_candidate,
                "label_status": label_status,
                "predictive_scoring_performed": False,
                "external_outcome_tuning_performed": False,
                "sha256": {
                    "family_soft": sha256_file(soft_path),
                    "sample_tsv": sha256_file(sample_tsv),
                    "characteristic_inventory": sha256_file(characteristic_path),
                    "execution_script": sha256_file(Path(__file__).resolve()),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(summary_path.read_text(encoding="utf-8"))

    print("Wrote:")
    for p in [
        sample_tsv,
        characteristic_path,
        summary_path,
        manifest_path,
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
