#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

VERSION = "v41.17"

def project_dir_from_script() -> Path:
    return Path(__file__).resolve().parent.parent

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

def extract_float(text: str, pattern: str):
    m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return float(m.group(1)) if m else None

def extract_word(text: str, pattern: str):
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip() if m else None

def rec(path: Path):
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}

def main():
    project = project_dir_from_script()
    results = project / "results" / "stability"
    outdir = project / "docs" / "closure"
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "v41_12b": results / "v41_12b_feature_stability_summary.txt",
        "v41_13": results / "v41_13_module_subspace_stability_summary.txt",
        "v41_14": results / "v41_14_external_module_transfer_summary.txt",
        "v41_15_mapping": results / "v41_15_mapping_audit.csv",
        "v41_15x": results / "v41_15x_exploratory_summary.txt",
        "v41_16_mapping": results / "v41_16_mapping_audit.csv",
        "v41_16": results / "v41_16_external_replication_summary.txt",
    }

    for p in paths.values():
        require(p)

    t12 = read_text(paths["v41_12b"])
    t13 = read_text(paths["v41_13"])
    t14 = read_text(paths["v41_14"])
    t15x = read_text(paths["v41_15x"])
    t16 = read_text(paths["v41_16"])

    values = {
        "v41_12b_real_jaccard": extract_float(t12, r"REAL mean per-panel Jaccard:\s*([0-9.]+)"),
        "v41_12b_null_jaccard": extract_float(t12, r"NULL mean per-panel Jaccard:\s*([0-9.]+)"),
        "v41_12b_mean_frozen_overlap": extract_float(t12, r"Mean overlap fraction:\s*([0-9.]+)"),
        "v41_13_subspace_real": extract_float(t13, r"FROZEN-SUBSPACE CAPTURE.*?REAL mean:\s*([0-9.]+)"),
        "v41_13_subspace_null": extract_float(t13, r"FROZEN-SUBSPACE CAPTURE.*?NULL mean:\s*([0-9.]+)"),
        "v41_13_corr_real": extract_float(t13, r"CORRELATION-NEIGHBORHOOD CAPTURE.*?REAL mean:\s*([0-9.]+)"),
        "v41_13_corr_null": extract_float(t13, r"CORRELATION-NEIGHBORHOOD CAPTURE.*?NULL mean:\s*([0-9.]+)"),
        "v41_14_GSE87371_corr_p": extract_float(t14, r"PRIMARY: GSE10846 -> GSE87371.*?Pairwise correlation-structure similarity.*?empirical p:\s*([0-9.eE+-]+)"),
        "v41_14_GSE87371_top4_p": extract_float(t14, r"PRIMARY: GSE10846 -> GSE87371.*?Top-4 gene-eigensubspace overlap.*?empirical p:\s*([0-9.eE+-]+)"),
        "v41_15x_top4_real": extract_float(t15x, r"PRIMARY EXPLORATORY: Top-4.*?REAL:\s*([0-9.]+)"),
        "v41_15x_top4_null": extract_float(t15x, r"PRIMARY EXPLORATORY: Top-4.*?NULL mean:\s*([0-9.]+)"),
        "v41_15x_top4_p": extract_float(t15x, r"PRIMARY EXPLORATORY: Top-4.*?empirical p:\s*([0-9.eE+-]+)"),
        "v41_16_top4_real": extract_float(t16, r"PRIMARY CONFIRMATORY: Top-4.*?REAL:\s*([0-9.]+)"),
        "v41_16_top4_null": extract_float(t16, r"PRIMARY CONFIRMATORY: Top-4.*?NULL mean:\s*([0-9.]+)"),
        "v41_16_top4_p": extract_float(t16, r"PRIMARY CONFIRMATORY: Top-4.*?empirical p:\s*([0-9.eE+-]+)"),
        "v41_16_status": extract_word(t16, r"v41\.16 CONFIRMATORY STATUS:\s*(\S+)"),
    }

    if values["v41_16_status"] != "FAIL":
        raise RuntimeError(f"Expected v41.16 FAIL, got {values['v41_16_status']!r}")

    lines = [
        f"=== Soft Spaces / DLBCL {VERSION} CLOSURE / EVIDENCE AUDIT ===",
        "",
        "PURPOSE",
        "-------",
        "Freeze the scientific status of the DLBCL application track after v41.16.",
        "No new model fitting or hypothesis testing is performed here.",
        "",
        "A. SUPPORTED FINDINGS",
        "--------------------",
        "v41.12b: REAL feature panels recur more consistently than matched NULL,",
        "but exact 16-gene identity is not a stable fixed biomarker panel.",
        "",
        "v41.13: within GSE10846, REAL panels preserve more frozen-subspace and",
        "correlation-neighborhood structure than matched NULL panels.",
        "",
        "B. EXPLORATORY / PARTIAL FINDINGS",
        "---------------------------------",
        "v41.14: external geometry transfer is mixed/partial. GSE87371 shows",
        "REAL-favorable direction on some metrics but no clean confirmatory result.",
        "",
        "v41.15x: the reduced 12-gene panel gives a strong exploratory top-4 signal",
        "in GSE117556, but this remains exploratory only.",
        "",
        "C. TECHNICALLY NON-EVALUABLE",
        "----------------------------",
        "v41.15: preregistered 16-gene replication was technically non-evaluable",
        "because 4/16 frozen coordinates were not reproducibly measurable.",
        "",
        "D. FAILED CONFIRMATORY CLAIM",
        "----------------------------",
        "v41.16: preregistered 12-gene GPL570 replication FAILED its frozen primary",
        "decision rule. REAL top-4 overlap exceeded NULL mean, but empirical",
        "p remained above 0.05.",
        "",
        "E. KEY NUMERICAL RECORD",
        "-----------------------",
        f"v41.12b REAL mean Jaccard: {values['v41_12b_real_jaccard']}",
        f"v41.12b NULL mean Jaccard: {values['v41_12b_null_jaccard']}",
        f"v41.12b mean frozen overlap fraction: {values['v41_12b_mean_frozen_overlap']}",
        "",
        f"v41.13 frozen-subspace REAL mean: {values['v41_13_subspace_real']}",
        f"v41.13 frozen-subspace NULL mean: {values['v41_13_subspace_null']}",
        f"v41.13 correlation-neighborhood REAL mean: {values['v41_13_corr_real']}",
        f"v41.13 correlation-neighborhood NULL mean: {values['v41_13_corr_null']}",
        "",
        f"v41.14 GSE87371 correlation-structure empirical p: {values['v41_14_GSE87371_corr_p']}",
        f"v41.14 GSE87371 top-4 eigensubspace empirical p: {values['v41_14_GSE87371_top4_p']}",
        "",
        f"v41.15x exploratory top-4 REAL: {values['v41_15x_top4_real']}",
        f"v41.15x exploratory top-4 NULL mean: {values['v41_15x_top4_null']}",
        f"v41.15x exploratory top-4 empirical p: {values['v41_15x_top4_p']}",
        "",
        f"v41.16 confirmatory top-4 REAL: {values['v41_16_top4_real']}",
        f"v41.16 confirmatory top-4 NULL mean: {values['v41_16_top4_null']}",
        f"v41.16 confirmatory top-4 empirical p: {values['v41_16_top4_p']}",
        f"v41.16 confirmatory status: {values['v41_16_status']}",
        "",
        "F. CLAIMS NOT ESTABLISHED",
        "------------------------",
        "- A universally stable fixed 16-gene DLBCL biomarker panel.",
        "- Confirmed external replication of the 12-gene/top-4 geometry claim.",
        "- Causal biological roles for the selected genes.",
        "- Clinical diagnostic or prognostic utility.",
        "- Superiority over state-of-the-art classical bioinformatics methods.",
        "- Quantum advantage.",
        "",
        "G. SCIENTIFIC CLOSURE",
        "--------------------",
        "The DLBCL track shows non-random internal structural signal and several",
        "REAL-over-NULL effects, including exploratory external geometry transfer.",
        "",
        "However, the prospectively frozen v41.16 replication did not satisfy",
        "its confirmatory p <= 0.05 rule. Therefore the specific 12-gene/top-4",
        "external module-transfer hypothesis is NOT confirmed.",
        "",
        "CLOSURE STATUS:",
        "PROMISING STRUCTURAL SIGNAL; NO CONFIRMED EXTERNAL MODULE-TRANSFER CLAIM.",
        "",
        "Any future DLBCL work should start from a genuinely new protocol or",
        "hypothesis, not by post-hoc retuning of v41.16.",
        "",
        "H. ARTIFACT INTEGRITY",
        "--------------------",
    ]

    artifacts = {}
    for key, p in paths.items():
        artifacts[key] = rec(p)
        lines.append(f"{key}:")
        lines.append(f"  path: {p}")
        lines.append(f"  sha256: {artifacts[key]['sha256']}")

    report_path = outdir / "DLBCL_v41_17_CLOSURE_AUDIT.txt"
    manifest_path = outdir / "DLBCL_v41_17_CLOSURE_AUDIT_manifest.json"

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "version": VERSION,
        "purpose": "DLBCL closure/evidence audit",
        "new_hypothesis_testing": False,
        "closure_status": "PROMISING STRUCTURAL SIGNAL; NO CONFIRMED EXTERNAL MODULE-TRANSFER CLAIM",
        "parsed_key_results": values,
        "artifacts": artifacts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"=== {VERSION} DLBCL CLOSURE AUDIT ===")
    print("v41.16 status:", values["v41_16_status"])
    print("v41.16 primary p:", values["v41_16_top4_p"])
    print("Wrote:")
    print(" ", report_path)
    print(" ", manifest_path)

if __name__ == "__main__":
    main()
