#!/usr/bin/env python3
"""
Soft Spaces / CML
v43.4a — Technical platform/subspace transport audit

Purpose
-------
Determine whether the frozen v43 direct-subspace representation can be
transported from development cohort GSE130404 to external cohort GSE44589
WITHOUT evaluating external response outcomes.

This script:
- verifies the frozen v43.4 preregistration;
- reconstructs the final development Top-256 support using ALL 96 GSE130404
  baseline samples and raw variance only;
- fits the final development StandardScaler on GSE130404 only;
- constructs the final development H and frozen-r=16 basis U16 using
  GSE130404 labels only;
- audits gene measurability on GPL570;
- quantifies how much of the frozen projector is retained on externally
  measurable coordinates;
- DOES NOT parse or evaluate GSE44589 response labels;
- DOES NOT train or score an external classifier.

Outcome
-------
TECHNICALLY EVALUABLE only if all required frozen coordinates are measurable
on GPL570.

If one or more frozen Top-256 coordinates are unavailable, status is
TECHNICALLY NON-EVALUABLE under the strict v43.4 transport rule.

No replacement genes are selected.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


VERSION = "v43.4a"

EXPECTED_V43_4_SHA256 = (
    "361f3c57ccc5687bedc829080f86412fdc7f297a61fd814120502f5bc2b836b9"
)

FROZEN_R = 16
TOP_K = 256


def project_dir_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_v43_4_protocol(project: Path) -> Path:
    candidates = [
        project / "docs" / "v43_4_CML_EXTERNAL_GENERALIZATION_PREREGISTRATION.txt",
        project / "code" / "v43_4_CML_EXTERNAL_GENERALIZATION_PREREGISTRATION.txt",
        project / "v43_4_CML_EXTERNAL_GENERALIZATION_PREREGISTRATION.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "v43_4_CML_EXTERNAL_GENERALIZATION_PREREGISTRATION.txt not found"
    )


def parse_tab_line(line: str):
    return [x.strip().strip('"') for x in line.rstrip("\r\n").split("\t")]


def parse_series_matrix_expression(path: Path):
    """
    Read expression matrix only plus sample metadata needed for DEVELOPMENT
    labels. For external GSE44589 this function is NOT used to parse response.
    """
    sample_meta = defaultdict(list)
    inside = False
    header = None
    probes = []
    rows = []

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not inside and line.startswith("!Sample_"):
                parts = parse_tab_line(line)
                sample_meta[parts[0]].append(parts[1:])
                continue

            if line.startswith("!series_matrix_table_begin"):
                inside = True
                continue

            if line.startswith("!series_matrix_table_end"):
                break

            if not inside:
                continue

            if header is None:
                header = parse_tab_line(line)
                continue

            if line.strip():
                parts = line.rstrip("\r\n").split("\t")
                probes.append(parts[0].strip().strip('"'))
                rows.append([float(x.strip().strip('"')) for x in parts[1:]])

    if header is None:
        raise RuntimeError(f"No series matrix table found in {path}")

    sample_ids = header[1:]

    X_probe = pd.DataFrame(
        np.asarray(rows, dtype=np.float64).T,
        index=sample_ids,
        columns=probes,
    )

    n = len(sample_ids)
    meta_rows = [{"geo_accession": gsm} for gsm in sample_ids]

    for key, occurrences in sample_meta.items():
        for occ_i, vals in enumerate(occurrences, 1):
            if len(vals) != n:
                continue
            field = key if len(occurrences) == 1 else f"{key}__{occ_i}"
            for i, v in enumerate(vals):
                meta_rows[i][field] = v

    return X_probe, pd.DataFrame(meta_rows).set_index("geo_accession")


def development_labels(meta):
    """
    Parse GSE130404 only.
    """
    y = []

    for gsm, row in meta.iterrows():
        chars = {}

        for k, v in row.items():
            if k.startswith("!Sample_characteristics_ch1") and ":" in str(v):
                name, value = str(v).split(":", 1)
                chars[name.strip().lower()] = value.strip()

        stage = chars.get("disease stage", "").strip().lower()
        resp = chars.get("bcr-abl1 at 3 month", "").strip()

        if stage != "diagnostic chronic phase":
            raise RuntimeError(
                f"{gsm}: unexpected development stage {stage!r}"
            )

        if resp == ">10%":
            y.append(1)
        elif resp == "<10%":
            y.append(0)
        else:
            raise RuntimeError(
                f"{gsm}: unresolved development response {resp!r}"
            )

    return np.asarray(y, dtype=int)


def parse_platform_mapping(path: Path):
    """
    Return:
        probe -> set(gene symbols)
        gene -> set(probes)
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    begin = next(
        i for i, x in enumerate(lines)
        if x.startswith("!platform_table_begin")
    )
    end = next(
        i for i, x in enumerate(lines)
        if x.startswith("!platform_table_end")
    )

    table = lines[begin + 1:end]
    header = table[0].split("\t")
    lut = {x.strip().lower(): x for x in header}

    id_col = lut.get("id")
    symbol_col = next(
        (
            lut[c]
            for c in ["symbol", "gene symbol", "gene_symbol", "genesymbol"]
            if c in lut
        ),
        None,
    )

    if id_col is None or symbol_col is None:
        raise RuntimeError(
            f"Cannot identify platform mapping columns in {path.name}: {header}"
        )

    probe_to_symbols = defaultdict(set)
    gene_to_probes = defaultdict(set)

    reader = csv.DictReader(
        table[1:],
        fieldnames=header,
        delimiter="\t",
    )

    for row in reader:
        probe = str(row.get(id_col, "")).strip()
        raw = str(row.get(symbol_col, "")).strip()

        if not probe or not raw:
            continue

        raw = (
            raw.replace("///", "|")
            .replace(";", "|")
            .replace(",", "|")
        )

        for sym in raw.split("|"):
            sym = sym.strip()

            if not sym or sym.upper() in {"---", "NA", "N/A", "NULL"}:
                continue

            probe_to_symbols[probe].add(sym)
            gene_to_probes[sym].add(probe)

    return dict(probe_to_symbols), dict(gene_to_probes)


def aggregate_probe_to_gene(X_probe, probe_to_symbols):
    gene_to_probes = defaultdict(list)

    for probe in X_probe.columns:
        for sym in probe_to_symbols.get(probe, []):
            gene_to_probes[sym].append(probe)

    out = {}

    for gene, probes in gene_to_probes.items():
        out[gene] = X_probe.loc[:, probes].mean(axis=1)

    if not out:
        raise RuntimeError("No development gene mappings produced.")

    return pd.DataFrame(out, index=X_probe.index)


def class_contrast_operator(X, y):
    X0 = X[y == 0]
    X1 = X[y == 1]

    H0 = np.einsum("ni,nj->ij", X0, X0) / len(X0)
    H1 = np.einsum("ni,nj->ij", X1, X1) / len(X1)

    H = H1 - H0
    return 0.5 * (H + H.T)


def main():
    project = project_dir_from_script()

    prereg_path = find_v43_4_protocol(project)
    prereg_sha = sha256_file(prereg_path)

    print(f"=== {VERSION} CML TECHNICAL TRANSPORT AUDIT ===")
    print("Expected v43.4 SHA:", EXPECTED_V43_4_SHA256)
    print("Actual v43.4 SHA:  ", prereg_sha)

    if prereg_sha != EXPECTED_V43_4_SHA256:
        raise SystemExit("FAIL: v43.4 preregistration SHA mismatch.")

    print("PASS: v43.4 preregistration verified.")
    print("External response labels used: NO")
    print()

    # Development files
    dev_series = (
        project
        / "data"
        / "external"
        / "GSE130404"
        / "raw"
        / "GSE130404_series_matrix.txt.gz"
    )

    dev_platform = (
        project
        / "data"
        / "external"
        / "GSE130404"
        / "platform"
        / "GPL10558_full_geo_table.txt"
    )

    # External platform annotation only.
    # No GSE44589 response metadata is read in this audit.
    ext_platform_candidates = [
        project / "data" / "external" / "GSE44589" / "platform" / "GPL570_full_geo_table.txt",
        project / "data" / "external" / "GSE44589" / "platform" / "GPL570_platform.txt",
        project / "data" / "external" / "GSE44589" / "raw" / "GPL570_full_geo_table.txt",
    ]

    ext_platform = next(
        (p for p in ext_platform_candidates if p.exists()),
        None,
    )

    for p in [dev_series, dev_platform]:
        if not p.exists():
            raise FileNotFoundError(p)

    if ext_platform is None:
        tried = "\n".join(str(p) for p in ext_platform_candidates)
        raise FileNotFoundError(
            "GPL570 platform annotation not found. Tried:\n" + tried
        )

    # Reconstruct final development-only support and basis.
    X_probe, dev_meta = parse_series_matrix_expression(dev_series)

    dev_probe_to_symbols, _ = parse_platform_mapping(dev_platform)

    X_gene = aggregate_probe_to_gene(
        X_probe,
        dev_probe_to_symbols,
    )

    y = development_labels(dev_meta)

    if X_gene.shape[0] != 96:
        raise RuntimeError(
            f"Expected 96 development samples, found {X_gene.shape[0]}"
        )

    # Frozen final development support:
    # raw variance over all 96 development samples, no external information.
    variances = X_gene.var(axis=0, ddof=1)

    # Deterministic tie-break by gene symbol.
    var_df = pd.DataFrame({
        "gene": variances.index.astype(str),
        "variance": variances.to_numpy(dtype=float),
    })

    var_df = var_df.sort_values(
        ["variance", "gene"],
        ascending=[False, True],
        kind="mergesort",
    )

    final_genes = var_df.head(TOP_K)["gene"].tolist()

    X_dev_raw = X_gene.loc[:, final_genes].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_dev = scaler.fit_transform(X_dev_raw)

    H = class_contrast_operator(X_dev, y)

    evals, U = np.linalg.eigh(H)
    order = np.argsort(np.abs(evals))[::-1]

    U16 = U[:, order[:FROZEN_R]]
    lambda16 = evals[order[:FROZEN_R]]

    # External platform measurability only.
    _, ext_gene_to_probes = parse_platform_mapping(ext_platform)

    measurable = np.asarray(
        [gene in ext_gene_to_probes for gene in final_genes],
        dtype=bool,
    )

    n_measurable = int(measurable.sum())
    n_missing = TOP_K - n_measurable

    missing_genes = [
        g for g, ok in zip(final_genes, measurable)
        if not ok
    ]

    # Quantify projector mass retained on measurable coordinates.
    # For P = U U^T, diagonal mass per coordinate is row squared norm.
    coord_mass = np.sum(U16 * U16, axis=1)

    total_mass = float(np.sum(coord_mass))  # should be r
    measurable_mass = float(np.sum(coord_mass[measurable]))
    missing_mass = float(np.sum(coord_mass[~measurable]))

    retained_fraction = (
        measurable_mass / total_mass
        if total_mass > 0
        else np.nan
    )

    status = (
        "TECHNICALLY EVALUABLE"
        if n_missing == 0
        else "TECHNICALLY NON-EVALUABLE"
    )

    outdir = project / "results" / "direct_subspace"
    outdir.mkdir(parents=True, exist_ok=True)

    mapping_path = outdir / "v43_4a_transport_gene_audit.csv"
    summary_path = outdir / "v43_4a_transport_audit_summary.txt"
    manifest_path = outdir / "v43_4a_manifest.json"
    frozen_basis_path = outdir / "v43_4a_frozen_development_basis.npz"

    pd.DataFrame({
        "gene": final_genes,
        "raw_variance_rank": np.arange(1, TOP_K + 1),
        "gpl570_measurable": measurable.astype(int),
        "projector_diagonal_mass": coord_mass,
    }).to_csv(mapping_path, index=False)

    np.savez_compressed(
        frozen_basis_path,
        genes=np.asarray(final_genes, dtype=str),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        U16=U16,
        eigenvalues=lambda16,
    )

    lines = [
        "=== Soft Spaces / CML v43.4a TECHNICAL PLATFORM/SUBSPACE TRANSPORT AUDIT ===",
        "",
        "DESIGN",
        "------",
        "Development cohort: GSE130404",
        "External platform: GPL570 / GSE44589",
        f"Frozen rank r: {FROZEN_R}",
        f"Final development support: Top-{TOP_K} raw-variance genes",
        "Development samples used for final representation: 96",
        "External response labels used: NO",
        "",
        "PLATFORM TRANSPORT",
        "------------------",
        f"Frozen development coordinates: {TOP_K}",
        f"Measurable on GPL570:            {n_measurable}",
        f"Missing on GPL570:               {n_missing}",
        f"Coordinate coverage:             {n_measurable / TOP_K:.6f}",
        "",
        "FROZEN PROJECTOR MASS",
        "---------------------",
        f"Total projector diagonal mass:   {total_mass:.6f}",
        f"Measurable-coordinate mass:      {measurable_mass:.6f}",
        f"Missing-coordinate mass:         {missing_mass:.6f}",
        f"Retained projector-mass fraction:{retained_fraction:.6f}",
        "",
        f"v43.4a STATUS: {status}",
        "",
        "STRICT PREREGISTERED RULE",
        "-------------------------",
        "All frozen coordinates must be measurable for faithful transport.",
        "No missing coordinate may be replaced by an outcome-selected gene.",
        "",
        "MISSING GENES",
        "-------------",
    ]

    if missing_genes:
        lines.extend(missing_genes)
    else:
        lines.append("NONE")

    lines += [
        "",
        "INTERPRETATION",
        "--------------",
        "This audit evaluates technical platform transport only.",
        "It does not use or evaluate GSE44589 response outcomes.",
        "",
        "If TECHNICALLY NON-EVALUABLE, v43.5 must not silently modify the",
        "frozen representation to rescue the external test.",
    ]

    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "version": VERSION,
        "status": status,
        "development_cohort": "GSE130404",
        "external_cohort": "GSE44589",
        "development_platform": "GPL10558",
        "external_platform": "GPL570",
        "frozen_r": FROZEN_R,
        "top_k": TOP_K,
        "development_n": int(X_gene.shape[0]),
        "n_measurable": n_measurable,
        "n_missing": n_missing,
        "coordinate_coverage": n_measurable / TOP_K,
        "total_projector_mass": total_mass,
        "measurable_projector_mass": measurable_mass,
        "missing_projector_mass": missing_mass,
        "retained_projector_mass_fraction": retained_fraction,
        "missing_genes": missing_genes,
        "external_response_labels_used": False,
        "sha256": {
            "v43_4_preregistration": prereg_sha,
            "development_series": sha256_file(dev_series),
            "development_platform": sha256_file(dev_platform),
            "external_platform": sha256_file(ext_platform),
            "execution_script": sha256_file(Path(__file__).resolve()),
            "frozen_basis_npz": sha256_file(frozen_basis_path),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(summary_path.read_text(encoding="utf-8"))

    print("Wrote:")
    for p in [
        mapping_path,
        frozen_basis_path,
        summary_path,
        manifest_path,
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
