#!/usr/bin/env python3
"""
Soft Spaces / CML
v43.4d — Automated outcome-blind pseudobulk + frozen-gene coverage audit
External candidate: GSE236233

Primary frozen external population:
    Lin-CD34+ diagnosis samples

Reason:
    one Lin-CD34+ sample exists for each of the 9 patients.

Statistical unit:
    patient

This script AUTOMATICALLY:
1. reuses/downloads GSE236233 family SOFT;
2. identifies the 9 Lin-CD34+ samples;
3. downloads each processed supplementary tar.gz from GEO;
4. safely extracts each archive;
5. detects the RNA count matrix and gene-feature table;
6. constructs one raw-count pseudobulk vector per patient;
7. measures coverage of the frozen 256 development coordinates;
8. measures retained frozen r=16 projector mass;
9. writes an auditable patient x gene pseudobulk matrix.

It DOES NOT:
- parse or use 12-month response labels;
- score external predictive performance;
- fit an external classifier;
- treat cells as independent patients;
- replace missing genes using outcome information.

Dependencies:
    numpy, pandas, scipy
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


VERSION = "v43.4d"
GSE = "GSE236233"
FROZEN_R = 16
TOP_K = 256

SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE236nnn/GSE236233/soft/GSE236233_family.soft.gz"
)


def project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        print("Using existing:", path)
        return

    print("Downloading:", url)
    print(" ->", path)
    urllib.request.urlretrieve(url, path)


def safe_extract_tar(tar_path: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    root = outdir.resolve()

    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            target = (outdir / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(
                    f"Unsafe path in archive {tar_path.name}: {member.name}"
                )

        tf.extractall(outdir)


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
                    "source": "",
                    "characteristics": [],
                    "supplementary": [],
                }
                samples.append(current)
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
            elif k == "Sample_characteristics_ch1":
                current["characteristics"].append(v)
            elif k.startswith("Sample_supplementary_file"):
                current["supplementary"].append(v)

    return samples


def patient_id(text: str):
    m = re.search(r"(CML\d+)", text, flags=re.I)
    return m.group(1).upper() if m else None


def is_primary_cd34(text: str):
    """
    Primary = Lin-CD34+ but not the narrower CD34+CD38- sample.
    Titles in this series include e.g. CML1_34p vs CML2_38n2.
    """
    t = text.lower().replace(" ", "")

    narrow = (
        "cd34+cd38-" in t
        or "_38n" in t
        or "38n]" in t
    )

    broad = (
        "lin-cd34+" in t
        or "_34p" in t
        or "34p]" in t
    )

    return broad and not narrow


def choose_supplementary_url(rec):
    urls = [
        u for u in rec["supplementary"]
        if u.lower().endswith((".tar.gz", ".tgz", ".tar"))
    ]

    if len(urls) != 1:
        raise RuntimeError(
            f"{rec['gsm']}: expected exactly one processed TAR archive, "
            f"found {len(urls)}: {urls}"
        )

    return urls[0]


def recursive_files(root: Path):
    return [p for p in root.rglob("*") if p.is_file()]


def score_matrix_candidate(path: Path):
    n = path.name.lower()

    score = 0

    if n.endswith(".mtx"):
        score += 100
    elif n.endswith(".mtx.gz"):
        score += 95
    else:
        return -1

    if "adt" in n or "antibody" in n or "protein" in n:
        score -= 100

    if "matrix" in n:
        score += 20

    if "rna" in n or "gene" in n:
        score += 10

    return score


def score_feature_candidate(path: Path):
    n = path.name.lower()

    if not n.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz", ".csv", ".csv.gz")):
        return -1

    score = 0

    if "feature" in n:
        score += 100

    if "gene" in n:
        score += 50

    if "barcode" in n:
        score -= 100

    if "meta" in n or "cell" in n or "adt" in n:
        score -= 30

    return score


def read_text_table_noheader(path: Path):
    sep = "," if ".csv" in path.name.lower() else "\t"
    compression = "gzip" if path.name.lower().endswith(".gz") else None

    return pd.read_csv(
        path,
        sep=sep,
        header=None,
        dtype=str,
        compression=compression,
        engine="python",
    )


def read_mtx(path: Path):
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "rb") as f:
            M = mmread(f)
    else:
        M = mmread(path)

    if sparse.issparse(M):
        return M.tocsr()

    return sparse.csr_matrix(np.asarray(M))


def select_matrix_and_features(extract_dir: Path):
    files = recursive_files(extract_dir)

    matrix_candidates = sorted(
        [
            (score_matrix_candidate(p), p)
            for p in files
            if score_matrix_candidate(p) >= 0
        ],
        key=lambda x: (-x[0], str(x[1])),
    )

    feature_candidates = sorted(
        [
            (score_feature_candidate(p), p)
            for p in files
            if score_feature_candidate(p) >= 0
        ],
        key=lambda x: (-x[0], str(x[1])),
    )

    if not matrix_candidates:
        raise RuntimeError(
            f"No RNA Matrix Market file found under {extract_dir}"
        )

    matrix_path = matrix_candidates[0][1]
    M = read_mtx(matrix_path)

    # Find a feature table whose number of rows matches one matrix dimension.
    compatible = []

    for score, p in feature_candidates:
        try:
            df = read_text_table_noheader(p)
        except Exception:
            continue

        if len(df) in M.shape:
            compatible.append(
                (
                    score,
                    p,
                    df,
                )
            )

    if not compatible:
        inventory = "\n".join(str(p) for p in files)
        raise RuntimeError(
            f"No feature/gene table compatible with matrix shape {M.shape} "
            f"under {extract_dir}.\nFiles:\n{inventory}"
        )

    compatible.sort(
        key=lambda x: (-x[0], str(x[1]))
    )

    feature_path = compatible[0][1]
    feature_df = compatible[0][2]

    return matrix_path, M, feature_path, feature_df


def feature_symbols(feature_df: pd.DataFrame):
    """
    10x features.tsv usually:
        col0 = Ensembl ID
        col1 = gene symbol
        col2 = feature type

    Prefer col1 if present. Otherwise use col0.
    """
    if feature_df.shape[1] >= 2:
        symbols = feature_df.iloc[:, 1].fillna("").astype(str)
    else:
        symbols = feature_df.iloc[:, 0].fillna("").astype(str)

    return symbols.tolist()


def pseudobulk_from_sample(extract_dir: Path):
    matrix_path, M, feature_path, feature_df = select_matrix_and_features(
        extract_dir
    )

    symbols = feature_symbols(feature_df)

    if len(symbols) == M.shape[0]:
        gene_by_cell = M
    elif len(symbols) == M.shape[1]:
        gene_by_cell = M.T.tocsr()
    else:
        raise RuntimeError(
            f"Feature count {len(symbols)} does not match matrix shape {M.shape}"
        )

    sums = np.asarray(
        gene_by_cell.sum(axis=1)
    ).ravel()

    gene_sum = defaultdict(float)

    for g, value in zip(symbols, sums):
        g = str(g).strip()

        if (
            not g
            or g.upper() in {"NA", "N/A", "NULL", "---"}
        ):
            continue

        gene_sum[g] += float(value)

    info = {
        "matrix_path": str(matrix_path),
        "feature_path": str(feature_path),
        "matrix_shape_original": list(M.shape),
        "gene_by_cell_shape": list(gene_by_cell.shape),
        "n_detected_gene_symbols": len(gene_sum),
        "total_counts": float(np.sum(sums)),
    }

    return dict(gene_sum), info


def main():
    project = project_dir()

    data_root = (
        project
        / "data"
        / "external"
        / GSE
    )

    meta_dir = data_root / "metadata"
    processed_dir = data_root / "processed"
    extract_root = data_root / "extracted"

    outdir = (
        project
        / "results"
        / "direct_subspace"
    )

    for p in [
        meta_dir,
        processed_dir,
        extract_root,
        outdir,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    soft_path = meta_dir / f"{GSE}_family.soft.gz"
    download(SOFT_URL, soft_path)

    samples = parse_soft(soft_path)

    primary = []

    for rec in samples:
        combined = " ".join(
            [
                rec["title"],
                rec["source"],
                *rec["characteristics"],
            ]
        )

        pid = patient_id(combined)

        if pid and is_primary_cd34(combined):
            rec = dict(rec)
            rec["patient_id"] = pid
            primary.append(rec)

    primary.sort(
        key=lambda r: int(
            re.search(r"\d+", r["patient_id"]).group()
        )
    )

    patients = [r["patient_id"] for r in primary]

    print(f"=== {VERSION} AUTOMATED PSEUDOBULK / GENE COVERAGE AUDIT ===")
    print("Candidate:", GSE)
    print("Frozen external population: Lin-CD34+")
    print("Statistical unit: patient")
    print("Detected patients:", ", ".join(patients))
    print("External outcome labels used: NO")
    print()

    if len(primary) != 9 or len(set(patients)) != 9:
        raise RuntimeError(
            f"Expected one Lin-CD34+ sample for each of 9 patients; "
            f"found {len(primary)} samples and {len(set(patients))} patients."
        )

    # Frozen development representation from v43.4a.
    basis_path = (
        outdir
        / "v43_4a_frozen_development_basis.npz"
    )

    if not basis_path.exists():
        raise FileNotFoundError(basis_path)

    frozen = np.load(
        basis_path,
        allow_pickle=False,
    )

    frozen_genes = [
        str(x)
        for x in frozen["genes"].tolist()
    ]

    U16 = np.asarray(
        frozen["U16"],
        dtype=float,
    )

    if len(frozen_genes) != TOP_K:
        raise RuntimeError(
            f"Expected {TOP_K} frozen genes, found {len(frozen_genes)}"
        )

    if U16.shape != (TOP_K, FROZEN_R):
        raise RuntimeError(
            f"Expected U16 shape {(TOP_K, FROZEN_R)}, found {U16.shape}"
        )

    coord_mass = np.sum(
        U16 * U16,
        axis=1,
    )

    pseudobulk = {}
    technical_records = []

    for i, rec in enumerate(primary, 1):
        gsm = rec["gsm"]
        pid = rec["patient_id"]
        url = choose_supplementary_url(rec)

        filename = url.rsplit("/", 1)[-1]
        archive_path = (
            processed_dir
            / filename
        )

        sample_extract_dir = (
            extract_root
            / gsm
        )

        download(url, archive_path)

        marker = (
            sample_extract_dir
            / ".extraction_complete"
        )

        if not marker.exists():
            if sample_extract_dir.exists():
                shutil.rmtree(sample_extract_dir)

            safe_extract_tar(
                archive_path,
                sample_extract_dir,
            )

            marker.write_text(
                "ok\n",
                encoding="utf-8",
            )
        else:
            print("Using extracted:", sample_extract_dir)

        gene_sum, info = pseudobulk_from_sample(
            sample_extract_dir
        )

        pseudobulk[pid] = gene_sum

        technical_records.append({
            "patient_id": pid,
            "gsm": gsm,
            "title": rec["title"],
            "url": url,
            "archive": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            **info,
        })

        print(
            f"[{i}/9] {pid} {gsm}: "
            f"{info['n_detected_gene_symbols']} gene symbols; "
            f"{info['gene_by_cell_shape'][1]} cells"
        )

    # Union pseudobulk matrix.
    all_genes = sorted(
        {
            g
            for d in pseudobulk.values()
            for g in d
        }
    )

    pseudo_df = pd.DataFrame(
        {
            pid: [
                pseudobulk[pid].get(g, 0.0)
                for g in all_genes
            ]
            for pid in patients
        },
        index=all_genes,
    ).T

    # Technical presence means represented in feature tables across all 9
    # primary patient samples. We record both any-patient and all-patient
    # presence; faithful fixed-coordinate external representation requires
    # presence in ALL 9.
    patient_gene_sets = {
        pid: set(pseudobulk[pid].keys())
        for pid in patients
    }

    coverage_rows = []

    for idx, gene in enumerate(frozen_genes):
        present_patients = [
            pid
            for pid in patients
            if gene in patient_gene_sets[pid]
        ]

        n_present = len(present_patients)

        coverage_rows.append({
            "coordinate_index": idx,
            "gene": gene,
            "projector_diagonal_mass": float(coord_mass[idx]),
            "patients_with_gene_feature": n_present,
            "present_all_9": int(n_present == 9),
            "present_any": int(n_present > 0),
            "present_patients": "|".join(present_patients),
        })

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    all9 = (
        coverage_df["present_all_9"] == 1
    )

    n_all9 = int(all9.sum())
    n_missing = TOP_K - n_all9

    total_mass = float(
        coverage_df[
            "projector_diagonal_mass"
        ].sum()
    )

    retained_mass = float(
        coverage_df.loc[
            all9,
            "projector_diagonal_mass"
        ].sum()
    )

    retained_fraction = (
        retained_mass / total_mass
    )

    missing_genes = coverage_df.loc[
        ~all9,
        "gene",
    ].tolist()

    # Strict transport status.
    strict_status = (
        "TECHNICALLY EVALUABLE"
        if n_missing == 0
        else "TECHNICALLY NON-EVALUABLE"
    )

    # Save outputs.
    pseudo_path = (
        outdir
        / "v43_4d_GSE236233_CD34_patient_pseudobulk_raw_counts.csv"
    )

    coverage_path = (
        outdir
        / "v43_4d_GSE236233_frozen_gene_coverage.csv"
    )

    technical_path = (
        outdir
        / "v43_4d_GSE236233_sample_technical_manifest.json"
    )

    summary_path = (
        outdir
        / "v43_4d_GSE236233_pseudobulk_coverage_summary.txt"
    )

    manifest_path = (
        outdir
        / "v43_4d_manifest.json"
    )

    pseudo_df.to_csv(
        pseudo_path
    )

    coverage_df.to_csv(
        coverage_path,
        index=False,
    )

    technical_path.write_text(
        json.dumps(
            technical_records,
            indent=2,
        ),
        encoding="utf-8",
    )

    cell_counts = [
        x["gene_by_cell_shape"][1]
        for x in technical_records
    ]

    summary_lines = [
        "=== Soft Spaces / CML v43.4d GSE236233 PSEUDOBULK + GENE COVERAGE AUDIT ===",
        "",
        "FROZEN DESIGN",
        "-------------",
        "External candidate: GSE236233",
        "Primary population: Lin-CD34+",
        "Statistical unit: patient",
        "Patients: 9",
        "Patient IDs: " + ", ".join(patients),
        "External response labels used: NO",
        "External predictive performance evaluated: NO",
        "",
        "AUTOMATED DATA INGESTION",
        "------------------------",
        "Processed GEO archives downloaded automatically: 9",
        f"Total cells across primary samples: {sum(cell_counts)}",
        f"Smallest patient sample (cells):    {min(cell_counts)}",
        f"Largest patient sample (cells):     {max(cell_counts)}",
        "",
        "PSEUDOBULK",
        "----------",
        "Aggregation: raw RNA counts summed across all cells within each",
        "Lin-CD34+ patient sample.",
        "Rows in patient-level gene universe: "
        + str(len(all_genes)),
        "",
        "FROZEN v43 COORDINATE COVERAGE",
        "------------------------------",
        f"Frozen coordinates:              {TOP_K}",
        f"Present in all 9 patients:       {n_all9}",
        f"Missing from >=1 patient:        {n_missing}",
        f"Strict coordinate coverage:      {n_all9 / TOP_K:.6f}",
        "",
        "FROZEN PROJECTOR MASS",
        "---------------------",
        f"Total projector mass:            {total_mass:.6f}",
        f"Retained all-patient mass:       {retained_mass:.6f}",
        f"Retained projector fraction:     {retained_fraction:.6f}",
        "",
        f"v43.4d STRICT STATUS: {strict_status}",
        "",
        "MISSING / NON-UNIVERSAL FROZEN GENES",
        "------------------------------------",
    ]

    if missing_genes:
        summary_lines.extend(
            missing_genes
        )
    else:
        summary_lines.append(
            "NONE"
        )

    summary_lines += [
        "",
        "INTERPRETATION",
        "--------------",
        "This is an outcome-blind technical transport audit.",
        "No response labels were used and no external classifier was scored.",
        "",
        "The raw pseudobulk matrix is preserved for the later preregistered",
        "external analysis if and only if a scientifically valid transport",
        "rule is frozen before outcome evaluation.",
        "",
        "Cells are never treated as independent patients.",
    ]

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "version": VERSION,
        "candidate": GSE,
        "primary_population": "Lin-CD34+",
        "statistical_unit": "patient",
        "patient_n": 9,
        "patient_ids": patients,
        "aggregation": "sum raw RNA counts across cells within patient sample",
        "frozen_coordinates": TOP_K,
        "present_all_9": n_all9,
        "missing_from_at_least_one_patient": n_missing,
        "coordinate_coverage": n_all9 / TOP_K,
        "total_projector_mass": total_mass,
        "retained_projector_mass": retained_mass,
        "retained_projector_mass_fraction": retained_fraction,
        "strict_status": strict_status,
        "external_response_labels_used": False,
        "external_predictive_performance_evaluated": False,
        "sha256": {
            "family_soft": sha256_file(soft_path),
            "frozen_development_basis": sha256_file(basis_path),
            "pseudobulk_csv": sha256_file(pseudo_path),
            "coverage_csv": sha256_file(coverage_path),
            "execution_script": sha256_file(Path(__file__).resolve()),
        },
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    print("Wrote:")
    for p in [
        pseudo_path,
        coverage_path,
        technical_path,
        summary_path,
        manifest_path,
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
