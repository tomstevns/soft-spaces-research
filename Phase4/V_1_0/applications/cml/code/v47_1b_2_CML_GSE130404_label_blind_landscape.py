#!/usr/bin/env python3
"""
Soft Spaces / CML
v47.1b-v47.2 — GSE130404 label-blind state-landscape build

Purpose
-------
Resolve the missing local GSE130404 matrix by reconstructing it directly
from the GEO family SOFT file, then build the preregistered v47.2 landscape.

STRICTLY LABEL-BLIND:
- BCR-ABL1 response labels are not parsed.
- No outcome field is used.
- No classifier is fitted.

Pipeline
--------
GSE130404 / GPL10558 sample VALUE tables
 -> probe-to-gene mapping
 -> mean across probes per gene
 -> restrict to frozen v44 transport-aware 15,890-gene universe
 -> Top-256 by raw across-patient variance
 -> per-gene cohort z-standardization (ddof=0)
 -> PCA-16
 -> kNN local geometry, k=10
 -> SHA-freeze geometry for later v47.3 label overlay
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


VERSION = "v47.2"
GSE = "GSE130404"
GPL = "GPL10558"
EXPECTED_N = 96
ELIGIBLE_N = 15890
TOP_K = 256
PCA_R = 16
KNN_K = 10

SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE130nnn/GSE130404/soft/GSE130404_family.soft.gz"
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


def split_gene_symbols(text):
    if text is None:
        return []

    text = str(text).strip()

    if not text or text in {"---", "NA", "nan"}:
        return []

    parts = re.split(
        r"\s*///\s*|\s*//\s*|\s*;\s*|\s*,\s*",
        text,
    )

    out = []

    for p in parts:
        p = re.sub(r"\s+", "", p.strip())
        if p and p not in {"---", "NA"}:
            out.append(p)

    return out


def parse_family_soft(soft_path: Path):
    """
    Parse only:
    - GPL10558 annotation table
    - SAMPLE ID + VALUE expression tables

    Intentionally DO NOT parse Sample_characteristics_ch1 or any
    outcome/response metadata.
    """
    probe_to_symbols = {}
    sample_values = {}

    current_entity = None
    current_id = None

    in_platform_table = False
    in_sample_table = False

    platform_header = None
    sample_header = None

    probe_col_idx = None
    symbol_col_idx = None
    sample_probe_idx = None
    sample_value_idx = None

    with gzip.open(
        soft_path,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if line.startswith("^PLATFORM"):
                current_entity = "PLATFORM"
                current_id = line.split("=", 1)[1].strip()
                in_platform_table = False
                in_sample_table = False
                platform_header = None
                probe_col_idx = None
                symbol_col_idx = None
                continue

            if line.startswith("^SAMPLE"):
                current_entity = "SAMPLE"
                current_id = line.split("=", 1)[1].strip()
                in_platform_table = False
                in_sample_table = False
                sample_header = None
                sample_probe_idx = None
                sample_value_idx = None
                sample_values[current_id] = {}
                continue

            # Ignore all metadata lines except table boundaries.
            if current_entity == "PLATFORM" and current_id == GPL:
                if line == "!platform_table_begin":
                    in_platform_table = True
                    platform_header = None
                    continue

                if line == "!platform_table_end":
                    in_platform_table = False
                    continue

                if in_platform_table:
                    fields = line.split("\t")

                    if platform_header is None:
                        platform_header = fields
                        norm = [x.strip().lower() for x in fields]

                        for cand in (
                            "id",
                            "id_ref",
                            "probe id",
                            "probe_id",
                        ):
                            if cand in norm:
                                probe_col_idx = norm.index(cand)
                                break

                        if probe_col_idx is None:
                            probe_col_idx = 0

                        for cand in (
                            "gene symbol",
                            "gene_symbol",
                            "symbol",
                            "symbol_reannotated",
                        ):
                            if cand in norm:
                                symbol_col_idx = norm.index(cand)
                                break

                        if symbol_col_idx is None:
                            for i, h in enumerate(norm):
                                if "symbol" in h:
                                    symbol_col_idx = i
                                    break

                        if symbol_col_idx is None:
                            raise RuntimeError(
                                "Could not identify GPL10558 gene-symbol column. "
                                f"Header: {platform_header}"
                            )

                        continue

                    max_idx = max(probe_col_idx, symbol_col_idx)

                    if len(fields) <= max_idx:
                        continue

                    probe = fields[probe_col_idx].strip()
                    symbols = split_gene_symbols(
                        fields[symbol_col_idx]
                    )

                    if probe and symbols:
                        probe_to_symbols[probe] = symbols

                    continue

            if current_entity == "SAMPLE":
                if line == "!sample_table_begin":
                    in_sample_table = True
                    sample_header = None
                    continue

                if line == "!sample_table_end":
                    in_sample_table = False
                    continue

                if in_sample_table:
                    fields = line.split("\t")

                    if sample_header is None:
                        sample_header = fields
                        norm = [x.strip().lower() for x in fields]

                        if "id_ref" not in norm:
                            raise RuntimeError(
                                f"{current_id}: no ID_REF."
                            )

                        if "value" not in norm:
                            raise RuntimeError(
                                f"{current_id}: no VALUE."
                            )

                        sample_probe_idx = norm.index("id_ref")
                        sample_value_idx = norm.index("value")
                        continue

                    max_idx = max(
                        sample_probe_idx,
                        sample_value_idx,
                    )

                    if len(fields) <= max_idx:
                        continue

                    probe = fields[sample_probe_idx].strip()

                    try:
                        value = float(
                            fields[sample_value_idx].strip()
                        )
                    except ValueError:
                        continue

                    sample_values[current_id][probe] = value

    return probe_to_symbols, sample_values


def main():
    project = project_dir()

    docs = project / "docs"
    ds = project / "results" / "direct_subspace"
    data_dir = (
        project
        / "data"
        / "external"
        / GSE
        / "metadata"
    )

    docs.mkdir(parents=True, exist_ok=True)
    ds.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    protocol = docs / "v47_0_CML_STATE_LANDSCAPE_PREREGISTRATION.txt"
    protocol_manifest = docs / "v47_0_CML_STATE_LANDSCAPE_PREREGISTRATION_manifest.json"
    eligible_path = ds / "v44_1b_eligible_genes.txt"

    for p in [protocol, protocol_manifest, eligible_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    pm = json.loads(
        protocol_manifest.read_text(
            encoding="utf-8"
        )
    )

    actual_protocol_sha = sha256_file(protocol)

    if actual_protocol_sha != pm["protocol_sha256"]:
        raise RuntimeError(
            "v47.0 protocol SHA mismatch."
        )

    eligible = [
        x.strip()
        for x in eligible_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]

    if len(eligible) != ELIGIBLE_N:
        raise RuntimeError(
            f"Expected {ELIGIBLE_N} eligible genes, found {len(eligible)}."
        )

    eligible_set = set(eligible)

    soft_path = data_dir / f"{GSE}_family.soft.gz"
    download_if_missing(SOFT_URL, soft_path)

    print("=== v47.1b-v47.2 LABEL-BLIND STATE LANDSCAPE ===")
    print("v47.0 protocol SHA:", actual_protocol_sha)
    print("Outcome labels parsed: NO")
    print("Predictive model fitted: NO")
    print()
    print("Parsing GEO expression and GPL10558 annotation...")

    probe_to_symbols, sample_values = parse_family_soft(soft_path)

    print("Annotated probes parsed:", len(probe_to_symbols))
    print("Sample expression tables:", len(sample_values))

    if len(sample_values) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} samples, found {len(sample_values)}."
        )

    symbol_to_probes = defaultdict(list)

    for probe, symbols in probe_to_symbols.items():
        for symbol in symbols:
            if symbol in eligible_set:
                symbol_to_probes[symbol].append(probe)

    measurable_genes = [
        g for g in eligible
        if len(symbol_to_probes.get(g, [])) > 0
    ]

    print("Eligible universe:", len(eligible))
    print("Measurable eligible genes:", len(measurable_genes))

    if len(measurable_genes) < TOP_K:
        raise RuntimeError(
            "Fewer than 256 transport-aware eligible genes measurable."
        )

    gsms = sorted(sample_values.keys())

    # Construct patient x eligible-gene matrix, outcome-blind.
    X = np.empty(
        (len(gsms), len(measurable_genes)),
        dtype=float,
    )

    incomplete = []

    for j, gene in enumerate(measurable_genes):
        probes = sorted(
            set(symbol_to_probes[gene])
        )

        for i, gsm in enumerate(gsms):
            vals = [
                sample_values[gsm][p]
                for p in probes
                if p in sample_values[gsm]
            ]

            if not vals:
                incomplete.append(
                    (gsm, gene)
                )
                X[i, j] = np.nan
            else:
                X[i, j] = float(
                    np.mean(vals)
                )

    if incomplete:
        raise RuntimeError(
            "Missing expression values for measurable genes. "
            f"First examples: {incomplete[:10]}"
        )

    if not np.all(np.isfinite(X)):
        raise RuntimeError(
            "Non-finite expression values."
        )

    # Label-blind raw variance.
    variances = np.var(
        X,
        axis=0,
        ddof=0,
    )

    # Deterministic descending variance; stable tie handling.
    order = np.argsort(
        -variances,
        kind="mergesort",
    )

    top_idx = order[:TOP_K]
    top_genes = [
        measurable_genes[i]
        for i in top_idx
    ]

    X256 = X[:, top_idx]
    var256 = variances[top_idx]

    means = np.mean(
        X256,
        axis=0,
    )

    sds = np.std(
        X256,
        axis=0,
        ddof=0,
    )

    zero = np.where(
        sds == 0
    )[0]

    if len(zero):
        raise RuntimeError(
            "Unexpected zero-SD Top256 genes: "
            + ", ".join(
                top_genes[i]
                for i in zero
            )
        )

    Z = (
        X256 - means
    ) / sds

    # Label-blind PCA-16.
    pca = PCA(
        n_components=PCA_R,
        svd_solver="full",
    )

    Y = pca.fit_transform(Z)

    # Local geometry in frozen PCA space.
    k = KNN_K

    if len(gsms) <= 20:
        k = max(
            3,
            len(gsms) // 3,
        )

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
    )

    nn.fit(Y)

    distances, indices = nn.kneighbors(Y)

    # Drop self neighbor at column 0.
    nbr_dist = distances[:, 1:]
    nbr_idx = indices[:, 1:]

    mean_knn_distance = np.mean(
        nbr_dist,
        axis=1,
    )

    eps = 1e-12

    density_proxy = 1.0 / (
        mean_knn_distance + eps
    )

    centroid = np.mean(
        Y,
        axis=0,
    )

    distance_to_centroid = np.linalg.norm(
        Y - centroid,
        axis=1,
    )

    anisotropy = np.zeros(
        len(gsms),
        dtype=float,
    )

    for i in range(len(gsms)):
        local = Y[nbr_idx[i], :]

        if local.shape[0] < 2:
            anisotropy[i] = np.nan
            continue

        C = np.cov(
            local,
            rowvar=False,
            ddof=1,
        )

        eig = np.linalg.eigvalsh(C)
        eig = np.maximum(eig, 0.0)
        eig = eig[::-1]

        anisotropy[i] = float(
            eig[0]
            / (eig.sum() + eps)
        )

    if not np.all(
        np.isfinite(anisotropy)
    ):
        raise RuntimeError(
            "Non-finite local anisotropy."
        )

    # Save label-blind inputs.
    matrix_out = (
        ds
        / "v47_2_GSE130404_label_blind_top256_expression.tsv"
    )

    pd.DataFrame(
        X256,
        index=gsms,
        columns=top_genes,
    ).rename_axis("gsm").to_csv(
        matrix_out,
        sep="\t",
    )

    z_out = (
        ds
        / "v47_2_GSE130404_label_blind_top256_z.tsv"
    )

    pd.DataFrame(
        Z,
        index=gsms,
        columns=top_genes,
    ).rename_axis("gsm").to_csv(
        z_out,
        sep="\t",
    )

    feature_out = (
        ds
        / "v47_2_label_blind_top256_features.tsv"
    )

    pd.DataFrame({
        "gene": top_genes,
        "raw_variance": var256,
        "cohort_mean": means,
        "cohort_sd_ddof0": sds,
    }).to_csv(
        feature_out,
        sep="\t",
        index=False,
    )

    pc_cols = [
        f"PC{i+1}"
        for i in range(PCA_R)
    ]

    geometry = pd.DataFrame(
        Y,
        index=gsms,
        columns=pc_cols,
    )

    geometry["mean_knn_distance"] = mean_knn_distance
    geometry["density_proxy"] = density_proxy
    geometry["local_anisotropy"] = anisotropy
    geometry["distance_to_centroid"] = distance_to_centroid
    geometry.index.name = "gsm"

    geometry_out = (
        ds
        / "v47_2_GSE130404_label_blind_geometry.tsv"
    )

    geometry.to_csv(
        geometry_out,
        sep="\t",
    )

    pca_loadings_out = (
        ds
        / "v47_2_PCA16_loadings.tsv"
    )

    pd.DataFrame(
        pca.components_.T,
        index=top_genes,
        columns=pc_cols,
    ).rename_axis("gene").to_csv(
        pca_loadings_out,
        sep="\t",
    )

    ev_out = (
        ds
        / "v47_2_PCA16_explained_variance.tsv"
    )

    pd.DataFrame({
        "component": pc_cols,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(
            pca.explained_variance_ratio_
        ),
    }).to_csv(
        ev_out,
        sep="\t",
        index=False,
    )

    geometry_sha = sha256_file(
        geometry_out
    )

    summary_out = (
        ds
        / "v47_2_state_landscape_summary.txt"
    )

    lines = [
        "=== Soft Spaces / CML v47.2 LABEL-BLIND STATE LANDSCAPE ===",
        "",
        "PROTOCOL",
        "--------",
        f"v47.0 SHA verified:              {actual_protocol_sha}",
        "",
        "LABEL INTEGRITY",
        "---------------",
        "Outcome labels parsed:           NO",
        "Outcome labels used:             NO",
        "Predictive classifier fitted:    NO",
        "",
        "INPUT",
        "-----",
        f"Cohort:                          {GSE}",
        f"Platform:                        {GPL}",
        f"Samples:                         {len(gsms)}",
        f"Transport-aware universe:        {len(eligible)}",
        f"Measurable eligible genes:       {len(measurable_genes)}",
        f"Label-blind variance Top-K:      {TOP_K}",
        "",
        "STATE SPACE",
        "-----------",
        f"PCA dimensions:                  {PCA_R}",
        f"PCA-16 cumulative variance:      {np.sum(pca.explained_variance_ratio_):.6f}",
        f"kNN neighborhood k:              {k}",
        "",
        "LOCAL GEOMETRY",
        "--------------",
        f"Mean kNN distance, cohort mean:  {np.mean(mean_knn_distance):.6f}",
        f"Density proxy, cohort mean:      {np.mean(density_proxy):.6f}",
        f"Local anisotropy, cohort mean:   {np.mean(anisotropy):.6f}",
        f"Centroid distance, cohort mean:  {np.mean(distance_to_centroid):.6f}",
        "",
        "GEOMETRY FREEZE",
        "---------------",
        f"Geometry SHA256:                 {geometry_sha}",
        "",
        "v47.2 STATUS: LABEL-BLIND GEOMETRY FROZEN",
        "",
        "NEXT:",
        "v47.3 may now attach clinical labels and perform the preregistered",
        "10,000-permutation centroid-separation test.",
    ]

    summary_out.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest_out = (
        ds
        / "v47_2_manifest.json"
    )

    manifest_out.write_text(
        json.dumps(
            {
                "version": VERSION,
                "status": "LABEL_BLIND_GEOMETRY_FROZEN",
                "cohort": GSE,
                "platform": GPL,
                "sample_n": len(gsms),
                "eligible_universe_n": len(eligible),
                "measurable_eligible_gene_n": len(measurable_genes),
                "top_k": TOP_K,
                "pca_r": PCA_R,
                "knn_k": k,
                "labels_parsed": False,
                "labels_used": False,
                "predictive_classifier_fitted": False,
                "pca_cumulative_explained_variance_ratio": float(
                    np.sum(
                        pca.explained_variance_ratio_
                    )
                ),
                "geometry_sha256": geometry_sha,
                "sha256": {
                    "v47_0_protocol": actual_protocol_sha,
                    "gse130404_family_soft": sha256_file(
                        soft_path
                    ),
                    "eligible_gene_file": sha256_file(
                        eligible_path
                    ),
                    "top256_matrix": sha256_file(
                        matrix_out
                    ),
                    "top256_z": sha256_file(
                        z_out
                    ),
                    "features": sha256_file(
                        feature_out
                    ),
                    "geometry": geometry_sha,
                    "pca_loadings": sha256_file(
                        pca_loadings_out
                    ),
                    "pca_explained_variance": sha256_file(
                        ev_out
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

    print()
    print(
        summary_out.read_text(
            encoding="utf-8"
        )
    )

    print("Wrote:")
    for p in [
        matrix_out,
        z_out,
        feature_out,
        geometry_out,
        pca_loadings_out,
        ev_out,
        summary_out,
        manifest_out,
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
