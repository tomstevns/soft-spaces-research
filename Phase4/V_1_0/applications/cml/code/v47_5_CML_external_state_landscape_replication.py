#!/usr/bin/env python3
"""
Soft Spaces / CML
v47.5 — External state-landscape replication in GSE44589

One-run preregister-and-execute script.

Scientific question
-------------------
Does an INDEPENDENT CML cohort show non-random clinical-group separation
inside a state geometry constructed entirely without outcome labels?

Important
---------
This is NOT the v44/v46 predictive test.
No classifier is transported.
No classifier is fitted.

The external cohort builds its own label-blind landscape using the
same frozen construction principle as v47:

    frozen transport-aware eligible universe (15,890 genes)
    -> Top-256 raw variance, label-blind
    -> cohort z-standardization, label-blind
    -> PCA-16, label-blind
    -> freeze geometry
    -> only then overlay clinical labels
    -> 10,000 label permutations

External cohort
---------------
GSE44589 / GPL570

Baseline samples:
    135

Available external outcome labels:
    MMR: 69
    no MMR: 59
    unavailable: 7

Thus the label-overlay test uses 128 evaluable baseline samples.

Endpoint note
-------------
GSE44589 does NOT reproduce the exact GSE130404 3-month BCR-ABL1 endpoint.
This is therefore a cross-endpoint external state-landscape replication.
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


VERSION = "v47.5"

GSE = "GSE44589"
GPL = "GPL570"

ELIGIBLE_N = 15890
TOP_K = 256
PCA_R = 16
KNN_K = 10

EXPECTED_BASELINE_N = 135
EXPECTED_MMR_N = 69
EXPECTED_NO_MMR_N = 59
EXPECTED_UNAVAILABLE_N = 7
EXPECTED_EVALUABLE_N = 128

N_PERM = 10_000
SEED = 47050001

SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE44nnn/GSE44589/soft/GSE44589_family.soft.gz"
)


def project_dir():
    return Path(__file__).resolve().parent.parent


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_if_missing(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        print("Using existing:", path)
        return

    print("Downloading:", url)
    print(" ->", path)
    urllib.request.urlretrieve(url, path)


def normalize_ws(x):
    return re.sub(r"\s+", " ", str(x).strip())


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


def parse_family_soft(soft_path):
    """
    Parse GPL570 annotation, sample VALUE tables, and sample-specific metadata.

    Metadata are retained only to identify:
        - baseline vs post-6-week samples
        - MMR vs no-MMR vs unavailable

    They are NOT used in feature selection, standardization or PCA.
    """
    probe_to_symbols = {}
    sample_values = {}
    sample_meta = defaultdict(list)

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
                continue

            if line.startswith("^SAMPLE"):
                current_entity = "SAMPLE"
                current_id = line.split("=", 1)[1].strip()
                sample_values[current_id] = {}
                sample_meta[current_id] = []
                in_platform_table = False
                in_sample_table = False
                sample_header = None
                continue

            if current_entity == "SAMPLE":
                if (
                    line.startswith("!Sample_title")
                    or line.startswith("!Sample_source_name_ch1")
                    or line.startswith("!Sample_characteristics_ch1")
                    or line.startswith("!Sample_description")
                ):
                    if "=" in line:
                        sample_meta[current_id].append(
                            normalize_ws(
                                line.split("=", 1)[1]
                            )
                        )

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
                        norm = [
                            x.strip().lower()
                            for x in fields
                        ]

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
                                "Could not identify GPL570 gene-symbol column."
                            )

                        continue

                    if len(fields) <= max(
                        probe_col_idx,
                        symbol_col_idx,
                    ):
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
                        norm = [
                            x.strip().lower()
                            for x in fields
                        ]

                        if (
                            "id_ref" not in norm
                            or "value" not in norm
                        ):
                            raise RuntimeError(
                                f"{current_id}: missing ID_REF/VALUE."
                            )

                        sample_probe_idx = norm.index("id_ref")
                        sample_value_idx = norm.index("value")
                        continue

                    if len(fields) <= max(
                        sample_probe_idx,
                        sample_value_idx,
                    ):
                        continue

                    probe = fields[sample_probe_idx].strip()

                    try:
                        value = float(
                            fields[sample_value_idx].strip()
                        )
                    except ValueError:
                        continue

                    sample_values[current_id][probe] = value

    return probe_to_symbols, sample_values, sample_meta


def classify_sample(meta_lines):
    """
    Reproduce the corrected v42/v44 GSE44589 interpretation:
    baseline samples are distinct from the 6-week samples.

    Outcome categories:
        MMR
        NO_MMR
        UNAVAILABLE

    The parser is deliberately conservative. It requires sample-specific
    text and later verifies the frozen 135 / 69 / 59 / 7 counts.
    """
    text = " | ".join(meta_lines)
    t = text.lower()

    # Timing
    post_patterns = [
        r"\b6\s*week",
        r"\b6-week",
        r"\bweek\s*6\b",
        r"\bafter\s+6\s*week",
        r"\bpost[- ]?6\s*week",
    ]

    is_post = any(
        re.search(p, t)
        for p in post_patterns
    )

    # Explicit baseline / pretreatment language if available.
    baseline_patterns = [
        r"\bbaseline\b",
        r"\bpre[- ]?treatment\b",
        r"\bpretreatment\b",
        r"\bbefore treatment\b",
        r"\bdiagnosis\b",
    ]

    is_baseline_explicit = any(
        re.search(p, t)
        for p in baseline_patterns
    )

    # In this GEO series, samples not marked as 6-week samples belong to
    # the pretreatment arm. This is frozen from the corrected prior audit.
    is_baseline = (
        is_baseline_explicit
        or not is_post
    )

    # Outcome
    unavailable_patterns = [
        r"\bnot available\b",
        r"\bunavailable\b",
        r"\bunknown\b",
        r"\bna\b",
        r"\bn/a\b",
    ]

    no_mmr_patterns = [
        r"\bno\s*mmr\b",
        r"\bnon[- ]?mmr\b",
        r"\bwithout\s*mmr\b",
        r"\bnot\s+achiev(?:e|ed|ing).*mmr\b",
        r"\bfailed.*mmr\b",
    ]

    mmr_patterns = [
        r"\bmmr\b",
        r"\bmajor molecular response\b",
    ]

    if any(
        re.search(p, t)
        for p in unavailable_patterns
    ):
        outcome = "UNAVAILABLE"

    elif any(
        re.search(p, t)
        for p in no_mmr_patterns
    ):
        outcome = "NO_MMR"

    elif any(
        re.search(p, t)
        for p in mmr_patterns
    ):
        outcome = "MMR"

    else:
        outcome = "UNRESOLVED"

    return is_baseline, outcome, text


def centroid_distance(Y, y):
    c0 = np.mean(
        Y[y == 0],
        axis=0,
    )

    c1 = np.mean(
        Y[y == 1],
        axis=0,
    )

    return float(
        np.linalg.norm(
            c1 - c0
        )
    )


def group_geometry_stats(df, y):
    rows = []

    for value, name in [
        (0, "MMR"),
        (1, "NO_MMR"),
    ]:
        mask = (
            y == value
        )

        rows.append({
            "group": name,
            "n": int(mask.sum()),
            "mean_density_proxy": float(
                df.loc[mask, "density_proxy"].mean()
            ),
            "median_density_proxy": float(
                df.loc[mask, "density_proxy"].median()
            ),
            "mean_local_anisotropy": float(
                df.loc[mask, "local_anisotropy"].mean()
            ),
            "median_local_anisotropy": float(
                df.loc[mask, "local_anisotropy"].median()
            ),
            "mean_distance_to_centroid": float(
                df.loc[mask, "distance_to_centroid"].mean()
            ),
            "median_distance_to_centroid": float(
                df.loc[mask, "distance_to_centroid"].median()
            ),
        })

    return pd.DataFrame(rows)


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

    docs.mkdir(
        parents=True,
        exist_ok=True,
    )

    ds.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    v470 = (
        docs
        / "v47_0_CML_STATE_LANDSCAPE_PREREGISTRATION.txt"
    )

    v474 = (
        ds
        / "v47_4_manifest.json"
    )

    eligible_path = (
        ds
        / "v44_1b_eligible_genes.txt"
    )

    for p in [
        v470,
        v474,
        eligible_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    m474 = json.loads(
        v474.read_text(
            encoding="utf-8"
        )
    )

    if m474.get("status") != "STABILITY PASS":
        raise RuntimeError(
            "v47.4 is not recorded as STABILITY PASS."
        )

    # =========================================================
    # FREEZE v47.5 PROTOCOL BEFORE DOWNLOADING / SCORING
    # =========================================================

    protocol = f"""Soft Spaces / CML
v47.5 — EXTERNAL STATE-LANDSCAPE REPLICATION PREREGISTRATION

STATUS
------
FROZEN BEFORE EXTERNAL LANDSCAPE CONSTRUCTION

EXTERNAL COHORT
---------------
GSE44589 / GPL570

FROZEN SAMPLE INTERPRETATION
----------------------------
Baseline samples:
    {EXPECTED_BASELINE_N}

Evaluable clinical labels:
    MMR: {EXPECTED_MMR_N}
    no MMR: {EXPECTED_NO_MMR_N}

Outcome unavailable:
    {EXPECTED_UNAVAILABLE_N}

Total evaluable:
    {EXPECTED_EVALUABLE_N}

ENDPOINT LIMIT
--------------
This endpoint differs from the GSE130404 3-month BCR-ABL1 endpoint.

Therefore this is:
    CROSS-ENDPOINT EXTERNAL STATE-LANDSCAPE REPLICATION

It is NOT exact endpoint replication.

LANDSCAPE CONSTRUCTION
----------------------
Use all {EXPECTED_BASELINE_N} baseline samples to construct the landscape.

Outcome labels MUST NOT influence:
- gene selection
- normalization
- PCA fitting
- kNN geometry

Frozen construction:

1. restrict to the existing v44 transport-aware eligible universe:
       15,890 genes

2. select:
       Top-256 genes by raw variance across all baseline samples

3. standardize each selected gene:
       z = (x - external_baseline_mean) / external_baseline_sd
       ddof=0

4. construct:
       PCA-16

5. local geometry:
       kNN k=10
       density proxy
       local anisotropy
       distance to external-cohort centroid

6. write and SHA-freeze the complete external geometry

7. ONLY THEN attach MMR / no-MMR labels to the 128 evaluable patients

PRIMARY EXTERNAL TEST
---------------------
Euclidean distance between MMR and no-MMR group centroids in the
external label-blind PCA-16 geometry.

NULL
----
{N_PERM} random permutations of the 128 evaluable labels.

Empirical one-sided p:

    (1 + # NULL >= observed) / (1 + {N_PERM})

PRIMARY EXTERNAL PASS RULE
--------------------------
PASS if:

    empirical p <= 0.05

No additional performance condition is introduced.

SECONDARY DESCRIPTIVES
----------------------
For MMR and no-MMR report:
- local density proxy
- local anisotropy
- distance to cohort centroid

Also report:
- cumulative variance explained by PCA-16
- overlap of external Top-256 with GSE130404 v47.2 Top-256

The Top-256 overlap is descriptive only and is NOT part of the PASS rule.

NO RESCUE
---------
If the primary external test fails:
- no feature tuning
- no PCA-rank tuning
- no alternative k
- no label-driven subset selection
- no gene substitution

CLAIM LIMIT
-----------
A PASS would support cross-endpoint external replication of non-random
clinical-group organization in an independently constructed label-blind
CML expression landscape.

It would NOT establish:
- exact endpoint replication
- prediction
- clinical utility
- causal biology
- dormancy mechanism
- quantum biology
- quantum advantage
"""

    protocol_path = (
        docs
        / "v47_5_CML_EXTERNAL_STATE_LANDSCAPE_PREREGISTRATION.txt"
    )

    protocol_path.write_text(
        protocol,
        encoding="utf-8",
    )

    protocol_sha = sha256_file(
        protocol_path
    )

    protocol_manifest_path = (
        docs
        / "v47_5_CML_EXTERNAL_STATE_LANDSCAPE_PREREGISTRATION_manifest.json"
    )

    protocol_manifest_path.write_text(
        json.dumps(
            {
                "version": VERSION,
                "status": "FROZEN_BEFORE_EXTERNAL_LANDSCAPE",
                "external_cohort": GSE,
                "platform": GPL,
                "baseline_n": EXPECTED_BASELINE_N,
                "mmr_n": EXPECTED_MMR_N,
                "no_mmr_n": EXPECTED_NO_MMR_N,
                "unavailable_n": EXPECTED_UNAVAILABLE_N,
                "evaluable_n": EXPECTED_EVALUABLE_N,
                "eligible_universe_n": ELIGIBLE_N,
                "top_k": TOP_K,
                "pca_r": PCA_R,
                "knn_k": KNN_K,
                "permutations": N_PERM,
                "random_seed": SEED,
                "primary_pass_p": 0.05,
                "endpoint_relation": "cross-endpoint",
                "protocol_sha256": protocol_sha,
                "source_sha256": {
                    "v47_0_protocol": sha256_file(v470),
                    "v47_4_manifest": sha256_file(v474),
                    "eligible_genes": sha256_file(eligible_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== v47.5 EXTERNAL STATE-LANDSCAPE REPLICATION ===")
    print("Protocol frozen BEFORE external construction.")
    print("Protocol SHA:", protocol_sha)
    print()

    soft_path = (
        data_dir
        / "GSE44589_family.soft.gz"
    )

    download_if_missing(
        SOFT_URL,
        soft_path,
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

    eligible_set = set(
        eligible
    )

    print("Parsing GSE44589...")
    probe_to_symbols, sample_values, sample_meta = parse_family_soft(
        soft_path
    )

    sample_class = {}

    for gsm in sorted(
        sample_values
    ):
        sample_class[gsm] = classify_sample(
            sample_meta[gsm]
        )

    baseline = [
        gsm
        for gsm, (is_baseline, outcome, txt)
        in sample_class.items()
        if is_baseline
    ]

    counts = defaultdict(int)

    for gsm in baseline:
        counts[
            sample_class[gsm][1]
        ] += 1

    print("Baseline samples:", len(baseline))
    print("MMR:", counts["MMR"])
    print("NO_MMR:", counts["NO_MMR"])
    print("UNAVAILABLE:", counts["UNAVAILABLE"])
    print("UNRESOLVED:", counts["UNRESOLVED"])

    if len(baseline) != EXPECTED_BASELINE_N:
        raise RuntimeError(
            f"Baseline count mismatch: {len(baseline)} != {EXPECTED_BASELINE_N}"
        )

    if counts["MMR"] != EXPECTED_MMR_N:
        raise RuntimeError(
            f"MMR count mismatch: {counts['MMR']} != {EXPECTED_MMR_N}"
        )

    if counts["NO_MMR"] != EXPECTED_NO_MMR_N:
        raise RuntimeError(
            f"NO_MMR count mismatch: {counts['NO_MMR']} != {EXPECTED_NO_MMR_N}"
        )

    if counts["UNAVAILABLE"] != EXPECTED_UNAVAILABLE_N:
        raise RuntimeError(
            f"UNAVAILABLE count mismatch: {counts['UNAVAILABLE']} != {EXPECTED_UNAVAILABLE_N}"
        )

    if counts["UNRESOLVED"] != 0:
        unresolved = [
            gsm for gsm in baseline
            if sample_class[gsm][1] == "UNRESOLVED"
        ]

        raise RuntimeError(
            "Unresolved baseline outcomes: "
            + ", ".join(unresolved)
        )

    symbol_to_probes = defaultdict(list)

    for probe, symbols in probe_to_symbols.items():
        for symbol in symbols:
            if symbol in eligible_set:
                symbol_to_probes[symbol].append(
                    probe
                )

    measurable_genes = [
        g
        for g in eligible
        if symbol_to_probes.get(g)
    ]

    if len(measurable_genes) < TOP_K:
        raise RuntimeError(
            "Fewer than 256 transport-aware genes measurable on GPL570."
        )

    baseline = sorted(
        baseline
    )

    X = np.empty(
        (
            len(baseline),
            len(measurable_genes),
        ),
        dtype=float,
    )

    missing = []

    for j, gene in enumerate(
        measurable_genes
    ):
        probes = sorted(
            set(
                symbol_to_probes[gene]
            )
        )

        for i, gsm in enumerate(
            baseline
        ):
            vals = [
                sample_values[gsm][p]
                for p in probes
                if p in sample_values[gsm]
            ]

            if not vals:
                X[i, j] = np.nan
                missing.append(
                    (gsm, gene)
                )
            else:
                X[i, j] = float(
                    np.mean(vals)
                )

    if missing:
        raise RuntimeError(
            "Missing expression values. First examples: "
            + repr(
                missing[:10]
            )
        )

    if not np.all(
        np.isfinite(X)
    ):
        raise RuntimeError(
            "Non-finite external expression matrix."
        )

    # ---------------------------------------------------------
    # LABEL-BLIND EXTERNAL LANDSCAPE
    # ---------------------------------------------------------

    variances = np.var(
        X,
        axis=0,
        ddof=0,
    )

    order = np.argsort(
        -variances,
        kind="mergesort",
    )

    top_idx = order[:TOP_K]

    top_genes = [
        measurable_genes[i]
        for i in top_idx
    ]

    X256 = X[
        :,
        top_idx,
    ]

    top_var = variances[
        top_idx
    ]

    mu = np.mean(
        X256,
        axis=0,
    )

    sd = np.std(
        X256,
        axis=0,
        ddof=0,
    )

    zero = np.where(
        sd == 0
    )[0]

    if len(zero):
        raise RuntimeError(
            "Zero-SD external Top256 genes: "
            + ", ".join(
                top_genes[i]
                for i in zero
            )
        )

    Z = (
        X256 - mu
    ) / sd

    pca = PCA(
        n_components=PCA_R,
        svd_solver="full",
    )

    Y = pca.fit_transform(
        Z
    )

    nn = NearestNeighbors(
        n_neighbors=KNN_K + 1,
        metric="euclidean",
    )

    nn.fit(
        Y
    )

    distances, indices = nn.kneighbors(
        Y
    )

    nbr_dist = distances[
        :,
        1:,
    ]

    nbr_idx = indices[
        :,
        1:,
    ]

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
        len(baseline),
        dtype=float,
    )

    for i in range(
        len(baseline)
    ):
        local = Y[
            nbr_idx[i],
            :,
        ]

        C = np.cov(
            local,
            rowvar=False,
            ddof=1,
        )

        eig = np.linalg.eigvalsh(
            C
        )

        eig = np.maximum(
            eig,
            0.0,
        )[::-1]

        anisotropy[i] = float(
            eig[0]
            / (
                eig.sum()
                + eps
            )
        )

    pc_cols = [
        f"PC{i}"
        for i in range(
            1,
            PCA_R + 1,
        )
    ]

    geometry = pd.DataFrame(
        Y,
        index=baseline,
        columns=pc_cols,
    )

    geometry[
        "mean_knn_distance"
    ] = mean_knn_distance

    geometry[
        "density_proxy"
    ] = density_proxy

    geometry[
        "local_anisotropy"
    ] = anisotropy

    geometry[
        "distance_to_centroid"
    ] = distance_to_centroid

    geometry.index.name = "gsm"

    geometry_out = (
        ds
        / "v47_5_GSE44589_label_blind_geometry.tsv"
    )

    geometry.to_csv(
        geometry_out,
        sep="\t",
    )

    geometry_sha = sha256_file(
        geometry_out
    )

    feature_out = (
        ds
        / "v47_5_GSE44589_label_blind_top256_features.tsv"
    )

    pd.DataFrame({
        "gene": top_genes,
        "raw_variance": top_var,
        "cohort_mean": mu,
        "cohort_sd_ddof0": sd,
    }).to_csv(
        feature_out,
        sep="\t",
        index=False,
    )

    pca_loadings_out = (
        ds
        / "v47_5_GSE44589_PCA16_loadings.tsv"
    )

    pd.DataFrame(
        pca.components_.T,
        index=top_genes,
        columns=pc_cols,
    ).rename_axis(
        "gene"
    ).to_csv(
        pca_loadings_out,
        sep="\t",
    )

    # Freeze complete external geometry before overlay.
    external_geometry_manifest = (
        ds
        / "v47_5_GSE44589_label_blind_geometry_manifest.json"
    )

    external_geometry_manifest.write_text(
        json.dumps(
            {
                "version": VERSION,
                "cohort": GSE,
                "status": "EXTERNAL_LABEL_BLIND_GEOMETRY_FROZEN",
                "baseline_n": len(baseline),
                "eligible_universe_n": len(eligible),
                "measurable_eligible_n": len(measurable_genes),
                "top_k": TOP_K,
                "pca_r": PCA_R,
                "knn_k": KNN_K,
                "labels_used_for_geometry": False,
                "pca16_cumulative_variance": float(
                    np.sum(
                        pca.explained_variance_ratio_
                    )
                ),
                "geometry_sha256": geometry_sha,
                "protocol_sha256": protocol_sha,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # ONLY NOW: CLINICAL LABEL OVERLAY
    # ---------------------------------------------------------

    evaluable = [
        gsm
        for gsm in baseline
        if sample_class[gsm][1]
        in {
            "MMR",
            "NO_MMR",
        }
    ]

    if len(evaluable) != EXPECTED_EVALUABLE_N:
        raise RuntimeError(
            f"Evaluable count mismatch: {len(evaluable)} != {EXPECTED_EVALUABLE_N}"
        )

    geometry_eval = geometry.loc[
        evaluable
    ].copy()

    y = np.asarray(
        [
            0
            if sample_class[gsm][1] == "MMR"
            else 1
            for gsm in evaluable
        ],
        dtype=int,
    )

    if int(np.sum(y == 0)) != EXPECTED_MMR_N:
        raise RuntimeError(
            "MMR overlay count mismatch."
        )

    if int(np.sum(y == 1)) != EXPECTED_NO_MMR_N:
        raise RuntimeError(
            "NO_MMR overlay count mismatch."
        )

    Y_eval = geometry_eval[
        pc_cols
    ].to_numpy(
        dtype=float
    )

    observed = centroid_distance(
        Y_eval,
        y,
    )

    rng = np.random.default_rng(
        SEED
    )

    null = np.empty(
        N_PERM,
        dtype=float,
    )

    for i in range(
        N_PERM
    ):
        yp = rng.permutation(
            y
        )

        null[i] = centroid_distance(
            Y_eval,
            yp,
        )

    exceed = int(
        np.sum(
            null >= observed
        )
    )

    p_emp = float(
        (1 + exceed)
        / (1 + N_PERM)
    )

    null_mean = float(
        np.mean(
            null
        )
    )

    null_sd = float(
        np.std(
            null,
            ddof=1,
        )
    )

    null_q025 = float(
        np.quantile(
            null,
            0.025,
        )
    )

    null_median = float(
        np.quantile(
            null,
            0.5,
        )
    )

    null_q975 = float(
        np.quantile(
            null,
            0.975,
        )
    )

    null_z = float(
        (
            observed
            - null_mean
        )
        / null_sd
    )

    primary_pass = (
        p_emp <= 0.05
    )

    status = (
        "EXTERNAL LANDSCAPE REPLICATION PASS"
        if primary_pass
        else "EXTERNAL LANDSCAPE REPLICATION FAIL"
    )

    group_stats = group_geometry_stats(
        geometry_eval,
        y,
    )

    group_stats_out = (
        ds
        / "v47_5_GSE44589_group_geometry_descriptives.tsv"
    )

    group_stats.to_csv(
        group_stats_out,
        sep="\t",
        index=False,
    )

    overlay = geometry_eval.copy()

    overlay.insert(
        0,
        "clinical_group",
        np.where(
            y == 0,
            "MMR",
            "NO_MMR",
        ),
    )

    overlay_out = (
        ds
        / "v47_5_GSE44589_geometry_with_labels.tsv"
    )

    overlay.to_csv(
        overlay_out,
        sep="\t",
    )

    null_out = (
        ds
        / "v47_5_GSE44589_centroid_null_10000.tsv"
    )

    pd.DataFrame({
        "permutation": np.arange(
            1,
            N_PERM + 1,
        ),
        "centroid_distance": null,
    }).to_csv(
        null_out,
        sep="\t",
        index=False,
    )

    # Descriptive feature overlap with development v47.2.
    dev_feature_path = (
        ds
        / "v47_2_label_blind_top256_features.tsv"
    )

    if not dev_feature_path.exists():
        raise FileNotFoundError(
            dev_feature_path
        )

    dev_features = set(
        pd.read_csv(
            dev_feature_path,
            sep="\t",
        )["gene"].astype(str)
    )

    ext_features = set(
        top_genes
    )

    overlap_n = len(
        dev_features.intersection(
            ext_features
        )
    )

    overlap_jaccard = (
        overlap_n
        / (
            len(dev_features)
            + len(ext_features)
            - overlap_n
        )
    )

    summary_out = (
        ds
        / "v47_5_external_state_landscape_replication_summary.txt"
    )

    lines = [
        "=== Soft Spaces / CML v47.5 EXTERNAL STATE-LANDSCAPE REPLICATION ===",
        "",
        "PROTOCOL",
        "--------",
        f"v47.5 protocol SHA:              {protocol_sha}",
        f"External cohort:                 {GSE}",
        f"Platform:                        {GPL}",
        "Endpoint relation:               CROSS-ENDPOINT",
        "",
        "EXTERNAL LABEL-BLIND LANDSCAPE",
        "------------------------------",
        f"Baseline samples used:           {len(baseline)}",
        f"Transport-aware universe:        {len(eligible)}",
        f"Measurable eligible genes:       {len(measurable_genes)}",
        f"Top-K variance genes:            {TOP_K}",
        f"PCA dimensions:                  {PCA_R}",
        f"PCA16 cumulative variance:       {np.sum(pca.explained_variance_ratio_):.6f}",
        f"kNN k:                           {KNN_K}",
        "Outcome labels used in geometry: NO",
        f"External geometry SHA256:        {geometry_sha}",
        "",
        "CLINICAL OVERLAY",
        "----------------",
        f"MMR:                             {int(np.sum(y == 0))}",
        f"NO_MMR:                          {int(np.sum(y == 1))}",
        f"Evaluable:                       {len(y)}",
        f"Outcome unavailable:             {EXPECTED_UNAVAILABLE_N}",
        "",
        "PRIMARY EXTERNAL TEST",
        "---------------------",
        f"Observed centroid distance:      {observed:.9f}",
        f"NULL mean:                       {null_mean:.9f}",
        f"NULL SD:                         {null_sd:.9f}",
        f"NULL q2.5%:                      {null_q025:.9f}",
        f"NULL median:                     {null_median:.9f}",
        f"NULL q97.5%:                     {null_q975:.9f}",
        f"Observed - NULL mean:            {observed - null_mean:+.9f}",
        f"Observed NULL-z:                 {null_z:+.6f}",
        f"NULL >= observed:                {exceed}",
        f"Empirical one-sided p:           {p_emp:.9f}",
        "",
        "FEATURE OVERLAP — DESCRIPTIVE ONLY",
        "----------------------------------",
        f"GSE130404 Top256 ∩ GSE44589 Top256: {overlap_n}/256",
        f"Top256 Jaccard:                  {overlap_jaccard:.6f}",
        "",
        "SECONDARY GROUP GEOMETRY",
        "------------------------",
    ]

    for _, row in group_stats.iterrows():
        lines += [
            f"{row['group']}:",
            f"  n:                              {int(row['n'])}",
            f"  mean density proxy:             {row['mean_density_proxy']:.9f}",
            f"  mean local anisotropy:          {row['mean_local_anisotropy']:.9f}",
            f"  mean distance to centroid:      {row['mean_distance_to_centroid']:.9f}",
        ]

    lines += [
        "",
        f"v47.5 STATUS: {status}",
        "",
        "INTERPRETATION LIMIT",
        "--------------------",
        "This is an independent-cohort, cross-endpoint replication test of",
        "non-random clinical-group organization inside an independently",
        "constructed label-blind CML expression landscape.",
        "",
        "It is not exact endpoint replication and does not establish",
        "prediction, prognosis, clinical utility, causal biology, dormancy",
        "mechanism, quantum biology, or quantum advantage.",
    ]

    summary_out.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest_out = (
        ds
        / "v47_5_manifest.json"
    )

    manifest_out.write_text(
        json.dumps(
            {
                "version": VERSION,
                "status": status,
                "cohort": GSE,
                "platform": GPL,
                "endpoint_relation": "cross-endpoint",
                "baseline_n": len(baseline),
                "evaluable_n": len(y),
                "mmr_n": int(
                    np.sum(y == 0)
                ),
                "no_mmr_n": int(
                    np.sum(y == 1)
                ),
                "unavailable_n": EXPECTED_UNAVAILABLE_N,
                "eligible_universe_n": len(eligible),
                "measurable_eligible_n": len(measurable_genes),
                "top_k": TOP_K,
                "pca_r": PCA_R,
                "pca16_cumulative_variance": float(
                    np.sum(
                        pca.explained_variance_ratio_
                    )
                ),
                "knn_k": KNN_K,
                "labels_used_for_geometry": False,
                "observed_centroid_distance": observed,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "null_z": null_z,
                "n_permutations": N_PERM,
                "random_seed": SEED,
                "null_ge_observed": exceed,
                "empirical_one_sided_p": p_emp,
                "primary_pass_threshold": 0.05,
                "primary_pass": bool(
                    primary_pass
                ),
                "top256_overlap_with_gse130404": overlap_n,
                "top256_jaccard_with_gse130404": overlap_jaccard,
                "protocol_sha256": protocol_sha,
                "external_geometry_sha256": geometry_sha,
                "sha256": {
                    "gse44589_family_soft": sha256_file(
                        soft_path
                    ),
                    "external_features": sha256_file(
                        feature_out
                    ),
                    "external_geometry": geometry_sha,
                    "external_pca_loadings": sha256_file(
                        pca_loadings_out
                    ),
                    "overlay": sha256_file(
                        overlay_out
                    ),
                    "group_descriptives": sha256_file(
                        group_stats_out
                    ),
                    "null_distribution": sha256_file(
                        null_out
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
        protocol_path,
        protocol_manifest_path,
        feature_out,
        geometry_out,
        external_geometry_manifest,
        pca_loadings_out,
        overlay_out,
        group_stats_out,
        null_out,
        summary_out,
        manifest_out,
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
