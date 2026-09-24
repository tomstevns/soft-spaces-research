#!/usr/bin/env python3
"""
Soft Spaces / CML
v50 FINAL DIRECTIONAL-STRUCTURE TEST

Plain-language question
-----------------------
Do the frozen Soft-Spaces Top-256 gene panels contain unusually clear and
locally coherent "change directions" compared with ordinary high-variance
real-gene panels?

This is the FINAL planned CML structural test.

No clinical labels.
No classifier.
No rescue tuning.

Operational definition
----------------------
For each cohort independently:

1. Frozen Top-256 -> cohort z-standardization -> PCA-16.
2. For every patient, find k=10 nearest neighbors.
3. In that local neighborhood, calculate the dominant local direction
   (first eigenvector of the local covariance matrix).
4. Measure:

   A. LOCAL DIRECTIONAL CONCENTRATION
      lambda1 / sum(lambda)
      Higher = local variation is more concentrated into one direction.

   B. LOCAL DIRECTIONAL COHERENCE
      For neighboring patients, compare their dominant local directions
      using absolute cosine |v_i dot v_j|.
      Higher = nearby patients tend to "point the same way".

Primary rule
------------
For EACH of the three cohorts, BOTH observed metrics must exceed the
95th percentile of a structured real-gene NULL.

Structured NULL:
- real patients
- real genes
- real covariance
- rank 257..2048 by cohort variance
- draw 256 genes without replacement
- 500 NULL panels per cohort

Overall PASS requires all 6 cohort/metric gates to pass.

Then a patient-resampling robustness test is run:
- 100 resamples
- 80% of patients
- frozen observed Top-256
- 10 structured NULL panels per resample
- for both metrics in all three cohorts:
    median signed z > 1.644854
    fraction z > 0 >= 0.75

Full support requires BOTH:
- primary all-six-gates PASS
- robustness all-six-gates PASS

Regardless of result, v50 writes a final CML closure.
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


# ============================================================
# FROZEN PARAMETERS
# ============================================================

VERSION = "v50"

TOP_K = 256
CONTROL_RANK_START = 257
CONTROL_RANK_END = 2048

PCA_R = 16
KNN_K = 10

PRIMARY_NULL_N = 500
PRIMARY_SEED = 50010001

ROBUST_N_RESAMPLES = 100
ROBUST_NULL_PER_RESAMPLE = 10
ROBUST_SEED = 50020001

SUBSAMPLE_N = {
    "A": 76,   # GSE130404: 76/96
    "B": 108,  # GSE44589: 108/135
    "C": 47,   # GSE14671: 47/59
}

Z_THRESHOLD = 1.6448536269514722
POSITIVE_FRACTION_THRESHOLD = 0.75

URLS = {
    "GSE130404": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/"
        "GSE130nnn/GSE130404/soft/GSE130404_family.soft.gz"
    ),
    "GSE44589": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/"
        "GSE44nnn/GSE44589/soft/GSE44589_family.soft.gz"
    ),
    "GSE14671": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/"
        "GSE14nnn/GSE14671/soft/GSE14671_family.soft.gz"
    ),
}


# ============================================================
# UTILITIES
# ============================================================

def project_dir():
    return Path(__file__).resolve().parent.parent


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(
        json.dumps(obj, indent=2),
        encoding="utf-8",
    )


def download_if_missing(url, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists() and path.stat().st_size > 0:
        print("Using existing:", path)
        return

    print("Downloading:", url)
    print(" ->", path)

    urllib.request.urlretrieve(
        url,
        path,
    )


def empirical_high(null, observed):
    null = np.asarray(
        null,
        dtype=float,
    )

    return float(
        (
            1
            + np.sum(
                null >= observed
            )
        )
        / (
            1 + len(null)
        )
    )


def split_symbols(text):
    text = str(text).strip()

    if not text or text in {
        "---",
        "NA",
        "nan",
    }:
        return []

    parts = re.split(
        r"\s*///\s*|\s*//\s*|\s*;\s*|\s*,\s*",
        text,
    )

    out = []

    for p in parts:
        p = re.sub(
            r"\s+",
            "",
            p.strip(),
        )

        if p and p not in {
            "---",
            "NA",
        }:
            out.append(p)

    return out


# ============================================================
# GEO EXPRESSION PARSER
# ============================================================

def parse_soft_expression(
    soft_path,
    platform_id,
    selected_sample_ids,
    eligible_order,
):
    """
    Parse platform annotation + expression only.
    No phenotype or response metadata are used.
    """

    sample_order = list(
        map(
            str,
            selected_sample_ids,
        )
    )

    sample_set = set(
        sample_order
    )

    eligible_set = set(
        eligible_order
    )

    probe_to_symbols = {}

    sample_values = {
        gsm: {}
        for gsm in sample_order
    }

    current_entity = None
    current_id = None

    in_platform = False
    in_sample = False

    platform_header = None
    sample_header = None

    probe_idx = None
    symbol_idx = None
    pidx = None
    vidx = None

    with gzip.open(
        soft_path,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for raw in f:
            line = raw.rstrip(
                "\r\n"
            )

            if line.startswith(
                "^PLATFORM"
            ):
                current_entity = "PLATFORM"
                current_id = line.split(
                    "=",
                    1,
                )[1].strip()

                in_platform = False
                in_sample = False
                platform_header = None
                continue

            if line.startswith(
                "^SAMPLE"
            ):
                current_entity = "SAMPLE"
                current_id = line.split(
                    "=",
                    1,
                )[1].strip()

                in_platform = False
                in_sample = False
                sample_header = None
                continue

            if (
                current_entity == "PLATFORM"
                and current_id == platform_id
            ):

                if line == "!platform_table_begin":
                    in_platform = True
                    platform_header = None
                    continue

                if line == "!platform_table_end":
                    in_platform = False
                    continue

                if in_platform:
                    fields = line.split(
                        "\t"
                    )

                    if platform_header is None:
                        platform_header = fields

                        norm = [
                            x.strip().lower()
                            for x in fields
                        ]

                        probe_idx = (
                            norm.index("id")
                            if "id" in norm
                            else 0
                        )

                        for cand in (
                            "gene symbol",
                            "gene_symbol",
                            "symbol",
                        ):
                            if cand in norm:
                                symbol_idx = norm.index(
                                    cand
                                )
                                break

                        if symbol_idx is None:
                            for i, h in enumerate(
                                norm
                            ):
                                if "symbol" in h:
                                    symbol_idx = i
                                    break

                        if symbol_idx is None:
                            raise RuntimeError(
                                f"No gene-symbol column on {platform_id}."
                            )

                        continue

                    if len(fields) <= max(
                        probe_idx,
                        symbol_idx,
                    ):
                        continue

                    probe = fields[
                        probe_idx
                    ].strip()

                    symbols = [
                        s
                        for s in split_symbols(
                            fields[
                                symbol_idx
                            ]
                        )
                        if s in eligible_set
                    ]

                    if probe and symbols:
                        probe_to_symbols[
                            probe
                        ] = symbols

                    continue

            if (
                current_entity == "SAMPLE"
                and current_id in sample_set
            ):

                if line == "!sample_table_begin":
                    in_sample = True
                    sample_header = None
                    continue

                if line == "!sample_table_end":
                    in_sample = False
                    continue

                if in_sample:
                    fields = line.split(
                        "\t"
                    )

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

                        pidx = norm.index(
                            "id_ref"
                        )

                        vidx = norm.index(
                            "value"
                        )

                        continue

                    if len(fields) <= max(
                        pidx,
                        vidx,
                    ):
                        continue

                    probe = fields[
                        pidx
                    ].strip()

                    if probe not in probe_to_symbols:
                        continue

                    try:
                        value = float(
                            fields[
                                vidx
                            ].strip()
                        )
                    except ValueError:
                        continue

                    sample_values[
                        current_id
                    ][probe] = value

    symbol_to_probes = defaultdict(
        list
    )

    for probe, symbols in probe_to_symbols.items():
        for symbol in symbols:
            symbol_to_probes[
                symbol
            ].append(
                probe
            )

    measurable = [
        gene
        for gene in eligible_order
        if symbol_to_probes.get(
            gene
        )
    ]

    return (
        sample_values,
        symbol_to_probes,
        measurable,
    )


def build_gene_matrix(
    sample_values,
    symbol_to_probes,
    sample_order,
    gene_order,
):
    X = np.empty(
        (
            len(sample_order),
            len(gene_order),
        ),
        dtype=float,
    )

    for j, gene in enumerate(
        gene_order
    ):
        probes = sorted(
            set(
                symbol_to_probes.get(
                    gene,
                    [],
                )
            )
        )

        if not probes:
            raise RuntimeError(
                f"No probes for {gene}."
            )

        for i, gsm in enumerate(
            sample_order
        ):
            vals = [
                sample_values[gsm][p]
                for p in probes
                if p in sample_values[gsm]
            ]

            if not vals:
                raise RuntimeError(
                    f"Missing expression for {gsm}, {gene}."
                )

            X[
                i,
                j,
            ] = float(
                np.mean(
                    vals
                )
            )

    if not np.all(
        np.isfinite(
            X
        )
    ):
        raise RuntimeError(
            "Non-finite expression matrix."
        )

    return X


def matrix_for_gene_list(
    X_all,
    gene_to_col,
    genes,
):
    idx = [
        gene_to_col[g]
        for g in genes
    ]

    return X_all[
        :,
        idx,
    ]


# ============================================================
# DIRECTIONAL GEOMETRY
# ============================================================

def directional_metrics(X):
    """
    Returns two simple local-direction metrics:

    1. median local directional concentration:
       lambda1 / sum(lambda)

    2. median local directional coherence:
       median absolute cosine between local dominant direction
       vectors of neighboring patients.
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    if X.ndim != 2:
        raise RuntimeError(
            "Expected 2D matrix."
        )

    if X.shape[1] != TOP_K:
        raise RuntimeError(
            f"Expected {TOP_K} genes."
        )

    sd = np.std(
        X,
        axis=0,
        ddof=0,
    )

    if np.any(
        sd == 0
    ):
        raise RuntimeError(
            "Zero-SD coordinate."
        )

    Z = (
        X - np.mean(
            X,
            axis=0,
        )
    ) / sd

    pca = PCA(
        n_components=PCA_R,
        svd_solver="full",
    )

    Y = pca.fit_transform(
        Z
    )

    k_eff = min(
        KNN_K,
        len(Y) - 1,
    )

    nn = NearestNeighbors(
        n_neighbors=k_eff + 1,
        metric="euclidean",
    )

    nn.fit(Y)

    _, nbr = nn.kneighbors(
        Y
    )

    nbr = nbr[
        :,
        1:,
    ]

    n = len(Y)

    concentration = np.empty(
        n,
        dtype=float,
    )

    directions = np.empty(
        (
            n,
            PCA_R,
        ),
        dtype=float,
    )

    for i in range(
        n
    ):
        local = Y[
            nbr[i],
            :,
        ]

        local = (
            local
            - np.mean(
                local,
                axis=0,
            )
        )

        C = np.cov(
            local,
            rowvar=False,
            ddof=1,
        )

        eigvals, eigvecs = np.linalg.eigh(
            C
        )

        order = np.argsort(
            eigvals
        )[::-1]

        eigvals = np.maximum(
            eigvals[
                order
            ],
            0.0,
        )

        eigvecs = eigvecs[
            :,
            order,
        ]

        concentration[
            i
        ] = float(
            eigvals[0]
            / (
                eigvals.sum()
                + 1e-12
            )
        )

        v = eigvecs[
            :,
            0,
        ]

        v = (
            v
            / (
                np.linalg.norm(
                    v
                )
                + 1e-12
            )
        )

        directions[
            i,
            :,
        ] = v

    coherence_values = []

    seen = set()

    for i in range(
        n
    ):
        for j in nbr[
            i
        ]:
            a = min(
                i,
                int(
                    j
                ),
            )

            b = max(
                i,
                int(
                    j
                ),
            )

            key = (
                a,
                b,
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            coherence_values.append(
                abs(
                    float(
                        np.dot(
                            directions[
                                a
                            ],
                            directions[
                                b
                            ],
                        )
                    )
                )
            )

    coherence_values = np.asarray(
        coherence_values,
        dtype=float,
    )

    return {
        "median_concentration": float(
            np.median(
                concentration
            )
        ),
        "median_coherence": float(
            np.median(
                coherence_values
            )
        ),
        "mean_concentration": float(
            np.mean(
                concentration
            )
        ),
        "mean_coherence": float(
            np.mean(
                coherence_values
            )
        ),
        "pca16_cumulative_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    project = project_dir()

    docs = (
        project
        / "docs"
    )

    ds = (
        project
        / "results"
        / "direct_subspace"
    )

    docs.mkdir(
        parents=True,
        exist_ok=True,
    )

    ds.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Required previous artifacts
    # --------------------------------------------------------

    v49_manifest_path = (
        docs
        / "v49_4_CML_STRUCTURED_GEOMETRY_CLOSURE_manifest.json"
    )

    eligible_path = (
        ds
        / "v44_1b_eligible_genes.txt"
    )

    A_raw_path = (
        ds
        / "v47_2_GSE130404_label_blind_top256_expression.tsv"
    )

    A_features_path = (
        ds
        / "v47_2_label_blind_top256_features.tsv"
    )

    B_geometry_path = (
        ds
        / "v47_5b_GSE44589_label_blind_geometry.tsv"
    )

    B_features_path = (
        ds
        / "v47_5b_GSE44589_label_blind_top256_features.tsv"
    )

    C_raw_path = (
        ds
        / "v48_3b_GSE14671_label_blind_top256_expression.tsv"
    )

    C_features_path = (
        ds
        / "v48_3b_GSE14671_label_blind_top256_features.tsv"
    )

    required = [
        v49_manifest_path,
        eligible_path,
        A_raw_path,
        A_features_path,
        B_geometry_path,
        B_features_path,
        C_raw_path,
        C_features_path,
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                p
            )

    v49 = json.loads(
        v49_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if v49.get(
        "status"
    ) != "CLOSED":
        raise RuntimeError(
            "v49 is not formally closed."
        )

    # ========================================================
    # v50.0 — FREEZE FINAL TEST
    # ========================================================

    protocol = f"""Soft Spaces / CML
v50.0 — FINAL SPECIAL CHANGE-DIRECTION TEST PREREGISTRATION

STATUS
------
FROZEN BEFORE v50 ANALYSIS

PURPOSE
-------
This is the final planned CML structural test.

Plain-language question:

    Do the frozen Top-256 Soft-Spaces gene panels contain unusually clear
    and locally coherent "change directions" compared with ordinary
    high-variance real-gene panels?

NO CLINICAL LABELS
------------------
No response, prognosis or treatment label may be parsed or used.

COHORTS
-------
A = GSE130404
B = GSE44589
C = GSE14671

OBSERVED PANEL
--------------
Already-frozen Top-256 gene set for each cohort.

REPRESENTATION
--------------
Within each cohort:

1. z-standardize frozen Top-256
2. PCA-{PCA_R}
3. for each patient find k={KNN_K} nearest neighbors
4. compute local covariance among those neighbors
5. first eigenvector = dominant local change direction

PRIMARY METRIC 1
----------------
LOCAL DIRECTIONAL CONCENTRATION

    lambda1 / sum(lambda)

Use cohort median.

Higher means local variation is concentrated into one dominant direction.

PRIMARY METRIC 2
----------------
LOCAL DIRECTIONAL COHERENCE

For neighboring patients compare their dominant local directions:

    |v_i dot v_j|

Use median across unique neighboring patient pairs.

Higher means nearby patients tend to point in the same local direction.

STRUCTURED REAL-GENE NULL
-------------------------
For each cohort independently:

- rank all measurable eligible genes by raw variance
- observed ranks 1..256 remain frozen
- control pool = ranks {CONTROL_RANK_START}..{CONTROL_RANK_END}
- each NULL replicate draws {TOP_K} real genes without replacement
- real patient expression and gene-gene covariance remain intact

Primary NULL panels per cohort:
    {PRIMARY_NULL_N}

PRIMARY PASS RULE
-----------------
For EACH cohort and EACH metric:

    observed metric > NULL q95
    AND empirical one-sided p <= 0.05

There are 6 gates:
    A concentration
    A coherence
    B concentration
    B coherence
    C concentration
    C coherence

Overall primary PASS requires all 6 gates.

PATIENT-RESAMPLING ROBUSTNESS
-----------------------------
Resamples:
    {ROBUST_N_RESAMPLES}

Patient counts:
    A = {SUBSAMPLE_N["A"]}
    B = {SUBSAMPLE_N["B"]}
    C = {SUBSAMPLE_N["C"]}

Structured NULL panels per cohort per resample:
    {ROBUST_NULL_PER_RESAMPLE}

For each cohort and each metric:

    signed z =
        (observed - NULL mean) / NULL SD

Robustness gate requires:
    median signed z > {Z_THRESHOLD:.12f}
    AND fraction z > 0 >= {POSITIVE_FRACTION_THRESHOLD:.2f}

Overall robustness PASS requires all 6 gates.

FINAL SUPPORT RULE
------------------
SPECIAL LOCAL CHANGE-DIRECTION STRUCTURE SUPPORTED

ONLY IF:
    primary all-six-gates PASS
    AND robustness all-six-gates PASS

Otherwise:
    NOT SUPPORTED

NO RESCUE
---------
No post-hoc:
- Top-K changes
- PCA-rank changes
- kNN changes
- control-pool changes
- metric changes
- threshold changes
- cohort subsets

FINALITY
--------
Regardless of PASS or FAIL, v50 writes a final CML closure.

CLAIM LIMIT
-----------
A PASS would support only:

    unusually concentrated and locally coherent directions of transcriptomic
    variation in the frozen Top-256 panels compared with structured
    high-variance real-gene controls.

It would not establish:
- clinical prediction
- treatment response
- prognosis
- causal disease mechanism
- dormancy
- quantum biology
- quantum advantage
"""

    protocol_path = (
        docs
        / "v50_0_CML_FINAL_DIRECTION_TEST_PREREGISTRATION.txt"
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
        / "v50_0_CML_FINAL_DIRECTION_TEST_PREREGISTRATION_manifest.json"
    )

    write_json(
        protocol_manifest_path,
        {
            "version": "v50.0",
            "status": "FROZEN_BEFORE_ANALYSIS",
            "final_planned_cml_structural_test": True,
            "clinical_labels_used": False,
            "top_k": TOP_K,
            "pca_r": PCA_R,
            "knn_k": KNN_K,
            "control_rank_start": CONTROL_RANK_START,
            "control_rank_end": CONTROL_RANK_END,
            "primary_null_n": PRIMARY_NULL_N,
            "primary_seed": PRIMARY_SEED,
            "robust_n_resamples": ROBUST_N_RESAMPLES,
            "robust_null_per_resample": ROBUST_NULL_PER_RESAMPLE,
            "robust_seed": ROBUST_SEED,
            "protocol_sha256": protocol_sha,
            "source_sha256": {
                "v49_closure_manifest": sha256_file(
                    v49_manifest_path
                ),
                "eligible_genes": sha256_file(
                    eligible_path
                ),
            },
        },
    )

    print("=" * 72)
    print("v50.0 FINAL DIRECTION TEST PROTOCOL FROZEN")
    print("Protocol SHA:", protocol_sha)
    print("Clinical labels used: NO")
    print("=" * 72)

    # ========================================================
    # Load fixed sample IDs and feature sets
    # ========================================================

    A_raw_df = pd.read_csv(
        A_raw_path,
        sep="\t",
        index_col=0,
    )

    A_samples = [
        str(x)
        for x in A_raw_df.index
    ]

    A_frozen = pd.read_csv(
        A_features_path,
        sep="\t",
    )["gene"].astype(str).tolist()

    B_geometry_df = pd.read_csv(
        B_geometry_path,
        sep="\t",
        index_col=0,
    )

    B_samples = [
        str(x)
        for x in B_geometry_df.index
    ]

    B_frozen = pd.read_csv(
        B_features_path,
        sep="\t",
    )["gene"].astype(str).tolist()

    C_raw_df = pd.read_csv(
        C_raw_path,
        sep="\t",
        index_col=0,
    )

    C_samples = [
        str(x)
        for x in C_raw_df.index
    ]

    C_frozen = pd.read_csv(
        C_features_path,
        sep="\t",
    )["gene"].astype(str).tolist()

    if len(
        A_samples
    ) != 96:
        raise RuntimeError(
            "Unexpected GSE130404 sample count."
        )

    if len(
        B_samples
    ) != 135:
        raise RuntimeError(
            "Unexpected GSE44589 sample count."
        )

    if len(
        C_samples
    ) != 59:
        raise RuntimeError(
            "Unexpected GSE14671 sample count."
        )

    eligible = [
        x.strip()
        for x in eligible_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]

    if len(
        eligible
    ) != 15890:
        raise RuntimeError(
            "Eligible universe is not 15,890 genes."
        )

    cohort_specs = {
        "A": {
            "gse": "GSE130404",
            "platform": "GPL10558",
            "samples": A_samples,
            "frozen": A_frozen,
        },
        "B": {
            "gse": "GSE44589",
            "platform": "GPL570",
            "samples": B_samples,
            "frozen": B_frozen,
        },
        "C": {
            "gse": "GSE14671",
            "platform": "GPL570",
            "samples": C_samples,
            "frozen": C_frozen,
        },
    }

    # ========================================================
    # Reconstruct full matrices and frozen control pools
    # ========================================================

    matrices = {}
    gene_to_col = {}
    control_pools = {}
    integrity = {}

    print()
    print("Reconstructing real-gene matrices...")

    for key, spec in cohort_specs.items():
        gse = spec[
            "gse"
        ]

        soft_path = (
            project
            / "data"
            / "external"
            / gse
            / "metadata"
            / f"{gse}_family.soft.gz"
        )

        download_if_missing(
            URLS[
                gse
            ],
            soft_path,
        )

        values, symbol_to_probes, measurable = parse_soft_expression(
            soft_path,
            spec[
                "platform"
            ],
            spec[
                "samples"
            ],
            eligible,
        )

        X = build_gene_matrix(
            values,
            symbol_to_probes,
            spec[
                "samples"
            ],
            measurable,
        )

        var = np.var(
            X,
            axis=0,
            ddof=0,
        )

        order = np.argsort(
            -var,
            kind="mergesort",
        )

        ranked = [
            measurable[
                i
            ]
            for i in order
        ]

        reconstructed_top = ranked[
            :TOP_K
        ]

        set_match = (
            set(
                reconstructed_top
            )
            == set(
                spec[
                    "frozen"
                ]
            )
        )

        if not set_match:
            raise RuntimeError(
                f"{gse}: frozen Top256 integrity FAIL."
            )

        pool = ranked[
            CONTROL_RANK_START - 1:
            CONTROL_RANK_END
        ]

        if len(
            pool
        ) != (
            CONTROL_RANK_END
            - CONTROL_RANK_START
            + 1
        ):
            raise RuntimeError(
                f"{gse}: control pool size error."
            )

        matrices[
            key
        ] = X

        gene_to_col[
            key
        ] = {
            g: i
            for i, g in enumerate(
                measurable
            )
        }

        control_pools[
            key
        ] = pool

        integrity[
            key
        ] = {
            "gse": gse,
            "measurable_genes": len(
                measurable
            ),
            "frozen_top256_set_match": True,
        }

        print(
            gse,
            "| measurable",
            len(
                measurable
            ),
            "| Top256 integrity PASS",
        )

    # ========================================================
    # Observed metrics
    # ========================================================

    observed_matrices = {}

    observed = {}

    for key, spec in cohort_specs.items():
        observed_matrices[
            key
        ] = matrix_for_gene_list(
            matrices[
                key
            ],
            gene_to_col[
                key
            ],
            spec[
                "frozen"
            ],
        )

        observed[
            key
        ] = directional_metrics(
            observed_matrices[
                key
            ]
        )

    # ========================================================
    # PRIMARY STRUCTURED NULL
    # ========================================================

    print()
    print("v50.1 — PRIMARY SPECIAL-DIRECTION TEST")
    print("--------------------------------------")

    primary_rows = []

    primary_gates = {}

    rng = np.random.default_rng(
        PRIMARY_SEED
    )

    for key in [
        "A",
        "B",
        "C",
    ]:
        null_concentration = np.empty(
            PRIMARY_NULL_N,
            dtype=float,
        )

        null_coherence = np.empty(
            PRIMARY_NULL_N,
            dtype=float,
        )

        for b in range(
            PRIMARY_NULL_N
        ):
            genes = rng.choice(
                control_pools[
                    key
                ],
                size=TOP_K,
                replace=False,
            ).tolist()

            Xnull = matrix_for_gene_list(
                matrices[
                    key
                ],
                gene_to_col[
                    key
                ],
                genes,
            )

            m = directional_metrics(
                Xnull
            )

            null_concentration[
                b
            ] = m[
                "median_concentration"
            ]

            null_coherence[
                b
            ] = m[
                "median_coherence"
            ]

        obs_conc = observed[
            key
        ][
            "median_concentration"
        ]

        obs_coh = observed[
            key
        ][
            "median_coherence"
        ]

        conc_q95 = float(
            np.quantile(
                null_concentration,
                0.95,
            )
        )

        coh_q95 = float(
            np.quantile(
                null_coherence,
                0.95,
            )
        )

        conc_p = empirical_high(
            null_concentration,
            obs_conc,
        )

        coh_p = empirical_high(
            null_coherence,
            obs_coh,
        )

        conc_pass = bool(
            obs_conc > conc_q95
            and conc_p <= 0.05
        )

        coh_pass = bool(
            obs_coh > coh_q95
            and coh_p <= 0.05
        )

        primary_gates[
            key
        ] = {
            "concentration_pass": conc_pass,
            "coherence_pass": coh_pass,
        }

        for b in range(
            PRIMARY_NULL_N
        ):
            primary_rows.append({
                "cohort": key,
                "null_rep": b + 1,
                "null_median_concentration": null_concentration[
                    b
                ],
                "null_median_coherence": null_coherence[
                    b
                ],
            })

        observed[
            key
        ][
            "concentration_null_mean"
        ] = float(
            np.mean(
                null_concentration
            )
        )

        observed[
            key
        ][
            "concentration_q95"
        ] = conc_q95

        observed[
            key
        ][
            "concentration_p"
        ] = conc_p

        observed[
            key
        ][
            "coherence_null_mean"
        ] = float(
            np.mean(
                null_coherence
            )
        )

        observed[
            key
        ][
            "coherence_q95"
        ] = coh_q95

        observed[
            key
        ][
            "coherence_p"
        ] = coh_p

        print(
            key,
            cohort_specs[
                key
            ][
                "gse"
            ],
            "| concentration",
            f"{obs_conc:.6f}",
            "p",
            f"{conc_p:.6f}",
            "=>",
            "PASS" if conc_pass else "FAIL",
        )

        print(
            key,
            cohort_specs[
                key
            ][
                "gse"
            ],
            "| coherence",
            f"{obs_coh:.6f}",
            "p",
            f"{coh_p:.6f}",
            "=>",
            "PASS" if coh_pass else "FAIL",
        )

    primary_pass = bool(
        all(
            g[
                "concentration_pass"
            ]
            and g[
                "coherence_pass"
            ]
            for g in primary_gates.values()
        )
    )

    primary_null_out = (
        ds
        / "v50_1_directional_structure_null_500.tsv"
    )

    pd.DataFrame(
        primary_rows
    ).to_csv(
        primary_null_out,
        sep="\t",
        index=False,
    )

    # ========================================================
    # ROBUSTNESS
    # ========================================================

    print()
    print("v50.2 — PATIENT-RESAMPLING ROBUSTNESS")
    print("--------------------------------------")

    rng_master = np.random.default_rng(
        ROBUST_SEED
    )

    robustness_rows = []

    for r in range(
        ROBUST_N_RESAMPLES
    ):
        for key in [
            "A",
            "B",
            "C",
        ]:
            idx = rng_master.choice(
                observed_matrices[
                    key
                ].shape[0],
                size=SUBSAMPLE_N[
                    key
                ],
                replace=False,
            )

            Xobs_sub = observed_matrices[
                key
            ][
                idx,
                :,
            ]

            obs = directional_metrics(
                Xobs_sub
            )

            local_rng = np.random.default_rng(
                ROBUST_SEED
                + 100000
                + r * 10
                + ord(
                    key
                )
            )

            null_conc = np.empty(
                ROBUST_NULL_PER_RESAMPLE,
                dtype=float,
            )

            null_coh = np.empty(
                ROBUST_NULL_PER_RESAMPLE,
                dtype=float,
            )

            for b in range(
                ROBUST_NULL_PER_RESAMPLE
            ):
                genes = local_rng.choice(
                    control_pools[
                        key
                    ],
                    size=TOP_K,
                    replace=False,
                ).tolist()

                Xnull_full = matrix_for_gene_list(
                    matrices[
                        key
                    ],
                    gene_to_col[
                        key
                    ],
                    genes,
                )

                Xnull_sub = Xnull_full[
                    idx,
                    :,
                ]

                nm = directional_metrics(
                    Xnull_sub
                )

                null_conc[
                    b
                ] = nm[
                    "median_concentration"
                ]

                null_coh[
                    b
                ] = nm[
                    "median_coherence"
                ]

            sd_conc = float(
                np.std(
                    null_conc,
                    ddof=1,
                )
            )

            sd_coh = float(
                np.std(
                    null_coh,
                    ddof=1,
                )
            )

            if sd_conc <= 0 or sd_coh <= 0:
                raise RuntimeError(
                    "Zero NULL SD in robustness."
                )

            z_conc = float(
                (
                    obs[
                        "median_concentration"
                    ]
                    - np.mean(
                        null_conc
                    )
                )
                / sd_conc
            )

            z_coh = float(
                (
                    obs[
                        "median_coherence"
                    ]
                    - np.mean(
                        null_coh
                    )
                )
                / sd_coh
            )

            robustness_rows.append({
                "resample": r + 1,
                "cohort": key,
                "z_concentration": z_conc,
                "z_coherence": z_coh,
                "observed_concentration": obs[
                    "median_concentration"
                ],
                "observed_coherence": obs[
                    "median_coherence"
                ],
            })

        if (
            r + 1
        ) % 10 == 0:
            print(
                f"Completed {r+1}/{ROBUST_N_RESAMPLES} resamples"
            )

    robust_df = pd.DataFrame(
        robustness_rows
    )

    robustness_gates = {}

    for key in [
        "A",
        "B",
        "C",
    ]:
        d = robust_df[
            robust_df[
                "cohort"
            ] == key
        ]

        median_z_conc = float(
            d[
                "z_concentration"
            ].median()
        )

        median_z_coh = float(
            d[
                "z_coherence"
            ].median()
        )

        positive_conc = float(
            np.mean(
                d[
                    "z_concentration"
                ] > 0
            )
        )

        positive_coh = float(
            np.mean(
                d[
                    "z_coherence"
                ] > 0
            )
        )

        conc_pass = bool(
            median_z_conc > Z_THRESHOLD
            and positive_conc
            >= POSITIVE_FRACTION_THRESHOLD
        )

        coh_pass = bool(
            median_z_coh > Z_THRESHOLD
            and positive_coh
            >= POSITIVE_FRACTION_THRESHOLD
        )

        robustness_gates[
            key
        ] = {
            "median_z_concentration": median_z_conc,
            "positive_fraction_concentration": positive_conc,
            "concentration_pass": conc_pass,
            "median_z_coherence": median_z_coh,
            "positive_fraction_coherence": positive_coh,
            "coherence_pass": coh_pass,
        }

        print(
            key,
            "| concentration median z",
            f"{median_z_conc:.3f}",
            "positive",
            f"{positive_conc:.3f}",
            "=>",
            "PASS" if conc_pass else "FAIL",
        )

        print(
            key,
            "| coherence median z",
            f"{median_z_coh:.3f}",
            "positive",
            f"{positive_coh:.3f}",
            "=>",
            "PASS" if coh_pass else "FAIL",
        )

    robustness_pass = bool(
        all(
            g[
                "concentration_pass"
            ]
            and g[
                "coherence_pass"
            ]
            for g in robustness_gates.values()
        )
    )

    robust_out = (
        ds
        / "v50_2_directional_structure_resampling.tsv"
    )

    robust_df.to_csv(
        robust_out,
        sep="\t",
        index=False,
    )

    # ========================================================
    # FINAL CML CLOSURE
    # ========================================================

    full_support = bool(
        primary_pass
        and robustness_pass
    )

    final_structural_claim = (
        "SPECIAL LOCAL CHANGE-DIRECTION STRUCTURE SUPPORTED"
        if full_support
        else
        "SPECIAL LOCAL CHANGE-DIRECTION STRUCTURE NOT SUPPORTED"
    )

    closure_path = (
        docs
        / "v50_3_CML_FINAL_STRUCTURAL_CLOSURE.txt"
    )

    lines = []

    lines.append(
        "Soft Spaces / CML"
    )

    lines.append(
        "v50.3 — FINAL STRUCTURAL CLOSURE"
    )

    lines.append(
        ""
    )

    lines.append(
        "STATUS"
    )

    lines.append(
        "------"
    )

    lines.append(
        "CML STRUCTURAL EXPLORATION CLOSED"
    )

    lines.append(
        ""
    )

    lines.append(
        f"Protocol SHA: {protocol_sha}"
    )

    lines.append(
        ""
    )

    lines.append(
        "FINAL TEST QUESTION"
    )

    lines.append(
        "-------------------"
    )

    lines.append(
        "Do the frozen Top-256 panels show unusually concentrated and"
    )

    lines.append(
        "locally coherent change directions compared with structured"
    )

    lines.append(
        "high-variance real-gene controls?"
    )

    lines.append(
        ""
    )

    lines.append(
        "Clinical labels used: NO"
    )

    lines.append(
        ""
    )

    lines.append(
        "PRIMARY RESULTS"
    )

    lines.append(
        "---------------"
    )

    for key in [
        "A",
        "B",
        "C",
    ]:
        gse = cohort_specs[
            key
        ][
            "gse"
        ]

        lines.append(
            f"{gse}:"
        )

        lines.append(
            "  observed median concentration: "
            f"{observed[key]['median_concentration']:.9f}"
        )

        lines.append(
            "  concentration NULL q95:       "
            f"{observed[key]['concentration_q95']:.9f}"
        )

        lines.append(
            "  concentration p:              "
            f"{observed[key]['concentration_p']:.9f}"
        )

        lines.append(
            "  concentration gate:           "
            f"{'PASS' if primary_gates[key]['concentration_pass'] else 'FAIL'}"
        )

        lines.append(
            "  observed median coherence:    "
            f"{observed[key]['median_coherence']:.9f}"
        )

        lines.append(
            "  coherence NULL q95:           "
            f"{observed[key]['coherence_q95']:.9f}"
        )

        lines.append(
            "  coherence p:                  "
            f"{observed[key]['coherence_p']:.9f}"
        )

        lines.append(
            "  coherence gate:               "
            f"{'PASS' if primary_gates[key]['coherence_pass'] else 'FAIL'}"
        )

        lines.append(
            ""
        )

    lines.append(
        "v50.1 PRIMARY OVERALL:"
    )

    lines.append(
        f"    {'PASS' if primary_pass else 'FAIL'}"
    )

    lines.append(
        ""
    )

    lines.append(
        "ROBUSTNESS RESULTS"
    )

    lines.append(
        "------------------"
    )

    for key in [
        "A",
        "B",
        "C",
    ]:
        gse = cohort_specs[
            key
        ][
            "gse"
        ]

        rg = robustness_gates[
            key
        ]

        lines.append(
            f"{gse}:"
        )

        lines.append(
            "  concentration median z:       "
            f"{rg['median_z_concentration']:.9f}"
        )

        lines.append(
            "  concentration positive frac:  "
            f"{rg['positive_fraction_concentration']:.9f}"
        )

        lines.append(
            "  concentration robustness:     "
            f"{'PASS' if rg['concentration_pass'] else 'FAIL'}"
        )

        lines.append(
            "  coherence median z:           "
            f"{rg['median_z_coherence']:.9f}"
        )

        lines.append(
            "  coherence positive frac:      "
            f"{rg['positive_fraction_coherence']:.9f}"
        )

        lines.append(
            "  coherence robustness:         "
            f"{'PASS' if rg['coherence_pass'] else 'FAIL'}"
        )

        lines.append(
            ""
        )

    lines.append(
        "v50.2 ROBUSTNESS OVERALL:"
    )

    lines.append(
        f"    {'PASS' if robustness_pass else 'FAIL'}"
    )

    lines.append(
        ""
    )

    lines.append(
        "FINAL STRUCTURAL RESULT"
    )

    lines.append(
        "-----------------------"
    )

    lines.append(
        final_structural_claim
    )

    lines.append(
        ""
    )

    lines.append(
        "FINAL CML INTERPRETATION"
    )

    lines.append(
        "------------------------"
    )

    lines.append(
        "The CML work has found reproducible statistical structure in"
    )

    lines.append(
        "several analyses, but broad geometric similarity alone was not"
    )

    lines.append(
        "shown to be specific to the frozen Soft-Spaces panels under the"
    )

    lines.append(
        "structured real-gene controls of v49."
    )

    lines.append(
        ""
    )

    if full_support:
        lines.append(
            "The final v50 test supports one narrower remaining observation:"
        )

        lines.append(
            "the frozen panels contain unusually concentrated and locally"
        )

        lines.append(
            "coherent directions of variation relative to structured"
        )

        lines.append(
            "high-variance real-gene controls."
        )

        lines.append(
            ""
        )

        lines.append(
            "This is a structural statistical result only."
        )

    else:
        lines.append(
            "The final v50 test did not establish that the frozen panels"
        )

        lines.append(
            "contain special local change-direction structure beyond"
        )

        lines.append(
            "ordinary high-variance real-gene controls."
        )

        lines.append(
            ""
        )

        lines.append(
            "Accordingly, no distinctive CML Soft-Spaces structural claim"
        )

        lines.append(
            "is established by the present series of tests."
        )

    lines.append(
        ""
    )

    lines.append(
        "No clinical, causal, dormancy, quantum-biological or quantum-"
    )

    lines.append(
        "advantage claim follows from this work."
    )

    lines.append(
        ""
    )

    lines.append(
        "NO RESCUE / NO FURTHER CML TUNING"
    )

    lines.append(
        "---------------------------------"
    )

    lines.append(
        "This was prospectively designated the final planned structural"
    )

    lines.append(
        "CML test. The CML structural exploration is closed regardless"
    )

    lines.append(
        "of PASS or FAIL."
    )

    closure = "\n".join(
        lines
    ) + "\n"

    closure_path.write_text(
        closure,
        encoding="utf-8",
    )

    closure_sha = sha256_file(
        closure_path
    )

    manifest_path = (
        docs
        / "v50_3_CML_FINAL_STRUCTURAL_CLOSURE_manifest.json"
    )

    write_json(
        manifest_path,
        {
            "version": "v50.3",
            "status": "CML_STRUCTURAL_EXPLORATION_CLOSED",
            "final_planned_test": True,
            "clinical_labels_used": False,
            "primary_pass": primary_pass,
            "robustness_pass": robustness_pass,
            "full_support": full_support,
            "final_structural_claim": final_structural_claim,
            "primary_gates": primary_gates,
            "robustness_gates": robustness_gates,
            "observed": observed,
            "protocol_sha256": protocol_sha,
            "closure_sha256": closure_sha,
            "sha256": {
                "execution_script": sha256_file(
                    Path(__file__).resolve()
                ),
                "primary_null": sha256_file(
                    primary_null_out
                ),
                "robustness_results": sha256_file(
                    robust_out
                ),
                "closure": closure_sha,
            },
        },
    )

    summary_path = (
        ds
        / "v50_final_CML_summary.txt"
    )

    summary = f"""=== Soft Spaces / CML v50 FINAL SUMMARY ===

Final planned structural test:
    COMPLETED

Primary special-direction test:
    {"PASS" if primary_pass else "FAIL"}

Patient-resampling robustness:
    {"PASS" if robustness_pass else "FAIL"}

FINAL STRUCTURAL RESULT:
    {final_structural_claim}

CML STRUCTURAL EXPLORATION:
    CLOSED

Protocol SHA:
    {protocol_sha}

Closure SHA:
    {closure_sha}
"""

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("v50 FINAL CML STRUCTURAL TEST COMPLETE")
    print("=" * 72)
    print(summary)
    print("Closure:")
    print(" ", closure_path)
    print("Manifest:")
    print(" ", manifest_path)
    print("Summary:")
    print(" ", summary_path)


if __name__ == "__main__":
    main()
