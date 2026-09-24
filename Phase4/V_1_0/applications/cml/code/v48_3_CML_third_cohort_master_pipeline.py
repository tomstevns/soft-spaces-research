#!/usr/bin/env python3
"""
Soft Spaces / CML
v48.3 MASTER PIPELINE — third-cohort geometric replication

One file, sequential execution, checkpoint-style outputs.

NEW DATASET
-----------
GSE14671 / GPL570
All 59 samples are used ONLY as a label-blind expression cohort.

NO clinical response labels are parsed or used anywhere.

PIPELINE
--------
v48.3a  technical audit / matrix construction
v48.3b  third-cohort label-blind Top256 -> PCA16 geometry
v48.3c  pairwise replication:
          GSE14671 vs GSE130404
          GSE14671 vs GSE44589
v48.3d  paired patient-resampling robustness
v48.3e  joint three-cohort invariance test
v48.3f  automatic closure

The COMPLETE protocol and all stop/pass rules are frozen before any
GSE14671 geometry result is calculated.

This is not a rescue of v47.
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
# FROZEN MASTER PARAMETERS
# ============================================================

MASTER_VERSION = "v48.3"

TOP_K = 256
PCA_R = 16
KNN_K = 10

THIRD_GSE = "GSE14671"
THIRD_GPL = "GPL570"
THIRD_EXPECTED_N = 59

PAIRWISE_NULL_N = 1000
PAIRWISE_SEED = 48310001

ROBUST_N_RESAMPLES = 100
ROBUST_NULL_PER_RESAMPLE = 20
ROBUST_DEV_N = 76       # 80% of 96
ROBUST_EXT_N = 108      # 80% of 135
ROBUST_THIRD_N = 47     # ~80% of 59
ROBUST_SEED = 48320001

JOINT_NULL_N = 500
JOINT_SEED = 48330001

Z_THRESHOLD = 1.6448536269514722
POSITIVE_FRACTION_THRESHOLD = 0.75

THIRD_SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE14nnn/GSE14671/soft/GSE14671_family.soft.gz"
)


# ============================================================
# GENERIC UTILITIES
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
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        print("Using existing:", path)
        return

    print("Downloading:", url)
    print(" ->", path)
    urllib.request.urlretrieve(url, path)


def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    den = np.linalg.norm(a) * np.linalg.norm(b)

    if den <= 0:
        raise RuntimeError("Zero norm in cosine similarity.")

    return float(np.dot(a, b) / den)


def wasserstein_1d(x, y):
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))

    if len(x) == 0 or len(y) == 0:
        raise RuntimeError("Empty distribution.")

    n = max(len(x), len(y))
    q = (np.arange(n, dtype=float) + 0.5) / n

    qx = np.quantile(x, q, method="linear")
    qy = np.quantile(y, q, method="linear")

    return float(np.mean(np.abs(qx - qy)))


def pairwise_upper(Y):
    Y = np.asarray(Y, dtype=float)

    diff = Y[:, None, :] - Y[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))

    iu = np.triu_indices(len(Y), k=1)

    return D[iu]


def local_anisotropy(Y, k=KNN_K):
    Y = np.asarray(Y, dtype=float)

    k_eff = min(k, len(Y) - 1)

    nn = NearestNeighbors(
        n_neighbors=k_eff + 1,
        metric="euclidean",
    )
    nn.fit(Y)

    _, idx = nn.kneighbors(Y)

    out = np.empty(len(Y), dtype=float)
    eps = 1e-12

    for i in range(len(Y)):
        local = Y[idx[i, 1:], :]
        C = np.cov(local, rowvar=False, ddof=1)

        eig = np.linalg.eigvalsh(C)
        eig = np.maximum(eig, 0.0)[::-1]

        out[i] = float(
            eig[0] / (eig.sum() + eps)
        )

    return out


def geometry_signature(X):
    """
    Cohort-internal label-blind geometry.

    X is raw expression with exactly the cohort's frozen Top256 features.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise RuntimeError("Expected 2D matrix.")

    if X.shape[1] != TOP_K:
        raise RuntimeError(
            f"Expected {TOP_K} features; got {X.shape[1]}."
        )

    sd = np.std(X, axis=0, ddof=0)

    if np.any(sd == 0):
        raise RuntimeError("Zero-SD Top256 coordinate.")

    Z = (
        X - np.mean(X, axis=0)
    ) / sd

    pca = PCA(
        n_components=PCA_R,
        svd_solver="full",
    )

    Y = pca.fit_transform(Z)

    spectrum = pca.explained_variance_.astype(float)
    spectrum /= spectrum.sum()

    pdist = pairwise_upper(Y)

    median_distance = float(
        np.median(pdist)
    )

    if median_distance <= 0:
        raise RuntimeError(
            "Non-positive median pairwise distance."
        )

    pdist_norm = pdist / median_distance

    return {
        "Y": Y,
        "spectrum": spectrum,
        "pdist_norm": pdist_norm,
        "anisotropy": local_anisotropy(Y),
        "cumvar": float(
            pca.explained_variance_ratio_.sum()
        ),
        "effdim": float(
            1.0 / np.sum(spectrum * spectrum)
        ),
    }


def pair_metrics(sig_a, sig_b):
    return {
        "spectrum_cosine": cosine(
            sig_a["spectrum"],
            sig_b["spectrum"],
        ),
        "distance_w1": wasserstein_1d(
            sig_a["pdist_norm"],
            sig_b["pdist_norm"],
        ),
        "anisotropy_w1": wasserstein_1d(
            sig_a["anisotropy"],
            sig_b["anisotropy"],
        ),
        "effdim_abs_diff": abs(
            sig_a["effdim"] - sig_b["effdim"]
        ),
    }


def empirical_high(null, observed):
    null = np.asarray(null, dtype=float)

    return float(
        (
            1
            + np.sum(null >= observed)
        )
        / (
            1 + len(null)
        )
    )


def empirical_low(null, observed):
    null = np.asarray(null, dtype=float)

    return float(
        (
            1
            + np.sum(null <= observed)
        )
        / (
            1 + len(null)
        )
    )


def split_symbols(text):
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


# ============================================================
# GEO SOFT PARSER — EXPRESSION ONLY, NO PHENOTYPES
# ============================================================

def parse_soft_expression(
    soft_path,
    platform_id,
    selected_sample_ids=None,
    selected_genes=None,
):
    """
    Parse:
    - platform gene-symbol annotation
    - sample ID_REF/VALUE tables

    Explicitly ignores phenotype / clinical metadata.
    """

    sample_filter = (
        None
        if selected_sample_ids is None
        else set(map(str, selected_sample_ids))
    )

    gene_filter = (
        None
        if selected_genes is None
        else set(map(str, selected_genes))
    )

    probe_to_symbols = {}
    sample_values = {}

    current_entity = None
    current_id = None

    in_platform = False
    in_sample = False

    platform_header = None
    sample_header = None

    probe_idx = None
    symbol_idx = None
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

                in_platform = False
                in_sample = False
                platform_header = None
                continue

            if line.startswith("^SAMPLE"):
                current_entity = "SAMPLE"
                current_id = line.split("=", 1)[1].strip()

                in_platform = False
                in_sample = False
                sample_header = None

                if (
                    sample_filter is None
                    or current_id in sample_filter
                ):
                    sample_values[current_id] = {}

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
                    fields = line.split("\t")

                    if platform_header is None:
                        platform_header = fields

                        norm = [
                            x.strip().lower()
                            for x in fields
                        ]

                        if "id" in norm:
                            probe_idx = norm.index("id")
                        else:
                            probe_idx = 0

                        for cand in (
                            "gene symbol",
                            "gene_symbol",
                            "symbol",
                        ):
                            if cand in norm:
                                symbol_idx = norm.index(cand)
                                break

                        if symbol_idx is None:
                            for i, h in enumerate(norm):
                                if "symbol" in h:
                                    symbol_idx = i
                                    break

                        if symbol_idx is None:
                            raise RuntimeError(
                                f"Cannot find symbol column on {platform_id}."
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

                    symbols = split_symbols(
                        fields[
                            symbol_idx
                        ]
                    )

                    if gene_filter is not None:
                        symbols = [
                            s
                            for s in symbols
                            if s in gene_filter
                        ]

                    if probe and symbols:
                        probe_to_symbols[
                            probe
                        ] = symbols

                    continue

            if (
                current_entity == "SAMPLE"
                and (
                    sample_filter is None
                    or current_id in sample_filter
                )
            ):

                if line == "!sample_table_begin":
                    in_sample = True
                    sample_header = None
                    continue

                if line == "!sample_table_end":
                    in_sample = False
                    continue

                if in_sample:
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

                        sample_probe_idx = norm.index(
                            "id_ref"
                        )

                        sample_value_idx = norm.index(
                            "value"
                        )

                        continue

                    if len(fields) <= max(
                        sample_probe_idx,
                        sample_value_idx,
                    ):
                        continue

                    probe = fields[
                        sample_probe_idx
                    ].strip()

                    if probe not in probe_to_symbols:
                        continue

                    try:
                        value = float(
                            fields[
                                sample_value_idx
                            ].strip()
                        )
                    except ValueError:
                        continue

                    sample_values[
                        current_id
                    ][probe] = value

    symbol_to_probes = defaultdict(list)

    for probe, symbols in probe_to_symbols.items():
        for symbol in symbols:
            symbol_to_probes[
                symbol
            ].append(probe)

    return sample_values, symbol_to_probes


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

    for j, gene in enumerate(gene_order):
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
                f"No annotated probes for gene {gene}."
            )

        for i, gsm in enumerate(sample_order):
            vals = [
                sample_values[gsm][probe]
                for probe in probes
                if probe in sample_values[gsm]
            ]

            if not vals:
                raise RuntimeError(
                    f"Missing expression for {gsm}, {gene}."
                )

            X[i, j] = float(
                np.mean(vals)
            )

    if not np.all(np.isfinite(X)):
        raise RuntimeError(
            "Non-finite expression matrix."
        )

    return X


# ============================================================
# MASTER
# ============================================================

def main():
    project = project_dir()

    docs = project / "docs"
    ds = project / "results" / "direct_subspace"

    third_data = (
        project
        / "data"
        / "external"
        / THIRD_GSE
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

    third_data.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Existing frozen inputs.
    # --------------------------------------------------------

    v482_manifest_path = (
        ds
        / "v48_2_manifest.json"
    )

    eligible_path = (
        ds
        / "v44_1b_eligible_genes.txt"
    )

    dev_raw_path = (
        ds
        / "v47_2_GSE130404_label_blind_top256_expression.tsv"
    )

    dev_features_path = (
        ds
        / "v47_2_label_blind_top256_features.tsv"
    )

    ext_geometry_path = (
        ds
        / "v47_5b_GSE44589_label_blind_geometry.tsv"
    )

    ext_features_path = (
        ds
        / "v47_5b_GSE44589_label_blind_top256_features.tsv"
    )

    ext_soft_path = (
        project
        / "data"
        / "external"
        / "GSE44589"
        / "metadata"
        / "GSE44589_family.soft.gz"
    )

    required = [
        v482_manifest_path,
        eligible_path,
        dev_raw_path,
        dev_features_path,
        ext_geometry_path,
        ext_features_path,
        ext_soft_path,
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    v482 = json.loads(
        v482_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if v482.get(
        "status"
    ) != "GEOMETRIC INVARIANCE ROBUSTNESS PASS":
        raise RuntimeError(
            "v48.2 must be a recorded robustness PASS before v48.3."
        )

    if v482.get(
        "clinical_labels_used"
    ) is not False:
        raise RuntimeError(
            "Unexpected label use recorded in v48.2."
        )

    # ========================================================
    # FREEZE COMPLETE MASTER PROTOCOL BEFORE NEW DATA RESULT
    # ========================================================

    protocol = f"""Soft Spaces / CML
v48.3 — THIRD-COHORT GEOMETRIC REPLICATION MASTER PREREGISTRATION

STATUS
------
FROZEN BEFORE GSE14671 GEOMETRIC ANALYSIS

RELATION TO PRIOR WORK
----------------------
v48.1:
    GEOMETRIC INVARIANCE PASS

v48.2:
    GEOMETRIC INVARIANCE ROBUSTNESS PASS

v48.3 is a prospective third-cohort extension.

It is not a rescue of v47 and does not use clinical outcome labels.

THIRD COHORT
------------
GSE14671 / GPL570

Frozen intended use:
    all {THIRD_EXPECTED_N} GEO samples

No clinical-response label is read or used.

The samples are treated only as an independent CML expression cohort.

SHARED TECHNICAL UNIVERSE
-------------------------
Use the existing v44 transport-aware eligible universe:
    15,890 genes

Each cohort retains its OWN label-blind Top-256 high-variance features.

Therefore exact gene identity is not required to match across cohorts.

GEOMETRIC REPRESENTATION
------------------------
For every cohort:

1. Top-256 raw-variance genes frozen for that cohort
2. per-gene cohort z-standardization, ddof=0
3. PCA-16
4. normalized eigen-spectrum
5. pairwise PCA-16 distances normalized by cohort median distance
6. local anisotropy, k={KNN_K}

No clinical labels.

------------------------------------------------------------
v48.3a — TECHNICAL AUDIT
------------------------------------------------------------
Requirements:
- GSE14671 family SOFT available
- exactly {THIRD_EXPECTED_N} sample expression tables
- GPL570 annotation usable
- at least {TOP_K} genes from the 15,890-gene eligible universe measurable
- complete finite expression values for selected genes

If any requirement fails:
    TECHNICALLY NON-EVALUABLE
    STOP MASTER PIPELINE BEFORE GEOMETRIC TESTING

------------------------------------------------------------
v48.3b — THIRD-COHORT LABEL-BLIND GEOMETRY
------------------------------------------------------------
Across all {THIRD_EXPECTED_N} samples:

1. restrict to measurable genes in frozen 15,890-gene universe
2. select Top-{TOP_K} by raw variance
3. construct PCA-{PCA_R}
4. freeze Top-256 and geometry by SHA

No outcome labels are parsed.

------------------------------------------------------------
v48.3c — PAIRWISE THIRD-COHORT REPLICATION
------------------------------------------------------------
Pairs:
    GSE14671 vs GSE130404
    GSE14671 vs GSE44589

Primary A:
    cosine similarity of normalized PCA-16 eigen-spectra
    HIGHER = more similar

Primary B:
    Wasserstein-1 distance between normalized pairwise-distance distributions
    LOWER = more similar

NULL:
    {PAIRWISE_NULL_N} covariance-destroyed GSE14671 geometries

For each NULL:
    independently permute patients within every frozen GSE14671 Top-256 gene

This preserves:
- third-cohort feature identity
- every gene's marginal values
- mean
- variance
- sample count

It destroys:
- cross-gene covariance
- patient-level multivariate organization

For EACH of the two cohort pairs, PASS requires BOTH:
    observed spectrum cosine > NULL q95
    observed distance W1 < NULL q05
    empirical p_spectrum <= 0.05
    empirical p_distance <= 0.05

v48.3c overall PASS requires BOTH cohort pairs to PASS.

If v48.3c FAILS:
    record FAIL
    do not tune
    continue only to predeclared robustness/joint characterization,
    which cannot convert the pairwise FAIL into a PASS.

------------------------------------------------------------
v48.3d — PATIENT-RESAMPLING ROBUSTNESS
------------------------------------------------------------
Paired three-cohort resamples:
    {ROBUST_N_RESAMPLES}

Patients per resample:
    GSE130404: {ROBUST_DEV_N}/96
    GSE44589:  {ROBUST_EXT_N}/135
    GSE14671:  {ROBUST_THIRD_N}/{THIRD_EXPECTED_N}

Frozen feature identities are retained.

Within every resample:
- re-standardize each cohort
- refit PCA-16
- compare GSE14671 with BOTH existing cohorts
- create {ROBUST_NULL_PER_RESAMPLE} covariance-destroyed third-cohort NULLs

Signed NULL-z:
    z_spectrum = (observed - null_mean) / null_sd
    z_distance = (null_mean - observed) / null_sd

For EACH pair and EACH primary metric, robustness requires:
    median signed z > {Z_THRESHOLD:.12f}
    fraction signed z > 0 >= {POSITIVE_FRACTION_THRESHOLD:.2f}

Overall robustness PASS:
    all four gates PASS

------------------------------------------------------------
v48.3e — JOINT THREE-COHORT INVARIANCE
------------------------------------------------------------
Observed worst-pair statistics across:
    GSE130404 vs GSE44589
    GSE130404 vs GSE14671
    GSE44589  vs GSE14671

Joint spectral statistic:
    MINIMUM pairwise spectrum cosine
    higher = better

Joint distance statistic:
    MAXIMUM pairwise normalized-distance W1
    lower = better

NULL:
    {JOINT_NULL_N} replicates

In EACH NULL replicate:
    independently destroy covariance in ALL THREE cohorts
    by within-gene patient permutation.

PASS requires BOTH:
    observed minimum cosine > NULL q95
    observed maximum W1 < NULL q05
    both empirical one-sided p <= 0.05

------------------------------------------------------------
v48.3f — AUTOMATIC CLOSURE
------------------------------------------------------------
The master script writes a final closure with:

- technical evaluability
- third-cohort feature composition
- pairwise replication PASS/FAIL
- resampling robustness PASS/FAIL
- joint three-cohort invariance PASS/FAIL
- all SHA hashes
- no-rescue statement

FINAL CLAIM RULE
----------------
"THREE-COHORT GEOMETRIC INVARIANCE SUPPORTED"
is allowed ONLY if:

    v48.3a technical PASS
    AND v48.3c pairwise overall PASS
    AND v48.3d robustness overall PASS
    AND v48.3e joint overall PASS

Otherwise the final claim is NOT SUPPORTED.

CLAIM LIMIT
-----------
Even a full PASS would support only:

    reproducible higher-level label-blind geometric similarity across
    these three CML transcriptomic cohorts under the frozen v48 tests.

It would NOT establish:
- prediction
- treatment-response replication
- clinical utility
- causal biology
- cancer dormancy
- disease-transition mechanism
- quantum biology
- quantum advantage

NO RESCUE
---------
No post-hoc:
- feature-count tuning
- PCA-rank tuning
- neighborhood tuning
- cohort subset tuning
- label selection
- NULL redefinition
- threshold adjustment

Any later change requires a NEW version and NEW hypothesis.
"""

    protocol_path = (
        docs
        / "v48_3_CML_THIRD_COHORT_MASTER_PREREGISTRATION.txt"
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
        / "v48_3_CML_THIRD_COHORT_MASTER_PREREGISTRATION_manifest.json"
    )

    write_json(
        protocol_manifest_path,
        {
            "version": MASTER_VERSION,
            "status": "FROZEN_BEFORE_GSE14671_GEOMETRIC_ANALYSIS",
            "clinical_labels_used": False,
            "third_cohort": THIRD_GSE,
            "third_platform": THIRD_GPL,
            "third_expected_n": THIRD_EXPECTED_N,
            "top_k": TOP_K,
            "pca_r": PCA_R,
            "knn_k": KNN_K,
            "pairwise_null_n": PAIRWISE_NULL_N,
            "pairwise_seed": PAIRWISE_SEED,
            "robust_n_resamples": ROBUST_N_RESAMPLES,
            "robust_null_per_resample": ROBUST_NULL_PER_RESAMPLE,
            "joint_null_n": JOINT_NULL_N,
            "joint_seed": JOINT_SEED,
            "protocol_sha256": protocol_sha,
            "source_sha256": {
                "v48_2_manifest": sha256_file(
                    v482_manifest_path
                ),
                "eligible_genes": sha256_file(
                    eligible_path
                ),
                "gse130404_raw_top256": sha256_file(
                    dev_raw_path
                ),
                "gse130404_features": sha256_file(
                    dev_features_path
                ),
                "gse44589_frozen_geometry": sha256_file(
                    ext_geometry_path
                ),
                "gse44589_features": sha256_file(
                    ext_features_path
                ),
                "gse44589_soft": sha256_file(
                    ext_soft_path
                ),
            },
        },
    )

    print("=" * 72)
    print("v48.3 MASTER PROTOCOL FROZEN")
    print("Protocol SHA:", protocol_sha)
    print("Clinical labels used: NO")
    print("=" * 72)
    print()

    # ========================================================
    # LOAD EXISTING RAW COHORTS A/B
    # ========================================================

    dev_df = pd.read_csv(
        dev_raw_path,
        sep="\t",
        index_col=0,
    )

    dev_features = pd.read_csv(
        dev_features_path,
        sep="\t",
    )["gene"].astype(str).tolist()

    if dev_df.shape != (96, TOP_K):
        raise RuntimeError(
            f"Unexpected GSE130404 matrix shape: {dev_df.shape}"
        )

    if list(
        dev_df.columns.astype(str)
    ) != dev_features:
        raise RuntimeError(
            "GSE130404 frozen feature order mismatch."
        )

    Xdev = dev_df.to_numpy(
        dtype=float
    )

    ext_geometry = pd.read_csv(
        ext_geometry_path,
        sep="\t",
        index_col=0,
    )

    ext_samples = [
        str(x)
        for x in ext_geometry.index
    ]

    if len(ext_samples) != 135:
        raise RuntimeError(
            f"Expected 135 GSE44589 baseline samples; got {len(ext_samples)}."
        )

    ext_features = pd.read_csv(
        ext_features_path,
        sep="\t",
    )["gene"].astype(str).tolist()

    if len(ext_features) != TOP_K:
        raise RuntimeError(
            "GSE44589 frozen feature count mismatch."
        )

    ext_values, ext_symbol_to_probes = parse_soft_expression(
        ext_soft_path,
        "GPL570",
        selected_sample_ids=ext_samples,
        selected_genes=ext_features,
    )

    Xext = build_gene_matrix(
        ext_values,
        ext_symbol_to_probes,
        ext_samples,
        ext_features,
    )

    # ========================================================
    # v48.3a — THIRD-COHORT TECHNICAL AUDIT
    # ========================================================

    print("v48.3a — TECHNICAL AUDIT")
    print("-------------------------")

    third_soft_path = (
        third_data
        / "GSE14671_family.soft.gz"
    )

    download_if_missing(
        THIRD_SOFT_URL,
        third_soft_path,
    )

    eligible = [
        x.strip()
        for x in eligible_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]

    if len(eligible) != 15890:
        raise RuntimeError(
            f"Expected 15,890 eligible genes; found {len(eligible)}."
        )

    third_values, third_symbol_to_probes = parse_soft_expression(
        third_soft_path,
        THIRD_GPL,
        selected_sample_ids=None,
        selected_genes=eligible,
    )

    third_samples = sorted(
        third_values.keys()
    )

    third_measurable = [
        gene
        for gene in eligible
        if third_symbol_to_probes.get(gene)
    ]

    audit_path = (
        ds
        / "v48_3a_GSE14671_technical_audit.txt"
    )

    technical_pass = (
        len(third_samples) == THIRD_EXPECTED_N
        and len(third_measurable) >= TOP_K
    )

    audit_text = f"""=== v48.3a GSE14671 TECHNICAL AUDIT ===

Protocol SHA:                    {protocol_sha}
Clinical labels parsed:          NO
Clinical labels used:            NO

Cohort:                          {THIRD_GSE}
Platform:                        {THIRD_GPL}
Expression sample tables:        {len(third_samples)}
Expected sample count:           {THIRD_EXPECTED_N}

Transport-aware universe:        {len(eligible)}
Measurable eligible genes:       {len(third_measurable)}
Required minimum:                {TOP_K}

v48.3a STATUS: {"TECHNICAL PASS" if technical_pass else "TECHNICALLY NON-EVALUABLE"}
"""

    audit_path.write_text(
        audit_text,
        encoding="utf-8",
    )

    print(audit_text)

    if not technical_pass:
        closure_path = (
            docs
            / "v48_3f_CML_THREE_COHORT_CLOSURE.txt"
        )

        closure_path.write_text(
            audit_text
            + "\nMASTER PIPELINE STOPPED BEFORE GEOMETRIC TESTING.\n",
            encoding="utf-8",
        )

        raise SystemExit(
            "v48.3 stopped: third cohort technically non-evaluable."
        )

    # Build all-measurable matrix only now.
    Xthird_all = build_gene_matrix(
        third_values,
        third_symbol_to_probes,
        third_samples,
        third_measurable,
    )

    # ========================================================
    # v48.3b — LABEL-BLIND THIRD GEOMETRY
    # ========================================================

    print()
    print("v48.3b — THIRD-COHORT LABEL-BLIND GEOMETRY")
    print("------------------------------------------")

    third_variance = np.var(
        Xthird_all,
        axis=0,
        ddof=0,
    )

    third_order = np.argsort(
        -third_variance,
        kind="mergesort",
    )

    third_top_idx = third_order[
        :TOP_K
    ]

    third_features = [
        third_measurable[i]
        for i in third_top_idx
    ]

    Xthird = Xthird_all[
        :,
        third_top_idx,
    ]

    third_sig = geometry_signature(
        Xthird
    )

    third_feature_out = (
        ds
        / "v48_3b_GSE14671_label_blind_top256_features.tsv"
    )

    pd.DataFrame({
        "gene": third_features,
        "raw_variance": third_variance[
            third_top_idx
        ],
    }).to_csv(
        third_feature_out,
        sep="\t",
        index=False,
    )

    third_raw_out = (
        ds
        / "v48_3b_GSE14671_label_blind_top256_expression.tsv"
    )

    pd.DataFrame(
        Xthird,
        index=third_samples,
        columns=third_features,
    ).rename_axis(
        "gsm"
    ).to_csv(
        third_raw_out,
        sep="\t",
    )

    third_geometry_out = (
        ds
        / "v48_3b_GSE14671_label_blind_geometry.tsv"
    )

    pc_cols = [
        f"PC{i}"
        for i in range(
            1,
            PCA_R + 1,
        )
    ]

    pd.DataFrame(
        third_sig["Y"],
        index=third_samples,
        columns=pc_cols,
    ).rename_axis(
        "gsm"
    ).to_csv(
        third_geometry_out,
        sep="\t",
    )

    third_geometry_sha = sha256_file(
        third_geometry_out
    )

    overlap_dev = len(
        set(third_features)
        & set(dev_features)
    )

    overlap_ext = len(
        set(third_features)
        & set(ext_features)
    )

    print("Samples:", len(third_samples))
    print("Measurable eligible genes:", len(third_measurable))
    print("PCA16 cumulative variance:", f"{third_sig['cumvar']:.6f}")
    print("Effective dimension:", f"{third_sig['effdim']:.6f}")
    print("Top256 overlap vs GSE130404:", f"{overlap_dev}/256")
    print("Top256 overlap vs GSE44589:", f"{overlap_ext}/256")
    print("Geometry SHA:", third_geometry_sha)

    # ========================================================
    # FIXED OBSERVED SIGNATURES FOR ALL THREE COHORTS
    # ========================================================

    dev_sig = geometry_signature(
        Xdev
    )

    ext_sig = geometry_signature(
        Xext
    )

    pair_ab = pair_metrics(
        dev_sig,
        ext_sig,
    )

    pair_ac = pair_metrics(
        dev_sig,
        third_sig,
    )

    pair_bc = pair_metrics(
        ext_sig,
        third_sig,
    )

    # ========================================================
    # v48.3c — PAIRWISE THIRD-COHORT REPLICATION
    # ========================================================

    print()
    print("v48.3c — PAIRWISE THIRD-COHORT REPLICATION")
    print("------------------------------------------")

    rng = np.random.default_rng(
        PAIRWISE_SEED
    )

    null_ac_cos = np.empty(
        PAIRWISE_NULL_N,
        dtype=float,
    )

    null_ac_w1 = np.empty(
        PAIRWISE_NULL_N,
        dtype=float,
    )

    null_bc_cos = np.empty(
        PAIRWISE_NULL_N,
        dtype=float,
    )

    null_bc_w1 = np.empty(
        PAIRWISE_NULL_N,
        dtype=float,
    )

    Xthird_null = np.empty_like(
        Xthird
    )

    for b in range(
        PAIRWISE_NULL_N
    ):
        for j in range(
            TOP_K
        ):
            Xthird_null[
                :,
                j,
            ] = rng.permutation(
                Xthird[
                    :,
                    j,
                ]
            )

        ns = geometry_signature(
            Xthird_null
        )

        mac = pair_metrics(
            dev_sig,
            ns,
        )

        mbc = pair_metrics(
            ext_sig,
            ns,
        )

        null_ac_cos[b] = mac[
            "spectrum_cosine"
        ]

        null_ac_w1[b] = mac[
            "distance_w1"
        ]

        null_bc_cos[b] = mbc[
            "spectrum_cosine"
        ]

        null_bc_w1[b] = mbc[
            "distance_w1"
        ]

        if (
            b + 1
        ) % 100 == 0:
            print(
                f"Completed {b+1}/{PAIRWISE_NULL_N} third-cohort NULL geometries"
            )

    def evaluate_pair(
        observed,
        null_cos,
        null_w1,
    ):
        q95_cos = float(
            np.quantile(
                null_cos,
                0.95,
            )
        )

        q05_w1 = float(
            np.quantile(
                null_w1,
                0.05,
            )
        )

        p_cos = empirical_high(
            null_cos,
            observed[
                "spectrum_cosine"
            ],
        )

        p_w1 = empirical_low(
            null_w1,
            observed[
                "distance_w1"
            ],
        )

        gate_cos = bool(
            observed[
                "spectrum_cosine"
            ] > q95_cos
            and p_cos <= 0.05
        )

        gate_w1 = bool(
            observed[
                "distance_w1"
            ] < q05_w1
            and p_w1 <= 0.05
        )

        return {
            "q95_cos": q95_cos,
            "q05_w1": q05_w1,
            "p_cos": p_cos,
            "p_w1": p_w1,
            "gate_cos": gate_cos,
            "gate_w1": gate_w1,
            "pass": bool(
                gate_cos
                and gate_w1
            ),
        }

    eval_ac = evaluate_pair(
        pair_ac,
        null_ac_cos,
        null_ac_w1,
    )

    eval_bc = evaluate_pair(
        pair_bc,
        null_bc_cos,
        null_bc_w1,
    )

    pairwise_pass = bool(
        eval_ac["pass"]
        and eval_bc["pass"]
    )

    print(
        "GSE14671 vs GSE130404:",
        "PASS" if eval_ac["pass"] else "FAIL",
        "| cosine",
        f"{pair_ac['spectrum_cosine']:.6f}",
        "| W1",
        f"{pair_ac['distance_w1']:.6f}",
    )

    print(
        "GSE14671 vs GSE44589:",
        "PASS" if eval_bc["pass"] else "FAIL",
        "| cosine",
        f"{pair_bc['spectrum_cosine']:.6f}",
        "| W1",
        f"{pair_bc['distance_w1']:.6f}",
    )

    pairwise_null_out = (
        ds
        / "v48_3c_pairwise_third_cohort_null_1000.tsv"
    )

    pd.DataFrame({
        "null_rep": np.arange(
            1,
            PAIRWISE_NULL_N + 1,
        ),
        "AC_spectrum_cosine": null_ac_cos,
        "AC_distance_w1": null_ac_w1,
        "BC_spectrum_cosine": null_bc_cos,
        "BC_distance_w1": null_bc_w1,
    }).to_csv(
        pairwise_null_out,
        sep="\t",
        index=False,
    )

    # ========================================================
    # v48.3d — THREE-COHORT PATIENT-RESAMPLING ROBUSTNESS
    # ========================================================

    print()
    print("v48.3d — PATIENT-RESAMPLING ROBUSTNESS")
    print("--------------------------------------")

    rng_master = np.random.default_rng(
        ROBUST_SEED
    )

    robust_rows = []

    for r in range(
        ROBUST_N_RESAMPLES
    ):
        ia = rng_master.choice(
            len(Xdev),
            size=ROBUST_DEV_N,
            replace=False,
        )

        ib = rng_master.choice(
            len(Xext),
            size=ROBUST_EXT_N,
            replace=False,
        )

        ic = rng_master.choice(
            len(Xthird),
            size=ROBUST_THIRD_N,
            replace=False,
        )

        A = Xdev[
            ia,
            :,
        ]

        B = Xext[
            ib,
            :,
        ]

        C = Xthird[
            ic,
            :,
        ]

        sa = geometry_signature(A)
        sb = geometry_signature(B)
        sc = geometry_signature(C)

        obs_ac = pair_metrics(
            sa,
            sc,
        )

        obs_bc = pair_metrics(
            sb,
            sc,
        )

        n_ac_cos = np.empty(
            ROBUST_NULL_PER_RESAMPLE,
            dtype=float,
        )

        n_ac_w1 = np.empty(
            ROBUST_NULL_PER_RESAMPLE,
            dtype=float,
        )

        n_bc_cos = np.empty(
            ROBUST_NULL_PER_RESAMPLE,
            dtype=float,
        )

        n_bc_w1 = np.empty(
            ROBUST_NULL_PER_RESAMPLE,
            dtype=float,
        )

        rng_null = np.random.default_rng(
            ROBUST_SEED
            + 100000
            + r
        )

        Cn = np.empty_like(C)

        for b in range(
            ROBUST_NULL_PER_RESAMPLE
        ):
            for j in range(
                TOP_K
            ):
                Cn[
                    :,
                    j,
                ] = rng_null.permutation(
                    C[
                        :,
                        j,
                    ]
                )

            sn = geometry_signature(
                Cn
            )

            nac = pair_metrics(
                sa,
                sn,
            )

            nbc = pair_metrics(
                sb,
                sn,
            )

            n_ac_cos[b] = nac[
                "spectrum_cosine"
            ]

            n_ac_w1[b] = nac[
                "distance_w1"
            ]

            n_bc_cos[b] = nbc[
                "spectrum_cosine"
            ]

            n_bc_w1[b] = nbc[
                "distance_w1"
            ]

        def signed_z_high(
            observed,
            null,
        ):
            sd = float(
                np.std(
                    null,
                    ddof=1,
                )
            )

            if sd <= 0:
                raise RuntimeError(
                    "Zero NULL SD."
                )

            return float(
                (
                    observed
                    - np.mean(null)
                )
                / sd
            )

        def signed_z_low(
            observed,
            null,
        ):
            sd = float(
                np.std(
                    null,
                    ddof=1,
                )
            )

            if sd <= 0:
                raise RuntimeError(
                    "Zero NULL SD."
                )

            return float(
                (
                    np.mean(null)
                    - observed
                )
                / sd
            )

        robust_rows.append({
            "resample": r + 1,
            "AC_z_spectrum": signed_z_high(
                obs_ac[
                    "spectrum_cosine"
                ],
                n_ac_cos,
            ),
            "AC_z_distance": signed_z_low(
                obs_ac[
                    "distance_w1"
                ],
                n_ac_w1,
            ),
            "BC_z_spectrum": signed_z_high(
                obs_bc[
                    "spectrum_cosine"
                ],
                n_bc_cos,
            ),
            "BC_z_distance": signed_z_low(
                obs_bc[
                    "distance_w1"
                ],
                n_bc_w1,
            ),
            "AC_anisotropy_w1": obs_ac[
                "anisotropy_w1"
            ],
            "BC_anisotropy_w1": obs_bc[
                "anisotropy_w1"
            ],
            "A_cumvar": sa["cumvar"],
            "B_cumvar": sb["cumvar"],
            "C_cumvar": sc["cumvar"],
        })

        if (
            r + 1
        ) % 10 == 0:
            print(
                f"Completed {r+1}/{ROBUST_N_RESAMPLES} paired three-cohort resamples"
            )

    robust = pd.DataFrame(
        robust_rows
    )

    robust_gates = {}

    for col in [
        "AC_z_spectrum",
        "AC_z_distance",
        "BC_z_spectrum",
        "BC_z_distance",
    ]:
        median_z = float(
            robust[
                col
            ].median()
        )

        positive_fraction = float(
            np.mean(
                robust[
                    col
                ] > 0
            )
        )

        robust_gates[
            col
        ] = {
            "median_z": median_z,
            "positive_fraction": positive_fraction,
            "pass": bool(
                median_z > Z_THRESHOLD
                and positive_fraction
                >= POSITIVE_FRACTION_THRESHOLD
            ),
        }

    robustness_pass = bool(
        all(
            x["pass"]
            for x in robust_gates.values()
        )
    )

    robust_out = (
        ds
        / "v48_3d_three_cohort_resampling_results.tsv"
    )

    robust.to_csv(
        robust_out,
        sep="\t",
        index=False,
    )

    print("Robustness gates:")
    for key, value in robust_gates.items():
        print(
            " ",
            key,
            "median z =",
            f"{value['median_z']:.3f}",
            "positive =",
            f"{value['positive_fraction']:.3f}",
            "=>",
            "PASS" if value["pass"] else "FAIL",
        )

    # ========================================================
    # v48.3e — JOINT THREE-COHORT INVARIANCE
    # ========================================================

    print()
    print("v48.3e — JOINT THREE-COHORT INVARIANCE")
    print("--------------------------------------")

    observed_min_cos = min(
        pair_ab[
            "spectrum_cosine"
        ],
        pair_ac[
            "spectrum_cosine"
        ],
        pair_bc[
            "spectrum_cosine"
        ],
    )

    observed_max_w1 = max(
        pair_ab[
            "distance_w1"
        ],
        pair_ac[
            "distance_w1"
        ],
        pair_bc[
            "distance_w1"
        ],
    )

    joint_rng = np.random.default_rng(
        JOINT_SEED
    )

    null_min_cos = np.empty(
        JOINT_NULL_N,
        dtype=float,
    )

    null_max_w1 = np.empty(
        JOINT_NULL_N,
        dtype=float,
    )

    An = np.empty_like(
        Xdev
    )

    Bn = np.empty_like(
        Xext
    )

    Cn = np.empty_like(
        Xthird
    )

    for b in range(
        JOINT_NULL_N
    ):
        for j in range(
            TOP_K
        ):
            An[
                :,
                j,
            ] = joint_rng.permutation(
                Xdev[
                    :,
                    j,
                ]
            )

            Bn[
                :,
                j,
            ] = joint_rng.permutation(
                Xext[
                    :,
                    j,
                ]
            )

            Cn[
                :,
                j,
            ] = joint_rng.permutation(
                Xthird[
                    :,
                    j,
                ]
            )

        sna = geometry_signature(
            An
        )

        snb = geometry_signature(
            Bn
        )

        snc = geometry_signature(
            Cn
        )

        nab = pair_metrics(
            sna,
            snb,
        )

        nac = pair_metrics(
            sna,
            snc,
        )

        nbc = pair_metrics(
            snb,
            snc,
        )

        null_min_cos[
            b
        ] = min(
            nab[
                "spectrum_cosine"
            ],
            nac[
                "spectrum_cosine"
            ],
            nbc[
                "spectrum_cosine"
            ],
        )

        null_max_w1[
            b
        ] = max(
            nab[
                "distance_w1"
            ],
            nac[
                "distance_w1"
            ],
            nbc[
                "distance_w1"
            ],
        )

        if (
            b + 1
        ) % 50 == 0:
            print(
                f"Completed {b+1}/{JOINT_NULL_N} joint NULL triads"
            )

    joint_q95_cos = float(
        np.quantile(
            null_min_cos,
            0.95,
        )
    )

    joint_q05_w1 = float(
        np.quantile(
            null_max_w1,
            0.05,
        )
    )

    joint_p_cos = empirical_high(
        null_min_cos,
        observed_min_cos,
    )

    joint_p_w1 = empirical_low(
        null_max_w1,
        observed_max_w1,
    )

    joint_cos_gate = bool(
        observed_min_cos
        > joint_q95_cos
        and joint_p_cos <= 0.05
    )

    joint_w1_gate = bool(
        observed_max_w1
        < joint_q05_w1
        and joint_p_w1 <= 0.05
    )

    joint_pass = bool(
        joint_cos_gate
        and joint_w1_gate
    )

    joint_null_out = (
        ds
        / "v48_3e_joint_three_cohort_null_500.tsv"
    )

    pd.DataFrame({
        "null_rep": np.arange(
            1,
            JOINT_NULL_N + 1,
        ),
        "minimum_pairwise_spectrum_cosine": null_min_cos,
        "maximum_pairwise_distance_w1": null_max_w1,
    }).to_csv(
        joint_null_out,
        sep="\t",
        index=False,
    )

    print(
        "Joint minimum cosine:",
        f"{observed_min_cos:.6f}",
        "NULL q95:",
        f"{joint_q95_cos:.6f}",
        "p:",
        f"{joint_p_cos:.6f}",
        "=>",
        "PASS" if joint_cos_gate else "FAIL",
    )

    print(
        "Joint maximum W1:",
        f"{observed_max_w1:.6f}",
        "NULL q05:",
        f"{joint_q05_w1:.6f}",
        "p:",
        f"{joint_p_w1:.6f}",
        "=>",
        "PASS" if joint_w1_gate else "FAIL",
    )

    # ========================================================
    # v48.3f — AUTOMATIC CLOSURE
    # ========================================================

    full_support = bool(
        technical_pass
        and pairwise_pass
        and robustness_pass
        and joint_pass
    )

    final_claim = (
        "THREE-COHORT GEOMETRIC INVARIANCE SUPPORTED"
        if full_support
        else "THREE-COHORT GEOMETRIC INVARIANCE NOT SUPPORTED"
    )

    closure_path = (
        docs
        / "v48_3f_CML_THREE_COHORT_CLOSURE.txt"
    )

    closure = f"""Soft Spaces / CML
v48.3f — THIRD-COHORT GEOMETRIC REPLICATION CLOSURE

STATUS
------
CLOSED

MASTER PROTOCOL SHA
-------------------
{protocol_sha}

LABEL INTEGRITY
---------------
Clinical labels parsed: NO
Clinical labels used:   NO
Classifier fitted:      NO

THIRD COHORT
------------
GSE14671 / GPL570
Samples:                         {len(third_samples)}
Measurable eligible genes:       {len(third_measurable)}
Top-256 selected label-blind:    YES
PCA16 cumulative variance:       {third_sig["cumvar"]:.9f}
Effective dimension:             {third_sig["effdim"]:.9f}
Geometry SHA256:                 {third_geometry_sha}

FEATURE IDENTITY — DESCRIPTIVE
------------------------------
GSE14671 vs GSE130404 Top256 overlap:
    {overlap_dev}/256

GSE14671 vs GSE44589 Top256 overlap:
    {overlap_ext}/256

PAIRWISE OBSERVED GEOMETRY
--------------------------
GSE130404 vs GSE44589:
    spectrum cosine:             {pair_ab["spectrum_cosine"]:.9f}
    distance W1:                 {pair_ab["distance_w1"]:.9f}

GSE130404 vs GSE14671:
    spectrum cosine:             {pair_ac["spectrum_cosine"]:.9f}
    distance W1:                 {pair_ac["distance_w1"]:.9f}
    p spectrum:                  {eval_ac["p_cos"]:.9f}
    p distance:                  {eval_ac["p_w1"]:.9f}
    STATUS:                      {"PASS" if eval_ac["pass"] else "FAIL"}

GSE44589 vs GSE14671:
    spectrum cosine:             {pair_bc["spectrum_cosine"]:.9f}
    distance W1:                 {pair_bc["distance_w1"]:.9f}
    p spectrum:                  {eval_bc["p_cos"]:.9f}
    p distance:                  {eval_bc["p_w1"]:.9f}
    STATUS:                      {"PASS" if eval_bc["pass"] else "FAIL"}

v48.3c PAIRWISE OVERALL:
    {"PASS" if pairwise_pass else "FAIL"}

RESAMPLING ROBUSTNESS
---------------------
AC spectrum median z:
    {robust_gates["AC_z_spectrum"]["median_z"]:.9f}

AC distance median z:
    {robust_gates["AC_z_distance"]["median_z"]:.9f}

BC spectrum median z:
    {robust_gates["BC_z_spectrum"]["median_z"]:.9f}

BC distance median z:
    {robust_gates["BC_z_distance"]["median_z"]:.9f}

v48.3d ROBUSTNESS OVERALL:
    {"PASS" if robustness_pass else "FAIL"}

JOINT THREE-COHORT TEST
-----------------------
Observed minimum pairwise spectrum cosine:
    {observed_min_cos:.9f}

NULL q95:
    {joint_q95_cos:.9f}

Empirical p:
    {joint_p_cos:.9f}

Spectral joint gate:
    {"PASS" if joint_cos_gate else "FAIL"}

Observed maximum pairwise normalized-distance W1:
    {observed_max_w1:.9f}

NULL q05:
    {joint_q05_w1:.9f}

Empirical p:
    {joint_p_w1:.9f}

Distance joint gate:
    {"PASS" if joint_w1_gate else "FAIL"}

v48.3e JOINT OVERALL:
    {"PASS" if joint_pass else "FAIL"}

FINAL CONCLUSION
----------------
{final_claim}

The stronger claim is permitted only because it was prospectively defined
to require ALL technical, pairwise, robustness and joint gates.

NO RESCUE / NO TUNING
---------------------
No post-hoc feature-count, PCA-rank, neighborhood, subgroup, NULL or
threshold tuning was used.

CLAIM LIMIT
-----------
Even if supported, this means only:

    higher-level label-blind transcriptomic geometry is reproducibly similar
    across these three CML cohorts under the frozen v48 tests.

It does NOT establish prediction, treatment-response replication, clinical
utility, causal biology, dormancy, disease-transition mechanism, quantum
biology or quantum advantage.

TRACK
-----
v48.3 MASTER PIPELINE:
    CLOSED
"""

    closure_path.write_text(
        closure,
        encoding="utf-8",
    )

    closure_sha = sha256_file(
        closure_path
    )

    master_manifest_path = (
        docs
        / "v48_3f_CML_THREE_COHORT_CLOSURE_manifest.json"
    )

    write_json(
        master_manifest_path,
        {
            "version": "v48.3f",
            "status": "CLOSED",
            "clinical_labels_used": False,
            "classifier_fitted": False,
            "third_cohort": THIRD_GSE,
            "technical_pass": technical_pass,
            "pairwise_overall_pass": pairwise_pass,
            "robustness_overall_pass": robustness_pass,
            "joint_overall_pass": joint_pass,
            "three_cohort_geometric_invariance_supported": full_support,
            "final_claim": final_claim,
            "protocol_sha256": protocol_sha,
            "closure_sha256": closure_sha,
            "third_geometry_sha256": third_geometry_sha,
            "pairwise": {
                "AB": pair_ab,
                "AC": {
                    **pair_ac,
                    **eval_ac,
                },
                "BC": {
                    **pair_bc,
                    **eval_bc,
                },
            },
            "robustness_gates": robust_gates,
            "joint": {
                "observed_min_pairwise_spectrum_cosine": observed_min_cos,
                "null_q95_min_cosine": joint_q95_cos,
                "p_spectrum": joint_p_cos,
                "spectrum_gate": joint_cos_gate,
                "observed_max_pairwise_distance_w1": observed_max_w1,
                "null_q05_max_w1": joint_q05_w1,
                "p_distance": joint_p_w1,
                "distance_gate": joint_w1_gate,
            },
            "sha256": {
                "master_script": sha256_file(
                    Path(__file__).resolve()
                ),
                "third_soft": sha256_file(
                    third_soft_path
                ),
                "third_features": sha256_file(
                    third_feature_out
                ),
                "third_raw_top256": sha256_file(
                    third_raw_out
                ),
                "third_geometry": third_geometry_sha,
                "pairwise_null": sha256_file(
                    pairwise_null_out
                ),
                "robustness_results": sha256_file(
                    robust_out
                ),
                "joint_null": sha256_file(
                    joint_null_out
                ),
                "closure": closure_sha,
            },
        },
    )

    summary_path = (
        ds
        / "v48_3_master_summary.txt"
    )

    summary = f"""=== Soft Spaces / CML v48.3 MASTER SUMMARY ===

Third cohort technical audit:
    {"PASS" if technical_pass else "FAIL"}

Pairwise third-cohort replication:
    {"PASS" if pairwise_pass else "FAIL"}

Patient-resampling robustness:
    {"PASS" if robustness_pass else "FAIL"}

Joint three-cohort invariance:
    {"PASS" if joint_pass else "FAIL"}

FINAL:
    {final_claim}

Master protocol SHA:
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
    print("v48.3 MASTER PIPELINE COMPLETE")
    print("=" * 72)
    print(summary)
    print("Closure:")
    print(" ", closure_path)
    print("Manifest:")
    print(" ", master_manifest_path)
    print("Summary:")
    print(" ", summary_path)


if __name__ == "__main__":
    main()
