#!/usr/bin/env python3
"""
Soft Spaces / CML
v48.2 — Geometric invariance resampling robustness

NEW TEST WITHIN THE v48 GEOMETRIC-INVARIANCE HYPOTHESIS.

Purpose
-------
Test whether the v48.1 label-blind cross-cohort geometric similarity is
robust to patient resampling in BOTH cohorts.

No clinical labels are read or used.
No classifier is fitted.
No feature tuning is performed.

Frozen feature identities:
- GSE130404: frozen v47.2 Top-256
- GSE44589: frozen v47.5b Top-256

Resampling:
- 200 independent paired subsamples
- 80% of each cohort, without replacement
- GSE130404: 76 / 96 patients
- GSE44589: 108 / 135 patients

Within each paired resample:
1. Re-standardize each cohort using only that resample.
2. Refit PCA-16 independently in each cohort.
3. Compute observed:
   A) normalized PCA-16 eigen-spectrum cosine similarity
   B) normalized PCA-16 pairwise-distance Wasserstein-1
4. Build 50 covariance-destroyed external NULL geometries by independently
   permuting patient values within every frozen GSE44589 gene.
5. Convert both primary metrics to signed NULL-z:
      z_spectrum = (observed - null_mean) / null_sd
      z_distance = (null_mean - observed) / null_sd
   Thus larger positive z always means stronger geometric similarity than NULL.

Primary robustness PASS
-----------------------
BOTH metrics must satisfy BOTH:
- median signed NULL-z > 1.644854
- fraction of resamples with signed NULL-z > 0 >= 0.75

Secondary descriptives
----------------------
- fraction with empirical one-sided p <= 0.05 for each primary metric
- local-anisotropy W1
- effective-dimension difference
- PCA16 cumulative explained variance in both cohorts

This tests internal robustness of the cross-cohort geometric-invariance
signal to patient sampling. It is not a third independent cohort replication.
"""

from pathlib import Path
import gzip
import hashlib
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


VERSION = "v48.2"

TOP_K = 256
PCA_R = 16
KNN_K = 10

N_RESAMPLES = 200
DEV_SUB_N = 76
EXT_SUB_N = 108
N_NULL_PER_RESAMPLE = 50
MASTER_SEED = 48020001

Z_THRESHOLD = 1.6448536269514722
POSITIVE_FRACTION_THRESHOLD = 0.75


def project_dir():
    return Path(__file__).resolve().parent.parent


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def pairwise_distances_upper(Y):
    Y = np.asarray(Y, dtype=float)
    diff = Y[:, None, :] - Y[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(len(Y), k=1)
    return D[iu]


def local_anisotropy(Y, k):
    Y = np.asarray(Y, dtype=float)

    k_eff = min(k, len(Y) - 1)

    nn = NearestNeighbors(
        n_neighbors=k_eff + 1,
        metric="euclidean",
    )
    nn.fit(Y)

    _, indices = nn.kneighbors(Y)
    nbr_idx = indices[:, 1:]

    eps = 1e-12
    out = np.empty(len(Y), dtype=float)

    for i in range(len(Y)):
        local = Y[nbr_idx[i], :]
        C = np.cov(local, rowvar=False, ddof=1)
        eig = np.linalg.eigvalsh(C)
        eig = np.maximum(eig, 0.0)[::-1]
        out[i] = float(
            eig[0] / (eig.sum() + eps)
        )

    return out


def geometry_signature(X):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise RuntimeError("Expected 2D expression matrix.")

    if X.shape[1] != TOP_K:
        raise RuntimeError(
            f"Expected {TOP_K} features, got {X.shape[1]}."
        )

    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)

    if np.any(sd == 0):
        raise RuntimeError("Zero-SD feature in resample.")

    Z = (X - mu) / sd

    pca = PCA(
        n_components=PCA_R,
        svd_solver="full",
    )

    Y = pca.fit_transform(Z)

    spec = pca.explained_variance_.astype(float)
    spec = spec / spec.sum()

    pdist = pairwise_distances_upper(Y)
    med = float(np.median(pdist))

    if med <= 0:
        raise RuntimeError(
            "Non-positive median pairwise distance."
        )

    pdist_norm = pdist / med

    aniso = local_anisotropy(
        Y,
        KNN_K,
    )

    effdim = float(
        1.0 / np.sum(spec * spec)
    )

    return {
        "spectrum": spec,
        "pdist_norm": pdist_norm,
        "anisotropy": aniso,
        "cumvar": float(
            pca.explained_variance_ratio_.sum()
        ),
        "effdim": effdim,
    }


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


def reconstruct_external_matrix(
    soft_path,
    sample_ids,
    gene_order,
):
    sample_ids = list(map(str, sample_ids))
    sample_set = set(sample_ids)
    gene_set = set(gene_order)

    probe_to_symbols = {}
    sample_values = {
        gsm: {}
        for gsm in sample_ids
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
                continue

            if (
                current_entity == "PLATFORM"
                and current_id == "GPL570"
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
                                "Could not identify GPL570 symbol column."
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
                        if s in gene_set
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

                        pidx = norm.index("id_ref")
                        vidx = norm.index("value")
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

    symbol_to_probes = defaultdict(list)

    for probe, symbols in probe_to_symbols.items():
        for symbol in symbols:
            symbol_to_probes[
                symbol
            ].append(probe)

    X = np.empty(
        (
            len(sample_ids),
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
                f"No GPL570 probes for frozen gene {gene}."
            )

        for i, gsm in enumerate(sample_ids):
            vals = [
                sample_values[gsm][p]
                for p in probes
                if p in sample_values[gsm]
            ]

            if not vals:
                raise RuntimeError(
                    f"Missing expression for {gsm}, {gene}."
                )

            X[i, j] = float(
                np.mean(vals)
            )

    return X


def empirical_p_high(null, observed):
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


def empirical_p_low(null, observed):
    return float(
        (
            1
            + np.sum(
                null <= observed
            )
        )
        / (
            1 + len(null)
        )
    )


def main():
    project = project_dir()

    docs = project / "docs"
    ds = project / "results" / "direct_subspace"

    v481_manifest_path = (
        ds
        / "v48_1_manifest.json"
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
        v481_manifest_path,
        dev_raw_path,
        dev_features_path,
        ext_geometry_path,
        ext_features_path,
        ext_soft_path,
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    v481 = json.loads(
        v481_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if v481.get("status") != "GEOMETRIC INVARIANCE PASS":
        raise RuntimeError(
            "v48.1 is not recorded as GEOMETRIC INVARIANCE PASS."
        )

    if v481.get("clinical_labels_used") is not False:
        raise RuntimeError(
            "v48.1 manifest unexpectedly reports clinical-label use."
        )

    # ---------------------------------------------------------
    # Freeze protocol BEFORE robustness execution.
    # ---------------------------------------------------------

    protocol = f"""Soft Spaces / CML
v48.2 — GEOMETRIC INVARIANCE RESAMPLING ROBUSTNESS PREREGISTRATION

STATUS
------
FROZEN BEFORE RESAMPLING EXECUTION

RELATION TO v48.1
-----------------
v48.1 produced a label-blind GEOMETRIC INVARIANCE PASS.

v48.2 tests patient-sampling robustness of that same higher-level
geometric-invariance hypothesis.

NO CLINICAL LABELS are used.

FEATURE IDENTITIES
------------------
Frozen throughout v48.2:

GSE130404:
    frozen v47.2 Top-256

GSE44589:
    frozen v47.5b Top-256

Feature identities are NOT reselected in v48.2.

Therefore v48.2 tests:
    patient-sampling robustness of geometry

It does NOT test:
    feature-selection robustness

RESAMPLING
----------
Paired independent subsampling without replacement.

Number of paired resamples:
    {N_RESAMPLES}

GSE130404:
    {DEV_SUB_N} / 96 patients

GSE44589:
    {EXT_SUB_N} / 135 patients

For each resample:
1. re-standardize each cohort within the selected patients;
2. refit PCA-16 independently in each cohort;
3. compute observed primary metrics;
4. create {N_NULL_PER_RESAMPLE} external covariance-destroyed NULL geometries.

PRIMARY METRIC A
----------------
Normalized PCA-16 eigen-spectrum cosine similarity.

Higher means more similar.

Within each resample:

    z_spectrum =
        (observed_cosine - mean(NULL_cosine))
        / sd(NULL_cosine)

PRIMARY METRIC B
----------------
Wasserstein-1 distance between normalized PCA-16 pairwise-distance
distributions.

Lower means more similar.

Within each resample:

    z_distance =
        (mean(NULL_W1) - observed_W1)
        / sd(NULL_W1)

Thus for BOTH primary metrics:
    positive z = stronger similarity than covariance-destroyed NULL

NULL CONSTRUCTION
-----------------
For each NULL replicate:

Within the selected GSE44589 patient subset, independently permute patient
values inside every frozen external Top-256 gene.

This preserves:
- selected external patient count
- every selected gene's exact marginal values
- mean
- variance
- frozen feature identity

It destroys:
- gene-gene covariance
- patient-level multivariate structure

PRIMARY ROBUSTNESS PASS RULE
----------------------------
BOTH primary metrics must satisfy BOTH:

1. median signed NULL-z > {Z_THRESHOLD:.12f}

AND

2. fraction of resamples with signed NULL-z > 0
   >= {POSITIVE_FRACTION_THRESHOLD:.2f}

SECONDARY DESCRIPTIVES
----------------------
Report:
- fraction empirical one-sided p <= 0.05 for spectrum
- fraction empirical one-sided p <= 0.05 for distance geometry
- local-anisotropy Wasserstein distance
- effective-dimension absolute difference
- PCA-16 cumulative variance in each cohort

CLAIM LIMIT
-----------
A PASS supports robustness of the v48 cross-cohort geometric-invariance
signal to patient subsampling.

It is NOT:
- a third independent cohort replication
- treatment-response replication
- prediction
- prognosis
- causal biology
- dormancy mechanism
- quantum biology
- quantum advantage

No rescue tuning after FAIL.
"""

    protocol_path = (
        docs
        / "v48_2_CML_GEOMETRIC_INVARIANCE_ROBUSTNESS_PREREGISTRATION.txt"
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
        / "v48_2_CML_GEOMETRIC_INVARIANCE_ROBUSTNESS_PREREGISTRATION_manifest.json"
    )

    protocol_manifest_path.write_text(
        json.dumps(
            {
                "version": VERSION,
                "status": "FROZEN_BEFORE_RESAMPLING",
                "clinical_labels_used": False,
                "classifier_fitted": False,
                "feature_identity_reselected": False,
                "n_resamples": N_RESAMPLES,
                "dev_subsample_n": DEV_SUB_N,
                "ext_subsample_n": EXT_SUB_N,
                "n_null_per_resample": N_NULL_PER_RESAMPLE,
                "master_seed": MASTER_SEED,
                "median_z_threshold": Z_THRESHOLD,
                "positive_fraction_threshold": POSITIVE_FRACTION_THRESHOLD,
                "protocol_sha256": protocol_sha,
                "source_sha256": {
                    "v48_1_manifest": sha256_file(
                        v481_manifest_path
                    ),
                    "dev_raw_top256": sha256_file(
                        dev_raw_path
                    ),
                    "dev_features": sha256_file(
                        dev_features_path
                    ),
                    "ext_geometry": sha256_file(
                        ext_geometry_path
                    ),
                    "ext_features": sha256_file(
                        ext_features_path
                    ),
                    "gse44589_soft": sha256_file(
                        ext_soft_path
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== v48.2 ROBUSTNESS PROTOCOL FROZEN ===")
    print("Protocol SHA:", protocol_sha)
    print("Clinical labels used: NO")
    print()

    # ---------------------------------------------------------
    # Load/reconstruct frozen raw matrices.
    # ---------------------------------------------------------

    dev_df = pd.read_csv(
        dev_raw_path,
        sep="\t",
        index_col=0,
    )

    dev_features = pd.read_csv(
        dev_features_path,
        sep="\t",
    )["gene"].astype(str).tolist()

    ext_geometry = pd.read_csv(
        ext_geometry_path,
        sep="\t",
        index_col=0,
    )

    ext_features = pd.read_csv(
        ext_features_path,
        sep="\t",
    )["gene"].astype(str).tolist()

    if dev_df.shape != (96, TOP_K):
        raise RuntimeError(
            f"Unexpected development matrix shape: {dev_df.shape}"
        )

    if len(ext_geometry) != 135:
        raise RuntimeError(
            f"Unexpected external sample count: {len(ext_geometry)}"
        )

    if len(dev_features) != TOP_K:
        raise RuntimeError(
            "Development frozen feature count mismatch."
        )

    if len(ext_features) != TOP_K:
        raise RuntimeError(
            "External frozen feature count mismatch."
        )

    if list(
        dev_df.columns.astype(str)
    ) != dev_features:
        raise RuntimeError(
            "Development feature order mismatch."
        )

    ext_samples = [
        str(x)
        for x in ext_geometry.index
    ]

    Xdev = dev_df.to_numpy(
        dtype=float
    )

    Xext = reconstruct_external_matrix(
        ext_soft_path,
        ext_samples,
        ext_features,
    )

    if Xext.shape != (
        135,
        TOP_K,
    ):
        raise RuntimeError(
            f"Unexpected external raw matrix shape: {Xext.shape}"
        )

    # ---------------------------------------------------------
    # Paired resampling.
    # ---------------------------------------------------------

    rng_master = np.random.default_rng(
        MASTER_SEED
    )

    rows = []

    for r in range(
        N_RESAMPLES
    ):
        dev_idx = rng_master.choice(
            Xdev.shape[0],
            size=DEV_SUB_N,
            replace=False,
        )

        ext_idx = rng_master.choice(
            Xext.shape[0],
            size=EXT_SUB_N,
            replace=False,
        )

        D = Xdev[
            dev_idx,
            :,
        ]

        E = Xext[
            ext_idx,
            :,
        ]

        dev_sig = geometry_signature(
            D
        )

        ext_sig = geometry_signature(
            E
        )

        obs_cos = cosine(
            dev_sig["spectrum"],
            ext_sig["spectrum"],
        )

        obs_w1 = wasserstein_1d(
            dev_sig["pdist_norm"],
            ext_sig["pdist_norm"],
        )

        obs_aniso_w1 = wasserstein_1d(
            dev_sig["anisotropy"],
            ext_sig["anisotropy"],
        )

        obs_effdim_diff = abs(
            dev_sig["effdim"]
            - ext_sig["effdim"]
        )

        null_cos = np.empty(
            N_NULL_PER_RESAMPLE,
            dtype=float,
        )

        null_w1 = np.empty(
            N_NULL_PER_RESAMPLE,
            dtype=float,
        )

        rng_null = np.random.default_rng(
            MASTER_SEED
            + 100000
            + r
        )

        Enull = np.empty_like(
            E
        )

        for b in range(
            N_NULL_PER_RESAMPLE
        ):
            for j in range(
                TOP_K
            ):
                Enull[
                    :,
                    j,
                ] = rng_null.permutation(
                    E[
                        :,
                        j,
                    ]
                )

            null_sig = geometry_signature(
                Enull
            )

            null_cos[b] = cosine(
                dev_sig["spectrum"],
                null_sig["spectrum"],
            )

            null_w1[b] = wasserstein_1d(
                dev_sig["pdist_norm"],
                null_sig["pdist_norm"],
            )

        cos_mean = float(
            np.mean(
                null_cos
            )
        )

        cos_sd = float(
            np.std(
                null_cos,
                ddof=1,
            )
        )

        w1_mean = float(
            np.mean(
                null_w1
            )
        )

        w1_sd = float(
            np.std(
                null_w1,
                ddof=1,
            )
        )

        if cos_sd <= 0 or w1_sd <= 0:
            raise RuntimeError(
                f"Resample {r+1}: zero NULL SD."
            )

        z_cos = float(
            (
                obs_cos
                - cos_mean
            )
            / cos_sd
        )

        z_w1 = float(
            (
                w1_mean
                - obs_w1
            )
            / w1_sd
        )

        p_cos = empirical_p_high(
            null_cos,
            obs_cos,
        )

        p_w1 = empirical_p_low(
            null_w1,
            obs_w1,
        )

        rows.append({
            "resample": r + 1,
            "observed_spectrum_cosine": obs_cos,
            "null_spectrum_mean": cos_mean,
            "null_spectrum_sd": cos_sd,
            "z_spectrum": z_cos,
            "empirical_p_spectrum": p_cos,
            "observed_distance_w1": obs_w1,
            "null_distance_mean": w1_mean,
            "null_distance_sd": w1_sd,
            "z_distance": z_w1,
            "empirical_p_distance": p_w1,
            "observed_anisotropy_w1": obs_aniso_w1,
            "observed_effective_dimension_abs_diff": obs_effdim_diff,
            "dev_pca16_cumulative_variance": dev_sig["cumvar"],
            "ext_pca16_cumulative_variance": ext_sig["cumvar"],
            "dev_effective_dimension": dev_sig["effdim"],
            "ext_effective_dimension": ext_sig["effdim"],
        })

        if (
            r + 1
        ) % 20 == 0:
            print(
                f"Completed {r+1}/{N_RESAMPLES} paired resamples"
            )

    res = pd.DataFrame(
        rows
    )

    median_z_cos = float(
        res[
            "z_spectrum"
        ].median()
    )

    median_z_w1 = float(
        res[
            "z_distance"
        ].median()
    )

    positive_cos = float(
        np.mean(
            res[
                "z_spectrum"
            ] > 0
        )
    )

    positive_w1 = float(
        np.mean(
            res[
                "z_distance"
            ] > 0
        )
    )

    p05_cos = float(
        np.mean(
            res[
                "empirical_p_spectrum"
            ] <= 0.05
        )
    )

    p05_w1 = float(
        np.mean(
            res[
                "empirical_p_distance"
            ] <= 0.05
        )
    )

    spectrum_pass = bool(
        median_z_cos > Z_THRESHOLD
        and positive_cos
        >= POSITIVE_FRACTION_THRESHOLD
    )

    distance_pass = bool(
        median_z_w1 > Z_THRESHOLD
        and positive_w1
        >= POSITIVE_FRACTION_THRESHOLD
    )

    primary_pass = bool(
        spectrum_pass
        and distance_pass
    )

    status = (
        "GEOMETRIC INVARIANCE ROBUSTNESS PASS"
        if primary_pass
        else "GEOMETRIC INVARIANCE ROBUSTNESS FAIL"
    )

    q_cos = res[
        "z_spectrum"
    ].quantile(
        [
            0.025,
            0.25,
            0.5,
            0.75,
            0.975,
        ]
    )

    q_w1 = res[
        "z_distance"
    ].quantile(
        [
            0.025,
            0.25,
            0.5,
            0.75,
            0.975,
        ]
    )

    results_out = (
        ds
        / "v48_2_geometric_invariance_resampling_results.tsv"
    )

    res.to_csv(
        results_out,
        sep="\t",
        index=False,
    )

    summary_out = (
        ds
        / "v48_2_geometric_invariance_robustness_summary.txt"
    )

    summary = f"""=== Soft Spaces / CML v48.2 GEOMETRIC INVARIANCE ROBUSTNESS ===

PROTOCOL
--------
v48.2 protocol SHA:              {protocol_sha}
Paired resamples:                {N_RESAMPLES}
GSE130404 patients/resample:     {DEV_SUB_N}
GSE44589 patients/resample:      {EXT_SUB_N}
NULL geometries/resample:        {N_NULL_PER_RESAMPLE}

LABEL / MODEL USE
-----------------
Clinical labels used:            NO
Classifier fitted:               NO
Feature identities reselected:   NO

PRIMARY A — SPECTRAL SHAPE ROBUSTNESS
-------------------------------------
Median signed NULL-z:            {median_z_cos:.6f}
Required median z:               > {Z_THRESHOLD:.6f}
Fraction z > 0:                  {positive_cos:.6f}
Required positive fraction:      >= {POSITIVE_FRACTION_THRESHOLD:.6f}

z q2.5%:                         {q_cos.loc[0.025]:.6f}
z q25%:                          {q_cos.loc[0.25]:.6f}
z median:                        {q_cos.loc[0.5]:.6f}
z q75%:                          {q_cos.loc[0.75]:.6f}
z q97.5%:                        {q_cos.loc[0.975]:.6f}

Fraction empirical p <= 0.05:    {p05_cos:.6f}
Spectrum robustness gate:        {"PASS" if spectrum_pass else "FAIL"}

PRIMARY B — DISTANCE GEOMETRY ROBUSTNESS
----------------------------------------
Median signed NULL-z:            {median_z_w1:.6f}
Required median z:               > {Z_THRESHOLD:.6f}
Fraction z > 0:                  {positive_w1:.6f}
Required positive fraction:      >= {POSITIVE_FRACTION_THRESHOLD:.6f}

z q2.5%:                         {q_w1.loc[0.025]:.6f}
z q25%:                          {q_w1.loc[0.25]:.6f}
z median:                        {q_w1.loc[0.5]:.6f}
z q75%:                          {q_w1.loc[0.75]:.6f}
z q97.5%:                        {q_w1.loc[0.975]:.6f}

Fraction empirical p <= 0.05:    {p05_w1:.6f}
Distance robustness gate:        {"PASS" if distance_pass else "FAIL"}

SECONDARY DESCRIPTIVES
----------------------
Median observed anisotropy W1:   {res["observed_anisotropy_w1"].median():.6f}
Median effective-dim abs diff:   {res["observed_effective_dimension_abs_diff"].median():.6f}

Median dev PCA16 cumulative var: {res["dev_pca16_cumulative_variance"].median():.6f}
Median ext PCA16 cumulative var: {res["ext_pca16_cumulative_variance"].median():.6f}

Median dev effective dimension:  {res["dev_effective_dimension"].median():.6f}
Median ext effective dimension:  {res["ext_effective_dimension"].median():.6f}

v48.2 STATUS: {status}

INTERPRETATION LIMIT
--------------------
This tests robustness of the v48.1 label-blind cross-cohort geometric
similarity to repeated patient subsampling in both cohorts.

A PASS supports patient-sampling robustness of the geometric-invariance
signal under the frozen v48 framework.

It is not a third independent cohort replication and does not establish
prediction, treatment-response replication, clinical utility, causal
biology, dormancy mechanism, quantum biology, or quantum advantage.

A FAIL is retained without rescue tuning.
"""

    summary_out.write_text(
        summary,
        encoding="utf-8",
    )

    manifest_out = (
        ds
        / "v48_2_manifest.json"
    )

    manifest_out.write_text(
        json.dumps(
            {
                "version": VERSION,
                "status": status,
                "clinical_labels_used": False,
                "classifier_fitted": False,
                "feature_identity_reselected": False,
                "n_resamples": N_RESAMPLES,
                "dev_subsample_n": DEV_SUB_N,
                "ext_subsample_n": EXT_SUB_N,
                "n_null_per_resample": N_NULL_PER_RESAMPLE,
                "master_seed": MASTER_SEED,
                "spectrum": {
                    "median_signed_null_z": median_z_cos,
                    "positive_z_fraction": positive_cos,
                    "fraction_empirical_p_le_0_05": p05_cos,
                    "pass": spectrum_pass,
                },
                "distance_geometry": {
                    "median_signed_null_z": median_z_w1,
                    "positive_z_fraction": positive_w1,
                    "fraction_empirical_p_le_0_05": p05_w1,
                    "pass": distance_pass,
                },
                "secondary": {
                    "median_anisotropy_w1": float(
                        res[
                            "observed_anisotropy_w1"
                        ].median()
                    ),
                    "median_effective_dimension_abs_diff": float(
                        res[
                            "observed_effective_dimension_abs_diff"
                        ].median()
                    ),
                    "median_dev_pca16_cumulative_variance": float(
                        res[
                            "dev_pca16_cumulative_variance"
                        ].median()
                    ),
                    "median_ext_pca16_cumulative_variance": float(
                        res[
                            "ext_pca16_cumulative_variance"
                        ].median()
                    ),
                },
                "primary_pass": primary_pass,
                "protocol_sha256": protocol_sha,
                "sha256": {
                    "resampling_results": sha256_file(
                        results_out
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
    print(summary)
    print("Wrote:")
    print(" ", protocol_path)
    print(" ", protocol_manifest_path)
    print(" ", results_out)
    print(" ", summary_out)
    print(" ", manifest_out)


if __name__ == "__main__":
    main()
