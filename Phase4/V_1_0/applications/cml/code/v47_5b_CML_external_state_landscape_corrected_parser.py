#!/usr/bin/env python3
"""
Soft Spaces / CML
v47.5b — Corrected GSE44589 external state-landscape replication

CORRECTION ONLY
---------------
v47.5 aborted BEFORE external geometry construction/scoring because its
sample-timepoint parser incorrectly mixed "pretreatment ..." treatment-history
characteristics into the timepoint decision. This reproduced the historical
v42.1b parser failure mode (198/198 baseline).

v47.5b changes ONLY sample metadata parsing:

Timepoint:
    !Sample_source_name_ch1 == "peripheral whole blood sample prior to treatment"
        -> BASELINE_PRETREATMENT

    !Sample_source_name_ch1 == "peripheral whole blood sample after 6 weeks of treatment"
        -> POST_6W

Outcome:
    sample-specific characteristic named "molecular response"
        MMR
        none
        not available

Expected frozen counts:
    baseline 135
      MMR 69
      none 59
      not available 7
    post-6-week 63

The already-written v47.5 preregistration remains unchanged.
No geometry or outcome score was produced by the aborted v47.5 run.

After count verification, this script executes the originally preregistered
external label-blind landscape test:
    15,890 transport-aware genes
    -> external baseline Top-256 raw variance, label-blind
    -> external cohort z-standardization
    -> PCA-16, label-blind
    -> SHA-freeze geometry
    -> overlay 128 MMR/none labels
    -> 10,000 label permutations

No classifier is fit.
No rescue/tuning is performed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


VERSION = "v47.5b"
GSE = "GSE44589"
GPL = "GPL570"

ELIGIBLE_N = 15890
TOP_K = 256
PCA_R = 16
KNN_K = 10

EXPECTED_TOTAL_N = 198
EXPECTED_BASELINE_N = 135
EXPECTED_POST_N = 63
EXPECTED_MMR_N = 69
EXPECTED_NONE_N = 59
EXPECTED_UNAVAILABLE_N = 7
EXPECTED_EVALUABLE_N = 128

N_PERM = 10_000
SEED = 47050001


def project_dir():
    return Path(__file__).resolve().parent.parent


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(x):
    return re.sub(r"\s+", " ", str(x).strip().strip('"'))


def split_gene_symbols(text):
    text = clean(text)

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
    Parse GPL570 annotation, expression tables, and only sample-specific
    metadata needed for corrected timepoint/outcome classification.

    Critically:
    - timepoint comes ONLY from Sample_source_name_ch1
    - molecular response comes ONLY from the named characteristic
    """
    probe_to_symbols = {}
    sample_values = {}
    sample_source = {}
    sample_chars = defaultdict(dict)

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
                sample_source[current_id] = ""
                sample_chars[current_id] = {}
                in_platform_table = False
                in_sample_table = False
                sample_header = None
                continue

            if current_entity == "SAMPLE":

                if (
                    line.startswith("!Sample_source_name_ch1")
                    and "=" in line
                ):
                    sample_source[current_id] = clean(
                        line.split("=", 1)[1]
                    )
                    continue

                if (
                    line.startswith("!Sample_characteristics_ch1")
                    and "=" in line
                ):
                    value = clean(
                        line.split("=", 1)[1]
                    )

                    if ":" in value:
                        key, val = value.split(":", 1)

                        sample_chars[current_id][
                            clean(key).lower()
                        ] = clean(val)

                    continue

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
                            clean(x).lower()
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

                    probe = clean(
                        fields[probe_col_idx]
                    )

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
                            clean(x).lower()
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

                    probe = clean(
                        fields[sample_probe_idx]
                    )

                    try:
                        value = float(
                            clean(
                                fields[sample_value_idx]
                            )
                        )
                    except ValueError:
                        continue

                    sample_values[current_id][probe] = value

    return (
        probe_to_symbols,
        sample_values,
        sample_source,
        sample_chars,
    )


def classify_corrected(source_name, chars):
    """
    Frozen corrected interpretation from v42.1c / v44.5b1.

    Timepoint uses ONLY sample-specific source_name.

    Outcome uses ONLY the sample-specific characteristic
    named 'molecular response'.
    """
    source = clean(source_name).lower()

    if source == "peripheral whole blood sample prior to treatment":
        timepoint = "BASELINE_PRETREATMENT"

    elif source == "peripheral whole blood sample after 6 weeks of treatment":
        timepoint = "POST_6W"

    else:
        timepoint = "UNRESOLVED"

    mr = None

    # Exact key preferred.
    if "molecular response" in chars:
        mr = clean(
            chars["molecular response"]
        ).lower()

    # Conservative fallback only for a key containing the exact phrase.
    if mr is None:
        hits = [
            clean(v).lower()
            for k, v in chars.items()
            if "molecular response" in k.lower()
        ]

        if len(hits) == 1:
            mr = hits[0]

        elif len(hits) > 1:
            raise RuntimeError(
                "Multiple molecular-response fields found."
            )

    if mr is None:
        outcome = "UNRESOLVED"

    elif mr == "mmr":
        outcome = "MMR"

    elif mr == "none":
        outcome = "NONE"

    elif mr == "not available":
        outcome = "NOT_AVAILABLE"

    else:
        outcome = "UNRESOLVED"

    return timepoint, outcome


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
        (1, "NONE_NO_MMR"),
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

    return pd.DataFrame(
        rows
    )


def main():
    project = project_dir()

    docs = project / "docs"
    ds = project / "results" / "direct_subspace"

    soft_path = (
        project
        / "data"
        / "external"
        / GSE
        / "metadata"
        / "GSE44589_family.soft.gz"
    )

    protocol_path = (
        docs
        / "v47_5_CML_EXTERNAL_STATE_LANDSCAPE_PREREGISTRATION.txt"
    )

    protocol_manifest_path = (
        docs
        / "v47_5_CML_EXTERNAL_STATE_LANDSCAPE_PREREGISTRATION_manifest.json"
    )

    eligible_path = (
        ds
        / "v44_1b_eligible_genes.txt"
    )

    for p in [
        protocol_path,
        protocol_manifest_path,
        eligible_path,
        soft_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    pm = json.loads(
        protocol_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    actual_protocol_sha = sha256_file(
        protocol_path
    )

    if actual_protocol_sha != pm["protocol_sha256"]:
        raise RuntimeError(
            "v47.5 preregistration SHA mismatch."
        )

    print("=== v47.5b CORRECTED EXTERNAL STATE-LANDSCAPE REPLICATION ===")
    print("Existing v47.5 protocol SHA:", actual_protocol_sha)
    print("Scientific protocol changed: NO")
    print("Parser corrected: YES")
    print("Previous v47.5 scoring performed: NO")
    print()

    (
        probe_to_symbols,
        sample_values,
        sample_source,
        sample_chars,
    ) = parse_family_soft(
        soft_path
    )

    if len(sample_values) != EXPECTED_TOTAL_N:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_N} samples, found {len(sample_values)}."
        )

    sample_class = {}

    for gsm in sorted(
        sample_values
    ):
        sample_class[gsm] = classify_corrected(
            sample_source.get(gsm, ""),
            sample_chars.get(gsm, {}),
        )

    time_counts = Counter(
        tp
        for tp, outcome in sample_class.values()
    )

    baseline = [
        gsm
        for gsm, (tp, outcome)
        in sample_class.items()
        if tp == "BASELINE_PRETREATMENT"
    ]

    post = [
        gsm
        for gsm, (tp, outcome)
        in sample_class.items()
        if tp == "POST_6W"
    ]

    unresolved_time = [
        gsm
        for gsm, (tp, outcome)
        in sample_class.items()
        if tp == "UNRESOLVED"
    ]

    baseline_outcomes = Counter(
        sample_class[gsm][1]
        for gsm in baseline
    )

    post_outcomes = Counter(
        sample_class[gsm][1]
        for gsm in post
    )

    print("CORRECTED SAMPLE AUDIT")
    print("----------------------")
    print("Total samples:", len(sample_values))
    print("Baseline:", len(baseline))
    print("Post 6 weeks:", len(post))
    print("Unresolved timepoint:", len(unresolved_time))
    print()
    print("Baseline MMR:", baseline_outcomes["MMR"])
    print("Baseline none:", baseline_outcomes["NONE"])
    print("Baseline not available:", baseline_outcomes["NOT_AVAILABLE"])
    print("Baseline unresolved outcome:", baseline_outcomes["UNRESOLVED"])
    print()
    print("Post-6w MMR:", post_outcomes["MMR"])
    print("Post-6w none:", post_outcomes["NONE"])
    print("Post-6w not available:", post_outcomes["NOT_AVAILABLE"])
    print("Post-6w unresolved outcome:", post_outcomes["UNRESOLVED"])
    print()

    # Strict historical-count gate before any geometry.
    expected_ok = (
        len(baseline) == EXPECTED_BASELINE_N
        and len(post) == EXPECTED_POST_N
        and len(unresolved_time) == 0
        and baseline_outcomes["MMR"] == EXPECTED_MMR_N
        and baseline_outcomes["NONE"] == EXPECTED_NONE_N
        and baseline_outcomes["NOT_AVAILABLE"] == EXPECTED_UNAVAILABLE_N
        and baseline_outcomes["UNRESOLVED"] == 0
        and post_outcomes["MMR"] == 29
        and post_outcomes["NONE"] == 30
        and post_outcomes["NOT_AVAILABLE"] == 4
        and post_outcomes["UNRESOLVED"] == 0
    )

    if not expected_ok:
        raise RuntimeError(
            "Corrected parser does not reproduce the frozen historical "
            "135/63 and 69/59/7 counts. STOPPED BEFORE GEOMETRY."
        )

    print("PARSER COUNT GATE: PASS")
    print("Proceeding with the already-preregistered external landscape.")
    print()

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

    symbol_to_probes = defaultdict(list)

    for probe, symbols in probe_to_symbols.items():
        for symbol in symbols:
            if symbol in eligible_set:
                symbol_to_probes[symbol].append(
                    probe
                )

    measurable_genes = [
        gene
        for gene in eligible
        if symbol_to_probes.get(gene)
    ]

    if len(measurable_genes) < TOP_K:
        raise RuntimeError(
            "Fewer than 256 eligible genes measurable on GPL570."
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
                raise RuntimeError(
                    f"Missing expression: {gsm}, {gene}"
                )

            X[i, j] = float(
                np.mean(vals)
            )

    if not np.all(
        np.isfinite(X)
    ):
        raise RuntimeError(
            "Non-finite expression matrix."
        )

    # LABEL-BLIND FEATURE SELECTION.
    variances = np.var(
        X,
        axis=0,
        ddof=0,
    )

    order = np.argsort(
        -variances,
        kind="mergesort",
    )

    top_idx = order[
        :TOP_K
    ]

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

    if np.any(
        sd == 0
    ):
        zero = np.where(
            sd == 0
        )[0]

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

    # LABEL-BLIND PCA.
    pca = PCA(
        n_components=PCA_R,
        svd_solver="full",
    )

    Y = pca.fit_transform(
        Z
    )

    # LABEL-BLIND LOCAL GEOMETRY.
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

    feature_out = (
        ds
        / "v47_5b_GSE44589_label_blind_top256_features.tsv"
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

    geometry_out = (
        ds
        / "v47_5b_GSE44589_label_blind_geometry.tsv"
    )

    geometry.to_csv(
        geometry_out,
        sep="\t",
    )

    geometry_sha = sha256_file(
        geometry_out
    )

    loadings_out = (
        ds
        / "v47_5b_GSE44589_PCA16_loadings.tsv"
    )

    pd.DataFrame(
        pca.components_.T,
        index=top_genes,
        columns=pc_cols,
    ).rename_axis(
        "gene"
    ).to_csv(
        loadings_out,
        sep="\t",
    )

    geometry_manifest_out = (
        ds
        / "v47_5b_GSE44589_label_blind_geometry_manifest.json"
    )

    geometry_manifest_out.write_text(
        json.dumps(
            {
                "version": VERSION,
                "status": "EXTERNAL_LABEL_BLIND_GEOMETRY_FROZEN",
                "scientific_protocol_version": "v47.5",
                "scientific_protocol_sha256": actual_protocol_sha,
                "parser_correction_only": True,
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # =========================================================
    # ONLY AFTER GEOMETRY SHA FREEZE: attach outcomes.
    # =========================================================
    evaluable = [
        gsm
        for gsm in baseline
        if sample_class[gsm][1]
        in {
            "MMR",
            "NONE",
        }
    ]

    if len(evaluable) != EXPECTED_EVALUABLE_N:
        raise RuntimeError(
            f"Expected {EXPECTED_EVALUABLE_N} evaluable, found {len(evaluable)}."
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

    if int(
        np.sum(y == 0)
    ) != EXPECTED_MMR_N:
        raise RuntimeError(
            "MMR overlay count mismatch."
        )

    if int(
        np.sum(y == 1)
    ) != EXPECTED_NONE_N:
        raise RuntimeError(
            "NONE overlay count mismatch."
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
        np.mean(null)
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
            observed - null_mean
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

    stats = group_geometry_stats(
        geometry_eval,
        y,
    )

    stats_out = (
        ds
        / "v47_5b_GSE44589_group_geometry_descriptives.tsv"
    )

    stats.to_csv(
        stats_out,
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
            "NONE_NO_MMR",
        ),
    )

    overlay_out = (
        ds
        / "v47_5b_GSE44589_geometry_with_labels.tsv"
    )

    overlay.to_csv(
        overlay_out,
        sep="\t",
    )

    null_out = (
        ds
        / "v47_5b_GSE44589_centroid_null_10000.tsv"
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

    dev_features_path = (
        ds
        / "v47_2_label_blind_top256_features.tsv"
    )

    if not dev_features_path.exists():
        raise FileNotFoundError(
            dev_features_path
        )

    dev_features = set(
        pd.read_csv(
            dev_features_path,
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

    overlap_jaccard = float(
        overlap_n
        / (
            2 * TOP_K
            - overlap_n
        )
    )

    summary_out = (
        ds
        / "v47_5b_external_state_landscape_replication_summary.txt"
    )

    lines = [
        "=== Soft Spaces / CML v47.5b CORRECTED EXTERNAL STATE-LANDSCAPE REPLICATION ===",
        "",
        "PARSER CORRECTION",
        "-----------------",
        "v47.5 scientific protocol changed: NO",
        "Timepoint parser corrected:       YES",
        "Previous v47.5 geometry/scoring:  NOT PERFORMED",
        f"v47.5 protocol SHA verified:      {actual_protocol_sha}",
        "",
        "CORRECTED SAMPLE AUDIT",
        "----------------------",
        f"Total samples:                    {len(sample_values)}",
        f"Baseline pretreatment:            {len(baseline)}",
        f"Post 6 weeks:                     {len(post)}",
        f"Baseline MMR:                     {baseline_outcomes['MMR']}",
        f"Baseline none / no MMR:           {baseline_outcomes['NONE']}",
        f"Baseline unavailable:             {baseline_outcomes['NOT_AVAILABLE']}",
        "",
        "EXTERNAL LABEL-BLIND LANDSCAPE",
        "------------------------------",
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
        f"None / no MMR:                   {int(np.sum(y == 1))}",
        f"Evaluable:                       {len(y)}",
        f"Unavailable:                     {EXPECTED_UNAVAILABLE_N}",
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
        f"GSE130404 Top256 intersection:   {overlap_n}/256",
        f"Top256 Jaccard:                  {overlap_jaccard:.6f}",
        "",
        "SECONDARY GROUP GEOMETRY",
        "------------------------",
    ]

    for _, row in stats.iterrows():
        lines += [
            f"{row['group']}:",
            f"  n:                              {int(row['n'])}",
            f"  mean density proxy:             {row['mean_density_proxy']:.9f}",
            f"  mean local anisotropy:          {row['mean_local_anisotropy']:.9f}",
            f"  mean distance to centroid:      {row['mean_distance_to_centroid']:.9f}",
        ]

    lines += [
        "",
        f"v47.5b STATUS: {status}",
        "",
        "INTERPRETATION LIMIT",
        "--------------------",
        "This is an independent-cohort, cross-endpoint replication test of",
        "clinical-group organization in an independently constructed",
        "label-blind CML expression landscape.",
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
        / "v47_5b_manifest.json"
    )

    manifest_out.write_text(
        json.dumps(
            {
                "version": VERSION,
                "status": status,
                "scientific_protocol_version": "v47.5",
                "scientific_protocol_sha256": actual_protocol_sha,
                "parser_correction_only": True,
                "prior_v47_5_scoring_performed": False,
                "cohort": GSE,
                "platform": GPL,
                "endpoint_relation": "cross-endpoint",
                "total_n": len(sample_values),
                "baseline_n": len(baseline),
                "post_6w_n": len(post),
                "evaluable_n": len(y),
                "mmr_n": int(np.sum(y == 0)),
                "no_mmr_n": int(np.sum(y == 1)),
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
                "primary_pass": bool(primary_pass),
                "top256_overlap_with_gse130404": overlap_n,
                "top256_jaccard_with_gse130404": overlap_jaccard,
                "external_geometry_sha256": geometry_sha,
                "sha256": {
                    "gse44589_family_soft": sha256_file(soft_path),
                    "external_features": sha256_file(feature_out),
                    "external_geometry": geometry_sha,
                    "external_pca_loadings": sha256_file(loadings_out),
                    "overlay": sha256_file(overlay_out),
                    "group_descriptives": sha256_file(stats_out),
                    "null_distribution": sha256_file(null_out),
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
        feature_out,
        geometry_out,
        geometry_manifest_out,
        loadings_out,
        overlay_out,
        stats_out,
        null_out,
        summary_out,
        manifest_out,
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
