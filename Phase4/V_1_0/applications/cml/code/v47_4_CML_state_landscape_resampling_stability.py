#!/usr/bin/env python3
"""
Soft Spaces / CML
v47.4 — State-landscape resampling stability

This is a one-run preregister-and-execute script.

It first freezes the stability protocol in ../docs and only then runs
the resampling analysis.

Question
--------
Is the v47.3 clinical-group separation robust when the label-blind
state landscape is rebuilt repeatedly on different patient subsets?

Design
------
- Cohort: GSE130404
- 300 repeated stratified 80% subsamples without replacement
- each subsample:
    * Top-256 raw variance selected anew WITHOUT using labels
    * cohort z-standardization fitted anew
    * PCA-16 fitted anew WITHOUT using labels
    * labels attached only after geometry construction
    * observed group-centroid distance computed
    * 500 within-subsample random label permutations
- no classifier
- no predictive threshold
- no outcome-driven feature/PCA selection

Primary stability quantities
----------------------------
For each resample:
    z_null = (observed centroid distance - null mean) / null SD

Frozen stability PASS:
    median z_null > 1.644854
    AND
    at least 75% of resamples have z_null > 0

This is a stability criterion, not a clinical-performance criterion.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


VERSION = "v47.4"
GSE = "GSE130404"
GPL = "GPL10558"

EXPECTED_N = 96
GOOD_N = 83
POOR_N = 13

ELIGIBLE_N = 15890
TOP_K = 256
PCA_R = 16

N_RESAMPLES = 300
SUBSAMPLE_FRACTION = 0.80
GOOD_SUB_N = 66
POOR_SUB_N = 10

N_PERM_PER_RESAMPLE = 500
MASTER_SEED = 47040001

Z_PASS_THRESHOLD = 1.6448536269514722
POSITIVE_FRACTION_THRESHOLD = 0.75


def project_dir():
    return Path(__file__).resolve().parent.parent


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    Parse:
    - GPL10558 annotation
    - sample VALUE tables
    - only the sample-specific metadata needed for the frozen 3-month label

    Feature selection / standardization / PCA never use the labels.
    """
    probe_to_symbols = {}
    sample_values = {}
    sample_labels = {}

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
                sample_values[current_id] = {}
                sample_labels[current_id] = None
                in_platform_table = False
                in_sample_table = False
                sample_header = None
                sample_probe_idx = None
                sample_value_idx = None
                continue

            if (
                current_entity == "SAMPLE"
                and line.startswith("!Sample_characteristics_ch1")
                and "=" in line
            ):
                value = normalize_ws(line.split("=", 1)[1])
                vl = value.lower()

                if (
                    "bcr" in vl
                    or "3 month" in vl
                    or "3-month" in vl
                    or "3 months" in vl
                ):
                    text = (
                        vl.replace("≤", "<=")
                        .replace("≥", ">=")
                        .replace("％", "%")
                    )

                    if re.search(r"(?:<|<=)\s*10\s*%?", text):
                        old = sample_labels[current_id]
                        if old is not None and old != 0:
                            raise RuntimeError(
                                f"Conflicting label for {current_id}"
                            )
                        sample_labels[current_id] = 0

                    elif re.search(r"(?:>|>=)\s*10\s*%?", text):
                        old = sample_labels[current_id]
                        if old is not None and old != 1:
                            raise RuntimeError(
                                f"Conflicting label for {current_id}"
                            )
                        sample_labels[current_id] = 1

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
                                "Could not identify GPL10558 symbol column."
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
                        norm = [
                            x.strip().lower()
                            for x in fields
                        ]

                        if "id_ref" not in norm or "value" not in norm:
                            raise RuntimeError(
                                f"{current_id}: missing ID_REF/VALUE."
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

    return probe_to_symbols, sample_values, sample_labels


def centroid_distance(Y, y):
    c0 = np.mean(Y[y == 0], axis=0)
    c1 = np.mean(Y[y == 1], axis=0)
    return float(np.linalg.norm(c1 - c0))


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

    v470_protocol = (
        docs
        / "v47_0_CML_STATE_LANDSCAPE_PREREGISTRATION.txt"
    )

    v472_manifest = ds / "v47_2_manifest.json"
    v473_manifest = ds / "v47_3_manifest.json"
    eligible_path = ds / "v44_1b_eligible_genes.txt"
    soft_path = data_dir / "GSE130404_family.soft.gz"

    for p in [
        v470_protocol,
        v472_manifest,
        v473_manifest,
        eligible_path,
        soft_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    m472 = json.loads(
        v472_manifest.read_text(encoding="utf-8")
    )

    m473 = json.loads(
        v473_manifest.read_text(encoding="utf-8")
    )

    if m472.get("status") != "LABEL_BLIND_GEOMETRY_FROZEN":
        raise RuntimeError("v47.2 geometry state mismatch.")

    if m473.get("primary_pass") is not True:
        raise RuntimeError(
            "v47.3 primary landscape separation is not recorded as PASS."
        )

    # ---------------------------------------------------------
    # Freeze v47.4 protocol BEFORE executing resampling.
    # ---------------------------------------------------------
    protocol = f"""Soft Spaces / CML
v47.4 — STATE-LANDSCAPE RESAMPLING STABILITY PREREGISTRATION

STATUS
------
FROZEN BEFORE RESAMPLING EXECUTION

PURPOSE
-------
Test whether the v47.3 clinical-group separation remains present when the
label-blind state landscape is repeatedly rebuilt on different subsets of
GSE130404 patients.

RESAMPLING
----------
Repeated stratified subsampling without replacement.

Number of resamples:
    {N_RESAMPLES}

Fraction:
    {SUBSAMPLE_FRACTION:.2f}

Per resample:
    GOOD / 3-month BCR-ABL1 <10%: {GOOD_SUB_N}
    POOR / 3-month BCR-ABL1 >10%: {POOR_SUB_N}
    total: {GOOD_SUB_N + POOR_SUB_N}

The labels are used only to preserve group counts during resampling and for
the later overlay test.

They are NOT used for feature selection, standardization or PCA.

LANDSCAPE REBUILD IN EACH RESAMPLE
----------------------------------
1. restrict to the frozen v44 transport-aware 15,890-gene universe;
2. select Top-256 by raw variance in the resampled patients, label-blind;
3. cohort z-standardize each selected gene, ddof=0;
4. fit PCA-16, label-blind;
5. only then overlay the frozen 3-month clinical labels.

PRIMARY RESAMPLE EFFECT
-----------------------
Observed Euclidean distance between the GOOD and POOR centroids in PCA-16.

WITHIN-RESAMPLE NULL
--------------------
Randomly permute the labels {N_PERM_PER_RESAMPLE} times while preserving
the same class counts.

For each resample:

    z_null =
        (observed_distance - mean(null_distance))
        / sd(null_distance)

PRIMARY STABILITY RULE
----------------------
PASS requires BOTH:

1. median z_null > {Z_PASS_THRESHOLD:.12f}

AND

2. fraction of resamples with z_null > 0 >= {POSITIVE_FRACTION_THRESHOLD:.2f}

The first criterion corresponds to a typical one-sided 5% normal-reference
separation at the median resample.

The second requires the direction of excess separation to persist in at
least 75% of resamples.

SECONDARY DESCRIPTIVES
----------------------
Report:
- median observed centroid distance
- median null distance
- median empirical within-resample p
- fraction with empirical p <= 0.05
- z_null quantiles
- Top-256 feature-overlap statistics relative to the original v47.2 Top-256

CLAIM LIMIT
-----------
A PASS supports resampling robustness of the label-blind state-geometry
separation inside GSE130404.

It does NOT establish:
- independent external replication
- prediction
- prognosis
- clinical utility
- causal biology
- dormancy mechanism
- quantum biology
- quantum advantage
"""

    protocol_path = (
        docs
        / "v47_4_CML_STATE_LANDSCAPE_STABILITY_PREREGISTRATION.txt"
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
        / "v47_4_CML_STATE_LANDSCAPE_STABILITY_PREREGISTRATION_manifest.json"
    )

    protocol_manifest_path.write_text(
        json.dumps(
            {
                "version": VERSION,
                "status": "FROZEN_BEFORE_RESAMPLING",
                "n_resamples": N_RESAMPLES,
                "subsample_fraction": SUBSAMPLE_FRACTION,
                "good_per_resample": GOOD_SUB_N,
                "poor_per_resample": POOR_SUB_N,
                "top_k": TOP_K,
                "pca_r": PCA_R,
                "n_permutations_per_resample": N_PERM_PER_RESAMPLE,
                "master_seed": MASTER_SEED,
                "median_z_pass_threshold": Z_PASS_THRESHOLD,
                "positive_fraction_threshold": POSITIVE_FRACTION_THRESHOLD,
                "protocol_sha256": protocol_sha,
                "source_sha256": {
                    "v47_0_protocol": sha256_file(v470_protocol),
                    "v47_2_manifest": sha256_file(v472_manifest),
                    "v47_3_manifest": sha256_file(v473_manifest),
                    "eligible_genes": sha256_file(eligible_path),
                    "gse130404_family_soft": sha256_file(soft_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lock_path = (
        docs
        / "v47_4_CML_state_landscape_stability_protocol_lock.py"
    )

    lock_path.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import hashlib\n"
        f'EXPECTED_SHA256="{protocol_sha}"\n'
        'p=Path(__file__).resolve().parent/'
        '"v47_4_CML_STATE_LANDSCAPE_STABILITY_PREREGISTRATION.txt"\n'
        'a=hashlib.sha256(p.read_bytes()).hexdigest()\n'
        'print("Expected:",EXPECTED_SHA256)\n'
        'print("Actual:  ",a)\n'
        'raise SystemExit("FAIL") if a!=EXPECTED_SHA256 '
        'else print("PASS: v47.4 protocol unchanged.")\n',
        encoding="utf-8",
    )

    print("=== v47.4 STATE-LANDSCAPE STABILITY ===")
    print("Protocol SHA:", protocol_sha)
    print("Protocol frozen BEFORE resampling.")
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
            f"Expected {ELIGIBLE_N} eligible genes."
        )

    eligible_set = set(eligible)

    probe_to_symbols, sample_values, sample_labels = parse_family_soft(
        soft_path
    )

    gsms = sorted(sample_values.keys())

    if len(gsms) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} samples, found {len(gsms)}."
        )

    y = np.asarray(
        [
            -1 if sample_labels.get(gsm) is None
            else int(sample_labels[gsm])
            for gsm in gsms
        ],
        dtype=int,
    )

    if np.any(y < 0):
        missing = [
            gsm for gsm, yy in zip(gsms, y)
            if yy < 0
        ]
        raise RuntimeError(
            "Unresolved labels: " + ", ".join(missing)
        )

    if int(np.sum(y == 0)) != GOOD_N or int(np.sum(y == 1)) != POOR_N:
        raise RuntimeError(
            "Frozen 83/13 label counts not reproduced."
        )

    symbol_to_probes = defaultdict(list)

    for probe, symbols in probe_to_symbols.items():
        for symbol in symbols:
            if symbol in eligible_set:
                symbol_to_probes[symbol].append(probe)

    measurable_genes = [
        g for g in eligible
        if symbol_to_probes.get(g)
    ]

    if len(measurable_genes) < TOP_K:
        raise RuntimeError(
            "Fewer than 256 eligible genes are measurable."
        )

    # Build full 96 x measurable-gene expression matrix once.
    X = np.empty(
        (len(gsms), len(measurable_genes)),
        dtype=float,
    )

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
                raise RuntimeError(
                    f"Missing expression: {gsm}, {gene}"
                )

            X[i, j] = float(
                np.mean(vals)
            )

    if not np.all(np.isfinite(X)):
        raise RuntimeError("Non-finite expression matrix.")

    # Original v47.2 Top256 for overlap descriptives only.
    original_features_path = (
        ds
        / "v47_2_label_blind_top256_features.tsv"
    )

    if not original_features_path.exists():
        raise FileNotFoundError(original_features_path)

    original_features = pd.read_csv(
        original_features_path,
        sep="\t",
    )

    original_top256 = set(
        original_features["gene"].astype(str)
    )

    good_idx = np.where(y == 0)[0]
    poor_idx = np.where(y == 1)[0]

    rng_master = np.random.default_rng(
        MASTER_SEED
    )

    rows = []

    for r in range(N_RESAMPLES):
        selected_good = rng_master.choice(
            good_idx,
            size=GOOD_SUB_N,
            replace=False,
        )

        selected_poor = rng_master.choice(
            poor_idx,
            size=POOR_SUB_N,
            replace=False,
        )

        idx = np.concatenate(
            [selected_good, selected_poor]
        )

        # Randomize row order without changing membership.
        idx = rng_master.permutation(idx)

        Xr = X[idx, :]
        yr = y[idx]

        # Completely label-blind variance selection.
        variances = np.var(
            Xr,
            axis=0,
            ddof=0,
        )

        order = np.argsort(
            -variances,
            kind="mergesort",
        )

        top_idx = order[:TOP_K]
        selected_genes = [
            measurable_genes[j]
            for j in top_idx
        ]

        overlap_n = len(
            original_top256.intersection(
                selected_genes
            )
        )

        Xt = Xr[:, top_idx]

        mu = np.mean(
            Xt,
            axis=0,
        )

        sd = np.std(
            Xt,
            axis=0,
            ddof=0,
        )

        if np.any(sd == 0):
            raise RuntimeError(
                f"Resample {r+1}: zero-SD Top256 gene."
            )

        Z = (
            Xt - mu
        ) / sd

        pca = PCA(
            n_components=PCA_R,
            svd_solver="full",
        )

        Y = pca.fit_transform(
            Z
        )

        observed = centroid_distance(
            Y,
            yr,
        )

        null = np.empty(
            N_PERM_PER_RESAMPLE,
            dtype=float,
        )

        # Derive deterministic independent seed per resample.
        rng_perm = np.random.default_rng(
            MASTER_SEED + 100000 + r
        )

        for pidx in range(
            N_PERM_PER_RESAMPLE
        ):
            yp = rng_perm.permutation(
                yr
            )

            null[pidx] = centroid_distance(
                Y,
                yp,
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

        if null_sd <= 0:
            raise RuntimeError(
                f"Resample {r+1}: null SD is zero."
            )

        z_null = float(
            (observed - null_mean)
            / null_sd
        )

        exceed = int(
            np.sum(
                null >= observed
            )
        )

        p_emp = float(
            (1 + exceed)
            / (1 + N_PERM_PER_RESAMPLE)
        )

        rows.append({
            "resample": r + 1,
            "n_good": int(np.sum(yr == 0)),
            "n_poor": int(np.sum(yr == 1)),
            "observed_centroid_distance": observed,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "z_null": z_null,
            "empirical_p": p_emp,
            "top256_overlap_with_v47_2": overlap_n,
            "top256_jaccard_with_v47_2": (
                overlap_n
                / (2 * TOP_K - overlap_n)
            ),
            "pca16_cumulative_variance": float(
                np.sum(
                    pca.explained_variance_ratio_
                )
            ),
        })

        if (r + 1) % 25 == 0:
            print(
                f"Completed {r+1}/{N_RESAMPLES} resamples"
            )

    res = pd.DataFrame(rows)

    median_z = float(
        res["z_null"].median()
    )

    positive_fraction = float(
        np.mean(
            res["z_null"] > 0
        )
    )

    p05_fraction = float(
        np.mean(
            res["empirical_p"] <= 0.05
        )
    )

    stability_pass = (
        median_z > Z_PASS_THRESHOLD
        and
        positive_fraction >= POSITIVE_FRACTION_THRESHOLD
    )

    status = (
        "STABILITY PASS"
        if stability_pass
        else "STABILITY FAIL"
    )

    q = res["z_null"].quantile(
        [0.025, 0.25, 0.5, 0.75, 0.975]
    )

    results_out = (
        ds
        / "v47_4_state_landscape_resampling_results.tsv"
    )

    res.to_csv(
        results_out,
        sep="\t",
        index=False,
    )

    summary_out = (
        ds
        / "v47_4_state_landscape_stability_summary.txt"
    )

    lines = [
        "=== Soft Spaces / CML v47.4 STATE-LANDSCAPE RESAMPLING STABILITY ===",
        "",
        "PROTOCOL",
        "--------",
        f"v47.4 protocol SHA:              {protocol_sha}",
        f"Resamples:                       {N_RESAMPLES}",
        f"Patients per resample:           {GOOD_SUB_N + POOR_SUB_N}",
        f"GOOD per resample:               {GOOD_SUB_N}",
        f"POOR per resample:               {POOR_SUB_N}",
        f"Permutations per resample:       {N_PERM_PER_RESAMPLE}",
        "",
        "LABEL-BLIND REBUILD",
        "-------------------",
        "Top-256 reselected each time:    YES, without labels",
        "Standardization refit each time: YES, without labels",
        "PCA-16 refit each time:          YES, without labels",
        "Classifier fitted:               NO",
        "",
        "PRIMARY STABILITY",
        "-----------------",
        f"Median NULL-z:                    {median_z:.6f}",
        f"Required median NULL-z:           > {Z_PASS_THRESHOLD:.6f}",
        f"Fraction NULL-z > 0:              {positive_fraction:.6f}",
        f"Required positive fraction:       >= {POSITIVE_FRACTION_THRESHOLD:.6f}",
        "",
        f"z q2.5%:                          {q.loc[0.025]:.6f}",
        f"z q25%:                           {q.loc[0.25]:.6f}",
        f"z median:                         {q.loc[0.5]:.6f}",
        f"z q75%:                           {q.loc[0.75]:.6f}",
        f"z q97.5%:                         {q.loc[0.975]:.6f}",
        "",
        "SECONDARY",
        "---------",
        f"Median observed distance:         {res['observed_centroid_distance'].median():.6f}",
        f"Median NULL mean distance:        {res['null_mean'].median():.6f}",
        f"Median empirical p:               {res['empirical_p'].median():.6f}",
        f"Fraction empirical p <= 0.05:     {p05_fraction:.6f}",
        f"Median Top256 overlap / 256:      {res['top256_overlap_with_v47_2'].median():.1f}",
        f"Mean Top256 overlap / 256:        {res['top256_overlap_with_v47_2'].mean():.3f}",
        f"Median Top256 Jaccard:            {res['top256_jaccard_with_v47_2'].median():.6f}",
        f"Median PCA16 cumulative variance: {res['pca16_cumulative_variance'].median():.6f}",
        "",
        f"v47.4 STATUS: {status}",
        "",
        "INTERPRETATION LIMIT",
        "--------------------",
        "This tests internal resampling robustness of the label-blind",
        "state-landscape separation within GSE130404.",
        "",
        "It is not independent external replication and does not establish",
        "prediction, prognosis, clinical utility, causality, dormancy",
        "mechanism, or quantum advantage.",
    ]

    summary_out.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest_out = (
        ds
        / "v47_4_manifest.json"
    )

    manifest_out.write_text(
        json.dumps(
            {
                "version": VERSION,
                "status": status,
                "n_resamples": N_RESAMPLES,
                "patients_per_resample": GOOD_SUB_N + POOR_SUB_N,
                "good_per_resample": GOOD_SUB_N,
                "poor_per_resample": POOR_SUB_N,
                "n_permutations_per_resample": N_PERM_PER_RESAMPLE,
                "median_z_null": median_z,
                "median_z_pass_threshold": Z_PASS_THRESHOLD,
                "positive_z_fraction": positive_fraction,
                "positive_fraction_threshold": POSITIVE_FRACTION_THRESHOLD,
                "fraction_empirical_p_le_0_05": p05_fraction,
                "median_empirical_p": float(
                    res["empirical_p"].median()
                ),
                "median_top256_overlap": float(
                    res["top256_overlap_with_v47_2"].median()
                ),
                "mean_top256_overlap": float(
                    res["top256_overlap_with_v47_2"].mean()
                ),
                "median_top256_jaccard": float(
                    res["top256_jaccard_with_v47_2"].median()
                ),
                "median_pca16_cumulative_variance": float(
                    res["pca16_cumulative_variance"].median()
                ),
                "classifier_fitted": False,
                "external_replication": False,
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
    print(
        summary_out.read_text(
            encoding="utf-8"
        )
    )

    print("Wrote:")
    print(" ", results_out)
    print(" ", summary_out)
    print(" ", manifest_out)


if __name__ == "__main__":
    main()
