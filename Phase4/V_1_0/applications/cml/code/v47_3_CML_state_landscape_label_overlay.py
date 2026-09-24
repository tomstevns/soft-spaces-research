#!/usr/bin/env python3
"""
Soft Spaces / CML
v47.3 — Clinical-label overlay on frozen label-blind state landscape

Purpose
-------
Overlay the GSE130404 3-month BCR-ABL1 response labels onto the already
SHA-frozen v47.2 PCA-16 geometry.

The geometry is NOT rebuilt or modified.

Primary preregistered test:
    Euclidean distance between the two clinical-group centroids in PCA-16

Null:
    10,000 random permutations of the frozen labels

Empirical one-sided p:
    (1 + # null >= observed) / (1 + Nperm)

Secondary descriptive summaries:
    local density proxy
    local anisotropy
    distance to cohort centroid

No classifier is fit.
No feature selection is rerun.
No PCA is rerun.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v47.3"
GSE = "GSE130404"
EXPECTED_N = 96
EXPECTED_GOOD_N = 83
EXPECTED_POOR_N = 13
N_PERM = 10_000
SEED = 47030001


def project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_ws(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip())


def parse_gse130404_labels(soft_path: Path):
    """
    Parse only sample-specific metadata needed for the 3-month BCR-ABL1 label.

    Frozen classes:
        GOOD / EMR:
            BCR-ABL1 at 3 months < 10%

        POOR:
            BCR-ABL1 at 3 months > 10%

    We inspect Sample_characteristics_ch1 lines sample by sample.
    """
    labels = {}
    current_gsm = None

    with gzip.open(
        soft_path,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if line.startswith("^SAMPLE"):
                current_gsm = line.split("=", 1)[1].strip()
                labels[current_gsm] = None
                continue

            if current_gsm is None:
                continue

            if (
                line.startswith("!Sample_characteristics_ch1")
                and "=" in line
            ):
                value = normalize_ws(
                    line.split("=", 1)[1]
                )

                vl = value.lower()

                # Only inspect characteristics explicitly mentioning
                # BCR-ABL / 3 month / molecular response style fields.
                if (
                    "bcr" not in vl
                    and "3 month" not in vl
                    and "3-month" not in vl
                    and "3 months" not in vl
                ):
                    continue

                # Normalize unicode / spacing variants.
                text = (
                    vl.replace("≤", "<=")
                    .replace("≥", ">=")
                    .replace("％", "%")
                )

                # We deliberately classify only explicit <10 or >10 style
                # 3-month BCR-ABL1 metadata.
                if re.search(
                    r"(?:<|<=)\s*10\s*%?",
                    text,
                ):
                    if labels[current_gsm] is not None and labels[current_gsm] != 0:
                        raise RuntimeError(
                            f"Conflicting label metadata for {current_gsm}: {value}"
                        )
                    labels[current_gsm] = 0

                elif re.search(
                    r"(?:>|>=)\s*10\s*%?",
                    text,
                ):
                    if labels[current_gsm] is not None and labels[current_gsm] != 1:
                        raise RuntimeError(
                            f"Conflicting label metadata for {current_gsm}: {value}"
                        )
                    labels[current_gsm] = 1

    return labels


def centroid_distance(Y: np.ndarray, y: np.ndarray) -> float:
    c0 = np.mean(Y[y == 0], axis=0)
    c1 = np.mean(Y[y == 1], axis=0)
    return float(
        np.linalg.norm(c1 - c0)
    )


def descriptive_group_stats(df: pd.DataFrame, y: np.ndarray):
    rows = []

    for label_value, label_name in [
        (0, "GOOD_EMR_LT10"),
        (1, "POOR_GT10"),
    ]:
        mask = y == label_value

        rows.append({
            "group": label_name,
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
    metadata = (
        project
        / "data"
        / "external"
        / GSE
        / "metadata"
    )

    protocol = docs / "v47_0_CML_STATE_LANDSCAPE_PREREGISTRATION.txt"
    protocol_manifest = docs / "v47_0_CML_STATE_LANDSCAPE_PREREGISTRATION_manifest.json"
    geometry_path = ds / "v47_2_GSE130404_label_blind_geometry.tsv"
    geometry_manifest = ds / "v47_2_manifest.json"
    soft_path = metadata / "GSE130404_family.soft.gz"

    for p in [
        protocol,
        protocol_manifest,
        geometry_path,
        geometry_manifest,
        soft_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    pm = json.loads(
        protocol_manifest.read_text(
            encoding="utf-8"
        )
    )

    gm = json.loads(
        geometry_manifest.read_text(
            encoding="utf-8"
        )
    )

    actual_protocol_sha = sha256_file(
        protocol
    )

    actual_geometry_sha = sha256_file(
        geometry_path
    )

    if actual_protocol_sha != pm["protocol_sha256"]:
        raise RuntimeError(
            "v47.0 protocol SHA mismatch."
        )

    if actual_geometry_sha != gm["geometry_sha256"]:
        raise RuntimeError(
            "v47.2 geometry SHA mismatch."
        )

    if gm.get("status") != "LABEL_BLIND_GEOMETRY_FROZEN":
        raise RuntimeError(
            "v47.2 geometry is not recorded as frozen."
        )

    if gm.get("labels_used") is not False:
        raise RuntimeError(
            "v47.2 manifest unexpectedly reports label use."
        )

    print("=== v47.3 CLINICAL-LABEL OVERLAY ===")
    print("v47.0 protocol SHA:", actual_protocol_sha)
    print("v47.2 geometry SHA:", actual_geometry_sha)
    print("Geometry rebuilt: NO")
    print("PCA refit: NO")
    print("Feature selection rerun: NO")
    print()

    geometry = pd.read_csv(
        geometry_path,
        sep="\t",
        index_col=0,
    )

    if len(geometry) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} geometry rows, found {len(geometry)}."
        )

    pc_cols = [
        f"PC{i}"
        for i in range(1, 17)
    ]

    missing_pc = [
        c for c in pc_cols
        if c not in geometry.columns
    ]

    if missing_pc:
        raise RuntimeError(
            f"Missing PCA columns: {missing_pc}"
        )

    for c in [
        "density_proxy",
        "local_anisotropy",
        "distance_to_centroid",
    ]:
        if c not in geometry.columns:
            raise RuntimeError(
                f"Missing frozen geometry metric: {c}"
            )

    # Attach labels only now.
    labels = parse_gse130404_labels(
        soft_path
    )

    y = []
    unresolved = []

    for gsm in geometry.index.astype(str):
        label = labels.get(
            gsm
        )

        if label is None:
            unresolved.append(
                gsm
            )
            y.append(
                -1
            )
        else:
            y.append(
                int(label)
            )

    y = np.asarray(
        y,
        dtype=int,
    )

    if unresolved:
        raise RuntimeError(
            "Unresolved 3-month BCR-ABL1 labels: "
            + ", ".join(unresolved)
        )

    good_n = int(
        np.sum(y == 0)
    )

    poor_n = int(
        np.sum(y == 1)
    )

    if good_n != EXPECTED_GOOD_N or poor_n != EXPECTED_POOR_N:
        raise RuntimeError(
            "Frozen label-count mismatch: "
            f"GOOD={good_n}, POOR={poor_n}; "
            f"expected {EXPECTED_GOOD_N}/{EXPECTED_POOR_N}."
        )

    Y = geometry[
        pc_cols
    ].to_numpy(
        dtype=float
    )

    observed = centroid_distance(
        Y,
        y,
    )

    rng = np.random.default_rng(
        SEED
    )

    null = np.empty(
        N_PERM,
        dtype=float,
    )

    for i in range(N_PERM):
        yp = rng.permutation(
            y
        )

        null[i] = centroid_distance(
            Y,
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

    null_q50 = float(
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

    effect_vs_null_mean = float(
        observed - null_mean
    )

    null_z = (
        float(
            (observed - null_mean)
            / null_sd
        )
        if null_sd > 0
        else float("nan")
    )

    # Freeze overlay table.
    overlay = geometry.copy()

    overlay.insert(
        0,
        "clinical_group",
        np.where(
            y == 1,
            "POOR_GT10",
            "GOOD_EMR_LT10",
        ),
    )

    overlay_out = (
        ds
        / "v47_3_GSE130404_geometry_with_labels.tsv"
    )

    overlay.to_csv(
        overlay_out,
        sep="\t",
    )

    group_stats = descriptive_group_stats(
        geometry,
        y,
    )

    group_stats_out = (
        ds
        / "v47_3_group_geometry_descriptives.tsv"
    )

    group_stats.to_csv(
        group_stats_out,
        sep="\t",
        index=False,
    )

    null_out = (
        ds
        / "v47_3_centroid_distance_null_10000.tsv"
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

    # Primary decision: preregistered p <= .05.
    primary_pass = (
        p_emp <= 0.05
    )

    status = (
        "PRIMARY LANDSCAPE SEPARATION PASS"
        if primary_pass
        else "PRIMARY LANDSCAPE SEPARATION FAIL"
    )

    summary_out = (
        ds
        / "v47_3_state_landscape_label_overlay_summary.txt"
    )

    lines = [
        "=== Soft Spaces / CML v47.3 CLINICAL-LABEL OVERLAY ===",
        "",
        "FROZEN GEOMETRY",
        "---------------",
        f"v47.2 geometry SHA:              {actual_geometry_sha}",
        "Geometry rebuilt:                NO",
        "PCA refit:                       NO",
        "Feature selection rerun:         NO",
        "",
        "CLINICAL LABELS",
        "---------------",
        "3-month BCR-ABL1 <10%:           83",
        "3-month BCR-ABL1 >10%:           13",
        "Total:                           96",
        "",
        "PRIMARY TEST",
        "------------",
        "Metric: PCA-16 group-centroid Euclidean distance",
        f"Observed centroid distance:      {observed:.9f}",
        "",
        f"Permutations:                    {N_PERM}",
        f"Random seed:                     {SEED}",
        f"NULL mean:                       {null_mean:.9f}",
        f"NULL SD:                         {null_sd:.9f}",
        f"NULL q2.5%:                      {null_q025:.9f}",
        f"NULL median:                     {null_q50:.9f}",
        f"NULL q97.5%:                     {null_q975:.9f}",
        f"Observed - NULL mean:            {effect_vs_null_mean:+.9f}",
        f"Observed NULL-z:                 {null_z:+.6f}",
        f"NULL >= observed:                {exceed}",
        f"Empirical one-sided p:           {p_emp:.9f}",
        "",
        f"v47.3 STATUS: {status}",
        "",
        "SECONDARY DESCRIPTIVE GEOMETRY",
        "------------------------------",
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
        "INTERPRETATION",
        "--------------",
        "The primary test asks whether independently defined clinical groups",
        "occupy more separated regions of the already-frozen label-blind",
        "PCA-16 state geometry than expected under random label assignment.",
        "",
        "No predictive classifier was fitted in this test.",
        "",
        "A PASS does not establish diagnosis, prognosis, causal biology,",
        "dormancy mechanism, clinical utility, or quantum advantage.",
    ]

    summary_out.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest_out = (
        ds
        / "v47_3_manifest.json"
    )

    manifest = {
        "version": VERSION,
        "status": status,
        "cohort": GSE,
        "sample_n": EXPECTED_N,
        "good_emr_lt10_n": good_n,
        "poor_gt10_n": poor_n,
        "geometry_rebuilt": False,
        "pca_refit": False,
        "feature_selection_rerun": False,
        "primary_test": "PCA16 group-centroid Euclidean distance",
        "observed_centroid_distance": observed,
        "n_permutations": N_PERM,
        "random_seed": SEED,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_q025": null_q025,
        "null_median": null_q50,
        "null_q975": null_q975,
        "observed_minus_null_mean": effect_vs_null_mean,
        "null_z": null_z,
        "null_ge_observed": exceed,
        "empirical_one_sided_p": p_emp,
        "primary_pass_threshold": 0.05,
        "primary_pass": bool(
            primary_pass
        ),
        "sha256": {
            "v47_0_protocol": actual_protocol_sha,
            "v47_2_geometry": actual_geometry_sha,
            "gse130404_family_soft": sha256_file(
                soft_path
            ),
            "overlay_table": sha256_file(
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
    }

    manifest_out.write_text(
        json.dumps(
            manifest,
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
        overlay_out,
        group_stats_out,
        null_out,
        summary_out,
        manifest_out,
    ]:
        print(
            " ",
            p,
        )


if __name__ == "__main__":
    main()
