#!/usr/bin/env python3
"""
Soft Spaces / CML
v48.4 — Diagnostic decomposition of v48.3 joint FAIL

Purpose
-------
Explain the already-frozen v48.3 joint three-cohort FAIL.

NO new hypothesis.
NO new threshold.
NO retuning.
NO rescoring against alternative rules.
NO rescue.

This script only reads already-created v48.3 artifacts and reports:

1. Which pair determined the minimum spectrum cosine.
2. Which pair determined the maximum distance-W1.
3. Whether the spectral joint gate failed, the distance gate failed, or both.
4. Absolute and relative margin to the preregistered NULL boundary.
5. Pairwise geometry ordering across AB, AC, BC.
6. A concise diagnostic interpretation.

Pair labels
-----------
A = GSE130404
B = GSE44589
C = GSE14671
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v48.4"


def project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pct_margin_high(observed: float, threshold: float):
    if threshold == 0:
        return None
    return (observed - threshold) / abs(threshold)


def pct_margin_low(observed: float, threshold: float):
    if threshold == 0:
        return None
    return (threshold - observed) / abs(threshold)


def fmt_pct(x):
    if x is None:
        return "NA"
    return f"{100.0*x:+.3f}%"


def main():
    project = project_dir()
    docs = project / "docs"
    ds = project / "results" / "direct_subspace"

    manifest_path = (
        docs
        / "v48_3f_CML_THREE_COHORT_CLOSURE_manifest.json"
    )

    closure_path = (
        docs
        / "v48_3f_CML_THREE_COHORT_CLOSURE.txt"
    )

    summary_path = (
        ds
        / "v48_3_master_summary.txt"
    )

    joint_null_path = (
        ds
        / "v48_3e_joint_three_cohort_null_500.tsv"
    )

    pairwise_null_path = (
        ds
        / "v48_3c_pairwise_third_cohort_null_1000.tsv"
    )

    robust_path = (
        ds
        / "v48_3d_three_cohort_resampling_results.tsv"
    )

    required = [
        manifest_path,
        closure_path,
        summary_path,
        joint_null_path,
        pairwise_null_path,
        robust_path,
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest.get("status") != "CLOSED":
        raise RuntimeError(
            "v48.3 closure is not recorded as CLOSED."
        )

    if manifest.get(
        "three_cohort_geometric_invariance_supported"
    ) is not False:
        raise RuntimeError(
            "v48.3 is not recorded as overall NOT SUPPORTED."
        )

    pairwise = manifest["pairwise"]
    joint = manifest["joint"]

    # ---------------------------------------------------------
    # Extract observed pairwise metrics.
    # ---------------------------------------------------------

    pairs = {
        "AB_GSE130404_vs_GSE44589": {
            "spectrum_cosine": float(
                pairwise["AB"]["spectrum_cosine"]
            ),
            "distance_w1": float(
                pairwise["AB"]["distance_w1"]
            ),
        },
        "AC_GSE130404_vs_GSE14671": {
            "spectrum_cosine": float(
                pairwise["AC"]["spectrum_cosine"]
            ),
            "distance_w1": float(
                pairwise["AC"]["distance_w1"]
            ),
        },
        "BC_GSE44589_vs_GSE14671": {
            "spectrum_cosine": float(
                pairwise["BC"]["spectrum_cosine"]
            ),
            "distance_w1": float(
                pairwise["BC"]["distance_w1"]
            ),
        },
    }

    spectrum_values = {
        k: v["spectrum_cosine"]
        for k, v in pairs.items()
    }

    distance_values = {
        k: v["distance_w1"]
        for k, v in pairs.items()
    }

    worst_spectrum_pair = min(
        spectrum_values,
        key=spectrum_values.get,
    )

    worst_distance_pair = max(
        distance_values,
        key=distance_values.get,
    )

    best_spectrum_pair = max(
        spectrum_values,
        key=spectrum_values.get,
    )

    best_distance_pair = min(
        distance_values,
        key=distance_values.get,
    )

    observed_min_cos = float(
        joint[
            "observed_min_pairwise_spectrum_cosine"
        ]
    )

    q95_min_cos = float(
        joint[
            "null_q95_min_cosine"
        ]
    )

    p_spectrum = float(
        joint["p_spectrum"]
    )

    spectrum_gate = bool(
        joint["spectrum_gate"]
    )

    observed_max_w1 = float(
        joint[
            "observed_max_pairwise_distance_w1"
        ]
    )

    q05_max_w1 = float(
        joint[
            "null_q05_max_w1"
        ]
    )

    p_distance = float(
        joint["p_distance"]
    )

    distance_gate = bool(
        joint["distance_gate"]
    )

    # ---------------------------------------------------------
    # Consistency checks.
    # ---------------------------------------------------------

    if not np.isclose(
        spectrum_values[
            worst_spectrum_pair
        ],
        observed_min_cos,
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Manifest inconsistency: worst pair cosine != joint observed minimum."
        )

    if not np.isclose(
        distance_values[
            worst_distance_pair
        ],
        observed_max_w1,
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Manifest inconsistency: worst pair W1 != joint observed maximum."
        )

    # ---------------------------------------------------------
    # Margins to preregistered joint thresholds.
    # ---------------------------------------------------------

    spectrum_abs_margin = (
        observed_min_cos
        - q95_min_cos
    )

    distance_abs_margin = (
        q05_max_w1
        - observed_max_w1
    )

    spectrum_rel_margin = pct_margin_high(
        observed_min_cos,
        q95_min_cos,
    )

    distance_rel_margin = pct_margin_low(
        observed_max_w1,
        q05_max_w1,
    )

    # ---------------------------------------------------------
    # Read NULLs for percentile position / z-like diagnostics.
    # Descriptive only; does not change PASS/FAIL.
    # ---------------------------------------------------------

    joint_null = pd.read_csv(
        joint_null_path,
        sep="\t",
    )

    null_min_cos = joint_null[
        "minimum_pairwise_spectrum_cosine"
    ].to_numpy(float)

    null_max_w1 = joint_null[
        "maximum_pairwise_distance_w1"
    ].to_numpy(float)

    cos_null_mean = float(
        np.mean(null_min_cos)
    )

    cos_null_sd = float(
        np.std(
            null_min_cos,
            ddof=1,
        )
    )

    w1_null_mean = float(
        np.mean(null_max_w1)
    )

    w1_null_sd = float(
        np.std(
            null_max_w1,
            ddof=1,
        )
    )

    cos_z = (
        (observed_min_cos - cos_null_mean)
        / cos_null_sd
        if cos_null_sd > 0
        else float("nan")
    )

    w1_z = (
        (w1_null_mean - observed_max_w1)
        / w1_null_sd
        if w1_null_sd > 0
        else float("nan")
    )

    cos_percentile = float(
        100.0
        * np.mean(
            null_min_cos
            <= observed_min_cos
        )
    )

    # For W1, lower is better. This is the percent of NULLs
    # with value >= observed (descriptive "better-than-NULL" position).
    w1_better_than_null_pct = float(
        100.0
        * np.mean(
            null_max_w1
            >= observed_max_w1
        )
    )

    # ---------------------------------------------------------
    # Robustness context from existing v48.3d only.
    # ---------------------------------------------------------

    robust = pd.read_csv(
        robust_path,
        sep="\t",
    )

    robust_diag = {}

    for col in [
        "AC_z_spectrum",
        "AC_z_distance",
        "BC_z_spectrum",
        "BC_z_distance",
    ]:
        robust_diag[col] = {
            "median": float(
                robust[col].median()
            ),
            "q025": float(
                robust[col].quantile(
                    0.025
                )
            ),
            "q975": float(
                robust[col].quantile(
                    0.975
                )
            ),
            "positive_fraction": float(
                np.mean(
                    robust[col] > 0
                )
            ),
        }

    # ---------------------------------------------------------
    # Determine failure mode.
    # ---------------------------------------------------------

    if (
        not spectrum_gate
        and not distance_gate
    ):
        failure_mode = (
            "BOTH JOINT GATES FAILED"
        )

    elif not spectrum_gate:
        failure_mode = (
            "SPECTRAL JOINT GATE FAILED ONLY"
        )

    elif not distance_gate:
        failure_mode = (
            "DISTANCE-GEOMETRY JOINT GATE FAILED ONLY"
        )

    else:
        failure_mode = (
            "UNEXPECTED: BOTH JOINT GATES PASS"
        )

    # ---------------------------------------------------------
    # Outputs.
    # ---------------------------------------------------------

    pair_table = pd.DataFrame([
        {
            "pair": pair_name,
            "spectrum_cosine": values[
                "spectrum_cosine"
            ],
            "distance_w1": values[
                "distance_w1"
            ],
            "is_worst_spectrum_pair": (
                pair_name
                == worst_spectrum_pair
            ),
            "is_worst_distance_pair": (
                pair_name
                == worst_distance_pair
            ),
            "is_best_spectrum_pair": (
                pair_name
                == best_spectrum_pair
            ),
            "is_best_distance_pair": (
                pair_name
                == best_distance_pair
            ),
        }
        for pair_name, values in pairs.items()
    ])

    pair_out = (
        ds
        / "v48_4_joint_fail_pairwise_diagnostic.tsv"
    )

    pair_table.to_csv(
        pair_out,
        sep="\t",
        index=False,
    )

    summary_out = (
        ds
        / "v48_4_joint_fail_diagnostic_summary.txt"
    )

    summary = f"""=== Soft Spaces / CML v48.4 JOINT-FAIL DIAGNOSTIC ===

PURPOSE
-------
Diagnostic decomposition of the already-frozen v48.3 joint FAIL.

NO new hypothesis.
NO new thresholds.
NO tuning.
NO rescue.

v48.3 FINAL STATUS
------------------
Pairwise third-cohort replication:      {"PASS" if manifest["pairwise_overall_pass"] else "FAIL"}
Patient-resampling robustness:          {"PASS" if manifest["robustness_overall_pass"] else "FAIL"}
Joint three-cohort invariance:           {"PASS" if manifest["joint_overall_pass"] else "FAIL"}
Overall three-cohort invariance claim:  {"SUPPORTED" if manifest["three_cohort_geometric_invariance_supported"] else "NOT SUPPORTED"}

FAILURE MODE
------------
{failure_mode}

PAIRWISE OBSERVED GEOMETRY
--------------------------
AB = GSE130404 vs GSE44589
    spectrum cosine:   {pairs["AB_GSE130404_vs_GSE44589"]["spectrum_cosine"]:.9f}
    distance W1:       {pairs["AB_GSE130404_vs_GSE44589"]["distance_w1"]:.9f}

AC = GSE130404 vs GSE14671
    spectrum cosine:   {pairs["AC_GSE130404_vs_GSE14671"]["spectrum_cosine"]:.9f}
    distance W1:       {pairs["AC_GSE130404_vs_GSE14671"]["distance_w1"]:.9f}

BC = GSE44589 vs GSE14671
    spectrum cosine:   {pairs["BC_GSE44589_vs_GSE14671"]["spectrum_cosine"]:.9f}
    distance W1:       {pairs["BC_GSE44589_vs_GSE14671"]["distance_w1"]:.9f}

WORST / BEST PAIRS
------------------
Worst spectrum pair:
    {worst_spectrum_pair}

Best spectrum pair:
    {best_spectrum_pair}

Worst distance pair:
    {worst_distance_pair}

Best distance pair:
    {best_distance_pair}

JOINT SPECTRAL GATE
-------------------
Observed minimum pairwise cosine:
    {observed_min_cos:.9f}

NULL q95:
    {q95_min_cos:.9f}

Absolute margin (observed - threshold):
    {spectrum_abs_margin:+.9f}

Relative margin:
    {fmt_pct(spectrum_rel_margin)}

Empirical p:
    {p_spectrum:.9f}

Descriptive NULL mean:
    {cos_null_mean:.9f}

Descriptive signed NULL-z:
    {cos_z:+.6f}

Observed percentile within joint NULL:
    {cos_percentile:.3f}%

Spectral joint gate:
    {"PASS" if spectrum_gate else "FAIL"}

JOINT DISTANCE-GEOMETRY GATE
----------------------------
Observed maximum pairwise W1:
    {observed_max_w1:.9f}

NULL q05:
    {q05_max_w1:.9f}

Margin in PASS direction (threshold - observed):
    {distance_abs_margin:+.9f}

Relative margin:
    {fmt_pct(distance_rel_margin)}

Empirical p:
    {p_distance:.9f}

Descriptive NULL mean:
    {w1_null_mean:.9f}

Descriptive signed NULL-z:
    {w1_z:+.6f}

Percent of NULL max-W1 >= observed:
    {w1_better_than_null_pct:.3f}%

Distance joint gate:
    {"PASS" if distance_gate else "FAIL"}

EXISTING RESAMPLING CONTEXT
---------------------------
AC spectrum:
    median z = {robust_diag["AC_z_spectrum"]["median"]:.6f}
    q2.5 = {robust_diag["AC_z_spectrum"]["q025"]:.6f}
    q97.5 = {robust_diag["AC_z_spectrum"]["q975"]:.6f}
    positive fraction = {robust_diag["AC_z_spectrum"]["positive_fraction"]:.6f}

AC distance:
    median z = {robust_diag["AC_z_distance"]["median"]:.6f}
    q2.5 = {robust_diag["AC_z_distance"]["q025"]:.6f}
    q97.5 = {robust_diag["AC_z_distance"]["q975"]:.6f}
    positive fraction = {robust_diag["AC_z_distance"]["positive_fraction"]:.6f}

BC spectrum:
    median z = {robust_diag["BC_z_spectrum"]["median"]:.6f}
    q2.5 = {robust_diag["BC_z_spectrum"]["q025"]:.6f}
    q97.5 = {robust_diag["BC_z_spectrum"]["q975"]:.6f}
    positive fraction = {robust_diag["BC_z_spectrum"]["positive_fraction"]:.6f}

BC distance:
    median z = {robust_diag["BC_z_distance"]["median"]:.6f}
    q2.5 = {robust_diag["BC_z_distance"]["q025"]:.6f}
    q97.5 = {robust_diag["BC_z_distance"]["q975"]:.6f}
    positive fraction = {robust_diag["BC_z_distance"]["positive_fraction"]:.6f}

DIAGNOSTIC INTERPRETATION
-------------------------
The v48.3 pairwise and resampling results remain unchanged.

The joint claim failed because the preregistered worst-pair aggregate
criterion was not satisfied.

This diagnostic identifies exactly which pair controlled each worst-case
joint statistic and how far that statistic lay from its preregistered
NULL boundary.

It does NOT convert the v48.3 joint FAIL into a PASS.

The correct scientific status remains:

    PAIRWISE THIRD-COHORT REPLICATION: SUPPORTED
    PATIENT-RESAMPLING ROBUSTNESS: SUPPORTED
    JOINT THREE-COHORT INVARIANCE: NOT SUPPORTED

No rescue inference is made.
"""

    summary_out.write_text(
        summary,
        encoding="utf-8",
    )

    manifest_out = (
        ds
        / "v48_4_manifest.json"
    )

    write_obj = {
        "version": VERSION,
        "status": "DIAGNOSTIC_COMPLETE",
        "changes_v48_3_status": False,
        "new_hypothesis": False,
        "new_thresholds": False,
        "retuning": False,
        "rescue": False,
        "failure_mode": failure_mode,
        "worst_spectrum_pair": worst_spectrum_pair,
        "worst_distance_pair": worst_distance_pair,
        "best_spectrum_pair": best_spectrum_pair,
        "best_distance_pair": best_distance_pair,
        "joint_spectrum": {
            "observed": observed_min_cos,
            "null_q95": q95_min_cos,
            "absolute_margin": spectrum_abs_margin,
            "relative_margin": spectrum_rel_margin,
            "empirical_p": p_spectrum,
            "null_mean": cos_null_mean,
            "signed_null_z": cos_z,
            "observed_percentile": cos_percentile,
            "gate_pass": spectrum_gate,
        },
        "joint_distance": {
            "observed": observed_max_w1,
            "null_q05": q05_max_w1,
            "pass_direction_margin": distance_abs_margin,
            "relative_margin": distance_rel_margin,
            "empirical_p": p_distance,
            "null_mean": w1_null_mean,
            "signed_null_z": w1_z,
            "percent_null_ge_observed": w1_better_than_null_pct,
            "gate_pass": distance_gate,
        },
        "pairwise_observed": pairs,
        "robustness_context": robust_diag,
        "source_sha256": {
            "v48_3_closure_manifest": sha256_file(
                manifest_path
            ),
            "v48_3_closure": sha256_file(
                closure_path
            ),
            "v48_3_summary": sha256_file(
                summary_path
            ),
            "v48_3_joint_null": sha256_file(
                joint_null_path
            ),
            "v48_3_pairwise_null": sha256_file(
                pairwise_null_path
            ),
            "v48_3_robustness_results": sha256_file(
                robust_path
            ),
            "execution_script": sha256_file(
                Path(__file__).resolve()
            ),
        },
    }

    manifest_out.write_text(
        json.dumps(
            write_obj,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(summary)
    print("Wrote:")
    print(" ", pair_out)
    print(" ", summary_out)
    print(" ", manifest_out)


if __name__ == "__main__":
    main()
