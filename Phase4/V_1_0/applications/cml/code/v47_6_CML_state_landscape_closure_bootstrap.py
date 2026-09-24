#!/usr/bin/env python3
"""
Soft Spaces / CML
v47.6 — State-Landscape closure bootstrap

Purpose
-------
Create the formal closure record for the v47 CML State-Landscape track.

NO new analysis.
NO rescoring.
NO tuning.
NO reinterpretation of failed external replication.

The closure preserves:
- v47.3 internal primary PASS
- v47.4 internal resampling STABILITY PASS
- v47.5 parser-abort before scoring
- v47.5b corrected parser
- v47.5b external cross-endpoint replication FAIL

Final closure:
    INTERNAL GEOMETRIC SIGNAL SUPPORTED
    INTERNAL RESAMPLING STABILITY SUPPORTED
    EXTERNAL CROSS-ENDPOINT REPLICATION NOT CONFIRMED
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

VERSION = "v47.6"


def project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    project = project_dir()
    docs = project / "docs"
    ds = project / "results" / "direct_subspace"

    docs.mkdir(parents=True, exist_ok=True)

    v470_protocol = docs / "v47_0_CML_STATE_LANDSCAPE_PREREGISTRATION.txt"
    v472_manifest = ds / "v47_2_manifest.json"
    v473_manifest = ds / "v47_3_manifest.json"
    v474_manifest = ds / "v47_4_manifest.json"
    v475_protocol = docs / "v47_5_CML_EXTERNAL_STATE_LANDSCAPE_PREREGISTRATION.txt"
    v475_protocol_manifest = docs / "v47_5_CML_EXTERNAL_STATE_LANDSCAPE_PREREGISTRATION_manifest.json"
    v475b_manifest = ds / "v47_5b_manifest.json"

    required = [
        v470_protocol,
        v472_manifest,
        v473_manifest,
        v474_manifest,
        v475_protocol,
        v475_protocol_manifest,
        v475b_manifest,
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    m472 = read_json(v472_manifest)
    m473 = read_json(v473_manifest)
    m474 = read_json(v474_manifest)
    p475m = read_json(v475_protocol_manifest)
    m475b = read_json(v475b_manifest)

    actual_v475_protocol_sha = sha256_file(v475_protocol)
    expected_v475_protocol_sha = p475m["protocol_sha256"]

    if actual_v475_protocol_sha != expected_v475_protocol_sha:
        raise RuntimeError("v47.5 preregistration SHA mismatch.")

    if m472.get("status") != "LABEL_BLIND_GEOMETRY_FROZEN":
        raise RuntimeError("v47.2 status mismatch.")

    if m473.get("primary_pass") is not True:
        raise RuntimeError("v47.3 is not recorded as primary PASS.")

    if m474.get("status") != "STABILITY PASS":
        raise RuntimeError("v47.4 is not recorded as STABILITY PASS.")

    if m475b.get("parser_correction_only") is not True:
        raise RuntimeError("v47.5b is not recorded as parser correction only.")

    if m475b.get("prior_v47_5_scoring_performed") is not False:
        raise RuntimeError("v47.5b does not preserve aborted v47.5 as unscored.")

    if m475b.get("primary_pass") is not False:
        raise RuntimeError("v47.5b is not recorded as external FAIL.")

    if m475b.get("status") != "EXTERNAL LANDSCAPE REPLICATION FAIL":
        raise RuntimeError("Unexpected v47.5b status.")

    v473_p = float(m473["empirical_one_sided_p"])
    v473_obs = float(m473["observed_centroid_distance"])
    v473_null_mean = float(m473["null_mean"])

    v474_median_z = float(m474["median_z_null"])
    v474_positive_fraction = float(m474["positive_z_fraction"])
    v474_p05_fraction = float(m474["fraction_empirical_p_le_0_05"])
    v474_overlap = float(m474["median_top256_overlap"])
    v474_jaccard = float(m474["median_top256_jaccard"])

    v475b_obs = float(m475b["observed_centroid_distance"])
    v475b_null_mean = float(m475b["null_mean"])
    v475b_null_z = float(m475b["null_z"])
    v475b_p = float(m475b["empirical_one_sided_p"])
    v475b_overlap = int(m475b["top256_overlap_with_gse130404"])
    v475b_jaccard = float(m475b["top256_jaccard_with_gse130404"])
    v475b_pca = float(m475b["pca16_cumulative_variance"])

    closure = f"""Soft Spaces / CML
v47.6 — STATE-LANDSCAPE TRACK CLOSURE

STATUS
------
CLOSED

FINAL CONCLUSION
----------------
INTERNAL GEOMETRIC SIGNAL SUPPORTED
INTERNAL RESAMPLING STABILITY SUPPORTED
EXTERNAL CROSS-ENDPOINT REPLICATION NOT CONFIRMED

The v47 CML State-Landscape track is closed without rescue tuning.

SCIENTIFIC QUESTION
-------------------
v47 tested whether clinically meaningful CML states occupy non-random
regions of a molecular expression geometry that is constructed WITHOUT
using the clinical outcome labels.

This differs from the earlier predictor track.

The central object in v47 is:
    label-blind state geometry

not:
    a transported or fitted outcome classifier

v47.0 — PREREGISTRATION
-----------------------
The state-landscape hypothesis was frozen before construction.

Frozen principle:
1. define the molecular coordinate system without outcome labels;
2. construct low-dimensional state geometry without outcome labels;
3. freeze that geometry;
4. only then attach clinical labels;
5. test whether the labels occupy the geometry non-randomly.

v47.2 — LABEL-BLIND DEVELOPMENT GEOMETRY
----------------------------------------
Cohort:
    GSE130404

Samples:
    96

Transport-aware eligible universe:
    15,890 genes

Feature selection:
    Top-256 raw variance
    outcome labels not used

State space:
    PCA-16
    outcome labels not used

PCA-16 cumulative explained variance:
    0.871397

Geometry status:
    FROZEN

Geometry SHA256:
    {m472["geometry_sha256"]}

Interpretation:
A high-variance, low-dimensional label-blind molecular state representation
could be constructed before clinical labels were introduced.

No biological basin, attractor, dormancy mechanism or causal transition
was established by this construction alone.

v47.3 — INTERNAL CLINICAL-LABEL OVERLAY
---------------------------------------
Clinical split:
    3-month BCR-ABL1 <10%: 83
    3-month BCR-ABL1 >10%: 13

Primary metric:
    Euclidean distance between group centroids in frozen PCA-16 geometry

Observed centroid distance:
    {v473_obs:.9f}

Permutation NULL mean:
    {v473_null_mean:.9f}

Permutations:
    10,000

Empirical one-sided p:
    {v473_p:.9f}

STATUS:
    PRIMARY LANDSCAPE SEPARATION PASS

Interpretation:
The independently defined clinical groups occupied more separated regions
of the already-frozen label-blind development geometry than expected under
random label assignment.

This was an internal geometric association, not prediction.

v47.4 — INTERNAL RESAMPLING STABILITY
-------------------------------------
Resamples:
    300

Patients per resample:
    76
    66 GOOD
    10 POOR

Within every resample:
- Top-256 reselected label-blind
- standardization refit label-blind
- PCA-16 refit label-blind
- 500 label permutations
- no classifier fitted

Median NULL-z:
    {v474_median_z:.6f}

Frozen threshold:
    > 1.644854

Fraction with NULL-z > 0:
    {v474_positive_fraction:.6f}

Frozen threshold:
    >= 0.750000

Fraction with empirical p <= 0.05:
    {v474_p05_fraction:.6f}

Median Top-256 overlap with original:
    {v474_overlap:.1f} / 256

Median Top-256 Jaccard:
    {v474_jaccard:.6f}

STATUS:
    STABILITY PASS

Interpretation:
The internal label-blind state-landscape separation was robust to repeated
patient subsampling and complete label-blind reconstruction of the
development landscape.

This remains internal resampling evidence, not independent replication.

v47.5 — FIRST EXTERNAL EXECUTION ATTEMPT
----------------------------------------
External cohort:
    GSE44589 / GPL570

The first v47.5 execution reproduced a known historical parser failure mode:
all 198 samples were incorrectly interpreted as baseline.

Observed erroneous parser counts:
    baseline: 198
    MMR: 98
    no-MMR: 0
    unavailable: 11
    unresolved: 89

The script stopped at the frozen sample-count gate.

IMPORTANT:
No external landscape was constructed.
No clinical geometry score was calculated.
No external primary outcome was observed.

This run is preserved as:
    PARSER ABORT BEFORE SCORING

It is neither a scientific PASS nor a scientific FAIL.

v47.5b — CORRECTED EXTERNAL EXECUTION
-------------------------------------
The scientific v47.5 protocol was NOT changed.

Only the sample-specific parser was corrected.

v47.5 protocol SHA256:
    {actual_v475_protocol_sha}

Corrected sample counts:
    total: 198
    baseline pretreatment: 135
    post 6 weeks: 63

Baseline:
    MMR: 69
    none / no-MMR: 59
    unavailable: 7

Evaluable clinical overlay:
    128

External transport-aware eligible universe:
    15,890 genes

Measurable eligible genes:
    15,884

External label-blind Top-K:
    256

External state space:
    PCA-16

PCA-16 cumulative explained variance:
    {v475b_pca:.6f}

Outcome labels used in external geometry:
    NO

External geometry SHA256:
    {m475b["external_geometry_sha256"]}

PRIMARY EXTERNAL TEST
---------------------
Observed centroid distance:
    {v475b_obs:.9f}

Permutation NULL mean:
    {v475b_null_mean:.9f}

Observed NULL-z:
    {v475b_null_z:+.6f}

Permutations:
    10,000

Empirical one-sided p:
    {v475b_p:.9f}

Frozen PASS rule:
    p <= 0.05

STATUS:
    EXTERNAL LANDSCAPE REPLICATION FAIL

The observed centroid distance remained above the NULL mean, but the
preregistered external test was not statistically significant.

Therefore the external state-landscape replication claim is NOT supported.

CROSS-COHORT FEATURE COMPOSITION
--------------------------------
GSE130404 Top-256 vs GSE44589 Top-256 overlap:
    {v475b_overlap} / 256

Jaccard:
    {v475b_jaccard:.6f}

This is descriptive only.

The relatively limited exact Top-256 overlap may motivate a future,
separately preregistered hypothesis about higher-level geometric invariance,
but it does NOT rescue the failed v47.5b external test.

WHAT v47 SUPPORTS
-----------------
1. GSE130404 contains a label-blind low-dimensional state geometry in which
   the frozen 3-month BCR-ABL1 groups are more separated than expected by
   random label assignment.

2. That internal group-separation signal is robust under repeated
   subsampling and complete label-blind landscape reconstruction.

3. The signal is therefore not explained solely by one fixed PCA fit or one
   fixed Top-256 feature list inside GSE130404.

WHAT v47 DOES NOT SUPPORT
-------------------------
1. Independent external confirmation of the clinical-group separation.

2. Exact endpoint replication.

3. A universal CML response landscape.

4. Clinical prediction or prognostic utility.

5. A causal biological state-transition mechanism.

6. Cancer dormancy or reactivation as an identified mechanism.

7. Quantum biology.

8. Quantum computational advantage.

FINAL INTERPRETATION
--------------------
The v47 evidence is internally coherent but externally incomplete.

The strongest defensible statement is:

    A reproducible label-blind molecular state geometry is associated with
    3-month response groups inside GSE130404, and this association is stable
    under internal resampling. However, the preregistered independent
    cross-endpoint replication in GSE44589 failed, so external
    generalizability of the clinical state separation is not confirmed.

Accordingly:

    INTERNAL GEOMETRIC SIGNAL SUPPORTED
    INTERNAL RESAMPLING STABILITY SUPPORTED
    EXTERNAL CROSS-ENDPOINT REPLICATION NOT CONFIRMED

NO-RESCUE DECISION
------------------
v47 is closed without:
- alternative PCA rank
- alternative kNN neighborhood size
- outcome-driven feature selection
- alternative external subgroup selection
- repeated external dataset shopping
- post-hoc threshold adjustment

Any future work must begin as a NEW hypothesis and NEW version.

POTENTIAL NEXT HYPOTHESIS
-------------------------
A scientifically distinct future track may ask whether the invariant object
is higher-level geometry rather than exact gene identity or binary outcome
separation.

Examples of quantities that could be prospectively tested include:
- subspace-angle structure
- local spectral structure
- neighborhood topology
- covariance geometry
- transition-direction structure
- geometry conserved despite changing feature identities

Such work must not be described as a rescue of v47.5b.

TRACK STATUS
------------
Soft Spaces / CML v47 State-Landscape:
    CLOSED
"""

    closure_path = docs / "v47_6_CML_STATE_LANDSCAPE_CLOSURE.txt"
    closure_path.write_text(closure, encoding="utf-8")
    closure_sha = sha256_file(closure_path)

    manifest = {
        "version": VERSION,
        "track": "CML State-Landscape",
        "status": "CLOSED",
        "final_conclusion": {
            "internal_geometric_signal": "SUPPORTED",
            "internal_resampling_stability": "SUPPORTED",
            "external_cross_endpoint_replication": "NOT_CONFIRMED",
        },
        "v47_3": {
            "status": m473["status"],
            "observed_centroid_distance": v473_obs,
            "null_mean": v473_null_mean,
            "empirical_one_sided_p": v473_p,
            "primary_pass": True,
        },
        "v47_4": {
            "status": m474["status"],
            "median_z_null": v474_median_z,
            "positive_z_fraction": v474_positive_fraction,
            "fraction_empirical_p_le_0_05": v474_p05_fraction,
            "median_top256_overlap": v474_overlap,
            "median_top256_jaccard": v474_jaccard,
        },
        "v47_5": {
            "status": "PARSER_ABORT_BEFORE_SCORING",
            "scientific_result": None,
            "geometry_constructed": False,
            "scoring_performed": False,
        },
        "v47_5b": {
            "status": m475b["status"],
            "parser_correction_only": True,
            "scientific_protocol_changed": False,
            "scientific_protocol_sha256": actual_v475_protocol_sha,
            "baseline_n": m475b["baseline_n"],
            "post_6w_n": m475b["post_6w_n"],
            "evaluable_n": m475b["evaluable_n"],
            "observed_centroid_distance": v475b_obs,
            "null_mean": v475b_null_mean,
            "null_z": v475b_null_z,
            "empirical_one_sided_p": v475b_p,
            "primary_pass": False,
            "top256_overlap_with_gse130404": v475b_overlap,
            "top256_jaccard_with_gse130404": v475b_jaccard,
        },
        "no_rescue": True,
        "future_work_requires_new_hypothesis": True,
        "closure_sha256": closure_sha,
        "sha256": {
            "v47_0_protocol": sha256_file(v470_protocol),
            "v47_2_manifest": sha256_file(v472_manifest),
            "v47_3_manifest": sha256_file(v473_manifest),
            "v47_4_manifest": sha256_file(v474_manifest),
            "v47_5_protocol": actual_v475_protocol_sha,
            "v47_5_protocol_manifest": sha256_file(v475_protocol_manifest),
            "v47_5b_manifest": sha256_file(v475b_manifest),
            "closure": closure_sha,
            "execution_script": sha256_file(Path(__file__).resolve()),
        },
    }

    manifest_path = docs / "v47_6_CML_STATE_LANDSCAPE_CLOSURE_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_path = ds / "v47_6_state_landscape_closure_summary.txt"

    summary = f"""=== Soft Spaces / CML v47.6 STATE-LANDSCAPE CLOSURE ===

INTERNAL GEOMETRIC SIGNAL:
    SUPPORTED

INTERNAL RESAMPLING STABILITY:
    SUPPORTED

EXTERNAL CROSS-ENDPOINT REPLICATION:
    NOT CONFIRMED

v47.3:
    p = {v473_p:.9f}
    PASS

v47.4:
    median NULL-z = {v474_median_z:.6f}
    positive-z fraction = {v474_positive_fraction:.6f}
    STABILITY PASS

v47.5:
    PARSER ABORT BEFORE SCORING
    no geometry
    no scientific external result

v47.5b:
    observed centroid distance = {v475b_obs:.9f}
    NULL mean = {v475b_null_mean:.9f}
    NULL-z = {v475b_null_z:+.6f}
    p = {v475b_p:.9f}
    EXTERNAL LANDSCAPE REPLICATION FAIL

NO RESCUE / NO TUNING

FINAL:
    INTERNAL GEOMETRIC SIGNAL SUPPORTED
    INTERNAL RESAMPLING STABILITY SUPPORTED
    EXTERNAL CROSS-ENDPOINT REPLICATION NOT CONFIRMED

TRACK:
    CLOSED

Closure SHA256:
    {closure_sha}
"""

    summary_path.write_text(summary, encoding="utf-8")

    print(summary)
    print("Wrote:")
    print(" ", closure_path)
    print(" ", manifest_path)
    print(" ", summary_path)


if __name__ == "__main__":
    main()
