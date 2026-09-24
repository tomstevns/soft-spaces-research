#!/usr/bin/env python3
"""
Soft Spaces / CML
v45.1 — GSE14671 cross-endpoint external preregistration bootstrap

Run from:
    ...\applications\cml\code\

Creates in ../docs:
- v45_1_CML_GSE14671_CROSS_ENDPOINT_PREREGISTRATION.txt
- v45_1_CML_GSE14671_CROSS_ENDPOINT_PREREGISTRATION_manifest.json
- v45_1_CML_GSE14671_protocol_lock.py

This stage freezes the external test BEFORE any GSE14671 model score.
"""

from pathlib import Path
import hashlib
import json


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    code_dir = Path(__file__).resolve().parent
    project = code_dir.parent
    docs = project / "docs"
    ds = project / "results" / "direct_subspace"

    docs.mkdir(parents=True, exist_ok=True)

    candidate_manifest = ds / "v45_0b_manifest.json"
    frozen_model = ds / "v44_5a_frozen_development_model.npz"
    closure_manifest = docs / "v44_7_CML_CLOSURE_AUDIT_manifest.json"

    for p in [candidate_manifest, frozen_model, closure_manifest]:
        if not p.exists():
            raise FileNotFoundError(p)

    cand = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    closure = json.loads(closure_manifest.read_text(encoding="utf-8"))

    if cand.get("status") != "CANDIDATE ELIGIBLE FOR v45.1 PREREGISTRATION":
        raise RuntimeError("v45.0b candidate audit is not eligible for v45.1.")

    if int(cand.get("sample_n", -1)) != 59:
        raise RuntimeError("v45.0b sample count mismatch.")

    if int(cand.get("responder_n", -1)) != 41:
        raise RuntimeError("v45.0b responder count mismatch.")

    if int(cand.get("non_responder_n", -1)) != 18:
        raise RuntimeError("v45.0b non-responder count mismatch.")

    if int(cand.get("unknown_n", -1)) != 0:
        raise RuntimeError("v45.0b contains unknown response labels.")

    if closure.get("status") != "CLOSED":
        raise RuntimeError("v44 closure is not recorded as CLOSED.")

    protocol = """Soft Spaces / CML
v45.1 — GSE14671 CROSS-ENDPOINT EXTERNAL PREREGISTRATION

STATUS
------
FROZEN BEFORE GSE14671 MODEL SCORING

PURPOSE
-------
Test whether the already-frozen v44 transport-aware CML direct-subspace
model preserves predictive information in a second independent CML cohort.

This is an EXTERNAL CROSS-ENDPOINT test.

It is NOT an exact replication of the GSE130404 development endpoint.

DEVELOPMENT MODEL
-----------------
Reuse v44.5a without modification:

- development cohort: GSE130404
- transport-aware eligible universe: 15,890 genes
- final Top-256 frozen coordinates
- frozen direct subspace rank: r = 16
- frozen U16
- frozen balanced L2 logistic regression

No refitting on GSE14671.

EXTERNAL COHORT
---------------
GEO:
    GSE14671

Platform:
    GPL570

Sample count:
    59

Expression material:
    pretreatment CML material according to the audited GEO series design

Outcome:
    12-month cytogenetic response to imatinib

Frozen external classes:

NEGATIVE CLASS / BETTER RESPONSE
    complete cytogenetic response (CCyR) after 12 months
    n = 41

POSITIVE CLASS / POOR RESPONSE
    >65% Ph-positive metaphases after 12 months of imatinib
    n = 18

Unknown:
    n = 0

CLASS ORIENTATION
-----------------
Positive class remains POOR RESPONSE, consistent with the orientation used
in the v44 development / external framework.

ENDPOINT DIFFERENCE
-------------------
GSE130404 development endpoint:
    3-month BCR-ABL1 response threshold

GSE14671 external endpoint:
    12-month cytogenetic response

Therefore any positive result is evidence of CROSS-ENDPOINT GENERALIZATION,
not exact endpoint replication.

TECHNICAL TRANSPORT GATE
------------------------
Before outcome scoring:

1. map the frozen final Top-256 coordinates to GSE14671 / GPL570;
2. require all 256 frozen coordinates to be technically measurable;
3. require non-zero across-sample variation for every frozen coordinate
   after the frozen external representation transform;
4. no gene substitution;
5. no rank reduction;
6. no post-hoc coordinate deletion.

If the technical gate fails:
    TECHNICALLY NON-EVALUABLE
and no predictive score may be interpreted.

FROZEN EXTERNAL REPRESENTATION
------------------------------
Use the same cross-platform external representation principle frozen for v44:

1. obtain one expression value per frozen gene per patient;
2. no external outcome information enters preprocessing;
3. per gene, standardize across the complete GSE14671 cohort:
       z = (x - cohort_mean) / cohort_sd
   with ddof = 0;
4. project standardized Top-256 vector into the frozen U16;
5. apply the frozen v44.5a logistic coefficients and intercept.

No external model refit.

No outcome-driven normalization choice.

PRIMARY METRIC
--------------
PR-AUC for POOR RESPONSE.

Reference prevalence:
    18 / 59 = 0.3050847458

SECONDARY METRICS
-----------------
ROC-AUC

Balanced accuracy at frozen threshold 0.5

Sensitivity at frozen threshold 0.5

Specificity at frozen threshold 0.5

FROZEN DIRECTION CHECK
----------------------
Mean frozen score in the >65% Ph-positive poor-response class must exceed
mean frozen score in the CCyR class.

PASS RULE
---------
PASS requires all of:

1. technical transport gate PASS;
2. PR-AUC > positive-class prevalence;
3. ROC-AUC > 0.5;
4. mean frozen score poor-response > mean frozen score CCyR;
5. no model refitting;
6. no outcome-driven tuning or rescue.

No p-value threshold is required for the preregistered single external cohort
decision.

INTERPRETATION
--------------
A PASS would support a second independent cross-cohort, cross-endpoint
generalization of the frozen v44 subspace.

A FAIL would be retained as a genuine external predictive failure.

A technical transport failure would be retained as TECHNICALLY NON-EVALUABLE.

NO RESCUE
---------
After model scoring begins, do NOT:

- change rank;
- change Top-256 coordinates;
- replace genes;
- remove inconvenient genes;
- change normalization;
- retune threshold;
- refit the classifier;
- redefine response classes;
- exclude patients based on model score;
- search alternative endpoints.

CLAIM LIMIT
-----------
Even a PASS does not establish:

- clinical utility;
- treatment recommendation;
- causal biology;
- superiority to established CML biomarkers;
- superiority to the best classical predictor;
- exact endpoint replication;
- quantum biology;
- quantum computational advantage.

RELATION TO PRIOR RESULTS
-------------------------
v42:
    feature-selection formulation
    INTERNAL FAIL

v43:
    direct-subspace formulation
    INTERNAL PASS
    external technical transport failure

v44:
    transport-aware direct-subspace formulation
    INTERNAL PASS
    strong internal stability
    GSE44589 primary external cross-endpoint PASS, modest
    GSE236233 secondary RNA-seq stress TECHNICALLY NON-EVALUABLE

v45.1:
    second independent GPL570 cross-endpoint test
    GSE14671
    frozen before model scoring
"""

    protocol_path = docs / "v45_1_CML_GSE14671_CROSS_ENDPOINT_PREREGISTRATION.txt"
    protocol_path.write_text(protocol, encoding="utf-8")
    protocol_sha = sha256_file(protocol_path)

    manifest = {
        "version": "v45.1",
        "status": "FROZEN_BEFORE_GSE14671_MODEL_SCORING",
        "external_cohort": "GSE14671",
        "platform": "GPL570",
        "sample_n": 59,
        "positive_class": {
            "definition": ">65% Ph-positive metaphases after 12 months of imatinib",
            "n": 18,
        },
        "negative_class": {
            "definition": "complete cytogenetic response (CCyR) after 12 months",
            "n": 41,
        },
        "positive_prevalence": 18 / 59,
        "endpoint_type": "cross-endpoint external generalization",
        "development_endpoint": "3-month BCR-ABL1 response threshold",
        "external_endpoint": "12-month cytogenetic response",
        "frozen_top_k": 256,
        "frozen_rank": 16,
        "external_representation": "per-gene cohort z-standardization, ddof=0",
        "external_refit": False,
        "outcome_tuning": False,
        "primary_metric": "PR-AUC",
        "secondary_metrics": [
            "ROC-AUC",
            "balanced_accuracy@0.5",
            "sensitivity@0.5",
            "specificity@0.5",
        ],
        "pass_rule": {
            "technical_transport": "PASS",
            "pr_auc": "> prevalence",
            "roc_auc": "> 0.5",
            "direction": "mean poor-response score > mean CCyR score",
            "refit": "NO",
            "outcome_tuning": "NO",
        },
        "protocol_sha256": protocol_sha,
        "source_sha256": {
            "v45_0b_manifest": sha256_file(candidate_manifest),
            "v44_5a_frozen_model": sha256_file(frozen_model),
            "v44_7_closure_manifest": sha256_file(closure_manifest),
        },
    }

    manifest_path = docs / "v45_1_CML_GSE14671_CROSS_ENDPOINT_PREREGISTRATION_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lock_code = (
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import hashlib\n\n"
        f'EXPECTED_SHA256 = "{protocol_sha}"\n'
        'HERE = Path(__file__).resolve().parent\n'
        'PATH = HERE / "v45_1_CML_GSE14671_CROSS_ENDPOINT_PREREGISTRATION.txt"\n'
        'actual = hashlib.sha256(PATH.read_bytes()).hexdigest()\n'
        'print("v45.1 GSE14671 preregistration integrity check")\n'
        'print("Expected:", EXPECTED_SHA256)\n'
        'print("Actual:  ", actual)\n'
        'if actual != EXPECTED_SHA256:\n'
        '    raise SystemExit("FAIL: v45.1 preregistration has changed.")\n'
        'print("PASS: v45.1 preregistration unchanged.")\n'
    )

    lock_path = docs / "v45_1_CML_GSE14671_protocol_lock.py"
    lock_path.write_text(lock_code, encoding="utf-8")

    print("=== v45.1 GSE14671 PREREGISTRATION COMPLETE ===")
    print()
    print("Created:")
    print(" ", protocol_path)
    print(" ", manifest_path)
    print(" ", lock_path)
    print()
    print("Protocol SHA256:", protocol_sha)
    print()
    print("FROZEN:")
    print("  GSE14671")
    print("  59 samples")
    print("  18 poor response (>65% Ph+) = positive")
    print("  41 CCyR = negative")
    print("  Top-256 unchanged")
    print("  r=16 unchanged")
    print("  no external refit")
    print("  no outcome tuning")
    print()
    print("NEXT: technical transport audit before any outcome scoring.")


if __name__ == "__main__":
    main()
