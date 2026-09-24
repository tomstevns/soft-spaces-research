#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json

def main():
    code_dir = Path(__file__).resolve().parent
    project = code_dir.parent
    docs = project / "docs"
    results = project / "results" / "direct_subspace"
    docs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    closure = """Soft Spaces / CML
v43.6 — CLOSURE AUDIT

STATUS
------
CLOSED

INTERNAL RESULTS
----------------
v43.2 direct-subspace REAL vs matched random-subspace NULL:
REAL mean PR-AUC = 0.455122
NULL mean PR-AUC = 0.387771
Delta = +0.067352
Empirical one-sided p = 0.000999001
STATUS = PASS

REAL mean ROC-AUC = 0.725895
NULL mean ROC-AUC = 0.684225
Empirical one-sided p = 0.002997

v43.3 internal subspace stability:
REAL mean normalized projector overlap = 0.687344
NULL mean = 0.056782
Delta = +0.630562
Empirical one-sided p = 0.000999001

EXTERNAL TRANSPORT
------------------
GSE44589 / GPL570:
after stable Entrez harmonization = 214 / 256 coordinates
retained projector mass fraction = 0.798870
STATUS = TECHNICALLY NON-EVALUABLE

GSE236233 / RNA-seq pseudobulk:
9 CML patients, Lin-CD34+ primary population
13,616 cells aggregated patient-wise
193 / 256 coordinates present in all 9 patients
retained projector mass fraction = 0.715551
STATUS = TECHNICALLY NON-EVALUABLE

External response performance was NOT evaluated.

CONCLUSION
----------
v43 supports an internal direct-subspace signal and strong internal
resampling stability, but the frozen Top-256 coordinate system was not
sufficiently cross-platform transportable for a faithful external test.

The external stage is therefore TECHNICALLY NON-EVALUABLE, not a negative
external predictive result.

v43 does not establish external generalization, clinical utility, causal
biology, superiority to established biomarkers, or quantum advantage.

NEXT
----
v44 starts from an outcome-blind cross-platform transport-aware gene universe
defined BEFORE supervised subspace construction.
"""

    closure_path = docs / "v43_6_CML_CLOSURE_AUDIT.txt"
    closure_path.write_text(closure, encoding="utf-8")
    closure_sha = hashlib.sha256(closure_path.read_bytes()).hexdigest()

    protocol = """Soft Spaces / CML
v44.0 — CROSS-PLATFORM DIRECT-SUBSPACE PREREGISTRATION

STATUS
------
FROZEN BEFORE v44 SUPERVISED MODELING

NEW HYPOTHESIS
--------------
v43 internally supported direct subspace projection but failed technical
external transport.

v44 defines a transport-aware gene universe BEFORE any supervised modeling.

DEVELOPMENT
-----------
GSE130404
96 baseline chronic-phase CML patients
y=1: BCR-ABL1 >10% IS at 3 months, n=13
y=0: BCR-ABL1 <10% IS at 3 months, n=83

OUTCOME-BLIND TECHNICAL REFERENCES
----------------------------------
GSE44589 / GPL570
GSE236233 / RNA-seq feature universe

External response outcomes MUST NOT be used to define the gene universe.

TRANSPORT-AWARE ELIGIBILITY
---------------------------
A development gene is eligible only if:
1. it has a usable development gene identity;
2. it is resolvable on GPL570 by exact symbol or stable Entrez ID;
3. it is represented in the GSE236233 RNA feature universe for all
   9 primary Lin-CD34+ patient samples.

No outcome-guided substitution is allowed.

If fewer than 256 genes satisfy the rule:
    STOP as technically insufficient.

If at least 256 genes satisfy it:
    use raw training-fold variance within this eligible universe to select
    Top-256 independently inside each outer training fold.

INTERNAL CV
-----------
Reuse exactly the frozen v42.2 100 held-out folds:
5-fold stratified CV x 20 repeats.

Within each training fold:
1. restrict to the transport-aware universe;
2. rank by raw training-fold variance;
3. select Top-256;
4. StandardScaler fit on training fold only;
5. H = mean(xx^T | y=1) - mean(xx^T | y=0);
6. symmetrize H;
7. diagonalize;
8. direct projection into candidate subspace.

CANDIDATE RANKS
---------------
r in {4, 8, 16}

Selection:
highest mean held-out PR-AUC;
tie -> smaller r.

CLASSIFIER
----------
Balanced L2 LogisticRegression
C=1.0
solver=liblinear

PRIMARY METRIC
--------------
PR-AUC

MATCHED NULL
------------
After rank selection is frozen:
compare REAL against random orthonormal subspaces of identical dimension,
using the same folds, transport-aware Top-256 pools, preprocessing,
classifier, and held-out samples.

Internal PASS iff:
REAL mean PR-AUC > NULL mean PR-AUC
AND empirical one-sided p <= 0.05.

STABILITY
---------
Only after internal PASS:
test projector-overlap stability against matched random subspaces.

EXTERNAL STAGE
--------------
Only after internal PASS and stability:
freeze a separate external protocol before response scoring.

The patient remains the statistical unit.
Single-cell data may be used only through preregistered patient-level
pseudobulk aggregation.

NO RESCUE
---------
If internal REAL-vs-NULL fails:
record FAIL and stop.

If later external transport fails:
record TECHNICALLY NON-EVALUABLE.

If later external predictive performance fails:
record FAIL.

No external outcome retuning.

CLAIM LIMIT
-----------
A positive v44 result would support a cross-platform transportable
CML-response subspace only.

It would not establish clinical utility, treatment recommendation,
causal genes, superiority to established biomarkers, or quantum advantage.
"""

    protocol_path = docs / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION.txt"
    protocol_path.write_text(protocol, encoding="utf-8")
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()

    manifest_path = docs / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION_manifest.json"
    manifest_path.write_text(json.dumps({
        "version": "v44.0",
        "status": "FROZEN_BEFORE_V44_SUPERVISED_MODELING",
        "development_cohort": "GSE130404",
        "candidate_ranks": [4, 8, 16],
        "primary_metric": "PR-AUC",
        "minimum_transport_universe_size": 256,
        "external_outcomes_allowed_for_universe": False,
        "technical_references": ["GSE44589/GPL570", "GSE236233/RNA-seq"],
        "protocol_sha256": protocol_sha,
        "v43_closure_sha256": closure_sha
    }, indent=2), encoding="utf-8")

    lock_code = """#!/usr/bin/env python3
from pathlib import Path
import hashlib

EXPECTED_SHA256 = "__SHA__"
HERE = Path(__file__).resolve().parent
path = HERE / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION.txt"
if not path.exists():
    path = HERE.parent / "docs" / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION.txt"
if not path.exists():
    raise FileNotFoundError("v44.0 preregistration not found")

actual = hashlib.sha256(path.read_bytes()).hexdigest()
print("v44.0 preregistration integrity check")
print("Expected:", EXPECTED_SHA256)
print("Actual:  ", actual)
if actual != EXPECTED_SHA256:
    raise SystemExit("FAIL: v44.0 preregistration has changed.")
print("PASS: v44.0 preregistration is unchanged.")
""".replace("__SHA__", protocol_sha)

    lock_path = docs / "v44_0_CML_protocol_lock.py"
    lock_path.write_text(lock_code, encoding="utf-8")

    v441 = """#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import pandas as pd

EXPECTED_PROTOCOL_SHA256 = "__SHA__"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    code_dir = Path(__file__).resolve().parent
    project = code_dir.parent
    docs = project / "docs"
    results = project / "results" / "direct_subspace"

    protocol = docs / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION.txt"
    actual = sha256_file(protocol)

    print("=== v44.1 TRANSPORT-AWARE UNIVERSE PRECHECK ===")
    print("Expected protocol SHA:", EXPECTED_PROTOCOL_SHA256)
    print("Actual protocol SHA:  ", actual)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise SystemExit("FAIL: v44.0 protocol SHA mismatch.")
    print("PASS: v44.0 protocol verified.")
    print("External response outcomes used: NO")
    print()

    gpl_path = results / "v43_4b_identifier_harmonization.csv"
    rna_path = results / "v43_4d_GSE236233_frozen_gene_coverage.csv"
    for p in [gpl_path, rna_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    gpl = pd.read_csv(gpl_path)
    rna = pd.read_csv(rna_path)

    gpl_ok = set(gpl.loc[
        gpl["match_status"].isin(["EXACT_SYMBOL", "ENTREZ_ID_MATCH"]),
        "development_gene"
    ].astype(str))

    rna_ok = set(rna.loc[
        rna["present_all_9"] == 1,
        "gene"
    ].astype(str))

    prior_intersection = sorted(gpl_ok & rna_ok)

    summary_path = results / "v44_1_transport_universe_precheck_summary.txt"
    list_path = results / "v44_1_prior256_crossplatform_intersection.txt"
    manifest_path = results / "v44_1_manifest.json"

    summary = [
        "=== Soft Spaces / CML v44.1 TRANSPORT-AWARE UNIVERSE PRECHECK ===",
        "",
        "External response outcomes used: NO",
        "",
        "PRIOR v43 TOP-256 DIAGNOSTIC",
        "---------------------------",
        "GPL570 transportable: " + str(len(gpl_ok)),
        "GSE236233 all-9 present: " + str(len(rna_ok)),
        "Cross-platform intersection: " + str(len(prior_intersection)),
        "",
        "STATUS",
        "------",
        "FULL-UNIVERSE REBUILD REQUIRED",
        "",
        "The old audit tables contain only the prior v43 Top-256.",
        "They are not sufficient to define the full v44 eligible universe.",
        "",
        "NEXT",
        "----",
        "Use v44.1b to rebuild the full development universe outcome-blind",
        "from GPL10558, GPL570 and the downloaded GSE236233 feature tables."
    ]

    summary_path.write_text("\\n".join(summary) + "\\n", encoding="utf-8")
    list_path.write_text("\\n".join(prior_intersection) + "\\n", encoding="utf-8")

    manifest_path.write_text(json.dumps({
        "version": "v44.1",
        "status": "FULL-UNIVERSE REBUILD REQUIRED",
        "prior256_gpl570_transportable": len(gpl_ok),
        "prior256_gse236233_all9_present": len(rna_ok),
        "prior256_crossplatform_intersection": len(prior_intersection),
        "external_outcomes_used": False,
        "sha256": {
            "v44_0_protocol": actual,
            "v43_4b": sha256_file(gpl_path),
            "v43_4d": sha256_file(rna_path)
        }
    }, indent=2), encoding="utf-8")

    print(summary_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
""".replace("__SHA__", protocol_sha)

    v441_path = code_dir / "v44_1_CML_transport_universe_precheck.py"
    v441_path.write_text(v441, encoding="utf-8")

    print("=== v44 BOOTSTRAP COMPLETE ===")
    print("Created:")
    for p in [closure_path, protocol_path, manifest_path, lock_path, v441_path]:
        print(" ", p)
    print()
    print("v43.6 closure SHA256:", closure_sha)
    print("v44.0 protocol SHA256:", protocol_sha)
    print()
    print("NEXT COMMAND:")
    print(r"python -X utf8 -u .\v44_1_CML_transport_universe_precheck.py")

if __name__ == "__main__":
    main()
