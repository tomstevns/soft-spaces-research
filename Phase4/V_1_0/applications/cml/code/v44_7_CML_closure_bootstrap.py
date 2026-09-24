#!/usr/bin/env python3
from pathlib import Path
import hashlib, json

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    code_dir=Path(__file__).resolve().parent
    project=code_dir.parent
    docs=project/"docs"
    ds=project/"results"/"direct_subspace"
    docs.mkdir(parents=True,exist_ok=True)

    required={
        "v44_1b": ds/"v44_1b_manifest.json",
        "v44_2": ds/"v44_2_manifest.json",
        "v44_3": ds/"v44_3_manifest.json",
        "v44_4": ds/"v44_4_manifest.json",
        "v44_5a": ds/"v44_5a_manifest.json",
        "v44_5b1": ds/"v44_5b1_manifest.json",
    }

    for k,p in required.items():
        if not p.exists():
            raise FileNotFoundError(f"{k}: {p}")

    m={k:json.loads(p.read_text(encoding="utf-8")) for k,p in required.items()}

    if m["v44_1b"].get("status")!="PASS TO v44.2":
        raise RuntimeError("v44.1b gate failed")
    if int(m["v44_2"].get("selected_r",-1))!=16:
        raise RuntimeError("v44.2 did not freeze r=16")
    if m["v44_3"].get("status")!="PASS":
        raise RuntimeError("v44.3 failed")
    if m["v44_4"].get("status")!="PASS":
        raise RuntimeError("v44.4 failed")
    if m["v44_5a"].get("status")!="TECHNICALLY EVALUABLE":
        raise RuntimeError("v44.5a not technically evaluable")
    if m["v44_5b1"].get("status")!="PASS":
        raise RuntimeError("v44.5b1 failed")

    closure = """Soft Spaces / CML
v44.7 — CLOSURE AUDIT

STATUS
------
CLOSED

OVERALL EVIDENCE CHAIN
----------------------
v44.1b transport-aware universe:
    15,890 eligible cross-platform genes
    PASS TO v44.2

v44.2 internal development rank selection:
    r=4  PR-AUC 0.425112
    r=8  PR-AUC 0.415784
    r=16 PR-AUC 0.477031
    frozen rank = 16

v44.3 internal REAL vs matched random-subspace NULL:
    REAL PR-AUC  = 0.477031
    NULL PR-AUC  = 0.392061
    delta        = +0.084970
    empirical p  = 0.000999001

    REAL ROC-AUC = 0.785870
    NULL ROC-AUC = 0.691650
    delta        = +0.094220
    empirical p  = 0.000999001

    STATUS: PASS

v44.4 internal subspace stability:
    REAL mean normalized projector overlap = 0.662641
    NULL mean                              = 0.056197
    delta                                  = +0.606444
    empirical p                            = 0.000999001

    STATUS: PASS

v44.5a primary external technical transport:
    final Top-256 transportable = 256 / 256
    retained projector mass     = 16 / 16
    retained fraction           = 1.000000

    STATUS: TECHNICALLY EVALUABLE

v44.5b1 primary external GSE44589 generalization:
    baseline samples            = 135
    evaluable                   = 128
    poor response / none        = 59
    MMR                         = 69
    unavailable                 = 7

    prevalence                  = 0.460938
    PR-AUC                      = 0.529572
    PR-AUC - prevalence         = +0.068634
    ROC-AUC                     = 0.577254
    balanced accuracy @0.5      = 0.526775
    sensitivity @0.5            = 0.169492
    specificity @0.5            = 0.884058

    frozen score direction preserved = YES

    STATUS: PASS

v44.6a secondary GSE236233 RNA-seq stress test:
    statistical unit = patient
    n = 9
    model refit = NO
    tuning = NO

    preregistered technical gate stopped execution BEFORE scoring because
    10 frozen coordinates had zero cross-patient SD after the frozen
    CPM -> log1p transform:

        MMP8
        ORM1
        PI3
        PGLYRP1
        OLFM4
        ORM2
        FCRLA
        IFNG
        CFC1B
        AQP9

    STATUS: TECHNICALLY NON-EVALUABLE

    No rescue was performed.
    No 246-gene model was evaluated.
    No epsilon or SD=1 substitution was used.
    No rank or normalization rule was changed.

FINAL SCIENTIFIC INTERPRETATION
-------------------------------
v44 supports a bounded conclusion:

A transport-aware learned r=16 multivariate subspace derived from GSE130404
contains reproducible internal CML-response information, exceeds matched
random subspaces internally, is strongly stable under the frozen resampling
design, transports completely to GSE44589, and preserves modest but
directionally consistent predictive information in an independent
cross-endpoint external cohort.

The primary external evidence is therefore positive but modest.

The GSE236233 cross-modality branch remains technically non-evaluable and
must not be represented as either positive or negative predictive replication.

WHAT v44 DOES NOT ESTABLISH
---------------------------
v44 does NOT establish:

- clinical utility
- clinical decision support
- treatment recommendation
- causal genes
- causal disease mechanism
- superiority to established CML biomarkers
- superiority to the best classical internal model
- exact replication of the GSE130404 endpoint
- successful RNA-seq cross-modality validation
- quantum biology
- quantum computational advantage

RELATION TO EARLIER CML TRACKS
------------------------------
v42:
    coordinate / feature-selection formulation
    INTERNAL VALIDATION FAIL

v43:
    direct-subspace formulation
    INTERNAL PASS
    strong internal stability
    external transport TECHNICALLY NON-EVALUABLE

v44:
    transport-aware direct-subspace formulation
    INTERNAL PASS
    strong internal stability
    complete primary technical transport
    PRIMARY EXTERNAL GENERALIZATION PASS
    secondary RNA-seq stress TECHNICALLY NON-EVALUABLE

CLOSURE DECISION
----------------
CML v44 is CLOSED.

No further tuning or rescue is justified inside this branch.

Any future CML work should begin as a new preregistered hypothesis/version
rather than modifying the v44 evidence chain.

FINAL STATUS
------------
INTERNAL DIRECT-SUBSPACE SIGNAL: SUPPORTED
INTERNAL STABILITY: SUPPORTED
PRIMARY EXTERNAL CROSS-ENDPOINT GENERALIZATION: SUPPORTED, MODEST
SECONDARY RNA-seq CROSS-MODALITY TEST: TECHNICALLY NON-EVALUABLE
CLINICAL CLAIM: NOT ESTABLISHED
QUANTUM ADVANTAGE CLAIM: NOT ESTABLISHED
"""

    closure_path=docs/"v44_7_CML_CLOSURE_AUDIT.txt"
    closure_path.write_text(closure,encoding="utf-8")
    closure_sha=sha256_file(closure_path)

    source_hashes={k:sha256_file(p) for k,p in required.items()}

    manifest={
        "version":"v44.7",
        "status":"CLOSED",
        "branch":"CML cross-platform transport-aware direct subspace",
        "final_status":{
            "internal_direct_subspace_signal":"SUPPORTED",
            "internal_stability":"SUPPORTED",
            "primary_external_cross_endpoint_generalization":"SUPPORTED_MODEST",
            "secondary_rnaseq_cross_modality":"TECHNICALLY_NON_EVALUABLE",
            "clinical_claim":"NOT_ESTABLISHED",
            "quantum_advantage_claim":"NOT_ESTABLISHED"
        },
        "closure_sha256":closure_sha,
        "source_sha256":source_hashes
    }

    manifest_path=docs/"v44_7_CML_CLOSURE_AUDIT_manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    lock_code = (
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import hashlib\n\n"
        f'EXPECTED_SHA256 = "{closure_sha}"\n'
        'HERE = Path(__file__).resolve().parent\n'
        'PATH = HERE / "v44_7_CML_CLOSURE_AUDIT.txt"\n'
        'actual = hashlib.sha256(PATH.read_bytes()).hexdigest()\n'
        'print("v44.7 CML closure integrity check")\n'
        'print("Expected:", EXPECTED_SHA256)\n'
        'print("Actual:  ", actual)\n'
        'if actual != EXPECTED_SHA256:\n'
        '    raise SystemExit("FAIL: v44.7 closure has changed.")\n'
        'print("PASS: v44.7 closure is unchanged.")\n'
    )

    lock_path=docs/"v44_7_CML_closure_lock.py"
    lock_path.write_text(lock_code,encoding="utf-8")

    print("=== v44.7 CML CLOSURE COMPLETE ===")
    print("Created:",closure_path)
    print("Created:",manifest_path)
    print("Created:",lock_path)
    print("Closure SHA256:",closure_sha)
    print()
    print("FINAL:")
    print("  INTERNAL SIGNAL: SUPPORTED")
    print("  INTERNAL STABILITY: SUPPORTED")
    print("  PRIMARY EXTERNAL GENERALIZATION: SUPPORTED, MODEST")
    print("  SECONDARY RNA-seq STRESS: TECHNICALLY NON-EVALUABLE")
    print("  CML v44: CLOSED")

if __name__=="__main__":
    main()
