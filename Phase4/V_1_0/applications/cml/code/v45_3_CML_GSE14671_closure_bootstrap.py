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

    candidate_manifest=ds/"v45_0b_manifest.json"
    prereg=docs/"v45_1_CML_GSE14671_CROSS_ENDPOINT_PREREGISTRATION.txt"
    prereg_manifest=docs/"v45_1_CML_GSE14671_CROSS_ENDPOINT_PREREGISTRATION_manifest.json"
    transport_summary=ds/"v45_2_GSE14671_technical_transport_summary.txt"
    resolution_summary=ds/"v45_2b_PTPN20_resolution_summary.txt"
    resolution_override=ds/"v45_2b_PTPN20_mapping_override.json"

    for x in [candidate_manifest,prereg,prereg_manifest,transport_summary,resolution_summary,resolution_override]:
        if not x.exists():
            raise FileNotFoundError(x)

    cand=json.loads(candidate_manifest.read_text(encoding="utf-8"))
    pre=json.loads(prereg_manifest.read_text(encoding="utf-8"))
    override=json.loads(resolution_override.read_text(encoding="utf-8"))

    if pre.get("protocol_sha256")!=sha256_file(prereg):
        raise RuntimeError("v45.1 preregistration SHA mismatch.")
    if cand.get("status")!="CANDIDATE ELIGIBLE FOR v45.1 PREREGISTRATION":
        raise RuntimeError("v45.0b candidate status mismatch.")
    if cand.get("sample_n")!=59 or cand.get("responder_n")!=41 or cand.get("non_responder_n")!=18:
        raise RuntimeError("v45.0b counts mismatch.")
    if override.get("target_gene")!="PTPN20" or override.get("status")!="UNRESOLVED":
        raise RuntimeError("v45.2b PTPN20 resolution state mismatch.")

    closure='''Soft Spaces / CML
v45.3 — GSE14671 CLOSURE AUDIT

STATUS
------
CLOSED — TECHNICALLY NON-EVALUABLE

SUMMARY
-------
GSE14671 was accepted as a second independent GPL570 CML candidate cohort:
59 pretreatment samples, 41 CCyR responders and 18 poor responders
(>65% Ph-positive metaphases after 12 months).

v45.1 preregistered the external cross-endpoint test before model scoring:
frozen v44.5a Top-256, frozen U16, frozen logistic model, no refit and no
outcome-driven tuning.

v45.2 performed technical transport only. Outcome labels were not used and
no predictive score was calculated.

Frozen Top-256 coordinates: 256
Mapped to GPL570: 255
Missing coordinate: PTPN20

The preregistered technical rule required all 256 frozen coordinates.
Therefore v45.2 was TECHNICALLY NON-EVALUABLE.

v45.2b then performed a targeted outcome-blind GPL570 annotation audit for
PTPN20.

GPL570 annotation rows: 54,675
Rows containing exact PTPN20 token: 0
Unique annotation-supported PTPN20 probes: 0
PTPN20 status: UNRESOLVED

No technical rescue was available under the frozen rules.

NO RESCUE
---------
Not performed:
- dropping PTPN20
- scoring a 255-gene model
- substituting another gene or proxy
- choosing a probe by outcome or predictive performance
- changing Top-256, U16 or rank
- refitting the classifier
- changing normalization or response definition after scoring

FINAL EVIDENCE STATE
--------------------
v45.0b candidate audit: PASS
v45.1 preregistration: FROZEN
v45.2 transport: 255/256; technical gate not satisfied
v45.2b PTPN20 resolution: UNRESOLVED
Predictive scoring: NOT PERFORMED

FINAL INTERPRETATION:
TECHNICALLY NON-EVALUABLE

This is neither an external predictive PASS nor an external predictive FAIL.

PROSPECTIVE REQUIREMENT LEARNED
-------------------------------
A future external cohort should be checked before predictive scoring for:
- independent CML population
- pretreatment material
- clearly defined response endpoint
- adequate patient count
- all 256 frozen coordinates measurable
- PTPN20 explicitly representable
- non-zero variation for every frozen coordinate
- no outcome-driven preprocessing or gene substitution

CLOSURE DECISION
----------------
CML v45 is CLOSED.
Any further external test begins as a new preregistered version.

FINAL STATUS
------------
GSE14671 CANDIDATE QUALITY: ACCEPTABLE
FROZEN COORDINATE TRANSPORT: 255/256
PTPN20 RESOLUTION: UNRESOLVED
PREDICTIVE SCORING: NOT PERFORMED
v45: TECHNICALLY NON-EVALUABLE
'''
    closure_path=docs/"v45_3_CML_GSE14671_CLOSURE_AUDIT.txt"
    closure_path.write_text(closure,encoding="utf-8")
    closure_sha=sha256_file(closure_path)

    manifest={
        "version":"v45.3",
        "status":"CLOSED_TECHNICALLY_NON_EVALUABLE",
        "dataset":"GSE14671",
        "platform":"GPL570",
        "sample_n":59,
        "responder_n":41,
        "non_responder_n":18,
        "frozen_top_k":256,
        "mapped_gene_n":255,
        "missing_gene":"PTPN20",
        "ptpn20_resolution":"UNRESOLVED",
        "predictive_scoring_performed":False,
        "closure_sha256":closure_sha,
        "source_sha256":{
            "v45_0b_manifest":sha256_file(candidate_manifest),
            "v45_1_protocol":sha256_file(prereg),
            "v45_1_manifest":sha256_file(prereg_manifest),
            "v45_2_transport_summary":sha256_file(transport_summary),
            "v45_2b_resolution_summary":sha256_file(resolution_summary),
            "v45_2b_mapping_override":sha256_file(resolution_override),
            "execution_script":sha256_file(Path(__file__).resolve())
        }
    }
    manifest_path=docs/"v45_3_CML_GSE14671_CLOSURE_AUDIT_manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    lock=(f'#!/usr/bin/env python3\nfrom pathlib import Path\nimport hashlib\n'
          f'EXPECTED_SHA256="{closure_sha}"\n'
          'p=Path(__file__).resolve().parent/"v45_3_CML_GSE14671_CLOSURE_AUDIT.txt"\n'
          'a=hashlib.sha256(p.read_bytes()).hexdigest()\n'
          'print("Expected:",EXPECTED_SHA256)\nprint("Actual:  ",a)\n'
          'raise SystemExit("FAIL") if a!=EXPECTED_SHA256 else print("PASS: v45.3 closure unchanged.")\n')
    lock_path=docs/"v45_3_CML_GSE14671_closure_lock.py"
    lock_path.write_text(lock,encoding="utf-8")

    print("=== v45.3 GSE14671 CLOSURE COMPLETE ===")
    print("Created:",closure_path)
    print("Created:",manifest_path)
    print("Created:",lock_path)
    print("Closure SHA256:",closure_sha)
    print("FINAL: v45 CLOSED — TECHNICALLY NON-EVALUABLE")

if __name__=="__main__":
    main()
