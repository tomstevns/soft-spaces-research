#!/usr/bin/env python3
from pathlib import Path
import hashlib, json

PROTOCOL = 'Soft Spaces / CML\nv47.0 — STATE-LANDSCAPE HYPOTHESIS PREREGISTRATION\n\nSTATUS\n------\nFROZEN BEFORE STATE-LANDSCAPE CONSTRUCTION\n\nPRIMARY HYPOTHESIS\n------------------\nCML molecular states may occupy structured, reproducible low-dimensional\nregions in expression space.\n\nThe geometry must be constructed WITHOUT using outcome labels.\n\nANALYSIS ORDER\n--------------\n1. define the expression coordinate system without outcome labels;\n2. construct a low-dimensional state representation without outcome labels;\n3. quantify local geometry without outcome labels;\n4. freeze the geometry;\n5. only then overlay known clinical labels;\n6. test whether clinical states occupy the geometry non-randomly.\n\nINITIAL COHORT\n--------------\nGSE130404\n\nCOORDINATE UNIVERSE\n-------------------\nUse the already-defined v44 transport-aware eligible universe:\n15,890 genes.\n\nFEATURE REDUCTION\n-----------------\nTop 256 genes by raw variance across the full GSE130404 cohort.\nOutcome labels must not be used.\n\nSTANDARDIZATION\n---------------\nPer gene:\nz = (x - cohort_mean) / cohort_sd\nwith ddof=0.\n\nSTATE SPACE\n-----------\nPrimary representation:\nPCA, 16 dimensions.\n\nNo response labels may be used to fit PCA.\n\nLOCAL GEOMETRY\n--------------\nFor each patient in PCA-16 state space compute:\n\n- mean k-nearest-neighbor distance\n- local density proxy = 1 / (mean kNN distance + epsilon)\n- local covariance\n- local anisotropy\n- distance to cohort centroid\n\nFrozen neighborhood size:\nk = 10\n\nLOCAL ANISOTROPY\n----------------\nWithin the k-neighbor set:\nanisotropy = lambda_1 / (sum(lambda_i) + epsilon)\n\nLABEL OVERLAY\n-------------\nClinical labels may be attached only AFTER the geometry matrix has been\nwritten and SHA-hashed.\n\nPRIMARY LABEL-OVERLAY TEST\n--------------------------\nObserved distance between clinical-group centroids in PCA-16.\n\nNull:\n10,000 random label permutations.\n\nEmpirical p:\n(1 + #NULL >= observed) / (1 + 10000)\n\nSECONDARY DESCRIPTIVE TESTS\n---------------------------\nCompare groups on:\n- local density proxy\n- local anisotropy\n- distance to cohort centroid\n\nCLAIM LIMIT\n-----------\nA positive result would support only that independently defined clinical\ngroups occupy different regions of a label-blind molecular state geometry.\n\nIt would NOT establish:\n- clinical utility\n- diagnosis\n- prognosis\n- causality\n- cancer dormancy mechanism\n- quantum biology\n- quantum advantage\n\nFALSIFICATION\n-------------\nThe hypothesis is weakened if:\n- the label-blind geometry is unstable under resampling;\n- group separation is indistinguishable from permutation;\n- local geometry is not reproducible;\n- similar geometry cannot later be reproduced externally.\n\nSEQUENCE\n--------\nv47.0 preregistration\nv47.1 technical input audit, no labels\nv47.2 label-blind PCA-16 landscape\nv47.3 label overlay + permutation test\nv47.4+ stability / external replication only if warranted\n'
AUDIT_SCRIPT = '#!/usr/bin/env python3\nfrom pathlib import Path\nimport hashlib, json\n\ndef project_dir():\n    return Path(__file__).resolve().parent.parent\n\ndef sha256_file(path):\n    h=hashlib.sha256()\n    with open(path,"rb") as f:\n        for chunk in iter(lambda:f.read(1024*1024),b""):\n            h.update(chunk)\n    return h.hexdigest()\n\ndef find_first_existing(candidates):\n    for p in candidates:\n        if p.exists():\n            return p\n    return None\n\ndef main():\n    project=project_dir()\n    ds=project/"results"/"direct_subspace"\n    docs=project/"docs"\n\n    protocol=docs/"v47_0_CML_STATE_LANDSCAPE_PREREGISTRATION.txt"\n    pmanifest=docs/"v47_0_CML_STATE_LANDSCAPE_PREREGISTRATION_manifest.json"\n\n    for p in [protocol,pmanifest]:\n        if not p.exists():\n            raise FileNotFoundError(p)\n\n    pm=json.loads(pmanifest.read_text(encoding="utf-8"))\n    actual=sha256_file(protocol)\n\n    if actual!=pm["protocol_sha256"]:\n        raise RuntimeError("v47.0 protocol SHA mismatch.")\n\n    eligible_candidates=[\n        ds/"v44_1b_eligible_genes.txt",\n        project/"results"/"v44_1b_eligible_genes.txt",\n        project/"data"/"v44_1b_eligible_genes.txt",\n    ]\n\n    eligible_path=find_first_existing(eligible_candidates)\n\n    if eligible_path is None:\n        raise FileNotFoundError(\n            "Could not locate v44_1b_eligible_genes.txt."\n        )\n\n    eligible=[\n        x.strip()\n        for x in eligible_path.read_text(encoding="utf-8").splitlines()\n        if x.strip()\n    ]\n\n    if len(eligible)!=15890:\n        raise RuntimeError(\n            f"Expected 15890 eligible genes, found {len(eligible)}."\n        )\n\n    candidates=[]\n    roots=[\n        project/"data",\n        project/"results",\n        project/"results"/"direct_subspace",\n    ]\n\n    for root in roots:\n        if root.exists():\n            for pattern in [\n                "*GSE130404*.csv",\n                "*GSE130404*.tsv",\n                "*GSE130404*.txt",\n            ]:\n                candidates.extend(root.rglob(pattern))\n\n    matrix_candidates=[]\n\n    for p in sorted(set(candidates)):\n        name=p.name.lower()\n\n        if any(x in name for x in [\n            "manifest","summary","audit","protocol","prereg","closure"\n        ]):\n            continue\n\n        if p.stat().st_size < 10000:\n            continue\n\n        matrix_candidates.append(p)\n\n    summary_out=ds/"v47_1_state_landscape_input_audit_summary.txt"\n    manifest_out=ds/"v47_1_manifest.json"\n\n    lines=[\n        "=== Soft Spaces / CML v47.1 STATE-LANDSCAPE INPUT AUDIT ===",\n        "",\n        f"v47.0 protocol SHA verified: {actual}",\n        "",\n        "LABEL USE",\n        "---------",\n        "Outcome labels used: NO",\n        "PCA fitted: NO",\n        "Predictive model fitted: NO",\n        "",\n        "TRANSPORT-AWARE UNIVERSE",\n        "------------------------",\n        f"Eligible gene file: {eligible_path}",\n        f"Eligible genes: {len(eligible)}",\n        "",\n        "GSE130404 MATRIX CANDIDATES",\n        "--------------------------",\n    ]\n\n    if matrix_candidates:\n        for p in matrix_candidates:\n            lines.append(f"{p} | {p.stat().st_size} bytes")\n\n        status="MATRIX_CANDIDATES_FOUND"\n\n        lines += [\n            "",\n            "STATUS: MATRIX CANDIDATES FOUND",\n            "",\n            "NEXT:",\n            "Use the exact label-blind expression matrix for v47.2."\n        ]\n    else:\n        status="INPUT_MATRIX_RESOLUTION_REQUIRED"\n\n        lines += [\n            "No sufficiently large existing GSE130404 matrix located.",\n            "",\n            "STATUS: INPUT MATRIX RESOLUTION REQUIRED",\n            "",\n            "Do not construct the landscape yet."\n        ]\n\n    summary_out.write_text("\\n".join(lines)+"\\n",encoding="utf-8")\n\n    manifest_out.write_text(json.dumps({\n        "version":"v47.1",\n        "status":status,\n        "outcome_labels_used":False,\n        "pca_fitted":False,\n        "predictive_model_fitted":False,\n        "eligible_gene_n":len(eligible),\n        "eligible_gene_file":str(eligible_path),\n        "matrix_candidates":[str(p) for p in matrix_candidates],\n        "sha256":{\n            "v47_0_protocol":actual,\n            "eligible_gene_file":sha256_file(eligible_path),\n            "execution_script":sha256_file(Path(__file__).resolve())\n        }\n    },indent=2),encoding="utf-8")\n\n    print(summary_out.read_text(encoding="utf-8"))\n    print("Wrote:")\n    print(" ",summary_out)\n    print(" ",manifest_out)\n\nif __name__=="__main__":\n    main()\n'

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
    docs.mkdir(parents=True,exist_ok=True)

    protocol_path=docs/"v47_0_CML_STATE_LANDSCAPE_PREREGISTRATION.txt"
    protocol_path.write_text(PROTOCOL,encoding="utf-8")
    protocol_sha=sha256_file(protocol_path)

    manifest_path=docs/"v47_0_CML_STATE_LANDSCAPE_PREREGISTRATION_manifest.json"
    manifest_path.write_text(json.dumps({
        "version":"v47.0",
        "status":"FROZEN_BEFORE_STATE_LANDSCAPE_CONSTRUCTION",
        "development_cohort":"GSE130404",
        "eligible_universe_n":15890,
        "feature_selection":"Top256 raw variance, label-blind",
        "state_space":"PCA-16, label-blind",
        "knn_k":10,
        "primary_overlay_test":"PCA16 group-centroid distance",
        "permutations":10000,
        "labels_used_for_geometry":False,
        "protocol_sha256":protocol_sha
    },indent=2),encoding="utf-8")

    lock_path=docs/"v47_0_CML_state_landscape_protocol_lock.py"
    lock_path.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import hashlib\n"
        f'EXPECTED_SHA256="{protocol_sha}"\n'
        'p=Path(__file__).resolve().parent/"v47_0_CML_STATE_LANDSCAPE_PREREGISTRATION.txt"\n'
        'a=hashlib.sha256(p.read_bytes()).hexdigest()\n'
        'print("Expected:",EXPECTED_SHA256)\n'
        'print("Actual:  ",a)\n'
        'raise SystemExit("FAIL") if a!=EXPECTED_SHA256 else print("PASS: v47.0 protocol unchanged.")\n',
        encoding="utf-8"
    )

    audit_path=code_dir/"v47_1_CML_state_landscape_input_audit.py"
    audit_path.write_text(AUDIT_SCRIPT,encoding="utf-8")

    print("=== v47.0 STATE-LANDSCAPE BOOTSTRAP COMPLETE ===")
    print("Protocol SHA256:",protocol_sha)
    print("Created:",protocol_path)
    print("Created:",manifest_path)
    print("Created:",lock_path)
    print("Created:",audit_path)
    print()
    print(r"NEXT: python -X utf8 -u .\v47_1_CML_state_landscape_input_audit.py")

if __name__=="__main__":
    main()
