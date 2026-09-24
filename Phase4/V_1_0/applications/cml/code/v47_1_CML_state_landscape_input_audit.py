#!/usr/bin/env python3
from pathlib import Path
import hashlib, json

def project_dir():
    return Path(__file__).resolve().parent.parent

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def find_first_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None

def main():
    project=project_dir()
    ds=project/"results"/"direct_subspace"
    docs=project/"docs"

    protocol=docs/"v47_0_CML_STATE_LANDSCAPE_PREREGISTRATION.txt"
    pmanifest=docs/"v47_0_CML_STATE_LANDSCAPE_PREREGISTRATION_manifest.json"

    for p in [protocol,pmanifest]:
        if not p.exists():
            raise FileNotFoundError(p)

    pm=json.loads(pmanifest.read_text(encoding="utf-8"))
    actual=sha256_file(protocol)

    if actual!=pm["protocol_sha256"]:
        raise RuntimeError("v47.0 protocol SHA mismatch.")

    eligible_candidates=[
        ds/"v44_1b_eligible_genes.txt",
        project/"results"/"v44_1b_eligible_genes.txt",
        project/"data"/"v44_1b_eligible_genes.txt",
    ]

    eligible_path=find_first_existing(eligible_candidates)

    if eligible_path is None:
        raise FileNotFoundError(
            "Could not locate v44_1b_eligible_genes.txt."
        )

    eligible=[
        x.strip()
        for x in eligible_path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    if len(eligible)!=15890:
        raise RuntimeError(
            f"Expected 15890 eligible genes, found {len(eligible)}."
        )

    candidates=[]
    roots=[
        project/"data",
        project/"results",
        project/"results"/"direct_subspace",
    ]

    for root in roots:
        if root.exists():
            for pattern in [
                "*GSE130404*.csv",
                "*GSE130404*.tsv",
                "*GSE130404*.txt",
            ]:
                candidates.extend(root.rglob(pattern))

    matrix_candidates=[]

    for p in sorted(set(candidates)):
        name=p.name.lower()

        if any(x in name for x in [
            "manifest","summary","audit","protocol","prereg","closure"
        ]):
            continue

        if p.stat().st_size < 10000:
            continue

        matrix_candidates.append(p)

    summary_out=ds/"v47_1_state_landscape_input_audit_summary.txt"
    manifest_out=ds/"v47_1_manifest.json"

    lines=[
        "=== Soft Spaces / CML v47.1 STATE-LANDSCAPE INPUT AUDIT ===",
        "",
        f"v47.0 protocol SHA verified: {actual}",
        "",
        "LABEL USE",
        "---------",
        "Outcome labels used: NO",
        "PCA fitted: NO",
        "Predictive model fitted: NO",
        "",
        "TRANSPORT-AWARE UNIVERSE",
        "------------------------",
        f"Eligible gene file: {eligible_path}",
        f"Eligible genes: {len(eligible)}",
        "",
        "GSE130404 MATRIX CANDIDATES",
        "--------------------------",
    ]

    if matrix_candidates:
        for p in matrix_candidates:
            lines.append(f"{p} | {p.stat().st_size} bytes")

        status="MATRIX_CANDIDATES_FOUND"

        lines += [
            "",
            "STATUS: MATRIX CANDIDATES FOUND",
            "",
            "NEXT:",
            "Use the exact label-blind expression matrix for v47.2."
        ]
    else:
        status="INPUT_MATRIX_RESOLUTION_REQUIRED"

        lines += [
            "No sufficiently large existing GSE130404 matrix located.",
            "",
            "STATUS: INPUT MATRIX RESOLUTION REQUIRED",
            "",
            "Do not construct the landscape yet."
        ]

    summary_out.write_text("\n".join(lines)+"\n",encoding="utf-8")

    manifest_out.write_text(json.dumps({
        "version":"v47.1",
        "status":status,
        "outcome_labels_used":False,
        "pca_fitted":False,
        "predictive_model_fitted":False,
        "eligible_gene_n":len(eligible),
        "eligible_gene_file":str(eligible_path),
        "matrix_candidates":[str(p) for p in matrix_candidates],
        "sha256":{
            "v47_0_protocol":actual,
            "eligible_gene_file":sha256_file(eligible_path),
            "execution_script":sha256_file(Path(__file__).resolve())
        }
    },indent=2),encoding="utf-8")

    print(summary_out.read_text(encoding="utf-8"))
    print("Wrote:")
    print(" ",summary_out)
    print(" ",manifest_out)

if __name__=="__main__":
    main()
