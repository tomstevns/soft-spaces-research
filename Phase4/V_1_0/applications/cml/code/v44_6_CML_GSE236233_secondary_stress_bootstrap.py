#!/usr/bin/env python3
from pathlib import Path
import hashlib, json

PROTOCOL = 'Soft Spaces / CML\nv44.6 — GSE236233 SECONDARY CROSS-MODALITY STRESS PREREGISTRATION\n\nSTATUS\n------\nFROZEN BEFORE GSE236233 MODEL SCORING\n\nROLE\n----\nSecondary cross-modality stress test only.\nThe primary external test remains GSE44589 v44.5b1.\n\nCOHORT / UNIT\n-------------\nGSE236233, Lin-CD34+ patient-level pseudobulk.\nStatistical unit: PATIENT.\n\nPUBLISHED 12-MONTH RESPONSE STRATA\n----------------------------------\nOptimal: CML1, CML2, CML3, CML4\nWarning: CML5, CML6, CML7\nFailure: CML8, CML9\n\nPRIMARY SECONDARY-STRESS COMPARISON\n-----------------------------------\nFailure = positive class\nOptimal = negative class\nWarning excluded from the binary decision metric and retained descriptively.\n\nFROZEN MODEL\n------------\nReuse v44.5a without modification:\n- final Top-256\n- U16\n- frozen balanced L2 logistic regression\n\nNo refit, no rank change, no gene substitution.\n\nFROZEN RNA TRANSFORM\n--------------------\nStarting from raw Lin-CD34+ patient pseudobulk counts:\n1. restrict to frozen Top-256\n2. patient CPM normalization over those 256 genes\n3. log1p(CPM)\n4. per-gene z-standardization across all 9 patients\n\nNo response labels enter this transformation.\n\nTECHNICAL GATE\n--------------\nAll 256 frozen coordinates must exist.\nAny zero cross-patient SD after transform => TECHNICALLY NON-EVALUABLE.\n\nPRIMARY METRIC\n--------------\nPR-AUC on Failure vs Optimal.\nReference prevalence = 2/6.\n\nSecondary:\nROC-AUC, balanced accuracy @0.5, sensitivity, specificity.\n\nFROZEN DIRECTION CHECK\n----------------------\nMean score Failure > mean score Optimal.\n\nDESCRIPTIVE ONLY\n----------------\nReport mean Optimal, Warning and Failure scores and whether\nOptimal < Warning < Failure.\n\nSECONDARY STRESS PASS RULE\n--------------------------\nPASS iff:\n1. 256/256 technically transportable\n2. PR-AUC > prevalence\n3. ROC-AUC > 0.5\n4. mean Failure score > mean Optimal score\n5. no outcome-driven tuning\n\nNo p-value threshold is required because the extreme-group test has only\n6 patients (4 Optimal, 2 Failure).\n\nNO RESCUE\n---------\nNo post-score normalization change, threshold tuning, gene substitution,\nrank change, model refit, or outcome-driven exclusion.\n\nCLAIM LIMIT\n-----------\nA PASS supports only directional cross-modality consistency of the frozen\ntransport-aware CML subspace. It does not establish clinical utility,\ncausal biology, superiority to established biomarkers, or quantum advantage.\n'
SCORER = '#!/usr/bin/env python3\nfrom pathlib import Path\nimport hashlib, json\nimport numpy as np\nimport pandas as pd\nfrom sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score, confusion_matrix\n\nTOP_K=256\nR=16\nOPTIMAL=["CML1","CML2","CML3","CML4"]\nWARNING=["CML5","CML6","CML7"]\nFAILURE=["CML8","CML9"]\nCATEGORY={**{x:"Optimal" for x in OPTIMAL}, **{x:"Warning" for x in WARNING}, **{x:"Failure" for x in FAILURE}}\n\ndef project_dir():\n    return Path(__file__).resolve().parent.parent\n\ndef sha256_file(path):\n    h=hashlib.sha256()\n    with open(path,"rb") as f:\n        for chunk in iter(lambda:f.read(1024*1024),b""):\n            h.update(chunk)\n    return h.hexdigest()\n\ndef sigmoid(x):\n    x=np.clip(np.asarray(x,dtype=float),-700,700)\n    return 1/(1+np.exp(-x))\n\ndef main():\n    project=project_dir()\n    docs=project/"docs"\n    ds=project/"results"/"direct_subspace"\n\n    protocol=docs/"v44_6_CML_GSE236233_SECONDARY_STRESS_PREREGISTRATION.txt"\n    pmanifest=docs/"v44_6_CML_GSE236233_SECONDARY_STRESS_PREREGISTRATION_manifest.json"\n    model_path=ds/"v44_5a_frozen_development_model.npz"\n    pseudo_path=ds/"v43_4d_GSE236233_CD34_patient_pseudobulk_raw_counts.csv"\n\n    for p in [protocol,pmanifest,model_path,pseudo_path]:\n        if not p.exists(): raise FileNotFoundError(p)\n\n    pm=json.loads(pmanifest.read_text(encoding="utf-8"))\n    expected=pm["protocol_sha256"]\n    actual=sha256_file(protocol)\n\n    print("=== v44.6a GSE236233 SECONDARY CROSS-MODALITY STRESS TEST ===")\n    print("Expected protocol SHA:",expected)\n    print("Actual protocol SHA:  ",actual)\n    if expected!=actual: raise SystemExit("FAIL: v44.6 protocol SHA mismatch.")\n    print("PASS: v44.6 protocol verified.")\n    print("External refit: NO")\n    print("External tuning: NO\\n")\n\n    model=np.load(model_path,allow_pickle=False)\n    genes=[str(x) for x in model["genes"].tolist()]\n    U16=np.asarray(model["U16"],dtype=float)\n    coef=np.asarray(model["logistic_coef"],dtype=float)\n    intercept=np.asarray(model["logistic_intercept"],dtype=float)\n\n    if len(genes)!=TOP_K: raise RuntimeError(f"Expected {TOP_K} frozen genes, found {len(genes)}")\n    if U16.shape!=(TOP_K,R): raise RuntimeError(f"Unexpected U16 shape {U16.shape}")\n\n    pseudo=pd.read_csv(pseudo_path,index_col=0)\n    pseudo.index=pseudo.index.astype(str)\n    pseudo.columns=pseudo.columns.astype(str)\n\n    patients=[f"CML{i}" for i in range(1,10)]\n    missing_patients=[p for p in patients if p not in pseudo.index]\n    if missing_patients: raise RuntimeError("Missing patients: "+", ".join(missing_patients))\n\n    missing_genes=[g for g in genes if g not in pseudo.columns]\n    if missing_genes:\n        raise RuntimeError("TECHNICALLY NON-EVALUABLE: missing frozen genes: "+", ".join(missing_genes))\n\n    Xraw=pseudo.loc[patients,genes].to_numpy(dtype=float)\n    if np.any(Xraw<0): raise RuntimeError("Negative pseudobulk counts found.")\n\n    lib=Xraw.sum(axis=1)\n    if np.any(lib<=0): raise RuntimeError("Zero patient library size across frozen Top-256.")\n\n    Xcpm=Xraw/lib[:,None]*1_000_000.0\n    Xlog=np.log1p(Xcpm)\n    mu=Xlog.mean(axis=0)\n    sd=Xlog.std(axis=0,ddof=0)\n\n    zero=np.where(sd==0)[0]\n    if len(zero):\n        raise RuntimeError("TECHNICALLY NON-EVALUABLE: zero-SD genes: "+", ".join(genes[i] for i in zero))\n\n    Z=(Xlog-mu)/sd\n    Zsub=Z@U16\n    score=sigmoid((Zsub@coef.T+intercept.reshape(1,-1)).ravel())\n\n    result=pd.DataFrame({\n        "patient_id":patients,\n        "response_stratum":[CATEGORY[p] for p in patients],\n        "frozen_probability":score,\n        "top256_raw_count_sum":lib\n    })\n\n    ext=result[result["response_stratum"].isin(["Optimal","Failure"])].copy()\n    y=(ext["response_stratum"]=="Failure").astype(int).to_numpy()\n    s=ext["frozen_probability"].to_numpy(float)\n\n    if len(y)!=6 or int(y.sum())!=2:\n        raise RuntimeError(f"Frozen extreme-group mismatch n={len(y)}, failures={int(y.sum())}")\n\n    prevalence=float(y.mean())\n    pr=float(average_precision_score(y,s))\n    roc=float(roc_auc_score(y,s))\n    pred=(s>=0.5).astype(int)\n    ba=float(balanced_accuracy_score(y,pred))\n    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()\n    sens=float(tp/(tp+fn)) if tp+fn else float("nan")\n    spec=float(tn/(tn+fp)) if tn+fp else float("nan")\n\n    mo=float(result.loc[result.response_stratum=="Optimal","frozen_probability"].mean())\n    mw=float(result.loc[result.response_stratum=="Warning","frozen_probability"].mean())\n    mf=float(result.loc[result.response_stratum=="Failure","frozen_probability"].mean())\n\n    direction=mf>mo\n    ordered=mo<mw<mf\n    status="SECONDARY STRESS PASS" if (pr>prevalence and roc>0.5 and direction) else "SECONDARY STRESS FAIL"\n\n    pred_out=ds/"v44_6a_GSE236233_patient_scores.csv"\n    summary_out=ds/"v44_6a_GSE236233_secondary_stress_summary.txt"\n    manifest_out=ds/"v44_6a_manifest.json"\n\n    result.to_csv(pred_out,index=False)\n\n    lines=[\n        "=== Soft Spaces / CML v44.6a GSE236233 SECONDARY CROSS-MODALITY STRESS TEST ===","",\n        "FROZEN DESIGN","-------------",\n        "External cohort: GSE236233",\n        "Population: Lin-CD34+ patient-level pseudobulk",\n        "Statistical unit: patient",\n        "Patients: 9",\n        "Frozen Top-K: 256",\n        "Frozen rank: 16",\n        "RNA transform: CPM -> log1p -> per-gene z-score across all 9 patients",\n        "External model refitting: NO",\n        "External tuning: NO","",\n        "PUBLISHED 12-MONTH RESPONSE STRATA","----------------------------------",\n        "Optimal: CML1, CML2, CML3, CML4",\n        "Warning: CML5, CML6, CML7",\n        "Failure: CML8, CML9","",\n        "PRIMARY SECONDARY-STRESS COMPARISON","-----------------------------------",\n        "Failure positive vs Optimal negative",\n        "Warning excluded from binary metric and retained descriptively",\n        f"Evaluable extreme-group n:       {len(y)}",\n        f"Failure n:                       {int(y.sum())}",\n        f"Optimal n:                       {int((y==0).sum())}",\n        f"Failure prevalence:              {prevalence:.6f}",\n        f"PR-AUC:                          {pr:.6f}",\n        f"PR-AUC - prevalence:             {pr-prevalence:+.6f}",\n        f"ROC-AUC:                         {roc:.6f}",\n        f"Balanced accuracy @0.5:          {ba:.6f}",\n        f"Sensitivity @0.5:                {sens:.6f}",\n        f"Specificity @0.5:                {spec:.6f}","",\n        "FROZEN DIRECTION","----------------",\n        f"Mean score Optimal:              {mo:.6f}",\n        f"Mean score Warning:              {mw:.6f}",\n        f"Mean score Failure:              {mf:.6f}",\n        f"Failure > Optimal:               {\'YES\' if direction else \'NO\'}",\n        f"Optimal < Warning < Failure:     {\'YES\' if ordered else \'NO\'}","",\n        "DECISION","--------",\n        "SECONDARY STRESS PASS requires PR-AUC > prevalence, ROC-AUC > 0.5,",\n        "and mean frozen score Failure > Optimal.","",\n        f"v44.6a STATUS: {status}","",\n        "INTERPRETATION LIMIT","--------------------",\n        "This is a secondary cross-modality stress test.",\n        "The primary binary comparison contains only 6 patients (4 Optimal, 2 Failure).",\n        "Warning patients are descriptive only.",\n        "No clinical-validation claim is permitted."\n    ]\n\n    summary_out.write_text("\\n".join(lines)+"\\n",encoding="utf-8")\n\n    manifest_out.write_text(json.dumps({\n        "version":"v44.6a","status":status,\n        "patient_n":9,"extreme_group_n":len(y),"failure_n":int(y.sum()),"optimal_n":int((y==0).sum()),\n        "failure_prevalence":prevalence,"pr_auc":pr,"pr_auc_minus_prevalence":pr-prevalence,\n        "roc_auc":roc,"balanced_accuracy":ba,"sensitivity":sens,"specificity":spec,\n        "mean_optimal_score":mo,"mean_warning_score":mw,"mean_failure_score":mf,\n        "direction_preserved":bool(direction),"ordered_means_descriptive":bool(ordered),\n        "external_refit":False,"external_tuning":False,\n        "sha256":{"v44_6_protocol":actual,"frozen_model":sha256_file(model_path),\n                  "gse236233_pseudobulk":sha256_file(pseudo_path),\n                  "execution_script":sha256_file(Path(__file__).resolve())}\n    },indent=2),encoding="utf-8")\n\n    print(summary_out.read_text(encoding="utf-8"))\n    print("Wrote:")\n    for p in [pred_out,summary_out,manifest_out]: print(" ",p)\n\nif __name__=="__main__":\n    main()\n'

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
    ds.mkdir(parents=True,exist_ok=True)

    primary_manifest=ds/"v44_5b1_manifest.json"
    frozen_model=ds/"v44_5a_frozen_development_model.npz"
    pseudobulk=ds/"v43_4d_GSE236233_CD34_patient_pseudobulk_raw_counts.csv"

    for p in [primary_manifest,frozen_model,pseudobulk]:
        if not p.exists(): raise FileNotFoundError(p)

    primary=json.loads(primary_manifest.read_text(encoding="utf-8"))
    if primary.get("status")!="PASS":
        raise RuntimeError("v44.5b1 primary external test is not recorded as PASS.")

    protocol_path=docs/"v44_6_CML_GSE236233_SECONDARY_STRESS_PREREGISTRATION.txt"
    protocol_path.write_text(PROTOCOL,encoding="utf-8")
    protocol_sha=sha256_file(protocol_path)

    manifest_path=docs/"v44_6_CML_GSE236233_SECONDARY_STRESS_PREREGISTRATION_manifest.json"
    manifest_path.write_text(json.dumps({
        "version":"v44.6",
        "status":"FROZEN_BEFORE_GSE236233_MODEL_SCORING",
        "role":"secondary cross-modality stress test",
        "external_cohort":"GSE236233",
        "population":"Lin-CD34+ patient pseudobulk",
        "patient_n":9,
        "optimal":["CML1","CML2","CML3","CML4"],
        "warning":["CML5","CML6","CML7"],
        "failure":["CML8","CML9"],
        "primary_comparison":"Failure vs Optimal",
        "frozen_rank":16,
        "top_k":256,
        "rna_transform":"Top256 CPM -> log1p -> per-gene z across 9 patients",
        "primary_metric":"PR-AUC",
        "outcome_tuning_allowed":False,
        "protocol_sha256":protocol_sha,
        "v44_5b1_manifest_sha256":sha256_file(primary_manifest),
        "frozen_model_sha256":sha256_file(frozen_model),
        "pseudobulk_sha256":sha256_file(pseudobulk)
    },indent=2),encoding="utf-8")

    lock_path=docs/"v44_6_CML_secondary_stress_protocol_lock.py"
    lock_path.write_text(
        '#!/usr/bin/env python3\nfrom pathlib import Path\nimport hashlib\n'
        f'EXPECTED_SHA256="{protocol_sha}"\n'
        'p=Path(__file__).resolve().parent/"v44_6_CML_GSE236233_SECONDARY_STRESS_PREREGISTRATION.txt"\n'
        'a=hashlib.sha256(p.read_bytes()).hexdigest()\n'
        'print("Expected:",EXPECTED_SHA256)\nprint("Actual:  ",a)\n'
        'raise SystemExit("FAIL") if a!=EXPECTED_SHA256 else print("PASS: v44.6 preregistration unchanged.")\n',
        encoding="utf-8"
    )

    scorer_path=code_dir/"v44_6a_CML_GSE236233_secondary_crossmodality_stress.py"
    scorer_path.write_text(SCORER,encoding="utf-8")

    print("=== v44.6 BOOTSTRAP COMPLETE ===")
    print("Protocol SHA256:",protocol_sha)
    print("Created:",protocol_path)
    print("Created:",manifest_path)
    print("Created:",lock_path)
    print("Created:",scorer_path)
    print()
    print(r"NEXT: python -X utf8 -u .\v44_6a_CML_GSE236233_secondary_crossmodality_stress.py")

if __name__=="__main__":
    main()
