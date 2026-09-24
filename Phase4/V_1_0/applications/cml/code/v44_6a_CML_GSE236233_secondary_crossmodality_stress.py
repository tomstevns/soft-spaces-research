#!/usr/bin/env python3
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score, confusion_matrix

TOP_K=256
R=16
OPTIMAL=["CML1","CML2","CML3","CML4"]
WARNING=["CML5","CML6","CML7"]
FAILURE=["CML8","CML9"]
CATEGORY={**{x:"Optimal" for x in OPTIMAL}, **{x:"Warning" for x in WARNING}, **{x:"Failure" for x in FAILURE}}

def project_dir():
    return Path(__file__).resolve().parent.parent

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def sigmoid(x):
    x=np.clip(np.asarray(x,dtype=float),-700,700)
    return 1/(1+np.exp(-x))

def main():
    project=project_dir()
    docs=project/"docs"
    ds=project/"results"/"direct_subspace"

    protocol=docs/"v44_6_CML_GSE236233_SECONDARY_STRESS_PREREGISTRATION.txt"
    pmanifest=docs/"v44_6_CML_GSE236233_SECONDARY_STRESS_PREREGISTRATION_manifest.json"
    model_path=ds/"v44_5a_frozen_development_model.npz"
    pseudo_path=ds/"v43_4d_GSE236233_CD34_patient_pseudobulk_raw_counts.csv"

    for p in [protocol,pmanifest,model_path,pseudo_path]:
        if not p.exists(): raise FileNotFoundError(p)

    pm=json.loads(pmanifest.read_text(encoding="utf-8"))
    expected=pm["protocol_sha256"]
    actual=sha256_file(protocol)

    print("=== v44.6a GSE236233 SECONDARY CROSS-MODALITY STRESS TEST ===")
    print("Expected protocol SHA:",expected)
    print("Actual protocol SHA:  ",actual)
    if expected!=actual: raise SystemExit("FAIL: v44.6 protocol SHA mismatch.")
    print("PASS: v44.6 protocol verified.")
    print("External refit: NO")
    print("External tuning: NO\n")

    model=np.load(model_path,allow_pickle=False)
    genes=[str(x) for x in model["genes"].tolist()]
    U16=np.asarray(model["U16"],dtype=float)
    coef=np.asarray(model["logistic_coef"],dtype=float)
    intercept=np.asarray(model["logistic_intercept"],dtype=float)

    if len(genes)!=TOP_K: raise RuntimeError(f"Expected {TOP_K} frozen genes, found {len(genes)}")
    if U16.shape!=(TOP_K,R): raise RuntimeError(f"Unexpected U16 shape {U16.shape}")

    pseudo=pd.read_csv(pseudo_path,index_col=0)
    pseudo.index=pseudo.index.astype(str)
    pseudo.columns=pseudo.columns.astype(str)

    patients=[f"CML{i}" for i in range(1,10)]
    missing_patients=[p for p in patients if p not in pseudo.index]
    if missing_patients: raise RuntimeError("Missing patients: "+", ".join(missing_patients))

    missing_genes=[g for g in genes if g not in pseudo.columns]
    if missing_genes:
        raise RuntimeError("TECHNICALLY NON-EVALUABLE: missing frozen genes: "+", ".join(missing_genes))

    Xraw=pseudo.loc[patients,genes].to_numpy(dtype=float)
    if np.any(Xraw<0): raise RuntimeError("Negative pseudobulk counts found.")

    lib=Xraw.sum(axis=1)
    if np.any(lib<=0): raise RuntimeError("Zero patient library size across frozen Top-256.")

    Xcpm=Xraw/lib[:,None]*1_000_000.0
    Xlog=np.log1p(Xcpm)
    mu=Xlog.mean(axis=0)
    sd=Xlog.std(axis=0,ddof=0)

    zero=np.where(sd==0)[0]
    if len(zero):
        raise RuntimeError("TECHNICALLY NON-EVALUABLE: zero-SD genes: "+", ".join(genes[i] for i in zero))

    Z=(Xlog-mu)/sd
    Zsub=Z@U16
    score=sigmoid((Zsub@coef.T+intercept.reshape(1,-1)).ravel())

    result=pd.DataFrame({
        "patient_id":patients,
        "response_stratum":[CATEGORY[p] for p in patients],
        "frozen_probability":score,
        "top256_raw_count_sum":lib
    })

    ext=result[result["response_stratum"].isin(["Optimal","Failure"])].copy()
    y=(ext["response_stratum"]=="Failure").astype(int).to_numpy()
    s=ext["frozen_probability"].to_numpy(float)

    if len(y)!=6 or int(y.sum())!=2:
        raise RuntimeError(f"Frozen extreme-group mismatch n={len(y)}, failures={int(y.sum())}")

    prevalence=float(y.mean())
    pr=float(average_precision_score(y,s))
    roc=float(roc_auc_score(y,s))
    pred=(s>=0.5).astype(int)
    ba=float(balanced_accuracy_score(y,pred))
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sens=float(tp/(tp+fn)) if tp+fn else float("nan")
    spec=float(tn/(tn+fp)) if tn+fp else float("nan")

    mo=float(result.loc[result.response_stratum=="Optimal","frozen_probability"].mean())
    mw=float(result.loc[result.response_stratum=="Warning","frozen_probability"].mean())
    mf=float(result.loc[result.response_stratum=="Failure","frozen_probability"].mean())

    direction=mf>mo
    ordered=mo<mw<mf
    status="SECONDARY STRESS PASS" if (pr>prevalence and roc>0.5 and direction) else "SECONDARY STRESS FAIL"

    pred_out=ds/"v44_6a_GSE236233_patient_scores.csv"
    summary_out=ds/"v44_6a_GSE236233_secondary_stress_summary.txt"
    manifest_out=ds/"v44_6a_manifest.json"

    result.to_csv(pred_out,index=False)

    lines=[
        "=== Soft Spaces / CML v44.6a GSE236233 SECONDARY CROSS-MODALITY STRESS TEST ===","",
        "FROZEN DESIGN","-------------",
        "External cohort: GSE236233",
        "Population: Lin-CD34+ patient-level pseudobulk",
        "Statistical unit: patient",
        "Patients: 9",
        "Frozen Top-K: 256",
        "Frozen rank: 16",
        "RNA transform: CPM -> log1p -> per-gene z-score across all 9 patients",
        "External model refitting: NO",
        "External tuning: NO","",
        "PUBLISHED 12-MONTH RESPONSE STRATA","----------------------------------",
        "Optimal: CML1, CML2, CML3, CML4",
        "Warning: CML5, CML6, CML7",
        "Failure: CML8, CML9","",
        "PRIMARY SECONDARY-STRESS COMPARISON","-----------------------------------",
        "Failure positive vs Optimal negative",
        "Warning excluded from binary metric and retained descriptively",
        f"Evaluable extreme-group n:       {len(y)}",
        f"Failure n:                       {int(y.sum())}",
        f"Optimal n:                       {int((y==0).sum())}",
        f"Failure prevalence:              {prevalence:.6f}",
        f"PR-AUC:                          {pr:.6f}",
        f"PR-AUC - prevalence:             {pr-prevalence:+.6f}",
        f"ROC-AUC:                         {roc:.6f}",
        f"Balanced accuracy @0.5:          {ba:.6f}",
        f"Sensitivity @0.5:                {sens:.6f}",
        f"Specificity @0.5:                {spec:.6f}","",
        "FROZEN DIRECTION","----------------",
        f"Mean score Optimal:              {mo:.6f}",
        f"Mean score Warning:              {mw:.6f}",
        f"Mean score Failure:              {mf:.6f}",
        f"Failure > Optimal:               {'YES' if direction else 'NO'}",
        f"Optimal < Warning < Failure:     {'YES' if ordered else 'NO'}","",
        "DECISION","--------",
        "SECONDARY STRESS PASS requires PR-AUC > prevalence, ROC-AUC > 0.5,",
        "and mean frozen score Failure > Optimal.","",
        f"v44.6a STATUS: {status}","",
        "INTERPRETATION LIMIT","--------------------",
        "This is a secondary cross-modality stress test.",
        "The primary binary comparison contains only 6 patients (4 Optimal, 2 Failure).",
        "Warning patients are descriptive only.",
        "No clinical-validation claim is permitted."
    ]

    summary_out.write_text("\n".join(lines)+"\n",encoding="utf-8")

    manifest_out.write_text(json.dumps({
        "version":"v44.6a","status":status,
        "patient_n":9,"extreme_group_n":len(y),"failure_n":int(y.sum()),"optimal_n":int((y==0).sum()),
        "failure_prevalence":prevalence,"pr_auc":pr,"pr_auc_minus_prevalence":pr-prevalence,
        "roc_auc":roc,"balanced_accuracy":ba,"sensitivity":sens,"specificity":spec,
        "mean_optimal_score":mo,"mean_warning_score":mw,"mean_failure_score":mf,
        "direction_preserved":bool(direction),"ordered_means_descriptive":bool(ordered),
        "external_refit":False,"external_tuning":False,
        "sha256":{"v44_6_protocol":actual,"frozen_model":sha256_file(model_path),
                  "gse236233_pseudobulk":sha256_file(pseudo_path),
                  "execution_script":sha256_file(Path(__file__).resolve())}
    },indent=2),encoding="utf-8")

    print(summary_out.read_text(encoding="utf-8"))
    print("Wrote:")
    for p in [pred_out,summary_out,manifest_out]: print(" ",p)

if __name__=="__main__":
    main()
