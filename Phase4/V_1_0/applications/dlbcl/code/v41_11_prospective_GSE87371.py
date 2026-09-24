#!/usr/bin/env python3
"""Soft Spaces Phase 4 v41.11 — prospective third-cohort test on GSE87371.

Frozen BEFORE GSE87371 label parsing:
- train GSE10846
- raw-variance top 256
- P dimension 16
- feature K 16
- COHORT_Z alignment
- fixed L2 logistic readout
- 200 matched NULL permutations

Blind predictions are saved before COO metadata is parsed.
No tuning on GSE87371. No Aer. No QPU.
"""

import csv, gzip, hashlib, json, shutil, time, urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

VERSION = "v41.11-dlbcl-prospective-gse87371-1"
URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE87nnn/GSE87371/matrix/GSE87371_series_matrix.txt.gz"
VARIANCE_POOL, P_DIM, FEATURE_K, N_NULL = 256, 16, 16, 200
NULL_SEED = 41_110_000

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"      exists: {path.name} ({path.stat().st_size/1024/1024:.1f} MiB)")
        return
    print(f"      downloading: {url}")
    with urllib.request.urlopen(url, timeout=180) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"      saved: {path.name} ({path.stat().st_size/1024/1024:.1f} MiB)")

def read_expression(path):
    rows, header, in_table = [], None, False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if line == "!series_matrix_table_begin":
                in_table = True; continue
            if line == "!series_matrix_table_end":
                break
            if not in_table:
                continue
            parts = next(csv.reader([line], delimiter="\t", quotechar='"'))
            if header is None:
                header = parts; continue
            rows.append(parts)
    if header is None:
        raise RuntimeError("No expression table.")
    ids = header[1:]
    probes, data = [], np.empty((len(rows), len(ids)), np.float32)
    for i, parts in enumerate(rows):
        probes.append(parts[0])
        data[i] = [np.nan if x.strip() in {"","NA","NaN","nan","NULL","null"} else float(x) for x in parts[1:]]
    return pd.DataFrame(data, index=pd.Index(probes, name="probe_id"), columns=ids)

def read_metadata(path):
    rows, occ = [], Counter()
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if line == "!series_matrix_table_begin":
                break
            if not line.startswith("!Sample_"):
                continue
            parts = next(csv.reader([line], delimiter="\t", quotechar='"'))
            key = parts[0]; occ[key] += 1
            rows.append({"key":key, "occurrence":occ[key],
                         "values":[" ".join(v.strip().strip('"').split()) for v in parts[1:]]})
    geo = [r for r in rows if r["key"] == "!Sample_geo_accession"]
    if len(geo) != 1:
        raise RuntimeError("Sample accession row problem.")
    return rows, geo[0]["values"]

def resolve_coo(rows, ids):
    n = len(ids); cand = []
    for r in rows:
        if len(r["values"]) != n: continue
        parsed, matched = [], 0
        for v in r["values"]:
            lv = v.lower()
            if lv.startswith("coo:"):
                parsed.append(v.split(":",1)[1].strip()); matched += 1
            else:
                parsed.append("")
        if matched:
            cand.append((matched, r, parsed))
    if not cand:
        raise RuntimeError("No explicit COO metadata row found.")
    cand.sort(key=lambda x: (-x[0], x[1]["occurrence"]))
    matched, row, parsed = cand[0]
    rec = []
    for gsm, raw in zip(ids, parsed):
        u = raw.upper().strip()
        if u in {"ABC","GCB"}: lab, status = u, "BINARY"
        elif u in {"NC","UNC","UNCLASSIFIED","UNCLASSIFIABLE"}: lab, status = "Unclassified", "NONBINARY"
        elif not u: lab, status = "", "MISSING"
        else: lab, status = raw, "OTHER"
        rec.append({"sample_id":gsm,"coo_label":lab,"coo_status":status,"coo_raw":raw})
    return pd.DataFrame(rec).set_index("sample_id"), f'{row["key"]}#{row["occurrence"]}', matched

def mapping(path):
    df = pd.read_csv(path, compression="gzip", dtype=str).fillna("")
    return df[["probe_id","gene_symbol"]].drop_duplicates()

def aggregate(expr, mp):
    mp = mp[mp["probe_id"].isin(expr.index)]
    groups = defaultdict(list)
    for p,g in mp.itertuples(index=False):
        if g: groups[g].append(p)
    genes = sorted(groups)
    out = np.empty((len(genes), expr.shape[1]), np.float32)
    for i,g in enumerate(genes):
        probes = list(dict.fromkeys(groups[g]))
        out[i] = np.nanmedian(expr.loc[probes].to_numpy(np.float32), axis=0)
    return pd.DataFrame(out, index=genes, columns=expr.columns)

def H(X,y):
    A,G = X[y==1], X[y==0]
    h = (A.T@A)/len(A) - (G.T@G)/len(G)
    return .5*(h+h.T)

def ss_score(h):
    e,U = np.linalg.eigh(h)
    top = np.argsort(np.abs(e))[::-1][:P_DIM]
    V = U[:,top]; p = np.sum(V*V, axis=1)
    coupling = p*(1-p)
    ae = np.abs(e)
    spectral = float(ae[top].sum()/ae.sum()) if ae.sum() else 0.0
    return coupling, spectral

def rank_scores(s, genes):
    return np.lexsort((genes.astype(str), -s))

def fit_model(X,y,idx):
    m = LogisticRegression(C=1.0, solver="liblinear", max_iter=5000, random_state=41110777)
    m.fit(X[:,idx], y)
    return m

def evaluate(y,p):
    pred = (p>=.5).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"roc_auc":float(roc_auc_score(y,p)),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "accuracy":float(accuracy_score(y,pred)),
            "f1":float(f1_score(y,pred)),
            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
            "predicted_ABC":int((pred==1).sum()),"predicted_GCB":int((pred==0).sum())}

def summ(x):
    a=np.asarray(x,float)
    return {"mean":float(a.mean()),"sd":float(a.std(ddof=1)),"min":float(a.min()),"max":float(a.max())}

def main():
    code_dir = Path(__file__).resolve().parent
    project = code_dir.parent
    train_expr = project/"data"/"processed"/"GSE10846_ABC_GCB_gene_expression.csv.gz"
    train_labels = project/"data"/"processed"/"GSE10846_ABC_GCB_sample_labels.csv"
    map_path = project/"data"/"processed"/"GSE10846_probe_to_gene_mapping.csv.gz"
    raw = project/"data"/"external"/"GSE87371"/"raw"/"GSE87371_series_matrix.txt.gz"
    meta_dir = project/"data"/"external"/"GSE87371"/"metadata"
    out = project/"results"/"prospective"
    meta_dir.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)

    blind_csv = out/"v41_11_GSE87371_BLIND_predictions.csv"
    labels_csv = meta_dir/"GSE87371_COO_resolved.csv"
    scored_csv = out/"v41_11_GSE87371_scored_predictions.csv"
    selected_csv = out/"v41_11_GSE87371_selected_genes.csv"
    null_csv = out/"v41_11_GSE87371_null_metrics.csv"
    summary_json = out/"v41_11_GSE87371_summary.json"

    print("=== v41.11 — prospective GSE87371 test ===")
    print("Frozen: COHORT_Z, pool=256, P=16, K=16, NULL=200")
    print("Blind predictions are saved before COO labels are parsed.\n")
    t0=time.perf_counter()

    print("[1/8] Downloading new cohort...")
    download(URL, raw); matrix_sha=sha256(raw)

    print("[2/8] Reading expression ONLY (labels hidden)...")
    ext_probe=read_expression(raw)
    print("      probe matrix:", ext_probe.shape)

    print("[3/8] Loading GSE10846 and mapping probes -> genes...")
    tr=pd.read_csv(train_expr,index_col=0)
    lab=pd.read_csv(train_labels,dtype=str).fillna("").set_index("sample_id").loc[tr.index]
    ytr=(lab["coo_label"]=="ABC").astype(int).to_numpy()
    ext_gene=aggregate(ext_probe,mapping(map_path))
    common=[g for g in tr.columns if g in ext_gene.index]
    Xtr=tr[common].to_numpy(float)
    Xex=ext_gene.loc[common,ext_probe.columns].T.to_numpy(float)
    genes=np.asarray(common,dtype=object)

    print("[4/8] Applying frozen corrected pipeline + COHORT_Z...")
    rv=np.var(Xtr,axis=0,ddof=1)
    order=np.lexsort((genes.astype(str),-rv))
    pool=order[:VARIANCE_POOL]
    pool_genes=genes[pool]
    Xtr_raw=Xtr[:,pool]; Xex_raw=Xex[:,pool]
    Xtr_z=StandardScaler().fit_transform(Xtr_raw)
    Xex_z=StandardScaler().fit_transform(Xex_raw)
    score,spec=ss_score(H(Xtr_z,ytr))
    rr=rank_scores(score,pool_genes); selected=rr[:FEATURE_K]
    model=fit_model(Xtr_z,ytr,selected)
    blind_prob=model.predict_proba(Xex_z[:,selected])[:,1]

    sel_rows=[{"rank":i+1,"gene":str(pool_genes[j]),"coupling_score":float(score[j]),
               "raw_training_variance":float(rv[pool[j]])} for i,j in enumerate(selected)]
    pd.DataFrame(sel_rows).to_csv(selected_csv,index=False)

    print("[5/8] Saving BLIND predictions before label reveal...")
    pd.DataFrame({"sample_id":ext_probe.columns,"p_ABC":blind_prob,
                  "predicted_coo_at_0_5":np.where(blind_prob>=.5,"ABC","GCB")}).to_csv(blind_csv,index=False)
    blind_sha=sha256(blind_csv)
    print("      blind SHA256:", blind_sha)

    print("[6/8] NOW parsing GSE87371 COO metadata...")
    rows, ids=read_metadata(raw)
    if ids != list(ext_probe.columns): raise RuntimeError("Metadata/expression order mismatch.")
    coo, source, matched=resolve_coo(rows,ids)
    coo.to_csv(labels_csv)
    print("      COO source:", source)
    print("      counts:", dict(Counter(coo["coo_label"])))

    binary=[s for s in ids if coo.loc[s,"coo_label"] in {"ABC","GCB"}]
    pos={s:i for i,s in enumerate(ids)}
    bp=np.array([pos[s] for s in binary],int)
    yext=np.array([1 if coo.loc[s,"coo_label"]=="ABC" else 0 for s in binary],int)
    p_real=blind_prob[bp]
    rm=evaluate(yext,p_real)
    pd.DataFrame({"sample_id":binary,"true_coo":[coo.loc[s,"coo_label"] for s in binary],
                  "y_true_ABC":yext,"p_ABC":p_real,
                  "predicted_coo":np.where(p_real>=.5,"ABC","GCB")}).to_csv(scored_csv,index=False)

    print(f"[7/8] Running {N_NULL} matched NULL controls...")
    rng=np.random.default_rng(NULL_SEED)
    na,nb,ns,nrows=[],[],[],[]
    for j in range(N_NULL):
        yn=rng.permutation(ytr)
        sc,sp=ss_score(H(Xtr_z,yn))
        nr=rank_scores(sc,pool_genes)[:FEATURE_K]
        nm=fit_model(Xtr_z,ytr,nr)
        pp=nm.predict_proba(Xex_z[:,nr])[:,1][bp]
        mm=evaluate(yext,pp)
        na.append(mm["roc_auc"]); nb.append(mm["balanced_accuracy"]); ns.append(sp)
        nrows.append({"null_rep":j+1,"external_roc_auc":mm["roc_auc"],
                      "external_balanced_accuracy":mm["balanced_accuracy"],
                      "external_accuracy":mm["accuracy"],"external_f1":mm["f1"],
                      "training_spectral_topP_abs_fraction":sp})
        if (j+1)%25==0: print(f"      NULL {j+1:3d}/{N_NULL}")
    pd.DataFrame(nrows).to_csv(null_csv,index=False)

    p_auc=(1+int(np.sum(np.asarray(na)>=rm["roc_auc"])))/(N_NULL+1)
    p_bal=(1+int(np.sum(np.asarray(nb)>=rm["balanced_accuracy"])))/(N_NULL+1)
    p_spec=(1+int(np.sum(np.asarray(ns)>=spec)))/(N_NULL+1)
    elapsed=time.perf_counter()-t0

    payload={
      "version":VERSION,"created_utc":datetime.now(timezone.utc).isoformat(),
      "scientific_status":"first new external cohort tested after freezing COHORT_Z in v41.10",
      "environment":{"python":platform.python_version(),"numpy":np.__version__,
                     "pandas":pd.__version__,"scikit_learn":sklearn.__version__},
      "dataset":{"accession":"GSE87371","series_url":URL,"series_matrix_sha256":matrix_sha,
                 "series_samples":int(ext_probe.shape[1]),"binary_evaluation_samples":len(binary),
                 "binary_counts":{"GCB":int((yext==0).sum()),"ABC":int((yext==1).sum())},
                 "coo_metadata_source":source},
      "frozen_before_label_reveal":{"training_cohort":"GSE10846","alignment":"COHORT_Z",
                 "alignment_uses_external_labels":False,"variance_pool":256,"p_dim":16,
                 "feature_k":16,"null_replicates":200},
      "blind_prediction_audit":{"predictions_saved_before_label_parse":True,
                 "blind_predictions_sha256":blind_sha,
                 "blind_predictions_file":str(blind_csv.relative_to(project))},
      "real_external_metrics":rm,
      "real_training_spectral_topP_abs_fraction":spec,
      "null_external_auc":summ(na),"null_external_balanced_accuracy":summ(nb),
      "null_training_spectral":summ(ns),
      "real_minus_null_mean":{"roc_auc":float(rm["roc_auc"]-np.mean(na)),
                              "balanced_accuracy":float(rm["balanced_accuracy"]-np.mean(nb)),
                              "spectral_topP_abs_fraction":float(spec-np.mean(ns))},
      "empirical_one_sided_p":{"external_auc":p_auc,"external_balanced_accuracy":p_bal,"training_spectral":p_spec},
      "selected_genes":sel_rows,
      "interpretation_limits":["GSE87371 is a new external cohort but has finite sample size.",
        "Only explicit ABC/GCB COO labels are scored.","COHORT_Z was frozen before performance inspection.",
        "External z-scoring uses expression values but no COO labels.","No parameter is tuned on GSE87371.",
        "No quantum advantage is claimed.","No Aer or QPU execution was performed."],
      "runtime_seconds":elapsed,
      "outputs":{"blind_predictions":str(blind_csv.relative_to(project)),
                 "resolved_labels":str(labels_csv.relative_to(project)),
                 "scored_predictions":str(scored_csv.relative_to(project)),
                 "selected_genes":str(selected_csv.relative_to(project)),
                 "null_metrics":str(null_csv.relative_to(project))}
    }
    summary_json.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")

    print("\n=== v41.11 PROSPECTIVE RESULT ===")
    print(f"Total / binary samples:       {ext_probe.shape[1]} / {len(binary)}")
    print(f"GCB / ABC:                    {(yext==0).sum()} / {(yext==1).sum()}")
    print(f"REAL external AUC:            {rm['roc_auc']:.4f}")
    print(f"NULL mean external AUC:       {np.mean(na):.4f}")
    print(f"REAL-NULL ΔAUC:               {rm['roc_auc']-np.mean(na):+.4f}")
    print(f"Empirical p(AUC):             {p_auc:.6g}")
    print(f"REAL external BalAcc:         {rm['balanced_accuracy']:.4f}")
    print(f"NULL mean external BalAcc:    {np.mean(nb):.4f}")
    print(f"REAL-NULL ΔBalAcc:            {rm['balanced_accuracy']-np.mean(nb):+.4f}")
    print(f"Empirical p(BalAcc):          {p_bal:.6g}")
    print(f"Empirical p(spectral):        {p_spec:.6g}")
    print(f"Blind SHA256:                 {blind_sha}")
    print(f"Runtime: {elapsed:.1f}s = {elapsed/60:.2f} min")
    print(f"Summary: {summary_json}")
    print("\nv41.11 COMPLETE.\nNo Aer or QPU execution performed.")

if __name__ == "__main__":
    main()
