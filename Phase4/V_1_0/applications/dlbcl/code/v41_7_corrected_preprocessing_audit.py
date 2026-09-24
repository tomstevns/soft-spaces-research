#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v41.7 — corrected preprocessing methods audit.

Change under test:
    OLD: StandardScaler -> variance ranking -> top 256
    NEW: raw training variance -> top 256 -> StandardScaler

Uses GSE10846 only and the exact 15 frozen v41.2 CV folds.
Keeps P dimension=16 and feature K=16 unchanged.
Runs 200 matched NULL permutations per fold.

No GSE31312 read. No Aer. No QPU.
"""

from __future__ import annotations
import json, platform, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest, wilcoxon
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

VERSION = "v41.7-dlbcl-corrected-preprocessing-audit-1"
VARIANCE_POOL = 256
P_DIM = 16
FEATURE_K = 16
N_NULL_PER_FOLD = 200
NULL_BASE_SEED = 41_700_000
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 41_700_900

def load_inputs(project_dir):
    expr_path = project_dir/"data"/"processed"/"GSE10846_ABC_GCB_gene_expression.csv.gz"
    labels_path = project_dir/"data"/"processed"/"GSE10846_ABC_GCB_sample_labels.csv"
    splits_path = project_dir/"results"/"baseline"/"v41_2_cv_splits.json"
    for p in (expr_path, labels_path, splits_path):
        if not p.exists():
            raise FileNotFoundError(p)
    Xdf = pd.read_csv(expr_path, index_col=0)
    labels = pd.read_csv(labels_path, dtype=str).fillna("").set_index("sample_id").loc[Xdf.index]
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    if payload["sample_order"] != Xdf.index.tolist():
        raise RuntimeError("Sample order mismatch.")
    X = Xdf.to_numpy(dtype=np.float64)
    y = (labels["coo_label"] == "ABC").astype(int).to_numpy()
    genes = np.asarray(Xdf.columns, dtype=object)
    folds = [(np.asarray(f["train_indices"], int), np.asarray(f["test_indices"], int)) for f in payload["folds"]]
    if X.shape != (350, 22154) or len(folds) != 15:
        raise RuntimeError("Unexpected locked input dimensions.")
    if Counter(labels["coo_label"]) != Counter({"GCB":183, "ABC":167}):
        raise RuntimeError("Unexpected class counts.")
    return X, y, genes, folds, splits_path

def preprocess_corrected(Xtr_raw, Xte_raw, genes):
    raw_var = np.var(Xtr_raw, axis=0, ddof=1)
    order = np.lexsort((genes.astype(str), -raw_var))
    chosen = order[:VARIANCE_POOL]
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_raw[:, chosen])
    Xte = scaler.transform(Xte_raw[:, chosen])
    return Xtr, Xte, genes[chosen], chosen, raw_var[chosen]

def H_operator(X, y):
    Xa, Xg = X[y==1], X[y==0]
    H = (Xa.T @ Xa)/len(Xa) - (Xg.T @ Xg)/len(Xg)
    return 0.5*(H + H.T)

def score(H):
    evals, evecs = np.linalg.eigh(H)
    top = np.argsort(np.abs(evals))[::-1][:P_DIM]
    U = evecs[:, top]
    p = np.sum(U*U, axis=1)
    coupling = p*(1-p)
    ae = np.abs(evals)
    frac = float(np.sum(ae[top])/np.sum(ae)) if np.sum(ae) > 0 else 0.0
    return coupling, frac

def rank_scores(scores, genes):
    return np.lexsort((genes.astype(str), -scores))

def readout(Xtr, ytr, Xte, yte, idx):
    m = LogisticRegression(C=1.0, solver="liblinear", max_iter=5000, random_state=41700777)
    m.fit(Xtr[:, idx], ytr)
    prob = m.predict_proba(Xte[:, idx])[:,1]
    pred = m.predict(Xte[:, idx])
    return float(roc_auc_score(yte, prob)), float(balanced_accuracy_score(yte, pred))

def summarize(a):
    a = np.asarray(a, float)
    return {"mean":float(a.mean()), "sd":float(a.std(ddof=1)), "min":float(a.min()), "max":float(a.max())}

def boot_ci(a, reps, seed):
    a = np.asarray(a, float)
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    pos = 0
    while pos < reps:
        n = min(10000, reps-pos)
        idx = rng.integers(0, len(a), size=(n, len(a)))
        means[pos:pos+n] = a[idx].mean(axis=1)
        pos += n
    return {"mean":float(a.mean()), "ci95_low":float(np.quantile(means,.025)), "ci95_high":float(np.quantile(means,.975))}

def main():
    code_dir = Path(__file__).resolve().parent
    project_dir = code_dir.parent
    out_dir = project_dir/"results"/"softspaces"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_csv = out_dir/"v41_7_corrected_preprocessing_fold_summary.csv"
    null_csv = out_dir/"v41_7_corrected_preprocessing_null_metrics.csv.gz"
    genes_csv = out_dir/"v41_7_corrected_preprocessing_selected_genes.csv"
    summary_json = out_dir/"v41_7_corrected_preprocessing_summary.json"

    X, y, genes, folds, splits_path = load_inputs(project_dir)

    print("=== Soft Spaces Phase 4 v41.7 — corrected preprocessing audit ===")
    print("OLD: scale -> variance rank")
    print("NEW: RAW variance rank -> top 256 -> scale")
    print("GSE10846 only; GSE31312 is NOT read.")
    print()

    fold_rows, null_rows, gene_rows = [], [], []
    real_aucs, real_bals, real_specs = [], [], []
    null_auc_mat = np.empty((15, N_NULL_PER_FOLD))
    null_bal_mat = np.empty((15, N_NULL_PER_FOLD))
    null_spec_mat = np.empty((15, N_NULL_PER_FOLD))

    t0 = time.perf_counter()

    for fi, (tr, te) in enumerate(folds):
        fold = fi+1
        ft = time.perf_counter()
        Xtr, Xte, pool_genes, pool_idx, pool_var = preprocess_corrected(X[tr], X[te], genes)
        ytr, yte = y[tr], y[te]

        rs, rspec = score(H_operator(Xtr, ytr))
        rr = rank_scores(rs, pool_genes)
        rauc, rbal = readout(Xtr, ytr, Xte, yte, rr[:FEATURE_K])
        real_aucs.append(rauc); real_bals.append(rbal); real_specs.append(rspec)

        for pos, li in enumerate(rr[:FEATURE_K], start=1):
            gene_rows.append({
                "fold":fold, "softspaces_rank":pos, "gene":str(pool_genes[li]),
                "raw_training_variance":float(pool_var[li]),
                "coupling_score":float(rs[li]), "global_gene_column":int(pool_idx[li])
            })

        rng = np.random.default_rng(NULL_BASE_SEED + fold)
        for j in range(N_NULL_PER_FOLD):
            yn = rng.permutation(ytr)
            ns, nspec = score(H_operator(Xtr, yn))
            nr = rank_scores(ns, pool_genes)
            nauc, nbal = readout(Xtr, ytr, Xte, yte, nr[:FEATURE_K])
            null_auc_mat[fi,j] = nauc
            null_bal_mat[fi,j] = nbal
            null_spec_mat[fi,j] = nspec
            null_rows.append({
                "fold":fold, "null_rep":j+1, "null_auc":nauc,
                "null_balanced_accuracy":nbal, "null_topP_abs_fraction":nspec
            })

        mn_auc = float(null_auc_mat[fi].mean())
        mn_bal = float(null_bal_mat[fi].mean())
        mn_spec = float(null_spec_mat[fi].mean())
        p_fold = (1 + int(np.sum(null_auc_mat[fi] >= rauc))) / (N_NULL_PER_FOLD+1)

        fold_rows.append({
            "fold":fold, "real_auc":rauc, "mean_null_auc":mn_auc, "delta_auc":rauc-mn_auc,
            "empirical_p_auc_one_sided":p_fold,
            "real_balanced_accuracy":rbal, "mean_null_balanced_accuracy":mn_bal,
            "delta_balanced_accuracy":rbal-mn_bal,
            "real_topP_abs_fraction":rspec, "mean_null_topP_abs_fraction":mn_spec,
            "delta_topP_abs_fraction":rspec-mn_spec,
            "fold_runtime_seconds":time.perf_counter()-ft
        })
        print(f"fold {fold:02d}/15  REAL AUC={rauc:.4f}  NULLmean={mn_auc:.4f}  Δ={rauc-mn_auc:+.4f}  p_perm={p_fold:.4f}")

    elapsed = time.perf_counter()-t0

    fold_df = pd.DataFrame(fold_rows)
    null_df = pd.DataFrame(null_rows)
    gene_df = pd.DataFrame(gene_rows)
    fold_df.to_csv(fold_csv, index=False)
    null_df.to_csv(null_csv, index=False, compression="gzip")
    gene_df.to_csv(genes_csv, index=False)

    da = fold_df["delta_auc"].to_numpy()
    db = fold_df["delta_balanced_accuracy"].to_numpy()
    ds = fold_df["delta_topP_abs_fraction"].to_numpy()

    auc_ci = boot_ci(da, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    bal_ci = boot_ci(db, BOOTSTRAP_REPS, BOOTSTRAP_SEED+1)
    spec_ci = boot_ci(ds, BOOTSTRAP_REPS, BOOTSTRAP_SEED+2)

    pos, neg, zero = int(np.sum(da>0)), int(np.sum(da<0)), int(np.sum(da==0))
    sign_p = float(binomtest(pos, pos+neg, p=.5, alternative="greater").pvalue) if pos+neg else 1.0
    try:
        wilc_p = float(wilcoxon(da, alternative="greater", zero_method="wilcox", method="auto").pvalue)
    except ValueError:
        wilc_p = 1.0

    real_mean_auc = float(np.mean(real_aucs))
    global_null_auc = null_auc_mat.mean(axis=0)
    global_p_auc = (1 + int(np.sum(global_null_auc >= real_mean_auc))) / (N_NULL_PER_FOLD+1)

    real_mean_spec = float(np.mean(real_specs))
    global_null_spec = null_spec_mat.mean(axis=0)
    global_p_spec = (1 + int(np.sum(global_null_spec >= real_mean_spec))) / (N_NULL_PER_FOLD+1)

    freq = gene_df["gene"].value_counts().rename_axis("gene").reset_index(name="selected_in_folds")
    top_recurrent = freq.head(30).to_dict(orient="records")

    payload = {
        "version":VERSION,
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__},
        "design":{
            "dataset":"GSE10846 ABC vs GCB","n_samples":350,"n_genes":22154,"cv_folds":15,
            "cv_source":str(splits_path.relative_to(project_dir)),
            "corrected_preprocessing":"raw training variance -> top 256 -> StandardScaler",
            "variance_pool":VARIANCE_POOL,"p_dim":P_DIM,"feature_k":FEATURE_K,
            "null_permutations_per_fold":N_NULL_PER_FOLD,"external_dataset_read":False
        },
        "real_auc":summarize(real_aucs),
        "real_balanced_accuracy":summarize(real_bals),
        "real_spectral":summarize(real_specs),
        "delta_real_minus_fold_null_mean":{
            "auc":{**summarize(da),"bootstrap95":auc_ci},
            "balanced_accuracy":{**summarize(db),"bootstrap95":bal_ci},
            "spectral":{**summarize(ds),"bootstrap95":spec_ci}
        },
        "paired_diagnostics_auc":{
            "positive_folds":pos,"zero_folds":zero,"negative_folds":neg,
            "sign_test_one_sided_p":sign_p,"wilcoxon_one_sided_p":wilc_p,
            "conditional_global_permutation_p":global_p_auc
        },
        "spectral_global_permutation_p":global_p_spec,
        "top_recurrent_selected_genes":top_recurrent,
        "interpretation_limits":[
            "Internal GSE10846 methods audit only; not external validation.",
            "P=16 and K=16 retained from earlier pipeline rather than re-optimized.",
            "The GSE31312 v41.6 result has already been observed historically.",
            "Any later GSE31312 re-test is post-hoc revised-pipeline testing, not a pristine first external validation.",
            "No quantum advantage is claimed.","No Aer or QPU execution was performed."
        ],
        "runtime_seconds":float(elapsed),
        "outputs":{
            "fold_summary":str(fold_csv.relative_to(project_dir)),
            "null_metrics":str(null_csv.relative_to(project_dir)),
            "selected_genes":str(genes_csv.relative_to(project_dir))
        }
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=== v41.7 SUMMARY ===")
    print(f"REAL mean AUC:           {np.mean(real_aucs):.4f}")
    print(f"Mean fold ΔAUC:          {np.mean(da):+.4f}  bootstrap95=[{auc_ci['ci95_low']:+.4f}, {auc_ci['ci95_high']:+.4f}]")
    print(f"Fold signs (+/0/-):      {pos}/{zero}/{neg}")
    print(f"Sign diagnostic p:       {sign_p:.6g}")
    print(f"Wilcoxon diagnostic p:   {wilc_p:.6g}")
    print(f"Global perm p(AUC):      {global_p_auc:.6g}")
    print(f"Mean spectral Δ:         {np.mean(ds):+.4f}  bootstrap95=[{spec_ci['ci95_low']:+.4f}, {spec_ci['ci95_high']:+.4f}]")
    print(f"Global perm p(spectral): {global_p_spec:.6g}")
    print()
    print("Most recurrent selected genes:")
    for rec in top_recurrent[:10]:
        print(f"  {rec['gene']:<24} {rec['selected_in_folds']:2d}/15 folds")
    print()
    print(f"Runtime:        {elapsed:.1f}s = {elapsed/60:.2f} min")
    print(f"Fold summary:   {fold_csv}")
    print(f"NULL metrics:   {null_csv}")
    print(f"Selected genes: {genes_csv}")
    print(f"Summary:        {summary_json}")
    print()
    print("v41.7 COMPLETE.")
    print("GSE31312 was not read.")
    print("No Aer or QPU execution performed.")

if __name__ == "__main__":
    main()
