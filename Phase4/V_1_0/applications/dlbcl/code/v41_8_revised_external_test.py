#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v41.8 — revised-pipeline external test on GSE31312.

Status
------
This is a post-hoc revised-pipeline external test. GSE31312 was already observed
in v41.6, so this is not a pristine first external validation.

Frozen revised pipeline from v41.7
----------------------------------
- Train: GSE10846 (183 GCB, 167 ABC)
- Test:  GSE31312 (237 GCB, 214 ABC), GEO GEP labels only
- raw training variance -> top 256 genes -> StandardScaler
- P dimension = 16
- feature K = 16
- H = <xx^T>_ABC - <xx^T>_GCB
- C_g = ||P V_g Q||_F^2 = p_g(1-p_g)
- fixed L2 logistic readout
- 200 matched NULL permutations on GSE10846 labels
- no tuning on GSE31312
- no Aer, no QPU
"""

from __future__ import annotations
import csv, gzip, json, platform, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

VERSION = "v41.8-dlbcl-revised-pipeline-external-test-1"
VARIANCE_POOL = 256
P_DIM = 16
FEATURE_K = 16
N_NULL = 200
NULL_BASE_SEED = 41_800_000

def read_series_matrix(path: Path) -> pd.DataFrame:
    rows, header, in_table = [], None, False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if line == "!series_matrix_table_begin":
                in_table = True
                continue
            if line == "!series_matrix_table_end":
                break
            if not in_table:
                continue
            parts = next(csv.reader([line], delimiter="\t", quotechar='"'))
            if header is None:
                header = parts
                continue
            if len(parts) != len(header):
                raise RuntimeError("Bad Series Matrix row width.")
            rows.append(parts)

    if header is None:
        raise RuntimeError("No Series Matrix table found.")

    sample_ids = header[1:]
    probe_ids = []
    data = np.empty((len(rows), len(sample_ids)), dtype=np.float32)

    for i, parts in enumerate(rows):
        probe_ids.append(parts[0])
        vals = []
        for x in parts[1:]:
            x = x.strip()
            vals.append(np.nan if x in {"", "NA", "NaN", "nan", "NULL", "null"} else float(x))
        data[i] = vals

    return pd.DataFrame(data, index=pd.Index(probe_ids, name="probe_id"), columns=sample_ids)

def read_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip", dtype=str).fillna("")
    need = {"probe_id", "gene_symbol"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"Mapping missing columns: {sorted(need - set(df.columns))}")
    return df[["probe_id", "gene_symbol"]].drop_duplicates()

def aggregate_probe_to_gene(expr_probe: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    common = mapping[mapping["probe_id"].isin(expr_probe.index)]
    grouped = defaultdict(list)
    for probe, gene in common.itertuples(index=False):
        if gene:
            grouped[gene].append(probe)

    genes = sorted(grouped)
    out = np.empty((len(genes), expr_probe.shape[1]), dtype=np.float32)
    for i, gene in enumerate(genes):
        probes = list(dict.fromkeys(grouped[gene]))
        out[i] = np.nanmedian(expr_probe.loc[probes].to_numpy(dtype=np.float32), axis=0)

    return pd.DataFrame(out, index=pd.Index(genes, name="gene_symbol"), columns=expr_probe.columns)

def h_operator(X, y):
    Xa, Xg = X[y == 1], X[y == 0]
    H = (Xa.T @ Xa) / len(Xa) - (Xg.T @ Xg) / len(Xg)
    return 0.5 * (H + H.T)

def mapping_score(H):
    evals, evecs = np.linalg.eigh(H)
    top = np.argsort(np.abs(evals))[::-1][:P_DIM]
    U = evecs[:, top]
    p = np.sum(U * U, axis=1)
    coupling = p * (1.0 - p)
    ae = np.abs(evals)
    spectral = float(np.sum(ae[top]) / np.sum(ae)) if np.sum(ae) > 0 else 0.0
    return coupling, spectral

def stable_rank(scores, genes):
    return np.lexsort((genes.astype(str), -scores))

def fit_predict(Xtr, ytr, Xext, idx):
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=5000, random_state=41800777)
    model.fit(Xtr[:, idx], ytr)
    prob = model.predict_proba(Xext[:, idx])[:, 1]
    pred = model.predict(Xext[:, idx])
    return prob, pred

def metrics(y, prob, pred):
    return {
        "roc_auc": float(roc_auc_score(y, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
    }

def summarize(x):
    a = np.asarray(x, float)
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
    }

def main():
    code_dir = Path(__file__).resolve().parent
    project_dir = code_dir.parent

    train_expr = project_dir/"data"/"processed"/"GSE10846_ABC_GCB_gene_expression.csv.gz"
    train_labels = project_dir/"data"/"processed"/"GSE10846_ABC_GCB_sample_labels.csv"
    mapping_path = project_dir/"data"/"processed"/"GSE10846_probe_to_gene_mapping.csv.gz"
    ext_matrix = project_dir/"data"/"external"/"GSE31312"/"raw"/"GSE31312_series_matrix.txt.gz"
    ext_labels = project_dir/"data"/"external"/"GSE31312"/"metadata"/"GSE31312_ABC_GCB_GEP_only.csv"

    for p in [train_expr, train_labels, mapping_path, ext_matrix, ext_labels]:
        if not p.exists():
            raise FileNotFoundError(p)

    out_dir = project_dir/"results"/"comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir/"v41_8_revised_external_predictions.csv"
    genes_csv = out_dir/"v41_8_revised_external_selected_genes.csv"
    pool_csv = out_dir/"v41_8_revised_external_variance_pool.csv"
    null_csv = out_dir/"v41_8_revised_external_null_metrics.csv"
    summary_json = out_dir/"v41_8_revised_external_summary.json"

    print("=== Soft Spaces Phase 4 v41.8 — revised-pipeline external test ===")
    print("NOTE: post-hoc revised-pipeline test; GSE31312 was already observed in v41.6.")
    print()

    t0 = time.perf_counter()

    print("[1/8] Loading GSE10846...")
    tr_df = pd.read_csv(train_expr, index_col=0)
    tr_lab = pd.read_csv(train_labels, dtype=str).fillna("").set_index("sample_id").loc[tr_df.index]
    ytr = (tr_lab["coo_label"] == "ABC").astype(int).to_numpy()
    tr_counts = Counter(tr_lab["coo_label"])
    if tr_counts != Counter({"GCB":183, "ABC":167}):
        raise RuntimeError(f"Unexpected GSE10846 counts: {tr_counts}")

    print("[2/8] Reading GSE31312 expression...")
    ext_probe = read_series_matrix(ext_matrix)

    print("[3/8] Mapping probes -> genes...")
    ext_gene = aggregate_probe_to_gene(ext_probe, read_mapping(mapping_path))

    print("[4/8] Loading GSE31312 GEP labels...")
    ex_lab = pd.read_csv(ext_labels, dtype=str).fillna("").set_index("sample_id")
    ext_ids = [s for s in ext_gene.columns if s in ex_lab.index]
    ex_lab = ex_lab.loc[ext_ids]
    yext = (ex_lab["gep_coo"] == "ABC").astype(int).to_numpy()
    ex_counts = Counter(ex_lab["gep_coo"])
    if ex_counts != Counter({"GCB":237, "ABC":214}):
        raise RuntimeError(f"Unexpected GSE31312 counts: {ex_counts}")

    print("[5/8] Aligning genes...")
    common_genes = [g for g in tr_df.columns if g in ext_gene.index]
    if len(common_genes) < VARIANCE_POOL:
        raise RuntimeError("Too few common genes.")

    Xtr_raw = tr_df[common_genes].to_numpy(dtype=np.float64)
    Xext_raw = ext_gene.loc[common_genes, ext_ids].T.to_numpy(dtype=np.float64)
    gene_names = np.asarray(common_genes, dtype=object)

    if not np.isfinite(Xtr_raw).all() or not np.isfinite(Xext_raw).all():
        raise RuntimeError("Non-finite values present.")

    print("[6/8] Corrected preprocessing: raw variance -> top 256 -> scale...")
    raw_var = np.var(Xtr_raw, axis=0, ddof=1)
    var_order = np.lexsort((gene_names.astype(str), -raw_var))
    pool_global = var_order[:VARIANCE_POOL]
    pool_genes = gene_names[pool_global]
    pool_var = raw_var[pool_global]

    scaler = StandardScaler()
    Xtr_pool = scaler.fit_transform(Xtr_raw[:, pool_global])
    Xext_pool = scaler.transform(Xext_raw[:, pool_global])

    pd.DataFrame({
        "variance_rank": np.arange(1, VARIANCE_POOL+1),
        "gene": pool_genes.astype(str),
        "raw_training_variance": pool_var,
    }).to_csv(pool_csv, index=False)

    print("[7/8] REAL mapping and external test...")
    real_score, real_spec = mapping_score(h_operator(Xtr_pool, ytr))
    real_rank = stable_rank(real_score, pool_genes)
    selected = real_rank[:FEATURE_K]
    prob, pred = fit_predict(Xtr_pool, ytr, Xext_pool, selected)
    real_m = metrics(yext, prob, pred)

    pd.DataFrame({
        "sample_id": ext_ids,
        "true_coo": ex_lab["gep_coo"].to_numpy(),
        "y_true_ABC": yext,
        "p_ABC": prob,
        "predicted_coo": np.where(pred == 1, "ABC", "GCB"),
        "correct": pred == yext,
    }).to_csv(pred_csv, index=False)

    selected_rows = []
    for r, idx in enumerate(selected, start=1):
        selected_rows.append({
            "rank": r,
            "gene": str(pool_genes[idx]),
            "coupling_score": float(real_score[idx]),
            "raw_training_variance": float(pool_var[idx]),
            "pool_index": int(idx),
        })
    pd.DataFrame(selected_rows).to_csv(genes_csv, index=False)

    print(f"      REAL external AUC={real_m['roc_auc']:.4f}")
    print(f"      REAL external BalAcc={real_m['balanced_accuracy']:.4f}")

    print(f"[8/8] Running {N_NULL} matched NULL controls...")
    rng = np.random.default_rng(NULL_BASE_SEED)
    null_rows, null_aucs, null_bals, null_specs = [], [], [], []

    for j in range(N_NULL):
        yn = rng.permutation(ytr)
        ns, nsp = mapping_score(h_operator(Xtr_pool, yn))
        nr = stable_rank(ns, pool_genes)
        nprob, npred = fit_predict(Xtr_pool, ytr, Xext_pool, nr[:FEATURE_K])
        nm = metrics(yext, nprob, npred)

        null_aucs.append(nm["roc_auc"])
        null_bals.append(nm["balanced_accuracy"])
        null_specs.append(nsp)
        null_rows.append({
            "null_rep": j+1,
            "external_roc_auc": nm["roc_auc"],
            "external_balanced_accuracy": nm["balanced_accuracy"],
            "external_accuracy": nm["accuracy"],
            "external_f1": nm["f1"],
            "training_spectral_topP_abs_fraction": nsp,
        })

        if (j+1) % 25 == 0:
            print(f"      NULL {j+1:3d}/{N_NULL}")

    pd.DataFrame(null_rows).to_csv(null_csv, index=False)

    p_auc = (1 + int(np.sum(np.asarray(null_aucs) >= real_m["roc_auc"]))) / (N_NULL+1)
    p_bal = (1 + int(np.sum(np.asarray(null_bals) >= real_m["balanced_accuracy"]))) / (N_NULL+1)
    p_spec = (1 + int(np.sum(np.asarray(null_specs) >= real_spec))) / (N_NULL+1)

    elapsed = time.perf_counter() - t0

    summary = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "historical_status": "post-hoc revised-pipeline external test; GSE31312 was already observed in v41.6",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "design": {
            "training_cohort": "GSE10846",
            "external_cohort": "GSE31312",
            "external_label_source": "gene expression profiling subgroup",
            "training_counts": dict(tr_counts),
            "external_counts": dict(ex_counts),
            "corrected_preprocessing": "raw GSE10846 variance -> top 256 -> StandardScaler",
            "variance_pool": VARIANCE_POOL,
            "p_dim": P_DIM,
            "feature_k": FEATURE_K,
            "null_replicates": N_NULL,
            "parameter_tuning_on_external": False,
        },
        "gene_alignment": {
            "common_gene_count": len(common_genes),
            "pool_gene_count": len(pool_genes),
        },
        "real_external_metrics": real_m,
        "real_training_spectral_topP_abs_fraction": real_spec,
        "null_external_auc": summarize(null_aucs),
        "null_external_balanced_accuracy": summarize(null_bals),
        "null_training_spectral": summarize(null_specs),
        "real_minus_null_mean": {
            "roc_auc": float(real_m["roc_auc"] - np.mean(null_aucs)),
            "balanced_accuracy": float(real_m["balanced_accuracy"] - np.mean(null_bals)),
            "spectral_topP_abs_fraction": float(real_spec - np.mean(null_specs)),
        },
        "empirical_one_sided_p": {
            "external_auc": p_auc,
            "external_balanced_accuracy": p_bal,
            "training_spectral": p_spec,
        },
        "selected_genes": selected_rows,
        "interpretation_limits": [
            "Not a pristine first external validation because GSE31312 was already observed in v41.6.",
            "Only the preprocessing order was revised from v41.6, as established in v41.7.",
            "P=16 and K=16 remain inherited rather than re-optimized.",
            "No parameter was tuned on GSE31312 in this run.",
            "No quantum advantage is claimed.",
            "No Aer or QPU execution was performed.",
        ],
        "runtime_seconds": float(elapsed),
        "outputs": {
            "external_predictions": str(pred_csv.relative_to(project_dir)),
            "selected_genes": str(genes_csv.relative_to(project_dir)),
            "variance_pool": str(pool_csv.relative_to(project_dir)),
            "null_metrics": str(null_csv.relative_to(project_dir)),
        },
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=== v41.8 SUMMARY ===")
    print(f"REAL external AUC:         {real_m['roc_auc']:.4f}")
    print(f"NULL mean external AUC:    {np.mean(null_aucs):.4f}")
    print(f"REAL-NULL ΔAUC:            {real_m['roc_auc'] - np.mean(null_aucs):+.4f}")
    print(f"Empirical p(AUC):          {p_auc:.6g}")
    print(f"REAL external BalAcc:      {real_m['balanced_accuracy']:.4f}")
    print(f"NULL mean external BalAcc: {np.mean(null_bals):.4f}")
    print(f"REAL-NULL ΔBalAcc:         {real_m['balanced_accuracy'] - np.mean(null_bals):+.4f}")
    print(f"Empirical p(BalAcc):       {p_bal:.6g}")
    print(f"REAL training spectral:    {real_spec:.4f}")
    print(f"NULL mean spectral:        {np.mean(null_specs):.4f}")
    print(f"Empirical p(spectral):     {p_spec:.6g}")
    print()
    print("Selected genes:")
    for row in selected_rows:
        print(f"  {row['rank']:2d}. {row['gene']}")
    print()
    print(f"Runtime:        {elapsed:.1f}s = {elapsed/60:.2f} min")
    print(f"Predictions:    {pred_csv}")
    print(f"Selected genes: {genes_csv}")
    print(f"Variance pool:  {pool_csv}")
    print(f"NULL metrics:   {null_csv}")
    print(f"Summary:        {summary_json}")
    print()
    print("v41.8 COMPLETE.")
    print("No Aer or QPU execution performed.")

if __name__ == "__main__":
    main()
