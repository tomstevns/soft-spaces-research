#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v41.6 — Independent external validation on GSE31312.

Purpose
-------
Train the frozen Soft Spaces biological mapping on ALL binary GSE10846 samples
and evaluate it ONCE on the independent GSE31312 GEP-defined ABC/GCB cohort.

No parameter tuning is allowed on GSE31312.

Frozen design inherited from v41.3/v41.4
----------------------------------------
Training cohort:
    GSE10846
    183 GCB + 167 ABC = 350 samples

External cohort:
    GSE31312
    237 GCB + 214 ABC = 451 samples
    COO source = GEO "gene expression profiling subgroup" only

Frozen mapping:
    variance pool = 256
    P dimension   = 16
    feature K     = 16

    H = <xx^T>_ABC - <xx^T>_GCB

    P = top-|E| 16-dimensional eigensubspace
    Q = I - P

    C_g = ||P V_g Q||_F^2 = p_g(1-p_g)

Readout:
    L2 LogisticRegression(C=1, liblinear)

External-validation discipline
------------------------------
1. Feature selection/ranking uses GSE10846 only.
2. Scaling parameters are fit on GSE10846 only.
3. The classifier is fit on GSE10846 only.
4. GSE31312 is used only for the final evaluation.
5. No threshold, K, P dimension or other parameter is tuned on GSE31312.

NULL robustness
---------------
For context only, this script also creates matched NULL rankings by permuting
GSE10846 labels. Every NULL model is trained entirely on GSE10846 and evaluated
on the same untouched GSE31312 external cohort.

IMPORTANT methodological note
-----------------------------
This script intentionally reproduces the v41.3/v41.4 preprocessing algorithm
exactly, including variance ranking after StandardScaler, so the external test
is a faithful validation of the already-observed pipeline rather than a revised
method. A later methods-cleanup version can address that design choice
separately without contaminating this frozen external test.

No Aer and no QPU execution.
"""

from __future__ import annotations

import csv
import gzip
import json
import platform
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


VERSION = "v41.6-dlbcl-independent-external-validation-1"

VARIANCE_POOL = 256
P_DIM = 16
FEATURE_K = 16

N_NULL = 200
NULL_BASE_SEED = 41_600_000


def clean(s: str) -> str:
    return " ".join(s.strip().strip('"').split())


def read_series_matrix(path: Path) -> pd.DataFrame:
    rows = []
    header = None
    in_table = False

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
                raise RuntimeError(
                    f"Bad Series Matrix row width: {len(parts)} != {len(header)}"
                )

            rows.append(parts)

    if header is None:
        raise RuntimeError("No expression table header found.")

    sample_ids = header[1:]
    probe_ids = []
    data = np.empty((len(rows), len(sample_ids)), dtype=np.float32)

    for i, parts in enumerate(rows):
        probe_ids.append(parts[0])
        vals = []
        for x in parts[1:]:
            x = x.strip()
            if x in {"", "NA", "NaN", "nan", "NULL", "null"}:
                vals.append(np.nan)
            else:
                vals.append(float(x))
        data[i, :] = vals

    return pd.DataFrame(
        data,
        index=pd.Index(probe_ids, name="probe_id"),
        columns=sample_ids,
    )


def read_probe_gene_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip", dtype=str).fillna("")
    required = {"probe_id", "gene_symbol"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Probe-gene mapping missing columns: {sorted(missing)}")
    return df[list(required)].drop_duplicates()


def aggregate_probe_to_gene(expr_probe: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    common = mapping[mapping["probe_id"].isin(expr_probe.index)].copy()
    if common.empty:
        raise RuntimeError("No probe IDs matched external expression matrix.")

    grouped = defaultdict(list)
    for probe, gene in common[["probe_id", "gene_symbol"]].itertuples(index=False):
        if gene:
            grouped[gene].append(probe)

    genes = sorted(grouped)
    out = np.empty((len(genes), expr_probe.shape[1]), dtype=np.float32)

    for i, gene in enumerate(genes):
        probes = list(dict.fromkeys(grouped[gene]))
        vals = expr_probe.loc[probes].to_numpy(dtype=np.float32)
        out[i, :] = np.nanmedian(vals, axis=0)

    return pd.DataFrame(
        out,
        index=pd.Index(genes, name="gene_symbol"),
        columns=expr_probe.columns,
    )


def class_contrast_operator(X, y):
    Xa = X[y == 1]
    Xg = X[y == 0]
    H = (Xa.T @ Xa) / len(Xa) - (Xg.T @ Xg) / len(Xg)
    return 0.5 * (H + H.T)


def mapping_score(H):
    evals, evecs = np.linalg.eigh(H)
    order = np.argsort(np.abs(evals))[::-1]
    top = order[:P_DIM]
    U = evecs[:, top]

    p_diag = np.sum(U * U, axis=1)
    coupling = p_diag * (1.0 - p_diag)

    abs_e = np.abs(evals)
    total_abs = float(np.sum(abs_e))
    top_abs_fraction = (
        float(np.sum(abs_e[top]) / total_abs)
        if total_abs > 0 else 0.0
    )

    return coupling, top_abs_fraction


def stable_rank(scores, genes):
    return np.lexsort((genes.astype(str), -scores))


def evaluate(y_true, prob, pred):
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
    }


def train_and_predict(Xtr, ytr, Xext, selected_idx):
    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        random_state=41_600_777,
    )
    model.fit(Xtr[:, selected_idx], ytr)

    prob = model.predict_proba(Xext[:, selected_idx])[:, 1]
    pred = model.predict(Xext[:, selected_idx])
    return model, prob, pred


def simple_summary(values):
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "sd": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def main() -> int:
    code_dir = Path(__file__).resolve().parent
    project_dir = code_dir.parent

    train_expr_path = (
        project_dir / "data" / "processed" /
        "GSE10846_ABC_GCB_gene_expression.csv.gz"
    )
    train_labels_path = (
        project_dir / "data" / "processed" /
        "GSE10846_ABC_GCB_sample_labels.csv"
    )
    mapping_path = (
        project_dir / "data" / "processed" /
        "GSE10846_probe_to_gene_mapping.csv.gz"
    )

    ext_matrix_path = (
        project_dir / "data" / "external" / "GSE31312" / "raw" /
        "GSE31312_series_matrix.txt.gz"
    )
    ext_labels_path = (
        project_dir / "data" / "external" / "GSE31312" / "metadata" /
        "GSE31312_ABC_GCB_GEP_only.csv"
    )

    out_dir = project_dir / "results" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = out_dir / "v41_6_external_predictions.csv"
    selected_genes_csv = out_dir / "v41_6_selected_genes.csv"
    null_csv = out_dir / "v41_6_external_null_metrics.csv"
    summary_json = out_dir / "v41_6_external_validation_summary.json"

    for p in [
        train_expr_path,
        train_labels_path,
        mapping_path,
        ext_matrix_path,
        ext_labels_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    print("=== Soft Spaces Phase 4 v41.6 — Independent external validation ===")
    print()
    print("Frozen design:")
    print(f"  variance pool = {VARIANCE_POOL}")
    print(f"  P dimension   = {P_DIM}")
    print(f"  feature K     = {FEATURE_K}")
    print("  NO tuning on GSE31312")
    print()

    t0 = time.perf_counter()

    print("[1/7] Loading GSE10846 training data...")
    Xtrain_df = pd.read_csv(train_expr_path, index_col=0)
    train_labels = pd.read_csv(train_labels_path, dtype=str).fillna("")
    train_labels = train_labels.set_index("sample_id").loc[Xtrain_df.index]

    ytrain = (train_labels["coo_label"] == "ABC").astype(int).to_numpy()
    train_counts = Counter(train_labels["coo_label"])

    if train_counts != Counter({"GCB": 183, "ABC": 167}):
        raise RuntimeError(f"Unexpected GSE10846 counts: {train_counts}")

    print(f"      shape: {Xtrain_df.shape}")
    print(f"      labels: {dict(train_counts)}")

    print("[2/7] Reading GSE31312 external expression matrix...")
    ext_probe = read_series_matrix(ext_matrix_path)
    print(f"      probe matrix: {ext_probe.shape}")

    print("[3/7] Mapping GSE31312 probes -> genes...")
    mapping = read_probe_gene_mapping(mapping_path)
    ext_gene = aggregate_probe_to_gene(ext_probe, mapping)
    print(f"      gene matrix: {ext_gene.shape}")

    print("[4/7] Loading external GEP-only labels...")
    ext_labels = pd.read_csv(ext_labels_path, dtype=str).fillna("")
    ext_labels = ext_labels.set_index("sample_id")

    ext_ids = [s for s in ext_gene.columns if s in ext_labels.index]
    ext_labels = ext_labels.loc[ext_ids]

    bad_ext = sorted(set(ext_labels["gep_coo"]) - {"ABC", "GCB"})
    if bad_ext:
        raise RuntimeError(f"Unexpected external binary labels: {bad_ext}")

    yext = (ext_labels["gep_coo"] == "ABC").astype(int).to_numpy()
    ext_counts = Counter(ext_labels["gep_coo"])

    if ext_counts != Counter({"GCB": 237, "ABC": 214}):
        raise RuntimeError(f"Unexpected GSE31312 binary counts: {ext_counts}")

    print(f"      external binary samples: {len(ext_ids)}")
    print(f"      labels: {dict(ext_counts)}")

    print("[5/7] Aligning common genes and freezing GSE10846 mapping...")
    common_genes = [
        g for g in Xtrain_df.columns
        if g in ext_gene.index
    ]

    if len(common_genes) < VARIANCE_POOL:
        raise RuntimeError(
            f"Only {len(common_genes)} common genes; need at least {VARIANCE_POOL}."
        )

    Xtr_raw = Xtrain_df[common_genes].to_numpy(dtype=np.float64)
    Xext_raw = ext_gene.loc[common_genes, ext_ids].T.to_numpy(dtype=np.float64)
    gene_names = np.asarray(common_genes, dtype=object)

    if not np.isfinite(Xtr_raw).all():
        raise RuntimeError("Training matrix has non-finite values.")
    if not np.isfinite(Xext_raw).all():
        raise RuntimeError("External matrix has non-finite values.")

    # Exact v41.3/v41.4 algorithm: StandardScaler FIRST, variance ranking SECOND.
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr_raw)
    Xext_scaled = scaler.transform(Xext_raw)

    var = np.var(Xtr_scaled, axis=0, ddof=1)
    variance_order = np.lexsort((gene_names.astype(str), -var))
    pool_idx_global = variance_order[:VARIANCE_POOL]

    Xtr_pool = Xtr_scaled[:, pool_idx_global]
    Xext_pool = Xext_scaled[:, pool_idx_global]
    pool_genes = gene_names[pool_idx_global]

    print(f"      common genes: {len(common_genes)}")
    print(f"      frozen pool: {len(pool_genes)}")

    print("[6/7] Training REAL mapping on all GSE10846 and testing once...")
    H_real = class_contrast_operator(Xtr_pool, ytrain)
    real_score, real_spectral = mapping_score(H_real)
    real_rank = stable_rank(real_score, pool_genes)
    selected_idx = real_rank[:FEATURE_K]

    model, prob, pred = train_and_predict(
        Xtr_pool, ytrain, Xext_pool, selected_idx
    )

    real_metrics = evaluate(yext, prob, pred)

    print(
        f"      EXTERNAL AUC={real_metrics['roc_auc']:.4f}  "
        f"BalAcc={real_metrics['balanced_accuracy']:.4f}  "
        f"Accuracy={real_metrics['accuracy']:.4f}"
    )

    pred_df = pd.DataFrame({
        "sample_id": ext_ids,
        "true_coo": ext_labels["gep_coo"].to_numpy(),
        "y_true_ABC": yext,
        "p_ABC": prob,
        "predicted_coo": np.where(pred == 1, "ABC", "GCB"),
        "correct": pred == yext,
    })
    pred_df.to_csv(predictions_csv, index=False)

    selected_rows = []
    for rank_pos, local_idx in enumerate(selected_idx, start=1):
        selected_rows.append({
            "rank": rank_pos,
            "gene": str(pool_genes[local_idx]),
            "coupling_score": float(real_score[local_idx]),
            "pool_index": int(local_idx),
        })
    pd.DataFrame(selected_rows).to_csv(selected_genes_csv, index=False)

    print(f"[7/7] Running {N_NULL} matched training-label NULL controls...")
    rng = np.random.default_rng(NULL_BASE_SEED)
    null_rows = []
    null_aucs = []
    null_bals = []
    null_spectral = []

    for j in range(N_NULL):
        ynull = rng.permutation(ytrain)

        if Counter(ynull.tolist()) != Counter(ytrain.tolist()):
            raise RuntimeError("NULL permutation changed class counts.")

        H_null = class_contrast_operator(Xtr_pool, ynull)
        null_score, null_spec = mapping_score(H_null)
        null_rank = stable_rank(null_score, pool_genes)

        _, nprob, npred = train_and_predict(
            Xtr_pool, ytrain, Xext_pool, null_rank[:FEATURE_K]
        )
        nm = evaluate(yext, nprob, npred)

        null_aucs.append(nm["roc_auc"])
        null_bals.append(nm["balanced_accuracy"])
        null_spectral.append(null_spec)

        null_rows.append({
            "null_rep": j + 1,
            "external_roc_auc": nm["roc_auc"],
            "external_balanced_accuracy": nm["balanced_accuracy"],
            "external_accuracy": nm["accuracy"],
            "external_f1": nm["f1"],
            "training_spectral_topP_abs_fraction": null_spec,
        })

        if (j + 1) % 25 == 0:
            print(f"      NULL {j+1:3d}/{N_NULL}")

    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(null_csv, index=False)

    empirical_p_auc = (
        1 + int(np.sum(np.asarray(null_aucs) >= real_metrics["roc_auc"]))
    ) / (N_NULL + 1)

    empirical_p_bal = (
        1 + int(np.sum(np.asarray(null_bals) >= real_metrics["balanced_accuracy"]))
    ) / (N_NULL + 1)

    empirical_p_spec = (
        1 + int(np.sum(np.asarray(null_spectral) >= real_spectral))
    ) / (N_NULL + 1)

    elapsed = time.perf_counter() - t0

    summary = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
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
            "training_samples": 350,
            "external_binary_samples": 451,
            "training_counts": dict(train_counts),
            "external_counts": dict(ext_counts),
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
        "real_external_metrics": real_metrics,
        "real_training_spectral_topP_abs_fraction": real_spectral,
        "null_external_auc": simple_summary(null_aucs),
        "null_external_balanced_accuracy": simple_summary(null_bals),
        "null_training_spectral": simple_summary(null_spectral),
        "real_minus_null_mean": {
            "roc_auc": float(real_metrics["roc_auc"] - np.mean(null_aucs)),
            "balanced_accuracy": float(
                real_metrics["balanced_accuracy"] - np.mean(null_bals)
            ),
            "spectral_topP_abs_fraction": float(
                real_spectral - np.mean(null_spectral)
            ),
        },
        "empirical_one_sided_p": {
            "external_auc": empirical_p_auc,
            "external_balanced_accuracy": empirical_p_bal,
            "training_spectral": empirical_p_spec,
        },
        "selected_genes": selected_rows,
        "methodological_note": (
            "This external validation intentionally reproduces the v41.3/v41.4 "
            "pipeline exactly, including variance ranking after StandardScaler. "
            "That design choice should be audited separately after this frozen "
            "validation and must not be changed retrospectively based on the "
            "GSE31312 result."
        ),
        "scientific_limits": [
            "The biological H is a proposed operator, not the original physical Hamiltonian.",
            "No exact +/-E symmetry is claimed.",
            "No quantum advantage is claimed.",
            "GSE31312 was not used for parameter tuning.",
            "NULL controls permute only GSE10846 training labels.",
            "No Aer or QPU execution was performed.",
        ],
        "runtime_seconds": float(elapsed),
        "outputs": {
            "external_predictions": str(predictions_csv.relative_to(project_dir)),
            "selected_genes": str(selected_genes_csv.relative_to(project_dir)),
            "null_metrics": str(null_csv.relative_to(project_dir)),
        },
    }

    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=== v41.6 EXTERNAL VALIDATION SUMMARY ===")
    print(f"REAL external AUC:        {real_metrics['roc_auc']:.4f}")
    print(f"NULL mean external AUC:   {np.mean(null_aucs):.4f}")
    print(
        f"REAL-NULL ΔAUC:           "
        f"{real_metrics['roc_auc'] - np.mean(null_aucs):+.4f}"
    )
    print(f"Empirical p(AUC):         {empirical_p_auc:.6g}")
    print()
    print(f"REAL external BalAcc:     {real_metrics['balanced_accuracy']:.4f}")
    print(f"NULL mean external BalAcc:{np.mean(null_bals):.4f}")
    print(
        f"REAL-NULL ΔBalAcc:        "
        f"{real_metrics['balanced_accuracy'] - np.mean(null_bals):+.4f}"
    )
    print(f"Empirical p(BalAcc):      {empirical_p_bal:.6g}")
    print()
    print(f"REAL training spectral:   {real_spectral:.4f}")
    print(f"NULL mean spectral:       {np.mean(null_spectral):.4f}")
    print(f"Empirical p(spectral):    {empirical_p_spec:.6g}")
    print()
    print(f"Runtime: {elapsed:.1f}s = {elapsed/60:.2f} min")
    print(f"Predictions:    {predictions_csv}")
    print(f"Selected genes: {selected_genes_csv}")
    print(f"NULL metrics:   {null_csv}")
    print(f"Summary:        {summary_json}")
    print()
    print("v41.6 COMPLETE.")
    print("No Aer or QPU execution performed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
