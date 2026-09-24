#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v41.4 — DLBCL robustness and permutation statistics.

Purpose
-------
Stress-test the v41.3 biological Soft Spaces mapping at the frozen primary
feature budget K=16.

This is NOT independent external validation. K=16 was selected after inspecting
v41.3 on the same GSE10846 dataset, so v41.4 tests robustness of that mapping
against many matched NULL permutations; it does not erase the post-selection
history.

Frozen from v41.3
-----------------
- Dataset: GSE10846 ABC vs GCB
- 350 samples, 22,154 genes
- EXACT same 15 CV folds frozen in v41.2
- Unsupervised training-fold variance pool = 256 genes
- Biological operator:
      H = <xx^T>_ABC - <xx^T>_GCB
- P dimension = 16, chosen by largest |E|
- Q = I - P
- Gene coupling score:
      C_g = ||P V_g Q||_F^2 = p_g(1-p_g)
- Primary feature budget K = 16
- Fixed readout = L2 LogisticRegression(C=1)

NULL design
-----------
For each frozen CV fold:
  - keep expression data fixed
  - keep the same 256-gene variance pool
  - permute training labels while preserving class counts exactly
  - rebuild H_NULL, P_NULL/Q_NULL and gene ranking
  - fit the same K=16 logistic readout
  - repeat N_NULL_PER_FOLD times

Primary robustness outputs
--------------------------
1. REAL AUC versus the per-fold NULL AUC distribution.
2. Per-fold delta:
       REAL AUC - mean(NULL AUC)
3. Bootstrap 95% CI over the 15 fold deltas.
4. Sign test and Wilcoxon signed-rank diagnostic over fold deltas.
5. Conditional global permutation diagnostic:
       compare mean REAL AUC to the distribution of mean NULL AUC,
       formed by matching NULL replicate j across all 15 folds.
6. Spectral concentration REAL versus NULL.

Caution
-------
Repeated CV folds overlap and therefore are not statistically independent.
The sign/Wilcoxon/bootstrap values are descriptive paired robustness diagnostics,
not a substitute for external validation.

No Aer or QPU execution is performed.
"""

from __future__ import annotations

import json
import platform
import time
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


VERSION = "v41.4-dlbcl-softspaces-robustness-1"

VARIANCE_POOL = 256
P_DIM = 16
PRIMARY_K = 16

# 200 matched NULL permutations/fold -> 3,000 NULL mappings total.
N_NULL_PER_FOLD = 200
NULL_BASE_SEED = 41_400_000

BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 41_400_900


def load_inputs(project_dir: Path):
    expr_path = project_dir / "data" / "processed" / "GSE10846_ABC_GCB_gene_expression.csv.gz"
    labels_path = project_dir / "data" / "processed" / "GSE10846_ABC_GCB_sample_labels.csv"
    splits_path = project_dir / "results" / "baseline" / "v41_2_cv_splits.json"

    for p in (expr_path, labels_path, splits_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    Xdf = pd.read_csv(expr_path, index_col=0)
    labels = pd.read_csv(labels_path, dtype=str).fillna("").set_index("sample_id")
    split_payload = json.loads(splits_path.read_text(encoding="utf-8"))

    labels = labels.loc[Xdf.index]
    y = (labels["coo_label"] == "ABC").astype(int).to_numpy()
    X = Xdf.to_numpy(dtype=np.float64)
    genes = np.asarray(Xdf.columns, dtype=object)

    if X.shape != (350, 22154):
        raise RuntimeError(f"Unexpected matrix shape: {X.shape}")

    if Counter(labels["coo_label"]) != Counter({"GCB": 183, "ABC": 167}):
        raise RuntimeError("Locked class counts do not match v41.2.")

    if split_payload["sample_order"] != Xdf.index.tolist():
        raise RuntimeError("v41.2 sample order does not match expression matrix.")

    folds = [
        (
            np.asarray(f["train_indices"], dtype=int),
            np.asarray(f["test_indices"], dtype=int),
        )
        for f in split_payload["folds"]
    ]

    if len(folds) != 15:
        raise RuntimeError(f"Expected 15 frozen folds, found {len(folds)}.")

    return Xdf, X, y, genes, folds, splits_path


def fit_train_scaler_and_variance_pool(X_train, X_test, genes):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    var = np.var(Xtr, axis=0, ddof=1)
    order = np.lexsort((genes.astype(str), -var))
    chosen = order[:VARIANCE_POOL]

    return Xtr[:, chosen], Xte[:, chosen], genes[chosen], chosen


def class_contrast_operator(X, y):
    Xa = X[y == 1]
    Xg = X[y == 0]

    Ha = (Xa.T @ Xa) / len(Xa)
    Hg = (Xg.T @ Xg) / len(Xg)
    H = Ha - Hg
    return 0.5 * (H + H.T)


def mapping_score(H):
    evals, evecs = np.linalg.eigh(H)

    order = np.argsort(np.abs(evals))[::-1]
    top = order[:P_DIM]
    U = evecs[:, top]

    p_diag = np.sum(U * U, axis=1)
    coupling = p_diag * (1.0 - p_diag)

    abs_e = np.abs(evals)
    total_abs = np.sum(abs_e)
    top_fraction = (
        float(np.sum(abs_e[top]) / total_abs)
        if total_abs > 0
        else 0.0
    )

    return coupling, top_fraction


def stable_rank(scores, genes):
    return np.lexsort((genes.astype(str), -scores))


def logistic_readout(Xtr, ytr, Xte, yte, idx):
    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        random_state=41_400_777,
    )
    model.fit(Xtr[:, idx], ytr)

    prob = model.predict_proba(Xte[:, idx])[:, 1]
    pred = model.predict(Xte[:, idx])

    return (
        float(roc_auc_score(yte, prob)),
        float(balanced_accuracy_score(yte, pred)),
    )


def bootstrap_mean_ci(values, reps, seed):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    # Chunked to keep memory tiny.
    chunk = 10_000
    means = np.empty(reps, dtype=float)
    pos = 0

    while pos < reps:
        n = min(chunk, reps - pos)
        idx = rng.integers(0, len(a), size=(n, len(a)))
        means[pos:pos+n] = np.mean(a[idx], axis=1)
        pos += n

    return {
        "mean": float(np.mean(a)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def simple_summary(values):
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "sd": float(np.std(a, ddof=1)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def main() -> int:
    code_dir = Path(__file__).resolve().parent
    project_dir = code_dir.parent

    out_dir = project_dir / "results" / "softspaces"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_csv = out_dir / "v41_4_robustness_fold_summary.csv"
    null_csv = out_dir / "v41_4_null_permutation_metrics.csv.gz"
    summary_json = out_dir / "v41_4_robustness_summary.json"

    Xdf, X, y, genes, folds, splits_path = load_inputs(project_dir)

    print("=== Soft Spaces Phase 4 v41.4 — DLBCL robustness ===")
    print(f"Samples: {X.shape[0]}")
    print(f"Genes: {X.shape[1]}")
    print(f"Frozen CV folds: {len(folds)}")
    print(f"Primary K: {PRIMARY_K}")
    print(f"NULL permutations/fold: {N_NULL_PER_FOLD}")
    print(f"Total NULL mappings: {len(folds) * N_NULL_PER_FOLD}")
    print()
    print("NOTE: robustness test on same dataset; not independent validation.")
    print()

    fold_rows = []
    null_rows = []

    # Matrix [fold, null replicate] enables matched global permutation diagnostic.
    null_auc_matrix = np.empty((len(folds), N_NULL_PER_FOLD), dtype=float)
    null_bal_matrix = np.empty((len(folds), N_NULL_PER_FOLD), dtype=float)
    null_spec_matrix = np.empty((len(folds), N_NULL_PER_FOLD), dtype=float)

    real_aucs = []
    real_bals = []
    real_specs = []

    t0 = time.perf_counter()

    for f_idx, (train_idx, test_idx) in enumerate(folds):
        fold_no = f_idx + 1
        ft0 = time.perf_counter()

        Xtr_raw = X[train_idx]
        Xte_raw = X[test_idx]
        ytr = y[train_idx]
        yte = y[test_idx]

        Xtr, Xte, pool_genes, _ = fit_train_scaler_and_variance_pool(
            Xtr_raw, Xte_raw, genes
        )

        # REAL
        H_real = class_contrast_operator(Xtr, ytr)
        real_score, real_spec = mapping_score(H_real)
        real_rank = stable_rank(real_score, pool_genes)
        real_auc, real_bal = logistic_readout(
            Xtr, ytr, Xte, yte, real_rank[:PRIMARY_K]
        )

        real_aucs.append(real_auc)
        real_bals.append(real_bal)
        real_specs.append(real_spec)

        rng = np.random.default_rng(NULL_BASE_SEED + fold_no)

        for j in range(N_NULL_PER_FOLD):
            y_null = rng.permutation(ytr)

            if Counter(y_null.tolist()) != Counter(ytr.tolist()):
                raise RuntimeError("NULL permutation did not preserve class counts.")

            H_null = class_contrast_operator(Xtr, y_null)
            null_score, null_spec = mapping_score(H_null)
            null_rank = stable_rank(null_score, pool_genes)

            null_auc, null_bal = logistic_readout(
                Xtr, ytr, Xte, yte, null_rank[:PRIMARY_K]
            )

            null_auc_matrix[f_idx, j] = null_auc
            null_bal_matrix[f_idx, j] = null_bal
            null_spec_matrix[f_idx, j] = null_spec

            null_rows.append({
                "fold": fold_no,
                "null_rep": j + 1,
                "null_auc": null_auc,
                "null_balanced_accuracy": null_bal,
                "null_topP_abs_fraction": null_spec,
            })

        fold_null_auc = null_auc_matrix[f_idx]
        fold_null_bal = null_bal_matrix[f_idx]
        fold_null_spec = null_spec_matrix[f_idx]

        mean_null_auc = float(np.mean(fold_null_auc))
        mean_null_bal = float(np.mean(fold_null_bal))
        mean_null_spec = float(np.mean(fold_null_spec))

        # One-sided empirical fold-level permutation diagnostics.
        p_auc = (
            1 + int(np.sum(fold_null_auc >= real_auc))
        ) / (N_NULL_PER_FOLD + 1)

        p_spec = (
            1 + int(np.sum(fold_null_spec >= real_spec))
        ) / (N_NULL_PER_FOLD + 1)

        fold_seconds = time.perf_counter() - ft0

        fold_rows.append({
            "fold": fold_no,
            "real_auc": real_auc,
            "mean_null_auc": mean_null_auc,
            "delta_auc": real_auc - mean_null_auc,
            "empirical_p_auc_one_sided": p_auc,
            "real_balanced_accuracy": real_bal,
            "mean_null_balanced_accuracy": mean_null_bal,
            "delta_balanced_accuracy": real_bal - mean_null_bal,
            "real_topP_abs_fraction": real_spec,
            "mean_null_topP_abs_fraction": mean_null_spec,
            "delta_topP_abs_fraction": real_spec - mean_null_spec,
            "empirical_p_spectral_one_sided": p_spec,
            "fold_runtime_seconds": fold_seconds,
        })

        print(
            f"fold {fold_no:02d}/15  "
            f"REAL AUC={real_auc:.4f}  "
            f"NULLmean={mean_null_auc:.4f}  "
            f"Δ={real_auc-mean_null_auc:+.4f}  "
            f"p_perm={p_auc:.4f}  "
            f"{fold_seconds:.1f}s"
        )

    total_seconds = time.perf_counter() - t0

    fold_df = pd.DataFrame(fold_rows)
    null_df = pd.DataFrame(null_rows)

    fold_df.to_csv(fold_csv, index=False)
    null_df.to_csv(null_csv, index=False, compression="gzip")

    delta_auc = fold_df["delta_auc"].to_numpy()
    delta_bal = fold_df["delta_balanced_accuracy"].to_numpy()
    delta_spec = fold_df["delta_topP_abs_fraction"].to_numpy()

    auc_boot = bootstrap_mean_ci(delta_auc, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    bal_boot = bootstrap_mean_ci(delta_bal, BOOTSTRAP_REPS, BOOTSTRAP_SEED + 1)
    spec_boot = bootstrap_mean_ci(delta_spec, BOOTSTRAP_REPS, BOOTSTRAP_SEED + 2)

    # Paired diagnostic tests across the 15 overlapping CV folds.
    n_pos_auc = int(np.sum(delta_auc > 0))
    n_neg_auc = int(np.sum(delta_auc < 0))
    n_nonzero_auc = n_pos_auc + n_neg_auc

    sign_p_auc = (
        float(binomtest(n_pos_auc, n_nonzero_auc, p=0.5, alternative="greater").pvalue)
        if n_nonzero_auc > 0
        else 1.0
    )

    try:
        wilcoxon_p_auc = float(
            wilcoxon(
                delta_auc,
                alternative="greater",
                zero_method="wilcox",
                correction=False,
                method="auto",
            ).pvalue
        )
    except ValueError:
        wilcoxon_p_auc = 1.0

    # Conditional global permutation diagnostic.
    observed_mean_real_auc = float(np.mean(real_aucs))
    null_global_mean_auc = np.mean(null_auc_matrix, axis=0)

    global_perm_p_auc = (
        1 + int(np.sum(null_global_mean_auc >= observed_mean_real_auc))
    ) / (N_NULL_PER_FOLD + 1)

    observed_mean_real_spec = float(np.mean(real_specs))
    null_global_mean_spec = np.mean(null_spec_matrix, axis=0)
    global_perm_p_spec = (
        1 + int(np.sum(null_global_mean_spec >= observed_mean_real_spec))
    ) / (N_NULL_PER_FOLD + 1)

    payload = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "frozen_design": {
            "dataset": "GSE10846 ABC vs GCB",
            "n_samples": 350,
            "n_genes": 22154,
            "cv_folds": len(folds),
            "cv_source": str(splits_path.relative_to(project_dir)),
            "variance_pool": VARIANCE_POOL,
            "p_dim": P_DIM,
            "primary_feature_budget_k": PRIMARY_K,
            "null_permutations_per_fold": N_NULL_PER_FOLD,
            "bootstrap_reps": BOOTSTRAP_REPS,
        },
        "primary_auc": {
            "real_auc_across_folds": simple_summary(real_aucs),
            "null_auc_all_permutations": simple_summary(null_df["null_auc"]),
            "delta_real_minus_fold_null_mean": {
                **simple_summary(delta_auc),
                "bootstrap95": auc_boot,
            },
            "positive_delta_folds": n_pos_auc,
            "negative_delta_folds": n_neg_auc,
            "zero_delta_folds": int(np.sum(delta_auc == 0)),
            "sign_test_one_sided_p_diagnostic": sign_p_auc,
            "wilcoxon_one_sided_p_diagnostic": wilcoxon_p_auc,
            "conditional_global_permutation": {
                "observed_mean_real_auc": observed_mean_real_auc,
                "null_mean_auc_distribution": simple_summary(null_global_mean_auc),
                "empirical_one_sided_p": global_perm_p_auc,
            },
        },
        "balanced_accuracy": {
            "real": simple_summary(real_bals),
            "delta_real_minus_fold_null_mean": {
                **simple_summary(delta_bal),
                "bootstrap95": bal_boot,
            },
        },
        "spectral_concentration": {
            "real_topP_abs_fraction": simple_summary(real_specs),
            "delta_real_minus_fold_null_mean": {
                **simple_summary(delta_spec),
                "bootstrap95": spec_boot,
            },
            "conditional_global_permutation": {
                "observed_mean_real": observed_mean_real_spec,
                "null_mean_distribution": simple_summary(null_global_mean_spec),
                "empirical_one_sided_p": global_perm_p_spec,
            },
        },
        "interpretation_limits": [
            "K=16 was selected after v41.3 on this same dataset.",
            "Repeated CV folds overlap and are not independent observations.",
            "Sign/Wilcoxon/bootstrap values are paired robustness diagnostics, not definitive inferential proof.",
            "External validation on a second cohort is required for independent confirmation.",
            "The biological H is a proposed operator, not the original physical Hamiltonian.",
            "No quantum advantage is claimed.",
            "No Aer or QPU execution was performed.",
        ],
        "total_runtime_seconds": float(total_seconds),
        "outputs": {
            "fold_summary": str(fold_csv.relative_to(project_dir)),
            "null_permutations": str(null_csv.relative_to(project_dir)),
        },
    }

    summary_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=== v41.4 SUMMARY ===")
    print(
        f"REAL mean AUC: {np.mean(real_aucs):.4f}"
    )
    print(
        f"NULL mean AUC (all permutations): "
        f"{np.mean(null_df['null_auc']):.4f}"
    )
    print(
        f"Mean fold ΔAUC: {np.mean(delta_auc):+.4f}  "
        f"bootstrap95=[{auc_boot['ci95_low']:+.4f}, {auc_boot['ci95_high']:+.4f}]"
    )
    print(
        f"Fold signs (+/0/-): "
        f"{n_pos_auc}/{int(np.sum(delta_auc == 0))}/{n_neg_auc}"
    )
    print(f"Sign diagnostic p:     {sign_p_auc:.6g}")
    print(f"Wilcoxon diagnostic p: {wilcoxon_p_auc:.6g}")
    print(
        f"Global conditional permutation p(AUC): "
        f"{global_perm_p_auc:.6g}"
    )
    print(
        f"Mean spectral Δ: {np.mean(delta_spec):+.4f}  "
        f"bootstrap95=[{spec_boot['ci95_low']:+.4f}, {spec_boot['ci95_high']:+.4f}]"
    )
    print(
        f"Global conditional permutation p(spectral): "
        f"{global_perm_p_spec:.6g}"
    )
    print()
    print(f"Total runtime: {total_seconds:.1f}s = {total_seconds/60:.2f} min")
    print(f"Fold summary: {fold_csv}")
    print(f"NULL metrics: {null_csv}")
    print(f"Summary:      {summary_json}")
    print()
    print("v41.4 COMPLETE.")
    print("No Aer or QPU execution performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
