#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v41.3 — DLBCL biological mapping probe.

Goal
----
Test a concrete, auditable mapping from DLBCL gene-expression data into the
Phase-3 Soft Spaces language without pretending that ordinary ML is itself
Soft Spaces.

This is an EXPLORATORY BIOLOGICAL MAPPING PROBE, not yet the final Soft Spaces
benchmark.

Frozen data
-----------
- GSE10846 ABC vs GCB: 350 samples, 22,154 genes
- Reuse EXACTLY the 15 CV folds frozen by v41.2

Mapping used in each training fold
----------------------------------
1. Standardize genes using training data only.
2. Reduce to the 256 most variable genes using training data only.
   This step is UNSUPERVISED.
3. Construct a symmetric class-contrast operator

       H = <x x^T>_ABC - <x x^T>_GCB

   from training samples only.

4. Spectrally decompose H.
5. Define P as the projector onto the K=16 eigenvectors with largest |E|.
   Q = I - P.
6. For each gene basis direction e_g, define the coordinate perturbation
   V_g = |g><g| and score its P-Q coupling

       C_g = || P V_g Q ||_F^2

   For rank-1 coordinate perturbations this equals

       C_g = p_g (1 - p_g),  p_g = <g|P|g>.

   This is a direct P/Q coupling quantity, not a univariate p-value.

REAL / NULL
-----------
REAL:
    H is built from the true ABC/GCB training labels.

NULL:
    H is built from a deterministic permutation of the training labels,
    preserving the exact class counts.

The unsupervised variance-screened gene set is identical for REAL and NULL.

Evaluation
----------
This script asks two questions:

A. Does REAL produce a different / stronger Soft Spaces structure than NULL?
   We report spectral concentration, P-Q coupling statistics, and REAL-vs-NULL
   top-gene overlap.

B. Is the REAL Soft Spaces ranking useful under a very small feature budget?
   For budgets 8, 16, 32, 64:
      - rank genes by C_g using training data only
      - fit the SAME L2 logistic-regression readout
      - evaluate on the frozen v41.2 test fold
      - compare REAL ranking vs NULL ranking

The logistic model is only a fixed readout for assessing feature ranking.
The claimed Soft Spaces object is the H -> P/Q -> coupling ranking, not the
logistic classifier.

Important scientific limits
---------------------------
- H above is a proposed biological operator, not the original physical
  Hamiltonian from Phase 3.
- No exact +/-E symmetry is assumed or claimed.
- No quantum advantage is claimed.
- No Aer or QPU execution is performed.
- A positive result is evidence that this mapping is worth pursuing, not proof
  that the Phase-3 physical mechanism has been transferred to biology.

Run
---
From applications\\dlbcl\\code:

    python -X utf8 -u .\\v41_3_softspaces_mapping_probe.py
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
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


VERSION = "v41.3-dlbcl-softspaces-mapping-probe-1"

VARIANCE_POOL = 256
P_DIM = 16
FEATURE_BUDGETS = [8, 16, 32, 64]
NULL_BASE_SEED = 41_300_000


def load_inputs(project_dir: Path):
    expr_path = (
        project_dir / "data" / "processed" /
        "GSE10846_ABC_GCB_gene_expression.csv.gz"
    )
    labels_path = (
        project_dir / "data" / "processed" /
        "GSE10846_ABC_GCB_sample_labels.csv"
    )
    splits_path = (
        project_dir / "results" / "baseline" /
        "v41_2_cv_splits.json"
    )

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

    folds = []
    for f in split_payload["folds"]:
        folds.append((
            np.asarray(f["train_indices"], dtype=int),
            np.asarray(f["test_indices"], dtype=int),
        ))

    if len(folds) != 15:
        raise RuntimeError(f"Expected 15 frozen folds, found {len(folds)}.")

    return Xdf, X, y, genes, folds, expr_path, labels_path, splits_path


def fit_train_scaler_and_variance_pool(X_train, X_test, genes):
    """
    Training-only standardization and training-only unsupervised variance pool.
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    var = np.var(Xtr, axis=0, ddof=1)

    # Stable ranking: descending variance, then gene name.
    # lexsort uses last key as primary.
    order = np.lexsort((genes.astype(str), -var))
    chosen = order[:VARIANCE_POOL]

    return Xtr[:, chosen], Xte[:, chosen], genes[chosen], chosen


def class_contrast_operator(X, y):
    """
    H = mean(xx^T | ABC) - mean(xx^T | GCB)
    """
    Xa = X[y == 1]
    Xg = X[y == 0]

    if len(Xa) == 0 or len(Xg) == 0:
        raise RuntimeError("Both classes are required.")

    Ha = (Xa.T @ Xa) / len(Xa)
    Hg = (Xg.T @ Xg) / len(Xg)
    H = Ha - Hg

    # Explicitly symmetrize against floating-point drift.
    return 0.5 * (H + H.T)


def softspaces_decompose_and_score(H):
    """
    Spectral decomposition and coordinate P-Q coupling score.
    """
    evals, evecs = np.linalg.eigh(H)

    # largest |E| first
    order = np.argsort(np.abs(evals))[::-1]
    top = order[:P_DIM]
    U = evecs[:, top]

    # diag(P) without explicitly constructing dense P
    p_diag = np.sum(U * U, axis=1)

    # C_g = ||P V_g Q||_F^2 = p_g (1-p_g)
    coupling = p_diag * (1.0 - p_diag)

    abs_e = np.abs(evals)
    total_abs = float(np.sum(abs_e))
    top_abs = float(np.sum(abs_e[top]))

    diagnostics = {
        "spectral_abs_sum": total_abs,
        "topP_abs_sum": top_abs,
        "topP_abs_fraction": top_abs / total_abs if total_abs > 0 else 0.0,
        "max_abs_eigenvalue": float(np.max(abs_e)),
        "median_abs_eigenvalue": float(np.median(abs_e)),
        "positive_eigenvalues": int(np.sum(evals > 1e-12)),
        "negative_eigenvalues": int(np.sum(evals < -1e-12)),
        "near_zero_eigenvalues": int(np.sum(np.abs(evals) <= 1e-12)),
        "coupling_mean": float(np.mean(coupling)),
        "coupling_median": float(np.median(coupling)),
        "coupling_max": float(np.max(coupling)),
    }

    return evals, coupling, diagnostics


def stable_rank(scores, genes):
    """
    Descending score, deterministic alphabetical tie-break.
    """
    return np.lexsort((genes.astype(str), -scores))


def fixed_logistic_readout(Xtr, ytr, Xte, yte, feature_idx):
    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        random_state=41_300_900,
    )
    model.fit(Xtr[:, feature_idx], ytr)
    prob = model.predict_proba(Xte[:, feature_idx])[:, 1]
    pred = model.predict(Xte[:, feature_idx])

    return {
        "roc_auc": float(roc_auc_score(yte, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "accuracy": float(accuracy_score(yte, pred)),
    }


def jaccard_top(rank_a, rank_b, k):
    a = set(rank_a[:k].tolist())
    b = set(rank_b[:k].tolist())
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def summarize(values):
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

    fold_csv = out_dir / "v41_3_mapping_probe_fold_metrics.csv"
    genes_csv = out_dir / "v41_3_mapping_probe_top_genes.csv"
    summary_json = out_dir / "v41_3_mapping_probe_summary.json"

    Xdf, X, y, genes, folds, expr_path, labels_path, splits_path = load_inputs(project_dir)

    print("=== Soft Spaces Phase 4 v41.3 — DLBCL biological mapping probe ===")
    print(f"Samples: {X.shape[0]}")
    print(f"Genes:   {X.shape[1]}")
    print(f"Frozen folds reused from v41.2: {len(folds)}")
    print(f"Variance pool per fold: {VARIANCE_POOL}")
    print(f"P dimension: {P_DIM}")
    print(f"Feature budgets: {FEATURE_BUDGETS}")
    print()
    print("Mapping:")
    print("  H_REAL = <xx^T>_ABC - <xx^T>_GCB")
    print("  P      = top-|E| spectral subspace of H")
    print("  Q      = I - P")
    print("  C_g    = ||P V_g Q||_F^2 = p_g(1-p_g)")
    print("  NULL   = class-count-preserving label permutation")
    print()

    fold_rows = []
    top_gene_rows = []

    t0 = time.perf_counter()

    for fold_no, (train_idx, test_idx) in enumerate(folds, start=1):
        ft0 = time.perf_counter()

        Xtr_raw = X[train_idx]
        Xte_raw = X[test_idx]
        ytr = y[train_idx]
        yte = y[test_idx]

        Xtr, Xte, pool_genes, pool_global_idx = fit_train_scaler_and_variance_pool(
            Xtr_raw, Xte_raw, genes
        )

        H_real = class_contrast_operator(Xtr, ytr)

        rng = np.random.default_rng(NULL_BASE_SEED + fold_no)
        y_null = rng.permutation(ytr)
        if Counter(y_null.tolist()) != Counter(ytr.tolist()):
            raise RuntimeError("NULL permutation failed to preserve class counts.")
        H_null = class_contrast_operator(Xtr, y_null)

        evals_real, score_real, diag_real = softspaces_decompose_and_score(H_real)
        evals_null, score_null, diag_null = softspaces_decompose_and_score(H_null)

        rank_real = stable_rank(score_real, pool_genes)
        rank_null = stable_rank(score_null, pool_genes)

        fold_seconds = time.perf_counter() - ft0

        # Structural diagnostics, once per fold.
        base_row = {
            "fold": fold_no,
            "variance_pool": VARIANCE_POOL,
            "p_dim": P_DIM,
            "real_topP_abs_fraction": diag_real["topP_abs_fraction"],
            "null_topP_abs_fraction": diag_null["topP_abs_fraction"],
            "delta_topP_abs_fraction": (
                diag_real["topP_abs_fraction"] - diag_null["topP_abs_fraction"]
            ),
            "real_max_abs_eigenvalue": diag_real["max_abs_eigenvalue"],
            "null_max_abs_eigenvalue": diag_null["max_abs_eigenvalue"],
            "real_coupling_mean": diag_real["coupling_mean"],
            "null_coupling_mean": diag_null["coupling_mean"],
            "real_coupling_max": diag_real["coupling_max"],
            "null_coupling_max": diag_null["coupling_max"],
            "top32_jaccard_real_null": jaccard_top(rank_real, rank_null, 32),
            "fold_runtime_seconds": fold_seconds,
        }

        # Save top genes for audit.
        for pos in range(min(64, len(rank_real))):
            i = rank_real[pos]
            top_gene_rows.append({
                "fold": fold_no,
                "model": "REAL",
                "rank": pos + 1,
                "gene": str(pool_genes[i]),
                "coupling_score": float(score_real[i]),
                "global_gene_column": int(pool_global_idx[i]),
            })
        for pos in range(min(64, len(rank_null))):
            i = rank_null[pos]
            top_gene_rows.append({
                "fold": fold_no,
                "model": "NULL",
                "rank": pos + 1,
                "gene": str(pool_genes[i]),
                "coupling_score": float(score_null[i]),
                "global_gene_column": int(pool_global_idx[i]),
            })

        # Fixed readout at multiple small budgets.
        for budget in FEATURE_BUDGETS:
            real_idx = rank_real[:budget]
            null_idx = rank_null[:budget]

            real_perf = fixed_logistic_readout(Xtr, ytr, Xte, yte, real_idx)
            null_perf = fixed_logistic_readout(Xtr, ytr, Xte, yte, null_idx)

            row = dict(base_row)
            row.update({
                "budget": budget,
                "real_roc_auc": real_perf["roc_auc"],
                "null_roc_auc": null_perf["roc_auc"],
                "delta_roc_auc": real_perf["roc_auc"] - null_perf["roc_auc"],
                "real_balanced_accuracy": real_perf["balanced_accuracy"],
                "null_balanced_accuracy": null_perf["balanced_accuracy"],
                "delta_balanced_accuracy": (
                    real_perf["balanced_accuracy"] -
                    null_perf["balanced_accuracy"]
                ),
                "real_accuracy": real_perf["accuracy"],
                "null_accuracy": null_perf["accuracy"],
                "delta_accuracy": real_perf["accuracy"] - null_perf["accuracy"],
                "topk_jaccard_real_null": jaccard_top(rank_real, rank_null, budget),
            })
            fold_rows.append(row)

        print(
            f"fold {fold_no:02d}/15  "
            f"Δspectral={base_row['delta_topP_abs_fraction']:+.4f}  "
            f"J32={base_row['top32_jaccard_real_null']:.3f}  "
            f"{fold_seconds:.2f}s"
        )

    total_seconds = time.perf_counter() - t0

    fold_df = pd.DataFrame(fold_rows)
    gene_df = pd.DataFrame(top_gene_rows)

    fold_df.to_csv(fold_csv, index=False)
    gene_df.to_csv(genes_csv, index=False)

    budget_summary = {}
    for budget in FEATURE_BUDGETS:
        d = fold_df[fold_df["budget"] == budget]
        budget_summary[str(budget)] = {
            "real_roc_auc": summarize(d["real_roc_auc"]),
            "null_roc_auc": summarize(d["null_roc_auc"]),
            "delta_roc_auc": summarize(d["delta_roc_auc"]),
            "real_balanced_accuracy": summarize(d["real_balanced_accuracy"]),
            "null_balanced_accuracy": summarize(d["null_balanced_accuracy"]),
            "delta_balanced_accuracy": summarize(d["delta_balanced_accuracy"]),
            "topk_jaccard_real_null": summarize(d["topk_jaccard_real_null"]),
            "positive_delta_auc_folds": int(np.sum(d["delta_roc_auc"] > 0)),
            "tie_delta_auc_folds": int(np.sum(d["delta_roc_auc"] == 0)),
            "negative_delta_auc_folds": int(np.sum(d["delta_roc_auc"] < 0)),
        }

    structural_fold = fold_df.drop_duplicates(subset=["fold"])
    structural_summary = {
        "delta_topP_abs_fraction": summarize(
            structural_fold["delta_topP_abs_fraction"]
        ),
        "real_topP_abs_fraction": summarize(
            structural_fold["real_topP_abs_fraction"]
        ),
        "null_topP_abs_fraction": summarize(
            structural_fold["null_topP_abs_fraction"]
        ),
        "top32_jaccard_real_null": summarize(
            structural_fold["top32_jaccard_real_null"]
        ),
        "real_coupling_mean": summarize(
            structural_fold["real_coupling_mean"]
        ),
        "null_coupling_mean": summarize(
            structural_fold["null_coupling_mean"]
        ),
    }

    payload = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "benchmark": {
            "task": "GSE10846 ABC vs GCB",
            "n_samples": int(X.shape[0]),
            "n_genes": int(X.shape[1]),
            "cv_folds": len(folds),
            "cv_source": str(splits_path.relative_to(project_dir)),
        },
        "mapping": {
            "status": "exploratory biological mapping probe",
            "variance_pool": VARIANCE_POOL,
            "variance_pool_supervised": False,
            "operator": "H = mean(xx^T|ABC) - mean(xx^T|GCB)",
            "p_definition": f"top {P_DIM} eigenvectors ranked by |E|",
            "q_definition": "I - P",
            "coordinate_perturbation": "V_g = |g><g|",
            "coupling_score": "C_g = ||P V_g Q||_F^2 = p_g(1-p_g)",
            "null": "deterministic permutation of training labels preserving counts",
            "plus_minus_E_claimed": False,
        },
        "readout": {
            "purpose": "evaluate feature ranking only",
            "model": "L2 LogisticRegression(C=1, liblinear)",
            "feature_budgets": FEATURE_BUDGETS,
            "important": (
                "The logistic classifier is not claimed to be Soft Spaces; "
                "the Soft Spaces object is the H -> P/Q -> coupling ranking."
            ),
        },
        "structural_summary": structural_summary,
        "budget_summary": budget_summary,
        "total_runtime_seconds": float(total_seconds),
        "outputs": {
            "fold_metrics": str(fold_csv.relative_to(project_dir)),
            "top_genes": str(genes_csv.relative_to(project_dir)),
        },
        "scientific_limits": [
            "The biological H is a proposed operator, not the original physical Hamiltonian.",
            "No exact +/-E symmetry is assumed or claimed.",
            "Positive results establish usefulness of this mapping only.",
            "No quantum advantage is claimed.",
            "No Aer or QPU execution was performed.",
        ],
    }

    summary_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=== v41.3 SUMMARY ===")
    s = structural_summary["delta_topP_abs_fraction"]
    print(
        "REAL-NULL spectral concentration delta: "
        f"{s['mean']:+.4f} ± {s['sd']:.4f}"
    )

    for budget in FEATURE_BUDGETS:
        b = budget_summary[str(budget)]
        print(
            f"K={budget:2d}  "
            f"REAL AUC={b['real_roc_auc']['mean']:.4f}  "
            f"NULL AUC={b['null_roc_auc']['mean']:.4f}  "
            f"Δ={b['delta_roc_auc']['mean']:+.4f}  "
            f"(+ / = / - folds: "
            f"{b['positive_delta_auc_folds']} / "
            f"{b['tie_delta_auc_folds']} / "
            f"{b['negative_delta_auc_folds']})"
        )

    print()
    print(f"Total runtime: {total_seconds:.1f}s = {total_seconds/60:.2f} min")
    print(f"Fold metrics: {fold_csv}")
    print(f"Top genes:    {genes_csv}")
    print(f"Summary:      {summary_json}")
    print()
    print("v41.3 COMPLETE.")
    print("No Aer or QPU execution performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
