#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v41.2 — Classical DLBCL baseline benchmark.

Purpose
-------
Establish locked classical baselines for the GSE10846 ABC-vs-GCB benchmark
before Soft Spaces is evaluated in v41.3.

Input
-----
data/processed/GSE10846_ABC_GCB_gene_expression.csv.gz
data/processed/GSE10846_ABC_GCB_sample_labels.csv

Models
------
1. Logistic Regression
2. Linear SVM
3. Random Forest
4. PCA + Logistic Regression
5. Univariate ANOVA feature ranking + Logistic Regression

Evaluation
----------
Repeated stratified 5-fold cross-validation, 3 repeats (15 test folds total).
The exact same folds are saved and MUST be reused by Soft Spaces v41.3.

Primary metrics:
  - ROC-AUC
  - Balanced accuracy
  - Accuracy
  - F1

Leakage control
---------------
All scaling, PCA and feature selection occur INSIDE sklearn Pipelines and are
therefore fitted only on training folds.

No model selection is performed using the test folds. Hyperparameters are
frozen below before execution.

No Aer or QPU execution is performed.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


VERSION = "v41.2-dlbcl-classical-baselines-1"

N_SPLITS = 5
N_REPEATS = 3
CV_RANDOM_STATE = 41_200_001

UNIVARIATE_K = 500
PCA_COMPONENTS = 50

MODELS = {
    "logistic_regression": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=5000,
            random_state=41_200_010,
        )),
    ]),
    "linear_svm": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LinearSVC(
            C=1.0,
            max_iter=10000,
            random_state=41_200_020,
        )),
    ]),
    "random_forest": RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=41_200_030,
    ),
    "pca50_logistic": Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(
            n_components=PCA_COMPONENTS,
            svd_solver="randomized",
            random_state=41_200_040,
        )),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=5000,
            random_state=41_200_041,
        )),
    ]),
    "anova500_logistic": Pipeline([
        ("select", SelectKBest(score_func=f_classif, k=UNIVARIATE_K)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=5000,
            random_state=41_200_050,
        )),
    ]),
}


def load_data(project_dir: Path):
    expr_path = (
        project_dir / "data" / "processed" /
        "GSE10846_ABC_GCB_gene_expression.csv.gz"
    )
    labels_path = (
        project_dir / "data" / "processed" /
        "GSE10846_ABC_GCB_sample_labels.csv"
    )

    if not expr_path.exists():
        raise FileNotFoundError(expr_path)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    Xdf = pd.read_csv(expr_path, index_col=0)
    labels = pd.read_csv(labels_path, dtype=str).fillna("")

    required = {"sample_id", "coo_label"}
    missing = required - set(labels.columns)
    if missing:
        raise RuntimeError(f"Missing label columns: {sorted(missing)}")

    labels = labels.set_index("sample_id")
    missing_ids = sorted(set(Xdf.index) - set(labels.index))
    if missing_ids:
        raise RuntimeError(f"Expression samples missing labels: {missing_ids[:10]}")

    labels = labels.loc[Xdf.index]

    bad = sorted(set(labels["coo_label"]) - {"ABC", "GCB"})
    if bad:
        raise RuntimeError(f"Unexpected COO labels: {bad}")

    y = (labels["coo_label"] == "ABC").astype(int).to_numpy()
    X = Xdf.to_numpy(dtype=np.float64)

    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite expression values found.")

    counts = labels["coo_label"].value_counts().to_dict()
    if counts != {"GCB": 183, "ABC": 167}:
        raise RuntimeError(f"Unexpected locked class counts: {counts}")

    return Xdf, labels, X, y, expr_path, labels_path


def get_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise RuntimeError(f"Model {type(model).__name__} has no ranking score method.")


def summarize(vals):
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "sd": float(np.std(a, ddof=1)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def main() -> int:
    code_dir = Path(__file__).resolve().parent
    project_dir = code_dir.parent
    results_dir = project_dir / "results" / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)

    fold_csv = results_dir / "v41_2_fold_metrics.csv"
    summary_json = results_dir / "v41_2_baseline_summary.json"
    splits_json = results_dir / "v41_2_cv_splits.json"

    print("=== Soft Spaces Phase 4 v41.2 — Classical DLBCL baselines ===")
    print()

    Xdf, labels, X, y, expr_path, labels_path = load_data(project_dir)

    print(f"Samples: {X.shape[0]}")
    print(f"Genes:   {X.shape[1]}")
    print(f"ABC:     {int(y.sum())}")
    print(f"GCB:     {int((1-y).sum())}")
    print()
    print(
        f"CV: RepeatedStratifiedKFold "
        f"{N_SPLITS}-fold x {N_REPEATS} repeats = {N_SPLITS*N_REPEATS} folds"
    )
    print(f"Models: {', '.join(MODELS)}")
    print()

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=CV_RANDOM_STATE,
    )

    # Materialize and freeze splits once. Every model sees identical folds.
    splits = list(cv.split(X, y))

    split_payload = {
        "version": VERSION,
        "cv": {
            "type": "RepeatedStratifiedKFold",
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "random_state": CV_RANDOM_STATE,
        },
        "sample_order": Xdf.index.tolist(),
        "folds": [
            {
                "fold_index": i,
                "train_indices": tr.tolist(),
                "test_indices": te.tolist(),
                "train_sample_ids": Xdf.index[tr].tolist(),
                "test_sample_ids": Xdf.index[te].tolist(),
            }
            for i, (tr, te) in enumerate(splits)
        ],
        "reuse_rule": (
            "Soft Spaces v41.3 must use these exact folds for paired comparison."
        ),
    }
    splits_json.write_text(
        json.dumps(split_payload, indent=2),
        encoding="utf-8",
    )

    all_rows = []
    model_timings = {}
    total_start = time.perf_counter()

    for model_i, (name, model) in enumerate(MODELS.items(), start=1):
        print(f"[{model_i}/{len(MODELS)}] {name}")
        start = time.perf_counter()

        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            fold_start = time.perf_counter()
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            score = get_score(model, Xte)
            fold_seconds = time.perf_counter() - fold_start

            row = {
                "model": name,
                "fold": fold_idx,
                "roc_auc": roc_auc_score(yte, score),
                "balanced_accuracy": balanced_accuracy_score(yte, pred),
                "accuracy": accuracy_score(yte, pred),
                "f1": f1_score(yte, pred),
                "fit_predict_seconds": fold_seconds,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
            all_rows.append(row)

            print(
                f"    fold {fold_idx:02d}/{len(splits)} "
                f"AUC={row['roc_auc']:.4f} "
                f"BalAcc={row['balanced_accuracy']:.4f} "
                f"{fold_seconds:.1f}s"
            )

        elapsed = time.perf_counter() - start
        model_timings[name] = elapsed
        print(f"    model time: {elapsed/60:.2f} min")
        print()

    total_seconds = time.perf_counter() - total_start
    fold_df = pd.DataFrame(all_rows)
    fold_df.to_csv(fold_csv, index=False)

    summary = {}
    for name in MODELS:
        d = fold_df[fold_df["model"] == name]
        summary[name] = {
            "roc_auc": summarize(d["roc_auc"]),
            "balanced_accuracy": summarize(d["balanced_accuracy"]),
            "accuracy": summarize(d["accuracy"]),
            "f1": summarize(d["f1"]),
            "runtime_seconds": float(model_timings[name]),
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
            "class_encoding": {"GCB": 0, "ABC": 1},
            "class_counts": {"GCB": int((1-y).sum()), "ABC": int(y.sum())},
        },
        "cross_validation": {
            "type": "RepeatedStratifiedKFold",
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "total_folds": len(splits),
            "random_state": CV_RANDOM_STATE,
            "splits_file": str(splits_json.relative_to(project_dir)),
        },
        "frozen_models": {
            "logistic_regression": "StandardScaler + L2 LogisticRegression(C=1)",
            "linear_svm": "StandardScaler + LinearSVC(C=1)",
            "random_forest": "500 trees, max_features=sqrt, class_weight=balanced",
            "pca50_logistic": "StandardScaler + PCA(50) + L2 LogisticRegression",
            "anova500_logistic": "SelectKBest(f_classif,k=500) + StandardScaler + L2 LogisticRegression",
        },
        "leakage_control": (
            "Scaling, PCA and feature selection fitted inside each training fold only."
        ),
        "summary": summary,
        "total_runtime_seconds": float(total_seconds),
        "outputs": {
            "fold_metrics": str(fold_csv.relative_to(project_dir)),
            "cv_splits": str(splits_json.relative_to(project_dir)),
        },
        "scope": (
            "Classical baseline benchmark only. "
            "No Soft Spaces REAL/NULL, Aer simulation, or QPU execution."
        ),
    }

    summary_json.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("=== v41.2 SUMMARY ===")
    for name in MODELS:
        s = summary[name]
        print(
            f"{name:22s} "
            f"AUC={s['roc_auc']['mean']:.4f} ± {s['roc_auc']['sd']:.4f}  "
            f"BalAcc={s['balanced_accuracy']['mean']:.4f} ± "
            f"{s['balanced_accuracy']['sd']:.4f}"
        )
    print()
    print(f"Total runtime: {total_seconds:.1f} s = {total_seconds/60:.2f} min")
    print(f"Fold metrics:  {fold_csv}")
    print(f"CV splits:     {splits_json}")
    print(f"Summary:       {summary_json}")
    print()
    print("v41.2 COMPLETE.")
    print("No Soft Spaces, Aer, or QPU execution performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
