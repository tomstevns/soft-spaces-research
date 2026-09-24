#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v41.9 — training-only threshold calibration audit.

Purpose
-------
Investigate the v41.8 cross-cohort threshold failure WITHOUT changing the
Soft Spaces ranking and WITHOUT tuning anything on GSE31312.

Key idea
--------
v41.8 had:
    external AUC ≈ 0.829
    balanced accuracy = 0.50 at threshold 0.5

That suggests ranking survived but the fixed probability threshold did not.

v41.9 therefore:
1. Reconstructs OUT-OF-FOLD predictions on GSE10846 using the corrected v41.7
   pipeline and the exact frozen 15 CV folds.
2. Chooses ONE classification threshold using GSE10846 OOF predictions only.
3. Trains the final corrected pipeline on all GSE10846 samples.
4. Applies that fixed training-only threshold to GSE31312.
5. Reports:
      - AUC (threshold independent)
      - default threshold 0.5 metrics
      - training-only calibrated threshold metrics
6. Does NOT tune threshold on GSE31312.

Frozen mapping
--------------
raw training variance -> top 256 genes -> StandardScaler
P dimension = 16
feature K = 16
H = <xx^T>_ABC - <xx^T>_GCB
C_g = ||P V_g Q||_F^2 = p_g(1-p_g)
L2 LogisticRegression(C=1, liblinear)

Historical status
-----------------
GSE31312 has already been observed in v41.6 and v41.8.
This is a post-hoc calibration audit, not pristine validation.

No Aer. No QPU.
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
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


VERSION = "v41.9-dlbcl-training-only-threshold-calibration-1"

VARIANCE_POOL = 256
P_DIM = 16
FEATURE_K = 16


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
            vals.append(
                np.nan
                if x in {"", "NA", "NaN", "nan", "NULL", "null"}
                else float(x)
            )
        data[i] = vals

    return pd.DataFrame(
        data,
        index=pd.Index(probe_ids, name="probe_id"),
        columns=sample_ids,
    )


def read_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip", dtype=str).fillna("")
    need = {"probe_id", "gene_symbol"}
    if not need.issubset(df.columns):
        raise RuntimeError(
            f"Mapping missing columns: {sorted(need - set(df.columns))}"
        )
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
        out[i] = np.nanmedian(
            expr_probe.loc[probes].to_numpy(dtype=np.float32),
            axis=0,
        )

    return pd.DataFrame(
        out,
        index=pd.Index(genes, name="gene_symbol"),
        columns=expr_probe.columns,
    )


def h_operator(X, y):
    Xa = X[y == 1]
    Xg = X[y == 0]
    H = (Xa.T @ Xa) / len(Xa) - (Xg.T @ Xg) / len(Xg)
    return 0.5 * (H + H.T)


def mapping_score(H):
    evals, evecs = np.linalg.eigh(H)
    top = np.argsort(np.abs(evals))[::-1][:P_DIM]
    U = evecs[:, top]
    p = np.sum(U * U, axis=1)
    coupling = p * (1.0 - p)
    return coupling


def stable_rank(scores, genes):
    return np.lexsort((genes.astype(str), -scores))


def corrected_fit_transform(Xtr_raw, Xte_raw, genes):
    raw_var = np.var(Xtr_raw, axis=0, ddof=1)
    order = np.lexsort((genes.astype(str), -raw_var))
    chosen = order[:VARIANCE_POOL]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_raw[:, chosen])
    Xte = scaler.transform(Xte_raw[:, chosen])

    return Xtr, Xte, genes[chosen], chosen, scaler


def train_model(Xtr, ytr, selected_idx):
    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        random_state=41_900_777,
    )
    model.fit(Xtr[:, selected_idx], ytr)
    return model


def metrics_at_threshold(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_ABC": int(np.sum(pred == 1)),
        "predicted_GCB": int(np.sum(pred == 0)),
    }


def choose_training_only_threshold(y, prob):
    """
    Choose threshold maximizing balanced accuracy using ONLY GSE10846 OOF probs.
    Deterministic tie-break:
      1) highest balanced accuracy
      2) closest threshold to 0.5
      3) lower threshold
    """
    candidates = np.unique(
        np.concatenate([
            np.array([0.0, 0.5, 1.0]),
            prob,
            np.nextafter(prob, 1.0),
        ])
    )

    rows = []
    for t in candidates:
        m = metrics_at_threshold(y, prob, float(t))
        rows.append((m["balanced_accuracy"], abs(float(t) - 0.5), float(t)))

    best = sorted(rows, key=lambda x: (-x[0], x[1], x[2]))[0]
    return best[2]


def main():
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
    splits_path = (
        project_dir / "results" / "baseline" /
        "v41_2_cv_splits.json"
    )
    mapping_path = (
        project_dir / "data" / "processed" /
        "GSE10846_probe_to_gene_mapping.csv.gz"
    )
    ext_matrix_path = (
        project_dir / "data" / "external" / "GSE31312" /
        "raw" / "GSE31312_series_matrix.txt.gz"
    )
    ext_labels_path = (
        project_dir / "data" / "external" / "GSE31312" /
        "metadata" / "GSE31312_ABC_GCB_GEP_only.csv"
    )

    for p in [
        train_expr_path,
        train_labels_path,
        splits_path,
        mapping_path,
        ext_matrix_path,
        ext_labels_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    out_dir = project_dir / "results" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_csv = out_dir / "v41_9_training_oof_predictions.csv"
    ext_pred_csv = out_dir / "v41_9_external_calibrated_predictions.csv"
    selected_csv = out_dir / "v41_9_final_selected_genes.csv"
    summary_json = out_dir / "v41_9_threshold_calibration_summary.json"

    print("=== Soft Spaces Phase 4 v41.9 — training-only threshold calibration ===")
    print("Threshold will be selected from GSE10846 OOF predictions only.")
    print("No threshold tuning on GSE31312.")
    print()

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Load GSE10846 + frozen folds
    # ------------------------------------------------------------------
    print("[1/7] Loading GSE10846 and frozen folds...")

    Xdf = pd.read_csv(train_expr_path, index_col=0)
    labels = (
        pd.read_csv(train_labels_path, dtype=str)
        .fillna("")
        .set_index("sample_id")
        .loc[Xdf.index]
    )

    X = Xdf.to_numpy(dtype=np.float64)
    y = (labels["coo_label"] == "ABC").astype(int).to_numpy()
    genes = np.asarray(Xdf.columns, dtype=object)

    split_payload = json.loads(
        splits_path.read_text(encoding="utf-8")
    )

    if split_payload["sample_order"] != Xdf.index.tolist():
        raise RuntimeError("Frozen fold sample order mismatch.")

    folds = [
        (
            np.asarray(f["train_indices"], dtype=int),
            np.asarray(f["test_indices"], dtype=int),
        )
        for f in split_payload["folds"]
    ]

    if len(folds) != 15:
        raise RuntimeError("Expected 15 frozen folds.")

    if Counter(labels["coo_label"]) != Counter({"GCB": 183, "ABC": 167}):
        raise RuntimeError("Unexpected GSE10846 counts.")

    # ------------------------------------------------------------------
    # OOF prediction generation
    # Repeated CV gives 3 OOF predictions per sample.
    # Average them per sample before threshold selection.
    # ------------------------------------------------------------------
    print("[2/7] Generating GSE10846 out-of-fold predictions...")

    per_sample_probs = defaultdict(list)
    oof_rows = []

    for fold_no, (tr, te) in enumerate(folds, start=1):
        Xtr, Xte, pool_genes, _, _ = corrected_fit_transform(
            X[tr], X[te], genes
        )

        ytr = y[tr]
        yte = y[te]

        s = mapping_score(h_operator(Xtr, ytr))
        r = stable_rank(s, pool_genes)
        selected = r[:FEATURE_K]

        model = train_model(Xtr, ytr, selected)
        prob = model.predict_proba(Xte[:, selected])[:, 1]

        for local_pos, sample_idx in enumerate(te):
            p = float(prob[local_pos])
            per_sample_probs[int(sample_idx)].append(p)

            oof_rows.append({
                "fold": fold_no,
                "sample_index": int(sample_idx),
                "sample_id": str(Xdf.index[sample_idx]),
                "true_coo": str(labels.iloc[sample_idx]["coo_label"]),
                "y_true_ABC": int(y[sample_idx]),
                "p_ABC": p,
            })

    if set(per_sample_probs) != set(range(len(Xdf))):
        raise RuntimeError("Not all samples received OOF predictions.")

    oof_mean_prob = np.array([
        np.mean(per_sample_probs[i])
        for i in range(len(Xdf))
    ])

    replicate_counts = Counter(
        len(v) for v in per_sample_probs.values()
    )

    print(f"      OOF predictions/sample: {dict(replicate_counts)}")

    pd.DataFrame(oof_rows).to_csv(oof_csv, index=False)

    # ------------------------------------------------------------------
    # Training-only threshold
    # ------------------------------------------------------------------
    print("[3/7] Choosing threshold from GSE10846 OOF predictions only...")

    threshold = choose_training_only_threshold(y, oof_mean_prob)

    oof_default = metrics_at_threshold(y, oof_mean_prob, 0.5)
    oof_calibrated = metrics_at_threshold(y, oof_mean_prob, threshold)

    print(f"      selected threshold: {threshold:.6f}")
    print(
        f"      OOF BalAcc: 0.5={oof_default['balanced_accuracy']:.4f}  "
        f"calibrated={oof_calibrated['balanced_accuracy']:.4f}"
    )

    # ------------------------------------------------------------------
    # Prepare external cohort
    # ------------------------------------------------------------------
    print("[4/7] Preparing GSE31312...")

    ext_probe = read_series_matrix(ext_matrix_path)
    ext_gene = aggregate_probe_to_gene(
        ext_probe,
        read_mapping(mapping_path),
    )

    ext_labels = (
        pd.read_csv(ext_labels_path, dtype=str)
        .fillna("")
        .set_index("sample_id")
    )

    ext_ids = [
        s for s in ext_gene.columns
        if s in ext_labels.index
    ]
    ext_labels = ext_labels.loc[ext_ids]

    yext = (
        ext_labels["gep_coo"] == "ABC"
    ).astype(int).to_numpy()

    if Counter(ext_labels["gep_coo"]) != Counter({"GCB": 237, "ABC": 214}):
        raise RuntimeError("Unexpected GSE31312 counts.")

    # ------------------------------------------------------------------
    # Final corrected model on all GSE10846
    # ------------------------------------------------------------------
    print("[5/7] Training final corrected model on all GSE10846...")

    common_genes = [
        g for g in Xdf.columns
        if g in ext_gene.index
    ]

    Xtr_raw = Xdf[common_genes].to_numpy(dtype=np.float64)
    Xext_raw = (
        ext_gene
        .loc[common_genes, ext_ids]
        .T
        .to_numpy(dtype=np.float64)
    )

    common_gene_names = np.asarray(common_genes, dtype=object)

    raw_var = np.var(Xtr_raw, axis=0, ddof=1)
    var_order = np.lexsort(
        (
            common_gene_names.astype(str),
            -raw_var,
        )
    )
    pool_global = var_order[:VARIANCE_POOL]
    pool_genes = common_gene_names[pool_global]
    pool_var = raw_var[pool_global]

    scaler = StandardScaler()
    Xtr_pool = scaler.fit_transform(
        Xtr_raw[:, pool_global]
    )
    Xext_pool = scaler.transform(
        Xext_raw[:, pool_global]
    )

    real_score = mapping_score(
        h_operator(Xtr_pool, y)
    )
    real_rank = stable_rank(
        real_score,
        pool_genes,
    )
    selected = real_rank[:FEATURE_K]

    final_model = train_model(
        Xtr_pool,
        y,
        selected,
    )

    ext_prob = final_model.predict_proba(
        Xext_pool[:, selected]
    )[:, 1]

    selected_rows = []
    for rank_pos, idx in enumerate(selected, start=1):
        selected_rows.append({
            "rank": rank_pos,
            "gene": str(pool_genes[idx]),
            "coupling_score": float(real_score[idx]),
            "raw_training_variance": float(pool_var[idx]),
        })

    pd.DataFrame(selected_rows).to_csv(
        selected_csv,
        index=False,
    )

    # ------------------------------------------------------------------
    # External evaluation: no tuning here
    # ------------------------------------------------------------------
    print("[6/7] Evaluating fixed thresholds on GSE31312...")

    ext_default = metrics_at_threshold(
        yext,
        ext_prob,
        0.5,
    )

    ext_calibrated = metrics_at_threshold(
        yext,
        ext_prob,
        threshold,
    )

    print(
        f"      external AUC: {ext_default['roc_auc']:.4f}"
    )
    print(
        f"      BalAcc at 0.5:       {ext_default['balanced_accuracy']:.4f}"
    )
    print(
        f"      BalAcc at train OOF: {ext_calibrated['balanced_accuracy']:.4f}"
    )

    ext_pred_df = pd.DataFrame({
        "sample_id": ext_ids,
        "true_coo": ext_labels["gep_coo"].to_numpy(),
        "y_true_ABC": yext,
        "p_ABC": ext_prob,
        "pred_0_5": np.where(
            ext_prob >= 0.5,
            "ABC",
            "GCB",
        ),
        "pred_training_only_threshold": np.where(
            ext_prob >= threshold,
            "ABC",
            "GCB",
        ),
    })

    ext_pred_df.to_csv(
        ext_pred_csv,
        index=False,
    )

    # ------------------------------------------------------------------
    # Distribution-shift diagnostics
    # ------------------------------------------------------------------
    print("[7/7] Writing calibration audit...")

    train_prob_by_class = {
        "GCB": {
            "mean": float(oof_mean_prob[y == 0].mean()),
            "median": float(np.median(oof_mean_prob[y == 0])),
            "min": float(oof_mean_prob[y == 0].min()),
            "max": float(oof_mean_prob[y == 0].max()),
        },
        "ABC": {
            "mean": float(oof_mean_prob[y == 1].mean()),
            "median": float(np.median(oof_mean_prob[y == 1])),
            "min": float(oof_mean_prob[y == 1].min()),
            "max": float(oof_mean_prob[y == 1].max()),
        },
    }

    ext_prob_by_class = {
        "GCB": {
            "mean": float(ext_prob[yext == 0].mean()),
            "median": float(np.median(ext_prob[yext == 0])),
            "min": float(ext_prob[yext == 0].min()),
            "max": float(ext_prob[yext == 0].max()),
        },
        "ABC": {
            "mean": float(ext_prob[yext == 1].mean()),
            "median": float(np.median(ext_prob[yext == 1])),
            "min": float(ext_prob[yext == 1].min()),
            "max": float(ext_prob[yext == 1].max()),
        },
    }

    elapsed = time.perf_counter() - t0

    payload = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "historical_status": (
            "post-hoc threshold calibration audit; "
            "GSE31312 was previously observed in v41.6 and v41.8"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "design": {
            "threshold_source": "GSE10846 averaged out-of-fold predictions only",
            "threshold_criterion": "maximize balanced accuracy",
            "external_threshold_tuning": False,
            "variance_pool": VARIANCE_POOL,
            "p_dim": P_DIM,
            "feature_k": FEATURE_K,
            "corrected_preprocessing": (
                "raw training variance -> top 256 -> StandardScaler"
            ),
            "oof_cv_source": str(
                splits_path.relative_to(project_dir)
            ),
            "oof_prediction_counts_per_sample": dict(replicate_counts),
        },
        "selected_threshold": float(threshold),
        "gse10846_oof": {
            "default_threshold_0_5": oof_default,
            "training_only_calibrated_threshold": oof_calibrated,
            "probability_distribution_by_class": train_prob_by_class,
        },
        "gse31312_external": {
            "default_threshold_0_5": ext_default,
            "training_only_calibrated_threshold": ext_calibrated,
            "probability_distribution_by_class": ext_prob_by_class,
        },
        "selected_genes": selected_rows,
        "interpretation_limits": [
            "GSE31312 was already observed before this calibration audit.",
            "The threshold is derived only from GSE10846 OOF predictions.",
            "No threshold or parameter is optimized on GSE31312 in this run.",
            "Improved balanced accuracy would support a threshold-shift explanation, not prove biological transportability by itself.",
            "AUC is unchanged by threshold calibration.",
            "No quantum advantage is claimed.",
            "No Aer or QPU execution was performed.",
        ],
        "runtime_seconds": float(elapsed),
        "outputs": {
            "training_oof_predictions": str(
                oof_csv.relative_to(project_dir)
            ),
            "external_predictions": str(
                ext_pred_csv.relative_to(project_dir)
            ),
            "selected_genes": str(
                selected_csv.relative_to(project_dir)
            ),
        },
    }

    summary_json.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=== v41.9 SUMMARY ===")
    print(f"Training-only threshold:      {threshold:.6f}")
    print(f"GSE10846 OOF AUC:             {oof_default['roc_auc']:.4f}")
    print(
        f"GSE10846 OOF BalAcc 0.5:      "
        f"{oof_default['balanced_accuracy']:.4f}"
    )
    print(
        f"GSE10846 OOF BalAcc calibrated:"
        f"{oof_calibrated['balanced_accuracy']:.4f}"
    )
    print()
    print(f"GSE31312 external AUC:        {ext_default['roc_auc']:.4f}")
    print(
        f"GSE31312 BalAcc 0.5:          "
        f"{ext_default['balanced_accuracy']:.4f}"
    )
    print(
        f"GSE31312 BalAcc train-only:   "
        f"{ext_calibrated['balanced_accuracy']:.4f}"
    )
    print(
        f"GSE31312 predicted ABC 0.5:   "
        f"{ext_default['predicted_ABC']}"
    )
    print(
        f"GSE31312 predicted ABC calib: "
        f"{ext_calibrated['predicted_ABC']}"
    )
    print()
    print(f"Runtime: {elapsed:.1f}s = {elapsed/60:.2f} min")
    print(f"OOF predictions:      {oof_csv}")
    print(f"External predictions: {ext_pred_csv}")
    print(f"Selected genes:       {selected_csv}")
    print(f"Summary:              {summary_json}")
    print()
    print("v41.9 COMPLETE.")
    print("No Aer or QPU execution performed.")


if __name__ == "__main__":
    main()
