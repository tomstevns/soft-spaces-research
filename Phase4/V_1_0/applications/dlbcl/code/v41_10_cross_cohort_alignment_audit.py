#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v41.10 — cross-cohort distribution alignment audit.

Purpose
-------
Investigate why v41.8/v41.9 produced:
    external AUC ~0.829
    but all GSE31312 probabilities far below 0.5.

This script keeps the corrected v41.7/v41.8 Soft Spaces ranking fixed from
GSE10846 and compares PREDEFINED, LABEL-FREE external alignment strategies.

IMPORTANT
---------
This is a post-hoc methods audit. GSE31312 labels have already been observed in
earlier runs. Results here are diagnostic, not pristine prospective validation.

Frozen Soft Spaces design
-------------------------
Training cohort: GSE10846
External cohort: GSE31312
raw GSE10846 variance -> top 256 genes
P dimension = 16
feature K = 16
H = <xx^T>_ABC - <xx^T>_GCB
C_g = ||P V_g Q||_F^2 = p_g(1-p_g)
L2 LogisticRegression(C=1, liblinear)

Alignment strategies
--------------------
A) TRAIN_SCALER
   Baseline from v41.8:
   fit StandardScaler on GSE10846 and apply it unchanged to GSE31312.

B) COHORT_Z
   Fit gene-wise mean/std separately in each cohort, WITHOUT labels.
   Train model on GSE10846 z-scores and apply to GSE31312 z-scores.

C) COHORT_ROBUST
   Fit gene-wise median/IQR separately in each cohort, WITHOUT labels.
   Train model on robust-scaled GSE10846 and apply to robust-scaled GSE31312.
   Soft Spaces ranking itself remains the frozen GSE10846 ranking from the
   corrected StandardScaler pipeline; only the logistic readout transform
   changes in this diagnostic strategy.

No strategy uses GSE31312 labels to compute alignment parameters.
No strategy is selected or optimized by external performance inside this script.

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


VERSION = "v41.10-dlbcl-cross-cohort-alignment-audit-1"

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


def aggregate_probe_to_gene(
    expr_probe: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    common = mapping[mapping["probe_id"].isin(expr_probe.index)]
    grouped = defaultdict(list)

    for probe, gene in common.itertuples(index=False):
        if gene:
            grouped[gene].append(probe)

    genes = sorted(grouped)
    out = np.empty(
        (len(genes), expr_probe.shape[1]),
        dtype=np.float32,
    )

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

    H = (
        (Xa.T @ Xa) / len(Xa)
        -
        (Xg.T @ Xg) / len(Xg)
    )

    return 0.5 * (H + H.T)


def mapping_score(H):
    evals, evecs = np.linalg.eigh(H)

    top = np.argsort(
        np.abs(evals)
    )[::-1][:P_DIM]

    U = evecs[:, top]

    p = np.sum(
        U * U,
        axis=1,
    )

    return p * (1.0 - p)


def stable_rank(scores, genes):
    return np.lexsort(
        (
            genes.astype(str),
            -scores,
        )
    )


def fit_logistic(Xtr, ytr, selected_idx):
    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        random_state=41_100_777,
    )

    model.fit(
        Xtr[:, selected_idx],
        ytr,
    )

    return model


def metrics_at_05(y, prob):
    pred = (
        prob >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "roc_auc":
            float(
                roc_auc_score(
                    y,
                    prob,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),

        "accuracy":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),

        "f1":
            float(
                f1_score(
                    y,
                    pred,
                )
            ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "predicted_ABC":
            int(
                np.sum(
                    pred == 1
                )
            ),

        "predicted_GCB":
            int(
                np.sum(
                    pred == 0
                )
            ),
    }


def robust_center_scale(X):
    median = np.median(
        X,
        axis=0,
    )

    q25 = np.quantile(
        X,
        0.25,
        axis=0,
    )

    q75 = np.quantile(
        X,
        0.75,
        axis=0,
    )

    iqr = q75 - q25

    bad = (
        ~np.isfinite(iqr)
        |
        (np.abs(iqr) < 1e-12)
    )

    iqr = iqr.copy()
    iqr[bad] = 1.0

    Xs = (
        X - median
    ) / iqr

    return Xs, median, iqr


def distribution_summary(X):
    return {
        "mean":
            float(
                np.mean(X)
            ),

        "sd":
            float(
                np.std(
                    X,
                    ddof=1,
                )
            ),

        "median":
            float(
                np.median(X)
            ),

        "q05":
            float(
                np.quantile(
                    X,
                    0.05,
                )
            ),

        "q95":
            float(
                np.quantile(
                    X,
                    0.95,
                )
            ),
    }


def probability_summary(prob, y):
    out = {}

    for cls, name in [
        (0, "GCB"),
        (1, "ABC"),
    ]:
        a = prob[
            y == cls
        ]

        out[name] = {
            "mean":
                float(
                    np.mean(a)
                ),

            "median":
                float(
                    np.median(a)
                ),

            "min":
                float(
                    np.min(a)
                ),

            "max":
                float(
                    np.max(a)
                ),
        }

    return out


def main():
    code_dir = Path(__file__).resolve().parent
    project_dir = code_dir.parent

    train_expr_path = (
        project_dir /
        "data" /
        "processed" /
        "GSE10846_ABC_GCB_gene_expression.csv.gz"
    )

    train_labels_path = (
        project_dir /
        "data" /
        "processed" /
        "GSE10846_ABC_GCB_sample_labels.csv"
    )

    mapping_path = (
        project_dir /
        "data" /
        "processed" /
        "GSE10846_probe_to_gene_mapping.csv.gz"
    )

    ext_matrix_path = (
        project_dir /
        "data" /
        "external" /
        "GSE31312" /
        "raw" /
        "GSE31312_series_matrix.txt.gz"
    )

    ext_labels_path = (
        project_dir /
        "data" /
        "external" /
        "GSE31312" /
        "metadata" /
        "GSE31312_ABC_GCB_GEP_only.csv"
    )

    for p in [
        train_expr_path,
        train_labels_path,
        mapping_path,
        ext_matrix_path,
        ext_labels_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                p
            )

    out_dir = (
        project_dir /
        "results" /
        "comparison"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gene_shift_csv = (
        out_dir /
        "v41_10_selected_gene_distribution_shift.csv"
    )

    predictions_csv = (
        out_dir /
        "v41_10_alignment_predictions.csv"
    )

    summary_json = (
        out_dir /
        "v41_10_alignment_audit_summary.json"
    )

    print(
        "=== Soft Spaces Phase 4 v41.10 "
        "— cross-cohort alignment audit ==="
    )

    print()

    print(
        "Three predefined label-free strategies:"
    )

    print(
        "  A TRAIN_SCALER"
    )

    print(
        "  B COHORT_Z"
    )

    print(
        "  C COHORT_ROBUST"
    )

    print()

    t0 = time.perf_counter()

    # ------------------------------------------------------------
    # Load training cohort
    # ------------------------------------------------------------
    print(
        "[1/7] Loading GSE10846..."
    )

    tr_df = pd.read_csv(
        train_expr_path,
        index_col=0,
    )

    tr_labels = (
        pd.read_csv(
            train_labels_path,
            dtype=str,
        )
        .fillna("")
        .set_index(
            "sample_id"
        )
        .loc[
            tr_df.index
        ]
    )

    ytr = (
        tr_labels[
            "coo_label"
        ] == "ABC"
    ).astype(int).to_numpy()

    if Counter(
        tr_labels[
            "coo_label"
        ]
    ) != Counter({
        "GCB": 183,
        "ABC": 167,
    }):
        raise RuntimeError(
            "Unexpected GSE10846 counts."
        )

    # ------------------------------------------------------------
    # Load external cohort
    # ------------------------------------------------------------
    print(
        "[2/7] Loading GSE31312..."
    )

    ext_probe = read_series_matrix(
        ext_matrix_path
    )

    ext_gene = aggregate_probe_to_gene(
        ext_probe,
        read_mapping(
            mapping_path
        ),
    )

    ext_labels = (
        pd.read_csv(
            ext_labels_path,
            dtype=str,
        )
        .fillna("")
        .set_index(
            "sample_id"
        )
    )

    ext_ids = [
        s
        for s in ext_gene.columns
        if s in ext_labels.index
    ]

    ext_labels = ext_labels.loc[
        ext_ids
    ]

    yext = (
        ext_labels[
            "gep_coo"
        ] == "ABC"
    ).astype(int).to_numpy()

    if Counter(
        ext_labels[
            "gep_coo"
        ]
    ) != Counter({
        "GCB": 237,
        "ABC": 214,
    }):
        raise RuntimeError(
            "Unexpected GSE31312 counts."
        )

    # ------------------------------------------------------------
    # Common genes + corrected raw variance pool
    # ------------------------------------------------------------
    print(
        "[3/7] Building corrected GSE10846 variance pool..."
    )

    common_genes = [
        g
        for g in tr_df.columns
        if g in ext_gene.index
    ]

    Xtr_raw = (
        tr_df[
            common_genes
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    Xext_raw = (
        ext_gene
        .loc[
            common_genes,
            ext_ids,
        ]
        .T
        .to_numpy(
            dtype=np.float64
        )
    )

    gene_names = np.asarray(
        common_genes,
        dtype=object,
    )

    if (
        not np.isfinite(
            Xtr_raw
        ).all()
        or
        not np.isfinite(
            Xext_raw
        ).all()
    ):
        raise RuntimeError(
            "Non-finite expression values."
        )

    raw_var = np.var(
        Xtr_raw,
        axis=0,
        ddof=1,
    )

    var_order = np.lexsort(
        (
            gene_names.astype(str),
            -raw_var,
        )
    )

    pool_global = var_order[
        :VARIANCE_POOL
    ]

    pool_genes = gene_names[
        pool_global
    ]

    Xtr_pool_raw = Xtr_raw[
        :,
        pool_global,
    ]

    Xext_pool_raw = Xext_raw[
        :,
        pool_global,
    ]

    # ------------------------------------------------------------
    # Frozen Soft Spaces ranking from corrected standard-scaled
    # GSE10846 training data
    # ------------------------------------------------------------
    print(
        "[4/7] Freezing corrected Soft Spaces ranking..."
    )

    train_scaler = StandardScaler()

    Xtr_std = train_scaler.fit_transform(
        Xtr_pool_raw
    )

    Xext_train_scaler = train_scaler.transform(
        Xext_pool_raw
    )

    ss_score = mapping_score(
        h_operator(
            Xtr_std,
            ytr,
        )
    )

    ss_rank = stable_rank(
        ss_score,
        pool_genes,
    )

    selected = ss_rank[
        :FEATURE_K
    ]

    selected_genes = pool_genes[
        selected
    ]

    print(
        "      selected genes:"
    )

    for i, g in enumerate(
        selected_genes,
        start=1,
    ):
        print(
            f"      {i:2d}. {g}"
        )

    # ------------------------------------------------------------
    # Strategy A: baseline train-scaler transport
    # ------------------------------------------------------------
    print(
        "[5/7] Strategy A — TRAIN_SCALER..."
    )

    model_a = fit_logistic(
        Xtr_std,
        ytr,
        selected,
    )

    prob_a = model_a.predict_proba(
        Xext_train_scaler[
            :,
            selected,
        ]
    )[:, 1]

    met_a = metrics_at_05(
        yext,
        prob_a,
    )

    # ------------------------------------------------------------
    # Strategy B: cohort-wise z-score
    # ------------------------------------------------------------
    print(
        "[6/7] Strategy B/C — label-free cohort alignment..."
    )

    ext_scaler = StandardScaler()

    Xext_cohort_z = ext_scaler.fit_transform(
        Xext_pool_raw
    )

    # Training remains GSE10846 z-scores.
    model_b = fit_logistic(
        Xtr_std,
        ytr,
        selected,
    )

    prob_b = model_b.predict_proba(
        Xext_cohort_z[
            :,
            selected,
        ]
    )[:, 1]

    met_b = metrics_at_05(
        yext,
        prob_b,
    )

    # ------------------------------------------------------------
    # Strategy C: cohort-wise robust scaling.
    # Ranking frozen; logistic readout retrained on robust-scaled
    # training selected features.
    # ------------------------------------------------------------
    Xtr_robust, _, _ = robust_center_scale(
        Xtr_pool_raw
    )

    Xext_robust, _, _ = robust_center_scale(
        Xext_pool_raw
    )

    model_c = fit_logistic(
        Xtr_robust,
        ytr,
        selected,
    )

    prob_c = model_c.predict_proba(
        Xext_robust[
            :,
            selected,
        ]
    )[:, 1]

    met_c = metrics_at_05(
        yext,
        prob_c,
    )

    # ------------------------------------------------------------
    # Per-gene shift audit
    # ------------------------------------------------------------
    print(
        "[7/7] Writing distribution audit..."
    )

    gene_rows = []

    for rank_pos, idx in enumerate(
        selected,
        start=1,
    ):
        tr = Xtr_pool_raw[
            :,
            idx,
        ]

        ex = Xext_pool_raw[
            :,
            idx,
        ]

        gene_rows.append({
            "rank":
                rank_pos,

            "gene":
                str(
                    pool_genes[
                        idx
                    ]
                ),

            "train_mean":
                float(
                    np.mean(tr)
                ),

            "external_mean":
                float(
                    np.mean(ex)
                ),

            "mean_shift_external_minus_train":
                float(
                    np.mean(ex)
                    -
                    np.mean(tr)
                ),

            "train_sd":
                float(
                    np.std(
                        tr,
                        ddof=1,
                    )
                ),

            "external_sd":
                float(
                    np.std(
                        ex,
                        ddof=1,
                    )
                ),

            "sd_ratio_external_over_train":
                float(
                    np.std(
                        ex,
                        ddof=1,
                    )
                    /
                    np.std(
                        tr,
                        ddof=1,
                    )
                ),

            "train_median":
                float(
                    np.median(tr)
                ),

            "external_median":
                float(
                    np.median(ex)
                ),
        })

    pd.DataFrame(
        gene_rows
    ).to_csv(
        gene_shift_csv,
        index=False,
    )

    pred_df = pd.DataFrame({
        "sample_id":
            ext_ids,

        "true_coo":
            ext_labels[
                "gep_coo"
            ].to_numpy(),

        "y_true_ABC":
            yext,

        "p_ABC_TRAIN_SCALER":
            prob_a,

        "p_ABC_COHORT_Z":
            prob_b,

        "p_ABC_COHORT_ROBUST":
            prob_c,
    })

    pred_df.to_csv(
        predictions_csv,
        index=False,
    )

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    payload = {
        "version":
            VERSION,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "historical_status":
            "post-hoc cross-cohort distribution alignment audit",

        "environment": {
            "python":
                platform.python_version(),

            "numpy":
                np.__version__,

            "pandas":
                pd.__version__,

            "scikit_learn":
                sklearn.__version__,
        },

        "design": {
            "training_cohort":
                "GSE10846",

            "external_cohort":
                "GSE31312",

            "external_alignment_uses_labels":
                False,

            "variance_pool":
                VARIANCE_POOL,

            "p_dim":
                P_DIM,

            "feature_k":
                FEATURE_K,

            "softspaces_ranking_source":
                "corrected GSE10846 pipeline from v41.7/v41.8",

            "strategies": [
                "TRAIN_SCALER",
                "COHORT_Z",
                "COHORT_ROBUST",
            ],
        },

        "selected_genes":
            [
                str(x)
                for x in selected_genes
            ],

        "raw_distribution": {
            "training_selected_features":
                distribution_summary(
                    Xtr_pool_raw[
                        :,
                        selected,
                    ]
                ),

            "external_selected_features":
                distribution_summary(
                    Xext_pool_raw[
                        :,
                        selected,
                    ]
                ),
        },

        "results": {
            "TRAIN_SCALER": {
                "description":
                    "v41.8 baseline; GSE10846 scaler transported unchanged",

                "metrics":
                    met_a,

                "probability_by_class":
                    probability_summary(
                        prob_a,
                        yext,
                    ),
            },

            "COHORT_Z": {
                "description":
                    "gene-wise mean/std estimated separately in each cohort; no external labels",

                "metrics":
                    met_b,

                "probability_by_class":
                    probability_summary(
                        prob_b,
                        yext,
                    ),
            },

            "COHORT_ROBUST": {
                "description":
                    "gene-wise median/IQR estimated separately in each cohort; no external labels",

                "metrics":
                    met_c,

                "probability_by_class":
                    probability_summary(
                        prob_c,
                        yext,
                    ),
            },
        },

        "interpretation_limits": [
            "GSE31312 has already been observed, so this is a post-hoc methods audit.",
            "No external labels are used to calculate alignment transforms.",
            "Three predefined strategies are reported side-by-side; this script does not declare or freeze a winner.",
            "COHORT_ROBUST changes the logistic readout scaling but keeps the Soft Spaces gene ranking frozen.",
            "Any strategy chosen for future prospective work must be frozen before testing a new independent cohort.",
            "No quantum advantage is claimed.",
            "No Aer or QPU execution was performed.",
        ],

        "runtime_seconds":
            float(
                elapsed
            ),

        "outputs": {
            "selected_gene_distribution_shift":
                str(
                    gene_shift_csv.relative_to(
                        project_dir
                    )
                ),

            "external_predictions":
                str(
                    predictions_csv.relative_to(
                        project_dir
                    )
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
    print(
        "=== v41.10 SUMMARY ==="
    )

    for name, m in [
        ("TRAIN_SCALER", met_a),
        ("COHORT_Z", met_b),
        ("COHORT_ROBUST", met_c),
    ]:
        print(
            f"{name:14s}  "
            f"AUC={m['roc_auc']:.4f}  "
            f"BalAcc={m['balanced_accuracy']:.4f}  "
            f"PredABC={m['predicted_ABC']:3d}"
        )

    print()
    print(
        f"Runtime: {elapsed:.1f}s = "
        f"{elapsed/60:.2f} min"
    )

    print(
        f"Gene shift:  {gene_shift_csv}"
    )

    print(
        f"Predictions: {predictions_csv}"
    )

    print(
        f"Summary:     {summary_json}"
    )

    print()
    print(
        "v41.10 COMPLETE."
    )

    print(
        "No Aer or QPU execution performed."
    )


if __name__ == "__main__":
    main()
