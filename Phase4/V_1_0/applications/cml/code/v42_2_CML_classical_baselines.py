#!/usr/bin/env python3
"""
Soft Spaces / CML
v42.2 — Frozen preprocessing + classical baselines

Implements the frozen v42.1d development design.

Development cohort:
    GSE130404
    96 baseline diagnostic chronic-phase CML samples

Outcome:
    y=1 : BCR-ABL1 >10% IS at 3 months (EMR failure)
    y=0 : BCR-ABL1 <10% IS at 3 months

Frozen evaluation:
    RepeatedStratifiedKFold(n_splits=5, n_repeats=20)
    master seed = 4202001
    100 held-out fold evaluations

Frozen preprocessing inside each outer training fold:
    raw variance -> Top 256 genes -> StandardScaler(train only)

Classical baselines:
    1. Logistic regression L2, class_weight="balanced"
    2. Logistic regression L1, class_weight="balanced"
    3. Linear SVM, class_weight="balanced"
    4. Univariate F-ranking + balanced logistic regression
       K selected from {8,16,32} using development-only inner CV

Primary metric:
    PR-AUC

Secondary:
    ROC-AUC
    balanced accuracy
    sensitivity
    specificity

NO SOFT-SPACES SCORING IS PERFORMED HERE.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import shutil
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


VERSION = "v42.2"

EXPECTED_DESIGN_SHA256 = (
    "3d397a9a5edec39c4809abf95daf031960d6f4caf14314492c17f3606b480fe2"
)

MASTER_SEED = 4202001
N_SPLITS = 5
N_REPEATS = 20
VARIANCE_POOL = 256
K_GRID = [8, 16, 32]

GPL = "GPL10558"

GPL_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    "?acc=GPL10558&targ=self&form=text&view=full"
)


def project_dir_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_design(project: Path) -> Path:
    candidates = [
        project / "docs" / "v42_1d_CML_FROZEN_DESIGN.txt",
        project / "code" / "v42_1d_CML_FROZEN_DESIGN.txt",
        project / "v42_1d_CML_FROZEN_DESIGN.txt",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError("v42_1d_CML_FROZEN_DESIGN.txt not found")


def download_text(url: str, path: Path):
    if path.exists() and path.stat().st_size > 0:
        print("[download] reuse:", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".part")

    print("[download]", url)
    print("        ->", path)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 SoftSpaces-v42.2",
            "Accept": "text/plain,text/*;q=0.9,*/*;q=0.1",
        },
    )

    with urllib.request.urlopen(req, timeout=180) as r:
        payload = r.read()

    if b"!platform_table_begin" not in payload:
        raise RuntimeError(
            "GEO response did not contain GPL10558 platform table."
        )

    tmp.write_bytes(payload)
    tmp.replace(path)

    print("[download] done:", f"{path.stat().st_size:,}", "bytes")


def parse_tab_line(line: str):
    return [x.strip().strip('"') for x in line.rstrip("\r\n").split("\t")]


def parse_gse130404(path: Path):
    """
    Return:
        X_probe : DataFrame samples x probes
        metadata: DataFrame indexed by GSM
    """
    sample_meta = defaultdict(list)
    inside = False
    header = None
    probes = []
    rows = []

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not inside and line.startswith("!Sample_"):
                parts = parse_tab_line(line)
                sample_meta[parts[0]].append(parts[1:])
                continue

            if line.startswith("!series_matrix_table_begin"):
                inside = True
                continue

            if line.startswith("!series_matrix_table_end"):
                break

            if not inside:
                continue

            if header is None:
                header = parse_tab_line(line)
                continue

            if line.strip():
                parts = line.rstrip("\r\n").split("\t")
                probes.append(parts[0].strip().strip('"'))
                rows.append(
                    [float(x.strip().strip('"')) for x in parts[1:]]
                )

    if header is None:
        raise RuntimeError("Series matrix header not found.")

    sample_ids = header[1:]
    M = np.asarray(rows, dtype=np.float64)

    if M.shape[1] != len(sample_ids):
        raise RuntimeError("Expression matrix sample dimension mismatch.")

    X_probe = pd.DataFrame(
        M.T,
        index=sample_ids,
        columns=probes,
    )

    # Parse sample characteristics.
    n = len(sample_ids)
    meta_rows = [{"geo_accession": gsm} for gsm in sample_ids]

    for key, occurrences in sample_meta.items():
        for occ_i, vals in enumerate(occurrences, 1):
            if len(vals) != n:
                continue
            field = key if len(occurrences) == 1 else f"{key}__{occ_i}"
            for i, v in enumerate(vals):
                meta_rows[i][field] = v

    parsed = []
    for row in meta_rows:
        chars = {}
        for k, v in row.items():
            if k.startswith("!Sample_characteristics_ch1") and ":" in str(v):
                name, value = str(v).split(":", 1)
                chars[name.strip().lower()] = value.strip()

        parsed.append(
            {
                "geo_accession": row["geo_accession"],
                "disease_stage": chars.get("disease stage", ""),
                "bcr_abl1_3m": chars.get("bcr-abl1 at 3 month", ""),
            }
        )

    meta = pd.DataFrame(parsed).set_index("geo_accession")

    if list(meta.index) != list(X_probe.index):
        raise RuntimeError("Metadata/expression sample order mismatch.")

    return X_probe, meta


def parse_platform_mapping(path: Path):
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    try:
        begin = next(
            i for i, line in enumerate(lines)
            if line.startswith("!platform_table_begin")
        )
        end = next(
            i for i, line in enumerate(lines)
            if line.startswith("!platform_table_end")
        )
    except StopIteration:
        raise RuntimeError("GPL10558 table markers not found.")

    table = lines[begin + 1:end]
    if len(table) < 2:
        raise RuntimeError("GPL10558 platform table empty.")

    header = table[0].split("\t")
    lut = {x.strip().lower(): x for x in header}

    id_col = lut.get("id")

    symbol_candidates = [
        "symbol",
        "gene symbol",
        "gene_symbol",
        "genesymbol",
    ]
    symbol_col = next(
        (lut[c] for c in symbol_candidates if c in lut),
        None,
    )

    if id_col is None or symbol_col is None:
        raise RuntimeError(
            "Could not identify GPL10558 ID/Symbol columns. "
            f"Columns: {header}"
        )

    probe_to_symbols = defaultdict(set)

    reader = csv.DictReader(
        table[1:],
        fieldnames=header,
        delimiter="\t",
    )

    for row in reader:
        probe = str(row.get(id_col, "")).strip()
        raw = str(row.get(symbol_col, "")).strip()

        if not probe or not raw:
            continue

        raw = raw.replace("///", "|").replace(";", "|").replace(",", "|")

        for sym in raw.split("|"):
            sym = sym.strip()
            if not sym:
                continue
            if sym.upper() in {"---", "NA", "N/A", "NULL"}:
                continue
            probe_to_symbols[probe].add(sym)

    return dict(probe_to_symbols), {
        "id_column": id_col,
        "symbol_column": symbol_col,
        "all_columns": header,
    }


def aggregate_probe_to_gene(X_probe: pd.DataFrame, probe_to_symbols: dict):
    gene_to_probes = defaultdict(list)

    for probe in X_probe.columns:
        for sym in probe_to_symbols.get(probe, []):
            gene_to_probes[sym].append(probe)

    out = {}

    for gene, probes in gene_to_probes.items():
        out[gene] = X_probe.loc[:, probes].mean(axis=1)

    if not out:
        raise RuntimeError("No genes mapped from GPL10558.")

    return pd.DataFrame(out, index=X_probe.index), gene_to_probes


def make_y(meta: pd.DataFrame):
    y = []
    for gsm, row in meta.iterrows():
        stage = str(row["disease_stage"]).strip().lower()
        resp = str(row["bcr_abl1_3m"]).strip()

        if stage != "diagnostic chronic phase":
            raise RuntimeError(
                f"{gsm}: unexpected disease stage {row['disease_stage']!r}"
            )

        if resp == ">10%":
            y.append(1)
        elif resp == "<10%":
            y.append(0)
        else:
            raise RuntimeError(
                f"{gsm}: unresolved 3-month BCR-ABL1 label {resp!r}"
            )

    y = np.asarray(y, dtype=int)

    counts = Counter(y.tolist())
    if counts != Counter({0: 83, 1: 13}):
        raise RuntimeError(f"Unexpected class counts: {counts}")

    return y


def top_variance_indices(X_train_raw, genes, k):
    var = np.var(X_train_raw, axis=0, ddof=1)
    genes = np.asarray(genes, dtype=object)

    order = np.lexsort(
        (
            genes.astype(str),
            -var,
        )
    )

    return order[:min(k, len(order))]


def confusion_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return (
        float(balanced_accuracy_score(y_true, y_pred)),
        float(sensitivity),
        float(specificity),
    )


def model_metrics(y_true, score, pred):
    return {
        "pr_auc": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "balanced_accuracy": confusion_metrics(y_true, pred)[0],
        "sensitivity": confusion_metrics(y_true, pred)[1],
        "specificity": confusion_metrics(y_true, pred)[2],
    }


def fit_baselines(Xtr, ytr, Xte, yte, outer_fold_seed):
    rows = []
    predictions = []

    # L2 logistic.
    l2 = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=outer_fold_seed,
    )
    l2.fit(Xtr, ytr)
    score = l2.predict_proba(Xte)[:, 1]
    pred = (score >= 0.5).astype(int)
    rows.append(("logistic_l2", None, model_metrics(yte, score, pred)))
    predictions.append(("logistic_l2", None, score, pred))

    # L1 logistic.
    l1 = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=outer_fold_seed,
    )
    l1.fit(Xtr, ytr)
    score = l1.predict_proba(Xte)[:, 1]
    pred = (score >= 0.5).astype(int)
    rows.append(("logistic_l1", None, model_metrics(yte, score, pred)))
    predictions.append(("logistic_l1", None, score, pred))

    # Linear SVM.
    svm = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=10000,
        random_state=outer_fold_seed,
    )
    svm.fit(Xtr, ytr)
    score = svm.decision_function(Xte)
    pred = (score >= 0.0).astype(int)
    rows.append(("linear_svm", None, model_metrics(yte, score, pred)))
    predictions.append(("linear_svm", None, score, pred))

    return rows, predictions


def choose_univariate_k(Xtr, ytr, seed):
    """
    Development-only inner CV to choose K from frozen {8,16,32}.
    The outer test fold is never touched.
    """
    inner = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=seed,
    )

    mean_scores = {}

    for k in K_GRID:
        vals = []

        for inner_train, inner_val in inner.split(Xtr, ytr):
            Xin = Xtr[inner_train]
            yin = ytr[inner_train]
            Xv = Xtr[inner_val]
            yv = ytr[inner_val]

            f, _ = f_classif(Xin, yin)
            f = np.nan_to_num(f, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
            idx = np.argsort(-f, kind="mergesort")[:k]

            model = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            )
            model.fit(Xin[:, idx], yin)
            score = model.predict_proba(Xv[:, idx])[:, 1]
            vals.append(average_precision_score(yv, score))

        mean_scores[k] = float(np.mean(vals))

    # Deterministic tie-break: smallest K.
    best_k = sorted(
        mean_scores,
        key=lambda k: (-mean_scores[k], k),
    )[0]

    return best_k, mean_scores


def summarize_metric(df, metric):
    vals = df[metric].to_numpy(dtype=float)
    return {
        "mean": float(np.mean(vals)),
        "sd": float(np.std(vals, ddof=1)),
        "q025": float(np.quantile(vals, 0.025)),
        "median": float(np.median(vals)),
        "q975": float(np.quantile(vals, 0.975)),
    }


def main():
    project = project_dir_from_script()

    design = find_design(project)
    actual_design_sha = sha256_file(design)

    print(f"=== {VERSION} CML CLASSICAL BASELINES ===")
    print("Expected frozen-design SHA:", EXPECTED_DESIGN_SHA256)
    print("Actual frozen-design SHA:  ", actual_design_sha)

    if actual_design_sha != EXPECTED_DESIGN_SHA256:
        raise SystemExit("FAIL: v42.1d frozen design SHA mismatch.")

    print("PASS: frozen design verified.")
    print("NO SOFT-SPACES SCORING.")
    print()

    series_path = (
        project
        / "data"
        / "external"
        / "GSE130404"
        / "raw"
        / "GSE130404_series_matrix.txt.gz"
    )

    if not series_path.exists():
        raise FileNotFoundError(series_path)

    platform_dir = (
        project
        / "data"
        / "external"
        / "GSE130404"
        / "platform"
    )
    platform_dir.mkdir(parents=True, exist_ok=True)

    gpl_path = platform_dir / "GPL10558_full_geo_table.txt"
    download_text(GPL_URL, gpl_path)

    print("\n[1/6] Loading GSE130404...")
    X_probe, meta = parse_gse130404(series_path)
    print("      samples x probes:", X_probe.shape)

    print("[2/6] Parsing GPL10558 annotation...")
    probe_to_symbols, mapping_info = parse_platform_mapping(gpl_path)
    print("      mapped platform probes:", f"{len(probe_to_symbols):,}")
    print("      symbol column:", mapping_info["symbol_column"])

    print("[3/6] Aggregating probes to genes...")
    X_gene, gene_to_probes = aggregate_probe_to_gene(
        X_probe,
        probe_to_symbols,
    )
    print("      samples x genes:", X_gene.shape)

    y = make_y(meta)
    print("      y=0:", int(np.sum(y == 0)))
    print("      y=1:", int(np.sum(y == 1)))
    print("      positive prevalence:", f"{np.mean(y):.6f}")
    print("      random PR-AUC reference:", f"{np.mean(y):.6f}")

    X_raw = X_gene.to_numpy(dtype=np.float64)
    genes = np.asarray(X_gene.columns, dtype=object)
    sample_ids = np.asarray(X_gene.index, dtype=object)

    if not np.isfinite(X_raw).all():
        raise RuntimeError("Non-finite gene expression values detected.")

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=MASTER_SEED,
    )

    fold_rows = []
    prediction_rows = []
    split_manifest = []

    print("[4/6] Running 5-fold x 20-repeat frozen CV...")

    for fold_no, (train_idx, test_idx) in enumerate(
        splitter.split(X_raw, y),
        start=1,
    ):
        ytr = y[train_idx]
        yte = y[test_idx]

        if len(np.unique(ytr)) != 2 or len(np.unique(yte)) != 2:
            raise RuntimeError(
                f"Fold {fold_no} lacks one class."
            )

        # Frozen raw variance Top-256 on OUTER TRAIN ONLY.
        idx_pool = top_variance_indices(
            X_raw[train_idx],
            genes,
            VARIANCE_POOL,
        )

        selected_genes = genes[idx_pool]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(
            X_raw[train_idx][:, idx_pool]
        )
        Xte = scaler.transform(
            X_raw[test_idx][:, idx_pool]
        )

        fold_seed = MASTER_SEED + fold_no

        base_rows, base_preds = fit_baselines(
            Xtr,
            ytr,
            Xte,
            yte,
            fold_seed,
        )

        for model, chosen_k, metrics in base_rows:
            fold_rows.append(
                {
                    "fold": fold_no,
                    "repeat": (fold_no - 1) // N_SPLITS + 1,
                    "fold_within_repeat": (fold_no - 1) % N_SPLITS + 1,
                    "model": model,
                    "chosen_k": chosen_k,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_pos_train": int(np.sum(ytr == 1)),
                    "n_pos_test": int(np.sum(yte == 1)),
                    **metrics,
                }
            )

        for model, chosen_k, score, pred in base_preds:
            for j, sample_i in enumerate(test_idx):
                prediction_rows.append(
                    {
                        "fold": fold_no,
                        "repeat": (fold_no - 1) // N_SPLITS + 1,
                        "model": model,
                        "chosen_k": chosen_k,
                        "sample_id": str(sample_ids[sample_i]),
                        "y_true": int(y[sample_i]),
                        "score": float(score[j]),
                        "y_pred": int(pred[j]),
                    }
                )

        # Univariate nested K selection.
        best_k, inner_scores = choose_univariate_k(
            Xtr,
            ytr,
            seed=fold_seed,
        )

        f, _ = f_classif(Xtr, ytr)
        f = np.nan_to_num(
            f,
            nan=-np.inf,
            posinf=np.inf,
            neginf=-np.inf,
        )
        uni_idx = np.argsort(
            -f,
            kind="mergesort",
        )[:best_k]

        uni = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=fold_seed,
        )

        uni.fit(Xtr[:, uni_idx], ytr)
        score = uni.predict_proba(Xte[:, uni_idx])[:, 1]
        pred = (score >= 0.5).astype(int)
        metrics = model_metrics(yte, score, pred)

        fold_rows.append(
            {
                "fold": fold_no,
                "repeat": (fold_no - 1) // N_SPLITS + 1,
                "fold_within_repeat": (fold_no - 1) % N_SPLITS + 1,
                "model": "univariate_logistic",
                "chosen_k": best_k,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_pos_train": int(np.sum(ytr == 1)),
                "n_pos_test": int(np.sum(yte == 1)),
                **metrics,
            }
        )

        for j, sample_i in enumerate(test_idx):
            prediction_rows.append(
                {
                    "fold": fold_no,
                    "repeat": (fold_no - 1) // N_SPLITS + 1,
                    "model": "univariate_logistic",
                    "chosen_k": best_k,
                    "sample_id": str(sample_ids[sample_i]),
                    "y_true": int(y[sample_i]),
                    "score": float(score[j]),
                    "y_pred": int(pred[j]),
                }
            )

        split_manifest.append(
            {
                "fold": fold_no,
                "repeat": (fold_no - 1) // N_SPLITS + 1,
                "fold_within_repeat": (fold_no - 1) % N_SPLITS + 1,
                "train_samples": sample_ids[train_idx].astype(str).tolist(),
                "test_samples": sample_ids[test_idx].astype(str).tolist(),
                "top256_genes": selected_genes.astype(str).tolist(),
                "univariate_best_k": int(best_k),
                "univariate_inner_mean_pr_auc": {
                    str(k): float(v)
                    for k, v in inner_scores.items()
                },
            }
        )

        if fold_no % 10 == 0:
            print(
                f"      completed {fold_no}/"
                f"{N_SPLITS * N_REPEATS} outer folds",
                flush=True,
            )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(prediction_rows)

    print("[5/6] Summarizing baseline performance...")

    methods = [
        "logistic_l2",
        "logistic_l1",
        "linear_svm",
        "univariate_logistic",
    ]

    summary_rows = []

    for model in methods:
        sub = fold_df[fold_df["model"] == model].copy()

        row = {
            "model": model,
            "valid_folds": int(len(sub)),
        }

        for metric in [
            "pr_auc",
            "roc_auc",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
        ]:
            s = summarize_metric(sub, metric)
            for key, value in s.items():
                row[f"{metric}_{key}"] = value

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    outdir = project / "results" / "baseline"
    outdir.mkdir(parents=True, exist_ok=True)

    folds_path = outdir / "v42_2_fold_metrics.csv"
    preds_path = outdir / "v42_2_oof_predictions.csv"
    summary_csv = outdir / "v42_2_baseline_summary.csv"
    summary_txt = outdir / "v42_2_baseline_summary.txt"
    splits_path = outdir / "v42_2_cv_splits.json"
    manifest_path = outdir / "v42_2_manifest.json"

    fold_df.to_csv(folds_path, index=False)
    pred_df.to_csv(preds_path, index=False)
    summary_df.to_csv(summary_csv, index=False)

    splits_path.write_text(
        json.dumps(
            {
                "version": VERSION,
                "master_seed": MASTER_SEED,
                "n_splits": N_SPLITS,
                "n_repeats": N_REPEATS,
                "folds": split_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        f"=== Soft Spaces / CML {VERSION} CLASSICAL BASELINES ===",
        "",
        "DESIGN",
        "------",
        "Cohort: GSE130404",
        "Samples: 96 baseline CP-CML",
        "Positive: BCR-ABL1 >10% at 3 months, n=13",
        "Negative: BCR-ABL1 <10% at 3 months, n=83",
        f"Positive prevalence / random PR-AUC reference: {np.mean(y):.6f}",
        "CV: RepeatedStratifiedKFold 5 x 20 = 100 folds",
        "Variance pool: Top-256 raw training-fold variance",
        "Primary metric: PR-AUC",
        "Soft Spaces used: NO",
        "",
        "RESULTS",
        "-------",
    ]

    for _, r in summary_df.iterrows():
        lines += [
            "",
            str(r["model"]),
            f"  valid folds: {int(r['valid_folds'])}",
            (
                f"  PR-AUC: {r['pr_auc_mean']:.6f} "
                f"[{r['pr_auc_q025']:.6f}, {r['pr_auc_q975']:.6f}]"
            ),
            (
                f"  ROC-AUC: {r['roc_auc_mean']:.6f} "
                f"[{r['roc_auc_q025']:.6f}, {r['roc_auc_q975']:.6f}]"
            ),
            f"  Balanced accuracy: {r['balanced_accuracy_mean']:.6f}",
            f"  Sensitivity: {r['sensitivity_mean']:.6f}",
            f"  Specificity: {r['specificity_mean']:.6f}",
        ]

    k_counts = (
        fold_df.loc[
            fold_df["model"] == "univariate_logistic",
            "chosen_k"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    lines += [
        "",
        "UNIVARIATE K SELECTION",
        "----------------------",
    ]

    for k in K_GRID:
        lines.append(f"K={k}: {int(k_counts.get(k, 0))} folds")

    # Entry condition: at least one baseline measurably above random
    # on mean PR-AUC OR mean ROC-AUC > 0.5.
    best_pr = float(summary_df["pr_auc_mean"].max())
    best_roc = float(summary_df["roc_auc_mean"].max())
    random_pr = float(np.mean(y))

    entry = bool(
        best_pr > random_pr
        or best_roc > 0.5
    )

    lines += [
        "",
        "SOFT-SPACES ENTRY CONDITION",
        "---------------------------",
        f"Best mean PR-AUC: {best_pr:.6f}",
        f"Random PR-AUC reference: {random_pr:.6f}",
        f"Best mean ROC-AUC: {best_roc:.6f}",
        f"ENTRY CONDITION SATISFIED: {entry}",
        "",
        "Interpret conservatively because only 13 positive patients are available.",
    ]

    summary_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    script_path = Path(__file__).resolve()

    manifest = {
        "version": VERSION,
        "development_cohort": "GSE130404",
        "samples": int(len(y)),
        "negative_n": int(np.sum(y == 0)),
        "positive_n": int(np.sum(y == 1)),
        "positive_prevalence": float(np.mean(y)),
        "primary_metric": "PR-AUC",
        "master_seed": MASTER_SEED,
        "cv": {
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "outer_evaluations": N_SPLITS * N_REPEATS,
        },
        "variance_pool": VARIANCE_POOL,
        "univariate_k_grid": K_GRID,
        "softspaces_used": False,
        "entry_condition_satisfied": entry,
        "best_mean_pr_auc": best_pr,
        "best_mean_roc_auc": best_roc,
        "mapping": {
            "platform": GPL,
            "mapped_genes": int(X_gene.shape[1]),
            "annotation_columns": mapping_info,
        },
        "sha256": {
            "frozen_design": actual_design_sha,
            "series_matrix": sha256_file(series_path),
            "platform_table": sha256_file(gpl_path),
            "cv_splits": sha256_file(splits_path),
            "execution_script": sha256_file(script_path),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("[6/6] Complete.\n")
    print(summary_txt.read_text(encoding="utf-8"))
    print("Wrote:")
    for p in [
        folds_path,
        preds_path,
        summary_csv,
        summary_txt,
        splits_path,
        manifest_path,
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
