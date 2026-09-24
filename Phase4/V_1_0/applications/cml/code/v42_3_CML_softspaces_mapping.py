#!/usr/bin/env python3
"""
Soft Spaces / CML
v42.3 — Development-only Soft-Spaces mapping

Purpose
-------
Apply the Soft-Spaces construction to GSE130404 using the EXACT frozen
v42.2 CV splits and preprocessing structure.

This stage is DEVELOPMENT ONLY.

It does:
- reuse v42.2 frozen CV splits;
- use Top-256 raw-variance training-fold genes;
- build class-contrast operator H from training data only;
- evaluate candidate subspace ranks r in {4,8,16};
- evaluate candidate panel sizes K in {8,16,32};
- score selected panels on held-out samples using balanced logistic regression;
- select r,K using development-only performance summaries.

It does NOT:
- use GSE44589;
- perform the final matched NULL test;
- make an external claim.

The selected r,K from this stage will be frozen for v42.4.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


VERSION = "v42.3"

EXPECTED_DESIGN_SHA256 = (
    "3d397a9a5edec39c4809abf95daf031960d6f4caf14314492c17f3606b480fe2"
)

R_GRID = [4, 8, 16]
K_GRID = [8, 16, 32]
VARIANCE_POOL = 256
MASTER_SEED = 4203001


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


def parse_tab_line(line: str):
    return [x.strip().strip('"') for x in line.rstrip("\r\n").split("\t")]


def parse_gse130404(path: Path):
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
                rows.append([float(x.strip().strip('"')) for x in parts[1:]])

    sample_ids = header[1:]
    M = np.asarray(rows, dtype=np.float64)

    X_probe = pd.DataFrame(M.T, index=sample_ids, columns=probes)

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

        parsed.append({
            "geo_accession": row["geo_accession"],
            "disease_stage": chars.get("disease stage", ""),
            "bcr_abl1_3m": chars.get("bcr-abl1 at 3 month", ""),
        })

    meta = pd.DataFrame(parsed).set_index("geo_accession")
    return X_probe, meta


def parse_platform_mapping(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    begin = next(i for i, x in enumerate(lines) if x.startswith("!platform_table_begin"))
    end = next(i for i, x in enumerate(lines) if x.startswith("!platform_table_end"))

    table = lines[begin + 1:end]
    header = table[0].split("\t")
    lut = {x.strip().lower(): x for x in header}

    id_col = lut.get("id")
    symbol_col = next(
        (lut[c] for c in ["symbol", "gene symbol", "gene_symbol", "genesymbol"] if c in lut),
        None,
    )

    if id_col is None or symbol_col is None:
        raise RuntimeError(f"Cannot identify mapping columns: {header}")

    probe_to_symbols = defaultdict(set)

    reader = csv.DictReader(table[1:], fieldnames=header, delimiter="\t")
    for row in reader:
        probe = str(row.get(id_col, "")).strip()
        raw = str(row.get(symbol_col, "")).strip()

        if not probe or not raw:
            continue

        raw = raw.replace("///", "|").replace(";", "|").replace(",", "|")
        for sym in raw.split("|"):
            sym = sym.strip()
            if sym and sym.upper() not in {"---", "NA", "N/A", "NULL"}:
                probe_to_symbols[probe].add(sym)

    return dict(probe_to_symbols)


def aggregate_probe_to_gene(X_probe, probe_to_symbols):
    gene_to_probes = defaultdict(list)

    for probe in X_probe.columns:
        for sym in probe_to_symbols.get(probe, []):
            gene_to_probes[sym].append(probe)

    out = {}
    for gene, probes in gene_to_probes.items():
        out[gene] = X_probe.loc[:, probes].mean(axis=1)

    return pd.DataFrame(out, index=X_probe.index)


def make_y(meta):
    y = []

    for gsm, row in meta.iterrows():
        stage = str(row["disease_stage"]).strip().lower()
        resp = str(row["bcr_abl1_3m"]).strip()

        if stage != "diagnostic chronic phase":
            raise RuntimeError(f"{gsm}: unexpected stage {stage}")

        if resp == ">10%":
            y.append(1)
        elif resp == "<10%":
            y.append(0)
        else:
            raise RuntimeError(f"{gsm}: unresolved response {resp!r}")

    return np.asarray(y, dtype=int)


def top_variance_indices(X_train_raw, genes, k):
    var = np.var(X_train_raw, axis=0, ddof=1)
    genes = np.asarray(genes, dtype=object)
    order = np.lexsort((genes.astype(str), -var))
    return order[:min(k, len(order))]


def class_contrast_operator(X, y):
    X0 = X[y == 0]
    X1 = X[y == 1]

    if len(X0) == 0 or len(X1) == 0:
        raise RuntimeError("Missing class in training fold.")

    H0 = np.mean([np.outer(x, x) for x in X0], axis=0)
    H1 = np.mean([np.outer(x, x) for x in X1], axis=0)

    H = H1 - H0
    return 0.5 * (H + H.T)


def softspaces_scores(H, r):
    evals, U = np.linalg.eigh(H)
    order = np.argsort(np.abs(evals))[::-1]
    U = U[:, order[:r]]

    p_diag = np.sum(U * U, axis=1)
    coupling = p_diag * (1.0 - p_diag)

    return coupling, evals[order]


def stable_rank(scores, genes):
    scores = np.asarray(scores)
    genes = np.asarray(genes, dtype=object)
    return np.lexsort((genes.astype(str), -scores))


def confusion_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    bal = balanced_accuracy_score(y_true, y_pred)
    return float(bal), float(sens), float(spec)


def evaluate_panel(Xtr, ytr, Xte, yte, idx, seed):
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
    )

    model.fit(Xtr[:, idx], ytr)
    score = model.predict_proba(Xte[:, idx])[:, 1]
    pred = (score >= 0.5).astype(int)

    bal, sens, spec = confusion_metrics(yte, pred)

    return {
        "pr_auc": float(average_precision_score(yte, score)),
        "roc_auc": float(roc_auc_score(yte, score)),
        "balanced_accuracy": bal,
        "sensitivity": sens,
        "specificity": spec,
    }


def main():
    project = project_dir_from_script()

    design_path = find_design(project)
    design_sha = sha256_file(design_path)

    if design_sha != EXPECTED_DESIGN_SHA256:
        raise SystemExit("FAIL: frozen v42.1d design SHA mismatch.")

    baseline_dir = project / "results" / "baseline"
    splits_path = baseline_dir / "v42_2_cv_splits.json"

    if not splits_path.exists():
        raise FileNotFoundError(splits_path)

    splits_doc = json.loads(splits_path.read_text(encoding="utf-8"))

    series_path = (
        project / "data" / "external" / "GSE130404" / "raw"
        / "GSE130404_series_matrix.txt.gz"
    )
    gpl_path = (
        project / "data" / "external" / "GSE130404" / "platform"
        / "GPL10558_full_geo_table.txt"
    )

    for p in [series_path, gpl_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    print(f"=== {VERSION} CML SOFT-SPACES DEVELOPMENT MAPPING ===")
    print("Frozen design SHA:", design_sha)
    print("Reusing v42.2 CV split manifest.")
    print("External cohort used: NO")
    print("Matched NULL test: NOT YET")
    print()

    X_probe, meta = parse_gse130404(series_path)
    probe_to_symbols = parse_platform_mapping(gpl_path)
    X_gene = aggregate_probe_to_gene(X_probe, probe_to_symbols)

    y = make_y(meta)

    genes_all = np.asarray(X_gene.columns, dtype=object)
    X_raw = X_gene.to_numpy(dtype=np.float64)
    sample_ids = np.asarray(X_gene.index.astype(str))

    if len(splits_doc["folds"]) != 100:
        raise RuntimeError("Expected exactly 100 frozen v42.2 folds.")

    gsm_to_i = {gsm: i for i, gsm in enumerate(sample_ids)}

    rows = []
    panel_rows = []

    for fold_rec in splits_doc["folds"]:
        fold = int(fold_rec["fold"])
        train_ids = fold_rec["train_samples"]
        test_ids = fold_rec["test_samples"]

        train_idx = np.asarray([gsm_to_i[x] for x in train_ids], dtype=int)
        test_idx = np.asarray([gsm_to_i[x] for x in test_ids], dtype=int)

        ytr = y[train_idx]
        yte = y[test_idx]

        pool_idx = top_variance_indices(
            X_raw[train_idx],
            genes_all,
            VARIANCE_POOL,
        )

        pool_genes = genes_all[pool_idx]

        # Verify exact same Top-256 as v42.2.
        frozen_pool = fold_rec["top256_genes"]
        if pool_genes.astype(str).tolist() != frozen_pool:
            raise RuntimeError(
                f"Fold {fold}: Top-256 pool mismatch from v42.2."
            )

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_raw[train_idx][:, pool_idx])
        Xte = scaler.transform(X_raw[test_idx][:, pool_idx])

        H = class_contrast_operator(Xtr, ytr)

        for r in R_GRID:
            coupling, evals = softspaces_scores(H, r)
            rank_idx = stable_rank(coupling, pool_genes)

            for k in K_GRID:
                chosen = rank_idx[:k]
                metrics = evaluate_panel(
                    Xtr,
                    ytr,
                    Xte,
                    yte,
                    chosen,
                    seed=MASTER_SEED + fold + 1000 * r + k,
                )

                rows.append({
                    "fold": fold,
                    "repeat": fold_rec["repeat"],
                    "fold_within_repeat": fold_rec["fold_within_repeat"],
                    "r": r,
                    "k": k,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_pos_train": int(np.sum(ytr == 1)),
                    "n_pos_test": int(np.sum(yte == 1)),
                    **metrics,
                })

                for position, idx in enumerate(chosen, 1):
                    panel_rows.append({
                        "fold": fold,
                        "r": r,
                        "k": k,
                        "rank": position,
                        "gene": str(pool_genes[idx]),
                        "coupling_score": float(coupling[idx]),
                    })

        if fold % 10 == 0:
            print(f"completed {fold}/100 folds", flush=True)

    df = pd.DataFrame(rows)
    panels = pd.DataFrame(panel_rows)

    summary_rows = []

    for r in R_GRID:
        for k in K_GRID:
            sub = df[(df["r"] == r) & (df["k"] == k)]

            summary_rows.append({
                "r": r,
                "k": k,
                "n_folds": int(len(sub)),
                "pr_auc_mean": float(sub["pr_auc"].mean()),
                "pr_auc_sd": float(sub["pr_auc"].std(ddof=1)),
                "pr_auc_q025": float(sub["pr_auc"].quantile(0.025)),
                "pr_auc_q975": float(sub["pr_auc"].quantile(0.975)),
                "roc_auc_mean": float(sub["roc_auc"].mean()),
                "balanced_accuracy_mean": float(sub["balanced_accuracy"].mean()),
                "sensitivity_mean": float(sub["sensitivity"].mean()),
                "specificity_mean": float(sub["specificity"].mean()),
            })

    summary = pd.DataFrame(summary_rows)

    # Frozen selection rule for v42.3:
    # highest mean PR-AUC, tie -> smaller r, then smaller K.
    order = summary.sort_values(
        ["pr_auc_mean", "r", "k"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    best = order.iloc[0]

    best_r = int(best["r"])
    best_k = int(best["k"])

    outdir = project / "results" / "softspaces"
    outdir.mkdir(parents=True, exist_ok=True)

    folds_csv = outdir / "v42_3_fold_metrics.csv"
    panels_csv = outdir / "v42_3_selected_panels.csv"
    summary_csv = outdir / "v42_3_grid_summary.csv"
    summary_txt = outdir / "v42_3_softspaces_mapping_summary.txt"
    manifest_path = outdir / "v42_3_manifest.json"

    df.to_csv(folds_csv, index=False)
    panels.to_csv(panels_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    lines = [
        f"=== Soft Spaces / CML {VERSION} DEVELOPMENT MAPPING ===",
        "",
        "DESIGN",
        "------",
        "Development cohort: GSE130404",
        "External cohort used: NO",
        "Frozen v42.2 splits reused: YES",
        "Top-256 raw training-fold variance pool reused: YES",
        "Soft-Spaces r grid: 4, 8, 16",
        "Panel K grid: 8, 16, 32",
        "Primary selection metric: mean held-out PR-AUC",
        "Matched NULL test: NOT YET",
        "",
        "GRID RESULTS",
        "------------",
    ]

    for _, x in summary.sort_values(["r", "k"]).iterrows():
        lines += [
            (
                f"r={int(x['r']):2d} K={int(x['k']):2d} | "
                f"PR-AUC {x['pr_auc_mean']:.6f} | "
                f"ROC-AUC {x['roc_auc_mean']:.6f} | "
                f"BA {x['balanced_accuracy_mean']:.6f} | "
                f"Sens {x['sensitivity_mean']:.6f} | "
                f"Spec {x['specificity_mean']:.6f}"
            )
        ]

    lines += [
        "",
        "DEVELOPMENT-ONLY SELECTION",
        "--------------------------",
        f"Selected r: {best_r}",
        f"Selected K: {best_k}",
        f"Selected mean PR-AUC: {float(best['pr_auc_mean']):.6f}",
        f"Selected mean ROC-AUC: {float(best['roc_auc_mean']):.6f}",
        f"Selected balanced accuracy: {float(best['balanced_accuracy_mean']):.6f}",
        "",
        "IMPORTANT",
        "---------",
        "This is NOT yet a Soft-Spaces success claim.",
        "The selected r,K must now be frozen and tested against matched NULL",
        "in v42.4 on the same frozen folds.",
    ]

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "version": VERSION,
        "development_cohort": "GSE130404",
        "external_used": False,
        "matched_null_tested": False,
        "r_grid": R_GRID,
        "k_grid": K_GRID,
        "selection_rule": "max mean held-out PR-AUC; tie smaller r then smaller K",
        "selected_r": best_r,
        "selected_k": best_k,
        "selected_mean_pr_auc": float(best["pr_auc_mean"]),
        "selected_mean_roc_auc": float(best["roc_auc_mean"]),
        "sha256": {
            "frozen_design": design_sha,
            "v42_2_cv_splits": sha256_file(splits_path),
            "series_matrix": sha256_file(series_path),
            "platform_table": sha256_file(gpl_path),
            "execution_script": sha256_file(Path(__file__).resolve()),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print(summary_txt.read_text(encoding="utf-8"))
    print("Wrote:")
    for p in [folds_csv, panels_csv, summary_csv, summary_txt, manifest_path]:
        print(" ", p)


if __name__ == "__main__":
    main()
