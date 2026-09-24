#!/usr/bin/env python3
"""
Soft Spaces / CML
v44.2 — Internal direct-subspace rank selection
using the frozen transport-aware gene universe from v44.1b.

Design
------
Development cohort:
    GSE130404

Outer CV:
    exact frozen v42.2 100 folds

Per outer training fold:
    1. restrict to v44.1b eligible cross-platform gene universe
    2. rank by RAW training-fold variance
    3. select Top-256
    4. StandardScaler fit on training only
    5. H = mean(xx^T | y=1) - mean(xx^T | y=0)
    6. symmetrize H
    7. eigendecompose
    8. direct projection for r in {4,8,16}
    9. balanced L2 logistic regression
   10. held-out evaluation

Primary rank-selection metric:
    mean held-out PR-AUC

Tie-break:
    smaller r

No matched NULL here.
No external outcome data here.
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


VERSION = "v44.2"
R_GRID = [4, 8, 16]
TOP_K = 256
MASTER_SEED = 4402001


def project_dir():
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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

    X_probe = pd.DataFrame(
        np.asarray(rows, dtype=np.float64).T,
        index=sample_ids,
        columns=probes,
    )

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
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    begin = next(
        i for i, x in enumerate(lines)
        if x.startswith("!platform_table_begin")
    )
    end = next(
        i for i, x in enumerate(lines)
        if x.startswith("!platform_table_end")
    )

    table = lines[begin + 1:end]
    header = table[0].split("\t")
    lut = {x.strip().lower(): x for x in header}

    id_col = lut.get("id")

    symbol_col = next(
        (
            lut[c]
            for c in [
                "symbol",
                "gene symbol",
                "gene_symbol",
                "genesymbol",
            ]
            if c in lut
        ),
        None,
    )

    if id_col is None or symbol_col is None:
        raise RuntimeError(
            f"Cannot identify GPL10558 mapping columns: {header}"
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

        raw = (
            raw.replace("///", "|")
            .replace(";", "|")
            .replace(",", "|")
        )

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

    if not out:
        raise RuntimeError("No gene mappings produced.")

    return pd.DataFrame(out, index=X_probe.index)


def make_y(meta):
    y = []

    for gsm, row in meta.iterrows():
        stage = str(row["disease_stage"]).strip().lower()
        resp = str(row["bcr_abl1_3m"]).strip()

        if stage != "diagnostic chronic phase":
            raise RuntimeError(
                f"{gsm}: unexpected stage {stage!r}"
            )

        if resp == ">10%":
            y.append(1)
        elif resp == "<10%":
            y.append(0)
        else:
            raise RuntimeError(
                f"{gsm}: unresolved response {resp!r}"
            )

    return np.asarray(y, dtype=int)


def class_contrast_operator(X, y):
    X0 = X[y == 0]
    X1 = X[y == 1]

    H0 = np.einsum("ni,nj->ij", X0, X0) / len(X0)
    H1 = np.einsum("ni,nj->ij", X1, X1) / len(X1)

    H = H1 - H0

    return 0.5 * (H + H.T)


def top_abs_eigenvectors(H, r):
    evals, U = np.linalg.eigh(H)
    order = np.argsort(np.abs(evals))[::-1]
    return U[:, order[:r]], evals[order]


def confusion_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    bal = balanced_accuracy_score(y_true, y_pred)

    return float(bal), float(sens), float(spec)


def evaluate(Xtr, ytr, Xte, yte, basis, seed):
    Ztr = Xtr @ basis
    Zte = Xte @ basis

    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
    )

    model.fit(Ztr, ytr)

    score = model.predict_proba(Zte)[:, 1]
    pred = (score >= 0.5).astype(int)

    bal, sens, spec = confusion_metrics(
        yte,
        pred,
    )

    return {
        "pr_auc": float(
            average_precision_score(yte, score)
        ),
        "roc_auc": float(
            roc_auc_score(yte, score)
        ),
        "balanced_accuracy": bal,
        "sensitivity": sens,
        "specificity": spec,
    }


def summarize(values):
    x = np.asarray(values, dtype=float)

    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "q025": float(np.quantile(x, 0.025)),
        "median": float(np.median(x)),
        "q975": float(np.quantile(x, 0.975)),
    }


def main():
    project = project_dir()
    docs = project / "docs"
    results = project / "results"

    protocol = (
        docs
        / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION.txt"
    )

    protocol_manifest = (
        docs
        / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION_manifest.json"
    )

    universe_manifest = (
        results
        / "direct_subspace"
        / "v44_1b_manifest.json"
    )

    universe_list = (
        results
        / "direct_subspace"
        / "v44_1b_eligible_genes.txt"
    )

    splits_path = (
        results
        / "baseline"
        / "v42_2_cv_splits.json"
    )

    for p in [
        protocol,
        protocol_manifest,
        universe_manifest,
        universe_list,
        splits_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    pm = json.loads(
        protocol_manifest.read_text(
            encoding="utf-8"
        )
    )

    expected_sha = pm["protocol_sha256"]
    actual_sha = sha256_file(protocol)

    print(
        "=== v44.2 CML TRANSPORT-AWARE DIRECT SUBSPACE RANK SELECTION ==="
    )
    print("Expected protocol SHA:", expected_sha)
    print("Actual protocol SHA:  ", actual_sha)

    if actual_sha != expected_sha:
        raise SystemExit(
            "FAIL: v44.0 protocol SHA mismatch."
        )

    um = json.loads(
        universe_manifest.read_text(
            encoding="utf-8"
        )
    )

    if um.get("status") != "PASS TO v44.2":
        raise RuntimeError(
            "v44.1b did not authorize v44.2."
        )

    eligible_genes = [
        x.strip()
        for x in universe_list.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]

    if len(eligible_genes) < TOP_K:
        raise RuntimeError(
            f"Eligible universe has only {len(eligible_genes)} genes."
        )

    print(
        "Transport-aware eligible universe:",
        len(eligible_genes),
    )
    print("External outcome data used: NO")
    print("Matched NULL: NOT YET")
    print()

    splits_doc = json.loads(
        splits_path.read_text(
            encoding="utf-8"
        )
    )

    folds = splits_doc.get("folds", [])

    if len(folds) != 100:
        raise RuntimeError(
            "Expected exactly 100 frozen v42.2 folds."
        )

    series_path = (
        project
        / "data"
        / "external"
        / "GSE130404"
        / "raw"
        / "GSE130404_series_matrix.txt.gz"
    )

    gpl_path = (
        project
        / "data"
        / "external"
        / "GSE130404"
        / "platform"
        / "GPL10558_full_geo_table.txt"
    )

    for p in [series_path, gpl_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    X_probe, meta = parse_gse130404(
        series_path
    )

    X_gene = aggregate_probe_to_gene(
        X_probe,
        parse_platform_mapping(gpl_path),
    )

    y = make_y(meta)

    sample_ids = np.asarray(
        X_gene.index.astype(str)
    )

    available = [
        g
        for g in eligible_genes
        if g in X_gene.columns
    ]

    if len(available) != len(eligible_genes):
        missing = sorted(
            set(eligible_genes)
            - set(available)
        )
        raise RuntimeError(
            "Eligible genes missing after development aggregation: "
            + ", ".join(missing[:20])
        )

    X_eligible = X_gene.loc[
        :,
        eligible_genes,
    ].to_numpy(
        dtype=np.float64
    )

    gsm_to_i = {
        gsm: i
        for i, gsm in enumerate(sample_ids)
    }

    gene_arr = np.asarray(
        eligible_genes,
        dtype=str,
    )

    rows = []
    pool_manifest = []

    for fold_rec in folds:
        fold = int(fold_rec["fold"])

        train_idx = np.asarray(
            [
                gsm_to_i[x]
                for x in fold_rec["train_samples"]
            ],
            dtype=int,
        )

        test_idx = np.asarray(
            [
                gsm_to_i[x]
                for x in fold_rec["test_samples"]
            ],
            dtype=int,
        )

        Xtr_raw_all = X_eligible[train_idx]
        Xte_raw_all = X_eligible[test_idx]

        # Raw training-fold variance.
        variances = np.var(
            Xtr_raw_all,
            axis=0,
            ddof=1,
        )

        # Deterministic sort:
        # descending variance, then ascending gene symbol.
        order = np.lexsort(
            (
                gene_arr,
                -variances,
            )
        )

        top_idx = order[:TOP_K]
        top_genes = gene_arr[top_idx].tolist()

        Xtr_raw = Xtr_raw_all[:, top_idx]
        Xte_raw = Xte_raw_all[:, top_idx]

        scaler = StandardScaler()

        Xtr = scaler.fit_transform(
            Xtr_raw
        )

        Xte = scaler.transform(
            Xte_raw
        )

        ytr = y[train_idx]
        yte = y[test_idx]

        H = class_contrast_operator(
            Xtr,
            ytr,
        )

        pool_manifest.append({
            "fold": fold,
            "repeat": fold_rec["repeat"],
            "fold_within_repeat": fold_rec["fold_within_repeat"],
            "top256_genes": top_genes,
        })

        for r in R_GRID:
            U_r, evals = top_abs_eigenvectors(
                H,
                r,
            )

            metrics = evaluate(
                Xtr,
                ytr,
                Xte,
                yte,
                U_r,
                seed=MASTER_SEED + 1000 * r + fold,
            )

            rows.append({
                "fold": fold,
                "repeat": fold_rec["repeat"],
                "fold_within_repeat": fold_rec["fold_within_repeat"],
                "r": r,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_pos_train": int(np.sum(ytr == 1)),
                "n_pos_test": int(np.sum(yte == 1)),
                "leading_abs_eigenvalue": float(
                    np.abs(evals[0])
                ),
                **metrics,
            })

        if fold % 10 == 0:
            print(
                f"completed {fold}/100 folds",
                flush=True,
            )

    df = pd.DataFrame(rows)

    summary_rows = []

    for r in R_GRID:
        sub = df[
            df["r"] == r
        ]

        row = {
            "r": r,
            "n_folds": len(sub),
        }

        for metric in [
            "pr_auc",
            "roc_auc",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
        ]:
            s = summarize(
                sub[metric].to_numpy()
            )

            for key, value in s.items():
                row[f"{metric}_{key}"] = value

        summary_rows.append(row)

    summary_df = pd.DataFrame(
        summary_rows
    )

    selected = summary_df.sort_values(
        ["pr_auc_mean", "r"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]

    selected_r = int(
        selected["r"]
    )

    outdir = (
        results
        / "direct_subspace"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds_out = (
        outdir
        / "v44_2_fold_metrics.csv"
    )

    rank_out = (
        outdir
        / "v44_2_rank_summary.csv"
    )

    pools_out = (
        outdir
        / "v44_2_fold_top256_manifest.json"
    )

    summary_out = (
        outdir
        / "v44_2_direct_subspace_summary.txt"
    )

    manifest_out = (
        outdir
        / "v44_2_manifest.json"
    )

    df.to_csv(
        folds_out,
        index=False,
    )

    summary_df.to_csv(
        rank_out,
        index=False,
    )

    pools_out.write_text(
        json.dumps(
            {
                "version": VERSION,
                "eligible_universe_n": len(eligible_genes),
                "top_k": TOP_K,
                "folds": pool_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "=== Soft Spaces / CML v44.2 TRANSPORT-AWARE DIRECT SUBSPACE RANK SELECTION ===",
        "",
        "DESIGN",
        "------",
        "Development cohort: GSE130404",
        f"Transport-aware eligible universe: {len(eligible_genes)} genes",
        "Fold-specific Top-256 by raw training variance: YES",
        "Frozen v42.2 100 folds reused: YES",
        "External response outcomes used: NO",
        "Matched NULL test: NOT YET",
        "",
        "RANK RESULTS",
        "------------",
    ]

    for _, x in summary_df.sort_values("r").iterrows():
        lines.append(
            f"r={int(x['r']):2d} | "
            f"PR-AUC {x['pr_auc_mean']:.6f} "
            f"[{x['pr_auc_q025']:.6f}, {x['pr_auc_q975']:.6f}] | "
            f"ROC-AUC {x['roc_auc_mean']:.6f} | "
            f"BA {x['balanced_accuracy_mean']:.6f} | "
            f"Sens {x['sensitivity_mean']:.6f} | "
            f"Spec {x['specificity_mean']:.6f}"
        )

    lines += [
        "",
        "DEVELOPMENT-ONLY SELECTION",
        "--------------------------",
        f"Selected r: {selected_r}",
        f"Selected mean PR-AUC: {float(selected['pr_auc_mean']):.6f}",
        f"Selected mean ROC-AUC: {float(selected['roc_auc_mean']):.6f}",
        f"Selected balanced accuracy: {float(selected['balanced_accuracy_mean']):.6f}",
        "",
        "IMPORTANT",
        "---------",
        "This is NOT a success claim.",
        "The selected rank must now be frozen and tested against matched",
        "random orthonormal subspaces in v44.3.",
    ]

    summary_out.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest_out.write_text(
        json.dumps(
            {
                "version": VERSION,
                "development_cohort": "GSE130404",
                "eligible_universe_n": len(eligible_genes),
                "top_k": TOP_K,
                "candidate_ranks": R_GRID,
                "selected_r": selected_r,
                "selected_mean_pr_auc": float(
                    selected["pr_auc_mean"]
                ),
                "selected_mean_roc_auc": float(
                    selected["roc_auc_mean"]
                ),
                "selected_balanced_accuracy": float(
                    selected[
                        "balanced_accuracy_mean"
                    ]
                ),
                "external_outcomes_used": False,
                "matched_null_tested": False,
                "sha256": {
                    "v44_0_protocol": actual_sha,
                    "v44_1b_eligible_genes": sha256_file(
                        universe_list
                    ),
                    "v42_2_cv_splits": sha256_file(
                        splits_path
                    ),
                    "fold_top256_manifest": sha256_file(
                        pools_out
                    ),
                    "execution_script": sha256_file(
                        Path(__file__).resolve()
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        summary_out.read_text(
            encoding="utf-8"
        )
    )

    print("Wrote:")
    for p in [
        folds_out,
        rank_out,
        pools_out,
        summary_out,
        manifest_out,
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
