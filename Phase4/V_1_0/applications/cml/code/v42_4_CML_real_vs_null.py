#!/usr/bin/env python3
"""
Soft Spaces / CML
v42.4 — Frozen REAL vs matched NULL validation

Frozen from v42.3:
    r = 4
    K = 32

Development cohort:
    GSE130404

Primary metric:
    mean held-out PR-AUC across the exact frozen v42.2 folds

Matched NULL:
    random K=32 panels sampled from the SAME fold-specific Top-256
    raw-variance pool used by REAL.

Each NULL realization:
    - uses the same 100 outer folds
    - uses the same train/test samples
    - uses the same scaler fit on each training fold
    - uses the same balanced logistic-regression evaluation machinery
    - differs only in feature-panel selection

Primary decision rule:
    REAL mean PR-AUC > NULL mean PR-AUC
    AND empirical one-sided p <= 0.05

This stage is INTERNAL DEVELOPMENT validation only.
No external cohort is used.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


VERSION = "v42.4"

EXPECTED_DESIGN_SHA256 = (
    "3d397a9a5edec39c4809abf95daf031960d6f4caf14314492c17f3606b480fe2"
)

FROZEN_R = 4
FROZEN_K = 32
VARIANCE_POOL = 256
N_NULL = 1000
MASTER_SEED = 4204001


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
        raise RuntimeError(f"Cannot identify GPL10558 mapping columns: {header}")

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
            raise RuntimeError(f"{gsm}: unexpected disease stage {stage!r}")

        if resp == ">10%":
            y.append(1)
        elif resp == "<10%":
            y.append(0)
        else:
            raise RuntimeError(f"{gsm}: unresolved outcome {resp!r}")

    return np.asarray(y, dtype=int)


def class_contrast_operator(X, y):
    X0 = X[y == 0]
    X1 = X[y == 1]

    H0 = np.einsum("ni,nj->ij", X0, X0) / len(X0)
    H1 = np.einsum("ni,nj->ij", X1, X1) / len(X1)

    H = H1 - H0
    return 0.5 * (H + H.T)


def softspaces_panel(Xtr, ytr, genes, r, k):
    H = class_contrast_operator(Xtr, ytr)

    evals, U = np.linalg.eigh(H)
    order = np.argsort(np.abs(evals))[::-1]
    U = U[:, order[:r]]

    p_diag = np.sum(U * U, axis=1)
    coupling = p_diag * (1.0 - p_diag)

    rank = np.lexsort((np.asarray(genes).astype(str), -coupling))
    return rank[:k]


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

    return (
        float(average_precision_score(yte, score)),
        float(roc_auc_score(yte, score)),
    )


def empirical_p(real, null):
    null = np.asarray(null, dtype=float)
    return float((1 + np.sum(null >= real)) / (1 + len(null)))


def main():
    project = project_dir_from_script()

    design = find_design(project)
    design_sha = sha256_file(design)

    if design_sha != EXPECTED_DESIGN_SHA256:
        raise SystemExit("FAIL: v42.1d frozen design SHA mismatch.")

    baseline_dir = project / "results" / "baseline"
    soft_dir = project / "results" / "softspaces"

    splits_path = baseline_dir / "v42_2_cv_splits.json"
    v423_manifest_path = soft_dir / "v42_3_manifest.json"

    for p in [splits_path, v423_manifest_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    v423_manifest = json.loads(
        v423_manifest_path.read_text(encoding="utf-8")
    )

    if int(v423_manifest["selected_r"]) != FROZEN_R:
        raise RuntimeError("v42.3 selected_r mismatch.")
    if int(v423_manifest["selected_k"]) != FROZEN_K:
        raise RuntimeError("v42.3 selected_k mismatch.")

    splits_doc = json.loads(
        splits_path.read_text(encoding="utf-8")
    )

    series_path = (
        project / "data" / "external" / "GSE130404" / "raw"
        / "GSE130404_series_matrix.txt.gz"
    )
    gpl_path = (
        project / "data" / "external" / "GSE130404" / "platform"
        / "GPL10558_full_geo_table.txt"
    )

    print(f"=== {VERSION} CML REAL vs MATCHED NULL ===")
    print(f"Frozen r={FROZEN_R}, K={FROZEN_K}")
    print(f"Matched NULL realizations: {N_NULL}")
    print("External cohort used: NO")
    print()

    X_probe, meta = parse_gse130404(series_path)
    X_gene = aggregate_probe_to_gene(
        X_probe,
        parse_platform_mapping(gpl_path),
    )

    y = make_y(meta)
    sample_ids = np.asarray(X_gene.index.astype(str))
    genes_all = np.asarray(X_gene.columns, dtype=object)
    X_raw = X_gene.to_numpy(dtype=np.float64)
    gsm_to_i = {gsm: i for i, gsm in enumerate(sample_ids)}

    if len(splits_doc["folds"]) != 100:
        raise RuntimeError("Expected 100 frozen folds.")

    # Precompute each frozen fold once.
    fold_cache = []
    real_fold_rows = []

    for fold_rec in splits_doc["folds"]:
        fold = int(fold_rec["fold"])
        train_idx = np.asarray(
            [gsm_to_i[x] for x in fold_rec["train_samples"]],
            dtype=int,
        )
        test_idx = np.asarray(
            [gsm_to_i[x] for x in fold_rec["test_samples"]],
            dtype=int,
        )

        frozen_pool_genes = fold_rec["top256_genes"]

        gene_to_global = {g: i for i, g in enumerate(genes_all.astype(str))}
        try:
            pool_global_idx = np.asarray(
                [gene_to_global[g] for g in frozen_pool_genes],
                dtype=int,
            )
        except KeyError as e:
            raise RuntimeError(f"Frozen Top-256 gene missing: {e}")

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(
            X_raw[train_idx][:, pool_global_idx]
        )
        Xte = scaler.transform(
            X_raw[test_idx][:, pool_global_idx]
        )

        ytr = y[train_idx]
        yte = y[test_idx]
        pool_genes = np.asarray(frozen_pool_genes, dtype=object)

        real_idx = softspaces_panel(
            Xtr,
            ytr,
            pool_genes,
            FROZEN_R,
            FROZEN_K,
        )

        real_pr, real_roc = evaluate_panel(
            Xtr,
            ytr,
            Xte,
            yte,
            real_idx,
            seed=MASTER_SEED + fold,
        )

        real_fold_rows.append({
            "fold": fold,
            "repeat": fold_rec["repeat"],
            "fold_within_repeat": fold_rec["fold_within_repeat"],
            "real_pr_auc": real_pr,
            "real_roc_auc": real_roc,
            "real_panel": "|".join(pool_genes[real_idx].astype(str)),
        })

        fold_cache.append({
            "fold": fold,
            "Xtr": Xtr,
            "Xte": Xte,
            "ytr": ytr,
            "yte": yte,
        })

    real_df = pd.DataFrame(real_fold_rows)
    real_mean_pr = float(real_df["real_pr_auc"].mean())
    real_mean_roc = float(real_df["real_roc_auc"].mean())

    print("REAL mean PR-AUC:", f"{real_mean_pr:.6f}")
    print("REAL mean ROC-AUC:", f"{real_mean_roc:.6f}")
    print()

    rng = np.random.default_rng(MASTER_SEED)

    null_rows = []

    for null_rep in range(1, N_NULL + 1):
        fold_pr = []
        fold_roc = []

        for fc in fold_cache:
            # Matched random K-panel from the same 256 coordinates.
            idx = rng.choice(
                VARIANCE_POOL,
                size=FROZEN_K,
                replace=False,
            )

            pr, roc = evaluate_panel(
                fc["Xtr"],
                fc["ytr"],
                fc["Xte"],
                fc["yte"],
                idx,
                seed=MASTER_SEED + 100000 * null_rep + fc["fold"],
            )

            fold_pr.append(pr)
            fold_roc.append(roc)

        null_rows.append({
            "null_rep": null_rep,
            "mean_pr_auc": float(np.mean(fold_pr)),
            "mean_roc_auc": float(np.mean(fold_roc)),
        })

        if null_rep % 50 == 0:
            print(f"NULL {null_rep}/{N_NULL}", flush=True)

    null_df = pd.DataFrame(null_rows)

    null_mean_pr = float(null_df["mean_pr_auc"].mean())
    null_mean_roc = float(null_df["mean_roc_auc"].mean())

    p_pr = empirical_p(
        real_mean_pr,
        null_df["mean_pr_auc"].to_numpy(),
    )
    p_roc = empirical_p(
        real_mean_roc,
        null_df["mean_roc_auc"].to_numpy(),
    )

    primary_pass = bool(
        real_mean_pr > null_mean_pr
        and p_pr <= 0.05
    )

    status = "PASS" if primary_pass else "FAIL"

    outdir = project / "results" / "softspaces"
    outdir.mkdir(parents=True, exist_ok=True)

    real_path = outdir / "v42_4_real_fold_metrics.csv"
    null_path = outdir / "v42_4_null_summary.csv"
    summary_path = outdir / "v42_4_real_null_summary.txt"
    manifest_path = outdir / "v42_4_manifest.json"

    real_df.to_csv(real_path, index=False)
    null_df.to_csv(null_path, index=False)

    lines = [
        f"=== Soft Spaces / CML {VERSION} REAL vs MATCHED NULL ===",
        "",
        "FROZEN DESIGN",
        "-------------",
        "Development cohort: GSE130404",
        f"Frozen Soft-Spaces rank r: {FROZEN_R}",
        f"Frozen panel size K: {FROZEN_K}",
        "Frozen v42.2 folds reused: YES",
        f"Matched NULL realizations: {N_NULL}",
        "External cohort used: NO",
        "",
        "PRIMARY: MEAN HELD-OUT PR-AUC",
        "-----------------------------",
        f"REAL:      {real_mean_pr:.6f}",
        f"NULL mean: {null_mean_pr:.6f}",
        f"Delta:     {real_mean_pr - null_mean_pr:+.6f}",
        f"NULL 2.5%: {float(null_df['mean_pr_auc'].quantile(0.025)):.6f}",
        f"NULL 97.5%:{float(null_df['mean_pr_auc'].quantile(0.975)):.6f}",
        f"Empirical one-sided p: {p_pr:.6g}",
        "",
        "SECONDARY: MEAN HELD-OUT ROC-AUC",
        "---------------------------------",
        f"REAL:      {real_mean_roc:.6f}",
        f"NULL mean: {null_mean_roc:.6f}",
        f"Delta:     {real_mean_roc - null_mean_roc:+.6f}",
        f"Empirical one-sided p: {p_roc:.6g}",
        "",
        "DEVELOPMENT DECISION",
        "--------------------",
        "PASS rule:",
        "REAL mean PR-AUC > NULL mean PR-AUC",
        "AND empirical one-sided p <= 0.05",
        "",
        f"v42.4 STATUS: {status}",
        "",
        "INTERPRETATION",
        "--------------",
        "PASS supports only an INTERNAL development-stage Soft-Spaces",
        "feature-selection effect relative to matched random panels.",
        "",
        "It does not establish external replication, clinical utility,",
        "causal biology, or superiority to the best classical baseline.",
    ]

    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "version": VERSION,
        "development_cohort": "GSE130404",
        "selected_r": FROZEN_R,
        "selected_k": FROZEN_K,
        "n_null": N_NULL,
        "master_seed": MASTER_SEED,
        "primary_metric": "mean held-out PR-AUC",
        "real_mean_pr_auc": real_mean_pr,
        "null_mean_pr_auc": null_mean_pr,
        "delta_pr_auc": real_mean_pr - null_mean_pr,
        "empirical_p_pr_auc": p_pr,
        "real_mean_roc_auc": real_mean_roc,
        "null_mean_roc_auc": null_mean_roc,
        "delta_roc_auc": real_mean_roc - null_mean_roc,
        "empirical_p_roc_auc": p_roc,
        "status": status,
        "external_used": False,
        "sha256": {
            "frozen_design": design_sha,
            "v42_2_cv_splits": sha256_file(splits_path),
            "v42_3_manifest": sha256_file(v423_manifest_path),
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
    print(summary_path.read_text(encoding="utf-8"))
    print("Wrote:")
    for p in [real_path, null_path, summary_path, manifest_path]:
        print(" ", p)


if __name__ == "__main__":
    main()
