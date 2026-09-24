#!/usr/bin/env python3
"""
Soft Spaces / CML
v44.3 — Transport-aware direct-subspace REAL vs matched random-subspace NULL

Frozen from v44.2:
    r = 16

Development cohort:
    GSE130404

Gene universe:
    v44.1b transport-aware eligible universe

Per-fold support:
    exact frozen v44.2 fold-specific Top-256 pools

Evaluation:
    exact frozen v42.2 100 CV folds

REAL:
    H = mean(xx^T | y=1) - mean(xx^T | y=0)
    top-|eigenvalue| r=16 eigenvectors
    direct projection
    balanced L2 logistic regression

NULL:
    random orthonormal 16D subspaces in the SAME fold-specific 256D space
    same preprocessing, classifier and held-out samples

Primary:
    mean held-out PR-AUC

PASS iff:
    REAL mean PR-AUC > NULL mean PR-AUC
    AND empirical one-sided p <= 0.05

No external outcomes are used.
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


VERSION = "v44.3"
FROZEN_R = 16
N_NULL = 1000
MASTER_SEED = 4403001


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
        raise RuntimeError(f"Cannot identify platform columns: {header}")

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

    if not out:
        raise RuntimeError("No gene mappings produced.")

    return pd.DataFrame(out, index=X_probe.index)


def make_y(meta):
    y = []

    for gsm, row in meta.iterrows():
        stage = str(row["disease_stage"]).strip().lower()
        resp = str(row["bcr_abl1_3m"]).strip()

        if stage != "diagnostic chronic phase":
            raise RuntimeError(f"{gsm}: unexpected stage {stage!r}")

        if resp == ">10%":
            y.append(1)
        elif resp == "<10%":
            y.append(0)
        else:
            raise RuntimeError(f"{gsm}: unresolved response {resp!r}")

    return np.asarray(y, dtype=int)


def class_contrast_operator(X, y):
    X0 = X[y == 0]
    X1 = X[y == 1]

    H0 = np.einsum("ni,nj->ij", X0, X0) / len(X0)
    H1 = np.einsum("ni,nj->ij", X1, X1) / len(X1)

    H = H1 - H0
    return 0.5 * (H + H.T)


def real_basis(Xtr, ytr, r):
    H = class_contrast_operator(Xtr, ytr)
    evals, U = np.linalg.eigh(H)
    order = np.argsort(np.abs(evals))[::-1]
    return U[:, order[:r]]


def random_basis(rng, ambient_dim, r):
    A = rng.normal(size=(ambient_dim, r))
    Q, _ = np.linalg.qr(A, mode="reduced")
    return Q[:, :r]


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

    return (
        float(average_precision_score(yte, score)),
        float(roc_auc_score(yte, score)),
    )


def empirical_p(real, null):
    null = np.asarray(null, dtype=float)
    return float((1 + np.sum(null >= real)) / (1 + len(null)))


def main():
    project = project_dir()
    docs = project / "docs"
    results = project / "results"
    ds = results / "direct_subspace"

    protocol = docs / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION.txt"
    protocol_manifest = docs / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION_manifest.json"
    v442_manifest = ds / "v44_2_manifest.json"
    pools_path = ds / "v44_2_fold_top256_manifest.json"
    splits_path = results / "baseline" / "v42_2_cv_splits.json"

    for p in [protocol, protocol_manifest, v442_manifest, pools_path, splits_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    pm = json.loads(protocol_manifest.read_text(encoding="utf-8"))
    expected_sha = pm["protocol_sha256"]
    actual_sha = sha256_file(protocol)

    print("=== v44.3 CML TRANSPORT-AWARE REAL vs MATCHED NULL ===")
    print("Expected protocol SHA:", expected_sha)
    print("Actual protocol SHA:  ", actual_sha)

    if expected_sha != actual_sha:
        raise SystemExit("FAIL: v44.0 protocol SHA mismatch.")

    m442 = json.loads(v442_manifest.read_text(encoding="utf-8"))

    if int(m442["selected_r"]) != FROZEN_R:
        raise RuntimeError(
            f"v44.2 selected r={m442['selected_r']}, expected frozen r={FROZEN_R}"
        )

    print("PASS: v44.0 protocol verified.")
    print(f"Frozen rank r={FROZEN_R}")
    print(f"NULL realizations={N_NULL}")
    print("External outcome data used: NO")
    print()

    pools_doc = json.loads(pools_path.read_text(encoding="utf-8"))
    pool_by_fold = {
        int(rec["fold"]): rec["top256_genes"]
        for rec in pools_doc["folds"]
    }

    splits_doc = json.loads(splits_path.read_text(encoding="utf-8"))

    if len(splits_doc.get("folds", [])) != 100:
        raise RuntimeError("Expected 100 frozen folds.")

    series_path = (
        project / "data" / "external" / "GSE130404"
        / "raw" / "GSE130404_series_matrix.txt.gz"
    )

    gpl_path = (
        project / "data" / "external" / "GSE130404"
        / "platform" / "GPL10558_full_geo_table.txt"
    )

    X_probe, meta = parse_gse130404(series_path)
    X_gene = aggregate_probe_to_gene(
        X_probe,
        parse_platform_mapping(gpl_path),
    )
    y = make_y(meta)

    sample_ids = np.asarray(X_gene.index.astype(str))
    gsm_to_i = {gsm: i for i, gsm in enumerate(sample_ids)}
    gene_to_i = {g: i for i, g in enumerate(X_gene.columns.astype(str))}
    X_raw = X_gene.to_numpy(dtype=np.float64)

    fold_cache = []
    real_rows = []

    for fold_rec in splits_doc["folds"]:
        fold = int(fold_rec["fold"])

        if fold not in pool_by_fold:
            raise RuntimeError(f"Missing v44.2 Top-256 pool for fold {fold}")

        genes = pool_by_fold[fold]

        train_idx = np.asarray(
            [gsm_to_i[x] for x in fold_rec["train_samples"]],
            dtype=int,
        )
        test_idx = np.asarray(
            [gsm_to_i[x] for x in fold_rec["test_samples"]],
            dtype=int,
        )

        pool_idx = np.asarray(
            [gene_to_i[g] for g in genes],
            dtype=int,
        )

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_raw[train_idx][:, pool_idx])
        Xte = scaler.transform(X_raw[test_idx][:, pool_idx])

        ytr = y[train_idx]
        yte = y[test_idx]

        U = real_basis(Xtr, ytr, FROZEN_R)

        pr, roc = evaluate(
            Xtr, ytr, Xte, yte, U,
            seed=MASTER_SEED + fold,
        )

        real_rows.append({
            "fold": fold,
            "pr_auc": pr,
            "roc_auc": roc,
        })

        fold_cache.append({
            "fold": fold,
            "Xtr": Xtr,
            "Xte": Xte,
            "ytr": ytr,
            "yte": yte,
        })

    real_df = pd.DataFrame(real_rows)
    real_pr = float(real_df["pr_auc"].mean())
    real_roc = float(real_df["roc_auc"].mean())

    print("REAL mean PR-AUC:", f"{real_pr:.6f}")
    print("REAL mean ROC-AUC:", f"{real_roc:.6f}")
    print()

    rng = np.random.default_rng(MASTER_SEED)
    null_rows = []

    for null_rep in range(1, N_NULL + 1):
        prs = []
        rocs = []

        for fc in fold_cache:
            Q = random_basis(
                rng,
                ambient_dim=fc["Xtr"].shape[1],
                r=FROZEN_R,
            )

            pr, roc = evaluate(
                fc["Xtr"],
                fc["ytr"],
                fc["Xte"],
                fc["yte"],
                Q,
                seed=MASTER_SEED + 100000 * null_rep + fc["fold"],
            )

            prs.append(pr)
            rocs.append(roc)

        null_rows.append({
            "null_rep": null_rep,
            "mean_pr_auc": float(np.mean(prs)),
            "mean_roc_auc": float(np.mean(rocs)),
        })

        if null_rep % 50 == 0:
            print(f"NULL {null_rep}/{N_NULL}", flush=True)

    null_df = pd.DataFrame(null_rows)

    null_pr = float(null_df["mean_pr_auc"].mean())
    null_roc = float(null_df["mean_roc_auc"].mean())

    p_pr = empirical_p(real_pr, null_df["mean_pr_auc"].to_numpy())
    p_roc = empirical_p(real_roc, null_df["mean_roc_auc"].to_numpy())

    status = (
        "PASS"
        if real_pr > null_pr and p_pr <= 0.05
        else "FAIL"
    )

    real_out = ds / "v44_3_real_fold_metrics.csv"
    null_out = ds / "v44_3_null_summary.csv"
    summary_out = ds / "v44_3_real_vs_null_summary.txt"
    manifest_out = ds / "v44_3_manifest.json"

    real_df.to_csv(real_out, index=False)
    null_df.to_csv(null_out, index=False)

    lines = [
        "=== Soft Spaces / CML v44.3 TRANSPORT-AWARE REAL vs MATCHED NULL ===",
        "",
        "FROZEN DESIGN",
        "-------------",
        "Development cohort: GSE130404",
        f"Frozen rank r: {FROZEN_R}",
        "Transport-aware fold-specific Top-256 pools reused: YES",
        "Frozen v42.2 100 folds reused: YES",
        f"Matched random orthonormal subspaces: {N_NULL}",
        "External outcome data used: NO",
        "",
        "PRIMARY: MEAN HELD-OUT PR-AUC",
        "-----------------------------",
        f"REAL:       {real_pr:.6f}",
        f"NULL mean:  {null_pr:.6f}",
        f"Delta:      {real_pr - null_pr:+.6f}",
        f"NULL 2.5%:  {float(null_df['mean_pr_auc'].quantile(0.025)):.6f}",
        f"NULL 97.5%: {float(null_df['mean_pr_auc'].quantile(0.975)):.6f}",
        f"Empirical one-sided p: {p_pr:.9f}",
        "",
        "SECONDARY: MEAN HELD-OUT ROC-AUC",
        "---------------------------------",
        f"REAL:       {real_roc:.6f}",
        f"NULL mean:  {null_roc:.6f}",
        f"Delta:      {real_roc - null_roc:+.6f}",
        f"Empirical one-sided p: {p_roc:.9f}",
        "",
        "DECISION",
        "--------",
        "PASS iff REAL mean PR-AUC > NULL mean PR-AUC",
        "and empirical one-sided p <= 0.05.",
        "",
        f"v44.3 STATUS: {status}",
        "",
        "INTERPRETATION LIMIT",
        "--------------------",
        "PASS supports only an INTERNAL transport-aware Soft-Spaces",
        "direct-subspace effect relative to matched random subspaces.",
        "",
        "It does not establish external replication, clinical utility,",
        "causal biology, superiority to established biomarkers,",
        "or quantum advantage.",
    ]

    summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_out.write_text(
        json.dumps(
            {
                "version": VERSION,
                "frozen_r": FROZEN_R,
                "n_null": N_NULL,
                "real_mean_pr_auc": real_pr,
                "null_mean_pr_auc": null_pr,
                "delta_pr_auc": real_pr - null_pr,
                "empirical_p_pr_auc": p_pr,
                "real_mean_roc_auc": real_roc,
                "null_mean_roc_auc": null_roc,
                "delta_roc_auc": real_roc - null_roc,
                "empirical_p_roc_auc": p_roc,
                "status": status,
                "external_outcomes_used": False,
                "sha256": {
                    "v44_0_protocol": actual_sha,
                    "v44_2_manifest": sha256_file(v442_manifest),
                    "v44_2_fold_top256_manifest": sha256_file(pools_path),
                    "v42_2_cv_splits": sha256_file(splits_path),
                    "execution_script": sha256_file(Path(__file__).resolve()),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(summary_out.read_text(encoding="utf-8"))
    print("Wrote:")
    for p in [real_out, null_out, summary_out, manifest_out]:
        print(" ", p)


if __name__ == "__main__":
    main()
