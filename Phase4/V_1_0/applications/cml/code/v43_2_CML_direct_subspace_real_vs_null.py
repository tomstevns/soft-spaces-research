#!/usr/bin/env python3
"""
Soft Spaces / CML
v43.2 — Frozen direct-subspace REAL vs matched random-subspace NULL

Frozen from v43.1:
    r = 16

Development cohort:
    GSE130404

Frozen evaluation:
    exact v42.2 100 CV folds
    exact fold-specific Top-256 raw-variance pools
    train-only StandardScaler
    balanced L2 logistic regression

REAL:
    build H on training fold
    take top-|eigenvalue| r=16 eigenvectors
    project train/test directly into U_r
    classify in r-dimensional space

NULL:
    for each fold and each null realization:
      draw Gaussian 256 x 16 matrix
      QR -> orthonormal Q_r
      project same train/test matrices
      fit same classifier
      score same held-out fold

Primary metric:
    mean held-out PR-AUC

PASS iff:
    REAL mean PR-AUC > NULL mean PR-AUC
    AND empirical one-sided p <= 0.05

No external cohort used.
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
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


VERSION = "v43.2"

EXPECTED_PROTOCOL_SHA256 = (
    "f32741ca64d2bc6c015374b0e5e0b25cf5820141d57bd3e5ba8a53a94a097059"
)

FROZEN_R = 16
N_NULL = 1000
MASTER_SEED = 4302001


def project_dir_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_protocol(project: Path) -> Path:
    candidates = [
        project / "docs" / "v43_0_CML_DIRECT_SUBSPACE_PROTOCOL.txt",
        project / "code" / "v43_0_CML_DIRECT_SUBSPACE_PROTOCOL.txt",
        project / "v43_0_CML_DIRECT_SUBSPACE_PROTOCOL.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("v43_0_CML_DIRECT_SUBSPACE_PROTOCOL.txt not found")


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


def real_subspace(Xtr, ytr, r):
    H = class_contrast_operator(Xtr, ytr)
    evals, U = np.linalg.eigh(H)
    order = np.argsort(np.abs(evals))[::-1]
    return U[:, order[:r]]


def random_orthonormal_subspace(rng, ambient_dim, r):
    A = rng.normal(size=(ambient_dim, r))
    Q, _ = np.linalg.qr(A, mode="reduced")
    return Q[:, :r]


def evaluate_projection(Xtr, ytr, Xte, yte, basis, seed):
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


def empirical_p(real_value, null_values):
    null_values = np.asarray(null_values, dtype=float)
    return float(
        (1 + np.sum(null_values >= real_value))
        / (1 + len(null_values))
    )


def main():
    project = project_dir_from_script()

    protocol_path = find_protocol(project)
    protocol_sha = sha256_file(protocol_path)

    print(f"=== {VERSION} CML DIRECT SUBSPACE REAL vs NULL ===")
    print("Expected protocol SHA:", EXPECTED_PROTOCOL_SHA256)
    print("Actual protocol SHA:  ", protocol_sha)

    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise SystemExit("FAIL: v43.0 protocol SHA mismatch.")

    print("PASS: v43.0 protocol verified.")
    print(f"Frozen rank r={FROZEN_R}")
    print(f"NULL realizations={N_NULL}")
    print("External cohort used: NO")
    print()

    splits_path = (
        project
        / "results"
        / "baseline"
        / "v42_2_cv_splits.json"
    )

    v431_manifest_path = (
        project
        / "results"
        / "direct_subspace"
        / "v43_1_manifest.json"
    )

    for p in [splits_path, v431_manifest_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    v431 = json.loads(v431_manifest_path.read_text(encoding="utf-8"))

    if int(v431["selected_r"]) != FROZEN_R:
        raise RuntimeError(
            f"v43.1 selected rank mismatch: expected {FROZEN_R}, "
            f"found {v431['selected_r']}"
        )

    splits_doc = json.loads(
        splits_path.read_text(encoding="utf-8")
    )

    if len(splits_doc.get("folds", [])) != 100:
        raise RuntimeError("Expected 100 frozen v42.2 folds.")

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

    X_probe, meta = parse_gse130404(series_path)
    X_gene = aggregate_probe_to_gene(
        X_probe,
        parse_platform_mapping(gpl_path),
    )
    y = make_y(meta)

    sample_ids = np.asarray(X_gene.index.astype(str))
    genes_all = np.asarray(X_gene.columns.astype(str))
    X_raw = X_gene.to_numpy(dtype=np.float64)

    gsm_to_i = {gsm: i for i, gsm in enumerate(sample_ids)}
    gene_to_i = {g: i for i, g in enumerate(genes_all)}

    fold_cache = []
    real_rows = []

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

        try:
            pool_idx = np.asarray(
                [gene_to_i[g] for g in frozen_pool_genes],
                dtype=int,
            )
        except KeyError as e:
            raise RuntimeError(f"Fold {fold}: frozen Top-256 gene missing: {e}")

        scaler = StandardScaler()

        Xtr = scaler.fit_transform(
            X_raw[train_idx][:, pool_idx]
        )
        Xte = scaler.transform(
            X_raw[test_idx][:, pool_idx]
        )

        ytr = y[train_idx]
        yte = y[test_idx]

        U_real = real_subspace(Xtr, ytr, FROZEN_R)

        real_pr, real_roc = evaluate_projection(
            Xtr,
            ytr,
            Xte,
            yte,
            U_real,
            seed=MASTER_SEED + fold,
        )

        real_rows.append({
            "fold": fold,
            "repeat": fold_rec["repeat"],
            "fold_within_repeat": fold_rec["fold_within_repeat"],
            "pr_auc": real_pr,
            "roc_auc": real_roc,
        })

        fold_cache.append({
            "fold": fold,
            "Xtr": Xtr,
            "Xte": Xte,
            "ytr": ytr,
            "yte": yte,
        })

    real_df = pd.DataFrame(real_rows)

    real_mean_pr = float(real_df["pr_auc"].mean())
    real_mean_roc = float(real_df["roc_auc"].mean())

    print("REAL mean PR-AUC:", f"{real_mean_pr:.6f}")
    print("REAL mean ROC-AUC:", f"{real_mean_roc:.6f}")
    print()

    rng = np.random.default_rng(MASTER_SEED)

    null_rows = []

    for null_rep in range(1, N_NULL + 1):
        prs = []
        rocs = []

        for fc in fold_cache:
            Q = random_orthonormal_subspace(
                rng,
                ambient_dim=fc["Xtr"].shape[1],
                r=FROZEN_R,
            )

            pr, roc = evaluate_projection(
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

    status = (
        "PASS"
        if real_mean_pr > null_mean_pr and p_pr <= 0.05
        else "FAIL"
    )

    outdir = project / "results" / "direct_subspace"
    outdir.mkdir(parents=True, exist_ok=True)

    real_path = outdir / "v43_2_real_fold_metrics.csv"
    null_path = outdir / "v43_2_null_summary.csv"
    summary_path = outdir / "v43_2_real_vs_null_summary.txt"
    manifest_path = outdir / "v43_2_manifest.json"

    real_df.to_csv(real_path, index=False)
    null_df.to_csv(null_path, index=False)

    lines = [
        f"=== Soft Spaces / CML {VERSION} DIRECT SUBSPACE REAL vs MATCHED NULL ===",
        "",
        "FROZEN DESIGN",
        "-------------",
        "Development cohort: GSE130404",
        f"Frozen direct-subspace rank r: {FROZEN_R}",
        "Frozen v42.2 splits reused: YES",
        "Frozen Top-256 pools reused: YES",
        "Feature selection used: NO",
        f"Matched random orthonormal subspaces: {N_NULL}",
        "External cohort used: NO",
        "",
        "PRIMARY: MEAN HELD-OUT PR-AUC",
        "-----------------------------",
        f"REAL:       {real_mean_pr:.6f}",
        f"NULL mean:  {null_mean_pr:.6f}",
        f"Delta:      {real_mean_pr - null_mean_pr:+.6f}",
        f"NULL 2.5%:  {float(null_df['mean_pr_auc'].quantile(0.025)):.6f}",
        f"NULL 97.5%: {float(null_df['mean_pr_auc'].quantile(0.975)):.6f}",
        f"Empirical one-sided p: {p_pr:.6g}",
        "",
        "SECONDARY: MEAN HELD-OUT ROC-AUC",
        "---------------------------------",
        f"REAL:       {real_mean_roc:.6f}",
        f"NULL mean:  {null_mean_roc:.6f}",
        f"Delta:      {real_mean_roc - null_mean_roc:+.6f}",
        f"Empirical one-sided p: {p_roc:.6g}",
        "",
        "DECISION",
        "--------",
        "PASS iff REAL mean PR-AUC > NULL mean PR-AUC",
        "and empirical one-sided p <= 0.05.",
        "",
        f"v43.2 STATUS: {status}",
        "",
        "INTERPRETATION LIMIT",
        "--------------------",
        "PASS would support only that the learned Soft-Spaces subspace",
        "preserves more response information than matched random subspaces",
        "of the same dimension.",
        "",
        "It would not establish clinical utility, causal biology,",
        "external replication, superiority to classical baselines,",
        "or quantum advantage.",
    ]

    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "version": VERSION,
        "development_cohort": "GSE130404",
        "frozen_r": FROZEN_R,
        "n_null": N_NULL,
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
            "v43_0_protocol": protocol_sha,
            "v43_1_manifest": sha256_file(v431_manifest_path),
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
    print(summary_path.read_text(encoding="utf-8"))
    print("Wrote:")
    for p in [real_path, null_path, summary_path, manifest_path]:
        print(" ", p)


if __name__ == "__main__":
    main()
