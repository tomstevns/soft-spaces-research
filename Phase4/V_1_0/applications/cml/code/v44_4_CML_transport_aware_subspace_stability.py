#!/usr/bin/env python3
"""
Soft Spaces / CML
v44.4 — Transport-aware direct-subspace stability

Frozen design
-------------
- Development cohort: GSE130404
- Frozen v42.2 outer CV: 100 folds
- Frozen v44.2 fold-specific transport-aware Top-256 pools
- Frozen rank r = 16

REAL stability
--------------
For each fold:
    construct the exact v44 REAL basis U_i in its fold-specific 256D space,
    then embed U_i into the union of all genes appearing in any fold's Top-256.

For each fold pair (i,j):
    S_ij = tr(P_i P_j) / r
         = || U_i^T U_j ||_F^2 / r

where the inner product is computed after embedding both bases in the
common union-gene coordinate system.

NULL stability
--------------
For each of 1000 matched realizations:
    draw one random orthonormal 16D basis independently inside each fold's
    SAME 256D support,
    embed in the same union-gene coordinates,
    calculate all 4950 pairwise normalized projector overlaps,
    record the mean overlap.

Primary stability test
----------------------
PASS iff:
    REAL mean overlap > NULL mean overlap
    AND empirical one-sided p <= 0.05.

No external outcome data are used.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


VERSION = "v44.4"
FROZEN_R = 16
TOP_K = 256
N_NULL = 1000
MASTER_SEED = 4404001


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
            for c in ["symbol", "gene symbol", "gene_symbol", "genesymbol"]
            if c in lut
        ),
        None,
    )

    if id_col is None or symbol_col is None:
        raise RuntimeError(
            f"Cannot identify platform mapping columns: {header}"
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


def embed_basis(local_basis, support_genes, union_index, union_dim):
    out = np.zeros(
        (union_dim, local_basis.shape[1]),
        dtype=np.float64,
    )

    rows = np.asarray(
        [union_index[g] for g in support_genes],
        dtype=int,
    )

    out[rows, :] = local_basis
    return out


def normalized_projector_overlap(U, V, r):
    cross = U.T @ V
    return float(
        np.sum(cross * cross) / r
    )


def empirical_p(real, null):
    null = np.asarray(null, dtype=float)
    return float(
        (1 + np.sum(null >= real))
        / (1 + len(null))
    )


def summary_stats(x):
    x = np.asarray(x, dtype=float)

    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q025": float(np.quantile(x, 0.025)),
        "q975": float(np.quantile(x, 0.975)),
        "sd": float(np.std(x, ddof=1)),
    }


def main():
    project = project_dir()
    docs = project / "docs"
    results = project / "results"
    ds = results / "direct_subspace"

    protocol = (
        docs
        / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION.txt"
    )

    protocol_manifest = (
        docs
        / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION_manifest.json"
    )

    v443_manifest = (
        ds
        / "v44_3_manifest.json"
    )

    pools_path = (
        ds
        / "v44_2_fold_top256_manifest.json"
    )

    splits_path = (
        results
        / "baseline"
        / "v42_2_cv_splits.json"
    )

    for p in [
        protocol,
        protocol_manifest,
        v443_manifest,
        pools_path,
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
        "=== v44.4 CML TRANSPORT-AWARE SUBSPACE STABILITY ==="
    )
    print("Expected protocol SHA:", expected_sha)
    print("Actual protocol SHA:  ", actual_sha)

    if expected_sha != actual_sha:
        raise SystemExit(
            "FAIL: v44.0 protocol SHA mismatch."
        )

    m443 = json.loads(
        v443_manifest.read_text(
            encoding="utf-8"
        )
    )

    if m443.get("status") != "PASS":
        raise RuntimeError(
            "v44.3 did not PASS; v44.4 is not authorized."
        )

    if int(m443["frozen_r"]) != FROZEN_R:
        raise RuntimeError(
            f"v44.3 frozen r={m443['frozen_r']}, expected {FROZEN_R}."
        )

    print("PASS: v44.0 protocol verified.")
    print("PASS: v44.3 authorization verified.")
    print(f"Frozen rank r={FROZEN_R}")
    print(f"NULL stability realizations={N_NULL}")
    print("External outcome data used: NO")
    print()

    pools_doc = json.loads(
        pools_path.read_text(
            encoding="utf-8"
        )
    )

    pool_by_fold = {
        int(rec["fold"]): list(rec["top256_genes"])
        for rec in pools_doc["folds"]
    }

    if len(pool_by_fold) != 100:
        raise RuntimeError(
            f"Expected 100 fold pools, found {len(pool_by_fold)}."
        )

    union_genes = sorted(
        set(
            g
            for genes in pool_by_fold.values()
            for g in genes
        )
    )

    union_index = {
        g: i
        for i, g in enumerate(union_genes)
    }

    union_dim = len(union_genes)

    print("Union of fold-specific Top-256 supports:", union_dim)

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

    gsm_to_i = {
        gsm: i
        for i, gsm in enumerate(sample_ids)
    }

    gene_to_i = {
        g: i
        for i, g in enumerate(
            X_gene.columns.astype(str)
        )
    }

    X_raw = X_gene.to_numpy(
        dtype=np.float64
    )

    real_embedded = {}
    fold_repeat = {}

    for fold_rec in folds:
        fold = int(fold_rec["fold"])
        repeat = int(fold_rec["repeat"])

        support = pool_by_fold[fold]

        missing = [
            g
            for g in support
            if g not in gene_to_i
        ]

        if missing:
            raise RuntimeError(
                f"Fold {fold}: genes missing from development matrix: "
                + ", ".join(missing[:20])
            )

        train_idx = np.asarray(
            [
                gsm_to_i[x]
                for x in fold_rec["train_samples"]
            ],
            dtype=int,
        )

        pool_idx = np.asarray(
            [
                gene_to_i[g]
                for g in support
            ],
            dtype=int,
        )

        scaler = StandardScaler()

        Xtr = scaler.fit_transform(
            X_raw[train_idx][:, pool_idx]
        )

        ytr = y[train_idx]

        U_local = real_basis(
            Xtr,
            ytr,
            FROZEN_R,
        )

        U_global = embed_basis(
            U_local,
            support,
            union_index,
            union_dim,
        )

        real_embedded[fold] = U_global
        fold_repeat[fold] = repeat

        if fold % 10 == 0:
            print(
                f"constructed REAL basis {fold}/100",
                flush=True,
            )

    pair_rows = []

    for i, j in combinations(
        sorted(real_embedded),
        2,
    ):
        overlap = normalized_projector_overlap(
            real_embedded[i],
            real_embedded[j],
            FROZEN_R,
        )

        pair_rows.append({
            "fold_i": i,
            "fold_j": j,
            "repeat_i": fold_repeat[i],
            "repeat_j": fold_repeat[j],
            "same_repeat": int(
                fold_repeat[i] == fold_repeat[j]
            ),
            "normalized_projector_overlap": overlap,
        })

    real_pairs = pd.DataFrame(
        pair_rows
    )

    if len(real_pairs) != 4950:
        raise RuntimeError(
            f"Expected 4950 REAL fold pairs, found {len(real_pairs)}."
        )

    real_stats = summary_stats(
        real_pairs[
            "normalized_projector_overlap"
        ].to_numpy()
    )

    same_repeat = real_pairs.loc[
        real_pairs["same_repeat"] == 1,
        "normalized_projector_overlap",
    ].to_numpy()

    different_repeat = real_pairs.loc[
        real_pairs["same_repeat"] == 0,
        "normalized_projector_overlap",
    ].to_numpy()

    same_repeat_mean = float(
        np.mean(same_repeat)
    )

    different_repeat_mean = float(
        np.mean(different_repeat)
    )

    print()
    print("REAL fold pairs:", len(real_pairs))
    print(
        "REAL mean normalized projector overlap:",
        f"{real_stats['mean']:.6f}",
    )
    print()

    rng = np.random.default_rng(
        MASTER_SEED
    )

    null_rows = []

    sorted_folds = sorted(
        pool_by_fold
    )

    for null_rep in range(
        1,
        N_NULL + 1,
    ):
        random_embedded = {}

        for fold in sorted_folds:
            support = pool_by_fold[fold]

            Q_local = random_basis(
                rng,
                TOP_K,
                FROZEN_R,
            )

            random_embedded[fold] = embed_basis(
                Q_local,
                support,
                union_index,
                union_dim,
            )

        overlaps = []

        for i, j in combinations(
            sorted_folds,
            2,
        ):
            overlaps.append(
                normalized_projector_overlap(
                    random_embedded[i],
                    random_embedded[j],
                    FROZEN_R,
                )
            )

        null_rows.append({
            "null_rep": null_rep,
            "mean_normalized_projector_overlap": float(
                np.mean(overlaps)
            ),
            "median_normalized_projector_overlap": float(
                np.median(overlaps)
            ),
        })

        if null_rep % 50 == 0:
            print(
                f"NULL stability {null_rep}/{N_NULL}",
                flush=True,
            )

    null_df = pd.DataFrame(
        null_rows
    )

    null_mean = float(
        null_df[
            "mean_normalized_projector_overlap"
        ].mean()
    )

    p_value = empirical_p(
        real_stats["mean"],
        null_df[
            "mean_normalized_projector_overlap"
        ].to_numpy(),
    )

    status = (
        "PASS"
        if real_stats["mean"] > null_mean
        and p_value <= 0.05
        else "FAIL"
    )

    pair_out = (
        ds
        / "v44_4_real_pairwise_stability.csv"
    )

    null_out = (
        ds
        / "v44_4_null_stability_summary.csv"
    )

    summary_out = (
        ds
        / "v44_4_subspace_stability_summary.txt"
    )

    manifest_out = (
        ds
        / "v44_4_manifest.json"
    )

    real_pairs.to_csv(
        pair_out,
        index=False,
    )

    null_df.to_csv(
        null_out,
        index=False,
    )

    lines = [
        "=== Soft Spaces / CML v44.4 TRANSPORT-AWARE SUBSPACE STABILITY ===",
        "",
        "FROZEN DESIGN",
        "-------------",
        "Development cohort: GSE130404",
        f"Frozen rank r: {FROZEN_R}",
        "Transport-aware fold-specific Top-256 pools reused: YES",
        "Frozen v42.2 100 folds reused: YES",
        f"Union support size: {union_dim}",
        f"REAL fold pairs: {len(real_pairs)}",
        f"Matched random stability realizations: {N_NULL}",
        "External outcome data used: NO",
        "",
        "REAL STABILITY",
        "--------------",
        f"Mean normalized projector overlap:   {real_stats['mean']:.6f}",
        f"Median:                              {real_stats['median']:.6f}",
        f"2.5% quantile:                       {real_stats['q025']:.6f}",
        f"97.5% quantile:                      {real_stats['q975']:.6f}",
        f"SD:                                  {real_stats['sd']:.6f}",
        f"Same-repeat mean:                    {same_repeat_mean:.6f}",
        f"Different-repeat mean:               {different_repeat_mean:.6f}",
        "",
        "MATCHED RANDOM NULL",
        "-------------------",
        f"NULL mean:                           {null_mean:.6f}",
        f"NULL 2.5%:                           {float(null_df['mean_normalized_projector_overlap'].quantile(0.025)):.6f}",
        f"NULL 97.5%:                          {float(null_df['mean_normalized_projector_overlap'].quantile(0.975)):.6f}",
        f"REAL - NULL mean:                    {real_stats['mean'] - null_mean:+.6f}",
        f"Empirical one-sided p:               {p_value:.9f}",
        "",
        "DECISION",
        "--------",
        "PASS iff REAL mean overlap > NULL mean overlap",
        "and empirical one-sided p <= 0.05.",
        "",
        f"v44.4 STATUS: {status}",
        "",
        "INTERPRETATION LIMIT",
        "--------------------",
        "This measures INTERNAL resampling stability.",
        "The repeated-CV training sets overlap substantially and therefore",
        "do not constitute independent external replication.",
        "",
        "Same-repeat versus different-repeat values are descriptive only.",
        "No biological interpretation is assigned to that difference.",
        "",
        "A PASS authorizes preparation of a separately frozen external",
        "generalization protocol before any external outcome scoring.",
    ]

    summary_out.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest_out.write_text(
        json.dumps(
            {
                "version": VERSION,
                "frozen_r": FROZEN_R,
                "top_k": TOP_K,
                "union_support_n": union_dim,
                "real_pair_n": len(real_pairs),
                "real_mean_overlap": real_stats["mean"],
                "real_median_overlap": real_stats["median"],
                "real_q025": real_stats["q025"],
                "real_q975": real_stats["q975"],
                "same_repeat_mean": same_repeat_mean,
                "different_repeat_mean": different_repeat_mean,
                "n_null": N_NULL,
                "null_mean_overlap": null_mean,
                "delta_mean_overlap": real_stats["mean"] - null_mean,
                "empirical_p": p_value,
                "status": status,
                "external_outcomes_used": False,
                "sha256": {
                    "v44_0_protocol": actual_sha,
                    "v44_3_manifest": sha256_file(
                        v443_manifest
                    ),
                    "v44_2_fold_top256_manifest": sha256_file(
                        pools_path
                    ),
                    "v42_2_cv_splits": sha256_file(
                        splits_path
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
        pair_out,
        null_out,
        summary_out,
        manifest_out,
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
