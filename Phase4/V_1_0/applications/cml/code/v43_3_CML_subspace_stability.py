#!/usr/bin/env python3
"""
Soft Spaces / CML
v43.3 — Internal subspace stability analysis

Prerequisite:
    v43.2 STATUS = PASS

Frozen rank:
    r = 16

Goal:
    Determine whether the learned Soft-Spaces subspaces are more mutually
    aligned across the 100 frozen development folds than matched random
    orthonormal subspaces defined on the same fold-specific Top-256 supports.

Primary stability statistic:
    normalized projector overlap

        S_ij = tr(P_i P_j) / r
             = ||U_i^T U_j||_F^2 / r

    where each fold basis is embedded into a common global gene-coordinate
    system before comparison.

Interpretation:
    S_ij = average squared cosine of the principal angles.
    0 means orthogonal subspaces.
    1 means identical subspaces.

Matched NULL:
    For each fold, draw a random Gaussian 256 x r matrix, QR-orthonormalize it,
    embed it into that fold's exact Top-256 gene support, and compute the same
    mean pairwise overlap across all 100 folds.

This is an INTERNAL resampling-stability analysis.
It is not external replication.
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

from sklearn.preprocessing import StandardScaler


VERSION = "v43.3"

EXPECTED_PROTOCOL_SHA256 = (
    "f32741ca64d2bc6c015374b0e5e0b25cf5820141d57bd3e5ba8a53a94a097059"
)

FROZEN_R = 16
N_NULL = 1000
MASTER_SEED = 4303001


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
        (
            lut[c]
            for c in ["symbol", "gene symbol", "gene_symbol", "genesymbol"]
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

            if (
                sym
                and sym.upper() not in {"---", "NA", "N/A", "NULL"}
            ):
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
                f"{gsm}: unexpected disease stage {stage!r}"
            )

        if resp == ">10%":
            y.append(1)
        elif resp == "<10%":
            y.append(0)
        else:
            raise RuntimeError(
                f"{gsm}: unresolved 3-month response {resp!r}"
            )

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


def embed_basis(local_basis, local_genes, global_gene_to_i, global_dim):
    """
    Embed a local 256 x r orthonormal basis into the union gene-coordinate
    system. Zero rows correspond to genes absent from that fold's Top-256.
    """
    E = np.zeros(
        (global_dim, local_basis.shape[1]),
        dtype=np.float64,
    )

    for local_i, gene in enumerate(local_genes):
        E[global_gene_to_i[gene], :] = local_basis[local_i, :]

    return E


def projector_overlap(Ua, Ub, r):
    """
    tr(Pa Pb)/r = ||Ua^T Ub||_F^2/r.
    Equivalent to mean squared cosine of the principal angles.
    """
    cross = Ua.T @ Ub
    return float(np.sum(cross * cross) / r)


def pairwise_overlaps(bases, repeats=None):
    rows = []

    n = len(bases)

    for i in range(n):
        for j in range(i + 1, n):
            s = projector_overlap(
                bases[i],
                bases[j],
                FROZEN_R,
            )

            rec = {
                "i": i,
                "j": j,
                "overlap": s,
            }

            if repeats is not None:
                rec["same_repeat"] = int(
                    repeats[i] == repeats[j]
                )

            rows.append(rec)

    return pd.DataFrame(rows)


def mean_pairwise_overlap_fast(bases):
    total = 0.0
    count = 0

    n = len(bases)

    for i in range(n):
        Ui = bases[i]

        for j in range(i + 1, n):
            Uj = bases[j]
            cross = Ui.T @ Uj
            total += float(
                np.sum(cross * cross) / FROZEN_R
            )
            count += 1

    return total / count


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

    print(f"=== {VERSION} CML SUBSPACE STABILITY ===")
    print("Expected protocol SHA:", EXPECTED_PROTOCOL_SHA256)
    print("Actual protocol SHA:  ", protocol_sha)

    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise SystemExit(
            "FAIL: v43.0 protocol SHA mismatch."
        )

    print("PASS: v43.0 protocol verified.")

    v432_manifest_path = (
        project
        / "results"
        / "direct_subspace"
        / "v43_2_manifest.json"
    )

    if not v432_manifest_path.exists():
        raise FileNotFoundError(v432_manifest_path)

    v432 = json.loads(
        v432_manifest_path.read_text(encoding="utf-8")
    )

    if v432.get("status") != "PASS":
        raise RuntimeError(
            "v43.3 is allowed only after v43.2 PASS."
        )

    if int(v432.get("frozen_r")) != FROZEN_R:
        raise RuntimeError(
            f"Expected frozen r={FROZEN_R}, "
            f"found r={v432.get('frozen_r')}"
        )

    print("PASS: v43.2 status verified.")
    print(f"Frozen rank r={FROZEN_R}")
    print(f"Matched stability NULL realizations={N_NULL}")
    print("External cohort used: NO")
    print()

    splits_path = (
        project
        / "results"
        / "baseline"
        / "v42_2_cv_splits.json"
    )

    if not splits_path.exists():
        raise FileNotFoundError(splits_path)

    splits_doc = json.loads(
        splits_path.read_text(encoding="utf-8")
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

    X_probe, meta = parse_gse130404(series_path)

    X_gene = aggregate_probe_to_gene(
        X_probe,
        parse_platform_mapping(gpl_path),
    )

    y = make_y(meta)

    sample_ids = np.asarray(
        X_gene.index.astype(str)
    )

    genes_all = np.asarray(
        X_gene.columns.astype(str)
    )

    X_raw = X_gene.to_numpy(
        dtype=np.float64
    )

    gsm_to_i = {
        gsm: i
        for i, gsm in enumerate(sample_ids)
    }

    gene_to_i = {
        g: i
        for i, g in enumerate(genes_all)
    }

    # Union of all genes appearing in any frozen Top-256 pool.
    union_genes = sorted(
        {
            gene
            for fold_rec in folds
            for gene in fold_rec["top256_genes"]
        }
    )

    global_gene_to_i = {
        g: i
        for i, g in enumerate(union_genes)
    }

    global_dim = len(union_genes)

    print("Union of frozen Top-256 genes:", global_dim)

    real_bases = []
    fold_gene_lists = []
    repeats = []
    fold_ids = []

    for fold_rec in folds:
        fold = int(fold_rec["fold"])

        train_idx = np.asarray(
            [
                gsm_to_i[x]
                for x in fold_rec["train_samples"]
            ],
            dtype=int,
        )

        frozen_pool_genes = list(
            fold_rec["top256_genes"]
        )

        try:
            pool_idx = np.asarray(
                [
                    gene_to_i[g]
                    for g in frozen_pool_genes
                ],
                dtype=int,
            )
        except KeyError as e:
            raise RuntimeError(
                f"Fold {fold}: frozen Top-256 gene missing: {e}"
            )

        scaler = StandardScaler()

        Xtr = scaler.fit_transform(
            X_raw[train_idx][:, pool_idx]
        )

        ytr = y[train_idx]

        U_local = real_subspace(
            Xtr,
            ytr,
            FROZEN_R,
        )

        U_global = embed_basis(
            U_local,
            frozen_pool_genes,
            global_gene_to_i,
            global_dim,
        )

        real_bases.append(U_global)
        fold_gene_lists.append(frozen_pool_genes)
        repeats.append(int(fold_rec["repeat"]))
        fold_ids.append(fold)

        if fold % 10 == 0:
            print(
                f"built REAL basis {fold}/100",
                flush=True,
            )

    # REAL pairwise stability.
    real_pairs = pairwise_overlaps(
        real_bases,
        repeats=repeats,
    )

    real_mean = float(
        real_pairs["overlap"].mean()
    )

    real_median = float(
        real_pairs["overlap"].median()
    )

    real_q025 = float(
        real_pairs["overlap"].quantile(0.025)
    )

    real_q975 = float(
        real_pairs["overlap"].quantile(0.975)
    )

    same_repeat_mean = float(
        real_pairs.loc[
            real_pairs["same_repeat"] == 1,
            "overlap",
        ].mean()
    )

    different_repeat_mean = float(
        real_pairs.loc[
            real_pairs["same_repeat"] == 0,
            "overlap",
        ].mean()
    )

    print()
    print(
        "REAL mean pairwise projector overlap:",
        f"{real_mean:.6f}",
    )

    # Matched random-subspace stability null.
    rng = np.random.default_rng(MASTER_SEED)

    null_rows = []

    for null_rep in range(1, N_NULL + 1):
        null_bases = []

        for genes in fold_gene_lists:
            Q_local = random_orthonormal_subspace(
                rng,
                ambient_dim=len(genes),
                r=FROZEN_R,
            )

            Q_global = embed_basis(
                Q_local,
                genes,
                global_gene_to_i,
                global_dim,
            )

            null_bases.append(Q_global)

        mean_overlap = mean_pairwise_overlap_fast(
            null_bases
        )

        null_rows.append({
            "null_rep": null_rep,
            "mean_pairwise_projector_overlap": mean_overlap,
        })

        if null_rep % 50 == 0:
            print(
                f"NULL {null_rep}/{N_NULL}",
                flush=True,
            )

    null_df = pd.DataFrame(null_rows)

    null_mean = float(
        null_df[
            "mean_pairwise_projector_overlap"
        ].mean()
    )

    p_stability = empirical_p(
        real_mean,
        null_df[
            "mean_pairwise_projector_overlap"
        ].to_numpy(),
    )

    delta = real_mean - null_mean

    outdir = (
        project
        / "results"
        / "direct_subspace"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_path = (
        outdir
        / "v43_3_real_pairwise_stability.csv"
    )

    null_path = (
        outdir
        / "v43_3_stability_null.csv"
    )

    summary_path = (
        outdir
        / "v43_3_subspace_stability_summary.txt"
    )

    manifest_path = (
        outdir
        / "v43_3_manifest.json"
    )

    real_pairs.to_csv(
        pair_path,
        index=False,
    )

    null_df.to_csv(
        null_path,
        index=False,
    )

    summary_lines = [
        "=== Soft Spaces / CML v43.3 SUBSPACE STABILITY ===",
        "",
        "DESIGN",
        "------",
        "Development cohort: GSE130404",
        f"Frozen rank r: {FROZEN_R}",
        "Frozen v42.2 folds reused: YES",
        "Frozen fold-specific Top-256 supports reused: YES",
        "External cohort used: NO",
        f"Matched random-subspace stability realizations: {N_NULL}",
        "",
        "PRIMARY STABILITY STATISTIC",
        "---------------------------",
        "S_ij = tr(P_i P_j)/r = ||U_i^T U_j||_F^2/r",
        "Equivalent to mean squared cosine of principal angles.",
        "",
        f"Number of REAL fold pairs: {len(real_pairs)}",
        f"REAL mean overlap:          {real_mean:.6f}",
        f"REAL median overlap:        {real_median:.6f}",
        f"REAL pairwise 2.5%:         {real_q025:.6f}",
        f"REAL pairwise 97.5%:        {real_q975:.6f}",
        "",
        f"REAL same-repeat mean:      {same_repeat_mean:.6f}",
        f"REAL different-repeat mean: {different_repeat_mean:.6f}",
        "",
        "MATCHED NULL",
        "------------",
        f"NULL mean overlap:          {null_mean:.6f}",
        f"Delta REAL - NULL:          {delta:+.6f}",
        f"NULL 2.5%:                  {float(null_df['mean_pairwise_projector_overlap'].quantile(0.025)):.6f}",
        f"NULL 97.5%:                 {float(null_df['mean_pairwise_projector_overlap'].quantile(0.975)):.6f}",
        f"Empirical one-sided p:      {p_stability:.6g}",
        "",
        "INTERPRETATION",
        "--------------",
        "This is an INTERNAL resampling-stability analysis.",
        "Higher REAL-than-NULL overlap means the learned r=16 subspaces",
        "are more mutually aligned than matched random subspaces on the",
        "same fold-specific gene supports.",
        "",
        "Because repeated-CV training sets overlap substantially, this",
        "must not be interpreted as independent external replication.",
        "",
        "No claim of clinical utility, causal biology, superiority to",
        "classical baselines, or quantum advantage is made.",
    ]

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "version": VERSION,
        "development_cohort": "GSE130404",
        "frozen_r": FROZEN_R,
        "n_folds": 100,
        "n_fold_pairs": int(len(real_pairs)),
        "union_gene_dimension": global_dim,
        "stability_metric": (
            "tr(P_i P_j)/r = ||U_i^T U_j||_F^2/r"
        ),
        "real_mean_overlap": real_mean,
        "real_median_overlap": real_median,
        "real_q025": real_q025,
        "real_q975": real_q975,
        "same_repeat_mean": same_repeat_mean,
        "different_repeat_mean": different_repeat_mean,
        "n_null": N_NULL,
        "null_mean_overlap": null_mean,
        "delta_real_minus_null": delta,
        "empirical_one_sided_p": p_stability,
        "external_used": False,
        "sha256": {
            "v43_0_protocol": protocol_sha,
            "v43_2_manifest": sha256_file(
                v432_manifest_path
            ),
            "v42_2_cv_splits": sha256_file(
                splits_path
            ),
            "series_matrix": sha256_file(
                series_path
            ),
            "platform_table": sha256_file(
                gpl_path
            ),
            "execution_script": sha256_file(
                Path(__file__).resolve()
            ),
        },
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    print("Wrote:")

    for p in [
        pair_path,
        null_path,
        summary_path,
        manifest_path,
    ]:
        print(" ", p)


if __name__ == "__main__":
    main()
