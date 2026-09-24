#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v41.11b — recover prospective GSE87371 scoring.

Why this exists
---------------
v41.11 successfully:
- downloaded GSE87371,
- generated BLIND predictions,
- saved them BEFORE label parsing,
- recorded SHA256:
  ded619e0f640f4d783e4267f7c75b570298375f8a5708db88d4137b26cfd0289

Then the metadata parser selected a descriptive definition row instead of the
actual per-sample COO characteristics row.

v41.11b does NOT regenerate the blind predictions.
It verifies the saved blind file hash, then parses only
!Sample_characteristics_ch1 rows and resolves the actual COO values.

Known GEO encoding includes values such as:
    coo: GC
    coo: ABC

For the binary benchmark:
    GC  -> GCB
    GCB -> GCB
    ABC -> ABC

PMBL / NC / unclassified / missing / other values are preserved but excluded
from the binary ABC-vs-GCB score.

After label recovery, the script evaluates the preserved blind predictions and
runs the same 200 matched NULL controls.

No parameter tuning on GSE87371.
No Aer.
No QPU.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


VERSION = "v41.11b-dlbcl-gse87371-metadata-recovery-1"

EXPECTED_BLIND_SHA256 = (
    "ded619e0f640f4d783e4267f7c75b570"
    "298375f8a5708db88d4137b26cfd0289"
)

VARIANCE_POOL = 256
P_DIM = 16
FEATURE_K = 16
N_NULL = 200
NULL_SEED = 41_110_000


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_sample_metadata(path: Path):
    rows = []
    occurrence = Counter()

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if line == "!series_matrix_table_begin":
                break

            if not line.startswith("!Sample_"):
                continue

            parts = next(
                csv.reader(
                    [line],
                    delimiter="\t",
                    quotechar='"',
                )
            )

            key = parts[0]

            values = [
                " ".join(
                    v.strip().strip('"').split()
                )
                for v in parts[1:]
            ]

            occurrence[key] += 1

            rows.append({
                "key": key,
                "occurrence": occurrence[key],
                "values": values,
            })

    access = [
        r
        for r in rows
        if r["key"] == "!Sample_geo_accession"
    ]

    if len(access) != 1:
        raise RuntimeError(
            f"Expected one Sample_geo_accession row, found {len(access)}."
        )

    return rows, access[0]["values"]


def parse_coo_value(text: str):
    """
    Parse actual per-sample COO characteristic.

    Returns normalized_label, status.
    """
    t = text.strip()
    low = t.lower()

    if not low.startswith("coo:"):
        return "", "NO_COO"

    raw = t.split(":", 1)[1].strip()
    u = raw.upper()

    if u in {"GC", "GCB"}:
        return "GCB", "BINARY"

    if u == "ABC":
        return "ABC", "BINARY"

    if u in {
        "PMBL",
        "PMBCL",
        "PRIMARY MEDIASTINAL B-CELL LYMPHOMA",
    }:
        return "PMBL", "NONBINARY"

    if u in {
        "NC",
        "UNC",
        "UNCLASSIFIED",
        "UNCLASSIFIABLE",
        "NON-CLASSIFIED",
        "NON CLASSIFIED",
    }:
        return "Unclassified", "NONBINARY"

    if u in {
        "",
        "NA",
        "N/A",
        "NONE",
        "UNKNOWN",
    }:
        return "", "MISSING"

    return raw, "OTHER"


def find_actual_coo_row(rows, sample_count):
    """
    Search ONLY Sample_characteristics_ch1 rows.

    Pick the row with the most actual values beginning with 'coo:'.
    Descriptive !Sample_description rows are ignored by construction.
    """
    candidates = []

    for r in rows:
        if r["key"] != "!Sample_characteristics_ch1":
            continue

        if len(r["values"]) != sample_count:
            continue

        n_coo = sum(
            v.lower().startswith("coo:")
            for v in r["values"]
        )

        if n_coo > 0:
            candidates.append(
                (
                    n_coo,
                    r["occurrence"],
                    r,
                )
            )

    if not candidates:
        raise RuntimeError(
            "No !Sample_characteristics_ch1 row containing actual 'coo:' values found."
        )

    candidates.sort(
        key=lambda x: (
            -x[0],
            x[1],
        )
    )

    n_coo, _, row = candidates[0]

    return row, n_coo, candidates


def resolve_coo(rows, sample_ids):
    row, matched, candidates = find_actual_coo_row(
        rows,
        len(sample_ids),
    )

    records = []

    for gsm, raw in zip(
        sample_ids,
        row["values"],
    ):
        label, status = parse_coo_value(
            raw
        )

        records.append({
            "sample_id": gsm,
            "coo_label": label,
            "coo_status": status,
            "coo_raw": raw,
        })

    df = pd.DataFrame(
        records
    ).set_index(
        "sample_id"
    )

    source = (
        f'{row["key"]}#'
        f'{row["occurrence"]}'
    )

    candidate_audit = [
        {
            "source":
                f'{r["key"]}#{r["occurrence"]}',
            "coo_prefixed_values":
                int(n),
        }
        for n, _, r in candidates
    ]

    return (
        df,
        source,
        matched,
        candidate_audit,
    )


def read_mapping(path: Path):
    return (
        pd.read_csv(
            path,
            compression="gzip",
            dtype=str,
        )
        .fillna("")[
            [
                "probe_id",
                "gene_symbol",
            ]
        ]
        .drop_duplicates()
    )


def read_expression(path: Path):
    rows = []
    header = None
    in_table = False

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as f:
        for raw in f:
            line = raw.rstrip(
                "\r\n"
            )

            if line == "!series_matrix_table_begin":
                in_table = True
                continue

            if line == "!series_matrix_table_end":
                break

            if not in_table:
                continue

            parts = next(
                csv.reader(
                    [line],
                    delimiter="\t",
                    quotechar='"',
                )
            )

            if header is None:
                header = parts
                continue

            rows.append(
                parts
            )

    if header is None:
        raise RuntimeError(
            "No expression table found."
        )

    ids = header[1:]

    probes = []

    data = np.empty(
        (
            len(rows),
            len(ids),
        ),
        dtype=np.float32,
    )

    for i, parts in enumerate(
        rows
    ):
        probes.append(
            parts[0]
        )

        data[i] = [
            np.nan
            if x.strip() in {
                "",
                "NA",
                "NaN",
                "nan",
                "NULL",
                "null",
            }
            else float(
                x
            )
            for x in parts[1:]
        ]

    return pd.DataFrame(
        data,
        index=pd.Index(
            probes,
            name="probe_id",
        ),
        columns=ids,
    )


def aggregate_probe_to_gene(
    expr,
    mapping,
):
    mp = mapping[
        mapping["probe_id"].isin(
            expr.index
        )
    ]

    groups = defaultdict(
        list
    )

    for probe, gene in mp.itertuples(
        index=False
    ):
        if gene:
            groups[
                gene
            ].append(
                probe
            )

    genes = sorted(
        groups
    )

    out = np.empty(
        (
            len(genes),
            expr.shape[1],
        ),
        dtype=np.float32,
    )

    for i, gene in enumerate(
        genes
    ):
        probes = list(
            dict.fromkeys(
                groups[
                    gene
                ]
            )
        )

        out[i] = np.nanmedian(
            expr.loc[
                probes
            ].to_numpy(
                dtype=np.float32
            ),
            axis=0,
        )

    return pd.DataFrame(
        out,
        index=genes,
        columns=expr.columns,
    )


def h_operator(
    X,
    y,
):
    Xa = X[
        y == 1
    ]

    Xg = X[
        y == 0
    ]

    H = (
        (Xa.T @ Xa) / len(Xa)
        -
        (Xg.T @ Xg) / len(Xg)
    )

    return 0.5 * (
        H + H.T
    )


def softspaces_score(
    H,
):
    evals, evecs = np.linalg.eigh(
        H
    )

    top = np.argsort(
        np.abs(
            evals
        )
    )[::-1][
        :P_DIM
    ]

    U = evecs[
        :,
        top,
    ]

    p = np.sum(
        U * U,
        axis=1,
    )

    coupling = (
        p * (
            1.0 - p
        )
    )

    abs_e = np.abs(
        evals
    )

    spectral = (
        float(
            np.sum(
                abs_e[
                    top
                ]
            )
            /
            np.sum(
                abs_e
            )
        )
        if np.sum(
            abs_e
        ) > 0
        else 0.0
    )

    return (
        coupling,
        spectral,
    )


def stable_rank(
    scores,
    genes,
):
    return np.lexsort(
        (
            genes.astype(
                str
            ),
            -scores,
        )
    )


def fit_model(
    X,
    y,
    selected,
):
    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        random_state=41_110_777,
    )

    model.fit(
        X[
            :,
            selected,
        ],
        y,
    )

    return model


def evaluate(
    y,
    prob,
):
    pred = (
        prob >= 0.5
    ).astype(
        int
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[
            0,
            1,
        ],
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

        "tn":
            int(
                tn
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "tp":
            int(
                tp
            ),

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


def summarize(
    values,
):
    a = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean":
            float(
                np.mean(
                    a
                )
            ),

        "sd":
            float(
                np.std(
                    a,
                    ddof=1,
                )
            ),

        "min":
            float(
                np.min(
                    a
                )
            ),

        "max":
            float(
                np.max(
                    a
                )
            ),
    }


def main():
    code_dir = Path(
        __file__
    ).resolve().parent

    project = (
        code_dir.parent
    )

    matrix_path = (
        project /
        "data" /
        "external" /
        "GSE87371" /
        "raw" /
        "GSE87371_series_matrix.txt.gz"
    )

    blind_csv = (
        project /
        "results" /
        "prospective" /
        "v41_11_GSE87371_BLIND_predictions.csv"
    )

    train_expr_path = (
        project /
        "data" /
        "processed" /
        "GSE10846_ABC_GCB_gene_expression.csv.gz"
    )

    train_labels_path = (
        project /
        "data" /
        "processed" /
        "GSE10846_ABC_GCB_sample_labels.csv"
    )

    mapping_path = (
        project /
        "data" /
        "processed" /
        "GSE10846_probe_to_gene_mapping.csv.gz"
    )

    meta_dir = (
        project /
        "data" /
        "external" /
        "GSE87371" /
        "metadata"
    )

    out_dir = (
        project /
        "results" /
        "prospective"
    )

    meta_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_csv = (
        meta_dir /
        "GSE87371_COO_resolved_v41_11b.csv"
    )

    scored_csv = (
        out_dir /
        "v41_11b_GSE87371_scored_predictions.csv"
    )

    null_csv = (
        out_dir /
        "v41_11b_GSE87371_null_metrics.csv"
    )

    summary_json = (
        out_dir /
        "v41_11b_GSE87371_summary.json"
    )

    for p in [
        matrix_path,
        blind_csv,
        train_expr_path,
        train_labels_path,
        mapping_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                p
            )

    print(
        "=== v41.11b — recover GSE87371 prospective scoring ==="
    )

    print()

    print(
        "[1/6] Verifying preserved BLIND prediction file..."
    )

    observed_blind_sha = sha256_file(
        blind_csv
    )

    print(
        f"      observed SHA256: "
        f"{observed_blind_sha}"
    )

    if observed_blind_sha != EXPECTED_BLIND_SHA256:
        raise RuntimeError(
            "Blind prediction SHA256 does NOT match the v41.11 pre-label file. "
            "Stop: prospective audit trail would be broken."
        )

    print(
        "      SHA256 MATCH: PASS"
    )

    blind_df = pd.read_csv(
        blind_csv,
        dtype={
            "sample_id":
                str,
        },
    )

    print(
        "[2/6] Parsing actual Sample_characteristics_ch1 COO row..."
    )

    rows, sample_ids = read_sample_metadata(
        matrix_path
    )

    coo_df, source, matched, candidate_audit = resolve_coo(
        rows,
        sample_ids,
    )

    coo_df.to_csv(
        resolved_csv
    )

    counts = Counter(
        coo_df[
            "coo_label"
        ]
    )

    print(
        f"      COO source: "
        f"{source}"
    )

    print(
        f"      COO-prefixed values: "
        f"{matched}"
    )

    print(
        f"      counts: "
        f"{dict(counts)}"
    )

    if set(
        blind_df[
            "sample_id"
        ]
    ) != set(
        sample_ids
    ):
        raise RuntimeError(
            "Blind prediction sample IDs do not match GEO sample IDs."
        )

    blind_df = (
        blind_df
        .set_index(
            "sample_id"
        )
        .loc[
            sample_ids
        ]
    )

    binary_ids = [
        s
        for s in sample_ids
        if coo_df.loc[
            s,
            "coo_label",
        ] in {
            "ABC",
            "GCB",
        }
    ]

    if len(
        binary_ids
    ) < 20:
        raise RuntimeError(
            "Too few binary ABC/GCB samples after corrected COO parsing."
        )

    yext = np.array(
        [
            1
            if coo_df.loc[
                s,
                "coo_label",
            ] == "ABC"
            else 0
            for s in binary_ids
        ],
        dtype=int,
    )

    prob_real = (
        blind_df
        .loc[
            binary_ids,
            "p_ABC",
        ]
        .to_numpy(
            dtype=float
        )
    )

    real_metrics = evaluate(
        yext,
        prob_real,
    )

    pd.DataFrame({
        "sample_id":
            binary_ids,

        "true_coo":
            [
                coo_df.loc[
                    s,
                    "coo_label",
                ]
                for s in binary_ids
            ],

        "p_ABC":
            prob_real,

        "predicted_coo":
            np.where(
                prob_real >= 0.5,
                "ABC",
                "GCB",
            ),
    }).to_csv(
        scored_csv,
        index=False,
    )

    print(
        "[3/6] Reconstructing frozen GSE10846 + COHORT_Z pipeline..."
    )

    train_df = pd.read_csv(
        train_expr_path,
        index_col=0,
    )

    train_labels = (
        pd.read_csv(
            train_labels_path,
            dtype=str,
        )
        .fillna("")
        .set_index(
            "sample_id"
        )
        .loc[
            train_df.index
        ]
    )

    ytrain = (
        train_labels[
            "coo_label"
        ] == "ABC"
    ).astype(
        int
    ).to_numpy()

    ext_probe = read_expression(
        matrix_path
    )

    ext_gene = aggregate_probe_to_gene(
        ext_probe,
        read_mapping(
            mapping_path
        ),
    )

    common = [
        g
        for g in train_df.columns
        if g in ext_gene.index
    ]

    Xtrain_raw = train_df[
        common
    ].to_numpy(
        dtype=float
    )

    Xext_raw = (
        ext_gene
        .loc[
            common,
            sample_ids,
        ]
        .T
        .to_numpy(
            dtype=float
        )
    )

    genes = np.asarray(
        common,
        dtype=object,
    )

    raw_var = np.var(
        Xtrain_raw,
        axis=0,
        ddof=1,
    )

    var_order = np.lexsort(
        (
            genes.astype(
                str
            ),
            -raw_var,
        )
    )

    pool = var_order[
        :VARIANCE_POOL
    ]

    pool_genes = genes[
        pool
    ]

    Xtrain_pool = Xtrain_raw[
        :,
        pool,
    ]

    Xext_pool = Xext_raw[
        :,
        pool,
    ]

    Xtrain_z = StandardScaler().fit_transform(
        Xtrain_pool
    )

    Xext_z = StandardScaler().fit_transform(
        Xext_pool
    )

    sample_pos = {
        s:
            i
        for i, s in enumerate(
            sample_ids
        )
    }

    binary_pos = np.array(
        [
            sample_pos[
                s
            ]
            for s in binary_ids
        ],
        dtype=int,
    )

    print(
        f"[4/6] Running {N_NULL} matched NULL controls..."
    )

    rng = np.random.default_rng(
        NULL_SEED
    )

    null_rows = []
    null_auc = []
    null_bal = []
    null_spec = []

    for j in range(
        N_NULL
    ):
        ynull = rng.permutation(
            ytrain
        )

        score, spectral = softspaces_score(
            h_operator(
                Xtrain_z,
                ynull,
            )
        )

        rank = stable_rank(
            score,
            pool_genes,
        )

        selected = rank[
            :FEATURE_K
        ]

        model = fit_model(
            Xtrain_z,
            ytrain,
            selected,
        )

        p_all = model.predict_proba(
            Xext_z[
                :,
                selected,
            ]
        )[
            :,
            1
        ]

        p = p_all[
            binary_pos
        ]

        m = evaluate(
            yext,
            p,
        )

        null_auc.append(
            m[
                "roc_auc"
            ]
        )

        null_bal.append(
            m[
                "balanced_accuracy"
            ]
        )

        null_spec.append(
            spectral
        )

        null_rows.append({
            "null_rep":
                j + 1,

            "external_roc_auc":
                m[
                    "roc_auc"
                ],

            "external_balanced_accuracy":
                m[
                    "balanced_accuracy"
                ],

            "external_accuracy":
                m[
                    "accuracy"
                ],

            "external_f1":
                m[
                    "f1"
                ],

            "training_spectral_topP_abs_fraction":
                spectral,
        })

        if (
            j + 1
        ) % 25 == 0:
            print(
                f"      NULL "
                f"{j+1:3d}/"
                f"{N_NULL}"
            )

    pd.DataFrame(
        null_rows
    ).to_csv(
        null_csv,
        index=False,
    )

    print(
        "[5/6] Computing prospective REAL-vs-NULL statistics..."
    )

    # Reconstruct REAL spectral statistic only.
    real_score, real_spectral = softspaces_score(
        h_operator(
            Xtrain_z,
            ytrain,
        )
    )

    p_auc = (
        1
        +
        int(
            np.sum(
                np.asarray(
                    null_auc
                )
                >=
                real_metrics[
                    "roc_auc"
                ]
            )
        )
    ) / (
        N_NULL + 1
    )

    p_bal = (
        1
        +
        int(
            np.sum(
                np.asarray(
                    null_bal
                )
                >=
                real_metrics[
                    "balanced_accuracy"
                ]
            )
        )
    ) / (
        N_NULL + 1
    )

    p_spec = (
        1
        +
        int(
            np.sum(
                np.asarray(
                    null_spec
                )
                >=
                real_spectral
            )
        )
    ) / (
        N_NULL + 1
    )

    elapsed = time.perf_counter() - time.perf_counter() + 0.0
    # Runtime below starts from this recovery script only.
    # Use a fresh timestamp for an honest but compact audit.
    # (The prospective blind generation runtime already belongs to v41.11.)

    print(
        "[6/6] Writing recovered prospective audit..."
    )

    payload = {
        "version":
            VERSION,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "prospective_audit_preserved":
            True,

        "blind_prediction_sha256_expected":
            EXPECTED_BLIND_SHA256,

        "blind_prediction_sha256_observed":
            observed_blind_sha,

        "blind_sha256_match":
            observed_blind_sha
            ==
            EXPECTED_BLIND_SHA256,

        "metadata_fix": {
            "incorrect_v41_11_source":
                "!Sample_description#12 descriptive definition row",

            "corrected_rule":
                "select only !Sample_characteristics_ch1 row containing actual coo: values",

            "corrected_source":
                source,

            "coo_prefixed_values":
                matched,

            "candidate_characteristics_rows":
                candidate_audit,

            "label_counts":
                dict(
                    counts
                ),
        },

        "binary_evaluation": {
            "n":
                len(
                    binary_ids
                ),

            "GCB":
                int(
                    np.sum(
                        yext == 0
                    )
                ),

            "ABC":
                int(
                    np.sum(
                        yext == 1
                    )
                ),
        },

        "real_external_metrics":
            real_metrics,

        "real_training_spectral_topP_abs_fraction":
            real_spectral,

        "null_external_auc":
            summarize(
                null_auc
            ),

        "null_external_balanced_accuracy":
            summarize(
                null_bal
            ),

        "null_training_spectral":
            summarize(
                null_spec
            ),

        "real_minus_null_mean": {
            "roc_auc":
                float(
                    real_metrics[
                        "roc_auc"
                    ]
                    -
                    np.mean(
                        null_auc
                    )
                ),

            "balanced_accuracy":
                float(
                    real_metrics[
                        "balanced_accuracy"
                    ]
                    -
                    np.mean(
                        null_bal
                    )
                ),

            "spectral":
                float(
                    real_spectral
                    -
                    np.mean(
                        null_spec
                    )
                ),
        },

        "empirical_one_sided_p": {
            "external_auc":
                p_auc,

            "external_balanced_accuracy":
                p_bal,

            "training_spectral":
                p_spec,
        },

        "interpretation": (
            "Blind predictions were not regenerated after label parsing failure; "
            "their preserved SHA256 was verified before scoring."
        ),

        "outputs": {
            "resolved_labels":
                str(
                    resolved_csv.relative_to(
                        project
                    )
                ),

            "scored_predictions":
                str(
                    scored_csv.relative_to(
                        project
                    )
                ),

            "null_metrics":
                str(
                    null_csv.relative_to(
                        project
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
        "=== v41.11b RECOVERED PROSPECTIVE RESULT ==="
    )

    print(
        f"Binary samples:             "
        f"{len(binary_ids)}"
    )

    print(
        f"GCB / ABC:                  "
        f"{np.sum(yext == 0)} / "
        f"{np.sum(yext == 1)}"
    )

    print()

    print(
        f"REAL external AUC:          "
        f"{real_metrics['roc_auc']:.4f}"
    )

    print(
        f"NULL mean external AUC:     "
        f"{np.mean(null_auc):.4f}"
    )

    print(
        f"REAL-NULL ΔAUC:             "
        f"{real_metrics['roc_auc'] - np.mean(null_auc):+.4f}"
    )

    print(
        f"Empirical p(AUC):           "
        f"{p_auc:.6g}"
    )

    print()

    print(
        f"REAL external BalAcc:       "
        f"{real_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"NULL mean external BalAcc:  "
        f"{np.mean(null_bal):.4f}"
    )

    print(
        f"REAL-NULL ΔBalAcc:          "
        f"{real_metrics['balanced_accuracy'] - np.mean(null_bal):+.4f}"
    )

    print(
        f"Empirical p(BalAcc):        "
        f"{p_bal:.6g}"
    )

    print()

    print(
        f"Blind SHA256 match:         "
        f"{observed_blind_sha == EXPECTED_BLIND_SHA256}"
    )

    print()

    print(
        f"Summary: "
        f"{summary_json}"
    )

    print()

    print(
        "v41.11b COMPLETE."
    )

    print(
        "No blind predictions were regenerated."
    )

    print(
        "No Aer or QPU execution performed."
    )


if __name__ == "__main__":
    main()
