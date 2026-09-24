#!/usr/bin/env python3
"""
Soft Spaces / CML
v44.5b — Primary external generalization test on GSE44589

FROZEN BY v44.5
----------------
Primary external cohort:
    GSE44589

Population:
    pretreatment / baseline samples only

Positive class:
    poor response / no MMR

Negative class:
    MMR

Expected prior audit counts:
    135 baseline
    69 MMR
    59 none
    7 unavailable
    128 evaluable

Frozen representation:
    final v44.5a Top-256
    frozen U16
    frozen logistic-regression coefficients/intercept

External normalization:
    per-cohort, per-gene z-standardization across ALL baseline GSE44589
    samples, performed before/independently of response labels.

Primary metric:
    PR-AUC

Secondary:
    ROC-AUC
    balanced accuracy
    sensitivity
    specificity

Decision:
    PASS only if:
      technical transport is already TECHNICALLY EVALUABLE,
      PR-AUC > positive-class prevalence,
      ROC-AUC > 0.5,
      frozen score direction is preserved,
      no outcome-driven tuning occurred.

No model refitting on GSE44589.
No gene substitution.
No rank change.
No threshold tuning.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


VERSION = "v44.5b"
GSE = "GSE44589"
TOP_K = 256
FROZEN_R = 16

SERIES_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE44nnn/GSE44589/matrix/GSE44589_series_matrix.txt.gz"
)

SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE44nnn/GSE44589/soft/GSE44589_family.soft.gz"
)


def project_dir():
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_if_missing(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        print("Using existing:", path)
        return

    print("Downloading:", url)
    print(" ->", path)
    urllib.request.urlretrieve(url, path)


def split_multi(raw):
    s = str(raw or "").strip()

    for sep in ("///", "//", ";", ",", "|"):
        s = s.replace(sep, "|")

    return [
        x.strip()
        for x in s.split("|")
        if x.strip()
        and x.strip().upper() not in {"---", "NA", "N/A", "NULL", "NONE"}
    ]


def parse_entrez(raw):
    out = set()

    for token in split_multi(raw):
        out.update(
            re.findall(
                r"(?<!\d)(\d+)(?!\d)",
                token,
            )
        )

    out.discard("0")
    return out


def normalize_header(x):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(x).strip().lower(),
    )


def parse_platform(path: Path):
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    b = next(
        i for i, x in enumerate(lines)
        if x.startswith("!platform_table_begin")
    )

    e = next(
        i for i, x in enumerate(lines)
        if x.startswith("!platform_table_end")
    )

    table = lines[b + 1:e]
    header = table[0].split("\t")
    lut = {
        normalize_header(x): x
        for x in header
    }

    id_col = next(
        (
            lut[x]
            for x in (
                "id",
                "idref",
                "probeid",
                "probesetid",
            )
            if x in lut
        ),
        None,
    )

    symbol_col = next(
        (
            lut[x]
            for x in (
                "symbol",
                "genesymbol",
                "officialgenesymbol",
            )
            if x in lut
        ),
        None,
    )

    entrez_col = next(
        (
            orig
            for n, orig in lut.items()
            if "entrez" in n
        ),
        None,
    )

    if id_col is None or symbol_col is None:
        raise RuntimeError(
            f"Cannot identify GPL570 probe/symbol columns: {header}"
        )

    symbol_to_probes = defaultdict(set)
    entrez_to_probes = defaultdict(set)

    reader = csv.DictReader(
        table[1:],
        fieldnames=header,
        delimiter="\t",
    )

    for row in reader:
        probe = str(
            row.get(id_col, "")
        ).strip()

        if not probe:
            continue

        symbols = set(
            split_multi(
                row.get(symbol_col, "")
            )
        )

        eids = (
            parse_entrez(
                row.get(entrez_col, "")
            )
            if entrez_col
            else set()
        )

        for sym in symbols:
            symbol_to_probes[sym].add(
                probe
            )

        for eid in eids:
            entrez_to_probes[eid].add(
                probe
            )

    return {
        "symbol_to_probes": dict(
            symbol_to_probes
        ),
        "entrez_to_probes": dict(
            entrez_to_probes
        ),
    }


def parse_series_matrix(path: Path):
    """
    Parse both expression table and all sample metadata lines.
    """
    metadata = defaultdict(list)
    inside = False
    header = None
    probes = []
    rows = []

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            if not inside and line.startswith("!Sample_"):
                parts = [
                    x.strip().strip('"')
                    for x in line.rstrip(
                        "\r\n"
                    ).split("\t")
                ]

                metadata[
                    parts[0]
                ].append(
                    parts[1:]
                )
                continue

            if line.startswith(
                "!series_matrix_table_begin"
            ):
                inside = True
                continue

            if line.startswith(
                "!series_matrix_table_end"
            ):
                break

            if not inside:
                continue

            if header is None:
                header = [
                    x.strip().strip('"')
                    for x in line.rstrip(
                        "\r\n"
                    ).split("\t")
                ]
                continue

            if line.strip():
                parts = line.rstrip(
                    "\r\n"
                ).split("\t")

                probes.append(
                    parts[0].strip().strip('"')
                )

                rows.append(
                    [
                        float(
                            x.strip().strip('"')
                        )
                        for x in parts[1:]
                    ]
                )

    if header is None:
        raise RuntimeError(
            "Could not parse GSE44589 series matrix."
        )

    sample_ids = header[1:]

    X_probe = pd.DataFrame(
        np.asarray(
            rows,
            dtype=np.float64,
        ).T,
        index=sample_ids,
        columns=probes,
    )

    n = len(sample_ids)

    meta_rows = [
        {"geo_accession": gsm}
        for gsm in sample_ids
    ]

    for key, occurrences in metadata.items():
        for occ_i, vals in enumerate(
            occurrences,
            1,
        ):
            if len(vals) != n:
                continue

            field = (
                key
                if len(occurrences) == 1
                else f"{key}__{occ_i}"
            )

            for i, value in enumerate(vals):
                meta_rows[i][field] = value

    meta_df = pd.DataFrame(
        meta_rows
    ).set_index(
        "geo_accession"
    )

    return X_probe, meta_df


def parse_soft_sample_metadata(path: Path):
    """
    Used only as an auxiliary metadata source.
    Expression comes from the series matrix.
    """
    records = {}
    current = None

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

            if line.startswith("^SAMPLE"):
                gsm = line.split(
                    "=",
                    1,
                )[1].strip()

                current = {
                    "gsm": gsm,
                    "fields": [],
                }

                records[gsm] = current
                continue

            if current is None:
                continue

            if line.startswith("!Sample_"):
                if "=" in line:
                    _, value = line.split(
                        "=",
                        1,
                    )

                    current[
                        "fields"
                    ].append(
                        value.strip()
                    )

    return records


def flatten_metadata_text(
    gsm,
    matrix_meta,
    soft_records,
):
    parts = []

    if gsm in matrix_meta.index:
        for x in matrix_meta.loc[gsm].tolist():
            if pd.notna(x):
                parts.append(
                    str(x)
                )

    if gsm in soft_records:
        parts.extend(
            soft_records[gsm][
                "fields"
            ]
        )

    return " || ".join(
        parts
    )


def infer_timepoint(text):
    """
    Outcome-blind sample timing classifier.

    Returns:
        baseline
        post6w
        unknown
    """
    t = text.lower()

    # Strong post-treatment patterns first.
    post_patterns = [
        r"\b6\s*week",
        r"\b6\s*wk",
        r"\b6w\b",
        r"post[- ]?treat",
        r"after\s+6",
        r"6\s*weeks?\s+(?:after|on)",
    ]

    if any(
        re.search(p, t)
        for p in post_patterns
    ):
        return "post6w"

    baseline_patterns = [
        r"\bpretreat",
        r"\bpre[- ]?treat",
        r"\bbaseline\b",
        r"\bdiagnos",
        r"\bbefore\s+treat",
        r"\b0\s*week",
        r"\b0\s*wk",
    ]

    if any(
        re.search(p, t)
        for p in baseline_patterns
    ):
        return "baseline"

    return "unknown"


def infer_response(text):
    """
    Response parser used only AFTER baseline cohort and expression transform
    are frozen in memory.

    Returns:
        0 -> MMR
        1 -> none / poor response
        None -> unavailable
    """
    t = text.lower()

    # Explicit no-MMR / none response first.
    poor_patterns = [
        r"\bno\s+mmr\b",
        r"\bnon[- ]?mmr\b",
        r"\bnone\b",
        r"\bno molecular response\b",
        r"\bpoor responder\b",
        r"\bnon[- ]?responder\b",
    ]

    if any(
        re.search(p, t)
        for p in poor_patterns
    ):
        return 1

    mmr_patterns = [
        r"\bmmr\b",
        r"major molecular response",
    ]

    if any(
        re.search(p, t)
        for p in mmr_patterns
    ):
        return 0

    return None


def locate_prior_v42_audit(project: Path):
    """
    Prefer the user's already-audited v42 sample classification if a
    sufficiently informative TSV/CSV exists.

    This avoids silently changing the earlier audited baseline/response
    classification.
    """
    candidates = []

    roots = [
        project / "results" / "dataset_audit",
        project / "results",
    ]

    for root in roots:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if not p.is_file():
                continue

            name = p.name.lower()

            if (
                "v42_1c" in name
                and p.suffix.lower() in {
                    ".csv",
                    ".tsv",
                    ".txt",
                }
            ):
                candidates.append(
                    p
                )

    return sorted(
        candidates
    )


def try_load_prior_classification(
    path: Path,
    sample_ids,
):
    """
    Try common field names without making assumptions about a particular
    v42.1c serialization. Returns dict or None.
    """
    try:
        sep = (
            "\t"
            if path.suffix.lower()
            in {".tsv", ".txt"}
            else ","
        )

        df = pd.read_csv(
            path,
            sep=sep,
            dtype=str,
        )
    except Exception:
        return None

    cols = {
        normalize_header(c): c
        for c in df.columns
    }

    gsm_col = next(
        (
            cols[x]
            for x in (
                "gsm",
                "geoaccession",
                "sample",
                "sampleid",
            )
            if x in cols
        ),
        None,
    )

    if gsm_col is None:
        return None

    time_col = next(
        (
            cols[x]
            for x in (
                "timepoint",
                "sampletimepoint",
                "baselinepost",
                "treatmenttimepoint",
            )
            if x in cols
        ),
        None,
    )

    response_col = next(
        (
            cols[x]
            for x in (
                "response",
                "responseclass",
                "mmr",
                "outcome",
            )
            if x in cols
        ),
        None,
    )

    if time_col is None:
        return None

    out = {}

    for _, row in df.iterrows():
        gsm = str(
            row[gsm_col]
        ).strip()

        if gsm not in sample_ids:
            continue

        t = str(
            row[time_col]
        ).strip().lower()

        if (
            "base" in t
            or "pre" in t
            or t in {"0", "0w"}
        ):
            timepoint = "baseline"
        elif (
            "6" in t
            or "post" in t
        ):
            timepoint = "post6w"
        else:
            timepoint = "unknown"

        response = None

        if response_col is not None:
            r = str(
                row[response_col]
            ).strip().lower()

            if (
                "no mmr" in r
                or "none" in r
                or "non" in r
            ):
                response = 1
            elif "mmr" in r:
                response = 0

        out[gsm] = {
            "timepoint": timepoint,
            "response": response,
            "source": str(path),
        }

    if len(out) < 100:
        return None

    return out


def resolve_probe_set(
    gene,
    matched_entrez_ids,
    platform,
):
    exact = sorted(
        platform[
            "symbol_to_probes"
        ].get(
            gene,
            set(),
        )
    )

    if exact:
        return exact, "EXACT_SYMBOL"

    eids = [
        x
        for x in str(
            matched_entrez_ids
        ).split("|")
        if x
    ]

    probes = set()

    for eid in eids:
        probes.update(
            platform[
                "entrez_to_probes"
            ].get(
                eid,
                set(),
            )
        )

    if probes:
        return sorted(
            probes
        ), "ENTREZ_ID_MATCH"

    return [], "UNRESOLVED"


def sigmoid(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    x = np.clip(
        x,
        -700,
        700,
    )

    return 1.0 / (
        1.0 + np.exp(-x)
    )


def main():
    project = project_dir()
    docs = project / "docs"
    ds = (
        project
        / "results"
        / "direct_subspace"
    )

    protocol = (
        docs
        / "v44_5_CML_EXTERNAL_GENERALIZATION_PREREGISTRATION.txt"
    )

    protocol_manifest = (
        docs
        / "v44_5_CML_EXTERNAL_GENERALIZATION_PREREGISTRATION_manifest.json"
    )

    transport_manifest = (
        ds
        / "v44_5a_manifest.json"
    )

    transport_audit = (
        ds
        / "v44_5a_GSE44589_transport_audit.csv"
    )

    frozen_model = (
        ds
        / "v44_5a_frozen_development_model.npz"
    )

    for p in [
        protocol,
        protocol_manifest,
        transport_manifest,
        transport_audit,
        frozen_model,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                p
            )

    pm = json.loads(
        protocol_manifest.read_text(
            encoding="utf-8"
        )
    )

    expected_sha = pm[
        "protocol_sha256"
    ]

    actual_sha = sha256_file(
        protocol
    )

    print(
        "=== v44.5b CML PRIMARY EXTERNAL GENERALIZATION: GSE44589 ==="
    )

    print(
        "Expected protocol SHA:",
        expected_sha,
    )

    print(
        "Actual protocol SHA:  ",
        actual_sha,
    )

    if expected_sha != actual_sha:
        raise SystemExit(
            "FAIL: v44.5 protocol SHA mismatch."
        )

    tm = json.loads(
        transport_manifest.read_text(
            encoding="utf-8"
        )
    )

    if (
        tm.get("status")
        != "TECHNICALLY EVALUABLE"
    ):
        raise RuntimeError(
            "v44.5a technical transport did not PASS."
        )

    if int(
        tm.get(
            "transportable_coordinates",
            0,
        )
    ) != TOP_K:
        raise RuntimeError(
            "v44.5a did not transport all 256 coordinates."
        )

    print(
        "PASS: v44.5 protocol verified."
    )

    print(
        "PASS: v44.5a 256/256 technical transport verified."
    )

    print()

    data_root = (
        project
        / "data"
        / "external"
        / GSE
    )

    raw_dir = (
        data_root
        / "raw"
    )

    meta_dir = (
        data_root
        / "metadata"
    )

    platform_dir = (
        data_root
        / "platform"
    )

    series_path = (
        raw_dir
        / "GSE44589_series_matrix.txt.gz"
    )

    soft_path = (
        meta_dir
        / "GSE44589_family.soft.gz"
    )

    gpl570_path = (
        platform_dir
        / "GPL570_full_geo_table.txt"
    )

    download_if_missing(
        SERIES_MATRIX_URL,
        series_path,
    )

    download_if_missing(
        SOFT_URL,
        soft_path,
    )

    if not gpl570_path.exists():
        raise FileNotFoundError(
            gpl570_path
        )

    X_probe, matrix_meta = parse_series_matrix(
        series_path
    )

    soft_records = parse_soft_sample_metadata(
        soft_path
    )

    platform = parse_platform(
        gpl570_path
    )

    sample_ids = set(
        X_probe.index.astype(
            str
        )
    )

    # Reuse prior v42.1c audit if a compatible classification table exists.
    prior = None
    prior_source = None

    for candidate in locate_prior_v42_audit(
        project
    ):
        loaded = try_load_prior_classification(
            candidate,
            sample_ids,
        )

        if loaded is not None:
            prior = loaded
            prior_source = str(
                candidate
            )
            break

    classification = {}

    for gsm in X_probe.index.astype(
        str
    ):
        text = flatten_metadata_text(
            gsm,
            matrix_meta,
            soft_records,
        )

        inferred_time = infer_timepoint(
            text
        )

        inferred_response = infer_response(
            text
        )

        if (
            prior is not None
            and gsm in prior
            and prior[gsm][
                "timepoint"
            ]
            != "unknown"
        ):
            timepoint = prior[
                gsm
            ][
                "timepoint"
            ]

            response = prior[
                gsm
            ][
                "response"
            ]

            source = (
                "PRIOR_V42_1C_AUDIT"
            )
        else:
            timepoint = (
                inferred_time
            )

            response = (
                inferred_response
            )

            source = (
                "GEO_METADATA"
            )

        classification[
            gsm
        ] = {
            "timepoint": timepoint,
            "response": response,
            "source": source,
        }

    baseline_ids = [
        gsm
        for gsm in X_probe.index.astype(
            str
        )
        if classification[
            gsm
        ][
            "timepoint"
        ]
        == "baseline"
    ]

    post_ids = [
        gsm
        for gsm in X_probe.index.astype(
            str
        )
        if classification[
            gsm
        ][
            "timepoint"
        ]
        == "post6w"
    ]

    unknown_time_ids = [
        gsm
        for gsm in X_probe.index.astype(
            str
        )
        if classification[
            gsm
        ][
            "timepoint"
        ]
        == "unknown"
    ]

    print(
        "Sample classification source:",
        prior_source
        if prior_source
        else "GEO metadata parser",
    )

    print(
        "Baseline samples:",
        len(baseline_ids),
    )

    print(
        "Post-6-week samples:",
        len(post_ids),
    )

    print(
        "Unknown timepoint:",
        len(unknown_time_ids),
    )

    # Safety gate: the previously audited structure must be reproduced.
    if (
        len(baseline_ids) != 135
        or len(post_ids) != 63
    ):
        raise RuntimeError(
            "ABORT BEFORE OUTCOME SCORING: "
            f"expected 135 baseline and 63 post-6w, "
            f"found {len(baseline_ids)} and {len(post_ids)}. "
            "Sample classification does not reproduce the frozen audit."
        )

    # -----------------------------
    # OUTCOME-BLIND EXPRESSION STEP
    # -----------------------------
    audit = pd.read_csv(
        transport_audit
    )

    model = np.load(
        frozen_model,
        allow_pickle=False,
    )

    frozen_genes = [
        str(x)
        for x in model[
            "genes"
        ].tolist()
    ]

    if len(
        frozen_genes
    ) != TOP_K:
        raise RuntimeError(
            "Frozen model does not contain exactly 256 genes."
        )

    audit = audit.sort_values(
        "coordinate_index"
    ).reset_index(
        drop=True
    )

    audit_genes = audit[
        "development_gene"
    ].astype(
        str
    ).tolist()

    if (
        audit_genes
        != frozen_genes
    ):
        raise RuntimeError(
            "Frozen model gene order and v44.5a transport audit disagree."
        )

    Xext = np.zeros(
        (
            len(
                baseline_ids
            ),
            TOP_K,
        ),
        dtype=np.float64,
    )

    external_mapping_rows = []

    for j, row in audit.iterrows():
        gene = str(
            row[
                "development_gene"
            ]
        )

        probes, mapping_type = resolve_probe_set(
            gene,
            row.get(
                "matched_entrez_ids",
                "",
            ),
            platform,
        )

        usable = [
            p
            for p in probes
            if p in X_probe.columns
        ]

        if not usable:
            raise RuntimeError(
                f"Frozen coordinate {j} {gene}: no usable GPL570 probes."
            )

        vals = (
            X_probe.loc[
                baseline_ids,
                usable,
            ]
            .mean(
                axis=1
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        Xext[
            :,
            j
        ] = vals

        external_mapping_rows.append({
            "coordinate_index": j,
            "development_gene": gene,
            "mapping_type": mapping_type,
            "n_gpl570_probes": len(
                usable
            ),
            "gpl570_probes": "|".join(
                usable
            ),
        })

    # Outcome-blind external cohort z-standardization.
    ext_mean = np.mean(
        Xext,
        axis=0,
    )

    ext_sd = np.std(
        Xext,
        axis=0,
        ddof=0,
    )

    zero_sd = np.where(
        ext_sd == 0
    )[0]

    if len(zero_sd):
        genes = [
            frozen_genes[
                i
            ]
            for i in zero_sd
        ]

        raise RuntimeError(
            "ABORT BEFORE OUTCOME SCORING: zero external SD for frozen genes: "
            + ", ".join(
                genes
            )
        )

    Zext = (
        Xext
        - ext_mean
    ) / ext_sd

    U16 = np.asarray(
        model[
            "U16"
        ],
        dtype=np.float64,
    )

    coef = np.asarray(
        model[
            "logistic_coef"
        ],
        dtype=np.float64,
    )

    intercept = np.asarray(
        model[
            "logistic_intercept"
        ],
        dtype=np.float64,
    )

    if U16.shape != (
        TOP_K,
        FROZEN_R,
    ):
        raise RuntimeError(
            f"Unexpected U16 shape {U16.shape}."
        )

    Zsub = Zext @ U16

    frozen_logit = (
        Zsub @ coef.T
        + intercept.reshape(
            1,
            -1,
        )
    ).ravel()

    frozen_prob = sigmoid(
        frozen_logit
    )

    # -------------------------------------
    # ONLY NOW CONSUME EXTERNAL OUTCOME LABELS
    # -------------------------------------
    y_all = np.asarray(
        [
            (
                classification[
                    gsm
                ][
                    "response"
                ]
                if classification[
                    gsm
                ][
                    "response"
                ]
                is not None
                else -1
            )
            for gsm in baseline_ids
        ],
        dtype=int,
    )

    evaluable_mask = (
        y_all >= 0
    )

    evaluable_ids = np.asarray(
        baseline_ids
    )[
        evaluable_mask
    ]

    y = y_all[
        evaluable_mask
    ]

    score = frozen_prob[
        evaluable_mask
    ]

    n_pos = int(
        np.sum(
            y == 1
        )
    )

    n_neg = int(
        np.sum(
            y == 0
        )
    )

    n_unavailable = int(
        np.sum(
            ~evaluable_mask
        )
    )

    print()
    print(
        "External response labels now consumed."
    )

    print(
        "Evaluable baseline samples:",
        len(y),
    )

    print(
        "Positive / poor response:",
        n_pos,
    )

    print(
        "Negative / MMR:",
        n_neg,
    )

    print(
        "Unavailable:",
        n_unavailable,
    )

    # Frozen audit-count gate.
    if (
        len(y) != 128
        or n_pos != 59
        or n_neg != 69
        or n_unavailable != 7
    ):
        raise RuntimeError(
            "ABORT: response classification does not reproduce frozen "
            f"v42.1c counts. Found evaluable={len(y)}, poor={n_pos}, "
            f"MMR={n_neg}, unavailable={n_unavailable}; "
            "expected 128, 59, 69, 7."
        )

    prevalence = float(
        np.mean(
            y
        )
    )

    pr_auc = float(
        average_precision_score(
            y,
            score,
        )
    )

    roc_auc = float(
        roc_auc_score(
            y,
            score,
        )
    )

    pred = (
        score >= 0.5
    ).astype(
        int
    )

    ba = float(
        balanced_accuracy_score(
            y,
            pred,
        )
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[
            0,
            1,
        ],
    ).ravel()

    sensitivity = (
        float(
            tp / (
                tp + fn
            )
        )
        if (
            tp + fn
        )
        else float(
            "nan"
        )
    )

    specificity = (
        float(
            tn / (
                tn + fp
            )
        )
        if (
            tn + fp
        )
        else float(
            "nan"
        )
    )

    mean_pos_score = float(
        np.mean(
            score[
                y == 1
            ]
        )
    )

    mean_neg_score = float(
        np.mean(
            score[
                y == 0
            ]
        )
    )

    direction_preserved = (
        mean_pos_score
        > mean_neg_score
    )

    status = (
        "PASS"
        if (
            pr_auc
            > prevalence
            and roc_auc
            > 0.5
            and direction_preserved
        )
        else "FAIL"
    )

    # Outputs.
    mapping_out = (
        ds
        / "v44_5b_GSE44589_frozen_coordinate_mapping.csv"
    )

    sample_out = (
        ds
        / "v44_5b_GSE44589_external_predictions.csv"
    )

    summary_out = (
        ds
        / "v44_5b_GSE44589_external_summary.txt"
    )

    manifest_out = (
        ds
        / "v44_5b_manifest.json"
    )

    pd.DataFrame(
        external_mapping_rows
    ).to_csv(
        mapping_out,
        index=False,
    )

    pred_df = pd.DataFrame({
        "gsm": evaluable_ids,
        "y_poor_response": y,
        "frozen_probability": score,
        "frozen_prediction_at_0_5": pred,
    })

    pred_df.to_csv(
        sample_out,
        index=False,
    )

    summary = [
        "=== Soft Spaces / CML v44.5b PRIMARY EXTERNAL GENERALIZATION: GSE44589 ===",
        "",
        "FROZEN DESIGN",
        "-------------",
        "Primary external cohort: GSE44589",
        "Population: pretreatment/baseline only",
        "Positive class: no MMR / poor response",
        "Negative class: MMR",
        f"Frozen Top-K: {TOP_K}",
        f"Frozen rank: {FROZEN_R}",
        "External normalization: per-cohort per-gene z-standardization",
        "External model refitting: NO",
        "External tuning: NO",
        "",
        "SAMPLE COUNTS",
        "-------------",
        f"Baseline samples:       {len(baseline_ids)}",
        f"Evaluable samples:      {len(y)}",
        f"Poor response / none:   {n_pos}",
        f"MMR:                    {n_neg}",
        f"Unavailable outcome:    {n_unavailable}",
        "",
        "PRIMARY METRIC",
        "--------------",
        f"Positive-class prevalence: {prevalence:.6f}",
        f"PR-AUC:                    {pr_auc:.6f}",
        f"PR-AUC - prevalence:       {pr_auc - prevalence:+.6f}",
        "",
        "SECONDARY METRICS",
        "-----------------",
        f"ROC-AUC:                   {roc_auc:.6f}",
        f"Balanced accuracy @0.5:    {ba:.6f}",
        f"Sensitivity @0.5:          {sensitivity:.6f}",
        f"Specificity @0.5:          {specificity:.6f}",
        "",
        "FROZEN DIRECTION CHECK",
        "----------------------",
        f"Mean score poor response:  {mean_pos_score:.6f}",
        f"Mean score MMR:            {mean_neg_score:.6f}",
        f"Direction preserved:       {'YES' if direction_preserved else 'NO'}",
        "",
        "DECISION",
        "--------",
        "PASS requires:",
        "PR-AUC > prevalence, ROC-AUC > 0.5, and frozen direction preserved.",
        "",
        f"v44.5b STATUS: {status}",
        "",
        "INTERPRETATION",
        "--------------",
        "This is the preregistered primary cross-endpoint external test.",
        "No external outcome-driven model tuning or rescue was performed.",
        "",
        "A PASS supports cross-cohort predictive generalization only.",
        "It does not establish clinical utility, causal biology,",
        "superiority to established CML biomarkers, or quantum advantage.",
    ]

    summary_out.write_text(
        "\n".join(
            summary
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_out.write_text(
        json.dumps(
            {
                "version": VERSION,
                "status": status,
                "primary_external_cohort": GSE,
                "baseline_n": len(
                    baseline_ids
                ),
                "evaluable_n": len(
                    y
                ),
                "positive_n": n_pos,
                "negative_n": n_neg,
                "unavailable_n": n_unavailable,
                "prevalence": prevalence,
                "pr_auc": pr_auc,
                "pr_auc_minus_prevalence": (
                    pr_auc
                    - prevalence
                ),
                "roc_auc": roc_auc,
                "balanced_accuracy": ba,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "mean_positive_score": mean_pos_score,
                "mean_negative_score": mean_neg_score,
                "direction_preserved": bool(
                    direction_preserved
                ),
                "external_refit": False,
                "external_tuning": False,
                "sample_classification_source": (
                    prior_source
                    if prior_source
                    else "GEO_METADATA"
                ),
                "sha256": {
                    "v44_5_protocol": actual_sha,
                    "v44_5a_manifest": sha256_file(
                        transport_manifest
                    ),
                    "frozen_model": sha256_file(
                        frozen_model
                    ),
                    "transport_audit": sha256_file(
                        transport_audit
                    ),
                    "series_matrix": sha256_file(
                        series_path
                    ),
                    "execution_script": sha256_file(
                        Path(
                            __file__
                        ).resolve()
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

    print(
        "Wrote:"
    )

    for p in [
        mapping_out,
        sample_out,
        summary_out,
        manifest_out,
    ]:
        print(
            " ",
            p,
        )


if __name__ == "__main__":
    main()
