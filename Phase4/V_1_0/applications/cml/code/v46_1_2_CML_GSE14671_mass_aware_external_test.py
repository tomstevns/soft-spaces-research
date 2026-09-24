#!/usr/bin/env python3
"""
Soft Spaces / CML
v46.1-v46.2 — Mass-aware transport preregistration + first GSE14671 scoring

ONE-RUN SCRIPT
--------------
This script performs two strictly ordered phases:

PHASE A — BEFORE outcome scoring
    1. verify v45 closure and v46.0 frozen-model importance audit;
    2. write and SHA-freeze the v46.1 mass-aware transport protocol;
    3. evaluate only the technical missing-coordinate mass gate.

PHASE B — ONLY IF PHASE A PASSES
    4. build the GSE14671 external representation outcome-blindly;
    5. set technically absent accepted coordinates to z=0
       (cohort mean in standardized coordinate space);
    6. only then attach frozen response labels;
    7. apply the frozen v44.5a U16 + logistic model;
    8. calculate the preregistered external metrics.

No model refit.
No gene substitution.
No outcome-driven preprocessing.
No threshold tuning.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
)


VERSION_PROTOCOL = "v46.1"
VERSION_SCORE = "v46.2"

TOP_K = 256
R = 16

MAX_LOST_PROJECTOR_MASS_FRACTION = 0.01
MAX_LOST_CLASSIFIER_WEIGHT_FRACTION = 0.01

GSE = "GSE14671"
EXPECTED_N = 59
EXPECTED_POSITIVE_N = 18
EXPECTED_NEGATIVE_N = 41


def project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -700, 700)
    return 1.0 / (1.0 + np.exp(-x))


def split_gene_symbols(text):
    if text is None:
        return []

    text = str(text).strip()

    if not text or text in {"---", "NA", "nan"}:
        return []

    parts = re.split(
        r"\s*///\s*|\s*//\s*|\s*;\s*|\s*,\s*",
        text,
    )

    out = []

    for p in parts:
        p = re.sub(r"\s+", "", p.strip())
        if p and p not in {"---", "NA"}:
            out.append(p)

    return out


def parse_family_soft(soft_path: Path):
    probe_to_symbols = {}
    sample_values = {}
    sample_source = {}

    current_entity = None
    current_id = None
    in_platform_table = False
    in_sample_table = False

    platform_header = None
    sample_header = None

    probe_col_idx = None
    symbol_col_idx = None
    sample_probe_idx = None
    sample_value_idx = None

    with gzip.open(
        soft_path,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if line.startswith("^PLATFORM"):
                current_entity = "PLATFORM"
                current_id = line.split("=", 1)[1].strip()
                in_platform_table = False
                in_sample_table = False
                platform_header = None
                probe_col_idx = None
                symbol_col_idx = None
                continue

            if line.startswith("^SAMPLE"):
                current_entity = "SAMPLE"
                current_id = line.split("=", 1)[1].strip()
                in_platform_table = False
                in_sample_table = False
                sample_header = None
                sample_probe_idx = None
                sample_value_idx = None
                sample_values[current_id] = {}
                sample_source[current_id] = ""
                continue

            if (
                current_entity == "SAMPLE"
                and line.startswith("!Sample_source_name_ch1")
                and "=" in line
            ):
                sample_source[current_id] = line.split("=", 1)[1].strip()
                continue

            if current_entity == "PLATFORM" and current_id == "GPL570":
                if line == "!platform_table_begin":
                    in_platform_table = True
                    platform_header = None
                    continue

                if line == "!platform_table_end":
                    in_platform_table = False
                    continue

                if in_platform_table:
                    fields = line.split("\t")

                    if platform_header is None:
                        platform_header = fields
                        norm = [x.strip().lower() for x in fields]

                        for cand in (
                            "id",
                            "id_ref",
                            "probe set id",
                            "probe_set_id",
                        ):
                            if cand in norm:
                                probe_col_idx = norm.index(cand)
                                break

                        if probe_col_idx is None:
                            probe_col_idx = 0

                        for cand in (
                            "gene symbol",
                            "gene_symbol",
                            "symbol",
                        ):
                            if cand in norm:
                                symbol_col_idx = norm.index(cand)
                                break

                        if symbol_col_idx is None:
                            for i, h in enumerate(norm):
                                if "gene" in h and "symbol" in h:
                                    symbol_col_idx = i
                                    break

                        if symbol_col_idx is None:
                            raise RuntimeError(
                                "Could not identify GPL570 gene-symbol column."
                            )

                        continue

                    max_idx = max(
                        probe_col_idx,
                        symbol_col_idx,
                    )

                    if len(fields) <= max_idx:
                        continue

                    probe = fields[probe_col_idx].strip()
                    symbols = split_gene_symbols(
                        fields[symbol_col_idx]
                    )

                    if probe and symbols:
                        probe_to_symbols[probe] = symbols

                    continue

            if current_entity == "SAMPLE":
                if line == "!sample_table_begin":
                    in_sample_table = True
                    sample_header = None
                    continue

                if line == "!sample_table_end":
                    in_sample_table = False
                    continue

                if in_sample_table:
                    fields = line.split("\t")

                    if sample_header is None:
                        sample_header = fields
                        norm = [
                            x.strip().lower()
                            for x in fields
                        ]

                        if "id_ref" not in norm:
                            raise RuntimeError(
                                f"{current_id}: no ID_REF column."
                            )

                        if "value" not in norm:
                            raise RuntimeError(
                                f"{current_id}: no VALUE column."
                            )

                        sample_probe_idx = norm.index("id_ref")
                        sample_value_idx = norm.index("value")
                        continue

                    max_idx = max(
                        sample_probe_idx,
                        sample_value_idx,
                    )

                    if len(fields) <= max_idx:
                        continue

                    probe = fields[sample_probe_idx].strip()

                    try:
                        value = float(
                            fields[sample_value_idx].strip()
                        )
                    except ValueError:
                        continue

                    sample_values[current_id][probe] = value

    return (
        probe_to_symbols,
        sample_values,
        sample_source,
    )


def classify_response(source_name: str):
    t = re.sub(
        r"\s+",
        " ",
        str(source_name).strip().lower(),
    )

    if (
        "complete cytogenetic response" in t
        or "ccyr" in t
    ):
        return 0

    if (
        "65% ph-positive" in t
        or "65% ph positive" in t
        or ">65% ph" in t
    ):
        return 1

    return None


def main():
    project = project_dir()

    docs = project / "docs"
    ds = project / "results" / "direct_subspace"
    metadata = (
        project
        / "data"
        / "external"
        / GSE
        / "metadata"
    )

    docs.mkdir(parents=True, exist_ok=True)
    ds.mkdir(parents=True, exist_ok=True)

    model_path = ds / "v44_5a_frozen_development_model.npz"
    v45_closure = docs / "v45_3_CML_GSE14671_CLOSURE_AUDIT.txt"
    v46_importance = ds / "v46_0_manifest.json"
    soft_path = metadata / "GSE14671_family.soft.gz"

    for p in [
        model_path,
        v45_closure,
        v46_importance,
        soft_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    imp = json.loads(
        v46_importance.read_text(
            encoding="utf-8"
        )
    )

    if imp.get("target_gene") != "PTPN20":
        raise RuntimeError(
            "v46.0 target gene is not PTPN20."
        )

    if imp.get("v45_reopened") is not False:
        raise RuntimeError(
            "Unexpected v46.0 provenance state."
        )

    model = np.load(
        model_path,
        allow_pickle=False,
    )

    genes = np.asarray(
        [
            str(x)
            for x in model["genes"].tolist()
        ]
    )

    U = np.asarray(
        model["U16"],
        dtype=float,
    )

    coef = np.asarray(
        model["logistic_coef"],
        dtype=float,
    ).reshape(-1)

    intercept = float(
        np.asarray(
            model["logistic_intercept"],
            dtype=float,
        ).reshape(-1)[0]
    )

    if len(genes) != TOP_K:
        raise RuntimeError(
            f"Expected {TOP_K} genes."
        )

    if U.shape != (TOP_K, R):
        raise RuntimeError(
            f"Unexpected U shape: {U.shape}"
        )

    if coef.shape[0] != R:
        raise RuntimeError(
            f"Unexpected classifier coefficient shape: {coef.shape}"
        )

    beta_gene = U @ coef
    leverage = np.sum(
        U ** 2,
        axis=1,
    )

    classifier_weight = beta_gene ** 2
    total_classifier_weight = float(
        classifier_weight.sum()
    )

    if total_classifier_weight <= 0:
        raise RuntimeError(
            "Zero classifier-weight mass."
        )

    # ---------------------------------------------------------
    # PHASE A1: technical mapping before response attachment.
    # ---------------------------------------------------------
    print(
        "=== v46.1-v46.2 MASS-AWARE GSE14671 TEST ==="
    )
    print()
    print(
        "PHASE A: technical transport before outcome scoring"
    )

    (
        probe_to_symbols,
        sample_values,
        sample_source,
    ) = parse_family_soft(
        soft_path
    )

    if len(sample_values) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} samples, "
            f"found {len(sample_values)}."
        )

    symbol_to_probes = defaultdict(list)

    for probe, symbols in probe_to_symbols.items():
        for symbol in symbols:
            symbol_to_probes[symbol].append(
                probe
            )

    missing_idx = []
    observed_idx = []

    for i, gene in enumerate(genes):
        probes = sorted(
            set(
                symbol_to_probes.get(
                    gene,
                    [],
                )
            )
        )

        if probes:
            observed_idx.append(i)
        else:
            missing_idx.append(i)

    missing_genes = [
        str(genes[i])
        for i in missing_idx
    ]

    lost_projector_mass = float(
        leverage[missing_idx].sum()
    )

    lost_projector_mass_fraction = (
        lost_projector_mass / float(R)
    )

    lost_classifier_weight = float(
        classifier_weight[
            missing_idx
        ].sum()
    )

    lost_classifier_weight_fraction = (
        lost_classifier_weight
        / total_classifier_weight
    )

    technical_gate_pass = (
        lost_projector_mass_fraction
        <= MAX_LOST_PROJECTOR_MASS_FRACTION
        and
        lost_classifier_weight_fraction
        <= MAX_LOST_CLASSIFIER_WEIGHT_FRACTION
    )

    # ---------------------------------------------------------
    # PHASE A2: write protocol BEFORE any outcome scoring.
    # ---------------------------------------------------------
    protocol = f"""Soft Spaces / CML
v46.1 — MASS-AWARE EXTERNAL TRANSPORT PREREGISTRATION

STATUS
------
FROZEN BEFORE GSE14671 PREDICTIVE SCORING

RATIONALE
---------
v45 used an exact 256/256 coordinate rule and was correctly closed as
TECHNICALLY NON-EVALUABLE when PTPN20 could not be represented on GPL570.

v46.0 subsequently quantified PTPN20 using only the already-frozen v44.5a
model and no GSE14671 outcome scores.

The v46 transport rule therefore replaces the binary all-or-nothing
coordinate criterion with a prospective model-mass criterion.

This does not reopen or reinterpret v45.

FROZEN TRANSPORT RULE
---------------------
Missing frozen coordinates are technically acceptable only if BOTH:

1. total lost projector-mass fraction <= 1.0%

and

2. total lost classifier-weight fraction <= 1.0%

Definitions:

For frozen U16:
    leverage_i = sum_k U[i,k]^2

Total projector mass:
    trace(U U^T) = r = 16

Lost projector-mass fraction:
    sum_missing leverage_i / 16

For frozen subspace classifier coefficient beta_sub:

    beta_gene = U16 @ beta_sub

Classifier-weight mass:
    beta_gene_i^2

Lost classifier-weight fraction:
    sum_missing beta_gene_i^2 / sum_all beta_gene_i^2

FROZEN MISSING-COORDINATE REPRESENTATION
----------------------------------------
For an accepted technically absent coordinate:

    z_i = 0

after external cohort per-gene z-standardization of measurable coordinates.

This is equivalent to imputing the external cohort mean in standardized
coordinate space.

No biological proxy gene is substituted.

No coefficient is altered.

No rank is changed.

No model is refit.

GSE14671 TECHNICAL STATE KNOWN BEFORE SCORING
---------------------------------------------
Frozen Top-K:
    256

Missing coordinates:
    {len(missing_genes)}

Missing genes:
    {", ".join(missing_genes) if missing_genes else "NONE"}

Lost projector-mass fraction:
    {lost_projector_mass_fraction:.12g}

Lost projector-mass percent:
    {100*lost_projector_mass_fraction:.9f}%

Lost classifier-weight fraction:
    {lost_classifier_weight_fraction:.12g}

Lost classifier-weight percent:
    {100*lost_classifier_weight_fraction:.9f}%

Technical mass gate:
    {"PASS" if technical_gate_pass else "FAIL"}

EXTERNAL REPRESENTATION
-----------------------
For measurable frozen genes:

1. GPL570 probes are mapped by Gene Symbol;
2. multiple probes mapping to one frozen gene are averaged arithmetically
   within each sample;
3. each measurable gene is standardized across all 59 GSE14671 samples
   using cohort mean and population SD (ddof=0);
4. accepted missing coordinates are set to z=0;
5. the 256-vector is projected through the frozen U16;
6. the frozen v44.5a logistic classifier is applied.

No response label enters steps 1-4.

EXTERNAL RESPONSE
-----------------
Positive:
    poor response / >65% Ph-positive metaphases after 12 months
    n = 18

Negative:
    CCyR after 12 months
    n = 41

Positive prevalence:
    18/59 = {18/59:.12g}

PRIMARY METRIC
--------------
PR-AUC

SECONDARY
---------
ROC-AUC
Balanced accuracy @0.5
Sensitivity @0.5
Specificity @0.5

FROZEN DIRECTION
----------------
Mean score poor-response > mean score CCyR

PASS RULE
---------
PASS requires:

1. mass-aware technical gate PASS;
2. PR-AUC > prevalence;
3. ROC-AUC > 0.5;
4. mean score poor-response > mean score CCyR;
5. no external refit;
6. no outcome-driven tuning or rescue.

CLAIM LIMIT
-----------
A PASS supports cross-endpoint predictive generalization under the
mass-aware v46 transport rule.

It does not establish clinical utility, causal biology, superiority to
established CML biomarkers, or quantum advantage.
"""

    protocol_path = (
        docs
        / "v46_1_CML_MASS_AWARE_GSE14671_PREREGISTRATION.txt"
    )

    protocol_path.write_text(
        protocol,
        encoding="utf-8",
    )

    protocol_sha = sha256_file(
        protocol_path
    )

    protocol_manifest = {
        "version": VERSION_PROTOCOL,
        "status": (
            "FROZEN_BEFORE_GSE14671_SCORING"
        ),
        "dataset": GSE,
        "max_lost_projector_mass_fraction": (
            MAX_LOST_PROJECTOR_MASS_FRACTION
        ),
        "max_lost_classifier_weight_fraction": (
            MAX_LOST_CLASSIFIER_WEIGHT_FRACTION
        ),
        "missing_coordinate_rule": "z=0 after cohort standardization",
        "frozen_top_k": TOP_K,
        "frozen_rank": R,
        "missing_genes": missing_genes,
        "lost_projector_mass_fraction": (
            lost_projector_mass_fraction
        ),
        "lost_classifier_weight_fraction": (
            lost_classifier_weight_fraction
        ),
        "technical_gate_pass": technical_gate_pass,
        "external_refit": False,
        "outcome_tuning": False,
        "protocol_sha256": protocol_sha,
        "source_sha256": {
            "v44_5a_model": sha256_file(
                model_path
            ),
            "v45_closure": sha256_file(
                v45_closure
            ),
            "v46_0_manifest": sha256_file(
                v46_importance
            ),
            "gse14671_family_soft": sha256_file(
                soft_path
            ),
        },
    }

    protocol_manifest_path = (
        docs
        / "v46_1_CML_MASS_AWARE_GSE14671_PREREGISTRATION_manifest.json"
    )

    protocol_manifest_path.write_text(
        json.dumps(
            protocol_manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    lock_path = (
        docs
        / "v46_1_CML_mass_aware_protocol_lock.py"
    )

    lock_code = (
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import hashlib\n\n"
        f'EXPECTED_SHA256="{protocol_sha}"\n'
        'p=Path(__file__).resolve().parent/'
        '"v46_1_CML_MASS_AWARE_GSE14671_PREREGISTRATION.txt"\n'
        'a=hashlib.sha256(p.read_bytes()).hexdigest()\n'
        'print("Expected:",EXPECTED_SHA256)\n'
        'print("Actual:  ",a)\n'
        'raise SystemExit("FAIL") if a!=EXPECTED_SHA256 '
        'else print("PASS: v46.1 protocol unchanged.")\n'
    )

    lock_path.write_text(
        lock_code,
        encoding="utf-8",
    )

    print(
        "v46.1 protocol SHA:",
        protocol_sha,
    )
    print(
        "Missing genes:",
        missing_genes,
    )
    print(
        "Lost projector mass:",
        f"{100*lost_projector_mass_fraction:.6f}%",
    )
    print(
        "Lost classifier weight:",
        f"{100*lost_classifier_weight_fraction:.6f}%",
    )
    print(
        "Mass-aware technical gate:",
        "PASS"
        if technical_gate_pass
        else "FAIL",
    )

    if not technical_gate_pass:
        raise SystemExit(
            "v46.1 TECHNICAL GATE FAIL. "
            "No outcome scoring performed."
        )

    # ---------------------------------------------------------
    # Build outcome-blind external standardized matrix.
    # ---------------------------------------------------------
    gsms = sorted(
        sample_values.keys()
    )

    Z = np.zeros(
        (len(gsms), TOP_K),
        dtype=float,
    )

    raw_means = np.full(
        TOP_K,
        np.nan,
    )

    raw_sd = np.full(
        TOP_K,
        np.nan,
    )

    for j in observed_idx:
        gene = str(
            genes[j]
        )

        probes = sorted(
            set(
                symbol_to_probes[gene]
            )
        )

        vals = []

        for gsm in gsms:
            pv = [
                sample_values[gsm][p]
                for p in probes
                if p in sample_values[gsm]
            ]

            if not pv:
                raise RuntimeError(
                    f"Missing expression for {gsm}, {gene}."
                )

            vals.append(
                float(
                    np.mean(pv)
                )
            )

        vals = np.asarray(
            vals,
            dtype=float,
        )

        mu = float(
            np.mean(vals)
        )

        sd = float(
            np.std(
                vals,
                ddof=0,
            )
        )

        if sd == 0:
            raise SystemExit(
                "v46.1 TECHNICALLY NON-EVALUABLE: "
                f"zero SD for measurable gene {gene}. "
                "No outcome scoring performed."
            )

        raw_means[j] = mu
        raw_sd[j] = sd

        Z[:, j] = (
            vals - mu
        ) / sd

    # Missing accepted coordinates remain exactly z=0.
    if not np.all(
        np.isfinite(Z)
    ):
        raise RuntimeError(
            "Non-finite standardized matrix."
        )

    representation_out = (
        ds
        / "v46_2_GSE14671_mass_aware_standardized_top256.tsv"
    )

    zdf = pd.DataFrame(
        Z,
        index=gsms,
        columns=genes,
    )

    zdf.index.name = "gsm"

    zdf.to_csv(
        representation_out,
        sep="\t",
    )

    # ---------------------------------------------------------
    # PHASE B: attach response labels only now.
    # ---------------------------------------------------------
    print()
    print(
        "PHASE B: technical representation frozen; "
        "attaching response labels"
    )

    y = []

    for gsm in gsms:
        cls = classify_response(
            sample_source[gsm]
        )

        if cls is None:
            raise RuntimeError(
                f"Unrecognized response label for {gsm}: "
                f"{sample_source[gsm]}"
            )

        y.append(
            cls
        )

    y = np.asarray(
        y,
        dtype=int,
    )

    pos_n = int(
        y.sum()
    )

    neg_n = int(
        len(y) - pos_n
    )

    if (
        pos_n != EXPECTED_POSITIVE_N
        or neg_n != EXPECTED_NEGATIVE_N
    ):
        raise RuntimeError(
            f"Response count mismatch: "
            f"positive={pos_n}, negative={neg_n}."
        )

    Zsub = Z @ U

    logit = (
        Zsub @ coef
        + intercept
    )

    score = sigmoid(
        logit
    )

    prevalence = float(
        np.mean(y)
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
    ).astype(int)

    ba = float(
        balanced_accuracy_score(
            y,
            pred,
        )
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = float(
        tp / (tp + fn)
    ) if (tp + fn) else float("nan")

    specificity = float(
        tn / (tn + fp)
    ) if (tn + fp) else float("nan")

    mean_pos = float(
        np.mean(
            score[y == 1]
        )
    )

    mean_neg = float(
        np.mean(
            score[y == 0]
        )
    )

    direction = (
        mean_pos > mean_neg
    )

    predictive_pass = (
        pr_auc > prevalence
        and
        roc_auc > 0.5
        and
        direction
    )

    status = (
        "PASS"
        if predictive_pass
        else "FAIL"
    )

    scores_out = (
        ds
        / "v46_2_GSE14671_mass_aware_patient_scores.tsv"
    )

    pd.DataFrame({
        "gsm": gsms,
        "response_class": [
            "POOR_RESPONSE"
            if yy == 1
            else "CCyR"
            for yy in y
        ],
        "frozen_probability": score,
        "frozen_logit": logit,
    }).to_csv(
        scores_out,
        sep="\t",
        index=False,
    )

    summary_out = (
        ds
        / "v46_2_GSE14671_mass_aware_external_summary.txt"
    )

    lines = [
        "=== Soft Spaces / CML v46.2 GSE14671 MASS-AWARE EXTERNAL TEST ===",
        "",
        "PROTOCOL",
        "--------",
        f"v46.1 SHA:                       {protocol_sha}",
        "",
        "TECHNICAL TRANSPORT",
        "-------------------",
        f"Frozen Top-K:                    {TOP_K}",
        f"Observed coordinates:            {len(observed_idx)}",
        f"Accepted missing coordinates:    {len(missing_idx)}",
        f"Missing genes:                   {', '.join(missing_genes) if missing_genes else 'NONE'}",
        f"Lost projector mass percent:     {100*lost_projector_mass_fraction:.9f}%",
        f"Allowed maximum:                 {100*MAX_LOST_PROJECTOR_MASS_FRACTION:.6f}%",
        f"Lost classifier weight percent:  {100*lost_classifier_weight_fraction:.9f}%",
        f"Allowed maximum:                 {100*MAX_LOST_CLASSIFIER_WEIGHT_FRACTION:.6f}%",
        "Missing-coordinate representation: z=0",
        "Technical mass-aware gate:       PASS",
        "",
        "EXTERNAL TEST",
        "-------------",
        f"Samples:                         {len(y)}",
        f"Poor response positive:          {pos_n}",
        f"CCyR negative:                   {neg_n}",
        f"Positive prevalence:             {prevalence:.6f}",
        "",
        "PRIMARY",
        "-------",
        f"PR-AUC:                          {pr_auc:.6f}",
        f"PR-AUC - prevalence:             {pr_auc-prevalence:+.6f}",
        "",
        "SECONDARY",
        "---------",
        f"ROC-AUC:                         {roc_auc:.6f}",
        f"Balanced accuracy @0.5:          {ba:.6f}",
        f"Sensitivity @0.5:                {sensitivity:.6f}",
        f"Specificity @0.5:                {specificity:.6f}",
        "",
        "FROZEN DIRECTION",
        "----------------",
        f"Mean score poor response:        {mean_pos:.6f}",
        f"Mean score CCyR:                 {mean_neg:.6f}",
        f"Poor response > CCyR:            {'YES' if direction else 'NO'}",
        "",
        "MODEL INTEGRITY",
        "---------------",
        "External refit:                  NO",
        "Outcome-driven preprocessing:   NO",
        "Gene substitution:              NO",
        "Threshold tuning:               NO",
        "",
        f"v46.2 STATUS: {status}",
        "",
        "INTERPRETATION LIMIT",
        "--------------------",
        "This is a cross-endpoint external test under the prospectively frozen",
        "v46.1 mass-aware transport rule.",
        "",
        "It does not establish clinical utility, causal biology, superiority",
        "to established CML biomarkers, or quantum advantage.",
    ]

    summary_out.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest_out = (
        ds
        / "v46_2_manifest.json"
    )

    manifest = {
        "version": VERSION_SCORE,
        "status": status,
        "dataset": GSE,
        "sample_n": len(y),
        "positive_n": pos_n,
        "negative_n": neg_n,
        "positive_prevalence": prevalence,
        "frozen_top_k": TOP_K,
        "observed_coordinate_n": len(observed_idx),
        "missing_coordinate_n": len(missing_idx),
        "missing_genes": missing_genes,
        "lost_projector_mass_fraction": (
            lost_projector_mass_fraction
        ),
        "lost_classifier_weight_fraction": (
            lost_classifier_weight_fraction
        ),
        "max_lost_projector_mass_fraction": (
            MAX_LOST_PROJECTOR_MASS_FRACTION
        ),
        "max_lost_classifier_weight_fraction": (
            MAX_LOST_CLASSIFIER_WEIGHT_FRACTION
        ),
        "missing_coordinate_representation": "z=0",
        "pr_auc": pr_auc,
        "pr_auc_minus_prevalence": (
            pr_auc - prevalence
        ),
        "roc_auc": roc_auc,
        "balanced_accuracy": ba,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "mean_positive_score": mean_pos,
        "mean_negative_score": mean_neg,
        "direction_preserved": bool(
            direction
        ),
        "external_refit": False,
        "outcome_driven_preprocessing": False,
        "gene_substitution": False,
        "threshold_tuning": False,
        "sha256": {
            "v46_1_protocol": protocol_sha,
            "v44_5a_model": sha256_file(
                model_path
            ),
            "standardized_matrix": sha256_file(
                representation_out
            ),
            "patient_scores": sha256_file(
                scores_out
            ),
            "execution_script": sha256_file(
                Path(__file__).resolve()
            ),
        },
    }

    manifest_out.write_text(
        json.dumps(
            manifest,
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
        "Created protocol:",
        protocol_path,
    )
    print(
        "Created protocol manifest:",
        protocol_manifest_path,
    )
    print(
        "Created protocol lock:",
        lock_path,
    )
    print(
        "Created result summary:",
        summary_out,
    )
    print(
        "Created result manifest:",
        manifest_out,
    )


if __name__ == "__main__":
    main()
