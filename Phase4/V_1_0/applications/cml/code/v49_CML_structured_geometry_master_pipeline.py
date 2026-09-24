#!/usr/bin/env python3
"""
Soft Spaces / CML
v49 MASTER PIPELINE — structured biological-control geometry test

NEW HYPOTHESIS
--------------
Do the frozen cohort-specific Top-256 label-blind geometries show stronger
three-cohort similarity than alternative HIGH-VARIANCE REAL-GENE panels
drawn from the same cohorts while preserving biological covariance?

This is NOT a rescue or re-scoring of v48.3.

No clinical labels are parsed or used anywhere.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

VERSION = "v49"
TOP_K = 256
CONTROL_RANK_START = 257
CONTROL_RANK_END = 2048
PCA_R = 16

PRIMARY_NULL_N = 1000
PRIMARY_SEED = 49020001

ROBUST_N_RESAMPLES = 100
ROBUST_NULL_PER_RESAMPLE = 20
ROBUST_DEV_N = 76
ROBUST_EXT_N = 108
ROBUST_THIRD_N = 47
ROBUST_SEED = 49030001

Z_THRESHOLD = 1.6448536269514722
POSITIVE_FRACTION_THRESHOLD = 0.75

URLS = {
    "GSE130404": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/GSE130404/soft/GSE130404_family.soft.gz",
    "GSE44589": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE44nnn/GSE44589/soft/GSE44589_family.soft.gz",
    "GSE14671": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14671/soft/GSE14671_family.soft.gz",
}


def project_dir():
    return Path(__file__).resolve().parent.parent


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def download_if_missing(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print("Using existing:", path)
        return
    print("Downloading:", url)
    print(" ->", path)
    urllib.request.urlretrieve(url, path)


def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 0:
        raise RuntimeError("Zero norm in cosine similarity.")
    return float(np.dot(a, b) / den)


def wasserstein_1d(x, y):
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    n = max(len(x), len(y))
    q = (np.arange(n, dtype=float) + 0.5) / n
    return float(np.mean(np.abs(
        np.quantile(x, q, method="linear")
        - np.quantile(y, q, method="linear")
    )))


def pairwise_upper(Y):
    Y = np.asarray(Y, dtype=float)
    diff = Y[:, None, :] - Y[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(len(Y), k=1)
    return D[iu]


def geometry_signature(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise RuntimeError("Expected 2D expression matrix.")
    if X.shape[1] != TOP_K:
        raise RuntimeError(f"Expected {TOP_K} features, got {X.shape[1]}.")

    sd = X.std(axis=0, ddof=0)
    if np.any(sd == 0):
        raise RuntimeError("Zero-SD coordinate.")

    Z = (X - X.mean(axis=0)) / sd

    pca = PCA(
        n_components=PCA_R,
        svd_solver="full",
    )
    Y = pca.fit_transform(Z)

    spectrum = pca.explained_variance_.astype(float)
    spectrum /= spectrum.sum()

    pdist = pairwise_upper(Y)
    med = float(np.median(pdist))
    if med <= 0:
        raise RuntimeError("Non-positive median pairwise distance.")

    return {
        "spectrum": spectrum,
        "pdist_norm": pdist / med,
        "cumvar": float(pca.explained_variance_ratio_.sum()),
        "effdim": float(1.0 / np.sum(spectrum * spectrum)),
    }


def pair_metrics(sig_a, sig_b):
    return {
        "cosine": cosine(sig_a["spectrum"], sig_b["spectrum"]),
        "w1": wasserstein_1d(sig_a["pdist_norm"], sig_b["pdist_norm"]),
    }


def triad_metrics(sa, sb, sc):
    ab = pair_metrics(sa, sb)
    ac = pair_metrics(sa, sc)
    bc = pair_metrics(sb, sc)

    return {
        "AB_cosine": ab["cosine"],
        "AC_cosine": ac["cosine"],
        "BC_cosine": bc["cosine"],
        "AB_w1": ab["w1"],
        "AC_w1": ac["w1"],
        "BC_w1": bc["w1"],
        "minimum_pairwise_cosine": min(ab["cosine"], ac["cosine"], bc["cosine"]),
        "maximum_pairwise_w1": max(ab["w1"], ac["w1"], bc["w1"]),
    }


def empirical_high(null, observed):
    null = np.asarray(null, dtype=float)
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def empirical_low(null, observed):
    null = np.asarray(null, dtype=float)
    return float((1 + np.sum(null <= observed)) / (1 + len(null)))


def split_symbols(text):
    text = str(text).strip()
    if not text or text in {"---", "NA", "nan"}:
        return []
    parts = re.split(r"\s*///\s*|\s*//\s*|\s*;\s*|\s*,\s*", text)
    out = []
    for p in parts:
        p = re.sub(r"\s+", "", p.strip())
        if p and p not in {"---", "NA"}:
            out.append(p)
    return out


def parse_soft_expression(soft_path, platform_id, selected_sample_ids, eligible_order):
    """
    Reads ONLY platform annotation and expression tables.
    Clinical phenotype metadata are ignored.
    """
    sample_order = list(map(str, selected_sample_ids))
    sample_set = set(sample_order)
    eligible_set = set(eligible_order)

    probe_to_symbols = {}
    sample_values = {gsm: {} for gsm in sample_order}

    current_entity = None
    current_id = None
    in_platform = False
    in_sample = False
    platform_header = None
    sample_header = None

    probe_idx = symbol_idx = pidx = vidx = None

    with gzip.open(soft_path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if line.startswith("^PLATFORM"):
                current_entity = "PLATFORM"
                current_id = line.split("=", 1)[1].strip()
                in_platform = in_sample = False
                platform_header = None
                continue

            if line.startswith("^SAMPLE"):
                current_entity = "SAMPLE"
                current_id = line.split("=", 1)[1].strip()
                in_platform = in_sample = False
                sample_header = None
                continue

            if current_entity == "PLATFORM" and current_id == platform_id:
                if line == "!platform_table_begin":
                    in_platform = True
                    platform_header = None
                    continue
                if line == "!platform_table_end":
                    in_platform = False
                    continue

                if in_platform:
                    fields = line.split("\t")

                    if platform_header is None:
                        platform_header = fields
                        norm = [x.strip().lower() for x in fields]
                        probe_idx = norm.index("id") if "id" in norm else 0

                        symbol_idx = None
                        for cand in ("gene symbol", "gene_symbol", "symbol"):
                            if cand in norm:
                                symbol_idx = norm.index(cand)
                                break
                        if symbol_idx is None:
                            for i, h in enumerate(norm):
                                if "symbol" in h:
                                    symbol_idx = i
                                    break
                        if symbol_idx is None:
                            raise RuntimeError(
                                f"Could not find gene-symbol column in {platform_id}."
                            )
                        continue

                    if len(fields) <= max(probe_idx, symbol_idx):
                        continue

                    probe = fields[probe_idx].strip()
                    symbols = [
                        s for s in split_symbols(fields[symbol_idx])
                        if s in eligible_set
                    ]
                    if probe and symbols:
                        probe_to_symbols[probe] = symbols
                    continue

            if current_entity == "SAMPLE" and current_id in sample_set:
                if line == "!sample_table_begin":
                    in_sample = True
                    sample_header = None
                    continue
                if line == "!sample_table_end":
                    in_sample = False
                    continue

                if in_sample:
                    fields = line.split("\t")

                    if sample_header is None:
                        sample_header = fields
                        norm = [x.strip().lower() for x in fields]
                        if "id_ref" not in norm or "value" not in norm:
                            raise RuntimeError(
                                f"{current_id}: missing ID_REF/VALUE."
                            )
                        pidx = norm.index("id_ref")
                        vidx = norm.index("value")
                        continue

                    if len(fields) <= max(pidx, vidx):
                        continue

                    probe = fields[pidx].strip()
                    if probe not in probe_to_symbols:
                        continue

                    try:
                        val = float(fields[vidx].strip())
                    except ValueError:
                        continue

                    sample_values[current_id][probe] = val

    symbol_to_probes = defaultdict(list)
    for probe, symbols in probe_to_symbols.items():
        for symbol in symbols:
            symbol_to_probes[symbol].append(probe)

    measurable_set = {
        gene for gene in eligible_order
        if symbol_to_probes.get(gene)
    }
    measurable = [
        gene for gene in eligible_order
        if gene in measurable_set
    ]

    return sample_values, symbol_to_probes, measurable


def build_gene_matrix(sample_values, symbol_to_probes, sample_order, gene_order):
    X = np.empty((len(sample_order), len(gene_order)), dtype=float)

    for j, gene in enumerate(gene_order):
        probes = sorted(set(symbol_to_probes.get(gene, [])))
        if not probes:
            raise RuntimeError(f"No probes for gene {gene}.")

        for i, gsm in enumerate(sample_order):
            vals = [
                sample_values[gsm][p]
                for p in probes
                if p in sample_values[gsm]
            ]
            if not vals:
                raise RuntimeError(f"Missing expression for {gsm}, {gene}.")
            X[i, j] = float(np.mean(vals))

    if not np.all(np.isfinite(X)):
        raise RuntimeError("Non-finite matrix.")

    return X


def rank_genes_by_variance(X, genes):
    var = np.var(X, axis=0, ddof=0)
    order = np.argsort(-var, kind="mergesort")
    ranked_genes = [genes[i] for i in order]
    ranked_var = var[order]
    return ranked_genes, ranked_var, order


def matrix_for_gene_list(X_all, gene_to_col, gene_list):
    idx = [gene_to_col[g] for g in gene_list]
    return X_all[:, idx]


def main():
    project = project_dir()
    docs = project / "docs"
    ds = project / "results" / "direct_subspace"

    docs.mkdir(parents=True, exist_ok=True)
    ds.mkdir(parents=True, exist_ok=True)

    journal_path = ds / "v49_master_journal.txt"

    def journal(text):
        print(text)
        with journal_path.open("a", encoding="utf-8") as f:
            f.write(str(text) + "\n")

    journal_path.write_text("", encoding="utf-8")

    # --------------------------------------------------------
    # Existing frozen artifacts
    # --------------------------------------------------------

    v484_manifest = ds / "v48_4_manifest.json"
    v483_closure_manifest = docs / "v48_3f_CML_THREE_COHORT_CLOSURE_manifest.json"
    eligible_path = ds / "v44_1b_eligible_genes.txt"

    dev_frozen_raw = ds / "v47_2_GSE130404_label_blind_top256_expression.tsv"
    dev_frozen_features = ds / "v47_2_label_blind_top256_features.tsv"

    ext_frozen_geometry = ds / "v47_5b_GSE44589_label_blind_geometry.tsv"
    ext_frozen_features = ds / "v47_5b_GSE44589_label_blind_top256_features.tsv"

    third_frozen_raw = ds / "v48_3b_GSE14671_label_blind_top256_expression.tsv"
    third_frozen_features = ds / "v48_3b_GSE14671_label_blind_top256_features.tsv"

    required = [
        v484_manifest,
        v483_closure_manifest,
        eligible_path,
        dev_frozen_raw,
        dev_frozen_features,
        ext_frozen_geometry,
        ext_frozen_features,
        third_frozen_raw,
        third_frozen_features,
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    diag = json.loads(v484_manifest.read_text(encoding="utf-8"))

    if diag.get("status") != "DIAGNOSTIC_COMPLETE":
        raise RuntimeError("v48.4 diagnostic is not complete.")

    if diag.get("rescue") is not False:
        raise RuntimeError("Unexpected v48.4 rescue flag.")

    # ========================================================
    # v49.0 — preregistration
    # ========================================================

    protocol = f"""Soft Spaces / CML
v49.0 — STRUCTURED BIOLOGICAL-CONTROL GEOMETRY PREREGISTRATION

STATUS
------
FROZEN BEFORE v49 ANALYSIS

RELATION TO v48
---------------
v48.3 remains:
    JOINT THREE-COHORT INVARIANCE NOT SUPPORTED

v48.4 showed that the covariance-destroyed joint NULL produced
near-isotropic geometries that were mutually more similar than the
real biological cohorts.

v49 is a NEW hypothesis and NEW NULL model.
v49 does NOT change, rescue or re-score the formal v48.3 FAIL.

HYPOTHESIS
----------
The frozen cohort-specific Top-256 label-blind geometries show stronger
three-cohort similarity than alternative high-variance REAL-GENE geometries
drawn from the same cohorts.

NO CLINICAL LABELS
------------------
Clinical outcome labels must not be parsed or used anywhere in v49.

COHORTS
-------
A = GSE130404, 96 frozen samples
B = GSE44589, 135 frozen pretreatment samples
C = GSE14671, 59 frozen samples

GENE UNIVERSE
-------------
Existing v44 transport-aware eligible universe:
    15,890 genes

For each cohort independently:
- reconstruct all measurable eligible genes
- rank by raw cohort variance
- verify reconstructed ranks 1..256 exactly match the already-frozen
  observed Top-256 feature SET for that cohort

If exact frozen-feature integrity fails:
    TECHNICAL STOP before primary scoring.

OBSERVED GEOMETRY
-----------------
Frozen Top-256 -> cohort z-standardization -> PCA-{PCA_R}

Observed joint spectrum statistic:
    minimum pairwise normalized-spectrum cosine across AB, AC, BC
    HIGHER = better

Observed joint distance statistic:
    maximum pairwise normalized-distance W1 across AB, AC, BC
    LOWER = better

PRIMARY STRUCTURED NULL
-----------------------
For EACH cohort independently:

1. rank measurable eligible genes by raw variance
2. exclude ranks 1..256
3. define fixed high-variance control pool:
       ranks {CONTROL_RANK_START}..{CONTROL_RANK_END}
4. each NULL replicate draws {TOP_K} genes without replacement
5. use untouched REAL expression values
6. z-standardize
7. PCA-{PCA_R}
8. compute joint triad metrics

NULL triads:
    {PRIMARY_NULL_N}

Seed:
    {PRIMARY_SEED}

The NULL preserves:
- real patients
- real genes
- real covariance
- non-Gaussian expression
- cohort/platform structure
- high-variance feature context
- sample sizes

It does NOT preserve the exact observed Top-256 identities.

PRIMARY PASS RULE
-----------------
BOTH:
1. observed minimum cosine > NULL q95 AND empirical p <= 0.05
2. observed maximum W1 < NULL q05 AND empirical p <= 0.05

v49.3 ROBUSTNESS
----------------
Frozen observed feature identities and frozen control pools.

Paired patient resamples:
    {ROBUST_N_RESAMPLES}

Patients/resample:
    A: {ROBUST_DEV_N}/96
    B: {ROBUST_EXT_N}/135
    C: {ROBUST_THIRD_N}/59

Structured NULL triads/resample:
    {ROBUST_NULL_PER_RESAMPLE}

Signed z:
    z_spectrum =
      (observed min cosine - NULL mean min cosine) / NULL SD

    z_distance =
      (NULL mean max W1 - observed max W1) / NULL SD

Robustness PASS requires BOTH metrics:
    median signed z > {Z_THRESHOLD:.12f}
    fraction signed z > 0 >= {POSITIVE_FRACTION_THRESHOLD:.2f}

FINAL SUPPORT RULE
------------------
STRUCTURED BIOLOGICAL-CONTROL GEOMETRIC INVARIANCE SUPPORTED

ONLY IF:
    technical integrity PASS
    AND primary structured-NULL PASS
    AND robustness PASS

Otherwise:
    NOT SUPPORTED

CLAIM LIMIT
-----------
A full PASS supports only:
the frozen higher-level label-blind geometries are more mutually similar
across these three CML cohorts than alternative high-variance real-gene
geometries under this frozen v49 test.

It does NOT establish prediction, response replication, clinical utility,
causal genes, causal mechanism, dormancy, quantum biology or quantum advantage.

NO RESCUE
---------
No post-hoc feature-count, control-pool rank, PCA-rank, sample-subset,
NULL-definition or threshold tuning.
"""

    protocol_path = docs / "v49_0_CML_STRUCTURED_GEOMETRY_PREREGISTRATION.txt"
    protocol_path.write_text(protocol, encoding="utf-8")
    protocol_sha = sha256_file(protocol_path)

    protocol_manifest = docs / "v49_0_CML_STRUCTURED_GEOMETRY_PREREGISTRATION_manifest.json"
    write_json(
        protocol_manifest,
        {
            "version": "v49.0",
            "status": "FROZEN_BEFORE_ANALYSIS",
            "clinical_labels_used": False,
            "top_k": TOP_K,
            "control_rank_start": CONTROL_RANK_START,
            "control_rank_end": CONTROL_RANK_END,
            "pca_r": PCA_R,
            "primary_null_n": PRIMARY_NULL_N,
            "primary_seed": PRIMARY_SEED,
            "robust_n_resamples": ROBUST_N_RESAMPLES,
            "robust_null_per_resample": ROBUST_NULL_PER_RESAMPLE,
            "robust_seed": ROBUST_SEED,
            "protocol_sha256": protocol_sha,
            "source_sha256": {
                "v48_4_manifest": sha256_file(v484_manifest),
                "v48_3_closure_manifest": sha256_file(v483_closure_manifest),
                "eligible_genes": sha256_file(eligible_path),
            },
        },
    )

    journal("=" * 72)
    journal("v49.0 PROTOCOL FROZEN")
    journal("Protocol SHA: " + protocol_sha)
    journal("Clinical labels used: NO")
    journal("=" * 72)

    # ========================================================
    # Sample IDs / frozen features
    # ========================================================

    A_raw_frozen_df = pd.read_csv(dev_frozen_raw, sep="\t", index_col=0)
    A_samples = [str(x) for x in A_raw_frozen_df.index]
    A_frozen = pd.read_csv(dev_frozen_features, sep="\t")["gene"].astype(str).tolist()

    B_geometry = pd.read_csv(ext_frozen_geometry, sep="\t", index_col=0)
    B_samples = [str(x) for x in B_geometry.index]
    B_frozen = pd.read_csv(ext_frozen_features, sep="\t")["gene"].astype(str).tolist()

    C_raw_frozen_df = pd.read_csv(third_frozen_raw, sep="\t", index_col=0)
    C_samples = [str(x) for x in C_raw_frozen_df.index]
    C_frozen = pd.read_csv(third_frozen_features, sep="\t")["gene"].astype(str).tolist()

    if len(A_samples) != 96:
        raise RuntimeError(f"GSE130404 sample count {len(A_samples)} != 96.")
    if len(B_samples) != 135:
        raise RuntimeError(f"GSE44589 sample count {len(B_samples)} != 135.")
    if len(C_samples) != 59:
        raise RuntimeError(f"GSE14671 sample count {len(C_samples)} != 59.")

    if not (len(A_frozen) == len(B_frozen) == len(C_frozen) == TOP_K):
        raise RuntimeError("Frozen Top256 count mismatch.")

    eligible = [
        x.strip()
        for x in eligible_path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if len(eligible) != 15890:
        raise RuntimeError(f"Eligible universe count {len(eligible)} != 15890.")

    # ========================================================
    # v49.1 — reconstruct full real-gene matrices and integrity
    # ========================================================

    journal("")
    journal("v49.1 — FULL REAL-GENE MATRIX RECONSTRUCTION")
    journal("-------------------------------------------")

    cohort_specs = {
        "A": {
            "gse": "GSE130404",
            "platform": "GPL10558",
            "samples": A_samples,
            "frozen": A_frozen,
        },
        "B": {
            "gse": "GSE44589",
            "platform": "GPL570",
            "samples": B_samples,
            "frozen": B_frozen,
        },
        "C": {
            "gse": "GSE14671",
            "platform": "GPL570",
            "samples": C_samples,
            "frozen": C_frozen,
        },
    }

    matrices = {}
    gene_lists = {}
    gene_to_col = {}
    rankings = {}
    control_pools = {}
    integrity = {}

    for key, spec in cohort_specs.items():
        gse = spec["gse"]
        soft_path = (
            project
            / "data"
            / "external"
            / gse
            / "metadata"
            / f"{gse}_family.soft.gz"
        )

        download_if_missing(URLS[gse], soft_path)

        journal(f"Parsing {key} {gse} ...")

        sample_values, symbol_to_probes, measurable = parse_soft_expression(
            soft_path,
            spec["platform"],
            spec["samples"],
            eligible,
        )

        if len(measurable) < CONTROL_RANK_END:
            raise RuntimeError(
                f"{gse}: only {len(measurable)} measurable eligible genes; "
                f"need at least {CONTROL_RANK_END}."
            )

        X = build_gene_matrix(
            sample_values,
            symbol_to_probes,
            spec["samples"],
            measurable,
        )

        ranked_genes, ranked_var, order = rank_genes_by_variance(
            X,
            measurable,
        )

        reconstructed_top = ranked_genes[:TOP_K]

        exact_set_match = (
            set(reconstructed_top) == set(spec["frozen"])
        )

        exact_order_match = (
            reconstructed_top == spec["frozen"]
        )

        integrity[key] = {
            "gse": gse,
            "measurable_n": len(measurable),
            "exact_top256_set_match": bool(exact_set_match),
            "exact_top256_order_match": bool(exact_order_match),
        }

        if not exact_set_match:
            fail_path = ds / "v49_1_integrity_failure.txt"
            fail_path.write_text(
                f"{gse}: reconstructed Top256 set does not match frozen Top256.\n"
                "v49 stopped before primary scoring.\n",
                encoding="utf-8",
            )
            raise RuntimeError(f"{gse}: frozen Top256 integrity FAIL.")

        matrices[key] = X
        gene_lists[key] = measurable
        gene_to_col[key] = {
            gene: i for i, gene in enumerate(measurable)
        }
        rankings[key] = ranked_genes

        control_pools[key] = ranked_genes[
            CONTROL_RANK_START - 1:
            CONTROL_RANK_END
        ]

        expected_pool_n = (
            CONTROL_RANK_END
            - CONTROL_RANK_START
            + 1
        )
        if len(control_pools[key]) != expected_pool_n:
            raise RuntimeError(f"{gse}: control pool size mismatch.")

        journal(f"{gse}: measurable eligible = {len(measurable)}")
        journal(f"{gse}: frozen Top256 set integrity = PASS")

    integrity_out = ds / "v49_1_full_matrix_integrity_audit.json"
    write_json(
        integrity_out,
        {
            "protocol_sha256": protocol_sha,
            "clinical_labels_used": False,
            "integrity": integrity,
            "control_pool": {
                "rank_start": CONTROL_RANK_START,
                "rank_end": CONTROL_RANK_END,
                "size": CONTROL_RANK_END - CONTROL_RANK_START + 1,
            },
            "status": "PASS",
        },
    )

    # ========================================================
    # Observed frozen Top256 geometries
    # ========================================================

    observed_sig = {}

    for key, spec in cohort_specs.items():
        Xobs = matrix_for_gene_list(
            matrices[key],
            gene_to_col[key],
            spec["frozen"],
        )
        observed_sig[key] = geometry_signature(Xobs)

    observed = triad_metrics(
        observed_sig["A"],
        observed_sig["B"],
        observed_sig["C"],
    )

    # ========================================================
    # v49.2 — primary structured real-gene NULL
    # ========================================================

    journal("")
    journal("v49.2 — PRIMARY STRUCTURED BIOLOGICAL-CONTROL TEST")
    journal("--------------------------------------------------")

    rng = np.random.default_rng(PRIMARY_SEED)
    null_rows = []

    for b in range(PRIMARY_NULL_N):
        null_sig = {}

        for key in ["A", "B", "C"]:
            genes = rng.choice(
                control_pools[key],
                size=TOP_K,
                replace=False,
            ).tolist()

            Xpanel = matrix_for_gene_list(
                matrices[key],
                gene_to_col[key],
                genes,
            )
            null_sig[key] = geometry_signature(Xpanel)

        tm = triad_metrics(
            null_sig["A"],
            null_sig["B"],
            null_sig["C"],
        )

        null_rows.append({
            "null_rep": b + 1,
            **tm,
        })

        if (b + 1) % 100 == 0:
            journal(f"Completed {b+1}/{PRIMARY_NULL_N} structured NULL triads")

    null_df = pd.DataFrame(null_rows)

    null_min_cos = null_df["minimum_pairwise_cosine"].to_numpy(dtype=float)
    null_max_w1 = null_df["maximum_pairwise_w1"].to_numpy(dtype=float)

    q95_cos = float(np.quantile(null_min_cos, 0.95))
    q05_w1 = float(np.quantile(null_max_w1, 0.05))

    p_cos = empirical_high(
        null_min_cos,
        observed["minimum_pairwise_cosine"],
    )

    p_w1 = empirical_low(
        null_max_w1,
        observed["maximum_pairwise_w1"],
    )

    spectrum_gate = bool(
        observed["minimum_pairwise_cosine"] > q95_cos
        and p_cos <= 0.05
    )

    distance_gate = bool(
        observed["maximum_pairwise_w1"] < q05_w1
        and p_w1 <= 0.05
    )

    primary_pass = bool(
        spectrum_gate and distance_gate
    )

    null_out = ds / "v49_2_structured_real_gene_null_1000.tsv"
    null_df.to_csv(null_out, sep="\t", index=False)

    primary_summary = ds / "v49_2_structured_geometry_summary.txt"

    primary_text = f"""=== v49.2 STRUCTURED BIOLOGICAL-CONTROL GEOMETRY TEST ===

Protocol SHA:
    {protocol_sha}

Clinical labels used:
    NO

Control genes:
    REAL genes with untouched patient-level covariance

Control rank pool:
    {CONTROL_RANK_START}..{CONTROL_RANK_END}

NULL triads:
    {PRIMARY_NULL_N}

OBSERVED
--------
AB spectrum cosine:
    {observed["AB_cosine"]:.9f}

AC spectrum cosine:
    {observed["AC_cosine"]:.9f}

BC spectrum cosine:
    {observed["BC_cosine"]:.9f}

Observed minimum pairwise cosine:
    {observed["minimum_pairwise_cosine"]:.9f}

AB distance W1:
    {observed["AB_w1"]:.9f}

AC distance W1:
    {observed["AC_w1"]:.9f}

BC distance W1:
    {observed["BC_w1"]:.9f}

Observed maximum pairwise W1:
    {observed["maximum_pairwise_w1"]:.9f}

SPECTRAL PRIMARY GATE
---------------------
NULL mean minimum cosine:
    {np.mean(null_min_cos):.9f}

NULL q95:
    {q95_cos:.9f}

Empirical p:
    {p_cos:.9f}

Gate:
    {"PASS" if spectrum_gate else "FAIL"}

DISTANCE PRIMARY GATE
---------------------
NULL mean maximum W1:
    {np.mean(null_max_w1):.9f}

NULL q05:
    {q05_w1:.9f}

Empirical p:
    {p_w1:.9f}

Gate:
    {"PASS" if distance_gate else "FAIL"}

v49.2 STATUS:
    {"STRUCTURED-NULL PRIMARY PASS" if primary_pass else "STRUCTURED-NULL PRIMARY FAIL"}
"""

    primary_summary.write_text(primary_text, encoding="utf-8")
    journal(primary_text)

    # ========================================================
    # v49.3 — patient-resampling robustness
    # ========================================================

    journal("")
    journal("v49.3 — PATIENT-RESAMPLING ROBUSTNESS")
    journal("-------------------------------------")

    subsample_sizes = {
        "A": ROBUST_DEV_N,
        "B": ROBUST_EXT_N,
        "C": ROBUST_THIRD_N,
    }

    rng_master = np.random.default_rng(ROBUST_SEED)
    robust_rows = []

    observed_full_matrices = {
        key: matrix_for_gene_list(
            matrices[key],
            gene_to_col[key],
            cohort_specs[key]["frozen"],
        )
        for key in ["A", "B", "C"]
    }

    for r in range(ROBUST_N_RESAMPLES):
        row_idx = {
            key: rng_master.choice(
                matrices[key].shape[0],
                size=subsample_sizes[key],
                replace=False,
            )
            for key in ["A", "B", "C"]
        }

        obs_sub_sig = {
            key: geometry_signature(
                observed_full_matrices[key][row_idx[key], :]
            )
            for key in ["A", "B", "C"]
        }

        obs_sub = triad_metrics(
            obs_sub_sig["A"],
            obs_sub_sig["B"],
            obs_sub_sig["C"],
        )

        local_rng = np.random.default_rng(
            ROBUST_SEED + 100000 + r
        )

        null_sub_cos = np.empty(
            ROBUST_NULL_PER_RESAMPLE,
            dtype=float,
        )

        null_sub_w1 = np.empty(
            ROBUST_NULL_PER_RESAMPLE,
            dtype=float,
        )

        for b in range(ROBUST_NULL_PER_RESAMPLE):
            nsig = {}

            for key in ["A", "B", "C"]:
                genes = local_rng.choice(
                    control_pools[key],
                    size=TOP_K,
                    replace=False,
                ).tolist()

                Xpanel = matrix_for_gene_list(
                    matrices[key],
                    gene_to_col[key],
                    genes,
                )

                nsig[key] = geometry_signature(
                    Xpanel[row_idx[key], :]
                )

            nt = triad_metrics(
                nsig["A"],
                nsig["B"],
                nsig["C"],
            )

            null_sub_cos[b] = nt[
                "minimum_pairwise_cosine"
            ]

            null_sub_w1[b] = nt[
                "maximum_pairwise_w1"
            ]

        sd_cos = float(np.std(null_sub_cos, ddof=1))
        sd_w1 = float(np.std(null_sub_w1, ddof=1))

        if sd_cos <= 0 or sd_w1 <= 0:
            raise RuntimeError(
                f"Resample {r+1}: zero structured-NULL SD."
            )

        z_cos = float(
            (
                obs_sub["minimum_pairwise_cosine"]
                - np.mean(null_sub_cos)
            )
            / sd_cos
        )

        z_w1 = float(
            (
                np.mean(null_sub_w1)
                - obs_sub["maximum_pairwise_w1"]
            )
            / sd_w1
        )

        robust_rows.append({
            "resample": r + 1,
            "observed_min_pairwise_cosine": obs_sub[
                "minimum_pairwise_cosine"
            ],
            "null_mean_min_pairwise_cosine": float(
                np.mean(null_sub_cos)
            ),
            "z_spectrum": z_cos,
            "observed_max_pairwise_w1": obs_sub[
                "maximum_pairwise_w1"
            ],
            "null_mean_max_pairwise_w1": float(
                np.mean(null_sub_w1)
            ),
            "z_distance": z_w1,
        })

        if (r + 1) % 10 == 0:
            journal(
                f"Completed {r+1}/{ROBUST_N_RESAMPLES} patient resamples"
            )

    robust_df = pd.DataFrame(robust_rows)

    median_z_cos = float(
        robust_df["z_spectrum"].median()
    )

    median_z_w1 = float(
        robust_df["z_distance"].median()
    )

    positive_cos = float(
        np.mean(
            robust_df["z_spectrum"] > 0
        )
    )

    positive_w1 = float(
        np.mean(
            robust_df["z_distance"] > 0
        )
    )

    robust_spectrum_gate = bool(
        median_z_cos > Z_THRESHOLD
        and positive_cos >= POSITIVE_FRACTION_THRESHOLD
    )

    robust_distance_gate = bool(
        median_z_w1 > Z_THRESHOLD
        and positive_w1 >= POSITIVE_FRACTION_THRESHOLD
    )

    robustness_pass = bool(
        robust_spectrum_gate
        and robust_distance_gate
    )

    robust_out = ds / "v49_3_structured_null_resampling_results.tsv"
    robust_df.to_csv(robust_out, sep="\t", index=False)

    robust_summary = ds / "v49_3_structured_null_robustness_summary.txt"

    robust_text = f"""=== v49.3 STRUCTURED-NULL PATIENT-RESAMPLING ROBUSTNESS ===

Resamples:
    {ROBUST_N_RESAMPLES}

Structured NULL triads per resample:
    {ROBUST_NULL_PER_RESAMPLE}

SPECTRUM
--------
Median signed z:
    {median_z_cos:.9f}

Required:
    > {Z_THRESHOLD:.9f}

Fraction z > 0:
    {positive_cos:.9f}

Required:
    >= {POSITIVE_FRACTION_THRESHOLD:.9f}

Gate:
    {"PASS" if robust_spectrum_gate else "FAIL"}

DISTANCE
--------
Median signed z:
    {median_z_w1:.9f}

Required:
    > {Z_THRESHOLD:.9f}

Fraction z > 0:
    {positive_w1:.9f}

Required:
    >= {POSITIVE_FRACTION_THRESHOLD:.9f}

Gate:
    {"PASS" if robust_distance_gate else "FAIL"}

v49.3 STATUS:
    {"STRUCTURED-NULL ROBUSTNESS PASS" if robustness_pass else "STRUCTURED-NULL ROBUSTNESS FAIL"}
"""

    robust_summary.write_text(
        robust_text,
        encoding="utf-8",
    )

    journal(robust_text)

    # ========================================================
    # v49.4 — closure
    # ========================================================

    full_support = bool(
        primary_pass
        and robustness_pass
    )

    final_claim = (
        "STRUCTURED BIOLOGICAL-CONTROL GEOMETRIC INVARIANCE SUPPORTED"
        if full_support
        else
        "STRUCTURED BIOLOGICAL-CONTROL GEOMETRIC INVARIANCE NOT SUPPORTED"
    )

    closure_path = docs / "v49_4_CML_STRUCTURED_GEOMETRY_CLOSURE.txt"

    closure = f"""Soft Spaces / CML
v49.4 — STRUCTURED BIOLOGICAL-CONTROL GEOMETRY CLOSURE

STATUS
------
CLOSED

Protocol SHA:
    {protocol_sha}

RELATION TO v48
---------------
v48.3 formal joint FAIL remains unchanged.

v49 used a prospectively frozen NEW structured biological-control NULL
consisting of real high-variance genes with untouched covariance.

Clinical labels used:
    NO

TECHNICAL / INTEGRITY
---------------------
GSE130404 measurable eligible genes:
    {integrity["A"]["measurable_n"]}

GSE44589 measurable eligible genes:
    {integrity["B"]["measurable_n"]}

GSE14671 measurable eligible genes:
    {integrity["C"]["measurable_n"]}

Frozen Top256 set integrity:
    GSE130404: PASS
    GSE44589:  PASS
    GSE14671:  PASS

PRIMARY STRUCTURED-NULL TEST
----------------------------
Observed minimum pairwise cosine:
    {observed["minimum_pairwise_cosine"]:.9f}

Structured NULL q95:
    {q95_cos:.9f}

Empirical spectral p:
    {p_cos:.9f}

Spectral gate:
    {"PASS" if spectrum_gate else "FAIL"}

Observed maximum pairwise W1:
    {observed["maximum_pairwise_w1"]:.9f}

Structured NULL q05:
    {q05_w1:.9f}

Empirical distance p:
    {p_w1:.9f}

Distance gate:
    {"PASS" if distance_gate else "FAIL"}

v49.2:
    {"PASS" if primary_pass else "FAIL"}

PATIENT-RESAMPLING ROBUSTNESS
-----------------------------
Spectrum median signed z:
    {median_z_cos:.9f}

Spectrum positive fraction:
    {positive_cos:.9f}

Spectrum robustness:
    {"PASS" if robust_spectrum_gate else "FAIL"}

Distance median signed z:
    {median_z_w1:.9f}

Distance positive fraction:
    {positive_w1:.9f}

Distance robustness:
    {"PASS" if robust_distance_gate else "FAIL"}

v49.3:
    {"PASS" if robustness_pass else "FAIL"}

FINAL
-----
{final_claim}

NO RESCUE / NO TUNING
---------------------
Control-pool ranks, Top-K, PCA rank, NULL size, resampling design and PASS
rules were frozen before the v49 result.

No post-hoc change was used.

CLAIM LIMIT
-----------
If supported, the result means only:

    the frozen higher-level label-blind transcriptomic geometries are more
    mutually similar across these three CML cohorts than alternative
    high-variance real-gene geometries under the frozen v49 structured
    biological-control test.

It does NOT establish prediction, response replication, clinical utility,
causal biology, dormancy, disease mechanism, quantum biology or quantum
advantage.

TRACK
-----
v49 CLOSED
"""

    closure_path.write_text(
        closure,
        encoding="utf-8",
    )

    closure_sha = sha256_file(
        closure_path
    )

    manifest_path = docs / "v49_4_CML_STRUCTURED_GEOMETRY_CLOSURE_manifest.json"

    write_json(
        manifest_path,
        {
            "version": "v49.4",
            "status": "CLOSED",
            "clinical_labels_used": False,
            "v48_status_changed": False,
            "technical_integrity_pass": True,
            "primary_structured_null_pass": primary_pass,
            "robustness_pass": robustness_pass,
            "full_support": full_support,
            "final_claim": final_claim,
            "observed": observed,
            "primary": {
                "control_rank_start": CONTROL_RANK_START,
                "control_rank_end": CONTROL_RANK_END,
                "n_null": PRIMARY_NULL_N,
                "spectrum_q95": q95_cos,
                "spectrum_p": p_cos,
                "spectrum_gate": spectrum_gate,
                "distance_q05": q05_w1,
                "distance_p": p_w1,
                "distance_gate": distance_gate,
            },
            "robustness": {
                "n_resamples": ROBUST_N_RESAMPLES,
                "null_per_resample": ROBUST_NULL_PER_RESAMPLE,
                "median_z_spectrum": median_z_cos,
                "positive_fraction_spectrum": positive_cos,
                "spectrum_gate": robust_spectrum_gate,
                "median_z_distance": median_z_w1,
                "positive_fraction_distance": positive_w1,
                "distance_gate": robust_distance_gate,
            },
            "protocol_sha256": protocol_sha,
            "closure_sha256": closure_sha,
            "sha256": {
                "execution_script": sha256_file(Path(__file__).resolve()),
                "integrity_audit": sha256_file(integrity_out),
                "primary_null_results": sha256_file(null_out),
                "primary_summary": sha256_file(primary_summary),
                "robustness_results": sha256_file(robust_out),
                "robustness_summary": sha256_file(robust_summary),
                "closure": closure_sha,
            },
        },
    )

    master_summary = ds / "v49_master_summary.txt"

    master_text = f"""=== Soft Spaces / CML v49 MASTER SUMMARY ===

Technical + frozen-feature integrity:
    PASS

Primary structured biological-control test:
    {"PASS" if primary_pass else "FAIL"}

Patient-resampling robustness:
    {"PASS" if robustness_pass else "FAIL"}

FINAL:
    {final_claim}

Protocol SHA:
    {protocol_sha}

Closure SHA:
    {closure_sha}
"""

    master_summary.write_text(
        master_text,
        encoding="utf-8",
    )

    journal("")
    journal("=" * 72)
    journal("v49 MASTER PIPELINE COMPLETE")
    journal("=" * 72)
    journal(master_text)
    journal("Closure:")
    journal("  " + str(closure_path))
    journal("Manifest:")
    journal("  " + str(manifest_path))
    journal("Summary:")
    journal("  " + str(master_summary))


if __name__ == "__main__":
    main()
