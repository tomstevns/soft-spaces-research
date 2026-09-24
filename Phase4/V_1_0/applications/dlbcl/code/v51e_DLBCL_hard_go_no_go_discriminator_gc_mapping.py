#!/usr/bin/env python3
"""
Soft Spaces / DLBCL
v51e MASTER GO/NO-GO DISCRIMINATOR — GC→GCB PARSER

Goal
----
One hard discriminator test to decide whether DLBCL behaves materially
differently from the CML result.

Core question
-------------
Does the ALREADY-FROZEN DLBCL Soft-Spaces gene panel separate ABC/GCB
more strongly than biologically realistic, high-variance REAL-GENE
control panels in independent cohorts?

This script is deliberately conservative:

- It does NOT invent or reconstruct a new Soft-Spaces panel.
- It first searches existing local v41 artifacts for an unambiguous
  previously-frozen DLBCL panel.
- If it cannot identify exactly one defensible frozen panel, it STOPS
  BEFORE scoring and prints the candidate files it found.
- No clinical-label tuning.
- No panel-size tuning.
- No post-hoc rescue.

Planned external cohorts
------------------------
Primary external cohort:
    GSE87371

Second external cohort:
    GSE159472

ABC/GCB labels are used only for the final separation test.
All control-gene selection is label-blind.

Structured control
------------------
For each cohort:
- reconstruct measurable genes from the platform
- rank by raw expression variance without labels
- build a control pool from high-variance real genes
- repeatedly draw matched panels with the SAME gene count as the
  frozen Soft-Spaces panel
- preserve real patients, real genes, and real covariance

Primary statistic
-----------------
ROC-AUC for ABC vs GCB from a simple L2 logistic classifier fitted
WITHIN repeated stratified CV using ONLY the fixed panel coordinates.

For controls, the same CV splits and classifier are used.

Per-cohort PASS:
    frozen-panel mean ROC-AUC > 95th percentile of structured-control NULL
    AND empirical p <= 0.05

Overall DLBCL discriminator PASS:
    BOTH external cohorts PASS

Interpretation
--------------
PASS:
    DLBCL shows evidence materially stronger than what ordinary
    high-variance real-gene panels provide.

FAIL:
    DLBCL does not demonstrate such a distinction under this test.

This is not a clinical model.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


# ============================================================
# FROZEN TEST PARAMETERS
# ============================================================

VERSION = "v51"

COHORTS = {
    "GSE87371": {
        "platform": "GPL570",
        "url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/"
            "GSE87nnn/GSE87371/soft/GSE87371_family.soft.gz"
        ),
    },
    "GSE159472": {
        "platform": "GPL570",
        "url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/"
            "GSE159nnn/GSE159472/soft/GSE159472_family.soft.gz"
        ),
    },
}

CV_SPLITS = 5
CV_REPEATS = 20
CV_SEED = 51010001

NULL_N = 500
NULL_SEED = 51020001

CONTROL_RANK_START = 17
CONTROL_RANK_END = 2048

MAX_PANEL_SIZE = 64

# ============================================================
# EXPLICITLY FROZEN v41.16 PANEL
# ============================================================

FROZEN_PANEL = [
    "DNER",
    "SPIC",
    "TCL1B",
    "MMP20",
    "CKAP2",
    "RNF183",
    "MYOCD",
    "UMODL1",
    "CTAG2",
    "HRK",
    "RGS13",
    "TCL1A",
]

FROZEN_PANEL_PROVENANCE = [
    "v41_16_PREREGISTRATION_manifest.json :: /frozen_panel",
    "v41_16_external_replication_manifest.json :: /frozen_panel",
    "v41_16_mapping_audit.csv :: frozen_gene",
]


# ============================================================
# PATH / HASH
# ============================================================

def project_dir():
    return Path(__file__).resolve().parent.parent


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(
        json.dumps(obj, indent=2),
        encoding="utf-8",
    )


def download_if_missing(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        print("Using existing:", path)
        return

    print("Downloading:", url)
    print(" ->", path)

    urllib.request.urlretrieve(url, path)


# ============================================================
# FROZEN-PANEL DISCOVERY
# ============================================================

GENE_RE = re.compile(r"\b[A-Z][A-Z0-9\-]{1,14}\b")


def plausible_gene(token):
    bad = {
        "PASS", "FAIL", "REAL", "NULL", "ABC", "GCB",
        "AUC", "ROC", "PCA", "SHA", "JSON", "TSV",
        "CSV", "TXT", "GPL", "GSE", "DLBCL", "CML",
        "TRUE", "FALSE", "NONE", "NA", "AND", "OR",
        "WITH", "WITHOUT", "STATUS", "SEED", "TOP",
    }
    return token not in bad and not token.isdigit()


def extract_gene_candidates_from_text(text):
    toks = [
        t for t in GENE_RE.findall(text)
        if plausible_gene(t)
    ]
    # preserve first occurrence order
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def discover_frozen_panel(dlbcl_root):
    """
    Search v41 artifacts for files likely to contain the frozen DLBCL panel.

    Preference:
    1. v41_16
    2. v41_15
    3. v41_13
    and filenames containing gene/frozen/panel/feature/subspace/manifest/protocol.

    We only accept a candidate automatically when exactly one candidate
    panel size in [4,64] is strongly repeated across >=2 relevant artifacts
    OR one JSON/TSV file explicitly exposes a gene list.
    """

    all_files = [
        p for p in dlbcl_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".txt", ".json", ".tsv", ".csv", ".py"}
        and "v41" in p.name.lower()
    ]

    if not all_files:
        raise RuntimeError(
            f"No v41 DLBCL artifacts found under {dlbcl_root}"
        )

    ranked = sorted(
        all_files,
        key=lambda p: (
            0 if "v41_16" in p.name.lower() else
            1 if "v41_15" in p.name.lower() else
            2 if "v41_13" in p.name.lower() else
            3,
            0 if any(k in p.name.lower() for k in (
                "gene", "frozen", "panel", "feature",
                "subspace", "manifest", "protocol"
            )) else 1,
            len(str(p)),
        )
    )

    explicit_candidates = []

    # JSON explicit lists
    for p in ranked:
        if p.suffix.lower() != ".json":
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        def walk(x, keypath=""):
            if isinstance(x, dict):
                for k, v in x.items():
                    walk(v, keypath + "/" + str(k))
            elif isinstance(x, list):
                if 4 <= len(x) <= MAX_PANEL_SIZE and all(
                    isinstance(v, str) for v in x
                ):
                    genes = [
                        str(v).strip()
                        for v in x
                        if plausible_gene(str(v).strip())
                        and GENE_RE.fullmatch(str(v).strip())
                    ]
                    if len(genes) == len(x):
                        explicit_candidates.append(
                            (p, keypath, genes)
                        )
                for v in x:
                    walk(v, keypath)
        walk(obj)

    # TSV/CSV one-column explicit lists
    for p in ranked:
        if p.suffix.lower() not in {".tsv", ".csv"}:
            continue
        try:
            sep = "\t" if p.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(p, sep=sep)
        except Exception:
            continue

        for col in df.columns:
            vals = df[col].dropna().astype(str).str.strip().tolist()
            if 4 <= len(vals) <= MAX_PANEL_SIZE and all(
                GENE_RE.fullmatch(v) and plausible_gene(v)
                for v in vals
            ):
                explicit_candidates.append(
                    (p, f"column:{col}", vals)
                )

    # Deduplicate exact lists
    uniq = {}
    for p, src, genes in explicit_candidates:
        key = tuple(genes)
        uniq.setdefault(key, []).append((p, src))

    if len(uniq) == 1:
        genes = list(next(iter(uniq.keys())))
        sources = next(iter(uniq.values()))
        return genes, sources, ranked

    # Fallback: look for repeated exact 12/16-style gene sets in text.
    text_sets = defaultdict(list)

    for p in ranked[:100]:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Search lines likely to contain gene lists.
        for line in text.splitlines():
            lower = line.lower()
            if not any(k in lower for k in (
                "gene", "frozen", "panel", "top16", "top12",
                "feature", "subspace"
            )):
                continue

            genes = extract_gene_candidates_from_text(line)

            if 4 <= len(genes) <= MAX_PANEL_SIZE:
                key = tuple(genes)
                text_sets[key].append((p, line[:200]))

    repeated = [
        (genes, srcs)
        for genes, srcs in text_sets.items()
        if len(srcs) >= 2
    ]

    if len(repeated) == 1:
        genes, sources = repeated[0]
        return list(genes), sources, ranked

    # No safe automatic resolution.
    discovery_report = dlbcl_root / "results" / "v51_panel_discovery_candidates.txt"
    discovery_report.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "v51 frozen-panel discovery could not identify exactly one safe panel.",
        "",
        "Relevant v41 files found:",
    ]

    for p in ranked[:50]:
        lines.append(str(p))

    lines.append("")
    lines.append("Explicit list candidates:")
    for genes, srcs in uniq.items():
        lines.append(f"GENES ({len(genes)}): {', '.join(genes)}")
        for src in srcs:
            lines.append(f"  {src[0]} :: {src[1]}")

    lines.append("")
    lines.append("Repeated text candidates:")
    for genes, srcs in repeated:
        lines.append(f"GENES ({len(genes)}): {', '.join(genes)}")
        for src in srcs[:10]:
            lines.append(f"  {src[0]}")

    discovery_report.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    raise RuntimeError(
        "Could not identify one unambiguous frozen DLBCL panel. "
        f"See {discovery_report}"
    )


# ============================================================
# GEO SOFT PARSER
# ============================================================

def split_symbols(text):
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


def parse_soft_all(soft_path, platform_id):
    """
    Parse:
    - platform probe->gene annotation
    - sample expression
    - minimal sample characteristic text needed ONLY to extract ABC/GCB

    No treatment-outcome labels are used.
    """

    probe_to_symbols = {}
    sample_values = {}
    sample_text = defaultdict(list)

    current_entity = None
    current_id = None

    in_platform = False
    in_sample = False

    platform_header = None
    sample_header = None

    probe_idx = None
    symbol_idx = None
    pidx = None
    vidx = None

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
                in_platform = False
                in_sample = False
                platform_header = None
                continue

            if line.startswith("^SAMPLE"):
                current_entity = "SAMPLE"
                current_id = line.split("=", 1)[1].strip()
                sample_values[current_id] = {}
                in_platform = False
                in_sample = False
                sample_header = None
                continue

            if current_entity == "SAMPLE" and current_id is not None:
                if line.startswith("!Sample_"):
                    sample_text[current_id].append(line)

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
                        for cand in (
                            "gene symbol",
                            "gene_symbol",
                            "symbol",
                        ):
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
                    syms = split_symbols(fields[symbol_idx])

                    if probe and syms:
                        probe_to_symbols[probe] = syms
                    continue

            if current_entity == "SAMPLE":
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

                    try:
                        value = float(fields[vidx].strip())
                    except ValueError:
                        continue

                    sample_values[current_id][probe] = value

    symbol_to_probes = defaultdict(list)

    for probe, syms in probe_to_symbols.items():
        for s in syms:
            symbol_to_probes[s].append(probe)

    return sample_values, symbol_to_probes, sample_text


def parse_abc_gcb_labels(sample_text):
    """
    v51d PARSER-ONLY FIX.

    Accept ONLY an explicit Sample_characteristics field with key "coo"
    (cell of origin). ABC maps to ABC; GC or GCB maps to the GCB class.

    Examples accepted:
        coo: ABC
        coo: GC
        coo: GCB

    Everything else is ignored, including PMBL and unrelated metadata.

    Returns:
        labels[gsm] = 1 (ABC), 0 (GCB), or None
        audit_rows   = parser audit records
    """

    labels = {}
    audit_rows = []

    for gsm, lines in sample_text.items():
        parsed = None
        matched_line = None
        all_coo_lines = []

        for raw in lines:
            if not raw.startswith("!Sample_characteristics"):
                continue

            rhs = raw.split("=", 1)[1].strip() if "=" in raw else raw.strip()

            if ":" not in rhs:
                continue

            key, value = rhs.split(":", 1)

            key_clean = key.strip().lower()
            value_clean = value.strip().strip('"').strip("'").upper()

            if key_clean != "coo":
                continue

            all_coo_lines.append(raw)

            if value_clean == "ABC":
                candidate = 1
            elif value_clean in {"GC", "GCB"}:
                # GSE87371 stores the germinal-center class as "GC".
                # For this frozen ABC-vs-GCB discriminator, GC maps to GCB.
                candidate = 0
            else:
                candidate = None

            if candidate is not None:
                if parsed is None:
                    parsed = candidate
                    matched_line = raw
                elif parsed != candidate:
                    # Conflicting COO labels: unresolved.
                    parsed = None
                    matched_line = "CONFLICTING_COO_VALUES"
                    break

        labels[gsm] = parsed

        if parsed == 1:
            status = "ABC"
        elif parsed == 0:
            status = "GCB"
        else:
            status = "UNRESOLVED"

        audit_rows.append({
            "gsm": gsm,
            "parsed_label": status,
            "matched_line": matched_line or "",
            "all_coo_lines": " || ".join(all_coo_lines),
        })

    return labels, audit_rows

def build_gene_matrix(
    sample_values,
    symbol_to_probes,
    sample_ids,
    genes,
):
    X = np.empty(
        (len(sample_ids), len(genes)),
        dtype=float,
    )

    for j, gene in enumerate(genes):
        probes = sorted(set(symbol_to_probes.get(gene, [])))

        if not probes:
            raise RuntimeError(
                f"No probes found for gene {gene}."
            )

        for i, gsm in enumerate(sample_ids):
            vals = [
                sample_values[gsm][p]
                for p in probes
                if p in sample_values[gsm]
            ]

            if not vals:
                raise RuntimeError(
                    f"Missing expression for {gsm}, {gene}."
                )

            X[i, j] = float(np.mean(vals))

    return X


def measurable_gene_matrix(
    sample_values,
    symbol_to_probes,
    sample_ids,
):
    genes = [
        g for g in symbol_to_probes.keys()
        if len(g) >= 2
    ]

    good_genes = []
    cols = []

    for gene in genes:
        probes = sorted(set(symbol_to_probes[gene]))

        col = []
        ok = True

        for gsm in sample_ids:
            vals = [
                sample_values[gsm][p]
                for p in probes
                if p in sample_values[gsm]
            ]

            if not vals:
                ok = False
                break

            col.append(float(np.mean(vals)))

        if ok:
            good_genes.append(gene)
            cols.append(col)

    X = np.asarray(cols, dtype=float).T

    return good_genes, X


# ============================================================
# CLASSIFIER / CV
# ============================================================

def cv_auc_fixed_panel(X, y, split_indices):
    aucs = []

    for train_idx, test_idx in split_indices:
        scaler = StandardScaler()

        Xtr = scaler.fit_transform(
            X[train_idx]
        )

        Xte = scaler.transform(
            X[test_idx]
        )

        clf = LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=1,
        )

        clf.fit(
            Xtr,
            y[train_idx],
        )

        score = clf.predict_proba(
            Xte
        )[:, 1]

        aucs.append(
            roc_auc_score(
                y[test_idx],
                score,
            )
        )

    return float(np.mean(aucs))


# ============================================================
# MAIN
# ============================================================

def main():
    project = project_dir()

    # Script can live either in cml/code or dlbcl/code.
    # Locate sibling DLBCL application defensively.
    phase4 = project.parents[2] if project.name.lower() in {"cml", "dlbcl"} else project

    candidate_roots = [
        project,
        project.parent / "dlbcl",
        project.parent / "DLBCL",
        phase4 / "applications" / "dlbcl",
        phase4 / "applications" / "DLBCL",
    ]

    dlbcl_root = None

    for root in candidate_roots:
        if root.exists():
            v41_hits = list(root.rglob("v41*"))
            if v41_hits:
                dlbcl_root = root
                break

    if dlbcl_root is None:
        raise RuntimeError(
            "Could not locate DLBCL application tree containing v41 artifacts."
        )

    docs = dlbcl_root / "docs"
    results = dlbcl_root / "results"

    docs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Soft Spaces / DLBCL v51e GO/NO-GO DISCRIMINATOR")
    print("DLBCL root:", dlbcl_root)
    print("=" * 72)

    # ========================================================
    # v51.0 frozen panel discovery
    # ========================================================

    # v51b: use the explicitly frozen v41.16 panel.
    # This avoids mistaking audit-status columns (e.g. "EXACT") for genes.
    frozen_genes = list(FROZEN_PANEL)
    frozen_sources = list(FROZEN_PANEL_PROVENANCE)
    ranked_files = []

    panel_n = len(frozen_genes)

    if not (4 <= panel_n <= MAX_PANEL_SIZE):
        raise RuntimeError(
            f"Frozen panel size {panel_n} is outside allowed range."
        )

    print()
    print("Frozen DLBCL panel identified:")
    print("N =", panel_n)
    print(", ".join(frozen_genes))
    print("Sources:")
    for src in frozen_sources:
        print(" ", src)

    # ========================================================
    # Freeze protocol BEFORE scoring
    # ========================================================

    protocol = f"""Soft Spaces / DLBCL
v51e.0 — HARD GO/NO-GO DISCRIMINATOR PREREGISTRATION

STATUS
------
FROZEN BEFORE EXTERNAL SCORING

PARSER-ONLY REVISION
--------------------
v51b, v51c and v51d all stopped before scientific scoring.

The v51d audit showed that GSE87371 stores the germinal-center category as:
    coo: GC
rather than:
    coo: GCB

v51e changes ONLY this metadata mapping:

    coo: ABC -> ABC
    coo: GC  -> GCB class
    coo: GCB -> GCB class

PMBL and Other remain excluded.

The frozen 12-gene panel, cohorts, CV, classifier, NULL construction,
NULL count and PASS thresholds are unchanged.

PURPOSE
-------
Determine whether DLBCL behaves materially differently from the CML result.

FROZEN DLBCL PANEL
------------------
Panel size:
    {panel_n}

Genes:
    {", ".join(frozen_genes)}

Panel source artifacts:
"""

    for src in frozen_sources:
        protocol += f"    {src}\n"

    protocol += f"""
EXTERNAL COHORTS
----------------
1. GSE87371
2. GSE159472

BIOLOGICAL LABEL
----------------
ABC vs GCB only.

No treatment-response endpoint is used.

PRIMARY TEST
------------
Within EACH external cohort:

1. Use the already-frozen DLBCL panel with no gene substitution.
2. Keep only samples with unambiguous ABC or GCB label.
3. Use RepeatedStratifiedKFold:
       {CV_SPLITS} folds x {CV_REPEATS} repeats
4. Train-fold StandardScaler only.
5. Balanced L2 logistic regression.
6. Primary metric:
       mean ROC-AUC across all frozen CV folds.

STRUCTURED REAL-GENE NULL
-------------------------
Within each cohort:

1. reconstruct all fully measurable genes
2. rank genes by raw variance WITHOUT labels
3. exclude the frozen panel genes
4. control pool = variance ranks
       {CONTROL_RANK_START}..{CONTROL_RANK_END}
   after exclusion
5. draw panels of exactly {panel_n} REAL genes
6. use same patients, same CV folds, same scaler/classifier
7. preserve real gene covariance

NULL panels per cohort:
    {NULL_N}

PER-COHORT PASS
---------------
Frozen-panel mean ROC-AUC:
    > structured NULL q95
AND
    empirical p <= 0.05

OVERALL GO/NO-GO RULE
---------------------
DLBCL DISTINCTIVE SIGNAL SUPPORTED

ONLY IF BOTH external cohorts PASS.

Otherwise:
    DLBCL DISTINCTIVE SIGNAL NOT SUPPORTED

NO RESCUE
---------
No post-hoc:
- panel substitution
- gene deletion
- feature-count change
- endpoint change
- classifier change
- CV change
- control-pool change
- threshold change

CLAIM LIMIT
-----------
A PASS would mean only that the already-frozen DLBCL panel separates ABC/GCB
more strongly than matched high-variance real-gene controls in both external
cohorts under this frozen test.

It would not establish clinical utility or causal biology.
"""

    protocol_path = docs / "v51e_0_DLBCL_GO_NO_GO_PREREGISTRATION.txt"
    protocol_path.write_text(protocol, encoding="utf-8")
    protocol_sha = sha256_file(protocol_path)

    write_json(
        docs / "v51e_0_DLBCL_GO_NO_GO_PREREGISTRATION_manifest.json",
        {
            "version": "v51e.0",
            "status": "FROZEN_BEFORE_EXTERNAL_SCORING",
            "frozen_panel_genes": frozen_genes,
            "frozen_panel_n": panel_n,
            "external_cohorts": list(COHORTS.keys()),
            "cv_splits": CV_SPLITS,
            "cv_repeats": CV_REPEATS,
            "null_n": NULL_N,
            "control_rank_start": CONTROL_RANK_START,
            "control_rank_end": CONTROL_RANK_END,
            "protocol_sha256": protocol_sha,
        },
    )

    print()
    print("Protocol SHA:", protocol_sha)
    print("Protocol frozen BEFORE scoring.")

    # ========================================================
    # Cohort tests
    # ========================================================

    cohort_results = {}

    for cohort_i, (gse, spec) in enumerate(COHORTS.items()):
        print()
        print("=" * 72)
        print("Testing", gse)
        print("=" * 72)

        soft_path = (
            dlbcl_root
            / "data"
            / "external"
            / gse
            / "metadata"
            / f"{gse}_family.soft.gz"
        )

        download_if_missing(
            spec["url"],
            soft_path,
        )

        (
            sample_values,
            symbol_to_probes,
            sample_text,
        ) = parse_soft_all(
            soft_path,
            spec["platform"],
        )

        labels, label_audit_rows = parse_abc_gcb_labels(
            sample_text
        )

        label_audit_path = (
            results
            / f"v51e_{gse}_ABC_GCB_label_parser_audit.tsv"
        )

        pd.DataFrame(
            label_audit_rows
        ).to_csv(
            label_audit_path,
            sep="\t",
            index=False,
        )

        sample_ids = [
            gsm
            for gsm, y in labels.items()
            if y is not None
        ]

        y = np.asarray(
            [labels[gsm] for gsm in sample_ids],
            dtype=int,
        )

        n_abc = int(np.sum(y == 1))
        n_gcb = int(np.sum(y == 0))

        print(
            "Evaluable ABC/GCB samples:",
            len(sample_ids),
            "| ABC:",
            n_abc,
            "| GCB:",
            n_gcb,
        )

        if len(sample_ids) < 30:
            raise RuntimeError(
                f"{gse}: TECHNICAL PARSER ABORT — too few evaluable ABC/GCB samples. "
                f"No scientific scoring performed. See {label_audit_path}"
            )

        if min(n_abc, n_gcb) < CV_SPLITS:
            raise RuntimeError(
                f"{gse}: TECHNICAL PARSER ABORT — too few samples in one class "
                f"for {CV_SPLITS}-fold CV (ABC={n_abc}, GCB={n_gcb}). "
                f"No scientific scoring performed. See {label_audit_path}"
            )

        missing_frozen = [
            g for g in frozen_genes
            if not symbol_to_probes.get(g)
        ]

        if missing_frozen:
            raise RuntimeError(
                f"{gse}: frozen panel technically non-evaluable; "
                f"missing genes: {missing_frozen}"
            )

        X_frozen = build_gene_matrix(
            sample_values,
            symbol_to_probes,
            sample_ids,
            frozen_genes,
        )

        all_genes, X_all = measurable_gene_matrix(
            sample_values,
            symbol_to_probes,
            sample_ids,
        )

        # Raw variance ranking, completely label-blind.
        variances = np.var(
            X_all,
            axis=0,
            ddof=0,
        )

        order = np.argsort(
            -variances,
            kind="mergesort",
        )

        ranked_genes = [
            all_genes[i]
            for i in order
        ]

        frozen_set = set(frozen_genes)

        ranked_controls = [
            g
            for g in ranked_genes
            if g not in frozen_set
        ]

        if len(ranked_controls) < CONTROL_RANK_END:
            raise RuntimeError(
                f"{gse}: insufficient measurable genes for control pool."
            )

        control_pool = ranked_controls[
            CONTROL_RANK_START - 1:
            CONTROL_RANK_END
        ]

        gene_to_col = {
            g: i
            for i, g in enumerate(all_genes)
        }

        cv = RepeatedStratifiedKFold(
            n_splits=CV_SPLITS,
            n_repeats=CV_REPEATS,
            random_state=CV_SEED + cohort_i,
        )

        split_indices = list(
            cv.split(
                np.zeros(
                    len(y)
                ),
                y,
            )
        )

        real_auc = cv_auc_fixed_panel(
            X_frozen,
            y,
            split_indices,
        )

        rng = np.random.default_rng(
            NULL_SEED + cohort_i
        )

        null_auc = np.empty(
            NULL_N,
            dtype=float,
        )

        for b in range(NULL_N):
            genes = rng.choice(
                control_pool,
                size=panel_n,
                replace=False,
            ).tolist()

            idx = [
                gene_to_col[g]
                for g in genes
            ]

            X_null = X_all[
                :,
                idx,
            ]

            null_auc[b] = cv_auc_fixed_panel(
                X_null,
                y,
                split_indices,
            )

            if (b + 1) % 50 == 0:
                print(
                    f"Completed {b+1}/{NULL_N} structured control panels"
                )

        q95 = float(
            np.quantile(
                null_auc,
                0.95,
            )
        )

        p = float(
            (
                1
                + np.sum(
                    null_auc >= real_auc
                )
            )
            / (
                1 + NULL_N
            )
        )

        pass_gate = bool(
            real_auc > q95
            and p <= 0.05
        )

        cohort_results[gse] = {
            "n": len(sample_ids),
            "n_abc": n_abc,
            "n_gcb": n_gcb,
            "real_auc": real_auc,
            "null_mean_auc": float(np.mean(null_auc)),
            "null_q95_auc": q95,
            "empirical_p": p,
            "pass": pass_gate,
        }

        out = results / f"v51_{gse}_structured_control_null_500.tsv"

        pd.DataFrame({
            "null_rep": np.arange(
                1,
                NULL_N + 1,
            ),
            "roc_auc": null_auc,
        }).to_csv(
            out,
            sep="\t",
            index=False,
        )

        print()
        print(
            gse,
            "| frozen AUC",
            f"{real_auc:.6f}",
            "| NULL mean",
            f"{np.mean(null_auc):.6f}",
            "| NULL q95",
            f"{q95:.6f}",
            "| p",
            f"{p:.6f}",
            "=>",
            "PASS" if pass_gate else "FAIL",
        )

    overall_pass = bool(
        all(
            cohort_results[gse]["pass"]
            for gse in COHORTS
        )
    )

    final_claim = (
        "DLBCL DISTINCTIVE SIGNAL SUPPORTED"
        if overall_pass
        else
        "DLBCL DISTINCTIVE SIGNAL NOT SUPPORTED"
    )

    closure_path = docs / "v51e_1_DLBCL_GO_NO_GO_CLOSURE.txt"

    lines = [
        "Soft Spaces / DLBCL",
        "v51e.1 — HARD GO/NO-GO DISCRIMINATOR CLOSURE",
        "",
        "STATUS",
        "------",
        "CLOSED",
        "",
        f"Protocol SHA: {protocol_sha}",
        "",
        "FROZEN PANEL",
        "------------",
        f"N = {panel_n}",
        ", ".join(frozen_genes),
        "",
        "STRUCTURED CONTROL",
        "------------------",
        "Real high-variance genes with real patient-level covariance.",
        "",
    ]

    for gse in COHORTS:
        r = cohort_results[gse]

        lines.extend([
            gse,
            "-" * len(gse),
            f"Evaluable n:            {r['n']}",
            f"ABC:                    {r['n_abc']}",
            f"GCB:                    {r['n_gcb']}",
            f"Frozen-panel ROC-AUC:   {r['real_auc']:.9f}",
            f"NULL mean ROC-AUC:      {r['null_mean_auc']:.9f}",
            f"NULL q95 ROC-AUC:       {r['null_q95_auc']:.9f}",
            f"Empirical p:            {r['empirical_p']:.9f}",
            f"STATUS:                 {'PASS' if r['pass'] else 'FAIL'}",
            "",
        ])

    lines.extend([
        "FINAL GO/NO-GO RESULT",
        "---------------------",
        final_claim,
        "",
        "INTERPRETATION",
        "--------------",
    ])

    if overall_pass:
        lines.extend([
            "The already-frozen DLBCL panel separated ABC/GCB more strongly",
            "than matched high-variance real-gene controls in BOTH external",
            "cohorts under the frozen v51 test.",
            "",
            "This indicates that DLBCL behaves materially differently from",
            "the negative CML structured-control result.",
        ])
    else:
        lines.extend([
            "The already-frozen DLBCL panel did not beat structured",
            "high-variance real-gene controls in BOTH external cohorts.",
            "",
            "Therefore this test does not establish that DLBCL behaves",
            "materially differently from the negative CML result.",
        ])

    lines.extend([
        "",
        "CLAIM LIMIT",
        "-----------",
        "This is a statistical discriminator test only.",
        "It does not establish clinical utility or causal biology.",
        "",
        "NO RESCUE / NO TUNING",
        "---------------------",
        "No post-hoc panel, classifier, endpoint, control-pool or threshold",
        "change was used.",
    ])

    closure_text = "\n".join(lines) + "\n"

    closure_path.write_text(
        closure_text,
        encoding="utf-8",
    )

    closure_sha = sha256_file(
        closure_path
    )

    manifest_path = docs / "v51e_1_DLBCL_GO_NO_GO_CLOSURE_manifest.json"

    write_json(
        manifest_path,
        {
            "version": "v51e.1",
            "status": "CLOSED",
            "frozen_panel_genes": frozen_genes,
            "frozen_panel_n": panel_n,
            "cohort_results": cohort_results,
            "overall_pass": overall_pass,
            "final_claim": final_claim,
            "protocol_sha256": protocol_sha,
            "closure_sha256": closure_sha,
            "execution_script_sha256": sha256_file(
                Path(__file__).resolve()
            ),
        },
    )

    summary_path = results / "v51e_DLBCL_go_no_go_summary.txt"

    summary = f"""=== Soft Spaces / DLBCL v51e GO/NO-GO SUMMARY ===

Frozen panel size:
    {panel_n}

GSE87371:
    {"PASS" if cohort_results["GSE87371"]["pass"] else "FAIL"}

GSE159472:
    {"PASS" if cohort_results["GSE159472"]["pass"] else "FAIL"}

FINAL:
    {final_claim}

Protocol SHA:
    {protocol_sha}

Closure SHA:
    {closure_sha}
"""

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("v51e DLBCL GO/NO-GO COMPLETE")
    print("=" * 72)
    print(summary)
    print("Closure:")
    print(" ", closure_path)
    print("Summary:")
    print(" ", summary_path)


if __name__ == "__main__":
    main()
