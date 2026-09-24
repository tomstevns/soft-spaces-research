#!/usr/bin/env python3
"""
Soft Spaces / CML
v48.0-v48.1 — Geometric invariance despite changing feature identity

NEW hypothesis. NOT a rescue of v47.

Question:
Can GSE130404 and GSE44589 preserve a shared higher-level label-blind
geometry even though their frozen Top-256 gene identities differ?

No clinical labels are used anywhere in v48.1.

Primary metrics:
A) cosine similarity of normalized PCA-16 eigen-spectra
B) Wasserstein-1 distance between normalized PCA-16 pairwise-distance
   distributions

NULL:
For GSE44589, independently permute patient values within each frozen gene,
preserving every gene's marginal values and variance while destroying
multivariate covariance. N=1000.

PASS requires BOTH:
A observed > NULL 95th percentile
B observed < NULL 5th percentile
"""

from pathlib import Path
import gzip, hashlib, json, re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

N_NULL = 1000
SEED = 48010001
TOP_K = 256
PCA_R = 16
KNN_K = 10


def project_dir():
    return Path(__file__).resolve().parent.parent


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cosine(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def w1(x, y):
    x = np.sort(np.asarray(x, float))
    y = np.sort(np.asarray(y, float))
    n = max(len(x), len(y))
    q = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(
        np.quantile(x, q, method="linear") -
        np.quantile(y, q, method="linear")
    )))


def pairwise_upper(Y):
    D = np.sqrt(np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2))
    iu = np.triu_indices(len(Y), 1)
    return D[iu]


def local_anisotropy(Y, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Y)
    _, idx = nn.kneighbors(Y)
    out = []
    for i in range(len(Y)):
        C = np.cov(Y[idx[i, 1:], :], rowvar=False, ddof=1)
        eig = np.maximum(np.linalg.eigvalsh(C), 0)[::-1]
        out.append(float(eig[0] / (eig.sum() + 1e-12)))
    return np.asarray(out)


def signature(X):
    X = np.asarray(X, float)
    if X.shape[1] != TOP_K:
        raise RuntimeError(f"Expected {TOP_K} features, got {X.shape[1]}")
    sd = X.std(axis=0, ddof=0)
    if np.any(sd == 0):
        raise RuntimeError("Zero-SD feature.")
    Z = (X - X.mean(axis=0)) / sd
    pca = PCA(n_components=PCA_R, svd_solver="full")
    Y = pca.fit_transform(Z)

    spec = pca.explained_variance_.astype(float)
    spec /= spec.sum()

    pdist = pairwise_upper(Y)
    pdist /= np.median(pdist)

    return {
        "spectrum": spec,
        "pdist": pdist,
        "anisotropy": local_anisotropy(Y, KNN_K),
        "cumvar": float(pca.explained_variance_ratio_.sum()),
        "effdim": float(1.0 / np.sum(spec * spec)),
    }


def split_symbols(text):
    text = str(text).strip()
    if not text or text in {"---", "NA", "nan"}:
        return []
    parts = re.split(r"\s*///\s*|\s*//\s*|\s*;\s*|\s*,\s*", text)
    return [re.sub(r"\s+", "", p.strip()) for p in parts
            if p.strip() and p.strip() not in {"---", "NA"}]


def reconstruct_external_matrix(soft_path, sample_ids, gene_order):
    sample_ids = list(map(str, sample_ids))
    sample_set = set(sample_ids)
    gene_set = set(gene_order)

    probe_to_symbols = {}
    sample_values = {gsm: {} for gsm in sample_ids}

    current_entity = None
    current_id = None
    in_platform = False
    in_sample = False
    platform_header = None
    sample_header = None

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

            if current_entity == "PLATFORM" and current_id == "GPL570":
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
                            raise RuntimeError("No GPL570 symbol column.")
                        continue

                    if len(fields) <= max(probe_idx, symbol_idx):
                        continue
                    probe = fields[probe_idx].strip()
                    syms = [s for s in split_symbols(fields[symbol_idx]) if s in gene_set]
                    if probe and syms:
                        probe_to_symbols[probe] = syms
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
    for probe, syms in probe_to_symbols.items():
        for s in syms:
            symbol_to_probes[s].append(probe)

    X = np.empty((len(sample_ids), len(gene_order)), float)
    for j, gene in enumerate(gene_order):
        probes = sorted(set(symbol_to_probes.get(gene, [])))
        if not probes:
            raise RuntimeError(f"No probes for frozen external gene {gene}")
        for i, gsm in enumerate(sample_ids):
            vals = [sample_values[gsm][p] for p in probes if p in sample_values[gsm]]
            if not vals:
                raise RuntimeError(f"Missing expression for {gsm}, {gene}")
            X[i, j] = float(np.mean(vals))
    return X


def main():
    project = project_dir()
    docs = project / "docs"
    ds = project / "results" / "direct_subspace"

    closure_manifest = docs / "v47_6_CML_STATE_LANDSCAPE_CLOSURE_manifest.json"
    dev_raw_path = ds / "v47_2_GSE130404_label_blind_top256_expression.tsv"
    dev_features_path = ds / "v47_2_label_blind_top256_features.tsv"
    ext_geometry_path = ds / "v47_5b_GSE44589_label_blind_geometry.tsv"
    ext_features_path = ds / "v47_5b_GSE44589_label_blind_top256_features.tsv"
    ext_soft_path = project / "data" / "external" / "GSE44589" / "metadata" / "GSE44589_family.soft.gz"

    for p in [closure_manifest, dev_raw_path, dev_features_path,
              ext_geometry_path, ext_features_path, ext_soft_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    closure = json.loads(closure_manifest.read_text(encoding="utf-8"))
    if closure.get("status") != "CLOSED":
        raise RuntimeError("v47 must be formally closed before v48.")

    protocol = f"""Soft Spaces / CML
v48.0 — GEOMETRIC INVARIANCE PREREGISTRATION

STATUS
------
FROZEN BEFORE v48.1 EXECUTION

NEW HYPOTHESIS
--------------
v48 is scientifically distinct from v47 and is NOT a rescue of v47.5b.

Hypothesis:
Two independent CML cohorts may preserve shared higher-level label-blind
geometry despite differing exact Top-256 gene identities.

NO CLINICAL LABELS are used.

Cohorts:
- GSE130404 frozen v47.2 label-blind Top-256 expression matrix
- GSE44589 frozen v47.5b label-blind Top-256 feature list and baseline samples

Primary A:
cosine similarity of normalized PCA-16 eigen-spectra.

Primary B:
Wasserstein-1 distance between PCA-16 pairwise-distance distributions,
after dividing distances by each cohort's own median.

NULL:
{N_NULL} replicates.
In each replicate independently permute patient values within every frozen
GSE44589 gene. This preserves every external gene's marginal values,
mean, variance, sample count and feature identity, while destroying
patient-level multivariate covariance.

PASS requires BOTH:
- observed spectrum cosine > NULL 95th percentile
- observed distance W1 < NULL 5th percentile

Empirical p-values for both must be <= 0.05.

Secondary:
Wasserstein distance between local-anisotropy distributions, k={KNN_K}.

Claim limit:
A PASS supports higher-level feature-identity-invariant geometric similarity
under this specific label-blind test. It does not establish prediction,
response replication, mechanism, dormancy, quantum biology or quantum advantage.

No rescue tuning after FAIL.
"""

    docs.mkdir(parents=True, exist_ok=True)
    protocol_path = docs / "v48_0_CML_GEOMETRIC_INVARIANCE_PREREGISTRATION.txt"
    protocol_path.write_text(protocol, encoding="utf-8")
    protocol_sha = sha256_file(protocol_path)

    protocol_manifest = docs / "v48_0_CML_GEOMETRIC_INVARIANCE_PREREGISTRATION_manifest.json"
    protocol_manifest.write_text(json.dumps({
        "version": "v48.0",
        "status": "FROZEN_BEFORE_V48_1_EXECUTION",
        "clinical_labels_used": False,
        "n_null": N_NULL,
        "seed": SEED,
        "pca_r": PCA_R,
        "knn_k": KNN_K,
        "protocol_sha256": protocol_sha,
        "source_sha256": {
            "v47_6_closure_manifest": sha256_file(closure_manifest),
            "dev_raw_top256": sha256_file(dev_raw_path),
            "dev_features": sha256_file(dev_features_path),
            "ext_geometry": sha256_file(ext_geometry_path),
            "ext_features": sha256_file(ext_features_path),
            "gse44589_soft": sha256_file(ext_soft_path),
        }
    }, indent=2), encoding="utf-8")

    print("=== v48.0 PREREGISTRATION FROZEN ===")
    print("Protocol SHA:", protocol_sha)
    print("Clinical labels used: NO")
    print()

    dev_df = pd.read_csv(dev_raw_path, sep="\t", index_col=0)
    dev_features = pd.read_csv(dev_features_path, sep="\t")["gene"].astype(str).tolist()
    ext_geometry = pd.read_csv(ext_geometry_path, sep="\t", index_col=0)
    ext_features = pd.read_csv(ext_features_path, sep="\t")["gene"].astype(str).tolist()

    if dev_df.shape != (96, 256):
        raise RuntimeError(f"Unexpected dev matrix shape {dev_df.shape}")
    if len(dev_features) != 256 or len(ext_features) != 256:
        raise RuntimeError("Frozen feature count mismatch.")
    if list(dev_df.columns.astype(str)) != dev_features:
        raise RuntimeError("Development feature order mismatch.")
    if len(ext_geometry) != 135:
        raise RuntimeError("Expected 135 frozen external samples.")

    ext_samples = [str(x) for x in ext_geometry.index]
    Xext = reconstruct_external_matrix(ext_soft_path, ext_samples, ext_features)

    dev_sig = signature(dev_df.to_numpy(float))
    ext_sig = signature(Xext)

    obs_cos = cosine(dev_sig["spectrum"], ext_sig["spectrum"])
    obs_w1 = w1(dev_sig["pdist"], ext_sig["pdist"])
    obs_aniso = w1(dev_sig["anisotropy"], ext_sig["anisotropy"])

    overlap = len(set(dev_features) & set(ext_features))
    jaccard = overlap / (512 - overlap)

    rng = np.random.default_rng(SEED)
    null_cos = np.empty(N_NULL)
    null_w1 = np.empty(N_NULL)
    null_aniso = np.empty(N_NULL)

    Xnull = np.empty_like(Xext)

    for b in range(N_NULL):
        for j in range(TOP_K):
            Xnull[:, j] = rng.permutation(Xext[:, j])

        ns = signature(Xnull)
        null_cos[b] = cosine(dev_sig["spectrum"], ns["spectrum"])
        null_w1[b] = w1(dev_sig["pdist"], ns["pdist"])
        null_aniso[b] = w1(dev_sig["anisotropy"], ns["anisotropy"])

        if (b + 1) % 100 == 0:
            print(f"Completed {b+1}/{N_NULL} NULL geometries")

    q95_cos = float(np.quantile(null_cos, 0.95))
    q05_w1 = float(np.quantile(null_w1, 0.05))

    p_cos = float((1 + np.sum(null_cos >= obs_cos)) / (1 + N_NULL))
    p_w1 = float((1 + np.sum(null_w1 <= obs_w1)) / (1 + N_NULL))

    gate_cos = bool(obs_cos > q95_cos)
    gate_w1 = bool(obs_w1 < q05_w1)
    primary_pass = bool(gate_cos and gate_w1 and p_cos <= 0.05 and p_w1 <= 0.05)

    status = "GEOMETRIC INVARIANCE PASS" if primary_pass else "GEOMETRIC INVARIANCE FAIL"

    null_out = ds / "v48_1_geometric_invariance_null_1000.tsv"
    pd.DataFrame({
        "null_rep": np.arange(1, N_NULL + 1),
        "spectrum_cosine": null_cos,
        "distance_wasserstein": null_w1,
        "anisotropy_wasserstein": null_aniso,
    }).to_csv(null_out, sep="\t", index=False)

    summary_out = ds / "v48_1_geometric_invariance_summary.txt"
    summary = f"""=== Soft Spaces / CML v48.1 GEOMETRIC INVARIANCE ===

v48.0 protocol SHA:              {protocol_sha}

RELATION TO v47
---------------
v47 closed:                      YES
v48 new hypothesis:              YES
v48 rescue of v47.5b:            NO

LABEL USE
---------
Clinical labels used:            NO
Classifier fitted:               NO

FEATURE IDENTITY
----------------
Top256 overlap:                  {overlap}/256
Jaccard:                         {jaccard:.6f}

LABEL-BLIND GEOMETRY
--------------------
GSE130404 PCA16 cumulative var:  {dev_sig["cumvar"]:.6f}
GSE44589  PCA16 cumulative var:  {ext_sig["cumvar"]:.6f}
GSE130404 effective dimension:   {dev_sig["effdim"]:.6f}
GSE44589  effective dimension:   {ext_sig["effdim"]:.6f}

PRIMARY A — SPECTRAL SHAPE
--------------------------
Observed cosine:                 {obs_cos:.9f}
NULL mean:                       {np.mean(null_cos):.9f}
NULL q95:                        {q95_cos:.9f}
Empirical p:                     {p_cos:.9f}
Gate observed > q95:             {"YES" if gate_cos else "NO"}

PRIMARY B — DISTANCE GEOMETRY
-----------------------------
Observed normalized W1:          {obs_w1:.9f}
NULL mean:                       {np.mean(null_w1):.9f}
NULL q05:                        {q05_w1:.9f}
Empirical p:                     {p_w1:.9f}
Gate observed < q05:             {"YES" if gate_w1 else "NO"}

SECONDARY — LOCAL ANISOTROPY
----------------------------
Observed anisotropy W1:          {obs_aniso:.9f}
NULL mean:                       {np.mean(null_aniso):.9f}
NULL median:                     {np.median(null_aniso):.9f}

v48.1 STATUS: {status}

INTERPRETATION LIMIT
--------------------
This test asks whether two independently constructed label-blind CML
expression geometries share higher-level shape beyond what is expected when
the external gene-gene covariance structure is destroyed while preserving
each external frozen gene's marginal values.

A PASS supports geometric similarity under this preregistered test only.
A FAIL is retained without rescue tuning.

Neither outcome establishes prediction, treatment-response replication,
causal biology, dormancy mechanism, quantum biology or quantum advantage.
"""
    summary_out.write_text(summary, encoding="utf-8")

    manifest_out = ds / "v48_1_manifest.json"
    manifest_out.write_text(json.dumps({
        "version": "v48.1",
        "status": status,
        "clinical_labels_used": False,
        "classifier_fitted": False,
        "v47_rescue": False,
        "top256_overlap": overlap,
        "top256_jaccard": jaccard,
        "observed": {
            "spectrum_cosine": obs_cos,
            "distance_wasserstein": obs_w1,
            "anisotropy_wasserstein": obs_aniso,
            "dev_pca16_cumulative_variance": dev_sig["cumvar"],
            "ext_pca16_cumulative_variance": ext_sig["cumvar"],
            "dev_effective_dimension": dev_sig["effdim"],
            "ext_effective_dimension": ext_sig["effdim"],
        },
        "null": {
            "n": N_NULL,
            "seed": SEED,
            "spectrum_cosine_q95": q95_cos,
            "distance_wasserstein_q05": q05_w1,
        },
        "primary": {
            "p_spectrum": p_cos,
            "p_distance": p_w1,
            "spectrum_gate": gate_cos,
            "distance_gate": gate_w1,
            "primary_pass": primary_pass,
        },
        "protocol_sha256": protocol_sha,
        "sha256": {
            "null_results": sha256_file(null_out),
            "execution_script": sha256_file(Path(__file__).resolve()),
        },
    }, indent=2), encoding="utf-8")

    print()
    print(summary)
    print("Wrote:")
    print(" ", protocol_path)
    print(" ", protocol_manifest)
    print(" ", null_out)
    print(" ", summary_out)
    print(" ", manifest_out)


if __name__ == "__main__":
    main()
