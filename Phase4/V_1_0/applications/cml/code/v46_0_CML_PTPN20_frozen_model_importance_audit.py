#!/usr/bin/env python3
"""
Soft Spaces / CML
v46.0 — PTPN20 frozen-model importance audit

Purpose
-------
Quantify how important PTPN20 actually is inside the already-frozen
v44.5a Top-256 / U16 / logistic model.

This is NOT a new external predictive test.
It does NOT reopen v45.
It does NOT use GSE14671 outcomes.
It does NOT refit or alter the model.

Exact model-level quantities
----------------------------
For frozen U in R^(256 x 16):

1. Subspace leverage of gene i:
       leverage_i = sum_k U[i,k]^2

   This is the diagonal element P_ii of the projector P = U U^T.

2. Fraction of total projector mass:
       leverage_i / r

   Since trace(P) = r = 16.

3. Effective classifier coefficient in original standardized
   256-gene coordinates:

       beta_gene = U @ beta_subspace

   Thus a +1 SD change in standardized gene i changes the classifier
   logit by exactly beta_gene[i], holding all other coordinates fixed.

4. Relative classifier-weight share:
       beta_gene[i]^2 / sum_j beta_gene[j]^2

Outputs include ranks and percentiles among all 256 frozen genes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v46.0"
TARGET = "PTPN20"
TOP_K = 256
R = 16


def project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rank_desc(values: np.ndarray) -> np.ndarray:
    """
    Rank 1 = largest value.
    Stable deterministic ranking.
    """
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def percentile_from_rank(rank: int, n: int) -> float:
    """
    Percentage of genes at or below this importance value.
    Top-ranked gene -> 100%.
    Bottom-ranked gene -> 100/n%.
    """
    return 100.0 * (n - rank + 1) / n


def main():
    project = project_dir()
    ds = project / "results" / "direct_subspace"
    docs = project / "docs"

    model_path = ds / "v44_5a_frozen_development_model.npz"
    v45_closure = docs / "v45_3_CML_GSE14671_CLOSURE_AUDIT.txt"

    for p in [model_path, v45_closure]:
        if not p.exists():
            raise FileNotFoundError(p)

    model = np.load(model_path, allow_pickle=False)

    required_arrays = [
        "genes",
        "U16",
        "logistic_coef",
        "logistic_intercept",
    ]

    for key in required_arrays:
        if key not in model.files:
            raise RuntimeError(
                f"Frozen model missing '{key}'. Available arrays: {model.files}"
            )

    genes = np.asarray([str(x) for x in model["genes"].tolist()])
    U = np.asarray(model["U16"], dtype=float)
    coef = np.asarray(model["logistic_coef"], dtype=float)

    if len(genes) != TOP_K:
        raise RuntimeError(f"Expected {TOP_K} genes, found {len(genes)}.")

    if U.shape != (TOP_K, R):
        raise RuntimeError(f"Expected U16 shape {(TOP_K, R)}, found {U.shape}.")

    beta_sub = coef.reshape(-1)

    if beta_sub.shape[0] != R:
        raise RuntimeError(
            f"Expected {R} subspace classifier coefficients, found {beta_sub.shape[0]}."
        )

    matches = np.where(genes == TARGET)[0]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {TARGET} coordinate, found {len(matches)}."
        )

    idx = int(matches[0])

    # Exact projector leverage.
    leverage = np.sum(U ** 2, axis=1)
    projector_mass_fraction = leverage / float(R)

    # Exact frozen classifier coefficient in original standardized coordinates.
    beta_gene = U @ beta_sub
    abs_beta_gene = np.abs(beta_gene)
    beta_sq = beta_gene ** 2

    beta_sq_sum = float(np.sum(beta_sq))
    if beta_sq_sum <= 0:
        raise RuntimeError("Frozen gene-space classifier has zero total weight.")

    classifier_weight_share = beta_sq / beta_sq_sum

    leverage_rank = rank_desc(leverage)
    abs_beta_rank = rank_desc(abs_beta_gene)
    weight_share_rank = rank_desc(classifier_weight_share)

    # Useful global summaries.
    target_leverage = float(leverage[idx])
    target_mass_fraction = float(projector_mass_fraction[idx])
    target_beta = float(beta_gene[idx])
    target_abs_beta = float(abs_beta_gene[idx])
    target_weight_share = float(classifier_weight_share[idx])

    leverage_median = float(np.median(leverage))
    leverage_mean = float(np.mean(leverage))
    abs_beta_median = float(np.median(abs_beta_gene))
    abs_beta_mean = float(np.mean(abs_beta_gene))

    # Mathematical consistency checks.
    trace_projector = float(np.sum(leverage))
    orthonormality_error = float(
        np.linalg.norm(U.T @ U - np.eye(R), ord="fro")
    )

    # If coordinate is set to its standardized mean z=0, the logit change
    # relative to an arbitrary original z value is -z_i * beta_i.
    # Here we report the exact per-1-SD sensitivity.
    logit_delta_plus_1sd = target_beta
    logit_delta_minus_1sd = -target_beta

    # Odds multiplier for +1 SD, purely algebraic from the frozen logistic model.
    odds_multiplier_plus_1sd = float(np.exp(np.clip(target_beta, -700, 700)))

    # Full ranking table.
    df = pd.DataFrame({
        "gene": genes,
        "subspace_leverage": leverage,
        "projector_mass_fraction": projector_mass_fraction,
        "effective_gene_beta": beta_gene,
        "abs_effective_gene_beta": abs_beta_gene,
        "classifier_weight_share": classifier_weight_share,
        "leverage_rank_desc": leverage_rank,
        "abs_beta_rank_desc": abs_beta_rank,
        "weight_share_rank_desc": weight_share_rank,
    })

    ranking_out = ds / "v46_0_frozen_gene_importance_ranking.tsv"
    df.sort_values(
        ["subspace_leverage", "abs_effective_gene_beta"],
        ascending=[False, False],
    ).to_csv(ranking_out, sep="\t", index=False)

    summary_out = ds / "v46_0_PTPN20_frozen_model_importance_summary.txt"

    n = len(genes)

    lines = [
        "=== Soft Spaces / CML v46.0 PTPN20 FROZEN-MODEL IMPORTANCE AUDIT ===",
        "",
        "SCOPE",
        "-----",
        "Frozen model: v44.5a",
        "External outcome data used: NO",
        "GSE14671 scores used: NO",
        "Model refit: NO",
        "Model modification: NO",
        "v45 reopened: NO",
        "",
        "MODEL CHECKS",
        "------------",
        f"Top-K:                           {TOP_K}",
        f"Subspace rank:                   {R}",
        f"sum(diag(U U^T)):                {trace_projector:.12f}",
        f"expected projector trace:        {R:.12f}",
        f"||U^T U - I||_F:                 {orthonormality_error:.12e}",
        "",
        "PTPN20 SUBSPACE IMPORTANCE",
        "---------------------------",
        f"subspace leverage P_ii:          {target_leverage:.12g}",
        f"mean leverage across 256:        {leverage_mean:.12g}",
        f"median leverage across 256:      {leverage_median:.12g}",
        f"projector-mass fraction:         {target_mass_fraction:.12g}",
        f"projector-mass percent:          {100*target_mass_fraction:.9f}%",
        f"leverage rank / 256:             {int(leverage_rank[idx])}",
        f"leverage percentile:             {percentile_from_rank(int(leverage_rank[idx]), n):.3f}%",
        "",
        "PTPN20 CLASSIFIER IMPORTANCE",
        "-----------------------------",
        f"effective gene beta:             {target_beta:+.12g}",
        f"absolute effective beta:         {target_abs_beta:.12g}",
        f"mean |beta_gene| across 256:     {abs_beta_mean:.12g}",
        f"median |beta_gene| across 256:   {abs_beta_median:.12g}",
        f"|beta| rank / 256:               {int(abs_beta_rank[idx])}",
        f"|beta| percentile:               {percentile_from_rank(int(abs_beta_rank[idx]), n):.3f}%",
        f"classifier weight share:         {target_weight_share:.12g}",
        f"classifier weight percent:       {100*target_weight_share:.9f}%",
        f"weight-share rank / 256:         {int(weight_share_rank[idx])}",
        "",
        "EXACT LOCAL SENSITIVITY",
        "-----------------------",
        "Holding all other standardized genes fixed:",
        f"logit change for PTPN20 +1 SD:   {logit_delta_plus_1sd:+.12g}",
        f"logit change for PTPN20 -1 SD:   {logit_delta_minus_1sd:+.12g}",
        f"odds multiplier for +1 SD:       {odds_multiplier_plus_1sd:.12g}",
        "",
        "INTERPRETATION",
        "--------------",
        "The leverage measures how strongly PTPN20 participates in the frozen",
        "16-dimensional subspace.",
        "",
        "The effective beta measures its exact first-order contribution to the",
        "frozen logistic logit per one standardized-expression unit.",
        "",
        "These are model-structural quantities. They do NOT establish that",
        "PTPN20 is a biological, diagnostic, prognostic, or causal CML marker.",
        "",
        "v45 remains closed and TECHNICALLY NON-EVALUABLE regardless of the",
        "importance found here. This audit is intended only to inform the",
        "prospective technical design of a future v46 external test.",
    ]

    summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_out = ds / "v46_0_manifest.json"

    manifest = {
        "version": VERSION,
        "status": "MODEL_IMPORTANCE_AUDIT_COMPLETE",
        "target_gene": TARGET,
        "external_outcomes_used": False,
        "model_refit": False,
        "model_modified": False,
        "v45_reopened": False,
        "top_k": TOP_K,
        "rank": R,
        "ptpn20": {
            "subspace_leverage": target_leverage,
            "projector_mass_fraction": target_mass_fraction,
            "projector_mass_percent": 100 * target_mass_fraction,
            "leverage_rank_desc": int(leverage_rank[idx]),
            "leverage_percentile": percentile_from_rank(
                int(leverage_rank[idx]), n
            ),
            "effective_gene_beta": target_beta,
            "abs_effective_gene_beta": target_abs_beta,
            "abs_beta_rank_desc": int(abs_beta_rank[idx]),
            "abs_beta_percentile": percentile_from_rank(
                int(abs_beta_rank[idx]), n
            ),
            "classifier_weight_share": target_weight_share,
            "classifier_weight_percent": 100 * target_weight_share,
            "classifier_weight_rank_desc": int(weight_share_rank[idx]),
            "logit_delta_plus_1sd": logit_delta_plus_1sd,
            "odds_multiplier_plus_1sd": odds_multiplier_plus_1sd,
        },
        "model_checks": {
            "projector_trace": trace_projector,
            "orthonormality_fro_error": orthonormality_error,
        },
        "sha256": {
            "frozen_model": sha256_file(model_path),
            "v45_closure": sha256_file(v45_closure),
            "ranking_tsv": sha256_file(ranking_out),
            "execution_script": sha256_file(Path(__file__).resolve()),
        },
    }

    manifest_out.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(summary_out.read_text(encoding="utf-8"))

    print("Wrote:")
    print(" ", ranking_out)
    print(" ", summary_out)
    print(" ", manifest_out)


if __name__ == "__main__":
    main()
