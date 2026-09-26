#!/usr/bin/env python3
r"""
Create the Phase 7 directory skeleton and install the frozen preregistration.

Default repository:
    C:\Users\tomst\PycharmProjects\SoftSpaces_Phase2

Usage:
    python install_phase7_preregistration.py
"""

from pathlib import Path
import sys

DEFAULT_REPO = Path(r"C:\Users\tomst\PycharmProjects\SoftSpaces_Phase2")

PREREG = "# Phase 7 Preregistration\n## Orthogonal Information Test: Soft-Space Prominence vs EN2\n\n**Version:** v70_0  \n**Status:** FROZEN BEFORE EXECUTION  \n**Purpose:** One final, falsifiable reopening test after formal project closure.\n\n## 1. Scientific question\n\nPhase 7 does **not** retest whether Soft Spaces can outperform EN2 on ground-state determinant selection. Phase 5 already tested that question and EN2 dominated the matched-budget ground-state selection task.\n\nPhase 7 asks instead:\n\n> **Does Soft-Space prominence contain predictive information about perturbation-induced eigenstate/subspace reorganisation that is not already captured by EN2?**\n\nThe target is therefore **orthogonal information**, not superiority on EN2's native energy-ranking task.\n\n## 2. Primary hypothesis\n\nFor candidate local eigenspaces matched to have similar EN2 importance, higher Soft-Space prominence predicts stronger perturbation-induced reorganisation.\n\nOperationally:\n\n> Among EN2-matched candidate pairs, Soft-Space prominence should predict larger subspace rotation / fidelity loss / eigenvector mixing under held-out perturbations better than EN2 itself.\n\n## 3. Null hypothesis\n\nAfter controlling for EN2, Soft-Space prominence contains no additional predictive information about perturbation-induced eigenspace reorganisation.\n\nEquivalent outcomes include:\n\n- chance-level performance after EN2 matching;\n- no incremental prediction beyond EN2;\n- a system-specific signal that fails independent replication.\n\n## 4. Frozen Soft Spaces mechanism\n\nPhase 7 must use the **frozen Phase 2/3 Soft Spaces mechanism**, not the v50_2–v50_6 molecular REAL proxy.\n\nCore quantity:\n\n`C_matrix = ||sum_m A_m||_F / sum_m ||A_m||_F`\n\nFor perturbation family `f`:\n\n`DeltaC_f(k) = C_matrix_REAL,f(k) - C_matrix_NULL,f(k)`\n\nRobust cross-family score:\n\n`S(k) = min(median DeltaC_dephasing(k), median DeltaC_transverse(k))`\n\nLocal prominence:\n\n`P(k) = S(k) - median_nearby S(k')`\n\nThe exact implementation must be inherited from the frozen Phase 2/3 source semantics and must not be altered after inspection of Phase 7 outcomes.\n\n## 5. EN2 control variable\n\nEN2 is treated as the energetic-importance control variable.\n\nWorking project-consistent EN2-like score:\n\n`EN2_i = |H_ir|^2 / (|H_ii - H_rr| + eps)`\n\nPhase 7 does not claim that this scalar is the complete selected-CI EN2 algorithm. It is used as the energetic ranking control.\n\n## 6. Systems\n\nPhase 7 must contain at least **three independent Hamiltonian families**:\n\n1. **H4/STO-3G geometry family**\n   - reuse the established geometry scan;\n   - fixed `(N_alpha, N_beta) = (2,2)` sector.\n\n2. **LiH/STO-3G geometry family**\n   - reuse the established LiH scan;\n   - same particle-number/spin-sector conventions as Phase 5.\n\n3. **Independent synthetic avoided-crossing family**\n   - parameterized Hamiltonians with genuine local near-degeneracies / avoided crossings;\n   - construction frozen before evaluating Phase 7 target metrics;\n   - not tuned to maximize Soft-Space performance.\n\nThe third family prevents the conclusion from resting only on H4/LiH chemistry.\n\n## 7. Candidate definition\n\nCandidate local eigenspaces are adjacent eigenstate pairs `(k, k+1)` satisfying the same eligibility rules used by the frozen Phase 2/3 mechanism.\n\nEligibility must be computed before target labels are examined.\n\nIneligible pairs must not be silently treated as negatives.\n\n## 8. EN2 matching\n\nThe central anti-confounding step is to compare candidates at approximately equal EN2 importance.\n\nPrimary method:\n\n- rank candidates by EN2;\n- divide into EN2 quantile bins within each Hamiltonian instance;\n- compare Soft-Space prominence only within bins.\n\nSecondary robustness method:\n\n- nearest-neighbour matching in `log10(EN2 + eps)`;\n- maximum standardized EN2-distance fixed before execution.\n\nNo matching tolerance may be changed after seeing outcomes.\n\n## 9. Held-out perturbations\n\nSoft-Space prominence is computed from the frozen discovery perturbation families.\n\nTargets must be measured using **held-out perturbations** not used to construct the Soft-Space score.\n\nMinimum held-out set:\n\n- one dephasing-like family;\n- one transverse-like family;\n- one mixed family;\n- one parameter-displacement perturbation relevant to the Hamiltonian family.\n\nHeld-out seeds must be disjoint from discovery seeds.\n\n## 10. Primary target: subspace reorganisation\n\nFor each candidate pair, let the unperturbed two-dimensional subspace be `U` and the perturbed subspace be `U'`.\n\nPrimary endpoint:\n\n`R_subspace = sin(theta_max(U, U'))`\n\nwhere `theta_max` is the largest principal angle between the two subspaces.\n\nInterpretation:\n\n- `0` = no subspace rotation;\n- values approaching `1` = strong reorganisation.\n\nThis is the primary endpoint.\n\n## 11. Secondary targets\n\nSecondary endpoints are reported but cannot rescue a failed primary endpoint:\n\n### T2 — subspace fidelity loss\nProjector-overlap / fidelity-style loss between `U` and `U'`.\n\n### T3 — eigenvector mixing\nChange in participation or overlap structure within a local spectral window.\n\n### T4 — local spectral response\nMagnitude of perturbation-induced local gap change or avoided-crossing displacement.\n\nNo new target may be introduced after results are inspected and then treated as confirmatory.\n\n## 12. Primary statistical comparison\n\nWithin EN2-matched strata, test whether Soft-Space prominence predicts `R_subspace`.\n\nPrimary comparison:\n\n- Model A: EN2 only\n- Model B: EN2 + Soft-Space prominence\n\nThe question is whether Model B yields reproducible out-of-sample improvement.\n\n## 13. Cross-validation\n\nCross-validation must separate Hamiltonian instances rather than randomly splitting candidate pairs from the same Hamiltonian.\n\nPreferred scheme:\n\n- leave-one-geometry-out within H4;\n- leave-one-geometry-out within LiH;\n- leave-one-parameter-instance-out within the synthetic family.\n\nA pair from a held-out Hamiltonian instance must not influence training.\n\n## 14. Frozen success criteria\n\nPhase 7 is a **CLEAR PASS** only if all of the following hold:\n\n1. **Primary pooled association**\n   - out-of-sample matched/partial association between Soft-Space prominence and `R_subspace` after EN2 control:\n   - `rho >= 0.25`.\n\n2. **Incremental predictive value**\n   - Model B improves out-of-sample prediction over Model A by:\n   - `Delta R^2 >= 0.05`,\n   - or an explicitly preregistered rank-prediction equivalent if a nonparametric model is used.\n\n3. **Replication**\n   - positive incremental effect in all three Hamiltonian families.\n\n4. **Per-family minimum**\n   - at least two of the three families individually show:\n   - matched/partial `rho >= 0.20`.\n\n5. **Direction consistency**\n   - at least 70% of held-out Hamiltonian instances show the preregistered positive direction.\n\n6. **No EN2-imbalance explanation**\n   - the result remains positive under the secondary nearest-neighbour EN2 matching.\n\nIf criteria 1–3 fail, Phase 7 is a **FAIL**.\n\nCriteria 4–6 are also required for a CLEAR PASS.\n\n## 15. Interpretation rules\n\n### CLEAR PASS permits\n\nA CLEAR PASS permits the claim:\n\n> Soft-Space prominence contains reproducible information about perturbation-induced eigenspace reorganisation that is not captured by EN2 energetic importance alone.\n\nIt does **not** permit claims of:\n\n- general computational superiority to EN2;\n- quantum speedup;\n- lower selected-CI cost;\n- commercial utility;\n- universal Hilbert-space behavior.\n\n### FAIL permits\n\nA FAIL supports:\n\n> After controlling for EN2, the tested Soft-Space prominence did not show robust transferable predictive information for eigenspace reorganisation across the preregistered Hamiltonian families.\n\nNo threshold, target, matching tolerance, system, or perturbation family may then be changed post hoc and presented as confirmation of v70_0.\n\n## 16. Stop rule\n\nThis is a one-shot application test.\n\nIf v70_0 fails:\n\n- Phase 7 closes;\n- no v70_1 threshold tuning;\n- no target shopping;\n- no removal of an inconvenient Hamiltonian family;\n- no replacement of the EN2 matching strategy after outcome inspection.\n\nFurther work would require a genuinely new theoretical hypothesis and a new preregistration.\n\n## 17. Why this is distinct from Phase 5\n\nPhase 5 asked whether Soft Spaces could compete with EN2 on molecular energetic/determinant-selection tasks.\n\nPhase 7 asks whether Soft Spaces measures a **different physical property**:\n\n> coherent perturbative susceptibility and local eigenspace reorganisation.\n\nIf Soft-Space prominence succeeds only when EN2 is not controlled, that is not evidence for orthogonal information.\n\nIf it succeeds after EN2 matching and independently across three Hamiltonian families, the project has identified information not directly encoded by EN2 energetic ranking.\n\n## 18. Frozen reporting table\n\n| Metric | H4 | LiH | Synthetic | Pooled |\n|---|---:|---:|---:|---:|\n| EN2-only predictive score | | | | |\n| EN2 + Soft Space predictive score | | | | |\n| Delta predictive score | | | | |\n| matched/partial rho | | | | |\n| positive-instance fraction | | | | |\n| CLEAR PASS component | | | | |\n\nAll preregistered results must be reported, including negative ones.\n\n## 19. Final preregistered question\n\n> **When energetic importance is held approximately constant, does the frozen Soft Spaces mechanism predict which local eigenspaces will reorganize most strongly under genuinely held-out perturbations?**\n\nThat is the sole confirmatory question of Phase 7.\n"

def main():
    repo = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_REPO
    if not (repo / ".git").exists():
        raise SystemExit(f"Not a Git repository root: {repo}")

    root = repo / "Phase7" / "V_1_0"

    dirs = [
        root / "code",
        root / "config",
        root / "data" / "raw",
        root / "data" / "processed",
        root / "docs" / "preregistration",
        root / "docs" / "methods",
        root / "results",
        root / "figures",
        root / "logs",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        keep = d / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")

    target = root / "docs" / "preregistration" / "v70_0_ORTHOGONAL_INFORMATION_PREREGISTRATION.md"
    target.write_text(PREREG, encoding="utf-8")

    readme = root / "README.md"
    readme.write_text(
        "# Phase 7 — Orthogonal Information Test\n\n"
        "Phase 7 is a one-shot preregistered test of whether the frozen Phase 2/3 "
        "Soft Spaces mechanism contains predictive information about perturbation-induced "
        "eigenspace reorganisation after controlling for EN2 energetic importance.\n\n"
        "Primary preregistration:\n\n"
        "`docs/preregistration/v70_0_ORTHOGONAL_INFORMATION_PREREGISTRATION.md`\n\n"
        "No post-hoc threshold or target tuning is permitted for the v70_0 claim.\n",
        encoding="utf-8",
    )

    print(f"Installed Phase 7 under: {root}")
    print(f"Preregistration: {target}")
    print()
    print("Next:")
    print("  git status -sb")
    print("  git add Phase7")
    print('  git commit -m "Add Phase 7 orthogonal-information preregistration"')
    print("  git push origin main")

if __name__ == "__main__":
    main()
