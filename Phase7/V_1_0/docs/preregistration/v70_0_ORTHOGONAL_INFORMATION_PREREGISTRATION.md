# Phase 7 Preregistration
## Orthogonal Information Test: Soft-Space Prominence vs EN2

**Version:** v70_0  
**Status:** FROZEN BEFORE EXECUTION  
**Purpose:** One final, falsifiable reopening test after formal project closure.

## 1. Scientific question

Phase 7 does **not** retest whether Soft Spaces can outperform EN2 on ground-state determinant selection. Phase 5 already tested that question and EN2 dominated the matched-budget ground-state selection task.

Phase 7 asks instead:

> **Does Soft-Space prominence contain predictive information about perturbation-induced eigenstate/subspace reorganisation that is not already captured by EN2?**

The target is therefore **orthogonal information**, not superiority on EN2's native energy-ranking task.

## 2. Primary hypothesis

For candidate local eigenspaces matched to have similar EN2 importance, higher Soft-Space prominence predicts stronger perturbation-induced reorganisation.

Operationally:

> Among EN2-matched candidate pairs, Soft-Space prominence should predict larger subspace rotation / fidelity loss / eigenvector mixing under held-out perturbations better than EN2 itself.

## 3. Null hypothesis

After controlling for EN2, Soft-Space prominence contains no additional predictive information about perturbation-induced eigenspace reorganisation.

Equivalent outcomes include:

- chance-level performance after EN2 matching;
- no incremental prediction beyond EN2;
- a system-specific signal that fails independent replication.

## 4. Frozen Soft Spaces mechanism

Phase 7 must use the **frozen Phase 2/3 Soft Spaces mechanism**, not the v50_2–v50_6 molecular REAL proxy.

Core quantity:

`C_matrix = ||sum_m A_m||_F / sum_m ||A_m||_F`

For perturbation family `f`:

`DeltaC_f(k) = C_matrix_REAL,f(k) - C_matrix_NULL,f(k)`

Robust cross-family score:

`S(k) = min(median DeltaC_dephasing(k), median DeltaC_transverse(k))`

Local prominence:

`P(k) = S(k) - median_nearby S(k')`

The exact implementation must be inherited from the frozen Phase 2/3 source semantics and must not be altered after inspection of Phase 7 outcomes.

## 5. EN2 control variable

EN2 is treated as the energetic-importance control variable.

Working project-consistent EN2-like score:

`EN2_i = |H_ir|^2 / (|H_ii - H_rr| + eps)`

Phase 7 does not claim that this scalar is the complete selected-CI EN2 algorithm. It is used as the energetic ranking control.

## 6. Systems

Phase 7 must contain at least **three independent Hamiltonian families**:

1. **H4/STO-3G geometry family**
   - reuse the established geometry scan;
   - fixed `(N_alpha, N_beta) = (2,2)` sector.

2. **LiH/STO-3G geometry family**
   - reuse the established LiH scan;
   - same particle-number/spin-sector conventions as Phase 5.

3. **Independent synthetic avoided-crossing family**
   - parameterized Hamiltonians with genuine local near-degeneracies / avoided crossings;
   - construction frozen before evaluating Phase 7 target metrics;
   - not tuned to maximize Soft-Space performance.

The third family prevents the conclusion from resting only on H4/LiH chemistry.

## 7. Candidate definition

Candidate local eigenspaces are adjacent eigenstate pairs `(k, k+1)` satisfying the same eligibility rules used by the frozen Phase 2/3 mechanism.

Eligibility must be computed before target labels are examined.

Ineligible pairs must not be silently treated as negatives.

## 8. EN2 matching

The central anti-confounding step is to compare candidates at approximately equal EN2 importance.

Primary method:

- rank candidates by EN2;
- divide into EN2 quantile bins within each Hamiltonian instance;
- compare Soft-Space prominence only within bins.

Secondary robustness method:

- nearest-neighbour matching in `log10(EN2 + eps)`;
- maximum standardized EN2-distance fixed before execution.

No matching tolerance may be changed after seeing outcomes.

## 9. Held-out perturbations

Soft-Space prominence is computed from the frozen discovery perturbation families.

Targets must be measured using **held-out perturbations** not used to construct the Soft-Space score.

Minimum held-out set:

- one dephasing-like family;
- one transverse-like family;
- one mixed family;
- one parameter-displacement perturbation relevant to the Hamiltonian family.

Held-out seeds must be disjoint from discovery seeds.

## 10. Primary target: subspace reorganisation

For each candidate pair, let the unperturbed two-dimensional subspace be `U` and the perturbed subspace be `U'`.

Primary endpoint:

`R_subspace = sin(theta_max(U, U'))`

where `theta_max` is the largest principal angle between the two subspaces.

Interpretation:

- `0` = no subspace rotation;
- values approaching `1` = strong reorganisation.

This is the primary endpoint.

## 11. Secondary targets

Secondary endpoints are reported but cannot rescue a failed primary endpoint:

### T2 — subspace fidelity loss
Projector-overlap / fidelity-style loss between `U` and `U'`.

### T3 — eigenvector mixing
Change in participation or overlap structure within a local spectral window.

### T4 — local spectral response
Magnitude of perturbation-induced local gap change or avoided-crossing displacement.

No new target may be introduced after results are inspected and then treated as confirmatory.

## 12. Primary statistical comparison

Within EN2-matched strata, test whether Soft-Space prominence predicts `R_subspace`.

Primary comparison:

- Model A: EN2 only
- Model B: EN2 + Soft-Space prominence

The question is whether Model B yields reproducible out-of-sample improvement.

## 13. Cross-validation

Cross-validation must separate Hamiltonian instances rather than randomly splitting candidate pairs from the same Hamiltonian.

Preferred scheme:

- leave-one-geometry-out within H4;
- leave-one-geometry-out within LiH;
- leave-one-parameter-instance-out within the synthetic family.

A pair from a held-out Hamiltonian instance must not influence training.

## 14. Frozen success criteria

Phase 7 is a **CLEAR PASS** only if all of the following hold:

1. **Primary pooled association**
   - out-of-sample matched/partial association between Soft-Space prominence and `R_subspace` after EN2 control:
   - `rho >= 0.25`.

2. **Incremental predictive value**
   - Model B improves out-of-sample prediction over Model A by:
   - `Delta R^2 >= 0.05`,
   - or an explicitly preregistered rank-prediction equivalent if a nonparametric model is used.

3. **Replication**
   - positive incremental effect in all three Hamiltonian families.

4. **Per-family minimum**
   - at least two of the three families individually show:
   - matched/partial `rho >= 0.20`.

5. **Direction consistency**
   - at least 70% of held-out Hamiltonian instances show the preregistered positive direction.

6. **No EN2-imbalance explanation**
   - the result remains positive under the secondary nearest-neighbour EN2 matching.

If criteria 1–3 fail, Phase 7 is a **FAIL**.

Criteria 4–6 are also required for a CLEAR PASS.

## 15. Interpretation rules

### CLEAR PASS permits

A CLEAR PASS permits the claim:

> Soft-Space prominence contains reproducible information about perturbation-induced eigenspace reorganisation that is not captured by EN2 energetic importance alone.

It does **not** permit claims of:

- general computational superiority to EN2;
- quantum speedup;
- lower selected-CI cost;
- commercial utility;
- universal Hilbert-space behavior.

### FAIL permits

A FAIL supports:

> After controlling for EN2, the tested Soft-Space prominence did not show robust transferable predictive information for eigenspace reorganisation across the preregistered Hamiltonian families.

No threshold, target, matching tolerance, system, or perturbation family may then be changed post hoc and presented as confirmation of v70_0.

## 16. Stop rule

This is a one-shot application test.

If v70_0 fails:

- Phase 7 closes;
- no v70_1 threshold tuning;
- no target shopping;
- no removal of an inconvenient Hamiltonian family;
- no replacement of the EN2 matching strategy after outcome inspection.

Further work would require a genuinely new theoretical hypothesis and a new preregistration.

## 17. Why this is distinct from Phase 5

Phase 5 asked whether Soft Spaces could compete with EN2 on molecular energetic/determinant-selection tasks.

Phase 7 asks whether Soft Spaces measures a **different physical property**:

> coherent perturbative susceptibility and local eigenspace reorganisation.

If Soft-Space prominence succeeds only when EN2 is not controlled, that is not evidence for orthogonal information.

If it succeeds after EN2 matching and independently across three Hamiltonian families, the project has identified information not directly encoded by EN2 energetic ranking.

## 18. Frozen reporting table

| Metric | H4 | LiH | Synthetic | Pooled |
|---|---:|---:|---:|---:|
| EN2-only predictive score | | | | |
| EN2 + Soft Space predictive score | | | | |
| Delta predictive score | | | | |
| matched/partial rho | | | | |
| positive-instance fraction | | | | |
| CLEAR PASS component | | | | |

All preregistered results must be reported, including negative ones.

## 19. Final preregistered question

> **When energetic importance is held approximately constant, does the frozen Soft Spaces mechanism predict which local eigenspaces will reorganize most strongly under genuinely held-out perturbations?**

That is the sole confirmatory question of Phase 7.
