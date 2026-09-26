# Soft Spaces Research Project

**Status:** Scientifically closed  
**Closure tag:** `soft-spaces-closure-v1.0`

Soft Spaces is a multi-phase research project investigating whether small local subspaces in quantum Hilbert spaces exhibit reproducible perturbative structure, and whether that structure can provide predictive or computational value.

The project progressed from the identification and characterization of a REAL-vs-NULL perturbative structure in low-qubit Hilbert spaces to increasingly demanding application tests, including molecular Hamiltonians, EN2 comparisons, hardware/noise proxies, and sensing-like susceptibility tasks.

## Final scientific conclusion

> **Soft Spaces appears to describe a reproducible perturbative eigenstructure in small quantum Hilbert spaces. The structure survives several REAL-vs-NULL and molecular-transfer tests, but no robust, transferable computational or sensing advantage over established baselines has been demonstrated.**

The project is therefore closed as an application-development effort. Its principal contribution is the identification and characterization of the structure itself, together with an explicit mapping of where practical utility was **not** demonstrated.

## Start here

For a rapid research-level overview, read:

1. [`START_HERE.md`](START_HERE.md) — project map and recommended reading order.
2. [`FINAL_CLAIMS_AND_LIMITS.md`](FINAL_CLAIMS_AND_LIMITS.md) — supported claims, unsupported claims, and terminology boundaries.
3. [`Final_Papers/README.md`](Final_Papers/README.md) — guide to the closing paper series.

The closing papers are:

- `Final_Papers/overview/Soft_Spaces_Closing_Overview.docx`
- `Final_Papers/paper_1_core/Soft_Spaces_I_Core_Phenomenon.docx`
- `Final_Papers/paper_2_molecular/Soft_Spaces_II_Molecular_Transfer_and_EN2.docx`
- `Final_Papers/paper_3_limits/Soft_Spaces_III_Limits_and_Closure.docx`

## Phase map

| Phase | Research question | Main outcome |
|---|---|---|
| Phase 1 | Can local structure be identified in small quantum Hilbert spaces? | Initial exploratory evidence and method development. |
| Phase 2 | Is the REAL-vs-NULL structure reproducible under controlled perturbations? | Reproducible hotspot structure established in tested low-qubit systems. |
| Phase 3 | Can the phenomenon be formalized and stress-tested? | Core perturbative mechanism formalized; closure focused on robustness and interpretation. |
| Phase 4 | Does the structure survive application-oriented and hardware/noise proxy tests? | Structural robustness observed in selected tests, but this did not establish general practical advantage. |
| Phase 5 | Does the previously established structure provide predictive or computational value in molecular quantum problems? | Molecular structure was detectable, but robust utility beyond EN2 was not demonstrated. |
| Phase 6 | Does the structure provide transferable sensing-like predictive value? | No preregistered clear-pass criterion was met; sensing application was not demonstrated. |
| Final | What remains scientifically supported? | Reproducible perturbative structure with clearly mapped practical limits. |

## Core mechanism

The Phase 2/3 mechanism uses a cancellation/coherence statistic:

`C_matrix = ||sum_m A_m||_F / sum_m ||A_m||_F`

For perturbation family `f`:

`DeltaC_f(k) = C_matrix_REAL,f(k) - C_matrix_NULL,f(k)`

A robust cross-family score and local prominence were then used to identify hotspot candidates.

In the original Phase 2/3 formulation, REAL used the eigenbasis of the structured Hamiltonian, while NULL controls preserved the eigenvalue spectrum and randomized the eigenbasis with seeded Haar-random controls.

## Important terminology boundary

The molecular work in `v50_2`–`v50_6` used a **current molecular REAL proxy**:

`REAL_i = |H_ir| / (|H_ii - H_rr| + eps)`

This is closely related to the EN2-style quantity:

`EN2_i = |H_ir|^2 / (|H_ii - H_rr| + eps)`

Therefore:

- the molecular proxy is not identical to the original Soft Spaces mechanism;
- near-identical proxy/EN2 rankings do not show that Soft Spaces equals EN2;
- later Phase 5 tests (`v50_7` onward) returned to the frozen Phase 2/3 mechanism.

## Key negative results

Negative results are part of the scientific closure:

- EN2-to-hotspot predictability was weak in the tested bridge.
- In matched-budget ground-state determinant selection, EN2 won all 40 comparisons.
- H4 tendencies in the preregistered EN2 gauntlet did not satisfy frozen clear-pass criteria.
- The H4 tendency did not replicate convincingly on LiH.
- The Phase 6 sensing gauntlet produced no clear-pass test.

## Reproducibility and data policy

The repository contains source code, preregistrations, result summaries, manifests, selected reproducibility artifacts, and the closing papers.

Very large public external datasets are intentionally excluded from normal Git tracking and should be re-downloaded from their original sources when required.

## Project status

No Phase 7 is planned from the existing evidence base.

A future reopening should require a genuinely new, independently motivated hypothesis rather than post-hoc benchmark or threshold selection.
