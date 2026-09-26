# START HERE — Soft Spaces Research Map

This file is the recommended entry point for an external researcher, reviewer, or collaborator.

## 1. What is the project about?

Soft Spaces investigates whether some local eigenstate subspaces in quantum Hilbert spaces show systematically different perturbative behavior in a physically structured REAL system than in matched NULL controls.

The central observation is a reproducible **REAL-vs-NULL perturbative structure** in tested low-qubit Hilbert spaces.

The later phases asked a harder question:

> Does that structure provide useful prediction, resource reduction, computational advantage, or sensing advantage?

The completed project did not demonstrate such an advantage robustly or transferably.

## 2. Recommended reading order

1. `Final_Papers/overview/Soft_Spaces_Closing_Overview.docx`
2. `Final_Papers/paper_1_core/Soft_Spaces_I_Core_Phenomenon.docx`
3. `Final_Papers/paper_2_molecular/Soft_Spaces_II_Molecular_Transfer_and_EN2.docx`
4. `Final_Papers/paper_3_limits/Soft_Spaces_III_Limits_and_Closure.docx`
5. `FINAL_CLAIMS_AND_LIMITS.md`

## 3. Six-phase research map

| Phase | Purpose | Research status |
|---|---|---|
| 1 | Exploratory development | Historical/exploratory |
| 2 | REAL-vs-NULL hotspot discovery and replication | Core positive result |
| 3 | Formalization and robustness testing | Core positive result with caveats |
| 4 | Application screening and hardware/noise proxies | Mixed |
| 5 | Molecular transfer and EN2 comparison | Structure transfers; robust utility not demonstrated |
| 6 | Sensing-like susceptibility tests | Application not demonstrated |
| Final | Consolidated papers and claim boundary | Project closed |

## 4. Core Phase 2/3 result

The method evaluates cancellation/reinforcement under perturbations using:

`C_matrix = ||sum_m A_m||_F / sum_m ||A_m||_F`

and:

`DeltaC_f(k) = C_matrix_REAL,f(k) - C_matrix_NULL,f(k)`

Cross-family robustness and local prominence identify hotspot candidates.

The strongest scientific result of the project is the reproducible REAL-vs-NULL perturbative structure found in the tested low-qubit systems.

## 5. Frozen mechanism versus molecular proxy

The Phase 5 molecular proxy used in `v50_2`–`v50_6` is not identical to the original Soft Spaces mechanism.

Proxy:

`REAL_i = |H_ir| / (|H_ii - H_rr| + eps)`

EN2-like score:

`EN2_i = |H_ir|^2 / (|H_ii - H_rr| + eps)`

From `v50_7` onward, the molecular tests returned to an adapted version of the frozen Phase 2/3 mechanism.

## 6. Molecular results

For H4/STO-3G, the tested sector was fixed `(N_alpha, N_beta) = (2,2)`, corresponding to `M_S = 0`.

This must not be described as a pure `S = 0` singlet sector unless `S^2` is explicitly enforced.

The Phase 2/3-style mechanism found reproducible molecular eigenstate-pair structure in H4 and LiH.

However, matched-budget ground-state selection strongly favored EN2:

- 40 comparisons
- Soft Spaces wins: 0
- EN2 wins: 40
- ties: 0

H4 showed some sub-threshold tendencies in the preregistered gauntlet, but these did not satisfy the frozen clear-pass criteria and did not replicate convincingly on LiH.

## 7. Phase 6 sensing result

Three frozen sensing-style tests were used:

- S1: leakage susceptibility
- S2: first-order gap sensitivity
- S3: second-order gap curvature

No test met the preregistered clear-pass criteria.

The sensing application is therefore **not demonstrated**.

## 8. Canonical final statement

> **Soft Spaces appears to describe a reproducible perturbative eigenstructure in small quantum Hilbert spaces. The structure survives several REAL-vs-NULL and molecular-transfer tests, but no robust, transferable computational or sensing advantage over established baselines has been demonstrated.**

## 9. Closure

Canonical closure tag:

`soft-spaces-closure-v1.0`

Further work should be treated as a new research program unless it directly reproduces, challenges, or theoretically extends the existing claim set.
