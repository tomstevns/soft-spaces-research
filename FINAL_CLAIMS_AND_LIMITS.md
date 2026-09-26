# FINAL CLAIMS AND LIMITS

This document defines the canonical scientific claim boundary for the completed Soft Spaces project.

## Supported claims

### Reproducible REAL-vs-NULL perturbative structure

Within the tested low-qubit systems, the project found reproducible differences between structured REAL models and matched NULL controls under controlled perturbations.

### Local hotspot structure

The Phase 2/3 framework identified local eigenstate-subspace regions with elevated robust REAL-vs-NULL response relative to nearby controls.

### Cross-family testing

The project used more than one perturbation family, including dephasing-type and transverse-type perturbations.

### Molecular structural transfer

Adapted Phase 2/3-style tests identified reproducible REAL-vs-NULL molecular eigenstate-pair structure in H4 and LiH.

This supports structural transfer of the phenomenon into the molecular test setting.

It does not establish practical molecular utility.

### Noise/hardware-proxy robustness in tested regimes

Selected Phase 4 experiments showed that REAL-vs-NULL performance differences could survive substantial noisy-circuit/hardware-proxy conditions in the tested setup.

This does not establish quantum advantage or deployment readiness.

## Not demonstrated

The project does not establish:

- general computational advantage;
- superiority to EN2;
- robust molecular resource advantage;
- transferable sensing advantage;
- quantum speedup;
- production-hardware advantage;
- a deployable algorithm;
- a commercial product claim;
- universality across arbitrary Hilbert spaces.

## Critical terminology boundaries

### Soft Spaces

Use this term for the Phase 2/3 perturbative cancellation/reinforcement mechanism and its faithful adaptations.

### Molecular REAL proxy

For `v50_2`–`v50_6`, use **molecular REAL proxy**.

`REAL_i = |H_ir| / (|H_ii - H_rr| + eps)`

EN2 uses approximately:

`EN2_i = |H_ir|^2 / (|H_ii - H_rr| + eps)`

Near-identical rankings do not establish that the original Soft Spaces mechanism is EN2.

### Fixed particle/spin sector

For the H4 molecular work, the tested sector is fixed `(N_alpha, N_beta) = (2,2)`, corresponding to `M_S = 0`.

Do not call it a pure singlet (`S = 0`) sector unless `S^2` has been explicitly enforced or verified.

### REAL and NULL

In the original Phase 2/3 formulation:

- REAL uses the eigenbasis of the structured Hamiltonian.
- NULL preserves the relevant spectrum while randomizing the eigenbasis using seeded Haar-random controls.

## Selected Phase 5 boundaries

### v50_7c

Interpretation: reproducible molecular REAL-vs-NULL eigenstate-pair structure.

Not justified: general molecular advantage.

### v50_8

Interpretation: the chosen EN2 bridge did not systematically predict Soft Spaces hotspots.

Caveat: eligibility handling should be considered before making a strong standalone claim.

### v50_9

Matched-budget ground-state selection:

- 40 comparisons
- Soft Spaces wins: 0
- EN2 wins: 40

Interpretation: EN2 dominated this tested selection task.

### v50_10

No H4 gauntlet test met the frozen clear-pass criteria.

Some H4 tasks showed moderate Soft-Spaces-over-EN2 tendencies, but below the preregistered evidence threshold.

### v50_11

The H4 tendency did not replicate convincingly on LiH.

Interpretation: no transferable molecular application advantage was established.

## Phase 6 boundary

The sensing gauntlet tested:

- leakage susceptibility;
- first-order gap sensitivity;
- second-order gap curvature.

None met the frozen clear-pass criteria.

Therefore:

> **Sensing application not demonstrated.**

## Canonical final claim

> **Soft Spaces appears to describe a reproducible perturbative eigenstructure in small quantum Hilbert spaces. The structure survives several REAL-vs-NULL and molecular-transfer tests, but no robust, transferable computational or sensing advantage over established baselines has been demonstrated.**

## Project status

The project is scientifically closed at:

`soft-spaces-closure-v1.0`

A future reopening should require:

- independent replication;
- a genuinely new physical hypothesis;
- a new theoretical derivation that changes the expected observable; or
- a clearly preregistered experiment motivated independently of prior failed benchmarks.

Post-hoc metric, threshold, or benchmark shopping is not a valid reason to reopen the closed application program.
