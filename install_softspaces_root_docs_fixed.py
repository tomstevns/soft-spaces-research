#!/usr/bin/env python3
r"""
Install the three canonical root-level documentation files for the
Soft Spaces repository.

Default repository root:
    C:\Users\tomst\PycharmProjects\SoftSpaces_Phase2

Usage:
    python install_softspaces_root_docs.py

or:
    python install_softspaces_root_docs.py "C:\path\to\SoftSpaces_Phase2"

The script creates or replaces:
    README.md
    START_HERE.md
    FINAL_CLAIMS_AND_LIMITS.md

Existing target files are backed up automatically.
The script does NOT run git commit or git push.
"""

from pathlib import Path
from datetime import datetime
import shutil
import sys

DEFAULT_REPO = Path(r"C:\Users\tomst\PycharmProjects\SoftSpaces_Phase2")

README = """# Soft Spaces Research Project

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
"""

START_HERE = """# START HERE — Soft Spaces Research Map

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
"""

FINAL_CLAIMS = """# FINAL CLAIMS AND LIMITS

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
"""

def backup(path: Path) -> None:
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = path.with_name(path.name + f".bak-{stamp}")
        shutil.copy2(path, target)
        print(f"Backup created: {target}")

def write_file(path: Path, content: str) -> None:
    backup(path)
    path.write_text(content.strip() + "\n", encoding="utf-8", newline="\n")
    print(f"Written: {path}")

def main() -> int:
    repo = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_REPO

    print(f"Repository: {repo}")

    if not repo.exists():
        print("ERROR: repository path does not exist.")
        return 2

    if not (repo / ".git").exists():
        print("ERROR: this does not look like the Git repository root.")
        return 3

    write_file(repo / "README.md", README)
    write_file(repo / "START_HERE.md", START_HERE)
    write_file(repo / "FINAL_CLAIMS_AND_LIMITS.md", FINAL_CLAIMS)

    print("\nInstallation complete.")
    print("\nNext commands:")
    print("git status -sb")
    print("git diff -- README.md START_HERE.md FINAL_CLAIMS_AND_LIMITS.md")
    print("git add README.md START_HERE.md FINAL_CLAIMS_AND_LIMITS.md")
    print('git commit -m "Add canonical research map and final claim boundaries"')
    print("git push origin main")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
