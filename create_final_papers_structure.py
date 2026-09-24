#!/usr/bin/env python3



from pathlib import Path


ROOT = Path("Final_Papers")

DIRECTORIES = [
    ROOT,
    ROOT / "overview",
    ROOT / "paper_1_core",
    ROOT / "paper_2_molecular",
    ROOT / "paper_3_limits",
    ROOT / "figures",
    ROOT / "supplementary",
    ROOT / "archive",
]


README_CONTENT = """# Soft Spaces — Final Papers

## Project status

**Status: CLOSED**

The Soft Spaces project established a reproducible perturbative eigenstructure
in small quantum Hilbert spaces and investigated whether that structure could
provide predictive or computational utility in downstream quantum applications.

The strongest positive result is the reproducible REAL-vs-NULL structure found
in the original low-qubit studies and later transferred, with structural
adaptation, to molecular eigenstate spaces.

Subsequent application testing did not demonstrate a robust, transferable
computational or sensing advantage over established baselines.

In particular:

- the molecular REAL proxy was found to behave very similarly to EN2;
- the original Phase-2/3 Soft-Spaces mechanism remained distinguishable from
  the simple EN2 determinant ranking;
- matched-budget ground-state determinant/subspace selection strongly favored EN2;
- the H4 predictive gauntlet showed moderate Soft-Spaces signals but did not
  satisfy the preregistered CLEAR-PASS criteria;
- the corresponding LiH replication did not reproduce the H4 advantage;
- the Phase-6 sensing/metrology gauntlet did not demonstrate a robust advantage
  over EN2 under the frozen success criteria.

The final scientific position is therefore:

> Soft Spaces appears to describe a reproducible perturbative eigenstructure in
> small quantum Hilbert spaces. The structure survives several REAL-vs-NULL and
> molecular-transfer tests, but no robust, transferable computational or sensing
> advantage over established baselines has been demonstrated.

This repository preserves both the positive structural findings and the negative
application results.

---

## Final paper series

The project is closed with four final papers.

### 1. Closing overview

Location:

`overview/Soft_Spaces_Closing_Overview.docx`

Purpose:

A compact overview of the full Soft Spaces research program, including discovery,
formalization, molecular transfer, application testing, negative results, and
scientific closure.

---

### 2. Paper I — Core phenomenon

Location:

`paper_1_core/Soft_Spaces_I_Core_Phenomenon.docx`

Focus:

- REAL-vs-NULL construction
- perturbation families
- C_matrix
- robust cross-family score
- local prominence
- hotspot structure
- replication across low-qubit Hilbert spaces
- interpretation of Soft Spaces as a perturbative eigenstructure phenomenon

This paper contains the strongest positive scientific result of the project.

---

### 3. Paper II — Molecular transfer and EN2

Location:

`paper_2_molecular/Soft_Spaces_II_Molecular_Transfer_and_EN2.docx`

Focus:

- H4 and LiH molecular Hamiltonians
- fixed-particle sectors
- molecular transfer of the original Phase-2/3 mechanism
- distinction between the molecular REAL proxy and the original Soft-Spaces mechanism
- EN2 comparison
- matched-budget ground-state subspace selection
- H4 predictive gauntlet
- LiH independent replication

Main conclusion:

The molecular Soft-Spaces structure is reproducible, but no robust computational
advantage over EN2 was demonstrated.

---

### 4. Paper III — Limits and scientific closure

Location:

`paper_3_limits/Soft_Spaces_III_Limits_and_Closure.docx`

Focus:

- application screening
- hardware/noise robustness
- predictive testing
- sensing/metrology testing
- negative results
- methodological boundaries
- preregistered stopping rules
- final scientific interpretation

Main conclusion:

Soft Spaces remains scientifically interesting as a structured perturbative
eigenstate phenomenon, but its practical computational utility has not been
demonstrated.

---

## Supporting material

### Figures

`figures/`

Use this directory for publication-quality figures used across the final papers.

### Supplementary material

`supplementary/`

Use this directory for:

- extended tables
- additional numerical summaries
- methodological appendices
- seed lists
- reproducibility notes
- selected result JSON/CSV files

### Archive

`archive/`

Use this directory for frozen bundles, ZIP files, or older final-paper snapshots.

---

## Reproducibility

The underlying scripts, result files, seed definitions, hashes, and historical
phase directories remain in the repository.

The final papers should be read together with the corresponding source code and
result artifacts from Phases 2–6.

Important reproducibility principles used in the later phases include:

- frozen seed sets;
- explicit REAL-vs-NULL controls;
- fixed-particle-sector molecular Hamiltonians;
- matched-budget comparisons;
- preregistered or predeclared success criteria;
- no post-hoc threshold retuning after failed application tests.

---

## Claim boundaries

The project does **not** establish:

- quantum advantage;
- industrial deployment readiness;
- a general computational advantage over EN2;
- a validated molecular chemistry acceleration method;
- a demonstrated sensing advantage;
- a current commercial product or business case.

The project **does** provide evidence for:

- reproducible perturbative structure in selected small quantum Hilbert spaces;
- nontrivial REAL-vs-NULL separation;
- identifiable local eigenstate-subspace hotspots;
- scientifically useful negative results that delimit the practical scope of
  the observed structure.

---

## Repository status

The research program is considered scientifically closed in its current form.

Future work should only reopen the project if a genuinely new hypothesis,
independent application class, or theoretical development provides a strong
scientific reason to do so.

It should not be reopened merely to search for another benchmark on which
Soft Spaces might outperform an existing method.
"""


def main() -> None:
    print("=" * 78)
    print("SOFT SPACES — CREATE FINAL PAPERS STRUCTURE")
    print("=" * 78)

    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[DIR ] {directory}")

    readme = ROOT / "README.md"
    readme.write_text(README_CONTENT, encoding="utf-8")
    print(f"[WRITE] {readme}")

    print("-" * 78)
    print("Final_Papers structure created successfully.")
    print(f"Root: {ROOT.resolve()}")
    print()
    print("Expected final paper locations:")
    print("  Final_Papers/overview/Soft_Spaces_Closing_Overview.docx")
    print("  Final_Papers/paper_1_core/Soft_Spaces_I_Core_Phenomenon.docx")
    print("  Final_Papers/paper_2_molecular/Soft_Spaces_II_Molecular_Transfer_and_EN2.docx")
    print("  Final_Papers/paper_3_limits/Soft_Spaces_III_Limits_and_Closure.docx")
    print("=" * 78)


if __name__ == "__main__":
    main()
