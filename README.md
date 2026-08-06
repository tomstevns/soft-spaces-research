# Soft Spaces

**Exploring hidden structure in finite-dimensional Hilbert spaces**

---

## Overview

Soft Spaces is an independent research project investigating whether finite-dimensional Hilbert spaces contain reproducible structural regions that differ significantly from suitable null models.

The project introduces the concepts of **REAL** and **NULL** subspaces and studies whether statistically reproducible geometric structure emerges across increasing Hilbert-space dimensions.

Rather than searching for individual quantum states, the project investigates whether **stable regions (Soft Spaces)** exist within Hilbert space itself.

---

# Main publication

**Soft Spaces in Low-Dimensional Hilbert Spaces:
A Reproducible REAL–NULL Separation from 4 to 8 Qubits**

Current review version:

**v9.12**

The paper presents

- the complete theoretical framework
- experimental methodology
- statistical validation
- independent reproduction
- discussion of limitations

---

# Repository contents

```
docs/
    Main paper (PDF)

src/
    Python source code

results/
    Experimental output

figures/
    Figures used in the paper

README.md
LICENSE
```

---

# Main results

The reviewed study demonstrates reproducible statistical separation between REAL and NULL subspaces for systems from

- 4 qubits
- 5 qubits
- 6 qubits
- 7 qubits
- 8 qubits

using independent pilot runs and multiple statistical measures, including

- entropy
- Cohen's d
- Jaccard similarity
- dominant-state statistics
- stability analysis

---

# Current research (Phase 2)

The repository also contains ongoing work extending the original study toward larger Hilbert spaces.

Current topics include

- 9-qubit systems
- 10-qubit systems
- continuation hypotheses
- ridge structures
- targeted interval analysis
- open-system evolution
- Lindblad dynamics

These studies are experimental and are **not part of the reviewed paper**.

---

# Reproducibility

The repository contains

- source code
- configuration
- parameter files
- selected experimental results

allowing independent verification of the published experiments.

---

# Citation

A permanent Zenodo DOI will be added after the first archived release.

Until then, please cite the GitHub repository.

GitHub repository:

https://github.com/tomstevns/soft-spaces-github

Zenodo

DOI:
10.5281/zenodo.21819815

https://doi.org/10.5281/zenodo.21819815

# Project philosophy

The project follows a simple principle:

> Hypotheses should be challenged by independent repetition.

Unexpected results are treated as opportunities to refine the underlying model rather than as failures.

---

# Status

| Phase | Status |
|--------|--------|
| Theory | ✓ |
| q4–q8 validation | ✓ |
| Independent q8 replication | ✓ |
| Review paper | ✓ |
| q9 investigation | Ongoing |
| q10 investigation | Ongoing |
| Open-system analysis | Ongoing |

---

# Author

Tom Stevns

Independent Researcher

Denmark

---

# License

MIT License
