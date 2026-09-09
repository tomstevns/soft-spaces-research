# Soft Spaces

## Exploring Hidden Structures in Low-Qubit Hilbert Spaces

This repository contains the research material for the **Soft Spaces** project.

The project investigates whether finite-dimensional Hilbert spaces contain
reproducible structural regions that emerge under perturbations and open-system
quantum dynamics.

The repository is organized by research phase.

---

## Phase 1 — Discovery and Structural Exploration

**Location:** [`Phase1/V_1_0`](Phase1/V_1_0)

Phase 1 contains the original exploratory work on low-qubit systems, primarily
covering **4–8 qubits**.

The work includes:

- numerical exploration of candidate soft regions
- REAL versus NULL controls
- hotspot recurrence and stability tests
- local kernel and ridge analysis
- open-system and Lindblad perturbations
- repeated-seed validation
- development of the original Soft Spaces methodology

Phase 1 established the empirical basis for the Soft Spaces hypothesis and
identified reproducible candidate structures in low-dimensional Hilbert spaces.

---

## Phase 2 — Dimensional Transfer and Physics Validation

**Location:** [`Phase2/V_1_0`](Phase2/V_1_0)

Phase 2 extends the investigation beyond the original 4–8 qubit regime and
focuses on whether the structures discovered in Phase 1 survive increasingly
strict physical and mathematical tests.

The work includes:

- extension to 9, 10, 11 and 12 qubits
- dimensional hotspot transfer
- perturbative confirmation
- REAL/NULL cancellation tests
- anisotropy analysis
- effective-subspace geometry
- dimensional prediction rules
- 12-qubit confirmation
- Qiskit circuit design and hardware-oriented tests
- analytic and numerical robustness audits
- independent coupling and basis-invariance tests

Phase 2 is intended to distinguish genuine structural effects from numerical,
spectral, basis-dependent or implementation artifacts.

---

## Repository Structure

```text
soft-spaces-research/
│
├── Phase1/
│   └── V_1_0/
│
├── Phase2/
│   └── V_1_0/
│
├── .gitignore
└── README.md
