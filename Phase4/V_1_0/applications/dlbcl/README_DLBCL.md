# Soft Spaces Phase 4 — DLBCL application

## Objective

Test whether the Soft Spaces formalism can recover and prioritize biologically
meaningful structure in a standard DLBCL bioinformatics benchmark.

The first benchmark is **cell-of-origin (COO)** structure in **GSE10846**:
primarily **ABC vs GCB**, with unclassified samples retained in the raw metadata
but excluded from the first binary benchmark unless explicitly stated.

## Dataset

- GEO accession: `GSE10846`
- Organism: *Homo sapiens*
- Platform: `GPL570` (Affymetrix Human Genome U133 Plus 2.0 Array)
- Public cohort: 420 samples

## Version plan

- `v41.0` — download, archive and inspect GSE10846 + GPL570 annotation
- `v41.1` — reproducible preprocessing and probe-to-gene mapping
- `v41.2` — classical baseline models
- `v41.3` — Soft Spaces REAL/NULL construction
- `v41.4` — locked comparative benchmark
- `v41.5` — independent validation / robustness

No QPU execution is required for the initial bioinformatics development.
