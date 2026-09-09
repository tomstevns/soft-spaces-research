#!/usr/bin/env python3
"""Soft Spaces Phase 2 v25.41 — sparse-Pauli spectral artifact audit.

Frozen question
---------------
Does the high-Q rule k -> 2k+1 identify a dynamical object transported between
independent Hamiltonians, or does it track fixed normalized spectral boundaries
created by using only five Pauli terms while the Hilbert dimension doubles?

This audit uses the already frozen v25.31 seeds and checkpoint.  It performs no
new hotspot search, changes no coordinate or threshold, and drops no seed.

The exact algebraic facts used are:

1. For k' = 2k+1, (k'+1)/2^(q+1) = (k+1)/2^q.
2. If s independent Pauli words generate a binary symplectic space of rank t,
   their representation is Clifford-equivalent to an operator on
   s-t/2 active qubits tensored with identity on q-s+t/2 spectators.
3. Equivalently, the physical spectrum is the 2^s-dimensional twisted regular
   spectrum repeated 2^(q-s) times.  With q=12 and s=5 the repeat is 128.

The program constructs that 32-dimensional regular representation directly,
checks the frozen spectral pairs and controls, and then re-analyses only the
already stored v25.31 checkpoint values.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PAULIS = ("I", "X", "Y", "Z")
QUBITS = 12
N_TERMS = 5
BASE_SEED = 50_000_000
BATCH_STRIDE = 1_000_000
BATCHES = 8
SEEDS_PER_BATCH = 5
EPS_NEIGHBOR = 0.05
FROZEN = (511, 1535)
LOCAL_RADIUS = 10
LOCAL_STEP = 2


SINGLE = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.diag([1, -1]).astype(np.complex128),
}


def multiplication_table() -> dict[tuple[str, str], tuple[complex, str]]:
    table: dict[tuple[str, str], tuple[complex, str]] = {}
    phases = (1.0 + 0j, -1.0 + 0j, 1j, -1j)
    for left in PAULIS:
        for right in PAULIS:
            product = SINGLE[left] @ SINGLE[right]
            matches = [
                (phase, result)
                for result in PAULIS
                for phase in phases
                if np.array_equal(product, phase * SINGLE[result])
            ]
            if len(matches) != 1:
                raise RuntimeError("Could not build exact single-qubit Pauli table")
            table[left, right] = matches[0]
    return table


MULTIPLY = multiplication_table()


def multiply_labels(left: str, right: str) -> tuple[complex, str]:
    phase = 1.0 + 0j
    output: list[str] = []
    for a, b in zip(left, right):
        local_phase, symbol = MULTIPLY[a, b]
        phase *= local_phase
        output.append(symbol)
    return phase, "".join(output)


def index_to_label(index: int, n_qubits: int) -> str:
    value = int(index) + 1
    digits: list[str] = []
    for _ in range(n_qubits):
        value, remainder = divmod(value, 4)
        digits.append(PAULIS[remainder])
    return "".join(reversed(digits))


def random_hamiltonian_terms(
    n_qubits: int, n_terms: int, seed: int
) -> list[tuple[str, float]]:
    """Exact copy of the frozen v25.31 Hamiltonian sampling rule."""
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, 4**n_qubits - 1, size=int(n_terms))
    coefficients = rng.uniform(-1.0, 1.0, size=int(n_terms))
    return [
        (index_to_label(int(index), n_qubits), float(coefficient))
        for index, coefficient in zip(indices, coefficients)
    ]


def binary_vector(label: str) -> np.ndarray:
    x = [symbol in ("X", "Y") for symbol in label]
    z = [symbol in ("Z", "Y") for symbol in label]
    return np.asarray(x + z, dtype=np.uint8)


def gf2_rank(matrix: np.ndarray) -> int:
    reduced = np.asarray(matrix, dtype=np.uint8).copy() & 1
    rows, columns = reduced.shape
    rank = 0
    for column in range(columns):
        pivot = next((row for row in range(rank, rows) if reduced[row, column]), None)
        if pivot is None:
            continue
        reduced[[rank, pivot]] = reduced[[pivot, rank]]
        for row in range(rows):
            if row != rank and reduced[row, column]:
                reduced[row] ^= reduced[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def symplectic_rank(labels: list[str]) -> tuple[int, int]:
    vectors = np.asarray([binary_vector(label) for label in labels], dtype=np.uint8)
    q = len(labels[0])
    x, z = vectors[:, :q], vectors[:, q:]
    commutators = (x @ z.T + z @ x.T) & 1
    return gf2_rank(vectors), gf2_rank(commutators)


def has_common_anticommuter(labels: list[str]) -> bool:
    vectors = np.asarray([binary_vector(label) for label in labels], dtype=np.uint8)
    q = len(labels[0])
    # <(x,z),(u,v)> = x.v + z.u.  Solve A (u,v)^T = 1.
    equations = np.concatenate([vectors[:, q:], vectors[:, :q]], axis=1)
    augmented = np.concatenate(
        [equations, np.ones((len(labels), 1), dtype=np.uint8)], axis=1
    )
    return gf2_rank(equations) == gf2_rank(augmented)


def regular_spectrum(terms: list[tuple[str, float]]) -> np.ndarray:
    """Spectrum of the 2^s-dimensional twisted left-regular representation."""
    labels = [label for label, _ in terms]
    s = len(labels)
    size = 1 << s
    identity = "I" * len(labels[0])
    representatives: list[tuple[complex, str]] = []
    label_to_mask: dict[str, int] = {}

    for mask in range(size):
        phase = 1.0 + 0j
        label = identity
        for generator in range(s):
            if (mask >> generator) & 1:
                local_phase, label = multiply_labels(label, labels[generator])
                phase *= local_phase
        if label in label_to_mask:
            raise RuntimeError("Pauli generators are not binary independent")
        label_to_mask[label] = mask
        representatives.append((phase, label))

    generators: list[np.ndarray] = []
    for generator_label in labels:
        matrix = np.zeros((size, size), dtype=np.complex128)
        for source, (basis_phase, basis_label) in enumerate(representatives):
            local_phase, output_label = multiply_labels(generator_label, basis_label)
            target = label_to_mask[output_label]
            target_phase = representatives[target][0]
            matrix[target, source] = basis_phase * local_phase / target_phase
        if not np.array_equal(matrix, matrix.conj().T):
            raise RuntimeError("Regular Pauli generator is not exactly Hermitian")
        generators.append(matrix)

    hamiltonian = sum(
        coefficient * generator
        for generator, (_, coefficient) in zip(generators, terms)
    )
    return np.linalg.eigvalsh(hamiltonian)


def dense_spectrum(terms: list[tuple[str, float]]) -> np.ndarray:
    dimension = 1 << len(terms[0][0])
    hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)
    for label, coefficient in terms:
        operator = np.asarray([[1.0 + 0j]])
        for symbol in label:
            operator = np.kron(operator, SINGLE[symbol])
        hamiltonian += coefficient * operator
    return np.linalg.eigvalsh(hamiltonian)


def regular_representation_implementation_error(seeds: list[int]) -> float:
    """Independent dense 8Q check of the reduced-spectrum construction."""
    errors: list[float] = []
    for seed in seeds:
        terms = random_hamiltonian_terms(8, N_TERMS, seed)
        labels = [label for label, _ in terms]
        binary_rank, _ = symplectic_rank(labels)
        if binary_rank != N_TERMS:
            raise RuntimeError("Dense implementation check requires independent labels")
        predicted = np.repeat(regular_spectrum(terms), 1 << (8 - binary_rank))
        observed = dense_spectrum(terms)
        errors.append(float(np.max(np.abs(predicted - observed))))
    return max(errors)


def controls(center: int) -> list[int]:
    return [
        coordinate
        for coordinate in range(center - LOCAL_RADIUS, center + LOCAL_RADIUS + 1, LOCAL_STEP)
        if coordinate != center
    ]


@dataclass(frozen=True)
class AlgebraRow:
    seed: int
    binary_rank: int
    symplectic_rank: int
    forced_multiplicity: int
    common_anticommuter: bool
    candidate_gaps: tuple[float, float, float, float]
    max_control_gap: float
    branch_gap_multiset_error: float


def algebra_row(seed: int) -> AlgebraRow:
    terms = random_hamiltonian_terms(QUBITS, N_TERMS, seed)
    labels = [label for label, _ in terms]
    binary_rank, commutator_rank = symplectic_rank(labels)
    if binary_rank != N_TERMS:
        raise RuntimeError("Frozen audit requires five independent Pauli generators")

    multiplicity = 1 << (QUBITS - binary_rank + commutator_rank // 2)
    reduced = regular_spectrum(terms)
    regular_repeat = 1 << (QUBITS - binary_rank)
    physical = np.repeat(reduced, regular_repeat)
    if physical.size != 1 << QUBITS:
        raise RuntimeError("Regular-spectrum replication has wrong dimension")

    pair_left = (511, 1535, 2559, 3583)
    candidate_gaps = tuple(float(physical[index + 1] - physical[index]) for index in pair_left)
    control_indices = [coordinate for center in FROZEN for coordinate in controls(center)]
    control_indices += [coordinate + (1 << 11) for coordinate in control_indices]
    max_control_gap = max(abs(float(physical[index + 1] - physical[index])) for index in control_indices)

    # A branches are first/third; B branches are second/fourth.  Spectral
    # reflection predicts identical unordered branch-gap pairs.
    a = np.sort(np.asarray([candidate_gaps[0], candidate_gaps[2]]))
    b = np.sort(np.asarray([candidate_gaps[1], candidate_gaps[3]]))
    branch_error = float(np.max(np.abs(a - b)))

    return AlgebraRow(
        seed=seed,
        binary_rank=binary_rank,
        symplectic_rank=commutator_rank,
        forced_multiplicity=multiplicity,
        common_anticommuter=has_common_anticommuter(labels),
        candidate_gaps=candidate_gaps,
        max_control_gap=max_control_gap,
        branch_gap_multiset_error=branch_error,
    )


def checkpoint_local_ranks(
    checkpoint: Path,
) -> tuple[dict[int, list[tuple]], float, list[list[float]], dict[int, list[int]], bool]:
    records = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 240:
        raise RuntimeError(f"Expected 240 frozen checkpoint records, found {len(records)}")

    raw: dict[tuple[str, int, int], list[float]] = {}
    for record in records:
        for coordinate_text, value in record["delta_c"].items():
            key = (str(record["family"]), int(coordinate_text), int(record["batch"]))
            raw.setdefault(key, []).append(float(value))
    collapsed = {key: float(np.median(values)) for key, values in raw.items()}

    rankings: dict[int, list[tuple]] = {}
    frozen_prominences: list[list[float]] = []
    for center in FROZEN:
        window = [center] + controls(center)
        rows: list[tuple] = []
        center_prominence: list[float] = []
        for coordinate in window:
            prominences: list[float] = []
            for batch in range(BATCHES):
                robust = min(
                    collapsed["dephasing", coordinate, batch],
                    collapsed["transverse", coordinate, batch],
                )
                background = float(np.median([
                    min(
                        collapsed["dephasing", other, batch],
                        collapsed["transverse", other, batch],
                    )
                    for other in window
                    if other != coordinate
                ]))
                prominences.append(float(robust - background))
            rows.append((
                coordinate,
                sum(value > 0.0 for value in prominences),
                float(np.median(prominences)),
                float(min(prominences)),
                float(max(prominences)),
            ))
            if coordinate == center:
                center_prominence = prominences
        rankings[center] = sorted(rows, key=lambda row: (row[1], row[2]), reverse=True)
        frozen_prominences.append(center_prominence)

    correlation = float(np.corrcoef(frozen_prominences[0], frozen_prominences[1])[0, 1])
    eligible_counts: dict[int, list[int]] = {}
    eligible_sets: dict[int, list[set[int]]] = {}
    for center in FROZEN:
        per_batch_sets = [
            {
                int(record["seed"])
                for record in records
                if int(record["batch"]) == batch and str(center) in record["delta_c"]
            }
            for batch in range(BATCHES)
        ]
        eligible_sets[center] = per_batch_sets
        eligible_counts[center] = [len(values) for values in per_batch_sets]
    same_sets = all(
        eligible_sets[FROZEN[0]][batch] == eligible_sets[FROZEN[1]][batch]
        for batch in range(BATCHES)
    )
    return rankings, correlation, frozen_prominences, eligible_counts, same_sets


def binomial_tail(successes: int, trials: int) -> float:
    return sum(math.comb(trials, value) for value in range(successes, trials + 1)) / (2**trials)


def render(rows: list[AlgebraRow], checkpoint: Path) -> str:
    ranks = Counter((row.binary_rank, row.symplectic_rank, row.forced_multiplicity) for row in rows)
    rankings, correlation, frozen_prominences, eligible_counts, same_eligible_sets = (
        checkpoint_local_ranks(checkpoint)
    )
    max_control_gap = max(row.max_control_gap for row in rows)
    max_branch_error = max(row.branch_gap_multiset_error for row in rows)
    anticommuters = sum(row.common_anticommuter for row in rows)
    all_four_near = sum(all(gap < EPS_NEIGHBOR for gap in row.candidate_gaps) for row in rows)
    any_branch_a_near = sum(min(row.candidate_gaps[0], row.candidate_gaps[2]) < EPS_NEIGHBOR for row in rows)
    any_branch_b_near = sum(min(row.candidate_gaps[1], row.candidate_gaps[3]) < EPS_NEIGHBOR for row in rows)
    dense_check_error = regular_representation_implementation_error(
        [BASE_SEED + offset for offset in range(SEEDS_PER_BATCH)]
    )

    lines = [
        "=== Soft Spaces Phase 2 v25.41 SPARSE-PAULI SPECTRAL ARTIFACT AUDIT ===",
        "Frozen inputs: v25.31 12Q generator, 40 seeds, five Pauli terms, existing 240-record checkpoint",
        "No coordinate, score, threshold, seed, or checkpoint value was changed.",
        "",
        "A. Exact normalized-position identity",
        "  (2*k+1+1)/2^(q+1) = (k+1)/2^q.",
        "  Chain A is exactly the 1/4 source-spectrum position; chain B is exactly 3/4.",
        "  After the two-branch lift, the four tested target pairs lie at 1/8, 3/8, 5/8, 7/8.",
        "",
        "B. Exact sparse-Pauli algebra audit",
    ]
    for (binary_rank, commutator_rank, multiplicity), count in sorted(ranks.items()):
        lines.append(
            f"  s={binary_rank}, symplectic rank={commutator_rank}, "
            f"forced eigenvalue multiplicity={multiplicity}: {count}/40 seeds"
        )
    lines.extend([
        f"  Common Pauli anticommuter (hence exact E <-> -E spectral symmetry): {anticommuters}/40",
        f"  Regular spectrum repeat factor 2^(12-5): {1 << (12-5)}",
        f"  Independent dense-8Q implementation-check maximum spectral error: {dense_check_error:.6e}",
        f"  Maximum gap among all frozen local-control pairs: {max_control_gap:.6e}",
        f"  Maximum difference between A/B unordered branch-gap pairs: {max_branch_error:.6e}",
        f"  Seeds with all four boundary gaps below {EPS_NEIGHBOR}: {all_four_near}/40",
        f"  Seeds with at least one eligible A branch: {any_branch_a_near}/40",
        f"  Seeds with at least one eligible B branch: {any_branch_b_near}/40",
        "",
        "  Dimension geometry with the frozen +/-10 step-2 controls:",
        "  target Q | forced regular repeat | controls forced inside a repeated eigenvalue / 10",
    ])
    for target_q in range(7, 13):
        repeat = 1 << (target_q - N_TERMS)
        offsets = tuple(range(-LOCAL_RADIUS, 0, LOCAL_STEP)) + tuple(
            range(LOCAL_STEP, LOCAL_RADIUS + 1, LOCAL_STEP)
        )
        inside = sum(offset % repeat != 0 for offset in offsets)
        lines.append(f"       {target_q:2d}Q | {repeat:21d} | {inside:2d}/10")
    lines.extend([
        "",
        "C. Existing-checkpoint local-label audit",
    ])
    for center, ordered in rankings.items():
        frozen_rank = 1 + [row[0] for row in ordered].index(center)
        frozen_row = next(row for row in ordered if row[0] == center)
        lines.append(
            f"  Frozen {center}: local rank {frozen_rank}/11; positive batches "
            f"{frozen_row[1]}/8; median prominence {frozen_row[2]:+.9f}"
        )
        lines.append(
            "    best control: "
            f"k={next(row for row in ordered if row[0] != center)[0]}, "
            f"positive batches={next(row for row in ordered if row[0] != center)[1]}/8, "
            f"median prominence={next(row for row in ordered if row[0] != center)[2]:+.9f}"
        )
    sign_patterns = [
        tuple(prominences[batch] > 0.0 for prominences in frozen_prominences)
        for batch in range(BATCHES)
    ]
    both_same = sum(pattern[0] == pattern[1] for pattern in sign_patterns)
    lines.extend([
        f"  Correlation of the two frozen batch-prominence vectors: {correlation:.9f}",
        f"  Batches with identical A/B prominence sign: {both_same}/8",
        f"  Eligible seed counts per batch, k=511: {eligible_counts[511]}",
        f"  Eligible seed counts per batch, k=1535: {eligible_counts[1535]}",
        f"  Exact same eligible seed sets for both candidates: {same_eligible_sets}",
        f"  One-candidate sign-test tail P[X>=6 | p=0.5]: {binomial_tail(6, 8):.9f}",
        "  The two 6/8 confirmations must not be treated as independent: both succeed in the same six batches",
        "  and fail in the same two batches.",
        "",
        "D. Mathematical reading",
        "  1. k->2k+1 exactly preserves normalized spectral rank; it is not evidence of state transport.",
        "  2. Five Pauli generators leave an exponentially growing spectator multiplicity as q increases.",
        "  3. The frozen candidates are exactly sparse-algebra block boundaries, while every +/-10 step-2",
        "     control pair lies inside a repeated eigenvalue block and therefore has exactly zero gap.",
        "  4. The doubled multiplicity forces the same boundary coordinates to obey k->2k+1.",
        "  5. Exact spectral reflection makes the two candidate branch-gap multisets identical per seed,",
        "     explaining their strongly coupled batch behaviour.",
        "  6. With the fixed +/-10 window, the forced flat-control fraction rises from 6/10 at 7Q",
        "     to 8/10 at 8Q and 10/10 from 9Q onward.  This supplies a direct structural explanation",
        "     for the apparent strengthening of the recurrence in the high-Q regime.",
        "",
        "Conclusion",
        "  The high-Q coordinate recurrence is mathematically explained by fixed normalized spectral positions",
        "  plus the degeneracy structure of a five-term Pauli algebra.  In this protocol it is therefore a",
        "  structural consequence of the sparse ensemble and candidate/control geometry, not evidence that a",
        "  physical hotspot has been transported between independently generated Hilbert spaces.",
        "",
        "Remaining empirical content",
        "  The positive REAL-NULL prominence at those algebraic boundaries is a real recorded statistic, and",
        "  each frozen boundary ranks first among its 11 stored local labels.  Its 6/8 sign evidence is modest",
        "  on its own (one-candidate sign-test tail 0.1445), and the two candidates are highly dependent.",
        "  A genuinely physical Phase-2 claim now requires a redesigned ensemble whose number/locality of",
        "  Hamiltonian terms scales with q, and controls matched to the same algebraic boundary class.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("v25_31_12q_checkpoint.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v25_41_sparse_pauli_spectral_artifact_audit_output.txt"),
    )
    args = parser.parse_args()

    seeds = [
        BASE_SEED + batch * BATCH_STRIDE + offset
        for batch in range(BATCHES)
        for offset in range(SEEDS_PER_BATCH)
    ]
    rows = [algebra_row(seed) for seed in seeds]
    report = render(rows, args.checkpoint)
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")

    if max(row.max_control_gap for row in rows) > 1.0e-12:
        raise SystemExit("Control pairs were not inside exact repeated spectral blocks")
    if max(row.branch_gap_multiset_error for row in rows) > 1.0e-10:
        raise SystemExit("Expected reflected A/B branch-gap identity failed")


if __name__ == "__main__":
    main()
