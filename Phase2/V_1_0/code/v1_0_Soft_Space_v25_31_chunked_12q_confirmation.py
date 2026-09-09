#!/usr/bin/env python3
"""Soft Spaces Phase 2 v25.31 — checkpointed 12Q confirmation.

Frozen before execution
-----------------------
Rule:                 k_next = 2*k + 1
11Q source hotspots:  511, 1535
12Q lifted pairs:     (511,512), (2559,2560),
                      (1535,1536), (3583,3584)
Local controls:       source coordinate +/-10, step 2
Perturbation families: dephasing Z/ZZ and transverse X/XX
Primary score:        prominence of min(DeltaC_dephasing, DeltaC_transverse)
Decision gate:        positive prominence rate >= 0.75

The program is mathematically equivalent to the target-specific v25.30 path,
but never builds the full B = V^dagger dH V matrix.  It evaluates only the
columns required by the frozen hotspots and controls.  Every family/rep result
is appended to a JSONL checkpoint, so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PAULIS = ("I", "X", "Y", "Z")
FAMILIES = ("dephasing", "transverse")


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def index_to_label(index: int, n_qubits: int) -> str:
    value = int(index) + 1
    digits: list[str] = []
    for _ in range(n_qubits):
        value, remainder = divmod(value, 4)
        digits.append(PAULIS[remainder])
    return "".join(reversed(digits))


def pauli_permutation_phase(label: str) -> tuple[np.ndarray, np.ndarray]:
    """Return row permutation and column phase for P|j>."""
    n_qubits = len(label)
    dim = 1 << n_qubits
    basis = np.arange(dim, dtype=np.int64)
    flip_mask = 0
    phase = np.ones(dim, dtype=np.complex128)
    for position, symbol in enumerate(label):
        bit_number = n_qubits - 1 - position
        bit = (basis >> bit_number) & 1
        if symbol in ("X", "Y"):
            flip_mask |= 1 << bit_number
        if symbol == "Z":
            phase *= 1.0 - 2.0 * bit
        elif symbol == "Y":
            phase *= 1j * (1.0 - 2.0 * bit)
    return basis ^ flip_mask, phase


def dense_pauli_sum(n_qubits: int, terms: Iterable[tuple[str, float]]) -> np.ndarray:
    dim = 1 << n_qubits
    columns = np.arange(dim, dtype=np.int64)
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    for label, coefficient in terms:
        rows, phase = pauli_permutation_phase(label)
        matrix[rows, columns] += float(coefficient) * phase
    return matrix


def apply_pauli_sum(columns: np.ndarray, terms: Iterable[tuple[str, float]]) -> np.ndarray:
    result = np.zeros_like(columns, dtype=np.complex128)
    for label, coefficient in terms:
        rows, phase = pauli_permutation_phase(label)
        result[rows, :] += float(coefficient) * phase[:, None] * columns
    return result


def random_hamiltonian_terms(n_qubits: int, n_terms: int, seed: int) -> list[tuple[str, float]]:
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, 4**n_qubits - 1, size=int(n_terms))
    coefficients = rng.uniform(-1.0, 1.0, size=int(n_terms))
    return [
        (index_to_label(int(index), n_qubits), float(coefficient))
        for index, coefficient in zip(indices, coefficients)
    ]


def family_terms(n_qubits: int, n_terms: int, seed: int, family: str) -> list[tuple[str, float]]:
    rng = np.random.default_rng(int(seed))
    symbol = "Z" if family == "dephasing" else "X"
    labels: list[str] = []
    for qubit in range(n_qubits):
        label = ["I"] * n_qubits
        label[qubit] = symbol
        labels.append("".join(label))
    for qubit in range(n_qubits - 1):
        label = ["I"] * n_qubits
        label[qubit] = symbol
        label[qubit + 1] = symbol
        labels.append("".join(label))
    selected = rng.choice(len(labels), size=int(n_terms), replace=int(n_terms) > len(labels))
    coefficients = rng.uniform(-1.0, 1.0, size=int(n_terms))
    return [(labels[int(index)], float(coefficient)) for index, coefficient in zip(selected, coefficients)]


def haar_unitary(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    diagonal = np.diag(r)
    phases = diagonal / np.abs(diagonal)
    q *= phases
    return q


def finite_median(values: Iterable[float]) -> float:
    array = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    return float(np.median(array)) if array.size else float("nan")


def positive_rate(values: Iterable[float]) -> float:
    array = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    return float(np.mean(array > 0.0)) if array.size else float("nan")


def source_controls(source_k: int, frozen: set[int], source_qubits: int, radius: int, step: int) -> list[int]:
    maximum = (1 << source_qubits) - 2
    low = max(0, int(source_k) - int(radius))
    high = min(maximum, int(source_k) + int(radius))
    return [
        coordinate
        for coordinate in range(low, high + 1, int(step))
        if coordinate != int(source_k) and coordinate not in frozen
    ]


def cancel_from_selected_columns(
    eigenvalues: np.ndarray,
    b_columns: np.ndarray,
    left_index: int,
    column_positions: dict[int, int],
    eps_neighbor: float,
    energy_reg: float,
) -> float | None:
    right_index = int(left_index) + 1
    if abs(float(eigenvalues[right_index] - eigenvalues[left_index])) >= float(eps_neighbor):
        return None

    pair_columns = b_columns[:, [column_positions[left_index], column_positions[right_index]]]
    q_mask = np.ones(eigenvalues.size, dtype=bool)
    q_mask[[left_index, right_index]] = False
    wq = pair_columns[q_mask, :]
    eq = eigenvalues[q_mask]
    reference_energy = 0.5 * (float(eigenvalues[left_index]) + float(eigenvalues[right_index]))
    delta_energy = reference_energy - eq
    regularizer = max(float(energy_reg), 1e-15)
    inverse = delta_energy / (delta_energy * delta_energy + regularizer * regularizer)

    # For row vector w=(a,b), the v25.30 contribution is
    # inv * [[|a|^2, conj(a)b], [conj(b)a, |b|^2]].
    a = wq[:, 0]
    b = wq[:, 1]
    total00 = np.sum(inverse * np.abs(a) ** 2)
    total11 = np.sum(inverse * np.abs(b) ** 2)
    total01 = np.sum(inverse * np.conj(a) * b)
    total = np.array([[total00, total01], [np.conj(total01), total11]], dtype=np.complex128)

    row_norms = np.sqrt(np.abs(a) ** 4 + np.abs(b) ** 4 + 2.0 * np.abs(np.conj(a) * b) ** 2)
    norm_sum = float(np.sum(np.abs(inverse) * row_norms))
    if norm_sum <= 0.0 or not np.isfinite(norm_sum):
        return None
    return float(np.linalg.norm(total, ord="fro") / norm_sum)


def model_cancel_values(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    perturbation_terms: list[tuple[str, float]],
    all_coordinates: list[int],
    source_dim: int,
    selected_indices: list[int],
    column_positions: dict[int, int],
    eps_neighbor: float,
    energy_reg: float,
) -> dict[int, float]:
    selected_vectors = eigenvectors[:, selected_indices]
    acted = apply_pauli_sum(selected_vectors, perturbation_terms)
    b_columns = eigenvectors.conj().T @ acted
    output: dict[int, float] = {}
    for source_k in all_coordinates:
        branch_values: list[float] = []
        for branch in range(2):
            left_index = int(source_k) + branch * source_dim
            value = cancel_from_selected_columns(
                eigenvalues,
                b_columns,
                left_index,
                column_positions,
                eps_neighbor,
                energy_reg,
            )
            if value is not None:
                branch_values.append(float(value))
        if branch_values:
            output[int(source_k)] = finite_median(branch_values)
    return output


@dataclass(frozen=True)
class Summary:
    source_k: int
    batches: int
    positive_batches: int
    positive_rate: float
    median_robust_delta_c: float
    median_local_background: float
    median_prominence: float
    min_prominence: float
    max_prominence: float
    dephasing_positive_rate: float
    transverse_positive_rate: float
    status: str


def load_checkpoint(path: Path) -> tuple[list[dict], set[tuple[int, int, str, int]]]:
    records: list[dict] = []
    completed: set[tuple[int, int, str, int]] = set()
    if not path.exists():
        return records, completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (int(record["batch"]), int(record["seed"]), str(record["family"]), int(record["rep"]))
        records.append(record)
        completed.add(key)
    return records, completed


def append_checkpoint(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def aggregate(
    records: list[dict],
    frozen: list[int],
    control_map: dict[int, list[int]],
    batches: int,
    threshold: float,
) -> list[Summary]:
    raw: dict[tuple[str, int, int], list[float]] = {}
    for record in records:
        family = str(record["family"])
        batch = int(record["batch"])
        for coordinate_text, value in record["delta_c"].items():
            raw.setdefault((family, int(coordinate_text), batch), []).append(float(value))
    collapsed = {key: finite_median(values) for key, values in raw.items()}

    summaries: list[Summary] = []
    for source_k in frozen:
        prominence: list[float] = []
        robust_values: list[float] = []
        backgrounds: list[float] = []
        dephasing_values: list[float] = []
        transverse_values: list[float] = []
        for batch in range(int(batches)):
            dephasing = collapsed.get(("dephasing", source_k, batch), float("nan"))
            transverse = collapsed.get(("transverse", source_k, batch), float("nan"))
            if not (np.isfinite(dephasing) and np.isfinite(transverse)):
                continue
            robust = min(float(dephasing), float(transverse))
            neighbour_scores: list[float] = []
            for control in control_map[source_k]:
                cd = collapsed.get(("dephasing", control, batch), float("nan"))
                ct = collapsed.get(("transverse", control, batch), float("nan"))
                if np.isfinite(cd) and np.isfinite(ct):
                    neighbour_scores.append(min(float(cd), float(ct)))
            background = finite_median(neighbour_scores)
            if not np.isfinite(background):
                continue
            dephasing_values.append(float(dephasing))
            transverse_values.append(float(transverse))
            robust_values.append(robust)
            backgrounds.append(background)
            prominence.append(robust - background)
        rate = positive_rate(prominence)
        summaries.append(Summary(
            source_k=source_k,
            batches=len(prominence),
            positive_batches=sum(value > 0.0 for value in prominence),
            positive_rate=rate,
            median_robust_delta_c=finite_median(robust_values),
            median_local_background=finite_median(backgrounds),
            median_prominence=finite_median(prominence),
            min_prominence=float(np.min(prominence)) if prominence else float("nan"),
            max_prominence=float(np.max(prominence)) if prominence else float("nan"),
            dephasing_positive_rate=positive_rate(dephasing_values),
            transverse_positive_rate=positive_rate(transverse_values),
            status=("RUN_INCOMPLETE"
                    if len(prominence) < int(batches)
                    else ("DIMENSIONAL_RULE_CONFIRMED"
                          if np.isfinite(rate) and rate >= float(threshold)
                          else "DIMENSIONAL_RULE_NOT_CONFIRMED")),
        ))
    return summaries


def write_report(
    path: Path,
    args: argparse.Namespace,
    summaries: list[Summary],
    elapsed: float,
    checkpoint_records: int,
) -> None:
    expected_records = args.batches * args.seeds_per_batch * len(FAMILIES) * args.reps
    lines = [
        "=== Soft Spaces Phase 2 v25.31 CHUNKED PREDECLARED 12Q CONFIRMATION ===",
        "Frozen rule: k_next = 2*k + 1",
        f"Frozen 12Q source coordinates: {args.frozen_candidates}",
        "Lifted target pairs: A=(511,512)/(2559,2560); B=(1535,1536)/(3583,3584)",
        f"Independent batches: {args.batches} x {args.seeds_per_batch} seeds = {args.batches * args.seeds_per_batch}",
        f"Reps={args.reps} | families={list(FAMILIES)} | local radius=+/-{args.local_radius} | step={args.local_step}",
        f"Gate: prominence positive-rate >= {args.threshold:.2f}",
        f"Checkpoint records: {checkpoint_records}/{expected_records}",
        f"Elapsed seconds (current invocation): {elapsed:.1f}",
        "",
        "source k | status | batches | prom pos | prom+ | med robust | med bg | med prom | min | max | dephase+ | transverse+",
    ]
    for summary in summaries:
        lines.append(
            f"{summary.source_k:8d} | {summary.status:30s} | {summary.batches:7d} | "
            f"{summary.positive_batches:8d} | {summary.positive_rate:5.3f} | "
            f"{summary.median_robust_delta_c:+10.6f} | {summary.median_local_background:+8.6f} | "
            f"{summary.median_prominence:+9.6f} | {summary.min_prominence:+8.6f} | "
            f"{summary.max_prominence:+8.6f} | {summary.dephasing_positive_rate:8.3f} | "
            f"{summary.transverse_positive_rate:11.3f}"
        )
    confirmed = sum(summary.status == "DIMENSIONAL_RULE_CONFIRMED" for summary in summaries)
    lines.extend(["", f"Decision summary: {confirmed}/{len(summaries)} frozen 12Q predictions confirmed.", ""])
    report = "\n".join(lines)
    print(report)
    path.write_text(report, encoding="utf-8")


def parse_candidates(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpointed v25.31 12Q confirmation")
    parser.add_argument("--n-qubits", type=int, default=12)
    parser.add_argument("--source-qubits", type=int, default=11)
    parser.add_argument("--frozen", type=str, default="511,1535")
    parser.add_argument("--n-terms", type=int, default=5)
    parser.add_argument("--perturb-terms", type=int, default=5)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-start", type=int, default=0,
                        help="First global batch index handled by this worker.")
    parser.add_argument("--batch-stop", type=int, default=None,
                        help="Exclusive global batch index; default = --batches.")
    parser.add_argument("--seeds-per-batch", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=50_000_000)
    parser.add_argument("--batch-stride", type=int, default=1_000_000)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--eps-neighbor", type=float, default=0.05)
    parser.add_argument("--energy-reg", type=float, default=1e-3)
    parser.add_argument("--local-radius", type=int, default=10)
    parser.add_argument("--local-step", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--checkpoint", type=Path, default=Path("v25_31_12q_checkpoint.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("v25_31_12q_confirmation_output.txt"))
    parser.add_argument("--max-new-seeds", type=int, default=None,
                        help="Smoke/resume control; omit for the complete frozen run.")
    args = parser.parse_args()
    args.frozen_candidates = parse_candidates(args.frozen)
    if args.batch_stop is None:
        args.batch_stop = args.batches

    if args.n_qubits != 12 or args.source_qubits != 11:
        raise ValueError("v25.31 is frozen for 11Q->12Q only")
    if args.frozen_candidates != [511, 1535]:
        raise ValueError("Frozen 12Q candidates must remain 511,1535")
    if args.batches < 3:
        raise ValueError("At least three independent batches are required")
    if not (0 <= args.batch_start <= args.batch_stop <= args.batches):
        raise ValueError("Require 0 <= batch-start <= batch-stop <= batches")

    frozen_set = set(args.frozen_candidates)
    control_map = {
        source_k: source_controls(
            source_k, frozen_set, args.source_qubits, args.local_radius, args.local_step
        )
        for source_k in args.frozen_candidates
    }
    all_coordinates: list[int] = []
    for source_k in args.frozen_candidates:
        for coordinate in [source_k] + control_map[source_k]:
            if coordinate not in all_coordinates:
                all_coordinates.append(coordinate)

    source_dim = 1 << args.source_qubits
    selected_indices = sorted({
        index
        for coordinate in all_coordinates
        for branch in range(2)
        for index in (coordinate + branch * source_dim, coordinate + branch * source_dim + 1)
    })
    column_positions = {index: position for position, index in enumerate(selected_indices)}

    records, completed = load_checkpoint(args.checkpoint)
    start_time = time.perf_counter()
    new_seed_count = 0

    for batch in range(args.batch_start, args.batch_stop):
        for seed_offset in range(args.seeds_per_batch):
            seed = args.base_seed + batch * args.batch_stride + seed_offset
            seed_keys = {
                (batch, seed, family, rep)
                for family in FAMILIES
                for rep in range(args.reps)
            }
            if seed_keys.issubset(completed):
                continue
            if args.max_new_seeds is not None and new_seed_count >= args.max_new_seeds:
                break

            seed_start = time.perf_counter()
            h_terms = random_hamiltonian_terms(args.n_qubits, args.n_terms, seed)
            hamiltonian = dense_pauli_sum(args.n_qubits, h_terms)
            eigenvalues, real_vectors = np.linalg.eigh(hamiltonian)
            del hamiltonian
            null_vectors = haar_unitary(
                eigenvalues.size,
                stable_hash_int(f"NULL_HAAR_BASIS_V25_27|{seed}"),
            )

            for family in FAMILIES:
                for rep in range(args.reps):
                    key = (batch, seed, family, rep)
                    if key in completed:
                        continue
                    perturb_seed = stable_hash_int(
                        f"V25.31|{family.upper()}|PERT|{seed}|rep{rep}"
                    )
                    perturbation_terms = family_terms(
                        args.n_qubits, args.perturb_terms, perturb_seed, family
                    )
                    real_cancel = model_cancel_values(
                        eigenvalues, real_vectors, perturbation_terms,
                        all_coordinates, source_dim, selected_indices, column_positions,
                        args.eps_neighbor, args.energy_reg,
                    )
                    null_cancel = model_cancel_values(
                        eigenvalues, null_vectors, perturbation_terms,
                        all_coordinates, source_dim, selected_indices, column_positions,
                        args.eps_neighbor, args.energy_reg,
                    )
                    delta_c = {
                        str(coordinate): float(real_cancel[coordinate] - null_cancel[coordinate])
                        for coordinate in all_coordinates
                        if coordinate in real_cancel and coordinate in null_cancel
                    }
                    record = {
                        "version": "v25.31",
                        "batch": batch,
                        "seed": seed,
                        "family": family,
                        "rep": rep,
                        "delta_c": delta_c,
                    }
                    append_checkpoint(args.checkpoint, record)
                    records.append(record)
                    completed.add(key)

            new_seed_count += 1
            elapsed_seed = time.perf_counter() - seed_start
            print(
                f"Completed seed {seed} (batch {batch + 1}/{args.batches}, "
                f"seed {seed_offset + 1}/{args.seeds_per_batch}) in {elapsed_seed:.1f}s",
                flush=True,
            )
            del real_vectors, null_vectors, eigenvalues

        if args.max_new_seeds is not None and new_seed_count >= args.max_new_seeds:
            break

    summaries = aggregate(records, args.frozen_candidates, control_map, args.batches, args.threshold)
    write_report(
        args.output,
        args,
        summaries,
        time.perf_counter() - start_time,
        len(records),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted safely; resume with the same command.", file=sys.stderr)
        raise
