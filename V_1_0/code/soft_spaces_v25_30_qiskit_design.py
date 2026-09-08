"""Soft Spaces Phase 2.2 — Qiskit design aligned with v25.30.

This is a small, inspectable reproduction of the frozen v25.30 high-Q test.
It preserves:

* harmonized 6Q->7Q, 7Q->8Q and 8Q->9Q source/target geometry;
* frozen chains A=(15,31,63) and B=(47,95,191);
* two target branches i=k and i=k+2**source_qubits;
* spectrum-matched Haar NULL;
* dephasing Z/ZZ and transverse X/XX perturbation families;
* local source-space controls, radius +/-10 and step 2;
* C_matrix cancellation, robust score and local prominence;
* the predeclared 0.75 positive-batch-rate gate.

The full C_matrix scan is evaluated exactly, as in v25.30. Qiskit Statevector
then independently audits selected transition amplitudes B[q,p]=<q|dH|p>.
Finally, genuine Hadamard-test circuits are exported to show how one complex
Pauli transition amplitude can be measured as Re and Im on a quantum backend.

Default execution is deliberately a 7Q smoke test (1 batch, 2 seeds, 1 rep).
Increase --batches, --seeds-per-batch and --reps only after the smoke test.

Install:
    python -m pip install qiskit qiskit-aer matplotlib

Run:
    python -X utf8 -u soft_spaces_v25_30_qiskit_design.py
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector


PAULIS = ("I", "X", "Y", "Z")
PAULI_MATS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

CHAIN_A = {7: 15, 8: 31, 9: 63}
CHAIN_B = {7: 47, 8: 95, 9: 191}


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def index_to_pauli_label(index: int, n_qubits: int) -> str:
    value = int(index) + 1  # exclude the all-I word exactly as in v25.30
    digits = []
    for _ in range(n_qubits):
        value, remainder = divmod(value, 4)
        digits.append(PAULIS[remainder])
    return "".join(reversed(digits))


def kron_label(label: str) -> np.ndarray:
    result = PAULI_MATS[label[0]]
    for symbol in label[1:]:
        result = np.kron(result, PAULI_MATS[symbol])
    return result


def random_pauli_hamiltonian(n_qubits: int, n_terms: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    dim = 2**n_qubits
    indices = rng.integers(0, 4**n_qubits - 1, size=n_terms)
    coefficients = rng.uniform(-1.0, 1.0, size=n_terms)
    matrix = np.zeros((dim, dim), dtype=complex)
    for index, coefficient in zip(indices, coefficients):
        matrix += float(coefficient) * kron_label(index_to_pauli_label(int(index), n_qubits))
    return matrix


def haar_unitary(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    diagonal = np.diag(r)
    phases = diagonal / np.abs(diagonal)
    return q * phases


def family_perturbation(
    n_qubits: int,
    n_terms: int,
    seed: int,
    family: str,
) -> tuple[np.ndarray, list[tuple[str, float]]]:
    rng = np.random.default_rng(int(seed))
    symbol = "Z" if family == "dephasing" else "X"
    labels: list[str] = []
    for qubit in range(n_qubits):
        word = ["I"] * n_qubits
        word[qubit] = symbol
        labels.append("".join(word))
    for qubit in range(n_qubits - 1):
        word = ["I"] * n_qubits
        word[qubit] = symbol
        word[qubit + 1] = symbol
        labels.append("".join(word))

    chosen = rng.choice(len(labels), size=n_terms, replace=n_terms > len(labels))
    coefficients = rng.uniform(-1.0, 1.0, size=n_terms)
    terms = [(labels[int(index)], float(coefficient)) for index, coefficient in zip(chosen, coefficients)]
    matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for label, coefficient in terms:
        matrix += coefficient * kron_label(label)
    return matrix, terms


def cancel_matrix(
    eigenvalues: np.ndarray,
    transformed_perturbation: np.ndarray,
    left_index: int,
    eps_neighbor: float,
    energy_regularizer: float,
) -> float | None:
    """Exact v25.30 C_matrix definition."""
    dim = eigenvalues.size
    right_index = left_index + 1
    if left_index < 0 or right_index >= dim:
        return None
    if abs(float(eigenvalues[right_index] - eigenvalues[left_index])) >= eps_neighbor:
        return None

    q_mask = np.ones(dim, dtype=bool)
    q_mask[[left_index, right_index]] = False
    wq = transformed_perturbation[q_mask, :][:, [left_index, right_index]]
    eq = eigenvalues[q_mask]
    reference_energy = 0.5 * (eigenvalues[left_index] + eigenvalues[right_index])
    delta_energy = reference_energy - eq
    regularizer = max(float(energy_regularizer), 1e-15)
    inverse = delta_energy / (delta_energy**2 + regularizer**2)

    total = np.zeros((2, 2), dtype=complex)
    norm_sum = 0.0
    for row in range(wq.shape[0]):
        vector = np.asarray(wq[row, :], dtype=complex).reshape(2, 1)
        contribution = float(inverse[row]) * (vector.conj() @ vector.T)
        contribution = 0.5 * (contribution + contribution.conj().T)
        total += contribution
        norm_sum += float(np.linalg.norm(contribution, ord="fro"))
    if norm_sum <= 0 or not np.isfinite(norm_sum):
        return None
    return float(np.linalg.norm(total, ord="fro") / norm_sum)


def finite_median(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def local_controls(source_k: int, frozen: set[int], source_qubits: int, radius: int, step: int) -> list[int]:
    maximum = 2**source_qubits - 2
    low = max(0, source_k - radius)
    high = min(maximum, source_k + radius)
    start = low
    if (start - source_k) % step != 0:
        start += step - ((start - source_k) % step)
    return [coordinate for coordinate in range(start, high + 1, step)
            if coordinate != source_k and coordinate not in frozen]


def qiskit_transition_amplitude(
    eigenvectors: np.ndarray,
    perturbation: np.ndarray,
    q_index: int,
    p_index: int,
) -> complex:
    """Independent Qiskit Statevector calculation of <q|dH|p>."""
    ket_p = Statevector(eigenvectors[:, p_index])
    evolved = ket_p.evolve(Operator(perturbation))
    ket_q = Statevector(eigenvectors[:, q_index])
    return complex(np.vdot(ket_q.data, evolved.data))


def preparation_unitary(eigenvectors: np.ndarray, column: int) -> np.ndarray:
    """Return a unitary whose first column is the requested eigenvector."""
    order = list(range(eigenvectors.shape[1]))
    order[0], order[column] = order[column], order[0]
    return eigenvectors[:, order]


def hadamard_transition_circuits(
    eigenvectors: np.ndarray,
    pauli_label: str,
    q_index: int,
    p_index: int,
    name: str,
) -> tuple[QuantumCircuit, QuantumCircuit]:
    """Circuits for Re and Im of <q|Pauli|p> using an ancilla Hadamard test."""
    n_qubits = int(np.log2(eigenvectors.shape[0]))
    up = preparation_unitary(eigenvectors, p_index)
    uq = preparation_unitary(eigenvectors, q_index)
    transition = uq.conj().T @ kron_label(pauli_label) @ up
    controlled = UnitaryGate(transition, label=f"Uq† {pauli_label} Up").control(1)

    real = QuantumCircuit(1 + n_qubits, 1, name=f"{name}_real")
    real.h(0)
    real.append(controlled, range(1 + n_qubits))
    real.h(0)
    real.measure(0, 0)

    imaginary = QuantumCircuit(1 + n_qubits, 1, name=f"{name}_imag")
    imaginary.h(0)
    imaginary.append(controlled, range(1 + n_qubits))
    imaginary.sdg(0)
    imaginary.h(0)
    imaginary.measure(0, 0)
    return real, imaginary


@dataclass
class ResultRow:
    branch: str
    source_k: int
    units: int
    positive: int
    positive_rate: float
    median_robust: float
    median_background: float
    median_prominence: float
    status: str


def run(args: argparse.Namespace) -> tuple[list[ResultRow], float, tuple[QuantumCircuit, QuantumCircuit]]:
    target_q = args.target_qubits
    source_q = target_q - 1
    source_dim = 2**source_q
    hotspots = [("A", CHAIN_A[target_q]), ("B", CHAIN_B[target_q])]
    frozen = {coordinate for _, coordinate in hotspots}
    controls = {
        coordinate: local_controls(coordinate, frozen, source_q, args.local_radius, args.local_step)
        for _, coordinate in hotspots
    }
    all_coordinates = sorted(frozen | {item for values in controls.values() for item in values})

    raw: dict[tuple[str, int, int], list[float]] = {}
    max_qiskit_error = 0.0
    representative: tuple[QuantumCircuit, QuantumCircuit] | None = None

    for batch in range(args.batches):
        # v25.30 separates target dimensions by 10,000,000-seed blocks:
        # 7Q -> base, 8Q -> base+10M, 9Q -> base+20M.
        dimension_seed_base = args.base_seed + (target_q - 7) * 10_000_000
        seed_start = dimension_seed_base + batch * args.batch_stride
        for offset in range(args.seeds_per_batch):
            seed = seed_start + offset
            h_real = random_pauli_hamiltonian(target_q, args.n_terms, seed)
            eigenvalues, real_vectors = np.linalg.eigh(h_real)
            null_vectors = haar_unitary(
                2**target_q,
                stable_hash_int(f"NULL_HAAR_BASIS_V25_29_1_{source_q}Q_TO_{target_q}Q|{seed}"),
            )

            for family in ("dephasing", "transverse"):
                for rep in range(args.reps):
                    perturb_seed = stable_hash_int(
                        f"V25.30|{source_q}Q->{target_q}Q|{family.upper()}|PERT|{seed}|rep{rep}"
                    )
                    delta_h, terms = family_perturbation(
                        target_q, args.perturb_terms or args.n_terms, perturb_seed, family
                    )
                    b_real = real_vectors.conj().T @ delta_h @ real_vectors
                    b_null = null_vectors.conj().T @ delta_h @ null_vectors

                    for source_k in all_coordinates:
                        real_cancel: list[float] = []
                        null_cancel: list[float] = []
                        for branch_index in range(2):
                            left = source_k + branch_index * source_dim
                            cr = cancel_matrix(eigenvalues, b_real, left, args.eps_neighbor, args.energy_reg)
                            cn = cancel_matrix(eigenvalues, b_null, left, args.eps_neighbor, args.energy_reg)
                            if cr is not None and cn is not None:
                                real_cancel.append(cr)
                                null_cancel.append(cn)
                        if real_cancel and null_cancel:
                            delta_c = finite_median(real_cancel) - finite_median(null_cancel)
                            raw.setdefault((family, source_k, batch), []).append(delta_c)

                    # Qiskit audit of representative B[q,p] elements.
                    if offset == 0 and rep == 0:
                        left = hotspots[0][1]
                        q_indices = [index for index in range(min(4, len(eigenvalues))) if index not in (left, left + 1)]
                        for vectors, matrix in ((real_vectors, b_real), (null_vectors, b_null)):
                            for q_index in q_indices:
                                amplitude = qiskit_transition_amplitude(vectors, delta_h, q_index, left)
                                max_qiskit_error = max(max_qiskit_error, abs(amplitude - matrix[q_index, left]))

                        if representative is None:
                            label, _ = terms[0]
                            q_index = q_indices[0]
                            representative = hadamard_transition_circuits(
                                real_vectors, label, q_index, left,
                                f"v25_30_{target_q}q_{family}",
                            )

    collapsed = {key: finite_median(values) for key, values in raw.items()}
    rows: list[ResultRow] = []
    for branch, source_k in hotspots:
        prominence_values: list[float] = []
        robust_values: list[float] = []
        background_values: list[float] = []
        for batch in range(args.batches):
            dephasing = collapsed.get(("dephasing", source_k, batch), float("nan"))
            transverse = collapsed.get(("transverse", source_k, batch), float("nan"))
            if not (np.isfinite(dephasing) and np.isfinite(transverse)):
                continue
            robust = min(dephasing, transverse)
            neighbour_scores = []
            for control in controls[source_k]:
                control_dephasing = collapsed.get(("dephasing", control, batch), float("nan"))
                control_transverse = collapsed.get(("transverse", control, batch), float("nan"))
                if np.isfinite(control_dephasing) and np.isfinite(control_transverse):
                    neighbour_scores.append(min(control_dephasing, control_transverse))
            background = finite_median(neighbour_scores)
            if np.isfinite(background):
                robust_values.append(robust)
                background_values.append(background)
                prominence_values.append(robust - background)

        rate = (sum(value > 0 for value in prominence_values) / len(prominence_values)
                if prominence_values else float("nan"))
        rows.append(ResultRow(
            branch=branch,
            source_k=source_k,
            units=len(prominence_values),
            positive=sum(value > 0 for value in prominence_values),
            positive_rate=rate,
            median_robust=finite_median(robust_values),
            median_background=finite_median(background_values),
            median_prominence=finite_median(prominence_values),
            status="UPWARD_VALIDATED" if np.isfinite(rate) and rate >= args.threshold else "UPWARD_NOT_VALIDATED",
        ))

    if representative is None:
        raise RuntimeError("No representative Qiskit transition circuit was produced.")
    return rows, max_qiskit_error, representative


def main() -> None:
    parser = argparse.ArgumentParser(description="v25.30-aligned Soft Spaces Qiskit test")
    parser.add_argument("--target-qubits", type=int, choices=(7, 8, 9), default=7)
    parser.add_argument("--n-terms", type=int, default=5)
    parser.add_argument("--perturb-terms", type=int, default=None)
    parser.add_argument("--seeds-per-batch", type=int, default=2)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--batch-stride", type=int, default=1_000_000)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--eps-neighbor", type=float, default=0.05)
    parser.add_argument("--energy-reg", type=float, default=1e-3)
    parser.add_argument("--local-radius", type=int, default=10)
    parser.add_argument("--local-step", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    rows, qiskit_error, circuits = run(args)
    source_q = args.target_qubits - 1
    print(f"=== v25.30 Qiskit design: {source_q}Q -> {args.target_qubits}Q ===")
    print("branch | source k | units | positive | rate | median robust | median bg | median prominence | status")
    for row in rows:
        print(
            f"{row.branch:^6s} | {row.source_k:8d} | {row.units:5d} | {row.positive:8d} | "
            f"{row.positive_rate:4.2f} | {row.median_robust:+13.6f} | "
            f"{row.median_background:+9.6f} | {row.median_prominence:+17.6f} | {row.status}"
        )
    print(f"Qiskit transition-amplitude audit max error: {qiskit_error:.3e}")

    stem = Path(__file__).with_name(f"v25_30_qiskit_hadamard_{args.target_qubits}q")
    circuits[0].draw("mpl", filename=str(stem) + "_real.png", fold=40)
    circuits[1].draw("mpl", filename=str(stem) + "_imag.png", fold=40)
    print(f"Qiskit diagrams: {stem}_real.png and {stem}_imag.png")


if __name__ == "__main__":
    main()
