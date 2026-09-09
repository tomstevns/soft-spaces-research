"""Soft Spaces Phase 2.2 — minimal REAL/NULL Qiskit reproduction.

The program keeps the original test's essential physics:

1. REAL is a seeded sum of random Pauli words.
2. NULL has exactly the same eigenvalues, but a seeded Haar-random eigenbasis.
3. The same small Pauli perturbation is applied to REAL and NULL.
4. A near-neighbour two-dimensional eigenspace is selected.
5. Qiskit circuits measure the basis-invariant projector overlap
       Ov = 1/2 Tr(P P') = 1/2 ||U^dagger U'||_F^2.

This is intentionally a four-qubit validation core. Once it agrees with the
classical reference, N_QUBITS can be increased and the predeclared Phase 2
target windows can be added without changing the overlap circuit.

Install:
    python -m pip install qiskit qiskit-aer matplotlib

Run:
    python soft_spaces_qiskit_core.py
"""

from __future__ import annotations

import hashlib
from itertools import product
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


N_QUBITS = 4
N_TERMS = 5
EPS_NEIGHBOR = 0.05
ETA = 0.01
BASE_SEED = 2501000
SHOTS = 20_000

PAULIS = ("I", "X", "Y", "Z")
PAULI_MATS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def kron_word(label: str) -> np.ndarray:
    out = PAULI_MATS[label[0]]
    for symbol in label[1:]:
        out = np.kron(out, PAULI_MATS[symbol])
    return out


def random_pauli_hamiltonian(n_qubits: int, n_terms: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = ["".join(word) for word in product(PAULIS, repeat=n_qubits)]
    labels.remove("I" * n_qubits)
    chosen = rng.integers(0, len(labels), size=n_terms)
    coeffs = rng.uniform(-1.0, 1.0, size=n_terms)
    dim = 2**n_qubits
    hamiltonian = np.zeros((dim, dim), dtype=complex)
    for index, coeff in zip(chosen, coeffs):
        hamiltonian += float(coeff) * kron_word(labels[int(index)])
    return hamiltonian


def haar_unitary(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    diagonal = np.diag(r)
    phases = np.ones_like(diagonal)
    nonzero = np.abs(diagonal) > 0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    return q * phases


def neighbour_pair(eigenvalues: np.ndarray, epsilon: float) -> tuple[int, int, float]:
    gaps = np.diff(eigenvalues)
    eligible = np.flatnonzero(np.abs(gaps) < epsilon)
    if eligible.size == 0:
        raise RuntimeError("No neighbouring eigenpair below EPS_NEIGHBOR for this seed.")
    left = int(eligible[np.argmin(np.abs(gaps[eligible]))])
    return left, left + 1, float(abs(gaps[left]))


def subspace_overlap(u0: np.ndarray, u1: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(u0.conj().T @ u1) ** 2))


def best_matching_neighbour(eigenvectors: np.ndarray, reference: np.ndarray) -> tuple[int, int, float]:
    best = (-1, -1, -1.0)
    for left in range(eigenvectors.shape[1] - 1):
        candidate = eigenvectors[:, [left, left + 1]]
        overlap = subspace_overlap(reference, candidate)
        if overlap > best[2]:
            best = (left, left + 1, overlap)
    return best


def complete_unitary(first_columns: np.ndarray, seed: int) -> np.ndarray:
    """Complete two orthonormal columns to a full unitary without changing them."""
    dim = first_columns.shape[0]
    rng = np.random.default_rng(seed)
    columns = [first_columns[:, 0], first_columns[:, 1]]
    while len(columns) < dim:
        vector = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        for column in columns:
            vector -= column * np.vdot(column, vector)
        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            columns.append(vector / norm)
    return np.column_stack(columns)


def overlap_circuit(
    perturbed_vector: np.ndarray,
    reference_subspace: np.ndarray,
    label: str,
) -> QuantumCircuit:
    """Prepare |u'> and measure whether it lies in the baseline 2D subspace."""
    basis = complete_unitary(reference_subspace, stable_seed(f"COMPLETE|{label}"))
    circuit = QuantumCircuit(N_QUBITS, N_QUBITS, name=label)
    circuit.append(StatePreparation(perturbed_vector, label="Prepare |u'>"), range(N_QUBITS))
    circuit.append(UnitaryGate(basis.conj().T, label="U_P^dagger"), range(N_QUBITS))
    circuit.measure(range(N_QUBITS), range(N_QUBITS))
    return circuit


def qiskit_overlap(
    perturbed_subspace: np.ndarray,
    reference_subspace: np.ndarray,
    label: str,
    shots: int,
) -> tuple[float, list[QuantumCircuit]]:
    """Estimate 1/2 Tr(PP') from two Qiskit measurement circuits."""
    simulator = AerSimulator()
    circuits = [
        overlap_circuit(perturbed_subspace[:, column], reference_subspace, f"{label}_{column}")
        for column in range(2)
    ]
    compiled = transpile(circuits, simulator, optimization_level=1)
    result = simulator.run(compiled, shots=shots, seed_simulator=stable_seed(label)).result()

    accepted = {format(0, f"0{N_QUBITS}b"), format(1, f"0{N_QUBITS}b")}
    probabilities = []
    for index in range(2):
        counts = result.get_counts(index)
        hit_count = sum(counts.get(bitstring, 0) for bitstring in accepted)
        probabilities.append(hit_count / shots)
    return float(np.mean(probabilities)), circuits


def exact_qiskit_statevector_check(
    perturbed_subspace: np.ndarray,
    reference_subspace: np.ndarray,
    label: str,
) -> float:
    """Shot-free Qiskit check of the same two circuits before measurement."""
    accepted_indices = (0, 1)
    values = []
    for column in range(2):
        basis = complete_unitary(reference_subspace, stable_seed(f"COMPLETE|{label}_{column}"))
        circuit = QuantumCircuit(N_QUBITS)
        circuit.append(StatePreparation(perturbed_subspace[:, column]), range(N_QUBITS))
        circuit.append(UnitaryGate(basis.conj().T), range(N_QUBITS))
        probabilities = Statevector.from_instruction(circuit).probabilities()
        values.append(float(sum(probabilities[index] for index in accepted_indices)))
    return float(np.mean(values))


def main() -> None:
    h_real = random_pauli_hamiltonian(N_QUBITS, N_TERMS, BASE_SEED)
    evals, evecs_real = np.linalg.eigh(h_real)
    left, right, gap = neighbour_pair(evals, EPS_NEIGHBOR)

    u_haar = haar_unitary(2**N_QUBITS, stable_seed(f"NULL_HAAR_BASIS|{BASE_SEED}"))
    h_null = u_haar @ np.diag(evals) @ u_haar.conj().T

    perturb_seed = stable_seed(f"PERT|{BASE_SEED}|eta{ETA:.6g}|rep0")
    delta_h = random_pauli_hamiltonian(N_QUBITS, N_TERMS, perturb_seed)

    evals_real_p, evecs_real_p = np.linalg.eigh(h_real + ETA * delta_h)
    evals_null_p, evecs_null_p = np.linalg.eigh(h_null + ETA * delta_h)

    real_0 = evecs_real[:, [left, right]]
    null_0 = u_haar[:, [left, right]]
    real_i, real_j, real_exact = best_matching_neighbour(evecs_real_p, real_0)
    null_i, null_j, null_exact = best_matching_neighbour(evecs_null_p, null_0)
    real_1 = evecs_real_p[:, [real_i, real_j]]
    null_1 = evecs_null_p[:, [null_i, null_j]]

    real_sv = exact_qiskit_statevector_check(real_1, real_0, "REAL")
    null_sv = exact_qiskit_statevector_check(null_1, null_0, "NULL")
    real_shots, real_circuits = qiskit_overlap(real_1, real_0, "REAL", SHOTS)
    null_shots, _ = qiskit_overlap(null_1, null_0, "NULL", SHOTS)

    print("Soft Spaces Phase 2.2 — Qiskit core")
    print(f"qubits={N_QUBITS}, terms={N_TERMS}, seed={BASE_SEED}, eta={ETA}")
    print(f"baseline pair=({left},{right}), gap={gap:.8f}")
    print()
    print("                    REAL          NULL        DELTA")
    print(f"NumPy exact     {real_exact:10.6f}  {null_exact:10.6f}  {real_exact-null_exact:+10.6f}")
    print(f"Qiskit exact    {real_sv:10.6f}  {null_sv:10.6f}  {real_sv-null_sv:+10.6f}")
    print(f"Qiskit shots    {real_shots:10.6f}  {null_shots:10.6f}  {real_shots-null_shots:+10.6f}")

    # Save one representative, genuine Qiskit circuit diagram.
    output = Path(__file__).with_name("soft_spaces_qiskit_overlap_circuit.png")
    real_circuits[0].draw("mpl", filename=str(output), fold=-1)
    print(f"\nCircuit diagram saved as: {output}")


if __name__ == "__main__":
    main()
