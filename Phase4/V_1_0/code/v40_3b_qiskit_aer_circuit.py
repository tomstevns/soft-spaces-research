#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v40.3 — explicit Qiskit/Aer circuit-level ranked-search test.

Scientific role
---------------
v40.0: application screening                         PASS
v40.1: controlled score-noise robustness             PASS
v40.2: classical finite-shot/readout/depolarizing
       hardware proxy                                PASS
v40.3: explicit Qiskit/Aer gate-level realization    THIS FILE

This experiment preserves the frozen Phase-4 ranked-search endpoint:

    use -E source information to rank +E targets,
    then compare REAL with the matched NULL control.

The difference from v40.2 is that source information is now read through
an actual Qiskit circuit executed by AerSimulator under a Qiskit NoiseModel.

IMPORTANT SCIENTIFIC SCOPE
--------------------------
This file is an explicit gate-level *measurement-channel realization* of the
v40.0 ranked-search application.

It is NOT yet a gate decomposition of the complete frozen Soft-Spaces
Hamiltonian/projector calculation.  The classical Phase-3 kernel still
computes the source prominence.  Each source prominence is monotonically
encoded into a qubit rotation, passed through an explicitly transpiled
Qiskit circuit, sampled with finite shots under gate/readout noise, and then
used to rank candidates.

Therefore a PASS establishes:

    the REAL-specific ranking advantage survives an explicit noisy
    circuit/readout channel implemented in Qiskit/Aer.

It does NOT establish:

    * execution on IBM hardware,
    * a full Hamiltonian-native quantum implementation,
    * quantum computational advantage.

Current software target
-----------------------
Designed for current IBM/Qiskit generation:
    Python >= 3.10
    qiskit ~= 2.5
    qiskit-aer ~= 0.17

Frozen dependencies
-------------------
Place this file beside:

    v31_7_Soft_Spaces_independent_replication.py
    v40_0_application_screening.py

Circuit construction
--------------------
For each -E candidate score s:

1. Map s monotonically to probability p in [0.05, 0.95].
2. Prepare qubit 0 with RY(theta), where:
       p = sin^2(theta/2)
3. Add a logically identity two-qubit noise-loading network:
       CX(0,q); CX(0,q)
   repeated across spectator qubits.
   In the noiseless circuit this leaves the encoded state unchanged exactly.
4. Measure qubit 0.
5. Aer applies:
       * one-qubit depolarizing error
       * two-qubit depolarizing error
       * symmetric readout error
       * finite-shot sampling
6. The measured P(1) ranks the +E target candidates.

Two-qubit gate-count sweep
--------------------------
Default total CX counts per candidate circuit:
    0, 10, 20, 50, 100, 200, 400

Primary predeclared checkpoint:
    N_CX = 20

Default noise:
    p_2q = 1.17e-3
    p_1q = 1.17e-4
    readout = 0.01
    shots = 4096

Primary endpoint
----------------
Precision@8, REAL versus NULL.

Same four gates:
C1 mean delta Precision@8 > 0
C2 95% bootstrap CI lower bound > 0
C3 one-sided exact sign-test p < 0.05
C4 one-sided Wilcoxon p < 0.05
"""

from __future__ import annotations

import argparse
import importlib
import math
import platform
import sys
from pathlib import Path

import numpy as np


VERSION = "v40.3b"

BASE_MODULE = "v31_7_Soft_Spaces_independent_replication"
APP_MODULE = "v40_0_application_screening"

DEFAULT_K = 8
DEFAULT_REPS = 200
DEFAULT_SHOTS = 4096
DEFAULT_READOUT = 0.01
DEFAULT_P2Q = 1.17e-3
DEFAULT_P1Q = 1.17e-4
DEFAULT_OPT_LEVEL = 0

BOOTSTRAPS = 200_000
BOOTSTRAP_SEED = 40_300_001
SIM_SEED = 40_300_101

CX_GRID = (0, 10, 20, 50, 100, 200, 400)
PRIMARY_CX = 20


def require_modern_python():
    if sys.version_info < (3, 10):
        raise SystemExit(
            "\n"
            "v40.3 targets the current Qiskit 2.5.x generation, which requires "
            "Python >= 3.10.\n\n"
            f"Current interpreter: Python {platform.python_version()}\n\n"
            "Create/use a Python 3.10+ environment before running v40.3.\n"
        )


def load_qiskit():
    try:
        import qiskit
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import (
            NoiseModel,
            ReadoutError,
            depolarizing_error,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "\nQiskit/Aer is not installed in this Python environment.\n\n"
            "Recommended current packages:\n"
            "  python -m pip install \"qiskit~=2.5\" \"qiskit-aer~=0.17\"\n"
        ) from exc

    return {
        "qiskit": qiskit,
        "QuantumCircuit": QuantumCircuit,
        "transpile": transpile,
        "AerSimulator": AerSimulator,
        "NoiseModel": NoiseModel,
        "ReadoutError": ReadoutError,
        "depolarizing_error": depolarizing_error,
    }


def load_softspaces():
    try:
        base = importlib.import_module(BASE_MODULE)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Could not import {BASE_MODULE}.py\n"
            "Place v40_3_qiskit_aer_circuit.py in the same directory as "
            f"{BASE_MODULE}.py."
        ) from exc

    try:
        app0 = importlib.import_module(APP_MODULE)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Could not import {APP_MODULE}.py\n"
            "Place v40_3_qiskit_aer_circuit.py in the same directory as "
            f"{APP_MODULE}.py."
        ) from exc

    return base, app0


def extract_prominence_and_pairs(app0, base, seed: int, model: str):
    result = app0.build_prominence(base, seed, model)

    # Support both v40.0 variants used during Phase 4:
    #   (prominence, pairs)
    #   (energies, prominence, pairs)
    if len(result) == 2:
        prominence, pairs = result
    elif len(result) == 3:
        _, prominence, pairs = result
    else:
        raise ValueError(
            "Unexpected return shape from "
            "v40_0_application_screening.build_prominence(): "
            f"{len(result)} values"
        )

    return prominence, pairs


def build_frozen_seed_data(app0, base, seeds):
    cache = {}

    for seed in seeds:
        cache[seed] = {}

        for model in ("REAL", "NULL"):
            prominence, pairs = extract_prominence_and_pairs(
                app0, base, seed, model
            )
            cache[seed][model] = (prominence, pairs)

    return cache


def monotone_probability_map(source: np.ndarray) -> np.ndarray:
    """
    Strictly monotone affine map into [0.05, 0.95].

    It preserves the source ranking in the ideal/noiseless limit while leaving
    headroom for finite-shot and noisy-circuit degradation.
    """
    lo = float(np.min(source))
    hi = float(np.max(source))

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite source prominence encountered.")

    if hi <= lo:
        return np.full_like(source, 0.5, dtype=float)

    z = (source - lo) / (hi - lo)
    return 0.05 + 0.90 * z


def theta_from_probability(p: float) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    return 2.0 * math.asin(math.sqrt(p))


def build_noise_model(qk, p1q: float, p2q: float, readout_error: float):
    NoiseModel = qk["NoiseModel"]
    ReadoutError = qk["ReadoutError"]
    depolarizing_error = qk["depolarizing_error"]

    noise_model = NoiseModel()

    if p1q > 0.0:
        oneq_error = depolarizing_error(p1q, 1)
        # We intentionally transpile into these one-qubit basis gates.
        noise_model.add_all_qubit_quantum_error(
            oneq_error,
            ["x", "sx"],
        )

    if p2q > 0.0:
        twoq_error = depolarizing_error(p2q, 2)
        noise_model.add_all_qubit_quantum_error(
            twoq_error,
            ["cx"],
        )

    if readout_error > 0.0:
        r = float(readout_error)
        ro_error = ReadoutError(
            [
                [1.0 - r, r],
                [r, 1.0 - r],
            ]
        )
        noise_model.add_all_qubit_readout_error(ro_error)

    return noise_model


def build_candidate_circuit(
    QuantumCircuit,
    n_qubits: int,
    probability: float,
    total_cx: int,
):
    """
    Build an explicit circuit whose noiseless P(q0=1) equals `probability`.

    The CX network is composed of identity pairs CX(0,q); CX(0,q).
    Hence the *logical* ideal transformation of every pair is identity, while
    the physical noisy simulator sees every CX separately.

    total_cx must be even.
    """
    if total_cx < 0:
        raise ValueError("total_cx must be >= 0.")
    if total_cx % 2 != 0:
        raise ValueError(
            "total_cx must be even because noise-loading blocks use CX pairs."
        )
    if n_qubits < 2:
        raise ValueError("At least 2 qubits are required.")

    qc = QuantumCircuit(n_qubits, 1)

    theta = theta_from_probability(probability)
    qc.ry(theta, 0)

    pair_count = total_cx // 2
    spectators = list(range(1, n_qubits))

    for pair_idx in range(pair_count):
        q = spectators[pair_idx % len(spectators)]
        qc.cx(0, q)
        qc.cx(0, q)

    qc.measure(0, 0)

    return qc


def transpile_candidate(
    qk,
    circuit,
    simulator,
    optimization_level: int,
):
    transpile = qk["transpile"]

    # optimization_level=0 is critical here: the paired CX gates are deliberate
    # noise-loading operations and must not be optimized away.
    tcirc = transpile(
        circuit,
        optimization_level=optimization_level,
        basis_gates=["rz", "sx", "x", "cx"],
        seed_transpiler=12345,
    )

    return tcirc


def circuit_depth_2q(circuit) -> int:
    """
    Count two-qubit instructions in the transpiled circuit.
    """
    count = 0

    for instruction in circuit.data:
        # Qiskit CircuitInstruction exposes .qubits in current releases.
        if len(instruction.qubits) == 2:
            count += 1

    return count



def run_candidate_batch(
    simulator,
    circuits,
    shots: int,
    seed_simulator: int,
):
    """
    Execute a list of transpiled circuits in one Aer job and return P(1)
    for each circuit in order.
    """
    if not circuits:
        return []

    result = simulator.run(
        circuits,
        shots=shots,
        seed_simulator=seed_simulator,
    ).result()

    probabilities = []

    for index, _circuit in enumerate(circuits):
        counts = result.get_counts(index)

        n1 = 0
        total = 0

        for bitstring, count in counts.items():
            clean = bitstring.replace(" ", "")
            bit = clean[-1]
            if bit == "1":
                n1 += int(count)
            total += int(count)

        if total <= 0:
            raise RuntimeError("Aer returned zero shots.")

        probabilities.append(float(n1) / float(total))

    return probabilities


def stable_sim_seed(
    seed: int,
    model: str,
    total_cx: int,
    rep: int,
    candidate_index: int,
) -> int:
    model_code = 1 if model == "REAL" else 2

    value = (
        SIM_SEED
        + 1_000_003 * int(seed)
        + 10_007 * model_code
        + 100_003 * int(total_cx)
        + 65_537 * int(rep)
        + 257 * int(candidate_index)
    )

    # Aer expects a normal positive integer seed range.
    return int(value % (2**31 - 1))


def rank_metrics(
    measured_source: np.ndarray,
    target: np.ndarray,
    positions: np.ndarray,
    k: int,
):
    records = [
        {
            "pos": int(pos),
            "source": float(src),
            "target": float(tgt),
        }
        for pos, src, tgt in zip(positions, measured_source, target)
    ]

    predicted = sorted(
        records,
        key=lambda row: (-row["source"], row["pos"]),
    )
    truth = sorted(
        records,
        key=lambda row: (-row["target"], row["pos"]),
    )

    true_top = {row["pos"] for row in truth[:k]}
    predicted_top = {row["pos"] for row in predicted[:k]}

    precision = len(true_top & predicted_top) / float(k)

    predicted_positions = {
        row["pos"]: rank + 1
        for rank, row in enumerate(predicted)
    }

    evaluations = max(
        predicted_positions[pos]
        for pos in true_top
    )

    relevance = {
        row["pos"]: float(k - rank)
        for rank, row in enumerate(truth[:k])
    }

    def dcg(rows):
        value = 0.0

        for rank, row in enumerate(rows[:k], start=1):
            rel = relevance.get(row["pos"], 0.0)
            value += (
                (2.0**rel - 1.0)
                / math.log2(rank + 1.0)
            )

        return value

    ideal = dcg(truth)
    ndcg = dcg(predicted) / ideal if ideal > 0.0 else float("nan")

    return float(precision), int(evaluations), float(ndcg)


def prepare_model_circuits(
    qk,
    simulator,
    n_qubits: int,
    prominence,
    pairs,
    total_cx: int,
    optimization_level: int,
):
    source = np.asarray(
        [float(prominence[neg]) for neg, _ in pairs],
        dtype=float,
    )
    target = np.asarray(
        [float(prominence[pos]) for _, pos in pairs],
        dtype=float,
    )
    positions = np.asarray(
        [int(pos) for _, pos in pairs],
        dtype=int,
    )

    probabilities = monotone_probability_map(source)

    transpiled = []
    actual_cx_counts = []

    for p in probabilities:
        qc = build_candidate_circuit(
            qk["QuantumCircuit"],
            n_qubits=n_qubits,
            probability=float(p),
            total_cx=total_cx,
        )

        tcirc = transpile_candidate(
            qk,
            qc,
            simulator,
            optimization_level=optimization_level,
        )

        transpiled.append(tcirc)
        actual_cx_counts.append(circuit_depth_2q(tcirc))

    return target, positions, transpiled, actual_cx_counts



def run_model_repetitions(
    simulator,
    target,
    positions,
    circuits,
    seed: int,
    model: str,
    total_cx: int,
    k: int,
    reps: int,
    shots: int,
    batch_reps: int = 20,
):
    p_vals = []
    e_vals = []
    n_vals = []

    n_candidates = len(circuits)

    for rep_start in range(0, reps, batch_reps):
        rep_stop = min(reps, rep_start + batch_reps)
        local_reps = rep_stop - rep_start

        batch_circuits = []
        for _ in range(local_reps):
            batch_circuits.extend(circuits)

        seed_value = stable_sim_seed(
            seed,
            model,
            total_cx,
            rep_start,
            0,
        )

        batch_probs = run_candidate_batch(
            simulator,
            batch_circuits,
            shots=shots,
            seed_simulator=seed_value,
        )

        expected = local_reps * n_candidates
        if len(batch_probs) != expected:
            raise RuntimeError(
                f"Unexpected Aer batch size: got {len(batch_probs)}, "
                f"expected {expected}."
            )

        cursor = 0
        for _ in range(local_reps):
            measured = np.asarray(
                batch_probs[cursor:cursor + n_candidates],
                dtype=float,
            )
            cursor += n_candidates

            p, e, n = rank_metrics(
                measured,
                target,
                positions,
                k,
            )

            p_vals.append(p)
            e_vals.append(e)
            n_vals.append(n)

    return (
        float(np.mean(p_vals)),
        float(np.mean(e_vals)),
        float(np.mean(n_vals)),
    )


def run_cx_level(
    qk,
    base,
    cache,
    seeds,
    total_cx: int,
    k: int,
    reps: int,
    shots: int,
    p1q: float,
    p2q: float,
    readout_error: float,
    optimization_level: int,
):
    noise_model = build_noise_model(
        qk,
        p1q=p1q,
        p2q=p2q,
        readout_error=readout_error,
    )

    simulator = qk["AerSimulator"](
        method="automatic",
        noise_model=noise_model,
    )

    rows = []

    print(f"[v40.3b] Starting CX={total_cx} across {len(seeds)} seed(s)...", flush=True)

    for seed_index, seed in enumerate(seeds, start=1):
        row = {"seed": seed, "total_cx": total_cx}
        print(
            f"[v40.3b] CX={total_cx}: seed {seed_index}/{len(seeds)} ({seed})",
            flush=True,
        )

        for model in ("REAL", "NULL"):
            prominence, pairs = cache[seed][model]

            if len(pairs) < k:
                raise ValueError(
                    f"Seed {seed} model {model}: only {len(pairs)} pairs for K={k}."
                )

            (
                target,
                positions,
                circuits,
                actual_cx_counts,
            ) = prepare_model_circuits(
                qk=qk,
                simulator=simulator,
                n_qubits=base.N_QUBITS,
                prominence=prominence,
                pairs=pairs,
                total_cx=total_cx,
                optimization_level=optimization_level,
            )

            # Scientific guard: paired CX identity blocks must survive transpilation.
            if any(cx != total_cx for cx in actual_cx_counts):
                raise RuntimeError(
                    f"Transpiler altered the deliberate CX loading network. "
                    f"Requested {total_cx}; observed {sorted(set(actual_cx_counts))}. "
                    "Run with optimization_level=0."
                )

            p, e, n = run_model_repetitions(
                simulator=simulator,
                target=target,
                positions=positions,
                circuits=circuits,
                seed=seed,
                model=model,
                total_cx=total_cx,
                k=k,
                reps=reps,
                shots=shots,
            )

            prefix = model.lower()
            row[f"{prefix}_p"] = p
            row[f"{prefix}_eval"] = e
            row[f"{prefix}_ndcg"] = n
            row[f"{prefix}_pairs"] = len(pairs)

        row["delta_p"] = row["real_p"] - row["null_p"]
        row["delta_ndcg"] = row["real_ndcg"] - row["null_ndcg"]
        row["eval_saved"] = row["null_eval"] - row["real_eval"]

        rows.append(row)

    return rows


def bootstrap_paired_ci(
    values: np.ndarray,
    bootstraps: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    n = values.size

    samples = rng.choice(
        values,
        size=(bootstraps, n),
        replace=True,
    )
    stats = np.mean(samples, axis=1)

    low, high = np.quantile(
        stats,
        [0.025, 0.975],
    )

    return (
        float(np.mean(values)),
        float(low),
        float(high),
    )


def summarize_level(
    base,
    rows,
    total_cx: int,
    bootstraps: int,
):
    d_p = np.asarray(
        [r["delta_p"] for r in rows],
        dtype=float,
    )
    d_n = np.asarray(
        [r["delta_ndcg"] for r in rows],
        dtype=float,
    )
    d_e = np.asarray(
        [r["eval_saved"] for r in rows],
        dtype=float,
    )

    p_mean, p_low, p_high = bootstrap_paired_ci(
        d_p,
        bootstraps,
        BOOTSTRAP_SEED + total_cx,
    )
    n_mean, n_low, n_high = bootstrap_paired_ci(
        d_n,
        bootstraps,
        BOOTSTRAP_SEED + total_cx + 1,
    )
    e_mean, e_low, e_high = bootstrap_paired_ci(
        d_e,
        bootstraps,
        BOOTSTRAP_SEED + total_cx + 2,
    )

    p_sign = base.exact_sign_test_positive(d_p)
    p_wilc = base.wilcoxon_signed_rank_positive(d_p)

    n_sign = base.exact_sign_test_positive(d_n)
    n_wilc = base.wilcoxon_signed_rank_positive(d_n)

    real_p = np.asarray(
        [r["real_p"] for r in rows],
        dtype=float,
    )
    null_p = np.asarray(
        [r["null_p"] for r in rows],
        dtype=float,
    )
    real_eval = np.asarray(
        [r["real_eval"] for r in rows],
        dtype=float,
    )
    null_eval = np.asarray(
        [r["null_eval"] for r in rows],
        dtype=float,
    )
    real_ndcg = np.asarray(
        [r["real_ndcg"] for r in rows],
        dtype=float,
    )
    null_ndcg = np.asarray(
        [r["null_ndcg"] for r in rows],
        dtype=float,
    )

    c1 = p_mean > 0.0
    c2 = p_low > 0.0
    c3 = p_sign[2] < 0.05
    c4 = p_wilc[1] < 0.05

    return {
        "total_cx": total_cx,
        "real_p": float(np.mean(real_p)),
        "null_p": float(np.mean(null_p)),
        "delta_p": p_mean,
        "delta_p_low": p_low,
        "delta_p_high": p_high,
        "p_sign": float(p_sign[2]),
        "p_wilc": float(p_wilc[1]),
        "real_eval": float(np.mean(real_eval)),
        "null_eval": float(np.mean(null_eval)),
        "eval_saved": e_mean,
        "eval_saved_low": e_low,
        "eval_saved_high": e_high,
        "real_ndcg": float(np.mean(real_ndcg)),
        "null_ndcg": float(np.mean(null_ndcg)),
        "delta_ndcg": n_mean,
        "delta_ndcg_low": n_low,
        "delta_ndcg_high": n_high,
        "ndcg_sign_p": float(n_sign[2]),
        "ndcg_wilc_p": float(n_wilc[1]),
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "c4": c4,
        "pass": all((c1, c2, c3, c4)),
    }


def render(
    qk,
    base,
    seeds,
    all_rows,
    summaries,
    k: int,
    reps: int,
    shots: int,
    p1q: float,
    p2q: float,
    readout_error: float,
    optimization_level: int,
    bootstraps: int,
):
    qiskit_version = getattr(
        qk["qiskit"],
        "__version__",
        "unknown",
    )

    try:
        import qiskit_aer
        aer_version = getattr(
            qiskit_aer,
            "__version__",
            "unknown",
        )
    except Exception:
        aer_version = "unknown"

    lines = [
        "=== Soft Spaces Phase 4 v40.3b QISKIT/AER CIRCUIT-LEVEL TEST ===",
        f"Qiskit version: {qiskit_version}",
        f"Qiskit Aer version: {aer_version}",
        f"Python version: {platform.python_version()}",
        f"Frozen Phase-3 kernel: {base.VERSION} / namespace {base.FROZEN_MODEL_VERSION}",
        f"Qubits per candidate circuit: {base.N_QUBITS}",
        f"Hamiltonian terms in frozen classical kernel: {base.N_TERMS}",
        f"Independent seeds: {seeds[0]} ... {seeds[-1]}",
        "Application: -E prominence -> circuit readout -> ranked search on +E",
        f"K: {k}",
        f"Circuit repetitions per seed/model/CX level: {reps}",
        f"Shots per candidate circuit: {shots}",
        f"Bootstrap replicates: {bootstraps}",
        "",
        "Aer noise model:",
        f"  one-qubit depolarizing p1q = {p1q:.6g}",
        f"  two-qubit depolarizing p2q = {p2q:.6g}",
        f"  symmetric readout error = {readout_error:.6g}",
        f"  transpiler optimization level = {optimization_level}",
        "",
        "Circuit encoding:",
        "  source prominence -> p in [0.05,0.95] -> RY(theta) on q0",
        "  deliberate paired CX blocks are logically identity in the noiseless circuit",
        "  q0 is measured with finite shots through the Aer noise model",
        "",
        "Scientific scope:",
        "  Explicit noisy Qiskit/Aer circuit channel: YES",
        "  Full Hamiltonian/projector quantum implementation: NO",
        "  Real IBM hardware execution: NO",
        "",
        "=== CX-SWEEP SUMMARY ===",
        "CX | REAL P@K | NULL P@K | deltaP | 95% CI deltaP | sign p | Wilcoxon p | REAL eval | NULL eval | saved | REAL NDCG | NULL NDCG | decision",
    ]

    for s in summaries:
        lines.append(
            f"{s['total_cx']:3d} | "
            f"{s['real_p']:.6f} | "
            f"{s['null_p']:.6f} | "
            f"{s['delta_p']:+.6f} | "
            f"[{s['delta_p_low']:+.6f},{s['delta_p_high']:+.6f}] | "
            f"{s['p_sign']:.8f} | "
            f"{s['p_wilc']:.8f} | "
            f"{s['real_eval']:.6f} | "
            f"{s['null_eval']:.6f} | "
            f"{s['eval_saved']:+.6f} | "
            f"{s['real_ndcg']:.6f} | "
            f"{s['null_ndcg']:.6f} | "
            f"{'PASS' if s['pass'] else 'FAIL'}"
        )

    primary = min(
        summaries,
        key=lambda s: abs(s["total_cx"] - PRIMARY_CX),
    )

    passed = [
        s["total_cx"]
        for s in summaries
        if s["pass"]
    ]

    max_pass = max(passed) if passed else None

    lines.extend([
        "",
        "=== PRIMARY PREDECLARED CHECKPOINT ===",
        f"Total CX gates per candidate circuit = {PRIMARY_CX}",
        f"Mean Precision@{k} REAL: {primary['real_p']:.6f}",
        f"Mean Precision@{k} NULL: {primary['null_p']:.6f}",
        f"Mean delta Precision@{k}: {primary['delta_p']:+.6f}",
        f"95% bootstrap CI: [{primary['delta_p_low']:+.6f}, {primary['delta_p_high']:+.6f}]",
        f"One-sided sign-test p: {primary['p_sign']:.8f}",
        f"One-sided Wilcoxon p: {primary['p_wilc']:.8f}",
        "",
        f"C1 mean delta Precision@{k} > 0: {'PASS' if primary['c1'] else 'FAIL'}",
        f"C2 95% bootstrap CI lower bound > 0: {'PASS' if primary['c2'] else 'FAIL'}",
        f"C3 exact sign-test p < 0.05: {'PASS' if primary['c3'] else 'FAIL'}",
        f"C4 Wilcoxon p < 0.05: {'PASS' if primary['c4'] else 'FAIL'}",
        "",
        f"FINAL v40.3b PRIMARY CHECKPOINT DECISION: {'PASS' if primary['pass'] else 'FAIL'}",
        "",
        "=== QISKIT/AER CIRCUIT ROBUSTNESS ENVELOPE ===",
        (
            f"Largest tested CX count satisfying all four Precision@{k} gates: "
            f"{max_pass}"
            if max_pass is not None
            else f"No tested CX count satisfied all four Precision@{k} gates."
        ),
        "",
        "Interpretation:",
        "  PASS means the REAL-specific ranked-search advantage survives the",
        "  explicit noisy Qiskit/Aer measurement-channel circuit at this checkpoint.",
        "",
        "  It does NOT mean the full Soft-Spaces Hamiltonian/projector model has",
        "  been executed natively as a quantum algorithm.",
        "",
        "  It does NOT mean the result has been demonstrated on IBM hardware.",
    ])

    primary_rows = all_rows[PRIMARY_CX]

    lines.extend([
        "",
        f"=== SEED-LEVEL RESULTS AT CX={PRIMARY_CX} ===",
        "seed | REAL P@K | NULL P@K | deltaP | REAL eval | NULL eval | saved | REAL NDCG | NULL NDCG | deltaNDCG",
    ])

    for r in primary_rows:
        lines.append(
            f"{r['seed']} | "
            f"{r['real_p']:.6f} | "
            f"{r['null_p']:.6f} | "
            f"{r['delta_p']:+.6f} | "
            f"{r['real_eval']:.6f} | "
            f"{r['null_eval']:.6f} | "
            f"{r['eval_saved']:+.6f} | "
            f"{r['real_ndcg']:.6f} | "
            f"{r['null_ndcg']:.6f} | "
            f"{r['delta_ndcg']:+.6f}"
        )

    return "\n".join(lines) + "\n"


def main():
    require_modern_python()
    qk = load_qiskit()
    base, app0 = load_softspaces()

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Top-K target set (default {DEFAULT_K}).",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPS,
        help=f"Circuit repetitions per seed/model/CX level (default {DEFAULT_REPS}).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help=f"Shots per candidate circuit (default {DEFAULT_SHOTS}).",
    )
    parser.add_argument(
        "--p1q",
        type=float,
        default=DEFAULT_P1Q,
        help=f"One-qubit depolarizing error (default {DEFAULT_P1Q}).",
    )
    parser.add_argument(
        "--p2q",
        type=float,
        default=DEFAULT_P2Q,
        help=f"Two-qubit depolarizing error (default {DEFAULT_P2Q}).",
    )
    parser.add_argument(
        "--readout-error",
        type=float,
        default=DEFAULT_READOUT,
        help=f"Symmetric readout error (default {DEFAULT_READOUT}).",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=DEFAULT_OPT_LEVEL,
        help="Must remain 0 unless you explicitly want the transpiler to alter the noise-loading CX pairs.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=BOOTSTRAPS,
        help=f"Bootstrap replicates (default {BOOTSTRAPS}).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast check: first seed, CX=20, 5 circuit repetitions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v40_3b_qiskit_aer_circuit_output.txt"),
    )

    args = parser.parse_args()

    if args.k < 1:
        raise ValueError("--k must be >= 1.")
    if args.reps < 1:
        raise ValueError("--reps must be >= 1.")
    if args.shots < 1:
        raise ValueError("--shots must be >= 1.")
    if not (0.0 <= args.p1q < 1.0):
        raise ValueError("--p1q must satisfy 0 <= p1q < 1.")
    if not (0.0 <= args.p2q < 1.0):
        raise ValueError("--p2q must satisfy 0 <= p2q < 1.")
    if not (0.0 <= args.readout_error < 0.5):
        raise ValueError("--readout-error must satisfy 0 <= r < 0.5.")
    if args.optimization_level != 0:
        raise ValueError(
            "--optimization-level must be 0 in the frozen v40.3 protocol "
            "so deliberate CX identity pairs are not optimized away."
        )
    if args.bootstraps < 1000:
        raise ValueError("--bootstraps must be at least 1000.")

    if args.smoke:
        seeds = base.HAMILTONIAN_SEEDS[:1]
        cx_grid = (PRIMARY_CX,)
        reps = min(args.reps, 5)
        bootstraps = min(args.bootstraps, 5000)
    else:
        seeds = base.HAMILTONIAN_SEEDS
        cx_grid = CX_GRID
        reps = args.reps
        bootstraps = args.bootstraps

    cache = build_frozen_seed_data(
        app0,
        base,
        seeds,
    )

    all_rows = {}
    summaries = []

    for total_cx in cx_grid:
        rows = run_cx_level(
            qk=qk,
            base=base,
            cache=cache,
            seeds=seeds,
            total_cx=total_cx,
            k=args.k,
            reps=reps,
            shots=args.shots,
            p1q=args.p1q,
            p2q=args.p2q,
            readout_error=args.readout_error,
            optimization_level=args.optimization_level,
        )

        all_rows[total_cx] = rows

        summaries.append(
            summarize_level(
                base=base,
                rows=rows,
                total_cx=total_cx,
                bootstraps=bootstraps,
            )
        )

    report = render(
        qk=qk,
        base=base,
        seeds=seeds,
        all_rows=all_rows,
        summaries=summaries,
        k=args.k,
        reps=reps,
        shots=args.shots,
        p1q=args.p1q,
        p2q=args.p2q,
        readout_error=args.readout_error,
        optimization_level=args.optimization_level,
        bootstraps=bootstraps,
    )

    print(report, end="")
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
