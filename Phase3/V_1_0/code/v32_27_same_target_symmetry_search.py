#!/usr/bin/env python3
"""
Soft Spaces Phase 3.2 — v32.27 Same-Target Symmetry Search
==========================================================

Purpose
-------
Search systematically for an additional Pauli-string symmetry S that can
explain the numerically observed transverse same-target evenness

    F_E(lambda) = F_E(-lambda),

which v32.25 established to near machine precision.

Target operator conditions
--------------------------
We search all 4^N Pauli strings S for the frozen 8Q reference model and test:

    S H0 S^dagger = +H0

and, for the transverse perturbation V_T,

    S V_T S^dagger = -V_T.

Such an operator would preserve each H0 eigenspace while flipping the sign of
the transverse perturbation, yielding a direct same-target lambda -> -lambda
symmetry.

For comparison, the dephasing parity is also measured:

    S V_D S^dagger = +/- V_D.

Primary criteria
----------------
A strong same-target symmetry candidate should have:

    relres(S H0 S^dagger, H0)        ~ machine precision
    relres(S V_T S^dagger, -V_T)     ~ machine precision

Optional desirable properties:

    S^2 = I
    S^dagger S = I

All Pauli strings satisfy the latter two exactly up to floating-point error.

Additional validation
---------------------
For every exact +/-E target pair and every candidate S:

1. Check that S preserves each target projector P_E.
2. Check that S preserves the complementary Q_E.
3. Directly test finite-lambda response covariance

       S_P^dagger F_E(lambda) S_P = F_E(-lambda)

   at the SAME target E.

This is the decisive operator-level validation.

Frozen model
------------
    * 8 qubits
    * 11 Hamiltonian terms
    * seed 25042000
    * exhaustive search over 4^8 = 65536 Pauli strings
    * both perturbation families
    * lambdas = 1e-2, 3e-3, 1e-3, 3e-4

Dependencies
------------
Must be beside this file:

    v32_14_degeneracy_closed.py
    v32_17_perturbative_response.py

Outputs
-------
    v32_27_pauli_search.csv
    v32_27_candidates.csv
    v32_27_validation.csv
    v32_27_summary.txt

Usage
-----
    python .\\v32_27_same_target_symmetry_search.py

Optional threshold:
    python .\\v32_27_same_target_symmetry_search.py --tol 1e-10
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


VERSION = "v32.27"
DENOM_EPS = 1.0e-30
PAULI_CHARS = "IXYZ"


# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

def load_module(filename: str, module_name: str):
    here = Path(__file__).resolve().parent
    path = here / filename

    if not path.exists():
        raise FileNotFoundError(f"{filename} must be beside this file.")

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def frob(a: np.ndarray) -> float:
    return float(np.linalg.norm(a, ord="fro"))


def rel_residual(a: np.ndarray, b: np.ndarray) -> float:
    return frob(a - b) / max(frob(a), frob(b), DENOM_EPS)


def herm(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def pauli_string_matrix(label: str) -> np.ndarray:
    mats = {
        "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }
    out = np.array([[1.0 + 0.0j]])
    for ch in label:
        out = np.kron(out, mats[ch])
    return out


def pair_records(ref: dict) -> List[dict]:
    out = []
    for a, b in ref["pairs"]:
        ea = float(ref["energies"][a])
        eb = float(ref["energies"][b])

        gm, gp = (a, b) if ea <= eb else (b, a)

        prom = max(
            ref["prominence"].get(gm, float("-inf")),
            ref["prominence"].get(gp, float("-inf")),
        )
        if not np.isfinite(prom):
            continue

        out.append(
            {
                "g_minus": int(gm),
                "g_plus": int(gp),
                "E_abs": float(abs(ref["energies"][gp])),
                "prominence": float(prom),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Frozen operators
# ---------------------------------------------------------------------------

def dense_H_from_reference(ref: dict) -> np.ndarray:
    U = np.asarray(ref["evecs"], dtype=np.complex128)
    evals = np.asarray(ref["evals"], dtype=float)
    return U @ np.diag(evals) @ U.conj().T


def family_operator(base, ref: dict, family: str) -> np.ndarray:
    return base.dense_pauli_sum(
        base.N_QUBITS,
        ref["perturbations"][family],
    )


# ---------------------------------------------------------------------------
# Fast Pauli conjugation sign tests
# ---------------------------------------------------------------------------

def single_pauli_conjugation_sign(a: str, b: str) -> int:
    """
    For single-qubit Paulis a,b:
      a b a = sign * b
    with sign = +1 if commute, -1 if anticommute.
    """
    if a == "I" or b == "I" or a == b:
        return +1
    return -1


def pauli_conjugation_sign(s: str, term: str) -> int:
    sign = 1
    for a, b in zip(s, term):
        sign *= single_pauli_conjugation_sign(a, b)
    return sign


def extract_term_labels(terms) -> List[str]:
    """
    Best-effort extraction of Pauli labels from the frozen term format.

    Supports:
      * (coeff, "IXYZ...")
      * ("IXYZ...", coeff)
      * dict-like with label/string/pauli keys
    """
    labels = []

    for term in terms:
        label = None

        if isinstance(term, dict):
            for key in ("label", "pauli", "string", "op"):
                if key in term and isinstance(term[key], str):
                    label = term[key]
                    break

        elif isinstance(term, (tuple, list)):
            for item in term:
                if (
                    isinstance(item, str)
                    and set(item.upper()).issubset(set(PAULI_CHARS))
                ):
                    label = item.upper()
                    break

        if label is None:
            raise RuntimeError(
                "Could not extract Pauli label from term. "
                f"Unsupported term format: {term!r}"
            )

        labels.append(label)

    return labels


def exact_termwise_parity(candidate: str, term_labels: List[str]):
    """
    Return:
      +1 if candidate conjugates every nonzero term with + parity,
      -1 if every term with - parity,
       0 if mixed.
    """
    signs = [pauli_conjugation_sign(candidate, t) for t in term_labels]
    if all(s == +1 for s in signs):
        return +1
    if all(s == -1 for s in signs):
        return -1
    return 0


# ---------------------------------------------------------------------------
# Same-target finite-lambda response
# ---------------------------------------------------------------------------

def family_operator_in_eigenbasis(base, ref: dict, family: str):
    V = family_operator(base, ref, family)
    U = np.asarray(ref["evecs"], dtype=np.complex128)
    return V, U.conj().T @ V @ U


def target_blocks(ref: dict, V_eig: np.ndarray, gi: int, eta: float):
    pidx = np.asarray(ref["groups"][gi], dtype=int)
    all_idx = np.arange(len(ref["evals"]), dtype=int)

    mask = np.ones(len(all_idx), dtype=bool)
    mask[pidx] = False
    qidx = all_idx[mask]

    E = float(np.mean(ref["evals"][pidx]))
    Eq = np.asarray(ref["evals"][qidx], dtype=float)

    B = V_eig[np.ix_(qidx, pidx)]
    W = V_eig[np.ix_(qidx, qidx)]

    return pidx, qidx, E, Eq, B, W


def exact_response(
    ref: dict,
    V_eig: np.ndarray,
    gi: int,
    eta: float,
    lam: float,
) -> np.ndarray:
    _, _, E, Eq, B, W = target_blocks(ref, V_eig, gi, eta)

    D = np.diag(E - Eq + 1j * eta)
    X = np.linalg.solve(D - lam * W, B)

    return (lam * lam) * herm(B.conj().T @ X)


def subspace_S_map(
    ref: dict,
    S: np.ndarray,
    gi: int,
) -> np.ndarray:
    U = np.asarray(ref["evecs"], dtype=np.complex128)
    idx = np.asarray(ref["groups"][gi], dtype=int)
    Ug = U[:, idx]
    return Ug.conj().T @ S @ Ug


def projector_from_group(ref: dict, gi: int) -> np.ndarray:
    U = np.asarray(ref["evecs"], dtype=np.complex128)
    idx = np.asarray(ref["groups"][gi], dtype=int)
    Ug = U[:, idx]
    return Ug @ Ug.conj().T


# ---------------------------------------------------------------------------
# Main search and validation
# ---------------------------------------------------------------------------

def search_candidates(
    n_qubits: int,
    H_term_labels: List[str],
    VT_term_labels: List[str],
    VD_term_labels: List[str],
):
    rows = []
    candidates = []

    total = 4 ** n_qubits
    count = 0

    for chars in itertools.product(PAULI_CHARS, repeat=n_qubits):
        label = "".join(chars)
        count += 1

        h_parity = exact_termwise_parity(label, H_term_labels)
        vt_parity = exact_termwise_parity(label, VT_term_labels)
        vd_parity = exact_termwise_parity(label, VD_term_labels)

        is_candidate = (h_parity == +1 and vt_parity == -1)

        row = {
            "version": VERSION,
            "label": label,
            "H_parity": int(h_parity),
            "V_transverse_parity": int(vt_parity),
            "V_dephasing_parity": int(vd_parity),
            "candidate_same_target_transverse": int(is_candidate),
        }
        rows.append(row)

        if is_candidate:
            candidates.append(row.copy())

        if count % 8192 == 0:
            print(f"  scanned {count}/{total} Pauli strings", flush=True)

    return rows, candidates


def validate_candidate(
    base,
    ref: dict,
    H: np.ndarray,
    VT: np.ndarray,
    VD: np.ndarray,
    VT_eig: np.ndarray,
    label: str,
    lambdas: List[float],
):
    S = pauli_string_matrix(label)

    h_resid = rel_residual(S @ H @ S.conj().T, H)
    vt_resid = rel_residual(S @ VT @ S.conj().T, -VT)

    # For dephasing report both signs and let the smaller one speak.
    vd_plus = rel_residual(S @ VD @ S.conj().T, +VD)
    vd_minus = rel_residual(S @ VD @ S.conj().T, -VD)

    pair_info = pair_records(ref)
    targets = sorted(
        set(p["g_minus"] for p in pair_info)
        | set(p["g_plus"] for p in pair_info)
    )

    validation_rows = []

    max_projector_resid = 0.0
    max_subspace_unitarity = 0.0
    max_response_covariance = 0.0

    eta = float(base.ENERGY_REG)

    for gi in targets:
        P = projector_from_group(ref, gi)
        projector_resid = frob(S @ P @ S.conj().T - P)

        SP = subspace_S_map(ref, S, gi)
        I = np.eye(SP.shape[0], dtype=np.complex128)
        subspace_unitarity = frob(SP.conj().T @ SP - I)

        max_projector_resid = max(
            max_projector_resid, projector_resid
        )
        max_subspace_unitarity = max(
            max_subspace_unitarity, subspace_unitarity
        )

        for lam in lambdas:
            Fp = exact_response(
                ref, VT_eig, gi, eta, +lam
            )
            Fm = exact_response(
                ref, VT_eig, gi, eta, -lam
            )

            mapped = SP.conj().T @ Fp @ SP
            covariance_resid = rel_residual(mapped, Fm)

            max_response_covariance = max(
                max_response_covariance,
                covariance_resid,
            )

            validation_rows.append(
                {
                    "version": VERSION,
                    "candidate": label,
                    "group": int(gi),
                    "energy": float(
                        np.mean(ref["evals"][ref["groups"][gi]])
                    ),
                    "lambda": float(lam),
                    "projector_residual": float(projector_resid),
                    "subspace_unitarity_residual": float(
                        subspace_unitarity
                    ),
                    "response_covariance_residual": float(
                        covariance_resid
                    ),
                }
            )

    summary = {
        "label": label,
        "H_residual": float(h_resid),
        "VT_minus_residual": float(vt_resid),
        "VD_plus_residual": float(vd_plus),
        "VD_minus_residual": float(vd_minus),
        "max_projector_residual": float(max_projector_resid),
        "max_subspace_unitarity_residual": float(
            max_subspace_unitarity
        ),
        "max_response_covariance_residual": float(
            max_response_covariance
        ),
    }

    return validation_rows, summary


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Soft Spaces Phase 3.2 v32.27 — exhaustive same-target "
            "Pauli symmetry search."
        )
    )

    p.add_argument(
        "--seed",
        type=int,
        default=25_042_000,
        help="Frozen reference seed. Default 25042000.",
    )

    p.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Numerical validation tolerance. Default 1e-10.",
    )

    p.add_argument(
        "--max-candidates",
        type=int,
        default=32,
        help=(
            "Maximum number of exact termwise candidates to validate "
            "with dense matrices. Default 32."
        ),
    )

    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[1e-2, 3e-3, 1e-3, 3e-4],
    )

    p.add_argument(
        "--output-dir",
        type=str,
        default=".",
    )

    return p.parse_args()


def main():
    args = parse_args()

    base = load_module(
        "v32_14_degeneracy_closed.py",
        "v3214_for_v3227",
    )
    v3217 = load_module(
        "v32_17_perturbative_response.py",
        "v3217_for_v3227",
    )

    if int(base.N_QUBITS) != 8:
        raise RuntimeError(
            f"v32.27 is frozen for 8Q; base reports {base.N_QUBITS}Q."
        )

    if int(base.N_TERMS) != 11:
        raise RuntimeError(
            f"v32.27 expects 11 Hamiltonian terms; "
            f"base reports {base.N_TERMS}."
        )

    if int(args.seed) != 25_042_000:
        print(
            "WARNING: this symmetry search is scientifically frozen for "
            "seed 25042000. Other seeds are exploratory.",
            flush=True,
        )

    lambdas = sorted(
        [float(x) for x in args.lambdas if float(x) > 0.0],
        reverse=True,
    )

    ref = v3217.build_reference(base, int(args.seed))
    ref["seed"] = int(args.seed)

    # Frozen Hamiltonian and perturbation term labels.
    h_terms = base.random_hamiltonian_terms(int(args.seed))

    H_labels = extract_term_labels(h_terms)
    VT_labels = extract_term_labels(ref["perturbations"]["transverse"])
    VD_labels = extract_term_labels(ref["perturbations"]["dephasing"])

    print(f"{VERSION}: Same-Target Symmetry Search")
    print(
        f"seed={args.seed}, qubits={base.N_QUBITS}, "
        f"search_space={4 ** base.N_QUBITS}"
    )
    print()
    print("Searching exact termwise Pauli parities ...", flush=True)

    search_rows, candidate_rows = search_candidates(
        n_qubits=int(base.N_QUBITS),
        H_term_labels=H_labels,
        VT_term_labels=VT_labels,
        VD_term_labels=VD_labels,
    )

    print()
    print(
        f"Found {len(candidate_rows)} exact termwise candidates with "
        "S H S = +H and S V_T S = -V_T."
    )

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    search_path = outdir / "v32_27_pauli_search.csv"
    candidate_path = outdir / "v32_27_candidates.csv"
    validation_path = outdir / "v32_27_validation.csv"
    summary_path = outdir / "v32_27_summary.txt"

    write_csv(search_path, search_rows)

    # Dense operators only after cheap exhaustive search.
    H = dense_H_from_reference(ref)
    VT, VT_eig = family_operator_in_eigenbasis(
        base, ref, "transverse"
    )
    VD = family_operator(base, ref, "dephasing")

    validation_rows = []
    validated = []

    for idx, row in enumerate(
        candidate_rows[: int(args.max_candidates)],
        start=1,
    ):
        label = row["label"]

        print(
            f"Validating candidate {idx}/"
            f"{min(len(candidate_rows), args.max_candidates)}: {label}",
            flush=True,
        )

        vrows, summary = validate_candidate(
            base=base,
            ref=ref,
            H=H,
            VT=VT,
            VD=VD,
            VT_eig=VT_eig,
            label=label,
            lambdas=lambdas,
        )

        validation_rows.extend(vrows)

        merged = dict(row)
        merged.update(summary)

        merged["passes_dense_operator_test"] = int(
            summary["H_residual"] <= args.tol
            and summary["VT_minus_residual"] <= args.tol
        )

        merged["passes_same_target_response_test"] = int(
            summary["max_projector_residual"] <= 1e-8
            and summary["max_response_covariance_residual"] <= 1e-8
        )

        validated.append(merged)

    # Include unvalidated exact candidates too.
    validated_labels = {r["label"] for r in validated}
    for row in candidate_rows:
        if row["label"] not in validated_labels:
            extra = dict(row)
            extra["H_residual"] = float("nan")
            extra["VT_minus_residual"] = float("nan")
            extra["VD_plus_residual"] = float("nan")
            extra["VD_minus_residual"] = float("nan")
            extra["max_projector_residual"] = float("nan")
            extra["max_subspace_unitarity_residual"] = float("nan")
            extra["max_response_covariance_residual"] = float("nan")
            extra["passes_dense_operator_test"] = -1
            extra["passes_same_target_response_test"] = -1
            validated.append(extra)

    write_csv(candidate_path, validated)
    write_csv(validation_path, validation_rows)

    passed_dense = [
        r for r in validated
        if r["passes_dense_operator_test"] == 1
    ]
    passed_response = [
        r for r in validated
        if r["passes_same_target_response_test"] == 1
    ]

    lines = [
        f"Soft Spaces Phase 3.2 {VERSION}",
        "Same-Target Symmetry Search",
        "",
        f"seed={args.seed}",
        f"qubits={base.N_QUBITS}",
        f"dimension={base.DIM}",
        f"hamiltonian_terms={base.N_TERMS}",
        f"search_space={4 ** base.N_QUBITS}",
        f"tolerance={args.tol:.3e}",
        "",
        "SEARCH CONDITION",
        "----------------",
        "Find Pauli strings S such that",
        "",
        "  S H0 S^dagger = +H0",
        "  S V_T S^dagger = -V_T",
        "",
        "This would provide a direct same-target lambda -> -lambda symmetry.",
        "",
        "RESULTS",
        "-------",
        f"exact termwise candidates found = {len(candidate_rows)}",
        f"dense-validated candidates      = {len(passed_dense)}",
        f"same-target response passes     = {len(passed_response)}",
        "",
    ]

    if passed_response:
        lines += [
            "STRONG CANDIDATES",
            "-----------------",
        ]

        for r in passed_response:
            lines += [
                f"candidate={r['label']}",
                f"  H residual                  = {r['H_residual']:.6e}",
                f"  transverse -V residual      = {r['VT_minus_residual']:.6e}",
                f"  dephasing +V residual       = {r['VD_plus_residual']:.6e}",
                f"  dephasing -V residual       = {r['VD_minus_residual']:.6e}",
                f"  max projector residual      = {r['max_projector_residual']:.6e}",
                (
                    "  max response covariance resid = "
                    f"{r['max_response_covariance_residual']:.6e}"
                ),
                "",
            ]
    elif candidate_rows:
        lines += [
            "NO FULL SAME-TARGET RESPONSE PASS",
            "---------------------------------",
            "Termwise candidates exist, but none of the validated candidates",
            "passed the full finite-lambda same-target covariance test.",
            "",
        ]
    else:
        lines += [
            "NO PAULI CANDIDATE FOUND",
            "------------------------",
            "No Pauli string simultaneously commutes with H0 termwise and",
            "anticommutes with the transverse perturbation termwise.",
            "",
            "If transverse evenness remains exact, its origin must therefore",
            "involve a non-Pauli symmetry or a more subtle algebraic cancellation.",
            "",
        ]

    lines += [
        "DECISION LOGIC",
        "--------------",
        "A candidate that passes both dense operator tests and finite-lambda",
        "same-target response covariance supplies the missing symmetry mechanism",
        "for F_E(lambda)=F_E(-lambda).",
        "",
        "If no such Pauli candidate exists, the next step should not be more",
        "feature fitting; it should be an algebraic search beyond the Pauli group.",
    ]

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("WROTE")
    print(f"  {search_path}")
    print(f"  {candidate_path}")
    print(f"  {validation_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
