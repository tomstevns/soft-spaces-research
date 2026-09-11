#!/usr/bin/env python3
"""
Soft Spaces Phase 3.2 — v32.26 Odd-Order Cancellation Mechanism
===============================================================

Purpose
-------
Resolve the mechanism behind the transverse even-order response observed in
v32.24-v32.25.

For a fixed target eigenspace P at energy E, define

    B = Q V P,
    W = Q V Q,
    R0 = (E I_Q - Q H0 Q + i eta I_Q)^(-1).

The third-order coefficient of the regularized Feshbach response is

    K3 = Herm[B^dagger R0 W R0 B].

Using complete degeneracy-closed Q eigenspaces P_G, insert resolutions of the
identity in Q:

    K3 = sum_{G,K} X_{G,K},

where

    X_{G,K}
      = Herm[
          B_G^dagger r_G W_{G,K} r_K B_K
        ],

    B_G   = P_G V P,
    W_GK  = P_G V P_K,
    r_G   = 1 / (E - E_G + i eta).

The global Gamma symmetry maps each Q group G to its +/-E partner bar(G).

v32.26 tests whether the odd-order cancellation for transverse perturbations
occurs through symmetry-related grouped contributions.

Primary tests
-------------
For every target group and perturbation family:

1. Reconstruct K3 exactly from the grouped (G,K) terms.

2. Pair every contribution X_{G,K} with the Gamma-related contribution
   X_{bar(G),bar(K)}.

3. Transform the partner contribution back into the same target P basis and
   test whether

       X_partner ~= - X_original

   for transverse perturbations.

4. Measure how much cancellation occurs pairwise before the global sum.

5. Repeat the same logic for the leading K5 grouped structure in a compressed
   diagnostic form.  K5 has four intermediate Q-group indices and is therefore
   combinatorially larger; v32.26 samples/aggregates symmetry-paired path tuples
   rather than materializing the full 4-index tensor.

Scientific goal
---------------
Distinguish two possibilities:

A. Pairwise symmetry cancellation:
       odd virtual paths cancel locally in Gamma-related pairs.

B. Collective cancellation:
       individual odd-path contributions are nonzero, but only the full sum
       cancels.

Either result is informative.  The script does not assume the answer.

Frozen pilot
------------
    * 8 qubits
    * 11 Hamiltonian terms
    * seed 25042000
    * Gamma = XIXXIIXX
    * both perturbation families
    * all exact +/-E target pairs
    * complete degeneracy-closed Q groups

Dependencies
------------
Must be beside this file:

    v32_14_degeneracy_closed.py
    v32_17_perturbative_response.py

Outputs
-------
    v32_26_k3_pair_terms.csv
    v32_26_target_summary.csv
    v32_26_k5_sampled_paths.csv
    v32_26_summary.txt

Usage
-----
    python .\\v32_26_odd_order_cancellation.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


VERSION = "v32.26"
DEFAULT_GAMMA = "XIXXIIXX"
DENOM_EPS = 1.0e-30


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
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def frob(a: np.ndarray) -> float:
    return float(np.linalg.norm(a, ord="fro"))


def herm(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def rel_residual(a: np.ndarray, b: np.ndarray) -> float:
    return frob(a - b) / max(frob(a), frob(b), DENOM_EPS)


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
# Gamma group map
# ---------------------------------------------------------------------------

def build_gamma_group_map(ref: dict, Gamma: np.ndarray) -> Dict[int, int]:
    U = np.asarray(ref["evecs"], dtype=np.complex128)
    groups = ref["groups"]

    projectors = []
    for g in groups:
        Ug = U[:, np.asarray(g, dtype=int)]
        projectors.append(Ug @ Ug.conj().T)

    mapping = {}

    for i, Pi in enumerate(projectors):
        mapped = Gamma @ Pi @ Gamma.conj().T

        best_j = None
        best_resid = float("inf")

        for j, Pj in enumerate(projectors):
            resid = frob(mapped - Pj)
            if resid < best_resid:
                best_resid = resid
                best_j = j

        if best_j is None:
            raise RuntimeError("Could not map Gamma group.")

        mapping[i] = int(best_j)

    return mapping


def subspace_gamma_map(
    ref: dict,
    Gamma: np.ndarray,
    g_from: int,
    g_to: int,
) -> np.ndarray:
    U = np.asarray(ref["evecs"], dtype=np.complex128)
    idx_from = np.asarray(ref["groups"][g_from], dtype=int)
    idx_to = np.asarray(ref["groups"][g_to], dtype=int)

    U_from = U[:, idx_from]
    U_to = U[:, idx_to]

    return U_to.conj().T @ Gamma @ U_from


# ---------------------------------------------------------------------------
# Family operator and grouped blocks
# ---------------------------------------------------------------------------

def family_operator_in_eigenbasis(base, ref: dict, family: str):
    perturbation = ref["perturbations"][family]
    V = base.dense_pauli_sum(base.N_QUBITS, perturbation)
    U = np.asarray(ref["evecs"], dtype=np.complex128)
    V_eig = U.conj().T @ V @ U
    return V, V_eig


def q_groups_for_target(ref: dict, target_g: int) -> List[int]:
    return [g for g in range(len(ref["groups"])) if g != target_g]


def target_energy(ref: dict, gi: int) -> float:
    return float(np.mean(ref["evals"][ref["groups"][gi]]))


def group_block(
    ref: dict,
    V_eig: np.ndarray,
    g_row: int,
    g_col: int,
) -> np.ndarray:
    i = np.asarray(ref["groups"][g_row], dtype=int)
    j = np.asarray(ref["groups"][g_col], dtype=int)
    return V_eig[np.ix_(i, j)]


def resolvent_scalar(
    ref: dict,
    target_g: int,
    q_g: int,
    eta: float,
) -> complex:
    E = target_energy(ref, target_g)
    Eq = target_energy(ref, q_g)
    return 1.0 / (E - Eq + 1j * eta)


# ---------------------------------------------------------------------------
# K3 grouped decomposition
# ---------------------------------------------------------------------------

def k3_group_term(
    ref: dict,
    V_eig: np.ndarray,
    target_g: int,
    g: int,
    k: int,
    eta: float,
) -> np.ndarray:
    """
    X_{G,K} = Herm[
        (P V P_G) r_G (P_G V P_K) r_K (P_K V P)
    ]
    represented in the target P basis.
    """
    B_g = group_block(ref, V_eig, g, target_g)   # P_G V P
    B_k = group_block(ref, V_eig, k, target_g)   # P_K V P
    W_gk = group_block(ref, V_eig, g, k)         # P_G V P_K

    r_g = resolvent_scalar(ref, target_g, g, eta)
    r_k = resolvent_scalar(ref, target_g, k, eta)

    raw = B_g.conj().T @ (r_g * W_gk) @ (r_k * B_k)
    return herm(raw)


def direct_k3(
    ref: dict,
    V_eig: np.ndarray,
    target_g: int,
    eta: float,
) -> np.ndarray:
    qgs = q_groups_for_target(ref, target_g)
    dimp = len(ref["groups"][target_g])
    total = np.zeros((dimp, dimp), dtype=np.complex128)

    for g in qgs:
        for k in qgs:
            total += k3_group_term(
                ref, V_eig, target_g, g, k, eta
            )

    return total


# ---------------------------------------------------------------------------
# K5 sampled grouped path contribution
# ---------------------------------------------------------------------------

def k5_path_term(
    ref: dict,
    V_eig: np.ndarray,
    target_g: int,
    path: Tuple[int, int, int, int],
    eta: float,
) -> np.ndarray:
    g1, g2, g3, g4 = path

    B1 = group_block(ref, V_eig, g1, target_g)
    B4 = group_block(ref, V_eig, g4, target_g)

    W12 = group_block(ref, V_eig, g1, g2)
    W23 = group_block(ref, V_eig, g2, g3)
    W34 = group_block(ref, V_eig, g3, g4)

    r1 = resolvent_scalar(ref, target_g, g1, eta)
    r2 = resolvent_scalar(ref, target_g, g2, eta)
    r3 = resolvent_scalar(ref, target_g, g3, eta)
    r4 = resolvent_scalar(ref, target_g, g4, eta)

    raw = (
        B1.conj().T
        @ (r1 * W12)
        @ (r2 * W23)
        @ (r3 * W34)
        @ (r4 * B4)
    )

    return herm(raw)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_target(
    base,
    ref: dict,
    Gamma: np.ndarray,
    gamma_group_map: Dict[int, int],
    family: str,
    V_eig: np.ndarray,
    target_g: int,
    rng: np.random.Generator,
    k5_samples: int,
):
    eta = float(base.ENERGY_REG)
    qgs = q_groups_for_target(ref, target_g)

    partner_target = gamma_group_map[target_g]
    Gp = subspace_gamma_map(
        ref=ref,
        Gamma=Gamma,
        g_from=target_g,
        g_to=partner_target,
    )

    # K3 grouped decomposition
    k3_rows = []
    total = np.zeros(
        (
            len(ref["groups"][target_g]),
            len(ref["groups"][target_g]),
        ),
        dtype=np.complex128,
    )
    pair_cancel_sum = np.zeros_like(total)

    visited = set()

    for g in qgs:
        for k in qgs:
            X = k3_group_term(
                ref, V_eig, target_g, g, k, eta
            )
            total += X

            bg = gamma_group_map[g]
            bk = gamma_group_map[k]

            key = (g, k)
            pkey = (bg, bk)

            if key in visited:
                continue

            visited.add(key)
            visited.add(pkey)

            # Partner contribution lives at the Gamma-partner target.
            if bg == partner_target or bk == partner_target:
                # These cannot serve as Q groups of partner target.
                continue

            Xp_partner_basis = k3_group_term(
                ref,
                V_eig,
                partner_target,
                bg,
                bk,
                eta,
            )

            # Bring partner contribution back to original target basis.
            Xp_back = Gp.conj().T @ Xp_partner_basis @ Gp

            anti_resid = rel_residual(Xp_back, -X)
            same_resid = rel_residual(Xp_back, +X)

            pair_cancel = X + Xp_back
            pair_cancel_sum += pair_cancel

            k3_rows.append(
                {
                    "version": VERSION,
                    "seed": int(ref["seed"]),
                    "family": family,
                    "target_group": int(target_g),
                    "partner_target_group": int(partner_target),
                    "g": int(g),
                    "k": int(k),
                    "bar_g": int(bg),
                    "bar_k": int(bk),
                    "X_norm": float(frob(X)),
                    "partner_norm": float(frob(Xp_back)),
                    "anti_pair_residual": float(anti_resid),
                    "same_pair_residual": float(same_resid),
                    "pair_sum_norm": float(frob(pair_cancel)),
                }
            )

    direct = direct_k3(ref, V_eig, target_g, eta)
    recon_resid = rel_residual(total, direct)

    # K5 sampling
    k5_rows = []

    valid_q_partner = [
        g for g in qgs if gamma_group_map[g] != partner_target
    ]

    if valid_q_partner:
        for _ in range(int(k5_samples)):
            path = tuple(
                int(x)
                for x in rng.choice(valid_q_partner, size=4, replace=True)
            )
            bpath = tuple(gamma_group_map[g] for g in path)

            X5 = k5_path_term(
                ref, V_eig, target_g, path, eta
            )

            X5p = k5_path_term(
                ref, V_eig, partner_target, bpath, eta
            )
            X5p_back = Gp.conj().T @ X5p @ Gp

            k5_rows.append(
                {
                    "version": VERSION,
                    "seed": int(ref["seed"]),
                    "family": family,
                    "target_group": int(target_g),
                    "partner_target_group": int(partner_target),
                    "path": "-".join(str(x) for x in path),
                    "bar_path": "-".join(str(x) for x in bpath),
                    "X5_norm": float(frob(X5)),
                    "partner_norm": float(frob(X5p_back)),
                    "anti_pair_residual": float(
                        rel_residual(X5p_back, -X5)
                    ),
                    "same_pair_residual": float(
                        rel_residual(X5p_back, +X5)
                    ),
                    "pair_sum_norm": float(frob(X5 + X5p_back)),
                }
            )

    target_summary = {
        "version": VERSION,
        "seed": int(ref["seed"]),
        "family": family,
        "target_group": int(target_g),
        "partner_target_group": int(partner_target),
        "target_energy": float(target_energy(ref, target_g)),
        "K3_norm": float(frob(direct)),
        "K3_reconstruction_residual": float(recon_resid),
        "K3_pair_cancel_sum_norm": float(frob(pair_cancel_sum)),
        "median_K3_anti_pair_residual": float(
            np.median([r["anti_pair_residual"] for r in k3_rows])
            if k3_rows else float("nan")
        ),
        "median_K3_same_pair_residual": float(
            np.median([r["same_pair_residual"] for r in k3_rows])
            if k3_rows else float("nan")
        ),
        "median_K3_pair_sum_norm": float(
            np.median([r["pair_sum_norm"] for r in k3_rows])
            if k3_rows else float("nan")
        ),
        "median_K5_anti_pair_residual": float(
            np.median([r["anti_pair_residual"] for r in k5_rows])
            if k5_rows else float("nan")
        ),
        "median_K5_same_pair_residual": float(
            np.median([r["same_pair_residual"] for r in k5_rows])
            if k5_rows else float("nan")
        ),
        "median_K5_pair_sum_norm": float(
            np.median([r["pair_sum_norm"] for r in k5_rows])
            if k5_rows else float("nan")
        ),
        "n_K3_pairs": int(len(k3_rows)),
        "n_K5_samples": int(len(k5_rows)),
    }

    return k3_rows, k5_rows, target_summary


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Soft Spaces Phase 3.2 v32.26 — odd-order cancellation mechanism."
        )
    )

    p.add_argument(
        "--seed",
        type=int,
        default=25_042_000,
        help="Frozen symmetry seed. Default 25042000.",
    )

    p.add_argument(
        "--gamma",
        type=str,
        default=DEFAULT_GAMMA,
        help=f"Pauli-string Gamma. Default {DEFAULT_GAMMA}.",
    )

    p.add_argument(
        "--k5-samples",
        type=int,
        default=200,
        help="Random grouped K5 paths per target. Default 200.",
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
        "v3214_for_v3226",
    )
    v3217 = load_module(
        "v32_17_perturbative_response.py",
        "v3217_for_v3226",
    )

    if int(base.N_QUBITS) != 8:
        raise RuntimeError(
            f"v32.26 is frozen for 8Q; base reports {base.N_QUBITS}Q."
        )

    if int(base.N_TERMS) != 11:
        raise RuntimeError(
            f"v32.26 expects 11 Hamiltonian terms; base reports "
            f"{base.N_TERMS}."
        )

    if int(args.seed) != 25_042_000:
        print(
            "WARNING: Gamma=XIXXIIXX was established for seed 25042000. "
            "Other seeds are exploratory unless Gamma is independently verified."
        )

    gamma_label = args.gamma.strip().upper()
    if len(gamma_label) != int(base.N_QUBITS):
        raise ValueError(
            f"Gamma string has length {len(gamma_label)}, "
            f"expected {base.N_QUBITS}."
        )

    ref = v3217.build_reference(base, int(args.seed))
    ref["seed"] = int(args.seed)

    Gamma = pauli_string_matrix(gamma_label)
    gamma_group_map = build_gamma_group_map(ref, Gamma)

    rng = np.random.default_rng(32_026)

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    all_k3_rows = []
    all_k5_rows = []
    all_target_summary = []

    print(f"{VERSION}: Odd-Order Cancellation Mechanism")
    print(
        f"seed={args.seed}, qubits={base.N_QUBITS}, "
        f"terms={base.N_TERMS}, Gamma={gamma_label}"
    )
    print()

    # Analyze only positive-energy targets to avoid duplicate +/-E reporting.
    pairs = pair_records(ref)
    positive_targets = sorted(set(p["g_plus"] for p in pairs))

    family_summaries = {}

    for family in base.FAMILIES:
        print(f"[{family}] ...", flush=True)

        _, V_eig = family_operator_in_eigenbasis(base, ref, family)

        fam_summaries = []

        for idx, target_g in enumerate(positive_targets, start=1):
            print(
                f"  target {idx}/{len(positive_targets)} "
                f"group={target_g}",
                flush=True,
            )

            k3_rows, k5_rows, ts = analyze_target(
                base=base,
                ref=ref,
                Gamma=Gamma,
                gamma_group_map=gamma_group_map,
                family=family,
                V_eig=V_eig,
                target_g=target_g,
                rng=rng,
                k5_samples=int(args.k5_samples),
            )

            all_k3_rows.extend(k3_rows)
            all_k5_rows.extend(k5_rows)
            all_target_summary.append(ts)
            fam_summaries.append(ts)

        family_summaries[family] = {
            "median_K3_norm": float(
                np.median([x["K3_norm"] for x in fam_summaries])
            ),
            "median_K3_reconstruction_residual": float(
                np.median(
                    [x["K3_reconstruction_residual"] for x in fam_summaries]
                )
            ),
            "median_K3_anti_pair_residual": float(
                np.nanmedian(
                    [x["median_K3_anti_pair_residual"] for x in fam_summaries]
                )
            ),
            "median_K3_same_pair_residual": float(
                np.nanmedian(
                    [x["median_K3_same_pair_residual"] for x in fam_summaries]
                )
            ),
            "median_K3_pair_sum_norm": float(
                np.nanmedian(
                    [x["median_K3_pair_sum_norm"] for x in fam_summaries]
                )
            ),
            "median_K5_anti_pair_residual": float(
                np.nanmedian(
                    [x["median_K5_anti_pair_residual"] for x in fam_summaries]
                )
            ),
            "median_K5_same_pair_residual": float(
                np.nanmedian(
                    [x["median_K5_same_pair_residual"] for x in fam_summaries]
                )
            ),
            "median_K5_pair_sum_norm": float(
                np.nanmedian(
                    [x["median_K5_pair_sum_norm"] for x in fam_summaries]
                )
            ),
        }

        s = family_summaries[family]

        print(
            f"  median ||K3||                        = "
            f"{s['median_K3_norm']:.6e}"
        )
        print(
            f"  median K3 reconstruction residual    = "
            f"{s['median_K3_reconstruction_residual']:.6e}"
        )
        print(
            f"  median K3 anti-pair residual         = "
            f"{s['median_K3_anti_pair_residual']:.6e}"
        )
        print(
            f"  median K3 same-pair residual         = "
            f"{s['median_K3_same_pair_residual']:.6e}"
        )
        print(
            f"  median sampled K5 anti-pair residual = "
            f"{s['median_K5_anti_pair_residual']:.6e}"
        )
        print()

    k3_path = outdir / "v32_26_k3_pair_terms.csv"
    target_path = outdir / "v32_26_target_summary.csv"
    k5_path = outdir / "v32_26_k5_sampled_paths.csv"
    summary_path = outdir / "v32_26_summary.txt"

    write_csv(k3_path, all_k3_rows)
    write_csv(target_path, all_target_summary)
    write_csv(k5_path, all_k5_rows)

    lines = [
        f"Soft Spaces Phase 3.2 {VERSION}",
        "Odd-Order Cancellation Mechanism",
        "",
        f"seed={args.seed}",
        f"qubits={base.N_QUBITS}",
        f"dimension={base.DIM}",
        f"hamiltonian_terms={base.N_TERMS}",
        f"Gamma={gamma_label}",
        f"k5_samples_per_target={args.k5_samples}",
        "",
        "QUESTION",
        "--------",
        "Do odd-order virtual-path contributions cancel pairwise under the",
        "Gamma-related +/-E mapping, or only collectively after summation?",
        "",
        "RESULTS",
        "-------",
    ]

    for family in base.FAMILIES:
        s = family_summaries[family]

        lines += [
            f"[{family}]",
            (
                "  median ||K3||                        = "
                f"{s['median_K3_norm']:.6e}"
            ),
            (
                "  median K3 reconstruction residual    = "
                f"{s['median_K3_reconstruction_residual']:.6e}"
            ),
            (
                "  median K3 anti-pair residual         = "
                f"{s['median_K3_anti_pair_residual']:.6e}"
            ),
            (
                "  median K3 same-pair residual         = "
                f"{s['median_K3_same_pair_residual']:.6e}"
            ),
            (
                "  median K3 pair-sum norm              = "
                f"{s['median_K3_pair_sum_norm']:.6e}"
            ),
            (
                "  median sampled K5 anti-pair residual = "
                f"{s['median_K5_anti_pair_residual']:.6e}"
            ),
            (
                "  median sampled K5 same-pair residual = "
                f"{s['median_K5_same_pair_residual']:.6e}"
            ),
            (
                "  median sampled K5 pair-sum norm      = "
                f"{s['median_K5_pair_sum_norm']:.6e}"
            ),
            "",
        ]

    lines += [
        "INTERPRETATION GUIDE",
        "--------------------",
        "If transverse anti-pair residuals are near machine precision while",
        "same-pair residuals are O(1), then odd-order cancellation is pairwise",
        "under Gamma-related virtual paths.",
        "",
        "If both anti-pair and same-pair residuals are O(1) while the final",
        "K3 norm is near zero, the cancellation is collective rather than",
        "simple pairwise cancellation.",
        "",
        "A clean pairwise result would provide the missing microscopic",
        "mechanism for the even-order transverse response observed in v32.25.",
    ]

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("WROTE")
    print(f"  {k3_path}")
    print(f"  {target_path}")
    print(f"  {k5_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
