#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\
v1_0_Soft_Space_v25_4_1_dimension_normalized_ranker.py

Purpose
-------
First pythonfile that should be changed by ChatGPT for our new Soft Space project
_______
v25.3.1 correction: controls are now paired and geometry-matched to the anchor descendants.\n\nEvidence step v19: add a physically meaningful, basis-invariant perturbation robustness test.

This version keeps the fully reproducible, statistical–numerical protocol (no model training)
and the REAL vs spectrum-matched Haar NULL baseline, while upgrading the perturbation analysis
to use a 2D-subspace overlap metric (projector overlap) for near-degenerate eigenpairs.

Key features
------------
- REAL vs NULL baseline: spectrum-matched Haar eigenbasis for NULL.
- Fixed-fraction stable selection per model (v15).
- Fine + coarse signatures (global quantiles).
- Perturbation robustness (v21.1) + Open-system (v23): apply the *same* small Hermitian perturbation ΔH in the
  computational basis to both H_REAL and H_NULL, re-diagonalize, then track each baseline
  near-degenerate 2D manifold to the best-matching *neighbor* pair in the perturbed spectrum.
  Robustness is quantified by:

    (i)   pair_retention_rate: whether the best-matching neighbor gap remains < eps
    (ii)  subspace_overlap: 0.5 * Tr(P P') = 0.5 * ||U^† U'||_F^2, invariant to basis
    (iii) optional coarse-signature retention and feature drifts (secondary)

Key refinements vs v17
----------------------
1) Perturbation robustness is now evidence-grade: basis-invariant 2D subspace overlap,
   instead of relying on entropy drift under re-sampled Haar bases.
2) NULL perturbation is performed by constructing H_NULL = U diag(E) U^† (Haar U) and
   applying the same ΔH in computational basis before re-diagonalizing.
3) Reporting includes overlap quantiles for interpretability.

Dependencies
------------
Only numpy (no Qiskit, no SciPy).

Notes on scaling
----------------
- For n_qubits=4, entropy ranges up to log2(d)=4 bits, so absolute medians shift.
  Interpretations rely on REAL–NULL deltas/effect sizes and batch convergence, not raw levels.
"""

from __future__ import annotations

import argparse
import sys
import math
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Dict, Optional, Any
from collections import Counter
from itertools import product

import numpy as np


# Console/output encoding guard for Windows shells (e.g. PowerShell cp1252)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass



# ----------------------------
# Utilities
# ----------------------------

def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def stable_hash_int(s: str) -> int:
    """Deterministic int hash independent of Python's hash randomization."""
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def bin_index(x: float, step: float) -> int:
    if step <= 0:
        raise ValueError("step must be > 0")
    return int(math.floor(float(x) / float(step) + 1e-12))


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def make_quantile_edges(values: np.ndarray, q_bins: int) -> np.ndarray:
    """
    Quantile edges for q_bins categories.
    Returns length q_bins+1, with edges[0]=-inf and edges[-1]=+inf.
    """
    if q_bins < 2:
        raise ValueError("q_bins must be >= 2")
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        edges = np.linspace(0.0, 1.0, q_bins + 1)
    else:
        qs = np.linspace(0.0, 1.0, q_bins + 1)
        try:
            edges = np.quantile(v, qs, method="linear")
        except TypeError:
            edges = np.quantile(v, qs)
    edges = np.asarray(edges, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.maximum.accumulate(edges)
    return edges


def quantile_bin(x: float, edges: np.ndarray, q_bins: int) -> int:
    idx = int(np.searchsorted(edges, float(x), side="right") - 1)
    return clamp_int(idx, 0, q_bins - 1)


def safe_median(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.median(x))


# ----------------------------
# Exact binomial tail (no SciPy)
# ----------------------------

def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return -math.inf
    p = float(p)
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return _log_choose(n, k) + k * math.log(p) + (n - k) * math.log(1.0 - p)


def _logsumexp(log_terms: List[float]) -> float:
    m = max(log_terms)
    if m == -math.inf:
        return -math.inf
    s = sum(math.exp(t - m) for t in log_terms)
    return m + math.log(s)


def binom_tail_ge(k: int, n: int, p: float) -> float:
    """Exact tail probability P(X >= k), X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    log_terms = [_log_binom_pmf(x, n, p) for x in range(k, n + 1)]
    return float(math.exp(_logsumexp(log_terms)))


# ----------------------------
# Pauli operator cache (dimension dependent; v24.8 lazy/on-demand)
# ----------------------------

PAULIS = ["I", "X", "Y", "Z"]
PAULI_MATS = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

def kron_n(mats: List[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


class PauliCache:
    """
    Lazy/on-demand Pauli cache.

    v24.5 precomputed *all* n-qubit Pauli operators except all-I. That is fine for 6-7 qubits
    but becomes memory-prohibitive at 8 qubits because 4^n - 1 dense matrices are enormous.
    In v24.8 we instead:
      - keep only small metadata in memory,
      - generate a dense operator matrix only when a sampled label is actually used,
      - memoize operators that are requested repeatedly.

    This preserves the statistical protocol while removing the up-front RAM blow-up.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = int(n_qubits)
        self.d = 2 ** self.n_qubits
        self._op_cache: Dict[str, np.ndarray] = {}

    @property
    def n_labels(self) -> int:
        return (4 ** self.n_qubits) - 1

    @staticmethod
    def build(n_qubits: int) -> "PauliCache":
        return PauliCache(int(n_qubits))

    def get_op(self, label: str) -> np.ndarray:
        op = self._op_cache.get(label)
        if op is None:
            op = kron_n([PAULI_MATS[p] for p in label])
            self._op_cache[label] = op
        return op

    def sample_label_indices(self, rng: np.random.Generator, n_terms: int) -> np.ndarray:
        """Sample integer indices in [0, 4^n - 2], each mapping to one non-all-I label."""
        return rng.integers(0, self.n_labels, size=int(n_terms))

    def index_to_label(self, idx: int) -> str:
        """
        Map 0..(4^n-2) to the ordered non-all-I Pauli labels that v24.5 would have produced
        by iterating product(PAULIS, repeat=n_qubits) and skipping all-I.
        """
        x = int(idx) + 1  # shift so 0 -> first non-all-I word, and all-I is excluded
        digits = []
        for _ in range(self.n_qubits):
            x, r = divmod(x, 4)
            digits.append(PAULIS[r])
        return "".join(reversed(digits))


def build_random_pauli_hamiltonian_cached(cache: PauliCache, n_terms: int, seed: int) -> np.ndarray:
    rng = rng_from_seed(seed)
    d = cache.d
    H = np.zeros((d, d), dtype=complex)

    idxs = cache.sample_label_indices(rng, int(n_terms))
    coeffs = rng.uniform(-1.0, 1.0, size=int(n_terms)).astype(float)
    for k in range(int(n_terms)):
        lab = cache.index_to_label(int(idxs[k]))
        H = H + float(coeffs[k]) * cache.get_op(lab)

    H = 0.5 * (H + H.conj().T)
    return H


def haar_random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    Z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    diag = np.diag(R)
    ph = diag / np.abs(diag)
    Q = Q * ph
    return Q


def null_haar_basis_hamiltonian(evals: np.ndarray, seed: int) -> np.ndarray:
    rng = rng_from_seed(stable_hash_int(f"NULL_HAAR_BASIS|{seed}"))
    d = evals.size
    U = haar_random_unitary(d, rng)
    H = U @ np.diag(evals) @ U.conj().T
    H = 0.5 * (H + H.conj().T)
    return H



def null_haar_basis_eigs(evals: np.ndarray, seed: int, tag: str = "NULL_HAAR_BASIS") -> Tuple[np.ndarray, np.ndarray]:
    """Return (evals, evecs) for the Haar-basis NULL without re-diagonalizing a full matrix."""
    rng = rng_from_seed(stable_hash_int(f"{tag}|{seed}"))
    d = evals.size
    U = haar_random_unitary(d, rng)
    return np.array(evals, dtype=float), U


def amplitude_entropy_bits(state: np.ndarray, eps: float = 1e-12) -> float:
    p = np.abs(state) ** 2
    p = p / (p.sum() + eps)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log2(p)))


def dominant_mask_by_mass(p: np.ndarray, keep_mass: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    keep_mass = float(keep_mass)
    if p.size == 0:
        return np.zeros_like(p, dtype=bool)
    keep_mass = max(0.0, min(1.0, keep_mass))
    order = np.argsort(-p, kind="mergesort")
    cum = 0.0
    mask = np.zeros(p.size, dtype=bool)
    for idx in order:
        mask[idx] = True
        cum += float(p[idx])
        if cum >= keep_mass and np.any(mask):
            break
    if not np.any(mask):
        mask[order[0]] = True
    return mask


def leakage_proxy_from_state(state: np.ndarray, keep_mask: np.ndarray, eps: float = 1e-12) -> float:
    p = np.abs(state) ** 2
    p = p / (p.sum() + eps)
    kept = float(np.sum(p[keep_mask])) if np.any(keep_mask) else 0.0
    return float(max(0.0, 1.0 - kept))


def leakage_proxy_fast(
    evals: np.ndarray,
    evecs: np.ndarray,
    i: int,
    j: int,
    times: List[float],
    keep_mass: float,
) -> Tuple[float, int]:
    vi = evecs[:, i]
    vj = evecs[:, j]
    psi0 = (vi + vj) / np.sqrt(2.0)

    p0 = np.abs(psi0) ** 2
    p0 = p0 / max(1e-12, float(np.sum(p0)))
    keep_mask = dominant_mask_by_mass(p0, keep_mass=keep_mass)
    dom = int(np.sum(keep_mask))
    dom = max(1, dom)

    Ei = float(evals[i])
    Ej = float(evals[j])

    leaks = []
    for t in times:
        ph_i = np.exp(-1j * Ei * float(t))
        ph_j = np.exp(-1j * Ej * float(t))
        psi_t = (ph_i * vi + ph_j * vj) / np.sqrt(2.0)
        leaks.append(leakage_proxy_from_state(psi_t, keep_mask))
    return float(np.mean(leaks)), dom



# ----------------------------
# Phase 2 targeted eigenpair zones (v25.2)
# ----------------------------

IndexInterval = Tuple[int, int]


def parse_index_intervals(spec: Optional[str], d: int) -> List[IndexInterval]:
    """
    Parse a comma-separated inclusive index-range specification such as:
        "0-511"
        "0-127,384-511"

    The interval refers to the *left* index i of a neighboring eigenpair (i, i+1).
    Valid left indices are therefore 0..d-2.

    A bare integer such as "200" is accepted as the singleton interval (200, 200).
    """
    if spec is None:
        return []

    raw = str(spec).strip()
    if raw == "":
        return []

    hi_valid = max(0, int(d) - 2)
    out: List[IndexInterval] = []

    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue

        if "-" in token:
            a_s, b_s = token.split("-", 1)
            a = int(a_s.strip())
            b = int(b_s.strip())
        else:
            a = b = int(token)

        if a > b:
            a, b = b, a

        # Clamp to legal neighboring-pair left indices.
        a = max(0, min(hi_valid, a))
        b = max(0, min(hi_valid, b))

        if a <= b:
            out.append((a, b))

    # Merge overlapping/adjacent intervals.
    out.sort()
    merged: List[IndexInterval] = []
    for a, b in out:
        if not merged or a > merged[-1][1] + 1:
            merged.append((a, b))
        else:
            pa, pb = merged[-1]
            merged[-1] = (pa, max(pb, b))
    return merged


def left_index_in_intervals(i: int, intervals: List[IndexInterval]) -> bool:
    """Return True when left eigenpair index i is inside at least one inclusive interval."""
    if not intervals:
        return True
    ii = int(i)
    return any(a <= ii <= b for a, b in intervals)


def format_intervals(intervals: List[IndexInterval]) -> str:
    if not intervals:
        return "FULL"
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in intervals)


def parse_anchor_pair(spec: str) -> Tuple[int, int]:
    """
    Parse an adjacent anchor pair such as "157-158".
    The current continuation implementation is defined for neighbouring
    eigenpair anchors only.
    """
    raw = str(spec).strip()
    if "-" not in raw:
        raise ValueError("--anchor must be an adjacent pair such as 157-158")
    a_s, b_s = raw.split("-", 1)
    a, b = int(a_s.strip()), int(b_s.strip())
    if b < a:
        a, b = b, a
    if b != a + 1:
        raise ValueError("--anchor currently requires adjacent indices, e.g. 157-158")
    return a, b


def anchor_descendant_centers(anchor_left: int, from_qubits: int, to_qubits: int) -> List[int]:
    """
    Predict the direct computational-basis descendants of an n-qubit
    neighbouring-pair anchor after embedding into a larger Hilbert space.

    For one extra qubit this yields two branches:
        i
        i + 2**from_qubits

    For k extra qubits it yields 2**k branches separated by 2**from_qubits.
    These are priors, not claims that the successor must lie exactly there.
    """
    if to_qubits < from_qubits:
        raise ValueError("target n_qubits must be >= anchor_from_qubits")
    base_dim = 2 ** int(from_qubits)
    branches = 2 ** int(to_qubits - from_qubits)
    return [int(anchor_left) + j * base_dim for j in range(branches)]


def windows_around_centers(centers: List[int], radius: int, d: int) -> List[IndexInterval]:
    hi_valid = max(0, int(d) - 2)
    r = max(0, int(radius))
    out: List[IndexInterval] = []
    for c in centers:
        cc = max(0, min(hi_valid, int(c)))
        out.append((max(0, cc - r), min(hi_valid, cc + r)))
    # Reuse parser-style merge logic.
    out.sort()
    merged: List[IndexInterval] = []
    for a, b in out:
        if not merged or a > merged[-1][1] + 1:
            merged.append((a, b))
        else:
            pa, pb = merged[-1]
            merged[-1] = (pa, max(pb, b))
    return merged


def intervals_overlap(x: IndexInterval, y: IndexInterval) -> bool:
    return not (x[1] < y[0] or y[1] < x[0])


def parse_int_offsets(spec: str) -> List[int]:
    """Parse comma-separated signed integer offsets, e.g. '-64,64'."""
    vals: List[int] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    if not vals:
        raise ValueError("--control_offsets must contain at least one signed integer offset")
    if len(set(vals)) != len(vals):
        raise ValueError("--control_offsets contains duplicate offsets")
    return vals


def paired_control_sets(
    anchor_left: int,
    from_qubits: int,
    to_qubits: int,
    offsets: List[int],
    radius: int,
    d: int,
    anchor_windows: List[IndexInterval],
) -> List[List[IndexInterval]]:
    """
    Build geometry-matched control sets.

    Each control set starts from a source-space anchor-left index shifted by a
    fixed offset at anchor_from_qubits, then applies the *same* descendant
    embedding rule as the real anchor.

    Example, 8Q -> 9Q, anchor_left=157, offsets [-64,+64]:
      control source 93  -> descendant centers [93,349]
      control source 221 -> descendant centers [221,477]

    With radius=16 this gives:
      set 1: 77-109,333-365
      set 2: 205-237,461-493

    Thus every control set has the same number of windows, same window width,
    and same branch separation as the predicted anchor set.
    """
    source_hi = (2 ** int(from_qubits)) - 2
    sets: List[List[IndexInterval]] = []

    for off in offsets:
        source_left = int(anchor_left) + int(off)
        if source_left < 0 or source_left > source_hi:
            raise ValueError(
                f"Control offset {off:+d} moves source left-index to {source_left}, "
                f"outside legal {from_qubits}Q range 0-{source_hi}."
            )

        centers = anchor_descendant_centers(
            anchor_left=source_left,
            from_qubits=int(from_qubits),
            to_qubits=int(to_qubits),
        )
        windows = windows_around_centers(centers, int(radius), d)

        # Require exact matched cardinality: one window per descendant branch.
        expected_branches = 2 ** int(to_qubits - from_qubits)
        if len(windows) != expected_branches:
            raise ValueError(
                f"Control offset {off:+d} produced merged/clipped windows ({windows}); "
                "choose a smaller --anchor_radius or a different offset."
            )

        # Controls must not overlap the predicted anchor windows.
        if any(intervals_overlap(w, a) for w in windows for a in anchor_windows):
            raise ValueError(
                f"Control offset {off:+d} overlaps the predicted anchor windows. "
                "Choose a larger-magnitude offset."
            )

        # Control sets must not overlap one another.
        for prev in sets:
            if any(intervals_overlap(w, p) for w in windows for p in prev):
                raise ValueError(
                    f"Control offset {off:+d} overlaps another control set. "
                    "Choose more widely separated offsets."
                )

        sets.append(windows)

    return sets


# ----------------------------
# Signatures / Candidates
# ----------------------------

SigAbs = Tuple[int, int, int, int]             # (dom, ent_bin_i, ent_bin_j, leak_bin)
SigQGlobal = Tuple[int, int, int, int]         # (dom, ent_q_lo, ent_q_hi, leak_bin)
SigQGCoarse = Tuple[int, int, int, int]        # (dom_bin, ent_q_lo, ent_q_hi, leak_bin_coarse)


def signature_key_abs(dom: int, ent_i: float, ent_j: float, leak: float, ent_step: float, leak_step: float) -> SigAbs:
    bi = bin_index(ent_i, ent_step)
    bj = bin_index(ent_j, ent_step)
    lo, hi = (bi, bj) if bi <= bj else (bj, bi)
    lb = bin_index(leak, leak_step)
    return (int(dom), int(lo), int(hi), int(lb))


def signature_key_q_global(dom: int, ent_i: float, ent_j: float, leak: float, edges: np.ndarray, q_bins: int, leak_step: float) -> SigQGlobal:
    qi = quantile_bin(ent_i, edges, q_bins)
    qj = quantile_bin(ent_j, edges, q_bins)
    lo, hi = (qi, qj) if qi <= qj else (qj, qi)
    lb = bin_index(leak, leak_step)
    return (int(dom), int(lo), int(hi), int(lb))


def dom_to_bin_ratio(dom: int, d: int) -> int:
    """
    Dimension-aware coarse binning for dom_count using dom/d ratio.
    Bins: [0..0.25], (0.25..0.5], (0.5..0.75], (0.75..1.0]
    Returns an integer in {0,1,2,3}.
    """
    dom = int(max(1, dom))
    d = int(max(1, d))
    r = float(dom) / float(d)
    if r <= 0.25:
        return 0
    if r <= 0.50:
        return 1
    if r <= 0.75:
        return 2
    return 3


def leak_bin_coarse_from_fine(leak_bin: int) -> int:
    # Collapse fine leakage bins: 0 -> 0, 1 -> 1, >=2 -> 2
    lb = int(leak_bin)
    if lb <= 0:
        return 0
    if lb == 1:
        return 1
    return 2


def signature_key_qg_coarse(
    dom: int,
    ent_i: float,
    ent_j: float,
    leak: float,
    d: int,
    edges: np.ndarray,
    q_bins_coarse: int,
    leak_step_fine: float
) -> SigQGCoarse:
    qi = quantile_bin(ent_i, edges, q_bins_coarse)
    qj = quantile_bin(ent_j, edges, q_bins_coarse)
    lo, hi = (qi, qj) if qi <= qj else (qj, qi)

    lb_fine = bin_index(leak, leak_step_fine)
    lb = leak_bin_coarse_from_fine(lb_fine)
    return (int(dom_to_bin_ratio(dom, d)), int(lo), int(hi), int(lb))


@dataclass(frozen=True)
class Candidate:
    seed: int
    model: str
    i: int
    j: int
    delta_e: float
    gap_out: float
    iso_log: float
    ent_i: float
    ent_j: float
    dom: int
    leak: float
    score: float
    sig_abs: SigAbs
    sig_qg: SigQGlobal
    sig_qg_coarse: SigQGCoarse


def interestingness_score(dom: int, leak: float, dom_weight: float = 0.6, leak_weight: float = 0.4) -> float:
    dom = max(1, int(dom))
    dom_term = 1.0 / (1.0 + float(dom))
    leak_term = 1.0 - float(leak)
    s = dom_weight * dom_term + leak_weight * leak_term
    return float(max(0.0, min(1.0, s)))


def find_neighbor_pairs(evals: np.ndarray, eps: float) -> List[Tuple[int, int, float]]:
    pairs = []
    for k in range(len(evals) - 1):
        de = float(abs(evals[k + 1] - evals[k]))
        if de < float(eps):
            pairs.append((k, k + 1, de))
    return pairs


def generate_candidates_for_seed(
    model: str,
    seed: int,
    cache: PauliCache,
    n_terms: int,
    eps_neighbor: float,
    ent_step: float,
    leak_step: float,
    times: List[float],
    keep_mass: float,
    iso_eps: float,
    target_intervals: Optional[List[IndexInterval]] = None,
) -> List[Candidate]:
    """
    Generate candidates for one seed under REAL or NULL_HAAR_BASIS (spectrum-matched).
    Quantile signatures are assigned later at the batch level (pooled edges).
    """
    H_real = build_random_pauli_hamiltonian_cached(cache, n_terms, seed)

    if model == "REAL":
        H = H_real
        evals, evecs = np.linalg.eigh(H)
    elif model == "NULL_HAAR_BASIS":
        evals_real, _ = np.linalg.eigh(H_real)
        H = null_haar_basis_hamiltonian(evals_real, seed=seed)
        evals, evecs = np.linalg.eigh(H)
    else:
        raise ValueError(f"Unknown model: {model}")

    pairs = find_neighbor_pairs(evals, eps_neighbor)

    # Phase 2 v25.2 targeted mode:
    # restrict analysis to neighboring eigenpairs whose left index i lies
    # inside one of the requested target intervals.  The eigensystem itself
    # is deliberately unchanged, preserving the v25.1 REAL/NULL physics.
    if target_intervals:
        pairs = [(i, j, de) for (i, j, de) in pairs if left_index_in_intervals(i, target_intervals)]

    if not pairs:
        return []

    cands: List[Candidate] = []
    for (i, j, de) in pairs:
        # Gap/isolation metrics for neighbor pair (i,i+1)
        left_gap = float(abs(evals[i] - evals[i-1])) if i > 0 else float('inf')
        right_gap = float(abs(evals[j+1] - evals[j])) if (j + 1) < len(evals) else float('inf')
        gap_out = min(left_gap, right_gap)
        if not np.isfinite(gap_out):
            gap_out = left_gap if np.isfinite(left_gap) else (right_gap if np.isfinite(right_gap) else 0.0)
        denom = max(float(de), float(iso_eps))
        iso_log = float(math.log10(gap_out / denom)) if (gap_out > 0.0 and denom > 0.0) else float('nan')
        ent_i = amplitude_entropy_bits(evecs[:, i])
        ent_j = amplitude_entropy_bits(evecs[:, j])
        leak, dom = leakage_proxy_fast(evals, evecs, i, j, times, keep_mass=keep_mass)
        score = interestingness_score(dom, leak)
        sig_abs = signature_key_abs(dom, ent_i, ent_j, leak, ent_step, leak_step)
        cands.append(
            Candidate(
                seed=seed,
                model=model,
                i=i, j=j, delta_e=de,
                gap_out=gap_out, iso_log=iso_log,
                ent_i=ent_i, ent_j=ent_j,
                dom=dom, leak=leak, score=score,
                sig_abs=sig_abs,
                sig_qg=(0, 0, 0, 0),           # filled later
                sig_qg_coarse=(0, 0, 0, 0),    # filled later
            )
        )
    return cands


def assign_quantile_signatures(
    cands: List[Candidate],
    d: int,
    edges_fine: np.ndarray,
    q_bins_fine: int,
    edges_coarse: np.ndarray,
    q_bins_coarse: int,
    leak_step: float
) -> List[Candidate]:
    out: List[Candidate] = []
    for c in cands:
        sig_qg = signature_key_q_global(c.dom, c.ent_i, c.ent_j, c.leak, edges_fine, q_bins_fine, leak_step)
        sig_qg_coarse = signature_key_qg_coarse(c.dom, c.ent_i, c.ent_j, c.leak, d, edges_coarse, q_bins_coarse, leak_step)
        out.append(Candidate(**{**c.__dict__, "sig_qg": sig_qg, "sig_qg_coarse": sig_qg_coarse}))
    return out


# ----------------------------
# Enrichment + batch summaries
# ----------------------------

@dataclass(frozen=True)
class FamRow:
    sig: Tuple[int, int, int, int]
    overall: int
    stable: int
    expected: float
    p_tail: float
    neglog10_p: float
    enrichment: float

# ----------------------------
# Perturbation robustness (v21.1)
# ----------------------------

def _subspace_overlap_2d(U: np.ndarray, Up: np.ndarray) -> float:
    """Basis-invariant overlap between 2D subspaces span(U) and span(Up)."""
    # overlap = 0.5 * Tr(P P') = 0.5 * ||U^† U'||_F^2 for orthonormal columns.
    M = U.conj().T @ Up
    return float(0.5 * np.sum(np.abs(M) ** 2))


def _best_match_neighbor_pair(evals_p: np.ndarray, Ei: float, Ej: float) -> Tuple[int, int, float]:
    """Pick k,k+1 in perturbed spectrum that best matches (Ei,Ej) (allow swap)."""
    d = int(evals_p.size)
    if d < 2:
        return 0, 0, float("inf")
    best_k = 0
    best_cost = float("inf")
    for k in range(d - 1):
        a = float(evals_p[k])
        b = float(evals_p[k + 1])
        c1 = (a - Ei) ** 2 + (b - Ej) ** 2
        c2 = (a - Ej) ** 2 + (b - Ei) ** 2
        c = c1 if c1 <= c2 else c2
        if c < best_cost:
            best_cost = c
            best_k = k
    gap = float(abs(float(evals_p[best_k + 1]) - float(evals_p[best_k])))
    return int(best_k), int(best_k + 1), gap


def perturbation_robustness_summary_for_batch(
    *,
    cache: PauliCache,
    seed_start: int,
    n_seeds: int,
    n_terms: int,
    perturb_terms: Optional[int],
    iso_eps: float,
    eps_neighbor: float,
    times: List[float],
    keep_mass: float,
    ent_step: float,
    leak_step: float,
    edges_coarse: np.ndarray,
    q_bins_coarse: int,
    stable_frac: float,
    eta_list: List[float],
    reps: int,
    pairs_per_seed: int,
) -> Dict[float, Dict[str, Dict[str, float]]]:
    """\
    Compute perturbation robustness summaries for both REAL and spectrum-matched Haar NULL.

    v19 change: Robustness is quantified with a *basis-invariant* 2D subspace overlap.

    For each base seed we construct:
      H_REAL (Pauli-sum) and its eigensystem (E, V).
      H_NULL = U diag(E) U^† where U is Haar-random (seeded) (spectrum-matched).

    For each eta, rep we construct a small Hermitian perturbation ΔH (Pauli-sum, seeded) and apply
    the same ΔH in the computational basis:
      H_REAL' = H_REAL + eta * ΔH
      H_NULL' = H_NULL + eta * ΔH
    Then we re-diagonalize both. Each baseline neighbor-pair manifold (i,i+1) is tracked to the
    best-matching *neighbor* pair (k,k+1) in the perturbed spectrum by energy proximity.

    Primary outputs per eta, per model:
      - pair_retention_rate: fraction with matched neighbor gap < eps_neighbor
      - subspace_overlap_(mean/p10/p50/p90): overlap between baseline and perturbed 2D subspaces
    Secondary outputs (kept for context): coarse-signature retention and feature drifts.
    """
    d = cache.d
    perturb_terms_eff = int(perturb_terms) if perturb_terms is not None else int(n_terms)

    # accumulators: eta -> model -> lists/sums
    out: Dict[float, Dict[str, Dict[str, object]]] = {}
    for eta in eta_list:
        out[float(eta)] = {
            "REAL": {
                "pairs": 0.0,
                "retained": 0.0,
                "sig_retained": 0.0,
                "d_ent": 0.0,
                "d_leak": 0.0,
                "ov_all": [],
                "ov_ret": [],
                "logR_ret": [],
                "gap_out_ret": [],
            },
            "NULL": {
                "pairs": 0.0,
                "retained": 0.0,
                "sig_retained": 0.0,
                "d_ent": 0.0,
                "d_leak": 0.0,
                "ov_all": [],
                "ov_ret": [],
                "logR_ret": [],
                "gap_out_ret": [],
            },
        }

    if not eta_list or reps <= 0 or n_seeds <= 0:
        return out

    for s in range(int(n_seeds)):
        seed = int(seed_start + s)

        # Base REAL spectrum/eigenvectors
        H_real = build_random_pauli_hamiltonian_cached(cache, n_terms, seed)
        evals_base, evecs_real_base = np.linalg.eigh(H_real)

        # Base NULL eigenvectors (Haar basis) with same eigenvalues, and the explicit NULL Hamiltonian
        _, evecs_null_base = null_haar_basis_eigs(evals_base, seed=seed, tag="NULL_HAAR_BASIS")
        H_null = evecs_null_base @ np.diag(evals_base) @ evecs_null_base.conj().T
        H_null = 0.5 * (H_null + H_null.conj().T)

        pairs = find_neighbor_pairs(evals_base, eps_neighbor)
        if not pairs:
            continue

        # Precompute base candidate features for both models; optionally select top pairs_per_seed by score.
        # We additionally store the baseline 2D subspace basis U (d x 2) for overlap computations.
        base_lists: Dict[str, List[Tuple[int,int,float,float,int,float,SigQGCoarse,np.ndarray]]] = {"REAL": [], "NULL": []}
        for model, evecs in [("REAL", evecs_real_base), ("NULL", evecs_null_base)]:
            feats: List[Tuple[int,int,float,float,int,float,SigQGCoarse,float,np.ndarray]] = []
            for (i, j, _de) in pairs:
                ent_i = amplitude_entropy_bits(evecs[:, i])
                ent_j = amplitude_entropy_bits(evecs[:, j])
                leak, dom = leakage_proxy_fast(evals_base, evecs, i, j, times, keep_mass=keep_mass)
                score = interestingness_score(dom, leak)
                sig_c = signature_key_qg_coarse(dom, ent_i, ent_j, leak, d, edges_coarse, q_bins_coarse, leak_step)
                U = np.column_stack([evecs[:, i], evecs[:, j]])
                left_gap0 = float(abs(evals_base[i] - evals_base[i-1])) if i > 0 else float('inf')
                right_gap0 = float(abs(evals_base[j+1] - evals_base[j])) if (j + 1) < len(evals_base) else float('inf')
                gap_out0 = min(left_gap0, right_gap0)
                if not np.isfinite(gap_out0):
                    gap_out0 = left_gap0 if np.isfinite(left_gap0) else (right_gap0 if np.isfinite(right_gap0) else 0.0)
                denom0 = max(float(_de), float(iso_eps))
                iso_log0 = float(math.log10(gap_out0 / denom0)) if (gap_out0 > 0.0 and denom0 > 0.0) else float('nan')
                feats.append((i, j, ent_i, ent_j, dom, leak, sig_c, score, gap_out0, iso_log0, U))
            feats.sort(key=lambda t: t[-2])  # lower score = more "stable/interesting"
            take = int(max(1, min(int(pairs_per_seed), len(feats))))
            base_lists[model] = [(i, j, ent_i, ent_j, dom, leak, sig_c, gap_out0, iso_log0, U) for (i, j, ent_i, ent_j, dom, leak, sig_c, _score, gap_out0, iso_log0, U) in feats[:take]]

        for eta in eta_list:
            eta = float(eta)
            for rep in range(int(reps)):
                # Perturbation ΔH: deterministic Pauli-sum, applied identically to REAL and NULL.
                seed_pert = stable_hash_int(f"PERT|{seed}|eta{eta:.6g}|rep{rep}")
                dH = build_random_pauli_hamiltonian_cached(cache, perturb_terms_eff, seed_pert)

                H_real_p = 0.5 * (H_real + (eta * dH) + (H_real + (eta * dH)).conj().T)
                evals_real_p, evecs_real_p = np.linalg.eigh(H_real_p)

                H_null_p = 0.5 * (H_null + (eta * dH) + (H_null + (eta * dH)).conj().T)
                evals_null_p, evecs_null_p = np.linalg.eigh(H_null_p)

                for model, evals_p, evecs_p, base_items in [
                    ("REAL", evals_real_p, evecs_real_p, base_lists["REAL"]),
                    ("NULL", evals_null_p, evecs_null_p, base_lists["NULL"]),
                ]:
                    for (i, j, ent_i0, ent_j0, _dom0, leak0, sig0, gap_out0, iso_log0, U0) in base_items:
                        Ei = float(evals_base[i]); Ej = float(evals_base[j])
                        k, l, gap = _best_match_neighbor_pair(evals_p, Ei, Ej)
                        if k == l:
                            continue
                        retained = (gap < float(eps_neighbor))

                        # perturbed features
                        ent_i1 = amplitude_entropy_bits(evecs_p[:, k])
                        ent_j1 = amplitude_entropy_bits(evecs_p[:, l])
                        leak1, dom1 = leakage_proxy_fast(evals_p, evecs_p, k, l, times, keep_mass=keep_mass)
                        sig1 = signature_key_qg_coarse(dom1, ent_i1, ent_j1, leak1, d, edges_coarse, q_bins_coarse, leak_step)

                        U1 = np.column_stack([evecs_p[:, k], evecs_p[:, l]])
                        ov = _subspace_overlap_2d(U0, U1)

                        acc = out[eta][model]
                        acc["pairs"] = float(acc["pairs"]) + 1.0
                        acc["retained"] = float(acc["retained"]) + (1.0 if retained else 0.0)
                        acc["sig_retained"] = float(acc["sig_retained"]) + (1.0 if (sig1 == sig0) else 0.0)
                        acc["d_ent"] = float(acc["d_ent"]) + 0.5 * (abs(ent_i1 - ent_i0) + abs(ent_j1 - ent_j0))
                        acc["d_leak"] = float(acc["d_leak"]) + abs(leak1 - leak0)
                        acc["ov_all"].append(float(ov))
                        if retained:
                            acc["ov_ret"].append(float(ov))
                            acc["logR_ret"].append(float(iso_log0))
                            acc["gap_out_ret"].append(float(gap_out0))

    # finalize to rates/means + overlap quantiles
    def _q(vals: List[float], q: float) -> float:
        if not vals:
            return 0.0
        return float(np.quantile(np.array(vals, dtype=float), q))

    for eta in eta_list:
        eta = float(eta)
        for model in ["REAL", "NULL"]:
            acc = out[eta][model]
            n = max(1.0, float(acc["pairs"]))
            acc["pair_retention_rate"] = float(acc["retained"]) / n
            acc["sig_coarse_retention_rate"] = float(acc["sig_retained"]) / n
            acc["mean_abs_d_entropy_bits"] = float(acc["d_ent"]) / n
            acc["mean_abs_d_leak"] = float(acc["d_leak"]) / n
            acc["pairs_total"] = float(acc["pairs"])

            ov_all = list(acc["ov_all"])
            ov_ret = list(acc["ov_ret"])
            acc["subspace_overlap_all_mean"] = float(np.mean(ov_all)) if ov_all else 0.0
            acc["subspace_overlap_all_p10"] = _q(ov_all, 0.10)
            acc["subspace_overlap_all_p50"] = _q(ov_all, 0.50)
            acc["subspace_overlap_all_p90"] = _q(ov_all, 0.90)

            acc["subspace_overlap_retained_mean"] = float(np.mean(ov_ret)) if ov_ret else 0.0
            acc["subspace_overlap_retained_p10"] = _q(ov_ret, 0.10)
            acc["subspace_overlap_retained_p50"] = _q(ov_ret, 0.50)
            acc["subspace_overlap_retained_p90"] = _q(ov_ret, 0.90)
            acc["retained_pairs_total"] = float(len(ov_ret))

            # remove raw sums/lists to keep output clean

            logR_ret = [float(x) for x in acc.get("logR_ret", []) if np.isfinite(x)]
            gap_out_ret = [float(x) for x in acc.get("gap_out_ret", []) if np.isfinite(x)]
            acc["iso_log_ret_mean"] = float(np.mean(logR_ret)) if logR_ret else float('nan')
            acc["iso_log_ret_p10"] = _q(logR_ret, 0.10) if logR_ret else float('nan')
            acc["iso_log_ret_p50"] = _q(logR_ret, 0.50) if logR_ret else float('nan')
            acc["iso_log_ret_p90"] = _q(logR_ret, 0.90) if logR_ret else float('nan')

            # logR quartile table on retained subset: overlap statistics by logR quartile
            _pairs_lr_ov = [(float(lr), float(ov)) for lr, ov in zip(acc.get("logR_ret", []), ov_ret) if np.isfinite(lr) and np.isfinite(ov)]
            edges_q = [float("nan"), float("nan"), float("nan")]
            quart_rows = []
            if len(_pairs_lr_ov) >= 8:
                lrs = np.array([p[0] for p in _pairs_lr_ov], dtype=float)
                ovs = np.array([p[1] for p in _pairs_lr_ov], dtype=float)
                q25, q50, q75 = np.quantile(lrs, [0.25, 0.5, 0.75])
                edges_q = [float(q25), float(q50), float(q75)]
                bins = [(-np.inf, q25), (q25, q50), (q50, q75), (q75, np.inf)]
                labels = ["Q1", "Q2", "Q3", "Q4"]
                for (lo, hi), lab in zip(bins, labels):
                    if lo == -np.inf:
                        msk = (lrs <= hi)
                    else:
                        msk = (lrs > lo) & (lrs <= hi)
                    ov_bin = ovs[msk]
                    if ov_bin.size == 0:
                        quart_rows.append({"q": lab, "n": 0, "ov_p10": float("nan"), "ov_p50": float("nan"), "ov_p90": float("nan")})
                    else:
                        quart_rows.append({
                            "q": lab,
                            "n": int(ov_bin.size),
                            "ov_p10": float(np.quantile(ov_bin, 0.10)),
                            "ov_p50": float(np.quantile(ov_bin, 0.50)),
                            "ov_p90": float(np.quantile(ov_bin, 0.90)),
                        })
            acc["logR_quartile_edges"] = edges_q
            acc["logR_quartiles_ret"] = quart_rows

            def _corr(a: List[float], b: List[float]) -> float:
                if len(a) < 2 or len(b) < 2 or len(a) != len(b):
                    return float('nan')
                aa = np.array(a, dtype=float)
                bb = np.array(b, dtype=float)
                m = np.isfinite(aa) & np.isfinite(bb)
                if int(np.sum(m)) < 2:
                    return float('nan')
                return float(np.corrcoef(aa[m], bb[m])[0, 1])

            # correlations on the retained subset (aligned pairs)
            _lr_pairs = [(float(lr), float(ov)) for lr, ov in zip(acc.get("logR_ret", []), ov_ret) if np.isfinite(lr) and np.isfinite(ov)]
            _go_pairs = [(float(go), float(ov)) for go, ov in zip(acc.get("gap_out_ret", []), ov_ret) if np.isfinite(go) and np.isfinite(ov)]
            acc["corr_iso_log_ov_ret"] = _corr([p[0] for p in _lr_pairs], [p[1] for p in _lr_pairs]) if _lr_pairs else float("nan")
            acc["corr_gap_out_ov_ret"] = _corr([p[0] for p in _go_pairs], [p[1] for p in _go_pairs]) if _go_pairs else float("nan")

            # remove raw sums/lists to keep output clean
            for k in ["pairs", "retained", "sig_retained", "d_ent", "d_leak", "ov_all", "ov_ret"]:
                acc.pop(k, None)

    # type-ignore: nested dict contains floats only after finalization
    return out  # type: ignore[return-value]




def family_rows(
    sigs: List[Tuple[int, int, int, int]],
    stable_mask: np.ndarray,
    alpha: float,
) -> Tuple[List[FamRow], Dict[Tuple[int, int, int, int], Tuple[int, int]], float]:
    overall = Counter(sigs)
    stable = Counter([sigs[i] for i in range(len(sigs)) if stable_mask[i]])

    n_all = len(sigs)
    n_stable = int(np.sum(stable_mask))
    stable_rate = n_stable / max(1, n_all)

    K = max(1, len(overall))
    rows: List[FamRow] = []
    counts: Dict[Tuple[int, int, int, int], Tuple[int, int]] = {}

    for sig, o in overall.items():
        st = stable.get(sig, 0)
        counts[sig] = (int(o), int(st))

        p_all = (o + alpha) / (n_all + alpha * K)
        p_st = (st + alpha) / (n_stable + alpha * K)
        enr = (p_st / p_all) / max(1e-12, stable_rate)

        p_tail = binom_tail_ge(int(st), int(o), stable_rate) if o > 0 else 1.0
        p_tail = max(1e-300, min(1.0, float(p_tail)))
        neglog10 = -math.log10(p_tail)

        rows.append(
            FamRow(
                sig=sig,
                overall=int(o),
                stable=int(st),
                expected=float(o) * stable_rate,
                p_tail=float(p_tail),
                neglog10_p=float(neglog10),
                enrichment=float(enr),
            )
        )

    rows.sort(key=lambda r: (-r.neglog10_p, -r.enrichment, -r.stable, -r.overall))
    return rows, counts, stable_rate


def top_k_families(rows: List[FamRow], k: int, min_overall: int, min_stable: int, p_tail_max: Optional[float]) -> List[FamRow]:
    out: List[FamRow] = []
    for r in rows:
        if r.overall >= min_overall and r.stable >= min_stable:
            if p_tail_max is None or r.p_tail <= float(p_tail_max):
                out.append(r)
        if len(out) >= k:
            break
    return out


def summarize_entropy_effect(ent_vals: np.ndarray, stable_mask: np.ndarray, rng: np.random.Generator, B: int = 200) -> Tuple[float, Tuple[float, float]]:
    all_med = float(np.median(ent_vals))
    st_med = float(np.median(ent_vals[stable_mask])) if np.any(stable_mask) else all_med
    eff = float(st_med - all_med)

    n = ent_vals.size
    idx_all = np.arange(n)
    effects = []
    for _ in range(int(B)):
        samp = rng.choice(idx_all, size=n, replace=True)
        samp_vals = ent_vals[samp]
        samp_mask = stable_mask[samp]
        all_m = float(np.median(samp_vals))
        st_m = float(np.median(samp_vals[samp_mask])) if np.any(samp_mask) else all_m
        effects.append(st_m - all_m)
    lo, hi = np.quantile(np.array(effects, dtype=float), [0.025, 0.975])
    return eff, (float(lo), float(hi))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return 0.0
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    pooled = math.sqrt(max(1e-12, ((x.size - 1) * vx + (y.size - 1) * vy) / max(1, (x.size + y.size - 2))))
    return float((mx - my) / pooled)


def stable_mask_from_scores_and_leak(
    scores: np.ndarray,
    leaks: np.ndarray,
    stable_frac: float,
    stable_leak_max: Optional[float],
    stable_leak_quantile: Optional[float],
) -> np.ndarray:
    """
    Fixed-fraction stable selection per model (v15):

    Select exactly k = round(stable_frac * n) (minimum 1 if n>0) items per model,
    based on score (lower is better). Optionally impose a leakage constraint:

      - absolute: leak <= stable_leak_max
      - or quantile: leak <= quantile(leak, stable_leak_quantile)

    If a leakage constraint is provided:
      - If eligible (leak <= threshold) count >= k: pick the best k among eligible by score.
      - If eligible count < k: pick all eligible, then fill remainder from ineligible by (score, leak).
    """
    n = int(scores.size)
    if n == 0:
        return np.zeros((0,), dtype=bool)

    stable_frac = float(stable_frac)
    stable_frac = max(0.0, min(1.0, stable_frac))
    k = int(round(stable_frac * n))
    k = max(1, min(n, k))

    order = np.lexsort((leaks, scores))  # score primary, leak secondary

    if stable_leak_max is None and stable_leak_quantile is None:
        chosen = order[:k]
        mask = np.zeros((n,), dtype=bool)
        mask[chosen] = True
        return mask

    if stable_leak_max is not None:
        leak_thr = float(stable_leak_max)
    else:
        q = float(stable_leak_quantile) if stable_leak_quantile is not None else 1.0
        q = max(0.0, min(1.0, q))
        leak_thr = float(np.quantile(leaks, q))

    eligible = leaks <= leak_thr
    eligible_order = [i for i in order if eligible[i]]

    chosen: List[int] = []
    if len(eligible_order) >= k:
        chosen = eligible_order[:k]
    else:
        chosen = eligible_order[:]
        need = k - len(chosen)
        if need > 0:
            for i in order:
                if eligible[i]:
                    continue
                chosen.append(i)
                need -= 1
                if need == 0:
                    break

    mask = np.zeros((n,), dtype=bool)
    mask[chosen] = True
    return mask


# ----------------------------
# Batch runner (paired REAL + NULL)
# ----------------------------

@dataclass
class BatchResult:
    batch_id: int
    seed_offset: int
    model: str
    n_candidates: int
    stable_rate: float
    stable_rate_scoreonly: float
    score_stats: Tuple[float, float, float]
    leak_stats: Tuple[float, float, float]
    ent_stats: Tuple[float, float, float]
    dom_stats: Tuple[float, float, float]  # mean, median, max
    gap_in_stats: Tuple[float, float, float]      # mean, median, p90
    gap_in_cond_stats: Tuple[float, float, float] # conditional on ΔE_in > iso_eps
    gap_in_cond_n: int
    gap_out_stats: Tuple[float, float, float]     # mean, median, p90
    iso_log_stats: Tuple[float, float, float]     # mean, median, p90 of log10(ΔE_out/max(ΔE_in, eps))
    ent_pool: np.ndarray
    leak_vals: np.ndarray
    dom_vals: np.ndarray
    top_qg: List[FamRow]
    top_qg_coarse: List[FamRow]
    effect_entropy_bits: float
    effect_ci: Tuple[float, float]
    top_keys_qg: List[SigQGlobal]
    top_keys_qg_coarse: List[SigQGCoarse]
    counts_qg: Dict[SigQGlobal, Tuple[int, int]]
    counts_qg_coarse: Dict[SigQGCoarse, Tuple[int, int]]


def compute_batch_result(
    *,
    batch_id: int,
    seed_offset: int,
    model: str,
    cands: List[Candidate],
    stable_frac: float,
    stable_leak_max: Optional[float],
    stable_leak_quantile: Optional[float],
    topK: int,
    min_overall: int,
    min_stable: int,
    alpha: float,
    p_tail_max: Optional[float],
    bootstrap: int,
    iso_eps: float,
) -> BatchResult:
    if not cands:
        return BatchResult(
            batch_id=batch_id, seed_offset=seed_offset, model=model,
            n_candidates=0,
            stable_rate=0.0, stable_rate_scoreonly=0.0,
            score_stats=(0.0, 0.0, 0.0),
            leak_stats=(0.0, 0.0, 0.0),
            ent_stats=(0.0, 0.0, 0.0),
            dom_stats=(0.0, 0.0, 0.0),
            gap_in_stats=(0.0, 0.0, 0.0),
            gap_in_cond_stats=(0.0, 0.0, 0.0), gap_in_cond_n=0,
            gap_out_stats=(0.0, 0.0, 0.0),
            iso_log_stats=(0.0, 0.0, 0.0),
            ent_pool=np.array([], dtype=float),
            leak_vals=np.array([], dtype=float),
            dom_vals=np.array([], dtype=float),
            top_qg=[], top_qg_coarse=[],
            effect_entropy_bits=0.0, effect_ci=(0.0, 0.0),
            top_keys_qg=[], top_keys_qg_coarse=[],
            counts_qg={}, counts_qg_coarse={},
        )

    scores = np.array([c.score for c in cands], dtype=float)
    leaks = np.array([c.leak for c in cands], dtype=float)
    doms = np.array([c.dom for c in cands], dtype=float)
    ent_pool = np.array([c.ent_i for c in cands] + [c.ent_j for c in cands], dtype=float)

    gap_in = np.array([c.delta_e for c in cands], dtype=float)
    gap_out = np.array([c.gap_out for c in cands], dtype=float)
    iso_log = np.array([c.iso_log for c in cands], dtype=float)
    # Robust statistics for gaps/isolations (finite-only)
    fin = np.isfinite(gap_in) & np.isfinite(gap_out) & np.isfinite(iso_log)
    gap_in_f = gap_in[fin]
    gap_out_f = gap_out[fin]
    iso_log_f = iso_log[fin]
    def _stats_mean_med_p90(a: np.ndarray) -> Tuple[float, float, float]:
        if a.size == 0:
            return (0.0, 0.0, 0.0)
        return (float(np.mean(a)), float(np.median(a)), float(np.quantile(a, 0.9)))
    gap_in_stats = _stats_mean_med_p90(gap_in_f)
    gap_out_stats = _stats_mean_med_p90(gap_out_f)
    iso_log_stats = _stats_mean_med_p90(iso_log_f)
    cond = gap_in_f > float(iso_eps)
    gap_in_cond = gap_in_f[cond]
    gap_in_cond_n = int(gap_in_cond.size)
    gap_in_cond_stats = _stats_mean_med_p90(gap_in_cond)

    mask_scoreonly = stable_mask_from_scores_and_leak(scores, leaks, stable_frac, None, None)
    stable_rate_scoreonly = float(np.sum(mask_scoreonly) / max(1, len(cands)))

    mask = stable_mask_from_scores_and_leak(scores, leaks, stable_frac, stable_leak_max, stable_leak_quantile)

    sigs_qg = [c.sig_qg for c in cands]
    rows_qg, counts_qg, stable_rate_qg = family_rows(sigs_qg, mask, alpha=alpha)
    top_qg = top_k_families(rows_qg, k=topK, min_overall=min_overall, min_stable=min_stable, p_tail_max=p_tail_max)

    sigs_qg_c = [c.sig_qg_coarse for c in cands]
    rows_qg_c, counts_qg_c, _ = family_rows(sigs_qg_c, mask, alpha=alpha)
    top_qg_c = top_k_families(rows_qg_c, k=topK, min_overall=min_overall, min_stable=min_stable, p_tail_max=p_tail_max)

    rng_eff = rng_from_seed(stable_hash_int(f"{model}|batch{batch_id}|eff"))
    eff, ci = summarize_entropy_effect(ent_pool, np.repeat(mask, 2), rng_eff, B=bootstrap)

    return BatchResult(
        batch_id=batch_id, seed_offset=seed_offset, model=model,
        n_candidates=len(cands),
        stable_rate=float(stable_rate_qg),
        stable_rate_scoreonly=float(stable_rate_scoreonly),
        score_stats=(float(scores.mean()), float(np.median(scores)), float(scores.max())),
        leak_stats=(float(leaks.mean()), float(np.median(leaks)), float(leaks.min())),
        ent_stats=(float(ent_pool.mean()), float(np.median(ent_pool)), float(ent_pool.max())),
        dom_stats=(float(doms.mean()), float(np.median(doms)), float(doms.max())),
        gap_in_stats=gap_in_stats,
        gap_in_cond_stats=gap_in_cond_stats, gap_in_cond_n=gap_in_cond_n,
        gap_out_stats=gap_out_stats,
        iso_log_stats=iso_log_stats,
        ent_pool=ent_pool,
        leak_vals=leaks,
        dom_vals=doms,
        top_qg=top_qg,
        top_qg_coarse=top_qg_c,
        effect_entropy_bits=eff,
        effect_ci=ci,
        top_keys_qg=[r.sig for r in top_qg],
        top_keys_qg_coarse=[r.sig for r in top_qg_c],
        counts_qg=counts_qg,
        counts_qg_coarse=counts_qg_c,
    )


def run_paired_batches(
    *,
    cache: PauliCache,
    n_terms: int,
    seeds_per_batch: int,
    n_batches: int,
    base_seed: int,
    batch_stride: int,
    eps_neighbor: float,
    ent_step: float,
    leak_step: float,
    times: List[float],
    keep_mass: float,
    iso_eps: float,
    stable_frac: float,
    stable_leak_max: Optional[float],
    stable_leak_quantile: Optional[float],
    topK: int,
    min_overall: int,
    min_stable: int,
    alpha: float,
    q_bins: int,
    q_bins_coarse: int,
    p_tail_max: Optional[float],
    bootstrap: int,
    perturb_eta: List[float],
    perturb_reps: int,
    perturb_seeds: int,
    perturb_terms: Optional[int],
    perturb_pairs_per_seed: int,
    # v22 open-system options
    open_system: bool,
    os_include_baselines: bool,
    os_pairs_per_batch: int,
    os_noise_model: str,
    os_gamma_phi: float,
    os_gamma_1: float,
    os_t_max: float,
    os_t_steps: int,
    os_dt_internal: Optional[float],
    os_states_mode: str,
    os_use_stable_pool: bool,
    os_logical_basis: str,
    os_noise_qubits: str,
    os_noise_subset: Optional[List[int]],
    os_report_quantiles: Tuple[float, float, float],
    os_time_mode: str,
    os_time_late_power: float,
    contrast_focus: bool,
    kernel_zoom: bool,
    kernel_top_centers: int,
    kernel_radius: int,
    kernel_min_neighbors: int,
    interference_test: bool,
    if_pairs_per_batch: int,
    if_use_stable_pool: bool,
    target_intervals: Optional[List[IndexInterval]],
) -> Tuple[List[BatchResult], List[BatchResult], List[Dict[str, float]], List[Dict[float, Dict[str, Dict[str, float]]]], List[Dict[str, Dict[str, object]]], List[Dict[str, Dict[str, object]]]]:
    real_results: List[BatchResult] = []
    null_results: List[BatchResult] = []
    scoreboards: List[Dict[str, float]] = []
    perturb_summaries: List[Dict[float, Dict[str, Dict[str, float]]]] = []
    open_system_summaries: List[Dict[str, Dict[str, object]]] = []
    interference_summaries: List[Dict[str, Dict[str, object]]] = []

    d = cache.d

    for b in range(n_batches):
        offset = base_seed + b * batch_stride
        t0 = time.time()

        real_cands: List[Candidate] = []
        null_cands: List[Candidate] = []

        for s in range(seeds_per_batch):
            seed = offset + s
            real_cands.extend(
                generate_candidates_for_seed(
                    model="REAL",
                    seed=seed,
                    cache=cache,
                    n_terms=n_terms,
                    eps_neighbor=eps_neighbor,
                    ent_step=ent_step,
                    leak_step=leak_step,
                    times=times,
                    keep_mass=keep_mass,
                    iso_eps=iso_eps,
                    target_intervals=target_intervals,
                )
            )
            null_cands.extend(
                generate_candidates_for_seed(
                    model="NULL_HAAR_BASIS",
                    seed=seed,
                    cache=cache,
                    n_terms=n_terms,
                    eps_neighbor=eps_neighbor,
                    ent_step=ent_step,
                    leak_step=leak_step,
                    times=times,
                    keep_mass=keep_mass,
                    iso_eps=iso_eps,
                    target_intervals=target_intervals,
                )
            )

        pooled_ent = np.array(
            [c.ent_i for c in real_cands] + [c.ent_j for c in real_cands] +
            [c.ent_i for c in null_cands] + [c.ent_j for c in null_cands],
            dtype=float
        )

        edges_fine = make_quantile_edges(pooled_ent, q_bins=q_bins) if pooled_ent.size else make_quantile_edges(np.array([0.0]), q_bins=q_bins)
        edges_coarse = make_quantile_edges(pooled_ent, q_bins=q_bins_coarse) if pooled_ent.size else make_quantile_edges(np.array([0.0]), q_bins=q_bins_coarse)

        real_cands = assign_quantile_signatures(real_cands, d, edges_fine, q_bins, edges_coarse, q_bins_coarse, leak_step)
        null_cands = assign_quantile_signatures(null_cands, d, edges_fine, q_bins, edges_coarse, q_bins_coarse, leak_step)

        # v19: perturbation robustness (optional; uses only a small prefix of seeds to control runtime)
        pert_summary: Dict[float, Dict[str, Dict[str, float]]] = {}
        if perturb_eta and int(perturb_seeds) > 0:
            seed_start = int(offset)
            n_use = int(min(int(perturb_seeds), int(seeds_per_batch)))
            pert_summary = perturbation_robustness_summary_for_batch(
                cache=cache,
                seed_start=seed_start,
                n_seeds=n_use,
                n_terms=n_terms,
                perturb_terms=perturb_terms,
                iso_eps=iso_eps,
                eps_neighbor=eps_neighbor,
                times=times,
                keep_mass=keep_mass,
                ent_step=ent_step,
                leak_step=leak_step,
                edges_coarse=edges_coarse,
                q_bins_coarse=q_bins_coarse,
                stable_frac=stable_frac,
                eta_list=list(perturb_eta),
                reps=int(perturb_reps),
                pairs_per_seed=int(perturb_pairs_per_seed),
            )
        perturb_summaries.append(pert_summary)

        # v22: open-system (Lindblad) evaluation (optional)
        os_summary: Dict[str, Dict[str, object]] = {}
        if bool(open_system):
            os_summary = open_system_summary_for_batch(
                include_baselines=bool(os_include_baselines),
                cache=cache,
                n_terms=n_terms,
                batch_id=b,
                seed_offset=offset,
                real_cands=real_cands,
                null_cands=null_cands,
                stable_frac=stable_frac,
                stable_leak_max=stable_leak_max,
                stable_leak_quantile=stable_leak_quantile,
                use_stable_pool=bool(os_use_stable_pool),
                os_pairs_per_batch=int(os_pairs_per_batch),
                noise_model=str(os_noise_model),
                gamma_phi=float(os_gamma_phi),
                gamma_1=float(os_gamma_1),
                t_max=float(os_t_max),
                t_steps=int(os_t_steps),
                dt_internal=os_dt_internal,
                states_mode=str(os_states_mode),
                q_report=tuple(os_report_quantiles),
                logical_basis=str(os_logical_basis),
                os_noise_qubits=str(os_noise_qubits),
                os_noise_subset=os_noise_subset,
                time_mode=str(os_time_mode),
                time_late_power=float(os_time_late_power),
                contrast_focus=bool(contrast_focus),
                kernel_zoom=bool(kernel_zoom),
                kernel_top_centers=int(kernel_top_centers),
                kernel_radius=int(kernel_radius),
                kernel_min_neighbors=int(kernel_min_neighbors),
            )
        open_system_summaries.append(os_summary)

        if_summary: Dict[str, Dict[str, object]] = {}
        if bool(interference_test):
            if_summary = interference_summary_for_batch(
                cache=cache,
                n_terms=n_terms,
                batch_id=b,
                seed_offset=offset,
                real_cands=real_cands,
                null_cands=null_cands,
                stable_frac=stable_frac,
                stable_leak_max=stable_leak_max,
                stable_leak_quantile=stable_leak_quantile,
                use_stable_pool=bool(if_use_stable_pool),
                if_pairs_per_batch=int(if_pairs_per_batch),
                q_report=(0.1, 0.5, 0.9),
            )
        interference_summaries.append(if_summary)

        elapsed = time.time() - t0
        print(f"Batch {b+1}/{n_batches} generated: REAL={len(real_cands)} NULL={len(null_cands)} (elapsed {elapsed:.1f}s)")

        R = compute_batch_result(
            batch_id=b,
            seed_offset=offset,
            model="REAL",
            cands=real_cands,
            stable_frac=stable_frac,
            stable_leak_max=stable_leak_max,
            stable_leak_quantile=stable_leak_quantile,
            topK=topK,
            min_overall=min_overall,
            min_stable=min_stable,
            alpha=alpha,
            p_tail_max=p_tail_max,
            bootstrap=bootstrap,
            iso_eps=iso_eps,
        )
        N = compute_batch_result(
            batch_id=b,
            seed_offset=offset,
            model="NULL_HAAR_BASIS",
            cands=null_cands,
            stable_frac=stable_frac,
            stable_leak_max=stable_leak_max,
            stable_leak_quantile=stable_leak_quantile,
            topK=topK,
            min_overall=min_overall,
            min_stable=min_stable,
            alpha=alpha,
            p_tail_max=p_tail_max,
            bootstrap=bootstrap,
            iso_eps=iso_eps,
        )

        real_results.append(R)
        null_results.append(N)

        sb = {}
        sb["delta_median_entropy_bits"] = safe_median(R.ent_pool) - safe_median(N.ent_pool)
        sb["delta_median_leak"] = safe_median(R.leak_vals) - safe_median(N.leak_vals)
        sb["delta_median_dom"] = safe_median(R.dom_vals) - safe_median(N.dom_vals)
        sb["entropy_cohens_d"] = cohens_d(R.ent_pool, N.ent_pool)
        scoreboards.append(sb)

    return real_results, null_results, scoreboards, perturb_summaries, open_system_summaries, interference_summaries



# ----------------------------
# Open-system (Lindblad) evaluation (v22)
# ----------------------------

def op_on_qubit(n_qubits: int, op2: np.ndarray, target: int) -> np.ndarray:
    """Embed a single-qubit operator op2 onto qubit 'target' (0-indexed, left-to-right)."""
    mats = []
    for q in range(n_qubits):
        mats.append(op2 if q == target else PAULI_MATS["I"])
    return kron_n(mats)

def build_noise_operators(
    n_qubits: int,
    noise_model: str,
    gamma_phi: float,
    gamma_1: float,
    subset: Optional[List[int]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build Lindblad jump operators L_k and precompute L_k^† L_k.

    noise_model:
      - 'dephasing'  : L_k = sqrt(gamma_phi) * Z_k
      - 'amp_damp'   : L_k = sqrt(gamma_1) * sigma_minus_k
      - 'both'       : union of the above
    """
    if subset is None:
        qubits = list(range(int(n_qubits)))
    else:
        qubits = [int(x) for x in subset]

    noise_model = str(noise_model).lower().strip()
    Ls: List[np.ndarray] = []

    # Single-qubit lowering operator sigma^- = |0><1|
    sig_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)

    if noise_model in ("dephasing", "both"):
        g = float(gamma_phi)
        if g > 0:
            for q in qubits:
                Ls.append(math.sqrt(g) * op_on_qubit(n_qubits, PAULI_MATS["Z"], q))

    if noise_model in ("amp_damp", "amplitude_damping", "both"):
        g = float(gamma_1)
        if g > 0:
            for q in qubits:
                Ls.append(math.sqrt(g) * op_on_qubit(n_qubits, sig_minus, q))

    if noise_model in ("depolar", "depolarizing"):
        # Simple depolarizing via Pauli jumps (not a true continuous-time depolarizing channel,
        # but a useful stress test). We distribute the rate equally among X,Y,Z on each qubit.
        g = float(gamma_phi)  # reuse gamma_phi as depolar rate
        if g > 0:
            per = g / 3.0
            for q in qubits:
                for p in ("X", "Y", "Z"):
                    Ls.append(math.sqrt(per) * op_on_qubit(n_qubits, PAULI_MATS[p], q))

    LLs = [L.conj().T @ L for L in Ls]
    return Ls, LLs


def lindblad_rhs(rho: np.ndarray, H: np.ndarray, Ls: List[np.ndarray], LLs: List[np.ndarray]) -> np.ndarray:
    """Compute d rho / dt for Lindblad master equation."""
    dr = -1j * (H @ rho - rho @ H)
    for L, LL in zip(Ls, LLs):
        dr += L @ rho @ L.conj().T - 0.5 * (LL @ rho + rho @ LL)
    return dr


def rk4_step(rho: np.ndarray, dt: float, H: np.ndarray, Ls: List[np.ndarray], LLs: List[np.ndarray]) -> np.ndarray:
    k1 = lindblad_rhs(rho, H, Ls, LLs)
    k2 = lindblad_rhs(rho + 0.5 * dt * k1, H, Ls, LLs)
    k3 = lindblad_rhs(rho + 0.5 * dt * k2, H, Ls, LLs)
    k4 = lindblad_rhs(rho + dt * k3, H, Ls, LLs)
    out = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # numerical hygiene
    out = 0.5 * (out + out.conj().T)
    tr = np.trace(out)
    if abs(tr) > 0:
        out = out / tr
    return out


def evolve_density_matrix(
    rho0: np.ndarray,
    H: np.ndarray,
    Ls: List[np.ndarray],
    LLs: List[np.ndarray],
    t_max: float,
    t_steps: int,
    dt_internal: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evolve rho0 from t=0..t_max over t_steps points.
    Returns (t_grid, rhos) where rhos has shape (t_steps, d, d).
    """
    t_steps = int(t_steps)
    if t_steps < 2:
        t_steps = 2
    t_grid = np.linspace(0.0, float(t_max), t_steps)
    dt = float(t_grid[1] - t_grid[0])
    if dt_internal is None:
        sub = 1
    else:
        sub = max(1, int(math.ceil(dt / float(dt_internal))))
    dt_sub = dt / sub

    d = H.shape[0]
    rhos = np.zeros((t_steps, d, d), dtype=complex)
    rho = rho0.copy()
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) > 0:
        rho = rho / tr
    rhos[0] = rho

    for k in range(1, t_steps):
        for _ in range(sub):
            rho = rk4_step(rho, dt_sub, H, Ls, LLs)
        rhos[k] = rho
    return t_grid, rhos


def pure_state_density(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    return np.outer(psi, psi.conj())


def logical_density_matrix(rho: np.ndarray, psi0: np.ndarray, psi1: np.ndarray) -> Tuple[np.ndarray, float]:
    """Project rho into span{psi0, psi1}; return normalized 2x2 logical density matrix and subspace weight."""
    psi0 = np.asarray(psi0, dtype=complex).reshape(-1)
    psi1 = np.asarray(psi1, dtype=complex).reshape(-1)
    P = np.outer(psi0, psi0.conj()) + np.outer(psi1, psi1.conj())
    w = float(np.real(np.trace(P @ rho)))
    w = float(np.clip(w, 0.0, 1.0))
    if w <= 1e-15:
        return np.zeros((2, 2), dtype=complex), 0.0
    rhoP = P @ rho @ P
    rhoP = 0.5 * (rhoP + rhoP.conj().T) / w
    a00 = np.vdot(psi0, rhoP @ psi0)
    a01 = np.vdot(psi0, rhoP @ psi1)
    a10 = np.vdot(psi1, rhoP @ psi0)
    a11 = np.vdot(psi1, rhoP @ psi1)
    rhoL = np.array([[a00, a01], [a10, a11]], dtype=complex)
    rhoL = 0.5 * (rhoL + rhoL.conj().T)
    tr = np.trace(rhoL)
    if abs(tr) > 1e-15:
        rhoL = rhoL / tr
    return rhoL, w


def _fix_phase(psi: np.ndarray) -> np.ndarray:
    """Deterministic global phase fix: rotate so the largest-magnitude component is real and non-negative."""
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    k = int(np.argmax(np.abs(psi)))
    a = psi[k]
    if abs(a) < 1e-15:
        return psi
    phase = np.exp(-1j * np.angle(a))
    psi2 = psi * phase
    # enforce non-negative real for the pivot element
    if np.real(psi2[k]) < 0:
        psi2 = -psi2
    return psi2


def noise_diagonal_logical_basis(
    psi0: np.ndarray,
    psi1: np.ndarray,
    Ls: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Choose a logical basis inside span{psi0, psi1} by diagonalizing the projected noise intensity:
        A = Σ_k (P L_k P)† (P L_k P)  (restricted to the 2D subspace)
    Returns (psi0_new, psi1_new) as orthonormal vectors.
    """
    # Orthonormal input basis assumed (eigenvectors).
    psi0 = np.asarray(psi0, dtype=complex).reshape(-1)
    psi1 = np.asarray(psi1, dtype=complex).reshape(-1)

    # Build 2x2 matrix A in the {psi0, psi1} basis.
    A = np.zeros((2, 2), dtype=complex)
    bra0 = psi0.conj()
    bra1 = psi1.conj()

    for L in Ls:
        # J = P L P restricted to subspace basis
        Lpsi0 = L @ psi0
        Lpsi1 = L @ psi1
        j00 = np.vdot(psi0, Lpsi0)
        j01 = np.vdot(psi0, Lpsi1)
        j10 = np.vdot(psi1, Lpsi0)
        j11 = np.vdot(psi1, Lpsi1)
        J = np.array([[j00, j01], [j10, j11]], dtype=complex)
        A += J.conj().T @ J

    # Hermitize for numerical stability
    A = 0.5 * (A + A.conj().T)

    # Eigen-decompose (ascending eigenvalues => first vector is "least noisy" in this proxy)
    w, U = np.linalg.eigh(A)

    # Compose new basis vectors in full Hilbert space
    u0 = U[:, 0]
    u1 = U[:, 1]
    psi0n = u0[0] * psi0 + u0[1] * psi1
    psi1n = u1[0] * psi0 + u1[1] * psi1

    # Normalize and fix phases deterministically
    psi0n = psi0n / max(np.linalg.norm(psi0n), 1e-15)
    psi1n = psi1n / max(np.linalg.norm(psi1n), 1e-15)
    psi0n = _fix_phase(psi0n)
    psi1n = _fix_phase(psi1n)

    # Re-orthonormalize (Gram-Schmidt) to suppress drift
    psi1n = psi1n - np.vdot(psi0n, psi1n) * psi0n
    psi1n = psi1n / max(np.linalg.norm(psi1n), 1e-15)
    psi1n = _fix_phase(psi1n)
    return psi0n, psi1n

def open_system_metrics_for_pair(
    *,
    H: np.ndarray,
    psi0: np.ndarray,
    psi1: np.ndarray,
    Ls: List[np.ndarray],
    LLs: List[np.ndarray],
    t_max: float,
    t_steps: int,
    dt_internal: Optional[float],
    states_mode: str,
) -> Dict[str, np.ndarray]:
    """Compute state- and structure-retention metrics for one 2D logical subspace."""
    psi0 = np.asarray(psi0, dtype=complex).reshape(-1)
    psi1 = np.asarray(psi1, dtype=complex).reshape(-1)
    P = np.outer(psi0, psi0.conj()) + np.outer(psi1, psi1.conj())

    test_states = [psi0, psi1]
    if str(states_mode).lower().strip() in ("zx", "z+x", "full", "4"):
        test_states += [
            (psi0 + psi1) / math.sqrt(2.0),
            (psi0 + 1j * psi1) / math.sqrt(2.0),
        ]

    t_grid = None
    leak_list = []
    subret_list = []
    fu_list = []
    fc_list = []
    coh_list = []
    pur_list = []
    drift_list = []

    for psi in test_states:
        rho0 = pure_state_density(psi)
        t_grid, rhos = evolve_density_matrix(
            rho0=rho0, H=H, Ls=Ls, LLs=LLs,
            t_max=t_max, t_steps=t_steps, dt_internal=dt_internal,
        )

        w = np.real(np.trace(P @ rhos, axis1=1, axis2=2))
        w = np.clip(w, 0.0, 1.0)
        leak = 1.0 - w
        subret = w.copy()

        fu = np.real(np.einsum("i,tij,j->t", psi.conj(), rhos, psi))
        fu = np.clip(fu, 0.0, 1.0)
        fc = np.clip(fu / np.clip(w, 1e-15, 1.0), 0.0, 1.0)

        coh = np.zeros_like(leak, dtype=float)
        pur = np.zeros_like(leak, dtype=float)
        for ti, rho in enumerate(rhos):
            rhoL, _ = logical_density_matrix(rho, psi0, psi1)
            coh[ti] = float(abs(rhoL[0, 1]))
            pur[ti] = float(np.real(np.trace(rhoL @ rhoL))) if rhoL.size else 0.0

        coh0 = float(coh[0])
        pur0 = float(pur[0])
        drift = leak + (1.0 - fc) + np.abs(coh - coh0) + np.abs(pur - pur0)

        leak_list.append(leak)
        subret_list.append(subret)
        fu_list.append(fu)
        fc_list.append(fc)
        coh_list.append(coh)
        pur_list.append(pur)
        drift_list.append(drift)

    return {
        "t": t_grid,
        "leak_mean": np.mean(np.stack(leak_list, axis=0), axis=0),
        "subspace_retention_mean": np.mean(np.stack(subret_list, axis=0), axis=0),
        "fid_uncond_mean": np.mean(np.stack(fu_list, axis=0), axis=0),
        "fid_cond_mean": np.mean(np.stack(fc_list, axis=0), axis=0),
        "logical_coherence_mean": np.mean(np.stack(coh_list, axis=0), axis=0),
        "logical_purity_mean": np.mean(np.stack(pur_list, axis=0), axis=0),
        "structure_drift_mean": np.mean(np.stack(drift_list, axis=0), axis=0),
    }



def reduce_timeseries(
    t: np.ndarray,
    y: np.ndarray,
    mode: str = "mean",
    late_power: float = 2.0,
) -> float:
    """Reduce a single time series y(t) to one scalar according to a chosen time mode."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return float("nan")

    mode = str(mode).lower().strip()
    if mode == "snapshot":
        return float(y[-1])
    if mode == "mean":
        return float(np.mean(y))
    if mode == "auc":
        if t.size < 2 or float(t[-1] - t[0]) <= 0.0:
            return float(np.mean(y))
        area = float(np.trapz(y, t))
        return float(area / float(t[-1] - t[0]))
    if mode == "late":
        if y.size == 1:
            return float(y[0])
        x = np.linspace(0.0, 1.0, y.size)
        w = np.power(x + 1e-12, float(late_power))
        w = w / np.sum(w)
        return float(np.sum(w * y))
    if mode == "peak":
        return float(np.max(y))
    if mode == "peak_post_init":
        if y.size <= 1:
            return float(y[-1])
        return float(np.max(y[1:]))
    raise ValueError(f"Unknown time reduction mode: {mode}")


def summarize_reduced_timeseries_across_pairs(
    t: np.ndarray,
    curves: np.ndarray,
    mode: str = "mean",
    late_power: float = 2.0,
    q: Tuple[float, float, float] = (0.1, 0.5, 0.9),
) -> Dict[str, float]:
    """Reduce each pair-curve to one scalar, then summarize across pairs by quantiles."""
    curves = np.asarray(curves, dtype=float)
    if curves.size == 0:
        return {
            "q_lo": float("nan"),
            "q_med": float("nan"),
            "q_hi": float("nan"),
            "mean": float("nan"),
        }
    vals = np.array([reduce_timeseries(t, row, mode=mode, late_power=late_power) for row in curves], dtype=float)
    qvals = np.quantile(vals, q)
    return {
        "q_lo": float(qvals[0]),
        "q_med": float(qvals[1]),
        "q_hi": float(qvals[2]),
        "mean": float(np.mean(vals)),
    }

def first_crossing_time(t: np.ndarray, y: np.ndarray, thresh: float, direction: str) -> float:
    """
    Return first time where y crosses thresh.
      - direction='below': first t where y < thresh
      - direction='above': first t where y > thresh
    If no crossing, return t[-1].
    """
    direction = str(direction).lower().strip()
    if direction == "below":
        idx = np.where(y < float(thresh))[0]
    else:
        idx = np.where(y > float(thresh))[0]
    if idx.size == 0:
        return float(t[-1])
    return float(t[int(idx[0])])


def summarize_timeseries_across_pairs(
    t: np.ndarray,
    curves: np.ndarray,
    q: Tuple[float, float, float] = (0.1, 0.5, 0.9),
) -> Dict[str, np.ndarray]:
    """
    curves shape: (n_pairs, t_steps). Returns quantiles per time and median AUC.
    """
    q = tuple(float(x) for x in q)
    qvals = np.quantile(curves, q, axis=0)
    auc = np.trapz(curves, t, axis=1)
    return {
        "q_lo": qvals[0],
        "q_med": qvals[1],
        "q_hi": qvals[2],
        "auc_med": np.median(auc) if auc.size else float("nan"),
    }



@dataclass(frozen=True)
class ContrastPair:
    key: Tuple[int, int, int]
    real: Candidate
    null: Candidate
    contrast_score: float


@dataclass(frozen=True)
class HotspotRow:
    center_key: Tuple[int, int, int]
    seed: int
    i: int
    j: int
    contrast_score: float
    hotspot_score: float
    real_ent_mean: float
    null_ent_mean: float
    real_dom: float
    null_dom: float
    real_leak: float
    null_leak: float
    real_sig_qg: Tuple[int, int, int, int]
    null_sig_qg: Tuple[int, int, int, int]
    real_sig_qg_coarse: Tuple[int, int, int, int]
    null_sig_qg_coarse: Tuple[int, int, int, int]
    position_tag: Tuple[int, int]


def candidate_pair_key(c: Candidate) -> Tuple[int, int, int]:
    return (int(c.seed), int(c.i), int(c.j))


def static_contrast_score(real_c: Candidate, null_c: Candidate, d: int) -> float:
    """
    Heuristic contrast score used to focus open-system budget on subspaces where
    REAL and NULL already differ statically.
    """
    ent_r = 0.5 * (float(real_c.ent_i) + float(real_c.ent_j))
    ent_n = 0.5 * (float(null_c.ent_i) + float(null_c.ent_j))
    ent_term = abs(ent_r - ent_n)
    dom_term = abs(float(real_c.dom) - float(null_c.dom)) / max(1.0, float(d))
    leak_term = abs(float(real_c.leak) - float(null_c.leak))
    score_term = abs(float(real_c.score) - float(null_c.score))
    sig_bonus = 0.0
    if real_c.sig_qg != null_c.sig_qg:
        sig_bonus += 0.25
    if real_c.sig_qg_coarse != null_c.sig_qg_coarse:
        sig_bonus += 0.25
    return float(ent_term + dom_term + leak_term + 0.5 * score_term + sig_bonus)


def build_contrast_pairs(real_pool: List[Candidate], null_pool: List[Candidate], d: int) -> List[ContrastPair]:
    real_map = {candidate_pair_key(c): c for c in real_pool}
    null_map = {candidate_pair_key(c): c for c in null_pool}
    keys = sorted(set(real_map.keys()) & set(null_map.keys()))
    out: List[ContrastPair] = []
    for key in keys:
        rc = real_map[key]
        nc = null_map[key]
        out.append(ContrastPair(
            key=key,
            real=rc,
            null=nc,
            contrast_score=static_contrast_score(rc, nc, d),
        ))
    out.sort(key=lambda x: (-x.contrast_score, x.key))
    return out


def build_hotspot_report(
    contrast_pairs: List[ContrastPair],
    hotspot_top: int,
) -> List[HotspotRow]:
    """Select top sparse singleton hotspots rather than requiring local kernels."""
    if not contrast_pairs or hotspot_top <= 0:
        return []
    rows: List[HotspotRow] = []
    used_exact: set = set()
    for cp in contrast_pairs:
        if len(rows) >= int(hotspot_top):
            break
        if cp.key in used_exact:
            continue
        rc, nc = cp.real, cp.null
        ent_r = 0.5 * (float(rc.ent_i) + float(rc.ent_j))
        ent_n = 0.5 * (float(nc.ent_i) + float(nc.ent_j))
        hotspot_score = float(cp.contrast_score + 0.25 * abs(ent_r - ent_n) + 0.10 * abs(float(rc.score) - float(nc.score)))
        rows.append(HotspotRow(
            center_key=cp.key,
            seed=int(cp.key[0]),
            i=int(cp.key[1]),
            j=int(cp.key[2]),
            contrast_score=float(cp.contrast_score),
            hotspot_score=hotspot_score,
            real_ent_mean=ent_r,
            null_ent_mean=ent_n,
            real_dom=float(rc.dom),
            null_dom=float(nc.dom),
            real_leak=float(rc.leak),
            null_leak=float(nc.leak),
            real_sig_qg=tuple(rc.sig_qg),
            null_sig_qg=tuple(nc.sig_qg),
            real_sig_qg_coarse=tuple(rc.sig_qg_coarse),
            null_sig_qg_coarse=tuple(nc.sig_qg_coarse),
            position_tag=(int(cp.key[1]), int(cp.key[2])),
        ))
        used_exact.add(cp.key)
    rows.sort(key=lambda r: (-r.hotspot_score, -r.contrast_score, r.center_key))
    return rows


def build_hotspot_dynamic_report(
    *,
    hotspot_rows: List[HotspotRow],
    contrast_pairs: List[ContrastPair],
    cache: PauliCache,
    n_terms: int,
    Ls: List[np.ndarray],
    LLs: List[np.ndarray],
    t_max: float,
    t_steps: int,
    dt_internal: Optional[float],
    states_mode: str,
    logical_basis: str,
    time_mode: str,
    time_late_power: float,
) -> List[Dict[str, object]]:
    """Evaluate each singleton hotspot dynamically and summarize its REAL-minus-NULL advantage."""
    if not hotspot_rows or not contrast_pairs:
        return []
    pair_map = {cp.key: cp for cp in contrast_pairs}
    out_rows: List[Dict[str, object]] = []
    for row in hotspot_rows:
        cp = pair_map.get(tuple(row.center_key))
        if cp is None:
            continue
        real_m = reduced_open_system_metrics_for_candidate(
            cache=cache, n_terms=n_terms, cand=cp.real, model="REAL",
            Ls=Ls, LLs=LLs, t_max=t_max, t_steps=t_steps, dt_internal=dt_internal,
            states_mode=states_mode, logical_basis=logical_basis,
            time_mode=time_mode, time_late_power=time_late_power,
        )
        null_m = reduced_open_system_metrics_for_candidate(
            cache=cache, n_terms=n_terms, cand=cp.null, model="NULL",
            Ls=Ls, LLs=LLs, t_max=t_max, t_steps=t_steps, dt_internal=dt_internal,
            states_mode=states_mode, logical_basis=logical_basis,
            time_mode=time_mode, time_late_power=time_late_power,
        )
        deltas = {k: float(real_m[k] - null_m[k]) for k in real_m.keys()}
        favorable = bool(deltas.get("logical_purity", 0.0) > 0.0 and deltas.get("structure_drift", 0.0) < 0.0)
        dynamic_gain = float(deltas.get("logical_purity", 0.0) - deltas.get("structure_drift", 0.0))
        out_rows.append({
            "center_key": list(row.center_key),
            "seed": int(row.seed),
            "i": int(row.i),
            "j": int(row.j),
            "position_tag": list(row.position_tag),
            "real_sig_qg_coarse": list(row.real_sig_qg_coarse),
            "null_sig_qg_coarse": list(row.null_sig_qg_coarse),
            "contrast_score": float(row.contrast_score),
            "hotspot_score": float(row.hotspot_score),
            "real_sig_qg": list(row.real_sig_qg),
            "null_sig_qg": list(row.null_sig_qg),
            "real_sig_qg_coarse": list(row.real_sig_qg_coarse),
            "null_sig_qg_coarse": list(row.null_sig_qg_coarse),
            "delta_fid_uncond_q50": deltas.get("fid_uncond", float("nan")),
            "delta_fid_cond_q50": deltas.get("fid_cond", float("nan")),
            "delta_leak_q50": deltas.get("leak", float("nan")),
            "delta_subspace_retention_q50": deltas.get("subspace_retention", float("nan")),
            "delta_logical_coherence_q50": deltas.get("logical_coherence", float("nan")),
            "delta_logical_purity_q50": deltas.get("logical_purity", float("nan")),
            "delta_structure_drift_q50": deltas.get("structure_drift", float("nan")),
            "dynamic_gain": dynamic_gain,
            "favorable": favorable,
        })
    out_rows.sort(key=lambda r: (-float(r["dynamic_gain"]), -float(r["delta_logical_purity_q50"]), float(r["delta_structure_drift_q50"]), -float(r["hotspot_score"])))
    return out_rows


def _summarize_hotspot_groups(groups: Dict[str, List[Dict[str, object]]], grouping: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for key, rows in groups.items():
        lp = np.array([float(r.get("delta_logical_purity_q50", float("nan"))) for r in rows], dtype=float)
        sd = np.array([float(r.get("delta_structure_drift_q50", float("nan"))) for r in rows], dtype=float)
        dg = np.array([float(r.get("dynamic_gain", float("nan"))) for r in rows], dtype=float)
        fav_mask = np.array([bool(r.get("favorable", False)) for r in rows], dtype=bool)
        anti_mask = np.array([(float(r.get("delta_logical_purity_q50", 0.0)) < 0.0) and (float(r.get("delta_structure_drift_q50", 0.0)) > 0.0) for r in rows], dtype=bool)
        mixed_mask = ~(fav_mask | anti_mask)
        fav = int(np.sum(fav_mask))
        anti = int(np.sum(anti_mask))
        mixed = int(np.sum(mixed_mask))
        batches = sorted({int(r.get("batch_id", -1)) for r in rows if int(r.get("batch_id", -1)) >= 0})
        batch_count = int(len(batches)) if batches else 1
        count = int(len(rows))
        max_batch_id = max([int(r.get("batch_id", -1)) for r in rows if int(r.get("batch_id", -1)) >= 0] or [batch_count])
        batch_cov = float(batch_count / max(1, max_batch_id))
        fav_rate = float(fav / max(1, count))
        anti_rate = float(anti / max(1, count))
        consistency = float(fav_rate - anti_rate)
        dynamic_gain_med = float(np.nanmedian(dg))
        logical_purity_med = float(np.nanmedian(lp))
        structure_drift_med = float(np.nanmedian(sd))
        direction_bonus = 1.0 if (logical_purity_med > 0.0 and structure_drift_med < 0.0) else (-1.0 if (logical_purity_med < 0.0 and structure_drift_med > 0.0) else 0.0)
        stability_score = float(batch_cov * consistency * (1.0 + max(0.0, dynamic_gain_med)) + 0.10 * direction_bonus)
        if fav_rate >= 0.75 and batch_count >= 2 and logical_purity_med > 0.0 and structure_drift_med < 0.0:
            label = "robust_favorable"
        elif anti_rate >= 0.75 and batch_count >= 2 and logical_purity_med < 0.0 and structure_drift_med > 0.0:
            label = "robust_anti"
        elif batch_count >= 2:
            label = "mixed_recurrent"
        else:
            label = "single_batch"
        out.append({
            "grouping": grouping,
            "class_key": key,
            "count": count,
            "batch_count": batch_count,
            "batches": batches,
            "favorable_count": fav,
            "anti_count": anti,
            "mixed_count": mixed,
            "favorable_rate": fav_rate,
            "anti_rate": anti_rate,
            "batch_coverage": batch_cov,
            "consistency": consistency,
            "stability_score": stability_score,
            "label": label,
            "dynamic_gain_med": dynamic_gain_med,
            "logical_purity_med": logical_purity_med,
            "structure_drift_med": structure_drift_med,
        })
    out.sort(key=lambda r: (-float(r["stability_score"]), -int(r["batch_count"]), -float(r["favorable_rate"]), -float(r["dynamic_gain_med"]), -int(r["count"]), r["class_key"]))
    return out

def build_hotspot_class_summary(hotspot_dynamic_rows: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    """Aggregate hotspots more coarsely so recurrence can accumulate across runs/batches."""
    if not hotspot_dynamic_rows:
        return {"by_pos": [], "by_pos_realcoarse": []}

    groups_pos: Dict[str, List[Dict[str, object]]] = {}
    groups_pos_real: Dict[str, List[Dict[str, object]]] = {}
    for row in hotspot_dynamic_rows:
        pos = tuple(row.get("position_tag", [None, None]))
        rs = tuple(row.get("real_sig_qg_coarse", []))
        key_pos = f"pos={pos}"
        key_pos_real = f"pos={pos}|R={rs}"
        groups_pos.setdefault(key_pos, []).append(row)
        groups_pos_real.setdefault(key_pos_real, []).append(row)

    return {
        "by_pos": _summarize_hotspot_groups(groups_pos, "position"),
        "by_pos_realcoarse": _summarize_hotspot_groups(groups_pos_real, "position+REAL_coarse"),
    }

def aggregate_hotspots_across_batches(open_system_summaries: List[Dict[str, Dict[str, object]]]) -> Dict[str, List[Dict[str, object]]]:
    all_rows: List[Dict[str, object]] = []
    for bi, osb in enumerate(open_system_summaries):
        if not isinstance(osb, dict):
            continue
        rows = osb.get("_hotspots_dynamic", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            rr = dict(row)
            rr["batch_id"] = int(bi + 1)
            all_rows.append(rr)
    return build_hotspot_class_summary(all_rows)


def hotspot_stability_panels(global_hot: Dict[str, List[Dict[str, object]]], top_n: int = 6) -> Dict[str, List[Dict[str, object]]]:
    by_pos = list(global_hot.get("by_pos", [])) if isinstance(global_hot, dict) else []
    robust_favorable = [r for r in by_pos if str(r.get("label", "")) == "robust_favorable"]
    robust_anti = [r for r in by_pos if str(r.get("label", "")) == "robust_anti"]
    mixed = [r for r in by_pos if str(r.get("label", "")) == "mixed_recurrent"]
    return {
        "robust_favorable": robust_favorable[:int(top_n)],
        "robust_anti": robust_anti[:int(top_n)],
        "mixed_recurrent": mixed[:int(top_n)],
    }



def reduced_open_system_metrics_for_candidate(
    *,
    cache: PauliCache,
    n_terms: int,
    cand: Candidate,
    model: str,
    Ls: List[np.ndarray],
    LLs: List[np.ndarray],
    t_max: float,
    t_steps: int,
    dt_internal: Optional[float],
    states_mode: str,
    logical_basis: str,
    time_mode: str,
    time_late_power: float,
) -> Dict[str, float]:
    """Compute reduced open-system metrics for one candidate pair."""
    H_real = build_random_pauli_hamiltonian_cached(cache, n_terms, cand.seed)
    if model == "REAL":
        H = H_real
    else:
        evals_real, _ = np.linalg.eigh(H_real)
        H = null_haar_basis_hamiltonian(evals_real, seed=cand.seed)

    evals, evecs = np.linalg.eigh(H)
    psi0 = evecs[:, int(cand.i)]
    psi1 = evecs[:, int(cand.j)]
    if str(logical_basis).lower().strip() == 'noise_diag':
        psi0, psi1 = noise_diagonal_logical_basis(psi0, psi1, Ls)

    met = open_system_metrics_for_pair(
        H=H, psi0=psi0, psi1=psi1, Ls=Ls, LLs=LLs,
        t_max=t_max, t_steps=t_steps, dt_internal=dt_internal, states_mode=states_mode,
    )
    t = np.asarray(met["t"], dtype=float)
    return {
        "fid_uncond": reduce_timeseries(t, np.asarray(met["fid_uncond_mean"], dtype=float), mode=time_mode, late_power=time_late_power),
        "fid_cond": reduce_timeseries(t, np.asarray(met["fid_cond_mean"], dtype=float), mode=time_mode, late_power=time_late_power),
        "leak": reduce_timeseries(t, np.asarray(met["leak_mean"], dtype=float), mode=time_mode, late_power=time_late_power),
        "subspace_retention": reduce_timeseries(t, np.asarray(met["subspace_retention_mean"], dtype=float), mode=time_mode, late_power=time_late_power),
        "logical_coherence": reduce_timeseries(t, np.asarray(met["logical_coherence_mean"], dtype=float), mode=time_mode, late_power=time_late_power),
        "logical_purity": reduce_timeseries(t, np.asarray(met["logical_purity_mean"], dtype=float), mode=time_mode, late_power=time_late_power),
        "structure_drift": reduce_timeseries(t, np.asarray(met["structure_drift_mean"], dtype=float), mode=time_mode, late_power=time_late_power),
    }




def broaden_kernel_selection(
    contrast_pairs: List[ContrastPair],
    kernel_top_centers: int,
    kernel_radius: int,
    kernel_min_neighbors: int,
    target_min_pairs: int,
    max_radius: int = 8,
    min_neighbors_floor: int = 2,
) -> Tuple[List[KernelZoomRow], int, int]:
    """
    Adaptive broadening for sparse hotspot regimes (especially 4 qubits).

    v24.9 policy:
      - a kernel is not allowed to collapse to a singleton if it can be avoided,
      - we broaden radius more aggressively before relaxing constraints,
      - we never go below min_neighbors_floor when reporting kernel rows.

    Returns (rows, used_radius, used_min_neighbors).
    """
    if not contrast_pairs:
        return [], int(kernel_radius), max(int(kernel_min_neighbors), int(min_neighbors_floor))

    radius0 = max(0, int(kernel_radius))
    min_neighbors0 = max(int(min_neighbors_floor), int(kernel_min_neighbors))
    target = max(1, int(target_min_pairs))

    best_rows: List[KernelZoomRow] = []
    best_cov = -1
    best_quality = float('-inf')
    best_params = (radius0, min_neighbors0)

    for radius in range(radius0, int(max_radius) + 1):
        rows = build_kernel_zoom_report(contrast_pairs, kernel_top_centers, radius, min_neighbors0)
        cov = int(sum(r.neighborhood_size for r in rows))
        qual = float(sum(r.kernel_score for r in rows)) if rows else float('-inf')
        if (cov > best_cov) or (cov == best_cov and qual > best_quality):
            best_rows = rows
            best_cov = cov
            best_quality = qual
            best_params = (radius, min_neighbors0)
        if rows and cov >= target:
            return rows, radius, min_neighbors0

    # If true sparse profitable hotspots cannot reach target coverage, keep the best multi-point set.
    if best_rows:
        return best_rows, best_params[0], best_params[1]

    # Last-resort fallback: return no kernel rows rather than silently degrading to singleton kernels.
    return [], radius0, min_neighbors0

def build_kernel_zoom_dynamic_report(
    *,
    kernel_rows: List[KernelZoomRow],
    contrast_pairs: List[ContrastPair],
    cache: PauliCache,
    n_terms: int,
    Ls: List[np.ndarray],
    LLs: List[np.ndarray],
    t_max: float,
    t_steps: int,
    dt_internal: Optional[float],
    states_mode: str,
    logical_basis: str,
    time_mode: str,
    time_late_power: float,
    kernel_radius: int,
) -> List[Dict[str, object]]:
    """Turn kernel zoom into an evidence-bearing local dynamic report."""
    if not kernel_rows or not contrast_pairs:
        return []

    out_rows: List[Dict[str, object]] = []
    radius = max(0, int(kernel_radius))
    for row in kernel_rows:
        seed = int(row.seed)
        ci = int(row.i)
        cj = int(row.j)
        neigh = [p for p in contrast_pairs if p.key[0] == seed and abs(p.key[1] - ci) <= radius and abs(p.key[2] - cj) <= radius]
        if not neigh:
            continue
        deltas = {
            "fid_uncond": [], "fid_cond": [], "leak": [], "subspace_retention": [],
            "logical_coherence": [], "logical_purity": [], "structure_drift": []
        }
        for cp in neigh:
            real_m = reduced_open_system_metrics_for_candidate(
                cache=cache, n_terms=n_terms, cand=cp.real, model="REAL",
                Ls=Ls, LLs=LLs, t_max=t_max, t_steps=t_steps, dt_internal=dt_internal,
                states_mode=states_mode, logical_basis=logical_basis,
                time_mode=time_mode, time_late_power=time_late_power,
            )
            null_m = reduced_open_system_metrics_for_candidate(
                cache=cache, n_terms=n_terms, cand=cp.null, model="NULL",
                Ls=Ls, LLs=LLs, t_max=t_max, t_steps=t_steps, dt_internal=dt_internal,
                states_mode=states_mode, logical_basis=logical_basis,
                time_mode=time_mode, time_late_power=time_late_power,
            )
            for key in deltas:
                deltas[key].append(real_m[key] - null_m[key])

        med_lp = float(np.median(np.asarray(deltas["logical_purity"], dtype=float)))
        med_sd = float(np.median(np.asarray(deltas["structure_drift"], dtype=float)))
        out_rows.append({
            "center_key": list(row.center_key),
            "seed": seed,
            "i": ci,
            "j": cj,
            "neighborhood_size": int(len(neigh)),
            "contrast_score": float(row.contrast_score),
            "local_mean": float(row.local_mean),
            "local_best": float(row.local_best),
            "kernel_score": float(row.kernel_score),
            "dynamic_gain": float(med_lp - med_sd),
            "delta_fid_uncond_q50": float(np.median(np.asarray(deltas["fid_uncond"], dtype=float))),
            "delta_fid_cond_q50": float(np.median(np.asarray(deltas["fid_cond"], dtype=float))),
            "delta_leak_q50": float(np.median(np.asarray(deltas["leak"], dtype=float))),
            "delta_subspace_retention_q50": float(np.median(np.asarray(deltas["subspace_retention"], dtype=float))),
            "delta_logical_coherence_q50": float(np.median(np.asarray(deltas["logical_coherence"], dtype=float))),
            "delta_logical_purity_q50": med_lp,
            "delta_structure_drift_q50": med_sd,
            "delta_logical_purity_best": float(np.max(np.asarray(deltas["logical_purity"], dtype=float))),
            "delta_structure_drift_best": float(np.min(np.asarray(deltas["structure_drift"], dtype=float))),
        })
    out_rows.sort(key=lambda r: (-float(r["dynamic_gain"]), -float(r["delta_logical_purity_q50"]), float(r["delta_structure_drift_q50"]), -float(r["kernel_score"])))
    return out_rows


def sample_candidates_by_iso_quartile(
    cands: List[Candidate],
    which: str,
    n: int,
    rng: np.random.Generator,
) -> List[Candidate]:
    """
    which: 'Q1' or 'Q4' based on iso_log quartiles within cands.
    """
    if not cands:
        return []
    vals = np.array([c.iso_log for c in cands], dtype=float)
    q25 = float(np.quantile(vals, 0.25))
    q75 = float(np.quantile(vals, 0.75))
    if which.upper() == "Q1":
        pool = [c for c in cands if float(c.iso_log) <= q25]
    else:
        pool = [c for c in cands if float(c.iso_log) >= q75]
    if not pool:
        pool = list(cands)
    if n >= len(pool):
        return pool
    idx = rng.choice(len(pool), size=int(n), replace=False)
    return [pool[int(i)] for i in idx]


def open_system_summary_for_batch(
    *,
    include_baselines: bool,
    cache: PauliCache,
    n_terms: int,
    batch_id: int,
    seed_offset: int,
    real_cands: List[Candidate],
    null_cands: List[Candidate],
    stable_frac: float,
    stable_leak_max: Optional[float],
    stable_leak_quantile: Optional[float],
    use_stable_pool: bool,
    os_pairs_per_batch: int,
    noise_model: str,
    gamma_phi: float,
    gamma_1: float,
    t_max: float,
    t_steps: int,
    dt_internal: Optional[float],
    states_mode: str,
    q_report: Tuple[float, float, float],
    logical_basis: str,
    os_noise_qubits: str,
    os_noise_subset: Optional[List[int]],
    time_mode: str,
    time_late_power: float,
    contrast_focus: bool,
    kernel_zoom: bool,
    kernel_top_centers: int,
    kernel_radius: int,
    kernel_min_neighbors: int,
) -> Dict[str, Dict[str, object]]:
    """Open-system summaries for REAL/NULL plus structure-retention metrics (v24.8)."""
    rng = rng_from_seed(stable_hash_int(f"OPEN_SYSTEM|{seed_offset}|{batch_id}"))

    def maybe_stable_pool(cands: List[Candidate]) -> List[Candidate]:
        if (not use_stable_pool) or (not cands):
            return cands
        scores = np.array([c.score for c in cands], dtype=float)
        leaks = np.array([c.leak for c in cands], dtype=float)
        m = stable_mask_from_scores_and_leak(scores, leaks, stable_frac, stable_leak_max, stable_leak_quantile)
        return [c for c, keep in zip(cands, m) if bool(keep)]
    
    real_pool = maybe_stable_pool(real_cands)
    null_pool = maybe_stable_pool(null_cands)
    contrast_pairs = build_contrast_pairs(real_pool, null_pool, cache.d) if bool(contrast_focus) else []
    hotspot_rows: List[HotspotRow] = []
    used_kernel_radius = int(kernel_radius)
    used_kernel_min_neighbors = int(kernel_min_neighbors)
    if bool(contrast_focus) and bool(kernel_zoom):
        hotspot_rows = build_hotspot_report(
            contrast_pairs,
            hotspot_top=max(int(kernel_top_centers), 12 if cache.n_qubits <= 4 else int(kernel_top_centers)),
        )
    
    subset = None
    if str(os_noise_qubits).lower().strip() == 'subset':
        subset = [0] if not os_noise_subset else [int(x) for x in os_noise_subset]
    Ls, LLs = build_noise_operators(cache.n_qubits, noise_model, gamma_phi, gamma_1, subset)
    hotspot_dynamic_rows: List[Dict[str, object]] = []
    hotspot_class_rows: Dict[str, List[Dict[str, object]]] = {"by_pos": [], "by_pos_realcoarse": []}
    if hotspot_rows:
        hotspot_dynamic_rows = build_hotspot_dynamic_report(
            hotspot_rows=hotspot_rows,
            contrast_pairs=contrast_pairs,
            cache=cache,
            n_terms=n_terms,
            Ls=Ls,
            LLs=LLs,
            t_max=t_max,
            t_steps=t_steps,
            dt_internal=dt_internal,
            states_mode=states_mode,
            logical_basis=logical_basis,
            time_mode=time_mode,
            time_late_power=time_late_power,
                    )
        hotspot_class_rows = build_hotspot_class_summary(hotspot_dynamic_rows)
    
    def eval_group(model: str, which: str, pool: List[Candidate]) -> Dict[str, object]:
        if bool(contrast_focus) and which.upper() == "Q4" and contrast_pairs:
            chosen_pairs: List[ContrastPair]
            if hotspot_rows:
                centers = {tuple(r.center_key) for r in hotspot_rows}
                local_pairs = [cp for cp in contrast_pairs if cp.key in centers]
                local_pairs.sort(key=lambda x: (-x.contrast_score, x.key))
                if local_pairs:
                    chosen_pairs = local_pairs[:int(min(int(os_pairs_per_batch), len(local_pairs)))]
                else:
                    chosen_pairs = contrast_pairs[:int(min(int(os_pairs_per_batch), len(contrast_pairs)))]
            else:
                chosen_pairs = contrast_pairs[:int(min(int(os_pairs_per_batch), len(contrast_pairs)))]
            if model == "REAL":
                samp = [cp.real for cp in chosen_pairs]
            else:
                samp = [cp.null for cp in chosen_pairs]
            contrast_stats = summarize_scalar_values([cp.contrast_score for cp in chosen_pairs], q_report)
        else:
            samp = sample_candidates_by_iso_quartile(pool, which, int(os_pairs_per_batch), rng)
            contrast_stats = None
        if not samp:
            return {"n_pairs": 0}
    
        leak_curves=[]; subret_curves=[]; fu_curves=[]; fc_curves=[]; coh_curves=[]; pur_curves=[]; drift_curves=[]
        t_ref = None
        t_fid90=[]; t_fid50=[]; t_leak10=[]
    
        for c in samp:
            H_real = build_random_pauli_hamiltonian_cached(cache, n_terms, c.seed)
            if model == "REAL":
                H = H_real
            else:
                evals_real, _ = np.linalg.eigh(H_real)
                H = null_haar_basis_hamiltonian(evals_real, seed=c.seed)
    
            evals, evecs = np.linalg.eigh(H)
            psi0 = evecs[:, int(c.i)]
            psi1 = evecs[:, int(c.j)]
            if str(logical_basis).lower().strip() == 'noise_diag':
                psi0, psi1 = noise_diagonal_logical_basis(psi0, psi1, Ls)
    
            met = open_system_metrics_for_pair(
                H=H, psi0=psi0, psi1=psi1, Ls=Ls, LLs=LLs,
                t_max=t_max, t_steps=t_steps, dt_internal=dt_internal, states_mode=states_mode,
            )
            t = met["t"]
            if t_ref is None:
                t_ref = t
            leak = met["leak_mean"]; subret = met["subspace_retention_mean"]; fu = met["fid_uncond_mean"]; fc = met["fid_cond_mean"]
            coh = met["logical_coherence_mean"]; pur = met["logical_purity_mean"]; drift = met["structure_drift_mean"]
    
            leak_curves.append(leak); subret_curves.append(subret); fu_curves.append(fu); fc_curves.append(fc)
            coh_curves.append(coh); pur_curves.append(pur); drift_curves.append(drift)
            t_fid90.append(first_crossing_time(t, fu, 0.9, "below"))
            t_fid50.append(first_crossing_time(t, fu, 0.5, "below"))
            t_leak10.append(first_crossing_time(t, leak, 0.1, "above"))
    
        leak_curves=np.stack(leak_curves,axis=0); subret_curves=np.stack(subret_curves,axis=0); fu_curves=np.stack(fu_curves,axis=0)
        fc_curves=np.stack(fc_curves,axis=0); coh_curves=np.stack(coh_curves,axis=0); pur_curves=np.stack(pur_curves,axis=0); drift_curves=np.stack(drift_curves,axis=0)
    
        def sums(curves):
            return summarize_timeseries_across_pairs(t_ref, curves, q_report), summarize_reduced_timeseries_across_pairs(t_ref, curves, mode=time_mode, late_power=time_late_power, q=q_report)
    
        leak_sum, leak_red = sums(leak_curves)
        subret_sum, subret_red = sums(subret_curves)
        fu_sum, fu_red = sums(fu_curves)
        fc_sum, fc_red = sums(fc_curves)
        coh_sum, coh_red = sums(coh_curves)
        pur_sum, pur_red = sums(pur_curves)
        drift_sum, drift_red = sums(drift_curves)
    
        out_block = {
            "n_pairs": int(leak_curves.shape[0]),
            "t": t_ref,
            "leak": leak_sum,
            "subspace_retention": subret_sum,
            "fid_uncond": fu_sum,
            "fid_cond": fc_sum,
            "logical_coherence": coh_sum,
            "logical_purity": pur_sum,
            "structure_drift": drift_sum,
            "reduced": {
                "mode": str(time_mode), "late_power": float(time_late_power),
                "leak": leak_red, "subspace_retention": subret_red,
                "fid_uncond": fu_red, "fid_cond": fc_red,
                "logical_coherence": coh_red, "logical_purity": pur_red,
                "structure_drift": drift_red,
            },
            "t_fid90_med": float(np.median(np.array(t_fid90))) if t_fid90 else float("nan"),
            "t_fid50_med": float(np.median(np.array(t_fid50))) if t_fid50 else float("nan"),
            "t_leak10_med": float(np.median(np.array(t_leak10))) if t_leak10 else float("nan"),
        }
        if contrast_stats is not None:
            out_block["contrast_focus"] = True
            out_block["contrast_score"] = contrast_stats
        return out_block

    out: Dict[str, Dict[str, object]] = {}
    out["REAL_Q4"] = eval_group("REAL", "Q4", real_pool)
    if bool(include_baselines):
        out["REAL_Q1"] = eval_group("REAL", "Q1", real_pool)
    out["NULL_Q4"] = eval_group("NULL", "Q4", null_pool)
    out["_meta"] = {
        "noise_model": str(noise_model), "gamma_phi": float(gamma_phi), "gamma_1": float(gamma_1),
        "t_max": float(t_max), "t_steps": int(t_steps), "states_mode": str(states_mode),
        "use_stable_pool": bool(use_stable_pool), "os_pairs_per_batch": int(os_pairs_per_batch),
        "logical_basis": str(logical_basis), "os_noise_qubits": str(os_noise_qubits),
        "os_noise_subset": subset if subset is not None else None,
        "time_mode": str(time_mode), "time_late_power": float(time_late_power), "contrast_focus": bool(contrast_focus),
        "kernel_zoom": bool(kernel_zoom), "hotspot_rows": int(len(hotspot_rows)), "kernel_top_centers": int(kernel_top_centers), "kernel_radius": int(kernel_radius),
        "kernel_min_neighbors": int(kernel_min_neighbors),
        "kernel_radius_used": int(used_kernel_radius),
        "kernel_min_neighbors_used": int(used_kernel_min_neighbors),
        "kernel_rows": int(len(hotspot_rows)),
    }
    if hotspot_rows:
        out["_hotspots"] = [
            {
                "center_key": list(r.center_key),
                "seed": int(r.seed),
                "i": int(r.i),
                "j": int(r.j),
                "position_tag": list(r.position_tag),
                "contrast_score": float(r.contrast_score),
                "hotspot_score": float(r.hotspot_score),
                "real_ent_mean": float(r.real_ent_mean),
                "null_ent_mean": float(r.null_ent_mean),
                "real_dom": float(r.real_dom),
                "null_dom": float(r.null_dom),
                "real_leak": float(r.real_leak),
                "null_leak": float(r.null_leak),
                "real_sig_qg_coarse": list(r.real_sig_qg_coarse),
                "null_sig_qg_coarse": list(r.null_sig_qg_coarse),
            }
            for r in hotspot_rows
        ]
    if hotspot_dynamic_rows:
        out["_hotspots_dynamic"] = hotspot_dynamic_rows
    if hotspot_class_rows.get("by_pos") or hotspot_class_rows.get("by_pos_realcoarse"):
        out["_hotspot_classes"] = hotspot_class_rows
    return out



def prob_dist(psi: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.abs(np.asarray(psi, dtype=complex).reshape(-1)) ** 2
    s = float(np.sum(p))
    return np.asarray(p / max(eps, s), dtype=float)


def shannon_entropy_probs(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    if p.size == 0:
        return 0.0
    p = np.clip(p, eps, 1.0)
    p = p / float(np.sum(p))
    return float(-np.sum(p * np.log2(p)))


def interference_metrics_for_pair(psi0: np.ndarray, psi1: np.ndarray) -> Dict[str, float]:
    psi0 = np.asarray(psi0, dtype=complex).reshape(-1)
    psi1 = np.asarray(psi1, dtype=complex).reshape(-1)
    psi_plus = (psi0 + psi1) / math.sqrt(2.0)
    psi_minus = (psi0 - psi1) / math.sqrt(2.0)
    psi_ip = (psi0 + 1j * psi1) / math.sqrt(2.0)

    p_plus = prob_dist(psi_plus)
    p_minus = prob_dist(psi_minus)
    p_ip = prob_dist(psi_ip)

    contrast_pm = float(np.sum(np.abs(p_plus - p_minus)))
    contrast_pi = float(np.sum(np.abs(p_plus - p_ip)))

    return {
        "contrast_pm": contrast_pm,
        "contrast_pi": contrast_pi,
        "ent_plus": shannon_entropy_probs(p_plus),
        "ent_minus": shannon_entropy_probs(p_minus),
        "ent_ip": shannon_entropy_probs(p_ip),
    }


def summarize_scalar_values(vals: List[float], q: Tuple[float, float, float] = (0.1, 0.5, 0.9)) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"q_lo": float("nan"), "q_med": float("nan"), "q_hi": float("nan"), "mean": float("nan")}
    qv = np.quantile(arr, q)
    return {"q_lo": float(qv[0]), "q_med": float(qv[1]), "q_hi": float(qv[2]), "mean": float(np.mean(arr))}


def interference_summary_for_batch(
    *,
    cache: PauliCache,
    n_terms: int,
    batch_id: int,
    seed_offset: int,
    real_cands: List[Candidate],
    null_cands: List[Candidate],
    stable_frac: float,
    stable_leak_max: Optional[float],
    stable_leak_quantile: Optional[float],
    use_stable_pool: bool,
    if_pairs_per_batch: int,
    q_report: Tuple[float, float, float],
) -> Dict[str, Dict[str, object]]:
    rng = rng_from_seed(stable_hash_int(f"INTERFERENCE|{seed_offset}|{batch_id}"))

    def maybe_stable_pool(cands: List[Candidate]) -> List[Candidate]:
        if (not use_stable_pool) or (not cands):
            return cands
        scores = np.array([c.score for c in cands], dtype=float)
        leaks = np.array([c.leak for c in cands], dtype=float)
        m = stable_mask_from_scores_and_leak(scores, leaks, stable_frac, stable_leak_max, stable_leak_quantile)
        return [c for c, keep in zip(cands, m) if bool(keep)]

    real_pool = maybe_stable_pool(real_cands)
    null_pool = maybe_stable_pool(null_cands)

    def eval_group(model: str, which: str, pool: List[Candidate]) -> Dict[str, object]:
        samp = sample_candidates_by_iso_quartile(pool, which, int(if_pairs_per_batch), rng)
        if not samp:
            return {"n_pairs": 0}
        c_pm=[]; c_pi=[]; e_p=[]; e_m=[]; e_i=[]
        for c in samp:
            H_real = build_random_pauli_hamiltonian_cached(cache, n_terms, c.seed)
            if model == "REAL":
                H = H_real
            else:
                evals_real, _ = np.linalg.eigh(H_real)
                H = null_haar_basis_hamiltonian(evals_real, seed=c.seed)
            evals, evecs = np.linalg.eigh(H)
            psi0 = evecs[:, int(c.i)]
            psi1 = evecs[:, int(c.j)]
            met = interference_metrics_for_pair(psi0, psi1)
            c_pm.append(met["contrast_pm"])
            c_pi.append(met["contrast_pi"])
            e_p.append(met["ent_plus"])
            e_m.append(met["ent_minus"])
            e_i.append(met["ent_ip"])
        return {
            "n_pairs": int(len(samp)),
            "contrast_pm": summarize_scalar_values(c_pm, q_report),
            "contrast_pi": summarize_scalar_values(c_pi, q_report),
            "ent_plus": summarize_scalar_values(e_p, q_report),
            "ent_minus": summarize_scalar_values(e_m, q_report),
            "ent_ip": summarize_scalar_values(e_i, q_report),
        }

    out: Dict[str, Dict[str, object]] = {}
    out["REAL_Q4"] = eval_group("REAL", "Q4", real_pool)
    out["NULL_Q4"] = eval_group("NULL", "Q4", null_pool)
    out["_meta"] = {
        "use_stable_pool": bool(use_stable_pool),
        "if_pairs_per_batch": int(if_pairs_per_batch),
    }
    return out


# ----------------------------
# Reporting
# ----------------------------

def open_system_delta_summary(osb: Dict[str, Dict[str, object]]) -> Dict[str, float]:
    """Compare REAL_Q4 and NULL_Q4 on reduced open-system metrics when both are present."""
    if not isinstance(osb, dict):
        return {}
    rq = osb.get("REAL_Q4")
    nq = osb.get("NULL_Q4")
    if not isinstance(rq, dict) or not isinstance(nq, dict):
        return {}
    if int(rq.get("n_pairs", 0)) <= 0 or int(nq.get("n_pairs", 0)) <= 0:
        return {}

    def _get_red(block: Dict[str, object], key: str, sub: str = "q_med") -> float:
        red = block.get("reduced", {})
        if not isinstance(red, dict):
            return float("nan")
        sec = red.get(key, {})
        if not isinstance(sec, dict):
            return float("nan")
        val = sec.get(sub, float("nan"))
        try:
            return float(val)
        except Exception:
            return float("nan")

    out = {
        "delta_reduced_fid_uncond_q50": _get_red(rq, "fid_uncond") - _get_red(nq, "fid_uncond"),
        "delta_reduced_fid_cond_q50": _get_red(rq, "fid_cond") - _get_red(nq, "fid_cond"),
        "delta_reduced_leak_q50": _get_red(rq, "leak") - _get_red(nq, "leak"),
        "delta_reduced_subspace_retention_q50": _get_red(rq, "subspace_retention") - _get_red(nq, "subspace_retention"),
        "delta_reduced_logical_coherence_q50": _get_red(rq, "logical_coherence") - _get_red(nq, "logical_coherence"),
        "delta_reduced_logical_purity_q50": _get_red(rq, "logical_purity") - _get_red(nq, "logical_purity"),
        "delta_reduced_structure_drift_q50": _get_red(rq, "structure_drift") - _get_red(nq, "structure_drift"),
        "delta_t_fid90_med": float(rq.get("t_fid90_med", float("nan"))) - float(nq.get("t_fid90_med", float("nan"))),
        "delta_t_leak10_med": float(rq.get("t_leak10_med", float("nan"))) - float(nq.get("t_leak10_med", float("nan"))),
    }
    return out

def format_top(rows: List[FamRow], label: str, show: int = 10) -> str:
    lines = []
    lines.append(f"Top signature families ({label}):")
    lines.append("Format: sig=(a,b,c,d) | overall | stable | expected | p_tail | -log10(p) | enrichment")
    for r in rows[:show]:
        lines.append(
            f"  {r.sig} | {r.overall:7d} | {r.stable:7d} | {r.expected:9.2f} | {r.p_tail:8.2e} | {r.neglog10_p:9.2f} | {r.enrichment:9.2f}x"
        )
    if len(rows) > show:
        lines.append(f"  ... ({len(rows)} total, showing {show})")
    return "\n".join(lines)



# ----------------------------
# v25.4.1 dimension-normalized continuation ranker
# ----------------------------

@dataclass(frozen=True)
class ContinuationRow:
    source_left: int
    target_centers: Tuple[int, ...]
    n_obs: int
    seed_recurrence: float
    batch_recurrence: float
    branch_balance: float
    anchor_compatibility: float
    static_contrast: float
    leakage_penalty: float
    drift_penalty: float
    continuation_score: float


def _robust_scale(vals: np.ndarray, floor: float = 1e-9) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return 1.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    # 1.4826*MAD estimates sigma under a Gaussian; floor avoids zero scales.
    return max(float(floor), 1.4826 * mad)


def _candidate_feature_vector(c: Candidate, d: int) -> np.ndarray:
    """
    v25.4.1 dimension-normalized REAL feature vector used for anchor compatibility.

    Components:
      1) mean amplitude entropy / log2(d)
         -> removes the trivial +~1 bit shift when going nQ -> (n+1)Q.
      2) dominant-support fraction dom/d
         -> already dimension normalized.
      3) leakage
         -> dimensionless.
      4) interestingness score
         -> dimensionless under the existing protocol.

    This change is intentionally narrow: it corrects cross-dimension comparison
    without changing the underlying REAL/NULL candidate physics.
    """
    nq = math.log2(max(2.0, float(d)))
    ent_norm = 0.5 * (float(c.ent_i) + float(c.ent_j)) / max(1.0, nq)
    return np.array([
        ent_norm,
        float(c.dom) / max(1.0, float(d)),
        float(c.leak),
        float(c.score),
    ], dtype=float)


def _source_anchor_profile(
    anchor_left: int,
    from_qubits: int,
    n_terms: int,
    seeds: List[int],
    eps_neighbor: float,
    ent_step: float,
    leak_step: float,
    times: List[float],
    keep_mass: float,
    iso_eps: float,
) -> Tuple[np.ndarray, int]:
    """
    Reconstruct the source-nQ REAL anchor on the same seed schedule.
    The exact adjacent source pair is used; no neighbourhood averaging is
    introduced here, so the prior remains auditable.
    """
    cache_src = PauliCache.build(int(from_qubits))
    feats: List[np.ndarray] = []
    interval = [(int(anchor_left), int(anchor_left))]
    for seed in seeds:
        cands = generate_candidates_for_seed(
            model="REAL",
            seed=int(seed),
            cache=cache_src,
            n_terms=int(n_terms),
            eps_neighbor=float(eps_neighbor),
            ent_step=float(ent_step),
            leak_step=float(leak_step),
            times=list(times),
            keep_mass=float(keep_mass),
            iso_eps=float(iso_eps),
            target_intervals=interval,
        )
        for c in cands:
            if int(c.i) == int(anchor_left):
                feats.append(_candidate_feature_vector(c, cache_src.d))
                break
    if not feats:
        raise ValueError(
            f"Source anchor {anchor_left}-{anchor_left+1} produced no REAL candidates "
            f"under eps_neighbor={eps_neighbor}. The continuation prior cannot be calibrated."
        )
    return np.median(np.vstack(feats), axis=0), len(feats)


def _normalize_01_by_rank(values: Dict[Tuple[int, int, int], float]) -> Dict[Tuple[int, int, int], float]:
    """
    Deterministic empirical-rank normalization to [0,1].
    Ties receive the same midpoint rank.
    """
    if not values:
        return {}
    keys = list(values.keys())
    arr = np.array([float(values[k]) for k in keys], dtype=float)
    order = np.argsort(arr, kind="mergesort")
    out = np.zeros(len(arr), dtype=float)

    pos = 0
    n = len(arr)
    while pos < n:
        end = pos + 1
        while end < n and arr[order[end]] == arr[order[pos]]:
            end += 1
        midpoint = 0.5 * (pos + end - 1)
        rank01 = 1.0 if n == 1 else midpoint / float(n - 1)
        for q in range(pos, end):
            out[order[q]] = rank01
        pos = end
    return {k: float(out[idx]) for idx, k in enumerate(keys)}


def continuation_rank_scan(
    *,
    n_qubits: int,
    anchor_from_qubits: int,
    anchor_left: int,
    n_terms: int,
    seeds_per_batch: int,
    batches: int,
    base_seed: int,
    batch_stride: int,
    eps_neighbor: float,
    ent_step: float,
    leak_step: float,
    times: List[float],
    keep_mass: float,
    iso_eps: float,
    top_n: int,
    w_anchor: float,
    w_static: float,
    w_recurrence: float,
    w_leak: float,
    w_drift: float,
    w_batch: float,
    w_branch: float,
) -> Tuple[List[ContinuationRow], Dict[str, Any]]:
    """
    Dimension-normalized static-first implementation of the Phase-1 continuation idea.

    It does NOT assume that the successor remains at the same coordinate.
    Instead every legal source-coordinate k competes.  For each k the target
    branches k + b*2**anchor_from_qubits are pooled and scored.

    Components:
      A  anchor_compatibility  : feature similarity to the source REAL anchor
      M  static_contrast       : matched REAL-vs-spectrum-Haar contrast
      R  seed_recurrence       : fraction of target seeds in which k recurs
      L  leakage_penalty       : lower REAL leakage is preferred
      D  drift_penalty         : lower across-seed/branch feature drift preferred
      B  batch_recurrence      : fraction of batches represented
      G  branch_balance        : consistency across descendant branches

    Dynamic retention/purity are deliberately not fabricated here.  They are
    Stage-2 tests for the top-ranked static successors.
    """
    nq = int(n_qubits)
    fq = int(anchor_from_qubits)
    if nq <= fq:
        raise ValueError("--continuation_scan requires n_qubits > anchor_from_qubits")
    if nq != fq + 1:
        raise ValueError(
            "v25.4 first implementation deliberately supports one-step continuation only "
            "(nQ -> n+1Q) so the validation is maximally auditable."
        )

    cache_tgt = PauliCache.build(nq)
    d_tgt = cache_tgt.d
    base_dim = 2 ** fq
    source_left_max = base_dim - 2
    n_branches = 2 ** (nq - fq)

    if not (0 <= int(anchor_left) <= source_left_max):
        raise ValueError(
            f"Source anchor left index must be in 0-{source_left_max} for {fq}Q."
        )

    seed_records: List[Tuple[int, int, int]] = []
    all_seeds: List[int] = []
    for b in range(int(batches)):
        offset = int(base_seed) + b * int(batch_stride)
        for s in range(int(seeds_per_batch)):
            seed = offset + s
            all_seeds.append(seed)
            seed_records.append((b, s, seed))

    source_profile, source_hits = _source_anchor_profile(
        anchor_left=int(anchor_left),
        from_qubits=fq,
        n_terms=int(n_terms),
        seeds=all_seeds,
        eps_neighbor=float(eps_neighbor),
        ent_step=float(ent_step),
        leak_step=float(leak_step),
        times=list(times),
        keep_mass=float(keep_mass),
        iso_eps=float(iso_eps),
    )

    # Generate full target pools once.  This is the search, not a four-zone scan.
    real_all: List[Candidate] = []
    null_all: List[Candidate] = []
    batch_of_seed: Dict[int, int] = {}
    for b, _s, seed in seed_records:
        batch_of_seed[int(seed)] = int(b)
        real_all.extend(generate_candidates_for_seed(
            model="REAL", seed=seed, cache=cache_tgt, n_terms=n_terms,
            eps_neighbor=eps_neighbor, ent_step=ent_step, leak_step=leak_step,
            times=times, keep_mass=keep_mass, iso_eps=iso_eps,
            target_intervals=None,
        ))
        null_all.extend(generate_candidates_for_seed(
            model="NULL_HAAR_BASIS", seed=seed, cache=cache_tgt, n_terms=n_terms,
            eps_neighbor=eps_neighbor, ent_step=ent_step, leak_step=leak_step,
            times=times, keep_mass=keep_mass, iso_eps=iso_eps,
            target_intervals=None,
        ))

    # Quantile signatures are needed by static_contrast_score().
    pooled_ent = np.array(
        [c.ent_i for c in real_all] + [c.ent_j for c in real_all] +
        [c.ent_i for c in null_all] + [c.ent_j for c in null_all],
        dtype=float
    )
    # The ranker uses fixed 16/8 pooled quantile signatures, matching the Phase-2
    # production settings used in the 9Q-11Q experiments.
    edges_f = make_quantile_edges(pooled_ent, q_bins=16)
    edges_c = make_quantile_edges(pooled_ent, q_bins=8)
    real_all = assign_quantile_signatures(real_all, d_tgt, edges_f, 16, edges_c, 8, leak_step)
    null_all = assign_quantile_signatures(null_all, d_tgt, edges_f, 16, edges_c, 8, leak_step)

    rmap = {candidate_pair_key(c): c for c in real_all}
    nmap = {candidate_pair_key(c): c for c in null_all}
    matched_keys = sorted(set(rmap) & set(nmap))

    if not matched_keys:
        raise ValueError("No matched REAL/NULL target candidates were generated.")

    # Global robust scales for anchor compatibility and drift.
    real_feats = np.vstack([_candidate_feature_vector(rmap[k], d_tgt) for k in matched_keys])

    # Cross-dimension anchor comparison needs physically sensible scale floors.
    # Otherwise a nearly constant feature (especially leakage or score) can have
    # a tiny MAD and dominate the distance for numerically insignificant changes.
    empirical_scales = np.array(
        [_robust_scale(real_feats[:, j]) for j in range(real_feats.shape[1])],
        dtype=float,
    )
    scale_floors = np.array([
        0.010,  # entropy/log2(d)
        0.010,  # dom/d
        0.002,  # leakage
        0.005,  # score
    ], dtype=float)
    scales = np.maximum(empirical_scales, scale_floors)

    raw_contrast: Dict[Tuple[int, int, int], float] = {}
    for k in matched_keys:
        raw_contrast[k] = static_contrast_score(rmap[k], nmap[k], d_tgt)
    contrast_rank = _normalize_01_by_rank(raw_contrast)

    obs_by_source: Dict[int, List[Tuple[Candidate, Candidate, float, int, int]]] = {
        k: [] for k in range(source_left_max + 1)
    }

    for key in matched_keys:
        seed, i, j = key
        source_k = int(i) % base_dim
        branch = int(i) // base_dim
        # Only coordinates that map to a legal neighbouring source pair.
        if source_k > source_left_max or branch >= n_branches:
            continue
        obs_by_source[source_k].append(
            (rmap[key], nmap[key], float(contrast_rank[key]), int(branch), int(batch_of_seed[int(seed)]))
        )

    rows: List[ContinuationRow] = []
    total_seeds = max(1, len(all_seeds))
    total_batches = max(1, int(batches))

    for source_k in range(source_left_max + 1):
        obs = obs_by_source[source_k]
        if not obs:
            continue

        feats = np.vstack([_candidate_feature_vector(rc, d_tgt) for rc, _nc, _m, _br, _ba in obs])
        zdist = np.mean(np.abs((feats - source_profile[None, :]) / scales[None, :]), axis=1)
        anchor_compat = float(np.mean(np.exp(-0.5 * zdist)))

        static_m = float(np.mean([m for _rc, _nc, m, _br, _ba in obs]))
        seed_rec = len({int(rc.seed) for rc, _nc, _m, _br, _ba in obs}) / float(total_seeds)
        batch_rec = len({ba for _rc, _nc, _m, _br, ba in obs}) / float(total_batches)

        branch_counts = [sum(1 for _rc, _nc, _m, br, _ba in obs if br == q) for q in range(n_branches)]
        if max(branch_counts) > 0:
            branch_balance = min(branch_counts) / float(max(branch_counts))
        else:
            branch_balance = 0.0

        leak_pen = float(np.mean([float(rc.leak) for rc, _nc, _m, _br, _ba in obs]))

        if feats.shape[0] > 1:
            med = np.median(feats, axis=0)
            drift_z = np.mean(np.abs((feats - med[None, :]) / scales[None, :]), axis=1)
            drift_pen = float(np.mean(drift_z))
        else:
            drift_pen = 1.0

        # Map the unbounded drift penalty to [0,1), preserving monotonicity.
        drift01 = drift_pen / (1.0 + drift_pen)

        score = (
            float(w_anchor) * anchor_compat +
            float(w_static) * static_m +
            float(w_recurrence) * seed_rec -
            float(w_leak) * leak_pen -
            float(w_drift) * drift01 +
            float(w_batch) * batch_rec +
            float(w_branch) * branch_balance
        )

        centers = tuple(int(source_k + b * base_dim) for b in range(n_branches))
        rows.append(ContinuationRow(
            source_left=int(source_k),
            target_centers=centers,
            n_obs=len(obs),
            seed_recurrence=float(seed_rec),
            batch_recurrence=float(batch_rec),
            branch_balance=float(branch_balance),
            anchor_compatibility=float(anchor_compat),
            static_contrast=float(static_m),
            leakage_penalty=float(leak_pen),
            drift_penalty=float(drift01),
            continuation_score=float(score),
        ))

    rows.sort(key=lambda r: (-r.continuation_score, r.source_left))

    anchor_rank = next((idx + 1 for idx, r in enumerate(rows) if r.source_left == int(anchor_left)), None)

    local_radius = 8
    local_rows = [r for r in rows if abs(int(r.source_left) - int(anchor_left)) <= local_radius]
    best_local = local_rows[0] if local_rows else None
    best_local_rank = (
        next((idx + 1 for idx, r in enumerate(rows) if best_local is not None and r.source_left == best_local.source_left), None)
        if best_local is not None else None
    )

    meta: Dict[str, Any] = {
        "source_profile": source_profile.tolist(),
        "source_hits": int(source_hits),
        "source_anchor_left": int(anchor_left),
        "anchor_rank": anchor_rank,
        "local_radius": local_radius,
        "best_local_source": None if best_local is None else int(best_local.source_left),
        "best_local_rank": best_local_rank,
        "empirical_scales": empirical_scales.tolist(),
        "effective_scales": scales.tolist(),
        "n_ranked": len(rows),
        "n_target_real": len(real_all),
        "n_target_null": len(null_all),
        "weights": {
            "A_anchor": float(w_anchor),
            "M_static": float(w_static),
            "R_seed": float(w_recurrence),
            "L_leak": float(w_leak),
            "D_drift": float(w_drift),
            "B_batch": float(w_batch),
            "G_branch": float(w_branch),
        },
    }
    return rows[:max(1, int(top_n))], meta


def write_continuation_report(
    output_path: str,
    rows: List[ContinuationRow],
    meta: Dict[str, Any],
    *,
    n_qubits: int,
    anchor_from_qubits: int,
    anchor_spec: str,
    seeds_per_batch: int,
    batches: int,
    base_seed: int,
) -> None:
    lines: List[str] = []
    lines.append("=== Soft Spaces Phase 2 v25.4.1 DIMENSION-NORMALIZED CONTINUATION RANKER ===")
    lines.append(
        f"Source anchor: {anchor_spec} at {anchor_from_qubits}Q -> search all legal successor coordinates at {n_qubits}Q"
    )
    lines.append(
        f"Seeds: {batches} x {seeds_per_batch} | base_seed={base_seed} | "
        f"target candidates REAL={meta['n_target_real']} NULL={meta['n_target_null']}"
    )
    lines.append(
        "Static-first score: C = wA*A + wM*M + wR*R - wL*L - wD*D + wB*B + wG*G"
    )
    lines.append(
        "A=source-anchor feature compatibility; M=REAL/NULL static contrast; R=seed recurrence; "
        "L=REAL leakage; D=feature drift; B=batch recurrence; G=descendant-branch balance."
    )
    lines.append(
        "v25.4.1 correction: anchor entropy is compared as entropy/log2(d), dom as dom/d, "
        "and robust scale floors prevent near-constant features from dominating A."
    )
    lines.append(
        "Dynamic retention and logical purity are intentionally deferred to Stage 2 for the top-ranked successors."
    )
    lines.append(f"Weights: {meta['weights']}")
    lines.append(f"Source-anchor profile hits: {meta['source_hits']}")
    lines.append(f"Source normalized profile [entropy/log2(d), dom/d, leakage, score]: {meta['source_profile']}")
    lines.append(f"Empirical target feature scales: {meta['empirical_scales']}")
    lines.append(f"Effective feature scales after floors: {meta['effective_scales']}")
    lines.append(
        f"Original source coordinate {meta['source_anchor_left']} rank: "
        f"{meta['anchor_rank']} / {meta['n_ranked']}"
    )
    lines.append(
        f"Best coordinate within ±{meta['local_radius']} of source anchor: "
        f"{meta['best_local_source']} (global rank {meta['best_local_rank']})"
    )
    lines.append("")
    lines.append(
        "Rank | source_k | target centers | C | A | M | R | L | D | B | G | observations"
    )
    for idx, r in enumerate(rows, start=1):
        lines.append(
            f"{idx:4d} | {r.source_left:8d} | {str(r.target_centers):>16s} | "
            f"{r.continuation_score:7.4f} | {r.anchor_compatibility:6.3f} | "
            f"{r.static_contrast:6.3f} | {r.seed_recurrence:6.3f} | "
            f"{r.leakage_penalty:6.3f} | {r.drift_penalty:6.3f} | "
            f"{r.batch_recurrence:6.3f} | {r.branch_balance:6.3f} | {r.n_obs:4d}"
        )
    lines.append("")
    lines.append("=== End of v25.4.1 dimension-normalized continuation ranker ===")
    report = "\n".join(lines)
    print(report)
    Path(output_path).write_text(report + "\n", encoding="utf-8")



# ----------------------------
# v25.5 physics diagnostic
# ----------------------------

@dataclass(frozen=True)
class PhysicsDiagnosticRow:
    source_k: int
    target_centers: Tuple[int, ...]
    real_hits: int
    null_hits: int
    kdelta_real_p50: float
    kdelta_null_p50: float
    delta_k_protection: float
    chi_real_p50: float
    chi_null_p50: float
    delta_chi_protection: float
    overlap_real_p50: float
    overlap_null_p50: float
    delta_overlap: float
    retention_real: float
    retention_null: float
    delta_retention: float


def _parse_candidate_list(spec: str) -> List[int]:
    vals: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("--diag_candidates must contain at least one integer.")
    # preserve user order, remove duplicates
    out: List[int] = []
    seen = set()
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _safe_p50(vals: List[float]) -> float:
    finite = [float(x) for x in vals if np.isfinite(x)]
    return float(np.median(finite)) if finite else float("nan")


def _pair_physics_metrics(
    *,
    evals: np.ndarray,
    evecs: np.ndarray,
    H_base: np.ndarray,
    i: int,
    dH: np.ndarray,
    eta: float,
    eps_neighbor: float,
    chi_reg: float,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Diagnostic metrics for the exact adjacent pair (i,i+1).

    Returns:
      K_delta       = ||P dH Q||_F / ||dH||_F
      chi_log10     = log10(1 + weighted spectral susceptibility)
      overlap       = 0.5 Tr(P P_eta) after perturbation
      retained      = 1 if best-matched perturbed neighbor gap < eps_neighbor else 0

    Important:
      P is an eigensubspace of H_base, so P H_base Q is trivially zero.
      We therefore probe the physically meaningful perturbative coupling P dH Q.
    """
    d = int(evals.size)
    j = int(i) + 1
    if i < 0 or j >= d:
        return None

    gap0 = float(abs(float(evals[j]) - float(evals[i])))
    if gap0 >= float(eps_neighbor):
        return None

    U = np.column_stack([evecs[:, i], evecs[:, j]])  # d x 2

    # Couplings from the candidate subspace to all eigenstates under dH.
    # W[b,a] = <b|dH|a>, with a in {i,j}.
    W = evecs.conj().T @ (dH @ U)
    qmask = np.ones(d, dtype=bool)
    qmask[[i, j]] = False
    Wq = W[qmask, :]
    coupling_sq = float(np.sum(np.abs(Wq) ** 2))
    dH_norm = float(np.linalg.norm(dH, ord="fro"))
    k_delta = math.sqrt(max(0.0, coupling_sq)) / max(1e-15, dH_norm)

    # Perturbative spectral susceptibility.
    Eq = evals[qmask]
    denom_i = (Eq - float(evals[i])) ** 2 + float(chi_reg) ** 2
    denom_j = (Eq - float(evals[j])) ** 2 + float(chi_reg) ** 2
    chi_raw = float(
        np.sum(np.abs(Wq[:, 0]) ** 2 / denom_i) +
        np.sum(np.abs(Wq[:, 1]) ** 2 / denom_j)
    )
    # Normalize by total perturbation power so REAL and NULL are directly comparable.
    chi_norm = chi_raw / max(1e-15, dH_norm ** 2)
    chi_log10 = float(math.log10(1.0 + max(0.0, chi_norm)))

    # Projector retention after the same perturbation.
    Hp = H_base + float(eta) * dH
    Hp = 0.5 * (Hp + Hp.conj().T)
    evals_p, evecs_p = np.linalg.eigh(Hp)
    k, l, gap_p = _best_match_neighbor_pair(
        evals_p, float(evals[i]), float(evals[j])
    )
    U1 = np.column_stack([evecs_p[:, k], evecs_p[:, l]])
    overlap = _subspace_overlap_2d(U, U1)
    retained = 1.0 if float(gap_p) < float(eps_neighbor) else 0.0

    return float(k_delta), float(chi_log10), float(overlap), float(retained)


def physics_diagnostic_scan(
    *,
    n_qubits: int,
    from_qubits: int,
    candidates: List[int],
    n_terms: int,
    seeds_per_batch: int,
    batches: int,
    base_seed: int,
    batch_stride: int,
    eps_neighbor: float,
    eta: float,
    reps: int,
    perturb_terms: Optional[int],
    chi_reg: float,
) -> Tuple[List[PhysicsDiagnosticRow], Dict[str, Any]]:
    """
    v25.5: fixed-candidate diagnostic physics layer.

    No new continuation score is fitted here.
    The purpose is to test whether physically motivated quantities discriminate:
      - the strongest static 9Q candidates, and
      - the old 8Q-anchor neighbourhood.

    The same deterministic dH is applied to REAL and spectrum-matched Haar NULL.
    """
    if int(n_qubits) < int(from_qubits):
        raise ValueError("--n_qubits must be >= --anchor_from_qubits for diagnostic mode.")
    if int(reps) <= 0:
        raise ValueError("--diag_reps must be > 0.")
    if float(eta) <= 0.0:
        raise ValueError("--diag_eta must be > 0.")
    if float(chi_reg) <= 0.0:
        raise ValueError("--diag_chi_reg must be > 0.")

    cache = PauliCache.build(int(n_qubits))
    d = int(cache.d)
    base_dim = 1 << int(from_qubits)
    n_branches = 1 << (int(n_qubits) - int(from_qubits))
    perturb_terms_eff = int(perturb_terms) if perturb_terms is not None else int(n_terms)

    max_source = base_dim - 2
    for k in candidates:
        if int(k) < 0 or int(k) > max_source:
            raise ValueError(
                f"Diagnostic source coordinate {k} is outside legal range 0-{max_source} "
                f"for {from_qubits}Q source space."
            )

    all_seeds: List[int] = []
    for b in range(int(batches)):
        start = int(base_seed) + b * int(batch_stride)
        all_seeds.extend(start + s for s in range(int(seeds_per_batch)))

    # candidate -> model -> metric lists
    acc: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    for source_k in candidates:
        acc[int(source_k)] = {
            "REAL": {"k": [], "chi": [], "ov": [], "ret": [], "hits": []},
            "NULL": {"k": [], "chi": [], "ov": [], "ret": [], "hits": []},
        }

    for seed in all_seeds:
        H_real = build_random_pauli_hamiltonian_cached(cache, int(n_terms), int(seed))
        evals, evecs_real = np.linalg.eigh(H_real)

        # Spectrum-matched Haar NULL, same eigenvalues.
        _, evecs_null = null_haar_basis_eigs(
            evals, seed=int(seed), tag="NULL_HAAR_BASIS"
        )
        H_null = evecs_null @ np.diag(evals) @ evecs_null.conj().T
        H_null = 0.5 * (H_null + H_null.conj().T)

        for rep in range(int(reps)):
            seed_pert = stable_hash_int(
                f"V25.5|PERT|{seed}|eta{float(eta):.9g}|rep{rep}"
            )
            dH = build_random_pauli_hamiltonian_cached(
                cache, perturb_terms_eff, seed_pert
            )

            for source_k in candidates:
                centers = [
                    int(source_k) + branch * base_dim
                    for branch in range(n_branches)
                ]

                for i in centers:
                    if i < 0 or i + 1 >= d:
                        continue

                    for model, H0, V0 in (
                        ("REAL", H_real, evecs_real),
                        ("NULL", H_null, evecs_null),
                    ):
                        m = _pair_physics_metrics(
                            evals=evals,
                            evecs=V0,
                            H_base=H0,
                            i=int(i),
                            dH=dH,
                            eta=float(eta),
                            eps_neighbor=float(eps_neighbor),
                            chi_reg=float(chi_reg),
                        )
                        if m is None:
                            continue
                        kd, chi, ov, ret = m
                        acc[int(source_k)][model]["k"].append(kd)
                        acc[int(source_k)][model]["chi"].append(chi)
                        acc[int(source_k)][model]["ov"].append(ov)
                        acc[int(source_k)][model]["ret"].append(ret)
                        acc[int(source_k)][model]["hits"].append(1.0)

    rows: List[PhysicsDiagnosticRow] = []
    for source_k in candidates:
        ar = acc[int(source_k)]["REAL"]
        an = acc[int(source_k)]["NULL"]

        kr = _safe_p50(ar["k"])
        kn = _safe_p50(an["k"])
        cr = _safe_p50(ar["chi"])
        cn = _safe_p50(an["chi"])
        orr = _safe_p50(ar["ov"])
        orn = _safe_p50(an["ov"])
        rr = float(np.mean(ar["ret"])) if ar["ret"] else float("nan")
        rn = float(np.mean(an["ret"])) if an["ret"] else float("nan")

        centers = tuple(
            int(source_k) + branch * base_dim
            for branch in range(n_branches)
        )
        rows.append(PhysicsDiagnosticRow(
            source_k=int(source_k),
            target_centers=centers,
            real_hits=len(ar["hits"]),
            null_hits=len(an["hits"]),
            kdelta_real_p50=kr,
            kdelta_null_p50=kn,
            delta_k_protection=(kn - kr) if np.isfinite(kr) and np.isfinite(kn) else float("nan"),
            chi_real_p50=cr,
            chi_null_p50=cn,
            delta_chi_protection=(cn - cr) if np.isfinite(cr) and np.isfinite(cn) else float("nan"),
            overlap_real_p50=orr,
            overlap_null_p50=orn,
            delta_overlap=(orr - orn) if np.isfinite(orr) and np.isfinite(orn) else float("nan"),
            retention_real=rr,
            retention_null=rn,
            delta_retention=(rr - rn) if np.isfinite(rr) and np.isfinite(rn) else float("nan"),
        ))

    meta = {
        "n_qubits": int(n_qubits),
        "from_qubits": int(from_qubits),
        "base_dim": int(base_dim),
        "branches": int(n_branches),
        "seeds": len(all_seeds),
        "reps": int(reps),
        "eta": float(eta),
        "chi_reg": float(chi_reg),
        "perturb_terms": int(perturb_terms_eff),
        "candidates": [int(x) for x in candidates],
    }
    return rows, meta


def write_physics_diagnostic_report(
    output_path: str,
    rows: List[PhysicsDiagnosticRow],
    meta: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("=== Soft Spaces Phase 2 v25.5 PHYSICS DIAGNOSTIC ===")
    lines.append(
        f"Fixed candidates: {meta['candidates']} | "
        f"{meta['from_qubits']}Q source geometry -> {meta['n_qubits']}Q target "
        f"({meta['branches']} descendant branches)"
    )
    lines.append(
        f"Seeds={meta['seeds']} | perturb reps/seed={meta['reps']} | "
        f"eta={meta['eta']} | dH_terms={meta['perturb_terms']} | "
        f"chi_regularizer={meta['chi_reg']}"
    )
    lines.append("")
    lines.append(
        "This is deliberately NOT a new continuation score. "
        "It tests three physically motivated diagnostics on a fixed candidate set."
    )
    lines.append(
        "K_delta = ||P dH Q||_F / ||dH||_F. Lower REAL than NULL => stronger perturbative protection."
    )
    lines.append(
        "chi = log10(1 + normalized spectral susceptibility). Lower REAL than NULL => weaker mixing susceptibility."
    )
    lines.append(
        "R_P = 0.5 Tr(P P_eta). Higher REAL than NULL => stronger basis-invariant subspace retention."
    )
    lines.append(
        "Retention = fraction whose energy-matched perturbed neighbor pair still has gap < eps_neighbor."
    )
    lines.append("")
    lines.append(
        "source_k | target centers | hits R/N | "
        "K_R | K_N | ΔK(N-R) | chi_R | chi_N | Δchi(N-R) | "
        "Ov_R | Ov_N | ΔOv(R-N) | Ret_R | Ret_N | ΔRet"
    )

    for r in rows:
        lines.append(
            f"{r.source_k:8d} | {str(r.target_centers):>16s} | "
            f"{r.real_hits:4d}/{r.null_hits:<4d} | "
            f"{r.kdelta_real_p50:7.4f} | {r.kdelta_null_p50:7.4f} | {r.delta_k_protection:+9.4f} | "
            f"{r.chi_real_p50:7.4f} | {r.chi_null_p50:7.4f} | {r.delta_chi_protection:+11.4f} | "
            f"{r.overlap_real_p50:7.4f} | {r.overlap_null_p50:7.4f} | {r.delta_overlap:+10.4f} | "
            f"{r.retention_real:6.3f} | {r.retention_null:6.3f} | {r.delta_retention:+7.3f}"
        )

    lines.append("")
    lines.append("Reading rule:")
    lines.append(
        "  Favorable REAL-specific protection points in the direction: ΔK > 0, Δchi > 0, ΔOv > 0, ΔRet > 0."
    )
    lines.append(
        "  Do not combine these into a fitted score yet; first ask whether the old anchor neighbourhood "
        "and/or the static top candidates separate consistently from NULL."
    )
    lines.append("")
    lines.append("=== End of v25.5 physics diagnostic ===")

    report = "\n".join(lines)
    print(report)
    Path(output_path).write_text(report + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Soft Spaces Phase 2 v25.5 physics diagnostic + v25.4.1 continuation ranker.")
    ap.add_argument("--n_qubits", type=int, default=4)
    ap.add_argument("--n_terms", type=int, default=5)
    ap.add_argument("--seeds_per_batch", type=int, default=5000)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--batch_stride", type=int, default=1000000)
    ap.add_argument("--eps_neighbor", type=float, default=0.05)
    ap.add_argument("--target_zones", type=str, default=None,
                    help="Direct inclusive left-eigenpair index intervals. Ignored when --anchor is supplied.")
    ap.add_argument("--anchor", type=str, default=None,
                    help="Phase-2 continuation prior as adjacent source anchor, e.g. 157-158.")
    ap.add_argument("--anchor_from_qubits", type=int, default=None,
                    help="Qubit count at which --anchor was identified, e.g. 8.")
    ap.add_argument("--anchor_radius", type=int, default=16,
                    help="Half-width in left-eigenpair indices around each predicted descendant.")
    ap.add_argument("--anchor_mode", choices=["anchors", "controls"], default="anchors",
                    help="Analyse predicted anchor windows or one geometry-matched paired control set.")
    ap.add_argument("--control_offsets", type=str, default="-64,64",
                    help="Signed source-space offsets used to construct matched paired controls, e.g. -64,64.")
    ap.add_argument("--control_set", type=int, default=1,
                    help="1-based control-set number used when --anchor_mode controls.")

    # v25.4: rank all legal nQ -> (n+1)Q successor coordinates.
    ap.add_argument("--continuation_scan", action="store_true",
                    help="Run the v25.4 static-first continuation ranker and exit.")
    ap.add_argument("--continuation_top", type=int, default=25,
                    help="Number of top successor coordinates to report.")
    ap.add_argument("--w_anchor", type=float, default=0.25)
    ap.add_argument("--w_static", type=float, default=0.30)
    ap.add_argument("--w_recurrence", type=float, default=0.15)
    ap.add_argument("--w_leak", type=float, default=0.05)
    ap.add_argument("--w_drift", type=float, default=0.10)
    ap.add_argument("--w_batch", type=float, default=0.05)
    ap.add_argument("--w_branch", type=float, default=0.10)

    # v25.5: fixed-candidate physics diagnostic; does not fit a new continuation score.
    ap.add_argument("--physics_diagnostic", action="store_true",
                    help="Run v25.5 K_delta / spectral susceptibility / projector-retention diagnostic and exit.")
    ap.add_argument("--diag_candidates", type=str,
                    default="67,193,180,192,131,251,157,160,165,0,114",
                    help="Comma-separated source coordinates to test.")
    ap.add_argument("--diag_eta", type=float, default=0.01,
                    help="Perturbation strength used for projector-retention diagnostic.")
    ap.add_argument("--diag_reps", type=int, default=3,
                    help="Independent deterministic perturbations per seed.")
    ap.add_argument("--diag_terms", type=int, default=None,
                    help="Pauli terms in diagnostic dH; default = --n_terms.")
    ap.add_argument("--diag_chi_reg", type=float, default=1e-3,
                    help="Energy regularizer in spectral susceptibility denominator.")

    ap.add_argument("--keep_mass", type=float, default=0.90)
    ap.add_argument("--ent_step", type=float, default=0.1)
    ap.add_argument("--leak_step", type=float, default=0.05)
    ap.add_argument("--times", type=float, nargs="+", default=[0.5, 1.0, 1.5])

    ap.add_argument("--stable_frac", type=float, default=0.01)
    ap.add_argument("--stable_leak_max", type=float, default=None)
    ap.add_argument("--stable_leak_quantile", type=float, default=None)

    # v19: perturbation robustness controls (disabled unless --perturb_eta is provided and --perturb_seeds > 0)
    ap.add_argument("--perturb_eta", type=float, nargs="+", default=[])
    ap.add_argument("--perturb_reps", type=int, default=3)
    ap.add_argument("--perturb_seeds", type=int, default=0, help="Number of seeds per batch used for perturbation robustness (0 disables).")
    ap.add_argument("--perturb_terms", type=int, default=None, help="Number of Pauli terms in ΔH (default: same as --n_terms).")
    ap.add_argument("--perturb_pairs_per_seed", type=int, default=5, help="Number of near-degenerate pairs per seed used in robustness summary (lowest-score subset).")
    ap.add_argument("--iso_eps", type=float, default=1e-8, help="Clamp for ΔE_in when forming log isolation ratio log10(ΔE_out/max(ΔE_in, iso_eps)).")
    ap.add_argument("--no_iso_quartiles", action="store_true", help="Disable logR quartile table in perturbation output.")
    ap.add_argument("--iso_quartiles_all_eta", action="store_true", help="Print logR quartile tables for all eta values (default: only max eta).")
    # v22: open-system (Lindblad) evaluation
    ap.add_argument("--open_system", action="store_true", help="Enable open-system (Lindblad) logical retention test on selected 2D subspaces.")
    ap.add_argument("--os_include_baselines", action="store_true", help="If set, evaluate REAL_Q4, REAL_Q1, and NULL_Q4 in each batch.")
    ap.add_argument("--os_pairs_per_batch", type=int, default=400, help="Number of near-degenerate pairs sampled per category per batch for open-system evaluation.")
    ap.add_argument("--os_noise_model", type=str, default="dephasing", choices=["dephasing", "amp_damp", "both", "depolar"], help="Noise model for Lindblad evolution.")
    ap.add_argument("--os_gamma_phi", type=float, default=0.01, help="Dephasing (or depolar) rate gamma.")
    ap.add_argument("--os_gamma_1", type=float, default=0.01, help="Amplitude damping rate gamma_1.")
    ap.add_argument("--os_t_max", type=float, default=5.0, help="Max evolution time for open-system evaluation.")
    ap.add_argument("--os_t_steps", type=int, default=25, help="Number of time points for open-system evaluation.")
    ap.add_argument("--os_dt_internal", type=float, default=None, help="Optional internal RK4 step size. If None, one RK4 step per output interval.")
    ap.add_argument("--os_states_mode", type=str, default="zx", choices=["z", "zx"], help="Logical probe states: 'z'=(|0_L>,|1_L>), 'zx'=adds |+_L>,|i+_L>.")
    ap.add_argument("--logical_basis", type=str, default="eigen", choices=["eigen", "noise_diag"],
                    help="Open-system: logical basis inside the 2D subspace. 'eigen' uses the eigenpair basis; "
                         "'noise_diag' diagonalizes the projected noise intensity A=Σ (P L_k P)†(P L_k P).")
    ap.add_argument("--os_noise_qubits", type=str, default="all", choices=["all", "subset"],
                    help="Open-system: apply noise jumps on all qubits or only a subset.")
    ap.add_argument("--os_noise_subset", type=int, nargs="+", default=None,
                    help="Open-system: if --os_noise_qubits=subset, list 0-based qubit indices (e.g., 0 1). If omitted, defaults to [0].")
    ap.add_argument("--os_use_stable_pool", action="store_true", help="Restrict open-system sampling to the stable pool (same stable selection per model).")
    ap.add_argument("--os_report_quantiles", type=float, nargs=3, default=[0.1, 0.5, 0.9], help="Quantiles to report for time series summaries, e.g., 0.1 0.5 0.9.")
    ap.add_argument("--os_time_mode", type=str, default="mean", choices=["snapshot", "mean", "auc", "late", "peak", "peak_post_init"], help="How to reduce open-system time curves to one scalar per pair for summary reporting. peak is legacy max over all times; peak_post_init excludes t=0 before taking the max.")
    ap.add_argument("--os_time_late_power", type=float, default=2.0, help="Power used for late-time weighting when --os_time_mode=late.")
    ap.add_argument("--contrast_focus", action="store_true", help="Prioritize matched REAL/NULL subspaces with high static contrast when sampling open-system pairs.")
    ap.add_argument("--kernel_zoom", action="store_true", help="Enable hierarchical zoom into local contrast kernels after building the contrast pool.")
    ap.add_argument("--kernel_top_centers", type=int, default=8, help="How many top contrast centers to keep in the kernel-zoom report per batch.")
    ap.add_argument("--kernel_radius", type=int, default=2, help="Neighborhood radius in (i,j)-pair space for kernel zoom.")
    ap.add_argument("--kernel_min_neighbors", type=int, default=3, help="Minimum matched neighbors required to emit a kernel row.")

    # v24.3: interference test (fotonik-inspireret)
    ap.add_argument("--interference_test", action="store_true", help="Enable interference analysis on sampled 2D subspaces.")
    ap.add_argument("--if_pairs_per_batch", type=int, default=200, help="Number of near-degenerate pairs sampled per category per batch for interference analysis.")
    ap.add_argument("--if_use_stable_pool", action="store_true", help="Restrict interference sampling to the stable pool (same stable selection per model).")

    ap.add_argument("--topK", type=int, default=25)
    ap.add_argument("--min_overall", type=int, default=3)
    ap.add_argument("--min_stable", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)

    ap.add_argument("--q_bins", type=int, default=10)
    ap.add_argument("--q_bins_coarse", type=int, default=6)
    ap.add_argument("--p_tail_max", type=float, default=None)
    ap.add_argument("--bootstrap", type=int, default=200)

    ap.add_argument("--output", type=str, default="v24_9_2_output.txt")
    args = ap.parse_args()


    if bool(args.physics_diagnostic):
        if args.anchor_from_qubits is None:
            raise ValueError("--physics_diagnostic requires --anchor_from_qubits")
        diag_candidates = _parse_candidate_list(args.diag_candidates)
        rows, meta = physics_diagnostic_scan(
            n_qubits=args.n_qubits,
            from_qubits=args.anchor_from_qubits,
            candidates=diag_candidates,
            n_terms=args.n_terms,
            seeds_per_batch=args.seeds_per_batch,
            batches=args.batches,
            base_seed=args.base_seed,
            batch_stride=args.batch_stride,
            eps_neighbor=args.eps_neighbor,
            eta=args.diag_eta,
            reps=args.diag_reps,
            perturb_terms=args.diag_terms,
            chi_reg=args.diag_chi_reg,
        )
        write_physics_diagnostic_report(args.output, rows, meta)
        return

    if bool(args.continuation_scan):
        if args.anchor is None or args.anchor_from_qubits is None:
            raise ValueError("--continuation_scan requires --anchor and --anchor_from_qubits")
        a0, _a1 = parse_anchor_pair(args.anchor)
        rows, meta = continuation_rank_scan(
            n_qubits=args.n_qubits,
            anchor_from_qubits=args.anchor_from_qubits,
            anchor_left=a0,
            n_terms=args.n_terms,
            seeds_per_batch=args.seeds_per_batch,
            batches=args.batches,
            base_seed=args.base_seed,
            batch_stride=args.batch_stride,
            eps_neighbor=args.eps_neighbor,
            ent_step=args.ent_step,
            leak_step=args.leak_step,
            times=args.times,
            keep_mass=args.keep_mass,
            iso_eps=args.iso_eps,
            top_n=args.continuation_top,
            w_anchor=args.w_anchor,
            w_static=args.w_static,
            w_recurrence=args.w_recurrence,
            w_leak=args.w_leak,
            w_drift=args.w_drift,
            w_batch=args.w_batch,
            w_branch=args.w_branch,
        )
        write_continuation_report(
            args.output, rows, meta,
            n_qubits=args.n_qubits,
            anchor_from_qubits=args.anchor_from_qubits,
            anchor_spec=args.anchor,
            seeds_per_batch=args.seeds_per_batch,
            batches=args.batches,
            base_seed=args.base_seed,
        )
        return

    cache = PauliCache.build(args.n_qubits)
    d = cache.d

    anchor_windows: List[IndexInterval] = []
    control_windows: List[IndexInterval] = []
    anchor_centers: List[int] = []

    if args.anchor is not None:
        if args.anchor_from_qubits is None:
            raise ValueError("--anchor_from_qubits is required when --anchor is used")

        a0, a1 = parse_anchor_pair(args.anchor)
        anchor_centers = anchor_descendant_centers(
            anchor_left=a0,
            from_qubits=int(args.anchor_from_qubits),
            to_qubits=int(args.n_qubits),
        )
        anchor_windows = windows_around_centers(
            anchor_centers, int(args.anchor_radius), d
        )
        control_offsets = parse_int_offsets(args.control_offsets)
        control_sets = paired_control_sets(
            anchor_left=a0,
            from_qubits=int(args.anchor_from_qubits),
            to_qubits=int(args.n_qubits),
            offsets=control_offsets,
            radius=int(args.anchor_radius),
            d=d,
            anchor_windows=anchor_windows,
        )

        if args.anchor_mode == "anchors":
            target_intervals = anchor_windows
        else:
            cs = int(args.control_set)
            if cs < 1 or cs > len(control_sets):
                raise ValueError(
                    f"--control_set must be between 1 and {len(control_sets)} "
                    f"for offsets {control_offsets}."
                )
            target_intervals = control_sets[cs - 1]
    else:
        control_offsets = []
        control_sets = []
        target_intervals = parse_index_intervals(args.target_zones, d)

    header = []
    header.append("=== Soft Spaces Phase 2 v25.4 (legacy v25.3.1 matched-control path) ===")
    header.append(f"Qubits: {args.n_qubits} (d={d}) | terms={args.n_terms}")
    header.append(f"Batches: {args.batches} × {args.seeds_per_batch} seeds (base_seed={args.base_seed}, stride={args.batch_stride})")
    header.append(f"Neighbor eps={args.eps_neighbor:.3f}")
    if args.anchor is not None:
        header.append(
            f"Anchor prior: {args.anchor} from {args.anchor_from_qubits}Q -> {args.n_qubits}Q | "
            f"predicted left-index centers={anchor_centers} | radius={args.anchor_radius}"
        )
        header.append(f"Predicted anchor windows: {format_intervals(anchor_windows)}")
        for j, (off, cset) in enumerate(zip(control_offsets, control_sets), start=1):
            header.append(
                f"Matched control set {j}: source offset={off:+d} | "
                f"windows={format_intervals(cset)}"
            )
        active_label = "ANCHOR" if args.anchor_mode == "anchors" else f"CONTROL SET {int(args.control_set)}"
        header.append(f"Active analysis mode: {active_label} | active intervals: {format_intervals(target_intervals)}")
    else:
        header.append(f"Direct target zones (left pair index i): {format_intervals(target_intervals)}")
    header.append(f"Dominant set: keep_mass={args.keep_mass:.2f} (mass-based; guarantees non-empty mask)")
    header.append(f"Bins (SigAbs): ent_step={args.ent_step:.3f} | leak_step={args.leak_step:.3f}")
    header.append(f"Bins (SigQ_GLOBAL fine): q_bins={args.q_bins} (pooled REAL+NULL per batch)")
    header.append(f"Bins (SigQG_COARSE): q_bins_coarse={args.q_bins_coarse} + dom_bin(dom/d) + leak_bin_coarse (pooled REAL+NULL per batch)")
    header.append(f"Leakage proxy times={args.times} (FAST analytic evolution in eigenpair)")
    header.append(f"Stable selection: stable_frac={args.stable_frac:.3f} per model (optional leak constraint max={args.stable_leak_max}, q={args.stable_leak_quantile})")
    header.append(f"Perturbation robustness: eta={args.perturb_eta} | reps={args.perturb_reps} | seeds={args.perturb_seeds} | pairs/seed={args.perturb_pairs_per_seed} | dH_terms={args.perturb_terms}")
    header.append(f"Open-system (v23): enabled={bool(args.open_system)} | include_baselines={bool(args.os_include_baselines)} | pairs_per_cat={args.os_pairs_per_batch} | noise={args.os_noise_model}(phi={args.os_gamma_phi},g1={args.os_gamma_1}) | t={args.os_t_max}/{args.os_t_steps} | states={args.os_states_mode} | basis={args.logical_basis} | noise_qubits={args.os_noise_qubits}:{args.os_noise_subset} | stable_pool={bool(args.os_use_stable_pool)} | time_mode={args.os_time_mode} | late_power={args.os_time_late_power} | contrast_focus={bool(args.contrast_focus)} | kernel_zoom={bool(args.kernel_zoom)}")
    header.append(f"Interference test: enabled={bool(args.interference_test)} | pairs_per_cat={args.if_pairs_per_batch} | stable_pool={bool(args.if_use_stable_pool)}")
    header.append(f"TopK={args.topK} | min_overall={args.min_overall} | min_stable={args.min_stable} | alpha={args.alpha}")
    header.append(f"Optional family filter: p_tail_max={args.p_tail_max}")
    header.append("Pauli ops: lazy on-demand cache; v25.4 continuation ranker added, legacy matched-control physics preserved")
    header.append("")
    header_text = "\n".join(header)

    with open(args.output, "w", encoding="utf-8") as f:
        def out(s: str = "") -> None:
            try:
                print(s)
            except UnicodeEncodeError:
                safe = s.encode("ascii", "replace").decode("ascii")
                print(safe)
            f.write(s + "\n")

        out(header_text)

        real, null, scoreboards, perturb_summaries, open_system_summaries, interference_summaries = run_paired_batches(
            cache=cache,
            n_terms=args.n_terms,
            seeds_per_batch=args.seeds_per_batch,
            n_batches=args.batches,
            base_seed=args.base_seed,
            batch_stride=args.batch_stride,
            eps_neighbor=args.eps_neighbor,
            ent_step=args.ent_step,
            leak_step=args.leak_step,
            times=list(args.times),
            keep_mass=args.keep_mass,
    iso_eps=args.iso_eps,
            stable_frac=args.stable_frac,
            stable_leak_max=args.stable_leak_max,
            stable_leak_quantile=args.stable_leak_quantile,
            topK=args.topK,
            min_overall=args.min_overall,
            min_stable=args.min_stable,
            alpha=args.alpha,
            q_bins=args.q_bins,
            q_bins_coarse=args.q_bins_coarse,
            p_tail_max=args.p_tail_max,
            bootstrap=args.bootstrap,
            perturb_eta=list(args.perturb_eta),
            perturb_reps=args.perturb_reps,
            perturb_seeds=args.perturb_seeds,
            perturb_terms=args.perturb_terms,
            perturb_pairs_per_seed=args.perturb_pairs_per_seed,
            open_system=bool(args.open_system),
            os_include_baselines=bool(args.os_include_baselines),
            os_pairs_per_batch=int(args.os_pairs_per_batch),
            os_noise_model=str(args.os_noise_model),
            os_gamma_phi=float(args.os_gamma_phi),
            os_gamma_1=float(args.os_gamma_1),
            os_t_max=float(args.os_t_max),
            os_t_steps=int(args.os_t_steps),
            os_dt_internal=args.os_dt_internal,
            os_states_mode=str(args.os_states_mode),
            os_logical_basis=str(args.logical_basis),
            os_noise_qubits=str(args.os_noise_qubits),
            os_noise_subset=args.os_noise_subset,
            os_use_stable_pool=bool(args.os_use_stable_pool),
            os_report_quantiles=tuple(float(x) for x in args.os_report_quantiles),
            os_time_mode=str(args.os_time_mode),
            os_time_late_power=float(args.os_time_late_power),
            contrast_focus=bool(args.contrast_focus),
            kernel_zoom=bool(args.kernel_zoom),
            kernel_top_centers=int(args.kernel_top_centers),
            kernel_radius=int(args.kernel_radius),
            kernel_min_neighbors=int(args.kernel_min_neighbors),
            interference_test=bool(args.interference_test),
            if_pairs_per_batch=int(args.if_pairs_per_batch),
            if_use_stable_pool=bool(args.if_use_stable_pool),
            target_intervals=target_intervals,
        )

        out("")
        for b in range(args.batches):
            out("----------------------------------------------")
            out(f"Batch {b+1}/{args.batches} (seed_offset={real[b].seed_offset})")

            R = real[b]
            out(f"Model: REAL | candidates={R.n_candidates} | stable_rate={R.stable_rate:.4f} (score-only ref={R.stable_rate_scoreonly:.4f})")
            out(f"  score(mean/median/max)={R.score_stats[0]:.3f}/{R.score_stats[1]:.3f}/{R.score_stats[2]:.3f}")
            out(f"  leakage(mean/median/min)={R.leak_stats[0]:.3f}/{R.leak_stats[1]:.3f}/{R.leak_stats[2]:.3f}")
            out(f"  entropy(mean/median/max)={R.ent_stats[0]:.3f}/{R.ent_stats[1]:.3f}/{R.ent_stats[2]:.3f}")
            out(f"  dom_count(mean/median/max)={R.dom_stats[0]:.2f}/{R.dom_stats[1]:.1f}/{R.dom_stats[2]:.0f}")
            out(f"  gaps: ΔE_in(mean/med/p90)={R.gap_in_stats[0]:.4g}/{R.gap_in_stats[1]:.4g}/{R.gap_in_stats[2]:.4g} | ΔE_out(mean/med/p90)={R.gap_out_stats[0]:.4g}/{R.gap_out_stats[1]:.4g}/{R.gap_out_stats[2]:.4g} | logR(mean/med/p90)={R.iso_log_stats[0]:.3f}/{R.iso_log_stats[1]:.3f}/{R.iso_log_stats[2]:.3f}")
            out(f"  gaps_cond(ΔE_in>iso_eps): n={R.gap_in_cond_n} | ΔE_in(mean/med/p90)={R.gap_in_cond_stats[0]:.4g}/{R.gap_in_cond_stats[1]:.4g}/{R.gap_in_cond_stats[2]:.4g}")
            out(f"  entropy effect (stable-overall)={R.effect_entropy_bits:+.3f} bits  CI95={R.effect_ci}")
            out("")
            out(format_top(R.top_qg, "REAL, SigQ_GLOBAL (fine)"))
            out("")
            out(format_top(R.top_qg_coarse, "REAL, SigQG_COARSE"))
            out("")

            N = null[b]
            out(f"Model: NULL_HAAR_BASIS | candidates={N.n_candidates} | stable_rate={N.stable_rate:.4f} (score-only ref={N.stable_rate_scoreonly:.4f})")
            out(f"  score(mean/median/max)={N.score_stats[0]:.3f}/{N.score_stats[1]:.3f}/{N.score_stats[2]:.3f}")
            out(f"  leakage(mean/median/min)={N.leak_stats[0]:.3f}/{N.leak_stats[1]:.3f}/{N.leak_stats[2]:.3f}")
            out(f"  entropy(mean/median/max)={N.ent_stats[0]:.3f}/{N.ent_stats[1]:.3f}/{N.ent_stats[2]:.3f}")
            out(f"  dom_count(mean/median/max)={N.dom_stats[0]:.2f}/{N.dom_stats[1]:.1f}/{N.dom_stats[2]:.0f}")
            out(f"  gaps: ΔE_in(mean/med/p90)={N.gap_in_stats[0]:.4g}/{N.gap_in_stats[1]:.4g}/{N.gap_in_stats[2]:.4g} | ΔE_out(mean/med/p90)={N.gap_out_stats[0]:.4g}/{N.gap_out_stats[1]:.4g}/{N.gap_out_stats[2]:.4g} | logR(mean/med/p90)={N.iso_log_stats[0]:.3f}/{N.iso_log_stats[1]:.3f}/{N.iso_log_stats[2]:.3f}")
            out(f"  gaps_cond(ΔE_in>iso_eps): n={N.gap_in_cond_n} | ΔE_in(mean/med/p90)={N.gap_in_cond_stats[0]:.4g}/{N.gap_in_cond_stats[1]:.4g}/{N.gap_in_cond_stats[2]:.4g}")
            out(f"  entropy effect (stable-overall)={N.effect_entropy_bits:+.3f} bits  CI95={N.effect_ci}")
            out("")
            out(format_top(N.top_qg, "NULL, SigQ_GLOBAL (fine)"))
            out("")
            out(format_top(N.top_qg_coarse, "NULL, SigQG_COARSE"))
            out("")

            # v19: Perturbation robustness summary (optional)
            ps = perturb_summaries[b] if b < len(perturb_summaries) else {}
            if ps:
                out("Perturbation robustness (2D subspace overlap, evidence-grade):")
                out(f"  settings: eta={args.perturb_eta} | reps={args.perturb_reps} | seeds_used={min(args.perturb_seeds, args.seeds_per_batch)} | pairs/seed={args.perturb_pairs_per_seed} | dH_terms={args.perturb_terms}")
                for eta in sorted(ps.keys()):
                    eta_f = float(eta)
                    for model in ["REAL", "NULL"]:
                        mtr = ps[eta_f][model]
                        out(
                            f"  eta={eta_f:.6g} | {model}: "
                            f"pair_retention={mtr['pair_retention_rate']:.3f} | "
                            f"subspace_ov_ret(mean/p10/p50/p90)="
                            f"{mtr['subspace_overlap_retained_mean']:.3f}/"
                            f"{mtr['subspace_overlap_retained_p10']:.3f}/"
                            f"{mtr['subspace_overlap_retained_p50']:.3f}/"
                            f"{mtr['subspace_overlap_retained_p90']:.3f} | "
                            f"subspace_ov_all_mean={mtr['subspace_overlap_all_mean']:.3f} | "f"logR_ret(p50)={mtr.get('iso_log_ret_p50', float('nan')):.2f} | "f"corr(logR,ov)_ret={mtr.get('corr_iso_log_ov_ret', float('nan')):.2f} | "
                            f"sig_coarse_ret={mtr['sig_coarse_retention_rate']:.3f} | "
                            f"mean|Δentropy|={mtr['mean_abs_d_entropy_bits']:.3f} bits | "
                            f"mean|Δleak|={mtr['mean_abs_d_leak']:.3f} | "
                            f"pairs={int(mtr['pairs_total'])} | retained_pairs={int(mtr['retained_pairs_total'])}"
                        )
                        # v21.1: logR quartile table (retained subset) for interpretability
                        if (not args.no_iso_quartiles) and (args.iso_quartiles_all_eta or eta_f == max(sorted(ps.keys()))):
                            qs = mtr.get("logR_quartiles_ret", [])
                            edges = mtr.get("logR_quartile_edges", [float("nan"), float("nan"), float("nan")])
                            if qs:
                                q25s = f"{edges[0]:.3f}" if np.isfinite(edges[0]) else "nan"
                                q50s = f"{edges[1]:.3f}" if np.isfinite(edges[1]) else "nan"
                                q75s = f"{edges[2]:.3f}" if np.isfinite(edges[2]) else "nan"
                                out(f"    logR quartiles (retained): q25={q25s} q50={q50s} q75={q75s}")
                                for row in qs:
                                    p10 = row.get("ov_p10", float("nan"))
                                    p50 = row.get("ov_p50", float("nan"))
                                    p90 = row.get("ov_p90", float("nan"))
                                    p10s = f"{p10:.3f}" if np.isfinite(p10) else "nan"
                                    p50s = f"{p50:.3f}" if np.isfinite(p50) else "nan"
                                    p90s = f"{p90:.3f}" if np.isfinite(p90) else "nan"
                                    out(f"      {row.get('q', 'Q?')}: n={int(row.get('n', 0))} | ov(p10/p50/p90)={p10s}/{p50s}/{p90s}")

                out("")

            # Open-system (v23) summary (optional)
            if bool(args.open_system):
                osb = open_system_summaries[b] if b < len(open_system_summaries) else {}
                meta = osb.get("_meta", {}) if isinstance(osb, dict) else {}
                if isinstance(osb, dict) and meta:
                    out("Open-system logical retention (Lindblad; v23):")
                    out(
                        f"  settings: noise={meta.get('noise_model')} | gamma_phi={meta.get('gamma_phi')} | "
                        f"gamma_1={meta.get('gamma_1')} | t_max={meta.get('t_max')} | t_steps={meta.get('t_steps')} | "
                        f"states={meta.get('states_mode')} | stable_pool={meta.get('use_stable_pool')} | pairs_per_cat={meta.get('os_pairs_per_batch')} | "
                        f"time_mode={meta.get('time_mode')} | late_power={meta.get('time_late_power')} | contrast_focus={meta.get('contrast_focus')} | kernel_zoom={meta.get('kernel_zoom')} | kernel_rows={meta.get('kernel_rows')} | radius_used={meta.get('kernel_radius_used')} | min_neighbors_used={meta.get('kernel_min_neighbors_used')}"
                    )

                    def _fmt_cat(cat: str) -> None:
                        if cat not in osb:
                            return
                        S = osb.get(cat, {})
                        n_pairs = int(S.get("n_pairs", 0)) if isinstance(S, dict) else 0
                        if n_pairs <= 0:
                            out(f"  {cat}: n_pairs=0")
                            return
                        t = np.array(S["t"], dtype=float)
                        mid = int((t.size - 1) // 2)
                        idxs = [0, mid, int(t.size - 1)]

                        def _pt(i: int) -> str:
                            return f"t={t[i]:.3g}"

                        leak = S["leak"]
                        fu = S["fid_uncond"]
                        fc = S["fid_cond"]

                        out(f"  {cat}: n_pairs={n_pairs} | t_fid90_med={S.get('t_fid90_med', float('nan')):.3g} | t_leak10_med={S.get('t_leak10_med', float('nan')):.3g}")
                        if bool(S.get("contrast_focus", False)):
                            cs = S.get("contrast_score", {})
                            if isinstance(cs, dict):
                                out(f"    contrast_score(q10/q50/q90/mean)={cs.get('q_lo', float('nan')):.3f}/{cs.get('q_med', float('nan')):.3f}/{cs.get('q_hi', float('nan')):.3f}/{cs.get('mean', float('nan')):.3f}")
                        out("    leak(q10/q50/q90): " + " | ".join(
                            f"{_pt(i)}={leak['q_lo'][i]:.3f}/{leak['q_med'][i]:.3f}/{leak['q_hi'][i]:.3f}" for i in idxs
                        ))
                        out("    fid_uncond(q10/q50/q90): " + " | ".join(
                            f"{_pt(i)}={fu['q_lo'][i]:.3f}/{fu['q_med'][i]:.3f}/{fu['q_hi'][i]:.3f}" for i in idxs
                        ))
                        out("    fid_cond(q10/q50/q90): " + " | ".join(
                            f"{_pt(i)}={fc['q_lo'][i]:.3f}/{fc['q_med'][i]:.3f}/{fc['q_hi'][i]:.3f}" for i in idxs
                        ))
                        subret = S.get("subspace_retention", {})
                        coh = S.get("logical_coherence", {})
                        pur = S.get("logical_purity", {})
                        drift = S.get("structure_drift", {})
                        out("    subspace_ret(q10/q50/q90): " + " | ".join(
                            f"{_pt(i)}={subret['q_lo'][i]:.3f}/{subret['q_med'][i]:.3f}/{subret['q_hi'][i]:.3f}" for i in idxs
                        ))
                        out("    logical_coh(q10/q50/q90): " + " | ".join(
                            f"{_pt(i)}={coh['q_lo'][i]:.3f}/{coh['q_med'][i]:.3f}/{coh['q_hi'][i]:.3f}" for i in idxs
                        ))
                        out("    structure_drift(q10/q50/q90): " + " | ".join(
                            f"{_pt(i)}={drift['q_lo'][i]:.3f}/{drift['q_med'][i]:.3f}/{drift['q_hi'][i]:.3f}" for i in idxs
                        ))
                        red = S.get("reduced", {})
                        out(f"    AUC medians: fid_uncond={fu.get('auc_med', float('nan')):.3f} | leak={leak.get('auc_med', float('nan')):.3f} | subspace_ret={subret.get('auc_med', float('nan')):.3f} | logical_coh={coh.get('auc_med', float('nan')):.3f}")
                        if isinstance(red, dict):
                            out(
                                f"    Reduced[{red.get('mode', 'mean')}] q10/q50/q90: "
                                f"fid_uncond={red.get('fid_uncond', {}).get('q_lo', float('nan')):.3f}/{red.get('fid_uncond', {}).get('q_med', float('nan')):.3f}/{red.get('fid_uncond', {}).get('q_hi', float('nan')):.3f} | "
                                f"fid_cond={red.get('fid_cond', {}).get('q_lo', float('nan')):.3f}/{red.get('fid_cond', {}).get('q_med', float('nan')):.3f}/{red.get('fid_cond', {}).get('q_hi', float('nan')):.3f} | "
                                f"leak={red.get('leak', {}).get('q_lo', float('nan')):.3f}/{red.get('leak', {}).get('q_med', float('nan')):.3f}/{red.get('leak', {}).get('q_hi', float('nan')):.3f}"
                            )
                            out(
                                f"    Reduced[{red.get('mode', 'mean')}] structure q10/q50/q90: "
                                f"subspace_ret={red.get('subspace_retention', {}).get('q_lo', float('nan')):.3f}/{red.get('subspace_retention', {}).get('q_med', float('nan')):.3f}/{red.get('subspace_retention', {}).get('q_hi', float('nan')):.3f} | "
                                f"logical_coh={red.get('logical_coherence', {}).get('q_lo', float('nan')):.3f}/{red.get('logical_coherence', {}).get('q_med', float('nan')):.3f}/{red.get('logical_coherence', {}).get('q_hi', float('nan')):.3f} | "
                                f"logical_purity={red.get('logical_purity', {}).get('q_lo', float('nan')):.3f}/{red.get('logical_purity', {}).get('q_med', float('nan')):.3f}/{red.get('logical_purity', {}).get('q_hi', float('nan')):.3f} | "
                                f"structure_drift={red.get('structure_drift', {}).get('q_lo', float('nan')):.3f}/{red.get('structure_drift', {}).get('q_med', float('nan')):.3f}/{red.get('structure_drift', {}).get('q_hi', float('nan')):.3f}"
                            )

                    _fmt_cat("REAL_Q4")
                    _fmt_cat("REAL_Q1")
                    _fmt_cat("NULL_Q4")
                    d_os = open_system_delta_summary(osb)
                    if d_os:
                        out(
                            "  REAL_Q4 - NULL_Q4 deltas (reduced q50): "
                            f"fid_uncond={d_os.get('delta_reduced_fid_uncond_q50', float('nan')):+.4f} | "
                            f"fid_cond={d_os.get('delta_reduced_fid_cond_q50', float('nan')):+.4f} | "
                            f"leak={d_os.get('delta_reduced_leak_q50', float('nan')):+.4f} | "
                            f"subspace_ret={d_os.get('delta_reduced_subspace_retention_q50', float('nan')):+.4f} | "
                            f"logical_coh={d_os.get('delta_reduced_logical_coherence_q50', float('nan')):+.4f} | "
                            f"logical_purity={d_os.get('delta_reduced_logical_purity_q50', float('nan')):+.4f} | "
                            f"structure_drift={d_os.get('delta_reduced_structure_drift_q50', float('nan')):+.4f} | "
                            f"t_fid90_med={d_os.get('delta_t_fid90_med', float('nan')):+.3g} | "
                            f"t_leak10_med={d_os.get('delta_t_leak10_med', float('nan')):+.3g}"
                        )
                    hs = osb.get("_hotspots_dynamic", []) if isinstance(osb, dict) else []
                    if hs:
                        out("  Hotspot evidence (sparse profitable singleton search):")
                        for row in hs[: min(8, len(hs))]:
                            ck = row.get("center_key", [None, None, None])
                            out(
                                f"    hotspot(seed,i,j)=({ck[0]},{ck[1]},{ck[2]}) | pos={tuple(row.get('position_tag', [None,None]))} | "
                                f"contrast={float(row.get('contrast_score', float('nan'))):.4f} | hotspot_score={float(row.get('hotspot_score', float('nan'))):.4f} | "
                                f"dynamic_gain={float(row.get('dynamic_gain', float('nan'))):+.4f} | favorable={bool(row.get('favorable', False))}"
                            )
                            out(
                                f"      deltas q50: fid_uncond={float(row.get('delta_fid_uncond_q50', float('nan'))):+.4f} | "
                                f"logical_purity={float(row.get('delta_logical_purity_q50', float('nan'))):+.4f} | "
                                f"structure_drift={float(row.get('delta_structure_drift_q50', float('nan'))):+.4f} | "
                                f"subspace_ret={float(row.get('delta_subspace_retention_q50', float('nan'))):+.4f}"
                            )
                    hcls = osb.get("_hotspot_classes", {}) if isinstance(osb, dict) else {}
                    if isinstance(hcls, dict) and (hcls.get("by_pos") or hcls.get("by_pos_realcoarse")):
                        out("  Hotspot class summary (within-batch, coarse recurrence buckets):")
                        for label, rows in [("by_pos", hcls.get("by_pos", [])), ("by_pos_realcoarse", hcls.get("by_pos_realcoarse", []))]:
                            if not rows:
                                continue
                            out(f"    {label}:")
                            for row in rows[: min(4, len(rows))]:
                                out(
                                    f"      {row.get('class_key','?')} | count={int(row.get('count',0))} | favorable={int(row.get('favorable_count',0))}/{int(row.get('count',0))} | anti={int(row.get('anti_count',0))}/{int(row.get('count',0))} | "
                                    f"fav_rate={float(row.get('favorable_rate', float('nan'))):.2f} | stab={float(row.get('stability_score', float('nan'))):+.3f} | label={row.get('label','?')} | dyn_gain_med={float(row.get('dynamic_gain_med', float('nan'))):+.4f} | "
                                    f"logical_purity_med={float(row.get('logical_purity_med', float('nan'))):+.4f} | structure_drift_med={float(row.get('structure_drift_med', float('nan'))):+.4f}"
                                )
                    out("")

            ov_fine = jaccard(R.top_keys_qg, N.top_keys_qg)
            ov_coarse = jaccard(R.top_keys_qg_coarse, N.top_keys_qg_coarse)
            out(f"Batch {b+1}: Jaccard(Top-{args.topK}) REAL vs NULL: fine={ov_fine:.3f} | coarse={ov_coarse:.3f}")
            out("")

            sb = scoreboards[b]
            out("Baseline scoreboard (REAL - NULL):")
            out(f"  delta median entropy (bits): {sb['delta_median_entropy_bits']:+.3f}")
            out(f"  delta median leakage       : {sb['delta_median_leak']:+.3f}")
            out(f"  delta median dom_count     : {sb['delta_median_dom']:+.3f}")
            out(f"  entropy Cohen's d          : {sb['entropy_cohens_d']:+.3f}")
            out("")

        out("----------------------------------------------")
        out("=== Convergence diagnostics (REAL) ===")
        real_sets_fine = [set(r.top_keys_qg) for r in real]
        real_sets_coarse = [set(r.top_keys_qg_coarse) for r in real]
        for i in range(len(real_sets_fine)):
            for j in range(i + 1, len(real_sets_fine)):
                out(f"REAL overlap fine:   Jaccard(Top-{args.topK}) batch{i+1} vs batch{j+1} = {jaccard(real_sets_fine[i], real_sets_fine[j]):.3f}")
                out(f"REAL overlap coarse: Jaccard(Top-{args.topK}) batch{i+1} vs batch{j+1} = {jaccard(real_sets_coarse[i], real_sets_coarse[j]):.3f}")

        out("")
        global_hot = aggregate_hotspots_across_batches(open_system_summaries) if bool(args.open_system) else {"by_pos": [], "by_pos_realcoarse": []}
        if global_hot.get("by_pos") or global_hot.get("by_pos_realcoarse"):
            out("=== Hotspot recurrence across batches (v24.9.2) ===")
            for label, rows in [("by_pos", global_hot.get("by_pos", [])), ("by_pos_realcoarse", global_hot.get("by_pos_realcoarse", []))]:
                if not rows:
                    continue
                out(f"{label}:")
                for row in rows[: min(8, len(rows))]:
                    out(
                        f"  {row.get('class_key','?')} | count={int(row.get('count',0))} | batch_count={int(row.get('batch_count',0))} | batches={row.get('batches',[])} | "
                        f"favorable={int(row.get('favorable_count',0))}/{int(row.get('count',0))} | anti={int(row.get('anti_count',0))}/{int(row.get('count',0))} | fav_rate={float(row.get('favorable_rate', float('nan'))):.2f} | anti_rate={float(row.get('anti_rate', float('nan'))):.2f} | "
                        f"stab={float(row.get('stability_score', float('nan'))):+.3f} | label={row.get('label','?')} | dyn_gain_med={float(row.get('dynamic_gain_med', float('nan'))):+.4f} | logical_purity_med={float(row.get('logical_purity_med', float('nan'))):+.4f} | structure_drift_med={float(row.get('structure_drift_med', float('nan'))):+.4f}"
                    )
            panels = hotspot_stability_panels(global_hot, top_n=6)
            if panels.get("robust_favorable") or panels.get("robust_anti") or panels.get("mixed_recurrent"):
                out("=== Hotspot stability panels (v24.9.2) ===")
                for panel_name in ["robust_favorable", "robust_anti", "mixed_recurrent"]:
                    rows = panels.get(panel_name, [])
                    if not rows:
                        continue
                    out(f"{panel_name}:")
                    for row in rows:
                        out(
                            f"  {row.get('class_key','?')} | batch_count={int(row.get('batch_count',0))} | fav_rate={float(row.get('favorable_rate', float('nan'))):.2f} | anti_rate={float(row.get('anti_rate', float('nan'))):.2f} | "
                            f"stab={float(row.get('stability_score', float('nan'))):+.3f} | dyn_gain_med={float(row.get('dynamic_gain_med', float('nan'))):+.4f} | logical_purity_med={float(row.get('logical_purity_med', float('nan'))):+.4f} | structure_drift_med={float(row.get('structure_drift_med', float('nan'))):+.4f}"
                        )
            out("")
        out("=== Notes (scientific reading) ===")
        if args.anchor is not None:
            out("0) Matched-control protocol: compare the anchor run against each control set in a separate run")
            out("   using identical seeds and all other parameters unchanged. Each set has identical window count,")
            out("   width, and descendant-branch geometry; only the source-space location is shifted.")
        out("1) Compare REAL vs NULL primarily via deltas/effect sizes and batch stability, not raw entropy levels (d differs with n_qubits).")
        out("2) Use fine families for within-model discovery; use coarse families for cross-model interpretability.")
        out("3) If results at n=4 resemble n=3 (stable deltas + stable overlaps), that is strong qualitative evidence the effect is not a 3-qubit artifact.")
        out("")
        out("=== End of v25.3.1 anchor matched-controls ===")


if __name__ == "__main__":
    main()
