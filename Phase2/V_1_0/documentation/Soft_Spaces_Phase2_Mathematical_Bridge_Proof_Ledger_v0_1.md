# Soft Spaces Phase 2
## Mathematical bridge and proof ledger v0.1

**Date:** 7 September 2026  
**Revision:** 2 — exact ancilla-lift theorem and first stability bounds added  
**Status:** Rigorous partial results; not a proof for the original independent ensemble

## 1. Purpose

The purpose of this note is to formulate, without post-hoc adjustment, the
mathematical question suggested by the Phase 2 computations and by the
projection-operator formalism of Breuer and Petruccione.

The target is to determine whether the observed continuation of Soft-Space
hotspots can be proved under the dimensional rule

\[
k_{q+1}=2k_q+1,
\]

and, in particular, whether the *strength* or *prominence* of a hotspot is
preserved when the Hilbert-space dimension is increased.

This note maintains four separate statuses:

1. **PROVED EXACTLY**
2. **PROVED UNDER STATED ASSUMPTIONS**
3. **OBSERVED NUMERICALLY**
4. **UNRESOLVED OR FALSE IN GENERAL**

No numerical recurrence is promoted to a theorem unless all assumptions
required for the theorem are stated explicitly.

## 2. Three objects that must not be conflated

There are three mathematically different constructions in the current work.

### 2.1 Arithmetic continuation of the candidate label

Define

\[
r_q(k)=2k+1.
\]

This is an exact map between integer labels. Equivalently, on an abstract
label space one may define the isometry

\[
W_q^{\mathrm{lab}}|k\rangle=|2k+1\rangle.
\]

This proves only the index identity. It does not, by itself, imply any
relation between Hamiltonians, eigenspaces, perturbations, or hotspot scores.

### 2.2 The two branch lifts used by v25.31

In the actual 11Q-to-12Q program, a frozen source coordinate \(k\) is tested
through two adjacent eigenvalue-index pairs:

\[
(k,k+1),\qquad (k+2^{11},k+2^{11}+1).
\]

Thus the relevant index injections are

\[
j_{11,0}(m)=m,
\qquad
j_{11,1}(m)=m+2^{11}.
\]

For the two frozen sources this gives

\[
\begin{aligned}
k=511 &: (511,512),\ (2559,2560),\\
k=1535 &: (1535,1536),\ (3583,3584).
\end{aligned}
\]

These branch injections are not the same map as \(r_q(k)=2k+1\). In the
experimental protocol the arithmetic rule selects the next frozen source
label, after which the two branch pairs are tested in the enlarged spectrum.

### 2.3 A physical Hilbert-space embedding

A physical embedding is an isometry

\[
J_q:\mathcal H_q\longrightarrow\mathcal H_{q+1}
\]

such as tensoring with a fixed state of the added qubit. A physical embedding
must act on state vectors, not merely on their energy-order labels.

At present it has **not** been proved that either \(W_q^{\mathrm{lab}}\) or the
branch injections are physical intertwiners of the independently generated
Hamiltonians used in Phase 2.

## 3. Exact Hilbert-space definition of the v25.31 pair

Let a finite-dimensional Hamiltonian have the ordered spectral decomposition

\[
H=\sum_{n=0}^{d-1}\lambda_n |u_n\rangle\langle u_n|,
\qquad
\lambda_0\leq\lambda_1\leq\cdots\leq\lambda_{d-1}.
\]

For an adjacent pair beginning at index \(i\), define

\[
P_i=|u_i\rangle\langle u_i|+|u_{i+1}\rangle\langle u_{i+1}|,
\qquad
Q_i=I-P_i,
\]

and

\[
E_i=\frac{\lambda_i+\lambda_{i+1}}{2}.
\]

The program admits the pair only when

\[
|\lambda_{i+1}-\lambda_i|<\varepsilon_{\mathrm{neighbor}},
\]

with \(\varepsilon_{\mathrm{neighbor}}=0.05\) in v25.31.

For regularizer \(\eta>0\), define the real regularized scalar resolvent

\[
g_\eta(x)=\frac{x}{x^2+\eta^2}.
\]

The regularized second-order effective operator on the pair is

\[
\Sigma_i^{(\eta)}(H,V)
=
P_iVQ_i\,g_\eta(E_i-Q_iHQ_i)\,Q_iVP_i.
\]

In the ordered pair basis \((|u_i\rangle,|u_{i+1}\rangle)\), put

\[
w_n=
\begin{pmatrix}
\langle u_n|V|u_i\rangle &
\langle u_n|V|u_{i+1}\rangle
\end{pmatrix},
\qquad n\notin\{i,i+1\}.
\]

Then

\[
\Sigma_i^{(\eta)}
=
\sum_{n\notin\{i,i+1\}}
g_\eta(E_i-\lambda_n)\,w_n^\dagger w_n.
\]

This is exactly the 2-by-2 matrix assembled by the v25.31 code.

## 4. The v25.31 coherence ratio

The code normalizes the Frobenius norm of the signed sum by the sum of the
Frobenius norms of its individual terms. Define

\[
C_i(H,V)
=
\frac{
\left\|\displaystyle\sum_{n\notin\{i,i+1\}}
g_\eta(E_i-\lambda_n)w_n^\dagger w_n\right\|_F
}{
\displaystyle\sum_{n\notin\{i,i+1\}}
|g_\eta(E_i-\lambda_n)|\,\|w_n^\dagger w_n\|_F
},
\]

whenever the denominator is non-zero.

Since \(w_n^\dagger w_n\) is rank one,

\[
\|w_n^\dagger w_n\|_F=\|w_n\|_2^2.
\]

### Lemma 1 — range of the score

\[
0\leq C_i(H,V)\leq 1.
\]

**Proof.** Non-negativity follows from the norm. The upper bound follows
directly from the triangle inequality:

\[
\left\|\sum_n A_n\right\|_F
\leq\sum_n\|A_n\|_F,
\qquad
A_n=g_\eta(E_i-\lambda_n)w_n^\dagger w_n.
\]

**Status: PROVED EXACTLY.**

### Interpretation correction

The variable is called `cancel` in the program, but a *larger* value of \(C_i\)
means that the signed second-order contributions cancel less strongly and
retain a larger coherent resultant. A smaller value means stronger mutual
cancellation. Thus the frozen quantity

\[
\Delta C_i=C_i^{\mathrm{REAL}}-C_i^{\mathrm{NULL}}
\]

is positive when the REAL model retains a more coherent effective contribution
than its spectrum-matched NULL model. This semantic point should be corrected
in the manuscript even though it does not alter any computed value.

### Lemma 2 — invariance inside the two-dimensional pair

For any 2-by-2 unitary change of basis \(U\) inside \(P_i\),

\[
w_n\mapsto w_nU
\]

leaves both the numerator and denominator of \(C_i\) unchanged.

**Proof.** The numerator transforms by unitary conjugation,
\(\Sigma_i\mapsto U^\dagger\Sigma_iU\), which preserves the Frobenius norm.
Moreover \(\|w_nU\|_2=\|w_n\|_2\), so every denominator term is unchanged.

**Status: PROVED EXACTLY.**

This establishes that the v25.31 score is intrinsic to the selected
two-dimensional eigensubspace and does not depend on the chosen basis inside
that subspace.

## 5. Exact reconstruction of hotspot prominence

Let \(f\in\{Z,X\}\) label the dephasing and transverse perturbation families,
let \(r\) label the three perturbation repetitions, and let \(b\) label an
independent five-seed batch.

The code first takes the median of the two branch scores and forms

\[
\Delta C_{f,k,b,r}
=C_{f,k,b,r}^{\mathrm{REAL}}-C_{f,k,b,r}^{\mathrm{NULL}}.
\]

It then collapses the repetitions:

\[
D_{f,k,b}=\operatorname{median}_r\Delta C_{f,k,b,r}.
\]

The robust two-family score is

\[
R_{k,b}=\min\{D_{Z,k,b},D_{X,k,b}\}.
\]

For the frozen local control set \(N_q(k)\), defined by coordinate offsets
\(\pm10\) in steps of 2 with frozen targets excluded, the local background is

\[
B_{k,b}=\operatorname{median}_{c\in N_q(k)}R_{c,b}.
\]

The batch prominence is

\[
\Pi_{k,b}=R_{k,b}-B_{k,b}.
\]

Finally, with eight independent batches, the frozen decision is

\[
\widehat p_k
=\frac{1}{8}\sum_{b=1}^{8}\mathbf 1\{\Pi_{k,b}>0\},
\qquad
\widehat p_k\geq0.75.
\]

This is the exact mathematical object whose preservation must ultimately be
addressed. Preservation of \(P_i\), \(\Sigma_i\), or \(C_i\) alone is not
sufficient: the target, both perturbation families, the NULL comparison, and
the local-control background all enter \(\Pi_{k,b}\).

## 6. A first rigorous prominence-stability theorem

The minimum and the median are both 1-Lipschitz with respect to the sup norm.
This gives an immediate conditional theorem.

### Theorem 1 — deterministic preservation of positive prominence

Suppose a source target \(k\) and its lifted target \(k'\) have paired control
sets. For a fixed batch, assume

\[
|D'_{f,k'}-D_{f,k}|\leq\varepsilon_T
\]

for both perturbation families, and

\[
|D'_{f,c'}-D_{f,c}|\leq\varepsilon_C
\]

for both families and every paired control \(c\leftrightarrow c'\). Then

\[
|\Pi'_{k'}-\Pi_k|\leq\varepsilon_T+\varepsilon_C.
\]

Consequently,

\[
\Pi_k>\varepsilon_T+\varepsilon_C
\quad\Longrightarrow\quad
\Pi'_{k'}>0.
\]

If one common bound \(\varepsilon\) applies to both target and controls, the
sufficient margin is

\[
\Pi_k>2\varepsilon.
\]

**Proof.** The two-family minimum changes by at most \(\varepsilon_T\) at the
target. Each control minimum changes by at most \(\varepsilon_C\), and the
median of the control scores therefore changes by at most
\(\varepsilon_C\). Apply the triangle inequality to the difference between
target score and background.

**Status: PROVED UNDER STATED ASSUMPTIONS.**

### Corollary — preservation of the 0.75 gate

If the assumptions of Theorem 1 hold batchwise and at least six of eight source
batches satisfy

\[
\Pi_{k,b}>\varepsilon_{T,b}+\varepsilon_{C,b},
\]

then the lifted target necessarily satisfies the frozen empirical gate
\(\widehat p_{k'}\geq0.75\).

**Status: PROVED UNDER STATED ASSUMPTIONS.**

This theorem identifies exactly what an operator-level estimate must achieve:
it must control not only the target but also the local background.

## 7. From Hilbert-space projectors to superoperators

Given a Hilbert-space projector \(P\), define the Hilbert-Schmidt
superoperator

\[
\mathcal P(A)=PAP.
\]

Then

\[
\mathcal P^2=\mathcal P.
\]

With \(\mathcal Q=\mathcal I-\mathcal P\),

\[
\mathcal P+\mathcal Q=\mathcal I,
\qquad
\mathcal Q^2=\mathcal Q,
\qquad
\mathcal P\mathcal Q=\mathcal Q\mathcal P=0.
\]

**Status: PROVED EXACTLY.**

This reproduces the projector algebra on pages 432–433 of Breuer and
Petruccione. However, their standard projection

\[
\mathcal P_{\mathrm{BP}}(\rho)
=\operatorname{tr}_B(\rho)\otimes\rho_B
\]

is not the same map as \(A\mapsto PAP\). The former retains the reduced state
and resets the environment to a fixed reference state. The latter retains a
two-dimensional Hilbert-space block and may produce a subnormalised operator.

Therefore the two projector constructions are algebraically analogous but not
identical.

## 8. The exact Nakajima–Zwanzig bridge

For Liouvillian \(\mathcal L(t)\), Breuer and Petruccione obtain the exact
memory kernel

\[
\mathcal K(t,s)
=\alpha^2\mathcal P\mathcal L(t)\mathcal G(t,s)
\mathcal Q\mathcal L(s)\mathcal P,
\]

where \(\mathcal G(t,s)\) propagates the \(\mathcal Q\)-part. This has the
explicit structure

\[
\mathcal P\longrightarrow\mathcal Q\longrightarrow\mathcal P.
\]

For a time-independent generator, the Laplace transform of a propagator gives
a resolvent:

\[
\int_0^\infty e^{-zt}e^{t\mathcal Q\mathcal L\mathcal Q}\,dt
=(z-\mathcal Q\mathcal L\mathcal Q)^{-1},
\]

whenever the integral converges. Thus the Nakajima–Zwanzig memory kernel is the
time-domain relative of the Feshbach self-energy

\[
\Sigma_P(E)=PVQ(E-QHQ)^{-1}QVP.
\]

The relation is structural and exact after the appropriate transform, but one
object lives in operator/Liouville space and the other in Hilbert space.

## 9. Exact lifting on operator space

Let \(J:\mathcal H_q\to\mathcal H_{q+1}\) be an isometry and define

\[
\mathfrak J(A)=JAJ^\dagger.
\]

If

\[
P_{q+1}=JP_qJ^\dagger,
\]

then the induced block superoperators satisfy

\[
\mathcal P_{q+1}\mathfrak J
=\mathfrak J\mathcal P_q.
\]

**Proof.** Direct substitution gives

\[
P_{q+1}JAJ^\dagger P_{q+1}
=JP_qAP_qJ^\dagger.
\]

**Status: PROVED EXACTLY.**

### Theorem 2 — conditional preservation of the memory kernel

If, in addition, the Liouvillians and \(\mathcal Q\)-propagators intertwine,

\[
\mathcal L_{q+1}(t)\mathfrak J
=\mathfrak J\mathcal L_q(t),
\]

and

\[
\mathcal G_{q+1}(t,s)\mathfrak J
=\mathfrak J\mathcal G_q(t,s),
\]

then

\[
\mathcal K_{q+1}(t,s)\mathfrak J
=\mathfrak J\mathcal K_q(t,s).
\]

**Proof.** Insert the intertwining relations successively into every factor of
the exact Nakajima–Zwanzig kernel.

**Status: PROVED UNDER STATED ASSUMPTIONS.**

This theorem gives a mathematically clean bridge, but its assumptions have not
yet been established for the Phase 2 ensemble.

## 10. The present obstruction

The Phase 2 Hamiltonians at different qubit numbers are independently
generated random Pauli sums. Their eigenvectors are then ordered by energy,
and the tested pairs are selected by these energy-order indices.

Consequently, the arithmetic identity

\[
k\mapsto2k+1
\]

does not imply

\[
H_{q+1}J_q=J_qH_q,
\qquad
P_{q+1}=J_qP_qJ_q^\dagger,
\]

or the corresponding relations for \(V_q\), \(\Sigma_q\), or the local
controls.

Without additional assumptions, a universal preservation theorem is false in
general: after fixing \(H_q,V_q,P_q\), one may choose an unrelated
\(H_{q+1},V_{q+1}\) whose target score or local-control background is arbitrary.
No label isometry can prevent this.

**Status: UNRESOLVED FOR THE ACTUAL RANDOM ENSEMBLE; FALSE FOR ARBITRARY
UNRELATED OPERATORS.**

This does not invalidate the numerical recurrence. It determines its correct
logical level: the current evidence concerns a reproducible statistical
property of a structured random ensemble, not deterministic transport of one
fixed Hamiltonian into the next dimension.

## 11. Candidate theorem that remains scientifically meaningful

The strongest plausible theorem now has two layers.

### Layer A — deterministic conditional theorem

Prove quantitative bounds of the form

\[
\|\Sigma_{q+1,k'}-U\Sigma_{q,k}U^\dagger\|_F
\leq\delta_q
\]

for a specified coupling of the \(q\)- and \((q+1)\)-qubit models, and propagate
those bounds through \(C\), REAL–NULL subtraction, the family minimum, and the
local median by Theorem 1.

### Layer B — ensemble theorem

For the actual independent random-Pauli ensemble, seek a statement about the
distribution of prominence, for example

\[
\Pr\bigl(\Pi_{q+1,r_q(k)}>0\bigr)\geq p_0,
\]

with \(p_0\geq0.75\) only if this follows from the ensemble and not from the
observed sample. This requires a symmetry, concentration result, or exact
distributional relation that is not yet known.

## 12. Proof ledger after the first pass

| Statement | Status |
|---|---|
| \(k\mapsto2k+1\) is an exact label map | PROVED EXACTLY |
| v25.31 uses two branch injections distinct from \(2k+1\) | PROVED EXACTLY FROM CODE |
| \(P_i,Q_i\) form a Hilbert-space decomposition | PROVED EXACTLY |
| The v25.31 score satisfies \(0\leq C_i\leq1\) | PROVED EXACTLY |
| \(C_i\) is invariant under basis changes inside the pair | PROVED EXACTLY |
| v25.31 prominence is the target robust score minus local median background | PROVED EXACTLY FROM CODE |
| A uniform score bound preserves prominence above an explicit margin | PROVED UNDER STATED ASSUMPTIONS |
| \(A\mapsto PAP\) gives projector algebra in operator space | PROVED EXACTLY |
| Nakajima–Zwanzig and the Feshbach resolvent share a transformed \(P\to Q\to P\) structure | PROVED STRUCTURALLY |
| Exact kernel preservation follows from exact intertwining | PROVED UNDER STATED ASSUMPTIONS |
| The label map intertwines the actual independently generated Hamiltonians | NOT PROVED; FALSE IN GENERAL |
| Hotspot prominence is universally preserved by \(2k+1\) | UNRESOLVED; CANNOT FOLLOW FROM THE LABEL MAP ALONE |
| 12Q frozen predictions passed the empirical 0.75 gate | OBSERVED NUMERICALLY: 2/2 |

## 13. Next rigorous work

1. Reconstruct the coordinate convention for every validated step from 7Q to
   12Q and separate candidate recurrence from spectral branch lifting.
2. Define an explicit coupled \(q\)-to-\(q+1\) random-Pauli ensemble. Determine
   whether a natural added-qubit construction can preserve the relevant
   operator blocks without building the desired conclusion into the model.
3. Derive perturbation bounds for the normalized ratio \(C_i\), including the
   case where its denominator becomes small.
4. Extend Theorem 1 from deterministic batch values to a distributional
   statement with clearly separated mathematical probability and finite-sample
   evidence.
5. Compare the exact Hilbert-space Feshbach formula with the operator-space
   Nakajima–Zwanzig kernel under \(\mathcal L=-i[H,\cdot]\).
6. Search explicitly for counterexamples under the weakest proposed
   assumptions before attempting stronger proofs.

## 14. First conclusion

The mathematical bridge can be built, but the first rigorous step changes its
shape. The project already has an exact projector-level mechanism and an exact
conditional stability theorem. What is missing is not additional algebra
around \(2k+1\); it is a justified relation between the operators and control
landscapes at successive qubit numbers.

That missing relation is now the central proof obligation.

## 15. An exact non-circular q-to-(q+1) construction

There is a natural enlarged model in which the spectral branch lift is exact.
It is a baseline model, not yet the original Phase 2 random ensemble.

Let \(d=2^q\) and write the enlarged Hilbert space with the added qubit first:

\[
\mathcal H_{q+1}=\mathbb C^2\otimes\mathcal H_q.
\]

Let \(|0\rangle,|1\rangle\) be the ancilla basis and let

\[
J_b:\mathcal H_q\to\mathcal H_{q+1},
\qquad
J_b|\psi\rangle=|b\rangle\otimes|\psi\rangle,
\qquad b\in\{0,1\}.
\]

Choose \(\Delta>0\) and define

\[
H^{\uparrow}
=|0\rangle\langle0|\otimes(H-\Delta I)
+|1\rangle\langle1|\otimes(H+\Delta I),
\]

and, for every perturbation family,

\[
V_f^{\uparrow}=I_2\otimes V_f.
\]

If

\[
2\Delta>\lambda_{\max}(H)-\lambda_{\min}(H),
\]

the two spectral branches do not interleave. In energy order their indices are

\[
j_{q,0}(i)=i,
\qquad
j_{q,1}(i)=i+d,
\]

which is precisely the branch-offset form used in v25.31.

### Theorem 3 — exact preservation of the effective pair operator

For the adjacent pair projector \(P_i\), define

\[
P_{i,b}^{\uparrow}
=J_bP_iJ_b^\dagger
=|b\rangle\langle b|\otimes P_i.
\]

Then, for each branch \(b\),

\[
\Sigma_{i,b}^{(\eta),\uparrow}
=J_b\Sigma_i^{(\eta)}J_b^\dagger.
\]

Consequently,

\[
C_{i,b}(H^{\uparrow},V_f^{\uparrow})=C_i(H,V_f).
\]

**Proof.** The eigenvectors of \(H^{\uparrow}\) are
\(J_b|u_n\rangle\), with eigenvalues \(\lambda_n+(-1)^{b+1}\Delta\).
Within one branch the pair midpoint and every coupled intermediate energy are
shifted by the same constant. Therefore

\[
(E_i+\delta_b)-(\lambda_n+\delta_b)=E_i-\lambda_n.
\]

Moreover

\[
\langle b',u_n|V_f^{\uparrow}|b,u_m\rangle
=\delta_{bb'}\langle u_n|V_f|u_m\rangle.
\]

The opposite branch makes no contribution, while every term within branch
\(b\) is the isometric image of the corresponding source term. Both the
Frobenius norm of their sum and the sum of their individual Frobenius norms are
unchanged. Hence both \(\Sigma\) and \(C\) are preserved.

**Status: PROVED EXACTLY FOR THE DECOUPLED-ANCILLA LIFT.**

### Corollary — exact preservation of the full v25.31 prominence functional

Suppose the following are all lifted by the same construction:

- the REAL Hamiltonian,
- the spectrum-matched NULL Hamiltonian,
- both perturbation families and every repetition,
- the target pair and every local-control pair.

Then every branch score, branch median, REAL–NULL difference, repetition
median, two-family minimum, local-control median, and batch prominence is
unchanged. Thus

\[
\Pi_{k,b}^{\uparrow}=\Pi_{k,b}
\]

for every batch, and the empirical positive-rate gate is preserved exactly.

**Status: PROVED EXACTLY FOR A COUPLED NULL AND DECOUPLED-ANCILLA LIFT.**

### Why this construction is not circular

The lift does not insert the labels \(511\), \(1535\), or any other hotspot
into the Hamiltonian. It preserves every admissible adjacent pair and every
control by the same mechanism. Therefore hotspot status is not assumed in the
construction.

However, exact preservation occurs because the ancilla is decoupled and the
same lower-dimensional operators are copied into both branches. This is an
existence and baseline theorem. The original Phase 2 experiments instead draw
new random Pauli Hamiltonians and full Haar NULL bases at each dimension.

Theorem 3 therefore proves the branch-lift mechanism, not the observed
\(2k+1\) recurrence in the independent ensemble.

## 16. First rigorous bounds away from the exact lift

The exact lift provides a zeroth-order point around which coupled models can be
studied.

For a self-adjoint operator \(A\), define

\[
g_\eta(A)=A(A^2+\eta^2I)^{-1}.
\]

The identity

\[
g_\eta(A)
=\frac12\left[(A-i\eta I)^{-1}+(A+i\eta I)^{-1}\right]
\]

and the resolvent identity imply the following.

### Lemma 3 — regularized-resolvent stability

For self-adjoint \(A,A'\),

\[
\|g_\eta(A')-g_\eta(A)\|
\leq\frac{\|A'-A\|}{\eta^2}.
\]

Also,

\[
\|g_\eta(A)\|\leq\frac{1}{2\eta}.
\]

**Proof.** Apply

\[
(A'-zI)^{-1}-(A-zI)^{-1}
=(A'-zI)^{-1}(A-A')(A-zI)^{-1}
\]

at \(z=\pm i\eta\), and use that each resolvent has norm at most
\(1/\eta\). The second bound is the scalar maximum of
\(|x|/(x^2+\eta^2)\).

**Status: PROVED EXACTLY.**

### Lemma 4 — effective-operator stability for fixed aligned projectors

Let

\[
\Sigma=P V Qg_\eta(A)QVP,
\qquad
\Sigma'=P V'Qg_\eta(A')QV'P,
\]

with the same aligned \(P,Q\). Put

\[
M=\max\{\|V\|,\|V'\|\},
\qquad
\delta_V=\|V'-V\|,
\qquad
\delta_A=\|A'-A\|.
\]

Then

\[
\|\Sigma'-\Sigma\|
\leq
\frac{M\delta_V}{\eta}
+\frac{M^2\delta_A}{\eta^2}.
\]

**Proof.** Add and subtract the two mixed products, apply submultiplicativity,
and use Lemma 3. The two perturbation-factor differences contribute together
at most \(2M\delta_V/(2\eta)=M\delta_V/\eta\).

**Status: PROVED UNDER THE FIXED-PROJECTOR ASSUMPTION.**

Projector motion has deliberately not been hidden in this lemma. When the
selected two-dimensional eigensubspace changes, a separate cluster-gap bound
such as a Davis–Kahan estimate is required. The small internal pair gap is not
the relevant denominator; the necessary quantity is the separation of the
two-dimensional pair from the surrounding \(Q\)-spectrum.

### Lemma 5 — stability of the normalized v25.31 ratio

Write \(C=N/D\) and \(C'=N'/D'\), where the numerators are the Frobenius norms
of the total effective operators and the denominators are the sums of the
individual contribution norms. Suppose

\[
D\geq d_0>0,
\qquad
D'\geq d_0>0,
\]

and

\[
|N'-N|\leq\delta_N,
\qquad
|D'-D|\leq\delta_D.
\]

Since \(0\leq N/D\leq1\),

\[
|C'-C|
\leq\frac{\delta_N+\delta_D}{d_0}.
\]

**Proof.** Use

\[
\left|\frac{N'}{D'}-\frac ND\right|
\leq
\frac{|N'-N|}{D'}
+\frac ND\frac{|D'-D|}{D'}.
\]

**Status: PROVED UNDER THE NON-VANISHING-DENOMINATOR ASSUMPTION.**

Together, Lemmas 3–5 and Theorem 1 provide a complete logical route:

\[
\text{operator perturbation}
\Longrightarrow
\text{score bound}
\Longrightarrow
\text{prominence bound}
\Longrightarrow
\text{preserved 0.75 gate}.
\]

The remaining work is to obtain valid bounds for projector motion and for the
denominator \(D\) under a physically meaningful coupled random-Pauli lift.

## 17. Numerical verification and deliberate falsification

The companion program
`v25_33_exact_ancilla_lift_verification.py` performs two tests without fitting
any score or moving any coordinate:

1. It constructs the exact decoupled-ancilla lift and verifies, over fixed
   random seeds, targets, controls, REAL/NULL models, and two perturbation
   families, that branch scores and prominences agree to floating-point
   precision.
2. It replaces the copied upper-dimensional perturbations by unrelated random
   operators and reports the resulting non-zero discrepancies as a deliberate
   counterexample to preservation without operator coupling.

The first test checks the implementation of Theorem 3. The second checks that
the conclusion has not been made automatic merely by doubling the dimension.

The frozen five-seed verification produced

\[
\max |\text{exact-lift discrepancy}|
=1.273981\times10^{-14},
\]

and therefore passed the predeclared numerical tolerance
\(2.0\times10^{-10}\). This maximum includes REAL branch scores, NULL branch
scores, REAL–NULL differences, and full target-versus-control prominence.

For the deliberately unrelated upper-dimensional perturbations, the smallest
of the five seedwise maximum discrepancies was

\[
4.196371\times10^{-1}.
\]

Thus the same code path both verifies the exact theorem and demonstrates that
dimension doubling without the operator coupling does not preserve the score.

**Status: NUMERICALLY VERIFIED IMPLEMENTATION OF THEOREM 3; DELIBERATE
UNRELATED-OPERATOR CONTROL FAILS PRESERVATION.**

## 18. Updated conclusion

An exact q-to-(q+1) preservation theorem now exists for the decoupled-ancilla
branch lift, and rigorous perturbation inequalities provide a route away from
that ideal point. This is a genuine mathematical advance, but it does not yet
prove the empirical recurrence for independently generated random-Pauli
Hamiltonians.

The next irreducible question is whether the actual ensemble admits a natural
coupling whose projector, effective-operator, and denominator errors satisfy
the derived prominence margin. If it does not, the correct final result will
be an exact conditional theorem plus a separate statistical recurrence, not a
universal deterministic law.

## 19. No universal positive coupling radius without margins

Let \(\mathscr C\) be a class of admissible models. Suppose it contains models
with positive prominence arbitrarily close to zero and perturbation directions
that cross the boundary \(\Pi=0\). Then no number \(\epsilon_*>0\), uniform over
\(\mathscr C\), can guarantee preservation of positive prominence.

**Proof.** For any proposed \(\epsilon_*>0\), choose an admissible model close
enough to the boundary that a boundary-crossing perturbation of norm below
\(\epsilon_*\) reverses the sign. This contradicts uniform preservation.

**Status: PROVED EXACTLY UNDER THE STATED RICHNESS ASSUMPTION.**

The Phase 2 ensemble has no presently established positive lower bound on:

1. hotspot prominence,
2. separation of the selected two-dimensional cluster from the \(Q\)-spectrum,
3. the denominator of the normalized coherence ratio.

Therefore the only currently justified *universal* guaranteed radius for the
unrestricted ensemble is zero. A non-zero theorem must be conditional or
probabilistic.

## 20. Spectral identity condition under controlled coupling

Let \(g_*\) be the smallest distance between any frozen target/control
two-level cluster and the remainder of its spectrum. If

\[
\|\delta H\|<\frac{g_*}{2},
\]

Weyl's eigenvalue perturbation bound ensures that the cluster cannot close its
gap to the surrounding spectrum. With

\[
\rho=\frac{\|\delta H\|}{g_*},
\]

the sufficient structural condition is therefore

\[
\rho<\frac12.
\]

This condition preserves isolation of the pair. It does not by itself bound
the full nonlinear prominence functional; that additionally requires the
score and denominator estimates developed in Sections 6 and 16.

**Status: PROVED AS A SUFFICIENT PAIR-ISOLATION CONDITION.**

## 21. v25.34 controlled-coupling result

The frozen stress test used:

- target \(7\),
- controls \(1,3,5,9,11,13\),
- the same five seeds as v25.33,
- two fixed perturbation families,
- an off-diagonal ancilla coupling \(X_a\otimes G_s\),
- the predeclared grid
  \(\rho\in\{0,10^{-4},3\cdot10^{-4},\ldots,3,10\}\).

No target, control, score, seed, or coupling direction was changed after the
run.

Three of five source seeds had positive prominence. The other two remained in
the report and were not reclassified or discarded. For all three positive
source seeds:

1. both branches satisfied the proved target-plus-control prominence
   inequality at every evaluated point in the structurally safe grid
   \(\rho\leq0.3<0.5\);
2. the prominence sign was observed to remain positive through the largest
   tested value \(\rho=10\);
3. the common finite-grid score certificate extended through \(\rho=3\), but
   this latter statement does not guarantee physical pair identity because
   \(\rho\geq0.5\).

At \(\rho=10\), one positive seed no longer satisfied the sufficient
target-plus-control error inequality, although its directly computed
prominence remained positive. This demonstrates that the theorem bound is
conservative rather than equivalent to the observed sign.

**Status: FINITE-GRID NUMERICAL CERTIFICATION ALONG ONE FIXED COUPLING
DIRECTION; NOT A CONTINUUM OR ENSEMBLE-WIDE THEOREM.**

## 22. Revised boundary of what is now known

We now have three nested statements:

1. **Exact theorem:** zero-coupling ancilla lift preserves the full prominence
   functional.
2. **Structurally controlled regime:** \(\rho<1/2\) guarantees that each frozen
   pair remains isolated from its \(Q\)-spectrum.
3. **Frozen numerical path:** at the evaluated safe-grid points through
   \(\rho=0.3\), all positive source instances satisfy the rigorous
   prominence-error inequality in both branches.

What remains unproved is a uniform analytic prominence bound over every
coupling direction in the interval \(0\leq\rho<1/2\), and any transfer of that
bound to the independently redrawn random-Pauli ensemble used in 8Q–12Q.

## 23. Basis-independent denominator identity

For one frozen two-level cluster, define

\[
A=Q(E-H)Q,
\qquad
g_\eta(A)=A(A^2+\eta^2 I)^{-1},
\qquad
T=PVQ.
\]

The numerator of the v25.31 coherence ratio is

\[
N=\|Tg_\eta(A)T^\dagger\|_F.
\]

Its denominator, originally implemented as the sum of the Frobenius norms of
the individual spectral contributions, has the basis-independent expression

\[
D=\operatorname{Tr}\!\left(T\,|g_\eta(A)|\,T^\dagger\right).
\]

**Proof.** Diagonalize \(A\) on \(Q\). If \(z_j\) is the coupling row from the
\(j\)-th \(Q\)-eigenvector to the selected pair and
\(w_j=(E-\lambda_j)/((E-\lambda_j)^2+\eta^2)\), then the implemented
denominator is

\[
\sum_j\|w_j z_j^\dagger z_j\|_F
=\sum_j |w_j|\,\|z_j\|_2^2,
\]

which is exactly the stated trace. This also proves that the denominator does
not depend on the chosen eigenbasis inside degenerate \(Q\)-eigenspaces.

**Status: PROVED EXACTLY.**

## 24. Uniform all-directions perturbation theorem

Let a frozen rank-two cluster have width \(w\), separation \(g>0\) from the
rest of the spectrum, projector \(P\), midpoint \(E\), and centered spectral
radius \(R\). Let \(H'=H+\delta H\), keep \(V\) fixed, and put
\(\epsilon=\|\delta H\|_2<g/2\). A stadium contour at distance \(g/2\) from
the cluster has length \(L=2w+\pi g\). The resolvent identity gives

\[
p:=\|P'-P\|_2
\leq
\frac{L\epsilon}{\pi g(g/2-\epsilon)}.
\]

For several frozen target/control clusters, the maximum of these bounds is
used. With \(M=\|V\|_2\), one obtains the following sufficient bounds:

\[
|E'-E|\leq \epsilon+2pR,
\]

\[
a:=\|A'-A\|_2
\leq p(4R+|E'-E|+\epsilon)+|E'-E|+\epsilon,
\]

and

\[
t:=\|P'VQ'-PVQ\|_2\leq2Mp.
\]

The rational resolvent representation of \(g_\eta\) yields

\[
\|g_\eta(A')-g_\eta(A)\|_2\leq\frac{a}{\eta^2},
\qquad
\|g_\eta(A)\|_2\leq\frac1{2\eta}.
\]

Consequently, for rank \(r=2\), a sufficient numerator error is

\[
\delta_N
\leq
\sqrt{2r}\left(\frac{Mt}{\eta}+\frac{M^2a}{\eta^2}\right).
\]

Using
\(\||g_\eta(A')|-|g_\eta(A)|\|_2\leq\sqrt{a/\eta^3}\), a deliberately
conservative denominator error is

\[
\delta_D
\leq
r\sqrt2\frac{Mt}{\eta}
+rM^2\sqrt{\frac{a}{\eta^3}}.
\]

If the unperturbed denominator satisfies \(D\geq D_0>\delta_D\), then

\[
|C'-C|
\leq
\frac{\delta_N+\delta_D}{D_0-\delta_D}.
\]

Applying this separately to REAL and NULL and then to the target and control
median shows that the full prominence changes by at most four times this
single-model score bound. Therefore every Hermitian perturbation direction
with a bound smaller than the original positive prominence preserves its
sign.

**Status: PROVED UNDER THE STATED GAP, FIXED-\(V\), AND NON-VANISHING-
DENOMINATOR ASSUMPTIONS.**

The theorem is uniform over perturbation directions; it is not a sampled-path
claim. Its constants are intentionally coarse, especially the square-root
bound for the absolute-value operator.

## 25. v25.35 numerical evaluation of the analytic theorem

The companion program
`v25_35_uniform_analytic_prominence_certificate.py` evaluates the theorem for
the same frozen target, controls, two operator families, and five seeds as the
preceding tests. Negative source-prominence seeds remain in the output and are
not eligible for a positive-sign preservation certificate.

The three positive source instances gave the following conservative candidate
radii, where \(\rho=\|\delta H\|_2/g_*\):

| Seed | Source prominence | Candidate \(\rho\) | Candidate \(\|\delta H\|_2\) |
|---:|---:|---:|---:|
| 2533001 | \(+1.410\times10^{-1}\) | \(1.09239\times10^{-15}\) | \(6.64582\times10^{-17}\) |
| 2533003 | \(+7.364\times10^{-2}\) | \(4.24309\times10^{-16}\) | \(3.93926\times10^{-17}\) |
| 2533004 | \(+2.353\times10^{-1}\) | \(1.32046\times10^{-15}\) | \(4.67094\times10^{-17}\) |

The common candidate radius is therefore

\[
\rho\leq4.24309\times10^{-16}.
\]

This tiny value is not an observed breakdown scale. It results mainly from
the worst-case \(\eta^{-2}\) and \(\eta^{-3/2}\) factors at
\(\eta=10^{-3}\). The v25.34 fixed coupling path remained positive many
orders of magnitude beyond it, showing how conservative the all-directions
bound is.

The displayed constants were computed in ordinary double precision. Since
the resulting radii are at or below the scale where floating-point input and
eigendecomposition errors matter, these numbers are **candidate evaluations
of a rigorous analytic theorem, not yet validated numerical certificates**.
Interval arithmetic or a high-precision calculation with explicit outward
error bounds is required before they may be presented as computer-assisted
proof for the frozen instances.

**Status: ANALYTIC THEOREM PROVED; DOUBLE-PRECISION NUMERICAL INSTANTIATION IS
DIAGNOSTIC AND AWAITS VALIDATED NUMERICS.**

## 26. Current mathematical boundary

The exact zero-coupling theorem has now been extended by an explicit
all-directions sufficient condition. Thus continuity is no longer merely
asserted: a complete symbolic chain from \(\|\delta H\|_2\) to projector
motion, score error, and prominence error has been written down.

Two gaps remain and must not be conflated:

1. The frozen-instance numerical constants require validated high-precision
   or interval evaluation, preferably after sharpening the denominator bound.
2. No coupling theorem has yet linked this ancilla neighbourhood to the
   independently redrawn random-Pauli 8Q–12Q ensemble.

The present result is therefore a rigorous conditional theorem about the
coupled baseline, not a proof of universal hotspot recurrence across the full
Phase 2 ensemble.

## 27. Ancilla-parity theorem: exact removal of the linear response

The controlled path used in v25.34 has

\[
H(\epsilon)=H^{\uparrow}+\epsilon(X_a\otimes G),
\qquad
V^{\uparrow}=I_a\otimes V,
\]

where \(H^{\uparrow}\) is block diagonal in the ancilla basis. Define the
ancilla parity operator

\[
\Gamma=Z_a\otimes I.
\]

Then

\[
\Gamma H(\epsilon)\Gamma=H(-\epsilon),
\qquad
\Gamma V^{\uparrow}\Gamma=V^{\uparrow}.
\]

Because the v25.31 ratio is invariant under simultaneous unitary conjugation
of the Hamiltonian, perturbation operator, and selected spectral projector,
every consistently tracked pair score obeys

\[
C(\epsilon)=C(-\epsilon).
\]

The same is true for REAL–NULL differences, the minimum over the two frozen
families, the control median, and the final prominence:

\[
\Pi(\epsilon)=\Pi(-\epsilon).
\]

Thus the linear perturbation coefficient vanishes wherever the local spectral
branches remain isolated. In particular,

\[
\Pi(\epsilon)=\Pi(0)+O(\epsilon^2),
\]

not merely \(\Pi(0)+O(|\epsilon|)\). This exact selection rule is absent from
the unrestricted all-directions estimate in v25.35 and explains why that
estimate is drastically more pessimistic than the structured numerical path.

**Status: PROVED EXACTLY FOR THE CONTROLLED ANCILLA PATH WITH BLOCK-DIAGONAL
\(H^{\uparrow}\) AND \(V^{\uparrow}\).**

## 28. v25.36 decomposition of the observed stability

The companion program
`v25_36_stability_mechanism_decomposition.py` retained the complete frozen
v25.34 design:

- target \(7\), controls \(1,3,5,9,11,13\);
- seeds 2533001–2533005, including the two negative source instances;
- both fixed perturbation families;
- the same deterministic \(X_a\otimes G_s\) direction;
- the same predeclared twelve-point \(\rho\)-grid.

Five overlapping mechanisms were measured separately. They are diagnostics,
not additive causal percentages.

### 28.1 Direct parity and response-order check

For every seed and branch, a log–log fit of
\(|\Pi(\rho)-\Pi(0)|\) over the frozen low-coupling grid
\(10^{-4}\leq\rho\leq10^{-1}\) gave exponents between

\[
1.999691\quad\text{and}\quad2.002899,
\]

with median

\[
2.000099.
\]

The maximum directly computed discrepancy between \(+\rho\) and \(-\rho\)
prominence was zero at the printed numerical precision. This is the numerical
signature predicted by the exact ancilla-parity theorem.

### 28.2 Normalization

The median cancellation between logarithmic numerator and denominator motion
was

\[
40.8\%.
\]

Normalization therefore supplies a substantial secondary damping mechanism
along this path, although it is not by itself the primary explanation.

### 28.3 REAL–NULL, target–control, and branch common motion

Across all non-zero grid points, the medians of the diagnostic cancellation
fractions were:

\[
\begin{array}{lr}
\text{REAL--NULL common motion} & 0.0\%,\\
\text{target--control common motion} & 0.0\%,\\
\text{same-direction branch motion} & 0.0\%.
\end{array}
\]

These mechanisms occur in individual cases—for example seed 2533003 has
strong target–control cancellation over part of the path—but they are not the
general explanation across the frozen five-seed set. The earlier suggestion
that REAL and NULL might usually move together is therefore not supported by
this decomposition.

### 28.4 Sorted indices versus physical subspace tracking

At every one of the 110 non-zero branch-points, the two eigenvectors selected
by maximum overlap with the initial rank-two projector were the same pair as
the sorted-index rule. Consequently:

\[
\max|\Pi_{\mathrm{sorted}}-\Pi_{\mathrm{tracked}}|=0
\]

at printed precision, with zero sign disagreements. The minimum projector
overlap fidelity over the complete path, including \(\rho=10\), was

\[
\frac12\operatorname{Tr}(P_0P_{\mathrm{tracked}})=0.944400.
\]

Thus the observed sign stability through \(\rho=10\) is not explained by the
algorithm silently switching to a different sorted spectral pair in these
frozen runs. This does not replace the analytic pair-isolation theorem, but it
removes the proposed index-tracking artefact as an explanation of this data.

**Status: NUMERICALLY VERIFIED DIAGNOSTIC DECOMPOSITION ON THE FROZEN PATH;
NOT AN ENSEMBLE-WIDE CAUSAL THEOREM.**

## 29. Revised explanation of the large numerical stability

The evidence now supports the following ordered explanation:

1. **Primary:** exact ancilla parity forces the first-order response to vanish,
   leaving quadratic small-coupling motion.
2. **Secondary:** numerator/denominator normalization cancels a median 40.8%
   of their logarithmic motion.
3. **Protection by margin:** the three positive source prominences begin a
   finite distance above zero.
4. **Instance-specific only:** REAL–NULL and target–control common motion can
   help particular seeds but are not general across all five.
5. **Rejected for these runs:** sorted-index replacement is not responsible
   for the persistence through \(\rho=10\).

This conclusion is sharper than the earlier qualitative explanation. It also
identifies the next mathematical improvement: replace the linear worst-case
v25.35 estimate by a parity-aware second-order bound for this structured
coupling. Such a bound may be many orders of magnitude less conservative, but
it will still apply only to the controlled ancilla model until a justified
coupling to the independent 8Q–12Q ensemble is established.

## 30. Exact Schur-complement reduction of the ancilla coupling

Write the controlled Hamiltonian as

\[
H(\epsilon)=
\begin{pmatrix}
A & \epsilon G\\
\epsilon G & D
\end{pmatrix},
\qquad \|G\|_2=1,
\]

where \(A=H-sI\) and \(D=H+sI\) are the two uncoupled ancilla branches. For an
eigenpair \(H(\epsilon)(\psi_A,\psi_D)^T=E(\psi_A,\psi_D)^T\), whenever
\(D-E\) is invertible,

\[
\psi_D=-\epsilon(D-E)^{-1}G\psi_A.
\]

Substitution into the first block gives the exact reduced equation

\[
\left[A-E-\epsilon^2G(D-E)^{-1}G\right]\psi_A=0.
\]

Thus the opposite branch enters through the self-energy

\[
\Sigma_A(E,\epsilon)=\epsilon^2G(D-E)^{-1}G.
\]

If \(\Gamma\) is the uncoupled distance between the two spectral branch
intervals, Weyl's inequality gives

\[
\operatorname{dist}(E,\sigma(D))\geq\Gamma-|\epsilon|.
\]

Therefore, for \(|\epsilon|<\Gamma\),

\[
\|\Sigma_A(E,\epsilon)\|_2
\leq
\frac{\epsilon^2\|G\|_2^2}{\Gamma-|\epsilon|}.
\]

The upper branch has the analogous formula. This is an explicit continuum
second-order bound, not a fitted power law. It explains structurally why the
branch dynamics are much less sensitive than the unrestricted linear
perturbation estimate in v25.35.

**Status: PROVED EXACTLY FOR THE TWO-BLOCK CONTROLLED ANCILLA HAMILTONIAN.**

This theorem bounds the effective branch Hamiltonian. It does not alone bound
the complete normalized coherence ratio, because that ratio also contains
moving spectral projectors, opposite-branch leakage, a Frobenius norm, and its
normalizing denominator.

## 31. Conditional parity-aware prominence theorem

For each frozen perturbation family \(f\), suppose the REAL–NULL difference at
the target satisfies

\[
|\Delta_{T,f}(\epsilon)-\Delta_{T,f}(0)|
\leq K_T\epsilon^2,
\]

and every frozen control satisfies

\[
|\Delta_{i,f}(\epsilon)-\Delta_{i,f}(0)|
\leq K_C\epsilon^2.
\]

The minimum over the two families is 1-Lipschitz in the sup norm, and the
median is also 1-Lipschitz in the sup norm. Hence the full prominence obeys

\[
|\Pi(\epsilon)-\Pi(0)|
\leq(K_T+K_C)\epsilon^2.
\]

If \(m=\Pi(0)>0\), then

\[
|\epsilon|<\sqrt{\frac{m}{K_T+K_C}}
\]

is sufficient to preserve positive prominence.

**Status: PROVED EXACTLY, CONDITIONAL ON UNIFORM SECOND-ORDER FAMILY-SCORE
BOUNDS.**

Unlike a Taylor argument, this statement does not require the identity of the
minimum family or median control to remain fixed. The remaining task is to
derive or validate suitable \(K_T\) and \(K_C\) for the complete v25.31 score.

## 32. v25.37 parity-aware curvature audit

The companion program `v25_37_parity_aware_second_order_bound.py` retained the
same target, controls, five seeds, two families, coupling direction, and
predeclared grid. No negative seed was removed.

For the exact branch self-energy, all five instances remained inside
\(|\epsilon|<\Gamma\), even at \(\rho=10\). The branch separations ranged from
\(3.790353\) to \(3.977583\), whereas \(|\epsilon|\) at \(\rho=10\) ranged
from \(0.353737\) to \(0.944067\). The corresponding rigorous Schur bounds at
\(\rho=10\) ranged from \(0.036411\) to \(0.306405\).

On the frozen structurally safe grid \(10^{-4}\leq\rho\leq0.3\), define

\[
K_{\Pi,\mathrm{grid}}
=\max\frac{|\Pi(\rho)-\Pi(0)|}{\rho^2}
\]

and

\[
K_{\mathrm{err,grid}}
=\max\frac{e_T(\rho)+e_C(\rho)}{\rho^2},
\]

where \(e_T\) and \(e_C\) are the already used target and maximum-control
REAL–NULL errors. The three positive source seeds gave data-derived margin
scales

\[
15.78717,\qquad5.712469,\qquad18.36531,
\]

so the common candidate scale is

\[
\rho_{\mathrm{candidate}}=5.712469.
\]

This is more than \(10^{16}\) times the common v25.35 linear worst-case
candidate \(4.24309\times10^{-16}\). The comparison quantifies how much
structure the unrestricted bound discarded.

However, \(K_{\mathrm{err,grid}}\) is the maximum of evaluated points, not a
validated supremum on a continuum interval. Consequently
\(\rho_{\mathrm{candidate}}\) must not be reported as a rigorous preservation
radius. The statements that are presently rigorous are:

1. exact evenness under ancilla parity;
2. exact quadratic Schur self-energy;
3. the conditional quadratic prominence theorem;
4. the already certified individual grid points through \(\rho=0.3\).

The measured exponents remained between 1.999691 and 2.002899, their median
was 2.000099, and the maximum \(+\rho/-\rho\) prominence residual was zero at
printed precision.

**Status: EXACT SECOND-ORDER HAMILTONIAN THEOREM PLUS FINITE-GRID FULL-SCORE
CURVATURE AUDIT; FULL CONTINUUM PROMINENCE RADIUS REMAINS UNPROVED.**

## 33. New irreducible boundary

The earlier linear all-directions analysis is no longer the appropriate model
for the frozen \(X_a\otimes G_s\) path. Its symmetry and block structure
provably suppress the first order. The next irreducible mathematical problem
is narrower:

> Establish a valid continuum upper bound on the second-order variation of
> every frozen REAL and NULL coherence ratio, including projector motion,
> inter-branch leakage, numerator norm, and denominator normalization.

This may be approached through explicit second-resolvent derivatives or
validated interval arithmetic. Until that bound is obtained, the value
\(5.712469\) is a scientifically useful scale estimate but not a theorem.
None of these results yet transfers automatically to the independently drawn
8Q–12Q random-Pauli ensemble.

## 34. Continuum curvature chain for the complete v25.31 score

Let \(H(x)=H_0+xK\), \(\|K\|_2=1\), and restrict \(|x|\leq r\), where every
selected rank-two cluster and every individual \(Q\)-eigenvalue remains
isolated. A fixed Riesz contour of length \(L\), whose minimum distance from
the perturbed spectrum is \(d>0\), gives

\[
\|P'(x)\|_2\leq\frac{L}{2\pi d^2},
\qquad
\|P''(x)\|_2\leq\frac{L}{\pi d^3}.
\]

These follow by differentiating the resolvent under the contour integral.
Writing

\[
A(x)=Q(x)(E(x)-H(x))Q(x),
\quad
T(x)=P(x)VQ(x),
\quad
G(x)=g_\eta(A(x)),
\]

the product rule supplies explicit uniform bounds \(a_1,a_2,t_1,t_2\) for
their first and second derivatives. The resolvent representation

\[
g_\eta(A)=\frac12\left[(A-i\eta I)^{-1}+(A+i\eta I)^{-1}\right]
\]

then yields

\[
\|G'\|_2\leq\frac{a_1}{\eta^2},
\qquad
\|G''\|_2
\leq\frac{2a_1^2}{\eta^3}+\frac{a_2}{\eta^2}.
\]

For the effective numerator matrix \(F=TGT^\dagger\), the product rule gives
uniform \(F_1,F_2\). Provided its Frobenius norm has a positive interval floor
\(N_*\),

\[
N(x)=\|F(x)\|_F
\]

satisfies

\[
|N'|\leq\|F'\|_F,
\qquad
|N''|\leq\|F''\|_F+\frac{\|F'\|_F^2}{N_*}.
\]

For the denominator, use the exact spectral identity

\[
D(x)=\sum_{j\in Q}
h_\eta(E(x)-\lambda_j(x))
\|R_j(x)VP(x)\|_F^2,
\]

where

\[
h_\eta(y)=\frac{|y|}{y^2+\eta^2}.
\]

Individual Riesz bounds for \(R_j',R_j''\), the standard isolated-eigenvalue
bound

\[
|\lambda_j''(x)|
\leq\frac{2}{g_j-2r},
\]

and, on an interval with \(|y|\geq y_*>0\),

\[
|h_\eta|\leq(y_*^2+\eta^2)^{-1/2},
\quad
|h_\eta'|\leq(y_*^2+\eta^2)^{-1},
\quad
|h_\eta''|\leq6(y_*^2+\eta^2)^{-3/2}
\]

produce explicit uniform bounds \(D_1,D_2\). If
\(D(x)\geq D_*>0\), the quotient \(C=N/D\), together with \(0\leq C\leq1\),
obeys

\[
|C''(x)|
\leq
\frac{N_2+D_2}{D_*}
+\frac{2(N_1D_1+D_1^2)}{D_*^2}.
\]

This completes the missing symbolic route from spectral gaps and denominator
floors to a continuum second-derivative bound for every complete normalized
v25.31 score. Ancilla parity gives \(C'(0)=0\), and Taylor's theorem therefore
gives

\[
|C(x)-C(0)|\leq\frac12K_Cx^2.
\]

Combining REAL and NULL, the family minimum, and the control median through
their 1-Lipschitz properties gives the continuum prominence bound of Section
31 with fully explicit constants.

**Status: PROVED SYMBOLICALLY UNDER EXPLICIT ISOLATION, POSITIVE-NUMERATOR,
AND POSITIVE-DENOMINATOR FLOOR CONDITIONS.**

## 35. v25.38 frozen-instance evaluation

The companion program
`v25_38_continuum_score_curvature_bound.py` evaluates every constant for the
same frozen target, controls, families, seeds, and \(X_a\otimes G_s\) path.
The two negative source-prominence seeds remain reported and are not eligible
for a positive-sign certificate.

The first coarse interval set, beginning at \(\rho=10^{-8}\), failed to prove
a positive numerator floor for any positive seed. This failed attempt is
retained as `v25_38_initial_coarse_interval_failure_output.txt`; it is not
silently discarded.

A declared multiscale resolution audit from \(10^{-16}\) through \(10^{-2}\)
then found valid intervals for all three positive source seeds:

| Seed | Chosen interval \(\rho\) | Prominence curvature bound | Candidate \(\rho\) |
|---:|---:|---:|---:|
| 2533001 | \(1.0\times10^{-10}\) | \(2.199729\times10^{21}\) | \(9.0\times10^{-11}\) |
| 2533003 | \(3.0\times10^{-10}\) | \(5.811156\times10^{20}\) | \(1.091269\times10^{-10}\) |
| 2533004 | \(1.0\times10^{-10}\) | \(6.644358\times10^{21}\) | \(9.0\times10^{-11}\) |

The common continuum candidate is therefore

\[
\rho\leq9.0\times10^{-11}.
\]

This is approximately \(2.1\times10^5\) times larger than the v25.35 common
linear candidate \(4.24309\times10^{-16}\), but vastly smaller than the
finite-grid second-order scale estimate \(5.712469\). The difference is the
price of bounding every derivative term independently in the worst permitted
direction.

The selected interval floors remained positive. Across the three positive
seeds, the reported denominator floors were between \(0.578406\) and
\(0.728454\), and the numerator floors between \(0.021392\) and \(0.120201\).

## 36. Exact status of the v25.38 number

The analytic inequalities are continuum statements, not grid interpolation.
If their input gaps, norms, numerator floors, and denominator floors are exact,
the resulting radius is a rigorous preservation certificate throughout the
stated interval.

The current constants were evaluated with ordinary double-precision
eigendecompositions. They have not been enclosed using outward-rounded
interval arithmetic. Therefore

\[
9.0\times10^{-11}
\]

is a **double-precision evaluation of a rigorous continuum formula**, not yet
a formally validated computer-assisted numerical theorem for the frozen
matrices. The factor 0.9 keeps the reported point inside the calculated
boundary, but it does not replace a complete rounding-error proof.

**Status: COMPLETE ANALYTIC CONTINUUM BOUND; NUMERICAL INSTANTIATION AWAITS
INTERVAL VALIDATION.**

## 37. Revised irreducible boundary

For the controlled ancilla model, the conceptual analytic chain is now
complete:

\[
\text{block coupling}
\Longrightarrow
\text{projector/resolvent curvature}
\Longrightarrow
\text{normalized-score curvature}
\Longrightarrow
\text{prominence curvature}
\Longrightarrow
\text{positive-radius condition}.
\]

What remains is primarily validation rather than a missing symbolic link:

1. enclose all frozen numerical constants with directed rounding or interval
   arithmetic;
2. preferably sharpen the very large independent worst-case constants;
3. separately establish—or refute—a physical intertwiner connecting this
   controlled model to the independently generated 8Q–12Q ensemble.

The first item can upgrade the controlled-model candidate into a
computer-assisted proof. It cannot solve the third item, which remains the
central boundary for the original Phase 2 recurrence claim.

## 38. v25.39 outward-rounded Arb validation

The companion program `v25_39_arb_interval_validation.py` re-evaluates the
complete v25.38 constant chain using python-flint 0.9.0 and Arb/Acb ball
arithmetic at 256-bit working precision. Every ball operation is outward
rounded.

The validated mathematical objects are the exact binary64 matrices produced
by the frozen v25.33 construction. This qualification is important: the
computer-assisted theorem concerns those stored finite matrices, not an
unspecified ideal random-number distribution before rounding.

The validation includes:

1. isolated Acb eigenvalue and eigenvector enclosures;
2. lower enclosures for every rank-two and rank-one spectral gap;
3. upper enclosures for the coupling and perturbation operator norms;
4. lower enclosures for every base numerator and denominator;
5. upper enclosures for every first- and second-derivative constant;
6. lower and upper enclosures for all five source prominences;
7. directed conversion from the certified \(\epsilon\)-radius to \(\rho\),
   using an upper enclosure of \(g_*\) in the denominator.

NumPy's `eigh` convention uses the lower triangle as the authoritative
Hermitian matrix. The NULL matrix contains roundoff-level asymmetry from its
matrix products. The Arb validation therefore reconstructs exactly the
Hermitian matrix represented by NumPy's lower-triangle convention before
diagonalization. This prevents the validation from silently changing the
model.

The exact lifted norm identities

\[
\|\operatorname{diag}(V,V)\|_2=\|V\|_2,
\qquad
\left\|\begin{pmatrix}0&G\\G&0\end{pmatrix}\right\|_2=\|G\|_2
\]

were used to avoid artificial multiplicities in the ball eigenvalue problems.

## 39. Validated result

All three positive source instances passed every isolation, numerator-floor,
denominator-floor, curvature, and margin condition:

| Seed | Validated interval \(\rho\) | Validated prominence lower bound | Curvature upper bound | Certified \(\rho\) lower bound |
|---:|---:|---:|---:|---:|
| 2533001 | \(1.0\times10^{-10}\) | \(0.140983066\) | \(2.19972891\times10^{21}\) | \(8.9\times10^{-11}\) |
| 2533003 | \(3.0\times10^{-10}\) | \(0.0736388287\) | \(5.81115634\times10^{20}\) | \(1.07914365\times10^{-10}\) |
| 2533004 | \(1.0\times10^{-10}\) | \(0.235286838\) | \(6.64435839\times10^{21}\) | \(8.9\times10^{-11}\) |

Thus the common outward-rounded lower bound is

\[
\boxed{\rho_{\mathrm{cert}}=8.9\times10^{-11}}.
\]

Every relevant numerator and denominator floor was strictly positive. The
smallest reported numerator floor was \(0.0213918989\), the smallest
denominator floor was \(0.578405821\), and the smallest spectral gap lower
bound entering the detailed derivative calculation was \(0.0289256360\).

The two negative instances were retained and independently interval-validated
as strictly negative. Their prominence upper bounds were

\[
-0.1563239983671948
\quad\text{and}\quad
-0.1536730119167410.
\]

They were not reclassified or used in the positive-radius calculation.

Repeating the complete interval calculation at 192, 256, and 384 bits produced
the same common certified radius and the same displayed negative upper bounds.
This precision audit is stored in
`v25_39_precision_stability_audit_output.txt`.

**Status: COMPUTER-ASSISTED INTERVAL PROOF FOR THE THREE FROZEN POSITIVE
BINARY64 CONTROLLED-ANCILLA INSTANCES; BOTH NEGATIVE INSTANCES ALSO
INTERVAL-VALIDATED AND RETAINED.**

## 40. What has and has not now been proved

For the frozen controlled ancilla path, the chain is complete both
symbolically and numerically:

\[
|\rho|\leq8.9\times10^{-11}
\quad\Longrightarrow\quad
\Pi(\rho)>0
\]

for each of the three initially positive instances, with all constants
enclosed by outward-rounded ball arithmetic.

As with any computer-assisted proof, this conclusion assumes the correctness
of the stated analytic inequalities, the python-flint/Arb implementation, and
the supplied verification program. The program and complete output are
retained for independent reproduction.

This result does **not** establish:

1. that the two negative seeds become positive;
2. that prominence is preserved for arbitrary coupling directions;
3. that \(\rho=5.712469\) is a rigorous continuum radius;
4. that the controlled ancilla Hamiltonian is a physical intertwiner for the
   independently generated 8Q–12Q random-Pauli ensemble;
5. a universal law for all Soft Spaces Phase 2 systems.

The numerical-validation gap identified in Section 37 is now closed. The
remaining central mathematical question is no longer rounding error or local
continuity; it is whether a justified structural map connects the proven
controlled model to the original independent qubit-by-qubit construction.

## 41. v25.40 audit of the independent random-Pauli construction

The companion program
`v25_40_independent_ensemble_coupling_audit.py` examines the most favourable
elementary coupling available without changing the frozen v25.31 generator.
For each of the 40 frozen 12Q seeds it generates both the 11Q and 12Q
five-term Pauli sums with the same seed and asks whether

\[
H_{12}=I_a\otimes H_{11}.
\]

The same-seed construction preserves the sequence of five continuous
coefficients, but the dimension-dependent integer-to-Pauli map resamples the
Pauli labels. The audit found:

| Condition | Result |
|---|---:|
| Exact (I_a\otimes H_{11}) copies | 0/40 |
| Seeds with at least one exactly copied Pauli term | 0/40 |
| Exactly copied Pauli terms | 0/200 |
| 12Q Hamiltonians preserving both leading-ancilla computational sectors | 0/40 |

Every tested 12Q Hamiltonian contained at least one term with (X) or (Y)
on the leading qubit. Hence the elementary computational ancilla sectors leak
in every audited instance.

There is a second, logically prior distinction. The coordinates used by
v25.30 and v25.31 are ordinal indices of eigenvalue/eigenvector pairs after
`numpy.linalg.eigh` sorting. They are not computational-basis labels. Thus the
arithmetic identity (k\mapsto2k+1) does not by itself define a physical
Hilbert-space embedding of the scored eigenvectors.

**Status: EXACT CODE-SEMANTICS RESULT AND EXACT FINITE AUDIT. THE DIRECT
BLOCK-ANCILLA INTERTWINER IS REFUTED FOR THE FROZEN FAVOURABLE SAME-SEED
COUPLING. THIS DOES NOT REFUTE THE EMPIRICAL RECURRENCE AND DOES NOT EXCLUDE
EVERY POSSIBLE NONLINEAR OR DISTRIBUTIONAL COUPLING.**

## 42. Revised endpoint of the mathematical bridge

The present work has established two distinct results:

1. a reproducible statistical recurrence in independently generated
   random-Pauli dimensions, including the frozen 12Q confirmation;
2. an exact and locally interval-certified preservation theorem for a
   deliberately coupled ancilla model.

v25.40 shows that result 2 is not a direct generative explanation of result 1.
The remaining non-circular route is to investigate whether the independent
ensemble has a distributional or spectral-order self-similarity that preserves
the hotspot statistic. Such a claim would need a new predeclared theorem or
falsification design; it cannot be inferred from the arithmetic index map.

## 43. v25.41 exact sparse-Pauli multiplicity theorem

Represent a (q)-qubit Pauli word, modulo its phase, by a vector in the binary
symplectic space (mathbb F_2^{2q}). Let the Pauli words occurring in (H)
span a subspace of binary dimension (s), and let the restricted symplectic
form have rank (t), where (t) is even.

By the symplectic normal form, a Clifford change of basis places this span on

\[
s-\frac t2
\]

active qubits. Therefore every Hamiltonian in the real span of those Pauli
words has the form

\[
UHU^\dagger=H_{\mathrm{active}}\otimes
I_{2^{q-s+t/2}}.
\]

Consequently every eigenvalue has multiplicity divisible by

\[
M=2^{q-s+t/2}.
\]

Equivalently, when the (s) sampled words are binary independent, the
(2^q)-dimensional physical spectrum is the (2^s)-dimensional twisted
left-regular spectrum repeated

\[
R=2^{q-s}
\]

times. This follows also by comparing characters: every non-identity Pauli has
zero trace in both representations, while the identity traces are (2^q) and
(2^s).

For all forty frozen v25.31 12Q Hamiltonians, (s=5). Thirty-five had
(t=4), forcing (M=512), and five had (t=2), forcing (M=256). The
common regular-spectrum repetition was (R=2^{12-5}=128).

**Status: EXACT THEOREM; FROZEN FORTY-SEED ALGEBRA CLASSIFICATION.**

## 44. The (2k+1) rule is normalized spectral-rank preservation

The recurrence obeys the identity

\[
\frac{(2k+1)+1}{2^{q+1}}=\frac{k+1}{2^q}.
\]

Thus it exactly preserves the normalized left-pair boundary. The two source
chains are

\[
k_A(q)=2^{q-2}-1,
\qquad
k_B(q)=3\,2^{q-2}-1,
\]

and therefore select the (1/4) and (3/4) source-spectrum positions. After
the frozen two-branch target lift, the four tested boundaries are the
(1/8,3/8,5/8,7/8) positions.

For a target dimension (Q), the five-generator regular spectrum is repeated
(R_Q=2^{Q-5}) times. The four candidate pairs have left indices

\[
4R_Q-1,\quad12R_Q-1,\quad20R_Q-1,\quad28R_Q-1,
\]

so they are exactly regular-algebra block boundaries. A local control displaced
by (d\in\{\pm2,\pm4,\ldots,\pm10\}) crosses such a boundary only when
(R_Q\mid d). Hence the number of the ten controls forced to lie inside an
exact repeated-eigenvalue block is:

| Target | (R_Q) | Forced within-block controls |
|---:|---:|---:|
| 7Q | 4 | 6/10 |
| 8Q | 8 | 8/10 |
| 9Q | 16 | 10/10 |
| 10Q | 32 | 10/10 |
| 11Q | 64 | 10/10 |
| 12Q | 128 | 10/10 |

This supplies an exact structural explanation for both the coordinate
recurrence and its apparent strengthening in the high-Q regime: the number of
Hamiltonian terms and the absolute control window were held fixed while the
spectral repetition length doubled.

**Status: PROVED EXACTLY FOR THE FROZEN FIVE-TERM ENSEMBLE AND GEOMETRY.**

## 45. Exact reflection symmetry and dependence of the two candidates

For five binary-independent Pauli words (P_i), non-degeneracy of the ambient
symplectic form makes the map

\[
r\longmapsto(\langle r,P_1\rangle,\ldots,
\langle r,P_5\rangle)
\]

surjective. Hence there exists a Pauli word (R) with
(langle R,P_i\rangle=1) for every (i). It anticommutes with all five
terms, and therefore

\[
RHR^\dagger=-H.
\]

The spectrum is exactly reflection-symmetric. The v25.41 audit found a common
anticommuter for 40/40 frozen Hamiltonians and equality of the unordered A/B
branch-gap pairs to a maximum floating-point discrepancy
(5.440093\times10^{-15}).

The checkpoint analysis then found:

- both candidates positive in the same six batches and negative in the same
  two batches;
- prominence-vector correlation (0.989390733);
- identical eligible seed sets in every batch;
- eligible seed counts ([4,3,3,3,2,5,2,1]) for each candidate;
- each candidate ranks first among its eleven stored local labels;
- the one-candidate exact sign-test tail for 6/8 positives is
  (37/256=0.14453125).

The two reported 6/8 confirmations are therefore not independent replications.
Moreover, the local controls are not algebraically matched controls: at 12Q
their adjacent eigenvalue gaps are exactly zero, while the candidates are
regular-spectrum boundaries.

**Status: EXACT FINITE CHECKPOINT AUDIT. THE LOCAL RANKS ARE REAL, BUT THE
ORIGINAL CONTROL COMPARISON DOES NOT ISOLATE A TRANSPORTED HOTSPOT EFFECT.**

## 46. Revised Phase 2 conclusion after v25.41

The high-Q coordinate recurrence is now mathematically explained within the
implemented protocol by three jointly sufficient facts:

1. (2k+1) preserves normalized spectral rank;
2. a fixed five-term Pauli algebra has exponentially growing spectator
   multiplicity as (q) increases;
3. the frozen candidates sit on repeated-algebra boundaries while the fixed
   local controls increasingly sit inside exactly flat repeated blocks.

Accordingly, the high-Q recurrence must not be presented as evidence that a
physical hotspot is transported between independently generated Hilbert
spaces. It is a structural consequence of the sparse ensemble and the
candidate/control geometry.

This conclusion does not alter or delete any numerical output. The positive
REAL-NULL boundary prominences remain correctly recorded, and both frozen
boundaries rank first locally in the stored 12Q data. What changes is their
interpretation and evidential weight.

The controlled-ancilla theorems and the v25.39 interval certificate remain
mathematically valid for their stated model, but they are separate from the
mechanism generating the original high-Q recurrence.

A renewed physical test would require, before execution:

1. a Hamiltonian ensemble whose number or locality of independent terms scales
   with (q), eliminating the growing spectator identity factor;
2. controls drawn from the same algebraic boundary class as the candidates;
3. independent candidate families or a dependence-aware hierarchical test;
4. reporting by effective sample size, including the number of eligible seeds
   in every batch;
5. a separate holdout set not involved in discovery.

**Status: THE ORIGINAL HIGH-Q RECURRENCE HAS REACHED A MATHEMATICAL ENDPOINT:
ITS INDEX PROGRESSION IS EXPLAINED AS A PROTOCOL-INDUCED SPARSE-ALGEBRA EFFECT.
A BROADER PHYSICAL SOFT-SPACE CLAIM REMAINS UNESTABLISHED.**

## 47. v25.42 degeneracy and the necessary basis-invariance condition

Let (E) be an eigenvalue of (H) with eigenspace (mathcal E_E) of
dimension (m>1). The Hamiltonian determines the full spectral projector

\[
P_E=\sum_{j=1}^{m}|e_j\rangle\langle e_j|,
\]

but it does not determine the individual basis vectors (|e_j\rangle). For
every (U_E\in U(m)), the rotated vectors

\[
|e'_j\rangle=\sum_{\ell=1}^{m}|e_\ell\rangle(U_E)_{\ell j}
\]

give the same Hamiltonian and the same projector (P_E).

The Phase-2 score, however, selects particular ordinal eigenvectors at
((i,i+1)). When either selected vector lies in an eigenspace of dimension
larger than the selected slice, a rotation (U_E) changes the selected
two-dimensional projector while leaving (H) unchanged. Therefore the score
can be a function of (H) alone only if it is invariant under every such
block rotation. Sufficient alternatives include scoring the complete spectral
projectors or imposing an independently specified and physically justified
basis-selection observable.

This is not a small numerical detail. In the five-term ensemble the forced
degenerate eigenspaces grow exponentially with (q), so the selected two-vector
slice becomes an increasingly small and arbitrary part of the physical
eigenspace.

**Status: EXACT NECESSARY INVARIANCE CONDITION. GENERAL INVARIANCE OF THE
V25.31 PAIR SCORE WAS NOT ESTABLISHED.**

## 48. v25.42 frozen falsification test

`v25_42_degenerate_basis_invariance_test.py` instantiated the exact issue in
the same five-term Pauli ensemble at 8Q, where complete calculations are
inexpensive. Before execution it fixed:

- twelve Hamiltonian seeds (25042000,ldots,25042011);
- the source coordinates 31 and 95 and their two target branches;
- the same local radius (pm10), step 2;
- dephasing (Z/ZZ) and transverse (X/XX) perturbations;
- the v25.31 coherence-ratio and prominence definitions;
- eight deterministic Haar rotations inside every degenerate eigenspace.

REAL and NULL eigenbases were rotated separately. Hamiltonians, eigenvalues,
perturbations, coordinates, controls, gap gate, and regularizer were held
fixed. No ineligible seed was removed; ten of the twenty-four seed-candidate
instances were eligible and fourteen were reported as ineligible.

The largest reconstruction discrepancies were

\[
\|H_{\mathrm{REAL}}'-H_{\mathrm{REAL}}\|_2
=1.434724\times10^{-14},
\]

and

\[
\|H_{\mathrm{NULL}}'-H_{\mathrm{NULL}}\|_2
=9.209561\times10^{-15}.
\]

Despite the unchanged Hamiltonians:

| Rotation | Eligible cases whose sampled range crossed zero | Baseline sign reversed at least once |
|---|---:|---:|
| REAL degenerate basis | 2/10 | 2/10 |
| NULL degenerate basis | 4/10 | 4/10 |

For example, seed 25042000 at (k=95) had baseline prominence
(-0.011960), while legal REAL-basis rotations produced the range
([-0.005876,+0.004988]). At (k=31), rotating only the NULL degenerate
basis produced ([-0.036861,+0.009417]) from baseline (+0.008740).

A single such sign reversal is a counterexample to hotspot-sign invariance.
The multiple frozen counterexamples therefore establish that the score is not
a well-defined observable of the degenerate Hamiltonian and perturbation alone.

**Status: BASIS-INVARIANCE FALSIFIED BY PREDECLARED FINITE COUNTEREXAMPLES;
HAMILTONIAN PRESERVATION NUMERICALLY VERIFIED TO APPROXIMATELY
(1.5\times10^{-14}).**

## 49. Final endpoint for the original five-term high-Q model

v25.41 explained the coordinate recurrence as normalized spectral-rank
preservation combined with sparse-Pauli multiplicity and unmatched controls.
v25.42 now shows that the remaining REAL-NULL hotspot sign can change under a
physically equivalent choice of eigenbasis for the same Hamiltonian.

Therefore the original five-term high-Q protocol cannot support the claim that
its selected ordinal two-vector hotspots are basis-independent physical
structures. The stored numerical results remain faithful records of the
implemented diagonalization convention, but they are not evidence for an
intrinsic transported hotspot observable.

Further repetitions of the same protocol cannot repair this defect. A new
experiment would have to change the scientific model, not merely increase the
seed count. The two principled options are:

1. replace ordinal two-vector slices by complete degenerate-eigenspace
   projectors and define boundary-matched controls; or
2. use a physically motivated Hamiltonian ensemble with sufficient scaled
   local terms to remove the growing spectator degeneracy, followed by an
   entirely new frozen holdout protocol.

The v25.33-v25.39 controlled-ancilla results remain valid within their stated
model. They do not restore basis invariance to the original five-term ensemble.

**FINAL STATUS: THE ORIGINAL HIGH-Q PHASE-2 CONSTRUCTION IS MATHEMATICALLY
EXHAUSTED. ITS RECURRENCE IS STRUCTURALLY EXPLAINED, AND ITS RESIDUAL HOTSPOT
SIGN IS BASIS-DEPENDENT. ANY CONTINUATION REQUIRES A NEW MODEL AND A NEW
PREDECLARED EXPERIMENT.**
