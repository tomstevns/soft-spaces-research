# Soft Spaces Phase 2.2 — Qiskit-test efter v25.30

## Formål

Testen skal undersøge, om Qiskit kan gengive den frosne v25.30-mekanisme uden at ændre målkoordinater, kontroller, perturbationsfamilier eller beslutningsregel.

Den primære størrelse er ikke projektor-overlap. Den er perturbativ matrix-cancellation:

```text
DeltaC_f(k) = C_matrix_REAL,f(k) - C_matrix_NULL,f(k)
S(k)        = min(DeltaC_dephasing(k), DeltaC_transverse(k))
prominence  = S(k) - median(S(lokale kontroller))
```

Et frosset branch-punkt er `UPWARD_VALIDATED`, når prominence er positiv i mindst 75 procent af de uafhængige batchenheder.

## Frossen geometri

| Kilde → mål | Gren A | Gren B | Løft til målrum |
|---|---:|---:|---|
| 6Q → 7Q | 15 | 47 | `i=k` og `i=k+64` |
| 7Q → 8Q | 31 | 95 | `i=k` og `i=k+128` |
| 8Q → 9Q | 63 | 191 | `i=k` og `i=k+256` |

Reglen er `k_next = 2*k + 1`. Hver koordinat sammenlignes med lokale kildekoordinater inden for ±10 med skridt 2. De samme kontroller løftes til begge målgrene.

## REAL og NULL

1. REAL bygges som fem seedede, tilfældige Pauli-led.
2. REAL diagonaliseres til egenværdier `E` og egenbasis `V_REAL`.
3. NULL får præcis de samme egenværdier, men en seedet Haar-randomiseret egenbasis `V_NULL`.
4. De samme perturbationer anvendes på REAL og NULL.

## Perturbationsfamilier

- Dephasing: lokale `Z`-led og nærmeste-nabo `ZZ`-led.
- Transverse: lokale `X`-led og nærmeste-nabo `XX`-led.

For hver familie beregnes overgangsmatricen

```text
B = V† DeltaH V
```

og derefter v25.30's regulerede `C_matrix` for det todimensionale naboegenrum og resten af Hilbert-rummet.

## Qiskit-laget

Qiskit `Statevector` efterprøver udvalgte komplekse overgangsamplituder

```text
B[q,p] = <q|DeltaH|p>
```

mod den klassiske matrixberegning. Maksimal numerisk fejl rapporteres.

Programmet konstruerer desuden to Hadamard-testkredsløb for et repræsentativt Pauli-led:

- ét kredsløb måler realdelen af `<q|P|p>`;
- ét kredsløb måler imaginærdelen.

Disse kredsløb viser hardwarevejen. Den fulde v25.30-statistik beregnes fortsat eksakt, fordi et komplet hardwareestimat ville kræve meget mange overgangsmålinger.

## Testtrin

### 1. 7Q smoke-test

```powershell
python -X utf8 -u .\soft_spaces_v25_30_qiskit_design.py --target-qubits 7 --seeds-per-batch 2 --batches 1 --reps 1
```

Formålet er programkontrol og Qiskit-overensstemmelse, ikke videnskabelig bekræftelse.

### 2. Styrket 7Q-test

```powershell
python -X utf8 -u .\soft_spaces_v25_30_qiskit_design.py --target-qubits 7 --seeds-per-batch 25 --batches 2 --reps 3
```

### 3. Harmoniseret 8Q- og 9Q-test

```powershell
python -X utf8 -u .\soft_spaces_v25_30_qiskit_design.py --target-qubits 8 --seeds-per-batch 25 --batches 2 --reps 3
python -X utf8 -u .\soft_spaces_v25_30_qiskit_design.py --target-qubits 9 --seeds-per-batch 25 --batches 2 --reps 3
```

Programmet anvender automatisk v25.30's separate seedblokke: +0 for 7Q, +10.000.000 for 8Q og +20.000.000 for 9Q.

## Falsifikationsregel

- Koordinaterne må ikke flyttes efter resultatet.
- En svigtende gren må ikke erstattes af en bedre lokal koordinat.
- `UPWARD_NOT_VALIDATED` er et gyldigt videnskabeligt resultat.
- Kun prominence-raten mod den frosne grænse 0,75 afgør status.
