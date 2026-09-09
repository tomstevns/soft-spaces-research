# Soft Spaces fase 2A — v25.32 Qiskit-hardwarepilot

## Formål

Dette er en reduceret 4-qubit gate-test af Soft Spaces-princippet. Den flytter
ikke den fulde klassiske 12Q-beregning over på en kvantecomputer.

Testen forbereder et todimensionalt REAL-underrum og tre fastlåste
NULL-kontroller med lavdybe Qiskit-kredsløb. Den samme perturbation påføres alle
modeller, kredsløbet inverteres, og overlevelsen i underrummet måles med shots.

De to fastlåste perturbationsfamilier er:

- dephasing: `Z0`
- transverse: `XXXX`

Godkendelseskriteriet er fastlagt før hardwarekørslen:

`normaliseret REAL - median(normaliseret NULL) > 0`

Begge familier skal bestå.

## Allerede gennemført

- Ideal Qiskit/Aer-simulering: 2/2 PASS.
- Generisk støjsimulering: 2/2 PASS.
- Qiskit-kredsløbsdiagram og resultatgraf er genereret.
- Hardwarevejen er implementeret, men er endnu ikke sendt til en fysisk IBM
  Quantum-processor.

Støjsimuleringen gav:

| Familie | REAL | Median NULL | Delta |
| --- | ---: | ---: | ---: |
| dephasing Z0 | 0.993311 | 0.962166 | +0.031144 |
| transverse XXXX | 0.948896 | 0.906558 | +0.042338 |

## Lokal simulator

```bash
python -m pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib pylatexenc
python soft_spaces_qiskit_hardware_pilot_v25_32.py --mode noisy
```

## IBM Quantum-hardware

Programmet indeholder ingen token. Når IBM-kontoen er gemt lokalt gennem
`QiskitRuntimeService`, kan den mindst belastede egnede processor vælges
automatisk:

```bash
python soft_spaces_qiskit_hardware_pilot_v25_32.py --mode hardware
```

En bestemt processor kan vælges med:

```bash
python soft_spaces_qiskit_hardware_pilot_v25_32.py --mode hardware --backend BACKEND_NAME
```

Hardwareoutputtet gemmer backendnavn, Runtime-job-id, shots, rå resultater og
den prædeklarerede PASS/FAIL-afgørelse. Først når denne kørsel er gennemført,
kan projektet hævde en egentlig kvantehardwarevalidering.
