#!/usr/bin/env python3
"""
setup_ibm_quantum_and_run.py

Gør tre ting:
1) Gemmer IBM Quantum API-key lokalt til channel "ibm_quantum_platform".
2) Tester forbindelsen og viser tilgængelige backends.
3) Starter v40_4_ibm_heron_prepare.py med samme Python/venv.

Kør fra din aktive .venv_qiskit:
    python -X utf8 -u .\setup_ibm_quantum_and_run.py
"""

from __future__ import annotations

import getpass
import subprocess
import sys
from pathlib import Path

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except ImportError:
    print("FEJL: qiskit_ibm_runtime er ikke installeret i denne Python.")
    print("Kontrollér at (.venv_qiskit) er aktiv.")
    raise SystemExit(1)


CHANNEL = "ibm_quantum_platform"
TARGET_PROGRAM = "v40_4_ibm_heron_prepare.py"


def get_api_key() -> str:
    print()
    print("IBM Quantum API-key skal kun indtastes denne ene gang.")
    print("Tegnene vises ikke på skærmen.")
    """token = getpass.getpass("IBM Quantum API-key: ").strip()"""
    token = "HcMUP66zUPoXu4Es9IQJRIkCZR8OsZiqvL0tUM5_FUwt"
    if not token:
        print("FEJL: Der blev ikke indtastet nogen API-key.")
        raise SystemExit(1)

    return token


def save_account(token: str) -> None:
    print()
    print(f"Gemmer IBM Quantum-konto for channel '{CHANNEL}' ...")

    QiskitRuntimeService.save_account(
        token=token,
        channel=CHANNEL,
        set_as_default=True,
        overwrite=True,
    )

    print("OK: Kontoen er gemt.")


def test_connection() -> None:
    print()
    print("Tester forbindelsen til IBM Quantum Platform ...")

    service = QiskitRuntimeService()
    backends = service.backends()

    print("OK: Forbindelsen virker.")
    print(f"Antal tilgængelige backends: {len(backends)}")

    for backend in backends:
        try:
            name = backend.name
        except Exception:
            name = str(backend)
        print(f"  - {name}")


def run_target() -> int:
    here = Path(__file__).resolve().parent
    target = here / TARGET_PROGRAM

    print()

    if not target.exists():
        print(f"IBM-forbindelsen er klar, men jeg fandt ikke:")
        print(f"  {target}")
        print()
        print(f"Læg dette setup-program i samme mappe som {TARGET_PROGRAM}.")
        return 2

    print(f"Starter nu {TARGET_PROGRAM} ...")
    print("=" * 72)

    command = [
        sys.executable,
        "-X",
        "utf8",
        "-u",
        str(target),
    ]

    completed = subprocess.run(command, cwd=here)

    print("=" * 72)
    print(f"{TARGET_PROGRAM} sluttede med exit code {completed.returncode}.")
    return completed.returncode


def main() -> None:
    token = get_api_key()

    try:
        save_account(token)
        test_connection()
    except Exception as exc:
        print()
        print("FEJL under IBM Quantum-login/test:")
        print(f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)

    raise SystemExit(run_target())


if __name__ == "__main__":
    main()
