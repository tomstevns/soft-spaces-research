#!/usr/bin/env python3
"""v41.16 preregistration integrity lock."""
from pathlib import Path
import hashlib

EXPECTED_SHA256 = "00f27acd8c4b3bc8fcdd58368235f1bb8538588aa8e9c40720eb885d9b555505"

HERE = Path(__file__).resolve().parent
p = HERE / "v41_16_PREREGISTRATION.txt"

if not p.exists():
    raise FileNotFoundError(p)

actual = hashlib.sha256(p.read_bytes()).hexdigest()

print("v41.16 preregistration integrity check")
print("Expected:", EXPECTED_SHA256)
print("Actual:  ", actual)

if actual != EXPECTED_SHA256:
    raise SystemExit("FAIL: v41.16 preregistration has changed.")

print("PASS: v41.16 preregistration is unchanged.")
