#!/usr/bin/env python3
"""v41.15 preregistration integrity check."""
from pathlib import Path
import hashlib, json

EXPECTED_SHA256 = "979f7f4880fa3936edcdcd1055641bf5922abcc00caa3917e0fed8c8e464b56b"

HERE = Path(__file__).resolve().parent
p = HERE / "v41_15_PREREGISTRATION.txt"

if not p.exists():
    raise FileNotFoundError(p)

actual = hashlib.sha256(p.read_bytes()).hexdigest()

print("v41.15 preregistration integrity check")
print("Expected:", EXPECTED_SHA256)
print("Actual:  ", actual)

if actual != EXPECTED_SHA256:
    raise SystemExit("FAIL: preregistration has changed.")

print("PASS: preregistration is unchanged.")
