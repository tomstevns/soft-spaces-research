#!/usr/bin/env python3
from pathlib import Path
import hashlib

EXPECTED_SHA256 = "856f7a161d8d76c114a32d98ca812f36b52b018ef8108c4065a924f68b265e7c"
HERE = Path(__file__).resolve().parent
path = HERE / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION.txt"
if not path.exists():
    path = HERE.parent / "docs" / "v44_0_CML_CROSS_PLATFORM_PREREGISTRATION.txt"
if not path.exists():
    raise FileNotFoundError("v44.0 preregistration not found")

actual = hashlib.sha256(path.read_bytes()).hexdigest()
print("v44.0 preregistration integrity check")
print("Expected:", EXPECTED_SHA256)
print("Actual:  ", actual)
if actual != EXPECTED_SHA256:
    raise SystemExit("FAIL: v44.0 preregistration has changed.")
print("PASS: v44.0 preregistration is unchanged.")
