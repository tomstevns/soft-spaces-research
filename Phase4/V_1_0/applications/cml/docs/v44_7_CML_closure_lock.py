#!/usr/bin/env python3
from pathlib import Path
import hashlib

EXPECTED_SHA256 = "c2b60ce46f7be903d9efa3b3c5dbe4f3d45e87609c7dd27816d88ee716695230"
HERE = Path(__file__).resolve().parent
PATH = HERE / "v44_7_CML_CLOSURE_AUDIT.txt"
actual = hashlib.sha256(PATH.read_bytes()).hexdigest()
print("v44.7 CML closure integrity check")
print("Expected:", EXPECTED_SHA256)
print("Actual:  ", actual)
if actual != EXPECTED_SHA256:
    raise SystemExit("FAIL: v44.7 closure has changed.")
print("PASS: v44.7 closure is unchanged.")
