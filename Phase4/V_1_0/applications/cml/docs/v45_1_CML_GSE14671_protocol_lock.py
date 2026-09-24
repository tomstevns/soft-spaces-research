#!/usr/bin/env python3
from pathlib import Path
import hashlib

EXPECTED_SHA256 = "c620936dc5832ac26fca09155b7b89d241120d7e3399e9c2c50e38585ad63b0d"
HERE = Path(__file__).resolve().parent
PATH = HERE / "v45_1_CML_GSE14671_CROSS_ENDPOINT_PREREGISTRATION.txt"
actual = hashlib.sha256(PATH.read_bytes()).hexdigest()
print("v45.1 GSE14671 preregistration integrity check")
print("Expected:", EXPECTED_SHA256)
print("Actual:  ", actual)
if actual != EXPECTED_SHA256:
    raise SystemExit("FAIL: v45.1 preregistration has changed.")
print("PASS: v45.1 preregistration unchanged.")
