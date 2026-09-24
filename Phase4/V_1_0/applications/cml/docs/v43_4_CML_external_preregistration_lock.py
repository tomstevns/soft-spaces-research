#!/usr/bin/env python3
from pathlib import Path
import hashlib

EXPECTED_SHA256 = "361f3c57ccc5687bedc829080f86412fdc7f297a61fd814120502f5bc2b836b9"

HERE = Path(__file__).resolve().parent
candidates = [
    HERE / "v43_4_CML_EXTERNAL_GENERALIZATION_PREREGISTRATION.txt",
    HERE.parent / "docs" / "v43_4_CML_EXTERNAL_GENERALIZATION_PREREGISTRATION.txt",
]

path = next((p for p in candidates if p.exists()), None)

if path is None:
    raise FileNotFoundError(
        "v43_4_CML_EXTERNAL_GENERALIZATION_PREREGISTRATION.txt not found"
    )

actual = hashlib.sha256(path.read_bytes()).hexdigest()

print("v43.4 external preregistration integrity check")
print("Expected:", EXPECTED_SHA256)
print("Actual:  ", actual)

if actual != EXPECTED_SHA256:
    raise SystemExit("FAIL: v43.4 preregistration has changed.")

print("PASS: v43.4 preregistration is unchanged.")
