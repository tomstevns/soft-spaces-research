#!/usr/bin/env python3
from pathlib import Path
import hashlib

EXPECTED_SHA256 = "3d397a9a5edec39c4809abf95daf031960d6f4caf14314492c17f3606b480fe2"

HERE = Path(__file__).resolve().parent
candidates = [
    HERE / "v42_1d_CML_FROZEN_DESIGN.txt",
    HERE.parent / "docs" / "v42_1d_CML_FROZEN_DESIGN.txt",
]

path = next((p for p in candidates if p.exists()), None)
if path is None:
    raise FileNotFoundError("v42_1d_CML_FROZEN_DESIGN.txt not found")

actual = hashlib.sha256(path.read_bytes()).hexdigest()

print("v42.1d frozen-design integrity check")
print("Expected:", EXPECTED_SHA256)
print("Actual:  ", actual)

if actual != EXPECTED_SHA256:
    raise SystemExit("FAIL: v42.1d frozen design has changed.")

print("PASS: v42.1d frozen design is unchanged.")
