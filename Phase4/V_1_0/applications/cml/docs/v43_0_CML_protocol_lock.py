#!/usr/bin/env python3
from pathlib import Path
import hashlib

EXPECTED_SHA256 = "f32741ca64d2bc6c015374b0e5e0b25cf5820141d57bd3e5ba8a53a94a097059"

HERE = Path(__file__).resolve().parent
candidates = [
    HERE / "v43_0_CML_DIRECT_SUBSPACE_PROTOCOL.txt",
    HERE.parent / "docs" / "v43_0_CML_DIRECT_SUBSPACE_PROTOCOL.txt",
]

path = next((p for p in candidates if p.exists()), None)
if path is None:
    raise FileNotFoundError("v43_0_CML_DIRECT_SUBSPACE_PROTOCOL.txt not found")

actual = hashlib.sha256(path.read_bytes()).hexdigest()
print("v43.0 protocol integrity check")
print("Expected:", EXPECTED_SHA256)
print("Actual:  ", actual)

if actual != EXPECTED_SHA256:
    raise SystemExit("FAIL: v43.0 protocol has changed.")

print("PASS: v43.0 protocol is unchanged.")
