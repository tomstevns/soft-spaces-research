#!/usr/bin/env python3
from pathlib import Path
import hashlib
EXPECTED_SHA256="3874a36af23a9f11363270319dd09e5dcff16a12957814ddf48bbf4f8916bd25"
p=Path(__file__).resolve().parent/"v47_0_CML_STATE_LANDSCAPE_PREREGISTRATION.txt"
a=hashlib.sha256(p.read_bytes()).hexdigest()
print("Expected:",EXPECTED_SHA256)
print("Actual:  ",a)
raise SystemExit("FAIL") if a!=EXPECTED_SHA256 else print("PASS: v47.0 protocol unchanged.")
