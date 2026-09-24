#!/usr/bin/env python3
from pathlib import Path
import hashlib

EXPECTED_SHA256="9536c2e1e69389eb7c1dedaef7028174647e5ed4925d459576494e5da7e59d4f"
p=Path(__file__).resolve().parent/"v46_1_CML_MASS_AWARE_GSE14671_PREREGISTRATION.txt"
a=hashlib.sha256(p.read_bytes()).hexdigest()
print("Expected:",EXPECTED_SHA256)
print("Actual:  ",a)
raise SystemExit("FAIL") if a!=EXPECTED_SHA256 else print("PASS: v46.1 protocol unchanged.")
