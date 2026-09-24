#!/usr/bin/env python3
from pathlib import Path
import hashlib
EXPECTED_SHA256="a9bf1d80cf872a0ea5f2e8a3a8dded2d9181b59bdc09dfd64bc50df5cb25049c"
p=Path(__file__).resolve().parent/"v47_4_CML_STATE_LANDSCAPE_STABILITY_PREREGISTRATION.txt"
a=hashlib.sha256(p.read_bytes()).hexdigest()
print("Expected:",EXPECTED_SHA256)
print("Actual:  ",a)
raise SystemExit("FAIL") if a!=EXPECTED_SHA256 else print("PASS: v47.4 protocol unchanged.")
