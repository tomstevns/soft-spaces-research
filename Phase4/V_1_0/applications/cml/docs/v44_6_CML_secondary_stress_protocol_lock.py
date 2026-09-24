#!/usr/bin/env python3
from pathlib import Path
import hashlib
EXPECTED_SHA256="7ee56ef21295b120f06712bc51c4f69450fe701d249601ce773ebef9ca827ef6"
p=Path(__file__).resolve().parent/"v44_6_CML_GSE236233_SECONDARY_STRESS_PREREGISTRATION.txt"
a=hashlib.sha256(p.read_bytes()).hexdigest()
print("Expected:",EXPECTED_SHA256)
print("Actual:  ",a)
raise SystemExit("FAIL") if a!=EXPECTED_SHA256 else print("PASS: v44.6 preregistration unchanged.")
