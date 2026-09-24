#!/usr/bin/env python3
from pathlib import Path
import hashlib
EXPECTED_SHA256="4a1131926769a8cfb1bf2007628da21299e5c9fc0021e9bc1184eec25af93114"
HERE=Path(__file__).resolve().parent
p=HERE/"v44_5_CML_EXTERNAL_GENERALIZATION_PREREGISTRATION.txt"
a=hashlib.sha256(p.read_bytes()).hexdigest()
print("Expected:",EXPECTED_SHA256)
print("Actual:  ",a)
raise SystemExit("FAIL") if a!=EXPECTED_SHA256 else print("PASS: v44.5 preregistration unchanged.")
