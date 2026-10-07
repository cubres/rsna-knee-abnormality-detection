"""Original synthetic demonstration; MIT, copyright 2026 cubres.

No real images, study identifiers, labels, models, files or network access.
Run from this directory: python demo_synthetic_geometry.py
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from mild_affine import synthetic_runtime_check

if __name__ == "__main__":
    torch.set_num_threads(2)
    result = synthetic_runtime_check(device="cpu")
    result["geometry_source_sha256"] = hashlib.sha256(
        Path(__file__).with_name("mild_affine.py").read_bytes()).hexdigest()
    result["scope"] = "Synthetic geometry runtime only; no native training, anatomy, speed or score claim."
    print(json.dumps(result, indent=2))
