"""Plot the released 3x11 analytic golden as a cover; no clinical data. MIT."""
import argparse
import csv
import hashlib
import io
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--corpus", type=Path, default=HERE / "corpus")
parser.add_argument("--out", type=Path, required=True,
                    help="New PNG destination; existing files are never overwritten")
args = parser.parse_args()
with (args.corpus / "manifest.csv").open(newline="", encoding="utf-8") as handle:
    row = next(r for r in csv.DictReader(handle) if r["case_id"] == "ramp_3x11_i16_m2")
relative = Path(row["golden_path"])
if relative.is_absolute() or ".." in relative.parts:
    raise RuntimeError("Golden path must remain relative to the corpus")
raw = (args.corpus / relative).read_bytes()
if hashlib.sha256(raw).hexdigest() != row["golden_sha256"]:
    raise RuntimeError("Golden source differs")
image = np.load(io.BytesIO(raw), allow_pickle=False)
if image.shape != (3, 11) or image.dtype != np.float32:
    raise RuntimeError("Unexpected golden shape or dtype")
fig = plt.figure(figsize=(6.4, 3.2), dpi=100, facecolor="#111827")
fig.text(.5, .88, "DICOM FIXTURES", ha="center", color="#f9fafb", fontsize=16, weight="bold")
fig.text(.5, .75, "48 synthetic cases", ha="center", color="#a7f3d0", fontsize=11)
ax = fig.add_axes((.18, .27, .64, .35))
ax.imshow(image, cmap="magma", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
ax.set_axis_off()
fig.text(.5, .17, "43 analytic goldens + 5 deliberate refusals", ha="center", color="#f9fafb", fontsize=8)
fig.text(.5, .055, "ACTUAL 3 × 11 RAMP · SIGNED INT16 · MONOCHROME2", ha="center", color="#9ca3af", fontsize=6.8)
buffer = io.BytesIO()
fig.savefig(buffer, format="png", dpi=100, facecolor=fig.get_facecolor())
plt.close(fig)
with args.out.open("xb") as stream:
    stream.write(buffer.getvalue())
print({"bytes": len(buffer.getvalue()),
       "sha256": hashlib.sha256(buffer.getvalue()).hexdigest(), "shape": image.shape})
