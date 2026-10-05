"""Reproduce the timing figure from public scalar measurements.

Usage: python render.py --output-dir ./new-render
Outputs go to a new directory; existing files remain intact.
"""
from pathlib import Path
import argparse
import io
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(Path(__file__).with_name("measured-data.json").read_text())
    labels = {
        "t_dino": "DINO family", "extra_family": "Released reader · gate HOLD",
        "coat_resgated": "CoAt · resolution gated", "coat_global96": "CoAt · global 96",
        "coat_repairv1": "CoAt · repair v1", "coat_d4": "CoAt · depth zones",
        "raptor": "Raptor family", "t_a5": "A5 head", "t_rad": "RadImageNet heads",
        "inventory": "Input inventory", "t_assembly": "T family assembly", "coat_family": "CoAt family assembly",
    }
    times = data["block_seconds"]
    assert set(times) == set(labels)
    assert all(math.isfinite(v) and v >= 0 for v in times.values())
    assert data["final_gate"]["status"] == "HOLD" and not data["reader_predictions_blended"]
    ordered = sorted(times, key=times.get, reverse=True)
    values = [times[k] for k in ordered]
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "font.size": 10})
    fig, ax = plt.subplots(figsize=(10.4, 6.8), facecolor="#f7f7f2")
    ax.set_facecolor("#f7f7f2")
    colors = ["#c95c42" if k == "extra_family" else "#367a78" for k in ordered]
    ax.barh(range(len(ordered)), values, color=colors, height=.66)
    ax.set_yticks(range(len(ordered)), [labels[k] for k in ordered])
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.16)
    ax.set_xlabel("Observed seconds per block")
    ax.xaxis.grid(True, color="#d9dfd9", linewidth=.6)
    ax.set_axisbelow(True)
    for index, value in enumerate(values):
        ax.text(value + 1.5, index, f"{value:.2f}", va="center", color="#243a40", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)
    fig.text(.04, .95, "Base models passed. The added reader hit its timing gate.", fontsize=17, weight="bold", color="#15373f")
    fig.text(.04, .908, f"RSNA Knee · V{data['notebook_version']} · {data['visible_test_studies']} visible test studies · Tesla T4 × 2", fontsize=11, color="#486268")
    elapsed = data["native_receipt_elapsed_seconds"]
    fig.text(.04, .055, f"11 base blocks passed. Reader blend withheld. Native receipt elapsed: {elapsed:.1f} s.", fontsize=10, color="#15373f")
    fig.text(.04, .026, "Child timing numbers were not retained; no full-test projection or official quality gain is claimed.", fontsize=9, color="#486268")
    fig.subplots_adjust(left=.31, right=.96, top=.85, bottom=.13)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for suffix in ["svg", "png"]:
        buffer = io.BytesIO()
        fig.savefig(buffer, format=suffix, dpi=160, metadata={"Date": None} if suffix == "svg" else None)
        with (args.output_dir / ("observed-block-times." + suffix)).open("xb") as stream:
            stream.write(buffer.getvalue())
    plt.close(fig)


if __name__ == "__main__":
    main()
