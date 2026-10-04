"""Plot the completed frozen-feature diagnostic; accepts aggregate results only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("evaluation", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    d = json.loads(a.evaluation.read_text())
    assert d["status"] == "REAL_FROZEN_EMBEDDING_REPORT_TARGET_DIAGNOSTIC_COMPLETE"
    assert len(d["per_target"]) == 12 and d["official_score"] is None
    assert d["all12_findings_retained"] and d["finite_macro_bootstrap_draws"] == 5000
    assert not d["production_promotion"] and not d["independent_clinical_validation"]
    assert not a.output.exists(), "Preserve existing figure directories"
    a.output.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "knee-masks-20261004"})
    rows = [*d["per_target"], {"target": "All 12 findings", "delta": d["addressed_minus_ordinary"],
                             "paired_ci95": d["paired_macro_ci95"]}]
    names = [r["target"] for r in rows]
    values = np.array([r["delta"] for r in rows]) * 100
    intervals = np.array([r["paired_ci95"] for r in rows]) * 100
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")
    for i, r in enumerate(rows):
        color = "#162e46" if i == 12 else "#216e72"
        ax.plot(intervals[i], [i, i], color=color, linewidth=2.2 if i == 12 else 1.5)
        ax.scatter(values[i], i, c=color, s=70 if i == 12 else 30, zorder=3)
    ax.axvline(0, color="#94a3b8", linewidth=1, linestyle="--")
    ax.axhline(11.5, color="#cbd5e1", linewidth=1)
    ax.set_yticks(range(len(rows)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Masked minus ordinary supervision: AUC percentage points")
    ax.set_xlim(-8.7, 4.8)
    ax.grid(axis="x", alpha=.12)
    for edge in ["top", "right", "left"]:
        ax.spines[edge].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="y", length=0)
    fig.suptitle("Report masks did not help this frozen representation", x=.05, ha="left",
                 fontsize=18, weight="bold", color="#162e46")
    fig.text(.05, .91, "MedSigLIP six-view features · matched linear heads · final epoch 20 · one reused fold",
             color="#475569", fontsize=10)
    fig.text(.05, .04, "Points: paired report-label AUC differences. Lines: 95% report-group bootstrap intervals (5,000 draws).\n"
             "866 studies / 815 held report groups; all 12 findings retained. Same-source pseudo-targets, not clinical or official validation.\n"
             "Macro AUC 0.84256 → 0.83391; Δ −0.00864 [−0.01658, −0.00073]. No production promotion.",
             fontsize=9, color="#475569")
    fig.subplots_adjust(left=.20, right=.95, top=.87, bottom=.17)
    fig.savefig(a.output / "masked-supervision-result.svg", metadata={"Date": None})
    fig.savefig(a.output / "masked-supervision-result.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
