"""Plot approved V69 aggregates only; no training or private data access.

SPDX-License-Identifier: MIT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def validate(data):
    need(data["schema"] == "knee-v69-public-failure-aggregate-v1", "unexpected aggregate schema")
    progress = data["progress"]
    need(progress["finite_gradient_groups_completed"] == progress["optimizer_calls_observed"] == 3,
         "completed-group accounting drift")
    need(progress["failed_accumulation_group"] == 4 and progress["hold_before_clipping"],
         "failure-boundary drift")
    need(not progress["fourth_optimizer_update_performed"], "fourth-update claim drift")
    snapshots = data["group_4_snapshots"]
    need([s["index_zero_based"] for s in snapshots] == [6, 7], "snapshot binding drift")
    need([s["phase"] for s in snapshots] == ["after_scaled_backward_before_unscale", "after_unscale_before_clip"],
         "observation phase drift")
    for snapshot in snapshots:
        need(snapshot["finite_gradient_parameters"] + snapshot["nonfinite_gradient_parameters"]
             + snapshot["missing_gradient_parameters"] == snapshot["gradient_parameters_observed"],
             "gradient partition failed")
        need((snapshot["gradient_parameters_observed"], snapshot["finite_gradient_parameters"],
              snapshot["nonfinite_gradient_parameters"], snapshot["missing_gradient_parameters"],
              snapshot["Inf_elements"], snapshot["NaN_elements"]) == (411, 410, 1, 0, 18, 54),
             "group-4 aggregate drift")
    need(data["localized_gradient"]["cause"] == "UNKNOWN"
         and not data["localized_gradient"]["AMP_cause_inferred"], "unqualified causal claim")
    need(data["prospective_precision_diagnostic"]["status"] == "SOURCE_ONLY"
         and not data["prospective_precision_diagnostic"]["executed"], "prospective execution claim")
    need(not data["qualifications"]["repair_demonstrated"]
         and data["qualifications"]["official_score"] is None, "unqualified repair or score")


def box(ax, x, text, color, edge):
    ax.add_patch(FancyBboxPatch((x, 4.35), 2.4, 1.2,
                              boxstyle="round,pad=0.06,rounding_size=0.08",
                              facecolor=color, edgecolor=edge, linewidth=1.4))
    ax.text(x + 1.2, 4.95, text, ha="center", va="center", fontsize=10.3, linespacing=1.55)


def render(data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=170)
    fig.patch.set_facecolor("#ffffff")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.subplots_adjust(left=0.015, right=0.99, top=0.985, bottom=0.02)
    ax.text(0.3, 6.17, "V69 stopped before optimizer update 4", fontsize=19, weight="bold", color="#182638")
    ax.text(0.3, 5.82, "Observed gradient failure; cause unknown; no repair or accuracy result", fontsize=11, color="#46566a")
    progress = data["progress"]
    first = data["groups_1_to_3"]
    failure = data["group_4_snapshots"][0]
    xs = [0.35, 3.3, 6.25, 9.2]
    box(ax, xs[0], f"Groups 1–{progress['finite_gradient_groups_completed']}\n"
        f"{first['finite_gradient_parameters_each_snapshot']}/"
        f"{first['gradient_parameters_observed_each_snapshot']} gradients finite\n"
        f"{progress['optimizer_calls_observed']} optimizer calls", "#eaf3f8", "#4c879e")
    box(ax, xs[1], f"Group {progress['failed_accumulation_group']} backward\n"
        f"{progress['microbatches_per_group']} accumulated microbatches\nProbe before unscale", "#f3f0e7", "#a38a4f")
    box(ax, xs[2], f"One stem gradient\n{failure['Inf_elements']} Inf + {failure['NaN_elements']} NaN\n"
        f"{failure['finite_gradient_parameters']} other parameters finite", "#fff0e8", "#c47b4d")
    box(ax, xs[3], "HOLD\nBefore clipping / update 4\nFourth update blocked", "#fceced", "#bd5b63")
    for left, right in zip(xs, xs[1:]):
        ax.add_patch(FancyArrowPatch((left + 2.48, 4.95), (right - 0.1, 4.95),
                                    arrowstyle="-|>", mutation_scale=17, linewidth=1.6, color="#526273"))

    ax.text(0.35, 3.87, "The two group-4 observations agree", fontsize=13, weight="bold", color="#182638")
    ax.text(0.35, 3.52, "Observed phase", fontsize=10.5, weight="bold")
    for x, label in ((5.15, "Finite parameters"), (7.65, "Nonfinite parameters"), (10.25, "Missing")):
        ax.text(x, 3.52, label, fontsize=10.5, weight="bold", ha="center")
    labels = ["6 · After scaled backward, before unscale", "7 · After unscale, before clipping"]
    for y, label, snapshot in zip((3.15, 2.76), labels, data["group_4_snapshots"]):
        ax.text(0.35, y, label, fontsize=10.5)
        for x, key in ((5.15, "finite_gradient_parameters"), (7.65, "nonfinite_gradient_parameters"),
                       (10.25, "missing_gradient_parameters")):
            ax.text(x, y, str(snapshot[key]), fontsize=11, ha="center")
    localized = data["localized_gradient"]
    shape = " × ".join(str(dimension) for dimension in localized["gradient_shape"])
    ax.text(0.35, 2.35, f"{localized['parameter_name']}  ·  {localized['gradient_dtype']}  ·  {shape}  ·  "
            f"scale {failure['grad_scaler_scale']:,.0f} / tracker {failure['grad_scaler_growth_tracker']}",
            fontsize=10.2, color="#46566a")

    ax.add_patch(FancyBboxPatch((0.35, 0.47), 11.2, 1.45,
                              boxstyle="round,pad=0.06,rounding_size=0.07",
                              facecolor="#f8fafc", edgecolor="#8190a2", linewidth=1.2, linestyle="--"))
    ax.text(0.55, 1.65, "SOURCE ONLY · proposed paired precision diagnostic", fontsize=12.7, weight="bold", color="#344b66")
    arms = data["prospective_precision_diagnostic"]["arms"]
    ax.text(0.55, 1.25, arms[0], fontsize=11.2)
    ax.text(6.3, 1.25, arms[1], fontsize=11.2)
    ax.text(0.55, 0.85, "Earn and record a common pre-group-4 boundary; preserve the finite gate.", fontsize=10.3)
    ax.text(0.55, 0.58, "No executed comparison or bitwise V69 replay is claimed.", fontsize=10.3, color="#46566a")
    ax.text(0.35, 0.1, "Data: approved run and gradient aggregates. The figure is not an accuracy or throughput measurement.",
            fontsize=8.5, color="#6b7684")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--svg", type=pathlib.Path, required=True)
    parser.add_argument("--png", type=pathlib.Path, required=True)
    args = parser.parse_args()
    raw = args.data.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    validate(data)
    need(args.svg.resolve() != args.png.resolve(), "output paths must differ")
    need(not args.svg.exists() and not args.png.exists(), "use fresh output filenames")
    fig = render(data)
    description = "Approved V69 failure aggregates; data SHA256 " + digest
    with args.svg.open("xb") as stream:
        fig.savefig(stream, format="svg", metadata={"Title": "V69 gradient failure", "Description": description,
                                                    "Creator": "Original MIT-licensed plot_failure.py"})
    with args.png.open("xb") as stream:
        fig.savefig(stream, format="png", metadata={"Description": description, "Software": "plot_failure.py"})
    plt.close(fig)
    print(json.dumps({"status": "PUBLIC_AGGREGATE_FIGURE_CREATED", "aggregate_sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
