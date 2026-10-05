"""Reproduce the measured V54 timing figure from explicit, deidentified JSON.

Example: python plot_epoch_timing.py --input epoch_observations.json
Existing output directories and files are never overwritten.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def read_observations(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != 1:
        raise ValueError("Expected schema_version 1")
    source = data["source"]
    if source["exact_version"] != 54:
        raise ValueError("This figure describes the frozen V54 experiment")
    for key in ("native_source_sha256", "bounded_log_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", source[key]):
            raise ValueError("Invalid source hash")
    if source["log_clock"] != "Kaggle platform elapsed seconds":
        raise ValueError("Unexpected timing clock")
    terminal = data["terminal"]
    if terminal["status"] != "HOLD_NATIVE_TRIAL" or terminal["partial_checkpoints_eligible"] is not False:
        raise ValueError("Frozen V54 ended HOLD with ineligible partial checkpoints")
    if terminal["quality_metric"] is not None or terminal["official_score"] is not None:
        raise ValueError("This timing artifact contains no quality or official score")
    if not re.fullmatch(r"[0-9a-f]{64}", terminal["receipt_sha256"]):
        raise ValueError("Invalid terminal receipt hash")
    if data["required_fit_updates_per_network"] != 3488 or data["networks_per_arm"] != 2:
        raise ValueError("Unexpected original fit contract")
    events = data["observations"]
    if len(events) != 4:
        raise ValueError("Expected four recorded completed-epoch events")
    series = {}
    for arm in ("ordinary", "peer_selection"):
        rows = sorted((e for e in events if e["arm"] == arm), key=lambda e: e["epoch"])
        if [e["epoch"] for e in rows] != [1, 2] or [e["updates"] for e in rows] != [872, 1744]:
            raise ValueError("Unexpected epoch/update observations")
        for row in rows:
            if row["event"] != "NATIVE_EPOCH_COMPLETE":
                raise ValueError("Only completed-epoch measurements are supported")
            seconds = row["platform_elapsed_seconds"]
            if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Elapsed seconds must be positive and finite")
        if rows[1]["platform_elapsed_seconds"] <= rows[0]["platform_elapsed_seconds"]:
            raise ValueError("Epoch event times must increase")
        series[arm] = rows
    return data, series, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Explicit V54 epoch_observations.json")
    parser.add_argument("--out", type=Path, help="A new output directory; existing directories are refused")
    args = parser.parse_args()
    data, series, input_sha = read_observations(args.input)
    out = args.out or Path.cwd() / ("epoch-timing-v54-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    arms = ("ordinary", "peer_selection")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("#f7f9fb")
    colors = {"ordinary": "#254b78", "peer_selection": "#b85b31"}
    labels = {"ordinary": "Ordinary BCE · two networks", "peer_selection": "Peer selection · two networks"}
    for arm in arms:
        rows = series[arm]
        # Separate marker shapes retain both nearly coincident observed arm events.
        axes[0].scatter([r["platform_elapsed_seconds"] / 60 for r in rows],
                        [r["updates"] for r in rows], s=150 if arm == "ordinary" else 85,
                        facecolors="none" if arm == "ordinary" else colors[arm],
                        edgecolors=colors[arm], linewidths=2,
                        marker="o" if arm == "ordinary" else "^", label=labels[arm])
    axes[0].axhline(3488, color="#6e7683", linestyle=(0, (4, 4)), linewidth=1.2)
    axes[0].text(29.5, 3560, "Required fit seal: 3,488 updates per network", fontsize=10, color="#515966")
    axes[0].set(xlabel="Kaggle platform elapsed time (minutes)",
                ylabel="Observed cumulative updates per network", xlim=(29, 55), ylim=(0, 3800))
    axes[0].yaxis.set_major_locator(MultipleLocator(872))
    axes[0].set_title("Two completed epochs", loc="left", pad=16)
    axes[0].grid(axis="y", color="#dde3e9", linewidth=.8)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, .83), fontsize=9)

    for i, arm in enumerate(arms):
        rows = series[arm]
        measured = (rows[1]["platform_elapsed_seconds"] - rows[0]["platform_elapsed_seconds"]) / 60
        axes[1].barh(i, measured, height=.48, color=colors[arm])
        axes[1].text(measured + .35, i, f"{measured:.2f} min", va="center", fontsize=12, color=colors[arm])
    axes[1].set(yticks=range(2), yticklabels=["Ordinary BCE", "Peer selection"],
                xlabel="Measured time from epoch 1 event to epoch 2 event", xlim=(0, 23))
    axes[1].invert_yaxis()
    axes[1].set_title("The second epoch took ~18 minutes", loc="left", pad=16)
    axes[1].grid(axis="x", color="#dde3e9", linewidth=.8)
    axes[1].set_axisbelow(True)
    fig.suptitle("Native MRI training: the evidence so far", x=.065, ha="left", fontsize=21,
                 fontweight="bold", color="#172b42")
    fig.text(.065, .9, "RSNA Knee · exact V54 live log · matched four-network experiment", color="#596573")
    fig.text(.065, .065, "Markers are completed-epoch events. Timing includes checkpoint writes and hashes.\n"
             "The first event also includes startup; no unobserved batch trajectory is drawn.\n"
             "Terminal: HOLD_NATIVE_TRIAL. Partial checkpoints are ineligible. No accuracy improvement is established.",
             fontsize=9, color="#596573", linespacing=1.5)
    fig.subplots_adjust(left=.085, right=.97, top=.78, bottom=.27, wspace=.43)
    for name in ("epoch_timing.svg", "epoch_timing.png"):
        fig.savefig(out / name, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    record = {
        "artifact": "Measured V54 completed-epoch timing",
        "input_json_sha256": input_sha,
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": data["source"],
        "observed_events": data["observations"],
        "terminal": data["terminal"],
        "measurement_scope": data["measurement_scope"],
        "quality_claim": False,
        "matplotlib_version": matplotlib.__version__,
        "figure_files": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                         for name in ("epoch_timing.svg", "epoch_timing.png")},
    }
    with (out / "figure_provenance.json").open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output_directory": str(out.resolve()), "input_sha256": input_sha,
                      "observed_events": len(data["observations"]), "quality_claim": False}))


if __name__ == "__main__":
    main()
