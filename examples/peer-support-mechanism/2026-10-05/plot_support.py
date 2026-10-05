"""Render aggregate support and a synthetic soft-target loss illustration.

Requires NumPy and Matplotlib. No models or patient records are loaded.
Usage: python plot_support.py --input v54_support_counts.json --output new-figure
"""
import argparse
import hashlib
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="Fresh output prefix; creates .png, .svg and .json")
    args = parser.parse_args()
    paths = [pathlib.Path(str(args.output) + suffix) for suffix in (".png", ".svg", ".json")]
    if any(p.exists() for p in paths):
        raise SystemExit("Output must be fresh; an intended output path already exists")
    raw = args.input.read_bytes()
    data = json.loads(raw)
    full = np.asarray(data["full_support_per_network"], dtype=np.int64)
    kept = np.asarray(data["peer_epoch2_support_per_network"], dtype=np.int64)
    names = data["finding_order"]
    if full.shape != (len(names), 2) or kept.shape != full.shape:
        raise ValueError("Expected one negative/positive aggregate pair per finding")
    if np.any(full <= 0) or np.any(kept < 0) or np.any(kept > full):
        raise ValueError("Invalid support counts")
    if not 0 < float(data["nominal_retain_fraction"]) <= 1:
        raise ValueError("Expected a positive nominal retention fraction")
    totals = full.sum(axis=0)
    selected = kept.sum(axis=0)
    rates = np.r_[selected.sum() / totals.sum(), selected / totals] * 100
    navy, copper, teal, grey = "#18364a", "#b46740", "#377f82", "#566574"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": navy, "text.color": navy,
                         "xtick.color": grey, "ytick.color": grey,
                         "svg.hashsalt": "knee-peer-support-v1"})
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), facecolor="#fffdf8",
                            gridspec_kw={"height_ratios": [1, 1.45]})
    for ax in axs.flat:
        ax.set_facecolor("#fffdf8")
    ax = axs[0, 0]
    bars = ax.bar(["All cells", "Target ≤ 0.5", "Target > 0.5"], rates,
                  color=[navy, teal, copper], width=.56)
    ax.axhline(data["nominal_retain_fraction"] * 100, color=grey, linestyle="--",
               label="Nominal retain = 80%")
    ax.bar_label(bars, labels=[f"{x:.2f}%" for x in rates], padding=5)
    ax.set_ylim(0, 103)
    ax.set_ylabel("Selected / available cells (%)")
    ax.set_title("A  Measured epoch-2 support", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.text(.02, -.23, f"{selected.sum():,} / {totals.sum():,} cells per peer network",
            transform=ax.transAxes, color=grey, clip_on=False)
    ax = axs[0, 1]
    n = np.arange(1, 5)
    k = np.maximum(1, np.floor(float(data["nominal_retain_fraction"]) * n)).astype(int)
    ax.bar(n, k / n * 100, color=teal, width=.6)
    for ni, ki in zip(n, k):
        ax.text(ni, ki / ni * 100 + 4, f"keep {ki}/{ni}", ha="center")
    ax.set_xticks(n)
    ax.set_ylim(0, 118)
    ax.set_xlabel("Cells in one finding/class stratum")
    ax.set_ylabel("Realized retention (%)")
    ax.set_title("B  Algorithm example: floor plus minimum one", loc="left", fontweight="bold")
    ax.text(.02, -.32, "A batch split 2 + 2 keeps 1 + 1: only 50%.",
            transform=ax.transAxes, color=grey, clip_on=False)
    ax = axs[1, 0]
    y = np.arange(len(names))
    ax.barh(y-.18, kept[:, 0]/full[:, 0]*100, height=.34, color=teal, label="Target ≤ 0.5")
    ax.barh(y+.18, kept[:, 1]/full[:, 1]*100, height=.34, color=copper, label="Target > 0.5")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 103)
    ax.set_xlabel("Selected / available cells (%)")
    ax.set_title("C  Measured support by finding", loc="left", fontweight="bold", pad=35)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=9,
              borderaxespad=0)
    ax = axs[1, 1]
    target = np.linspace(.0001, .9999, 501)
    entropy = -target * np.log(target) - (1-target) * np.log1p(-target)
    ax.plot(target, entropy, color=copper, linewidth=2.6)
    ax.scatter([.1, .5, .9], [-v*np.log(v)-(1-v)*np.log1p(-v) for v in (.1,.5,.9)],
               color=copper, s=32)
    ax.set_xlabel("Synthetic soft target y; prediction p = y")
    ax.set_ylabel("Binary cross-entropy (nats)")
    ax.set_ylim(0, .83)
    ax.set_title("D  Synthetic illustration: raw BCE includes target entropy", loc="left", fontweight="bold")
    ax.text(.5, .77, "BCE(y, p) = H(y) + KL(Ber(y) || Ber(p))", ha="center", fontsize=10)
    ax.text(.5, .06, "Perfectly matching ambiguous targets still have higher loss.\nLow raw loss alone does not establish cleaner labels.",
            ha="center", va="bottom", fontsize=10, color=grey)
    fig.suptitle("What a tiny-batch peer filter actually selects", x=.075, ha="left",
                 fontsize=23, fontweight="bold")
    fig.text(.075, .925, "Knee V54: aggregate support evidence and two mathematical examples", fontsize=12, color=grey)
    fig.text(.075, .032, "V54 ended HOLD after 2/4 epochs. Partial weights are ineligible; no accuracy or official-score gain is shown.\n"
             "Classes use report-derived soft targets (>0.5), not verified diagnoses. Equal counts do not establish equal selected identities.",
             fontsize=10, color=grey)
    fig.subplots_adjust(left=.12, right=.975, top=.865, bottom=.13, hspace=.46, wspace=.30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=144, facecolor=fig.get_facecolor())
    fig.savefig(paths[1], facecolor=fig.get_facecolor(), metadata={"Date": None})
    plt.close(fig)
    record = {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "source_version": data["source_exact_version"],
        "source_native_sha256": data["source_native_sha256"],
        "available_cells_per_network": int(totals.sum()),
        "selected_cells_per_network": int(selected.sum()),
        "realized_retention_percent": rates.tolist(),
        "panels_A_C": "Measured deidentified V54 epoch-2 aggregates; one peer network",
        "panels_B_D": "Mathematical examples; no real target values used",
        "terminal_status": data["terminal_status"],
        "quality_or_speed_or_official_score_claim": None,
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "files": {p.name: {"bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths[:2]},
    }
    with paths[2].open("x") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
