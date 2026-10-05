#!/usr/bin/env python3
"""Render a descriptive aggregate AUC figure; never fit or evaluate a model.

Original code, Copyright (c) 2026 cubres. MIT licensed; see LICENSE.
Usage: python render_proxy_auc_delta.py --input aggregate_proxy_auc.json \
    --output-dir new_figure_directory
Requires Python 3.9+ and Matplotlib. Output directory must not already exist.
"""

import argparse
import hashlib
import io
import json
import math
import pathlib
import platform
import sys

SCHEMA = "knee-v55-exposed-report-proxy-auc-v1"
OUTPUT_NAMES = ("knee_v55_proxy_auc_delta.png", "knee_v55_proxy_auc_delta.svg")


def unique_object(pairs):
    """Reject duplicate JSON keys rather than silently accepting the last one."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def finite_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(field + " must be a finite number")
    if not math.isfinite(value):
        raise ValueError(field + " must be finite")
    return float(value)


def validate(data):
    """Validate the aggregate fixture and its explicitly restricted scope."""
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("Unsupported aggregate schema")
    scope = data.get("scope", {})
    expected_scope = {
        "studies": 128,
        "previously_exposed_panel": True,
        "report_overlap": "UNKNOWN",
        "patient_independence_proven": False,
        "official_kaggle_score": None,
        "automatic_promotion": False,
        "error_bars": False,
        "significance_claim": False,
    }
    if scope != expected_scope:
        raise ValueError("Scope must retain the exposed local-proxy restrictions")
    design = data.get("matched_training_design", {})
    if design != {"models_per_arm": 2, "epochs_per_network": 4,
                  "updates_per_network": 3488}:
        raise ValueError("Expected matched 2-model, 4-epoch, 3488-update design")
    rows = data.get("per_finding", [])
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("Exactly 12 aggregate finding rows are required")
    names = set()
    deltas = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
                "finding", "ordinary_auc", "peer_selection_auc", "delta_auc"}:
            raise ValueError("Each row must contain only aggregate AUC fields")
        name = row["finding"]
        if not isinstance(name, str) or not name or len(name) > 32 or name in names:
            raise ValueError("Finding names must be unique, short, nonempty strings")
        names.add(name)
        ordinary = finite_number(row["ordinary_auc"], "ordinary_auc")
        peer = finite_number(row["peer_selection_auc"], "peer_selection_auc")
        delta = finite_number(row["delta_auc"], "delta_auc")
        if not (0.0 <= ordinary <= 1.0 and 0.0 <= peer <= 1.0):
            raise ValueError("AUC must lie in [0, 1]")
        if not math.isclose(peer - ordinary, delta, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Row delta must equal peer-selection minus ordinary AUC")
        deltas.append(delta)
    macro = data.get("macro_auc", {})
    if set(macro) != {"ordinary", "peer_selection", "delta"}:
        raise ValueError("Missing macro AUC fields")
    ordinary = finite_number(macro["ordinary"], "macro ordinary AUC")
    peer = finite_number(macro["peer_selection"], "macro peer-selection AUC")
    delta = finite_number(macro["delta"], "macro delta")
    if not (0.0 <= ordinary <= 1.0 and 0.0 <= peer <= 1.0):
        raise ValueError("Macro AUC must lie in [0, 1]")
    checks = ((peer - ordinary, delta),
              (math.fsum(deltas) / 12, delta),
              (math.fsum(row["ordinary_auc"] for row in rows) / 12, ordinary),
              (math.fsum(row["peer_selection_auc"] for row in rows) / 12, peer))
    if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in checks):
        raise ValueError("Macro AUC must equal the unweighted mean of all 12 findings")
    gains = sum(value > 0.0 for value in deltas)
    losses = sum(value < 0.0 for value in deltas)
    if data.get("direction_counts") != {"gains": gains, "losses": losses, "ties": 12-gains-losses}:
        raise ValueError("Direction counts do not match the signed deltas")
    return rows, gains, losses


def make_figure(data, input_sha256):
    # Matplotlib is the only optional dependency. No input code is imported.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    rows, gains, losses = validate(data)
    rows = sorted(rows, key=lambda row: (-row["delta_auc"], row["finding"]))
    bg, ink, muted = "#FCFCFA", "#152D3A", "#53646F"
    gain, loss, grid = "#087E86", "#C76643", "#E1E7E7"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.unicode_minus": True, "svg.fonttype": "none",
        "svg.hashsalt": input_sha256,
    })
    fig = plt.figure(figsize=(15.8, 10.8), facecolor=bg)
    fig.text(0.065, 0.952, "Knee V55 · Peer selection vs ordinary training",
             color=ink, fontsize=23, weight="bold", va="top")
    fig.text(0.065, 0.908, "12 findings on a previously exposed 128-study report-proxy panel",
             color=muted, fontsize=14, va="top")
    fig.text(0.065, 0.865, "LOCAL PROXY  ·  NO KAGGLE LEADERBOARD RESULT  ·  NO PROMOTION",
             color="#815928", fontsize=11.2, weight="bold",
             bbox={"facecolor": "#F5EDDC", "edgecolor": "none", "boxstyle": "round,pad=0.55"})

    for x, width in ((0.065, 0.31), (0.393, 0.225), (0.636, 0.299)):
        fig.add_artist(FancyBboxPatch((x, 0.761), width, 0.073,
                       boxstyle="round,pad=0.008,rounding_size=0.007",
                       linewidth=0.8, edgecolor=grid, facecolor="#F3F6F4",
                       transform=fig.transFigure))
    fig.text(0.081, 0.816, "MACRO Δ AUC", color=muted, fontsize=10.3, weight="bold", va="top")
    fig.text(0.081, 0.790, "{:+.10f}".format(data["macro_auc"]["delta"]),
             color=ink, fontsize=21, weight="bold", va="top")
    fig.text(0.409, 0.816, "FINDING DIRECTIONS", color=muted, fontsize=10.3, weight="bold", va="top")
    fig.text(0.409, 0.788, "{} gains / {} losses".format(gains, losses),
             color=ink, fontsize=17, weight="bold", va="top")
    fig.text(0.652, 0.816, "MATCHED TRAINING BUDGET", color=muted, fontsize=10.3, weight="bold", va="top")
    fig.text(0.652, 0.787, "2-model ensembles · 4 epochs", color=ink, fontsize=13, weight="bold", va="top")
    fig.text(0.652, 0.765, "3,488 updates per network", color=muted, fontsize=10.8, va="top")

    ax = fig.add_axes([0.19, 0.292, 0.56, 0.411], facecolor=bg)
    ax.set_ylim(-0.7, 11.7)
    ax.set_xlim(-0.0087, 0.0194)
    for index in range(12):
        if index % 2 == 0:
            ax.axhspan(index-0.5, index+0.5, color="#F1F4F2", zorder=0)
    ax.xaxis.set_major_locator(MultipleLocator(0.005))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "{:+.3f}".format(value) if value else "0.000"))
    ax.grid(axis="x", color=grid, linewidth=0.8, zorder=1)
    ax.axvline(0.0, color=ink, linewidth=1.3, zorder=3)
    ax.axvline(data["macro_auc"]["delta"], color="#70818A", linewidth=1.0,
               linestyle=(0, (3, 3)), zorder=2)
    positions = list(reversed(range(12)))
    for y, row in zip(positions, rows):
        delta = row["delta_auc"]
        color = gain if delta > 0.0 else loss
        ax.barh(y, delta, height=0.29, color=color, alpha=0.88, zorder=4)
        ax.scatter([delta], [y], s=35, color=color, edgecolor=bg, linewidth=0.6, zorder=5)
        ax.text(delta + (0.00032 if delta >= 0.0 else -0.00032), y,
                "{:+.7f}".format(delta), ha="left" if delta >= 0.0 else "right",
                va="center", fontsize=10.8, color=ink)
    ax.set_yticks(positions, [row["finding"] for row in rows], fontsize=11.5, color=ink)
    ax.tick_params(axis="y", length=0, pad=11)
    ax.tick_params(axis="x", length=0, pad=9, labelcolor=muted)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Δ report-proxy AUC (peer-selection − ordinary) · raw AUC units",
                  color=ink, fontsize=11.2, labelpad=16)
    fig.text(0.19, 0.728, "Signed change by finding", color=ink, fontsize=12.2, weight="bold")
    fig.text(0.19, 0.709, "Sorted by Δ AUC; positive values favor peer selection", color=muted, fontsize=10)

    table = fig.add_axes([0.799, 0.292, 0.148, 0.411], facecolor=bg)
    table.set_ylim(-0.7, 11.7)
    table.set_xlim(0.0, 1.0)
    table.axis("off")
    fig.text(0.799, 0.729, "Report-proxy AUC", color=ink, fontsize=12.2, weight="bold")
    fig.text(0.799, 0.708, "Ordinary", color=muted, fontsize=9.8)
    fig.text(0.880, 0.708, "Peer selection", color=muted, fontsize=9.8)
    for y, row in zip(positions, rows):
        table.text(0.01, y, "{:.6f}".format(row["ordinary_auc"]),
                   color=muted, fontsize=10.7, va="center")
        table.text(0.55, y, "{:.6f}".format(row["peer_selection_auc"]),
                   color=ink, fontsize=10.7, va="center")

    legend = [Line2D([], [], color=gain, marker="o", linewidth=3, label="Higher proxy AUC"),
              Line2D([], [], color=loss, marker="o", linewidth=3, label="Lower proxy AUC"),
              Line2D([], [], color="#70818A", linewidth=1.0, linestyle=(0, (3, 3)), label="Macro mean Δ")]
    fig.legend(handles=legend, loc="center left", bbox_to_anchor=(0.19, 0.222),
               ncol=3, frameon=False, fontsize=10.5, handlelength=2.4, columnspacing=2.4)
    fig.add_artist(Line2D([0.065, 0.947], [0.188, 0.188], color=grid, linewidth=1.0,
                         transform=fig.transFigure))
    fig.text(0.065, 0.162, "DESCRIPTIVE COMPARISON ONLY", color=ink, fontsize=10.6, weight="bold")
    fig.text(0.065, 0.136, "Previously exposed held fold. Report overlap: UNKNOWN. Patient independence: unproven.",
             color=muted, fontsize=11.2)
    fig.text(0.065, 0.110, "No error bars or significance claim. This local report-proxy result does not establish a Kaggle score gain.",
             color=muted, fontsize=11.2)
    fig.text(0.065, 0.084, "Unweighted macro AUC: ordinary {:.10f} → peer selection {:.10f}. No automatic promotion.".format(
             data["macro_auc"]["ordinary"], data["macro_auc"]["peer_selection"]),
             color=muted, fontsize=10.6)
    fig.text(0.065, 0.048, "Values are rounded for display; the portable JSON preserves full precision. Aggregate data only · cubres · Knee V55",
             color="#74818A", fontsize=9.4)
    return fig, matplotlib.__version__


def render(input_path, output_dir):
    input_path = pathlib.Path(input_path)
    output_dir = pathlib.Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Refusing to overwrite an existing output directory: " + str(output_dir))
    if not output_dir.parent.is_dir():
        raise ValueError("Output parent must already exist; no parent directories are created")
    if input_path.stat().st_size > 1_000_000:
        raise ValueError("Aggregate input exceeds the 1 MB limit")
    payload = input_path.read_bytes()
    data = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object)
    rows, gains, losses = validate(data)
    digest = hashlib.sha256(payload).hexdigest()
    fig, matplotlib_version = make_figure(data, digest)
    outputs = {}
    try:
        for name in OUTPUT_NAMES:
            stream = io.BytesIO()
            if name.endswith(".png"):
                fig.savefig(stream, format="png", dpi=180, facecolor=fig.get_facecolor(),
                            metadata={"Software": "cubres aggregate proxy AUC renderer"})
            else:
                fig.savefig(stream, format="svg", facecolor=fig.get_facecolor(),
                            metadata={"Date": None, "Creator": "cubres aggregate proxy AUC renderer"})
            outputs[name] = stream.getvalue()
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    # Validate and render in memory first. Exclusive mkdir and file creation
    # prevent accidental replacement. No source file or existing directory is edited.
    output_dir.mkdir(parents=False, exist_ok=False)
    files = {}
    for name, content in outputs.items():
        with (output_dir / name).open("xb") as handle:
            handle.write(content)
        files[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    receipt = {
        "status": "PASS_AGGREGATE_VALIDATION_AND_RENDER_ONLY",
        "schema": SCHEMA,
        "input_sha256": digest,
        "row_count": len(rows), "gains": gains, "losses": losses,
        "macro_delta_auc": data["macro_auc"]["delta"],
        "scope": data["scope"], "matched_training_design": data["matched_training_design"],
        "display_order": "descending delta_auc, then finding name",
        "python_version": platform.python_version(),
        "matplotlib_version": matplotlib_version,
        "files": files,
        "models_executed": False, "private_data_read": False,
        "source_receipts_required_at_render_time": False,
    }
    with (output_dir / "render_receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Portable aggregate JSON fixture")
    parser.add_argument("--output-dir", required=True, help="New, nonexistent output directory")
    args = parser.parse_args()
    try:
        receipt = render(args.input, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, "Render refused: " + str(error) + "\n")
    print(json.dumps({"status": receipt["status"], "files": sorted(receipt["files"]),
                      "input_sha256": receipt["input_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
