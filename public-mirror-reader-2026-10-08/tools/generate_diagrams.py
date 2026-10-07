"""Generate original explanatory figures; no MRI, model, observations or scores.

Copyright (c) 2026 cubres. MIT license; see ../LICENSES/MIT.txt.
Run once per fresh figure directory, preserving existing exports.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
CREAM = "#f8f5ed"
INK = "#203142"
TEAL = "#247e87"
COPPER = "#b66d48"
BLUE = "#506e9a"
COLORS = (TEAL, "#62a3a4", BLUE, "#90a2ba", COPPER)


def canvas(height):
    figure, axes = plt.subplots(figsize=(16, height), dpi=150)
    figure.patch.set_facecolor(CREAM)
    axes.set_facecolor(CREAM)
    axes.set_xlim(0, 16)
    axes.set_ylim(0, height)
    axes.axis("off")
    return figure, axes


def text(axes, x, y, content, size=14, color=INK, **kwargs):
    axes.text(x, y, content, fontsize=size, color=color, fontfamily="DejaVu Sans", **kwargs)


def box(axes, x, y, width, height, title, detail, edge=TEAL):
    axes.add_patch(FancyBboxPatch((x, y), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.12", linewidth=1.4,
        edgecolor=edge, facecolor="#fffdf8"))
    text(axes, x + width / 2, y + height - .28, title, 12.5, ha="center", va="top", weight="bold")
    text(axes, x + width / 2, y + height / 2 - .17, detail, 10.75, ha="center", va="center", linespacing=1.6)


def arrow(axes, start, stop, color=INK):
    axes.add_patch(FancyArrowPatch(start, stop, arrowstyle="-|>",
        mutation_scale=14, linewidth=1.5, color=color, connectionstyle="arc3"))


def export(figure, stem):
    FIGURES.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        destination = FIGURES / (stem + "." + extension)
        if destination.exists():
            raise RuntimeError("Preserve existing figure and choose a fresh export directory: " + str(destination))
        figure.savefig(destination, dpi=150, bbox_inches="tight", facecolor=CREAM)
    plt.close(figure)


def pipeline():
    figure, axes = canvas(7.7)
    text(axes, .25, 7.2, "One study. Two anatomical views. One finding vector.", 25, weight="bold")
    text(axes, .25, 6.65, "An explanatory diagram of the source-reviewed recipe — not an MRI, result or accuracy measurement.", 12)
    box(axes, .3, 3.25, 3.05, 2.05, "Build the MRI volume", "140 mm physical crop\n384 × 384 uint8 slices\n96 slices · 5 acquisition slots")
    x = .48
    for count, color in zip((26, 22, 18, 12, 18), COLORS):
        width = 2.7 * count / 96
        axes.add_patch(Rectangle((x, 3.48), width - .02, .15, facecolor=color, edgecolor="none"))
        x += width
    box(axes, 3.9, 3.25, 2.9, 2.05, "Select 94 triplets", "Adjacent slices at each center\n32 px border crop → 320\nCenter sets the mirror axis")
    arrow(axes, (3.4, 4.25), (3.85, 4.25))
    box(axes, 7.4, 4.58, 3.05, 1.36, "Plain view", "Channel normalization", edge=BLUE)
    box(axes, 7.4, 2.65, 3.05, 1.36, "Anatomical mirror", "Mirror uint8, then normalize", edge=COPPER)
    arrow(axes, (6.85, 4.28), (7.35, 5.20), BLUE)
    arrow(axes, (6.85, 4.22), (7.35, 3.35), COPPER)
    box(axes, 11.12, 4.58, 4.15, 1.36, "CoAtNet + finding attention", "94 windows → 12 logits → sigmoid", edge=BLUE)
    box(axes, 11.12, 2.65, 4.15, 1.36, "Same checkpoint + attention", "94 windows → 12 logits → sigmoid", edge=COPPER)
    arrow(axes, (10.5, 5.20), (11.07, 5.20), BLUE)
    arrow(axes, (10.5, 3.32), (11.07, 3.32), COPPER)
    box(axes, 7.6, .65, 3.45, 1.10, "Mean probabilities", "(plain + mirror) / 2")
    box(axes, 11.65, .65, 3.62, 1.10, "Ordinal ranks", "Per finding, across all studies")
    arrow(axes, (13.05, 2.60), (10.4, 1.80))
    arrow(axes, (15.35, 5.18), (15.65, 5.18))
    arrow(axes, (15.65, 5.18), (15.65, 2.20))
    arrow(axes, (15.65, 2.20), (10.70, 1.75))
    arrow(axes, (11.1, 1.17), (11.60, 1.17))
    text(axes, .3, 1.40, "SOURCE REVIEW PASSED", 12, color=TEAL, weight="bold")
    text(axes, .3, .98, "Native pixel / CUDA equivalence: UNRUN", 12)
    text(axes, .3, .60, "No new official score claimed", 12)
    export(figure, "anatomical_mirror_pipeline_v2")


def mirror_order():
    figure, axes = canvas(5.8)
    text(axes, .25, 5.28, "Reflect the acquisition before normalizing the channels", 24, weight="bold")
    text(axes, .25, 4.77, "The triplet center chooses the operation; the three ImageNet normalization pairs stay channel-specific.", 12)
    rows = ((3.25, "SAGITTAL CENTER < 48", ("k − 1", "k", "k + 1"), ("k + 1", "k", "k − 1"), COPPER),
            (1.35, "CORONAL / AXIAL CENTER ≥ 48", ("← image width →",), ("→ width reversed ←",), BLUE))
    for y, title, before, after, color in rows:
        text(axes, .35, y + .82, title, 13, color=color, weight="bold")
        for x, content in ((.35, before), (5.70, after)):
            unit = 4.0 / len(content)
            for i, label in enumerate(content):
                axes.add_patch(FancyBboxPatch((x + i * unit, y), unit - .06, .67,
                    boxstyle="round,pad=0.02,rounding_size=.06", edgecolor=color,
                    facecolor=("#ecdfce", "#d9e4e9", "#d7e7df")[i % 3], linewidth=1.2))
                text(axes, x + (i + .5) * unit, y + .335, label, 12, ha="center", va="center")
        arrow(axes, (4.50, y + .335), (5.60, y + .335), color)
        arrow(axes, (9.85, y + .335), (10.6, y + .335), color)
    box(axes, 10.70, 1.05, 4.5, 3.1, "Normalize each channel", "mean: 0.485, 0.456, 0.406\nstd:     0.229, 0.224, 0.225\n\nThen run the same trained reader")
    text(axes, .35, .50, "Schematic only. Channel reversal after normalization changes this recipe because channel statistics differ.", 12)
    export(figure, "mirror_before_normalization_v2")


if __name__ == "__main__":
    pipeline()
    mirror_order()
