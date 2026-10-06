"""Render an original illustrative SVG from synthetic receipt metrics only."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path


HELPER_SHA256 = "a47e8dfa175706b61946b3c21c819711c7047e682cd9e6c97e022d63e8a64dbc"
VERIFIER_SHA256 = "c8cdd272492482e9ef7290fc97cde85c424865a85aa9748b9ba34f3283bcd221"


def render(receipt):
    if receipt["status"] != "PASS_SYNTHETIC_GEOMETRY_ONLY" or receipt["parameter_combination_count"] != 81:
        raise ValueError("expected the reviewed 81-point synthetic receipt")
    if receipt["helper_sha256"] != HELPER_SHA256 or receipt["verifier_sha256"] != VERIFIER_SHA256:
        raise ValueError("synthetic receipt source hashes drifted")
    if receipt["real_ids_images_targets_models_used"] is not False or receipt["runtime"]["device"] != "cpu":
        raise ValueError("the diagram requires synthetic CPU evidence")
    center = receipt["source_center_outside_fraction_max_81_points"]
    missing = receipt["mean_missing_bilinear_mass_max_81_points"]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (center, missing)):
        raise ValueError("padding metrics must be finite real fractions")
    if not 0.0 <= missing <= center <= 0.10:
        raise ValueError("padding metrics violate the frozen diagram bounds")
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="980" viewBox="0 0 1440 980" role="img" aria-labelledby="title desc">',
        '<title id="title">A single mild affine grid shared across a study</title>',
        '<desc id="desc">Invented study UID, epoch and seed select four dedicated PCG64 draws. One inverse grid is reused across twelve image windows and three channels per window, then sampled with bilinear zero padding before normalization. Synthetic CPU measurements distinguish center-outside fraction and missing bilinear mass.</desc>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="#1d817a"/></marker></defs>',
        '<rect width="1440" height="980" fill="#f4f7f7"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#193546} .caption{fill:#516a77} .small{font-size:19px} .body{font-size:24px} .label{font-size:22px;font-weight:700} .card{fill:#fff;stroke:#d5e2e5;stroke-width:2} .link{fill:none;stroke:#1d817a;stroke-width:3;marker-end:url(#arrow)}</style>',
    ]

    def text(x, y, value, cls="body", extra=""):
        elements.append(f'<text x="{x}" y="{y}" class="{cls}" {extra}>{html.escape(str(value))}</text>')

    def card(x, y, w, h):
        elements.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" class="card"/>')

    text(48, 64, "One study. One transform.", extra='style="font-size:44px;font-weight:700"')
    text(48, 105, "Deterministic in-plane geometry for raw-image windows", "caption body")
    text(1392, 64, "SYNTHETIC · CPU CHECKS", "caption small", 'text-anchor="end"')

    for x in (48, 516, 984):
        card(x, 143, 408, 142)
    text(70, 181, "1   Stable study key", "label")
    text(70, 221, "exact UID + epoch + seed", "body")
    text(70, 255, "SHA256 → owned PCG64", "caption small")
    text(538, 181, "2   Four small draws", "label")
    text(538, 221, "±5°   ·   ±2% x/y   ·   .98–1.02", "body")
    text(538, 255, "Independent of training RNG", "caption small")
    text(1006, 181, "3   One inverse grid", "label")
    text(1006, 221, "Same map for every plane", "body")
    text(1006, 255, "Shared 2D coordinates · no 3D rotation", "caption small")
    elements += ['<path d="M462,214 H502" class="link"/>', '<path d="M930,214 H970" class="link"/>']

    card(48, 330, 600, 337)
    card(792, 330, 600, 337)
    text(70, 370, "Raw study windows", "label")
    text(70, 403, "12 windows × 3 adjacent channels", "caption small")
    text(814, 370, "Coherent augmented study", "label")
    text(814, 403, "One grid reused across all 36 planes", "caption small")

    # Twelve invented windows. The green wireframe is schematic rather than a
    # medical image or a claim about the measured appearance of real anatomy.
    for offset, warped in ((0, False), (744, True)):
        for i in range(12):
            x, y = 76 + offset + (i % 4) * 133, 430 + (i // 4) * 67
            elements.append(f'<rect x="{x}" y="{y}" width="108" height="49" rx="5" fill="#e8f0f2" stroke="#afc4cc"/>')
            transform = f'translate({x+54},{y+24.5}) rotate(5) scale(.98) translate(-54,-24.5)' if warped else f'translate({x},{y})'
            elements.append(f'<g transform="{transform}" fill="none" stroke="{("#19897f" if warped else "#466d86")}" stroke-width="1.6">')
            for u in (20, 42, 64, 86):
                elements.append(f'<path d="M{u},7 V42"/>')
            for v in (11, 24, 37):
                elements.append(f'<path d="M12,{v} H96"/>')
            elements.append('<circle cx="66" cy="17" r="4" fill="#c89330" stroke="none"/></g>')
    elements.append('<path d="M672,500 H768" class="link"/>')
    text(720, 452, "shared", "caption small", 'text-anchor="middle"')
    text(720, 477, "grid", "caption small", 'text-anchor="middle"')
    text(70, 643, "Finite raw values in [0,1]", "caption small")
    text(814, 643, "Normalize channels afterward", "caption small")

    for x in (48, 516, 984):
        card(x, 707, 408, 178)
    text(70, 747, "Interpolation", "label")
    text(70, 788, "Bilinear · zero padding", "body")
    text(70, 824, "align_corners=False", "caption small")
    text(70, 860, "Float32/64; autocast disabled locally", "caption small")
    text(538, 747, "Actual-grid hold gate", "label")
    text(538, 790, "Center-outside fraction ≤ 10%", "body")
    text(538, 828, "Pixel-center bounds govern the gate", "caption small")
    text(538, 860, "No clipping or recovery fallback", "caption small")
    text(1006, 747, "81 configurations screened", "label")
    text(1006, 791, f"{100*center:.2f}% center outside", "body")
    text(1006, 829, f"{100*missing:.2f}% missing bilinear mass", "body")
    text(1006, 860, "Different metrics; finite screen only", "caption small")
    text(48, 936, "Illustration only. Synthetic checks establish geometry and RNG behavior, not anatomical safety or a score gain.", "caption small")
    elements.append('</svg>')
    return "\n".join(elements) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new .svg path; existing files are preserved")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".svg":
        raise ValueError("only a new SVG output is supported")
    folder = Path(__file__).resolve().parent
    if hashlib.sha256((folder / "mild_affine.py").read_bytes()).hexdigest() != HELPER_SHA256:
        raise ValueError("reviewed helper hash drift")
    if hashlib.sha256((folder / "verify_geometry.py").read_bytes()).hexdigest() != VERIFIER_SHA256:
        raise ValueError("reviewed verifier hash drift")
    receipt = json.loads((folder / "verification_cpu_normal.json").read_text())
    if receipt["helper_sha256"] != HELPER_SHA256:
        raise ValueError("synthetic receipt/helper mismatch")
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(render(receipt))


if __name__ == "__main__":
    main()
