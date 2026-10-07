"""Reproduce original protocol and measured progress figures from this packet.

Writes only new output files; synthetic illustration and timestamped log progress
are clearly separated from measured model quality. No native data/model reads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


INK = '#17283c'
MUTED = '#586e83'
BLUE = '#126eab'
ORANGE = '#bd601c'
BG = '#f7f9fc'
ROOT = Path(__file__).resolve().parent.parent


def box(ax, x, y, width, height, title, body, color=BLUE, fill='white'):
    patch = FancyBboxPatch((x, y), width, height,
                          boxstyle='round,pad=0.014,rounding_size=0.018',
                          facecolor=fill, edgecolor=color, linewidth=1.5)
    ax.add_patch(patch)
    ax.text(x + width/2, y + height*0.72, title, ha='center', va='center',
            fontsize=12.4, color=color, fontweight='bold')
    ax.text(x + width/2, y + height*0.35, body, ha='center', va='center',
            fontsize=10.7, color=INK, linespacing=1.55)


def arrow(ax, a, b, color=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=17,
                                color=color, linewidth=1.6,
                                connectionstyle='arc3,rad=0'))


def save_new(fig, path, fmt):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        fig.savefig(handle, format=fmt, dpi=150, bbox_inches='tight', pad_inches=.18,
                    facecolor=fig.get_facecolor(),
                    metadata={'Date': None} if fmt == 'svg' else None)


def protocol_figure():
    fig, ax = plt.subplots(figsize=(15, 9), layout='constrained')
    fig.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(.04, .96, 'A small treatment, a matched comparison',
            fontsize=22, color=INK, fontweight='bold')
    ax.text(.04, .915,
            'Registered native protocol • model quality is still pending',
            fontsize=12.5, color=MUTED)
    box(ax, .27, .755, .46, .11, 'Shared original initialization',
        'Two networks per arm • fixed fit order • 3,487 fit studies', BLUE, '#edf5fb')
    arrow(ax, (.39, .741), (.26, .69), BLUE)
    arrow(ax, (.61, .741), (.74, .69), ORANGE)
    box(ax, .065, .475, .39, .19, 'PLAIN BCE  /  GPU 0',
        'Unaugmented raw windows → normalization\n4 epochs • 3,488 AdamW updates per network\nTwo networks; matched initial states', BLUE, '#edf5fb')
    box(ax, .545, .475, .39, .19, 'MILD AFFINE BCE  /  GPU 1',
        'One coherent 2D transform → normalization\n±5° rotation • ±2% shift • 0.98–1.02 scale\nSame 4 epochs and 3,488 updates per network', ORANGE, '#fff4e9')
    arrow(ax, (.26, .460), (.39, .41), BLUE)
    arrow(ax, (.74, .460), (.61, .41), ORANGE)
    box(ax, .21, .265, .58, .125, 'Seal predictions before held-target numerics',
        'Both complete fits + 3 ordered 862 × 12 prediction files\nPlain • mild affine • zero-update reference', '#32685c', '#eef8f4')
    arrow(ax, (.50, .25), (.50, .205), '#32685c')
    box(ax, .18, .055, .64, .13, 'Then compare against the private report proxy',
        'Affine − plain macro AUC • paired whole-study resampling\nExposed development panel; no official-score or clinical claim', '#6c528c', '#f4f0fa')
    ax.text(.04, .012, 'Illustration of the planned gates. Snapshot V59: both arms in epoch 1; results unverified.',
            fontsize=10.2, color=MUTED)
    return fig


def progress_figure(evidence):
    events = evidence['v59']['training_progress_events']
    fig, ax = plt.subplots(figsize=(11.5, 6), layout='constrained')
    fig.set_facecolor(BG); ax.set_facecolor('white')
    for arm, label, color, marker in [
            ('plain_bce', 'Plain BCE', BLUE, 'o'),
            ('mild_affine_bce', 'Mild affine BCE', ORANGE, 'x')]:
        selected = [event for event in events if event['arm'] == arm]
        ax.plot([e['log_time_seconds']/60 for e in selected],
                [e['updates_per_network'] for e in selected],
                marker=marker, color=color, linewidth=2.2,
                markersize=9, label=label, alpha=.8)
    ax.set_title('Measured partial training progress', fontsize=18,
                 color=INK, fontweight='bold', loc='left', pad=18)
    ax.set_xlabel('Kaggle log event time (minutes)', fontsize=12, color=INK)
    ax.set_ylabel('Logged optimizer updates per network', fontsize=12, color=INK)
    ax.set_xlim(3.15, 8.25); ax.set_ylim(40, 216)
    ax.set_yticks([64, 128, 192])
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper left', frameon=False)
    fig.text(.03, -.10,
             'V59 / session 356180297 • observed 2026-10-07 21:09:51 UTC • epoch 1 in both arms\n'
             'The traces nearly coincide. These counts are log evidence, not final step-state seals or a quality metric.',
             fontsize=10.2, color=MUTED)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='A new figure directory; existing files are preserved.')
    parser.add_argument('--png-preview-dir', type=Path)
    args = parser.parse_args()
    evidence = json.loads((ROOT / 'EVIDENCE.json').read_text())
    for name, fig in [('paired-native-protocol', protocol_figure()),
                      ('observed-training-progress', progress_figure(evidence))]:
        save_new(fig, args.output_dir / (name + '.svg'), 'svg')
        if args.png_preview_dir:
            save_new(fig, args.png_preview_dir / (name + '.png'), 'png')
        plt.close(fig)
