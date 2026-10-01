#!/usr/bin/env python
"""
Publication figures for the GreenPEFT paper (workflow spec section 21).

Generates the two figures tied to the tables in results/benchmark/ and results/pareto/:

    Figure A  results/benchmark/figures/figA_feasibility_matrix.png
    Figure E  results/pareto/figures/figE_pareto_frontier.png

Figures B, C, D and F already exist elsewhere in the repo (see readme.md).

Design notes, since both figures are print targets rather than screens:

  Figure A encodes "successful seeds out of 3" as a SINGLE-HUE sequential ramp, not as a
  red/amber/green status triple. The data's job is ordered magnitude, and a red/green pair
  measures only Delta E 4.1 under deuteranopia -- the most common colour-vision deficiency --
  so the status framing would put the figure's whole message in a channel many readers cannot
  see. Every cell additionally prints its count, and all-OOM cells are hatched, so the value
  never rests on colour alone.

  Figure E uses colour for ONE distinction -- on the Pareto front or dominated -- with method
  identity carried by marker shape and direct labels. Five categorical hues in a scatter cannot
  clear the all-pairs colour-separation floor, and the figure's actual message is which
  configurations are non-dominated, so that is what colour encodes.

  Light mode only: these render into a paper on white. Colour values come from the project's
  validated palette, re-checked against a #ffffff surface.

Usage:
    python analysis/figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from greenpeft_data import BACKBONE_ORDER, METHOD_ORDER, ROOT, ordered  # noqa: E402

BENCH_DIR = ROOT / 'results' / 'benchmark'
PARETO_DIR = ROOT / 'results' / 'pareto'

# --- palette (validated against a #ffffff print surface) -------------------------
SURFACE = '#ffffff'
INK = '#0b0b0b'          # primary
INK_2 = '#52514e'        # secondary
MUTED = '#898781'        # axis / labels, and recessive marks
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
SERIES = '#2a78d6'       # categorical slot 1 -- the emphasised state

# Single-hue sequential steps for 0..3 successful seeds. 0 is the neutral gray
# midpoint because it means "did not run", an absence rather than a low value.
COUNT_COLORS = ['#f0efec', '#9ec5f4', '#5598e7', '#1c5cab']

# Marker shape carries method identity in Figure E, so identity survives greyscale printing.
METHOD_MARKERS = {'full_ft': 'o', 'lisa': 's', 'lora': '^', 'lora_fa': 'D', 'qlora': 'v'}


def _style(ax):
    """Recessive chrome: hairline grid, no top/right spines, muted tick labels."""
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=1.0)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_color(INK_2)


def figure_a() -> Path:
    """Method x backbone feasibility: successful seeds of 3 under the 15.6 GB budget."""
    grid = pd.read_csv(BENCH_DIR / 'feasibility_matrix.csv')

    methods = ordered(grid['method'], METHOD_ORDER)
    backbones = ordered(grid['backbone'], BACKBONE_ORDER)
    counts = (grid.pivot(index='method', columns='backbone', values='runs_successful')
              .reindex(index=methods, columns=backbones))
    attempted = (grid.pivot(index='method', columns='backbone', values='runs_attempted')
                 .reindex(index=methods, columns=backbones))

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    fig.patch.set_facecolor(SURFACE)

    cmap = ListedColormap(COUNT_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(counts.values, cmap=cmap, norm=norm, aspect='auto')

    for i, method in enumerate(methods):
        for j, backbone in enumerate(backbones):
            n_ok = int(counts.loc[method, backbone])
            n_try = int(attempted.loc[method, backbone])
            # Dark fills need light text; the two palest steps need dark text.
            ax.text(j, i, f'{n_ok}/{n_try}', ha='center', va='center',
                    fontsize=10, fontweight='medium',
                    color='#ffffff' if n_ok >= 2 else INK)
            if n_ok == 0:
                # Second channel for total failure, so it is not merely "pale".
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch='///', edgecolor=MUTED,
                                           linewidth=0.0, zorder=1))

    ax.set_xticks(range(len(backbones)))
    ax.set_xticklabels(backbones)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xticks([x - 0.5 for x in range(1, len(backbones))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(methods))], minor=True)
    # A 2px surface gap between adjacent cells, drawn as minor gridlines.
    ax.grid(which='minor', color=SURFACE, linewidth=2.0)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_color(INK_2)
    for side in ax.spines.values():
        side.set_visible(False)

    ax.set_xlabel('Backbone scale', color=INK_2, fontsize=10, labelpad=8)
    ax.set_title('Figure A  Feasibility under a 15.6 GB budget\n'
                 'successful seeds of 3 per (method, backbone); hatched = every seed OOM',
                 color=INK, fontsize=11, loc='left', pad=12)

    # The zero swatch carries the same hatch as the cells, so legend and plot agree.
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=GRID,
                             hatch='///' if i == 0 else None)
               for i, c in enumerate(COUNT_COLORS)]
    labels = ['0 (all OOM)', '1 of 3', '2 of 3', '3 of 3']
    leg = ax.legend(handles, labels, title='Successful seeds', frameon=False,
                    bbox_to_anchor=(1.01, 1.0), loc='upper left', fontsize=9,
                    title_fontsize=9)
    leg.get_title().set_color(INK_2)
    for text in leg.get_texts():
        text.set_color(INK_2)

    out = BENCH_DIR / 'figures' / 'figA_feasibility_matrix.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    return out


def figure_e() -> Path:
    """Empirical Pareto frontier over measured configurations: accuracy vs energy."""
    measured = pd.read_csv(PARETO_DIR / 'pareto_measured.csv')

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.set_axisbelow(True)
    ax.grid(axis='both', color=GRID, linewidth=0.8)

    # No connecting line. Non-dominance is computed over FOUR objectives (accuracy, VRAM,
    # carbon, time) while this figure projects two of them, so a point can be non-dominated
    # overall yet sit below-left of another here. Joining the highlighted points in energy
    # order produces a sawtooth that implies an ordering this projection cannot support.
    front = measured[measured['pareto_optimal']].sort_values('energy_kwh')

    for _, r in measured.iterrows():
        on_front = bool(r['pareto_optimal'])
        ax.scatter(r['energy_kwh'], r['accuracy'],
                   marker=METHOD_MARKERS.get(r['method'], 'o'),
                   s=130 if on_front else 70,
                   facecolor=SERIES if on_front else 'none',
                   edgecolor=SERIES if on_front else MUTED,
                   linewidth=1.8 if on_front else 1.2,
                   zorder=4 if on_front else 3)

    # Direct labels on the front only -- labelling all 14 would collide, and the dominated
    # points are context rather than the message. Offsets alternate so the dense low-energy
    # cluster does not stack its labels on top of each other.
    # All labels go to the RIGHT of their mark, alternating up/down. Left-side offsets ran
    # off the y-axis for the low-energy cluster, where two points nearly coincide.
    for i, (_, r) in enumerate(front.iterrows()):
        dy = 9 if i % 2 == 0 else -16
        ax.annotate(f'{r["method"]} / {r["backbone"]}',
                    (r['energy_kwh'], r['accuracy']),
                    textcoords='offset points', xytext=(12, dy),
                    ha='left', fontsize=8, color=INK_2)

    # Headroom so edge labels clear the axes and the method legend.
    xlo, xhi = measured['energy_kwh'].min(), measured['energy_kwh'].max()
    ax.set_xlim(xlo - (xhi - xlo) * 0.05, xhi + (xhi - xlo) * 0.22)
    ylo, yhi = measured['accuracy'].min(), measured['accuracy'].max()
    ax.set_ylim(ylo - (yhi - ylo) * 0.12, yhi + (yhi - ylo) * 0.10)

    ax.set_xlabel('Training energy (kWh, valid telemetry pass only)', color=INK_2, fontsize=10)
    ax.set_ylabel('Accuracy (mean over seeds)', color=INK_2, fontsize=10)
    ax.set_title('Figure E  Empirical Pareto frontier\n'
                 f'{len(front)} of {len(measured)} measured configurations are non-dominated '
                 'over accuracy, VRAM, carbon and time;\nthis view projects two of those four '
                 'axes, so a non-dominated point may appear dominated here',
                 color=INK, fontsize=11, loc='left', pad=12)

    shape_handles = [Line2D([], [], marker=METHOD_MARKERS[m], color='none',
                            markerfacecolor='none', markeredgecolor=INK_2,
                            markeredgewidth=1.3, markersize=8, label=m)
                     for m in METHOD_ORDER if m in set(measured['method'])]
    state_handles = [
        Line2D([], [], marker='o', color='none', markerfacecolor=SERIES,
               markeredgecolor=SERIES, markersize=9, label='on Pareto front'),
        Line2D([], [], marker='o', color='none', markerfacecolor='none',
               markeredgecolor=MUTED, markeredgewidth=1.2, markersize=8, label='dominated'),
    ]
    leg1 = ax.legend(handles=state_handles, frameon=False, loc='lower right',
                     fontsize=9, title='Status', title_fontsize=9)
    leg1.get_title().set_color(INK_2)
    for t in leg1.get_texts():
        t.set_color(INK_2)
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=shape_handles, frameon=False, bbox_to_anchor=(1.01, 1.0),
                     loc='upper left', fontsize=9, title='Method', title_fontsize=9)
    leg2.get_title().set_color(INK_2)
    for t in leg2.get_texts():
        t.set_color(INK_2)

    out = PARETO_DIR / 'figures' / 'figE_pareto_frontier.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> int:
    missing = [p for p in (BENCH_DIR / 'feasibility_matrix.csv',
                           PARETO_DIR / 'pareto_measured.csv') if not p.exists()]
    if missing:
        print('error: run analysis/benchmark_analysis.py and analysis/pareto_analysis.py first; '
              f'missing {[str(p.relative_to(ROOT)) for p in missing]}', file=sys.stderr)
        return 2

    for path in (figure_a(), figure_e()):
        print(f'wrote {path.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
