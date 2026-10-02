"""Component 2 -- single-target figure: validation accuracy vs peak GPU memory.

A standalone, single-column figure for the memory-constrained story: where each configuration
sits against the 15.6 GB device budget, and which configurations are non-dominated on
(minimise memory, maximise accuracy).

Publication settings: 300 DPI, 10 pt base font, sns.despine(), Okabe-Ito colorblind-safe
palette, method carried by marker shape so the figure survives grayscale printing.

    python paper/scripts/fig_memory_pareto.py            # real measurements
    python paper/scripts/fig_memory_pareto.py --demo     # watermarked synthetic data

Reads paper/data/table2_complete_cells.csv (the 13 cells complete at all three seeds) and
writes paper/figures/fig8_memory_pareto.{pdf,png}. Error bars are the across-seed SDs.

Optional: `pip install adjustText` for overlap-free labels; falls back to fixed offsets.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

try:
    from adjustText import adjust_text
    HAVE_ADJUSTTEXT = True
except ImportError:
    HAVE_ADJUSTTEXT = False

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'paper' / 'data' / 'table2_complete_cells.csv'
FIG = ROOT / 'paper' / 'figures'

SIZE_OF = {'tiny': '0.5B', 'small': '1.1B', 'medium': '1.5B', 'large': '3.0B'}
SIZE_ORDER = ['0.5B', '1.1B', '1.5B', '3.0B']
OKABE_ITO = {'0.5B': '#0072B2', '1.1B': '#009E73', '1.5B': '#E69F00', '3.0B': '#D55E00'}
MARKER_OF = {'lora': 'o', 'qlora': '^', 'lora_fa': 'D', 'lisa': 'v', 'full_ft': 's'}
LABEL_OF = {'lora': 'LoRA', 'qlora': 'QLoRA', 'lora_fa': 'LoRA-FA',
            'lisa': 'LISA', 'full_ft': 'Full FT'}
METHOD_ORDER = ['lora', 'qlora', 'lora_fa', 'lisa', 'full_ft']

VRAM_BUDGET_GB = 15.6


def style():
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 10,
        'axes.titlesize': 10, 'legend.fontsize': 8, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
        'savefig.dpi': 300, 'figure.dpi': 300,
    })


def load_real():
    df = pd.read_csv(DATA)
    df = df[df['accuracy_count'] == 3].copy()
    df['size'] = df['backbone'].map(SIZE_OF)
    df['acc'] = df['accuracy_mean'] * 100
    df['acc_sd'] = df['accuracy_std'] * 100
    return df.rename(columns={'peak_gpu_memory_gb_mean': 'vram',
                              'peak_gpu_memory_gb_std': 'vram_sd'})


def load_demo():
    rng = np.random.default_rng(1)
    rows = []
    for size, scale in zip(SIZE_ORDER, [0.5, 1.1, 1.5, 3.0]):
        for method in METHOD_ORDER:
            if size != '0.5B' and method == 'full_ft':
                continue
            rows.append({'size': size, 'method': method,
                         'acc': 88 + 4 * np.log1p(scale) + rng.normal(0, 0.6),
                         'acc_sd': abs(rng.normal(0.4, 0.2)),
                         'vram': 2.5 * scale + rng.normal(0, 1.0) + 2.0,
                         'vram_sd': abs(rng.normal(0.4, 0.2))})
    return pd.DataFrame(rows)


def pareto_front(x, y):
    """Non-dominated indices for (minimise x, maximise y), returned sorted by x."""
    order = np.argsort(x)
    front, best = [], -np.inf
    for i in order:
        if y[i] > best:
            front.append(i)
            best = y[i]
    return np.array(front)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', action='store_true')
    args = ap.parse_args()

    style()
    df = (load_demo() if args.demo else load_real()).reset_index(drop=True)

    # 3.5 in is one IEEE column; 3.0 in tall keeps the aspect close to square.
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    for _, r in df.iterrows():
        ax.errorbar(r['vram'], r['acc'], xerr=r['vram_sd'], yerr=r['acc_sd'],
                    fmt=MARKER_OF[r['method']], ms=5.5, mfc=OKABE_ITO[r['size']],
                    mec='#222222', mew=0.6, ecolor='#999999', elinewidth=0.8,
                    capsize=1.8, zorder=3)

    ax.set_xlabel('Peak GPU memory (GB)')
    ax.set_ylabel('Validation accuracy (%)')
    ax.set_xlim(right=VRAM_BUDGET_GB * 1.08)
    ax.margins(y=0.22)

    # ---- Pareto staircase ---------------------------------------------------------------
    f = pareto_front(df['vram'].values.astype(float), df['acc'].values.astype(float))
    fx, fy = df['vram'].values[f], df['acc'].values[f]
    sx, sy = [fx[0]], [fy[0]]
    for i in range(1, len(fx)):                        # hold accuracy, then step up
        sx += [fx[i], fx[i]]
        sy += [sy[-1], fy[i]]
    ax.plot(sx, sy, ls='--', lw=1.1, color='#444444', zorder=2)
    ax.fill_between(sx, sy, ax.get_ylim()[1], color='#0072B2', alpha=0.06, lw=0, zorder=0)

    # ---- hardware budget ----------------------------------------------------------------
    ax.axvline(VRAM_BUDGET_GB, ls=':', lw=1.2, color='#D55E00', zorder=1)
    ax.annotate('15.6 GB budget', xy=(VRAM_BUDGET_GB, ax.get_ylim()[0]),
                xytext=(-4, 6), textcoords='offset points', rotation=90,
                ha='right', va='bottom', fontsize=7, color='#D55E00')

    # ---- label the frontier only --------------------------------------------------------
    texts = [ax.text(df['vram'][i], df['acc'][i],
                     '%s %s' % (df['size'][i], LABEL_OF[df['method'][i]]),
                     fontsize=6.5, color='#222222') for i in f]
    if HAVE_ADJUSTTEXT:
        adjust_text(texts, x=df['vram'].values, y=df['acc'].values, ax=ax,
                    expand=(1.6, 2.0), force_text=(0.8, 1.4), force_static=(0.5, 1.0),
                    ensure_inside_axes=True, max_move=40, time_lim=3,
                    arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.5,
                                    shrinkA=4, shrinkB=6))
    else:
        for t in texts:
            x, y = t.get_position()
            t.set_position((x, y + 0.2))

    sns.despine(ax=ax)

    # Two-group legend placed BELOW the axes: the in-axes corners are either occupied by
    # data or by the budget line's rotated label, and overlapping either is not acceptable.
    size_h = [Line2D([], [], marker='o', ls='', mfc=OKABE_ITO[s], mec='#222222',
                     mew=0.6, ms=5, label=s) for s in SIZE_ORDER]
    meth_h = [Line2D([], [], marker=MARKER_OF[m], ls='', mfc='white', mec='#222222',
                     mew=0.8, ms=5, label=LABEL_OF[m])
              for m in METHOD_ORDER if m in set(df['method'])]
    front_h = [Line2D([], [], ls='--', lw=1.1, color='#444444', label='Pareto frontier')]
    # Anchored in AXES coordinates so it sits just under the x-label; savefig's
    # bbox_inches='tight' then grows the canvas to include it, with no floating gap.
    ax.legend(handles=size_h + meth_h + front_h, loc='upper center',
              bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False,
              fontsize=7, handletextpad=0.35, columnspacing=1.0, labelspacing=0.3)

    if args.demo:
        ax.text(0.5, 0.5, 'SYNTHETIC', transform=ax.transAxes, fontsize=26,
                color='#D55E00', alpha=0.22, ha='center', va='center',
                rotation=30, zorder=10, fontweight='bold')

    fig.tight_layout()
    name = 'fig8_memory_pareto_demo' if args.demo else 'fig8_memory_pareto'
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / ('%s.pdf' % name), bbox_inches='tight')
    fig.savefig(FIG / ('%s.png' % name), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('wrote %s.{pdf,png}  (%d cells%s)'
          % (FIG / name, len(df), ', SYNTHETIC' if args.demo else ', measured'))
    print('Pareto-optimal (min memory, max accuracy):')
    for i in f:
        print('   %-5s %-8s  %5.2f GB  %.2f%%'
              % (df['size'][i], LABEL_OF[df['method'][i]], df['vram'][i], df['acc'][i]))
    if not HAVE_ADJUSTTEXT:
        print('note: adjustText not installed; used fixed label offsets.')


if __name__ == '__main__':
    main()
