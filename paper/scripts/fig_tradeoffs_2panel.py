"""Component 1 -- two-panel PEFT trade-off figure (accuracy vs energy, accuracy vs VRAM).

Publication settings: 300 DPI, 10 pt base font, sns.despine(), Okabe-Ito colorblind-safe
palette, marker shape carrying method identity so the figure survives grayscale printing.
Exports vector PDF (for LaTeX) and 300 DPI PNG, sized for an IEEE double-column span.

Data
----
By default this reads the real measurements in paper/data/table2_complete_cells.csv: the 13
method x backbone cells that completed all three seeds, with mean and SD per cell. Error bars
are those SDs, on both axes.

    python paper/scripts/fig_tradeoffs_2panel.py

--demo generates clearly-labelled synthetic data instead, so the script runs standalone
outside the repository. Demo output is watermarked SYNTHETIC across both panels and written
under a _demo name, because a plausible-looking fake figure must never be mistakable for a
measured one in this project.

    python paper/scripts/fig_tradeoffs_2panel.py --demo

Optional: `pip install adjustText` for overlap-free point labels. Without it the script falls
back to fixed offsets and still runs.
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
except ImportError:                                  # graceful fallback, not an error
    HAVE_ADJUSTTEXT = False

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'paper' / 'data' / 'table2_complete_cells.csv'
FIG = ROOT / 'paper' / 'figures'

# Backbone keys in the data -> parameter count used for the colour legend.
SIZE_OF = {'tiny': '0.5B', 'small': '1.1B', 'medium': '1.5B', 'large': '3.0B'}
SIZE_ORDER = ['0.5B', '1.1B', '1.5B', '3.0B']

# Okabe-Ito: colourblind-safe, and the four chosen entries also separate in grayscale.
OKABE_ITO = {'0.5B': '#0072B2', '1.1B': '#009E73', '1.5B': '#E69F00', '3.0B': '#D55E00'}

# Method identity is carried by SHAPE, so colour can be spent entirely on model size.
MARKER_OF = {'lora': 'o', 'qlora': '^', 'lora_fa': 'D', 'lisa': 'v', 'full_ft': 's'}
LABEL_OF = {'lora': 'LoRA', 'qlora': 'QLoRA', 'lora_fa': 'LoRA-FA',
            'lisa': 'LISA', 'full_ft': 'Full FT'}
METHOD_ORDER = ['lora', 'qlora', 'lora_fa', 'lisa', 'full_ft']

VRAM_BUDGET_GB = 15.6          # the Tesla T4 budget every run was held to


def style():
    """IEEE-ish publication styling applied once, before any axes exist."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 10,
        'axes.titlesize': 10, 'legend.fontsize': 8.5, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
        'savefig.dpi': 300, 'figure.dpi': 300,
    })


def load_real():
    df = pd.read_csv(DATA)
    df = df[df['accuracy_count'] == 3].copy()        # complete cells only
    df['size'] = df['backbone'].map(SIZE_OF)
    df['acc'] = df['accuracy_mean'] * 100
    df['acc_sd'] = df['accuracy_std'] * 100
    return df.rename(columns={
        'energy_wh_mean': 'energy', 'energy_wh_std': 'energy_sd',
        'peak_gpu_memory_gb_mean': 'vram', 'peak_gpu_memory_gb_std': 'vram_sd'})


def load_demo():
    """Synthetic stand-in with the same column structure. Shapes the eye, proves nothing."""
    rng = np.random.default_rng(0)
    rows = []
    for size, scale in zip(SIZE_ORDER, [0.5, 1.1, 1.5, 3.0]):
        for method in METHOD_ORDER:
            if size != '0.5B' and method == 'full_ft':
                continue                              # mirrors the real coverage gap
            rows.append({
                'size': size, 'method': method,
                'acc': 88 + 4 * np.log1p(scale) + rng.normal(0, 0.6),
                'acc_sd': abs(rng.normal(0.4, 0.2)),
                'energy': 1.0 * scale + rng.normal(0, 0.3) + 1.0,
                'energy_sd': abs(rng.normal(0.05, 0.02)),
                'vram': 2.5 * scale + rng.normal(0, 1.0) + 2.0,
                'vram_sd': abs(rng.normal(0.4, 0.2)),
            })
    return pd.DataFrame(rows)


def pareto_front(x, y):
    """Indices of the non-dominated set for (minimise x, maximise y).

    Point i is dominated when some j is no worse on both axes and strictly better on one.
    Returned indices are sorted by x so the caller can draw a monotone staircase.
    """
    idx = np.argsort(x)
    front, best_y = [], -np.inf
    for i in idx:
        if y[i] > best_y:                             # strictly better accuracy than anything cheaper
            front.append(i)
            best_y = y[i]
    return np.array(front)


def draw_front(ax, x, y, labels=None, shade=True):
    """Dashed staircase through the non-dominated points, with the dominated-free region shaded."""
    f = pareto_front(np.asarray(x, float), np.asarray(y, float))
    fx, fy = np.asarray(x, float)[f], np.asarray(y, float)[f]

    # Staircase: hold accuracy until the next cheaper-and-better point appears.
    sx, sy = [fx[0]], [fy[0]]
    for i in range(1, len(fx)):
        sx += [fx[i], fx[i]]
        sy += [sy[-1], fy[i]]
    ax.plot(sx, sy, ls='--', lw=1.1, color='#444444', zorder=2,
            label='Pareto frontier (2-D)')

    if shade:
        # Everything up-and-left of the staircase is the region no measured point dominates.
        ymax = ax.get_ylim()[1]
        ax.fill_between(sx, sy, ymax, color='#0072B2', alpha=0.06, lw=0, zorder=0)
    return f


def scatter_panel(ax, df, xcol, xerr):
    for _, r in df.iterrows():
        ax.errorbar(r[xcol], r['acc'], xerr=r[xerr], yerr=r['acc_sd'],
                    fmt=MARKER_OF[r['method']], ms=6.0, mfc=OKABE_ITO[r['size']],
                    mec='#222222', mew=0.6, ecolor='#999999', elinewidth=0.8,
                    capsize=2, zorder=3)


def label_front(ax, df, xcol, front_idx):
    """Annotate only the non-dominated points.

    Labelling all 13 would restate the colour legend and crowd the axes; the frontier is
    what a reader actually needs named, so each gets size and method.
    """
    texts = []
    for i in front_idx:
        r = df.iloc[i]
        texts.append(ax.text(r[xcol], r['acc'],
                             '%s %s' % (r['size'], LABEL_OF[r['method']]),
                             fontsize=7, color='#222222'))
    if HAVE_ADJUSTTEXT:
        # Pass every marker coordinate, not just the labelled ones, so labels are repelled
        # by all 13 points rather than only by each other. Frontier points sit on the
        # upper-left boundary, so the shaded region above them is the natural escape space.
        adjust_text(texts, x=df[xcol].values, y=df['acc'].values, ax=ax,
                    expand=(1.6, 2.0), force_text=(0.8, 1.4), force_static=(0.5, 1.0),
                    ensure_inside_axes=True, max_move=40, time_lim=3,
                    arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.5,
                                    shrinkA=4, shrinkB=6))
    else:
        for t in texts:                               # fixed offset fallback
            x, y = t.get_position()
            t.set_position((x, y + 0.15))
            t.set_ha('center')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', action='store_true',
                    help='use watermarked synthetic data instead of the measurements')
    ap.add_argument('--carbon', action='store_true',
                    help='put estimated operational CO2e on panel (a) instead of energy. '
                         'Same measurement: CO2e = energy x 0.65 kg/kWh, so this is a '
                         'relabelled axis, not a second result. Use one or the other, '
                         'never both figures in one paper.')
    args = ap.parse_args()

    style()
    df = load_demo() if args.demo else load_real()

    # CO2e is a fixed linear rescaling of measured energy, so it is an axis choice.
    if args.carbon:
        df = df.copy()
        df['carbon'] = df['energy'] * 0.65          # Wh x 0.65 g/Wh = g CO2e
        df['carbon_sd'] = df['energy_sd'] * 0.65
        xcol, xerr = 'carbon', 'carbon_sd'
        xlabel = 'Estimated operational CO$_2$e per run (g)'
        xtitle = '(a) Carbon cost'
    else:
        xcol, xerr = 'energy', 'energy_sd'
        xlabel = 'GPU energy per run (Wh)'
        xtitle = '(a) Energy efficiency'

    # 7.16 in spans both columns of an IEEE page.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.5))

    # ---- panel (a): accuracy vs energy (or its CO2e rescaling) -------------------------
    scatter_panel(ax1, df, xcol, xerr)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Validation accuracy (%)')
    ax1.set_title(xtitle, loc='left', fontweight='bold')
    ax1.margins(x=0.12, y=0.22)
    f1 = draw_front(ax1, df[xcol].values, df['acc'].values)
    label_front(ax1, df, xcol, f1)

    # ---- panel (b): accuracy vs peak VRAM ----------------------------------------------
    scatter_panel(ax2, df, 'vram', 'vram_sd')
    ax2.set_xlabel('Peak GPU memory (GB)')
    ax2.set_ylabel('Validation accuracy (%)')
    ax2.set_title('(b) Memory constraint', loc='left', fontweight='bold')
    # Headroom on the right so the budget line and its label are never clipped.
    ax2.set_xlim(right=VRAM_BUDGET_GB * 1.08)
    ax2.margins(y=0.22)
    f2 = draw_front(ax2, df['vram'].values, df['acc'].values)
    ax2.axvline(VRAM_BUDGET_GB, ls=':', lw=1.2, color='#D55E00', zorder=1)
    ax2.annotate('15.6 GB budget', xy=(VRAM_BUDGET_GB, ax2.get_ylim()[0]),
                 xytext=(-4, 6), textcoords='offset points', rotation=90,
                 ha='right', va='bottom', fontsize=7.5, color='#D55E00')
    label_front(ax2, df, 'vram', f2)

    for ax in (ax1, ax2):
        sns.despine(ax=ax)

    # ---- shared top legend: colour = size, shape = method ------------------------------
    size_h = [Line2D([], [], marker='o', ls='', mfc=OKABE_ITO[s], mec='#222222',
                     mew=0.6, ms=6, label=s) for s in SIZE_ORDER]
    meth_h = [Line2D([], [], marker=MARKER_OF[m], ls='', mfc='white', mec='#222222',
                     mew=0.8, ms=6, label=LABEL_OF[m])
              for m in METHOD_ORDER if m in set(df['method'])]
    front_h = [Line2D([], [], ls='--', lw=1.1, color='#444444', label='Pareto frontier')]

    leg = fig.legend(handles=size_h + meth_h + front_h,
                     loc='upper center', bbox_to_anchor=(0.5, 1.10),
                     ncol=5, frameon=False, handletextpad=0.4, columnspacing=1.2)
    leg.set_in_layout(True)

    if args.demo:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, 'SYNTHETIC', transform=ax.transAxes, fontsize=28,
                    color='#D55E00', alpha=0.22, ha='center', va='center',
                    rotation=30, zorder=10, fontweight='bold')

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    name = 'fig7_tradeoffs'
    if args.carbon:
        name = 'fig2_accuracy_carbon'
    if args.demo:
        name += '_demo'
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / ('%s.pdf' % name), bbox_inches='tight')
    fig.savefig(FIG / ('%s.png' % name), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('wrote %s.{pdf,png}  (%d cells%s)'
          % (FIG / name, len(df), ', SYNTHETIC' if args.demo else ', measured'))
    if not HAVE_ADJUSTTEXT:
        print('note: adjustText not installed; used fixed label offsets. '
              'pip install adjustText for overlap-free labels.')


if __name__ == '__main__':
    main()
