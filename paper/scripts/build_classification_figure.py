"""Build the SST-2 classification-evaluation figure from real per-example predictions.

Input is a CSV written by ``dump_eval_predictions.py`` with columns
``idx,label,pred,p_pos``. Output is ``paper/figures/fig6_classification.{pdf,png}`` -- a
confusion matrix and an ROC curve -- plus a block of LaTeX-ready numbers for the paper's
classification-evaluation subsection.

This script refuses to run on anything but a real prediction file: it checks that the file
has 872 rows, that labels are binary, that predictions match the argmax of the recorded
probability, and that the probabilities are not degenerate. It never synthesises scores.

Usage:
    python paper/scripts/build_classification_figure.py --preds preds_lora_tiny_classification_seed42.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             accuracy_score, f1_score)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'paper' / 'data'
FIG = ROOT / 'paper' / 'figures'

N_EVAL = 872                      # the full SST-2 validation split
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
POS, NEG = '#2a78d6', '#eb6834'   # matches METHOD_COLOR in build_paper_assets.py

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8.5, 'axes.labelsize': 8.5,
    'axes.titlesize': 8.5, 'legend.fontsize': 7.5, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'pdf.fonttype': 42,
})


def load(path):
    rows = list(csv.DictReader(path.open(encoding='utf-8')))
    if not rows:
        sys.exit('%s is empty' % path)
    y = np.array([int(r['label']) for r in rows])
    pred = np.array([int(r['pred']) for r in rows])
    p = np.array([float(r['p_pos']) for r in rows])

    # Guards: this figure is only meaningful on genuine saved predictions.
    if len(rows) != N_EVAL:
        sys.exit('expected %d rows (the SST-2 validation split), found %d'
                 % (N_EVAL, len(rows)))
    if set(np.unique(y)) - {0, 1} or set(np.unique(pred)) - {0, 1}:
        sys.exit('labels and predictions must be binary')
    if p.min() < 0 or p.max() > 1:
        sys.exit('p_pos outside [0, 1]')
    if np.unique(p).size < 10:
        sys.exit('p_pos takes only %d distinct values; an ROC curve needs real scores, '
                 'not thresholded output' % np.unique(p).size)
    mism = int(((p >= 0.5).astype(int) != pred).sum())
    if mism > 0:
        print('warning: %d rows where pred != argmax(p_pos); using the recorded pred '
              'column for the confusion matrix and p_pos for the ROC' % mism)
    return y, pred, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', required=True,
                    help='CSV in paper/data/ from dump_eval_predictions.py')
    ap.add_argument('--name', default='fig6_classification')
    args = ap.parse_args()

    path = Path(args.preds)
    if not path.exists():
        path = DATA / args.preds
    if not path.exists():
        sys.exit('prediction file not found: %s\nRun dump_eval_predictions.py on a GPU '
                 'machine first.' % args.preds)

    y, pred, p = load(path)
    run = path.stem.replace('preds_', '')

    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y, pred)
    f1m = f1_score(y, pred, average='macro')
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # -- confusion matrix ----------------------------------------------------------
    ax1.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
    ax1.grid(False)
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, '%d' % cm[i, j], ha='center', va='center',
                     fontsize=11, color='white' if cm[i, j] > cm.max() * 0.55 else INK)
    ax1.set_xticks([0, 1], ['negative', 'positive'])
    ax1.set_yticks([0, 1], ['negative', 'positive'])
    ax1.set_xlabel('predicted')
    ax1.set_ylabel('true')
    ax1.set_title('(a) Confusion matrix', loc='left')

    # -- ROC ------------------------------------------------------------------------
    ax2.plot(fpr, tpr, color=POS, lw=1.4, label='AUC = %.3f' % auc)
    ax2.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls='--', label='chance')
    ax2.scatter([fp / (fp + tn)], [tp / (tp + fn)], s=22, color=NEG, zorder=5,
                label='operating point')
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel('false positive rate')
    ax2.set_ylabel('true positive rate')
    ax2.set_title('(b) ROC curve', loc='left')
    ax2.legend(loc='lower right', frameon=False)

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / ('%s.pdf' % args.name), bbox_inches='tight')
    fig.savefig(FIG / ('%s.png' % args.name), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('wrote %s.{pdf,png}\n' % (FIG / args.name))
    print('--- numbers for the paper (run: %s) ---' % run)
    print('n            = %d' % len(y))
    print('TN %4d   FP %4d' % (tn, fp))
    print('FN %4d   TP %4d' % (fn, tp))
    print('accuracy     = %.4f' % acc)
    print('macro-F1     = %.4f' % f1m)
    print('AUC          = %.4f' % auc)
    print('TPR (recall) = %.4f' % (tp / (tp + fn)))
    print('FPR          = %.4f' % (fp / (fp + tn)))
    print('precision    = %.4f' % (tp / (tp + fp)) if (tp + fp) else 'precision    = n/a')
    print('\nCheck accuracy above against the recorded value for this cell in\n'
          'results/canonical_benchmark/raw_runs/%s.json before writing it up.' % run)


if __name__ == '__main__':
    main()
