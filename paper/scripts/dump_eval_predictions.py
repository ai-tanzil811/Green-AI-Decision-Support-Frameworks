"""Re-run one benchmark cell and save per-example SST-2 predictions.

Why this script exists
----------------------
The benchmark evaluates with ``AutoModelForSequenceClassification`` (num_labels=2), so
class scores exist at evaluation time, but ``evaluate_classification`` in
``notebooks/green_peft_benchmark_execution.ipynb`` keeps only ``logits.argmax(-1)`` and
returns aggregate accuracy / macro-F1. No per-example prediction survives in
``results/canonical_benchmark/raw_runs/``, so a confusion matrix and an ROC curve cannot be
produced from the saved artifacts.

Aggregate accuracy pins TP+TN exactly, and macro-F1 constrains the error split, but only
loosely: macro-F1 is flat near a balanced split, so at the 4-decimal precision stored in the
raw JSON the feasible TP range spans roughly +-20 counts with the sign unidentified. That is
too weak for a confusion matrix, and an ROC curve needs per-threshold scores that do not
exist at any precision. This script therefore recovers the real thing by re-running one cell.

What it does
------------
Executes the benchmark notebook's definition cells (environment, settings, data, measurement
layer, model builders, task pipelines, ``run_one``) so the pipeline is identical to the one
that produced the published numbers, then replaces ``evaluate_classification`` with a version
that also records the positive-class probability for each of the 872 validation examples, and
runs a single configuration.

Because this is a fresh run, its accuracy will be close to but need not exactly equal the
recorded value for the same cell (library versions and non-deterministic GPU kernels). The
script prints both so the difference can be reported honestly. It writes to its own raw
directory and never touches ``results/canonical_benchmark/``.

Usage (on a GPU box -- Kaggle T4, Colab T4, or any CUDA machine)
---------------------------------------------------------------
    python paper/scripts/dump_eval_predictions.py --method lora --backbone tiny --seed 42

    # first run on a fresh machine also needs the pinned packages:
    python paper/scripts/dump_eval_predictions.py --method lora --backbone tiny --seed 42 \
        --install --gpu 0

``--backbone tiny`` is Qwen2.5-0.5B and trains in roughly 110 s on a T4 at the benchmark's
300-step budget. Output: ``paper/data/preds_<run_id>.csv`` with columns
``idx,label,pred,p_pos``. Feed that to ``build_classification_figure.py``.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'notebooks' / 'green_peft_benchmark_execution.ipynb'
RECORDED = ROOT / 'results' / 'canonical_benchmark' / 'raw_runs'
OUT_DIR = ROOT / 'paper' / 'data'

# Notebook cells holding definitions only. Cell 11 is the sweep driver and is never run.
DEF_CELLS = range(2, 10)


def load_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    return [''.join(nb['cells'][i]['source']) for i in DEF_CELLS]


def patch_env_cell(src, install, gpu):
    """Turn off the pip install and point CUDA at the requested device."""
    if not install:
        src = src.replace('INSTALL = True', 'INSTALL = False', 1)
    src = re.sub(r"os\.environ\['CUDA_VISIBLE_DEVICES'\] = '\d+'",
                 "os.environ['CUDA_VISIBLE_DEVICES'] = '%d'" % gpu, src, count=1)
    return src


def make_recording_evaluator(ns, sink):
    """A drop-in evaluate_classification that also records per-example scores.

    Mirrors the notebook's own implementation (batch size 32, same tokenizer call, argmax
    prediction) so the returned accuracy and macro-F1 are computed exactly as before. The
    only addition is the softmax probability of the positive class.
    """
    torch = ns['torch']
    accuracy_score, f1_score = ns['accuracy_score'], ns['f1_score']
    model_device, MAX_SEQ_LEN = ns['model_device'], ns['MAX_SEQ_LEN']

    @torch.no_grad()
    def evaluate_classification(model, tok, eval_ds, task_cfg):
        model.eval()
        tf, lf = task_cfg['text_field'], task_cfg['label_field']
        preds, golds, probs = [], [], []
        bs = 32
        dev = model_device(model)
        for i in range(0, len(eval_ds), bs):
            chunk = eval_ds[i:i + bs]
            enc = tok(chunk[tf], truncation=True, max_length=MAX_SEQ_LEN,
                      padding=True, return_tensors='pt').to(dev)
            logits = model(**enc).logits.float()
            preds.extend(logits.argmax(-1).cpu().tolist())
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
            golds.extend(chunk[lf])
        n = len(golds)
        majority = max(sum(golds), n - sum(golds)) / max(n, 1)
        sink.clear()
        sink.extend(zip(golds, preds, probs))
        return {
            'accuracy': round(float(accuracy_score(golds, preds)), 4),
            'f1_macro': round(float(f1_score(golds, preds, average='macro')), 4),
            'majority_baseline': round(float(majority), 4),
            'n_eval': n,
        }

    return evaluate_classification


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--method', default='lora',
                    choices=['full_ft', 'lora', 'qlora', 'lora_fa', 'lisa'])
    ap.add_argument('--backbone', default='tiny',
                    choices=['tiny', 'small', 'medium', 'large'],
                    help='tiny=Qwen2.5-0.5B, small=TinyLlama-1.1B, '
                         'medium=Qwen2.5-1.5B, large=Qwen2.5-3B')
    ap.add_argument('--seed', type=int, default=42, choices=[13, 42, 2024])
    ap.add_argument('--install', action='store_true',
                    help='run the notebook pip install (needed once per fresh machine)')
    ap.add_argument('--gpu', type=int, default=1,
                    help="CUDA_VISIBLE_DEVICES index; the notebook uses 1 (Kaggle's "
                         'second T4). Use 0 on a single-GPU box.')
    args = ap.parse_args()

    if not NOTEBOOK.exists():
        sys.exit('benchmark notebook not found at %s' % NOTEBOOK)

    cells = load_cells()
    cells[0] = patch_env_cell(cells[0], args.install, args.gpu)

    ns = {'__name__': '__main__'}
    for i, src in zip(DEF_CELLS, cells):
        print('--- notebook cell %d ---' % i)
        exec(compile(src, '<nb cell %d>' % i, 'exec'), ns)

    if not ns.get('HAS_GPU'):
        sys.exit('\nNo CUDA device visible. This script must run on a GPU; the 4-bit QLoRA '
                 'path and the NVML measurement layer both require one.')

    # Keep the fresh run out of the published benchmark directory. run_one() short-circuits
    # when the target JSON already exists, so a private directory also guarantees it trains.
    rerun_dir = ROOT / 'results' / 'prediction_rerun' / 'raw_runs'
    rerun_dir.mkdir(parents=True, exist_ok=True)
    ns['RAW_DIR'] = rerun_dir

    sink = []
    ns['evaluate_classification'] = make_recording_evaluator(ns, sink)

    rid = ns['run_id_for'](args.method, args.backbone, 'classification', args.seed)
    print('\n=== re-running %s to capture per-example predictions ===' % rid)
    record = ns['run_one'](args.method, args.backbone, 'classification', args.seed)

    if record.get('status') != 'ok':
        sys.exit('run did not complete: status=%s' % record.get('status'))
    if not sink:
        sys.exit('evaluator produced no predictions')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / ('preds_%s.csv' % rid)
    with out.open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['idx', 'label', 'pred', 'p_pos'])
        for i, (label, pred, p) in enumerate(sink):
            w.writerow([i, int(label), int(pred), '%.6f' % p])

    rerun_acc = record['performance']['accuracy']
    print('\nwrote %s  (%d rows)' % (out, len(sink)))
    print('re-run accuracy : %.4f   macro-F1 %.4f'
          % (rerun_acc, record['performance']['f1_macro']))

    ref = RECORDED / ('%s.json' % rid)
    if ref.exists():
        rec = json.loads(ref.read_text())['performance']
        print('recorded        : %.4f   macro-F1 %.4f'
              % (rec['accuracy'], rec['f1_macro']))
        print('difference      : %+.4f accuracy' % (rerun_acc - rec['accuracy']))
        print('\nReport both numbers in the paper. The figure describes this replicate, not\n'
              'the recorded run, unless the two agree to the reported precision.')
    else:
        print('no recorded run at %s for comparison' % ref)

    print('\nnext: python paper/scripts/build_classification_figure.py --preds %s' % out.name)


if __name__ == '__main__':
    main()
