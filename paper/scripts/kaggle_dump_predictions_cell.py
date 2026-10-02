# ==================== PASTE AS A NEW CELL IN THE KAGGLE NOTEBOOK ====================
# Captures per-example SST-2 predictions so the paper can show a real confusion matrix and
# ROC curve. The benchmark evaluates with AutoModelForSequenceClassification (num_labels=2),
# so class scores exist at evaluation time -- Cell 7's evaluate_classification() just keeps
# logits.argmax(-1) and discards them. This re-runs ONE configuration with an evaluator that
# also records the positive-class probability.
#
# WHERE: run after Cell 8 (the one defining run_one). Do NOT run Cell 10, the sweep driver.
# TIME:  about 2 minutes on a T4 for the default configuration.
# SAFETY: writes to its own directory; results/canonical_benchmark/ is never touched.

METHOD, BACKBONE, SEED = 'lora', 'tiny', 42      # tiny = Qwen2.5-0.5B, ~110 s of training

import csv
from pathlib import Path

# 1. Keep the fresh run out of the published benchmark directory. run_one() short-circuits
#    when the target JSON already exists, so a private directory also guarantees it trains.
RAW_DIR = Path('/kaggle/working/prediction_rerun')
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 2. Drop-in replacement for Cell 7's evaluator. Identical logic -- same batch size, same
#    tokenizer call, same argmax prediction -- so accuracy and macro-F1 are computed exactly
#    as before. The only addition is softmax(logits)[:, 1].
_sink = []

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
    _sink.clear()
    _sink.extend(zip(golds, preds, probs))
    return {
        'accuracy': round(float(accuracy_score(golds, preds)), 4),
        'f1_macro': round(float(f1_score(golds, preds, average='macro')), 4),
        'majority_baseline': round(float(majority), 4),
        'n_eval': n,
    }

# 3. Run it.
rid = run_id_for(METHOD, BACKBONE, 'classification', SEED)
print('=== re-running %s to capture per-example predictions ===' % rid)
record = run_one(METHOD, BACKBONE, 'classification', SEED)
assert record.get('status') == 'ok', 'run did not complete: %s' % record.get('status')
assert _sink, 'evaluator produced no predictions'

# 4. Write the CSV. Download it from the Kaggle output pane into paper/data/.
out = Path('/kaggle/working/preds_%s.csv' % rid)
with out.open('w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['idx', 'label', 'pred', 'p_pos'])
    for i, (label, pred, p) in enumerate(_sink):
        w.writerow([i, int(label), int(pred), '%.6f' % p])

print('\nwrote %s  (%d rows)' % (out, len(_sink)))
print('re-run accuracy : %.4f   macro-F1 %.4f'
      % (record['performance']['accuracy'], record['performance']['f1_macro']))
print('\nCompare that against the recorded value for this cell in\n'
      'results/canonical_benchmark/raw_runs/%s.json. A fresh run need not match it\n'
      'exactly (library versions, non-deterministic kernels), so report both numbers and\n'
      'describe the figure as a replicate of that configuration.' % rid)
print('\nNext, locally:\n'
      '  python paper/scripts/build_classification_figure.py --preds %s' % out.name)
