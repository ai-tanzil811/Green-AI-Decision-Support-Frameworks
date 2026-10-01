#!/usr/bin/env python
"""
Predicted vs actual for GreenPEFT recommendations (workflow spec section 18).

This is the bridge between the surrogate and real-world evidence. Nothing else in the project
closes the loop: the surrogate predicts, the decision engine prescribes, and until a prediction
is checked against a real run, the whole chain rests on cross-validation alone.

Two modes, because they answer different questions:

  --from-benchmark   Compare predictions against the 41 configurations that were already
                     measured. These configurations were in the surrogate's training set, so
                     the result is IN-SAMPLE and is recorded as such. It is a plumbing check:
                     it proves the candidate -> feature -> model path reproduces what the model
                     was fitted on. It is NOT evidence of generalisation; the out-of-sample
                     numbers live in results/surrogate/cv_metrics.csv.

  --record FILE      Compare predictions against a NEW real run recorded in a JSON file. This
                     is the genuine out-of-sample path, and the only mode that produces new
                     evidence. Run it after an actual GPU experiment.

Both modes append to results/recommendation_validation.csv with an `in_sample` flag, so the two
kinds of row can never be averaged together by accident.

Usage:
    python experiments/validate_recommendation.py --from-benchmark
    python experiments/validate_recommendation.py --record my_run.json

The --record JSON format (a single object or a list of them):

    {
      "method": "lora", "backbone": "small", "family": "llama", "params_b": 1.1,
      "rank": 16, "quant_bits": 16,
      "actual_accuracy": 0.941, "actual_peak_gpu_memory_gb": 9.3,
      "actual_energy_kwh": 0.00176, "actual_wall_clock_seconds": 105.4,
      "notes": "T4, SST-2, 1 epoch"
    }

Any actual_* field may be omitted; errors are computed only for what you supply.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'analysis'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / 'green_peft_cli' / 'green_peft_pkg'))

from greenpeft_data import ROOT, load_canonical_runs  # noqa: E402
from green_peft.recommender import GreenPEFTArtifacts, predict_all  # noqa: E402

OUT_CSV = ROOT / 'results' / 'recommendation_validation.csv'

# actual column -> predicted column
PAIRS = {
    'actual_accuracy': 'pred_accuracy',
    'actual_peak_gpu_memory_gb': 'pred_peak_vram_gb',
    'actual_energy_kwh': 'pred_energy_kwh',
    'actual_wall_clock_seconds': 'pred_wall_clock_s',
}

CONFIG_COLS = ['method', 'backbone', 'family', 'params_b', 'rank', 'quant_bits']


def predict(configs: pd.DataFrame) -> pd.DataFrame:
    """Run the shipped surrogate over configuration rows, keeping scope/confidence."""
    art = GreenPEFTArtifacts.load()
    needed = set(CONFIG_COLS) - {'backbone'}
    missing = needed - set(configs.columns)
    if missing:
        raise ValueError(f'configurations are missing required columns: {sorted(missing)}')
    return predict_all(art, configs.copy())


def compare(predicted: pd.DataFrame, in_sample: bool, source: str) -> pd.DataFrame:
    """One row per (configuration, target) with signed and relative error."""
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    rows = []

    for _, r in predicted.iterrows():
        for actual_col, pred_col in PAIRS.items():
            if actual_col not in r.index or pd.isna(r[actual_col]):
                continue
            actual, pred = float(r[actual_col]), float(r[pred_col])
            target = actual_col.replace('actual_', '')

            band = r.get(f'{pred_col}_band_pct')
            lo, hi = r.get(f'{pred_col}_lo'), r.get(f'{pred_col}_hi')
            within = (bool(lo <= actual <= hi)
                      if lo is not None and not pd.isna(lo) else None)

            rows.append({
                'recorded_at': stamp,
                'source': source,
                'in_sample': in_sample,
                'config_id': r.get('config_id', f"{r['method']}_{r.get('backbone', '?')}"),
                'method': r['method'],
                'backbone': r.get('backbone'),
                'family': r['family'],
                'params_b': r['params_b'],
                'target': target,
                'actual': actual,
                'predicted': pred,
                'error': pred - actual,
                'abs_error': abs(pred - actual),
                'pct_error': (pred - actual) / actual * 100 if actual else None,
                'band_pct': band,
                'within_band': within,
                'confidence': r.get('confidence'),
                'scope': r.get('scope'),
                'notes': r.get('notes', ''),
            })

    if not rows:
        raise ValueError(
            'No actual_* values to compare against. Every record had them missing or null -- '
            'if you started from recorded_run.template.json, fill in the measurements from a '
            'real run first.')
    return pd.DataFrame(rows)


def from_benchmark() -> pd.DataFrame:
    """Back-test against already-measured configurations. In-sample by construction."""
    runs = load_canonical_runs()
    configs = runs[CONFIG_COLS + ['config_id']].copy()
    for actual_col, metric in [('actual_accuracy', 'accuracy'),
                               ('actual_peak_gpu_memory_gb', 'peak_gpu_memory_gb'),
                               ('actual_energy_kwh', 'energy_kwh'),
                               ('actual_wall_clock_seconds', 'wall_clock_seconds')]:
        configs[actual_col] = runs[metric].values

    predicted = predict(configs)
    return compare(predicted, in_sample=True, source='benchmark_backtest')


def from_file(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding='utf-8'))
    records = payload if isinstance(payload, list) else [payload]
    configs = pd.DataFrame(records)
    if 'family' not in configs.columns:
        raise ValueError('each record needs a "family" field; the surrogate one-hot encodes it')
    predicted = predict(configs)
    return compare(predicted, in_sample=False, source=f'recorded:{path.name}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--from-benchmark', action='store_true',
                   help='Back-test against the already-measured configurations (in-sample).')
    g.add_argument('--record', type=Path,
                   help='JSON file of real runs with actual_* metrics (out-of-sample).')
    ap.add_argument('--replace', action='store_true',
                    help='Overwrite results/recommendation_validation.csv instead of appending.')
    args = ap.parse_args()

    try:
        new = from_benchmark() if args.from_benchmark else from_file(args.record)
    except (ValueError, FileNotFoundError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUT_CSV.exists() and not args.replace:
        existing = pd.read_csv(OUT_CSV)
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    combined.to_csv(OUT_CSV, index=False)

    print(f'wrote {OUT_CSV.relative_to(ROOT)}  ({len(new)} new rows, {len(combined)} total)')

    print('\nError by target for this batch '
          f'({"IN-SAMPLE" if new["in_sample"].all() else "out-of-sample"}):')
    summary = (new.groupby('target')
               .agg(n=('actual', 'size'),
                    MAE=('abs_error', 'mean'),
                    mean_pct_error=('pct_error', 'mean'),
                    mean_abs_pct_error=('pct_error', lambda s: s.abs().mean()),
                    within_band=('within_band', 'mean'))
               .round(5))
    print(summary.to_string())

    if new['in_sample'].all():
        print('\nThese rows are IN-SAMPLE: every configuration above was in the surrogate\'s '
              'training set, so\nthe errors understate real-world error. They verify the '
              'prediction path, not generalisation.')
        print('For out-of-sample error see results/surrogate/cv_metrics.csv, or record a new '
              'run with --record.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
