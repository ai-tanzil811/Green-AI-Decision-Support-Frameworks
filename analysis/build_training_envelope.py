#!/usr/bin/env python
"""
Derive the surrogate's training envelope from the dataset and record it in model_metadata.json.

The envelope is the region of configuration space the surrogate was actually fitted on:
parameter range, model families, methods, measured tiers, and which (method, tier) cells were
measured together. green_peft.confidence reads it to decide how much to trust each prediction,
so it must come from the data rather than from a hand-maintained constant (workflow section 26).

Usage:
    python analysis/build_training_envelope.py [--check]

--check verifies that the recorded envelope matches the dataset and exits non-zero if not,
without writing anything. Use it in CI to catch a dataset change that silently invalidates
the recorded envelope.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / 'data' / 'processed' / 'greenpeft_ml_ready_dataset.csv'

# Both copies must stay in step: the repo-level audit record and the one shipped in the wheel.
METADATA_PATHS = [
    ROOT / 'models' / 'model_metadata.json',
    ROOT / 'green_peft_cli' / 'green_peft_pkg' / 'green_peft' / 'data' / 'model_metadata.json',
]


def build_envelope(csv_path: Path = DATASET) -> dict:
    df = pd.read_csv(csv_path)
    train = df[(df['analysis_role'] == 'surrogate_train') & (df['is_empirical'] == 1)]

    # One row per configuration -- the two measurement passes share every pre-run column,
    # so collapsing here keeps the counts meaningful.
    configs = train.drop_duplicates('config_id')

    tiers = configs.groupby('backbone')['params_b'].first()
    if tiers.nunique() != len(tiers):
        raise ValueError(f'Backbone tiers do not have distinct parameter counts: {tiers.to_dict()}')

    return {
        'derived_from': str(csv_path.relative_to(ROOT)).replace('\\', '/'),
        'n_configs': int(len(configs)),
        'n_measurement_rows': int(len(train)),
        'params_b': {'min': float(configs['params_b'].min()),
                     'max': float(configs['params_b'].max())},
        'families': sorted(configs['family'].unique().tolist()),
        'methods': sorted(configs['method'].unique().tolist()),
        'quant_bits': sorted(int(x) for x in configs['quant_bits'].unique()),
        'rank': sorted(float(x) for x in configs['rank'].unique()),
        'gpu_total_gb': float(configs['gpu_total_gb'].dropna().unique()[0]),
        'backbone_tiers': {str(b): float(v) for b, v in tiers.items()},
        'measured_cells': sorted({f'{m}|{b}'
                                  for m, b in zip(configs['method'], configs['backbone'])}),
        'task': 'SST-2 classification',
        'note': ('Scope of direct evidence. A candidate outside this envelope is an '
                 'extrapolation; see green_peft/confidence.py. The leave-one-tier-out '
                 'protocol only ever held out a tier INSIDE the params_b range, so error '
                 'bands do not cover candidates beyond it.'),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='Verify the recorded envelope matches the dataset; do not write.')
    args = ap.parse_args()

    if not DATASET.exists():
        print(f'error: dataset not found at {DATASET}', file=sys.stderr)
        return 2

    envelope = build_envelope()

    if args.check:
        stale = []
        for path in METADATA_PATHS:
            if not path.exists():
                stale.append(f'{path}: missing')
                continue
            recorded = json.loads(path.read_text(encoding='utf-8')).get('training_envelope')
            if recorded != envelope:
                stale.append(f'{path}: training_envelope does not match the dataset')
        if stale:
            for s in stale:
                print(f'STALE  {s}', file=sys.stderr)
            print('\nRe-run without --check to refresh.', file=sys.stderr)
            return 1
        print(f'OK  training_envelope matches {DATASET.name} in '
              f'{len(METADATA_PATHS)} metadata file(s)')
        return 0

    for path in METADATA_PATHS:
        if not path.exists():
            print(f'skip (missing): {path}')
            continue
        meta = json.loads(path.read_text(encoding='utf-8'))
        meta['training_envelope'] = envelope
        path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
        print(f'updated: {path.relative_to(ROOT)}')

    print(f'\nEnvelope: {envelope["n_configs"]} configs, '
          f'{envelope["params_b"]["min"]:g}-{envelope["params_b"]["max"]:g}B, '
          f'families={envelope["families"]}, '
          f'{len(envelope["measured_cells"])} measured (method, tier) cells')
    return 0


if __name__ == '__main__':
    sys.exit(main())
