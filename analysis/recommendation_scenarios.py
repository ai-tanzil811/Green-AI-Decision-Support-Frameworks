#!/usr/bin/env python
"""
Run the decision engine over the paper's example constraint scenarios (workflow sections 17, 22).

Produces Table 4 of the paper -- "Example GreenPEFT recommendations" -- as:

    results/recommendations/scenarios.csv       one row per (scenario, ranked candidate)
    results/recommendations/scenarios.json      the same, with the constraint set per scenario

Every row carries the candidate's scope and confidence, so a recommendation that rests on an
extrapolation can never be read as though it rested on a measured cell.

Usage:
    python analysis/recommendation_scenarios.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / 'green_peft_cli' / 'green_peft_pkg'))

from greenpeft_data import ROOT  # noqa: E402
from green_peft.recommender import (  # noqa: E402
    Constraints, GreenPEFTArtifacts, recommend,
)

OUT_DIR = ROOT / 'results' / 'recommendations'
TOP_K = 3

# The scenarios named in readme.md and workflow section 17, plus two evidence-floor variants
# that show what the engine recommends when it is restricted to measured configurations.
SCENARIOS = [
    ('tight_vram_balanced',
     'An 8 GB consumer GPU, 0.90 accuracy floor, balanced preferences',
     Constraints(max_vram_gb=8.0, min_accuracy=0.90), 'balanced'),
    ('standard_balanced',
     'A 16 GB GPU, 0.90 accuracy floor, balanced preferences',
     Constraints(max_vram_gb=16.0, min_accuracy=0.90), 'balanced'),
    ('high_accuracy',
     'A 16 GB GPU, 0.95 accuracy floor, accuracy weighted heavily',
     Constraints(max_vram_gb=16.0, min_accuracy=0.95), 'high_accuracy'),
    ('strict_carbon',
     'A 16 GB GPU, 0.90 accuracy floor, carbon weighted heavily',
     Constraints(max_vram_gb=16.0, min_accuracy=0.90), 'strict_carbon'),
    ('standard_balanced_measured_only',
     'As standard_balanced, but restricted to directly measured configurations',
     Constraints(max_vram_gb=16.0, min_accuracy=0.90, min_confidence='HIGH'), 'balanced'),
    ('high_accuracy_measured_only',
     'As high_accuracy, but restricted to directly measured configurations',
     Constraints(max_vram_gb=16.0, min_accuracy=0.95, min_confidence='HIGH'), 'high_accuracy'),
]

REPORT_COLS = ['backbone', 'model_id', 'family', 'params_b', 'method', 'rank', 'quant_bits',
               'pred_accuracy', 'pred_peak_vram_gb', 'pred_energy_kwh',
               'pred_carbon_kgco2eq', 'pred_wall_clock_s',
               'gei', 'on_pareto_front', 'confidence', 'scope', 'scope_reason']


def run_all() -> tuple[pd.DataFrame, list[dict]]:
    art = GreenPEFTArtifacts.load()
    rows, payload = [], []

    for name, description, constraints, profile in SCENARIOS:
        result = recommend(art, constraints, profile=profile,
                           gpu_vram_gb=constraints.max_vram_gb, top_k=TOP_K)

        record = {
            'scenario': name,
            'description': description,
            'profile': profile,
            'constraints': {k: v for k, v in vars(constraints).items() if v is not None},
            'n_candidates': result['n_candidates'],
            'n_feasible': result['n_feasible'],
            'n_implausible': result.get('n_implausible', 0),
            'model_status': result.get('model_status'),
            'scope_summary': result.get('scope_summary'),
            'recommendations': [],
        }

        if result['ranked'] is None:
            # A scenario with no feasible candidate is a result, not an error. Record why.
            record['dropped_summary'] = result['dropped_summary']
            rows.append({'scenario': name, 'rank': None, 'feasible': False,
                         'reason': ' | '.join(result['dropped_summary'])})
        else:
            for i, (_, r) in enumerate(result['ranked'].iterrows(), start=1):
                row = {'scenario': name, 'profile': profile, 'rank': i, 'feasible': True}
                row.update({c: r[c] for c in REPORT_COLS if c in r.index})
                rows.append(row)
                record['recommendations'].append(
                    {k: (v.item() if hasattr(v, 'item') else v) for k, v in row.items()})

        payload.append(record)

    return pd.DataFrame(rows), payload


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table, payload = run_all()

    csv_path = OUT_DIR / 'scenarios.csv'
    json_path = OUT_DIR / 'scenarios.json'
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(payload, indent=2, default=str) + '\n', encoding='utf-8')

    print(f'wrote {csv_path.relative_to(ROOT)}  ({len(table)} rows)')
    print(f'wrote {json_path.relative_to(ROOT)}  ({len(payload)} scenarios)')

    print('\nTop recommendation per scenario:')
    for rec in payload:
        if not rec['recommendations']:
            print(f'  {rec["scenario"]:34s} -> no feasible candidate')
            continue
        top = rec['recommendations'][0]
        print(f'  {rec["scenario"]:34s} -> {top["method"]} on {top["backbone"]:10s} '
              f'acc={top["pred_accuracy"]:.4f}  vram={top["pred_peak_vram_gb"]:5.2f}GB  '
              f'[{top["confidence"]}]')

    print('\nRestricting to measured configurations changes the answer: compare '
          'standard_balanced with\nstandard_balanced_measured_only. Both are legitimate; they '
          'differ in how much evidence backs them.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
