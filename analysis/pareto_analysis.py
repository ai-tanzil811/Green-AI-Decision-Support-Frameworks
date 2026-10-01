#!/usr/bin/env python
"""
Pareto analysis for GreenPEFT (workflow spec section 14).

Writes two CSVs to results/pareto/:

    pareto_candidates.csv   every candidate the decision engine can score, with its predicted
                            outcomes, pareto_optimal flag, GEI under each preference profile,
                            and its scope/confidence label
    pareto_measured.csv     the same analysis over the 41 MEASURED configurations -- the
                            empirical frontier for the paper's Figure E, with no surrogate
                            involved

Objectives: maximise accuracy, minimise VRAM, minimise carbon, minimise wall-clock. Carbon
stands in for energy because the two are related by a fixed factor under this project's
assumption; counting both would double-weight the same axis (workflow section 14).

Dominance uses green_peft.recommender.pareto_front, the same implementation the CLI uses, so
the measured frame is renamed to the engine's `pred_*` column names rather than getting a
second copy of the sorting logic.

Usage:
    python analysis/pareto_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / 'green_peft_cli' / 'green_peft_pkg'))

from greenpeft_data import CARBON_FACTOR, ROOT, load_canonical_runs  # noqa: E402
from green_peft.recommender import (  # noqa: E402
    GEI_PROFILES, Constraints, GreenPEFTArtifacts, apply_constraints,
    build_candidates, pareto_front, predict_all, score_gei,
)

OUT_DIR = ROOT / 'results' / 'pareto'

OBJECTIVES = ['pred_accuracy', 'pred_peak_vram_gb', 'pred_carbon_kgco2eq', 'pred_wall_clock_s']


def score_every_profile(df: pd.DataFrame) -> pd.DataFrame:
    """GEI under each shipped preference profile, so one table serves all three."""
    out = df.copy()
    for name, weights in GEI_PROFILES.items():
        scored, _ = score_gei(df, weights)
        out[f'gei_{name}'] = scored['gei']
    return out


def candidate_frontier() -> pd.DataFrame:
    art = GreenPEFTArtifacts.load()
    predicted = predict_all(art, build_candidates(art))

    # Physically impossible extrapolations are excluded before dominance: a negative VRAM
    # prediction dominates everything on the memory axis and would corrupt the whole front.
    scorable = apply_constraints(predicted, Constraints())
    excluded = len(predicted) - len(scorable)

    scorable = scorable.copy()
    scorable['pareto_optimal'] = pareto_front(scorable)
    scorable = score_every_profile(scorable)

    cols = (['backbone', 'model_id', 'family', 'params_b', 'method', 'rank', 'quant_bits']
            + OBJECTIVES + ['pred_energy_kwh', 'pareto_optimal']
            + [f'gei_{n}' for n in GEI_PROFILES]
            + [c for c in ('confidence', 'scope', 'scope_reason', 'scope_caveat')
               if c in scorable.columns])
    out = scorable[cols].sort_values('gei_balanced', ascending=False)
    out.attrs['excluded'] = excluded
    return out


def measured_frontier() -> pd.DataFrame:
    """Empirical frontier over configurations that were actually run. No surrogate involved."""
    runs = load_canonical_runs()
    agg = (runs.groupby(['method', 'backbone', 'family', 'params_b'], as_index=False)
           .agg(n_seeds=('config_id', 'count'),
                accuracy=('accuracy', 'mean'),
                peak_gpu_memory_gb=('peak_gpu_memory_gb', 'mean'),
                energy_kwh=('energy_kwh', 'mean'),
                wall_clock_seconds=('wall_clock_seconds', 'mean')))
    agg['carbon_kgco2eq'] = agg['energy_kwh'] * CARBON_FACTOR

    # Rename to the engine's column names so dominance has one implementation, not two.
    renamed = agg.rename(columns={
        'accuracy': 'pred_accuracy',
        'peak_gpu_memory_gb': 'pred_peak_vram_gb',
        'carbon_kgco2eq': 'pred_carbon_kgco2eq',
        'wall_clock_seconds': 'pred_wall_clock_s',
        'energy_kwh': 'pred_energy_kwh',
    })
    renamed['pareto_optimal'] = pareto_front(renamed)
    renamed = score_every_profile(renamed)

    out = renamed.rename(columns={
        'pred_accuracy': 'accuracy',
        'pred_peak_vram_gb': 'peak_gpu_memory_gb',
        'pred_carbon_kgco2eq': 'carbon_kgco2eq',
        'pred_wall_clock_s': 'wall_clock_seconds',
        'pred_energy_kwh': 'energy_kwh',
    })
    return out.sort_values('gei_balanced', ascending=False)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = candidate_frontier()
    measured = measured_frontier()

    for name, table in [('pareto_candidates.csv', candidates),
                        ('pareto_measured.csv', measured)]:
        path = OUT_DIR / name
        table.to_csv(path, index=False)
        print(f'wrote {path.relative_to(ROOT)}  ({len(table)} rows, '
              f'{int(table["pareto_optimal"].sum())} Pareto-optimal)')

    print(f'\n{candidates.attrs["excluded"]} candidates were excluded before dominance for '
          'physically impossible predictions.')

    print('\nEmpirical Pareto frontier over measured configurations '
          '(accuracy up; VRAM, carbon, time down):')
    front = measured[measured['pareto_optimal']]
    show = ['method', 'backbone', 'params_b', 'n_seeds', 'accuracy',
            'peak_gpu_memory_gb', 'energy_kwh', 'wall_clock_seconds']
    print(front[show].round(5).to_string(index=False))
    print(f'\n{len(front)}/{len(measured)} measured configurations are non-dominated. '
          'A configuration being dominated\nmeans another measured option was at least as good '
          'on every axis -- not that the method is bad.')

    if 'confidence' in candidates.columns:
        pf = candidates[candidates['pareto_optimal']]
        print('\nPredicted frontier by evidence level:')
        print('  ' + ', '.join(f'{n} {s}' for s, n in pf['confidence'].value_counts().items()))
        print('  Only the HIGH-confidence rows sit on cells that were actually measured; the '
              'rest are\n  surrogate extrapolations and carry the error bands in '
              'models/model_metadata.json.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
