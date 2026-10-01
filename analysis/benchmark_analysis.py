#!/usr/bin/env python
"""
Benchmark summary tables for the GreenPEFT paper (workflow spec section 20).

Writes four CSVs to results/benchmark/:

    method_summary.csv      per method: mean/std/min/max of each outcome, plus feasibility
    backbone_summary.csv    per backbone scale: the same
    feasibility_matrix.csv  method x backbone: successful runs, OOM runs, OOM rate
    seed_variance.csv       per configuration: across-seed spread of each outcome

Two sources, deliberately kept apart:
  - outcome statistics come from the 41 canonical measured configurations
  - feasibility comes from the full 60-configuration grid, because the 19 OOM cells exist
    only in the sweep file

OOM runs never receive a substitute numeric value. A configuration that did not run has no
accuracy and no energy, and averaging over only the survivors is reported alongside the OOM
rate so the survivorship is visible rather than hidden.

Usage:
    python analysis/benchmark_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from greenpeft_data import (  # noqa: E402
    BACKBONE_ORDER, GPU_BUDGET_GB, METHOD_ORDER, METRICS, ROOT,
    load_canonical_runs, load_feasibility, ordered,
)

OUT_DIR = ROOT / 'results' / 'benchmark'
AGG = ['mean', 'std', 'min', 'max', 'count']


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = ['_'.join(c).rstrip('_') if isinstance(c, tuple) else c for c in df.columns]
    return df


def by_group(runs: pd.DataFrame, sweep: pd.DataFrame, key: str,
             order: list[str]) -> pd.DataFrame:
    """Outcome statistics for one grouping key, joined to that key's feasibility counts."""
    stats = _flatten(runs.groupby(key)[METRICS].agg(AGG).reset_index())

    feas = (sweep.groupby(key)
            .agg(runs_attempted=('status', 'size'),
                 runs_successful=('feasible', 'sum'))
            .reset_index())
    feas['runs_oom'] = feas['runs_attempted'] - feas['runs_successful']
    feas['oom_rate'] = (feas['runs_oom'] / feas['runs_attempted']).round(4)

    out = feas.merge(stats, on=key, how='left')
    out[key] = pd.Categorical(out[key], categories=ordered(out[key], order), ordered=True)
    return out.sort_values(key)


def feasibility_matrix(sweep: pd.DataFrame) -> pd.DataFrame:
    """Long-form method x backbone feasibility, the source for paper Figure A."""
    grid = (sweep.groupby(['method', 'backbone'])
            .agg(runs_attempted=('status', 'size'),
                 runs_successful=('feasible', 'sum'))
            .reset_index())
    grid['runs_oom'] = grid['runs_attempted'] - grid['runs_successful']
    grid['oom_rate'] = (grid['runs_oom'] / grid['runs_attempted']).round(4)
    grid['outcome'] = grid.apply(
        lambda r: 'all_ok' if r['runs_oom'] == 0
        else ('all_oom' if r['runs_successful'] == 0 else 'mixed'), axis=1)

    grid['method'] = pd.Categorical(grid['method'],
                                    categories=ordered(grid['method'], METHOD_ORDER), ordered=True)
    grid['backbone'] = pd.Categorical(grid['backbone'],
                                      categories=ordered(grid['backbone'], BACKBONE_ORDER),
                                      ordered=True)
    return grid.sort_values(['method', 'backbone'])


def seed_variance(runs: pd.DataFrame) -> pd.DataFrame:
    """Across-seed spread per configuration -- the noise floor any method claim must clear."""
    rows = []
    for (method, backbone), g in runs.groupby(['method', 'backbone'], observed=True):
        row = {'method': method, 'backbone': backbone,
               'params_b': g['params_b'].iloc[0], 'n_seeds': len(g),
               'seeds': ','.join(str(s) for s in sorted(g['seed'].dropna().tolist()))}
        for m in METRICS:
            row[f'{m}_mean'] = g[m].mean()
            row[f'{m}_std'] = g[m].std()
            row[f'{m}_range'] = g[m].max() - g[m].min()
            # Relative spread makes targets with different units comparable.
            row[f'{m}_cv_pct'] = (g[m].std() / g[m].mean() * 100) if g[m].mean() else None
        rows.append(row)

    out = pd.DataFrame(rows)
    out['method'] = pd.Categorical(out['method'],
                                   categories=ordered(out['method'], METHOD_ORDER), ordered=True)
    out['backbone'] = pd.Categorical(out['backbone'],
                                     categories=ordered(out['backbone'], BACKBONE_ORDER),
                                     ordered=True)
    return out.sort_values(['method', 'backbone'])


def main() -> int:
    runs = load_canonical_runs()
    sweep = load_feasibility()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tables = {
        'method_summary.csv': by_group(runs, sweep, 'method', METHOD_ORDER),
        'backbone_summary.csv': by_group(runs, sweep, 'backbone', BACKBONE_ORDER),
        'feasibility_matrix.csv': feasibility_matrix(sweep),
        'seed_variance.csv': seed_variance(runs),
    }
    for name, table in tables.items():
        path = OUT_DIR / name
        table.to_csv(path, index=False)
        print(f'wrote {path.relative_to(ROOT)}  ({len(table)} rows)')

    print(f'\nSources: {len(runs)} canonical configurations (outcomes), '
          f'{len(sweep)} grid cells (feasibility: '
          f'{int(sweep["feasible"].sum())} ok / {int((~sweep["feasible"]).sum())} OOM) '
          f'under a {GPU_BUDGET_GB} GB budget.')

    print('\nFeasibility matrix (successful seeds of 3):')
    pivot = (tables['feasibility_matrix.csv']
             .pivot(index='method', columns='backbone', values='runs_successful'))
    print(pivot.to_string())

    print('\nAcross-seed coefficient of variation, worst configuration per metric:')
    sv = tables['seed_variance.csv']
    for m in METRICS:
        worst = sv.loc[sv[f'{m}_cv_pct'].idxmax()]
        print(f'  {m:20s}: {worst[f"{m}_cv_pct"]:5.2f}%  '
              f'({worst["method"]} / {worst["backbone"]})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
