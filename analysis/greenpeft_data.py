"""
Shared loaders for GreenPEFT analysis scripts.

One place that knows how to turn the raw files into analysis-ready frames, so every script
and table in results/ is built from the same definitions (workflow section 26: no hidden
preprocessing, every transformation reproducible in code).

Two frames matter:

  load_canonical_runs()  -- the 41 measured configurations, with the duplicate measurement
                            passes collapsed and the energy-validity filter applied.
  load_feasibility()     -- the full 60-configuration grid including the 19 OOM cells, which
                            exist only in the sweep file; the ML-ready CSV drops them.

The collapse rules match green-peft.ipynb section 4b exactly. The notebook keeps its own
inline copy on purpose -- it shows the derivation and asserts the preconditions in front of the
reader -- so if you change a rule here, change it there too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ML_READY = ROOT / 'data' / 'processed' / 'greenpeft_ml_ready_dataset.csv'
SWEEP_RAW = ROOT / 'results' / 'canonical_benchmark' / 'metrics' / 'sweep_raw.csv'

CARBON_FACTOR = 0.65          # kgCO2eq per kWh -- project assumption, not a measurement
GPU_BUDGET_GB = 15.6          # the T4 budget that defined feasibility

OUTCOME_COLS = [
    'accuracy', 'peak_gpu_memory_gb', 'wall_clock_seconds',
    'energy_kwh', 'carbon_kgco2eq', 'carbon_derived_kg',
    'energy_per_acc_point', 'vram_efficiency', 'throughput_proxy',
]
PASS_COLS = ['run_id', 'measurement_pass', 'energy_measurement_valid']

METRICS = ['accuracy', 'peak_gpu_memory_gb', 'energy_kwh', 'wall_clock_seconds']
BACKBONE_ORDER = ['tiny', 'small', 'medium', 'large']
METHOD_ORDER = ['full_ft', 'lisa', 'lora', 'lora_fa', 'qlora']


def load_canonical_runs(csv_path: Path = ML_READY) -> pd.DataFrame:
    """One row per measured configuration (method, backbone, seed).

    The benchmark measured each configuration twice. The `remeasured` pass recorded a flat
    ~10 W implied power -- the NVML idle floor rather than loaded training power -- and is
    flagged energy_measurement_valid == 0. Pooling the passes understates mean energy by ~42%
    and doubles every sample count, so they are collapsed here:

      accuracy             identical across passes (asserted) -> take either
      VRAM, wall-clock     genuine per-run measurements       -> mean, spread kept as *_pass_sd
      energy               valid telemetry pass only
      carbon               re-derived as energy * 0.65
    """
    df = pd.read_csv(csv_path)
    raw = df[(df['analysis_role'] == 'surrogate_train') & (df['is_empirical'] == 1)].copy()
    g = raw.groupby('config_id')

    if set(g.size().unique()) != {2}:
        raise ValueError(f'Expected exactly 2 measurement passes per config_id, '
                         f'found sizes {sorted(g.size().unique())}')
    if set(g['accuracy'].nunique().unique()) != {1}:
        raise ValueError('Accuracy differs between measurement passes; the collapse rule '
                         '"take either pass" is no longer valid')
    if set(g['energy_measurement_valid'].sum().unique()) != {1}:
        raise ValueError('Expected exactly one valid-energy pass per config_id')
    if not np.allclose(raw['carbon_kgco2eq'] / raw['energy_kwh'], CARBON_FACTOR, atol=1e-4):
        raise ValueError('carbon_kgco2eq is not exactly energy_kwh * 0.65')

    excluded = OUTCOME_COLS + PASS_COLS + ['config_id']
    static_cols = [c for c in raw.columns
                   if c not in excluded and g[c].nunique(dropna=False).max() == 1]

    runs = g[static_cols].first().reset_index()
    runs = runs.merge(g['accuracy'].first().rename('accuracy').reset_index(), on='config_id')

    for col in ['peak_gpu_memory_gb', 'wall_clock_seconds']:
        stats = (g[col].agg(['mean', 'std'])
                 .rename(columns={'mean': col, 'std': f'{col}_pass_sd'}).reset_index())
        runs = runs.merge(stats, on='config_id')

    valid = raw.loc[raw['energy_measurement_valid'] == 1, ['config_id', 'energy_kwh']]
    runs = runs.merge(valid, on='config_id', how='left')
    runs['carbon_kgco2eq'] = runs['energy_kwh'] * CARBON_FACTOR

    runs['seed'] = runs['config_id'].str.extract(r'seed(\d+)$')[0].astype('Int64')
    # config_base is the configuration identity WITHOUT the seed -- the correct grouping key
    # for cross-validation. config_id contains the seed and so does not separate seeds.
    runs['config_base'] = runs['config_id'].str.replace(r'_seed\d+$', '', regex=True)

    return runs.sort_values(['method', 'backbone', 'seed']).reset_index(drop=True)


def load_feasibility(csv_path: Path = SWEEP_RAW) -> pd.DataFrame:
    """The full 60-configuration grid, including the 19 that went OOM.

    OOM runs have no accuracy, energy, memory or runtime -- the run never completed. They are
    kept as rows with null metrics and status 'oom'; they are never given a substitute numeric
    value (workflow section 20).
    """
    sweep = pd.read_csv(csv_path)
    expected = {'run_id', 'method', 'backbone', 'seed', 'status'}
    missing = expected - set(sweep.columns)
    if missing:
        raise ValueError(f'{csv_path} is missing columns: {sorted(missing)}')

    sweep['feasible'] = sweep['status'].eq('ok')
    if sweep.loc[sweep['status'] == 'oom', 'accuracy'].notna().any():
        raise ValueError('An OOM run carries an accuracy value; OOM must stay null')
    return sweep


def ordered(values, order: list[str]) -> list[str]:
    """Known values in canonical order, then anything unexpected, so nothing is dropped."""
    present = list(dict.fromkeys(values))
    return [v for v in order if v in present] + [v for v in present if v not in order]
