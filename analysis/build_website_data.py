#!/usr/bin/env python
"""
Export the data layer the showcase website runs on.

The website is a static site: it cannot unpickle the surrogate bundle, so the full candidate
grid is predicted here, once, and written out as JSON. The browser then reproduces only the
parts of the decision engine that are pure arithmetic -- the feasibility gate, the Pareto
test and GEI normalisation -- so what a visitor sees is the same computation the CLI runs,
not a mock-up of it.

Two files are written into website/data/:

    candidates.js    the 70-candidate zoo x method grid with surrogate predictions, error
                     bands, scope and confidence -- exactly what predict_all() produces.
    evidence.js      measured benchmark tables, the feasibility grid, surrogate CV metrics
                     and the empirical back-test summary, all read from results/.

They are written as `window.<NAME> = {...};` assignments rather than bare .json so the page
loads identically from a Vercel deployment and from a local file:// open -- a static site
should not need a server just to read its own data.

Figures already generated under results/ are copied to website/figures/ so the site serves
the real artifacts rather than redrawn approximations.

Run with the interpreter whose scikit-learn matches models/model_metadata.json's
`sklearn_version`; a mismatched unpickle is refused rather than silently trusted.

Usage:
    python analysis/build_website_data.py
    python analysis/build_website_data.py --check    # verify committed JSON is current
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / 'green_peft_cli' / 'green_peft_pkg'))

from greenpeft_data import ROOT  # noqa: E402
from green_peft.recommender import (  # noqa: E402
    DEFAULT_ELECTRICITY_USD_PER_KWH, DEFAULT_GPU_RENTAL_USD_PER_HOUR, GEI_PROFILES,
    VRAM_SAFETY_MARGIN, GreenPEFTArtifacts, build_candidates, predict_all,
)

OUT_DIR = ROOT / 'website' / 'data'
FIG_DIR = ROOT / 'website' / 'figures'

# (source, destination name, caption) -- the figures the evidence section displays.
FIGURES = [
    (ROOT / 'results' / 'benchmark' / 'figures' / 'figA_feasibility_matrix.png',
     'figA_feasibility_matrix.png',
     'Figure A - Which (method, backbone) cells completed on a 16 GB T4 and which went OOM.'),
    (ROOT / 'results' / 'pareto' / 'figures' / 'figE_pareto_frontier.png',
     'figE_pareto_frontier.png',
     'Figure E - Measured Pareto frontier projected onto energy and accuracy.'),
    (ROOT / 'results' / 'canonical_benchmark' / 'figures' / 'benchmark_overview.png',
     'benchmark_overview.png',
     'Benchmark overview - accuracy, peak VRAM, energy and wall-clock across the 60-run sweep.'),
    (ROOT / 'results' / 'surrogate' / 'figures' / 'actual_vs_predicted_peak_gpu_memory_gb.png',
     'actual_vs_predicted_peak_gpu_memory_gb.png',
     'Surrogate validation - out-of-fold predicted vs measured peak VRAM.'),
    (ROOT / 'results' / 'surrogate' / 'figures' / 'actual_vs_predicted_energy_kwh.png',
     'actual_vs_predicted_energy_kwh.png',
     'Surrogate validation - out-of-fold predicted vs measured training energy.'),
    (ROOT / 'results' / 'surrogate' / 'figures' / 'actual_vs_predicted_accuracy.png',
     'actual_vs_predicted_accuracy.png',
     'Surrogate validation - out-of-fold predicted vs measured SST-2 accuracy.'),
    (ROOT / 'results' / 'surrogate' / 'figures' / 'actual_vs_predicted_wall_clock_seconds.png',
     'actual_vs_predicted_wall_clock_seconds.png',
     'Surrogate validation - out-of-fold predicted vs measured wall-clock time.'),
]

# Columns carried into the browser. Everything the engine needs downstream of prediction,
# and nothing it does not -- the payload is fetched on every page load.
CANDIDATE_COLS = [
    'backbone', 'model_id', 'family', 'params_b', 'method', 'rank', 'quant_bits',
    'pred_accuracy', 'pred_peak_vram_gb', 'pred_energy_kwh', 'pred_carbon_kgco2eq',
    'pred_wall_clock_s', 'pred_cost_usd',
    'pred_accuracy_band_pct', 'pred_peak_vram_gb_band_pct',
    'pred_energy_kwh_band_pct', 'pred_wall_clock_s_band_pct',
    'scope', 'confidence', 'scope_reason', 'scope_caveat', 'implausible',
]

METHOD_LABEL = {
    'full_ft': 'Full fine-tuning',
    'lora': 'LoRA',
    'qlora': 'QLoRA',
    'lora_fa': 'LoRA-FA',
    'lisa': 'LISA',
}


def _round(value, digits):
    """Trim float noise so the committed JSON diffs only when the numbers move."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return round(float(value), digits)


def _records(df: pd.DataFrame, digits: int = 6) -> list[dict]:
    out = []
    for row in df.to_dict(orient='records'):
        clean = {}
        for k, v in row.items():
            if isinstance(v, (int, bool)) and not isinstance(v, bool):
                clean[k] = int(v)
            elif isinstance(v, bool):
                clean[k] = bool(v)
            elif isinstance(v, float):
                clean[k] = _round(v, digits)
            elif pd.isna(v):
                clean[k] = None
            else:
                clean[k] = v
        out.append(clean)
    return out


def build_candidates_payload() -> dict:
    """Predict the whole zoo x method grid with the packaged surrogate bundle."""
    meta = json.loads((ROOT / 'models' / 'model_metadata.json').read_text(encoding='utf-8'))

    artifacts = GreenPEFTArtifacts.load()      # packaged v3 bundle, same as the CLI default
    grid = build_candidates(artifacts)
    predicted = predict_all(artifacts, grid)

    missing = [c for c in CANDIDATE_COLS if c not in predicted.columns]
    if missing:
        raise RuntimeError(f'predict_all() did not produce expected columns: {missing}')

    envelope = meta['training_envelope']
    validation = meta['validation']

    return {
        'generated_from': 'green_peft.recommender.predict_all (packaged surrogate bundle)',
        'schema_version': artifacts.schema_version,
        'model_status': artifacts.model_status,
        'engine': {
            'vram_safety_margin': VRAM_SAFETY_MARGIN,
            'grid_carbon_kg_per_kwh': artifacts.grid_carbon_kg_per_kwh,
            'electricity_usd_per_kwh': DEFAULT_ELECTRICITY_USD_PER_KWH,
            'gpu_rental_usd_per_hour': DEFAULT_GPU_RENTAL_USD_PER_HOUR,
            'gei_profiles': {k: list(v) for k, v in GEI_PROFILES.items()},
            'gei_weight_order': ['accuracy', 'memory', 'carbon', 'time'],
        },
        'envelope': {
            'params_b_min': envelope['params_b']['min'],
            'params_b_max': envelope['params_b']['max'],
            'families': envelope['families'],
            'methods': envelope['methods'],
            'measured_cells': sorted(envelope['measured_cells']),
            'backbone_tiers': envelope['backbone_tiers'],
            'gpu_total_gb': envelope['gpu_total_gb'],
            'n_configs': envelope['n_configs'],
            'n_measurement_rows': envelope['n_measurement_rows'],
            'task': envelope['task'],
        },
        'status_per_target': validation['status_per_target'],
        'method_labels': METHOD_LABEL,
        'candidates': _records(predicted[CANDIDATE_COLS], digits=8),
    }


def build_evidence_payload() -> dict:
    """Measured results, read straight from the artifacts reproduce.py regenerates."""
    results = ROOT / 'results'

    measured = pd.read_csv(results / 'pareto' / 'pareto_measured.csv')
    feasibility = pd.read_csv(results / 'benchmark' / 'feasibility_matrix.csv')
    by_method = pd.read_csv(results / 'benchmark' / 'method_summary.csv')
    by_backbone = pd.read_csv(results / 'benchmark' / 'backbone_summary.csv')
    seed_var = pd.read_csv(results / 'benchmark' / 'seed_variance.csv')
    cv = pd.read_csv(results / 'surrogate' / 'cv_metrics.csv')
    backtest = pd.read_csv(results / 'recommendation_validation.csv')
    manifest = json.loads(
        (results / 'canonical_benchmark' / 'manifest.json').read_text(encoding='utf-8'))
    scenarios = json.loads(
        (results / 'recommendations' / 'scenarios.json').read_text(encoding='utf-8'))

    summary_cols = ['runs_attempted', 'runs_successful', 'runs_oom', 'oom_rate',
                    'accuracy_mean', 'accuracy_std', 'peak_gpu_memory_gb_mean',
                    'peak_gpu_memory_gb_std', 'energy_kwh_mean',
                    'wall_clock_seconds_mean']

    bt = (backtest.assign(abs_pct_error=backtest['pct_error'].abs())
                  .groupby('target')
                  .agg(n=('target', 'size'),
                       within_band=('within_band', 'mean'),
                       mean_abs_pct_error=('abs_pct_error', 'mean'),
                       median_abs_pct_error=('abs_pct_error', 'median'))
                  .reset_index())

    return {
        'manifest': {
            'platform': manifest['platform'],
            'gpu': manifest['gpu'],
            'n_raw_runs': manifest['n_raw_runs'],
            'seeds': manifest['seeds'],
            'train_steps': manifest['train_steps'],
            'batch_size': manifest['batch_size'],
            'methods': manifest['methods'],
            'backbones': manifest['backbones'],
            'grid_carbon_kg_per_kwh': manifest['grid_carbon_kg_per_kwh'],
            'generated_at': manifest['generated_at'],
        },
        'measured': _records(measured, digits=8),
        'feasibility': _records(feasibility, digits=4),
        'by_method': _records(by_method[['method'] + summary_cols], digits=6),
        'by_backbone': _records(by_backbone[['backbone'] + summary_cols], digits=6),
        'seed_variance': _records(seed_var, digits=6),
        'cv_metrics': _records(cv, digits=8),
        'backtest': _records(bt, digits=6),
        'backtest_n_configs': int(backtest['config_id'].nunique()),
        'scenarios': scenarios,
    }


def copy_figures() -> list[dict]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for src, name, caption in FIGURES:
        if not src.exists():
            raise FileNotFoundError(f'{src} is missing -- run python reproduce.py first')
        shutil.copyfile(src, FIG_DIR / name)
        manifest.append({'file': f'figures/{name}', 'caption': caption,
                         'source': str(src.relative_to(ROOT)).replace('\\', '/')})
    return manifest


def write(path: Path, payload: dict, check: bool, global_name: str) -> bool:
    text = (f'/* Generated by analysis/build_website_data.py -- do not edit by hand. */\n'
            f'window.{global_name} = '
            + json.dumps(payload, indent=1, sort_keys=False) + ';\n')
    if check:
        current = path.read_text(encoding='utf-8') if path.exists() else ''
        same = current == text
        print(f'{"OK  " if same else "DIFF"}  {path.relative_to(ROOT)}')
        return same
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    print(f'wrote {path.relative_to(ROOT)}  ({len(text) / 1024:.0f} KB)')
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true',
                    help='Verify the committed JSON matches the artifacts; write nothing.')
    args = ap.parse_args()

    candidates = build_candidates_payload()
    evidence = build_evidence_payload()
    evidence['figures'] = copy_figures() if not args.check else [
        {'file': f'figures/{name}', 'caption': caption,
         'source': str(src.relative_to(ROOT)).replace('\\', '/')}
        for src, name, caption in FIGURES]

    ok = write(OUT_DIR / 'candidates.js', candidates, args.check, 'GREENPEFT_CANDIDATES')
    ok &= write(OUT_DIR / 'evidence.js', evidence, args.check, 'GREENPEFT_EVIDENCE')

    if not args.check:
        n = len(candidates['candidates'])
        bad = sum(1 for c in candidates['candidates'] if c['implausible'])
        print(f'\n{n} candidates predicted, {bad} flagged physically implausible, '
              f'{len(evidence["measured"])} measured configurations, '
              f'{len(FIGURES)} figures copied.')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
