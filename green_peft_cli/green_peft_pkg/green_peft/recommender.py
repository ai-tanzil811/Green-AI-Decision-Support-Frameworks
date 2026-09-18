"""
GreenPEFT decision engine.

Loads the surrogate models trained in Cell 13 (surrogate_models.joblib) and the
model/method catalog written by Cell 3 (configs/backbones.yaml, configs/methods/*.yaml),
then answers: "given VRAM / carbon / accuracy / time constraints, which PEFT
configuration should I run?" -- WITHOUT running it. This is RQ3 + RQ4 together:
zero-shot prediction (surrogate) feeding constraint-aware prescription (this module).

Design notes:
- Candidates are drawn from the FULL model_zoo in backbones.yaml, not just the tiers
  that were actually benchmarked. Predicting for an untested backbone is the entire
  point of having a surrogate instead of just looking up aggregated_gei.csv.
- GEI normalization (S_Acc, S_Mem, S_C, S_T) follows the formula in the README exactly:
  higher-is-better for accuracy, lower-is-better for memory/carbon/time. Normalization
  bounds are taken from the FEASIBLE, CONSTRAINT-FILTERED candidate set -- not the whole
  catalog -- because GEI is meant to score trade-offs among options the user could
  actually choose, not options that were already ruled out.
- A candidate whose predicted VRAM/accuracy/carbon/time we don't trust (feasibility
  classifier says P(fits) < FEASIBILITY_THRESHOLD) is dropped before scoring, not
  penalized within GEI -- an infeasible config isn't a worse option, it isn't an option.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml

CAT_FEATURES = ['method', 'family']
NUM_FEATURES = ['params_b', 'rank', 'quant_bits', 'is_adapter_method']
FEASIBILITY_THRESHOLD = 0.5

# Grid/cost assumptions -- mirror Cell 2 defaults. Override via GreenPEFTArtifacts if
# your notebook used different constants, so predicted carbon/cost stay consistent
# with what was actually measured.
DEFAULT_GRID_CARBON_KG_PER_KWH = 0.650
DEFAULT_ELECTRICITY_USD_PER_KWH = 0.12
DEFAULT_GPU_RENTAL_USD_PER_HOUR = 0.35

GEI_PROFILES = {
    # (w_accuracy, w_memory, w_carbon, w_time) -- must sum to 1.0
    'balanced':      (0.35, 0.25, 0.25, 0.15),
    'strict_carbon':  (0.20, 0.20, 0.50, 0.10),
    'high_accuracy':  (0.60, 0.15, 0.15, 0.10),
}


def _method_rank(method_cfg: dict) -> int:
    return int(method_cfg.get('rank', 0))


def _method_quant_bits(method_name: str) -> int:
    if method_name == 'qlora':
        return 4
    if method_name == 'full_ft':
        return 32
    return 16


def _method_is_adapter(method_name: str) -> int:
    return int(method_name in ('lora', 'qlora', 'lora_fa'))


@dataclass
class GreenPEFTArtifacts:
    """Everything the engine needs, loaded from one export directory.

    Expected layout (this is exactly what Cells 1-3 and 13 already write):
        artifacts_dir/
          results/surrogate_models.joblib
          configs/backbones.yaml        (has a top-level `model_zoo:` key)
          configs/methods/*.yaml
    """
    model_zoo: dict
    method_configs: dict
    models: dict                      # {'fits':..., 'accuracy':..., 'peak_gpu_memory_gb':..., 'energy_kwh':..., ['wall_clock_seconds':...]}
    grid_carbon_kg_per_kwh: float = DEFAULT_GRID_CARBON_KG_PER_KWH
    electricity_usd_per_kwh: float = DEFAULT_ELECTRICITY_USD_PER_KWH
    gpu_rental_usd_per_hour: float = DEFAULT_GPU_RENTAL_USD_PER_HOUR

    @classmethod
    def load(cls, artifacts_dir: str | Path) -> 'GreenPEFTArtifacts':
        root = Path(artifacts_dir)
        joblib_path = root / 'results' / 'surrogate_models.joblib'
        backbones_path = root / 'configs' / 'backbones.yaml'
        methods_dir = root / 'configs' / 'methods'

        if not joblib_path.exists():
            raise FileNotFoundError(
                f'{joblib_path} not found. Run Cell 13 in the notebook and download/export '
                f'the peft_bench/ folder (or just results/ + configs/) to this location.')
        if not backbones_path.exists():
            raise FileNotFoundError(f'{backbones_path} not found (from Cell 3).')

        models = joblib.load(joblib_path)
        backbones_yaml = yaml.safe_load(open(backbones_path))
        model_zoo = backbones_yaml.get('model_zoo') or backbones_yaml.get('backbones')

        method_configs = {}
        if methods_dir.exists():
            for p in methods_dir.glob('*.yaml'):
                cfg = yaml.safe_load(open(p))
                method_configs[cfg['method']] = cfg
        else:
            raise FileNotFoundError(f'{methods_dir} not found (from Cell 3).')

        return cls(model_zoo=model_zoo, method_configs=method_configs, models=models)


@dataclass
class Constraints:
    max_vram_gb: Optional[float] = None
    max_carbon_kgco2eq: Optional[float] = None
    min_accuracy: Optional[float] = None
    max_time_seconds: Optional[float] = None


def build_candidates(artifacts: GreenPEFTArtifacts,
                     backbones: Optional[list[str]] = None,
                     methods: Optional[list[str]] = None) -> pd.DataFrame:
    """Cross-product of model_zoo x methods with surrogate feature columns filled in."""
    backbones = backbones or list(artifacts.model_zoo.keys())
    methods = methods or list(artifacts.method_configs.keys())

    rows = []
    for bname in backbones:
        bb = artifacts.model_zoo[bname]
        for mname in methods:
            mcfg = artifacts.method_configs.get(mname, {'method': mname})
            rows.append({
                'backbone': bname, 'model_id': bb.get('model_id', bname),
                'method': mname, 'family': bb.get('family', 'unknown'),
                'params_b': float(bb['params_b']),
                'rank': _method_rank(mcfg),
                'quant_bits': _method_quant_bits(mname),
                'is_adapter_method': _method_is_adapter(mname),
            })
    return pd.DataFrame(rows)


def predict_all(artifacts: GreenPEFTArtifacts, candidates: pd.DataFrame) -> pd.DataFrame:
    """Run every surrogate model over the candidate table; derive carbon and cost."""
    X = candidates[CAT_FEATURES + NUM_FEATURES]
    out = candidates.copy()

    fits_model = artifacts.models.get('fits')
    if fits_model is not None:
        proba = fits_model.predict_proba(X)
        classes = list(fits_model.named_steps['model'].classes_) if hasattr(fits_model, 'named_steps') \
            else list(fits_model.classes_)
        fit_idx = classes.index(1) if 1 in classes else -1
        out['p_fits'] = proba[:, fit_idx]
    else:
        out['p_fits'] = np.nan

    for target, col in [('accuracy', 'pred_accuracy'),
                        ('peak_gpu_memory_gb', 'pred_peak_vram_gb'),
                        ('energy_kwh', 'pred_energy_kwh'),
                        ('wall_clock_seconds', 'pred_wall_clock_s')]:
        model = artifacts.models.get(target)
        out[col] = model.predict(X) if model is not None else np.nan

    out['pred_carbon_kgco2eq'] = out['pred_energy_kwh'] * artifacts.grid_carbon_kg_per_kwh
    hours = out['pred_wall_clock_s'].fillna(0) / 3600.0
    out['pred_cost_usd'] = (out['pred_energy_kwh'] * artifacts.electricity_usd_per_kwh
                            + hours * artifacts.gpu_rental_usd_per_hour)
    return out


def apply_constraints(df: pd.DataFrame, constraints: Constraints,
                      gpu_vram_gb: Optional[float] = None) -> pd.DataFrame:
    """Constraint filtering (RQ3): drop candidates the surrogate doesn't trust or that
    violate the user's stated budget. Every drop is explainable -- callers can diff
    df vs the return value to see exactly which rows were cut and why, by re-checking
    each condition."""
    out = df[df['p_fits'] >= FEASIBILITY_THRESHOLD].copy()
    vram_cap = constraints.max_vram_gb or gpu_vram_gb
    if vram_cap is not None:
        out = out[out['pred_peak_vram_gb'] <= vram_cap]
    if constraints.max_carbon_kgco2eq is not None:
        out = out[out['pred_carbon_kgco2eq'] <= constraints.max_carbon_kgco2eq]
    if constraints.min_accuracy is not None:
        out = out[out['pred_accuracy'] >= constraints.min_accuracy]
    if constraints.max_time_seconds is not None:
        out = out[out['pred_wall_clock_s'] <= constraints.max_time_seconds]
    return out


def _time_available(df: pd.DataFrame) -> bool:
    return 'pred_wall_clock_s' in df.columns and df['pred_wall_clock_s'].notna().any()


def pareto_front(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: True where no other row is at-least-as-good on every ACTIVE axis
    and strictly better on at least one (accuracy up; VRAM, carbon, [time] down).
    If no wall-clock model was trained (pred_wall_clock_s is all-NaN), time is dropped
    from the comparison entirely rather than left in as NaN -- NaN comparisons are
    always False in numpy, which would silently make every candidate "non-dominated."""
    if df.empty:
        return pd.Series([], dtype=bool)
    acc = df['pred_accuracy'].values
    mem = df['pred_peak_vram_gb'].values
    car = df['pred_carbon_kgco2eq'].values
    use_time = _time_available(df)
    tim = df['pred_wall_clock_s'].values if use_time else None
    n = len(df)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_or_equal = (acc[j] >= acc[i]) and (mem[j] <= mem[i]) and (car[j] <= car[i])
            strictly_better = (acc[j] > acc[i]) or (mem[j] < mem[i]) or (car[j] < car[i])
            if use_time:
                better_or_equal = better_or_equal and (tim[j] <= tim[i])
                strictly_better = strictly_better or (tim[j] < tim[i])
            if better_or_equal and strictly_better:
                dominated[i] = True
                break
    return pd.Series(~dominated, index=df.index)


def _normalize(series: pd.Series, higher_is_better: bool) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(1.0, index=series.index)   # only one distinct value -> no trade-off to score
    s = (series - lo) / (hi - lo)
    return s if higher_is_better else (1.0 - s)


def score_gei(df: pd.DataFrame, weights: tuple[float, float, float, float]) -> tuple[pd.DataFrame, bool]:
    """Adds S_Acc, S_Mem, S_C, [S_T] and gei columns, normalized over THIS df only --
    call this on the constraint-filtered candidate set, not the full catalog.
    Returns (scored_df, time_was_used). If no wall-clock surrogate was trained, the
    time term is dropped from the sum and the remaining three weights are renormalized
    to sum to 1 -- silently defaulting a missing objective to 0 would understate GEI
    for every candidate uniformly, and dropping the row entirely would discard otherwise
    valid accuracy/memory/carbon predictions over a target this run never produced."""
    w_acc, w_mem, w_c, w_t = weights
    out = df.copy()
    out['S_Acc'] = _normalize(out['pred_accuracy'], higher_is_better=True)
    out['S_Mem'] = _normalize(out['pred_peak_vram_gb'], higher_is_better=False)
    out['S_C'] = _normalize(out['pred_carbon_kgco2eq'], higher_is_better=False)

    use_time = _time_available(out)
    if use_time:
        out['S_T'] = _normalize(out['pred_wall_clock_s'], higher_is_better=False)
        out['gei'] = w_acc * out['S_Acc'] + w_mem * out['S_Mem'] + w_c * out['S_C'] + w_t * out['S_T']
    else:
        out['S_T'] = np.nan
        renorm = w_acc + w_mem + w_c
        out['gei'] = (w_acc * out['S_Acc'] + w_mem * out['S_Mem'] + w_c * out['S_C']) / renorm
    return out, use_time


def recommend(artifacts: GreenPEFTArtifacts, constraints: Constraints,
             profile: str = 'balanced', custom_weights: Optional[tuple] = None,
             gpu_vram_gb: Optional[float] = None,
             backbones: Optional[list[str]] = None, methods: Optional[list[str]] = None,
             top_k: int = 3) -> dict:
    weights = custom_weights or GEI_PROFILES.get(profile)
    if weights is None:
        raise ValueError(f'Unknown profile "{profile}", choose from {list(GEI_PROFILES)} '
                         f'or pass custom_weights=(w_acc, w_mem, w_carbon, w_time)')
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f'weights must sum to 1.0, got {weights} (sum={sum(weights):.3f})')

    candidates = build_candidates(artifacts, backbones, methods)
    predicted = predict_all(artifacts, candidates)
    filtered = apply_constraints(predicted, constraints, gpu_vram_gb=gpu_vram_gb)

    result = {
        'n_candidates': len(predicted), 'n_feasible': len(filtered),
        'constraints': constraints, 'profile': profile, 'weights': weights,
        'ranked': None, 'pareto_only': None, 'dropped_summary': None,
    }
    if filtered.empty:
        # Explain WHY nothing survived rather than just returning empty -- this is the
        # difference between a decision engine and a silent failure.
        reasons = []
        no_fit = predicted[predicted['p_fits'] < FEASIBILITY_THRESHOLD]
        reasons.append(f"{len(no_fit)}/{len(predicted)} candidates predicted infeasible "
                       f"(p_fits < {FEASIBILITY_THRESHOLD})")
        vram_cap = constraints.max_vram_gb or gpu_vram_gb
        if vram_cap is not None:
            over = predicted[predicted['pred_peak_vram_gb'] > vram_cap]
            reasons.append(f"{len(over)}/{len(predicted)} exceed {vram_cap} GB VRAM cap")
        if constraints.min_accuracy is not None:
            under = predicted[predicted['pred_accuracy'] < constraints.min_accuracy]
            reasons.append(f"{len(under)}/{len(predicted)} predicted below "
                           f"{constraints.min_accuracy} accuracy floor")
        result['dropped_summary'] = reasons
        return result

    scored, time_used = score_gei(filtered, weights)
    scored['on_pareto_front'] = pareto_front(filtered)
    scored = scored.sort_values('gei', ascending=False)

    result['ranked'] = scored.head(top_k)
    result['pareto_only'] = scored[scored['on_pareto_front']].sort_values('gei', ascending=False)
    result['time_objective_used'] = time_used
    return result


def explain(result: dict) -> str:
    """Human-readable summary of a recommend() result."""
    if result['ranked'] is None:
        lines = [f"No candidate satisfies every constraint out of {result['n_candidates']} tried.",
                 'Reasons:']
        lines += [f'  - {r}' for r in result['dropped_summary']]
        lines.append('Loosen one constraint (VRAM cap, carbon cap, or accuracy floor) and retry.')
        return '\n'.join(lines)

    top = result['ranked'].iloc[0]
    lines = [
        f"Recommendation ({result['profile']} profile, weights={result['weights']}):",
        f"  {top['method']} on {top['backbone']} ({top['model_id']}, {top['params_b']}B params)",
        f"  predicted accuracy   : {top['pred_accuracy']:.4f}",
        f"  predicted peak VRAM  : {top['pred_peak_vram_gb']:.2f} GB",
        f"  predicted energy     : {top['pred_energy_kwh']:.6f} kWh",
        f"  predicted carbon     : {top['pred_carbon_kgco2eq']:.6f} kgCO2eq",
        f"  predicted wall-clock : {top['pred_wall_clock_s']:.1f} s"
        if not pd.isna(top['pred_wall_clock_s']) else "  predicted wall-clock : (no time model trained)",
        f"  GEI score            : {top['gei']:.4f}"
        + ('  [on Pareto front]' if top['on_pareto_front'] else '  [dominated by another feasible option -- see note below]'),
        f"  {result['n_feasible']}/{result['n_candidates']} candidates satisfied all constraints, "
        f"{result['pareto_only'].shape[0]} of those are Pareto-optimal.",
    ]
    if not result.get('time_objective_used', True):
        lines.append('  Note: no wall-clock surrogate is in this artifacts export, so GEI and the '
                     'Pareto front were computed over accuracy/VRAM/carbon only (weights renormalized '
                     'over those three). Add wall_clock_seconds to Cell 13\'s REGRESSION_TARGETS and '
                     're-run to include a time objective.')
    if not top['on_pareto_front']:
        lines.append('  Note: top GEI pick is not Pareto-optimal under these exact weights -- this can '
                     'happen when weights trade off two close options; check pareto_only for alternatives.')
    return '\n'.join(lines)
