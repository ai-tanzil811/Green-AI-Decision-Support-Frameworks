"""Pre-run feature generation — the single source of truth for both training and inference.

Workflow spec section 7: "Every feature used for prediction must be knowable before the
training run starts", and "The training and inference paths must call the same function."

Every feature here is derived from configuration metadata alone. Nothing observed at run time
(accuracy, energy, peak VRAM, wall-clock) is ever read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Feature contract, in the order the fitted ColumnTransformer expects.
NUM_FEATURES = [
    'params_b', 'log_params_b', 'quant_bits', 'rank', 'log1p_rank',
    'is_adapter_method', 'adapter_capacity', 'trainable_param_pct', 'trainable_frac',
    'trainable_params_b', 'weight_bytes_gb', 'optimizer_state_gb', 'gradient_gb',
    'memory_proxy_gb', 'vram_pressure', 'params_x_bits', 'log_params_x_bits',
    'is_quantized', 'is_full_finetune', 'gpu_total_gb',
]
CAT_FEATURES = ['method', 'family']
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

# Columns that carry outcome information. Asserted absent from the feature matrix.
LEAKY_COLUMNS = frozenset({
    'accuracy', 'peak_gpu_memory_gb', 'energy_kwh', 'carbon_kgco2eq', 'wall_clock_seconds',
    'status', 'fits', 'energy_per_acc_point', 'vram_efficiency', 'throughput_proxy',
    'carbon_derived_kg',
})

# Measured trainable-parameter fractions per (method, params_b), from the benchmark.
# NOTE: the column is named `_pct` but is stored as a FRACTION (full_ft = 1.0). The fitted
# model divides it by 100 again, so `trainable_frac` is 100x smaller than the true fraction.
# That is a naming quirk, not a bug: it was applied identically when the model was fitted,
# so it must be reproduced here exactly. Do not "correct" it without refitting.
_TRAINABLE_ANCHORS = {
    'full_ft': [(0.5, 1.000000)],
    'lisa':    [(0.5, 0.724441), (1.1, 0.936649), (1.5, 0.848823)],
    'lora':    [(0.5, 0.004362), (1.1, 0.004340), (1.5, 0.002817)],
    'lora_fa': [(0.5, 0.001589), (1.1, 0.001565), (1.5, 0.001039)],
    'qlora':   [(0.5, 0.006822), (1.1, 0.008132), (1.5, 0.004884), (3.0, 0.004324)],
}

ADAPTER_METHODS = frozenset({'lora', 'qlora', 'lora_fa'})


def estimate_trainable_param_pct(method: str, params_b: float) -> float:
    """Interpolate the trainable fraction from measured anchors, log-linear in model size.

    Outside the measured range the nearest anchor is held flat rather than extrapolated:
    a linear fit through two points can go negative, which is not a possible fraction.
    """
    anchors = _TRAINABLE_ANCHORS.get(method)
    if not anchors:
        return 1.0 if method == 'full_ft' else 0.005
    if len(anchors) == 1:
        return anchors[0][1]
    xs = np.log10([a[0] for a in anchors])
    ys = np.array([a[1] for a in anchors], dtype=float)
    return float(np.interp(np.log10(max(params_b, 1e-6)), xs, ys, left=ys[0], right=ys[-1]))


def build_pre_run_features(configs: pd.DataFrame, gpu_total_gb: float = 15.6) -> pd.DataFrame:
    """Expand raw configuration rows into the full pre-run feature matrix.

    Required columns: method, family, params_b, rank, quant_bits.
    Optional: is_adapter_method, trainable_param_pct, gpu_total_gb (each derived if absent).
    """
    required = {'method', 'family', 'params_b', 'rank', 'quant_bits'}
    missing = required - set(configs.columns)
    if missing:
        raise ValueError(f'missing required configuration columns: {sorted(missing)}')

    df = configs.copy()
    df['params_b'] = df['params_b'].astype(float)
    df['rank'] = df['rank'].astype(float)
    df['quant_bits'] = df['quant_bits'].astype(float)

    if 'gpu_total_gb' not in df.columns:
        df['gpu_total_gb'] = float(gpu_total_gb)
    if 'is_adapter_method' not in df.columns:
        df['is_adapter_method'] = df['method'].isin(ADAPTER_METHODS).astype(float)
    if 'trainable_param_pct' not in df.columns:
        df['trainable_param_pct'] = [
            estimate_trainable_param_pct(m, p)
            for m, p in zip(df['method'], df['params_b'])
        ]

    # -- scale --
    df['log_params_b'] = np.log10(df['params_b'])
    df['params_x_bits'] = df['params_b'] * df['quant_bits']
    df['log_params_x_bits'] = np.log10(df['params_x_bits'])

    # -- memory physics: Adam holds 2 fp32 moments + 1 gradient per TRAINABLE parameter --
    df['weight_bytes_gb'] = df['params_b'] * 1e9 * (df['quant_bits'] / 8) / 1e9
    df['trainable_frac'] = (df['trainable_param_pct'] / 100.0).fillna(0.0)
    df['trainable_params_b'] = df['params_b'] * df['trainable_frac']
    df['optimizer_state_gb'] = df['trainable_params_b'] * 1e9 * 8 / 1e9
    df['gradient_gb'] = df['trainable_params_b'] * 1e9 * 4 / 1e9
    df['memory_proxy_gb'] = (df['weight_bytes_gb'] + df['optimizer_state_gb']
                             + df['gradient_gb'])
    df['vram_pressure'] = df['memory_proxy_gb'] / df['gpu_total_gb'].replace(0, np.nan)

    # -- method topology --
    df['is_quantized'] = (df['quant_bits'] < 16).astype(int)
    df['adapter_capacity'] = df['rank'] * df['is_adapter_method']
    df['log1p_rank'] = np.log1p(df['rank'])
    df['is_full_finetune'] = (df['method'] == 'full_ft').astype(int)

    leaked = LEAKY_COLUMNS & set(ALL_FEATURES)
    if leaked:                                   # workflow 3.4
        raise AssertionError(f'target leakage in feature contract: {sorted(leaked)}')

    absent = [c for c in ALL_FEATURES if c not in df.columns]
    if absent:
        raise AssertionError(f'feature builder failed to produce: {absent}')
    return df[ALL_FEATURES]
