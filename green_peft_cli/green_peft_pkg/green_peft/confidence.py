"""
Per-candidate scope and confidence for GreenPEFT surrogate predictions.

The surrogate is fitted on 41 configurations spanning four model scales (0.5-3.0B), two
model families (qwen, llama) and five methods, all on one 15.6 GB T4. The candidate zoo in
configs/backbones.yaml is deliberately wider than that -- predicting for an unmeasured
backbone is the whole point of having a surrogate. But a prediction for Mistral-7B and a
prediction for the exact tier the model was fitted on are not equally trustworthy, and the
engine must not present them identically (workflow sections 13 and 23).

This module answers two questions per candidate:

  1. Where does it sit relative to the training envelope?  -> `scope`
  2. How wrong is the prediction likely to be?             -> `confidence` + error bands

Both answers are derived from `training_envelope` and the measured cross-validation error in
models/model_metadata.json -- never hardcoded here. The bands are the surrogate's own
out-of-sample MAPE, so they are honest about what was actually tested rather than being an
invented uncertainty.

A caveat the bands cannot express on their own: the extrapolation protocol
(leave-one-backbone-out) only ever held out a tier *inside* 0.5-3.0B. It never tested
prediction beyond that range, so for OUT_OF_RANGE_SCALE candidates the reported band is a
lower bound on the real error, not an estimate of it. `scope_caveat` says so explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# Ordered weakest-to-strongest; a candidate takes the weakest scope that applies.
SCOPE_ORDER = [
    'OUT_OF_RANGE_SCALE',   # params_b outside the measured 0.5-3.0B span
    'UNSEEN_FAMILY',        # architecture family never in training
    'UNSEEN_METHOD_SCALE',  # method and scale both measured, but never together
    'INTERPOLATED_SCALE',   # inside the span, not on a measured tier
    'MEASURED',             # exactly a measured (method, tier) cell
]

SCOPE_CONFIDENCE = {
    'MEASURED': 'HIGH',
    'INTERPOLATED_SCALE': 'MEDIUM',
    'UNSEEN_METHOD_SCALE': 'MEDIUM',
    'UNSEEN_FAMILY': 'LOW',
    'OUT_OF_RANGE_SCALE': 'LOW',
}

# Which validation protocol's error applies to each scope. MEASURED candidates are the
# interpolation case the GroupKFold protocol measures; everything else is extrapolation.
SCOPE_PROTOCOL = {
    'MEASURED': 'groupkfold_config_id',
    'INTERPOLATED_SCALE': 'leave_one_tier_out',
    'UNSEEN_METHOD_SCALE': 'leave_one_tier_out',
    'UNSEEN_FAMILY': 'leave_one_tier_out',
    'OUT_OF_RANGE_SCALE': 'leave_one_tier_out',
}

SCOPE_REASON = {
    'MEASURED':
        'this (method, scale) cell was measured directly',
    'INTERPOLATED_SCALE':
        'scale sits between measured tiers',
    'UNSEEN_METHOD_SCALE':
        'method and scale were each measured, but never together',
    'UNSEEN_FAMILY':
        'model family never appears in training data',
    'OUT_OF_RANGE_SCALE':
        'parameter count falls outside the measured range',
}

TARGET_COLUMN = {
    'accuracy': 'pred_accuracy',
    'peak_gpu_memory_gb': 'pred_peak_vram_gb',
    'energy_kwh': 'pred_energy_kwh',
    'wall_clock_seconds': 'pred_wall_clock_s',
}

# Physically possible ranges for each prediction. The shipped regressors are linear (Ridge),
# so far outside the training range they extrapolate straight through these limits: the zoo's
# 0.135B entries draw NEGATIVE peak VRAM, and its 7.6B entries draw accuracy above 1.0.
#
# This matters beyond cosmetics. A candidate predicted at -1.0 GB passes any VRAM budget
# trivially, and because GEI normalisation takes its bounds from the feasible set, one
# negative value rescales the memory objective for every other candidate. Such a prediction
# is not a cheap option, it is a broken one, so it is dropped with a stated reason rather
# than scored (workflow section 12: never discard a candidate silently).
PLAUSIBLE_RANGE = {
    'pred_accuracy': (0.0, 1.0),
    'pred_peak_vram_gb': (0.0, None),
    'pred_energy_kwh': (0.0, None),
    'pred_wall_clock_s': (0.0, None),
}


def plausibility_violations(row: pd.Series) -> str:
    """Semicolon-separated list of physically impossible predictions; '' when all are sane."""
    bad = []
    for col, (lo, hi) in PLAUSIBLE_RANGE.items():
        if col not in row.index:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        if lo is not None and value <= lo:
            bad.append(f'{col}={value:.4g} <= {lo:g}')
        elif hi is not None and value > hi:
            bad.append(f'{col}={value:.4g} > {hi:g}')
    return '; '.join(bad)


@dataclass
class TrainingEnvelope:
    """The measured region of configuration space, plus per-target out-of-sample error."""
    params_b_min: float
    params_b_max: float
    families: frozenset
    methods: frozenset
    backbone_tiers: dict          # {'tiny': 0.5, ...}
    measured_cells: frozenset     # {'lora|tiny', ...}
    mape_by_protocol: dict        # {target: {protocol: mape_pct}}
    status_by_target: dict        # {target: 'VALIDATED' | 'PRELIMINARY' | ...}
    tier_tolerance: float = 1e-6  # params_b match tolerance for "on a measured tier"

    @classmethod
    def from_metadata(cls, meta: dict) -> 'TrainingEnvelope':
        env = meta.get('training_envelope')
        if not env:
            raise KeyError(
                "model_metadata.json has no 'training_envelope' block. Regenerate it with "
                "analysis/build_training_envelope.py so confidence reporting has a measured "
                "envelope to compare against.")

        detail = meta.get('validation', {}).get('per_target_detail', {})
        mape, status = {}, {}
        for target, blocks in detail.items():
            mape[target] = {p: b['MAPE_pct'] for p, b in blocks.items()
                            if isinstance(b, dict) and 'MAPE_pct' in b}
            status[target] = blocks.get('status_governing', 'unknown')

        return cls(
            params_b_min=float(env['params_b']['min']),
            params_b_max=float(env['params_b']['max']),
            families=frozenset(env['families']),
            methods=frozenset(env['methods']),
            backbone_tiers=dict(env['backbone_tiers']),
            measured_cells=frozenset(env['measured_cells']),
            mape_by_protocol=mape,
            status_by_target=status,
        )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'TrainingEnvelope':
        path = path or Path(__file__).parent / 'data' / 'model_metadata.json'
        with open(path, encoding='utf-8') as f:
            return cls.from_metadata(json.load(f))

    # --- scope classification ----------------------------------------------------
    def _tier_for(self, params_b: float) -> Optional[str]:
        """The measured tier whose parameter count this candidate matches, if any."""
        for tier, tier_params in self.backbone_tiers.items():
            if abs(float(tier_params) - params_b) <= self.tier_tolerance:
                return tier
        return None

    def classify(self, params_b: float, family: str, method: str) -> dict:
        """Weakest applicable scope for one candidate, with a human-readable reason."""
        params_b = float(params_b)
        reasons = []
        scopes = []

        if not (self.params_b_min <= params_b <= self.params_b_max):
            scopes.append('OUT_OF_RANGE_SCALE')
            reasons.append(f'{params_b:g}B outside measured '
                           f'{self.params_b_min:g}-{self.params_b_max:g}B')
        if family not in self.families:
            scopes.append('UNSEEN_FAMILY')
            reasons.append(f'family "{family}" not in {sorted(self.families)}')

        tier = self._tier_for(params_b)
        if tier is None:
            if 'OUT_OF_RANGE_SCALE' not in scopes:
                scopes.append('INTERPOLATED_SCALE')
                reasons.append(f'{params_b:g}B between measured tiers')
        elif f'{method}|{tier}' in self.measured_cells:
            scopes.append('MEASURED')
            reasons.append(f'({method}, {tier}) measured directly')
        else:
            scopes.append('UNSEEN_METHOD_SCALE')
            reasons.append(f'({method}, {tier}) never measured together')

        if method not in self.methods:
            scopes.append('UNSEEN_METHOD_SCALE')
            reasons.append(f'method "{method}" not in {sorted(self.methods)}')

        scope = min(scopes, key=SCOPE_ORDER.index)
        return {
            'scope': scope,
            'confidence': SCOPE_CONFIDENCE[scope],
            'scope_reason': '; '.join(reasons),
        }

    # --- error bands -------------------------------------------------------------
    def band_pct(self, target: str, scope: str) -> Optional[float]:
        """Expected relative error (MAPE %) for this target under this scope."""
        protocol = SCOPE_PROTOCOL[scope]
        by_protocol = self.mape_by_protocol.get(target, {})
        if protocol in by_protocol:
            return float(by_protocol[protocol])
        # Fall back to the worst measured protocol rather than silently reporting no band.
        return max(by_protocol.values()) if by_protocol else None


def annotate(df: pd.DataFrame, envelope: TrainingEnvelope) -> pd.DataFrame:
    """Add scope, confidence and per-target error bands to a predicted-candidate frame.

    Adds: scope, confidence, scope_reason, scope_caveat, and for each predicted target a
    `<col>_band_pct` plus `<col>_lo` / `<col>_hi` interval.
    """
    out = df.copy()
    if out.empty:
        for c in ['scope', 'confidence', 'scope_reason', 'scope_caveat', 'implausible']:
            out[c] = pd.Series(dtype='object')
        return out

    classified = pd.DataFrame(
        [envelope.classify(r.params_b, r.family, r.method) for r in out.itertuples()],
        index=out.index)
    out = pd.concat([out, classified], axis=1)

    # The extrapolation protocol never left the 0.5-3.0B span, so beyond it the band
    # understates the error. Say so on the candidate itself, not only in the docs.
    out['scope_caveat'] = ''
    beyond = out['scope'] == 'OUT_OF_RANGE_SCALE'
    out.loc[beyond, 'scope_caveat'] = (
        'error band is a LOWER BOUND: the surrogate was never validated outside '
        f'{envelope.params_b_min:g}-{envelope.params_b_max:g}B')

    for target, col in TARGET_COLUMN.items():
        if col not in out.columns:
            continue
        bands = out['scope'].map(lambda s: envelope.band_pct(target, s))
        out[f'{col}_band_pct'] = bands
        # Half-width from the magnitude, so the interval brackets the estimate even where the
        # regressor extrapolated to a negative value (those rows are dropped as implausible
        # below, but the band must not silently invert before that happens).
        half = out[col].abs() * bands / 100.0
        out[f'{col}_lo'] = out[col] - half
        out[f'{col}_hi'] = out[col] + half

    out['implausible'] = out.apply(plausibility_violations, axis=1)
    return out


def summarize(df: pd.DataFrame) -> str:
    """One-line-per-scope tally, for the end of a recommendation report."""
    if df.empty or 'scope' not in df.columns:
        return ''
    counts = df['scope'].value_counts()
    parts = [f'{counts[s]} {s}' for s in SCOPE_ORDER[::-1] if s in counts]
    return ', '.join(parts)
