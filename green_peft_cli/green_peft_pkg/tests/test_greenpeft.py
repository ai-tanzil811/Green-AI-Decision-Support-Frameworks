"""Tests for the failure modes this package actually hit (workflow spec section 27).

Run:  python -m pytest tests/ -q
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from green_peft import __version__                                    # noqa: E402
from green_peft.banner import render                                  # noqa: E402
from green_peft.features import (ALL_FEATURES, LEAKY_COLUMNS,         # noqa: E402
                                 build_pre_run_features,
                                 estimate_trainable_param_pct)
from green_peft.recommender import (Constraints, GreenPEFTArtifacts,  # noqa: E402
                                    VRAM_SAFETY_MARGIN, apply_constraints,
                                    build_candidates, pareto_front, predict_all)

CONFIG = pd.DataFrame([{'method': 'lora', 'family': 'qwen', 'params_b': 1.5,
                        'rank': 16, 'quant_bits': 16}])


# ---------------------------------------------------------------- feature generation
def test_features_are_deterministic():
    a = build_pre_run_features(CONFIG)
    b = build_pre_run_features(CONFIG)
    pd.testing.assert_frame_equal(a, b)


def test_feature_contract_shape_and_order():
    out = build_pre_run_features(CONFIG)
    assert list(out.columns) == ALL_FEATURES
    assert len(out) == 1


def test_no_target_leakage_in_feature_contract():
    """The bug class that matters most: an outcome column reaching the feature matrix."""
    assert not (LEAKY_COLUMNS & set(ALL_FEATURES))


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match='missing required'):
        build_pre_run_features(CONFIG.drop(columns=['params_b']))


def test_memory_proxy_is_weights_plus_optimizer_plus_gradient():
    f = build_pre_run_features(CONFIG).iloc[0]
    assert f.memory_proxy_gb == pytest.approx(
        f.weight_bytes_gb + f.optimizer_state_gb + f.gradient_gb)


def test_full_finetune_trains_every_parameter():
    assert estimate_trainable_param_pct('full_ft', 7.0) == 1.0


def test_trainable_fraction_never_negative_when_extrapolating():
    """Log-linear interpolation clamps outside the measured range; a linear fit would
    otherwise cross zero and produce a negative fraction, which is not possible."""
    for method in ('lora', 'qlora', 'lora_fa', 'lisa'):
        for params_b in (0.01, 0.5, 7.0, 70.0, 500.0):
            assert estimate_trainable_param_pct(method, params_b) > 0


# ---------------------------------------------------------------- carbon
def test_carbon_is_derived_from_energy_not_modelled():
    art = GreenPEFTArtifacts.load()
    assert 'carbon_kgco2eq' not in art.models, 'carbon must not be a separate model'
    pred = predict_all(art, build_candidates(art))
    assert np.allclose(pred.pred_carbon_kgco2eq,
                       pred.pred_energy_kwh * art.grid_carbon_kg_per_kwh)


# ---------------------------------------------------------------- feasibility gate
def test_nan_vram_prediction_is_dropped_not_silently_kept():
    """NaN >= x is False in numpy, so a NaN must be excluded explicitly and counted."""
    df = pd.DataFrame({'pred_peak_vram_gb': [4.0, np.nan, 40.0],
                       'pred_accuracy': [0.9] * 3, 'pred_carbon_kgco2eq': [0.001] * 3,
                       'pred_wall_clock_s': [100.0] * 3})
    kept = apply_constraints(df, Constraints(), gpu_vram_gb=16)
    assert list(kept.index) == [0], 'only the in-budget, non-NaN row may survive'


def test_safety_margin_is_applied_to_the_budget():
    cap = 16.0
    usable = cap * (1 - VRAM_SAFETY_MARGIN)
    df = pd.DataFrame({'pred_peak_vram_gb': [usable - 0.1, usable + 0.1],
                       'pred_accuracy': [0.9, 0.9], 'pred_carbon_kgco2eq': [0.001, 0.001],
                       'pred_wall_clock_s': [100.0, 100.0]})
    assert list(apply_constraints(df, Constraints(), gpu_vram_gb=cap).index) == [0]


# ---------------------------------------------------------------- pareto
def test_pareto_front_on_known_dominance():
    df = pd.DataFrame({
        'pred_accuracy':       [0.90, 0.95, 0.80],
        'pred_peak_vram_gb':   [8.0, 4.0, 16.0],
        'pred_carbon_kgco2eq': [0.002, 0.001, 0.004],
        'pred_wall_clock_s':   [100.0, 90.0, 200.0]})
    front = pareto_front(df)
    assert bool(front.iloc[1]), 'row 1 dominates on every axis and must be on the front'
    assert not bool(front.iloc[0]) and not bool(front.iloc[2]), 'rows 0 and 2 are dominated'


# ---------------------------------------------------------------- artifacts
def test_bundled_model_loads_without_an_artifacts_dir():
    art = GreenPEFTArtifacts.load()
    assert set(art.models) >= {'accuracy', 'peak_gpu_memory_gb',
                               'energy_kwh', 'wall_clock_seconds'}
    assert art.schema_version == '3.0'


def test_recommendation_only_returns_feasible_candidates():
    art = GreenPEFTArtifacts.load()
    pred = predict_all(art, build_candidates(art))
    kept = apply_constraints(pred, Constraints(max_vram_gb=16.0))
    assert (kept.pred_peak_vram_gb <= 16.0 * (1 - VRAM_SAFETY_MARGIN)).all()


# ---------------------------------------------------------------- banner
def test_banner_has_no_ansi_codes_when_color_disabled():
    assert '\033[' not in render(version=__version__, color=False)


def test_banner_names_the_author():
    assert 'Ashraful Islam Tanzil' in render(color=False)


def test_banner_goes_to_stderr_so_json_stays_parseable():
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, '-m', 'green_peft.cli', 'recommend',
                        '--vram', '16', '--json'],
                       cwd=root, capture_output=True, text=True,
                       env={**__import__('os').environ, 'PYTHONPATH': str(root)})
    json.loads(r.stdout)                      # raises if the banner leaked into stdout
    assert 'GreenPEFT' in r.stderr
