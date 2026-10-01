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
                                    build_candidates, explain, pareto_front,
                                    predict_all, recommend, score_gei)

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


# ---------------------------------------------------------------- scope & confidence
def test_training_envelope_is_recorded_and_matches_the_dataset():
    """The envelope must be derived, not hand-maintained (workflow section 26)."""
    art = GreenPEFTArtifacts.load()
    assert art.envelope is not None, (
        'no training_envelope in model_metadata.json -- '
        'run analysis/build_training_envelope.py')
    env = art.envelope
    assert env.params_b_min == 0.5 and env.params_b_max == 3.0
    assert env.families == frozenset({'qwen', 'llama'})
    assert len(env.measured_cells) == 14


@pytest.mark.parametrize('params_b,family,method,expected', [
    (1.1, 'llama', 'lora', 'MEASURED'),              # a cell that was actually measured
    (1.1, 'llama', 'full_ft', 'UNSEEN_METHOD_SCALE'),  # full_ft only ever ran at 0.5B
    (1.7, 'llama', 'lora', 'INTERPOLATED_SCALE'),    # between the 1.5B and 3.0B tiers
    (2.7, 'phi2', 'qlora', 'UNSEEN_FAMILY'),         # family never trained on
    (7.6, 'qwen', 'qlora', 'OUT_OF_RANGE_SCALE'),    # beyond the measured range
])
def test_scope_classification(params_b, family, method, expected):
    env = GreenPEFTArtifacts.load().envelope
    assert env.classify(params_b, family, method)['scope'] == expected


def test_out_of_range_candidates_are_flagged_as_lower_bound():
    """Leave-one-tier-out never left 0.5-3.0B, so beyond it the band understates the error."""
    art = GreenPEFTArtifacts.load()
    pred = predict_all(art, build_candidates(art))
    beyond = pred[pred['scope'] == 'OUT_OF_RANGE_SCALE']
    assert len(beyond), 'the zoo should contain candidates outside the measured range'
    assert beyond['scope_caveat'].str.contains('LOWER BOUND').all()
    assert (beyond['confidence'] == 'LOW').all()


def test_error_bands_widen_outside_the_measured_envelope():
    """An extrapolation must not be reported with the same precision as a measured cell."""
    art = GreenPEFTArtifacts.load()
    pred = predict_all(art, build_candidates(art))
    measured = pred[pred['scope'] == 'MEASURED']['pred_peak_vram_gb_band_pct'].iloc[0]
    extrapolated = pred[pred['scope'] == 'UNSEEN_FAMILY']['pred_peak_vram_gb_band_pct'].iloc[0]
    assert extrapolated > measured, (
        f'extrapolated band ({extrapolated}%) must exceed measured band ({measured}%)')


def test_bands_bracket_the_point_estimate():
    art = GreenPEFTArtifacts.load()
    pred = predict_all(art, build_candidates(art))
    for col in ['pred_accuracy', 'pred_peak_vram_gb', 'pred_energy_kwh', 'pred_wall_clock_s']:
        assert (pred[f'{col}_lo'] <= pred[col]).all()
        assert (pred[col] <= pred[f'{col}_hi']).all()


def test_min_confidence_keeps_only_measured_cells():
    art = GreenPEFTArtifacts.load()
    pred = predict_all(art, build_candidates(art))
    kept = apply_constraints(pred, Constraints(min_confidence='HIGH'))
    assert len(kept), 'at least the measured cells should survive a HIGH floor'
    assert (kept['scope'] == 'MEASURED').all()
    assert len(kept) < len(pred), 'the floor must actually exclude something'


def test_recommendation_surfaces_confidence_in_its_output():
    """A point estimate with no evidence label is exactly what workflow section 13 forbids."""
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, '-m', 'green_peft.cli', '--no-banner',
                        'recommend', '--vram', '16', '--accuracy', '0.90'],
                       cwd=root, capture_output=True, text=True,
                       env={**__import__('os').environ, 'PYTHONPATH': str(root)})
    assert r.returncode == 0, r.stderr
    assert 'evidence' in r.stdout and 'confidence' in r.stdout
    assert 'PRELIMINARY' in r.stdout, 'the audited surrogate status must be stated'


def test_json_output_reports_status_and_scope():
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, '-m', 'green_peft.cli', 'recommend',
                        '--vram', '16', '--json'],
                       cwd=root, capture_output=True, text=True,
                       env={**__import__('os').environ, 'PYTHONPATH': str(root)})
    payload = json.loads(r.stdout)
    assert payload['has_confidence'] is True
    assert payload['model_status'] == 'PRELIMINARY'
    assert payload['ranked'][0]['confidence'] in {'LOW', 'MEDIUM', 'HIGH'}


def test_physically_impossible_predictions_are_dropped_not_scored():
    """Ridge extrapolates linearly: the 0.135B zoo entries draw negative VRAM and the 7.6B
    entries draw accuracy above 1.0. A negative VRAM prediction satisfies every budget and
    then rescales the GEI memory objective for every other candidate, so it must not survive
    into scoring."""
    art = GreenPEFTArtifacts.load()
    pred = predict_all(art, build_candidates(art))

    flagged = pred[pred['implausible'] != '']
    assert len(flagged), 'the zoo should contain at least one non-physical extrapolation'
    assert (flagged['pred_peak_vram_gb'] <= 0).any() or (flagged['pred_accuracy'] > 1).any()

    kept = apply_constraints(pred, Constraints(max_vram_gb=16.0))
    assert (kept['pred_peak_vram_gb'] > 0).all(), 'negative VRAM survived the feasibility gate'
    assert (kept['pred_accuracy'] <= 1.0).all(), 'accuracy above 1.0 survived the gate'
    assert len(kept) < len(apply_constraints(pred.drop(columns=['implausible']),
                                             Constraints(max_vram_gb=16.0))), \
        'the plausibility check must actually exclude candidates'


def test_implausible_candidates_are_reported_not_hidden():
    """Workflow section 12: every dropped candidate needs a stated reason."""
    art = GreenPEFTArtifacts.load()
    res = recommend(art, Constraints(max_vram_gb=16.0), profile='balanced')
    assert res['n_implausible'] > 0
    assert str(res['n_implausible']) in explain(res)
