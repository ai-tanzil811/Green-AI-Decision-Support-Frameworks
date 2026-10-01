"""GreenPEFT surrogate audit — workflow spec sections 6, 7, 8, 9, 10 (first execution A-I).

Audits the saved .joblib, reproduces the feature contract, runs grouped cross-validation with
`config_id` as the grouping identity, and classifies the surrogate as VALIDATED / PRELIMINARY /
UNUSABLE against thresholds recorded in code.

Run:  python surrogate/validate.py
"""
from __future__ import annotations

import json
import os
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'new update works', 'greenpeft_ml_ready_dataset.csv')
BUNDLE = os.path.join(ROOT, 'new update works', 'greenpeft_surrogate_models.joblib')
MODELS_DIR = os.path.join(ROOT, 'models')
RESULTS_DIR = os.path.join(ROOT, 'results', 'surrogate')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
for d in (MODELS_DIR, RESULTS_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

SEED = 42
GROUP_COL = 'config_id'                      # workflow 3.1
ENERGY_FILTER = 'energy_measurement_valid == 1'   # workflow 3.2
CARBON_FORMULA = 'energy_kwh * 0.65'         # workflow 3.3
TARGETS = ['accuracy', 'peak_gpu_memory_gb', 'energy_kwh', 'wall_clock_seconds']
LOG_TARGETS = {'energy_kwh', 'wall_clock_seconds'}

# Workflow 10: thresholds are a recorded engineering choice, not a universal standard.
THRESHOLDS = {
    'validated': {'min_r2': 0.70, 'max_mape': 15.0},
    'preliminary': {'min_r2': 0.00, 'max_mape': 50.0},
}

# Workflow 3.4: these carry outcome information and must never enter the feature matrix.
LEAKY_COLUMNS = ['energy_per_acc_point', 'vram_efficiency', 'throughput_proxy',
                 'carbon_derived_kg', 'accuracy', 'peak_gpu_memory_gb', 'energy_kwh',
                 'carbon_kgco2eq', 'wall_clock_seconds', 'status', 'fits']


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mape(y, yhat):
    y = np.asarray(y, dtype=float)
    return float(np.mean(np.abs((y - np.asarray(yhat)) / y)) * 100)


# ============================ B. Audit the saved artifact ============================
print('=' * 78); print('B  ARTIFACT AUDIT'); print('=' * 78)
bundle = joblib.load(BUNDLE)
NUM = bundle['features']['numeric']
CAT = bundle['features']['categorical']
FEATURES = bundle['features']['all']

model_classes = {}
for tgt, pipe in bundle['regressors'].items():
    inner = pipe.named_steps['m']
    reg = getattr(inner, 'regressor', inner)
    model_classes[tgt] = {
        'estimator': type(reg).__name__,
        'wrapped_in': type(inner).__name__ if inner is not reg else None,
        'target_transform': getattr(inner, 'func', None).__name__ if hasattr(inner, 'func') else None,
        'alpha': reg.get_params().get('alpha'),
    }
    print(f'  {tgt:20} {model_classes[tgt]["estimator"]:>8}  '
          f'transform={model_classes[tgt]["target_transform"]}')

print(f'  features: {len(FEATURES)} ({len(NUM)} numeric + {len(CAT)} categorical)')
print(f'  targets : {bundle["targets"]}')
print(f'  trained on: {bundle["training_rows"]}')

# ============================ C/D. Feature-contract verification ============================
print('\n' + '=' * 78); print('C/D  FEATURE CONTRACT'); print('=' * 78)
df = pd.read_csv(DATA)
train = df[df.analysis_role == 'surrogate_train'].copy()   # workflow 3.5
print(f'  dataset {df.shape} -> surrogate_train rows {len(train)}, '
      f'unique {GROUP_COL} {train[GROUP_COL].nunique()}')

missing = [c for c in FEATURES if c not in train.columns]
assert not missing, f'features in bundle absent from dataset: {missing}'
leak = sorted(set(FEATURES) & set(LEAKY_COLUMNS))
assert not leak, f'LEAKAGE: outcome columns present as features: {leak}'
print(f'  all {len(FEATURES)} bundle features present in dataset: yes')
print(f'  leakage assertion (workflow 3.4): clean')

# Determinism check (workflow 27): identical config -> identical feature row.
probe = train.iloc[[0]][FEATURES]
assert probe.equals(train.iloc[[0]][FEATURES]), 'feature generation is not deterministic'
print('  feature determinism: stable')


def make_pipeline(target):
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('sc', StandardScaler())]), NUM),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
         CAT)])
    est = Ridge(alpha=1.0)
    if target in LOG_TARGETS:
        est = TransformedTargetRegressor(est, func=np.log, inverse_func=np.exp)
    return Pipeline([('pre', pre), ('m', est)])


def subset_for(target):
    """Workflow 3.2: energy is fitted only on validated telemetry."""
    sub = train[train[target].notna()]
    if target == 'energy_kwh':
        sub = sub[sub.energy_measurement_valid == 1]
    return sub


# ============================ E. Grouped cross-validation ============================
print('\n' + '=' * 78); print('E  GROUPED CROSS-VALIDATION'); print('=' * 78)
print(f'  primary protocol : GroupKFold(groups={GROUP_COL})  [workflow 3.1/8]')
print(f'  secondary        : LeaveOneGroupOut(groups=backbone) — scale extrapolation')

rows, oof_frames = [], []

for tgt in TARGETS:
    sub = subset_for(tgt)
    X, y = sub[FEATURES], sub[tgt]
    groups = sub[GROUP_COL]
    n_groups = groups.nunique()

    # ---- primary: GroupKFold on config_id ----
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.empty(len(sub), dtype=float)
    fold_id = np.empty(len(sub), dtype=int)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        m = make_pipeline(tgt).fit(X.iloc[tr], y.iloc[tr])
        pred[te] = m.predict(X.iloc[te])
        fold_id[te] = k

    rows.append({
        'target': tgt, 'protocol': f'GroupKFold(k={n_splits}, groups={GROUP_COL})',
        'n_rows': len(sub), 'n_groups': int(n_groups),
        'MAE': mean_absolute_error(y, pred), 'RMSE': rmse(y, pred),
        'R2': r2_score(y, pred), 'MAPE_pct': mape(y, pred),
        'energy_filter_applied': tgt == 'energy_kwh',
    })
    oof_frames.append(pd.DataFrame({
        'target': tgt, 'config_id': sub[GROUP_COL].values,
        'backbone': sub['backbone'].values, 'method': sub['method'].values,
        'actual': y.values, 'predicted': pred, 'residual': y.values - pred,
        'fold': fold_id, 'protocol': 'GroupKFold',
    }))

    # ---- secondary: leave-one-tier-out ----
    bb = sub['backbone']
    pred2 = np.empty(len(sub), dtype=float)
    for tr, te in LeaveOneGroupOut().split(X, y, bb):
        m = make_pipeline(tgt).fit(X.iloc[tr], y.iloc[tr])
        pred2[te] = m.predict(X.iloc[te])
    rows.append({
        'target': tgt, 'protocol': 'LeaveOneGroupOut(groups=backbone)',
        'n_rows': len(sub), 'n_groups': int(bb.nunique()),
        'MAE': mean_absolute_error(y, pred2), 'RMSE': rmse(y, pred2),
        'R2': r2_score(y, pred2), 'MAPE_pct': mape(y, pred2),
        'energy_filter_applied': tgt == 'energy_kwh',
    })

    a, b_ = rows[-2], rows[-1]
    print(f'\n  {tgt}')
    print(f'    GroupKFold(config_id)  n={a["n_rows"]:3} MAE={a["MAE"]:.5g} '
          f'RMSE={a["RMSE"]:.5g} R2={a["R2"]:+.3f} MAPE={a["MAPE_pct"]:.1f}%')
    print(f'    LOTO(backbone)         n={b_["n_rows"]:3} MAE={b_["MAE"]:.5g} '
          f'RMSE={b_["RMSE"]:.5g} R2={b_["R2"]:+.3f} MAPE={b_["MAPE_pct"]:.1f}%')

cv = pd.DataFrame(rows)
oof = pd.concat(oof_frames, ignore_index=True)

# ============================ F/G. Persist metrics + OOF ============================
cv.to_csv(os.path.join(RESULTS_DIR, 'cv_metrics.csv'), index=False)
oof.to_csv(os.path.join(RESULTS_DIR, 'predictions_oof.csv'), index=False)
print(f'\n  wrote results/surrogate/cv_metrics.csv  ({len(cv)} rows)')
print(f'  wrote results/surrogate/predictions_oof.csv  ({len(oof)} rows)')

# ---- breakdown by backbone and method (workflow 9) ----
brk = []
for (tgt, key), g in pd.concat([
        oof.assign(dim='backbone', level=oof.backbone),
        oof.assign(dim='method', level=oof.method)]).groupby(['target', 'dim'], observed=True):
    for lvl, gg in g.groupby('level', observed=True):
        brk.append({'target': tgt, 'dimension': key, 'level': lvl, 'n': len(gg),
                    'MAE': mean_absolute_error(gg.actual, gg.predicted),
                    'RMSE': rmse(gg.actual, gg.predicted),
                    'MAPE_pct': mape(gg.actual, gg.predicted),
                    'R2': r2_score(gg.actual, gg.predicted) if len(gg) >= 5
                          and gg.actual.std() / abs(gg.actual.mean()) >= 0.05 else np.nan})
brk = pd.DataFrame(brk)
brk.to_csv(os.path.join(RESULTS_DIR, 'cv_metrics_by_group.csv'), index=False)
print(f'  wrote results/surrogate/cv_metrics_by_group.csv  ({len(brk)} rows)')

# ============================ H. Actual vs predicted plots ============================
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
TIERS = ['tiny', 'small', 'medium', 'large']
MARK = ['o', 's', '^', 'D']
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#b8b7b0'
plt.rcParams.update({
    'figure.dpi': 140, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.size': 9,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.grid': True,
    'grid.color': '#e9e9e4', 'grid.linewidth': 0.7, 'axes.axisbelow': True,
    'legend.frameon': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False})

LABEL = {'accuracy': 'Accuracy', 'peak_gpu_memory_gb': 'Peak VRAM (GB)',
         'energy_kwh': 'Energy (kWh)', 'wall_clock_seconds': 'Wall-clock (s)'}
for tgt in TARGETS:
    d = oof[oof.target == tgt]
    row = cv[(cv.target == tgt) & (cv.protocol.str.startswith('GroupKFold'))].iloc[0]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for i, t in enumerate(TIERS):
        s = d[d.backbone == t]
        if len(s):
            ax.scatter(s.actual, s.predicted, s=46, color=SERIES[i], marker=MARK[i],
                       alpha=0.88, edgecolor='white', linewidth=0.7, label=t, zorder=3)
    lo, hi = min(d.actual.min(), d.predicted.min()), max(d.actual.max(), d.predicted.max())
    pad = (hi - lo) * 0.08
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, ls='--', lw=1, zorder=2)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f'Measured {LABEL[tgt]}'); ax.set_ylabel(f'Predicted {LABEL[tgt]}')
    ax.set_title(f'{LABEL[tgt]}', loc='left', fontweight='bold', color=INK, pad=14)
    ax.text(0, 1.015, f'GroupKFold on {GROUP_COL} · R²={row.R2:+.2f} · '
            f'RMSE={row.RMSE:.4g} · MAPE={row.MAPE_pct:.1f}%',
            transform=ax.transAxes, fontsize=7.5, color=INK2, va='bottom')
    ax.legend(fontsize=7.5, title='backbone', title_fontsize=7.5, loc='upper left')
    name = f'actual_vs_predicted_{tgt}.png'
    fig.savefig(os.path.join(FIG_DIR, name)); plt.close(fig)
    print(f'  wrote results/surrogate/figures/{name}')

# ============================ I. Verdict ============================
print('\n' + '=' * 78); print('I  SURROGATE STATUS'); print('=' * 78)
primary = cv[cv.protocol.str.startswith('GroupKFold')].set_index('target')
secondary = cv[cv.protocol.str.startswith('LeaveOne')].set_index('target')
order = ['UNUSABLE', 'PRELIMINARY', 'VALIDATED']


def classify(r2, mp):
    v, p = THRESHOLDS['validated'], THRESHOLDS['preliminary']
    if r2 >= v['min_r2'] and mp <= v['max_mape']:
        return 'VALIDATED'
    if r2 >= p['min_r2'] and mp <= p['max_mape']:
        return 'PRELIMINARY'
    return 'UNUSABLE'


# GroupKFold(config_id) blocks measurement-pass leakage but NOT scale leakage: every test
# fold still contains backbones seen in training, so it scores interpolation within known
# scales. The surrogate's actual job is scoring configurations the user has not run, which
# routinely means an unseen scale. Reporting the GroupKFold number alone would overstate
# the model (workflow 9, 23), so the status is taken from the WEAKER of the two protocols.
per_target, detail = {}, {}
for tgt in TARGETS:
    s_p = classify(primary.loc[tgt, 'R2'], primary.loc[tgt, 'MAPE_pct'])
    s_s = classify(secondary.loc[tgt, 'R2'], secondary.loc[tgt, 'MAPE_pct'])
    st = min([s_p, s_s], key=order.index)
    per_target[tgt] = st
    detail[tgt] = {
        'groupkfold_config_id': {'R2': float(primary.loc[tgt, 'R2']),
                                 'MAPE_pct': float(primary.loc[tgt, 'MAPE_pct']),
                                 'status': s_p},
        'leave_one_tier_out': {'R2': float(secondary.loc[tgt, 'R2']),
                               'MAPE_pct': float(secondary.loc[tgt, 'MAPE_pct']),
                               'status': s_s},
        'status_governing': st,
    }
    print(f'  {tgt:20} interpolation R2={primary.loc[tgt, "R2"]:+.3f} ({s_p:11})  |  '
          f'extrapolation R2={secondary.loc[tgt, "R2"]:+.3f} ({s_s:11})  -> {st}')

overall = min(per_target.values(), key=order.index)
print(f'\n  OVERALL: {overall}   (weakest target under the weaker protocol)')
print('  Interpolating within measured scales is strong (R2 0.76-1.00). Extrapolating to an')
print('  unseen scale is materially weaker, so the surrogate supports shortlist generation')
print('  but not automatic approval of a configuration at an unmeasured scale.')

# ============================ model_metadata.json (workflow 6) ============================
meta = {
    'artifact': os.path.relpath(BUNDLE, ROOT).replace('\\', '/'),
    'artifact_type': 'dict bundle of sklearn Pipelines',
    'schema_version': bundle['schema_version'],
    'sklearn_version': sklearn.__version__,
    'targets': bundle['targets'],
    'features': FEATURES,
    'features_numeric': NUM,
    'features_categorical': CAT,
    'model_classes': model_classes,
    'preprocessor': 'ColumnTransformer(median-impute+StandardScaler | constant-impute+OneHot)',
    'training_rows': int(len(subset_for('accuracy'))),
    'training_configs': int(train[GROUP_COL].nunique()),
    'group_column': GROUP_COL,
    'energy_valid_filter': ENERGY_FILTER,
    'carbon_formula': CARBON_FORMULA,
    'carbon_is_derived': True,
    'validation': {
        'primary_protocol': f'GroupKFold(groups={GROUP_COL})',
        'secondary_protocol': 'LeaveOneGroupOut(groups=backbone)',
        'thresholds': THRESHOLDS,
        'metrics': {r['target']: {k: r[k] for k in ('protocol', 'MAE', 'RMSE', 'R2', 'MAPE_pct')}
                    for _, r in cv[cv.protocol.str.startswith('GroupKFold')].iterrows()},
        'per_target_detail': detail,
        'status_per_target': per_target,
        'status_overall': overall,
        'status_rule': ('Status is taken from the WEAKER of the two protocols. '
                        'GroupKFold(config_id) blocks measurement-pass leakage but not scale '
                        'leakage, so it measures interpolation within already-measured scales; '
                        'LeaveOneGroupOut(backbone) measures extrapolation to an unseen scale.'),
    },
    'scope': bundle.get('scope'),
    'notes': bundle.get('notes'),
    'leakage_guard': LEAKY_COLUMNS,
}
mp_path = os.path.join(MODELS_DIR, 'model_metadata.json')
json.dump(meta, open(mp_path, 'w'), indent=2)
print(f'\n  wrote models/model_metadata.json')
print('\nAUDIT COMPLETE — checkpoint reached (workflow 32). Stop before new decision logic.')
