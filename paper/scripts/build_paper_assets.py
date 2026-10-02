"""
Build every number, table input and figure used in paper/main.tex.

Reads only existing project files (no new experiments):
  data/processed/greenpeft_ml_ready_dataset.csv     -> measured runs (via analysis/greenpeft_data.py)
  data/processed/greenpeft_surrogate_models.joblib  -> feature list of the audited surrogate
  green_peft CLI (installed package)                -> decision-support scenarios

Writes:
  paper/figures/fig1_pipeline.pdf
  paper/figures/fig2_accuracy_energy.pdf
  paper/figures/fig3_accuracy_vram.pdf
  paper/figures/fig4_surrogate_loto.pdf
  paper/figures/fig5_pareto_scenarioA.pdf
  paper/data/*.csv                                  -> the values transcribed into the tables

Primary comparisons use only method x backbone cells in which all three seeds completed
(clean-results policy of the paper workflow). Surrogate CV is recomputed on the 41 canonical
configurations (passes collapsed, valid energy pass only).

Run from the repository root:  python paper/scripts/build_paper_assets.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper'
FIG = PAPER / 'figures'
OUT = PAPER / 'data'
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / 'analysis'))
from greenpeft_data import CARBON_FACTOR, ML_READY, load_canonical_runs  # noqa: E402

BUNDLE = ROOT / 'data' / 'processed' / 'greenpeft_surrogate_models.joblib'
TARGETS = ['accuracy', 'peak_gpu_memory_gb', 'energy_kwh', 'wall_clock_seconds']
LOG_TARGETS = {'energy_kwh', 'wall_clock_seconds'}
THRESHOLDS = {'validated': (0.70, 15.0), 'preliminary': (0.0, 50.0)}  # surrogate/validate.py

BACKBONE_LABEL = {'tiny': '0.5B', 'small': '1.1B', 'medium': '1.5B', 'large': '3.0B'}
BACKBONE_NAME = {'tiny': 'Qwen2.5-0.5B', 'small': 'TinyLlama-1.1B',
                 'medium': 'Qwen2.5-1.5B', 'large': 'Qwen2.5-3B'}
METHOD_LABEL = {'full_ft': 'Full FT', 'lora': 'LoRA', 'qlora': 'QLoRA',
                'lora_fa': 'LoRA-FA', 'lisa': 'LISA'}
METHOD_ORDER = ['full_ft', 'lora', 'qlora', 'lora_fa', 'lisa']
BACKBONE_ORDER = ['tiny', 'small', 'medium', 'large']

# Validated with the dataviz palette checker (all-pairs, light surface). Full fine-tuning is the
# reference baseline and is drawn in neutral ink on purpose; every method also has its own marker
# and direct labels, so identity never rests on colour alone.
METHOD_COLOR = {'full_ft': '#52514e', 'lora': '#2a78d6', 'qlora': '#eb6834',
                'lora_fa': '#1baf7a', 'lisa': '#4a3aa7'}
METHOD_MARKER = {'full_ft': 's', 'lora': 'o', 'qlora': '^', 'lora_fa': 'D', 'lisa': 'v'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8.5, 'axes.labelsize': 8.5,
    'axes.titlesize': 8.5, 'legend.fontsize': 7.5, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'pdf.fonttype': 42,
})


# ----------------------------------------------------------------------------- data
runs = load_canonical_runs()                       # 41 configurations
seeds = runs.groupby(['method', 'backbone']).size()
complete = seeds[seeds == 3].index                 # cells with all three seeds
primary = runs.set_index(['method', 'backbone']).loc[complete].reset_index()

audit = {
    'n_configurations': int(len(runs)),
    'n_cells_measured': int(len(seeds)),
    'n_cells_complete': int(len(complete)),
    'n_runs_complete': int(len(primary)),
    'cells_excluded_from_primary': [f'{m}|{b} (n={n})' for (m, b), n in seeds.items() if n != 3],
}

# Two-pass audit numbers quoted in the Methods section, recomputed from the ML-ready table.
raw = pd.read_csv(ML_READY)
raw = raw[(raw.analysis_role == 'surrogate_train') & (raw.is_empirical == 1)].copy()
raw['implied_w'] = raw.energy_kwh * 3.6e6 / raw.wall_clock_seconds
for p, g in raw.groupby('measurement_pass'):
    audit[f'implied_power_{p}'] = {'mean': g.implied_w.mean(), 'sd': g.implied_w.std(),
                                   'min': g.implied_w.min(), 'max': g.implied_w.max()}
valid = raw[raw.energy_measurement_valid == 1].energy_kwh
audit['energy_cv_valid_pass'] = float(valid.std() / valid.mean())
audit['energy_cv_pooled'] = float(raw.energy_kwh.std() / raw.energy_kwh.mean())
audit['dataset_shape'] = list(pd.read_csv(ML_READY).shape)
audit['accuracy_range'] = [float(runs.accuracy.min()), float(runs.accuracy.max())]

# ----------------------------------------------------------------- Table 2: results
agg = (primary.assign(energy_wh=primary.energy_kwh * 1000,
                      co2_g=primary.energy_kwh * CARBON_FACTOR * 1000)
       .groupby(['backbone', 'method'])
       [['accuracy', 'peak_gpu_memory_gb', 'energy_wh', 'wall_clock_seconds', 'co2_g']]
       .agg(['mean', 'std', 'count']))
agg.columns = [f'{a}_{b}' for a, b in agg.columns]
agg = agg.reset_index()
agg['bo'] = agg.backbone.map(BACKBONE_ORDER.index)
agg['mo'] = agg.method.map(METHOD_ORDER.index)
agg = agg.sort_values(['bo', 'mo']).drop(columns=['bo', 'mo'])
agg.to_csv(OUT / 'table2_complete_cells.csv', index=False)

# Paired within-scale comparison at 0.5B (the only scale where full FT is a complete cell).
t = agg[agg.backbone == 'tiny'].set_index('method')
ref = t.loc['full_ft']
paired = pd.DataFrame({
    'acc_delta_pp': (t.accuracy_mean - ref.accuracy_mean) * 100,
    'energy_saved_pct': (1 - t.energy_wh_mean / ref.energy_wh_mean) * 100,
    'vram_saved_pct': (1 - t.peak_gpu_memory_gb_mean / ref.peak_gpu_memory_gb_mean) * 100,
    'time_saved_pct': (1 - t.wall_clock_seconds_mean / ref.wall_clock_seconds_mean) * 100,
}).drop(index='full_ft')
paired.to_csv(OUT / 'paired_vs_fullft_0p5b.csv')


# ------------------------------------------------- Table 3: surrogate, recomputed CV
bundle = joblib.load(BUNDLE)
NUM, CAT = bundle['features']['numeric'], bundle['features']['categorical']
FEATURES = bundle['features']['all']
assert not set(FEATURES) & {'accuracy', 'peak_gpu_memory_gb', 'energy_kwh',
                            'wall_clock_seconds', 'carbon_kgco2eq'}, 'leakage'
missing = [c for c in FEATURES if c not in runs.columns]
assert not missing, f'features missing from canonical frame: {missing}'


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


def status(r2, mape):
    for name, (r2min, mmax) in THRESHOLDS.items():
        if r2 >= r2min and mape <= mmax:
            return name.upper()
    return 'UNUSABLE'


PROTOCOLS = {
    'pass': ('Pass-grouped (config_id), 5-fold', GroupKFold(n_splits=5), 'config_id'),
    'cell': ('Seed-grouped (method x backbone), 5-fold', GroupKFold(n_splits=5), 'config_base'),
    'loto': ('Leave-one-tier-out (backbone)', LeaveOneGroupOut(), 'backbone'),
}
X = runs[FEATURES]
rows, oof = [], []
for tgt in TARGETS:
    y = runs[tgt].values
    for key, (label, splitter, gcol) in PROTOCOLS.items():
        pred = np.full(len(y), np.nan)
        for tr, te in splitter.split(X, y, runs[gcol]):
            pred[te] = make_pipeline(tgt).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        mape = float(np.mean(np.abs((y - pred) / y)) * 100)
        r2 = float(r2_score(y, pred))
        rows.append({'target': tgt, 'protocol': key, 'protocol_label': label,
                     'n_rows': len(y), 'n_groups': int(runs[gcol].nunique()),
                     'MAE': mean_absolute_error(y, pred),
                     'RMSE': float(np.sqrt(np.mean((y - pred) ** 2))),
                     'R2': r2, 'MAPE_pct': mape, 'status_at_threshold': status(r2, mape)})
        if key == 'loto':
            oof.append(pd.DataFrame({'target': tgt, 'backbone': runs.backbone,
                                     'method': runs.method, 'actual': y, 'predicted': pred}))
cv = pd.DataFrame(rows)
cv.to_csv(OUT / 'table3_surrogate_cv_41configs.csv', index=False)
oof = pd.concat(oof)
oof.to_csv(OUT / 'loto_predictions_41configs.csv', index=False)
# Status is governed by the weakest protocol for each target.
RANK = ['VALIDATED', 'PRELIMINARY', 'UNUSABLE']
audit['governing_status_recomputed'] = {
    t: max(g.status_at_threshold, key=RANK.index) for t, g in cv.groupby('target')}


# ------------------------------------------------- Table 4: decision-support scenarios
SCENARIOS = [
    ('A', ['--vram', '16', '--accuracy', '0.90', '--profile', 'balanced']),
    ('A-M', ['--vram', '16', '--accuracy', '0.90', '--profile', 'balanced',
             '--min-confidence', 'HIGH']),
    ('B', ['--vram', '8', '--accuracy', '0.90', '--profile', 'strict_carbon']),
    ('C', ['--vram', '16', '--accuracy', '0.93', '--profile', 'high_accuracy']),
]
scen_rows, scen_json = [], {}
for name, args in SCENARIOS:
    res = subprocess.run(['green-peft', 'recommend', *args, '--top-k', '5', '--json'],
                         capture_output=True, text=True, check=True, cwd=ROOT)
    d = json.loads(res.stdout)
    scen_json[name] = d
    top = d['ranked'][0]
    scen_rows.append({
        'scenario': name, 'args': ' '.join(args), 'n_candidates': d['n_candidates'],
        'n_feasible': d['n_feasible'], 'model_id': top['model_id'], 'method': top['method'],
        'params_b': top['params_b'], 'pred_accuracy': top['pred_accuracy'],
        'pred_vram_gb': top['pred_peak_vram_gb'], 'pred_energy_wh': top['pred_energy_kwh'] * 1000,
        'pred_co2_g': top['pred_carbon_kgco2eq'] * 1000, 'pred_time_s': top['pred_wall_clock_s'],
        'gei': top['gei'], 'pareto': top['on_pareto_front'], 'scope': top['scope'],
        'confidence': top['confidence'], 'caveat': top.get('scope_caveat', ''),
        'energy_band_pct': top['pred_energy_kwh_band_pct'],
        'runner_up': f"{d['ranked'][1]['model_id']} {d['ranked'][1]['method']}"
                     if len(d['ranked']) > 1 else '',
    })
pd.DataFrame(scen_rows).to_csv(OUT / 'table4_scenarios.csv', index=False)
(OUT / 'scenarios_raw.json').write_text(json.dumps(scen_json, indent=1, default=str))
(OUT / 'audit_numbers.json').write_text(json.dumps(audit, indent=1, default=float))


# ======================================================================== FIGURES
def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# -- Figure 1: methodology pipeline -----------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 1.25))
ax.set_axis_off()
ax.set_xlim(0, 6)
ax.set_ylim(0, 1)
steps = [
    ('PEFT\nconfiguration', 'method, backbone,\nrank, bit width'),
    ('Fine-tuning', 'SST-2, Tesla T4,\n300 steps, 3 seeds'),
    ('Resource\nmeasurement', 'accuracy, VRAM,\nenergy, runtime'),
    ('Audited\ndataset', '41 configurations,\nvalid energy pass'),
    ('Surrogate\nmodel', '4 Ridge models,\npre-run metadata'),
    ('Decision\nsupport', 'constraints, Pareto,\npreference profile'),
]
for i, (title, sub) in enumerate(steps):
    x0 = i + 0.03
    ax.add_patch(FancyBboxPatch((x0, 0.04), 0.86, 0.92, boxstyle='round,pad=0,rounding_size=0.05',
                                linewidth=0.8, edgecolor=INK2,
                                facecolor='#eef4fc' if i in (4, 5) else '#f6f6f4'))
    ax.text(x0 + 0.43, 0.70, title, ha='center', va='center', fontsize=6.9, color=INK,
            fontweight='bold', linespacing=1.05)
    ax.text(x0 + 0.43, 0.27, sub, ha='center', va='center', fontsize=5.6, color=INK2,
            linespacing=1.1)
    if i < len(steps) - 1:
        ax.annotate('', xy=(i + 1.03, 0.5), xytext=(i + 0.89, 0.5),
                    arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.8, mutation_scale=6))
save(fig, 'fig1_pipeline')


# -- Figures 2 & 3: measured trade-offs, complete cells only --------------------------
def tradeoff(xcol, xlabel, name, offsets, xscale=1.0, budget=None):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for m in METHOD_ORDER:
        sub = agg[agg.method == m]
        if sub.empty:
            continue
        ax.errorbar(sub[f'{xcol}_mean'] * xscale, sub.accuracy_mean,
                    xerr=sub[f'{xcol}_std'] * xscale, yerr=sub.accuracy_std,
                    fmt=METHOD_MARKER[m], ms=5.5, color=METHOD_COLOR[m], mec='white', mew=0.8,
                    ecolor=METHOD_COLOR[m], elinewidth=0.8, capsize=1.8, label=METHOD_LABEL[m],
                    zorder=3)
        for _, r in sub.iterrows():
            dx, dy, ha = offsets.get((m, r.backbone), (4, -3, 'left'))
            ax.annotate(BACKBONE_LABEL[r.backbone], (r[f'{xcol}_mean'] * xscale, r.accuracy_mean),
                        xytext=(dx, dy), textcoords='offset points', fontsize=6.3, color=INK2,
                        ha=ha)
    if budget is not None:
        ax.axvline(budget, color=INK2, lw=0.8, ls='--', zorder=1)
        ax.text(budget - 0.15, 0.868, '15.6 GB budget', fontsize=6.3,
                color=INK2, rotation=90, va='bottom', ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('SST-2 validation accuracy')
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=5,
              handletextpad=0.1, columnspacing=0.7, borderaxespad=0.1, fontsize=6.8)
    save(fig, name)


# Per-point label offsets (points, points, alignment), set by eye to avoid collisions.
tradeoff('energy_wh', 'Measured GPU energy per run (Wh)', 'fig2_accuracy_energy', {
    ('lora', 'small'): (-5, 2, 'right'), ('lora_fa', 'medium'): (0, -13, 'center'),
    ('lisa', 'medium'): (0, 7, 'center'), ('qlora', 'small'): (5, -6, 'left'),
    ('lora', 'tiny'): (0, 6, 'center'), ('lora_fa', 'tiny'): (5, -5, 'left'),
    ('qlora', 'tiny'): (-5, -3, 'right'),
    ('qlora', 'large'): (-5, -8, 'right'),
})
tradeoff('peak_gpu_memory_gb', 'Measured peak GPU memory (GB)', 'fig3_accuracy_vram', {
    ('lisa', 'medium'): (-5, -9, 'right'), ('lora', 'tiny'): (5, 2, 'left'),
    ('lora_fa', 'tiny'): (5, -8, 'left'), ('qlora', 'tiny'): (-5, -8, 'right'), ('lora', 'small'): (0, 6, 'center'),
    ('lora_fa', 'medium'): (0, 6, 'center'), ('qlora', 'large'): (5, -2, 'left'),
}, budget=15.6)


# -- Figure 4: leave-one-tier-out parity -------------------------------------------
UNITS = {'accuracy': ('Accuracy', 1, ''), 'peak_gpu_memory_gb': ('Peak VRAM', 1, ' (GB)'),
         'energy_kwh': ('Energy', 1000, ' (Wh)'), 'wall_clock_seconds': ('Runtime', 1, ' (s)')}
# 2x2 at column width rather than a 1x4 strip at text width: this keeps it a
# single-column float, so it does not compete with the methodology figure and Table I
# for the scarce double-column slots.
fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.3))
axes = axes.ravel()
for ax, tgt in zip(axes, TARGETS):
    o = oof[oof.target == tgt]
    name, k, unit = UNITS[tgt]
    a, p = o.actual * k, o.predicted * k
    lo, hi = min(a.min(), p.min()), max(a.max(), p.max())
    pad = (hi - lo) * 0.06
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, lw=0.8, ls='--', zorder=1)
    ax.scatter(a, p, s=14, color='#2a78d6', edgecolor='white', linewidth=0.5, zorder=3)
    r = cv[(cv.target == tgt) & (cv.protocol == 'loto')].iloc[0]
    ax.text(0.04, 0.96, f'$R^2$ = {r.R2:.2f}\nMAPE = {r.MAPE_pct:.1f}%', transform=ax.transAxes,
            va='top', fontsize=6.6, color=INK)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(name + unit, color=INK)
    ax.tick_params(labelsize=6.5)
for ax in axes[2:]:                       # x label only on the bottom row
    ax.set_xlabel('Measured')
for ax in (axes[0], axes[2]):             # y label only on the left column
    ax.set_ylabel('Predicted')
fig.tight_layout(w_pad=0.8, h_pad=0.8)
save(fig, 'fig4_surrogate_loto')


# -- Figure 5: Pareto view of scenario A ---------------------------------------------
# The frontier is computed by the CLI over four objectives (accuracy, VRAM, energy-derived
# carbon, runtime); this is its projection onto accuracy vs energy.
dA = scen_json['A']
res = subprocess.run(['green-peft', 'recommend', '--vram', '16', '--accuracy', '0.90',
                      '--profile', 'balanced', '--top-k', '200', '--json'],
                     capture_output=True, text=True, check=True, cwd=ROOT)
cand = pd.DataFrame(json.loads(res.stdout)['ranked'])
cand.to_csv(OUT / 'fig5_scenarioA_candidates.csv', index=False)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
dom = cand[~cand.on_pareto_front]
par = cand[cand.on_pareto_front]
ax.scatter(dom.pred_energy_kwh * 1000, dom.pred_accuracy, s=18, facecolor='white',
           edgecolor='#9a9893', linewidth=0.8, label='Dominated', zorder=2)
ax.scatter(par.pred_energy_kwh * 1000, par.pred_accuracy, s=20, color='#2a78d6',
           edgecolor='white', linewidth=0.5, label='Pareto-optimal (4 objectives)', zorder=3)
top = cand.iloc[0]
ax.scatter([top.pred_energy_kwh * 1000], [top.pred_accuracy], s=90, marker='*',
           color='#eb6834', edgecolor='white', linewidth=0.6, label='Recommended (balanced)',
           zorder=4)
ax.annotate(f"{top.model_id.split('/')[-1]} + {METHOD_LABEL[top.method]}",
            (top.pred_energy_kwh * 1000, top.pred_accuracy), xytext=(6, -10),
            textcoords='offset points', fontsize=6.3, color=INK)
ax.set_xscale('log')
ax.set_xticks([1, 2, 5, 10, 20])
ax.set_xticklabels(['1', '2', '5', '10', '20'])
ax.set_xlabel('Predicted GPU energy per run (Wh, log scale)')
ax.set_ylabel('Predicted accuracy')
ax.legend(frameon=False, loc='lower right', fontsize=6.6, handletextpad=0.3)
save(fig, 'fig5_pareto_scenarioA')

print(json.dumps(audit, indent=1, default=float))
print(cv[['target', 'protocol', 'n_rows', 'n_groups', 'MAE', 'RMSE', 'R2', 'MAPE_pct',
          'status_at_threshold']].to_string(index=False))
print(agg.round(4).to_string(index=False))
print(paired.round(2).to_string())
print(pd.DataFrame(scen_rows).drop(columns=['args']).round(4).to_string(index=False))
