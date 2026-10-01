# GreenPEFT — Surrogate audit report

Covers tasks **A–I** of §32 of `greenpeft_agent_workflow.md`. Per that section, work stops at
this checkpoint; no decision-support logic (candidate generator, constraints, Pareto, GEI,
recommender) has been built yet.

**Verdict: the surrogate is `PRELIMINARY`.** Energy alone reaches `VALIDATED`.

Reproduce with `python surrogate/validate.py`.

---

## 1. Repository inventory (§5)

| File / directory | Purpose | Status | Used by |
|---|---|---|---|
| `sample datasets/*.csv` | 4 raw input sources | Preserved, untouched | Preprocessing notebook |
| `new update works/greenpeft_ml_ready_dataset.csv` | Harmonized dataset, 477×46 | Current | Surrogate, audit |
| `new update works/greenpeft_surrogate_models.joblib` | Current surrogate bundle, 4 regressors | **Audited** | CLI (pending integration) |
| `new update works/greenpeft_cv_metrics.json` | LOTO metrics from the preprocessing notebook | Current | Paper |
| `new update works/DATA_METHODOLOGY.md` | Data provenance and preprocessing rules | Current | Paper |
| `new update works/data_preprocessing_and_feature_engineering.ipynb` | Stages 1–8, regenerates all of the above | Current | Everything |
| `new update works/figures/fig01..07*.png` | Paper figures | Current | Paper |
| `surrogate/validate.py` | **New** — artifact audit + grouped CV | New | This report |
| `models/model_metadata.json` | **New** — machine-readable artifact metadata (§6) | New | Downstream tooling |
| `results/surrogate/*` | **New** — CV metrics, OOF predictions, diagnostics | New | This report |
| `CITATION.cff` | **New** — Kaggle + HF DOIs | New | Citation |
| `green_peft_cli/green_peft_pkg/` | Installable CLI, `recommender.py` + `cli.py` | Pre-existing, **not yet rewired** | §17 |
| `model/artifacts_export/` | Export bundle from the earlier 60-run study | Legacy, superseded | Nothing current |
| `GreenPEFT/surrogate_models.joblib` | Old surrogate (142 KB, 60-run era) | Legacy, untracked | Nothing current |
| `results/canonical_benchmark/`, `results/working_run/` | Raw per-run JSON from the original sweep | Preserved (source of truth) | Provenance |
| `Green_PEFT.ipynb`, `green-project-patched.ipynb` | Benchmark execution notebooks | Preserved | Kaggle sweep |
| `website/` | Project site | Preserved | — |

### Files removed

Only regenerable bytecode, per Rule 1 ("do not destroy existing results"):

- `__pycache__/` — contained `data_preprocessing.cpython-312.pyc`, orphaned (its source no
  longer exists) and committed to git by accident
- `green_peft_cli/green_peft_pkg/green_peft/__pycache__/`

### Removal candidates — **not** actioned, awaiting your decision

These are genuine redundancy, but all are git-tracked research outputs, so Rule 1 applies:

| Candidate | Size | Why it is a candidate | Risk |
|---|---|---|---|
| `results/working_run/archives/export_bundle/` | ~500 KB | Byte-identical copies of files in `results/working_run/` and `results/canonical_benchmark/` | Low — pure triplication |
| `model/artifacts_export/` | 197 KB | Export of the superseded 60-run study; its `.joblib` is the old model | Low — superseded, git-tracked |
| `GreenPEFT/surrogate_models.joblib` | 142 KB | Old surrogate, duplicate of the one in `model/artifacts_export/` | **Untracked** — deletion is unrecoverable |

Every raw run JSON exists in three places (`canonical_benchmark/`, `working_run/`,
`working_run/archives/export_bundle/`). Collapsing to one copy would reclaim roughly 1 MB and
remove the risk of a future script reading a stale copy.

---

## 2. Artifact audit (§6)

`greenpeft_surrogate_models.joblib` is a dict bundle, schema 3.0, built with scikit-learn 1.9.1.

| Target | Estimator | Target transform |
|---|---|---|
| `accuracy` | Ridge (α=1.0) | none |
| `peak_gpu_memory_gb` | Ridge (α=1.0) | none |
| `energy_kwh` | Ridge (α=1.0) | `log` / `exp` |
| `wall_clock_seconds` | Ridge (α=1.0) | `log` / `exp` |

Each target is a full `Pipeline` with its own `ColumnTransformer` (median-impute + standardize
for 20 numeric features; constant-impute + one-hot for `method` and `family`), so the artifact
consumes a raw config `DataFrame` with no preprocessing to reimplement at the call site.

22 features, 4 targets, fitted on 41 configurations. **No carbon model exists**, correctly —
carbon is derived as `energy_kwh × 0.65` (§3.3).

Full contents recorded in [`models/model_metadata.json`](models/model_metadata.json).

---

## 3. Feature contract (§7)

Verified rather than assumed:

- All 22 bundle features are present in the dataset.
- **Leakage assertion passes.** No outcome column (`accuracy`, `energy_kwh`, `peak_gpu_memory_gb`,
  `wall_clock_seconds`, `carbon_kgco2eq`, `status`, `fits`) and none of the four reporting ratios
  (`energy_per_acc_point`, `vram_efficiency`, `throughput_proxy`, `carbon_derived_kg`) appears in
  the feature matrix (§3.4).
- Feature generation is deterministic for a fixed configuration (§27).
- Only `analysis_role == surrogate_train` rows are used; `scale_reference` and `context_only`
  are excluded (§3.5).

---

## 4. Grouped validation (§8)

Two protocols, because they answer different questions.

**Primary — `GroupKFold(groups=config_id)`**, as specified in §3.1/§8. Blocks measurement-pass
leakage: both passes of a configuration always land in the same fold.

**Secondary — `LeaveOneGroupOut(groups=backbone)`**. Holds out an entire model-scale tier.

The energy filter `energy_measurement_valid == 1` is applied before fitting and evaluation
(§3.2), which is why energy has n=41 rows where other targets have n=82.

| Target | n | MAE | RMSE | R² | MAPE | Protocol |
|---|---|---|---|---|---|---|
| accuracy | 82 | 0.0061 | 0.0092 | **+0.764** | 0.7% | GroupKFold |
| accuracy | 82 | 0.0149 | 0.0181 | +0.082 | 1.6% | Leave-one-tier-out |
| peak_gpu_memory_gb | 82 | 0.760 | 1.043 | **+0.923** | 11.5% | GroupKFold |
| peak_gpu_memory_gb | 82 | 2.076 | 2.500 | +0.555 | 39.9% | Leave-one-tier-out |
| energy_kwh | 41 | 5.86e-05 | 7.08e-05 | **+0.996** | 2.6% | GroupKFold |
| energy_kwh | 41 | 2.41e-04 | 3.81e-04 | +0.897 | 10.6% | Leave-one-tier-out |
| wall_clock_seconds | 82 | 6.42 | 7.91 | **+0.984** | 4.6% | GroupKFold |
| wall_clock_seconds | 82 | 34.56 | 42.72 | +0.531 | 23.0% | Leave-one-tier-out |

Outputs: [`results/surrogate/cv_metrics.csv`](results/surrogate/cv_metrics.csv),
[`predictions_oof.csv`](results/surrogate/predictions_oof.csv) (287 rows with `config_id`,
`backbone`, `method`, `actual`, `predicted`, `residual`, `fold`, `target`),
[`cv_metrics_by_group.csv`](results/surrogate/cv_metrics_by_group.csv) (per backbone and method),
and four actual-vs-predicted plots in `results/surrogate/figures/`.

### The gap between the two protocols is the finding

| Target | GroupKFold R² | Leave-one-tier-out R² | Gap |
|---|---|---|---|
| accuracy | +0.764 | +0.082 | **−0.682** |
| wall_clock_seconds | +0.984 | +0.531 | −0.453 |
| peak_gpu_memory_gb | +0.923 | +0.555 | −0.368 |
| energy_kwh | +0.996 | +0.897 | −0.100 |

`GroupKFold(config_id)` blocks measurement-pass leakage but **not scale leakage**: every test
fold still contains backbones that appear in training, so it scores *interpolation within
already-measured scales*. The surrogate's actual job is scoring configurations the user has not
run, which routinely means an unmeasured scale.

Reporting the GroupKFold numbers alone would overstate the model, so **status is taken from the
weaker of the two protocols** (§9 "do not hide poor performance", §23 "claims must remain
conservative").

---

## 5. Status (§10)

Thresholds are recorded in `surrogate/validate.py` as an engineering choice, not a universal
standard: `VALIDATED` requires R² ≥ 0.70 and MAPE ≤ 15%; `PRELIMINARY` requires R² ≥ 0 and
MAPE ≤ 50%.

| Target | Interpolation | Extrapolation | Governing |
|---|---|---|---|
| `energy_kwh` | VALIDATED | VALIDATED | **VALIDATED** |
| `peak_gpu_memory_gb` | VALIDATED | PRELIMINARY | PRELIMINARY |
| `wall_clock_seconds` | VALIDATED | PRELIMINARY | PRELIMINARY |
| `accuracy` | VALIDATED | PRELIMINARY | PRELIMINARY |

**Overall: `PRELIMINARY`.**

The surrogate is sound enough to generate and rank a shortlist, and energy prediction is strong
enough to quote directly. It is not trustworthy for automatic approval of a configuration at a
scale the benchmark never measured.

---

## 6. Implementation report (§28)

**What already worked.** The preprocessing pipeline, the feature contract and the artifact
structure all held up under audit. The bundle is self-contained, the leakage guard passes, and
the `config_id` grouping identity was already present in the dataset, so the correct validation
protocol was available without changing the data. Energy prediction is genuinely strong.

**What was changed.** Nothing in the dataset or the model. Added: `surrogate/validate.py`,
`models/model_metadata.json`, the `results/surrogate/` outputs, `CITATION.cff`, and a data- and
model-availability section in `DATA_METHODOLOGY.md`. Removed: two `__pycache__` directories.

**What is experimentally validated.** Energy prediction, under both protocols
(R² 0.90 extrapolating, 0.996 interpolating; MAPE 2.6–10.6%). The measurement-validity findings
behind it — the duplicate-pass structure and the idle-floor telemetry defect — are reproducible
from the raw files.

**What remains preliminary.** Accuracy, peak VRAM and wall-clock all degrade sharply when a
scale tier is held out. Accuracy in particular falls to R² = 0.082, though its 1.6% MAPE means
the absolute error stays small — within-tier accuracy variance is so low (CV 0.005–0.016) that
R² is close to undefined there. Read accuracy by MAPE, not R².

**What additional experiment is actually necessary.** One, and it is not a full re-sweep:

1. **More model scales.** Four tiers is the binding constraint on every extrapolation result
   above. The existing 13-model zoo would roughly triple scale diversity and is the single
   change most likely to move three of the four targets from PRELIMINARY to VALIDATED.
2. **Re-measure energy with verified telemetry** (secondary). Would double the energy sample
   from 41 to 82 and remove the need for the regime exclusion — but energy is already the one
   VALIDATED target, so this is lower priority than scale diversity.

Neither justifies re-running the whole benchmark (§19).

---

## 7. Conflict to resolve before §20–21

§20 requires reporting OOM runs and OOM rate; §21 Figure A requires a feasibility matrix of
successful vs OOM configurations across method × backbone. **The current dataset cannot produce
either**: the 19 infeasible configurations were dropped at ingest on instruction, and the
feasibility classifier was removed with them.

The raw source (`sample datasets/surrogate_dataset.csv`) still contains all 120 rows, and the
pipeline restores them with a one-line change — `INCLUDE_INFEASIBLE_RUNS = True` in the
notebook's Stage 0. Re-running then regenerates the feasibility matrix and the classifier
(previously balanced accuracy 0.974, ROC-AUC 0.995).

This needs a decision before §20 and §21 can be completed as written.
