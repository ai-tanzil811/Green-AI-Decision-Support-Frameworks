# GreenPEFT — Agent Work Specification

## 0. Mission

You are an implementation/research agent working on the **GreenPEFT** project:

> **GreenPEFT: A Multi-Objective Green AI Decision Support Framework**

The project is no longer just a benchmark. The goal is to build a **sustainability-aware decision-support system for selecting PEFT configurations under practical constraints** such as GPU VRAM, minimum accuracy, energy, training time, and user preference.

The core system should eventually perform:

```text
User Constraints
      ↓
Candidate Configuration Generation
      ↓
Pre-run Feature Engineering
      ↓
Surrogate Prediction (.joblib)
      ↓
Constraint Filtering
      ↓
Pareto Frontier
      ↓
Preference / GEI
      ↓
Recommendation + Trade-off Explanation
      ↓
Optional Real-GPU Validation
      ↓
Model/Data Update
```

Do not treat the surrogate `.joblib` as the entire system. It is one component inside the decision-support pipeline.

---

# 1. Current project state

The current GreenPEFT data pipeline has already produced:

- `greenpeft_ml_ready_dataset.csv`
- a trained `.joblib` model bundle / surrogate artifact
- `DATA_METHODOLOGY.md`
- an existing preprocessing and feature-engineering notebook

The methodology document states that the harmonized dataset is:

- **477 rows × 46 columns**
- **82 rows** currently assigned to `surrogate_train`
- **369 rows** assigned to `scale_reference`
- **26 rows** assigned to `context_only`

The four input sources have different roles. The project's own PEFT fine-tuning sweep is the gold-standard empirical source. The inference leaderboard is scale context only, and the frontier pretraining-energy source is context only.

Do **not** combine measurements from different physical regimes merely because their column names look similar.

---

# 2. Primary empirical benchmark

The project's own benchmark covers:

```text
4 backbone scales
×
5 methods
×
3 seeds
=
60 unique configurations
```

Backbones:

```text
tiny    = 0.5B
small   = 1.1B
medium  = 1.5B
large   = 3.0B
```

Methods:

```text
full_ft
lisa
lora
lora_fa
qlora
```

GPU budget:

```text
15.6 GB usable GPU memory
NVIDIA T4 environment
```

Task:

```text
SST-2 classification
```

The original empirical CSV supplied for the project contains the following outcome fields:

```text
status
fits
accuracy
peak_gpu_memory_gb
energy_kwh
carbon_kgco2eq
wall_clock_seconds
```

The benchmark contains 41 successful configurations and 19 OOM configurations.

Important observed feasibility pattern:

```text
0.5B:
  full_ft  ✓
  lisa     ✓
  lora     ✓
  lora_fa  ✓
  qlora    ✓

1.1B:
  full_ft  OOM
  lisa     ✓
  lora     ✓
  lora_fa  ✓
  qlora    ✓

1.5B:
  full_ft  OOM
  lisa     ✓
  lora     mixed: 1 OOM, 2 successful
  lora_fa  ✓
  qlora    ✓

3.0B:
  full_ft  OOM
  lisa     OOM
  lora     OOM
  lora_fa  OOM
  qlora    ✓
```

Do not describe any method as globally "best" from this table. It is a multi-objective and feasibility study.

---

# 3. Critical data-quality rules

The existing methodology identified several important issues. Preserve these rules in all downstream work.

## 3.1 Measurement passes

The benchmark source contains:

```text
60 configurations measured twice
=
120 benchmark rows
```

The two measurements are not two independent configurations.

Every pair shares a `config_id`.

Never use a random row-level train/test split that allows one measurement pass of the same configuration to enter training and the other pass to enter testing.

Use:

```text
config_id
```

as the grouping identity.

Preferred validation:

```text
GroupKFold
or
GroupShuffleSplit
```

with the group set to `config_id`.

## 3.2 Energy telemetry validity

The methodology found that the two measurement regimes disagree strongly for energy.

The remeasured pass reports an almost constant ~10 W implied power and is interpreted as the NVML idle floor rather than loaded training power.

Therefore:

```text
energy_measurement_valid == 1
```

must be respected for energy modeling.

Do not train the energy surrogate on rows where:

```text
energy_measurement_valid == 0
```

Keep those rows in the dataset for auditability, but exclude them from valid energy-target fitting.

## 3.3 Carbon is derived

The methodology found:

```text
carbon_kgco2eq ≈ energy_kwh × 0.65
```

Carbon should therefore NOT be treated as an independent machine-learning target.

Use:

```text
predicted_carbon = predicted_energy × 0.65
```

when the project uses the same fixed grid-intensity assumption.

Clearly label this as derived carbon.

## 3.4 No target leakage

The following are reporting/derived variables and must not become model input features:

```text
energy_per_acc_point
vram_efficiency
throughput_proxy
carbon_derived_kg
```

They contain outcome information and therefore leak targets.

The feature-generation pipeline must assert that target-derived columns are excluded.

## 3.5 Do not mix source roles

Respect:

```text
analysis_role == surrogate_train
```

for empirical surrogate fitting.

Do not use:

```text
scale_reference
context_only
```

as if they were empirical PEFT fine-tuning observations.

The inference leaderboard contains inference measurements, not PEFT fine-tuning measurements.

The frontier-energy source contains pretraining-from-scratch energy, not single-GPU PEFT training energy.

---

# 4. Immediate priority

DO NOT start by running a new large Kaggle sweep.

The immediate job is to determine whether the existing `.joblib` surrogate is valid enough to support a preliminary decision-support layer.

The execution order is:

```text
1. Inspect current repository
2. Inspect dataset schema and provenance
3. Inspect .joblib contents
4. Reproduce feature generation
5. Perform grouped surrogate validation
6. Diagnose model weaknesses
7. Improve/retrain surrogate only if needed
8. Build candidate generator
9. Build constraint engine
10. Build Pareto engine
11. Build GEI/preference engine
12. Build recommendation engine
13. Integrate into CLI
14. Add recommendation validation mode
15. Generate figures/tables/reports
16. Identify only the most valuable missing real experiments
```

Do not reorder these casually.

---

# 5. Step 1 — Repository audit

Before changing anything:

- inspect the repository tree
- find the current CSV
- find the `.joblib`
- find the preprocessing notebook
- find existing surrogate training code
- find current CLI code
- find existing analysis scripts
- find README and methodology files

Do not delete or overwrite existing research outputs.

Create a small inventory:

```text
file
purpose
status
used by
```

If the project already has equivalent functionality, extend it instead of duplicating it.

---

# 6. Step 2 — Audit the `.joblib`

The agent must inspect the saved Joblib artifact before retraining.

Determine exactly:

```text
artifact type
contained objects
preprocessor
feature names
target names
models
model class
hyperparameters
training metadata
scalers/encoders
version information if available
```

Produce a machine-readable metadata file:

```text
models/model_metadata.json
```

At minimum include:

```json
{
  "artifact": "...",
  "targets": [],
  "features": [],
  "model_classes": {},
  "training_rows": 0,
  "group_column": "config_id",
  "energy_valid_filter": "energy_measurement_valid == 1",
  "carbon_formula": "energy_kwh * 0.65"
}
```

Never invent metadata. Use the artifact contents.

---

# 7. Step 3 — Reproduce the feature pipeline

The surrogate must use exactly the same feature-generation logic used during training.

Pre-run features currently include concepts such as:

```text
log_params_b
params_x_bits
log_params_x_bits

weight_bytes_gb
optimizer_state_gb
gradient_gb
memory_proxy_gb
vram_pressure

adapter_capacity
log1p_rank
is_quantized
is_full_finetune

trainable_frac
trainable_params_b
```

The important rule is:

> Every feature used for prediction must be knowable before the training run starts.

Never use observed accuracy, observed energy, observed VRAM, or observed runtime as an input feature.

Create or preserve a single reusable feature-generation function.

Example interface:

```python
features = build_pre_run_features(config_dataframe)
```

The training and inference paths must call the same function.

---

# 8. Step 4 — Proper surrogate validation

Evaluate targets separately:

```text
accuracy
peak_gpu_memory_gb
energy_kwh
wall_clock_seconds
```

Do not create an independent carbon model.

For each target report:

```text
MAE
RMSE
R²
```

Prefer:

```text
GroupKFold(n_splits=...)
```

with:

```text
groups = config_id
```

For energy:

```text
energy_measurement_valid == 1
```

must be applied before fitting/evaluation.

Generate:

```text
results/surrogate/cv_metrics.csv
results/surrogate/predictions_oof.csv
```

OOF predictions should contain:

```text
config_id
backbone
method
actual
predicted
residual
fold
target
```

---

# 9. Surrogate diagnostics

Generate at least:

### Plot 1

```text
Actual vs Predicted Accuracy
```

### Plot 2

```text
Actual vs Predicted VRAM
```

### Plot 3

```text
Actual vs Predicted Energy
```

### Plot 4

```text
Actual vs Predicted Wall Clock
```

Also report model performance by:

```text
backbone
method
```

when there are enough samples.

Do not hide poor performance.

If accuracy or VRAM prediction generalization is weak, report it honestly and classify the surrogate as preliminary rather than validated for autonomous decision-making.

---

# 10. Surrogate model decision

After validation, choose one of three states:

```text
VALIDATED
PRELIMINARY
UNUSABLE
```

These labels are internal engineering states, not paper-ranking labels.

Suggested logic:

```text
VALIDATED
  only when grouped holdout performance is consistently acceptable

PRELIMINARY
  when useful for shortlist generation but not trustworthy for automatic approval

UNUSABLE
  when predictions are too poor to support candidate filtering
```

Do not invent universal thresholds. Record the thresholds used in code/configuration.

If the current model is weak, improve it with:

```text
feature corrections
better preprocessing
simpler model
regularization
hyperparameter tuning
target-specific models
```

Do not make the architecture more complex merely for novelty.

---

# 11. Candidate Configuration Generator

Build:

```text
decision/candidate_generator.py
```

Purpose:

Generate possible PEFT configurations before any training is run.

Input example:

```python
generate_candidates(
    max_vram_gb=16,
    min_accuracy=0.90,
    models=[...],
    methods=[...],
    ranks=[16],
    quant_bits=[4, 16, 32]
)
```

The generator should support the project's known method/backbone combinations and should be extensible.

Each candidate should have all information needed to generate pre-run features.

Example:

```text
backbone
family
params_b
method
rank
quant_bits
is_adapter_method
gpu_total_gb
```

Do not use future/observed outcome values at this stage.

---

# 12. Constraint Engine

Create:

```text
decision/constraints.py
```

Supported constraints should include:

```text
maximum VRAM
minimum accuracy
maximum energy
maximum runtime
maximum carbon
```

Not every user has to provide every constraint.

Example:

```bash
green-peft recommend \
  --vram 16 \
  --accuracy 0.90 \
  --profile balanced
```

The engine should return:

```text
all candidates
→ feasible candidates
→ infeasible candidates with reason
```

Each infeasible candidate should have a reason such as:

```text
Predicted VRAM exceeds budget
Predicted accuracy below threshold
Predicted energy exceeds limit
```

Do not silently discard candidates.

---

# 13. Feasibility uncertainty

The surrogate is not ground truth.

Every predicted candidate should retain:

```text
prediction
uncertainty/confidence
```

If formal uncertainty estimates are unavailable, implement a transparent preliminary confidence mechanism based on model support / extrapolation distance and clearly document it.

Never display unsupported claims such as:

```text
99% guaranteed
```

unless such a probability is actually calibrated.

---

# 14. Pareto Engine

Create:

```text
decision/pareto.py
```

Objectives:

```text
maximize accuracy
minimize VRAM
minimize energy
minimize wall-clock time
```

Carbon is redundant with energy under the project's fixed conversion and should not be counted as an independent objective.

Implement non-dominated sorting.

Output:

```text
pareto_optimal = True/False
```

and a Pareto frontier table:

```text
results/pareto/pareto_candidates.csv
```

---

# 15. GEI / Preference Engine

Create:

```text
decision/gei.py
```

The project's current preference profiles are:

### Balanced

```text
accuracy = 0.35
VRAM     = 0.25
energy   = 0.25
time     = 0.15
```

### Strict Carbon

```text
accuracy = 0.20
VRAM     = 0.20
energy   = 0.50
time     = 0.10
```

### High Accuracy

```text
accuracy = 0.60
VRAM     = 0.15
energy   = 0.15
time     = 0.10
```

Treat these as the project's current configurable preference profiles.

Do NOT claim that these weights are scientifically universal.

Make the weights configurable.

Recommended sequence:

```text
feasibility
→ Pareto frontier
→ preference profile
→ GEI score
→ recommendation
```

Do not use GEI alone to decide whether something is feasible.

---

# 16. Recommendation Engine

Create:

```text
decision/recommender.py
```

Input:

```text
candidate configurations
predicted metrics
constraints
preference profile
```

Output a structured recommendation object.

Example:

```json
{
  "method": "lora",
  "backbone": "small",
  "rank": 16,
  "quant_bits": 16,
  "predicted_accuracy": 0.94,
  "predicted_vram_gb": 9.7,
  "predicted_energy_kwh": 0.0018,
  "predicted_wall_clock_seconds": 101,
  "predicted_carbon_kgco2eq": 0.00117,
  "pareto_optimal": true,
  "profile": "balanced"
}
```

Also return:

```text
top alternatives
why candidates were filtered
trade-offs
confidence/status
```

The recommendation engine must never make unsupported claims of certainty.

---

# 17. CLI integration

Connect the decision engine to the existing:

```text
green_peft_cli
```

Maintain compatibility with existing commands where practical.

Target command:

```bash
green-peft recommend \
  --vram 16 \
  --accuracy 0.90 \
  --profile balanced
```

Also support example scenarios such as:

```bash
green-peft recommend --vram 8 --accuracy 0.90 --profile balanced

green-peft recommend --vram 16 --accuracy 0.95 --profile high_accuracy

green-peft recommend --vram 16 --accuracy 0.90 --profile strict_carbon
```

The output should be readable for a non-expert researcher.

---

# 18. Recommendation validation mode

Build an optional workflow:

```text
recommend
   ↓
select candidate
   ↓
run actual experiment
   ↓
record actual metrics
   ↓
compare predicted vs actual
```

Create:

```text
experiments/validate_recommendation.py
```

The result should calculate:

```text
accuracy error
VRAM error
energy error
runtime error
```

and save:

```text
results/recommendation_validation.csv
```

This is the bridge between the surrogate and real-world evidence.

---

# 19. Do not immediately rerun the entire benchmark

Current empirical evidence is already useful.

Only propose additional real experiments after completing:

```text
surrogate validation
decision-support validation
uncertainty analysis
```

Additional experiments should be chosen because they answer a specific unresolved research question.

Possible high-value gaps:

```text
validated energy re-measurements
additional model scale
additional task
additional GPU
additional PEFT method
```

Do not add experiments merely to increase the row count.

---

# 20. Benchmark statistics

Create a reproducible analysis script:

```text
analysis/benchmark_analysis.py
```

Produce:

```text
results/benchmark/method_summary.csv
results/benchmark/backbone_summary.csv
results/benchmark/feasibility_matrix.csv
results/benchmark/seed_variance.csv
```

For successful runs report:

```text
mean
std
minimum
maximum
```

for:

```text
accuracy
VRAM
energy
runtime
```

For feasibility report:

```text
successful runs
OOM runs
OOM rate
```

Do not turn OOM into a fake numeric energy or accuracy value.

---

# 21. Figures required for the research paper

At minimum generate:

### Figure A — Feasibility Matrix

```text
method × backbone
```

showing successful/OOM configurations.

### Figure B — Accuracy vs Energy

Only valid empirical energy measurements.

### Figure C — Accuracy vs VRAM

### Figure D — Accuracy vs Runtime

### Figure E — Pareto Frontier

### Figure F — Surrogate Actual vs Predicted

Separate panels or separate figures are acceptable.

Keep figures publication-oriented:

```text
clean
white background
minimal clutter
consistent typography
scientific
readable axis labels
legend only when necessary
```

Do not overdecorate.

---

# 22. Tables required for the paper

Produce:

## Table 1 — Benchmark configuration

```text
Backbone
Parameters
Method
Rank
Quantization
GPU budget
```

## Table 2 — Empirical results

```text
Method
Backbone
Accuracy mean ± std
VRAM mean ± std
Energy mean ± std
Runtime mean ± std
Feasibility
```

## Table 3 — Surrogate validation

```text
Target
MAE
RMSE
R²
Validation strategy
```

## Table 4 — Example GreenPEFT recommendations

For several realistic constraint profiles:

```text
Constraints
Recommended configuration
Predicted outcomes
Pareto status
Preference profile
```

---

# 23. Research claims must remain conservative

The current evidence supports:

```text
pilot benchmark
+
empirical resource measurements
+
preliminary surrogate
+
multi-objective decision-support prototype
```

Do NOT claim without new evidence:

```text
universal best PEFT method
zero-shot universal PEFT selection
hardware-independent prediction
generalization to arbitrary model scales
generalization to arbitrary tasks
production-ready autonomous decision system
```

The current methodology explicitly identifies:

```text
four model scales
single NVIDIA T4
single SST-2 task
one validated energy telemetry pass
```

as limitations.

Use this to scope the claims rather than hide it.

---

# 24. Carbon methodology

The current carbon assumption is:

```text
carbon_kgco2eq = energy_kwh × 0.65
```

Treat 0.65 kgCO2eq/kWh as a project assumption / fixed conversion factor.

Do not present carbon as independently measured when it is derived.

Report:

```text
energy = empirical measurement
carbon = derived estimate
```

---

# 25. Suggested project structure

Use or adapt this structure without breaking existing code:

```text
GreenPEFT/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── greenpeft_surrogate_v1.joblib
│   └── model_metadata.json
│
├── preprocessing/
│   ├── build_features.py
│   └── schema.py
│
├── surrogate/
│   ├── train.py
│   ├── validate.py
│   ├── predict.py
│   └── uncertainty.py
│
├── decision/
│   ├── candidate_generator.py
│   ├── constraints.py
│   ├── pareto.py
│   ├── gei.py
│   └── recommender.py
│
├── experiments/
│   ├── run_experiment.py
│   └── validate_recommendation.py
│
├── analysis/
│   ├── benchmark_analysis.py
│   ├── surrogate_analysis.py
│   └── figures.py
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_benchmark_analysis.ipynb
│   ├── 03_surrogate_training.ipynb
│   ├── 04_surrogate_validation.ipynb
│   └── 05_decision_support.ipynb
│
├── results/
│   ├── benchmark/
│   ├── surrogate/
│   ├── pareto/
│   └── recommendations/
│
├── green_peft_cli/
│
├── DATA_METHODOLOGY.md
└── README.md
```

If equivalent files already exist, reuse them rather than duplicating functionality.

---

# 26. Reproducibility requirements

Every script must:

- use explicit random seeds where relevant
- save results to deterministic paths
- preserve input data
- record model/version metadata when practical
- avoid hidden preprocessing
- avoid notebook-only state
- be runnable from a clean environment

Provide one reproducibility command or script for:

```text
benchmark analysis
surrogate validation
decision-support simulation
figure generation
```

---

# 27. Testing requirements

Create basic tests for:

## Feature generation

Given one known configuration, feature values should remain deterministic.

## Candidate generation

Check that candidate combinations are created correctly.

## Constraint filtering

Check obvious feasible/infeasible examples.

## Pareto

Use a tiny hand-built dataset with known dominance relationships.

## GEI

Check normalization and weight handling.

## Recommendation

Check that only feasible candidates can be selected as recommendations.

## Carbon

Check:

```text
carbon = energy * 0.65
```

## Leakage

Assert that target-derived columns never enter the feature matrix.

---

# 28. Final deliverables

The agent should finish with these outputs:

```text
1. Existing dataset preserved and versioned
2. Existing .joblib audited
3. model_metadata.json
4. grouped CV validation results
5. OOF prediction file
6. benchmark summary tables
7. feasibility matrix
8. Pareto analysis
9. candidate generator
10. constraint engine
11. GEI/preference engine
12. recommendation engine
13. CLI integration
14. recommendation validation workflow
15. research figures
16. tests
17. updated README
18. concise implementation report
```

The implementation report should answer:

```text
What already worked?
What was changed?
What is experimentally validated?
What remains preliminary?
What additional experiment is actually necessary?
```

---

# 29. Agent behavior rules

Follow these rules throughout implementation:

### Rule 1
Do not destroy existing results.

### Rule 2
Do not silently change the dataset definition.

### Rule 3
Do not leak target information into features.

### Rule 4
Do not mix source roles.

### Rule 5
Do not treat repeated measurement passes as independent configurations.

### Rule 6
Do not train energy models on invalid telemetry.

### Rule 7
Do not build a separate carbon model.

### Rule 8
Do not make unsupported claims about model generalization.

### Rule 9
Do not run a huge new GPU benchmark before determining what is actually missing.

### Rule 10
Prefer simple, maintainable implementation over unnecessary complexity.

### Rule 11
Every important transformation should be reproducible in code.

### Rule 12
If an existing implementation is correct, preserve it instead of rewriting it only for style.

---

# 30. Definition of success

The GreenPEFT prototype is considered functionally complete when this works end-to-end:

```text
User:
GPU VRAM = 16 GB
minimum accuracy = 0.90
profile = balanced
        ↓
GreenPEFT generates candidates
        ↓
pre-run features are generated
        ↓
.joblib predicts accuracy / VRAM / energy / runtime
        ↓
infeasible candidates are filtered
        ↓
Pareto-efficient candidates identified
        ↓
GEI/profile preference applied
        ↓
recommendation returned
        ↓
alternatives + trade-offs + uncertainty shown
```

The system should also be able to export the recommendation and its supporting predictions to CSV/JSON for research reporting.

---

# 31. Final research narrative

The final system should communicate this progression:

```text
Empirical PEFT measurements
        ↓
Resource-aware feature engineering
        ↓
Surrogate prediction
        ↓
Constraint-aware candidate screening
        ↓
Multi-objective Pareto analysis
        ↓
Preference-aware decision support
        ↓
Actionable PEFT recommendation
        ↓
Optional real-world validation
```

The central contribution is therefore:

> A sustainability-aware decision-support framework that uses empirical PEFT measurements, pre-run resource features, predictive modeling, feasibility constraints, Pareto analysis, and configurable practitioner preferences to support PEFT selection under resource constraints.

Do not present the system as a universally optimal selector. Present it as a **decision-support framework whose current empirical scope is limited to the validated benchmark regime**.

---

# 32. Recommended first execution

Start with ONLY these tasks:

```text
A. Inspect repository
B. Inspect .joblib
C. Verify the exact feature list
D. Verify target models
E. Run GroupKFold validation
F. Produce cv_metrics.csv
G. Produce OOF predictions
H. Produce actual-vs-predicted plots
I. Report whether the surrogate is VALIDATED, PRELIMINARY, or UNUSABLE
```

Stop at that checkpoint before implementing new decision logic.

The next implementation stage should begin only after the surrogate audit is complete.
