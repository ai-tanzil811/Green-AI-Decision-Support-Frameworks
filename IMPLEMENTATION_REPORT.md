# GreenPEFT — Implementation Report

Answers the five questions in section 28 of `greenpeft_agent_workflow.md`: what already worked,
what was changed, what is experimentally validated, what remains preliminary, and what
additional experiment is actually necessary.

Scope of this pass: an audit of `green-peft.ipynb` against the dataset, a rebuild of that
notebook, a confidence layer for the recommender, and the analysis deliverables that were
still missing. The dataset and the `.joblib` bundle were treated as read-only inputs
throughout (Rule 1).

---

## 1. What already worked

Most of the pipeline was in place and correct.

| Component | State found | Evidence |
|---|---|---|
| `.joblib` audit and `model_metadata.json` | Complete and accurate | `models/model_metadata.json` |
| Grouped CV + OOF predictions | Present | `results/surrogate/cv_metrics.csv`, `predictions_oof.csv` |
| Surrogate status call | Correct and conservative (`PRELIMINARY`) | metadata `status_overall` |
| Candidate generator, constraint engine, Pareto, GEI, recommender | Implemented | `green_peft_cli/.../recommender.py` |
| CLI | Working end-to-end | `green-peft recommend --vram 16 --accuracy 0.90` |
| Test suite | 16 tests, all passing | `green_peft_cli/green_peft_pkg/tests/` |

These live in `green_peft_cli/` rather than the `decision/` tree sketched in section 25. That
is the intended outcome under section 25's own instruction to reuse equivalent files rather
than duplicate them, so nothing was moved.

The project's section 30 definition of success already held: constraints in, candidates
generated, surrogate predictions, feasibility filtering, Pareto set, GEI ranking,
recommendation out.

---

## 2. What was changed

### 2.1 `green-peft.ipynb` — rebuilt

The notebook pooled the dataset's two measurement passes as if they were independent runs.
They are not: the 82 `surrogate_train` rows are **41 configurations measured twice**, paired by
`config_id`, and the `remeasured` pass is flagged `energy_measurement_valid == 0` because it
recorded a flat ~10 W implied power — the NVML idle floor, not loaded training power.

Consequences in the version as run:

| Defect | Effect |
|---|---|
| No energy-validity filter, no pass collapse | Mean energy understated by **40–42%** in every table and figure; every sample count doubled (n=6 printed where 3 seeds exist) |
| Paired PEFT-vs-full-FT merge on a two-pass frame | 2×2 cross product per seed: 12 rows instead of 3, half of them comparing one method's valid pass against the other's invalid one, producing "−202% energy saved" |
| `acc_per_energy` over pooled passes | Within-pass SD is 102–152; pooled SD 728–1445. The published method ranking was an artifact of telemetry, not of the methods |
| `GroupKFold(groups=config_id)` described as preventing seed leakage | `config_id` **contains the seed**, so it separated only the measurement passes. Two seeds trained while the third tested |
| `energy_kwh`, `peak_gpu_memory_gb`, `wall_clock_seconds` used as input features to predict accuracy | Violates Rule 3 and section 7: all three are in the project's own `leakage_guard`. A model needing measured energy to predict accuracy cannot help anyone choose a configuration |

What the notebook now does:

- **Section 4b (new)** collapses the passes into one canonical row per configuration, with each
  collapse rule backed by an assertion rather than an assumption, and prints the size of the
  error it corrects.
- All EDA, comparison and frontier sections run on that 41-row frame.
- The predictive section was rebuilt to predict **cost** (accuracy, VRAM, energy, runtime) from
  **pre-run features only**, taken verbatim from `model_metadata.json`, with an asserted
  leakage guard. It is grouped on `config_base` (method + backbone) and reports both
  interpolation and extrapolation protocols.
- The false leakage claim in the markdown was corrected and the mechanism explained.
- Six markdown cells and one code cell contained literal `U+FFFD` replacement characters from a
  bad encoding round-trip (the title read "GreenPEFT Surrogate Dataset � Research Notebook").
  Repaired.
- `ci='sd'` → `errorbar='sd'`; the former was removed in seaborn 0.13 and raises `TypeError`
  on a current install.
- The loader now falls back to local paths, so the notebook runs outside Kaggle.

The rebuilt notebook was executed end to end: 20 code cells, 0 errors.

**Effect on the headline number.** Removing the leaked features and grouping seeds correctly
moves accuracy R² from the reported **0.793** to **0.39 interpolating and 0.08 extrapolating**.
The original figure was not a surrogate result; it was a model reading measured cost back out.

### 2.2 Recommender — scope, confidence and error bands

The engine emitted bare point estimates for candidates far outside the measured envelope. Of
the 14 backbones in `model_zoo`, only 4 are measured tiers, 5 lie outside the 0.5–3.0B
parameter range, and 4 are families (`gpt_neox`, `phi2`, `phi3`, `mistral`) never seen in
training. Sections 13 and 23 require otherwise.

Added `green_peft/confidence.py`:

- `training_envelope` is **derived from the dataset** by `analysis/build_training_envelope.py`
  and recorded in both `model_metadata.json` copies, with a `--check` mode for CI. Nothing is
  hardcoded.
- Each candidate gets a `scope` (`MEASURED` → `OUT_OF_RANGE_SCALE`), a `confidence`
  (HIGH/MEDIUM/LOW), a stated reason, and per-target error bands drawn from the surrogate's own
  measured MAPE — the interpolation figure for measured cells, the extrapolation figure
  otherwise.
- Candidates beyond the measured parameter range carry an explicit caveat that the band is a
  **lower bound**: the leave-one-tier-out protocol only ever held out a tier *inside*
  0.5–3.0B, so it never tested prediction beyond that range.
- `--min-confidence {LOW,MEDIUM,HIGH}` restricts the candidate set to a chosen evidence level.

### 2.3 A correctness bug this surfaced

Writing a test that error bands must bracket their point estimate exposed a live defect: the
Ridge regressors extrapolate linearly past physical limits.

- The 0.135B zoo entries draw **negative peak VRAM** (down to −1.20 GB).
- The 7.2–7.6B entries draw **accuracy above 1.0** (up to 1.19).

12 of 70 candidates were affected. This was not cosmetic. A negative VRAM prediction satisfies
every budget trivially, so it passed the feasibility gate; and because GEI normalisation takes
its bounds from the feasible set, one negative value rescaled the memory objective for every
other candidate. Removing them changes the top candidate's balanced GEI from 0.8070 to 0.8339.

Such candidates are now dropped before scoring with a stated reason and a count in the report,
never silently (section 12).

### 2.4 New deliverables

| Deliverable | Path |
|---|---|
| Benchmark summary tables (§20) | `analysis/benchmark_analysis.py` → `results/benchmark/{method_summary,backbone_summary,feasibility_matrix,seed_variance}.csv` |
| Pareto analysis (§14) | `analysis/pareto_analysis.py` → `results/pareto/{pareto_candidates,pareto_measured}.csv` |
| Recommendation validation (§18) | `experiments/validate_recommendation.py` → `results/recommendation_validation.csv` |
| Example recommendations, Table 4 (§17, §22) | `analysis/recommendation_scenarios.py` → `results/recommendations/scenarios.{csv,json}` |
| Figures A and E (§21) | `analysis/figures.py` |
| Shared loaders | `analysis/greenpeft_data.py` |
| One-command reproduction (§26) | `reproduce.py` |
| Tests | 16 → 30, all passing |

`validate_recommendation.py` has two modes, kept apart because they answer different questions.
`--from-benchmark` back-tests against already-measured configurations and records the rows as
`in_sample=True`; it verifies the candidate → feature → model path but is **not** evidence of
generalisation. `--record FILE` compares against a new real run and is the only mode that
produces new evidence.

---

## 3. What is experimentally validated

Only one thing, and only under one protocol.

**`energy_kwh`** — R² 0.88 interpolating, 0.90 extrapolating to an unseen tier; MAPE 11.1% and
10.6%. It clears the project's recorded thresholds under both protocols. The log-target Ridge
is the model that achieves this; the RandomForest is UNUSABLE on this target (R² −0.02 and
−0.13), because a tree ensemble cannot extrapolate a monotone trend past its training range.

Also solid, though not a model result:

- **The feasibility pattern.** 41 successful and 19 OOM across the 5×4×3 grid, directly
  observed. `full_ft` fits only at 0.5B; `qlora` is the only method that fits at 3.0B.
- **The paired PEFT-vs-full-FT comparison at 0.5B**, now that it is computed correctly:

  | method | accuracy Δ (pp) | energy saved | memory saved | time saved |
  |---|---|---|---|---|
  | lora | +1.22 | 49.0% | 61.0% | 17.2% |
  | lora_fa | +1.03 | 45.7% | 61.3% | 20.5% |
  | qlora | +0.65 | 10.3% | 75.2% | **−26.7%** |
  | lisa | −1.84 | 47.4% | 54.1% | 45.6% |

  The qlora row is the interesting one and the broken analysis obscured it: qlora buys the most
  memory of any method and almost no energy, because 4-bit dequantisation makes it the slowest.
  Memory savings and energy savings are not the same axis.

---

## 4. What remains preliminary

- **`accuracy`** — R² 0.39 interpolating, **0.08** extrapolating. Barely better than predicting
  the mean on an unseen scale. Accuracy spans only 0.086 across the entire benchmark, so there
  is very little signal to fit, and most of what remains is the `params_b` trend.
- **`peak_gpu_memory_gb`** — R² 0.79 / 0.59, MAPE 18% / 38%. The feasibility gate rests on this
  model, so the 10% VRAM safety margin is load-bearing, not decoration.
- **`wall_clock_seconds`** — R² 0.62 / 0.54, MAPE 17% / 23%.
- **Overall status: PRELIMINARY.** Good enough to shortlist, not to approve without measuring.

Two further limits worth stating in the paper:

- **The scope/method confound.** `full_ft` exists only at 0.5B and `qlora` is the only method at
  3.0B, so no method comparison is like-for-like across the full range. Marginal means by
  method mix backbones and are not a ranking.
- **A systematic VRAM measurement artifact.** Across-seed CV reaches 34% for `qlora/small`, but
  it is not random: seed 13 reports consistently higher peak VRAM than seeds 42 and 2024 in
  *every* qlora cell (small 5.52 vs 3.17/3.19 GB; medium 8.57 vs 6.02/4.66; tiny 3.33 vs
  2.39/2.57). That pattern looks like run-order or allocator-fragmentation effect rather than
  seed variance, and it puts a floor under VRAM prediction error that no model change can
  remove.

### A documentation inconsistency left in place

`model_metadata.json`'s free-text `notes` field makes two claims. Inspecting the bundle itself
settles which is true:

- *"Fitted on UNIQUE CONFIGS collapsed from duplicate measurement passes"* — **true**. The
  bundle records `training_rows = {'regressor_configs': 41}`, so the four regressors were fitted
  on the 41 collapsed configurations, not on the 82 raw rows. The models themselves are sound.
- *"All metrics are leave-one-tier-out"* — **false**. `cv_metrics.csv` reports
  `GroupKFold(config_id)` as the primary protocol and records `n_rows=82, n_groups=41` for
  accuracy, VRAM and wall-clock. Only `energy_kwh` was evaluated at 41 rows.

So the defect is narrower than the prose suggests: the **fit** is correct, the **validation** was
run on the uncollapsed frame for three of four targets, and `GroupKFold(config_id)` does not
separate seeds in any case. The interpolation figures in the metadata are therefore optimistic;
the leave-one-tier-out figures are not affected and are the ones to quote.

Nothing here was edited, because correcting it means re-running the validation (and deciding
whether to regenerate the shipped metadata), which touches the published artifact — a decision
for the project owner under Rule 2. The recommendation is to re-run CV on the 41 canonical
configurations grouped by `config_base`, as `green-peft.ipynb` section 8 now does, and
regenerate the metadata from that. The `.joblib` weights themselves need no change.

---

## 5. What additional experiment is actually necessary

In priority order.

1. **A second *valid* energy measurement pass.** Exactly one telemetry pass per configuration is
   trustworthy, so the project's only VALIDATED target has **no replication at all** — every
   energy figure is a single measurement with no error bar. Re-measure the existing 41
   configurations with NVML under load, verified against a wall-plug reading. This needs no new
   configurations and would upgrade the strongest result from "measured once" to "measured
   twice and agreeing".
2. **Fill the method × backbone matrix, not the seed count.** `full_ft` at 1.1B and `lisa`,
   `lora`, `lora_fa` at 3.0B are the cells that would break the scope/method confound. They
   went OOM at 15.6 GB, so they need either a larger GPU or gradient checkpointing — which is
   itself a finding worth reporting.
3. **One backbone outside 0.5–3.0B, measured.** Every error band the CLI reports for an
   out-of-range candidate is currently a lower bound extrapolated from within-range evidence.
   A single measured point at ~7B would convert the recommender's largest guesses into
   something calibrated, and would test whether the energy model's good extrapolation holds.
4. **A second task.** Every number in the project is SST-2. Nothing currently distinguishes
   "PEFT behaves this way" from "PEFT behaves this way on SST-2".

What is **not** worth doing: adding seeds to cells that already have three. Across-seed CV is
under 6% for accuracy, energy and runtime; the uncertainty that matters lives in scale
extrapolation and in the single-pass energy telemetry, not in seed noise.

---

## 6. Reproducing this

```bash
python reproduce.py           # regenerate every derived artifact
python reproduce.py --check   # verify recorded artifacts match the data, write nothing
```

The input dataset and the `.joblib` bundle are read-only; no step modifies them.
