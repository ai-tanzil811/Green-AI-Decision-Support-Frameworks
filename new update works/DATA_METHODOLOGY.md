# Data acquisition, harmonization and preprocessing

How the GreenPEFT dataset was assembled, what was accepted or rejected and why, and every
transformation applied between the raw source files and the modelling table.

All of it is executable: [`data_preprocessing_and_feature_engineering.ipynb`](data_preprocessing_and_feature_engineering.ipynb)
reproduces this document's every number from the four raw files in `sample datasets/`.

**Result:** 4 raw sources → 1 harmonized dataset, `greenpeft_ml_ready_dataset.csv`
(477 rows × 46 columns), of which 82 rows are valid for surrogate fitting.

---

## 1. Acquisition

Four files were obtained. Three were accepted with different roles; one was rejected.

| Source | Rows | What it actually measures | Verdict |
|---|---|---|---|
| `surrogate_dataset.csv` | 120 | Our own PEFT fine-tuning sweep on a single T4 | **Accepted** — gold standard |
| `Open LLM-Perf Leaderboard.csv` | 370 | HuggingFace *inference* throughput of pre-trained models | **Accepted** as scale context |
| `llmenergy.csv` | 27 | Published *pretraining* energy for frontier models | **Accepted** as context only |
| `llm_comparison_dataset.csv` | 200 | Claimed model comparison metrics | **Rejected** |

### 1.1 Primary source — our benchmark

`surrogate_dataset.csv` is the only source that measures what the paper is about: PEFT
fine-tuning runs where energy, memory, runtime and accuracy were recorded together on known
hardware. 4 backbone scales (0.5B / 1.1B / 1.5B / 3.0B) × 5 methods (full fine-tuning, LoRA,
QLoRA, LoRA-FA, LISA) × 3 seeds, measured twice — 120 rows covering 60 unique configurations.

### 1.2 Secondary source — Open LLM-Perf Leaderboard

Genuine leaderboard data with real model names, spanning 24 distinct parameter counts from
0.007B to 70B — far wider scale coverage than our own sweep. But it measures **inference** of
already-trained models. There is no fine-tuning method, no adapter rank and no training energy
in it, and none was invented: those columns are left null.

It is retained as `scale_reference` and is **not** used to fit the surrogate. Blending
inference peak memory with fine-tuning peak memory would mix two different physical quantities
under one column name.

One exact duplicate row was present and dropped (370 → 369).

### 1.3 Tertiary source — frontier pretraining energy

`llmenergy.csv` holds real published figures, but for **pretraining from scratch** across up to
200,000 GPUs. Its energy values are 9+ orders of magnitude above a single-GPU PEFT run.

Two safeguards: its method is tagged `pretrain_full_scratch`, a label deliberately outside the
paper's five-method vocabulary so that no `groupby('method')` can average it into `full_ft`; and
it is tagged `context_only` so it never reaches the model.

One row (Gemini 1.5 Pro) reported `"Not disclosed"` for parameter count and was dropped, since a
row with no scale cannot be placed in the schema (27 → 26).

### 1.4 Rejected source

`llm_comparison_dataset.csv` was rejected on three independent grounds:

1. **Model names match no real release.** `DeepSeek-4`, `Llama-8`, `Llama-5`, `DeepSeek-3` — no
   vendor has shipped these versions.
2. **Metrics are unitless.** `Compute Power` and `Energy Efficiency` are bare numbers with no
   stated unit and no way to verify what they measure.
3. **No parameter-count column.** The only size-like signal is a digit suffix inside a model
   name already flagged as fabricated.

It is excluded entirely rather than merged with a quality flag. A flag would still leave the
rows available to a careless `groupby`.

---

## 2. Harmonization

All sources are mapped onto one 17-column schema plus 3 provenance columns.

```
run_id, method, backbone, family, params_b, rank, quant_bits, is_adapter_method,
gpu_total_gb, status, fits, trainable_param_pct, accuracy, peak_gpu_memory_gb,
energy_kwh, carbon_kgco2eq, wall_clock_seconds
+ data_source, is_empirical, data_quality_flag
```

### Mapping rules

**Parameter counts** are parsed from free text (`7B`, `700M`, `1.1B`) by regex and normalized to
billions. A string that does not match returns null — it is never defaulted to 0 or guessed from
surrounding text.

**Quantization width** maps dtype strings (`fp16`, `bf16`, `int4`, `nf4`, …) to bit widths. The
brief's default (32 for full fine-tuning, 16 otherwise) applies *only* where the source states
nothing; it never overrides an explicit value that simply isn't in the lookup table.

**Architecture family** is keyword-matched against the controlled vocabulary
`{llama, qwen, mistral, gemma, phi, falcon}`. Anything else maps to `other` rather than being
forced into the nearest-sounding bucket — GPT-NeoX, MPT, ChatGLM, Baichuan, OPT and T5 genuinely
are outside that vocabulary.

**Deduplication is on `run_id` alone.** The brief specified deduplicating on
`(family, params_b, method, quant_bits, gpu_total_gb, rank)`, but our seed replicates share every
one of those columns by design. Applying it literally would have collapsed 6 real measurements
per cell to 1, destroying 100 of 120 measurements and all seed-variance information.

### Scope filter

Configurations that exceeded the 15.6 GB VRAM budget are outside the study's scope and are
dropped at ingest — 38 rows (19 configurations × 2 passes). The remaining models therefore
predict cost *given* that a configuration runs.

This is controlled by `INCLUDE_INFEASIBLE_RUNS = False` in the setup cell; setting it to `True`
restores those rows and re-enables a feasibility classifier.

---

## 3. Missingness diagnosis

The missingness matrix is **blocky**: entire columns are 100% absent for an entire source.

| Column | Our benchmark | Frontier pretraining | Inference leaderboard |
|---|---|---|---|
| `method`, `rank`, `is_adapter_method` | 0% | 0% | **100%** |
| `gpu_total_gb` | 0% | **100%** | **100%** |
| `accuracy`, `peak_gpu_memory_gb` | 0% | **100%** | 0% |
| `energy_kwh`, `wall_clock_seconds` | 0% | 41–52% | **100%** |

This is the signature of *structural* missingness — each source measured a different thing —
not of data that went astray. Every 100% block is a question the source never asked.

**Consequence: no cross-block imputation.** Filling `energy_kwh` for the leaderboard rows from a
column median would fabricate training energy for models that were never fine-tuned. The
pipeline asserts this rather than trusting it.

---

## 4. Noise handling

Five audits run before any model is fitted. The last is the consequential one.

### 4.1 Duplicate rows
Exact `run_id` duplicates removed. (0 found after the leaderboard's own dedup.)

### 4.2 Collinear target detection
`carbon_kgco2eq / energy_kwh` was computed per row: the ratio is 0.649988 to 0.650015, standard
deviation 4.8×10⁻⁶. **Carbon is exactly energy × 0.65** — a grid-intensity unit conversion, not
an independent measurement. It is therefore derived rather than modelled; fitting it as a
separate target would report the same model twice under two names.

### 4.3 Physical-plausibility bounds
Values outside physically possible ranges are nulled (accuracy ∉ [0,1], negative energy, etc.).
The parameter bound is 5000B, wide enough to admit genuine trillion-scale MoE models — Grok 3 at
2700B and Claude 3 Opus at 2000B are legitimate entries, not outliers.

### 4.4 Seed-replicate outliers
Within each `(backbone, method)` cell, a robust modified z-score (median absolute deviation,
threshold 3.5) flags anomalous seeds. Two were flagged. They are **flagged, not deleted** —
seed variance is a real physical result the paper reports.

### 4.5 Measurement-regime audit — the consequential one

The 120 benchmark rows are **60 configurations measured twice**, not 120 independent runs.
Pairing them by configuration shows:

| Metric | Correlation across passes | Ratio (pass 2 / pass 1) |
|---|---|---|
| `accuracy` | **+1.0000** | **1.0000** |
| `peak_gpu_memory_gb` | +0.9994 | 0.921 |
| `wall_clock_seconds` | +0.9970 | 1.081 |
| `energy_kwh` | +0.9562 | **0.169** |

Two findings follow.

**Accuracy is byte-identical between passes.** The pair carries zero additional information for
that target, and treating the passes as independent would double the apparent sample size. Under
random k-fold it would also place a configuration's own duplicate in both train and test folds.

**The passes disagree on energy by 6×.** Converting energy and runtime into implied average power
identifies which one is wrong:

| Pass | Implied power | Range |
|---|---|---|
| original | 59.9 W | 38.4 – 65.6 W |
| remeasured | **9.97 W ± 0.10** | 9.85 – 10.23 W |

The T4 has a 70 W TDP and idles near 10 W. The remeasured pass reports a near-constant 9.97 W
across a sixfold range of model sizes — no accelerator under training load behaves that way.
That is the NVML **idle floor**, not a load measurement. The original pass, at 38–66 W under a
70 W ceiling, is physically consistent.

Energy from the remeasured pass is marked invalid (`energy_measurement_valid = 0`) rather than
deleted, so the decision stays visible and reversible in the exported file. Pooling both regimes
had inflated the energy target's coefficient of variation from 0.509 to 0.936, which is what made
energy appear unpredictable before the audit.

---

## 5. Missing-value treatment

The governing rule: **an imputed value must be recoverable, not invented.**

| Case | Treatment | Why |
|---|---|---|
| Absent categorical (`method`) | Explicit level `none_inference_only` | A mode-fill would assert a method that was never applied |
| Absent numeric (`rank`) | `0` **plus** indicator `has_adapter_config` | Lets the model distinguish "rank 0" from "no adapter exists" |
| `gpu_total_gb` unknown | Within-source median + `gpu_budget_known` | Never imputed across sources |
| `trainable_param_pct` | Median of the same `(method, backbone)` | Deterministic property of the config, not a measurement — genuinely recoverable |
| `params_b` unparseable | **Row dropped** | A row with no scale cannot be modelled at all |
| Outcome metrics | **Never imputed** | Within the in-scope data there is nothing to impute: every retained run reported every metric |

---

## 6. Feature engineering

14 model-input features, derived from GPU memory physics rather than column arithmetic. Every
one is knowable **before a run starts** — that is what makes the surrogate usable for planning.

**Scale.** `log_params_b`, `params_x_bits`, `log_params_x_bits` — energy follows a power law in
parameter count, so the log terms linearize it.

**Memory physics.** `weight_bytes_gb` + `optimizer_state_gb` + `gradient_gb` → `memory_proxy_gb`,
and `vram_pressure` as its ratio to the device budget. Adam holds two fp32 moments plus one
gradient per *trainable* parameter — which is precisely why adapter methods collapse the memory
cost while full fine-tuning does not. Encoding that relationship explicitly lets a linear model
capture it.

**Method topology.** `adapter_capacity` (rank × is_adapter), `log1p_rank`, `is_quantized`,
`is_full_finetune`, `trainable_frac`, `trainable_params_b`.

### Leakage quarantine

Four ratios — `energy_per_acc_point`, `vram_efficiency`, `throughput_proxy`, `carbon_derived_kg` —
are computed for **reporting only**. They contain the targets, so using them as model inputs
would be leakage. The modelling stage asserts that none appears in the feature list.

---

## 7. The output dataset

`greenpeft_ml_ready_dataset.csv` — 477 rows × 46 columns. One file, with an `analysis_role`
column so no consumer has to guess what a row is for:

| Role | Rows | Valid for |
|---|---|---|
| `surrogate_train` | 82 | Fitting the PEFT surrogate (41 configs × 2 passes) |
| `scale_reference` | 369 | Inference-side scale context, 0.007–70B |
| `context_only` | 26 | Frontier pretraining; energy scale 9+ orders larger |

Keeping all three in one file with explicit roles beats shipping three files: provenance travels
with the data, and the exclusions stay auditable instead of becoming folklore.

### Provenance columns

| Column | Meaning |
|---|---|
| `data_source` | Which raw file the row came from |
| `data_quality_flag` | Free-text caveat carried from ingest |
| `is_empirical` | Whether the row is a measurement or a published figure |
| `config_id` | Configuration identity, shared by both measurement passes |
| `measurement_pass` | `original` / `remeasured` / `single` |
| `energy_measurement_valid` | 0 where telemetry failed the plausibility audit |
| `is_seed_outlier` | Flagged by the MAD test, retained |
| `analysis_role`, `split_reason` | Role and the reason for it, in plain text |

---

## 8. Reproducing this

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn scipy
jupyter nbconvert --execute --inplace data_preprocessing_and_feature_engineering.ipynb
```

Paths resolve relative to the notebook; set `GREENPEFT_ROOT` to override. Runtime is about
50 seconds. The notebook regenerates the dataset, the model bundle, the CV metrics and all seven
figures from the four raw files alone — deleting every output first is a valid test, and one the
pipeline passes.

### Known limitations of the data

- **Four model scales** (0.5/1.1/1.5/3.0B) is the binding constraint on every extrapolation result.
- **Energy rests on one validated telemetry pass** — 41 configurations. Re-running the sweep with
  verified telemetry would double that sample and remove the need for the regime exclusion.
- **Single device, single task**: one NVIDIA T4, SST-2 classification.
- **Carbon figures are derived**, applying one static grid intensity (0.65 kgCO₂eq/kWh) uniformly.
  They are linear rescalings of measured energy and carry no independent measurement uncertainty.
