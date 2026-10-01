# GreenPEFT Paper Reproduction Guide

This guide defines the smallest useful project bundle for reproducing the GreenPEFT paper:
data acquisition, preprocessing, benchmark analysis, surrogate-model audit, recommendations,
and the reported tables and figures.

The repository is currently a working tree. This document is a cleanup manifest as well as a
reproduction guide: files marked **Keep** are part of the paper bundle; files marked **Archive**
or **Review** should not be deleted until their contents and any external references have been
checked.

## 1. Pipeline at a glance

```text
four source CSV files
        |
        v
data_preprocessing_and_feature_engineering.ipynb
        |
        v
new update works/greenpeft_ml_ready_dataset.csv
        |
        +--> green-peft.ipynb --------------------+
        |                                          |
        +--> surrogate/validate.py                 |
        |        -> models/model_metadata.json      |
        |        -> results/surrogate/              |
        |                                          |
        +--> analysis/greenpeft_data.py             |
                 + benchmark_analysis.py            |
                 + pareto_analysis.py               |
                 + recommendation_scenarios.py      |
                 + figures.py                        |
                 + build_hf_export.py               |
        |                                          |
        v                                          v
results/benchmark, pareto, recommendations, figures, surrogate
        |
        v
experiments/validate_recommendation.py
        |
        v
results/recommendation_validation.csv
```

`reproduce.py` runs the derived-artifact steps in dependency order. It does not modify the
source datasets or the surrogate bundle, but it does regenerate derived metadata and results.

## 2. Data collection and provenance

The raw inputs are in `sample datasets/` and are documented in
`new update works/DATA_METHODOLOGY.md`.

| Source | Role in the paper | Keep? |
| --- | --- | --- |
| `sample datasets/surrogate_dataset.csv` | Primary PEFT fine-tuning measurements | **Keep** |
| `sample datasets/Open LLM-Perf Leaderboard.csv` | Inference-scale reference only | **Keep** |
| `sample datasets/llmenergy.csv` | Pretraining-energy context only | **Keep** |
| `sample datasets/llm_comparison_dataset.csv` | Rejected due to unverifiable names and units | **Archive** |

The accepted sources are harmonized into
`new update works/greenpeft_ml_ready_dataset.csv` (477 rows and 46 columns). The roles are
explicit in `analysis_role`:

- `surrogate_train`: 82 rows from 41 configurations measured in two passes.
- `scale_reference`: 369 inference-scale rows, retained for context and not used to fit the
  surrogate.
- `context_only`: 26 published pretraining-energy rows, retained for context and not used to
  fit the surrogate.

The two measurement passes are not treated as independent observations. The remeasured energy
pass reports the NVML idle floor and is marked `energy_measurement_valid = 0`. The canonical
loader keeps the valid energy pass, averages the genuine VRAM/runtime measurements, and retains
the pass-level spread. Carbon is derived as `energy_kwh * 0.65` rather than fitted as an
independent target.

## 3. What the paper bundle must contain

### Inputs and provenance

- `sample datasets/` accepted source files
- `new update works/greenpeft_ml_ready_dataset.csv`
- `new update works/DATA_METHODOLOGY.md`
- `new update works/data_preprocessing_and_feature_engineering.ipynb`
- `results/canonical_benchmark/` including `configs/`, `raw_runs/`, `metrics/`, `figures/`,
  and `manifest.json`

### Models and model metadata

- `new update works/greenpeft_surrogate_models.joblib` as the audited surrogate input
- `models/model_metadata.json` as the repository-level audit record
- `model/artifacts_export/` as the reproducible CLI/export bundle
- `models/` and `model/` documentation and notices required by the selected release

The package copy under `green_peft_cli/green_peft_pkg/green_peft/data/` is needed only when the
published CLI package is being rebuilt or tested. It is not required for the paper's analysis
if `model/artifacts_export/` is retained.

### Code and notebooks

- `reproduce.py`
- `analysis/greenpeft_data.py`
- `analysis/build_training_envelope.py`
- `analysis/benchmark_analysis.py`
- `analysis/pareto_analysis.py`
- `analysis/recommendation_scenarios.py`
- `analysis/figures.py`
- `analysis/build_hf_export.py`
- `surrogate/validate.py`
- `experiments/validate_recommendation.py`
- `green-peft.ipynb`, the main benchmark and surrogate notebook
- `green_peft_cli/green_peft_pkg/green_peft/` and its tests if the recommender is part of the
  paper's artifact

`green-project-patched.ipynb` and `green_peft_Audit/green-peft-audit.ipynb` are **Review**:
retain them only if the paper cites their audit trail or they contain a result not reproduced by
the main notebook and scripts.

### Final outputs

Keep the paper-facing outputs:

- `results/benchmark/`
- `results/pareto/`
- `results/recommendations/`
- `results/surrogate/`
- `results/recommendation_validation.csv`
- figures referenced by the manuscript, including canonical benchmark figures
- `models/model_metadata.json`

`results/working_run/`, `results/notebook_assets/`, and `results/cache/` are **Archive/Review**.
They are not needed by the clean reproduction path described above.

## 4. Files to exclude from the clean bundle

These are generated or environment-specific and are not paper evidence:

- `.venv/`
- `.git/`
- `__pycache__/`
- `*.pyc`, `*.pyo`, `*.pyd`
- `.pytest_cache/`
- package build caches and temporary notebook checkpoints
- `results/cache/`

The virtual environment should be recreated from the documented dependencies rather than copied
into the paper archive. Deleting it is **Destructive** until the environment has been recreated
and the reproduction checks have passed.

## 5. Cleanup decisions and risk labels

| Action | Risk | Decision rule |
| --- | --- | --- |
| Keep source scripts, accepted datasets, canonical benchmark, model artifacts, and final results | Safe | Required for reproduction or provenance |
| Add this guide and an archive manifest | Safe | Documentation only |
| Move `results/working_run/` outside the paper bundle | Review | Verify no manuscript, notebook, or script reads it |
| Move `results/notebook_assets/` and `results/cache/` outside the paper bundle | Safe | They are rendering/cache support, not evidence |
| Archive rejected `llm_comparison_dataset.csv` | Review | Keep it if the paper discusses the rejection decision |
| Remove `.pytest_cache/`, `__pycache__/`, and bytecode | Safe | Regenerable and ignored; verify no tracked copies first |
| Rebuild and then remove `.venv/` | Destructive | Only after a clean environment passes the checks |
| Remove duplicate benchmark copies from `working_run/` | Destructive | Only after SHA-256 inventory and backup |
| Remove `api.txt` | Destructive | Only after confirming it contains no required provenance or secrets |

Do not overwrite `results/recommendation_validation.csv` before preserving the current tracked
change. It is currently modified relative to Git and may contain the latest validation results.

## 6. Reproduction commands

From the repository root:

```powershell
python reproduce.py --list
python reproduce.py --check
```

The full regeneration command is intentionally separate because it writes derived artifacts:

```powershell
python reproduce.py
```

The focused test command is:

```powershell
python -m pytest green_peft_cli/green_peft_pkg/tests -q
```

For a clean paper archive, first verify that `python reproduce.py --check` and the test suite
pass, then copy only the **Keep** paths above. Do not delete the working tree until the copied
archive has been opened and the checks have been run from its new location.

## 7. Paper claims supported by this bundle

The benchmark is an SST-2 classification study on an NVIDIA Tesla T4 with four backbone tiers,
five fine-tuning methods, and three seeds. The full attempted grid contains 60 configurations;
41 configurations have usable measured outcomes and 19 are OOM outcomes retained in the
feasibility analysis.

The surrogate is a planning aid, not a replacement for measurement. The energy target is the
strongest validated prediction in the current audit. Accuracy, peak VRAM, and wall-clock
predictions remain preliminary and should be described with their reported validation metrics,
scope, and error bands. Candidates outside the measured 0.5B-3.0B envelope are extrapolations.

## 8. Reproducibility checklist

- [ ] Preserve the accepted raw source CSV files.
- [ ] Preserve the harmonized ML-ready dataset and methodology notebook.
- [ ] Preserve the canonical benchmark manifest, configs, raw runs, and metrics.
- [ ] Preserve the surrogate bundle and metadata.
- [ ] Preserve the analysis, validation, reproduction, and test code.
- [ ] Regenerate derived outputs with `python reproduce.py`.
- [ ] Run `python reproduce.py --check`.
- [ ] Run the package tests.
- [ ] Record Python, package, CUDA, GPU, and operating-system versions.
- [ ] Keep a SHA-256 manifest of the final paper archive.
- [ ] Archive or delete Review/Destructive items only after an external backup exists.