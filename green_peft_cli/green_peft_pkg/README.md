# green-peft

Constraint-aware PEFT strategy recommender. Wraps the surrogate models trained in
Cell 13 of the GreenPEFT notebook so you can ask "which method + model size should I
use?" from the command line, without running anything.

## Install

```bash
pip install -e .
```

## Export artifacts from your notebook run

The CLI needs two things out of your Kaggle/Colab `peft_bench/` folder:

```
peft_bench/results/surrogate_models.joblib   # written by Cell 13
peft_bench/configs/backbones.yaml            # written by Cell 3 (has model_zoo:)
peft_bench/configs/methods/*.yaml            # written by Cell 3
```

Zip `results/` and `configs/` together, download from Kaggle, unzip locally into
e.g. `./peft_bench_export/`, and point `--artifacts-dir` at it.

This repository includes a ready-to-use export at:

```text
../../surrogate_artifacts_export/artifacts_export/
├── configs/backbones.yaml
├── configs/methods/*.yaml
├── configs/tasks.yaml
├── results/surrogate_models.joblib
├── results/surrogate_cv_metrics.json
└── results/surrogate_dataset.csv
```

From the repository root, install the package and use that export directly:

```powershell
cd green_peft_cli/green_peft_pkg
python -m pip install -e .
green-peft list-zoo --artifacts-dir ../../surrogate_artifacts_export/artifacts_export
```

## Usage

```bash
# Basic recommendation under a VRAM + accuracy constraint
green-peft recommend --artifacts-dir ./peft_bench_export \
    --vram 16 --accuracy 0.90 --profile balanced

# Strict carbon budget, custom weight profile, JSON output for scripting
green-peft recommend --artifacts-dir ./peft_bench_export \
    --vram 24 --carbon 0.003 --weights 0.3,0.2,0.4,0.1 --top-k 5 --json

# See every backbone x method combination the engine can currently score
green-peft list-zoo --artifacts-dir ./peft_bench_export
```

## What it actually does

1. Builds a candidate table crossing every backbone in your `model_zoo` (including
   ones you never ran) with every configured method.
2. Runs the four surrogate models (feasibility classifier + accuracy / peak-VRAM /
   energy / wall-clock regressors) over every candidate.
3. Drops candidates the feasibility classifier doesn't trust, or that violate your
   `--vram` / `--carbon` / `--accuracy` / `--time` constraints.
4. Computes the Pareto front and a GEI score (same formula as the paper: weighted sum
   of normalized accuracy/memory/carbon/time scores) over the surviving candidates.
5. Prints the top-k ranked options with a plain-English explanation, or an explicit
   breakdown of why nothing survived if your constraints are infeasible together.

Predictions are only as reliable as the surrogate's leave-one-tier-out validation
(`results/surrogate_cv_metrics.json` from Cell 13) says they are for that target --
check `regression.<target>.best_model` R2 there before trusting a specific number.

## Current validation and limitations

The included export reports leave-one-tier-out feasibility accuracy of `0.7833`.
Selected regression models report the following validation results:

| Target | Model | MAE | R2 |
| --- | --- | ---: | ---: |
| Accuracy | Gradient boosting | 0.0182 | -0.4573 |
| Peak VRAM | Gradient boosting | 3.9703 GB | -0.5154 |
| Energy | Ridge | 0.000563 kWh | 0.3740 |
| Wall-clock time | Gradient boosting | 33.12 s | 0.4077 |

Use the CLI for experiment planning, shortlist generation, and explicit budget checks.
It does not run fine-tuning, measure the target hardware, or guarantee performance.
The benchmark evidence is primarily SST-2 classification on a Tesla T4; recommendations
for unbenchmarked catalog entries are extrapolations. Validate the selected option with
a real run and retain the JSON output alongside the measured results.
