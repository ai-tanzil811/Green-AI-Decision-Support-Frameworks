---
license: mit
library_name: sklearn
tags:
  - green-ai
  - peft
  - lora
  - qlora
  - energy-efficiency
  - carbon-footprint
  - surrogate-model
  - decision-support
---

# GreenPEFT Surrogate Models (v3.0)

Pre-run cost surrogates for parameter-efficient fine-tuning (PEFT). Given a configuration that
has **not been run yet**, they predict what it will cost: accuracy, peak VRAM, training energy
and wall-clock time. They are the prediction component of the GreenPEFT decision-support
framework, not a standalone recommender.

- **Code and CLI:** [`green-peft` on PyPI](https://pypi.org/project/green-peft/) ·
  [GitHub](https://github.com/ai-tanzil811/Green-AI-Decision-Support-Frameworks)
- **Training data:** [greenpeft-surrogate-data](https://www.kaggle.com/dsv/20178095)
  (DOI `10.34740/KAGGLE/DSV/20178095`)

## Status: PRELIMINARY

**This model is a shortlisting aid, not an approval to skip measuring.** Of the four targets,
only energy clears the project's thresholds under both validation protocols.

| Target | Interp. R² | Extrap. R² | Interp. MAPE | Extrap. MAPE | Status |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Accuracy | 0.764 | 0.082 | 0.7% | 1.6% | PRELIMINARY |
| Peak VRAM (GB) | 0.923 | 0.555 | 11.5% | 39.9% | PRELIMINARY |
| Energy (kWh) | 0.996 | 0.897 | 2.6% | 10.6% | **VALIDATED** |
| Wall-clock (s) | 0.984 | 0.531 | 4.6% | 23.0% | PRELIMINARY |

Two protocols are reported because they answer different questions. *Interpolation*
(`GroupKFold(groups=config_id)`) tests a new seed of an already-measured configuration.
*Extrapolation* (`LeaveOneGroupOut(groups=backbone)`) tests an unseen model scale. The
governing status is the weaker of the two, since the point of a surrogate is to screen
configurations nobody has measured.

> **Read the interpolation column with care.** `config_id` contains the seed, so grouping on it
> separates only the dataset's two measurement passes — two seeds of a configuration can train
> while the third is tested. Re-validated with seeds held together (grouping on method +
> backbone), accuracy R² falls from 0.764
> to 0.39 and energy from 0.996
> to 0.88. The **extrapolation column is unaffected** by this and is the column to trust.

## Intended use

Screening candidate PEFT configurations under VRAM, accuracy, energy, carbon or runtime
constraints, to decide which ones are worth actually running.

**Out of scope:** automatic production approval, hardware other than the measured one, tasks
other than SST-2, and any claim that one PEFT method is universally best. The benchmark is a
feasibility and multi-objective study, not a method ranking.

## Training data and envelope

Fitted on 41 configurations, collapsed from
82 measurement rows (two passes each):

| Property | Value |
| :--- | :--- |
| Parameter range | 0.5–3.0 B |
| Model families | llama, qwen |
| Methods | full_ft, lisa, lora, lora_fa, qlora |
| Measured (method, scale) cells | 14 |
| Task | SST-2 classification |
| GPU budget | 15.6 GB (NVIDIA T4) |

A candidate outside this envelope is an **extrapolation**. The `green-peft` CLI labels every
prediction with a scope and confidence level and attaches the matching error band; predictions
beyond 3.0 B carry an explicit warning that the band is a *lower bound*,
because the extrapolation protocol only ever held out a tier inside the measured range.

## Important caveats

**Carbon is derived, not measured.** `carbon_kgco2eq = energy_kwh × 0.65`
exactly. There is no carbon model in this bundle and there should not be one — derive it from
the energy prediction and label it as an assumption.

**Energy telemetry.** The source benchmark measured each configuration twice. The second pass
recorded a flat ~10 W implied power — the NVML idle floor rather than loaded training power —
and is excluded via `energy_measurement_valid == 1`. Pooling both passes understates mean
energy by ~42%. Only one valid energy pass exists per configuration, so **energy has no
replication**; a second valid pass is the highest-value missing experiment.

**Feasibility is out of scope.** Feasible PEFT configurations only. Configurations exceeding the 15.6 GB VRAM budget are outside this study and were dropped at ingest, so these models predict cost GIVEN that a configuration runs -- they do not predict whether it will run.

**Linear extrapolation.** The regressors are Ridge. Far outside the measured range they
extrapolate past physical limits — negative VRAM below ~0.4 B, accuracy above 1.0 above ~7 B.
The CLI drops such predictions before scoring; if you call the bundle directly, check for them.

## Usage

```python
import joblib, pandas as pd

bundle = joblib.load("greenpeft_surrogate_models.joblib")
models = bundle["regressors"]              # accuracy, peak_gpu_memory_gb, energy_kwh, wall_clock_seconds
features = bundle["features"]["all"]       # ordered column list; also ["numeric"] / ["categorical"]

# Build pre-run features with the SAME function used at training time:
#   pip install green-peft
from green_peft.features import build_pre_run_features

config = pd.DataFrame([{
    "method": "lora", "family": "llama", "params_b": 1.1,
    "rank": 16, "quant_bits": 16,
}])
X = build_pre_run_features(config)

energy = models["energy_kwh"].predict(X)[0]
carbon = energy * bundle["carbon_intensity_kg_per_kwh"]
print(f"predicted energy {energy:.6f} kWh -> derived carbon {carbon:.6f} kgCO2eq")
```

Or, with scope and error bands handled for you:

```bash
pip install green-peft
green-peft recommend --vram 16 --accuracy 0.90 --profile balanced
green-peft recommend --vram 16 --accuracy 0.90 --min-confidence HIGH   # measured cells only
```

## Feature contract

Never use an observed outcome as an input. Every feature must be knowable before the run
starts. `bundle["features"]["all"]` is the authoritative ordered list (22
columns); these are **targets, not inputs**, and are guarded against in the training code:

`energy_per_acc_point`, `vram_efficiency`, `throughput_proxy`, `carbon_derived_kg`, `accuracy`, `peak_gpu_memory_gb`, `energy_kwh`, `carbon_kgco2eq`, `wall_clock_seconds`, `status`, `fits`

## Files

| File | Contents |
| :--- | :--- |
| `greenpeft_surrogate_models.joblib` | The v3.0 bundle: four sklearn Pipelines plus metadata |
| `model_metadata.json` | Features, targets, CV metrics, status, training envelope |
| `configs/` | Backbone catalog, task and method definitions |

Fitted with scikit-learn 1.9.1; loading with a different minor version may
warn or fail.

## Citation

```bibtex
@misc{ashraful_islam_tanzil_2026,
  title={greenpeft-surrogate-data},
  url={https://www.kaggle.com/dsv/20178095},
  DOI={10.34740/KAGGLE/DSV/20178095},
  publisher={Kaggle},
  author={Ashraful Islam Tanzil},
  year={2026}
}
```
