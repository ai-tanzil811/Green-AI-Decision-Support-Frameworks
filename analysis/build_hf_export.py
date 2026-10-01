#!/usr/bin/env python
"""
Assemble the Hugging Face model repository payload for the v3 surrogate bundle.

Writes results/hf_export/, ready to upload to https://huggingface.co/ai-tanzil/GreenPEFT:

    README.md                      model card, generated from model_metadata.json
    greenpeft_surrogate_models.joblib
    model_metadata.json
    configs/backbones.yaml, configs/tasks.yaml, configs/methods/*.yaml

The card's performance numbers are read out of model_metadata.json rather than typed in, so
they cannot drift from the artifact they describe.

Usage:
    python analysis/build_hf_export.py

Then (needs an HF token with write access):
    hf upload ai-tanzil/GreenPEFT results/hf_export . --repo-type=model
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from greenpeft_data import ROOT  # noqa: E402

OUT = ROOT / 'results' / 'hf_export'
BUNDLE = ROOT / 'new update works' / 'greenpeft_surrogate_models.joblib'
METADATA = ROOT / 'models' / 'model_metadata.json'
CONFIGS = ROOT / 'green_peft_cli' / 'green_peft_pkg' / 'green_peft' / 'data' / 'configs'

REPO_ID = 'ai-tanzil/GreenPEFT'

TARGET_LABEL = {
    'accuracy': 'Accuracy',
    'peak_gpu_memory_gb': 'Peak VRAM (GB)',
    'energy_kwh': 'Energy (kWh)',
    'wall_clock_seconds': 'Wall-clock (s)',
}


def metrics_table(meta: dict) -> str:
    detail = meta['validation']['per_target_detail']
    rows = ['| Target | Interp. R² | Extrap. R² | Interp. MAPE | Extrap. MAPE | Status |',
            '| :--- | ---: | ---: | ---: | ---: | :--- |']
    for target, label in TARGET_LABEL.items():
        d = detail[target]
        i, e = d['groupkfold_config_id'], d['leave_one_tier_out']
        status = d['status_governing']
        mark = f'**{status}**' if status == 'VALIDATED' else status
        rows.append(f'| {label} | {i["R2"]:.3f} | {e["R2"]:.3f} | '
                    f'{i["MAPE_pct"]:.1f}% | {e["MAPE_pct"]:.1f}% | {mark} |')
    return '\n'.join(rows)


def build_card(meta: dict) -> str:
    env = meta['training_envelope']
    v = meta['validation']
    return f"""---
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

# GreenPEFT Surrogate Models (v{meta['schema_version']})

Pre-run cost surrogates for parameter-efficient fine-tuning (PEFT). Given a configuration that
has **not been run yet**, they predict what it will cost: accuracy, peak VRAM, training energy
and wall-clock time. They are the prediction component of the GreenPEFT decision-support
framework, not a standalone recommender.

- **Code and CLI:** [`green-peft` on PyPI](https://pypi.org/project/green-peft/) ·
  [GitHub](https://github.com/ai-tanzil811/Green-AI-Decision-Support-Frameworks)
- **Training data:** [greenpeft-surrogate-data](https://www.kaggle.com/dsv/20178095)
  (DOI `10.34740/KAGGLE/DSV/20178095`)

## Status: {v['status_overall']}

**This model is a shortlisting aid, not an approval to skip measuring.** Of the four targets,
only energy clears the project's thresholds under both validation protocols.

{metrics_table(meta)}

Two protocols are reported because they answer different questions. *Interpolation*
(`GroupKFold(groups=config_id)`) tests a new seed of an already-measured configuration.
*Extrapolation* (`LeaveOneGroupOut(groups=backbone)`) tests an unseen model scale. The
governing status is the weaker of the two, since the point of a surrogate is to screen
configurations nobody has measured.

> **Read the interpolation column with care.** `config_id` contains the seed, so grouping on it
> separates only the dataset's two measurement passes — two seeds of a configuration can train
> while the third is tested. Re-validated with seeds held together (grouping on method +
> backbone), accuracy R² falls from {v['per_target_detail']['accuracy']['groupkfold_config_id']['R2']:.3f}
> to 0.39 and energy from {v['per_target_detail']['energy_kwh']['groupkfold_config_id']['R2']:.3f}
> to 0.88. The **extrapolation column is unaffected** by this and is the column to trust.

## Intended use

Screening candidate PEFT configurations under VRAM, accuracy, energy, carbon or runtime
constraints, to decide which ones are worth actually running.

**Out of scope:** automatic production approval, hardware other than the measured one, tasks
other than SST-2, and any claim that one PEFT method is universally best. The benchmark is a
feasibility and multi-objective study, not a method ranking.

## Training data and envelope

Fitted on {meta['training_configs']} configurations, collapsed from
{meta['training_rows']} measurement rows (two passes each):

| Property | Value |
| :--- | :--- |
| Parameter range | {env['params_b']['min']}–{env['params_b']['max']} B |
| Model families | {', '.join(env['families'])} |
| Methods | {', '.join(env['methods'])} |
| Measured (method, scale) cells | {len(env['measured_cells'])} |
| Task | {env['task']} |
| GPU budget | {env['gpu_total_gb']} GB (NVIDIA T4) |

A candidate outside this envelope is an **extrapolation**. The `green-peft` CLI labels every
prediction with a scope and confidence level and attaches the matching error band; predictions
beyond {env['params_b']['max']} B carry an explicit warning that the band is a *lower bound*,
because the extrapolation protocol only ever held out a tier inside the measured range.

## Important caveats

**Carbon is derived, not measured.** `carbon_kgco2eq = energy_kwh × {meta['carbon_formula'].split('*')[-1].strip()}`
exactly. There is no carbon model in this bundle and there should not be one — derive it from
the energy prediction and label it as an assumption.

**Energy telemetry.** The source benchmark measured each configuration twice. The second pass
recorded a flat ~10 W implied power — the NVML idle floor rather than loaded training power —
and is excluded via `energy_measurement_valid == 1`. Pooling both passes understates mean
energy by ~42%. Only one valid energy pass exists per configuration, so **energy has no
replication**; a second valid pass is the highest-value missing experiment.

**Feasibility is out of scope.** {meta['scope']}

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

config = pd.DataFrame([{{
    "method": "lora", "family": "llama", "params_b": 1.1,
    "rank": 16, "quant_bits": 16,
}}])
X = build_pre_run_features(config)

energy = models["energy_kwh"].predict(X)[0]
carbon = energy * bundle["carbon_intensity_kg_per_kwh"]
print(f"predicted energy {{energy:.6f}} kWh -> derived carbon {{carbon:.6f}} kgCO2eq")
```

Or, with scope and error bands handled for you:

```bash
pip install green-peft
green-peft recommend --vram 16 --accuracy 0.90 --profile balanced
green-peft recommend --vram 16 --accuracy 0.90 --min-confidence HIGH   # measured cells only
```

## Feature contract

Never use an observed outcome as an input. Every feature must be knowable before the run
starts. `bundle["features"]["all"]` is the authoritative ordered list ({len(meta['features'])}
columns); these are **targets, not inputs**, and are guarded against in the training code:

`{'`, `'.join(meta['leakage_guard'])}`

## Files

| File | Contents |
| :--- | :--- |
| `greenpeft_surrogate_models.joblib` | The v{meta['schema_version']} bundle: four sklearn Pipelines plus metadata |
| `model_metadata.json` | Features, targets, CV metrics, status, training envelope |
| `configs/` | Backbone catalog, task and method definitions |

Fitted with scikit-learn {meta['sklearn_version']}; loading with a different minor version may
warn or fail.

## Citation

```bibtex
@misc{{ashraful_islam_tanzil_2026,
  title={{greenpeft-surrogate-data}},
  url={{https://www.kaggle.com/dsv/20178095}},
  DOI={{10.34740/KAGGLE/DSV/20178095}},
  publisher={{Kaggle}},
  author={{Ashraful Islam Tanzil}},
  year={{2026}}
}}
```
"""


def main() -> int:
    for path in (BUNDLE, METADATA, CONFIGS):
        if not path.exists():
            print(f'error: missing {path}', file=sys.stderr)
            return 2

    meta = json.loads(METADATA.read_text(encoding='utf-8'))
    if 'training_envelope' not in meta:
        print('error: model_metadata.json has no training_envelope; '
              'run analysis/build_training_envelope.py first', file=sys.stderr)
        return 2

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    shutil.copy2(BUNDLE, OUT / BUNDLE.name)
    shutil.copy2(METADATA, OUT / 'model_metadata.json')
    shutil.copytree(CONFIGS, OUT / 'configs')
    (OUT / 'README.md').write_text(build_card(meta), encoding='utf-8')

    total = 0
    print(f'wrote {OUT.relative_to(ROOT)}/')
    for f in sorted(OUT.rglob('*')):
        if f.is_file():
            size = f.stat().st_size
            total += size
            print(f'  {f.relative_to(OUT).as_posix():45} {size / 1024:8.1f} KB')
    print(f'  {"total":45} {total / 1024:8.1f} KB')

    print(f'\nReady to publish to {REPO_ID}. Needs an HF token with write access:')
    print(f'    hf auth login')
    print(f'    hf upload {REPO_ID} {OUT.relative_to(ROOT).as_posix()} . --repo-type=model')
    return 0


if __name__ == '__main__':
    sys.exit(main())
