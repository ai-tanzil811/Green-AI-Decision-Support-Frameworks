---
license: mit
library_name: scikit-learn
pipeline_tag: tabular-regression
tags:
- green-ai
- peft
- sustainability
- resource-estimation
- scikit-learn
---

# GreenPEFT Surrogate Artifacts

This repository contains the GreenPEFT surrogate models used by the
`green-peft` constraint-aware PEFT recommendation CLI.

## Files

- `results/surrogate_models.joblib`: trained feasibility classifier and regressors.
- `results/surrogate_cv_metrics.json`: leave-one-tier-out validation metrics.
- `results/surrogate_dataset.csv`: feature and target table used for training.
- `configs/backbones.yaml`: candidate backbone catalog.
- `configs/methods/`: PEFT method configurations.
- `configs/tasks.yaml`: task metadata.
- `LICENSE`: license for GreenPEFT-owned artifacts.
- `THIRD_PARTY_NOTICES.md`: upstream model and dataset references.

## Intended use

Use these artifacts to shortlist PEFT methods and backbone sizes under VRAM,
carbon, accuracy, and runtime constraints. They are research and experiment-
planning artifacts, not a general-purpose language model and not a guarantee
that a proposed configuration will train successfully on new hardware or tasks.

The evidence base is primarily SST-2 classification on an NVIDIA Tesla T4.
The export reports leave-one-tier-out feasibility accuracy of approximately
0.7833. Accuracy and peak-VRAM regression remain weak under this validation,
so every recommendation should be verified with an actual measured run.

The benchmark-derived surrogate dataset is published on Kaggle:

[ashrafulislamtanzil/greenpeft-surrogate-data](https://www.kaggle.com/datasets/ashrafulislamtanzil/greenpeft-surrogate-data)

## Use with the CLI

```bash
pip install -e path/to/green_peft_cli/green_peft_pkg
green-peft recommend \
  --artifacts-dir path/to/artifacts_export \
  --vram 16 --accuracy 0.90 --profile balanced --json
```

## Loading warning

`surrogate_models.joblib` is a serialized Python artifact. Do not load it from
an untrusted source. Use a pinned environment and review the artifact before
loading it in a sensitive or production process.

## License

The GreenPEFT-owned files are released under the MIT License in `LICENSE`.
The license does not grant rights to upstream models, tokenizers, datasets, or
software listed in `THIRD_PARTY_NOTICES.md`. Review each upstream repository's
current license and usage terms before redistributing or deploying a referenced
backbone or dataset.
