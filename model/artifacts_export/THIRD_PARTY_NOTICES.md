# Third-Party Notices

The GreenPEFT surrogate export is authored by this project, but its feature
catalog and training evidence reference third-party models, datasets, and
software. This notice is a reference list, not a replacement for the terms
published by each upstream provider.

## Referenced Hugging Face model repositories

The catalog in `configs/backbones.yaml` references these repositories:

- [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)
- [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
- [HuggingFaceTB/SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B)
- [EleutherAI/pythia-1.4b](https://huggingface.co/EleutherAI/pythia-1.4b)
- [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)
- [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)

Do not assume that these repositories share the MIT license. Check each
repository's model card, LICENSE file, and usage restrictions at release time.
Publishing this surrogate export does not publish or redistribute those model
weights.

## Referenced dataset

The benchmark configuration references
[stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2). Review the
dataset card and terms before redistributing dataset-derived files. The
surrogate dataset contains benchmark features and targets; confirm that its
redistribution is permitted for your intended use.

## Software dependencies

The CLI and training workflow use third-party Python packages, including
scikit-learn, pandas, NumPy, joblib, PyYAML, PyTorch, Transformers, PEFT,
Datasets, Accelerate, BitsAndBytes, CodeCarbon, and related dependencies.
Their licenses are governed by the packages installed in the target
environment. Generate a pinned software bill of materials for a production
release.
