# GreenPEFT: A Multi-Objective Green AI Decision Support Framework

> **Sustainable Parameter-Efficient Fine-Tuning of Large Language Models via Predictive Zero-Shot Constraint Filtering and Green Efficiency Index (GEI)**

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![CodeCarbon](https://img.shields.io/badge/CodeCarbon-2.8%2B-brightgreen.svg)](https://codecarbon.io/)
[![PyPI](https://img.shields.io/badge/PyPI-green--peft%200.3.0-blue.svg)](https://pypi.org/project/green-peft/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-ai--tanzil%2FGreenPEFT-yellow.svg)](https://huggingface.co/ai-tanzil/GreenPEFT)

---

## 📌 Executive Summary

**GreenPEFT** shifts the paradigm of sustainable machine learning from static post-hoc benchmarking (*"Which method was most energy-efficient in past runs?"*) to **predictive, constraint-aware decision support** (*"Which PEFT configuration dynamically satisfies user VRAM, carbon, energy, and accuracy constraints while maximizing resource efficiency?"*).

Instead of requiring practitioners to run expensive empirical sweeps prior to selecting a fine-tuning strategy, GreenPEFT leverages a **zero-shot surrogate regressor** trained on pre-training metadata features (backbone parameters, adapter rank, quantization bits, dataset size) to forecast task performance and environmental cost. Combined with a multi-objective decision engine, Pareto-frontier filtering, and the **Green Efficiency Index (GEI)**, GreenPEFT prescribes the optimal adaptation strategy in sub-second inference time.

---

## 🔄 End-to-End Pipeline Overview

The GreenPEFT framework operates across **5 integrated stages**, seamlessly connecting data collection, empirical benchmarking, surrogate regression, constraint-aware decision support, and CLI execution.

```mermaid
flowchart TD
    subgraph Stage1 ["Stage 1: Multi-Source Data Harmonization"]
        A1["Raw Datasets (data/raw/)<br>• surrogate_dataset.csv<br>• Open LLM-Perf Leaderboard<br>• llmenergy.csv"] --> A2["Feature Engineering & Filtering<br>(notebooks/data_preprocessing...)"]
        A2 --> A3["ML-Ready Harmonized Dataset<br>(data/processed/greenpeft_ml_ready_dataset.csv)<br>477 rows × 46 features"]
    end

    subgraph Stage2 ["Stage 2: Empirical Benchmarking Sweep"]
        B1["Backbones (0.5B - 3.0B)<br>• Qwen2.5, TinyLlama, Llama3"] --> B2["PEFT Strategies<br>• Full-FT, LoRA, QLoRA<br>• LoRA-FA, LISA"]
        B2 --> B3["NVIDIA Tesla T4 Grid Sweep<br>(CodeCarbon + Peak VRAM Tracker)"]
        B3 --> B4["Canonical Benchmark Artifacts<br>(results/canonical_benchmark/)"]
    end

    subgraph Stage3 ["Stage 3: Zero-Shot Surrogate Regressor"]
        C1["Pre-Run Metadata Features<br>(Backbone params, Adapter rank, Quant bits, Dataset size)"] --> C2["Ridge Regressors (surrogate/validate.py)<br>• Peak VRAM (GB)<br>• Energy (kWh)<br>• Wall-Clock Time (s)<br>• Task Accuracy"]
        C2 --> C3["Audited Artifacts<br>(data/processed/greenpeft_surrogate_models.joblib)<br>+ models/model_metadata.json"]
    end

    subgraph Stage4 ["Stage 4: Constraint Engine & GEI Scoring"]
        D1["User Constraints<br>(VRAM, Carbon, Accuracy floor)"] --> D2["Feasibility Filter & Scope Check"]
        D2 --> D3["Pareto Frontier Extraction"]
        D3 --> D4["Green Efficiency Index (GEI)<br>Multi-Objective Scoring"]
        D4 --> D5["Recommended PEFT Strategy"]
    end

    subgraph Stage5 ["Stage 5: System Interface & Back-Testing"]
        E1["CLI Tool (green-peft recommend)"] --> E2["Empirical Run Verification<br>(experiments/validate_recommendation.py)"]
        E2 --> E3["Validation Log & Error Tracking<br>(results/recommendation_validation.csv)"]
    end

    A3 --> C1
    B4 --> C1
    C3 --> D2
```

### Stage Breakdown:
1. **Data Harmonization (`data/raw/` → `data/processed/`):** Merges multi-source fine-tuning and inference traces into a unified 477×46 feature dataset with rigorous measurement-validity filtering.
2. **Empirical Benchmarking (`results/canonical_benchmark/`):** Executes 60 systematic sweeps across 5 PEFT strategies (Full-FT, LoRA, QLoRA, LoRA-FA, LISA) on NVIDIA Tesla T4 GPUs with CodeCarbon tracking.
3. **Surrogate Regressor (`surrogate/validate.py`):** Fits zero-shot Ridge pipelines using pre-run metadata to estimate Peak VRAM, Energy (kWh), Wall-Clock Time, and Accuracy.
4. **Decision Engine & GEI (`green_peft/recommender.py`):** Filters infeasible candidates against hard resource constraints, constructs the non-dominated Pareto frontier, and scores survivors using the Green Efficiency Index.
5. **CLI & Empirical Back-Testing (`green_peft_cli`):** Provides instant CLI recommendations (`green-peft recommend`) and continuous back-testing against measured targets.

---

## 📑 Core Documentation & Repository Artifacts

| Category | File / Path | Description & Purpose |
| :--- | :--- | :--- |
| **Pipeline Reproduction** | [`reproduce.py`](reproduce.py) | Single command to verify and regenerate every derived artifact and test suite. |
| **Main Research Notebook** | [`notebooks/green-peft.ipynb`](notebooks/green-peft.ipynb) | End-to-end research notebook covering EDA, Pareto frontiers, and surrogate evaluations. |
| **Preprocessing Notebook** | [`notebooks/data_preprocessing_and_feature_engineering.ipynb`](notebooks/data_preprocessing_and_feature_engineering.ipynb) | Stages 1–8 feature pipeline generating the harmonized ML-ready dataset. |
| **Benchmark Notebook** | [`notebooks/green_peft_benchmark_execution.ipynb`](notebooks/green_peft_benchmark_execution.ipynb) | Kaggle T4 benchmark execution pipeline notebook. |
| **Surrogate Audit Report** | [`docs/AUDIT_REPORT.md`](docs/AUDIT_REPORT.md) | Audit analysis, GroupKFold / LOTO evaluation, and surrogate validation status. |
| **Implementation Report** | [`docs/IMPLEMENTATION_REPORT.md`](docs/IMPLEMENTATION_REPORT.md) | System completeness, audit envelope validation, and empirical scope breakdown. |
| **Reproduction Guide** | [`docs/PAPER_REPRODUCTION_GUIDE.md`](docs/PAPER_REPRODUCTION_GUIDE.md) | Paper reproduction guide and clean repository manifest. |
| **Data Provenance Guide** | [`docs/DATA_METHODOLOGY.md`](docs/DATA_METHODOLOGY.md) | Data sources, feature dictionary, and measurement pass filtering rules. |
| **Audited Model Metadata** | [`models/model_metadata.json`](models/model_metadata.json) | Machine-readable artifact metadata, training envelope, and error bounds. |
| **CLI Export Bundle** | [`models/artifacts_export/`](models/artifacts_export/) | Portable export bundle containing trained models and configurations for CLI offline use. |
| **Canonical Benchmark** | [`results/canonical_benchmark/`](results/canonical_benchmark/) | Verified raw JSON runs, aggregate metrics, and figures from the 60-run Kaggle T4 study. |
| **CLI Package** | [`green_peft_cli/green_peft_pkg/`](green_peft_cli/green_peft_pkg/) | Source code and test suite for the published `green-peft` PyPI package. |
| **Interactive Dashboard** | [`website/`](website/) | Web interface for constraint exploration and interactive recommendations. |

---

## 📊 Empirical Pilot Findings (NVIDIA Tesla T4)

Below are the empirical findings from the 60-run benchmark grid executed on Kaggle using an **NVIDIA Tesla T4 GPU (16 GB VRAM)** on the SST-2 classification task ($k=3$ random seeds):

| Strategy | Backbone | Params | Accuracy (mean ± std) | Wall-clock (s) | Peak VRAM (GB) | Energy (kWh) | Carbon ($\text{kgCO}_2\text{eq}$) | GEI Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full_ft** | Qwen2.5-Tiny | 0.5B | **0.9122 ± 0.0117** | 222.77 | 13.48 | 0.00382 | 0.00248 | 0.964 |
| **lisa** | Qwen2.5-Tiny | 0.5B | 0.5089 ± 0.0165 | **182.87** | **4.84** | **0.00324** | **0.00211** | **0.999** |
| **lisa** | TinyLlama-Small | 1.1B | 0.6122 ± 0.0901 | 467.91 | 9.43 | 0.00833 | 0.00541 | 0.822 |
| **lisa** | Qwen2.5-Medium | 1.5B | 0.4889 ± 0.0342 | 636.33 | 14.88 | 0.01124 | 0.00731 | 0.751 |

### Key Takeaways:
1. **Full-FT Memory Ceiling:** Full Fine-Tuning hits a hard VRAM ceiling at 0.5B parameters ($13.48$ GB peak VRAM). At $\ge 1.1\text{B}$, Full-FT immediately fails with Out-Of-Memory (OOM) errors.
2. **LISA Hardware Scaling:** LISA enables a 1.5B model to execute within a 16 GB VRAM budget ($14.88$ GB peak), where Full-FT fails completely.
3. **GEI Dominance:** LISA on 0.5B achieves the highest GEI score ($0.999$) due to its compact $4.84$ GB memory footprint and minimal carbon emissions.

---

## ⚖️ Green Efficiency Index (GEI) Formula

The **Green Efficiency Index (GEI)** synthesizes multi-dimensional objectives into a single decision score:

$$\text{GEI}(\mathbf{c}) = w_{\text{Acc}} \hat{S}_{\text{Acc}} + w_{\text{Mem}} \hat{S}_{\text{Mem}} + w_{C} \hat{S}_{C} + w_{T} \hat{S}_{T}$$

Where normalized metric scores are defined as:
$$\hat{S}_{\text{Acc}} = \frac{\text{Acc} - \text{Acc}_{\min}}{\text{Acc}_{\max} - \text{Acc}_{\min}}, \quad \hat{S}_{M} = 1 - \frac{M - M_{\min}}{M_{\max} - M_{\min}} \quad (M \in \{\text{Mem}, C, T\})$$

Weights satisfy $\sum w_i = 1$ based on user preference profiles:
- **Balanced Profile:** $w = [0.35, 0.25, 0.25, 0.15]$
- **Strict Carbon Profile:** $w = [0.20, 0.20, 0.50, 0.10]$
- **High-Accuracy Profile:** $w = [0.60, 0.15, 0.15, 0.10]$

---

## 📁 Clean Repository Structure

```text
.
├── README.md                           # Main documentation & pipeline overview
├── LICENSE                             # MIT License
├── CITATION.cff                        # Citation metadata
├── reproduce.py                        # Single command to reproduce all artifacts & tests
├── data/                               # Harmonized & raw datasets
│   ├── raw/                            # Source datasets (surrogate_dataset, LLM-Perf, etc.)
│   └── processed/                      # Harmonized 477x46 dataset & surrogate bundle (.joblib)
├── notebooks/                          # Categorized research & execution notebooks
│   ├── green-peft.ipynb                # Primary research & surrogate evaluation notebook
│   ├── green_peft_benchmark_execution.ipynb  # Kaggle T4 empirical benchmark execution notebook
│   ├── data_preprocessing_and_feature_engineering.ipynb # Feature pipeline (Stages 1–8)
│   └── legacy/                         # Historical audit notebooks
├── docs/                               # Comprehensive project documentation & reports
│   ├── AUDIT_REPORT.md                 # Surrogate audit & cross-validation metrics
│   ├── IMPLEMENTATION_REPORT.md        # System implementation & validation status
│   ├── PAPER_REPRODUCTION_GUIDE.md     # Step-by-step reproduction guide & file manifest
│   ├── DATA_METHODOLOGY.md             # Data harmonization rules & feature dictionary
│   ├── greenpeft_agent_workflow.md     # Agent specification & task roadmap
│   ├── progress_summary.md             # Development summary & milestones
│   ├── context.md                      # Strategic project context
│   └── author.md                       # Author profile & contact
├── analysis/                           # Reproduction & analysis pipeline scripts
│   ├── greenpeft_data.py               # Shared data loader & canonical filtering
│   ├── build_training_envelope.py      # Measured envelope builder for metadata
│   ├── benchmark_analysis.py           # Summary table generator
│   ├── pareto_analysis.py              # Pareto frontier extractor
│   ├── recommendation_scenarios.py     # Recommendation scenario builder (Table 4)
│   ├── figures.py                      # Publication figure generators
│   └── build_hf_export.py              # Model card & HF export package builder
├── surrogate/                          # Surrogate validation
│   └── validate.py                     # Grouped CV & extrapolation auditor
├── experiments/                        # Empirical validation & back-testing
│   ├── validate_recommendation.py      # Predicted vs actual empirical back-testing
│   └── recorded_run.template.json      # Template for logging new empirical runs
├── models/                             # Audited model metadata & CLI export bundle
│   ├── model_metadata.json             # Machine-readable model envelope & audit record
│   └── artifacts_export/               # Export bundle used by CLI recommender
├── green_peft_cli/                     # Installable green-peft CLI package
│   └── green_peft_pkg/                 # Package source, pyproject.toml & pytest suite
├── results/                            # Canonical & derived experimental outputs
│   ├── canonical_benchmark/            # Source-of-truth 60-run Kaggle T4 sweep
│   ├── benchmark/                      # Feasibility matrix & summary tables
│   ├── pareto/                         # Pareto frontiers & candidate sets
│   ├── recommendations/                # Decision scenario JSON/CSV outputs
│   ├── surrogate/                      # CV metrics & actual vs predicted figures
│   ├── hf_export/                      # Published HuggingFace artifact export
│   └── recommendation_validation.csv   # Predicted vs actual validation log
└── website/                            # Interactive Web UI dashboard
```

---

## 🚀 Quickstart & Installation

### 1. Installation

Install the published CLI package directly from PyPI:

```bash
pip install green-peft
```

Or install locally from source:

```bash
pip install -e green_peft_cli/green_peft_pkg
```

### 2. Run Constraint-Aware Recommendation CLI

Request a balanced recommendation under explicit resource constraints:

```bash
green-peft recommend --vram 16 --accuracy 0.90 --profile balanced
```

Request a strict carbon-constrained recommendation with JSON output:

```bash
green-peft recommend --vram 16 --carbon 0.003 --accuracy 0.90 --profile strict_carbon --json
```

List available candidate backbones and methods in the zoo:

```bash
green-peft list-zoo --artifacts-dir ./models/artifacts_export
```

### 3. Reproduce All Derived Artifacts

Run the single-command reproduction pipeline to verify all tables, figures, and unit tests:

```bash
python reproduce.py
```

Run in `--check` mode to verify recorded artifacts without mutating the workspace:

```bash
python reproduce.py --check
```

---

## 📜 Citation

If you use **GreenPEFT** in your research, please cite:

```bibtex
@misc{ashraful_islam_tanzil_2026,
    author       = {Ashraful Islam Tanzil},
    title        = {GreenPEFT: A Multi-Objective Green AI Decision Support Framework},
    year         = {2026},
    url          = {https://huggingface.co/ai-tanzil/GreenPEFT},
    doi          = {10.57967/hf/10504},
    publisher    = {Hugging Face}
}
```

---
*License: [MIT License](LICENSE)*
