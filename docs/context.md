# GreenPEFT: Master Research & Execution Context

> **A Multi-Objective Green AI Decision Support Framework for Sustainable Parameter-Efficient Fine-Tuning of Large Language Models**

**Status:** 🟡 Active Research / Systems & Framework Development  
**Primary Hardware:** Kaggle Notebooks / Single NVIDIA Tesla T4 (16 GB VRAM)  
**Target Venues:** SustaiNLP, EMNLP Efficiency Workshops, MLSys (Tools/Systems Track)

---

## 1. Executive Summary & Core Pivot

- **Core Shift:** Transitioned from a conventional, static benchmarking paper (*"Which PEFT uses less energy?"*) into a predictive, constraint-aware **Green AI Decision Support Framework** (*"Which PEFT method is optimal under user-defined VRAM, carbon, energy, and accuracy constraints?"*)[cite: 5].
- **Primary Novelty (RQ4):** Building a lightweight **surrogate / predictive model** that estimates accuracy, peak VRAM, training time, energy (kWh), and carbon ($kgCO_2eq$) directly from cheap pre-training metadata (model scale, adapter rank, quantization bits, dataset size, task type)[cite: 5]. This eliminates the need to run exhaustive candidate trainings before giving a recommendation[cite: 5].

---

## 2. Research Questions

- **RQ1 (Sustainability Benchmarking):** How do popular adaptation strategies (Full FT, LoRA, QLoRA, LoRA-FA, LISA) compare across task performance, peak GPU memory, training time, energy draw, carbon emissions, and monetary cost under identical controlled environments[cite: 5]?
- **RQ2 (Multi-Objective Trade-offs):** Which PEFT configurations lie on the Pareto-frontier when accuracy is jointly traded off against energy, carbon, and hardware footprint[cite: 5]?
- **RQ3 (Decision Support):** Can a constraint-aware decision engine reliably select feasible, Pareto-optimal PEFT strategies under explicit user constraints (e.g., VRAM $\le 16\text{ GB}$, carbon budget, accuracy floor)[cite: 5]?
- **RQ4 (Predictive Recommendation — Core Novelty):** Can a lightweight surrogate regressor accurately predict performance and sustainability metrics for unseen model/hardware/task combinations using pre-training features, enabling zero-shot strategy selection[cite: 5]?

---

## 3. Key Research Architecture & Pipeline

```text
                        USER CONSTRAINTS
                  (VRAM, Carbon, Accuracy, Time)
                               │
                               ▼
                   CHEAP PRE-TRAINING FEATURES
             (Param Count, Rank, Quant Bits, Task)
                               │
                               ▼
                   SURROGATE PREDICTIVE MODEL
             (Predicts Accuracy, VRAM, Energy, Carbon)
                               │
                               ▼
                    GREEN AI DECISION ENGINE
                     (Constraint Filtering)
                               │
                               ▼
                    PARETO & GEI EVALUATION
                               │
                               ▼
                   RECOMMENDED PEFT STRATEGY
                     + TRADE-OFF EXPLANATION
```[cite: 5]

---

## 4. Current Empirical Status & Pilot Observations

### Prior PDF / Synthetic Phase (Mode-A)
- Earlier iterations relied on **Mode-A synthetic mock data profiles** to validate Pareto plotting, hypervolume indicator computation, and multi-criteria scoring scripts[cite: 5].
- Single real smoke-test was a 6-second CPU test (Iowa grid), excluded from empirical comparisons[cite: 5].

### Kaggle T4 Pilot Experiment (2026-08-10)
- **Environment:** NVIDIA Tesla T4 (16 GB VRAM)[cite: 5].
- **Execution Rate:** 12 of 60 planned runs completed successfully; 48 failed due to VRAM Out-Of-Memory (OOM) errors or dependency conflicts (`bitsandbytes`, `torchao`)[cite: 5].
- **Empirical Findings:**
  - **Full Fine-Tuning (0.5B):** Succeeded with **91.2% accuracy**[cite: 5]. Full FT beyond 0.5B exceeded the 16 GB VRAM limit[cite: 5].
  - **LISA Performance:** 0.5B model achieved 50.9% accuracy; 1.1B model achieved 61.2%; 1.5B model achieved 48.9% (indicates LISA implementation requires hyperparameter tuning/ablation)[cite: 5].
  - **Failures:** OOMs logged as valid resource-feasibility data points; dependency errors flagged as environment issues to fix[cite: 5].

---

## 5. Comprehensive Strategic Roadmap (Tiers 1–3)

### Tier 1 — Core Methodological Requirements (Mandatory)
1. **Real Mode-B GPU Execution:** Complete at least one clean, real CodeCarbon-tracked grid (minimum 2 backbones × 1 task × 3 PEFT methods across $k=3$ seeds)[cite: 5].
2. **Principled Stochastic Variance:** For any mock/simulated pipeline components, utilize true Gaussian distributions (`np.random.normal`) rather than deterministic seed-modulo formulas to protect statistical integrity[cite: 5].

### Tier 2 — Novelty & Systems Upgrade (Main Paper Contributions)
1. **Surrogate / Predictive Model:** Train lightweight regressors (Linear Regression, Gradient Boosting/XGBoost) predicting accuracy, energy, carbon, and peak memory from metadata features[cite: 5].
2. **Modern Method Integration:** Extend beyond standard baselines by implementing **DoRA** and/or **GaLore**[cite: 5].
3. **Defensible GEI Weights:** Replace arbitrary scoring weights ($0.35/0.15/0.25/0.25$) with weights derived from Analytic Hierarchy Process (AHP) surveys or public Hugging Face deployment preferences[cite: 5].
4. **Scale Validation of Engine:** Benchmark recommendation stability, coverage, and constraint satisfaction over a broad grid (VRAM $\in \{8, 16, 24, 48\text{ GB}\}$, 5 carbon budget levels, 3 accuracy floors)[cite: 5].
5. **Explicit Differentiation Section:** Include a dedicated comparison section/table against Chen et al. (2024) and Kaur et al. (2026) detailing inputs, outputs, metric formulas, and constraint handling[cite: 5].

### Tier 3 — Tooling, Systems Polish & Publication
1. **Open-Source CLI Tool Release:** Package an executable CLI tool (`green-peft recommend --vram 16 --carbon 0.05 --accuracy 0.85`) to position the work as a systems contribution[cite: 5].
2. **Cross-Hardware Generalization:** Train surrogate models on one GPU class (e.g., A100) and evaluate prediction error when deploying recommendations to a different class (e.g., T4)[cite: 5].
3. **Targeted Venue Strategy:** Target specialized efficient ML / sustainability venues (SustaiNLP, Efficient NLP @ EMNLP/ACL, MLSys workshops) first before scaling to broader general ML conferences[cite: 5].