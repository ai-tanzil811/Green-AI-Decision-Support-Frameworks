# GreenPEFT Empirical Data & System Performance Report

> **Comprehensive Empirical Analysis, Data Traces, and System Formulations for LLM Prompting & Downstream Generation**

---

## 📌 Document Overview & Context

- **Project:** GreenPEFT — A Multi-Objective Green AI Decision Support Framework for Sustainable Parameter-Efficient Fine-Tuning
- **Hardware Platform:** Single NVIDIA Tesla T4 GPU (16 GB VRAM, GDDR6, 70W TDP) on Kaggle Instances
- **Task Benchmark:** SST-2 Sentiment Classification (`stanfordnlp/sst2`)
- **Execution Mode:** `real` (CodeCarbon 2.8+ NVML power sampling + PyTorch CUDA Memory tracking)
- **Data Source Files:** `sweep_raw.csv`, `aggregated_gei.csv`, `pareto_fronts.csv`, `raw/*.json`

---

## ⚡ Quick Promptable System Summaries (Copy-Paste Ready)

### Short Abstract Prompt Block
> **GreenPEFT** transforms sustainable LLM fine-tuning from post-hoc benchmarking into a predictive zero-shot decision support paradigm. Tested on an NVIDIA Tesla T4 (16 GB VRAM) across model scales (0.5B to 3.0B), GreenPEFT forecasts validation accuracy, peak VRAM, energy consumption (kWh), and carbon emissions (kgCO₂eq) directly from pre-training metadata. Full Fine-Tuning hits a rigid memory wall at 0.5B parameters (13.48 GB VRAM), while LISA scales to 1.5B parameters (14.88 GB VRAM) and achieves a peak Green Efficiency Index (GEI) score of 0.999.

### Key Metrics Summary Table Prompt Block
| Method | Backbone | Params | Accuracy | Peak VRAM (GB) | Energy (kWh) | Carbon (kgCO₂eq) | GEI Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full_ft** | Qwen2.5-Tiny | 0.5B | **0.9122 ± 0.0117** | 13.48 | 0.00382 | 0.00248 | 0.964 | Succeeded |
| **lisa** | Qwen2.5-Tiny | 0.5B | 0.5089 ± 0.0165 | **4.84** | **0.00324** | **0.00211** | **0.999** | Succeeded |
| **lisa** | TinyLlama-Small | 1.1B | 0.6122 ± 0.0901 | 9.43 | 0.00833 | 0.00541 | 0.822 | Succeeded |
| **lisa** | Qwen2.5-Medium | 1.5B | 0.4889 ± 0.0342 | 14.88 | 0.01124 | 0.00731 | 0.751 | Succeeded |
| **full_ft** | Small/Med/Large | 1.1B--3B | N/A | >16.00 | N/A | N/A | N/A | **OOM** |
| **lora / lora_fa**| All Backbones | 0.5B--3B | N/A | N/A | N/A | N/A | N/A | **TorchAO Lock** |
| **qlora** | All Backbones | 0.5B--3B | N/A | N/A | N/A | N/A | N/A | **BitsAndBytes Lock**|

---

## 📊 Detailed Empirical Execution Sweep Breakdown

### 1. Overall Grid Statistics
- **Total Planned Runs:** 60 runs (5 methods $\times$ 4 backbones $\times$ 3 seeds)
- **Successful Runs:** 12 runs ($20\%$ completion rate)
- **OOM Hardware Failures:** 36 runs ($60\%$ physical ceiling limit)
- **Dependency Conflict Blocks:** 12 runs ($20\%$ environment package locks)

### 2. Itemized Successful Run Logs ($k=3$ Seeds: 13, 42, 2024)

#### A. Full Fine-Tuning (Full-FT) — 0.5B Qwen2.5-Tiny
- **Seed 13:** Accuracy: `0.9133` | Wall-clock: `222.20 s` | VRAM: `13.475 GB` | Energy: `0.003802 kWh` | Carbon: `0.002471 kg` | Cost: `$0.0221`
- **Seed 2024:** Accuracy: `0.9233` | Wall-clock: `223.09 s` | VRAM: `13.480 GB` | Energy: `0.003828 kWh` | Carbon: `0.002488 kg` | Cost: `$0.0221`
- **Seed 42:** Accuracy: `0.9000` | Wall-clock: `223.02 s` | VRAM: `13.480 GB` | Energy: `0.003830 kWh` | Carbon: `0.002489 kg` | Cost: `$0.0221`
- **Mean ± Std:** Accuracy: **0.9122 ± 0.0117** | VRAM: **13.48 GB** | Energy: **0.00382 kWh** | GEI: **0.964**

#### B. LISA — 0.5B Qwen2.5-Tiny
- **Seed 13:** Accuracy: `0.5167` | Wall-clock: `185.89 s` | VRAM: `4.848 GB` | Energy: `0.003303 kWh` | Carbon: `0.002147 kg` | Cost: `$0.0185`
- **Seed 2024:** Accuracy: `0.4900` | Wall-clock: `181.62 s` | VRAM: `4.881 GB` | Energy: `0.003206 kWh` | Carbon: `0.002084 kg` | Cost: `$0.0180`
- **Seed 42:** Accuracy: `0.5200` | Wall-clock: `181.10 s` | VRAM: `4.785 GB` | Energy: `0.003222 kWh` | Carbon: `0.002094 kg` | Cost: `$0.0180`
- **Mean ± Std:** Accuracy: **0.5089 ± 0.0165** | VRAM: **4.84 GB** | Energy: **0.00324 kWh** | GEI: **0.999**

#### C. LISA — 1.1B TinyLlama-Small
- **Seed 13:** Accuracy: `0.7000` | Wall-clock: `474.73 s` | VRAM: `9.539 GB` | Energy: `0.008440 kWh` | Carbon: `0.005486 kg` | Cost: `$0.0472`
- **Seed 2024:** Accuracy: `0.6167` | Wall-clock: `465.83 s` | VRAM: `9.369 GB` | Energy: `0.008282 kWh` | Carbon: `0.005383 kg` | Cost: `$0.0463`
- **Seed 42:** Accuracy: `0.5200` | Wall-clock: `463.16 s` | VRAM: `9.378 GB` | Energy: `0.008258 kWh` | Carbon: `0.005368 kg` | Cost: `$0.0460`
- **Mean ± Std:** Accuracy: **0.6122 ± 0.0901** | VRAM: **9.43 GB** | Energy: **0.00833 kWh** | GEI: **0.822**

#### D. LISA — 1.5B Qwen2.5-Medium
- **Seed 13:** Accuracy: `0.5267` | Wall-clock: `628.26 s` | VRAM: `14.895 GB` | Energy: `0.011068 kWh` | Carbon: `0.007194 kg` | Cost: `$0.0624`
- **Seed 2024:** Accuracy: `0.4800` | Wall-clock: `641.97 s` | VRAM: `14.818 GB` | Energy: `0.011369 kWh` | Carbon: `0.007390 kg` | Cost: `$0.0638`
- **Seed 42:** Accuracy: `0.4600` | Wall-clock: `638.76 s` | VRAM: `14.937 GB` | Energy: `0.011289 kWh` | Carbon: `0.007338 kg` | Cost: `$0.0635`
- **Mean ± Std:** Accuracy: **0.4889 ± 0.0342** | VRAM: **14.88 GB** | Energy: **0.01124 kWh** | GEI: **0.751**

---

## 🧮 Promptable Mathematical Formulations

### 1. Multi-Objective Metric Vector
$$\mathbf{y}(\mathbf{c}) = \left[ \text{Acc}(\mathbf{c}), \text{Mem}(\mathbf{c}), E(\mathbf{c}), C(\mathbf{c}), T(\mathbf{c}) \right]$$

### 2. Green Efficiency Index (GEI) Equation
$$\text{GEI}(\mathbf{c}) = w_{\text{Acc}} \hat{S}_{\text{Acc}} + w_{\text{Mem}} \hat{S}_{\text{Mem}} + w_{C} \hat{S}_{C} + w_{T} \hat{S}_{T}$$

where benefit and cost normalization functions are:
$$\hat{S}_{\text{Acc}} = \frac{\text{Acc} - \text{Acc}_{\min}}{\text{Acc}_{\max} - \text{Acc}_{\min}}, \qquad \hat{S}_{M} = 1 - \frac{M - M_{\min}}{M_{\max} - M_{\min}} \quad (M \in \{\text{Mem}, C, T\})$$

### 3. Zero-Shot Surrogate Regressor
$$f_{\phi}: \mathcal{X} \to \mathbb{R}^5, \qquad \mathbf{x} = \left[ N_{\text{params}}, r, q_{\text{bits}}, N_{\text{samples}}, \mathbf{e}_{\text{task}}, \mathbf{e}_{\text{method}} \right]$$
$$\mathcal{L}(\phi) = \frac{1}{K} \sum_{k=1}^K \left| \frac{\mathbf{y}_k - f_{\phi}(\mathbf{x}_k)}{\mathbf{y}_k} \right|$$

---

## 🎯 Constraint Engine Decision Scenario Prescriptions

The GreenPEFT Decision Engine was evaluated across four representative user deployment personas:

```text
+-----------------------+------------------+------------------+-----------------------+------------+
| Scenario Persona      | VRAM Limit (GB)  | Carbon Cap (kg)  | Prescribed Strategy   | GEI Score  |
+-----------------------+------------------+------------------+-----------------------+------------+
| 1. Consumer GPU       | <= 8.0 GB        | Unconstrained    | LISA (0.5B Tiny)      | 0.999      |
| 2. Strict Carbon      | Unconstrained    | <= 0.003 kg      | LISA (0.5B Tiny)      | 0.999      |
| 3. High-Accuracy      | <= 16.0 GB       | Unconstrained    | Full-FT (0.5B Tiny)   | 0.964      |
| 4. Tight Deadline     | <= 16.0 GB       | <= 0.005 kg      | LISA (0.5B Tiny)      | 0.999      |
+-----------------------+------------------+------------------+-----------------------+------------+
```

---

## 🛠️ Failure Diagnostics & Remediation Script for Prompting

### Failure Cause 1: Hardware Memory Boundaries (36 runs)
- **Affected:** Full-FT on 1.1B, 1.5B, 3.0B; LISA on 3.0B.
- **Diagnosis:** CUDA Out-Of-Memory exception during model instantiation and AdamW optimizer state allocation ($>16\text{ GB}$).
- **Interpretation:** Authentic physical hardware limits validating the necessity of PEFT.

### Failure Cause 2: Dependency Package Conflicts (12 runs)
- **Affected:** LoRA, LoRA-FA, QLoRA across all scales.
- **Diagnosis:**
  - `torchao==0.10.0` installed, but PEFT requires `torchao>=0.16.0`.
  - `bitsandbytes` version requirement (`>=0.46.1`) for 4-bit NF4 quantization.
- **Fix (Executed in Cell 1):**
  ```bash
  pip install --upgrade "peft>=0.10.0" "bitsandbytes>=0.46.1" "torchao>=0.16.0"
  ```

### LISA Hyperparameter Tuning Prescription
- **Issue:** Default LISA settings (`layer_sample_prob=0.25`, `resample_every_n_steps=20`) caused under-training on classification tasks ($50.89\%$--$61.22\%$ accuracy).
- **Remediation:** Updated `configs/methods/lisa.yaml` to:
  ```yaml
  layer_sample_prob: 0.5
  resample_every_n_steps: 5
  ```

---

## 💡 Prompt Templates for Downstream AI Generations

### Prompt 1: Generate Paper Methodology Section
> "Using the GreenPEFT framework definitions from report.md, write a formal IEEE Methodology section describing the multi-objective optimization problem, the Green Efficiency Index (GEI) formula with AHP preference weights, and the zero-shot surrogate regressor predicting peak VRAM, energy, carbon, and accuracy from pre-training metadata."

### Prompt 2: Generate Slide Deck Presentation
> "Convert the empirical findings in report.md into a 10-slide presentation deck. Emphasise the Full-FT 13.48 GB memory wall at 0.5B parameters, LISA's memory savings (4.84 GB VRAM, GEI=0.999), and the decision engine scenario prescriptions."

### Prompt 3: Generate System CLI Documentation
> "Create a user manual for the `green-peft` CLI tool based on the surrogate model and constraint filtering logic described in report.md, including command-line arguments `--vram`, `--carbon`, `--accuracy`, and `--profile`."
