# GreenPEFT Data Preprocessing & Feature Engineering Guide

## Executive Overview
This document provides a comprehensive step-by-step breakdown of the data cleaning, OOM filtering, domain feature engineering, group median imputation, and categorical encoding pipeline applied to the aggregated LLM surrogate dataset.

---

## 1. Pipeline Overview & Objectives

| Step | Action Name | Primary Objective | Output Artifact / Target |
| :--- | :--- | :--- | :--- |
| **Step 1** | Data Loading & Verification | Load raw aggregated dataset (`957` rows) | Initial Dataframe `df_raw` |
| **Step 2** | OOM Failure Row Filtering | Isolate Out-Of-Memory failure runs (`65` rows) | [`oom_failed_runs.csv`](file:///c:/Users/Tanzil/Downloads/green/oom_failed_runs.csv) |
| **Step 3** | Domain Feature Engineering | Compute 5 LLM scaling features | Feature columns (`log_params_b`, etc.) |
| **Step 4** | Missing Value Imputation | Impute missing accuracy via Group Medians | **0 Missing Values** across `892` rows |
| **Step 5** | Categorical Encoding & Matrix Construction | One-Hot Encode `method`, `backbone`, `family` | ML Feature Matrix (`324` columns) |
| **Step 6** | Validation & Artifact Export | Export clean, production-ready datasets | [`clean_usable_surrogate_dataset.csv`](file:///c:/Users/Tanzil/Downloads/green/clean_usable_surrogate_dataset.csv) & [`ml_ready_surrogate_features.csv`](file:///c:/Users/Tanzil/Downloads/green/ml_ready_surrogate_features.csv) |

---

## 2. Engineered Domain Features

We engineer 5 domain-informed features capturing LLM parameter scaling laws and hardware VRAM headroom:

1. **`log_params_b`**:
   $$\text{log\_params\_b} = \log_2(\text{params\_b})$$
   *Linearizes non-linear parameter scaling relationships.*

2. **`trainable_params_b`**:
   $$\text{trainable\_params\_b} = \text{params\_b} \times \text{trainable\_param\_pct}$$
   *Calculates exact active trainable weights in billions.*

3. **`effective_vram_ratio`**:
   $$\text{effective\_vram\_ratio} = \frac{\text{gpu\_total\_gb}}{\text{params\_b} + 10^{-5}}$$
   *Quantifies available VRAM headroom per parameter.*

4. **`vram_per_param_mb`**:
   $$\text{vram\_per\_param\_mb} = \frac{\text{gpu\_total\_gb} \times 1024}{\text{params\_b} + 10^{-5}}$$
   *Converts headroom into megabytes per parameter.*

5. **`is_quantized`**:
   $$\text{is\_quantized} = \mathbb{I}(\text{quant\_bits} < 16)$$
   *Binary flag indicating sub-16bit quantization precision.*

---

## 3. Usable Dataset Schema (`clean_usable_surrogate_dataset.csv`)

- `run_id` (str): Unique configuration run identifier
- `method` (str): Fine-tuning strategy (`full_ft`, `lisa`, `lora`, `lora_fa`, `qlora`)
- `backbone` (str): Model size category (`tiny`, `small`, `medium`, `large`, `xlarge`, `huge`)
- `family` (str): Architecture family (`llama`, `qwen`, `mistral`, `gemma`, `gpt`, `deepseek`, etc.)
- `params_b` (float): Parameter count in billions
- `rank` (int): Adapter rank (0 for full_ft/lisa, 16 for PEFT)
- `quant_bits` (int): Bit precision (4, 8, 16, 32)
- `is_adapter_method` (int): Binary indicator (1 for PEFT, 0 for full_ft)
- `gpu_total_gb` (float): Available GPU VRAM in GB
- `status` (str): Run status (`ok`)
- `fits` (int): 1 (Fitted successfully in VRAM)
- `trainable_param_pct` (float): Ratio of trainable parameters (0.0 to 1.0)
- `accuracy` (float): Model metric score (Imputed via Group Medians)
- `peak_gpu_memory_gb` (float): Measured peak VRAM allocation in GB
- `energy_kwh` (float): Energy consumed in kWh
- `carbon_kgco2eq` (float): Carbon footprint in $\text{kgCO}_2\text{eq}$
- `wall_clock_seconds` (float): Execution duration in seconds
- `log_params_b` (float): Engineered $\log_2(\text{params\_b})$
- `trainable_params_b` (float): Active trainable parameters in billions
- `effective_vram_ratio` (float): VRAM headroom ratio
- `vram_per_param_mb` (float): Headroom in MB per parameter
- `is_quantized` (int): Binary quantization flag

---

## 4. Final Missing Value Audit

| Column Name | Raw Missing Count | Processed Missing Count | Status |
| :--- | :--- | :--- | :--- |
| `run_id`, `method`, `backbone`, `family` | 0 | **0** | Clean |
| `params_b`, `rank`, `quant_bits`, `is_adapter_method` | 0 | **0** | Clean |
| `gpu_total_gb`, `status`, `fits`, `trainable_param_pct` | 0 | **0** | Clean |
| `peak_gpu_memory_gb` | 65 (OOM) | **0** (Filtered OOMs) | Clean |
| `energy_kwh` | 65 (OOM) | **0** (Filtered OOMs) | Clean |
| `carbon_kgco2eq` | 65 (OOM) | **0** (Filtered OOMs) | Clean |
| `wall_clock_seconds` | 65 (OOM) | **0** (Filtered OOMs) | Clean |
| `accuracy` | 392 | **0** (Group Median Imputed) | **100% Complete** |
