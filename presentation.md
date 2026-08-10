# Sustainable PEFT Benchmark: Methodology & Implementation Guide

## 1. Project Overview
**A Real‑Measurement Pipeline for Energy‑Aware Parameter‑Efficient Fine‑Tuning (PEFT)**  
We systematically benchmark 5 PEFT methods across 4 model sizes (0.5B – 3B) on real hardware (NVIDIA T4 GPU) – measuring not just accuracy, but **time, VRAM, energy (kWh), carbon footprint, and dollar cost** – to identify the greenest, most efficient fine‑tuning strategies for real‑world deployment.

---

## 2. Proposed Methodology

| Component | Details |
|---|---|
| **Backbone Tiers** | `Qwen2.5-0.5B`, `TinyLlama-1.1B`, `Qwen2.5-1.5B`, `Qwen2.5-3B` (all Apache‑2.0, no gatekeeping) |
| **PEFT Methods** | `full_ft` (baseline), `lora` (LoRA), `qlora` (4‑bit QLoRA), `lora_fa` (LoRA with Frozen A), `lisa` (Layer‑wise Importance Sampled AdamW) |
| **Tasks** | `SST-2` (classification), `CNN/DailyMail` (summarization), `GSM8K` (instruction following) |
| **Hardware** | Kaggle GPU T4 x2 (16 GB VRAM) – Power metering via NVML |
| **Metrics** | Wall‑clock time, Peak VRAM (PyTorch + NVML combined), Energy (kWh, integrated from real‑time wattage), Carbon (kgCO₂eq via grid intensity), Cost (electricity + GPU rental proxy) |
| **Analytics** | Weighted scoring profiles, Green Efficiency Index (GEI), True Pareto fronts, Decision engine (Algorithm 1) |

---

## 3. Implementation Pipeline (Notebook Workflow)

1. **Environment Setup** → Install libraries, detect GPU, create folders.
2. **Configuration** → Define backbones, tasks, methods, training hyperparameters.
3. **Real Data Loading** → Download SST-2 / CNN‑DM / GSM8K from HuggingFace.
4. **Measurement Layer** → Background thread samples GPU power (watts) at 2 Hz.
5. **Model Builders** → Instantiate each PEFT method with consistent seeds.
6. **Training Loop** → Each run is measured end‑to‑end; OOMs are recorded as valid "infeasible" results.
7. **Evaluation** → Accuracy (SST-2), ROUGE‑L (CNN‑DM), Exact‑Match (GSM8K).
8. **Aggregation & Scoring** → Min‑max normalisation; 3 practitioner profiles (Balanced, Accuracy‑First, Carbon‑Constrained).
9. **GEI & Pareto** → Compute GEI = `P^α / ((E/E_ref)^β * (M/M_ref)^γ * (T/T_ref)^δ)`; extract non‑dominated configurations.
10. **Decision Engine** → Recommend optimal method given user constraints (VRAM, carbon budget, time, min accuracy).
11. **Export** → Produce CSVs, plots, and a downloadable zip bundle.

---

## 4. Usable Instructions (How to Run)

### Platform
- **Use Kaggle** (Notebook → Settings → Accelerator = **GPU T4 x2**, Internet = **ON**).  
  (Colab Free works for smoke tests only – sessions die and disk is wiped.)

### Step‑by‑Step

1. **Upload** the notebook (`green‑project.ipynb`) to a new Kaggle notebook.
2. **Set Accelerator** to T4 x2. Do **not** use P100 (QLoRA fails on compute capability < 7.0).
3. **First run (Smoke Test)** – In Cell 2, keep `RUN_MODE = 'smoke'`. Run all cells **top‑to‑bottom** (Cells 1 → 16). This takes ~10 minutes and confirms the pipeline works.
4. **If errors occur** – check the troubleshooting table below.
5. **Real Sweep** – After smoke passes, in Cell 2 set `RUN_MODE = 'real'`, and in Cell 1 set `INSTALL = False`. Restart the session and run all cells. The sweep takes **4–10 hours**.
6. **Resume after session death** – Just re‑run Cell 10; it skips already‑completed runs.
7. **Download results** – After Cell 16, download the zip from the Kaggle Output panel.

### Key Hyperparameters (Cell 2)
| Parameter | Smoke | Real (Suggested) |
|---|---|---|
| `TRAIN_STEPS` | 20 | 100–500 |
| `BATCH_SIZE` | 4 | 4 |
| `GRAD_ACCUM` | 2 | 2 |
| `TRAIN_EXAMPLES` | 200 | 1000 |
| `SEEDS` | `[42]` | `[13, 42, 2024]` |

### Troubleshooting Quick‑Fix
| Symptom | Fix |
|---|---|
| `ImportError: torchao` | In Cell 1, add: `subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'torchao>=0.16.0'], check=False)` |
| `CUDA OOM` on all runs | Lower `BATCH_SIZE` to 2 and `MAX_SEQ_LEN` to 128 in Cell 2. |
| Dataset download fails | Notebook Settings → Internet = **ON**. |
| QLoRA skipped | Your GPU is P100 – switch to **T4 x2** in Accelerator settings. |

---

## 5. Presentation Prompts (For Tomorrow's PPT)

Copy these bullet points directly into your slides.

---

### 🎯 Slide 1: Onboarding (Title Slide)
- **Title:** Sustainable Parameter‑Efficient Fine‑Tuning: A Real‑Hardware Benchmark  
- **Subtitle:** Measuring Energy, Carbon, and Cost Across PEFT Methods  
- **Your Name / Team / Date**

---

### 📌 Slide 2: Agenda
- Problem Statement & Motivation  
- Literature Review & Gap Analysis  
- Proposed Methodology  
- System Pipeline  
- Current Progress (What We Have Done)  
- Next Steps (What We Are Going to Do)  
- Expected Outcomes

---

### ❓ Slide 3: Why Are We Doing This? (Problem Statement)
- **The Scale Problem:** Fine‑tuning large language models (LLMs) is computationally expensive and environmentally costly.  
- **The Blind Spot:** Most research focuses solely on accuracy and parameter count – ignoring **energy, carbon, and cost**.  
- **The Need:** Practitioners lack empirical, hardware‑validated data to choose the greenest PEFT method for their constraints.  
- **Our Goal:** Provide a reproducible, real‑measurement benchmark that reveals the true sustainability trade‑offs of 5 popular PEFT methods.

---

### 📚 Slide 4: Literature Review & Gap Analysis
- **Existing Work:**  
  - PEFT papers report FLOPs or parameter counts (LoRA, QLoRA, LISA).  
  - Some works estimate energy theoretically via FLOPs‑to‑watts approximations.  
  - Carbon tracking (e.g., ML CO₂ Impact) uses coarse global averages.  
- **The Gaps:**  
  1. No direct, real‑time **hardware power sampling** for PEFT methods.  
  2. No unified benchmark across multiple **tiers** (0.5B → 3B).  
  3. No **decision engine** that maps practitioner constraints (VRAM, budget, time) to a recommended method.  
  4. Most studies ignore **infrastructure cost** (GPU rental).  
- **Our Contribution:** A full‑stack measurement pipeline + Green Efficiency Index (GEI) + Pareto‑based decision support.

---

### ⚙️ Slide 5: Proposed Methodology (Overview)
- **Scope:** 5 PEFT methods × 4 model sizes × 3 tasks × 3 seeds = 180 runs (scalable).  
- **Hardware:** Real NVIDIA T4 GPU with NVML power metering (2 samples/sec).  
- **Metrics:** Accuracy, Time, VRAM, Energy (kWh), Carbon (kgCO₂eq), Cost (USD).  
- **Analytics:**  
  - Weighted Scoring (3 personas)  
  - GEI (Green Efficiency Index)  
  - Multi‑objective Pareto Fronts  
  - Decision Engine (Algorithm 1)

---

### 🧪 Slide 6: PEFT Methods Under Test
| Method | Description | Trainable Params (relative) |
|---|---|---|
| **Full‑FT** | Full weight updates (baseline) | 100% |
| **LoRA** | Low‑Rank Adaptation (rank=16) | ~0.5‑1.5% |
| **QLoRA** | 4‑bit quantized LoRA | ~0.5‑1.5% |
| **LoRA‑FA** | LoRA with frozen down‑projection (A) | ~0.3‑0.8% |
| **LISA** | Layer‑wise Importance Sampled AdamW | ~5‑10% (dynamic) |

**Why these?** They represent the most widely adopted PEFT techniques in industry and academia.

---

### 📏 Slide 7: Measurement Infrastructure
- **Energy & Power:** Background thread calls `pynvml.nvmlDeviceGetPowerUsage()` at 2 Hz. Energy (J) = ∫ Watts dt → converted to kWh.  
- **VRAM:** Max of PyTorch’s `max_memory_allocated()` and NVML’s `used` memory (catches bitsandbytes allocations).  
- **Carbon:** `kWh × grid intensity` (default: 0.650 kgCO₂/kWh, configurable to renewable 0.150).  
- **Cost:** `(kWh × $0.12) + (hours × $0.35)` – electricity + spot GPU rental.

---

### 🧮 Slide 8: Evaluation Framework (GEI + Pareto + Decision)
- **GEI Formula (Paper’s core metric):**  
  `GEI(m) = P(m)^α / [ (E/E_ref)^β * (M/M_ref)^γ * (T/T_ref)^δ ]`  
  - `α=0.40` (accuracy weight), `β=0.25`, `γ=0.20`, `δ=0.15`  
  - Higher GEI = greener per unit of accuracy.
- **Pareto Front:** Identifies non‑dominated configurations (no other run is better on all objectives simultaneously).  
- **Decision Engine (Algorithm 1):**  
  Takes user constraints (VRAM ≤ 16GB, carbon ≤ 0.05kg, time ≤ 600s) → filters feasible set → returns highest‑GEI recommendation.

---

### 🔄 Slide 9: Implementation Pipeline (The Notebook)
1. **Cell 1‑3:** Environment, configs, write YAML files.  
2. **Cell 4:** Download real datasets (HuggingFace).  
3. **Cell 5‑7:** Build measurement layer, model builders, task pipelines.  
4. **Cell 8‑10:** Run the sweep – each run saves a JSON. Resumable.  
5. **Cell 11‑13:** Aggregate → Weighted Scores → GEI → Pareto.  
6. **Cell 14:** Decision Engine – scenario‑based recommendations.  
7. **Cell 15‑16:** Plots + Export zip bundle.

> **Key Design:** OOMs are recorded as `status: "oom"` – they are findings (infeasibility on 16GB) not errors.

---

### ✅ Slide 10: What We Have Done
- [x] Designed and implemented the full measurement pipeline (NVML power sampling, VRAM tracking).  
- [x] Integrated 5 PEFT methods (Full‑FT, LoRA, QLoRA, LoRA‑FA, LISA) into a unified training harness.  
- [x] Set up real dataset loading (SST‑2, CNN/DM, GSM8K) with caching.  
- [x] Built the analytics stack: Weighted scoring, GEI, Pareto front extraction, Decision Engine.  
- [x] Successfully ran **smoke tests** – pipeline is stable and error‑free.  
- [x] Created visualisation suite and export bundle for results.

---

### 🚀 Slide 11: What We Are Going to Do
- [ ] Execute the **full real sweep** (across all seeds, tiers, and tasks) – expected 4‑10 hours.  
- [ ] Collect and analyse the final aggregated data.  
- [ ] Generate publication‑ready tables and figures.  
- [ ] Validate the Decision Engine against real‑world scenarios (carbon‑constrained, high‑accuracy, tight deadline).  
- [ ] Prepare the final dataset and code release for reproducibility.  
- [ ] Write the methodology section and integrate results into the paper.

---

### 📊 Slide 12: Expected Outcomes (For Discussion)
- A clear ranking of PEFT methods by **GEI** – which one is truly the greenest?  
- Quantitative evidence of **energy‑accuracy trade‑offs** – is LoRA always better than Full‑FT?  
- A **decision matrix** that tells a practitioner: *“If you have a 16GB GPU and a 1‑hour budget, use X.”*  
- An open‑source, replicable benchmark that the community can extend.

---

## 6. Appendix: Quick Commands for PPT Slides
- Use the **bold** phrases as slide titles.  
- Keep bullet points concise (max 5 per slide).  
- Include a screenshot of the pipeline diagram (from the notebook’s markdown) to visualise the flow.  
- For the methodology slide, show the formula for GEI clearly.

---

Good luck with your presentation tomorrow! You have a solid, working pipeline and a compelling story. 🍀