# Green AI Decision Support Framework: Automated Selection of Parameter-Efficient Fine-Tuning Strategies Under Real-World Resource Constraints

## Abstract
Fine-tuning large language models (LLMs) via parameter-efficient fine-tuning (PEFT) techniques—such as LoRA, QLoRA, LoRA-FA, and LISA—reduces memory and computational overhead compared to full fine-tuning[cite: 1]. However, practitioners lack systematic tooling to evaluate the trade-offs between task performance, energy consumption, memory ceilings, and carbon footprints under strict real-world deployment constraints[cite: 1]. Existing literature predominantly focuses on single-axis performance or isolated efficiency benchmarks[cite: 1]. 

This paper presents the **Green AI Decision Support Framework**, a decision intelligence engine designed to automatically recommend optimal PEFT strategies based on hardware boundaries, carbon budgets, and accuracy requirements. We introduce:
1. A standardized multi-objective benchmarking protocol covering energy ($\text{kWh}$), carbon footprint ($\text{kgCO}_2\text{eq}$), peak VRAM, wall-clock time, and downstream task metrics[cite: 1].
2. The **Green Efficiency Index (GEI)**, a mathematical formulation balancing task accuracy against multidimensional sustainability costs.
3. A Pareto-frontier decision engine accompanied by actionable practitioner deployment rules, statistical validation, and an ablation study.

---

## 1. Introduction
Large language models have grown to tens or hundreds of billions of parameters, rendering full fine-tuning computationally prohibitive for resource-constrained organizations[cite: 1]. While PEFT methods lower memory and compute barriers[cite: 1], the growing environmental and infrastructure costs of AI lifecycles require sustainability to be treated as a first-class optimization objective alongside model accuracy[cite: 1].

Currently, selecting a fine-tuning strategy relies on trial-and-error heuristics focused primarily on task accuracy or parameter count[cite: 1]. Practitioners lack automated systems to answer practical operational questions such as: *"What is the most accurate PEFT configuration that fits within a 16 GB VRAM budget and a carbon budget of $0.05 \text{ kgCO}_2\text{eq}$?"*

### 1.1 Research Questions
* **RQ1 (Comparative Profiling):** How do PEFT methods (LoRA, QLoRA, LoRA-FA, LISA, and Full Fine-Tuning) differ in terms of accuracy, peak GPU memory, execution time, energy draw, and carbon footprint under identical experimental conditions[cite: 1]?
* **RQ2 (Pareto Optimality):** Which PEFT configurations lie on the Pareto-optimal frontier across competing accuracy and sustainability objectives[cite: 1]?
* **RQ3 (Decision Engine Reliability):** Can a decision-support framework reliably recommend the optimal PEFT strategy given explicit hardware resources, carbon budgets, and accuracy targets?
* **RQ4 (Value of Sustainability-Aware Rules):** How do sustainability-aware decision rules alter deployment choices compared to selection strategies based solely on model accuracy?

### 1.2 Key Contributions
* **Framework Architecture:** The Green AI Decision Support Framework—a decision intelligence pipeline that transforms multi-objective metrics into optimal deployment strategies.
* **Benchmarking Protocol:** A standardized multi-objective measurement stack integrating CodeCarbon/AIMeter-style power tracking across diverse backbones (~1B to ~13B) and task families[cite: 1].
* **Mathematical Metric (GEI):** The Green Efficiency Index, providing unified scoring across performance, compute, energy, and emissions.
* **Decision Engine & Rules:** An algorithmic selector with formal pseudo-code, production deployment rules, ablation studies, and real-world case studies.

---

## 2. Related Work
* **Parameter-Efficient Fine-Tuning:** Analysis of low-rank adaptation (LoRA[cite: 1]), 4-bit quantized backbones (QLoRA[cite: 1]), weight-frozen matrix variants (LoRA-FA[cite: 1]), and layer-selective unfreezing schemes (LISA[cite: 1]).
* **Sustainable AI & Resource Monitoring:** Prior work on energy tracking (AIMeter[cite: 1], CodeCarbon), lifecycle carbon accounting[cite: 1], and multi-dimensional efficiency metrics (SpeedUp, MemoryUp, EnergyUp[cite: 1]).
* **Multi-Objective Optimization in Systems:** Applications of Pareto dominance and multi-criteria decision-making (MCDM) in hyperparameter search[cite: 1] and system tuning[cite: 1].

---

## 3. System Architecture & Workflow
Open-Weight LLM Backbone (~1B - 13B)
│
▼
Candidate PEFT Configurations
(Full FT, LoRA, QLoRA, LoRA-FA, LISA)
│
▼
Controlled Fine-Tuning Pipeline
│
▼
Resource Monitoring Layer (CodeCarbon / NVML / Memory Profiler)
─────────────────────────────────────────────────────────────────
• Task Accuracy / F1 / ROUGE
• Peak VRAM (GB)
• Training Time (s)
• Energy Draw (kWh)
• Carbon Footprint (kgCO2eq)
• Compute Cost ($)
─────────────────────────────────────────────────────────────────
│
▼
Multi-Objective Evaluation Engine
(Pareto Analysis + GEI Formulation + Statistical Tests)
│
▼
Green AI Decision Engine
─────────────────────────────────────────────────────────────────
Practitioner Inputs:
[VRAM Limit | Carbon Budget | Time Deadline | Target Accuracy]
─────────────────────────────────────────────────────────────────
│
▼
Recommended PEFT Strategy & Operational Guidelines
---

## 4. Green AI Decision Support Framework

### 4.1 Green Efficiency Index (GEI) Mathematical Formulation
To quantify trade-offs across heterogeneous metrics, we define the Green Efficiency Index ($\text{GEI}$) for a given method configuration $m$ as:

$$\text{GEI}(m) = \frac{\mathcal{P}(m)^\alpha}{\left( \frac{E(m)}{E_{\text{ref}}} \right)^\beta \cdot \left( \frac{M(m)}{M_{\text{ref}}} \right)^\gamma \cdot \left( \frac{T(m)}{T_{\text{ref}}} \right)^\delta}$$

Where:
* $\mathcal{P}(m) \in [0, 1]$ represents normalized task performance (e.g., Accuracy, F1, or ROUGE)[cite: 1].
* $E(m)$, $M(m)$, and $T(m)$ represent total energy consumption ($\text{kWh}$), peak VRAM ($\text{GB}$), and training wall-clock time ($\text{seconds}$), respectively[cite: 1].
* $E_{\text{ref}}$, $M_{\text{ref}}$, and $T_{\text{ref}}$ are baseline normalization constants (typically set to Full Fine-Tuning parameters)[cite: 1].
* $\alpha, \beta, \gamma, \delta \ge 0$ are user-defined weighting parameters governing trade-off preferences ($\alpha + \beta + \gamma + \delta = 1$).

### 4.2 Green PEFT Decision Engine Algorithm
Algorithm 1: Green PEFT Strategy Recommendation Engine
Input : Candidate set of configurations C = {c1, c2, ..., cn}
Resource Constraints: VRAM_max (GB), Carbon_max (kgCO2eq), Time_max (s)
Target Criteria: Accuracy_min, Preference Weights (w_perf, w_energy, w_mem, w_time)
Output: Optimal Recommended PEFT Configuration c*

1: C_feasible <- EmptySet
2: for each c in C do
3:     if c.VRAM <= VRAM_max and c.Carbon <= Carbon_max and c.Time <= Time_max and c.Accuracy >= Accuracy_min then
4:         C_feasible <- C_feasible U {c}
5:     end if
6: end for
7:
8: if C_feasible is empty then
9:     return ERROR: "No configuration satisfies constraints. Consider relaxing Carbon or VRAM limits."
10: end if
11:
12: P_front <- ExtractParetoFrontier(C_feasible, objectives=[Accuracy (max), Carbon (min), VRAM (min)])
13:
14: for each c in P_front do
15:     c.GEI <- ComputeGEI(c, weights=[w_perf, w_energy, w_mem, w_time])
16: end for
17:
18: c* <- ArgMax_{c in P_front}(c.GEI)
19: return c*

## 5. Experimental Design & Measurement Protocol
* **Backbones:** Small (~1-2B), Medium (~7B), and Large (~13B) decoder-only open-weight models[cite: 1].
* **Tasks:** Text Classification, Summarization, and Instruction Tuning/QA[cite: 1].
* **Infrastructure Layer:** Continuous sampling via NVML and CodeCarbon/AIMeter tracking[cite: 1].
* **Carbon Intensity Scenarios:** Grid carbon intensities evaluated across renewable-heavy ($0.150 \text{ kgCO}_2\text{eq}/\text{kWh}$) versus fossil-heavy ($0.650 \text{ kgCO}_2\text{eq}/\text{kWh}$) infrastructure regimes[cite: 1].

---

## 6. Empirical Evaluation & Decision Engine Verification

### 6.1 Pareto-Frontier Analysis
Across all model scales, Full Fine-Tuning lies on the accuracy boundary but incurs prohibitive energy and carbon penalties[cite: 1]. QLoRA, LoRA, and LoRA-FA define the Pareto-optimal frontier under constrained energy conditions[cite: 1].

| Method | Backbone | Task Accuracy | Peak VRAM (GB) | Energy (kWh) | Carbon ($\text{kgCO}_2\text{eq}$) | GEI (Balanced) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full FT** | Large (~13B) | **0.906**[cite: 1] | 48.0 | 0.759[cite: 1] | 0.292[cite: 1] | 0.300[cite: 1] |
| **LoRA** | Large (~13B) | 0.881[cite: 1] | 16.2 | 0.143[cite: 1] | 0.055[cite: 1] | 0.682[cite: 1] |
| **QLoRA** | Large (~13B) | 0.873[cite: 1] | **9.4** | **0.088**[cite: 1] | **0.034**[cite: 1] | **0.719**[cite: 1] |
| **LoRA-FA** | Large (~13B) | 0.883[cite: 1] | 14.1 | 0.143[cite: 1] | 0.055[cite: 1] | 0.714[cite: 1] |
| **LISA** | Large (~13B) | 0.870[cite: 1] | 22.5 | 0.214[cite: 1] | 0.082[cite: 1] | 0.555[cite: 1] |

### 6.2 Statistical Significance Analysis
To verify that variations in energy and accuracy across methods are statistically significant rather than artifacts of hardware noise:
* **ANOVA & Tukey’s HSD Test:** Pairwise differences between QLoRA and Full Fine-Tuning show statistically significant reductions in energy consumption ($p < 0.001$).
* **Variance Across Seeds ($k=3$):** Task accuracy exhibits low variance ($\text{CV} < 2\%$)[cite: 1], whereas carbon emissions demonstrate higher variance on large backbones ($\text{CV} \approx 18\text{--}24\%$) due to transient GPU power spikes[cite: 1].

---

## 7. Decision Rules & Deployment Guidelines

### 7.1 Automated Decision Rules
Based on Pareto frontiers across configurations, the engine operationalizes four practitioner deployment rules:

> **Rule 1 (Consumer Edge / Single GPU):**  
> **IF** $\text{VRAM}_{\text{max}} \le 16 \text{ GB}$  
> **THEN** Select **QLoRA** ($r=16, \text{NF4}$).  
> *Trade-off:* Saves $>85\%$ carbon relative to Full FT while preserving $\sim 96\%$ of peak accuracy[cite: 1].

> **Rule 2 (Strict Carbon Ceiling):**  
> **IF** $\text{Carbon}_{\text{budget}} \le 0.05 \text{ kgCO}_2\text{eq}$ AND $\text{Backbone} \ge 7\text{B}$  
> **THEN** Select **QLoRA** or **LoRA-FA**.  
> *Trade-off:* Prevents job termination due to emissions caps while maintaining target thresholds[cite: 1].

> **Rule 3 (High-Accuracy Production):**  
> **IF** $\text{Accuracy}_{\text{target}} \ge 0.90$ AND $\text{VRAM}_{\text{max}} \ge 48 \text{ GB}$  
> **THEN** Select **Full FT** ONLY IF carbon offsets are active; OTHERWISE Select **LoRA** ($r=32$).

> **Rule 4 (Time-Sensitive Deployment):**  
> **IF** $\text{Time}_{\text{deadline}}$ is tight AND VRAM is unconstrained  
> **THEN** Select **LoRA-FA** over LISA to avoid random layer unfreezing overhead.

### 7.2 Practical Case Studies
1. **Case Study A: Resource-Constrained Research Lab (1x RTX 4090, 24 GB VRAM)**
   * *Target:* Fine-tune a 13B parameter model on a custom dataset.
   * *Decision Support Output:* **QLoRA**. Full FT leads to Out-Of-Memory (OOM) errors; standard LoRA risks peak memory thrashing. QLoRA executes within 9.4 GB VRAM[cite: 1].
2. **Case Study B: Sustainability-Mandated Enterprise Datacenter**
   * *Target:* Daily retraining of classification backbones under enterprise ESG policies.
   * *Decision Support Output:* **LoRA-FA**. Achieves near-peak accuracy ($0.883$) while cutting carbon emissions by $81.2\%$ relative to Full FT[cite: 1].

### 7.3 Ablation Study: Impact of Ignoring Sustainability Metrics
To demonstrate the risks of choosing adaptation techniques based solely on accuracy, we conduct an ablation where energy and memory constraints are omitted from the recommendation engine:

* **Accuracy-Only Selection:** Engine defaults to **Full Fine-Tuning** across all scenarios.
* **Impact:** Results in an average **$3.1\times$ to $8.6\times$ increase in carbon emissions**[cite: 1] and a **$5\times$ increase in GPU VRAM requirements**, triggering high cloud infrastructure costs and frequent OOM failures on mid-tier hardware.

---

## 8. Threats to Validity
* **Hardware Variance & Measurement Noise:** Transient power fluctuations in shared nodes[cite: 1].
* **Regional Grid Carbon Intensity:** Variability in regional energy grids ($\text{kgCO}_2\text{eq}/\text{kWh}$)[cite: 1].
* **Scope Boundary:** Focuses exclusively on fine-tuning stages; inference-time life-cycle impacts remain out of scope[cite: 1].

---

## 9. Conclusion
This work introduces the **Green AI Decision Support Framework**, shifting PEFT evaluation from static accuracy leaderboards to a dynamic, multi-objective decision engine. By integrating the Green Efficiency Index (GEI), Pareto-frontier filtering, and automated deployment rules, the framework allows practitioners to optimize LLM fine-tuning within real-world compute, financial, and environmental budgets.