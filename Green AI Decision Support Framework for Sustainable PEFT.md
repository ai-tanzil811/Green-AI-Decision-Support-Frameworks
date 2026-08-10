# Green AI Decision Support Framework for Sustainable PEFT

> **A sustainability-aware framework for selecting Parameter-Efficient Fine-Tuning (PEFT) strategies under real-world performance, memory, energy, carbon, and resource constraints.**

**Status:** 🟡 Active Research / Experimental Development  
**Primary Environment:** Kaggle GPU / NVIDIA T4 16 GB  
**Research Stage:** Pilot real-data experiment completed; full controlled benchmark in progress

---

## 1. Overview

Large language model (LLM) fine-tuning can require substantial computational resources. While Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA and QLoRA reduce trainable parameters and memory requirements, choosing the most appropriate method is not simply a matter of selecting the method with the highest accuracy.

A method may achieve slightly higher accuracy while consuming substantially more:

- GPU memory
- Training time
- Energy
- Carbon emissions
- Computational cost

This project proposes a **Green AI Decision Support Framework** that treats PEFT selection as a multi-objective decision problem.

Instead of asking:

> **"Which PEFT method performs best?"**

the project asks:

> **"Which PEFT strategy is most appropriate under a given combination of accuracy, hardware, time, energy, carbon, and resource constraints?"**

The long-term goal is to develop a practical decision-support system capable of recommending an appropriate fine-tuning strategy without requiring practitioners to exhaustively train every candidate method.

---

# 2. Research Motivation

Current PEFT research often emphasizes:

- task performance
- parameter efficiency
- memory reduction

However, practical deployment also involves:

- GPU availability
- peak memory requirements
- training duration
- electricity consumption
- carbon emissions
- monetary cost
- sustainability constraints

A practitioner with a 16 GB GPU and a strict carbon budget may not want the method that achieves the absolute highest accuracy.

They may instead need the method that provides the best **performance–resource–sustainability trade-off**.

This motivates the proposed Green AI Decision Support Framework.

---

# 3. Central Research Idea

The project is structured around the following pipeline:

```text
                     USER CONSTRAINTS
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        GPU Memory     Carbon Budget   Accuracy Target
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                GREEN AI DECISION ENGINE
                           ▲
                           │
                  EXPERIMENTAL DATA
                           ▲
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
      LoRA               QLoRA             LoRA-FA
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
                  CONTROLLED TRAINING
                           │
                           ▼
                 RESOURCE MONITORING
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
    Accuracy           GPU Memory          Energy
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                     Carbon / Cost
                           │
                           ▼
                 MULTI-OBJECTIVE ANALYSIS
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
              Pareto Analysis      GEI
                  │                 │
                  └────────┬────────┘
                           ▼
                 GREEN RECOMMENDATION
                           │
                           ▼
                PRACTICAL GUIDELINES
```

---

# 4. Research Questions

The current research is organized around four questions.

### RQ1 — Sustainability Benchmarking

How do different PEFT strategies compare in terms of:

- task performance
- GPU memory
- training time
- energy consumption
- carbon emissions
- computational cost?

### RQ2 — Multi-Objective Trade-offs

Which PEFT configurations lie on the Pareto frontier when performance and sustainability objectives are considered simultaneously?

### RQ3 — Decision Support

Can a sustainability-aware decision framework recommend an appropriate PEFT strategy based on:

- available GPU memory
- model size
- accuracy requirements
- carbon budget
- training-time constraints?

### RQ4 — Predictive Recommendation

Can a lightweight predictive model estimate the performance and sustainability characteristics of candidate PEFT strategies using inexpensive pre-training features, allowing recommendations without exhaustively running every candidate method?

> **RQ4 represents the main planned novelty upgrade beyond a conventional benchmark.**

---

# 5. Proposed Contributions

The project is being developed around the following contributions.

## 5.1 Controlled Sustainability Benchmark

A controlled experimental protocol for comparing PEFT methods under identical:

- model
- dataset
- training configuration
- hardware
- random seeds
- evaluation conditions

---

## 5.2 Multi-Objective Sustainability Evaluation

The framework evaluates PEFT methods using multiple objectives rather than accuracy alone:

```text
Performance
Memory
Training Time
Energy
Carbon
Cost
```

---

## 5.3 Pareto-Based Analysis

Rather than assuming one universal "best" method, the framework identifies configurations that provide favorable trade-offs across competing objectives.

A configuration is considered Pareto-efficient when no other configuration improves one objective without worsening at least one other objective.

---

## 5.4 Green Efficiency Index (GEI)

A composite decision metric is used to summarize sustainability and performance characteristics.

The current framework considers normalized dimensions such as:

- performance
- memory efficiency
- time efficiency
- energy efficiency
- carbon efficiency

The exact weighting methodology remains an active research component and will be refined before final publication.

---

## 5.5 Green AI Decision Engine

The decision engine accepts practical constraints such as:

```text
Available GPU memory
Model size
Accuracy requirement
Carbon budget
Training-time constraint
```

and produces a recommended PEFT configuration from the experimentally validated candidate set.

---

## 5.6 Planned Predictive Decision Layer

The major planned novelty enhancement is a lightweight surrogate/predictive model.

Instead of requiring:

```text
Run LoRA
Run QLoRA
Run LoRA-FA
Run DoRA
...
↓
Compare
↓
Recommend
```

the proposed system aims to learn:

```text
Cheap pre-training features
        ↓
Predictive Model
        ↓
Predicted Accuracy
Predicted Energy
Predicted Carbon
Predicted Memory
        ↓
Decision Engine
        ↓
Recommended PEFT
```

This would allow the framework to recommend methods for previously unseen combinations without requiring exhaustive experimentation.

---

# 6. Current Experimental Scope

The project is intentionally designed around a manageable research scope suitable for a university-level project with limited computational resources.

## Hardware

Initial experiments use:

```text
GPU: NVIDIA Tesla T4
GPU Memory: 16 GB
Environment: Kaggle Notebook
```

The current experimental design treats the T4 as a **single-GPU environment**.

---

## Models

The experimental model range currently includes small open-weight decoder models in approximately the:

```text
0.5B – 3B parameter range
```

The current study should **not** claim experimental validation for 7B–13B models unless those models are actually tested.

Larger models may be discussed as future work.

---

## Methods

The planned candidate methods include:

### Full Fine-Tuning

All model parameters are updated.

Purpose:

- reference baseline
- performance comparison
- resource feasibility analysis

---

### LoRA

Low-Rank Adaptation.

Purpose:

- parameter-efficient baseline
- memory/performance comparison

---

### QLoRA

Quantized LoRA fine-tuning.

Purpose:

- reduced memory requirements
- low-resource experimentation
- sustainability comparison

---

### LoRA-FA

LoRA with frozen A matrices.

Purpose:

- additional PEFT comparison
- memory/resource efficiency analysis

---

### LISA

LISA is being investigated as an additional PEFT strategy.

However, the current implementation requires further validation and tuning before its results should be treated as a definitive comparison against the other PEFT methods.

---

### Future Candidate Methods

Possible future extensions include:

- DoRA
- GaLore

These are **not required for the first complete experimental milestone**.

---

# 7. Dataset and Task

The first controlled task is:

```text
SST-2
```

The task provides a manageable classification benchmark for establishing the experimental pipeline.

The project may later expand to additional tasks such as:

- summarization
- instruction following
- question answering

However, expansion will occur only after the initial controlled experiment is stable.

---

# 8. Experimental Design

The experiment attempts to control the following variables:

```text
Model
Dataset
PEFT Method
Random Seed
Training Configuration
Hardware
Evaluation Procedure
```

The primary measured variables are:

```text
Accuracy
F1 Score
Training Time
Peak GPU Memory
Energy Consumption
Carbon Emissions
Cost
```

---

# 9. Real Experimental Measurement

The project distinguishes carefully between:

### Measured Data

Values obtained directly from real training runs.

Examples:

- measured training duration
- measured GPU power
- measured GPU memory
- actual model evaluation score

### Estimated Data

Values calculated from measured quantities.

For example:

```text
Energy → Carbon estimation
```

using an appropriate carbon-intensity assumption.

### Synthetic Data

Synthetic values were previously used for validating the analysis pipeline.

Synthetic results are **not treated as empirical research findings**.

The final paper will clearly distinguish:

```text
Synthetic / Pipeline Validation
                ≠
Real Experimental Evidence
```

---

# 10. Energy and Carbon Measurement

GPU energy consumption is monitored using NVIDIA GPU telemetry where available.

The experimental pipeline is designed to capture:

```text
GPU Power
GPU Utilization
GPU Memory
Training Duration
```

Energy is derived from power consumption over time.

Carbon emissions are then estimated from energy consumption and the selected grid carbon-intensity assumption.

The final paper will report the measurement methodology and assumptions explicitly.

---

# 11. Pilot Real Experiment

A first real Kaggle experiment has already been completed.

This experiment is considered a **pilot experiment**, not the final benchmark.

## Pilot Environment

```text
Platform: Kaggle
GPU: NVIDIA T4
VRAM: 16 GB
```

## Planned Run Grid

The initial experiment was designed around approximately:

```text
Multiple backbones
×
Multiple PEFT methods
×
Multiple seeds
```

However, only a subset of the planned runs completed successfully.

---

# 12. Pilot Results

The first real run successfully produced empirical measurements for several configurations.

Observed successful configurations included:

| Configuration | Accuracy |
|---|---:|
| Full FT + 0.5B | ~91.2% |
| LISA + 0.5B | ~50.9% |
| LISA + 1.1B | ~61.2% |
| LISA + 1.5B | ~48.9% |

These values are **pilot observations**, not final benchmark conclusions.

The successful runs also captured sustainability/resource measurements such as:

- training time
- peak GPU memory
- energy
- carbon
- GEI

---

# 13. Important Pilot Finding: GPU Memory Boundary

The pilot experiment revealed an important hardware constraint.

Full Fine-Tuning of the smallest tested model completed successfully, while larger Full-FT configurations exceeded the available 16 GB GPU memory under the tested configuration.

This should be interpreted as:

> **Under the experimental configuration, Full Fine-Tuning exceeded the available 16 GB GPU memory for the larger tested models.**

It should **not** be generalized to:

> "Full Fine-Tuning is impossible for models above 0.5B."

The result is hardware- and configuration-dependent.

---

# 14. Pilot Failure Analysis

A major purpose of the first run was identifying implementation and environment problems.

The initial run experienced a substantial number of unsuccessful configurations.

The primary issues included:

### 14.1 Out-of-Memory Errors

Full Fine-Tuning of larger models exceeded the available GPU memory.

These failures are useful resource-feasibility observations and should be retained rather than silently discarded.

---

### 14.2 Dependency Problems

Several PEFT configurations were blocked by compatibility issues involving packages such as:

```text
torchao
bitsandbytes
```

These failures are treated as **environment/implementation failures**, not scientific results.

The experimental environment must be stabilized before drawing comparisons among the affected PEFT methods.

---

# 15. Current Interpretation of LISA Results

The pilot experiment produced low classification performance for the current LISA configuration.

Therefore, the present LISA result should **not** be interpreted as evidence that LISA is inherently unsuitable.

The more appropriate interpretation is:

> The current LISA configuration produced poor performance in the pilot classification experiment and requires implementation/configuration validation before definitive comparison.

A future ablation may investigate different sampling configurations.

For example:

```text
Sampling ratio:
0.25
0.50
0.70
```

while keeping other settings controlled.

Potential outcomes can then be compared across:

```text
Accuracy
Memory
Energy
Carbon
```

---

# 16. Current Decision Engine Status

The decision engine has already been prototyped.

However, the current pilot dataset is incomplete.

LoRA, QLoRA, and LoRA-FA were not successfully evaluated in the initial run.

Therefore:

> **The current recommendation output must not be presented as the final Green AI recommendation.**

For example, if the current incomplete dataset selects LISA because it has the highest GEI among the available observations, this only means:

> LISA achieved the highest GEI among the currently available pilot observations.

It does **not** mean:

> LISA is the globally optimal PEFT method.

The decision engine becomes scientifically meaningful only after the candidate methods have sufficient comparable real measurements.

---

# 17. Research Workflow

The final intended workflow is:

```text
                ┌──────────────────────┐
                │ User Requirements    │
                ├──────────────────────┤
                │ GPU Memory           │
                │ Accuracy Target      │
                │ Carbon Budget        │
                │ Time Constraint      │
                └──────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Candidate PEFT Methods │
              ├────────────────────────┤
              │ Full FT                │
              │ LoRA                   │
              │ QLoRA                  │
              │ LoRA-FA                │
              │ LISA                   │
              └───────────┬────────────┘
                          │
                          ▼
                 Controlled Training
                          │
                          ▼
              ┌────────────────────────┐
              │ Monitoring Layer       │
              ├────────────────────────┤
              │ Accuracy               │
              │ GPU Memory             │
              │ Time                   │
              │ Energy                 │
              │ Carbon                 │
              │ Cost                   │
              └───────────┬────────────┘
                          │
                          ▼
                 Experimental Dataset
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
       Pareto Analysis              GEI
             │                         │
             └────────────┬────────────┘
                          ▼
              Predictive / Surrogate
                    Model Layer
                          │
                          ▼
                Decision Intelligence
                          │
                          ▼
                  PEFT Recommendation
                          │
                          ▼
               Practical Guidelines
```

---

# 18. Planned Predictive Model

The strongest planned novelty component is a surrogate model.

## Input Features

Potential features include:

```text
Model Parameter Count
Dataset Size
Sequence Length
Task Type
PEFT Method
LoRA Rank
Quantization Level
GPU Memory
Hardware Characteristics
```

## Predicted Outputs

The model will attempt to predict:

```text
Accuracy
Peak Memory
Training Time
Energy
Carbon
```

---

# 19. Candidate Predictive Models

The project intentionally starts with simple models.

Possible candidates:

- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost

The goal is not to build a complicated neural network.

The goal is:

> **Can a lightweight predictive model provide sufficiently useful estimates for decision-making?**

This keeps the method understandable and feasible for a university research project.

---

# 20. Predictive Decision Workflow

The intended future workflow is:

```text
New Model / Task / Hardware
             │
             ▼
     Extract Cheap Features
             │
             ▼
       Surrogate Model
             │
       ┌─────┼─────┐
       ▼     ▼     ▼
   Accuracy Energy Carbon
       │     │     │
       └─────┼─────┘
             ▼
      Constraint Engine
             │
             ▼
       Pareto / GEI
             │
             ▼
       Recommendation
```

This is the key mechanism that can transform the work from a conventional benchmark into a **predictive decision-support system**.

---

# 21. Green Efficiency Index

The Green Efficiency Index (GEI) is intended to combine multiple objectives into an interpretable sustainability score.

Conceptually:

```text
GEI =
Performance contribution
+
Memory efficiency
+
Time efficiency
+
Energy efficiency
+
Carbon efficiency
```

All metrics must first be normalized so that different units can be compared.

The weighting scheme remains under development.

The final research version should avoid arbitrary weights without justification.

Possible approaches include:

### Approach A — Scenario-Based Constraints

Examples:

```text
Accuracy-first
Carbon-constrained
Memory-constrained
Time-constrained
Balanced
```

### Approach B — AHP

Use pairwise comparisons to establish relative importance of objectives.

### Approach C — Pareto-first Decision Making

Use Pareto filtering before applying a secondary ranking method.

The final choice will be based on feasibility and methodological defensibility.

---

# 22. Pareto Analysis

The framework treats sustainability as a multi-objective optimization problem.

Objectives include:

```text
Maximize:
    Accuracy

Minimize:
    Memory
    Time
    Energy
    Carbon
    Cost
```

A configuration that is dominated across all relevant objectives is removed from the candidate set.

The remaining configurations form the Pareto frontier.

---

# 23. Decision Scenarios

The decision engine will eventually support scenarios such as:

### Scenario 1 — Accuracy Priority

Constraint:

```text
Accuracy ≥ target
```

Then select the most sustainable configuration satisfying the target.

---

### Scenario 2 — Memory-Constrained

Example:

```text
GPU VRAM ≤ 8 GB
```

Only feasible methods are considered.

---

### Scenario 3 — Carbon-Constrained

Example:

```text
Carbon ≤ predefined budget
```

Configurations exceeding the budget are rejected.

---

### Scenario 4 — Time-Constrained

Example:

```text
Training Time ≤ deadline
```

Methods exceeding the deadline are excluded.

---

### Scenario 5 — Balanced

Consider multiple objectives simultaneously using Pareto analysis and GEI.

---

# 24. Experimental Milestones

The research should proceed in stages.

## Milestone 1 — Complete One Real Experimental Block

Minimum target:

```text
2 Backbones
×
1 Task
×
3 PEFT Methods
×
3 Seeds
```

Preferred initial methods:

```text
LoRA
QLoRA
LoRA-FA
```

This milestone is more important than immediately expanding the number of models or methods.

---

## Milestone 2 — Expand the Benchmark

After the first block works reliably:

```text
Additional backbone
+
Additional PEFT method
+
Additional task
```

Potential future methods:

```text
DoRA
GaLore
```

---

## Milestone 3 — Build the Surrogate Model

Use the real experimental dataset to train a lightweight predictive model.

---

## Milestone 4 — Validate the Decision Engine

Systematically evaluate recommendations across many constraint combinations.

Potential dimensions:

```text
VRAM:
8 / 16 / 24 / 48 GB

Carbon:
Low / Medium / High

Accuracy:
Multiple target levels

Time:
Multiple limits
```

Measure:

- recommendation coverage
- constraint satisfaction
- recommendation stability
- prediction error
- regret compared with exhaustive evaluation

---

## Milestone 5 — Generalization

If sufficient data becomes available:

```text
Hardware A
     ↓
Train predictive model
     ↓
Test on Hardware B
```

This can evaluate whether the framework generalizes beyond the hardware on which it was trained.

---

# 25. What Counts as a Successful Recommendation?

A recommendation should satisfy the user's explicit constraints.

For example:

```text
GPU = 16 GB
Accuracy Target = 90%
Carbon Budget = X
```

The system should:

1. Remove infeasible configurations.
2. Remove configurations below the accuracy target.
3. Evaluate remaining candidates.
4. Identify Pareto-efficient candidates.
5. Apply the selected decision rule.
6. Return the recommended method.
7. Explain the trade-off.

Example output:

```text
Recommended Method: QLoRA

Reason:
- Fits available GPU memory
- Meets accuracy threshold
- Lower estimated energy than alternatives
- Lower estimated carbon
- Acceptable training time

Trade-off:
- Slightly lower expected accuracy than Full FT
```

---

# 26. Reproducibility

The final project should record:

```text
Python version
PyTorch version
Transformers version
PEFT version
bitsandbytes version
CUDA version
GPU model
GPU memory
Dataset version
Model version
Random seeds
Training configuration
Energy measurement method
Carbon intensity assumption
```

All raw experimental results should be preserved.

---

# 27. Research Integrity Rules

The project follows several rules.

### Rule 1

Synthetic data must never be presented as real experimental evidence.

### Rule 2

Failed experiments must be distinguished from successful measurements.

### Rule 3

OOM should be recorded as a resource-feasibility outcome when appropriate.

### Rule 4

Dependency failures are implementation/environment failures, not model-performance results.

### Rule 5

A recommendation generated from an incomplete candidate set must be labeled as provisional.

### Rule 6

A method should not be claimed to be superior unless comparable experimental evidence exists.

### Rule 7

Carbon values must include their estimation assumptions.

---

# 28. Current Limitations

The current project has several limitations.

### Limited Hardware

Experiments are primarily conducted on an NVIDIA T4 16 GB environment.

### Limited Model Scale

The current empirical scope is focused on smaller models because of available computational resources.

### Limited Task Coverage

The initial task is SST-2 classification.

### Incomplete PEFT Coverage

The pilot run did not successfully evaluate all planned PEFT methods.

### LISA Validation

The current LISA configuration requires further validation.

### Predictive Layer Not Yet Implemented

The surrogate model is a planned contribution and has not yet been experimentally validated.

### Cross-Hardware Generalization

Cross-hardware validation has not yet been performed.

---

# 29. What This Project Is NOT

This project is not primarily trying to:

- invent a new PEFT algorithm
- compete with state-of-the-art LLM architectures
- perform massive-scale training
- claim universal superiority of one PEFT method
- claim that one GPU represents all hardware environments

Instead, the project focuses on:

> **Sustainability-aware decision-making for PEFT selection under practical resource constraints.**

---

# 30. Why the Research Can Be Novel

A simple paper asking:

> "Which PEFT method uses less energy?"

would be relatively incremental.

The stronger research direction is:

> **Can a system learn the relationship between model/task/hardware characteristics and PEFT sustainability outcomes, then use that knowledge to recommend a suitable PEFT strategy under explicit user constraints?**

This creates three layers:

```text
Layer 1
Experimental Evidence

        ↓

Layer 2
Predictive Modeling

        ↓

Layer 3
Decision Support
```

The benchmark supplies evidence.

The surrogate model provides prediction.

The decision engine provides practical recommendations.

---

# 31. Current Project Status

## Completed

- [x] Research problem definition
- [x] Green AI Decision Support concept
- [x] Research questions
- [x] Experimental methodology
- [x] PEFT candidate design
- [x] Sustainability metrics
- [x] GEI concept
- [x] Pareto analysis concept
- [x] Decision-engine prototype
- [x] Kaggle experimental environment
- [x] First real pilot experiment
- [x] Real GPU measurements
- [x] Initial energy/carbon measurements
- [x] Initial multi-seed observations
- [x] Initial GPU memory/OOM observations

## In Progress

- [ ] Fix PEFT dependency environment
- [ ] Complete LoRA experiments
- [ ] Complete QLoRA experiments
- [ ] Complete LoRA-FA experiments
- [ ] Validate/tune LISA
- [ ] Establish complete real experimental dataset
- [ ] Refine GEI methodology
- [ ] Validate decision engine

## Planned

- [ ] Train surrogate predictive model
- [ ] Predict accuracy/energy/carbon/memory
- [ ] Evaluate unseen configurations
- [ ] Large constraint-grid evaluation
- [ ] Recommendation stability analysis
- [ ] Optional DoRA/GaLore extension
- [ ] Cross-hardware validation
- [ ] Release decision-support tool
- [ ] Final research paper

---

# 32. Recommended Immediate Next Step

The most important next action is **not** to add more algorithms.

The next action is:

```text
Fix Environment
      ↓
Run LoRA
      ↓
Run QLoRA
      ↓
Run LoRA-FA
      ↓
3 Seeds
      ↓
2 Backbones
      ↓
Real Experimental Dataset
```

A smaller complete real experiment is more scientifically valuable than a large incomplete synthetic benchmark.

---

# 33. Research Roadmap

```text
                         CURRENT
                            │
                            ▼
                  ┌─────────────────┐
                  │ Pilot Experiment│
                  │      ✓          │
                  └────────┬────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Complete Real Benchmark│
              │      ← NEXT            │
              └────────────┬───────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ Real Dataset     │
                 └────────┬─────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Surrogate Model    │
                └────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Decision Intelligence│
              └──────────┬───────────┘
                         │
                         ▼
                ┌────────────────┐
                │ Validation     │
                └───────┬────────┘
                        │
                        ▼
               ┌──────────────────┐
               │ Research Paper   │
               └──────────────────┘
```

---

# 34. Final Research Vision

The ultimate system is intended to work as follows:

```text
User enters:

Model = 1.5B
GPU = 16 GB
Accuracy target = 90%
Carbon budget = X
Time budget = Y

                 ↓

        Green AI Decision Engine

                 ↓

      Predict candidate outcomes

                 ↓

       Remove infeasible methods

                 ↓

          Pareto analysis

                 ↓

        GEI / Decision Ranking

                 ↓

      Recommended PEFT Strategy

                 ↓

      Explanation of trade-offs
```

The final objective is therefore not simply:

> **"Find the best PEFT method."**

It is:

> **"Provide evidence-based, sustainability-aware recommendations for choosing PEFT strategies under real-world computational constraints."**

---

# 35. Project Philosophy

This project deliberately follows a **measurement-first, evidence-first** approach.

The order of development is:

```text
Measure
  ↓
Validate
  ↓
Compare
  ↓
Model
  ↓
Recommend
```

not:

```text
Assume
  ↓
Recommend
  ↓
Find evidence later
```

This distinction is essential for making the eventual research paper scientifically defensible.

---

## Current Bottom Line

The project has successfully moved beyond the proposal stage.

The first real Kaggle experiment has demonstrated that:

- the experimental pipeline can execute real training;
- real GPU/resource measurements can be collected;
- hardware memory limits can be observed empirically;
- the sustainability analysis pipeline can process real observations.

However, the pilot experiment is **not yet a complete benchmark**.

The immediate objective is therefore:

> **Complete a small but clean real PEFT benchmark before expanding the system with predictive modeling and large-scale decision support.**

Once that real dataset exists, the next major research step is the **surrogate predictive model**, which is intended to transform the project from a conventional PEFT benchmark into a predictive, constraint-aware Green AI decision-support framework.