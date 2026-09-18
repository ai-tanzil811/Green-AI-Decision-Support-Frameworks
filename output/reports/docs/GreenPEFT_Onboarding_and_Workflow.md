# GreenPEFT — Onboarding, Workflow & Results Evaluation

**Status:** First real Kaggle run complete. Pipeline is proven. Two bugs are blocking full results.
**Last updated from:** `report.md` (real Kaggle run, T4 16GB, SST-2, `run_mode=real`)

---

## 1. How to read this document

This file has four parts:

| Part | What it tells you |
|---|---|
| 2. Onboarding | How a new team member gets the notebook running from zero |
| 3. Workflow | What the pipeline actually does, step by step |
| 4. Results evaluation | Honest read of your real run — what worked, what didn't, and why |
| 5. Downstream use | What you can and cannot claim in the paper/slides right now |
| 6. Future directions | The fix list, in priority order, to get full results |

---

## 2. Onboarding (for a new team member)

### 2.1 What you need before you start

| Requirement | Why |
|---|---|
| Kaggle account | Free GPU (T4 x2, 16 GB), 30 GPU-hours/week |
| No HuggingFace token needed | All 4 backbone models are open, ungated |
| No local install needed | Everything runs in the Kaggle notebook |

### 2.2 First-time setup (5 steps)

```
1. Go to kaggle.com -> Create -> New Notebook -> File -> Import Notebook
2. Upload: group01_v2_real_pipeline.ipynb
3. Right panel -> Settings -> Accelerator = GPU T4 x2
4. Right panel -> Settings -> Internet = ON
5. Run cells 1 -> 16, top to bottom, one at a time
```

### 2.3 The one cell you edit

Everything you control lives in **Cell 2**. You do not need to touch any other cell.

| Setting | What it controls | Start with |
|---|---|---|
| `RUN_MODE` | `smoke` / `real` / `synthetic` | `smoke` first, always |
| `METHODS` | which PEFT methods to test | leave as-is once bug is fixed (§6) |
| `SEEDS` | how many repeats per config | `[13, 42, 2024]` |
| `TRAIN_STEPS` | how long each run trains | `100` (see §6.3 — likely too low for LISA) |

### 2.4 Your first run — do this exactly once

1. Leave `RUN_MODE = 'smoke'`. Run all cells. Takes ~10 minutes.
2. Check the last cell produced a `.zip` in the Kaggle Output panel.
3. Only after smoke succeeds, switch to `RUN_MODE = 'real'`, set `INSTALL = False` in Cell 1, restart the session, and run all cells again.

**Do not skip smoke mode.** It is what caught that LoRA/QLoRA were broken before wasting the full sweep on them.

---

## 3. Workflow — what the pipeline does

```
 HuggingFace (SST-2 dataset)
          │
          ▼
 ┌─────────────────────────┐
 │  Cell 4: download data   │
 └────────────┬─────────────┘
              ▼
 ┌─────────────────────────────────────────┐
 │  Cell 8: run_one(method, backbone, seed)  │  <- runs 60 times
 │    1. build model + attach PEFT method    │
 │    2. train for TRAIN_STEPS               │
 │    3. measure: time, VRAM, watts -> kWh    │
 │    4. evaluate accuracy                    │
 │    5. save one JSON per run                │
 └────────────┬─────────────────────────────┘
              ▼
 ┌─────────────────────────┐
 │  Cell 11: aggregate JSON  │  ->  sweep_raw.csv, aggregated.csv
 └────────────┬─────────────┘
              ▼
 ┌─────────────────────────┐
 │  Cell 12: weighted score  │  ->  aggregated_scored.csv (score_balanced, 0-1)
 └────────────┬─────────────┘
              ▼
 ┌─────────────────────────┐
 │  Cell 13: GEI + Pareto    │  ->  aggregated_gei.csv, pareto_fronts.csv
 └────────────┬─────────────┘
              ▼
 ┌─────────────────────────┐
 │  Cell 14: decision engine │  ->  "given VRAM/carbon/accuracy limits, pick a method"
 └────────────┬─────────────┘
              ▼
 ┌─────────────────────────┐
 │  Cell 15/16: plots + zip  │  ->  what you download and put in the paper
 └───────────────────────────┘
```

**One naming trap to know about:** the pipeline actually produces *two* different "goodness scores," and your `report.md` used the term GEI for both. Keep them separate:

| Name | Cell | Formula shape | Range |
|---|---|---|---|
| `score_balanced` | 12 | weighted **sum** of normalized metrics | 0 to 1 |
| `gei` | 13 | accuracy **raised to a power**, divided by energy/memory/time ratios | can exceed 1 |

The 0.964 / 0.999 numbers in your report are `score_balanced`-shaped (linear, capped near 1), not the Cell 13 `gei` formula from the paper's §4.1. When you write the methodology section, pick one name per metric and cite the matching equation — don't call both "GEI."

---

## 4. Results evaluation — what your real run actually shows

### 4.1 The scoreboard, in plain terms

| Category | Runs | Share | What it means |
|---|---|---|---|
| ✅ Succeeded | 12 / 60 | 20% | Full FT (tiny only) + LISA (tiny, small, medium) |
| ❌ Out of memory | 36 / 60 | 60% | Full FT above 0.5B; LISA at 3B — **real hardware limit** |
| ❌ Dependency error | 12 / 60 | 20% | LoRA, QLoRA, LoRA-FA — **not a hardware limit, a bug** |

### 4.2 What genuinely worked — trust these numbers

**Full FT, 0.5B model:** 91.2% accuracy in 100 steps, 13.48 GB VRAM. This is a real, working, well-trained result. It also confirms the training pipeline, tokenizer, and evaluation code are all correct — because if they were broken, you would not get 91% on a real task.

**The memory wall is real:** Full FT could not run past 0.5B parameters on a 16 GB GPU. That is exactly the paper's core motivation (§1: *"full fine-tuning is computationally prohibitive"*) — and now you have a measured number to cite instead of a general claim.

### 4.3 What did NOT work, and why — two separate problems

**Problem A — LoRA / QLoRA / LoRA-FA never actually ran (a bug, fixable in one line)**

```
Cause:  Kaggle's pre-installed torchao (0.10.0) is older than what
        peft's LoRA path expects (>=0.16.0). bitsandbytes needs
        >=0.46.1 for QLoRA's 4-bit path. Neither got upgraded.

Effect: Every LoRA/QLoRA/LoRA-FA run failed before training started.
        This is an environment problem, not a finding about the methods.
```

This matters because LoRA and QLoRA are the two methods your paper's abstract and Rule 1/Rule 2 lean on most heavily. Right now you have **zero real measurements for them.** The fix is in §6.1.

**Problem B — LISA ran, but the accuracy (48–61%) is close to random guessing**

SST-2 is roughly balanced, so guessing one class every time scores about 51%. LISA's 50.9%, 61.2%, and 48.9% across the three tiers are *at or barely above* that floor. This is not "LISA is a worse method" — it's "LISA did not learn."

The likely cause, reading Cell 6/8 of the notebook:

```
LISA unfreezes only ~25% of layers, and swaps which 25%
every 20 steps. With only 100 total training steps, each
layer subset gets trained for roughly 20 steps before being
frozen again and swapped out — not enough steps for any
subset to learn anything durable, especially at the
low learning rate (1e-5) reused from Full FT.
```

This is a **hyperparameter problem**, not a hardware problem — it ran, it just needs a longer or gentler schedule. Fix in §6.3.

### 4.4 Quick self-check table (for your own sanity, and for reviewers)

| Question | Answer from this run |
|---|---|
| Does the pipeline measure real energy/VRAM/time? | Yes — Full FT numbers are internally consistent and plausible for a T4 |
| Does the pipeline train a real model? | Yes — 91% accuracy proves real learning happened |
| Can we compare LoRA vs QLoRA vs Full FT yet? | **No** — LoRA/QLoRA never ran |
| Can we claim LISA is worse than Full FT? | **No** — LISA under-trained, this isn't a fair comparison yet |
| Can we claim "PEFT is necessary above 0.5B on 16GB"? | **Yes** — this is a genuine, measured result |

---

## 5. Downstream use — what to do with what you have today

### 5.1 Safe to use in the paper / slides right now

- The Full FT memory-wall number (13.48 GB at 0.5B, OOM above it) — cite this directly, it's real.
- The pipeline description itself (Cells 1–16, the measurement method, NVML power sampling) — this is a real, working measurement protocol and is publishable as your methodology.
- The 20%/60%/20% success/OOM/dependency-error breakdown, framed honestly as "first sweep, environment issue found and being fixed" — reviewers respect a documented, resolved bug far more than silently missing data.

### 5.2 Not safe to use yet

- Any accuracy or GEI comparison between full_ft, lora, qlora, lora_fa, lisa as a group — you only have 2 of 5 methods with valid runs.
- The "LISA (0.5B Tiny), GEI=0.999" recommendation from the decision-engine scenarios — this recommendation is only correct *because* LoRA/QLoRA are missing from the candidate pool, not because LISA actually beat them. Re-run before quoting Scenario 1/2/4 in your report.
- Any claim about LISA's accuracy vs Full FT's accuracy — the comparison isn't fair until LISA is properly trained.

### 5.3 What is fine to present as "in progress"

If a deadline is close, it's fine to show the current results as a **pilot / proof of concept slide**, explicitly labeled as such, followed by "full 5-method sweep in progress, expected [date]." That is a normal and credible thing to show — a working pipeline with one clean result is a stronger status update than no pipeline at all.

---

## 6. Future directions — fix list, in priority order

### 6.1 Priority 1: unblock LoRA / QLoRA / LoRA-FA (do this first)

Add a pinned upgrade at the top of **Cell 1**, before the existing `REQUIRED` install:

```python
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade',
                 'peft>=0.12.0', 'bitsandbytes>=0.46.1', 'torchao>=0.16.0'],
                check=False)
```

Then restart the Kaggle session (package upgrades need a fresh kernel to take effect) and re-run `RUN_MODE = 'smoke'` before trusting it. This single fix should recover the 12 dependency-blocked runs, and with reruns, the corresponding runs across all 4 tiers (36 runs currently marked broken by this bug's downstream effects).

### 6.2 Priority 2: give LISA a fair chance

In Cell 2's `METHOD_CONFIGS` (or the `lisa` block), change:

| Setting | Old | New | Why |
|---|---|---|---|
| `resample_every_n_steps` | 20 | 50 | let each layer subset actually learn before swapping |
| `layer_sample_prob` | 0.25 | 0.4 | more capacity training at once |
| `learning_rate` | 1e-5 (shared with full_ft) | 5e-5 | LISA trains far fewer params, so it can tolerate a higher rate |

Also consider raising `TRAIN_STEPS` from 100 to 300 for a fairer read — 100 steps is a tight budget for any layer-sampling method.

### 6.3 Priority 3: re-run the full sweep

Once 6.1 and 6.2 are in place:
1. `RUN_MODE = 'smoke'` — confirm no errors, ~10 min.
2. `RUN_MODE = 'real'` — full 60-run sweep (or more, once LoRA/QLoRA succeed, expect closer to 4–8 hrs since they're cheap methods).
3. Delete the old `raw/*.json` files for `lora`, `qlora`, `lora_fa`, and any `lisa` run — the sweep driver (Cell 10) skips runs whose JSON already exists, so stale broken results won't be overwritten automatically.

### 6.4 Priority 4: fill in the paper's remaining scope

| Gap vs. paper's stated scope | What to do |
|---|---|
| Only `classification` (SST-2) tested | Set `TASKS = ['classification', 'summarization']` once the primary sweep is clean |
| Only up to 3B backbone tested | Paper claims "~1B to ~13B" — either add a larger tier or narrow the claim to match what was actually run |
| No statistical significance test run yet | Once ≥3 seeds exist for every method, run ANOVA/Tukey HSD (§6.2 of the paper) on the aggregated CSV |
| Ablation study (§7.3, accuracy-only selection) | Re-run the decision engine (Cell 14) with weights forcing `w_perf=1.0`, compare to the sustainability-aware recommendation |

### 6.5 Nice-to-have, not urgent

- Add `carbon_kg_per_kwh = 0.150` (renewable grid) as a second scenario, to match §5 of the paper's "renewable vs fossil" comparison.
- Log GPU temperature alongside power draw — useful for explaining the ~18–24% carbon variance the paper predicts at large scale.

---

## 7. One-paragraph status summary (for a quick standup update)

> The GreenPEFT pipeline is confirmed working end-to-end on Kaggle: real data, real training, real GPU power measurement. Full fine-tuning scored 91% accuracy at 0.5B parameters and then hit a genuine 16GB memory wall above that — a clean, citable result. Two issues are blocking the full comparison: a package-version bug stopped LoRA/QLoRA/LoRA-FA from running at all, and LISA's training schedule was too short to learn properly. Both have known one-line fixes. Next sweep should recover the full 5-method comparison the paper needs.
