# GreenPEFT — Session Progress Summary

## 1. Pipeline bugs found and fixed

The original notebook ran, but several bugs silently invalidated the numbers it produced. All are fixed in the patched cells.

| Bug | Symptom | Fix |
|---|---|---|
| Energy telemetry never worked | Every run reported exactly `70.0 W` (the fallback constant) | NVML probed with a real call at startup, falls back to `nvidia-smi`, every run tagged with `energy_source` |
| LISA wasn't training | Accuracy stuck at chance (0.489–0.509) | HF Trainer builds the optimizer once; the callback now only masks `requires_grad`, never adds new params after the fact, and purges stale Adam state for frozen layers |
| Full-FT ran in pure fp32, PEFT methods didn't | Confounded the memory/energy comparison | `FULL_FT_MIXED_PRECISION` flag, mixed precision by default |
| Kaggle "T4 x2" silently sharded across both GPUs | Energy/VRAM under-reported ~2x | `CUDA_VISIBLE_DEVICES=0` pinned before `import torch` |
| `torch.cuda.is_bf16_supported()` true via emulation on T4 | Slower runs, wrong dtype assumption | bf16 gated on compute capability ≥ 8.0 |
| Hard-coded LoRA target modules | Silently empty adapters on non-Qwen/Llama architectures (pythia, phi-2, Phi-3) | Per-architecture `resolve_target_modules()`, generic `find_decoder_layers()` |
| Padding every SST-2 sentence to 256 tokens (avg ~12 tokens) | ~95% of measured energy spent on padding | Dynamic padding via `DataCollatorWithPadding` |
| Classifier head randomly initialized at default scale | Loss started at 8+ instead of ~0.69 (chance) | `std=1e-3` init on the `score`/`classifier` head |
| Silent pip install failures (`check=False`) | 45/60 runs failed on missing/mismatched `torchao`/`bitsandbytes` with no warning | Install failures now print stdout/stderr and a hard verification block runs before the sweep |
| pip resolver bumped torch 2.10→2.14 (new CUDA 13 stack) as a side effect of upgrading `bitsandbytes`/`torchao` | Risk of breaking on Kaggle's driver | `torch==2.10.0+cu128` pinned explicitly in `REQUIRED` |
| Memory fragmentation across sequential in-process runs | `lora_medium_seed13` OOM'd while identical seeds 42/2024 succeeded | Identified as an artifact of the long-lived process; subprocess-per-run isolation recommended (not yet implemented) |
| `params.total` undercounts model size for QLoRA rows | bitsandbytes packs two 4-bit values per byte, halving the element count `model.parameters()` sees | Surrogate work uses the backbone's declared `params_b`, never `params.total` |

## 2. Real benchmark results (60 runs)

- 4 backbones (tiny 0.5B, small 1.1B, medium 1.5B, large 3.0B) × 5 methods (full_ft, lora, qlora, lora_fa, lisa) × 3 seeds = 60 configs.
- **41 `ok`, 19 `oom`** — and every failure is now a genuine physical result, not an import/dependency error.
- Clean scaling story: `full_ft` fails past 0.5B, `lisa`/`lora_fa` fail at 3B, **`qlora` is the only method that survives every tier including 3B** (7.45 GB peak).
- Pareto-front analysis: LoRA/LoRA-FA **dominate** full-FT at every tested tier (higher accuracy, less energy, less VRAM) — a clean, directly defensible RQ2 result.
- CodeCarbon cross-check runs alongside NVML; the two carbon estimates differ by a consistent ~40% due to differing grid-intensity assumptions (your static 0.65 kg/kWh vs CodeCarbon's regional lookup) — not a measurement disagreement. Worth one sentence in the methodology section.

## 3. Surrogate regressor (RQ4 — the paper's stated "core novelty")

Built and validated against the real 60-run dataset using **leave-one-tier-out (LOTO) cross-validation** — deliberately not random k-fold, since random k-fold let same-backbone/method seeds leak across train/test and made accuracy MAE look 2.3x better than it actually performs on a genuinely unseen scale.

**Two models:**
1. **Feasibility classifier** (all 60 runs) — predicts whether a config fits the VRAM budget.
2. **Performance regressors** (41 feasible runs) — predict accuracy, peak VRAM, energy, and wall-clock from cheap pre-training metadata (`params_b`, `rank`, `quant_bits`, `method`, `family`).

**Honest results:**
- **Energy: works.** R² = 0.374 under LOTO, beats the mean-predict baseline on every held-out tier. Energy scales smoothly and predictably with `params_b × quant_bits`.
- **Accuracy & peak VRAM: don't extrapolate yet.** Negative R² under LOTO — with only 4 distinct model scales in the training set, there isn't enough scale diversity to learn a curve that generalizes past 1.5B. This is a **data limitation, not a modeling problem**: the fix is running more of the already-built 13-model `MODEL_ZOO`, not a fancier regressor.
- This asymmetry (energy predictable, accuracy/VRAM not) is itself a reportable finding about what "cheap metadata" can and can't forecast at this data density.

All of this is now reusable: `surrogate_dataset.csv`, `surrogate_models.joblib` (production models fit on all 60 rows), and `surrogate_cv_metrics.json` (the honest LOTO numbers, ready to drop into the paper) are generated by the delivered **Cell 13** code.

## 4. Decision engine + CLI tool (RQ3 + Tier 3 roadmap item)

Built one shared recommendation engine (`green_peft/recommender.py`), used by two front-ends:

- **`green-peft` CLI** (installable package, `pip install -e .`) — `green-peft recommend --vram 16 --accuracy 0.90 --profile balanced`. Cross-products the *full* 13-model zoo × 5 methods (including the 9 backbones never actually run), runs all 4 surrogate models, applies constraint filtering, computes the Pareto front and GEI score (same weighted-sum formula as the README), and explains the top pick or, if nothing qualifies, exactly why.
- **Cell 14** (in-notebook version) — same logic, reads `MODEL_ZOO`/`METHOD_CONFIGS`/`production_models` straight from memory for live iteration during the defense.

**One real bug caught by testing against your actual data, not by inspection:** when the wall-clock surrogate model was absent, `NaN` silently propagated into both the Pareto comparison (NaN comparisons are always `False` in numpy, so *every* candidate came out "non-dominated" — 41/41) and the GEI sum (NaN poisons the whole score). Fixed by detecting the missing objective and excluding it from both calculations, with weights renormalized over the remaining three and an explicit note printed. Verified against the real bundle: 41 feasible → 14 genuinely Pareto-optimal, not 41/41.

Delivered artifacts:
- `green_peft_cli.zip` — the installable package (recommender + CLI + `pyproject.toml` + README).
- `surrogate_artifacts_export.zip` — your actual trained `surrogate_models.joblib` (all 4 targets), `surrogate_dataset.csv`, `surrogate_cv_metrics.json`, and the `configs/` the CLI needs — ready to run without touching Kaggle again.

## 5. Where this leaves the roadmap

- **Tier 1 (core methodology):** done, exceeded the stated minimum grid.
- **Tier 2 (novelty):**
  - Surrogate regressor — **done**, with an honest limitation documented (needs more scale diversity for accuracy/VRAM).
  - DoRA/GaLore integration — not started.
  - AHP-derived GEI weights — not started (still the arbitrary README weights).
  - Broad VRAM/carbon/accuracy grid validation — not started.
  - Differentiation section vs. prior work — writing task, not blocked technically.
- **Tier 3 (tooling):**
  - CLI tool — **done and tested** against real data.
  - Cross-hardware generalization (A100→T4) — not started, no A100 runs exist yet.
  - Venue strategy — open decision, not technically blocked.

## 6. Recommended next steps, in priority order

1. Widen `SWEEP_TIERS` using the already-built `MODEL_ZOO` (13 entries, only 4 used) to get more scale points into the surrogate before claiming accuracy/VRAM extrapolation works.
2. Isolate each sweep run in its own subprocess to eliminate the memory-fragmentation-driven OOM nondeterminism seen in `lora_medium_seed13`.
3. Re-run Cell 13 after widening tiers; check whether accuracy/VRAM R² turns positive — report the before/after as a methodological finding either way.
4. Decide on AHP survey vs. deployment-frequency-derived GEI weights to replace the arbitrary defaults.
5. Write the differentiation section against Chen et al. (2024) / Kaur et al. (2026) once the above numbers are locked.
