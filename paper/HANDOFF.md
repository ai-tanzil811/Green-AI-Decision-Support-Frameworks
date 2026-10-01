# GreenPEFT paper: author hand-off note

This note is for the author and is not part of the paper. It follows section 17 of
`GREENPEFT_PAPER_AGENT_PROMPT_MERGED.md`.

## Files

| File | What it is |
|---|---|
| `main.tex` | Paper draft, IEEEtran conference class. Each number has a `% [src: ...]` comment naming its source file. |
| `references.bib` | 23 entries, all cited. |
| `figures/fig1..fig5.{pdf,png}` | The five figures, as vector PDF and 300 dpi PNG. |
| `data/*.csv`, `data/*.json` | The values the tables were transcribed from. |
| `scripts/build_paper_assets.py` | Regenerates everything in `figures/` and `data/` from the repository. Run it from the repo root. It reads project files and the installed `green-peft` CLI, and writes only inside `paper/`. |

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
There is no LaTeX installation on this machine, so the draft has **not been compiled**. I
checked brace and environment balance, labels and references, citation keys, figure paths
and table column counts with a script. Compile it once in Overleaf or TeX Live and look at
the two wide tables (`\resizebox` scales them to text width).

## Measured / derived / predicted

```text
Measured:  accuracy, peak GPU memory (NVML), GPU energy (NVML power integration, valid pass
           only), wall-clock time, for 41 completed configurations on one Tesla T4.
Derived:   operational CO2e = energy_kwh x 0.65 (measured energy -> Table II column;
           predicted energy -> Table IV column). Never modelled, never called measured.
Predicted: every value in Table IV and Fig. 5; the y-axis of Fig. 4.
Excluded from primary figures (and why): the 1.5B LoRA cell (only 2 of 3 seeds). Primary
           Table II and Figs. 2-3 use the 13 complete cells (39 runs). The cell stays in the
           surrogate's training data (41 configurations).
Not measured / not claimed: embodied carbon, e-waste, hardware lifetime; directly measured
           CO2; cross-task or cross-hardware generalisation; a full methods x backbones x
           seeds grid; method rankings; LISA hyperparameter effects; out-of-sample
           recommendation validation (none recorded).
```

## Decisions I made that you should confirm

1. **Hugging Face DOI.** I used `10.57967/hf/10690` (GreenPEFT-v2, revision 148b027). It is
   the DOI in `CITATION.cff`, `DATA_METHODOLOGY.md` and the CLI banner, and it identifies the
   model the paper describes. `10.57967/hf/10504` is not used anywhere in the paper.
2. **Cross-validation recomputed on the 41 canonical configurations**, as the workflow
   recommends. I added a third protocol, seed-grouped (`GroupKFold` on method x backbone),
   because `config_id` contains the seed and so the pass-grouped protocol does not separate
   seeds. The recomputed values match `IMPLEMENTATION_REPORT.md`, for example VRAM
   leave-one-tier-out R2 0.59 and runtime 0.54. The canonical leave-one-tier-out values are
   the recomputed ones (VRAM 0.585, runtime 0.544), not the AUDIT_REPORT values (0.555,
   0.531). The status calls do not change: energy is VALIDATED, the other three targets are
   PRELIMINARY, and the overall status is PRELIMINARY.
3. **Venue.** None was named, so I used IEEEtran conference with IEEE numbered citations. If
   you have a target venue and page limit, tell me and I will reformat and cut. The draft is
   likely over 6 pages in two-column format.
4. **Scenario A-M** (`--min-confidence HIGH`) is an extra fourth scenario. It shows the
   recommendation when the search is restricted to measured cells, because A, B and C all
   rank unmeasured backbones first.
5. **Source tags** are LaTeX comments, so they stay out of the PDF but remain in the source.

## Unverified claims and missing files

- **Training-set size.** The cached dataset directory is named `classification_4000_872`,
  which suggests a 4,000-example training subset. The raw JSON confirms only `n_eval = 872`.
  Because I could not confirm the 4,000, the paper gives only the evaluation size. Add the
  training size if you can confirm it.
- **Energy sampling rate.** "Roughly 5 Hz" is inferred from 531 samples over 108.6 s in one
  run. The configured sampling interval is not recorded in the files I read.
- **70 W TDP for the T4.** Stated in `DATA_METHODOLOGY.md`, not checked against NVIDIA's
  datasheet.
- **Citations checked from memory**, because the publisher page was unreachable or I did not
  fetch it: Schwartz et al. (CACM 63(12), DOI 10.1145/3381831), Houlsby et al. (ICML 2019),
  Wolf et al. (EMNLP 2020 demos), PEFT library, scikit-learn. All others were checked against
  arXiv, the ACL Anthology, JMLR or the CodeCarbon repository.
- **LoRA-FA title.** The arXiv record now shows a revised title ("Efficient and Effective Low
  Rank Representation Fine-tuning", revised May 2026). The bib uses the original v1 title and
  notes the revision. Pick one.
- **Dropped related work.** I could not find Chen et al. (2024) or Kaur et al. (2026) in any
  supplied file, so they are not cited.
- **Affiliation.** Only "United International University" is given, taken from the CLI
  banner. Add your department, city and email.
- **Methodology figure.** `methodology.png` at the repo root still has the issues listed in
  workflow section 8 ("Ingest 4 sources", "RLAW DATA", the duplicated label, "zero-shot").
  The paper does not use it. Fig. 1 is a new, corrected pipeline figure.
- **Stale metadata note.** `models/model_metadata.json` still says "All metrics are
  leave-one-tier-out", which is false. The paper does not repeat it, but the shipped
  metadata should be corrected.
- **The CLI's error bands** come from the 82-row audit metrics, which are up to 2.3 MAPE
  points wider than the recomputed 41-configuration values. The paper says so in the RQ2
  subsection.

## Final audit (workflow section 16)

- Data: no duplicate rows; energy uses only `energy_measurement_valid == 1`; every plotted
  cell has 3 seeds and complete outcomes; the word OOM does not appear in `main.tex`.
- Model: leakage guard asserted in the build script; all protocols reported and labelled;
  overall status PRELIMINARY; carbon is never modelled; predicted values are labelled
  predicted.
- Claims: no e-waste or direct-CO2 claims; no claim of a complete grid; limitations section
  covers the required items.
- Citations: all 23 keys resolve and none are fabricated. The items listed above are
  unverified.
