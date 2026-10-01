# GreenPEFT showcase site

The public site for **GreenPEFT: A Multi-Objective Green AI Decision Support
Framework**. It is a static site — no build step, no framework, no runtime
dependencies — but it is not a brochure: the recommendation lab runs the real
decision engine against the real surrogate predictions.

## Structure

```text
website/
├── index.html              # the whole page
├── styles.css              # design system (forest / paper bands, components)
├── js/
│   ├── engine.js           # browser port of green_peft/recommender.py
│   ├── charts.js           # hand-rolled SVG chart primitives
│   └── app.js              # section rendering and interaction
├── data/
│   ├── candidates.js       # 70 surrogate-predicted candidates   (generated)
│   └── evidence.js         # measured benchmark + validation data (generated)
├── figures/                # PNGs copied from results/            (generated)
└── assets/author.jpg       # author portrait
```

## How the recommendation lab stays honest

The surrogate bundle is a pickled scikit-learn object, so the browser cannot run
inference. Instead the split is:

| Step | Where it runs |
| :--- | :--- |
| Surrogate inference over the 70-candidate catalogue | Python, ahead of time |
| Scope, confidence and error bands | Python, ahead of time |
| Plausibility drop, VRAM gate, accuracy / carbon / time filters | Browser |
| Pareto non-dominance test | Browser |
| GEI min–max normalisation and ranking | Browser |

Everything in the browser column is pure arithmetic ported one-to-one from
`green_peft/recommender.py`, so moving a slider produces the same answer the CLI
would. The site reproduces all six scenarios in
`results/recommendations/scenarios.json` exactly — feasible counts, implausible
counts, the top pick and its GEI to four decimal places.

**If `recommender.py` changes, `js/engine.js` has to change with it.**

## Regenerating the data layer

`data/*.js` and `figures/` are derived artifacts. Rebuild them with:

```bash
python analysis/build_website_data.py          # write
python analysis/build_website_data.py --check  # verify, write nothing
```

Both steps also run as part of `python reproduce.py`. Use an interpreter whose
scikit-learn matches the `sklearn_version` in `models/model_metadata.json`;
unpickling across versions is not trustworthy.

The data files are written as `window.GREENPEFT_* = {...}` assignments rather
than bare `.json`, so the page behaves identically when opened from a deployment
and from a local `file://` double-click.

## Deploying

### Vercel (current)

The repository root holds `vercel.json`, which publishes this directory with no
build command, plus `.vercelignore` so a deployment does not upload the
notebooks, datasets and virtualenv. Import the repository in Vercel and accept
the defaults; `outputDirectory` is already set to `website`.

### Netlify

The root `netlify.toml` publishes this directory too, and `website/netlify.toml`
works for a drag-and-drop of this folder on its own.

### Local preview

Open `index.html` directly, or serve the folder:

```bash
python -m http.server 8000 --directory website
```

## Conventions worth keeping

- **No fabricated numbers.** Every figure on the page traces back to a file in
  `results/` or `models/`. Illustrative values would have to be labelled as such;
  currently there are none.
- **Scope travels with the prediction.** A candidate outside the measured
  envelope is never shown with the same authority as a measured one.
- **Two bands, alternating.** Forest sections are instrument surfaces (hero, lab,
  pipeline, GEI, CLI); paper sections are for reading (problem, evidence, scope,
  research). Components read `--bg`, `--fg`, `--accent` and friends from the band,
  so the same markup works on either.
- **Motion is optional.** Everything animated respects
  `prefers-reduced-motion: reduce` and renders in its final state without it.
