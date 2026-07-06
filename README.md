# TabPFN — Seminar Deck

A single-file HTML slide deck explaining **TabPFN** (Hollmann, Müller, Eggensperger & Hutter — *arXiv:2207.01848*; TabPFN v2, *Nature 2025*), prepared for the seminar course
**วิทยาศาสตรบัณฑิต สาขาวิชา เทคโนโลยีปัญญาประดิษฐ์ (AIT)**.

Presenter: **คณิณ แคลวิณ ชมวิวัฒน์**

**Live deck:** https://tabfn-explained.vercel.app

## What it is

`tabpfn_seminar.html` is a self-contained, image-driven presentation (no build step, no
dependencies). Open it in any browser, or view the deployed version above.

**Controls:** `←` / `→` (or click left/right half of the screen) to navigate · `F` for
fullscreen · print to PDF to export.

## The 5 slides

| # | Slide | Point | Figures |
|---|-------|-------|---------|
| 1 | Title | The problem trees still win — what if you never train on your data? | `god.png` |
| 2 | The idea | Pretrained once, then a single forward pass replaces the whole AutoML pipeline | `simplicity.webp` |
| 3 | How it works | Two-way (feature + sample) attention over the table → a predicted distribution; trained once, inference = one forward pass | `architecture_layer.png`, `Attention_TabPFN.png` |
| 4 | Trained on synthetic worlds → behaves like a GP | Prior = synthetic structural-causal-model datasets (Occam's razor) → smooth, calibrated, fits any shape | SCM figure, `boundaries.webp`, `diverse_function.webp` |
| 5 | Results & limits | Matches ~1 h of AutoML tuning in ~0.6 s (230× CPU / 5,700× GPU); v1 limits: ≤1k rows, ≤100 features, ≤10 classes, numerical, classification | `fig_speed.png`, `fig_rank.png` |

## Files

| File | Description |
|------|-------------|
| `tabpfn_seminar.html` | The complete slide deck (HTML + CSS + navigation JS). |
| `vercel.json` | Rewrites `/` → `tabpfn_seminar.html` for the Vercel deployment. |
| `god.png` | Title-slide image. |
| `simplicity.webp` | AutoML pipeline vs a single TabPFN forward pass (via Max Kuhn, R/Pharma 2025). |
| `architecture_layer.png` | Full TabPFN v2 architecture: encoders → 12× {feature attention · sample attention · MLP} → 5,000-bin head → distribution. |
| `Attention_TabPFN.png` | The 2D attention figure: feature attention (columns) + sample attention (rows). |
| `Structural-causal-model-...webp` | Structural causal model (SCM) that generates the synthetic training data (the prior). |
| `boundaries.webp` | Decision boundaries across methods — TabPFN is smooth/GP-like vs blocky trees (Hollmann et al., Fig. 4). |
| `diverse_function.webp` | TabPFN fitting a broad class of functions (sin, x², \|x\|, step, noise). |
| `fig_speed.png` | Accuracy vs. compute budget — TabPFN sits top-left. |
| `fig_rank.png` | Mean rank across 18 numerical OpenML-CC18 datasets (lower = better). |
| `Gaussian.png`, `PFN_Transformer.png` | Extra reference figures (not currently placed in the 5-slide cut). |

## Deploying

The deck is deployed as a static site on Vercel. To redeploy after edits:

```bash
vercel deploy --prod --yes
```

## Sources

- Hollmann, Müller, Eggensperger & Hutter, *TabPFN: A Transformer That Solves Small
  Tabular Classification Problems in a Second* — arXiv:2207.01848 (ICLR 2023).
- Hollmann et al., TabPFN v2 — *Nature*, 2025.
- Max Kuhn, *TabPFN: A Deep-Learning Solution for Tabular Data* — R/Pharma 2025.
