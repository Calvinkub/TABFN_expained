# TabPFN — Seminar Deck & Explainer

A single-file HTML slide deck explaining **TabPFN**, prepared for the seminar course
**วิทยาศาสตรบัณฑิต สาขาวิชา เทคโนโลยีปัญญาประดิษฐ์ (AIT)**.

- **Presenter:** คณิณ แคลวิณ ชมวิวัฒน์
- **Live deck:** https://tabfn-explained.vercel.app
- **Papers:** Hollmann, Müller, Eggensperger & Hutter — *arXiv:2207.01848* (ICLR 2023, "v1")
  and the extended **TabPFN v2** — *Nature*, 2025.

This README doubles as a written explainer: the slide deck is the short version, everything
below is the detail.

---

## 1. The problem

Tabular data (spreadsheets, databases) is the most common data type in real machine
learning. Yet for years, **gradient-boosted decision trees** — XGBoost, LightGBM, CatBoost —
kept beating deep learning on it, and they still win many Kaggle competitions. The usual ML
recipe is also slow: for every new dataset you *train a fresh model* and then *tune
hyperparameters* with cross-validation.

**TabPFN asks a different question:** what if you never train a model on your data at all?

---

## 2. The core idea — in-context learning for tables

TabPFN is a **single Transformer that is pretrained once and then frozen forever**. To use it
you do *not* train:

1. You hand it your **entire training table** (features `X_train` + labels `y_train`) **and**
   the **test rows** you want predicted — all as **one input sequence** (a "prompt").
2. It reads the labeled examples and predicts the test labels in **one forward pass**.
3. **No gradient steps, no weight updates** happen on your data.

This is **in-context learning** — the same trick that lets a large language model "learn" a
task from examples in its prompt — applied to tables. The scikit-learn-style `.fit()` call
does **not** train; it just stores your rows to feed in as context. The learning already
happened, once, during pretraining.

> **"So is it actually trained?"** Yes — but there are **two separate phases** people conflate:
>
> | Phase | When | What happens |
> |-------|------|--------------|
> | **① Pre-training** | Once, by the authors | Real **gradient descent** on millions of synthetic datasets → the frozen weights you download |
> | **② Inference (your use)** | Every prediction | **One forward pass**, your table as the prompt — no weight updates |

---

## 3. How it was trained — Prior-Data Fitted Networks (PFNs)

TabPFN is a **Prior-Data Fitted Network (PFN)**: a network trained *once, offline* to
**approximate Bayesian inference**. Instead of being trained on real data, it is trained on a
huge stream of **synthetic datasets** drawn from a carefully designed **prior**.

**The prior** mixes two families of data-generating mechanisms (each dataset is sampled from
one or the other, roughly 50/50):

- **Structural Causal Models (SCMs)**
- **Bayesian Neural Networks (BNNs)**

### The SCM data-generating pipeline (per synthetic dataset)

1. **Sample a causal graph (a DAG).** The prior *prefers simple structures* — fewer nodes get
   higher probability (a built-in **Occam's razor**, e.g. a log-scaled-uniform distribution
   over the number of nodes).
2. **Parameterize each node** as a mechanism `zᵢ = fᵢ(parents, εᵢ)` — a small function with
   its own randomly sampled weights and a noise term `εᵢ`.
3. **Sample the noise and propagate** the values through the graph (cause → effect) to compute
   a value at every node.
4. **Pick columns:** choose some nodes to be the **features `X`** and one node to be the
   **target `y`**.
5. **Emit one labeled dataset**, then repeat with a brand-new random graph.

### The training objective

A 12-layer Transformer is trained on **~18,000 batches × 512 synthetic datasets each**
(about 20 hours on 8 GPUs) to minimize:

```
L = E[ −log q_θ( y_test | x_test, D_train ) ]
```

i.e. it learns to output the **posterior predictive distribution** of the target given the
labeled context. Because it has seen an enormous space of possible data-generating
mechanisms, it learns a general rule for "given these examples, what's the label?" — with
predictions that are **smooth, calibrated, and biased toward simple explanations**, and that
need **zero hyperparameter tuning**.

---

## 4. The architecture (TabPFN v2)

```
Input table ("prompt")
   │   X features encoder     y targets encoder
   ▼            ▼                    ▼
        192-d embeddings
   ▼
 12 ×  ┌───────────────────────────────────┐
       │ LayerNorm → Feature attention      │   (across columns → interactions)
       │ LayerNorm → Sample attention       │   (across rows → similar cases)
       │ LayerNorm → MLP                    │   (+ residual connections)
       └───────────────────────────────────┘
   ▼
 Linear 192-d → GELU 768-d → Linear 5000-d
   ▼
 5000 range-bins → predicted distribution over y
```

**Two-way ("2D") attention is the key upgrade in v2:**

- **Feature attention** runs *across columns* within a row → captures **interactions** and
  spline-like effects between predictors.
- **Sample attention** runs *across rows* within a column → compares a point to **similar
  training rows** (e.g. serial/temporal correlation, look-alike records). Crucially, **test
  rows attend only to training rows, never to each other** (an attention mask), which prevents
  leakage and is what makes it in-context learning.

The final MLP head turns the test vector into a **full predicted distribution** (a piece-wise
constant / Riemann distribution over 5,000 bins), not just a single point estimate.

> **v1 vs v2:** v1 (ICLR 2023) used one token per row and attended only across rows. v2
> (Nature 2025) adds the second axis (feature attention) → true 2D attention, and extends the
> method to larger data and regression.

**Permutation invariance:** there is **no positional encoding** over the training rows, so the
model treats them as an unordered *set* — reordering the rows does not change the prediction.

---

## 5. How it behaves — like a Gaussian Process

Because it approximates Bayesian inference with a simplicity-biased prior, TabPFN behaves much
like a **Gaussian Process (GP)**:

- **Smooth, calibrated decision boundaries** — where trees produce blocky, axis-aligned
  regions, TabPFN's boundaries are smooth (see `boundaries.webp`, comparable to the GP column).
- **Fits a very broad class of functions** — sin(x)+x, x², |x|, step functions, and both
  homoscedastic and heteroscedastic noise, where linear models and even CatBoost struggle
  (see `diverse_function.webp`).
- Returns a **full predictive distribution** with sensible uncertainty, not just a point.

---

## 6. Results

- **Matches ~1 hour of AutoML tuning in ~0.6 seconds.**
- **230× faster on CPU, 5,700× faster on GPU** than tuned baselines, at equal or better
  accuracy (ROC AUC) than XGBoost, CatBoost, Auto-sklearn, and AutoGluon.
- **Best mean rank** across 18 numerical OpenML-CC18 datasets (`fig_rank.png`), sitting
  top-left on the accuracy-vs-compute plot (`fig_speed.png`).
- Its errors are **uncorrelated with trees**, so **ensembling TabPFN + AutoGluon** wins
  outright.

---

## 7. Limitations (v1) — and *why*

| Limitation | Why it exists |
|------------|---------------|
| **Small data only — ≤ 1,000 training rows** | Attention is O(n²) over the whole training set, and the prior only ever saw *small* synthetic datasets — it can't extrapolate past that size. |
| **≤ 100 features, ≤ 10 classes** | The input uses a fixed-size encoding trained on bounded dimensions. |
| **Weak on categorical / missing values** | The prior is built mostly from *numerical* SCM / BNN mechanisms. |
| **Classification only, no regression** | v1 is trained to output class posteriors, not a continuous target. |
| **GPU strongly preferred** | Practical inference speed relies on CUDA hardware. |

**TabPFN v2 (Nature 2025)** lifts several of these — larger datasets, regression, and the 2D
attention architecture described above.

**When to reach for it:** small tabular problems where you want a strong, tuning-free baseline
in seconds — a great first move in a Kaggle tabular competition, and a strong ensemble member.

---

## 8. The deck

`tabpfn_seminar.html` is a **self-contained** presentation — plain HTML + CSS + a few lines of
navigation JavaScript, no build step and no dependencies. Open it in any browser or view the
deployed version.

**Controls:** `←` / `→` (or click the left/right half of the screen) to navigate ·
`F` for fullscreen · print to PDF to export.

### The 5 slides

| # | Slide | Point | Figures |
|---|-------|-------|---------|
| 1 | Title | Trees still win — what if you never train on your data? | `god.png` |
| 2 | The idea | Pretrained once, then one forward pass replaces the whole AutoML pipeline | `simplicity.webp` |
| 3 | How it works | Two-way (feature + sample) attention → a predicted distribution; trained once, inference = one pass | `architecture_layer.png`, `Attention_TabPFN.png` |
| 4 | Trained on synthetic worlds → GP-like behaviour | SCM prior (Occam) → smooth, calibrated, fits any shape | SCM figure, `boundaries.webp`, `diverse_function.webp` |
| 5 | Results & limits | Matches ~1 h of AutoML in ~0.6 s (230×/5,700×); v1 limits and v2 | `fig_speed.png`, `fig_rank.png` |

---

## 9. Files

| File | Description |
|------|-------------|
| `tabpfn_seminar.html` | The complete slide deck (HTML + CSS + navigation JS). |
| `vercel.json` | Rewrites `/` → `tabpfn_seminar.html` for the Vercel deployment. |
| `god.png` | Title-slide image. |
| `simplicity.webp` | AutoML pipeline vs a single TabPFN forward pass (via Max Kuhn, R/Pharma 2025). |
| `architecture_layer.png` | Full TabPFN v2 architecture (encoders → 12× {feature attn · sample attn · MLP} → 5,000-bin head → distribution). |
| `Attention_TabPFN.png` | The 2D attention figure: feature attention (columns) + sample attention (rows). |
| `Structural-causal-model-…​.webp` | The structural causal model (SCM) that generates the synthetic training data. |
| `boundaries.webp` | Decision boundaries across methods — TabPFN smooth/GP-like vs blocky trees (Hollmann et al., Fig. 4). |
| `diverse_function.webp` | TabPFN fitting a broad class of functions (sin, x², \|x\|, step, noise). |
| `fig_speed.png` | Accuracy vs. compute budget — TabPFN top-left. |
| `fig_rank.png` | Mean rank across 18 numerical OpenML-CC18 datasets (lower = better). |
| `Gaussian.png`, `PFN_Transformer.png` | Extra reference figures (not placed in the current 5-slide cut). |

---

## 10. Deploying

The deck is a static site on Vercel. To redeploy after edits:

```bash
vercel deploy --prod --yes
```

The root URL serves the deck via the `vercel.json` rewrite.

---

## 11. Sources

- Hollmann, Müller, Eggensperger & Hutter, *TabPFN: A Transformer That Solves Small Tabular
  Classification Problems in a Second* — **arXiv:2207.01848** (ICLR 2023).
- Hollmann et al., **TabPFN v2** — *Nature*, 2025.
- Max Kuhn, *TabPFN: A Deep-Learning Solution for Tabular Data* — **R/Pharma 2025**
  ([talk](https://opensource.posit.co/resources/videos/2025-12-17_max-kuhn-tabpfn-a-deep-learning-solution-for-tabular-data/)).
