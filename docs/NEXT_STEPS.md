# Next Steps

Short continuation checklist. **`CONTEXT.md` is the live summary** of what is already done; use this file for rubric / submission tasks that may still be pending.

## Current state (`feat/project-notebook`)

- Notebook runs **end-to-end** on RTX 4070 with full data and training; checkpoints + deploy weights in `models/` (git-ignored locally).
- **Windows/Jupyter**: `NUM_WORKERS=0` in the notebook (avoids multiprocessing pickling issues). Linux devs may increase for faster I/O.
- **Docker**: `docker compose build && docker compose up` serves Streamlit on **8501** with `./models` bind-mounted.
- **Optional re-run without retraining**: set env `SKIP_TRAINING=1` before `jupyter nbconvert --execute` if `*_best.pt` files already exist.
- **Docs**: `docs/REFERENCES.md` lists datasets, papers, and external code links (content was aligned with the older `cursor/add-next-steps-doc-3ac2` branch; no separate merge needed for that file).

## Still worth doing (course / repo hygiene)

### 1. Git / default branch

- Merge **`feat/project-notebook` → `main`** on GitHub (or locally + push) so `main` is not stuck on the initial-only commit.
- After `main` is updated and you are satisfied, delete the stale remote branch **`cursor/add-next-steps-doc-3ac2`** if it is still listed on GitHub (its doc/runtime fixes were folded into `feat` via merge).

### 2. Submission artefacts

- Export `mortgage_collateral_risk_dl.ipynb` to **PDF** if the brief requires it.
- Presentation + recording if required.
- Double-check rubric sections **5.1–5.8 + bonus** are visibly covered in the notebook narrative.

### 3. Hosted demo (optional but strong)

- **Hugging Face Space** or **Render** using the existing `Dockerfile` + mounted or bundled small deploy weights (respect host RAM; consider a lighter CNN only if the free tier OOMs).

### 4. Nice-to-haves

- If you add new experiments, append a line to `CONTEXT.md` under test metrics so the next session knows the canonical numbers.
- Keep `docs/REFERENCES.md` in sync if you cite new papers in the notebook.

## Reference — environment quick start

```bash
git clone https://github.com/strodmens/mortgage-collateral-risk-dl.git
cd mortgage-collateral-risk-dl
git checkout feat/project-notebook   # until main is updated
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
# CUDA: pick the cu12x wheel index that matches your driver from https://pytorch.org/get-started/locally/
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Kaggle: `~/.kaggle/kaggle.json` (or Windows `%USERPROFILE%\.kaggle\kaggle.json`). GloVe: optional `data/glove.6B.100d.txt` from Stanford NLP.

## Resolved dataset notes (precise counts)

These complement `docs/REFERENCES.md` and match the cleaned pipelines in the notebook:

- **SoCal (CNN)**: **15,474** images; bands reasonably balanced (~3k per band) — no need to collapse to four bands for count reasons alone.
- **London (RNN)**: **1,019** raw listings → **~996** usable rows after cleaning; mean **~252** words after HTML strip; essentially all rows meet the ≥15-word filter.
- **Airbnb NYC**: downloaded for completeness; **no** rich description column for the RNN (`NAME` is very short). Joint modelling uses **band-matched** SoCal images + London text, which is the intended design.

## Approximate training wall-clock (RTX 4070)

Order-of-magnitude only — actual time depends on drivers, `NUM_WORKERS`, and early stopping. A first **full** local run with `NUM_WORKERS=0` (Windows) was on the order of **~1 hour** end-to-end including CNN + transfer + RNN + joint + explainability.

Rough cell-group sanity check (not a promise):

- CNN from scratch (many epochs, ~10k+ train images): tens of minutes.
- ResNet-50 transfer (two phases): tens of minutes combined.
- LSTMs (small text set): minutes.
- Joint + Grad-CAM / token explanations: remainder.

Re-run faster with `SKIP_TRAINING=1` when checkpoints already exist (see `CONTEXT.md`).

## Rubric cross-check (implemented in notebook — verify before submit)

Skim the notebook sections against the brief:

- **Business integration**: `LOW` / `MEDIUM` / `HIGH` rule, side-by-side examples, comparison of CNN vs RNN vs rule vs joint.
- **Explainability**: Grad-CAM (CNN) and token-level attribution (RNN).
- **Deployment path**: Streamlit + optional Docker; export/deploy weights (`cnn_transfer_deploy.pt`, `lstm_deploy.pt`, `vocab.json`).
