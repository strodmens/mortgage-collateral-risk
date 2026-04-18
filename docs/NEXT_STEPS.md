# Next Steps

Short continuation checklist. **`CONTEXT.md` is the live summary** of what is already done; use this file for rubric / submission tasks that may still be pending.

## Current state (repository)

- **Branch:** `main` carries the full project (same tip as `feat/project-notebook` after merge). Use `main` for clones and grading.
- Notebook runs **end-to-end** on RTX 4070 with full data and training; checkpoints + deploy weights live under `models/` (git-ignored).
- **Windows/Jupyter:** `NUM_WORKERS=0` in the notebook (avoids multiprocessing pickling issues). Linux devs may increase for faster I/O.
- **Docker:** `docker compose build && docker compose up` serves Streamlit on **8501** with `./models` bind-mounted.
- **Optional re-run without retraining:** set `SKIP_TRAINING=1` before `jupyter nbconvert --execute` if `*_best.pt` files already exist.
- **Docs:** `docs/REFERENCES.md` lists datasets, papers, and external links.

## Done (recent housekeeping)

- [x] Merge feature work into **`main`** and push to GitHub.
- [x] Remove stale remote branch **`cursor/add-next-steps-doc-3ac2`**.

## Still worth doing (course / submission)

### 1. Submission artefacts

- Export **`mortgage_collateral_risk_dl.ipynb`** to **PDF** (or HTML print-to-PDF) if the brief requires a non-ipynb hand-in.
- **Presentation deck** + **recording** if the course asks for them.
- Walk **`Final_Project_Advanced_ML.docx.pdf`** rubric **5.1–5.8 + bonus** against notebook headings and markdown cells — tick each item explicitly in the notebook or cover it in the deck so markers do not have to hunt.

### 2. Hosted demo (optional but strong for marks / portfolio)

- **Hugging Face Space** or **Render** using the repo `Dockerfile`.
- **Constraint:** weights are not in git — either (a) document “upload `cnn_transfer_deploy.pt`, `lstm_deploy.pt`, `vocab.json` to Space secrets / files” in a Space README, or (b) use a Git LFS / release asset if the host supports it.
- If the free tier **OOMs** on ResNet-50, deploy a lighter CNN for inference only and say so in the write-up (keep full results in the notebook).

### 3. Repo polish (optional, improves first impression)

- Add a **root `README.md`**: one-screen clone → venv → `pip install -r requirements.txt` → CUDA note → `streamlit run app.py` / `docker compose up` → where to put Kaggle + model files.
- Confirm **GitHub default branch** is **`main`** (Settings → General → Default branch).
- **`feat/project-notebook`:** keep or delete the remote branch once you are sure you only work from `main`.

### 4. Nice-to-haves

- After any new training run, update **`CONTEXT.md`** test metrics so “canonical numbers” stay accurate.
- If you cite new papers in the notebook, append them to **`docs/REFERENCES.md`**.

## Reference — environment quick start

```bash
git clone https://github.com/strodmens/mortgage-collateral-risk-dl.git
cd mortgage-collateral-risk-dl
git checkout main
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
# CUDA: pick the cu12x wheel index that matches your driver from https://pytorch.org/get-started/locally/
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Kaggle: `~/.kaggle/kaggle.json` (or Windows `%USERPROFILE%\.kaggle\kaggle.json`). GloVe: optional `data/glove.6B.100d.txt` from Stanford NLP.

## Resolved dataset notes (precise counts)

These complement `docs/REFERENCES.md` and match the cleaned pipelines in the notebook:

- **SoCal (CNN):** **15,474** images; bands reasonably balanced (~3k per band) — no need to collapse to four bands for count reasons alone.
- **London (RNN):** **1,019** raw listings → **~996** usable rows after cleaning; mean **~252** words after HTML strip; essentially all rows meet the ≥15-word filter.
- **Airbnb NYC:** downloaded for completeness; **no** rich description column for the RNN (`NAME` is very short). Joint modelling uses **band-matched** SoCal images + London text, which is the intended design.

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

- **Business integration:** `LOW` / `MEDIUM` / `HIGH` rule, side-by-side examples, comparison of CNN vs RNN vs rule vs joint.
- **Explainability:** Grad-CAM (CNN) and token-level attribution (RNN).
- **Deployment path:** Streamlit + optional Docker; export/deploy weights (`cnn_transfer_deploy.pt`, `lstm_deploy.pt`, `vocab.json`).
- **Ethics / limitations:** notebook §12 already discusses bias and limitations — re-read once before submit so wording still matches your final models and claims.
