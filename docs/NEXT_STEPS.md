# Next Steps

Short continuation checklist. **`CONTEXT.md` is the live summary** of what is already done; use this file for rubric / submission tasks that may still be pending.

## Current state (post-merge `feat/project-notebook`)

- Notebook runs **end-to-end** on RTX 4070 with full data and training; checkpoints + deploy weights in `models/` (git-ignored locally).
- **Windows/Jupyter**: `NUM_WORKERS=0` in the notebook (avoids multiprocessing issues). Linux devs may increase for faster I/O.
- **Docker**: `docker compose build && docker compose up` serves Streamlit on **8501** with `./models` bind-mounted.
- **Optional re-run without retraining**: set env `SKIP_TRAINING=1` before `jupyter nbconvert --execute` if `*_best.pt` files already exist.
- **Docs**: `docs/REFERENCES.md` lists datasets, papers, and external code links.

## Still worth doing (course / repo hygiene)

### 1. Git / default branch

- Merge **`feat/project-notebook` → `main`** on GitHub (or locally + push) so `main` is not stuck on the initial-only commit.
- Delete stale remote branch **`cursor/add-next-steps-doc-3ac2`** after confirming its unique commits are merged (this merge brought in `NEXT_STEPS.md`, `REFERENCES.md`, and an expanded `CONTEXT.md` without overwriting the notebook fixes).

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
git checkout main    # after merge; until then use feat/project-notebook
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
# CUDA: pick the cu12x wheel index that matches your driver from https://pytorch.org/get-started/locally/
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Kaggle: `~/.kaggle/kaggle.json` (or Windows `%USERPROFILE%\.kaggle\kaggle.json`). GloVe: optional `data/glove.6B.100d.txt` from Stanford NLP.

## Resolved dataset notes (from earlier analysis)

- SoCal: **~15k** images, reasonably spread across 5 bands.
- London: **~1k** listings, long descriptions after cleaning.
- Airbnb: weak text for RNN; joint pipeline uses **band-matched** SoCal + London pairs.
