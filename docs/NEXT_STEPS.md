# Next Steps

Actionable continuation checklist for the Mortgage Collateral Risk DL project.
Read `CONTEXT.md` and `docs/PLAN.md` first — this file is the short-form "what to do next" derived from them, kept in git so any device/session can pick up work.

## Current state

- Notebook `mortgage_collateral_risk_dl.ipynb` executed **end-to-end successfully** (88/88 cells, zero errors).
- CPU baseline run complete (reduced epochs + subsampled data). All model artefacts in `models/`.
- Streamlit `app.py` loads trained models and produces predictions correctly.
- Active branch: `cursor/add-next-steps-doc-3ac2` (PR #2), base `feat/project-notebook` (PR #1).
- **Next priority**: re-run on the 4070 GPU with full data and proper epoch counts for production results.

## Next actions (in order)

### 1. Environment on the 4070 machine

```bash
git clone https://github.com/strodmens/mortgage-collateral-risk-dl.git
cd mortgage-collateral-risk-dl
git checkout feat/project-notebook   # or the latest branch with fixes
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Credentials / assets

```bash
export KAGGLE_API_TOKEN=<your-token>   # from kaggle.com/settings → API
cd data && wget https://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip glove.6B.100d.txt && rm glove.6B.zip
```

### 3. GPU-quality run of the notebook

Before running, adjust these settings in the notebook for GPU:
- Cell 3: change `NUM_WORKERS = 0` → `NUM_WORKERS = 4`
- Cell 31: the `MAX_TRAIN_SAMPLES` check auto-detects CUDA and uses the full dataset
- Cell 38: change `epochs=5, patience=3` → `epochs=30, patience=7`
- Cell 42: change `epochs=3, patience=2` → `epochs=15, patience=5`
- Cell 43: change `epochs=3, patience=2` → `epochs=20, patience=5`
- Cell 54: change `epochs=5, patience=3` → `epochs=30, patience=7`
- Cell 57: change `epochs=5, patience=3` → `epochs=30, patience=7`
- Cell 72: change `range(5)` → `range(30)` and `Epoch .../5` → `Epoch .../30`

Then run all cells top-to-bottom. Datasets auto-download via Kaggle CLI.

### 4. Resolved dataset questions

- SoCal: **15,474 images**, well-balanced 5 bands (~3k per band) — no need to collapse.
- London text: **996 samples** after cleaning, mean 252 words, all ≥15 words — good.
- Airbnb: no usable description column — joint model correctly pairs SoCal+London by band.

### 5. Business integration

- Apply `LOW` / `MEDIUM` / `HIGH` rule (≥2 bands below claimed band logic).
- Produce the required 20-example side-by-side table (CNN/RNN/truth/flag).
- Produce the comparison chart: CNN vs RNN vs combined-rule vs joint macro-F1.

### 6. Explainability

- Grad-CAM on CNN predictions.
- Attention or SHAP on RNN predictions.

### 7. Deployment

- `streamlit run app.py` locally (Docker available).
- Deploy to one of the available hosts:
  - **Hugging Face Spaces** — original plan, free, supports Streamlit directly.
  - **Render** (render.com) — Docker support, more RAM on paid tiers.
  - **Vercel** (vercel.com) — available, better suited for static front-ends.
  - **Custom domain** — can point at whichever host is chosen.
- If memory is tight on the free tier → swap ResNet-50 → MobileNet-V3 for the deployed artefact only.
- Record screen capture for submission.

### 8. Submission packaging

- Export notebook to PDF.
- Presentation deck + recording.
- Verify rubric sections 5.1–5.8 + bonus are each addressed in the notebook.

## Open decisions to revisit after the first real run

- 5 bands vs 4 bands (depends on SoCal per-band counts).
- Keep joint CNN backbone frozen vs partially unfreeze `layer4`.
- Final deployed CNN: ResNet-50 vs MobileNet-V3.
- RNN dataset: stick with London or fall back to Airbnb NYC.
