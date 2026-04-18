# Next Steps

Actionable continuation checklist for the Mortgage Collateral Risk DL project.
Read `CONTEXT.md` and `docs/PLAN.md` first — this file is the short-form "what to do next" derived from them, kept in git so any device/session can pick up work.

## Current state

- Notebook `mortgage_collateral_risk_dl.ipynb` is code-complete (all 12 planned todos done).
- It has **not** yet been executed end-to-end on real data.
- Active branch at time of writing: `feat/project-notebook`, PR #1 open.

## Next actions (in order)

### 1. Environment on the 4070 machine

- Clone repo, checkout `feat/project-notebook`.
- Create venv, `pip install -r requirements.txt`.
- Replace CPU torch with CUDA build:

```bash
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Credentials / assets

- Kaggle API token at `~/.kaggle/kaggle.json` (or `%USERPROFILE%\.kaggle\kaggle.json` on Windows).
- (Optional) Download `glove.6B.100d.txt` into `data/` to enable the GloVe BiLSTM variation.

### 3. First real run of the notebook (top-to-bottom)

- Auto-downloads: SoCal house images, London real-estate text, Airbnb NYC fallback.
- Watch for dataset issues flagged in the plan:
  - SoCal: enough samples per price band? If not → collapse 5 bands → 4.
  - London text: avg description ≥15 words after HTML stripping?

### 4. Modeling passes

- CNN: from-scratch 4-block → ResNet-50 transfer (freeze → fine-tune `layer3`/`layer4`/`fc`).
- RNN: single-layer LSTM (learned embeddings) → BiLSTM 2-layer + GloVe-100d.
- Joint model: ResNet-50 GAP (2048-d) + BiLSTM hidden (256-d) → 2 FC → softmax, both backbones frozen initially.

### 5. Business integration

- Apply `LOW` / `MEDIUM` / `HIGH` rule (≥2 bands below claimed band logic).
- Produce the required 20-example side-by-side table (CNN/RNN/truth/flag).
- Produce the comparison chart: CNN vs RNN vs combined-rule vs joint macro-F1.

### 6. Explainability

- Grad-CAM on CNN predictions.
- Attention or SHAP on RNN predictions.

### 7. Deployment

- `streamlit run app.py` locally.
- Deploy to Hugging Face Spaces.
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
