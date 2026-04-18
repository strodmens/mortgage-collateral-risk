# Project Context

Quick context file so any AI assistant (on any machine) can pick up where work left off. Read this plus `docs/PLAN.md` before doing anything.

## What this project is

Advanced Machine Learning course **final group project** (solo submission by Raivo Strods). Course brief is in `Final_Project_Advanced_ML.docx.pdf`. The task is to build a CNN (images) + RNN (text) pipeline in the same business domain, integrate their outputs at the business layer, and deploy a prototype.

## Decisions already made

- **Domain**: real estate / mortgage collateral underwriting (a continuation of the previous credit-risk project — see `previous_credit_risk_scoring_final_project.ipynb`).
- **Target**: 5 ordinal price bands (Budget / Economy / Mid-Range / Premium / Luxury) from quantile binning — deliberately mirrors the `loan_grade` target of the previous project.
- **Datasets**:
  - CNN: Kaggle `ted8080/house-prices-and-images-socal`
  - RNN: Kaggle `kanchana1990/real-estate-data-london-2024` (or fallback Airbnb NYC)
  - Joint-model paired subset: constructed by band-matching within the existing datasets
- **Framework**: PyTorch 2.x (CPU installed on the Mac, **CUDA build required on the 4070 machine** — see below).
- **CNN architectures**: from-scratch 4-block CNN + ResNet-50 transfer learning (2-phase: freeze → fine-tune `layer3`/`layer4`/`fc`).
- **RNN architectures**: single-layer LSTM with learned embeddings + bidirectional stacked 2-layer LSTM with GloVe-100d.
- **Joint model**: ResNet-50 GAP features (2048-d) + BiLSTM hidden state (256-d) → 2 FC layers → softmax (both backbones frozen).
- **Business rule**: `LOW` / `MEDIUM` / `HIGH` collateral risk flag based on how far each model's predicted band is below the applicant's claimed band.
- **Deployment**: Streamlit (`app.py`) targeting Hugging Face Spaces.

## Repo layout

```
.
├── Final_Project_Advanced_ML.docx.pdf       # Course brief (source of truth for grading)
├── previous_credit_risk_scoring_final_project.ipynb   # Prior project for reference
├── mortgage_collateral_risk_dl.ipynb        # Main deliverable notebook (88 cells)
├── app.py                                   # Streamlit deployment prototype
├── requirements.txt                         # Python dependencies
├── docs/
│   └── PLAN.md                              # Full implementation plan
└── CONTEXT.md                               # This file
```

Data and trained models are git-ignored — they live in `data/` and `models/` locally.

## Status at last handoff

All 12 planned todos are complete in the notebook. The notebook is a code-complete template but has **not yet been executed end-to-end with real data** — that's the next step on the 4070 machine.

## Next steps on the 4070 machine

```bash
git clone https://github.com/strodmens/mortgage-collateral-risk-dl.git
cd mortgage-collateral-risk-dl
git checkout feat/project-notebook
python -m venv .venv && .venv\Scripts\activate   # Windows; use source .venv/bin/activate on Linux
pip install -r requirements.txt
# Replace CPU torch with CUDA build:
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then:

1. Set up a Kaggle API token at `~/.kaggle/kaggle.json` (or `%USERPROFILE%\.kaggle\kaggle.json` on Windows).
2. Open `mortgage_collateral_risk_dl.ipynb` in Cursor/Jupyter and run cells top-to-bottom. The data-loading cell will auto-download the three Kaggle datasets.
3. Optional: download `glove.6B.100d.txt` from Stanford NLP into `data/` to enable the GloVe-backed BiLSTM variation.
4. After training completes, run `streamlit run app.py` to test the prototype locally, then deploy to Hugging Face Spaces.

## Open decisions / things to revisit after first real run

- Whether the SoCal dataset has enough samples per price band (if not, collapse to 4 bands).
- Whether the London text dataset descriptions are long enough after HTML stripping (target ≥15 words avg).
- Whether to keep the joint model's CNN backbone frozen or partially unfreeze once baseline numbers are in.
- Final choice of deployed CNN (ResNet-50 vs MobileNet-V3) based on HF Spaces memory.

## Useful references

- `docs/PLAN.md` — full plan with architecture diagram and rubric mapping.
- `Final_Project_Advanced_ML.docx.pdf` — official grading rubric (sections 5.1–5.8 + bonus).
- `previous_credit_risk_scoring_final_project.ipynb` — previous course project for framing consistency.
- PR #1 on GitHub: https://github.com/strodmens/mortgage-collateral-risk-dl/pull/1
