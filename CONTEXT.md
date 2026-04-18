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
- **Framework**: PyTorch 2.x with CUDA on the RTX 4070 (use the PyTorch wheel that matches your driver, e.g. cu124).
- **CNN architectures**: from-scratch 4-block CNN + ResNet-50 transfer learning (2-phase: freeze → fine-tune `layer3`/`layer4`/`fc`).
- **RNN architectures**: single-layer LSTM with learned embeddings + bidirectional stacked 2-layer LSTM with GloVe-100d.
- **Joint model**: ResNet-50 GAP features (2048-d) + BiLSTM hidden state (256-d) → 2 FC layers → softmax (both backbones frozen).
- **Business rule**: `LOW` / `MEDIUM` / `HIGH` collateral risk flag based on how far each model's predicted band is below the applicant's claimed band.
- **Deployment**: Streamlit (`app.py`) plus **Docker** (`Dockerfile`, `docker-compose.yml`) — Hugging Face Spaces, Render, or similar (see hosting table below).

## Local hardware (primary development machine)

- **GPU**: NVIDIA GeForce RTX 4070 (CUDA)
- **CPU**: Intel Core i7-13700K
- **RAM**: 32 GB
- **Storage**: Samsung 980 NVMe 1 TB
- **Docker**: available locally (CPU image is default; GPU compose optional)

Note: Cursor Cloud Agents run on CPU-only VMs — full training is done locally. The notebook is tuned for **Windows + Jupyter** (`NUM_WORKERS=0` avoids multiprocessing pickling issues). On Linux you can raise `NUM_WORKERS` for faster loading.

## Hosting / deployment options

| Platform | Notes |
|----------|-------|
| **Vercel** (vercel.com) | Good for static front-ends; limited for ML model serving |
| **Render** (render.com) | Supports Docker; free tier is tight on RAM for ResNet-50 |
| **Custom domain** | Can CNAME to whichever host you choose |
| **Hugging Face Spaces** | Original plan — Streamlit + modest model sizes |

Pick a host that fits ResNet-50 + LSTM in memory; use Docker (`docker compose up`) for a reproducible Streamlit runtime. Weights live in `models/` on the host (git-ignored) and are bind-mounted into the container.

## Repo layout

```
.
├── Final_Project_Advanced_ML.docx.pdf       # Course brief (source of truth for grading)
├── previous_credit_risk_scoring_final_project.ipynb   # Prior project for reference
├── mortgage_collateral_risk_dl.ipynb        # Main deliverable notebook (88 cells)
├── app.py                                   # Streamlit deployment prototype
├── requirements.txt                         # Python dependencies (includes torch)
├── requirements-app.txt                     # Used by Dockerfiles (torch installed separately)
├── Dockerfile / Dockerfile.gpu              # CPU and GPU Streamlit images
├── docker-compose.yml / docker-compose.gpu.yml
├── docs/
│   ├── PLAN.md                              # Full implementation plan
│   ├── NEXT_STEPS.md                        # Short continuation checklist (may lag reality — verify)
│   └── REFERENCES.md                        # Papers, datasets, external links
└── CONTEXT.md                               # This file
```

Data and trained models are git-ignored — they live in `data/` and `models/` locally.

## Status at last handoff

- Notebook **executed end-to-end** on the RTX 4070 (88/88 cells). Checkpoints under `models/`; deploy artefacts: `cnn_transfer_deploy.pt`, `lstm_deploy.pt`, `vocab.json`.
- **Optional**: set `SKIP_TRAINING=1` when re-running via `nbconvert` to load existing `*_best.pt` checkpoints instead of retraining.
- Streamlit smoke-tested; Docker CPU image builds and serves on port 8501 with `./models` mounted.

### Reference test-set scores (full training run, RTX 4070)

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| CNN from scratch | 24.0% | 0.210 |
| ResNet-50 transfer | 51.8% | 0.518 |
| LSTM v1 (learned embeddings) | 20.7% | 0.206 |
| LSTM v2 (BiLSTM + GloVe) | 24.7% | 0.245 |
| Joint CNN + LSTM | 43.3% | 0.429 |

(Baseline chance for 5 balanced classes ≈ 20%.)

### Historical note — earlier CPU baseline (reduced epochs / subsample)

An earlier cloud CPU run (fewer epochs, ~2k images) produced lower numbers (~15–33% accuracy range). Those runs validated the pipeline; the table above reflects the **full** local GPU training.

## Quick start (clone → env → data)

```bash
git clone https://github.com/strodmens/mortgage-collateral-risk-dl.git
cd mortgage-collateral-risk-dl
git checkout feat/project-notebook   # or main after merge
python -m venv .venv && .venv\Scripts\activate   # Windows; use source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
# CUDA (pick index URL matching your driver / CUDA version from pytorch.org):
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

1. **Kaggle**: `~/.kaggle/kaggle.json` (Windows: `%USERPROFILE%\.kaggle\kaggle.json`) or use the Kaggle CLI after `kaggle login`.
2. Run `mortgage_collateral_risk_dl.ipynb` top-to-bottom (datasets auto-download when missing).
3. Optional: `data/glove.6B.100d.txt` from Stanford GloVe for the BiLSTM branch.
4. **Docker**: from repo root, `docker compose build && docker compose up` — requires `./models` with deploy weights (see above).

## Open decisions / things to revisit

- Whether to collapse from 5 → 4 price bands if a band stays too thin after filtering.
- Joint model: keep CNN frozen vs partially unfreeze `layer4` after baseline review.
- Deployed CNN: ResNet-50 vs lighter backbone if a host tier is memory-limited.

## Useful references

- `docs/PLAN.md` — full plan with architecture diagram and rubric mapping.
- `docs/NEXT_STEPS.md` — checklist (cross-check against this file; `CONTEXT.md` is the live summary).
- `docs/REFERENCES.md` — datasets, papers, external code links.
- `Final_Project_Advanced_ML.docx.pdf` — official grading rubric (sections 5.1–5.8 + bonus).
- `previous_credit_risk_scoring_final_project.ipynb` — previous course project for framing consistency.
- `academics/` — local PDF copies (git-ignored); filenames referenced in `docs/REFERENCES.md`.
- PR #1: https://github.com/strodmens/mortgage-collateral-risk-dl/pull/1
