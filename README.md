# Mortgage Collateral Risk Assistant

A Streamlit app that assesses mortgage collateral risk using two deep-learning models:

- **ResNet-50 (CNN)** — classifies a property photo into a value tier
- **Bidirectional LSTM (RNN)** — classifies a listing description into a value tier

When both models disagree with the applicant's claimed value tier, the app flags the loan as **HIGH**, **MEDIUM**, or **LOW** risk.

---

## Live demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]((https://mortgage-collateral-risk.streamlit.app/))

---

## Deploy to Streamlit Community Cloud

### 1. Fork or clone this repo (must be public)

### 2. Host the model weights on Hugging Face Hub

The three required files are too large for a plain GitHub push. Upload them to a Hugging Face **model** or **dataset** repo:

| File | Size |
|------|------|
| `vocab.json` | ~80 KB |
| `lstm_deploy.pt` | ~4.3 MB |
| `cnn_transfer_deploy.pt` | ~92 MB |

```bash
# Install the Hub CLI once
pip install huggingface_hub

# Login and upload (replace YOUR_HF_REPO with e.g. yourname/mortgage-weights)
huggingface-cli login
huggingface-cli upload YOUR_HF_REPO vocab.json
huggingface-cli upload YOUR_HF_REPO lstm_deploy.pt
huggingface-cli upload YOUR_HF_REPO cnn_transfer_deploy.pt
```

### 3. Add a Streamlit secret

In **Streamlit Cloud → App settings → Secrets**, add:

```toml
MODEL_ASSETS_REPO = "yourname/mortgage-weights"
# Optional: MODEL_ASSETS_REPO_TYPE = "dataset"   (default is "model")
# Optional: MODEL_ASSETS_PREFIX  = "weights"      (if files are in a subfolder)
```

The app reads this secret as an environment variable and downloads the weights on first startup using `huggingface_hub`.

### 4. Deploy

Point Streamlit Cloud at:
- **Repository**: `strodmens/mortgage-collateral-risk-dl`
- **Branch**: `main`
- **Main file**: `app.py`

---

## Local development

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# Place model files in models/
# models/vocab.json
# models/cnn_transfer_deploy.pt
# models/lstm_deploy.pt

streamlit run app.py
```

---

## Resource requirements

| Resource | Minimum |
|----------|---------|
| RAM | 1 GB (tight on free Streamlit tier; 2 GB recommended) |
| Disk | ~700 MB for CPU PyTorch install |
| GPU | Not required — CPU inference only |

> **Note:** The Streamlit Community Cloud free tier (1 GB RAM) is borderline.
> If the app crashes on load, upgrade to the **Teams** tier or host it in a Docker
> container (see `Dockerfile`).

---

## Architecture

```
app.py
├── ensure_model_artifacts()   Downloads weights from HF Hub if MODEL_ASSETS_REPO is set
├── load_models()              Loads ResNet-50 + BiLSTM with @st.cache_resource
├── predict_image()            CNN inference on uploaded photo
├── predict_text()             LSTM inference on listing description
└── risk_flag()                Combines predictions vs. claimed tier → HIGH/MEDIUM/LOW
```

## Files

```
app.py                  Streamlit app (single file)
requirements.txt        Python dependencies (CPU-only PyTorch)
Dockerfile              Docker deployment (CPU)
models/                 ← git-ignored; place weights here for local runs
```
