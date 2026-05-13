"""Mortgage Collateral Risk Assistant — Streamlit Deployment."""
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Heavy imports (torch) are deferred until load_models() so Streamlit can bind
# port 8501 and satisfy HF health checks before PyTorch + ResNet allocate RAM.

# Default: ``models/`` next to this file → ``/app/models`` in the Docker image (WORKDIR /app).
_mdir_env = os.environ.get("MODEL_DIR", "").strip()
MODEL_DIR = Path(_mdir_env) if _mdir_env else Path(__file__).resolve().parent / "models"
REQUIRED_MODEL_FILES = ("vocab.json", "cnn_transfer_deploy.pt", "lstm_deploy.pt")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def _missing_model_files():
    return [f for f in REQUIRED_MODEL_FILES if not (MODEL_DIR / f).is_file()]


def ensure_model_artifacts():
    """Ensure ``models/`` exists and contains deploy weights.

    If files are missing and ``MODEL_ASSETS_REPO`` is set, download from the
    Hugging Face Hub (works well on Spaces). Optional env:

    - ``MODEL_ASSETS_REPO`` — e.g. ``yourname/mortgage-deploy-files``
    - ``MODEL_ASSETS_REPO_TYPE`` — ``model`` (default), ``dataset``, or ``space``
    - ``MODEL_ASSETS_PREFIX`` — path inside that repo, e.g. ``weights`` if files
      live at ``weights/vocab.json`` (no leading/trailing slashes required)
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    missing = _missing_model_files()
    if not missing:
        return None

    repo = os.environ.get("MODEL_ASSETS_REPO", "").strip()
    if repo:
        rtype = os.environ.get("MODEL_ASSETS_REPO_TYPE", "model").strip().lower()
        if rtype not in ("model", "dataset", "space"):
            rtype = "model"
        prefix = os.environ.get("MODEL_ASSETS_PREFIX", "").strip().strip("/")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            return (
                f"Missing {missing} and huggingface_hub is not installed: {e}. "
                "Add the files under `models/` in the Space repo."
            )
        try:
            for fname in REQUIRED_MODEL_FILES:
                if (MODEL_DIR / fname).is_file():
                    continue
                path_in_repo = f"{prefix}/{fname}" if prefix else fname
                cached = hf_hub_download(
                    repo_id=repo,
                    filename=path_in_repo,
                    repo_type=rtype,
                )
                shutil.copy2(cached, MODEL_DIR / fname)
        except Exception as e:
            return (
                f"Could not download model files from Hub repo `{repo}` "
                f"(type={rtype}): {e}"
            )

    missing = _missing_model_files()
    if missing:
        return (
            f"Missing files in `models/`: **{', '.join(missing)}**. "
            "Either commit them to this Space (use Git LFS for `.pt` files) "
            "or set Space secret **MODEL_ASSETS_REPO** to a Hub model/dataset "
            "repo that contains `vocab.json`, `cnn_transfer_deploy.pt`, and "
            "`lstm_deploy.pt` (optional **MODEL_ASSETS_REPO_TYPE** = `dataset` "
            "or `model`, optional **MODEL_ASSETS_PREFIX** if they sit in a subfolder)."
        )
    return None


@st.cache_resource
def load_models():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    import torchvision.models as models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class LSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                     num_layers=2, bidirectional=True, dropout=0.4):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=bidirectional,
                               dropout=dropout if num_layers > 1 else 0.0)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

        def forward(self, x):
            embedded = self.dropout(self.embedding(x))
            lengths = (x != 0).sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.clamp(min=1), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed)
            if self.bidirectional:
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                hidden = hidden[-1]
            return self.fc(self.dropout(hidden))

    with open(MODEL_DIR / "vocab.json") as f:
        config = json.load(f)

    cnn = models.resnet50(weights=None)
    cnn.fc = nn.Sequential(
        nn.Linear(2048, 256), nn.ReLU(True), nn.Dropout(0.4),
        nn.Linear(256, config["num_classes"]))
    cnn.load_state_dict(torch.load(MODEL_DIR / "cnn_transfer_deploy.pt",
                                   map_location=device, weights_only=True))
    cnn.eval().to(device)

    lstm = LSTMClassifier(
        len(config["word2idx"]), config["embed_dim"],
        config["hidden_dim"], config["num_classes"])
    lstm.load_state_dict(torch.load(MODEL_DIR / "lstm_deploy.pt",
                                    map_location=device, weights_only=True))
    lstm.eval().to(device)

    return cnn, lstm, config, device


def predict_image(cnn, img_pil, config, device):
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    tensor = transform(img_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = cnn(tensor)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    return pred, probs


def predict_text(lstm, text, config, device):
    import torch
    import torch.nn.functional as F

    word2idx = config["word2idx"]
    max_len = config["max_seq_len"]
    tokens = tokenize(text)[:max_len]
    indices = [word2idx.get(t, word2idx.get("<UNK>", 1)) for t in tokens]
    indices += [0] * (max_len - len(indices))
    tensor = torch.tensor([indices], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = lstm(tensor)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    return pred, probs


def risk_flag(img_band, txt_band, claimed_band):
    ig = claimed_band - img_band
    tg = claimed_band - txt_band
    if ig >= 2 and tg >= 2:
        return "HIGH"
    elif ig >= 2 or tg >= 2:
        return "MEDIUM"
    return "LOW"


st.set_page_config(page_title="Mortgage Collateral Risk", layout="wide")
st.title("Mortgage Collateral Risk Assistant")
st.markdown(
    "Upload a property photo and paste its listing description "
    "to assess collateral risk."
)

_setup_error = ensure_model_artifacts()
if _setup_error:
    st.error(_setup_error)
    st.stop()

with st.spinner("Loading PyTorch models (CPU; first load can take 1–2 minutes on small hardware)…"):
    cnn, lstm, config, _device = load_models()
bands = config["band_names"]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Property Photo")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    img_pred, img_probs = None, None
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True)
        img_pred, img_probs = predict_image(cnn, img, config, _device)
        st.metric("CNN Prediction", bands[img_pred],
                  f"{img_probs[img_pred]*100:.1f}% confidence")

with col2:
    st.subheader("Listing Description")
    text_input = st.text_area("Paste description", height=200)
    txt_pred, txt_probs = None, None
    if text_input.strip():
        txt_pred, txt_probs = predict_text(lstm, text_input, config, _device)
        st.metric("RNN Prediction", bands[txt_pred],
                  f"{txt_probs[txt_pred]*100:.1f}% confidence")

st.divider()
if img_pred is not None and txt_pred is not None:
    claimed = st.selectbox("Applicant's claimed value tier", bands, index=2)
    claimed_id = bands.index(claimed)
    flag = risk_flag(img_pred, txt_pred, claimed_id)
    colour = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[flag]
    st.markdown(f"### Collateral Risk: :{colour}[{flag}]")
    if flag == "LOW":
        st.success(
            "Both models support the claimed value. Auto-proceed recommended."
        )
    elif flag == "MEDIUM":
        st.warning(
            "One model diverges from the claimed value. "
            "Secondary review recommended."
        )
    else:
        st.error(
            "Both models suggest the collateral is over-valued. "
            "Manual appraisal required."
        )
