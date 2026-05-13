# CPU image for Streamlit (`app.py`). PyTorch CPU wheels keep the image portable
# (no NVIDIA driver needed inside or outside the container for inference).
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Matches app.py default ``MODEL_DIR`` = ``/app/models`` (repo folder ``models/``, not ``/models``).
RUN mkdir -p models

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt /tmp/requirements-app.txt

RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r /tmp/requirements-app.txt

COPY app.py /app/app.py

# Deploy weights + vocab (required for HF Spaces; no host mount there).
# Local `docker compose` still bind-mounts `./models` over this path at runtime.
COPY models/ /app/models/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
