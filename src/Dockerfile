# Web API container for the University Chatbot.
#
# The build context is the repository root (config/ lives outside src/):
#
#   gcloud builds submit . \
#     --tag us-central1-docker.pkg.dev/PROJECT/REPO/web-api:latest \
#     --dockerfile src/Dockerfile
#
# gcloud run deploy cannot use `--source ./src` for this app because the
# FastAPI code imports the top-level `config` package, which would not be
# part of a src-only build context.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache

# libgomp1 (OpenMP) is required by PyTorch at import time
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first, so sentence-transformers does not pull the
# multi-gigabyte CUDA wheels from PyPI
COPY src/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Cache the embedding model inside the image to eliminate cold-start
# downloads. config/ is copied first so the cached model always matches
# EMBEDDING_MODEL in the application settings.
COPY config ./config
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from config.settings import settings; from sentence_transformers import SentenceTransformer; SentenceTransformer(settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE)"

COPY src ./src

# The model is baked into the image — never look it up over the network
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=8080

EXPOSE 8080

# Cloud Run injects PORT (default 8080)
CMD exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}
