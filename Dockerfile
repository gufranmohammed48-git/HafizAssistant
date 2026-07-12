FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached separately from app code for faster rebuilds)
# Cython + build tools MUST come BEFORE nemo_toolkit (youtokentome needs Cython) because youtokentome
# (NeMo dep) needs to compile from source on Python 3.10.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir cython setuptools wheel && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt && \
    pip uninstall -y pyarrow && \
    pip install --no-cache-dir 'pyarrow==13.0.0'

# Copy app code
COPY app.py ./
COPY whisper_transcribe.py ./

# Download the FastConformer model at build time (cached in image).
# Production model: shahabazkc10/fastconformer-quran-bucket (4.13% WER,
# 0.93% on held-out unseen reciters, supports full Arabic voice diversity
# with diacritics). Replaces the older mohammed/fastconformer-quran-ar
# which was biased to Alafasy's voice.
# Override at runtime with MODEL_PATH env var.
ARG MODEL_REPO=shahabazkc10/fastconformer-quran-bucket
ARG MODEL_FILENAME=fastconformer-quran.nemo
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${MODEL_REPO}', filename='${MODEL_FILENAME}', local_dir='/data')"

# Default model path (can be overridden by MODEL_PATH env var at runtime)
ENV MODEL_PATH=/data/fastconformer-quran.nemo \
    PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

# Use shell form to expand $PORT (works for both Railway and docker-compose)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --log-level info"]
