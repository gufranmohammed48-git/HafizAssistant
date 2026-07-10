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
    pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Copy app code
COPY app.py .

# Download the FastConformer model at build time (cached in image)
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='mohammed/fastconformer-quran-ar', filename='phase3_full_finetune/phase3_full_finetune_wer0.1432.nemo', local_dir='/data')"

# Set the model path
ENV MODEL_PATH=/data/phase3_full_finetune/phase3_full_finetune_wer0.1432.nemo
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Railway sets $PORT. Default to 8080 if not set.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --log-level info"]
