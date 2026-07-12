# FastConformer Quran Arabic ASR — Backend

Real-time streaming Arabic Quran speech recognition using
[shahabazkc10/fastconformer-quran-bucket](https://huggingface.co/shahabazkc10/fastconformer-quran-bucket)
(4.13% WER overall, **0.93% WER on held-out unseen reciters**, includes full
diacritics in output, supports diverse Arabic voices).

This is the backend for [Hifzapp](https://github.com/gufranmohammed48-git/Hifzapp)
— a live Quran recitation tracker that highlights words as you recite.

## Quick start (local)

```bash
# Build + run
docker compose up --build

# Logs
docker compose logs -f fastconformer

# Health check
curl http://127.0.0.1:8080/healthz
# → {"status":"ok","model":"fastconformer-quran-ar"}

# WebSocket endpoint
ws://127.0.0.1:8080/ws
```

The model (~459MB) is downloaded into a named volume on first build, so subsequent
restarts don't re-download.

## Production (DigitalOcean / any VPS)

This image runs fine on a $24/mo DO droplet (4GB/2vCPU/nyc3). The repo
includes the model baked in at build time, so cold-start time is just the
~30s model load, not the ~5min download.

For HTTPS + WebSocket termination, front it with nginx (see
`/workspace/fastconformer-deploy/nginx.conf` for a working config that
includes self-signed cert generation).

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/healthz` | Health check (used by Railway/Render/DO healthchecks) |
| GET | `/readyz` | Readiness — returns 200 only after model is loaded |
| WS | `/ws` | Streaming audio + partial transcripts |
| GET | `/api/debug-audio` | Download saved audio chunks as zip (debug) |

## WebSocket protocol

**Send audio**: raw 16-bit PCM, mono, 16kHz, little-endian binary frames.

**Send control** (JSON text frames):
- `{"type": "commit"}` — finalize current utterance
- `{"type": "finalize"}` — transcribe full buffer
- `{"type": "reset"}` — clear audio buffer

**Receive** (JSON text frames):
- `{"type": "partial", "text": "...", "full_text": "..."}` — incremental words
- `{"type": "committed", "text": "..."}` — committed text
- `{"type": "final", "text": "...", "words": [...]}` — final with word timestamps
- `{"type": "error", "message": "..."}` — error

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `MODEL_PATH` | `/data/fastconformer-quran.nemo` | Override the model file |
| `PORT` | `8080` | Listening port (Railway/Compose auto-set) |
| `PYTHONUNBUFFERED` | `1` | Force flush stdout for log streaming |

## Build args (Dockerfile)

| Arg | Default | Purpose |
|-----|---------|---------|
| `MODEL_REPO` | `shahabazkc10/fastconformer-quran-bucket` | HuggingFace repo to download from |
| `MODEL_FILENAME` | `fastconformer-quran.nemo` | Filename inside that repo |

## Files

```
fastconformer_hf/
├── Dockerfile             # Python 3.10-slim, builds nemo_toolkit, downloads model
├── docker-compose.yml     # Local dev + production deploy
├── .dockerignore          # Excludes dev files from build context
├── .gitignore             # Standard Python ignore
├── requirements.txt       # Pinned Python deps
├── app.py                 # FastAPI + WebSocket + ASR model
├── whisper_transcribe.py  # Optional Whisper fallback (kept for reference)
├── railway.toml           # Railway deploy config (legacy, unused — DO is current)
├── railway.json           # Railway schema (legacy, unused)
└── README.md              # This file
```
