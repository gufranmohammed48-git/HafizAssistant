# FastConformer Quran Arabic ASR — Backend

Real-time streaming Arabic Quran speech recognition using
[mohammed/fastconformer-quran-ar](https://huggingface.co/mohammed/fastconformer-quran-ar)
(0.14% WER on tarteel-ai/everyayah validation).

## Endpoints

- `GET /healthz` — health check
- `WS  /ws` — WebSocket streaming endpoint

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
