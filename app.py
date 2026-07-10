"""
FastConformer Quran Arabic ASR — WebSocket streaming server
Uses the mohammed/fastconformer-quran-ar model from Hugging Face.
Designed for Hugging Face Spaces (Docker SDK).
"""
import os
import json
import asyncio
import logging
import numpy as np
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import nemo.collections.asr as nemo_asr

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("fastconformer-quran")

app = FastAPI(title="FastConformer Quran ASR")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup. Path is set by Dockerfile (downloads from HF).
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/data/phase3_full_finetune_wer0.1432.nemo"
)

log.info(f"Loading FastConformer model from {MODEL_PATH} ...")
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPE_Model.restore_from(MODEL_PATH, map_location="cpu")
asr_model.eval()
# Caching for streaming (cache-aware local attention)
asr_model.change_attention_model("rel_pos_local_attn", [256, 256])
asr_model.change_subsampling_conv_chunking_factor(1)
log.info("Model loaded. Ready for streaming inference.")

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"

# Each WebSocket session maintains a streaming state
class StreamingSession:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.buffer = np.zeros(SAMPLE_RATE * 60, dtype=np.float32)  # 60s rolling buffer
        self.buffer_len = 0
        self.committed_text = ""
        # For RNN-T streaming we feed pre-encoded chunks
        self.sample_offset = 0
        self.last_result_text = ""

    async def process_chunk(self, audio_int16: np.ndarray):
        """Take raw PCM int16 audio, run streaming inference, send word events."""
        # Normalize int16 → float32 [-1, 1]
        audio = audio_int16.astype(np.float32) / 32768.0

        # Append to rolling buffer
        if self.buffer_len + len(audio) > len(self.buffer):
            # Drop oldest half to keep buffer bounded
            shift = len(self.buffer) // 2
            self.buffer[:self.buffer_len - shift] = self.buffer[shift:self.buffer_len]
            self.buffer_len -= shift
            self.sample_offset += shift
        self.buffer[self.buffer_len:self.buffer_len + len(audio)] = audio
        self.buffer_len += len(audio)

        # Transcribe using NeMo's transcribe (offline on each chunk for simplicity)
        # For TRUE streaming we'd use the RNN-T greedy decoder with cache, but the
        # easier path is: each chunk is the rolling buffer, get full hypothesis,
        # diff against committed_text to extract new words.
        try:
            # Slice the active audio
            audio_active = self.buffer[:self.buffer_len].copy()
            # Use NeMo's transcribe on a numpy array — returns (text, ...)
            hyp = asr_model.transcribe(
                audio_active,
                return_hypotheses=True,
            )
            if isinstance(hyp, tuple):
                hyp = hyp[0]
            if isinstance(hyp, list):
                hyp = hyp[0] if hyp else None
            if hyp is None:
                return
            text = hyp.text if hasattr(hyp, "text") else str(hyp)
            text = (text or "").strip()
            if not text:
                return
            # Extract new words beyond what we've already committed
            new_text = text
            if self.committed_text and text.startswith(self.committed_text):
                new_text = text[len(self.committed_text):].lstrip()
            elif self.committed_text:
                # Try to find common prefix (in case of small variations)
                common = 0
                for a, b in zip(self.committed_text, text):
                    if a == b:
                        common += 1
                    else:
                        break
                new_text = text[common:].lstrip()
            if new_text:
                # Send partial word(s) as a JSON message
                await self.websocket.send_json({
                    "type": "partial",
                    "text": new_text,
                    "full_text": text,
                })
                self.last_result_text = text
        except Exception as e:
            log.exception(f"Streaming error: {e}")
            await self.websocket.send_json({
                "type": "error",
                "message": str(e),
            })

    async def commit(self):
        """Mark current text as committed."""
        self.committed_text = self.last_result_text
        await self.websocket.send_json({
            "type": "committed",
            "text": self.committed_text,
        })

    async def finalize(self):
        """Run final transcription on full buffer."""
        try:
            audio_active = self.buffer[:self.buffer_len].copy()
            if len(audio_active) < SAMPLE_RATE * 0.3:  # < 300ms, skip
                return
            hyp = asr_model.transcribe(audio_active, return_hypotheses=True)
            if isinstance(hyp, tuple):
                hyp = hyp[0]
            if isinstance(hyp, list):
                hyp = hyp[0] if hyp else None
            if hyp is None:
                return
            text = (hyp.text or "").strip()
            # Try to get word-level timestamps
            words = []
            if hasattr(hyp, "timestamp") and hyp.timestamp:
                words = [{"word": w, "start": s, "end": e}
                         for w, s, e in hyp.timestamp.get("word", [])]
            await self.websocket.send_json({
                "type": "final",
                "text": text,
                "words": words,
            })
        except Exception as e:
            log.exception(f"Finalize error: {e}")


@app.get("/")
async def root():
    return HTMLResponse("<h1>FastConformer Quran ASR</h1><p>WebSocket endpoint: /ws</p>")


@app.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok", "model": "fastconformer-quran-ar"})


@app.websocket("/ws")
async def ws_transcribe(websocket: WebSocket):
    await websocket.accept()
    session = StreamingSession(websocket)
    log.info("WebSocket connected")
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg:
                # Audio chunk: raw PCM int16 mono at 16kHz
                raw = msg["bytes"]
                audio = np.frombuffer(raw, dtype=np.int16)
                await session.process_chunk(audio)
            elif "text" in msg:
                # Control message
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "commit":
                        await session.commit()
                    elif data.get("type") == "finalize":
                        await session.finalize()
                    elif data.get("type") == "reset":
                        session.buffer_len = 0
                        session.sample_offset = 0
                        session.committed_text = ""
                        session.last_result_text = ""
                        await websocket.send_json({"type": "reset"})
                except json.JSONDecodeError:
                    log.warning(f"Bad JSON: {msg['text']}")
    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as e:
        log.exception(f"WebSocket error: {e}")
