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

# Monkey-patch json.dumps to handle numpy types globally. NeMo's transcribe() internally
# writes a manifest file with float32 timestamps and crashes without this.
_orig_json_dumps = json.dumps
def _safe_dumps(obj, **kwargs):
    kwargs.setdefault('cls', NumpyJSONEncoder)
    return _orig_json_dumps(obj, **kwargs)
json.dumps = _safe_dumps


class NumpyJSONEncoder(json.JSONEncoder):
    """Handle numpy types that the default JSON encoder rejects."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.str_,)):
            return str(obj)
        return super().default(obj)


async def _safe_send_json(ws, data):
    """Send JSON via WebSocket, with numpy type coercion."""
    return await ws.send_text(json.dumps(data, cls=NumpyJSONEncoder, ensure_ascii=False))

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
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL_PATH, map_location="cpu")
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

        # Only transcribe the LAST 2 seconds (most recent audio) to keep latency low
        # and avoid transcribing all the silence before speech started.
        last_two_sec = SAMPLE_RATE * 2
        start = max(0, self.buffer_len - last_two_sec)
        log.info(f"process_chunk: buffer_len={self.buffer_len}, using last {len(audio_int16) if False else (self.buffer_len-start)} samples")
        # For TRUE streaming we'd use the RNN-T greedy decoder with cache, but the
        # easier path is: each chunk is the rolling buffer, get full hypothesis,
        # diff against committed_text to extract new words.
        try:
            # Slice the active audio (last 2 seconds)
            audio_active = self.buffer[start:self.buffer_len].copy()
            # NeMo's hybrid FastConformer transcribe() only accepts a list of file
            # paths. Write the audio to a temp WAV and pass that.
            import tempfile, soundfile as sf
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, audio_active, SAMPLE_RATE)
            hyp = asr_model.transcribe(
                [tmp.name],
                return_hypotheses=True,
            )
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
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
                log.info(f"PARTIAL: {text!r} (new: {new_text!r})")
                await _safe_send_json(self.websocket, {
                    "type": "partial",
                    "text": new_text,
                    "full_text": text,
                })
                self.last_result_text = text
            else:
                log.info(f"NO_NEW_TEXT: full={text!r}")
        except Exception as e:
            log.exception(f"Streaming error: {e}")
            await _safe_send_json(self.websocket, {
                "type": "error",
                "message": str(e),
            })

    async def commit(self):
        """Mark current text as committed."""
        self.committed_text = self.last_result_text
        await _safe_send_json(self.websocket, {
            "type": "committed",
            "text": self.committed_text,
        })

    async def finalize(self):
        """Run final transcription on full buffer."""
        try:
            audio_active = self.buffer[:self.buffer_len].copy()
            if len(audio_active) < SAMPLE_RATE * 0.3:  # < 300ms, skip
                return
            import tempfile, soundfile as sf
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, audio_active, SAMPLE_RATE)
            hyp = asr_model.transcribe([tmp.name], return_hypotheses=True)
            try: os.unlink(tmp.name)
            except Exception: pass
            if isinstance(hyp, tuple):
                hyp = hyp[0]
            if isinstance(hyp, list):
                hyp = hyp[0] if hyp else None
            if hyp is None:
                return
            text = (hyp.text or "").strip()
            # Try to get word-level timestamps (coerce numpy types to native Python)
            words = []
            if hasattr(hyp, "timestamp") and hyp.timestamp:
                def _coerce(v):
                    # numpy float32/float64/int64 → Python float/int
                    if hasattr(v, "item"):
                        try: return v.item()
                        except Exception: return float(v)
                    return v
                for w, s, e in hyp.timestamp.get("word", []):
                    words.append({"word": _coerce(w), "start": _coerce(s), "end": _coerce(e)})
            # Also coerce text in case the model returns a numpy string
            text_str = str(text) if text is not None else ""
            await _safe_send_json(self.websocket, {
                "type": "final",
                "text": text_str,
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
                        await _safe_send_json(websocket, {"type": "reset"})
                except json.JSONDecodeError:
                    log.warning(f"Bad JSON: {msg['text']}")
    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as e:
        log.exception(f"WebSocket error: {e}")
