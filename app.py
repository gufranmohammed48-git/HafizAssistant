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
        self.chunk_count = 0

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
        self.chunk_count += 1

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
            # DEBUG: save named copy AFTER writing (for inspecting what model is getting)
            try:
                import shutil
                debug_path = f"/tmp/debug_chunks/chunk_{self.chunk_count:04d}.wav"
                os.makedirs("/tmp/debug_chunks", exist_ok=True)
                shutil.copy(tmp.name, debug_path)
                if self.chunk_count < 5 or self.chunk_count % 10 == 0:
                    log.info(f"DEBUG_SAVED: {debug_path} audio_len={len(audio_active)}")
            except Exception as e:
                log.warning(f"DEBUG_SAVE_FAILED: {e}")
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


@app.get("/api/debug-audio")
async def debug_audio():
    """Return the LAST 10 saved debug chunks as a zip. ~640KB max."""
    import re, zipfile, io
    from fastapi.responses import StreamingResponse
    if not os.path.isdir("/tmp/debug_chunks"):
        return JSONResponse({"error": "no chunks saved yet"}, status_code=404)
    # CRITICAL: sort by NUMERIC chunk number, not alphabetically!
    # "chunk_0001" < "chunk_0010" lexically, but 1 < 10 numerically.
    # Default lex sort gave chunk_0136..0145 (oldest) instead of chunk_0001..0010 (newest).
    files = sorted(
        [f for f in os.listdir("/tmp/debug_chunks") if f.endswith(".wav")],
        key=lambda f: int(re.search(r"chunk_(\d+)", f).group(1))
    )
    if not files:
        return JSONResponse({"error": "debug dir is empty"}, status_code=404)
    files = files[-10:]  # last 10 only
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for f in files:
            zf.write(os.path.join("/tmp/debug_chunks", f), f)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=debug_last_{len(files)}.zip"},
    )



class WhisperSession:
    """Accumulates 3s of audio and transcribes with Whisper (handles all Arabic voices)."""
    def __init__(self, websocket):
        self.websocket = websocket
        self.audio_buffer = []  # list of int16 arrays
        self.total_samples = 0
        self.sample_rate = 16000
        self.min_chunk_samples = self.sample_rate * 3  # 3 seconds minimum
        self.transcribe_count = 0

    async def process_chunk(self, audio_int16: np.ndarray):
        self.audio_buffer.append(audio_int16.copy())
        self.total_samples += len(audio_int16)
        # Once we have 3s, transcribe and reset
        if self.total_samples >= self.min_chunk_samples:
            await self._transcribe_buffer()

    async def _transcribe_buffer(self):
        if not self.audio_buffer:
            return
        # Concatenate all chunks
        full_audio = np.concatenate(self.audio_buffer)
        # Convert to float32 normalized
        audio_float = full_audio.astype(np.float32) / 32768.0
        self.audio_buffer = []
        self.total_samples = 0
        self.transcribe_count += 1

        try:
            # Send "transcribing" status
            await _safe_send_json(self.websocket, {
                "type": "transcribing",
                "buffer_seconds": len(audio_float) / self.sample_rate,
            })
            # Run whisper in thread pool to not block
            import whisper_transcribe
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None, whisper_transcribe.transcribe_audio, audio_float, self.sample_rate
            )
            if text and text.strip():
                log.info(f"WHISPER_RESPONSE #{self.transcribe_count}: {text!r}")
                await _safe_send_json(self.websocket, {
                    "type": "partial",
                    "text": text,
                    "full_text": text,
                    "model": "whisper",
                })
            else:
                log.info(f"WHISPER_EMPTY #{self.transcribe_count}")
                await _safe_send_json(self.websocket, {
                    "type": "empty",
                    "model": "whisper",
                })
        except Exception as e:
            log.exception(f"Whisper error: {e}")
            await _safe_send_json(self.websocket, {"type": "error", "message": str(e)})

    async def finalize(self):
        # Transcribe any remaining audio
        if self.total_samples > 0:
            await self._transcribe_buffer()


@app.websocket("/ws/whisper")
async def ws_whisper(websocket: WebSocket):
    """Whisper-based transcription (handles all Arabic reciters, slower than FastConformer)."""
    await websocket.accept()
    # Warm up the model on first connection
    import whisper_transcribe
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, whisper_transcribe.warmup)
    except Exception as e:
        log.warning(f"Whisper warmup failed: {e}")
    session = WhisperSession(websocket)
    log.info("Whisper WebSocket connected")
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg:
                raw = msg["bytes"]
                audio = np.frombuffer(raw, dtype=np.int16)
                await session.process_chunk(audio)
            elif "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "finalize":
                        await session.finalize()
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        await session.finalize()
        log.info("Whisper WebSocket closed")


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
