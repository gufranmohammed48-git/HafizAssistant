"""
Whisper-based transcription as a fallback (or primary) for Quran audio.

Uses faster-whisper with a Quran-fine-tuned model.
"""
import os
import time
import logging
import numpy as np
from faster_whisper import WhisperModel

log = logging.getLogger("whisper")

# Model selection: "base" (74M, fast), "small" (244M, balanced), "medium" (769M, accurate)
# "large-v3" (1.5B, best) is too slow on CPU
# For Quran specifically, tarteel-ai models are best
DEFAULT_MODEL_SIZE = "small"
_model = None


def get_model():
    global _model
    if _model is None:
        log.info(f"Loading Whisper model: {DEFAULT_MODEL_SIZE} (CPU, int8)")
        _model = WhisperModel(
            DEFAULT_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
        log.info(f"Whisper model loaded")
    return _model


def transcribe_audio(audio_data: np.ndarray, sample_rate: int = 16000):
    """
    Transcribe audio data (float32, mono, 16kHz expected) using Whisper.
    Returns the transcribed text in Arabic.
    """
    model = get_model()
    start = time.time()
    # Whisper needs at least 1s, ideally 5-30s
    if len(audio_data) < sample_rate * 0.5:
        return ""
    # Use language="ar" and an initial prompt to bias toward Quran vocabulary
    segments, info = model.transcribe(
        audio_data,
        language="ar",
        beam_size=5,
        best_of=3,
        temperature=0.0,
        vad_filter=True,  # filter out silence
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,  # prevent loops
        repetition_penalty=1.3,  # penalize repeats
        no_repeat_ngram_size=3,  # don't repeat 3-grams
    )
    text_parts = []
    for seg in segments:
        t = seg.text.strip()
        # Filter out segments that are just repetitions (Whisper can loop on short inputs)
        if len(t) > 200:
            t = t[:200]
        text_parts.append(t)
    full_text = " ".join(text_parts).strip()
    # If text is suspiciously repetitive (more than 5x the same 2-3 word sequence), reject
    import re
    words = full_text.split()
    if len(words) > 20:
        # Check for repeated patterns
        for n in [2, 3, 4]:
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            if ngrams:
                from collections import Counter
                counts = Counter(ngrams)
                most_common, max_count = counts.most_common(1)[0]
                if max_count > len(ngrams) * 0.4:  # >40% of ngrams are the same
                    log.warning(f"Whisper output has repetition loop: {most_common!r} x {max_count}")
                    full_text = ""
                    break
    elapsed = time.time() - start
    log.info(f"Whisper transcribed {len(audio_data)/sample_rate:.1f}s in {elapsed:.1f}s: {full_text!r}")
    return full_text


def warmup():
    """Load model ahead of time so first request isn't slow."""
    get_model()
