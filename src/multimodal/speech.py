"""
Speech I/O Pipeline
====================
Whisper STT + TTS (Coqui/Bark) for voice-operable agent interface.

ML System Design decisions documented inline.

Design decision: speech as a translation layer, not a separate path.
Voice input resolves to a text query string before hitting the planner.
The agent graph is identical for text and voice — no special casing.
This keeps the architecture clean and testable.

Usage:
    processor = SpeechProcessor()
    transcript = processor.transcribe("query.wav")
    # transcript.text → drop into agent as query string
"""

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TranscriptResult:
    text: str                  # clean transcript → becomes agent query
    language: str              # detected language (e.g. "en", "es")
    confidence: float          # avg log-prob — proxy for transcription quality
    duration_seconds: float    # audio duration


class SpeechProcessor:
    """
    Whisper STT + TTS pipeline.

    Design decision: use openai/whisper (open source) over Whisper API.
    - No per-call cost at development time
    - Full control over model size (tiny → large-v3)
    - Runs locally on GPU/CPU — no network dependency

    Model size tradeoff:
        whisper-tiny   : ~39M params, fastest, lowest accuracy
        whisper-base   : ~74M params, good for English
        whisper-small  : ~244M params, good multilingual
        whisper-large-v3: ~1.5B params, best accuracy, slowest

    For our latency budget (<500ms): whisper-base on GPU, whisper-tiny on CPU.

    Design decision: resample to 16kHz mono before Whisper.
    Whisper was trained on 16kHz mono audio. Feeding 44.1kHz stereo:
    - ~2.75x more samples than expected → slower processing
    - Two channels → Whisper expects one
    - Speech frequencies all below 8kHz → above that is wasted data
    Always match inference-time input to model training distribution.
    """

    TARGET_SR = 16000   # Whisper's expected sample rate
    TARGET_CHANNELS = 1  # mono

    def __init__(self, model_size: str = "base", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self._model = None   # lazy load — don't load until first use
        self._tts = None

    def _load_whisper(self):
        if self._model is None:
            import whisper
            print(f"Loading Whisper {self.model_size} on {self.device}...")
            self._model = whisper.load_model(self.model_size, device=self.device)

    def _resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        Resample audio to TARGET_SR (16kHz) mono.

        Design decision: use scipy for resampling (no extra deps).
        librosa is more accurate but adds a heavy dependency.
        For speech resampling, scipy is sufficient.
        """
        from scipy.signal import resample_poly
        from math import gcd

        # Convert stereo to mono by averaging channels
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if orig_sr != self.TARGET_SR:
            g = gcd(orig_sr, self.TARGET_SR)
            audio = resample_poly(audio, self.TARGET_SR // g, orig_sr // g)

        return audio.astype(np.float32)

    def load_audio(self, path: str) -> tuple[np.ndarray, int]:
        """
        Load audio file and return (samples, sample_rate).
        Supports: wav, mp3, m4a, flac, ogg.
        """
        try:
            import soundfile as sf
            audio, sr = sf.read(path)
        except Exception:
            # Fallback to whisper's built-in audio loader
            import whisper
            audio = whisper.load_audio(path)
            sr = self.TARGET_SR  # whisper.load_audio already resamples

        return audio, sr

    def transcribe(self, audio_path: str) -> TranscriptResult:
        """
        Transcribe audio file to text using Whisper.

        Returns TranscriptResult with text, language, confidence, duration.
        The .text field drops directly into the agent as a query string.
        """
        self._load_whisper()

        audio, orig_sr = self.load_audio(audio_path)

        # Resample to 16kHz mono
        audio_16k = self._resample(audio, orig_sr)
        duration = len(audio_16k) / self.TARGET_SR

        # Transcribe
        import whisper
        result = self._model.transcribe(
            audio_16k,
            language=None,          # auto-detect language
            task="transcribe",      # not translate
            fp16=(self.device == "cuda"),
            verbose=False,
        )

        # Compute confidence from segment log probs
        segments = result.get("segments", [])
        avg_logprob = (
            np.mean([s.get("avg_logprob", 0) for s in segments])
            if segments else 0.0
        )
        # Convert log prob to [0,1] confidence proxy
        confidence = float(np.exp(avg_logprob))

        return TranscriptResult(
            text=result["text"].strip(),
            language=result.get("language", "en"),
            confidence=confidence,
            duration_seconds=duration,
        )

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> TranscriptResult:
        """Transcribe from raw bytes (e.g. from a web upload)."""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            return self.transcribe(tmp_path)
        finally:
            os.unlink(tmp_path)

    def synthesize(self, text: str, out_path: str = None) -> bytes:
        """
        Convert text to speech using Coqui TTS.

        Design decision: Coqui TTS over Bark.
        Coqui: faster, lower memory, good quality for short utterances.
        Bark: more expressive, supports emotions/music, much slower.
        For recommendation responses (2-3 sentences), Coqui is sufficient.

        Returns audio bytes (wav format).
        """
        try:
            from TTS.api import TTS
            if self._tts is None:
                self._tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name

            self._tts.tts_to_file(text=text, file_path=tmp_path)

            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()

            if out_path:
                Path(out_path).write_bytes(audio_bytes)

            os.unlink(tmp_path)
            return audio_bytes

        except ImportError:
            print("TTS not installed. Run: pip install TTS")
            return b""

    def process_voice_query(self, audio_path: str) -> dict:
        """
        Full voice query pipeline: audio → transcript → agent-ready dict.

        Returns dict compatible with agent invocation:
            {query: str, language: str, is_voice: bool}
        """
        transcript = self.transcribe(audio_path)

        # Low confidence warning — may want to ask user to repeat
        if transcript.confidence < 0.3:
            print(f"  ⚠ Low transcription confidence: {transcript.confidence:.2f}")
            print(f"  Transcript: '{transcript.text}'")

        return {
            "query": transcript.text,
            "language": transcript.language,
            "is_voice": True,
            "confidence": transcript.confidence,
        }