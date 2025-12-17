"""Kokoro TTS backend implementation."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..config import TTSConfig
from ..core.logging import get_logger
from .base import TTSInterface

log = get_logger()


class KokoroTTS(TTSInterface):
    """Kokoro-82M text-to-speech backend."""

    SAMPLE_RATE = 24000
    REPO_ID = "hexgrad/Kokoro-82M"

    def __init__(self, config: TTSConfig):
        self.config = config
        self.pipeline = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def initialize(self) -> None:
        """Load the Kokoro model."""
        log.debug("Loading Kokoro model...")

        loop = asyncio.get_event_loop()

        def load_model():
            from kokoro import KPipeline
            return KPipeline(lang_code=self.config.kokoro_lang, repo_id=self.REPO_ID)

        self.pipeline = await loop.run_in_executor(self._executor, load_model)
        log.debug("Kokoro model loaded")

    async def synthesize(self, text: str) -> np.ndarray | None:
        """Generate TTS audio from text.

        Args:
            text: The text to synthesize.

        Returns:
            Audio as numpy array or None if synthesis failed.
        """
        import time

        if not self.pipeline:
            log.error("Kokoro pipeline not initialized")
            return None

        loop = asyncio.get_event_loop()

        def generate():
            all_audio = []
            for _, _, audio in self.pipeline(text, voice=self.config.kokoro_voice):
                all_audio.append(audio)
            if not all_audio:
                return None
            return np.concatenate(all_audio)

        try:
            start = time.perf_counter()
            audio = await loop.run_in_executor(self._executor, generate)
            elapsed = time.perf_counter() - start
            log.trace(f"Kokoro TTS synthesis: {elapsed:.3f}s")
            return audio
        except Exception as e:
            log.error(f"TTS generation failed: {e}")
            return None

    def get_sample_rate(self) -> int:
        """Return the sample rate (24kHz for Kokoro)."""
        return self.SAMPLE_RATE

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=False)
        self.pipeline = None
