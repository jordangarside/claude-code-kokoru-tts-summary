"""Abstract base class for TTS backends."""

from abc import ABC, abstractmethod

import numpy as np


class TTSInterface(ABC):
    """Abstract base class for text-to-speech backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the TTS backend (load models, etc.)."""
        pass

    @abstractmethod
    async def synthesize(self, text: str) -> np.ndarray | None:
        """Synthesize speech from text.

        Args:
            text: The text to synthesize.

        Returns:
            Audio as numpy array (float32, mono) or None if synthesis failed.
        """
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Return the sample rate of generated audio."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass
