"""Abstract base class for summarization backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto


class SummaryType(Enum):
    """Type of content being summarized."""

    SHORT_RESPONSE = auto()  # <300 chars, no tool calls
    LONG_RESPONSE = auto()  # Long or contains tool calls
    PERMISSION_REQUEST = auto()  # Permission announcement


@dataclass
class SummaryRequest:
    """Request for summarization."""

    content: str
    summary_type: SummaryType
    metadata: dict | None = None  # e.g., tool_name for permissions


@dataclass
class SummaryResult:
    """Result of summarization."""

    text: str
    model_used: str
    tokens_used: int | None = None


class SummarizerInterface(ABC):
    """Abstract base class for summarization backends."""

    @abstractmethod
    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        """Summarize content for TTS.

        Args:
            request: The summarization request.

        Returns:
            SummaryResult with the text suitable for TTS.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the summarizer backend is available."""
        pass
