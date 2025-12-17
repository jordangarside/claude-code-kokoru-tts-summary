"""Ollama summarization backend."""

import logging

import httpx

from ..config import SummarizerConfig
from .base import SummarizerInterface, SummaryRequest, SummaryResult, SummaryType
from .prompts import get_prompt_and_params

log = logging.getLogger("tts-server")


class OllamaSummarizer(SummarizerInterface):
    """Ollama-based summarization backend for local LLM inference."""

    def __init__(self, config: SummarizerConfig):
        self.config = config
        self.base_url = config.ollama_url.rstrip("/")
        # Longer timeout for local inference which can be slower
        self.client = httpx.AsyncClient(timeout=60.0)

    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        """Summarize content using Ollama API.

        Args:
            request: The summarization request.

        Returns:
            SummaryResult with the summarized text.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        prompt, temperature, max_tokens = get_prompt_and_params(request.summary_type)
        model = self._get_model(request.summary_type)

        # Use Ollama's OpenAI-compatible endpoint
        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": request.content},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()

        # Check for error in response
        if "error" in data:
            raise ValueError(f"Ollama API error: {data['error'].get('message', 'Unknown error')}")

        text = data["choices"][0]["message"]["content"]
        tokens_used = data.get("usage", {}).get("total_tokens")

        return SummaryResult(
            text=text,
            model_used=model,
            tokens_used=tokens_used,
        )

    async def health_check(self) -> bool:
        """Check if Ollama is accessible.

        Returns:
            True if Ollama is accessible, False otherwise.
        """
        try:
            # Check if Ollama is running by hitting the tags endpoint
            response = await self.client.get(
                f"{self.base_url}/api/tags",
                timeout=2.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    def _get_model(self, summary_type: SummaryType) -> str:
        """Get the appropriate model for the summary type.

        Args:
            summary_type: The type of summary.

        Returns:
            The model name to use.
        """
        if summary_type == SummaryType.LONG_RESPONSE:
            return self.config.ollama_model_large
        else:
            return self.config.ollama_model_small

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
