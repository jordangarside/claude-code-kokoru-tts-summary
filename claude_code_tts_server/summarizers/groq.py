"""Groq summarization backend."""

import logging

import httpx

from ..config import SummarizerConfig
from .base import SummarizerInterface, SummaryRequest, SummaryResult, SummaryType
from .prompts import get_prompt_and_params

log = logging.getLogger("tts-server")


class GroqSummarizer(SummarizerInterface):
    """Groq-based summarization backend."""

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, config: SummarizerConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=10.0)

    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        """Summarize content using Groq API.

        Args:
            request: The summarization request.

        Returns:
            SummaryResult with the summarized text.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
            ValueError: If the API key is not configured.
        """
        if not self.config.groq_api_key:
            raise ValueError("SUMMARY_GROQ_API_KEY not configured")

        prompt, temperature, max_tokens = get_prompt_and_params(request.summary_type)
        model = self._get_model(request.summary_type)

        response = await self.client.post(
            self.BASE_URL,
            headers={"Authorization": f"Bearer {self.config.groq_api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": request.content},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()

        # Check for API error in response
        if "error" in data:
            raise ValueError(f"Groq API error: {data['error'].get('message', 'Unknown error')}")

        text = data["choices"][0]["message"]["content"]
        tokens_used = data.get("usage", {}).get("total_tokens")

        return SummaryResult(
            text=text,
            model_used=model,
            tokens_used=tokens_used,
        )

    async def health_check(self) -> bool:
        """Check if Groq API is accessible.

        Returns:
            True if the API is accessible, False otherwise.
        """
        if not self.config.groq_api_key:
            return False

        try:
            # Just check if we can reach the API (models endpoint is lightweight)
            response = await self.client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {self.config.groq_api_key}"},
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
            return self.config.groq_model_large
        else:
            return self.config.groq_model_small

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
