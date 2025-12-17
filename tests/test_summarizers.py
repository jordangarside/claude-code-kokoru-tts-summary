"""Tests for summarizer backends."""

import pytest
from pytest_httpx import HTTPXMock

from claude_code_tts_server.config import SummarizerConfig
from claude_code_tts_server.summarizers.base import SummaryRequest, SummaryType
from claude_code_tts_server.summarizers.groq import GroqSummarizer
from claude_code_tts_server.summarizers.ollama import OllamaSummarizer
from claude_code_tts_server.summarizers.prompts import (
    PROMPT_LONG_RESPONSE,
    PROMPT_PERMISSION_REQUEST,
    PROMPT_SHORT_RESPONSE,
    get_prompt_and_params,
)


class TestPrompts:
    """Tests for prompt selection."""

    def test_short_response_prompt(self):
        """Test short response prompt selection."""
        prompt, temp, max_tokens = get_prompt_and_params(SummaryType.SHORT_RESPONSE)
        assert prompt == PROMPT_SHORT_RESPONSE
        assert temp == 0.3
        assert max_tokens == 2048

    def test_long_response_prompt(self):
        """Test long response prompt selection."""
        prompt, temp, max_tokens = get_prompt_and_params(SummaryType.LONG_RESPONSE)
        assert prompt == PROMPT_LONG_RESPONSE
        assert temp == 0.3
        assert max_tokens == 2048

    def test_permission_request_prompt(self):
        """Test permission request prompt selection."""
        prompt, temp, max_tokens = get_prompt_and_params(SummaryType.PERMISSION_REQUEST)
        assert prompt == PROMPT_PERMISSION_REQUEST
        assert temp == 0.1
        assert max_tokens == 50


class TestGroqSummarizer:
    """Tests for GroqSummarizer."""

    @pytest.fixture
    def summarizer(self, summarizer_config):
        """Create a GroqSummarizer instance."""
        return GroqSummarizer(summarizer_config)

    @pytest.mark.asyncio
    async def test_summarize_short_response(self, summarizer, httpx_mock: HTTPXMock):
        """Test summarizing a short response."""
        httpx_mock.add_response(
            url="https://api.groq.com/openai/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Cleaned text"}}],
                "usage": {"total_tokens": 50},
            },
        )

        request = SummaryRequest(
            content="Hello world",
            summary_type=SummaryType.SHORT_RESPONSE,
        )
        result = await summarizer.summarize(request)

        assert result.text == "Cleaned text"
        assert result.model_used == "test-model-small"
        assert result.tokens_used == 50

    @pytest.mark.asyncio
    async def test_summarize_long_response(self, summarizer, httpx_mock: HTTPXMock):
        """Test summarizing a long response."""
        httpx_mock.add_response(
            url="https://api.groq.com/openai/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "I updated the files."}}],
                "usage": {"total_tokens": 100},
            },
        )

        request = SummaryRequest(
            content="A very long response " * 50,
            summary_type=SummaryType.LONG_RESPONSE,
        )
        result = await summarizer.summarize(request)

        assert result.text == "I updated the files."
        assert result.model_used == "test-model-large"

    @pytest.mark.asyncio
    async def test_summarize_permission_request(self, summarizer, httpx_mock: HTTPXMock):
        """Test summarizing a permission request."""
        httpx_mock.add_response(
            url="https://api.groq.com/openai/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Permission requested: Run tests"}}],
            },
        )

        request = SummaryRequest(
            content="Tool: Bash. Input: {\"command\": \"pytest\"}",
            summary_type=SummaryType.PERMISSION_REQUEST,
            metadata={"tool_name": "Bash"},
        )
        result = await summarizer.summarize(request)

        assert result.text == "Permission requested: Run tests"
        assert result.model_used == "test-model-small"

    @pytest.mark.asyncio
    async def test_summarize_no_api_key(self):
        """Test that missing API key raises error."""
        config = SummarizerConfig(groq_api_key=None)
        summarizer = GroqSummarizer(config)

        request = SummaryRequest(
            content="Test",
            summary_type=SummaryType.SHORT_RESPONSE,
        )

        with pytest.raises(ValueError, match="not configured"):
            await summarizer.summarize(request)

    @pytest.mark.asyncio
    async def test_health_check_success(self, summarizer, httpx_mock: HTTPXMock):
        """Test successful health check."""
        httpx_mock.add_response(
            url="https://api.groq.com/openai/v1/models",
            status_code=200,
        )

        result = await summarizer.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, summarizer, httpx_mock: HTTPXMock):
        """Test failed health check."""
        httpx_mock.add_response(
            url="https://api.groq.com/openai/v1/models",
            status_code=401,
        )

        result = await summarizer.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_no_api_key(self):
        """Test health check with no API key."""
        config = SummarizerConfig(groq_api_key=None)
        summarizer = GroqSummarizer(config)

        result = await summarizer.health_check()
        assert result is False


class TestOllamaSummarizer:
    """Tests for OllamaSummarizer."""

    @pytest.fixture
    def ollama_config(self):
        """Create Ollama config."""
        return SummarizerConfig(
            backend="ollama",
            ollama_url="http://localhost:11434",
            ollama_model_large="qwen3:4b-instruct-2507-q4_K_M",
            ollama_model_small="qwen3:4b-instruct-2507-q4_K_M",
        )

    @pytest.fixture
    def summarizer(self, ollama_config):
        """Create an OllamaSummarizer instance."""
        return OllamaSummarizer(ollama_config)

    @pytest.mark.asyncio
    async def test_summarize_short_response(self, summarizer, httpx_mock: HTTPXMock):
        """Test summarizing a short response."""
        httpx_mock.add_response(
            url="http://localhost:11434/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Cleaned text"}}],
                "usage": {"total_tokens": 50},
            },
        )

        request = SummaryRequest(
            content="Hello world",
            summary_type=SummaryType.SHORT_RESPONSE,
        )
        result = await summarizer.summarize(request)

        assert result.text == "Cleaned text"
        assert result.model_used == "qwen3:4b-instruct-2507-q4_K_M"  # small model for short responses
        assert result.tokens_used == 50

    @pytest.mark.asyncio
    async def test_summarize_long_response(self, summarizer, httpx_mock: HTTPXMock):
        """Test summarizing a long response."""
        httpx_mock.add_response(
            url="http://localhost:11434/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "I updated the files."}}],
                "usage": {"total_tokens": 100},
            },
        )

        request = SummaryRequest(
            content="A very long response " * 50,
            summary_type=SummaryType.LONG_RESPONSE,
        )
        result = await summarizer.summarize(request)

        assert result.text == "I updated the files."
        assert result.model_used == "qwen3:4b-instruct-2507-q4_K_M"  # large model for long responses

    @pytest.mark.asyncio
    async def test_health_check_success(self, summarizer, httpx_mock: HTTPXMock):
        """Test successful health check."""
        httpx_mock.add_response(
            url="http://localhost:11434/api/tags",
            status_code=200,
        )

        result = await summarizer.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, summarizer, httpx_mock: HTTPXMock):
        """Test failed health check."""
        httpx_mock.add_response(
            url="http://localhost:11434/api/tags",
            status_code=500,
        )

        result = await summarizer.health_check()
        assert result is False
