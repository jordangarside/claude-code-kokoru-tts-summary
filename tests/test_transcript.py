"""Tests for transcript parsing."""

import json
from pathlib import Path

import pytest

from claude_code_tts_server.core.transcript import parse_transcript


class TestParseTranscript:
    """Tests for parse_transcript function."""

    def test_parse_simple_transcript(self, sample_transcript_jsonl):
        """Test parsing a simple transcript."""
        result = parse_transcript(sample_transcript_jsonl)

        assert result is not None
        assert "happy to help" in result.content
        assert not result.has_tool_calls
        assert result.length > 0

    def test_parse_transcript_with_tools(self, sample_transcript_with_tools):
        """Test parsing a transcript with tool calls."""
        result = parse_transcript(sample_transcript_with_tools)

        assert result is not None
        assert "[Tool: Bash]" in result.content
        assert result.has_tool_calls
        assert "pytest" in result.content

    def test_parse_transcript_with_interrupt(self, sample_transcript_with_interrupt):
        """Test parsing a transcript with an interrupt."""
        result = parse_transcript(sample_transcript_with_interrupt)

        assert result is not None
        # Should only include content after the interrupt
        assert "won't proceed" in result.content
        # Should not include the dangerous command
        assert "rm -rf" not in result.content

    def test_parse_nonexistent_file(self):
        """Test parsing a nonexistent file."""
        result = parse_transcript("/nonexistent/path/transcript.jsonl")
        assert result is None

    def test_parse_empty_file(self, tmp_path):
        """Test parsing an empty file."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.touch()

        result = parse_transcript(empty_file)
        assert result is None

    def test_parse_transcript_truncates_long_values(self, tmp_path):
        """Test that long tool input values are truncated."""
        transcript_file = tmp_path / "long_values.jsonl"

        long_value = "x" * 200
        entries = [
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "Do something"}]}
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Edit",
                            "input": {"content": long_value}
                        }
                    ]
                }
            },
        ]

        with open(transcript_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        result = parse_transcript(transcript_file)

        assert result is not None
        assert "..." in result.content
        # Original 200 chars should be truncated to 150 + "..."
        assert len(long_value) not in [len(part) for part in result.content.split()]

    def test_parse_transcript_with_string_content(self, tmp_path):
        """Test parsing transcript where user content is a string (context summarization)."""
        transcript_file = tmp_path / "string_content.jsonl"

        entries = [
            # Old assistant content that should be excluded
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Old content before user message."}]
                }
            },
            # User message with string content (from context summarization)
            {
                "type": "user",
                "message": {
                    "content": "User asked about something specific"
                }
            },
            # New assistant content after user message
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "New content after user message."}]
                }
            },
        ]

        with open(transcript_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        result = parse_transcript(transcript_file)

        assert result is not None
        # Should only include content after the user message (string content)
        assert "New content" in result.content
        assert "Old content" not in result.content

    def test_parse_transcript_truncates_long_content(self, tmp_path):
        """Test that very long content is truncated."""
        transcript_file = tmp_path / "long_content.jsonl"

        # Create content that exceeds the default limit
        long_text = "x" * 25000  # Over 20k default limit

        entries = [
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "Start"}]}
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": long_text}]
                }
            },
        ]

        with open(transcript_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        result = parse_transcript(transcript_file)

        assert result is not None
        assert result.truncated is True
        assert "[Earlier content truncated...]" in result.content
        assert result.length < 25000  # Should be truncated

    def test_parse_transcript_custom_max_length(self, tmp_path):
        """Test custom max content length."""
        transcript_file = tmp_path / "custom_limit.jsonl"

        entries = [
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "Start"}]}
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "A" * 500}]
                }
            },
        ]

        with open(transcript_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        # With default limit, should not truncate
        result = parse_transcript(transcript_file)
        assert result.truncated is False

        # With small limit, should truncate
        result = parse_transcript(transcript_file, max_content_length=100)
        assert result.truncated is True
        assert "[Earlier content truncated...]" in result.content
