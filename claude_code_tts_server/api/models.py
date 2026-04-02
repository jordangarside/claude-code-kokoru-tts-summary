"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field


class TranscriptSummarizeRequest(BaseModel):
    """Request for the /summarize/transcript endpoint (Claude Code JSONL)."""

    transcript_content: str = Field(
        description="Transcript JSONL content",
    )


class SummarizeTextRequest(BaseModel):
    """Request for the /summarize/text endpoint."""

    content: str = Field(description="Conversation text to summarize")
    has_tool_calls: bool = Field(
        default=False, description="Whether content includes tool calls"
    )


class PermissionRequest(BaseModel):
    """Request for the /permission endpoint."""

    tool_name: str = Field(description="Name of the tool requesting permission")
    tool_input: dict = Field(description="Tool input parameters")


class SpeakRequest(BaseModel):
    """Request for the /speak endpoint (direct TTS)."""

    text: str = Field(description="Text to convert to speech")


class QueueStatusResponse(BaseModel):
    """Response for the /queue endpoint."""

    pending_requests: int = Field(description="Requests waiting for summarization")
    pending_messages: int = Field(description="Messages waiting for TTS generation")
    ready_audio: int = Field(description="Audio files ready to play")
    is_playing: bool = Field(description="Whether audio is currently playing")
    current_text: str | None = Field(description="Text currently being played")


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""

    status: str = Field(description="Server status")
    tts_ready: bool = Field(description="Whether TTS backend is initialized")
    summarizer_ready: bool = Field(description="Whether summarizer is available")
    queue_depth: int = Field(description="Total items in queue")


class MessageResponse(BaseModel):
    """Response when a message is queued."""

    message_id: str = Field(description="ID of the queued message")
    status: str = Field(description="Status message")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(description="Error message")
    detail: str | None = Field(default=None, description="Additional details")
