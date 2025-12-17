# CLAUDE.md

## Project Overview

Claude Code TTS Server - Audio feedback for Claude Code via text-to-speech.

## Architecture

```
claude_code_tts_server/           # Python package
├── main.py                       # FastAPI entry point, CLI
├── config.py                     # Pydantic settings
├── api/
│   ├── routes.py                 # REST endpoints
│   └── models.py                 # Request/response models
├── core/
│   ├── audio_manager.py          # Async workers, queue management
│   ├── context.py                # Request ID context, logging utilities
│   ├── playback.py               # Audio playback
│   ├── sounds.py                 # Chime/drop tone generation
│   └── transcript.py             # JSONL transcript parsing
├── summarizers/
│   ├── base.py                   # SummarizerInterface ABC
│   ├── groq.py                   # Groq implementation
│   ├── ollama.py                 # Ollama implementation
│   └── prompts.py                # System prompts
└── tts/
    ├── base.py                   # TTSInterface ABC
    └── kokoro.py                 # Kokoro implementation

claude-code-hooks/                # Shell script wrappers
├── summary-tts.sh                # Stop hook -> POST /summarize
└── permission-tts.sh             # PermissionRequest hook -> POST /permission
```

**API Flow:**
1. Claude Code triggers hook with JSON on stdin
2. Hook script POSTs to TTS server API
3. Server parses transcript and summarizes via Groq
4. Server generates TTS via Kokoro
5. Audio queued and played with interrupt logic

**Key Interfaces:**
- `SummarizerInterface` - Abstract base for LLM summarization backends
- `TTSInterface` - Abstract base for TTS generation backends

## Environment Variables

All settings in `.env` file. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SUMMARY_BACKEND` | `groq` | `groq` or `ollama` |
| `SUMMARY_GROQ_API_KEY` | - | Required for Groq |
| `SUMMARY_OLLAMA_MODEL_LARGE` | `llama3.1:8b` | Ollama large model |
| `SUMMARY_OLLAMA_MODEL_SMALL` | `llama3.2:1b` | Ollama small model |
| `SUMMARY_AUDIO_PORT` | `20202` | Server port |

See `.env.example` for all options.

## Running

```bash
# Start TTS server
uv run tts-server

# For remote usage, SSH with reverse tunnel
ssh -R 20202:localhost:20202 user@server
```

## Testing

```bash
uv sync --group dev
uv run pytest tests/ -v
```

## Debugging

Hook output logs (written to script directory):
- `~/.claude/hooks/summary-tts.output`
- `~/.claude/hooks/permission-tts.output`

## Commit Messages

Do not include the "Generated with Claude Code" line in commit messages.

Commits should still include the `Co-Authored-By` line.
