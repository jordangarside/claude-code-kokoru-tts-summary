#!/bin/bash

# Claude Code Permission Request Hook
# Announces permission requests via TTS using Groq for summarization
#
# Required environment variables:
#   SUMMARY_GROQ_API_KEY        - Groq API key for summarization
#
# Optional environment variables:
#   SUMMARY_GROQ_MODEL_SMALL    - Groq model (default: llama-3.1-8b-instant)
#   SUMMARY_AUDIO_PORT          - Port for Kokoro TTS server (default: 20202)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_NAME="$(basename "$0" .sh)"
ERROR_LOG="$SCRIPT_DIR/$SCRIPT_NAME.output"

AUDIO_PORT="${SUMMARY_AUDIO_PORT:-20202}"
GROQ_MODEL="${SUMMARY_GROQ_MODEL_SMALL:-llama-3.1-8b-instant}"

# Clear previous log
rm -f "$ERROR_LOG"

# Log success with input and output
log_result() {
  local input="$1"
  local output="$2"
  {
    echo "=== $(date) ==="
    echo ""
    echo "--- Input ---"
    echo "$input" | jq '.' 2>/dev/null || echo "$input"
    echo ""
    echo "--- Output ---"
    echo "$output"
  } > "$ERROR_LOG"
}

# Guard: This script is meant to be called by Claude Code hooks, not directly
if [ -t 0 ]; then
  echo "Error: This script is a Claude Code hook and should not be run directly." >&2
  exit 1
fi

# Read hook input from stdin
input=$(cat)

# Extract tool info
tool_name=$(echo "$input" | jq -r '.tool_name // empty')
tool_input=$(echo "$input" | jq -r '.tool_input // empty')

if [ -z "$tool_name" ]; then
  exit 0
fi

# Check required env vars
if [ -z "$SUMMARY_GROQ_API_KEY" ]; then
  exit 0  # Silent exit - don't block permission dialog
fi

# Quick check if TTS server is available (non-blocking)
if ! (echo >/dev/tcp/localhost/"$AUDIO_PORT") 2>/dev/null; then
  exit 0
fi

# Build a description of the permission request
case "$tool_name" in
  Bash)
    command=$(echo "$tool_input" | jq -r '.command // empty')
    description="Run bash command: $command"
    ;;
  Write)
    file_path=$(echo "$tool_input" | jq -r '.file_path // empty')
    description="Write to file: $file_path"
    ;;
  Edit)
    file_path=$(echo "$tool_input" | jq -r '.file_path // empty')
    description="Edit file: $file_path"
    ;;
  Read)
    file_path=$(echo "$tool_input" | jq -r '.file_path // empty')
    description="Read file: $file_path"
    ;;
  Glob|Grep)
    pattern=$(echo "$tool_input" | jq -r '.pattern // empty')
    description="Search with $tool_name: $pattern"
    ;;
  WebFetch)
    url=$(echo "$tool_input" | jq -r '.url // empty')
    description="Fetch URL: $url"
    ;;
  *)
    description="Use $tool_name tool"
    ;;
esac

# Summarize for TTS using cheap model
SYSTEM_PROMPT="Convert this permission request into a brief spoken announcement (under 15 words). Include the tool type (Bash, Write, Edit, etc.) and what it's for. Start with 'Permission requested:'. Example: 'Permission requested: Bash command to check GitHub API'. No quotes, no special characters. Output ONLY the announcement."

groq_response=$(curl -s --max-time 5 -X POST "https://api.groq.com/openai/v1/chat/completions" \
    -H "Authorization: Bearer $SUMMARY_GROQ_API_KEY" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg system "$SYSTEM_PROMPT" --arg content "$description" --arg model "$GROQ_MODEL" '{
      model: $model,
      messages: [
        {role: "system", content: $system},
        {role: "user", content: $content}
      ],
      temperature: 0.1,
      max_tokens: 50
    }')")

if [ $? -ne 0 ]; then
  exit 0
fi

summary=$(echo "$groq_response" | jq -r '.choices[0].message.content // empty')

if [ -z "$summary" ]; then
  # Fallback to simple announcement
  summary="Permission requested for $tool_name"
fi

# Log input and output
log_result "$input" "$summary"

# Send to TTS server (non-blocking, fire and forget)
echo "$summary" | nc -w 1 localhost "$AUDIO_PORT" >/dev/null 2>&1 &

exit 0
