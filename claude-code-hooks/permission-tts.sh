#!/bin/bash

# Claude Code TTS Permission Hook
# Posts permission request to TTS server for announcement
#
# This is a thin wrapper that forwards the PermissionRequest hook payload
# to the TTS server. All summarization logic is handled by the server.
#
# Optional environment variables:
#   SUMMARY_AUDIO_PORT - Port for TTS server (default: 20202)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_NAME="$(basename "$0" .sh)"
LOG_FILE="$SCRIPT_DIR/$SCRIPT_NAME.output"

TTS_PORT="${SUMMARY_AUDIO_PORT:-20202}"
TTS_URL="http://localhost:${TTS_PORT}"

# Log to file
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
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

# Exit silently if no tool name
if [ -z "$tool_name" ]; then
  exit 0
fi

# Quick check if TTS server is available (non-blocking)
if ! (echo >/dev/tcp/localhost/"$TTS_PORT") 2>/dev/null; then
  exit 0
fi

# Save input for debugging
echo "$input" | jq '.' > "$SCRIPT_DIR/$SCRIPT_NAME.input" 2>/dev/null

# POST to TTS server (fire and forget in background)
{
  response=$(curl -s -X POST "${TTS_URL}/permission" \
    -H "Content-Type: application/json" \
    -d "$input" \
    --max-time 5 2>&1)

  exit_code=$?

  if [ $exit_code -eq 0 ]; then
    message_id=$(echo "$response" | jq -r '.message_id // empty' 2>/dev/null)
    log "Permission queued for $tool_name: $message_id"
  else
    log "Failed to queue permission for $tool_name: curl exit code $exit_code"
  fi
} &

exit 0
