// TTS Audio Plugin for OpenCode
//
// Provides audio feedback via a TTS server — session summaries,
// permission announcements, and MCP tool output speech.
//
// Installation: copy this file to .opencode/plugins/tts-plugin.ts
//
// Environment variables:
//   SUMMARY_AUDIO_PORT - TTS server port (default: 20202)
//   TTS_MCP_PREFIX     - Tool name prefix for MCP speech (default: "mcp__")

import type { Plugin } from "@opencode-ai/plugin"

const MAX_CONTENT_LENGTH = 20_000
const MAX_VALUE_LENGTH = 150

export const TTSAudioPlugin: Plugin = async ({ client }) => {
  const TTS_PORT = process.env.SUMMARY_AUDIO_PORT ?? "20202"
  const TTS_URL = `http://localhost:${TTS_PORT}`
  const MCP_PREFIX = process.env.TTS_MCP_PREFIX ?? "mcp__"

  // Track last processed assistant message per session to avoid duplicates
  const lastProcessedMessageId = new Map<string, string>()

  async function postToTTS(
    path: string,
    body: Record<string, unknown>,
  ): Promise<void> {
    try {
      await fetch(`${TTS_URL}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(10_000),
      })
    } catch {
      // Server unavailable — silent, same as shell hooks
    }
  }

  function truncateValue(value: unknown): string {
    const str =
      typeof value === "string" ? value : JSON.stringify(value) ?? ""
    return str.length > MAX_VALUE_LENGTH
      ? str.slice(0, MAX_VALUE_LENGTH) + "..."
      : str
  }

  return {
    // Session lifecycle events: idle (summarize) and deleted (cleanup)
    event: async ({ event }) => {
      const props = (event as any)?.properties

      if (event.type === "session.idle") {
        const sessionID = props?.sessionID as string | undefined
        if (!sessionID) return

        let messages: any[]
        try {
          const response = await client.session.messages({
            path: { id: sessionID },
          })
          messages = response.data ?? []
        } catch {
          return
        }

        // Find last assistant message
        const lastAssistant = [...messages]
          .reverse()
          .find((m: any) => m.info?.role === "assistant")
        if (!lastAssistant) return

        const msgId = lastAssistant.info.id as string
        if (!msgId) return

        // Deduplication: skip if we already processed this message
        if (lastProcessedMessageId.get(sessionID) === msgId) return

        const parts: any[] = lastAssistant.parts ?? []

        // Skip messages with errors or summaries (compacted)
        if (lastAssistant.info.error || lastAssistant.info.summary) return

        // Skip if any tool parts are still pending/running
        const hasPendingTools = parts.some(
          (p: any) =>
            p.type === "tool" &&
            (p.state?.status === "pending" || p.state?.status === "running"),
        )
        if (hasPendingTools) return

        // Build content from parts, matching core/transcript.py format
        let hasToolCalls = false
        const contentParts: string[] = []

        for (const part of parts) {
          if (part.type === "text" && part.text) {
            contentParts.push(part.text)
          } else if (part.type === "tool") {
            hasToolCalls = true
            const toolName = part.tool ?? "unknown"
            const input = part.state?.input ?? {}
            const params = Object.entries(input).map(
              ([k, v]) => `${k}: ${truncateValue(v)}`,
            )
            contentParts.push(`[Tool: ${toolName}] ${params.join(", ")}`)
          }
        }

        const content = contentParts.join("\n\n")
        if (!content.trim()) return

        // Mark as processed before sending (avoid races)
        lastProcessedMessageId.set(sessionID, msgId)

        // Truncate from beginning if too long (keep most recent)
        const finalContent =
          content.length > MAX_CONTENT_LENGTH
            ? "[Earlier content truncated...]\n\n" +
              content.slice(-MAX_CONTENT_LENGTH)
            : content

        await postToTTS("/summarize/text", {
          content: finalContent,
          has_tool_calls: hasToolCalls,
        })
      }

      if (event.type === "session.deleted") {
        // Cleanup: session.deleted uses properties.info.id for session ID
        const sessionID = props?.info?.id as string | undefined
        if (sessionID) lastProcessedMessageId.delete(sessionID)
      }
    },

    // Permission announcements — non-blocking (fire-and-forget)
    "permission.ask": async (input: any) => {
      const toolName =
        input.metadata?.tool ??
        input.metadata?.toolName ??
        input.type ??
        input.title ??
        "unknown"
      const toolInput = { description: input.title, ...input.metadata }

      // void — returns immediately, does not block the permission UI
      void postToTTS("/permission", {
        tool_name: toolName,
        tool_input: toolInput,
      })
    },

    // MCP tool output speech — non-blocking
    "tool.execute.after": async (input: any, output: any) => {
      if (!input.tool?.startsWith(MCP_PREFIX)) return

      const text = output.output?.trim()
      if (!text) return

      void postToTTS("/speak", { text })
    },
  }
}
