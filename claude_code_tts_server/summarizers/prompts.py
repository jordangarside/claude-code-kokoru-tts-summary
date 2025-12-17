"""System prompts for summarization."""

# Short response prompt (cleaning for TTS)
# Used for responses < 300 chars with no tool calls
PROMPT_SHORT_RESPONSE = """Convert this text for text-to-speech by removing markdown formatting and code blocks. Expand abbreviated units (0.2s -> 0.2 seconds, 100ms -> 100 milliseconds, 5MB -> 5 megabytes). Expand ALL file extensions to full names (.py -> Python, .js -> JavaScript, .yaml -> YAML, .html -> HTML). Output ONLY the cleaned text."""

# Long response / tool use summarization prompt
# Used for responses >= 300 chars or containing tool calls
PROMPT_LONG_RESPONSE = """Summarize the following Claude Code response for text-to-speech. Write 1-3 sentences in first-person AS IF YOU ARE Claude Code.

Rules:
- ACTIONS (edited files, ran commands): use past tense. Example: I updated the config and ran the tests.
- CREATIVE CONTENT (stories, poems, jokes you wrote): summarize what was created. Example: I told a story about a clockmaker who discovers a mysterious automaton.
- EXPLANATIONS: summarize what was explained. Example: I explained how the authentication system works.
- QUESTIONS: keep as-is. Example: How would you like to proceed?
- No markdown, no bullet points, no code blocks
- Expand abbreviated units (0.2s -> 0.2 seconds, 100ms -> 100 milliseconds, 5MB -> 5 megabytes)
- Expand ALL file extensions to full names (.py -> Python, .js -> JavaScript, .yaml -> YAML, .html -> HTML)
- Output ONLY the first-person summary"""

# Permission request prompt
PROMPT_PERMISSION_REQUEST = """Convert this permission request into a brief spoken announcement (under 30 words). Start with 'Permission requested:'. No quotes, no special characters. Output ONLY the announcement.

Examples:
Input: Tool: Bash. Description: Install dependencies. Input: {"command":"npm install","description":"Install dependencies"}
Output: Permission requested: Command to install node dependencies

Input: Tool: WebFetch. Input: {"url":"https://docs.python.org/3/library/json.html","prompt":"How do I parse JSON?"}
Output: Permission requested: Fetch Python documentation page

Input: Tool: Edit. Input: {"file_path":"/src/auth.js","old_string":"token","new_string":"sessionToken"}
Output: Permission requested: Edit auth.js file

Input: Tool: Bash. Description: Show working tree status. Input: {"command":"git status","description":"Show working tree status"}
Output: Permission requested: Command to show working tree status

Input: Tool: Bash. Input: {"command":"docker ps -a"}
Output: Permission requested: Command to list all Docker containers"""


def get_prompt_and_params(summary_type: "SummaryType") -> tuple[str, float, int]:
    """Get prompt and generation parameters for a summary type.

    Args:
        summary_type: The type of summary being generated.

    Returns:
        Tuple of (prompt, temperature, max_tokens).
    """
    from .base import SummaryType

    if summary_type == SummaryType.SHORT_RESPONSE:
        return PROMPT_SHORT_RESPONSE, 0.3, 2048
    elif summary_type == SummaryType.LONG_RESPONSE:
        return PROMPT_LONG_RESPONSE, 0.3, 2048
    else:  # PERMISSION_REQUEST
        return PROMPT_PERMISSION_REQUEST, 0.1, 50
