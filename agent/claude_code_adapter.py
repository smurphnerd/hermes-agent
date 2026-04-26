"""Claude Code CLI adapter for Hermes Agent.

Routes conversations through the `claude -p` CLI command instead of calling
an API directly. Claude Code manages its own tools, auth, and conversation
state — this adapter is a thin subprocess wrapper.

Wire protocol: hermes sends the latest user message as a prompt, Claude Code
returns a JSON result blob. Multi-turn uses --resume <session_id>.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Env vars that would override Claude Code's own auth. When hermes spawns
# `claude -p`, the subprocess inherits hermes's env — which often contains
# ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN left over from when the user
# was on the plain `anthropic` provider. Those would point claude at a
# different (usually exhausted) account instead of its own saved OAuth login.
_CLAUDE_AUTH_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_API_KEY",
)


def _dump_failing_command(cmd: List[str], cwd: Optional[str]) -> None:
    """Write a shell script that replays the exact failing `claude -p`
    invocation so the user can run it from their shell and compare."""
    import shlex
    from hermes_constants import get_hermes_home

    dump_dir = get_hermes_home() / "logs"
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / "claude_code_last_failing_command.sh"

    lines = ["#!/usr/bin/env bash", "# Last failing `claude -p` invocation from hermes.", "# Run this directly in your shell to reproduce."]
    if cwd:
        lines.append(f"cd {shlex.quote(cwd)}")
    lines.append("")
    lines.append("exec " + " ".join(shlex.quote(tok) for tok in cmd))
    lines.append("")

    dump_path.write_text("\n".join(lines))
    os.chmod(dump_path, 0o755)
    logger.info(
        "Wrote replay script for the failing claude -p invocation to %s — "
        "run it from your shell to reproduce exactly what hermes sent.",
        dump_path,
    )


def _env_for_claude_subprocess() -> Dict[str, str]:
    """Build subprocess env, stripping anthropic auth vars that would
    override Claude Code's own OAuth login."""
    env = dict(os.environ)
    stripped = [v for v in _CLAUDE_AUTH_ENV_VARS if v in env]
    for var in stripped:
        env.pop(var, None)
    if stripped:
        logger.info(
            "Claude Code subprocess env: stripped %s so `claude` uses its own OAuth login",
            ", ".join(stripped),
        )
    return env

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "completed": "stop",
}


class ClaudeCodeError(Exception):
    def __init__(self, message: str, result: Optional[Dict] = None):
        super().__init__(message)
        self.result = result


def find_claude_binary() -> str:
    path = shutil.which("claude")
    if path is None:
        raise FileNotFoundError(
            "claude CLI not found on PATH. Install Claude Code: "
            "https://docs.anthropic.com/en/docs/claude-code"
        )
    return path


def build_claude_code_command(
    prompt: str,
    model: str,
    *,
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None,
    effort: Optional[str] = None,
    max_budget_usd: Optional[float] = None,
    disable_tools: bool = False,
    allowed_tools: Optional[List[str]] = None,
    persist_session: bool = True,
    extra_flags: Optional[List[str]] = None,
) -> List[str]:
    cmd = ["claude", "-p", "--output-format", "json", "--dangerously-skip-permissions"]
    cmd.extend(["--model", model])
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    if session_id:
        cmd.extend(["--resume", session_id])
    if effort:
        cmd.extend(["--effort", effort])
    if max_budget_usd is not None:
        cmd.extend(["--max-budget-usd", str(max_budget_usd)])
    if disable_tools:
        cmd.append("--tools=")
    if allowed_tools:
        cmd.extend(["--allowedTools", ",".join(allowed_tools)])
    if not persist_session:
        cmd.append("--no-session-persistence")
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.append(prompt)
    return cmd


async def run_claude_code(
    prompt: str,
    model: str,
    *,
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None,
    effort: Optional[str] = None,
    max_budget_usd: Optional[float] = None,
    disable_tools: bool = False,
    allowed_tools: Optional[List[str]] = None,
    persist_session: bool = True,
    timeout: Optional[float] = None,
    extra_flags: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    process_holder: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cmd = build_claude_code_command(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        session_id=session_id,
        effort=effort,
        max_budget_usd=max_budget_usd,
        disable_tools=disable_tools,
        allowed_tools=allowed_tools,
        persist_session=persist_session,
        extra_flags=extra_flags,
    )

    # Build a redacted view of the command for logging: show flag names
    # and a size-summary for each value so we can diagnose issues without
    # dumping 20k tokens of system prompt into the log.
    def _redact(value: str) -> str:
        if value is None:
            return "<None>"
        n = len(value)
        if n <= 60:
            return repr(value)
        return f"<{n} chars>"

    redacted = []
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        if tok.startswith("--") and i + 1 < len(cmd) and not cmd[i + 1].startswith("-"):
            redacted.append(f"{tok}={_redact(cmd[i + 1])}")
            i += 2
        else:
            redacted.append(_redact(tok) if len(tok) > 60 else tok)
            i += 1
    logger.info("Claude Code command: %s", " ".join(redacted))

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=_env_for_claude_subprocess(),
    )

    if process_holder is not None:
        process_holder["process"] = process

    try:
        if timeout is not None:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        else:
            stdout, stderr = await process.communicate()
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise

    stdout_str = stdout.decode("utf-8", errors="replace").strip()
    stderr_str = stderr.decode("utf-8", errors="replace").strip()

    if stderr_str:
        logger.info("Claude Code stderr (exit=%s): %s", process.returncode, stderr_str[:2000])

    if not stdout_str:
        raise ClaudeCodeError(
            f"Claude Code returned empty output (exit code {process.returncode}). "
            f"stderr: {stderr_str[:500]}"
        )

    try:
        result = json.loads(stdout_str)
    except json.JSONDecodeError as e:
        raise ClaudeCodeError(
            f"Failed to parse Claude Code JSON output: {e}. "
            f"Raw output: {stdout_str[:500]}"
        )

    if result.get("is_error"):
        # Log the full raw result so we can see exactly what claude returned
        # (api_error_status, request_id, service_tier, etc.) when diagnosing
        # quota/billing failures.
        logger.warning("Claude Code returned is_error=true. Raw result: %s", json.dumps(result)[:3000])
        # Dump the exact failing invocation to a replay script so the user
        # can run it verbatim from a shell and compare against hermes.
        try:
            _dump_failing_command(cmd, cwd)
        except Exception as _dump_err:
            logger.debug("Failed to dump replay script: %s", _dump_err)
        raise ClaudeCodeError(
            result.get("result", "Unknown Claude Code error"),
            result=result,
        )

    return result


def _format_tools_for_prompt(tools: List[Dict]) -> str:
    """Format OpenAI-style tool definitions into an XML block for the prompt.

    Returns a string like:
        <available-tools>
        [{"name": "web_search", "description": "...", "parameters": {...}}, ...]
        </available-tools>

        To call a tool, output one or more <tool_call> blocks: ...
    """
    if not tools:
        return ""
    formatted = []
    for tool in tools:
        func = tool.get("function", tool)
        formatted.append({
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
        })
    lines = [
        "<available-tools>",
        json.dumps(formatted, ensure_ascii=False),
        "</available-tools>",
        "",
        "To call a tool, output one or more <tool_call> blocks in your response EXACTLY like this:",
        '<tool_call>{"name": "tool_name", "arguments": {"arg1": "value1"}}</tool_call>',
        "",
        "Rules for tool calls:",
        "- You may include text before, between, or after tool_call blocks.",
        "- Each <tool_call> block must contain valid JSON with \"name\" and \"arguments\" keys.",
        "- arguments must be an object matching the tool's parameter schema.",
        "- You will receive tool results in <tool_result> blocks and can then continue.",
        "- When you have enough information to answer, respond with plain text (no tool_call blocks).",
    ]
    return "\n".join(lines)


_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)

_TOOL_RESULT_PATTERN = re.compile(
    r"<tool_result[^>]*>\s*.*?\s*</tool_result>", re.DOTALL
)


def _parse_tool_calls_from_text(text: str) -> Tuple[str, List[SimpleNamespace]]:
    """Extract <tool_call> blocks from assistant text.

    Returns (cleaned_text, tool_calls) where cleaned_text has the blocks
    removed and tool_calls is a list of SimpleNamespace objects matching
    the shape expected by run_agent.py's tool execution loop.
    """
    matches = list(_TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        return text, []

    tool_calls = []
    for match in matches:
        raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed tool_call JSON: %s", raw[:200])
            continue

        name = parsed.get("name", "")
        arguments = parsed.get("arguments", {})
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)

        call_id = f"call_{uuid.uuid4().hex[:24]}"
        tool_calls.append(SimpleNamespace(
            id=call_id,
            type="function",
            function=SimpleNamespace(
                name=name,
                arguments=arguments,
            ),
        ))

    cleaned = _TOOL_CALL_PATTERN.sub("", text).strip()
    return cleaned, tool_calls


def _format_tool_results_for_prompt(messages: List[Dict[str, Any]]) -> str:
    """Format trailing tool-result messages into <tool_result> blocks.

    When the agent loop executes tools and feeds results back, the messages
    list ends with an assistant message followed by one or more role=tool
    messages. This function converts those into text that gets appended to
    the next prompt so Claude Code sees the results.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            parts.append(
                f'<tool_result tool_call_id="{tool_call_id}">\n'
                f"{content}\n"
                f"</tool_result>"
            )
    return "\n\n".join(parts)


def normalize_claude_code_response(
    result: Dict[str, Any],
    parse_tool_calls: bool = False,
) -> Tuple[SimpleNamespace, str]:
    content = result.get("result", "") or ""
    stop_reason = result.get("stop_reason", "end_turn")
    finish_reason = _STOP_REASON_MAP.get(stop_reason, "stop")

    raw_usage = result.get("usage", {})
    input_tokens = raw_usage.get("input_tokens", 0)
    output_tokens = raw_usage.get("output_tokens", 0)

    usage = SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_creation_input_tokens=raw_usage.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=raw_usage.get("cache_read_input_tokens", 0),
        total_cost_usd=result.get("total_cost_usd", 0),
    )

    content = _TOOL_RESULT_PATTERN.sub("", content).strip()

    tool_calls = None
    if parse_tool_calls:
        content, parsed_calls = _parse_tool_calls_from_text(content)
        if parsed_calls:
            tool_calls = parsed_calls

    msg = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        reasoning=None,
        reasoning_details=None,
        usage=usage,
        claude_code_session_id=result.get("session_id"),
    )

    return msg, finish_reason


def _extract_text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content) if content else ""


def build_claude_code_kwargs(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict]] = None,
    max_tokens: Optional[int] = None,
    reasoning_config: Optional[Dict] = None,
    session_id: Optional[str] = None,
    max_budget_usd: Optional[float] = None,
    disable_tools: Optional[bool] = None,
    allowed_tools: Optional[List[str]] = None,
    persist_session: bool = True,
    timeout: Optional[float] = None,
    extra_flags: Optional[List[str]] = None,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    hermes_system_prompt = None
    transcript: List[Tuple[str, str]] = []
    latest_user_prompt = ""
    trailing_tool_results: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        text = _extract_text_from_content(msg.get("content", ""))
        if role == "system":
            hermes_system_prompt = text
        elif role in ("user", "assistant"):
            transcript.append((role, text))
            trailing_tool_results = []
        elif role == "tool":
            trailing_tool_results.append(msg)

    # The last user turn is what we're asking claude to respond to.
    # Everything before it is prior conversation history.
    if transcript and transcript[-1][0] == "user":
        latest_user_prompt = transcript[-1][1]
        prior_turns = transcript[:-1]
    else:
        prior_turns = transcript

    parts: List[str] = []
    if hermes_system_prompt:
        parts.append(
            "<system-instructions>\n"
            f"{hermes_system_prompt}\n"
            "</system-instructions>"
        )

    # Inject tool definitions so the model can call hermes tools via XML.
    if tools:
        tools_block = _format_tools_for_prompt(tools)
        if tools_block:
            parts.append(tools_block)

    if prior_turns:
        history_lines = ["<conversation-history>"]
        for role, text in prior_turns:
            label = "User" if role == "user" else "Assistant"
            history_lines.append(f"{label}: {text}")
        history_lines.append("</conversation-history>")
        parts.append("\n".join(history_lines))

    # If this call is a tool-result continuation (the agent loop executed
    # tools and is feeding results back), include the results so the model
    # can see them and continue.
    if trailing_tool_results:
        tool_results_block = _format_tool_results_for_prompt(trailing_tool_results)
        if tool_results_block:
            parts.append(
                "Here are the results of the tool calls you made:\n\n"
                + tool_results_block
                + "\n\nContinue based on these results."
            )
    else:
        parts.append(latest_user_prompt)

    user_prompt = "\n\n".join(p for p in parts if p)

    effort = None
    if reasoning_config and isinstance(reasoning_config, dict):
        effort = reasoning_config.get("effort")

    # Default: leave Claude Code's internal tools enabled. Claude Code
    # handles tool execution internally and returns the final text result.
    # Hermes's own tool layer is skipped for claude_code api_mode (tools
    # can't round-trip through a subprocess), so Claude Code's built-in
    # tools are the only way the agent gets file/bash access.
    # Opt out with HERMES_CLAUDE_CODE_DISABLE_TOOLS=1.
    if disable_tools is None:
        disable_tools = os.environ.get(
            "HERMES_CLAUDE_CODE_DISABLE_TOOLS", ""
        ).strip().lower() in ("1", "true", "yes")

    kwargs: Dict[str, Any] = {
        "prompt": user_prompt,
        "model": model,
    }
    if effort:
        kwargs["effort"] = effort
    if max_budget_usd is not None:
        kwargs["max_budget_usd"] = max_budget_usd
    if disable_tools:
        kwargs["disable_tools"] = True
    if allowed_tools:
        kwargs["allowed_tools"] = allowed_tools
    if not persist_session:
        kwargs["persist_session"] = False
    if timeout is not None:
        kwargs["timeout"] = timeout
    if extra_flags:
        kwargs["extra_flags"] = extra_flags
    if cwd:
        kwargs["cwd"] = cwd

    return kwargs
