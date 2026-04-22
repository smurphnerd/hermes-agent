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
import shutil
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
    cmd = ["claude", "-p", "--output-format", "json"]
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
        cmd.extend(["--tools", ""])
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


def normalize_claude_code_response(
    result: Dict[str, Any],
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

    msg = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=None,
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
    disable_tools: bool = False,
    allowed_tools: Optional[List[str]] = None,
    persist_session: bool = True,
    timeout: Optional[float] = None,
    extra_flags: Optional[List[str]] = None,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    system_prompt = None
    user_prompt = ""

    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            system_prompt = _extract_text_from_content(msg.get("content", ""))
        elif role == "user":
            user_prompt = _extract_text_from_content(msg.get("content", ""))

    effort = None
    if reasoning_config and isinstance(reasoning_config, dict):
        effort = reasoning_config.get("effort")

    kwargs: Dict[str, Any] = {
        "prompt": user_prompt,
        "model": model,
    }

    if system_prompt:
        kwargs["system_prompt"] = system_prompt
    if session_id:
        kwargs["session_id"] = session_id
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
