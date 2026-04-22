# Claude Code CLI Adapter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new model adapter that uses the `claude -p` CLI command as its backend, allowing hermes to route conversations through Claude Code instead of directly calling an API.

**Architecture:** The adapter follows the existing functional pattern (`agent/anthropic_adapter.py`, `agent/bedrock_adapter.py`, etc.) — a set of module-level functions that build kwargs, execute the subprocess, and normalize the response back to hermes's OpenAI-compatible `SimpleNamespace` shape. Multi-turn is achieved via `claude -p --resume <session_id>` which lets Claude Code manage its own conversation state. Tool use is delegated to Claude Code's built-in tools (Bash, Read, Edit, etc.) rather than passing hermes tools through — this makes the adapter a "delegate everything to Claude Code" mode.

**Tech Stack:** Python 3.10+, `subprocess` (async via `asyncio.create_subprocess_exec`), JSON parsing of `claude -p --output-format json` output.

---

## Design Decisions

### Why delegate tools to Claude Code?

Claude Code has its own powerful tool set (Bash, Read, Edit, Write, WebSearch, etc.) with permission management, sandboxing, and hooks. Trying to convert hermes's OpenAI-format tool definitions into Claude Code's format would be complex and fragile. Instead, this adapter treats Claude Code as an autonomous agent: hermes sends the prompt, Claude Code decides what tools to use, and hermes gets back the final result.

### Multi-turn via session resume

Rather than converting hermes's full message history into a single prompt each turn, we use `claude -p --resume <session_id>` to let Claude Code maintain its own conversation state. The adapter stores the Claude Code session ID on the hermes session and reuses it across turns.

### What gets passed through

- **System prompt** → `--system-prompt` flag
- **Model selection** → `--model` flag  
- **The latest user message** → piped as the prompt argument
- **Effort/reasoning** → `--effort` flag
- **Max budget** → `--max-budget-usd` flag (optional safety cap)

### What does NOT get passed through

- Hermes tool definitions (Claude Code uses its own tools)
- Temperature/top_p/sampling params (Claude Code doesn't expose these)
- Structured reasoning blocks (Claude Code handles thinking internally)

---

## Tasks

### Task 1: Create the adapter module skeleton

**Files:**
- Create: `agent/claude_code_adapter.py`
- Test: `tests/test_claude_code_adapter.py`

**Step 1: Write the failing test for `build_claude_code_command`**

```python
# tests/test_claude_code_adapter.py
"""Tests for the Claude Code CLI adapter."""

import json
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
import pytest


def test_build_command_basic():
    """Build a basic claude -p command with just a prompt."""
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(
        prompt="Hello world",
        model="sonnet",
    )
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "--output-format" in cmd
    idx = cmd.index("--output-format")
    assert cmd[idx + 1] == "json"
    assert "--model" in cmd
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "sonnet"
    # Prompt should be the last positional argument
    assert cmd[-1] == "Hello world"


def test_build_command_with_system_prompt():
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(
        prompt="Hi",
        model="opus",
        system_prompt="You are a helpful robot",
    )
    assert "--system-prompt" in cmd
    idx = cmd.index("--system-prompt")
    assert cmd[idx + 1] == "You are a helpful robot"


def test_build_command_with_resume():
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(
        prompt="Continue please",
        model="sonnet",
        session_id="abc-123-def",
    )
    assert "--resume" in cmd
    idx = cmd.index("--resume")
    assert cmd[idx + 1] == "abc-123-def"


def test_build_command_with_effort():
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(
        prompt="Think hard",
        model="opus",
        effort="high",
    )
    assert "--effort" in cmd
    idx = cmd.index("--effort")
    assert cmd[idx + 1] == "high"


def test_build_command_with_max_budget():
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(
        prompt="Do something",
        model="sonnet",
        max_budget_usd=5.0,
    )
    assert "--max-budget-usd" in cmd
    idx = cmd.index("--max-budget-usd")
    assert cmd[idx + 1] == "5.0"


def test_build_command_disables_tools_when_requested():
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(
        prompt="Just chat",
        model="sonnet",
        disable_tools=True,
    )
    assert "--tools" in cmd
    idx = cmd.index("--tools")
    assert cmd[idx + 1] == ""


def test_build_command_with_allowed_tools():
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(
        prompt="Edit some files",
        model="sonnet",
        allowed_tools=["Bash", "Read", "Edit"],
    )
    assert "--allowedTools" in cmd
    idx = cmd.index("--allowedTools")
    assert cmd[idx + 1] == "Bash,Read,Edit"


def test_build_command_no_session_persistence():
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(
        prompt="One shot",
        model="sonnet",
        persist_session=False,
    )
    assert "--no-session-persistence" in cmd
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/Smurphy/.hermes/hermes-agent && python -m pytest tests/test_claude_code_adapter.py -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```python
# agent/claude_code_adapter.py
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
import shutil
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def find_claude_binary() -> str:
    """Locate the claude CLI binary on PATH."""
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
    """Build the argument list for a `claude -p` subprocess call.

    Returns a list suitable for subprocess.run() or asyncio.create_subprocess_exec().
    """
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

    # Prompt is the final positional argument
    cmd.append(prompt)

    return cmd
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/Smurphy/.hermes/hermes-agent && python -m pytest tests/test_claude_code_adapter.py -v`
Expected: PASS (all test_build_command_* tests)

**Step 5: Commit**

```bash
git add agent/claude_code_adapter.py tests/test_claude_code_adapter.py
git commit -m "feat: add claude code adapter skeleton with build_claude_code_command"
```

---

### Task 2: Implement `run_claude_code` async subprocess execution

**Files:**
- Modify: `agent/claude_code_adapter.py`
- Modify: `tests/test_claude_code_adapter.py`

**Step 1: Write the failing test for `run_claude_code`**

Add to `tests/test_claude_code_adapter.py`:

```python
@pytest.mark.asyncio
async def test_run_claude_code_success():
    """run_claude_code executes subprocess and returns parsed JSON."""
    from agent.claude_code_adapter import run_claude_code

    fake_result = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "Hello!",
        "stop_reason": "end_turn",
        "session_id": "test-session-123",
        "total_cost_usd": 0.05,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }

    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(
        return_value=(json.dumps(fake_result).encode(), b"")
    )
    mock_process.returncode = 0

    with patch("agent.claude_code_adapter.asyncio") as mock_asyncio:
        mock_asyncio.create_subprocess_exec = AsyncMock(return_value=mock_process)
        mock_asyncio.subprocess = asyncio.subprocess

        result = await run_claude_code(
            prompt="Hello",
            model="sonnet",
        )

    assert result["result"] == "Hello!"
    assert result["session_id"] == "test-session-123"
    assert result["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_run_claude_code_error():
    """run_claude_code raises on non-zero exit with error info."""
    from agent.claude_code_adapter import run_claude_code, ClaudeCodeError

    fake_result = {
        "type": "result",
        "subtype": "error",
        "is_error": True,
        "result": "Something went wrong",
        "stop_reason": "stop_sequence",
        "session_id": "err-session",
    }

    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(
        return_value=(json.dumps(fake_result).encode(), b"some stderr")
    )
    mock_process.returncode = 1

    with patch("agent.claude_code_adapter.asyncio") as mock_asyncio:
        mock_asyncio.create_subprocess_exec = AsyncMock(return_value=mock_process)
        mock_asyncio.subprocess = asyncio.subprocess

        with pytest.raises(ClaudeCodeError) as exc_info:
            await run_claude_code(prompt="fail", model="sonnet")

        assert "Something went wrong" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_claude_code_timeout():
    """run_claude_code raises TimeoutError when subprocess exceeds timeout."""
    from agent.claude_code_adapter import run_claude_code

    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    mock_process.kill = AsyncMock()
    mock_process.wait = AsyncMock()

    with patch("agent.claude_code_adapter.asyncio") as mock_asyncio:
        mock_asyncio.create_subprocess_exec = AsyncMock(return_value=mock_process)
        mock_asyncio.subprocess = asyncio.subprocess
        mock_asyncio.TimeoutError = asyncio.TimeoutError
        mock_asyncio.wait_for = AsyncMock(side_effect=asyncio.TimeoutError())

        with pytest.raises(asyncio.TimeoutError):
            await run_claude_code(prompt="slow", model="sonnet", timeout=1.0)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/Smurphy/.hermes/hermes-agent && python -m pytest tests/test_claude_code_adapter.py::test_run_claude_code_success -v`
Expected: FAIL (function not defined)

**Step 3: Write implementation**

Add to `agent/claude_code_adapter.py`:

```python
class ClaudeCodeError(Exception):
    """Raised when claude -p returns an error result."""

    def __init__(self, message: str, result: Optional[Dict] = None):
        super().__init__(message)
        self.result = result


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
    """Execute a `claude -p` subprocess and return the parsed JSON result.

    Args:
        prompt: The user message to send.
        model: Model name/alias (e.g. "sonnet", "opus", "claude-sonnet-4-6").
        system_prompt: Optional system prompt override.
        session_id: Resume an existing Claude Code session.
        effort: Reasoning effort level.
        max_budget_usd: Maximum dollar spend for this call.
        disable_tools: If True, disable all Claude Code tools.
        allowed_tools: Whitelist of Claude Code tool names.
        persist_session: Whether Claude Code persists the session to disk.
        timeout: Subprocess timeout in seconds. None = no timeout.
        extra_flags: Additional CLI flags to pass through.
        cwd: Working directory for the subprocess.

    Returns:
        Parsed JSON dict from claude's stdout.

    Raises:
        ClaudeCodeError: When claude returns an error result.
        asyncio.TimeoutError: When the subprocess exceeds timeout.
        FileNotFoundError: When the claude binary is not found.
    """
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

    logger.info("Claude Code command: %s", " ".join(cmd[:6]) + " ...")

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
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
        logger.debug("Claude Code stderr: %s", stderr_str[:500])

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
        raise ClaudeCodeError(
            result.get("result", "Unknown Claude Code error"),
            result=result,
        )

    return result
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/Smurphy/.hermes/hermes-agent && python -m pytest tests/test_claude_code_adapter.py -v`
Expected: PASS (all tests including new async ones)

**Step 5: Commit**

```bash
git add agent/claude_code_adapter.py tests/test_claude_code_adapter.py
git commit -m "feat: add run_claude_code async subprocess execution"
```

---

### Task 3: Implement `normalize_claude_code_response`

**Files:**
- Modify: `agent/claude_code_adapter.py`
- Modify: `tests/test_claude_code_adapter.py`

**Step 1: Write the failing test**

Add to `tests/test_claude_code_adapter.py`:

```python
def test_normalize_response_basic():
    """Normalize a successful Claude Code JSON result to SimpleNamespace."""
    from agent.claude_code_adapter import normalize_claude_code_response

    raw = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "The answer is 42.",
        "stop_reason": "end_turn",
        "session_id": "sess-abc",
        "total_cost_usd": 0.10,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_creation_input_tokens": 5000,
            "cache_read_input_tokens": 3000,
        },
    }

    msg, finish_reason = normalize_claude_code_response(raw)

    assert msg.content == "The answer is 42."
    assert finish_reason == "stop"
    assert msg.tool_calls is None
    assert msg.role == "assistant"


def test_normalize_response_end_turn_maps_to_stop():
    from agent.claude_code_adapter import normalize_claude_code_response

    raw = {"result": "hi", "stop_reason": "end_turn", "usage": {}}
    msg, reason = normalize_claude_code_response(raw)
    assert reason == "stop"


def test_normalize_response_max_tokens():
    from agent.claude_code_adapter import normalize_claude_code_response

    raw = {"result": "partial...", "stop_reason": "max_tokens", "usage": {}}
    msg, reason = normalize_claude_code_response(raw)
    assert reason == "length"


def test_normalize_response_preserves_session_id():
    from agent.claude_code_adapter import normalize_claude_code_response

    raw = {"result": "hi", "stop_reason": "end_turn", "session_id": "s-123", "usage": {}}
    msg, reason = normalize_claude_code_response(raw)
    assert msg.claude_code_session_id == "s-123"


def test_normalize_response_usage():
    from agent.claude_code_adapter import normalize_claude_code_response

    raw = {
        "result": "hi",
        "stop_reason": "end_turn",
        "total_cost_usd": 0.25,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 1000,
            "cache_read_input_tokens": 500,
        },
    }
    msg, _ = normalize_claude_code_response(raw)
    assert msg.usage.prompt_tokens == 100
    assert msg.usage.completion_tokens == 50
    assert msg.usage.total_tokens == 150
    assert msg.usage.total_cost_usd == 0.25
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/Smurphy/.hermes/hermes-agent && python -m pytest tests/test_claude_code_adapter.py::test_normalize_response_basic -v`
Expected: FAIL (function not defined)

**Step 3: Write implementation**

Add to `agent/claude_code_adapter.py`:

```python
# Map Claude Code stop_reason to hermes/OpenAI finish_reason
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "completed": "stop",
}


def normalize_claude_code_response(
    result: Dict[str, Any],
) -> Tuple[SimpleNamespace, str]:
    """Normalize a Claude Code JSON result to match hermes's expected shape.

    Returns (assistant_message, finish_reason) where assistant_message is a
    SimpleNamespace with .content, .role, .tool_calls, .usage, and
    .claude_code_session_id attributes.
    """
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
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/Smurphy/.hermes/hermes-agent && python -m pytest tests/test_claude_code_adapter.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add agent/claude_code_adapter.py tests/test_claude_code_adapter.py
git commit -m "feat: add normalize_claude_code_response"
```

---

### Task 4: Implement `build_claude_code_kwargs` (the adapter's main entry point)

This is the function that `run_agent.py:_build_api_kwargs` will call. It converts hermes's internal state into the kwargs needed for `run_claude_code`.

**Files:**
- Modify: `agent/claude_code_adapter.py`
- Modify: `tests/test_claude_code_adapter.py`

**Step 1: Write the failing test**

Add to `tests/test_claude_code_adapter.py`:

```python
def test_build_kwargs_extracts_system_and_user():
    """build_claude_code_kwargs extracts system prompt and latest user message."""
    from agent.claude_code_adapter import build_claude_code_kwargs

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello there"},
    ]

    kwargs = build_claude_code_kwargs(
        model="sonnet",
        messages=messages,
    )

    assert kwargs["model"] == "sonnet"
    assert kwargs["system_prompt"] == "You are helpful."
    assert kwargs["prompt"] == "Hello there"


def test_build_kwargs_multi_turn_extracts_latest_user():
    from agent.claude_code_adapter import build_claude_code_kwargs

    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "First reply"},
        {"role": "user", "content": "Second message"},
    ]

    kwargs = build_claude_code_kwargs(
        model="opus",
        messages=messages,
    )

    assert kwargs["prompt"] == "Second message"
    assert kwargs["system_prompt"] == "Be concise."


def test_build_kwargs_reasoning_config():
    from agent.claude_code_adapter import build_claude_code_kwargs

    kwargs = build_claude_code_kwargs(
        model="opus",
        messages=[{"role": "user", "content": "think hard"}],
        reasoning_config={"effort": "high"},
    )

    assert kwargs["effort"] == "high"


def test_build_kwargs_no_system():
    from agent.claude_code_adapter import build_claude_code_kwargs

    kwargs = build_claude_code_kwargs(
        model="sonnet",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert kwargs.get("system_prompt") is None
    assert kwargs["prompt"] == "hi"


def test_build_kwargs_multimodal_content_list():
    """When content is a list (multimodal), extract text parts."""
    from agent.claude_code_adapter import build_claude_code_kwargs

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
    ]

    kwargs = build_claude_code_kwargs(model="sonnet", messages=messages)
    assert "What is in this image?" in kwargs["prompt"]
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/Smurphy/.hermes/hermes-agent && python -m pytest tests/test_claude_code_adapter.py::test_build_kwargs_extracts_system_and_user -v`
Expected: FAIL

**Step 3: Write implementation**

Add to `agent/claude_code_adapter.py`:

```python
def _extract_text_from_content(content) -> str:
    """Extract plain text from a message content field.

    Content may be a string or a list of content blocks (multimodal format).
    """
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
    """Build kwargs for run_claude_code() from hermes's internal message list.

    Extracts the system prompt (if any) and the latest user message from the
    OpenAI-format message list. Earlier conversation turns are ignored because
    Claude Code maintains its own conversation state via --resume.
    """
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
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/Smurphy/.hermes/hermes-agent && python -m pytest tests/test_claude_code_adapter.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agent/claude_code_adapter.py tests/test_claude_code_adapter.py
git commit -m "feat: add build_claude_code_kwargs entry point"
```

---

### Task 5: Wire the adapter into `run_agent.py`

**Files:**
- Modify: `run_agent.py` (3 integration points)
- Modify: `hermes_cli/providers.py` (add `determine_api_mode` mapping)

**Step 1: Add `"claude_code"` to the `api_mode` detection in `run_agent.py`**

In `run_agent.py` around line 841, add `"claude_code"` to the valid api_mode set:

```python
# Change:
if api_mode in {"chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse"}:
# To:
if api_mode in {"chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse", "claude_code"}:
```

And add a provider auto-detection block around line 863 (after the bedrock elif):

```python
elif self.provider == "claude-code":
    self.api_mode = "claude_code"
```

**Step 2: Add client initialization for `claude_code` mode**

In `run_agent.py` around line 1076 (the `if self.api_mode == "anthropic_messages":` block), add a new branch before the `else:` at line 1149:

```python
elif self.api_mode == "claude_code":
    # Claude Code CLI adapter — no API client needed, uses subprocess.
    from agent.claude_code_adapter import find_claude_binary
    self._claude_code_binary = find_claude_binary()
    self._claude_code_session_id = None
    self.client = None
    self._client_kwargs = {}
    if not self.quiet_mode:
        print(f"🤖 AI Agent initialized with model: {self.model} (Claude Code CLI)")
```

**Step 3: Add `_build_api_kwargs` branch for `claude_code`**

In `run_agent.py:_build_api_kwargs` (around line 6654), add before the `codex_responses` branch:

```python
if self.api_mode == "claude_code":
    from agent.claude_code_adapter import build_claude_code_kwargs
    return {
        "__claude_code__": True,
        **build_claude_code_kwargs(
            model=self.model,
            messages=api_messages,
            reasoning_config=self.reasoning_config,
            session_id=getattr(self, "_claude_code_session_id", None),
        ),
    }
```

**Step 4: Add execution branch in `_interruptible_api_call`**

In `run_agent.py:_interruptible_api_call` (the `_call` inner function around line 5154), add a branch for claude_code. Since `run_claude_code` is async, we need to run it in an event loop from the thread:

```python
elif self.api_mode == "claude_code":
    from agent.claude_code_adapter import run_claude_code, normalize_claude_code_response
    api_kwargs_copy = {k: v for k, v in api_kwargs.items() if not k.startswith("__")}
    loop = asyncio.new_event_loop()
    try:
        raw_result = loop.run_until_complete(run_claude_code(**api_kwargs_copy))
    finally:
        loop.close()
    # Store session_id for multi-turn resume
    if raw_result.get("session_id"):
        self._claude_code_session_id = raw_result["session_id"]
    result["response"] = normalize_claude_code_response(raw_result)
```

**Step 5: Add response normalization branch**

In `run_agent.py` around line 10704 (the response normalization block), add:

```python
elif self.api_mode == "claude_code":
    # normalize_claude_code_response already called in _interruptible_api_call
    # — the response is already a (SimpleNamespace, str) tuple.
    assistant_message, finish_reason = response
```

**Step 6: Add to `hermes_cli/providers.py`**

In `determine_api_mode`, add a provider check:

```python
if provider == "claude-code":
    return "claude_code"
```

**Step 7: Run existing tests to verify no regressions**

Run: `cd /Users/Smurphy/.hermes/hermes-agent && python -m pytest tests/ -x -q --timeout=30 2>&1 | tail -20`
Expected: PASS (no regressions)

**Step 8: Commit**

```bash
git add run_agent.py hermes_cli/providers.py
git commit -m "feat: wire claude_code adapter into run_agent api_mode dispatch"
```

---

### Task 6: Add integration test (live subprocess, optional)

**Files:**
- Modify: `tests/test_claude_code_adapter.py`

**Step 1: Write a live integration test (skipped by default)**

Add to `tests/test_claude_code_adapter.py`:

```python
import shutil


@pytest.mark.skipif(
    shutil.which("claude") is None,
    reason="claude CLI not installed",
)
@pytest.mark.asyncio
async def test_live_claude_code_roundtrip():
    """Integration test: actually call claude -p and verify the response shape."""
    from agent.claude_code_adapter import run_claude_code, normalize_claude_code_response

    raw = await run_claude_code(
        prompt="Reply with exactly the word 'pong'. Nothing else.",
        model="sonnet",
        disable_tools=True,
        persist_session=False,
        timeout=30.0,
    )

    assert raw["type"] == "result"
    assert raw["stop_reason"] in ("end_turn", "completed")
    assert isinstance(raw["result"], str)
    assert len(raw["result"]) > 0

    msg, finish_reason = normalize_claude_code_response(raw)
    assert msg.role == "assistant"
    assert finish_reason == "stop"
    assert msg.claude_code_session_id is not None


@pytest.mark.skipif(
    shutil.which("claude") is None,
    reason="claude CLI not installed",
)
@pytest.mark.asyncio
async def test_live_claude_code_multi_turn():
    """Integration test: verify multi-turn via --resume."""
    from agent.claude_code_adapter import run_claude_code, normalize_claude_code_response

    # Turn 1
    raw1 = await run_claude_code(
        prompt="Remember the number 7742. Reply with just 'ok'.",
        model="sonnet",
        disable_tools=True,
        timeout=30.0,
    )
    session_id = raw1["session_id"]
    assert session_id

    # Turn 2 — resume
    raw2 = await run_claude_code(
        prompt="What number did I ask you to remember? Reply with just the number.",
        model="sonnet",
        session_id=session_id,
        disable_tools=True,
        timeout=30.0,
    )

    assert "7742" in raw2["result"]
```

**Step 2: Run integration tests**

Run: `cd /Users/Smurphy/.hermes/hermes-agent && python -m pytest tests/test_claude_code_adapter.py::test_live_claude_code_roundtrip -v`
Expected: PASS (if claude CLI is installed)

**Step 3: Commit**

```bash
git add tests/test_claude_code_adapter.py
git commit -m "test: add live integration tests for claude code adapter"
```

---

### Task 7: Add `claude-code` provider to config/CLI

**Files:**
- Modify: `hermes_cli/providers.py` (add provider definition)

**Step 1: Check existing provider definitions**

Read `hermes_cli/providers.py` to find the `HERMES_OVERLAYS` or provider registry and add a `claude-code` entry with:
- `transport`: a new value or reuse existing (depends on the registry pattern)
- No `api_key` required (Claude Code handles its own auth)
- Default model: `"sonnet"` (Claude Code resolves aliases)

The exact code depends on the provider registry structure found in the file. The provider should be usable as:

```bash
hermes --provider claude-code --model sonnet "Hello"
```

**Step 2: Test the CLI invocation manually**

Run: `cd /Users/Smurphy/.hermes/hermes-agent && python hermes_cli/cli.py --provider claude-code --model sonnet "test" 2>&1 | head -20`

**Step 3: Commit**

```bash
git add hermes_cli/providers.py
git commit -m "feat: register claude-code as a provider in hermes CLI"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | `build_claude_code_command` — CLI arg builder | `agent/claude_code_adapter.py`, `tests/test_claude_code_adapter.py` |
| 2 | `run_claude_code` — async subprocess executor | same |
| 3 | `normalize_claude_code_response` — response normalizer | same |
| 4 | `build_claude_code_kwargs` — main entry point | same |
| 5 | Wire into `run_agent.py` + `providers.py` | `run_agent.py`, `hermes_cli/providers.py` |
| 6 | Live integration tests | `tests/test_claude_code_adapter.py` |
| 7 | Register `claude-code` provider in CLI | `hermes_cli/providers.py` |

Total: 1 new file + 2 modified files + 1 test file. ~300 lines of adapter code, ~200 lines of tests.
