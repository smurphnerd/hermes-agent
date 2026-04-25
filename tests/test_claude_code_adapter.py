"""Tests for the Claude Code CLI adapter."""

import asyncio
import json
import shutil
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

import pytest


# ── build_claude_code_command tests ──────────────────────────────────

def test_build_command_basic():
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(prompt="Hello world", model="sonnet")
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "--output-format" in cmd
    idx = cmd.index("--output-format")
    assert cmd[idx + 1] == "json"
    assert "--model" in cmd
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "sonnet"
    assert cmd[-1] == "Hello world"


def test_build_command_with_system_prompt():
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(prompt="Hi", model="opus", system_prompt="You are a helpful robot")
    assert "--system-prompt" in cmd
    idx = cmd.index("--system-prompt")
    assert cmd[idx + 1] == "You are a helpful robot"


def test_build_command_with_resume():
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(prompt="Continue please", model="sonnet", session_id="abc-123-def")
    assert "--resume" in cmd
    idx = cmd.index("--resume")
    assert cmd[idx + 1] == "abc-123-def"


def test_build_command_with_effort():
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(prompt="Think hard", model="opus", effort="high")
    assert "--effort" in cmd
    idx = cmd.index("--effort")
    assert cmd[idx + 1] == "high"


def test_build_command_with_max_budget():
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(prompt="Do something", model="sonnet", max_budget_usd=5.0)
    assert "--max-budget-usd" in cmd
    idx = cmd.index("--max-budget-usd")
    assert cmd[idx + 1] == "5.0"


def test_build_command_disables_tools_when_requested():
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(prompt="Just chat", model="sonnet", disable_tools=True)
    assert "--tools" in cmd
    idx = cmd.index("--tools")
    assert cmd[idx + 1] == ""


def test_build_command_with_allowed_tools():
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(prompt="Edit some files", model="sonnet", allowed_tools=["Bash", "Read", "Edit"])
    assert "--allowedTools" in cmd
    idx = cmd.index("--allowedTools")
    assert cmd[idx + 1] == "Bash,Read,Edit"


def test_build_command_no_session_persistence():
    from agent.claude_code_adapter import build_claude_code_command

    cmd = build_claude_code_command(prompt="One shot", model="sonnet", persist_session=False)
    assert "--no-session-persistence" in cmd


# ── run_claude_code tests ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_claude_code_success():
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

        result = await run_claude_code(prompt="Hello", model="sonnet")

    assert result["result"] == "Hello!"
    assert result["session_id"] == "test-session-123"
    assert result["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_run_claude_code_error():
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
async def test_run_claude_code_empty_output():
    from agent.claude_code_adapter import run_claude_code, ClaudeCodeError

    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"", b"error details"))
    mock_process.returncode = 1

    with patch("agent.claude_code_adapter.asyncio") as mock_asyncio:
        mock_asyncio.create_subprocess_exec = AsyncMock(return_value=mock_process)
        mock_asyncio.subprocess = asyncio.subprocess

        with pytest.raises(ClaudeCodeError, match="empty output"):
            await run_claude_code(prompt="fail", model="sonnet")


@pytest.mark.asyncio
async def test_run_claude_code_invalid_json():
    from agent.claude_code_adapter import run_claude_code, ClaudeCodeError

    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"not json at all", b""))
    mock_process.returncode = 0

    with patch("agent.claude_code_adapter.asyncio") as mock_asyncio:
        mock_asyncio.create_subprocess_exec = AsyncMock(return_value=mock_process)
        mock_asyncio.subprocess = asyncio.subprocess

        with pytest.raises(ClaudeCodeError, match="Failed to parse"):
            await run_claude_code(prompt="fail", model="sonnet")


# ── normalize_claude_code_response tests ─────────────────────────────

def test_normalize_response_basic():
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
    _, reason = normalize_claude_code_response(raw)
    assert reason == "stop"


def test_normalize_response_max_tokens():
    from agent.claude_code_adapter import normalize_claude_code_response

    raw = {"result": "partial...", "stop_reason": "max_tokens", "usage": {}}
    _, reason = normalize_claude_code_response(raw)
    assert reason == "length"


def test_normalize_response_preserves_session_id():
    from agent.claude_code_adapter import normalize_claude_code_response

    raw = {"result": "hi", "stop_reason": "end_turn", "session_id": "s-123", "usage": {}}
    msg, _ = normalize_claude_code_response(raw)
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


# ── build_claude_code_kwargs tests ───────────────────────────────────

def test_build_kwargs_wraps_system_prompt_in_user_message():
    """Hermes's system prompt is NOT passed via --system-prompt (which would
    replace Claude Code's default and trip Anthropic's quota checks). Instead
    it's prepended to the user message wrapped in <system-instructions> tags.
    """
    from agent.claude_code_adapter import build_claude_code_kwargs

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello there"},
    ]
    kwargs = build_claude_code_kwargs(model="sonnet", messages=messages)
    assert kwargs["model"] == "sonnet"
    assert "system_prompt" not in kwargs
    assert "<system-instructions>" in kwargs["prompt"]
    assert "You are helpful." in kwargs["prompt"]
    assert "</system-instructions>" in kwargs["prompt"]
    assert kwargs["prompt"].endswith("Hello there")


def test_build_kwargs_multi_turn_extracts_latest_user():
    from agent.claude_code_adapter import build_claude_code_kwargs

    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "First reply"},
        {"role": "user", "content": "Second message"},
    ]
    kwargs = build_claude_code_kwargs(model="opus", messages=messages)
    assert kwargs["prompt"].endswith("Second message")
    assert "Be concise." in kwargs["prompt"]
    assert "system_prompt" not in kwargs


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


def test_build_kwargs_inlines_prior_turns_as_transcript():
    """Prior user/assistant turns must be inlined into the prompt.
    Hermes is the source of truth for transcript state — relying on
    --resume <session_id> drops context across agent re-spawn.
    """
    from agent.claude_code_adapter import build_claude_code_kwargs

    messages = [
        {"role": "user", "content": "what's 2+2"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "and times 3"},
    ]
    kwargs = build_claude_code_kwargs(model="sonnet", messages=messages)
    assert "<conversation-history>" in kwargs["prompt"]
    assert "User: what's 2+2" in kwargs["prompt"]
    assert "Assistant: 4" in kwargs["prompt"]
    assert "</conversation-history>" in kwargs["prompt"]
    assert kwargs["prompt"].endswith("and times 3")
    # Latest user turn must NOT also appear inside the history block.
    history_block = kwargs["prompt"].split("</conversation-history>")[0]
    assert "and times 3" not in history_block


def test_build_kwargs_does_not_pass_session_id():
    """Adapter no longer forwards session_id — context is inlined instead."""
    from agent.claude_code_adapter import build_claude_code_kwargs

    kwargs = build_claude_code_kwargs(
        model="sonnet",
        messages=[{"role": "user", "content": "hi"}],
        session_id="some-old-session",
    )
    assert "session_id" not in kwargs


def test_build_kwargs_disables_tools_by_default():
    """Hermes provides its own tool layer; Claude Code's internal tools
    are disabled by default to avoid multi-minute agentic runs on chat replies.
    """
    from agent.claude_code_adapter import build_claude_code_kwargs

    kwargs = build_claude_code_kwargs(
        model="sonnet",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert kwargs.get("disable_tools") is True


def test_build_kwargs_tools_opt_in_via_env(monkeypatch):
    from agent.claude_code_adapter import build_claude_code_kwargs

    monkeypatch.setenv("HERMES_CLAUDE_CODE_ENABLE_TOOLS", "1")
    kwargs = build_claude_code_kwargs(
        model="sonnet",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert "disable_tools" not in kwargs


def test_build_kwargs_explicit_disable_tools_overrides_env(monkeypatch):
    from agent.claude_code_adapter import build_claude_code_kwargs

    monkeypatch.setenv("HERMES_CLAUDE_CODE_ENABLE_TOOLS", "1")
    kwargs = build_claude_code_kwargs(
        model="sonnet",
        messages=[{"role": "user", "content": "hi"}],
        disable_tools=True,
    )
    assert kwargs.get("disable_tools") is True


def test_build_kwargs_multimodal_content_list():
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


# ── Live integration tests (skipped if claude CLI not installed) ─────

@pytest.mark.skipif(
    shutil.which("claude") is None,
    reason="claude CLI not installed",
)
@pytest.mark.asyncio
async def test_live_claude_code_roundtrip():
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
    from agent.claude_code_adapter import run_claude_code

    raw1 = await run_claude_code(
        prompt="Remember the number 7742. Reply with just 'ok'.",
        model="sonnet",
        disable_tools=True,
        timeout=30.0,
    )
    session_id = raw1["session_id"]
    assert session_id

    raw2 = await run_claude_code(
        prompt="What number did I ask you to remember? Reply with just the number.",
        model="sonnet",
        session_id=session_id,
        disable_tools=True,
        timeout=30.0,
    )
    assert "7742" in raw2["result"]
