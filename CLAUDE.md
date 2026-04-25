# CLAUDE.md — `sean/local` fork: Claude Code CLI adapter

This fork adds a **Claude Code CLI adapter** to hermes-agent, letting hermes route conversations through the local `claude -p` binary (Claude Code's headless mode) instead of calling Anthropic's API directly. Claude Code handles its own auth (OAuth login), tools, and session state — the adapter is a thin subprocess wrapper.

This file scopes everything you need to know to work on the adapter itself. For the rest of hermes, see `AGENTS.md`.

## Where the code lives

- `agent/claude_code_adapter.py` — the adapter. Builds the `claude -p` argv, spawns the subprocess, parses the JSON result, normalizes it into the `SimpleNamespace` shape hermes expects elsewhere.
- `hermes_cli/providers.py` — registers the `claude-code` provider with `transport="claude_code"` (see `HermesOverlay` at L158).
- `hermes_cli/runtime_provider.py` — resolves `provider="claude-code"` → `api_mode="claude_code"` (L858). `_VALID_API_MODES` at L137 includes `claude_code`.
- `hermes_cli/model_switch.py` — credential-presence check falls back to "is the `claude` binary on PATH?" for this provider (L948–L961).
- `run_agent.py` — main agent loop has `api_mode == "claude_code"` branches at the call sites listed below.
- `tests/test_claude_code_adapter.py` — unit tests for the adapter.

## Wire protocol

`claude -p --output-format json --model <m> [--resume <sid>] <prompt>` → stdout is one JSON blob:

```json
{
  "result": "...assistant text...",
  "stop_reason": "end_turn",
  "session_id": "...",
  "usage": { "input_tokens": N, "output_tokens": N, "cache_creation_input_tokens": N, "cache_read_input_tokens": N },
  "total_cost_usd": 0.0,
  "is_error": false
}
```

`is_error: true` triggers `ClaudeCodeError` and a replay-script dump (see below). Multi-turn uses `--resume <session_id>` returned from the previous call; hermes stashes it on `self._claude_code_session_id` (`run_agent.py:1154`, set at L5195).

Streaming is not supported by `claude -p`, so the streaming path is routed to the non-streaming path (commit `420500c6`). Don't try to wire SSE into this adapter — it would mean re-implementing on top of `--output-format stream-json`, which is a separate effort.

## The system-prompt gotcha (important)

**Do not pass hermes's system prompt via `--system-prompt`.** Doing so replaces Claude Code's default identity/tool instructions, and Anthropic's quota check then rejects the call with `"out of extra usage"` *before it reaches the API* — observed on Pro OAuth with Opus 4.6.

The workaround in `build_claude_code_kwargs` prepends the hermes system prompt to the user message wrapped in `<system-instructions>...</system-instructions>` tags. Claude Code keeps its own default system prompt and treats the tagged block as a secondary instruction layer, which keeps the call on the standard tier.

The `--system-prompt` flag is still implemented in `build_claude_code_command` for completeness, but `build_claude_code_kwargs` no longer sets it. If you re-introduce it, expect billing-tier rejections on OAuth accounts.

## Conversation history is inlined, not resumed

The adapter does **not** use `--resume <session_id>`. The full prior transcript (every `user`/`assistant` turn except the latest user one) is inlined into the prompt as `<conversation-history>...</conversation-history>` before the latest user message.

Rationale: hermes is the source of truth for transcript state — it persists per-thread to SQLite (`gateway/session.py:1171`). The previous design stored `_claude_code_session_id` only in memory on the agent instance and forwarded just the latest user message, expecting `--resume` to carry context. On any agent re-spawn (gateway restart, idle, error) the session_id was lost and discord threads silently went contextless. Inlining transcript every call eliminates that cross-call dependency.

Trade-off: no claude-side prompt-cache reuse across hermes turns. Hermes's own caching layer compensates.

## Tools are disabled by default

`build_claude_code_kwargs` defaults `disable_tools=True` (which becomes `--tools ""` on the CLI). Hermes already provides its own tool layer; leaving Claude Code's internal tools enabled turned every chat reply into an agentic run (Read/Bash/Glob/etc. fired before the final text), routinely taking 5–10 minutes and tripping the 300s watchdog.

Opt back in by setting `HERMES_CLAUDE_CODE_ENABLE_TOOLS=1` (or by passing `disable_tools=False` explicitly from a caller).

## Auth env-var stripping (important)

When hermes spawns `claude`, the subprocess inherits hermes's env. If the user previously used the plain `anthropic` provider, `ANTHROPIC_API_KEY` / `CLAUDE_CODE_OAUTH_TOKEN` / etc. will be set, and `claude` would use *those* (typically an exhausted account) instead of its own saved OAuth login.

`_env_for_claude_subprocess` (L61) strips these before spawn:

```
ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_TOKEN,
CLAUDE_CODE_OAUTH_TOKEN, CLAUDE_CODE_API_KEY
```

If you add a new auth-related env var that `claude` reads, add it to `_CLAUDE_AUTH_ENV_VARS`.

## Failure-mode debugging

On `is_error: true`, the adapter:

1. Logs the full raw result (including `api_error_status`, `request_id`, `service_tier`) — `claude_code_adapter.py:229`.
2. Writes a replay shell script to `<HERMES_HOME>/logs/claude_code_last_failing_command.sh` (`_dump_failing_command`). Run it from your shell to reproduce the exact invocation hermes sent and compare against an interactive `claude -p` run.

On stale-timeout (default 300s, see `_compute_non_stream_stale_timeout` in `run_agent.py`), the watchdog calls `process.kill()` on the live `claude -p` subprocess. The handle is plumbed via the `process_holder` dict passed to `run_claude_code`; the watchdog reads it from `request_client_holder["process"]`. Without this, the subprocess kept running past the timeout and its eventual output was dropped on the floor.

The command logger (L165–L183) redacts long values (`<N chars>`) so the system-prompt block doesn't fill the log.

## Response normalization

`normalize_claude_code_response` (L244) returns `(msg, finish_reason)` where `msg` is a `SimpleNamespace` that mimics an OpenAI/Anthropic message object: `role`, `content`, `tool_calls=None`, `reasoning=None`, plus a `usage` namespace and `claude_code_session_id`. Tool calls are always `None` because Claude Code executes its own tools internally — hermes only sees the final text result.

`_STOP_REASON_MAP` collapses `end_turn` / `stop_sequence` / `completed` → `"stop"` and `max_tokens` → `"length"` to match the OpenAI finish-reason vocabulary the rest of hermes uses. There's a separate finish-reason extraction path in `run_agent.py` that handles the tuple return shape (see commit `5664370c`).

## Call sites in `run_agent.py`

- `L841, L865` — api_mode validation / setting.
- `L1151–1154` — init: `find_claude_binary()`, init `_claude_code_session_id = None`.
- `L5186–5196` — main inference call: build kwargs → `run_claude_code` → normalize → store session_id.
- `L6383, L6688` — branches that skip OpenAI/Anthropic client setup (subprocess, no client).
- `L8645` — claude_code is treated as anthropic-shaped for the surrounding logic.
- `L9354, L9526, L10771` — response-handling branches that consume the already-normalized SimpleNamespace.

When adding new logic to the inference loop, search for `claude_code` and check whether your new branch needs to handle it (usually: same as `anthropic_messages`, but no streaming and no tool_calls).

## Testing

```
uv run pytest tests/test_claude_code_adapter.py -xvs
```

The tests stub `asyncio.create_subprocess_exec`. They do not actually invoke `claude`. End-to-end: install Claude Code, `claude login`, then run hermes with `--provider claude-code --model claude-opus-4-7`.

## Conventions for this fork

- Keep the adapter narrow. It's a subprocess wrapper, not a re-implementation of Claude Code's protocol. Anything that requires deep integration (interactive tools, streaming, MCP) belongs upstream in Claude Code, not here.
- Preserve the redacted command logging — the failure modes (quota, auth, model name) are nearly impossible to debug without it.
- Never re-introduce `--system-prompt` for hermes's full prompt. Tag-wrap it in the user message instead.
- When debugging an `is_error` failure, *always* run the dumped replay script before changing adapter code — most failures are auth/quota issues, not adapter bugs.
