"""ultron_rescue: turn a tool call the model wrote as text back into a real tool call.

LiteLLM proxy hook for ultron's local tiers. Small models sometimes stop calling tools and write
the command they meant to run in a ```bash block instead:

    The smoke test script is synced. Now I'll look up its last commit hash.

    ```bash
    cd ~/src/game && git log -1 --format=%H -- Scripts/smoke/run_smoke.sh | head -10
    ```

The client sees a text-only reply, ends the turn, and nothing runs. The model then copies that
reply on every later turn: a coding agent on the sonnet tier (Qwen3.5-9B) repeated one for 6 hours
after a loop_breaker force note (2026-09-30). Replays of that transcript: a corrective note fixed 0/4,
mtplx tool_choice "required" (a prompt hint, not constrained decoding) 0/4.

This hook rewrites such a reply on its way out: the trailing block becomes a tool_use for the
request's shell tool (Bash, bash, terminal…), or for a declared tool when the block reads
`ToolName {json}`. The block is dropped from the text and stop_reason becomes tool_use, so the
transcript shows a normal tool call, which also stops the copying. Only when all hold:

  - a local tier (not cloud/*) and the request declares tools
  - the reply made no tool call and ended normally (end_turn / stop)
  - the text ends with its only fenced block, a shell one (bash/sh/shell/zsh/console or none)
  - the text before it announces the action ("Now I'll…", "Let me…", "I need to…")

Covers /v1/messages, streaming and not; chat/completions passes through untouched. Streaming
holds a text block back from its first ``` until the block can no longer be rescued (or ends),
so other code blocks still arrive, a little later. The client's own permission checks still
apply to the rescued call. Mode: ~/.ultron/rescue-mode or ULTRON_RESCUE_MODE=enforce|shadow|off
(default enforce). Per-request opt-out: header `x-rescue: off`. Log: ~/.litellm/rescue.jsonl.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any

SHELL_TOOLS = ("bash", "shell", "terminal", "run_shell_command", "execute_command", "exec")
SHELL_LANGS = {"", "bash", "sh", "shell", "zsh", "console", "terminal"}
FENCE = "```"
# The model said it was about to act, not that it was showing an example.
ANNOUNCE_RE = re.compile(
    r"\b(I'll|I will|I'm going to|I am going to|let me|let's|now I|next,? I|I need to|running|checking)\b",
    re.I,
)
ANNOUNCE_WINDOW = 400  # chars of narration before the block that are searched
DONE_STOPS = ("end_turn", "stop")

LOG_PATH = os.path.expanduser(os.environ.get("ULTRON_RESCUE_LOG", "~/.litellm/rescue.jsonl"))
MODE_FILE = os.path.expanduser(os.environ.get("ULTRON_RESCUE_MODE_FILE", "~/.ultron/rescue-mode"))


# ----------------------------------------------------------------------------- detection

def split_trailing_block(text: str) -> tuple[str, str, str] | None:
    """(narration, lang, body) when the text ends with its only fenced block, else None."""
    if text.count(FENCE) != 2:
        return None
    start = text.index(FENCE)
    tail = text[start:].rstrip()
    if not tail.endswith(FENCE) or len(tail) < 2 * len(FENCE):
        return None
    inner = tail[len(FENCE):-len(FENCE)]
    nl = inner.find("\n")
    if nl < 0:  # ```cmd``` on one line is inline code, not a block
        return None
    return text[:start], inner[:nl].strip().lower(), inner[nl + 1:].strip()


def tool_specs(tools: Any) -> dict[str, dict[str, Any]]:
    """name -> input schema, for Anthropic ({name, input_schema}) or OpenAI ({function: {...}}) tools."""
    out: dict[str, dict[str, Any]] = {}
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t["function"] if isinstance(t.get("function"), dict) else t
        if fn.get("name"):
            out[str(fn["name"])] = fn.get("input_schema") or fn.get("parameters") or {}
    return out


def shell_tool(specs: dict[str, dict[str, Any]]) -> tuple[str, str] | None:
    """(tool name, command argument) of the request's shell tool."""
    for name, schema in specs.items():
        if name.lower() not in SHELL_TOOLS:
            continue
        props = schema.get("properties") or {}
        if "command" in props or not props:
            return name, "command"
        required = [k for k in schema.get("required") or [] if (props.get(k) or {}).get("type") == "string"]
        if len(required) == 1:
            return name, required[0]
    return None


def tool_call_for(text: str, tools: Any) -> tuple[str, str, dict[str, Any]] | None:
    """(narration, tool name, input) when this text reply should have been a tool call."""
    parts = split_trailing_block(text)
    if parts is None:
        return None
    narration, lang, body = parts
    if lang not in SHELL_LANGS or not body or not ANNOUNCE_RE.search(narration[-ANNOUNCE_WINDOW:]):
        return None
    specs = tool_specs(tools)
    head = body.split(None, 1)[0]
    if head in specs and head.lower() not in SHELL_TOOLS:  # `ListAgents {"filter": "running"}`
        rest = body[len(head):].strip()
        if rest.startswith("(") and rest.endswith(")"):
            rest = rest[1:-1].strip()
        if not rest:
            return narration, head, {}
        try:
            args = json.loads(rest)
        except ValueError:
            return None
        return (narration, head, args) if isinstance(args, dict) else None
    shell = shell_tool(specs)
    if shell is None:
        return None
    if lang == "console":
        body = "\n".join(line[2:] if line.startswith("$ ") else line for line in body.splitlines())
    return narration, shell[0], {shell[1]: body}


def _rescuable_tail(tail: str) -> bool:
    """Could text that starts at its first ``` still end as a single trailing block?"""
    n = tail.count(FENCE)
    if n > 2:
        return False
    if n == 2:
        return not tail[tail.index(FENCE, len(FENCE)) + len(FENCE):].strip()
    return True


def _new_id() -> str:
    return "toolu_" + uuid.uuid4().hex[:24]


# ----------------------------------------------------------------------------- streaming (/v1/messages SSE)

def _sse(ev: dict[str, Any]) -> str:
    return f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"


def _parse_sse(text: str) -> list[dict[str, Any]] | None:
    """Events in one chunk of Anthropic SSE, or None if it isn't that."""
    events = []
    for block in text.split("\n\n"):
        data = [line[5:].strip() for line in block.splitlines() if line.startswith("data:")]
        if not data:
            if block.strip() and not all(line.startswith(("event:", ":")) for line in block.splitlines() if line):
                return None
            continue
        try:
            ev = json.loads("\n".join(data))
        except ValueError:
            return None
        if not isinstance(ev, dict) or "type" not in ev:
            return None
        events.append(ev)
    return events


class StreamRewriter:
    """Rewrites one /v1/messages SSE stream. feed() takes a chunk and returns the chunks to send
    in its place; finish() returns whatever is still held when the upstream stream ends."""

    def __init__(self, tools: Any, enforce: bool) -> None:
        self.tools, self.enforce = tools, enforce
        self.text: dict[int, str] = {}   # text block index -> its full text so far
        self.sent: dict[int, int] = {}   # chars of it already passed on
        self.free: set[int] = set()      # text blocks that can no longer be rescued
        self.pending: int | None = None  # text block whose stop event (and held tail) wait for the verdict
        self.max_index = -1
        self.tool_used = False
        self.passthrough = False
        self.as_bytes = True
        self.rescued: tuple[str, str, dict[str, Any]] | None = None  # (narration, tool, input) for the log
        self.applied = False

    def _out(self, events: list[dict[str, Any]]) -> list[Any]:
        return [(_sse(ev).encode() if self.as_bytes else _sse(ev)) for ev in events]

    def _release(self, idx: int) -> list[dict[str, Any]]:
        """Text of block idx not yet passed on, as one delta event."""
        rest = self.text[idx][self.sent[idx]:]
        self.sent[idx] = len(self.text[idx])
        return [{"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": rest}}] if rest else []

    def _flush_pending(self) -> list[dict[str, Any]]:
        if self.pending is None:
            return []
        idx, self.pending = self.pending, None
        return self._release(idx) + [{"type": "content_block_stop", "index": idx}]

    def _on_event(self, ev: dict[str, Any]) -> list[dict[str, Any]]:
        t = ev.get("type")
        idx = ev.get("index")
        if t == "content_block_start":
            out = self._flush_pending()
            self.max_index = max(self.max_index, int(idx or 0))
            block = ev.get("content_block") or {}
            if block.get("type") == "tool_use":
                self.tool_used = True
            elif block.get("type") == "text":
                self.text[idx], self.sent[idx] = block.get("text") or "", 0
                block["text"] = ""
                out.append(ev)
                return out + self._text_progress(idx)
            return out + [ev]
        if t == "content_block_delta" and idx in self.text and (ev.get("delta") or {}).get("type") == "text_delta":
            self.text[idx] += ev["delta"].get("text") or ""
            return self._text_progress(idx)
        if t == "content_block_stop" and idx in self.text:
            if self.text[idx][self.sent[idx]:] and idx not in self.free:
                self.pending = idx  # held tail may be a trailing shell block: wait for message_delta
                return []
            return self._release(idx) + [ev]
        if t == "message_delta":
            return self._verdict(ev)
        if t == "ping":
            return [ev]
        return self._flush_pending() + [ev]

    def _text_progress(self, idx: int) -> list[dict[str, Any]]:
        """Pass on the part of block idx that can't be the start of a rescued block."""
        full, sent = self.text[idx], self.sent[idx]
        if idx in self.free:
            return self._release(idx)
        fence = full.find(FENCE)
        if fence >= 0:
            if not _rescuable_tail(full[fence:]):
                self.free.add(idx)
                return self._release(idx)
            safe = fence
        else:
            safe = len(full.rstrip("`"))  # a fence may be arriving split across deltas
        if safe <= sent:
            return []
        self.sent[idx] = safe
        return [{"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": full[sent:safe]}}]

    def _verdict(self, ev: dict[str, Any]) -> list[dict[str, Any]]:
        stop = (ev.get("delta") or {}).get("stop_reason")
        idx = self.pending
        call = None
        if idx is not None and not self.tool_used and stop in DONE_STOPS:
            call = tool_call_for(self.text[idx], self.tools)
        if call is None:
            return self._flush_pending() + [ev]
        self.rescued = call
        if not self.enforce:
            return self._flush_pending() + [ev]
        self.pending, self.applied = None, True
        _, name, args = call
        new = self.max_index + 1
        ev = {**ev, "delta": {**(ev.get("delta") or {}), "stop_reason": "tool_use"}}
        return [
            {"type": "content_block_stop", "index": idx},
            {"type": "content_block_start", "index": new,
             "content_block": {"type": "tool_use", "id": _new_id(), "name": name, "input": {}}},
            {"type": "content_block_delta", "index": new,
             "delta": {"type": "input_json_delta", "partial_json": json.dumps(args)}},
            {"type": "content_block_stop", "index": new},
            ev,
        ]

    def feed(self, chunk: Any) -> list[Any]:
        if self.passthrough:
            return [chunk]
        if isinstance(chunk, (bytes, bytearray)):
            self.as_bytes, text = True, bytes(chunk).decode("utf-8", "replace")
        elif isinstance(chunk, str):
            self.as_bytes, text = False, chunk
        else:
            text = None
        events = _parse_sse(text) if text is not None else None
        if events is None:  # not Anthropic SSE: stop interfering, hand back what's held
            self.passthrough = True
            return self._out(self._flush_pending()) + [chunk]
        out: list[dict[str, Any]] = []
        for ev in events:
            out.extend(self._on_event(ev))
        return self._out(out)

    def finish(self) -> list[Any]:
        return self._out(self._flush_pending())


# ----------------------------------------------------------------------------- non-streaming

def _plain(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    return dict(getattr(block, "__dict__", {}))


def rewrite_response(resp: Any, tools: Any, enforce: bool) -> tuple[str, str, dict[str, Any]] | None:
    """Rescue a non-streaming /v1/messages response in place; returns the call found."""
    if not isinstance(resp, dict) or resp.get("stop_reason") not in DONE_STOPS:
        return None
    content = [_plain(b) for b in resp.get("content") or []]
    if not content or any(b.get("type") == "tool_use" for b in content) or content[-1].get("type") != "text":
        return None
    call = tool_call_for(content[-1].get("text") or "", tools)
    if call is None or not enforce:
        return call
    narration, name, args = call
    if narration.strip():
        content[-1] = {**content[-1], "text": narration.rstrip()}
    else:
        content.pop()
    content.append({"type": "tool_use", "id": _new_id(), "name": name, "input": args})
    resp["content"] = content
    resp["stop_reason"] = "tool_use"
    return call


# ----------------------------------------------------------------------------- LiteLLM glue

def current_mode() -> str:
    """Mode file (live, no restart) wins over ULTRON_RESCUE_MODE from the env."""
    try:
        v = open(MODE_FILE).read().strip().lower()
    except OSError:
        v = ""
    return v if v in ("enforce", "shadow", "off") else os.environ.get("ULTRON_RESCUE_MODE", "enforce").lower()


def _headers(data: dict[str, Any]) -> dict[str, Any]:
    h = (data.get("proxy_server_request") or {}).get("headers") or {}
    return {str(k).lower(): v for k, v in h.items()} if isinstance(h, dict) else {}


def applies(data: dict[str, Any]) -> bool:
    """Local tier, tools declared, not opted out."""
    model = str(data.get("model") or "")
    if not data.get("tools") or model.startswith(("cloud/", "media/")):
        return False
    return str(_headers(data).get("x-rescue", "")).lower() != "off"


def _log(data: dict[str, Any], call: tuple[str, str, dict[str, Any]], mode: str, applied: bool, stream: bool) -> None:
    h = _headers(data)
    key = f"cc:{h['x-claude-code-session-id']}:{h.get('x-claude-code-agent-id') or 'main'}" if h.get("x-claude-code-session-id") else None
    entry = {
        "ts": time.time(), "mode": mode, "applied": applied, "model": data.get("model"), "stream": stream,
        "key": key, "call_id": data.get("litellm_call_id"), "tool": call[1],
        "input": json.dumps(call[2])[:300], "narration": re.sub(r"\s+", " ", call[0])[-160:],
    }
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass


try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # tests without litellm installed
    CustomLogger = object  # type: ignore[misc,assignment]


class UltronRescue(CustomLogger):  # type: ignore[misc,valid-type]
    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        mode = current_mode()
        if mode == "off" or not applies(request_data):
            async for chunk in response:
                yield chunk
            return
        rw = StreamRewriter(request_data.get("tools"), mode == "enforce")
        async for chunk in response:
            try:
                out = rw.feed(chunk)
            except Exception:  # the proxy must never break a stream because of this hook
                rw.passthrough, out = True, [chunk]
            for c in out:
                yield c
        try:
            tail = rw.finish()
        except Exception:
            tail = []
        for c in tail:
            yield c
        if rw.rescued is not None:
            _log(request_data, rw.rescued, mode, rw.applied, True)

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        try:
            mode = current_mode()
            if mode != "off" and not data.get("stream") and applies(data):
                call = rewrite_response(response, data.get("tools"), mode == "enforce")
                if call is not None:
                    _log(data, call, mode, mode == "enforce", False)
        except Exception:
            pass
        return response


proxy_handler_instance = UltronRescue()
