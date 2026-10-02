"""loop_breaker: stop agents from repeating the same tool call forever.

LiteLLM proxy pre-call hook for ultron (LiteLLM :4000 -> llama-swap -> mtplx tiers).
Every request carries the agent's whole history, so the hook is stateless: it reads the
trailing tool steps, and when the model keeps making the same call and getting the same
result back, it escalates:

    warn   append a note at the END of the history (prompt prefix / KV cache untouched)
    force  a firmer note naming the repeated calls: don't repeat them, change course with a
           different call (tools stay available; see apply()). It used to say "reply in text",
           which sent a sonnet-tier agent into a 6 h text-only stall (2026-09-30; see ultron_rescue)
    stop   answer the turn itself via mock_response; no backend call, no HTTP error
           (clients retry errors)

A "repeat" = same tool + same canonical args (keys sorted, whitespace trimmed, digits KEPT:
node ids / offsets / screenshot indexes change legitimately) AND the same result after
masking volatile fields (timestamps, durations, execution counters, uuids, hex ids,
countdowns). Any change in the call or its result ends the run, and so does a real user turn:
once enforcement has handed the turn back, the next prompt starts a fresh count (without that,
an enforced session could never recover). A model that loops again is caught again.

Thresholds come from ~430k real tool calls (two Unreal Engine MCP harnesses, pi, Claude Code,
Hermes): legitimate identical call+result runs never exceeded 3 for ordinary tools or 12 for
polling. Mode: ~/.ultron/loop-breaker-mode (set from the Wanda panel, read per request) or
LOOP_BREAKER_MODE=enforce|shadow|off (default enforce). Per-request opt-out:
header `x-loop-breaker: off` or metadata {"loop_breaker": "off"}.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any

# ----------------------------------------------------------------------------- thresholds

LEVELS = ("none", "warn", "force", "stop")
THRESHOLDS = {  # run length at which each level starts: (warn, force, stop)
    "default": (4, 6, 8),
    "poll": (10, 16, 20),
    "error": (3, 4, 6),
    "cycle": (5, 6, 8),  # full repetitions of a 2-4 step cycle
    "poll_cycle": (10, 16, 20),  # cycle containing a poll/wait step (hourly check + wakeup)
}
MAX_STEPS = 256  # trailing tool steps inspected; enough for the stop threshold of any rule

# Tools/args that legitimately repeat while waiting on something (status polls, PIE input,
# waits). Matching only raises the thresholds; a stuck poll is still stopped at 20.
POLL_RE = re.compile(
    r"poll|status|is_in_play|pie_inject|pie_call_function|pie_get_object|get_stat_group"
    r"|bt_state|search_logs|tail_log|get_scenario|import_status|capture_verified"
    r"|\bsleep\b|\bwait|wakeup|monitor|taskoutput|bashoutput|listagents|\btail\b|git (?:log|status)"
    r"|\bpump\b",  # FR harness drive scripts: `<script>.py pump` returns busy until done
    re.I,
)
# A result that says the work is still in progress marks a poll, whatever the tool is.
_BUSY = r"(busy|running|pending|queued|in[_ -]?progress|compiling|building|loading|waiting|importing|baking)"
BUSY_RE = re.compile(
    r"""["']?(state|status|phase)["']?\s*[:=]\s*["']?""" + _BUSY + r"\b"  # {"state": "busy"}
    r"|\b(still|currently)\s+" + _BUSY + r"\b"  # "still building: True" (a navmesh poll)
    r"|\b(is_)?" + _BUSY + r"""["']?\s*[:=]\s*true\b""",  # building: True / "is_compiling": true
    re.I,
)
# Args that describe a call rather than define it (Claude Code Bash `description` varied
# across 27 values in a 95-call `git status` loop).
SHELL_TOOLS = {"bash", "shell", "terminal", "run_shell_command", "execute_command", "exec"}
COSMETIC_ARGS = {"description", "reason", "explanation", "thought", "summary"}

# Result keys whose values change on every call without meaning progress.
VOLATILE_KEY_RE = re.compile(
    r"timestamp|(^|_)(ts|time|at|ms|elapsed|duration|took|latency|uptime|now|date)($|_)"
    r"|_time$|^time_|_ms$|execution_count|execution_time|request_id|trace_id|saved_packages"
    r"|^kernel$|^pid$|^id$",
    re.I,
)
VOLATILE_TEXT = [
    (re.compile(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d(:\d\d(\.\d+)?)?(Z|[+-]\d\d:?\d\d)?"), "<ts>"),
    (re.compile(r"\b\d{8}[-_T]?\d{6}\b"), "<ts>"),  # 20260927-101112 file stamps
    (re.compile(r"\b\d{1,2}:\d\d:\d\d(\.\d+)?\b"), "<clock>"),
    (re.compile(r"\b1[6-9]\d{8}(\.\d+)?\b|\b1[6-9]\d{11}\b"), "<epoch>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<uuid>"),
    (re.compile(r"\b0x[0-9a-f]{6,}\b|\b[0-9a-f]{16,}\b", re.I), "<hex>"),
    (re.compile(r"\b(in|after|for|took|elapsed:?)\s+\d+(\.\d+)?\s*(ms|s|sec|seconds|m|min)\b", re.I), r"\1 <dur>"),
    (re.compile(r"\b\d+(\.\d+)?\s*(ms|s)\b"), "<dur>"),
]

NOTE_PREFIX = "[ultron loop-breaker]"
LOG_PATH = os.path.expanduser(os.environ.get("LOOP_BREAKER_LOG", "~/.litellm/loop-breaker.jsonl"))


# ----------------------------------------------------------------------------- history parsing

class Step:
    """One assistant turn that made tool calls, plus the results those calls got.

    Plain class, not a dataclass: LiteLLM imports callback modules without registering them
    in sys.modules, and @dataclass fails there."""

    def __init__(self, calls: list[tuple[str, str, str]] | None = None) -> None:
        self.calls = calls or []  # (id, name, canonical args)
        self.results: dict[str, tuple[str, bool]] = {}  # id -> (raw text, is_error)
        self._rkey: str | None = None
        self.boundary = False  # marks a real user turn; detection only looks after the last one

    @property
    def names(self) -> list[str]:
        return [n for _, n, _ in self.calls]

    def complete(self) -> bool:
        return bool(self.calls) and all(cid in self.results for cid, _, _ in self.calls)

    def call_key(self) -> str:
        return "\x1e".join(sorted(f"{n}\x1f{a}" for _, n, a in self.calls))

    def result_key(self) -> str:
        if self._rkey is not None:
            return self._rkey
        parts = sorted(
            f"{n}\x1f{'E' if self.results[c][1] else 'R'}\x1f{normalize_result(self.results[c][0])}"
            for c, n, _ in self.calls
        )
        self._rkey = hashlib.sha1("\x1e".join(parts).encode()).hexdigest()
        return self._rkey

    def all_errors(self) -> bool:
        return all(self.results[c][1] for c, _, _ in self.calls)

    def is_poll(self) -> bool:
        return any(POLL_RE.search(n) or POLL_RE.search(a) for _, n, a in self.calls) or any(
            BUSY_RE.search(text[:4000]) for text, _ in self.results.values()
        )


def _canon_str(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.replace("\r\n", "\n").split("\n")).strip()


def _canon(value: Any, drop: set[str]) -> Any:
    if isinstance(value, dict):
        return {k: _canon(v, set()) for k, v in sorted(value.items()) if k not in drop}
    if isinstance(value, list):
        return [_canon(v, set()) for v in value]
    if isinstance(value, str):
        return _canon_str(value)
    return value


def canonical_args(name: str, args: Any) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            return _canon_str(args)
    drop = COSMETIC_ARGS if (name.lower() in SHELL_TOOLS or "wakeup" in name.lower()) else set()
    return json.dumps(_canon(args, drop), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_volatile(v) for k, v in sorted(value.items()) if not VOLATILE_KEY_RE.search(k)}
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    return value


def normalize_result(text: str) -> str:
    stripped = text.strip()
    if stripped[:1] in "{[":
        try:
            text = json.dumps(_strip_volatile(json.loads(stripped)), sort_keys=True)
        except ValueError:
            pass
    # Tool-loop warnings that clients append (Hermes) differ per repeat; drop them.
    text = re.sub(r"\[Tool loop warning:[^\]]*\]", "", text)
    text = text.replace("\r\n", "\n")
    for rx, sub in VOLATILE_TEXT:
        text = rx.sub(sub, text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _content_text(content: Any) -> str:
    """Flatten OpenAI/Anthropic content (str or list of blocks) to comparable text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict):
                t = block.get("type")
                if t in ("text", "input_text", "output_text"):
                    out.append(block.get("text", ""))
                elif t == "tool_result":
                    out.append(_content_text(block.get("content")))
                elif t in ("image", "image_url"):
                    src = block.get("source") or block.get("image_url") or {}
                    data = (src.get("data") or src.get("url") or "") if isinstance(src, dict) else str(src)
                    out.append("<image:" + hashlib.sha1(str(data).encode()).hexdigest()[:12] + ">")
                else:
                    out.append(json.dumps(block, sort_keys=True, default=str))
        return "\n".join(out)
    return json.dumps(content, sort_keys=True, default=str)


def extract_steps(messages: list[dict[str, Any]]) -> list[Step]:
    """Tool steps in order, from OpenAI (tool_calls / role tool) or Anthropic
    (tool_use / tool_result blocks) histories. Mixed histories are handled too."""
    steps: list[Step] = []
    by_id: dict[str, Step] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            step = Step()
            for i, tc in enumerate(msg.get("tool_calls") or []):
                fn = tc.get("function") or {}
                cid = tc.get("id") or f"_oa{len(steps)}_{i}"
                name = fn.get("name") or tc.get("name") or "?"
                step.calls.append((cid, name, canonical_args(name, fn.get("arguments", tc.get("arguments")))))
            if isinstance(content, list):
                for i, b in enumerate(content):
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        cid = b.get("id") or f"_an{len(steps)}_{i}"
                        name = b.get("name") or "?"
                        step.calls.append((cid, name, canonical_args(name, b.get("input", {}))))
            if step.calls:
                steps.append(step)
                for cid, _, _ in step.calls:
                    by_id[cid] = step
        elif role == "tool":
            cid = msg.get("tool_call_id")
            step = by_id.get(cid) if cid else (steps[-1] if steps else None)
            if step is not None:
                if cid is None:  # id-less tool message: pair with the first unanswered call
                    cid = next((c for c, _, _ in step.calls if c not in step.results), None)
                if cid is not None:
                    text = _content_text(content)
                    err = bool(msg.get("is_error")) or text.lstrip().lower().startswith(("error", "traceback"))
                    step.results[cid] = (text, err)
        elif role == "user":
            blocks = content if isinstance(content, list) else []
            results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
            for b in results:
                step = by_id.get(b.get("tool_use_id"))
                if step is not None:
                    step.results[b.get("tool_use_id")] = (_content_text(b.get("content")), bool(b.get("is_error")))
            if not results and not str(_content_text(content)).startswith(NOTE_PREFIX):
                # A real user turn (not tool results, not our own note): the agent loop was
                # handed back to the human, who has seen what happened. Count afresh from here,
                # so an enforced session can recover; if the model loops again it is caught again.
                boundary = Step()
                boundary.boundary = True
                steps.append(boundary)
    return steps


# ----------------------------------------------------------------------------- detection

class Verdict:
    def __init__(self, level: str = "none", rule: str = "", run: int = 0,
                 tools: list[str] | None = None, result_preview: str = "",
                 calls: list[tuple[str, str]] | None = None) -> None:
        self.level, self.rule, self.run = level, rule, run
        self.tools = tools or []
        self.result_preview = result_preview
        self.calls = calls or []  # (name, canonical args) of the repeated calls, in order


def _level(run: int, rule: str) -> str:
    warn, force, stop = THRESHOLDS[rule]
    return "stop" if run >= stop else "force" if run >= force else "warn" if run >= warn else "none"


def detect(steps: list[Step]) -> Verdict:
    tail = list(steps[-MAX_STEPS:])
    for i in range(len(tail) - 1, -1, -1):
        if tail[i].boundary:
            tail = tail[i + 1:]
            break
    # Only complete steps count; the newest step must be complete (its results are what the
    # model is about to react to).
    if not tail or not tail[-1].complete():
        return Verdict()
    while tail and not tail[0].complete():
        tail.pop(0)
    keys = [s.call_key() for s in tail]
    last = tail[-1]
    best = Verdict()

    # Same call + same result, back to back.
    lk, lr = keys[-1], last.result_key()
    run, i = 0, len(tail) - 1
    while i >= 0 and keys[i] == lk and tail[i].result_key() == lr:
        run, i = run + 1, i - 1
    if run > 1:
        rule = "error" if last.all_errors() else "poll" if last.is_poll() else "default"
        lvl = _level(run, rule)
        if LEVELS.index(lvl) > LEVELS.index(best.level):
            best = Verdict(lvl, rule, run, last.names, calls=[(n, a) for _, n, a in last.calls])

    # A-B / A-B-C cycles whose results also repeat.
    for period in (2, 3, 4):
        if len(tail) < period * 2:
            continue
        window = period * (THRESHOLDS["poll_cycle"][2] + 1)
        sig = [(keys[j], tail[j].result_key()) for j in range(max(0, len(tail) - window), len(tail))]
        n = 0
        while n + period < len(sig) and sig[-1 - n] == sig[-1 - n - period]:
            n += 1
        reps = (n + period) // period
        if len({sig[-1 - k][0] for k in range(period)}) < 2:
            continue  # a single repeated call is the rule above, not a cycle
        rule = "poll_cycle" if any(s.is_poll() for s in tail[-period:]) else "cycle"
        lvl = _level(reps, rule)
        if LEVELS.index(lvl) > LEVELS.index(best.level):
            calls = list(dict.fromkeys((n, a) for s in tail[-period:] for _, n, a in s.calls))
            best = Verdict(lvl, f"{rule}{period}", reps, [n for s in tail[-period:] for n in s.names], calls=calls)

    if best.level != "none":
        text = next(iter(last.results.values()))[0]
        best.result_preview = re.sub(r"\s+", " ", text)[:300]
    return best


# ----------------------------------------------------------------------------- interventions

def note_text(v: Verdict) -> str:
    tools = ", ".join(dict.fromkeys(v.tools))
    if "cycle" in v.rule:
        what = f"You have repeated the same cycle of calls ({tools}) {v.run} times and every result came back unchanged."
    elif v.rule == "error":
        what = f"You have called `{tools}` with the same arguments {v.run} times in a row and it failed the same way every time."
    else:
        what = f"You have called `{tools}` with the same arguments {v.run} times in a row and it returned the same result every time."
    if v.level == "force":
        # Names the exact calls and asks for a different one. Never "reply in text": that ends the
        # agent turn, and a small model then copies its text-only reply on every later turn.
        listed = "".join(f"\n- {_call_label(n, a)}" for n, a in v.calls[:4])
        return (
            f"{NOTE_PREFIX} {what.rstrip('.')}{':' + listed if listed else '.'}\n"
            "Do not repeat these exact calls: they will return the same thing, and the next identical call "
            "will end this turn. Keep working, but change course: make a different tool call (different "
            "arguments, a different command or tool, or a check of why this isn't working). Only if nothing "
            "else can work, tell the user what is blocking you."
        )
    tail = (
        " Calling it again will not change anything. Use the result you already have, try a different "
        "approach, or tell the user what is blocking you."
    )
    return f"{NOTE_PREFIX} {what}{tail}"


def _call_label(name: str, args: str, limit: int = 200) -> str:
    """One repeated call for the force note: a shell tool shows its command, others their args."""
    shown = args
    try:
        parsed = json.loads(args)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        strings = [v for v in parsed.values() if isinstance(v, str)]
        if isinstance(parsed.get("command"), str):
            shown = parsed["command"]
        elif len(parsed) == 1 and strings:
            shown = strings[0]
    if len(shown) > limit:
        shown = shown[:limit] + "..."
    return f"{name} `{shown}`"


def stop_text(v: Verdict) -> str:
    tools = ", ".join(dict.fromkeys(v.tools))
    return (
        f"{NOTE_PREFIX} Stopped this turn: the model called `{tools}` {v.run} times in a row with no change "
        f"in the result, and did not change course after being told to. Last result: "
        f"\"{v.result_preview[:200]}\". Send a new instruction (or switch to a larger model tier) to continue."
    )


def _is_anthropic(data: dict[str, Any], call_type: str) -> bool:
    return call_type == "anthropic_messages" or any(
        isinstance(m, dict) and isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result") for b in m["content"])
        for m in data.get("messages", [])[-4:]
    )


def _append_note(msgs: list[dict[str, Any]], note: str, anthropic: bool) -> None:
    """Add the note after everything else, so the cached prompt prefix is untouched."""
    if not anthropic:
        msgs.append({"role": "user", "content": note})
        return
    last = msgs[-1]
    if last.get("role") == "user":  # Anthropic: tool_result must stay in this user turn
        content = last.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        msgs[-1] = {**last, "content": [*(content or []), {"type": "text", "text": note}]}
    else:
        msgs.append({"role": "user", "content": [{"type": "text", "text": note}]})


# Marks stop text for a streaming /v1/messages request (see _patch_anthropic_mock_streaming).
# A prefix, not a str subclass: LiteLLM's pydantic params coerce subclasses back to str.
STREAM_MARK = "\u2063loop-breaker-stream\u2063"


def apply(data: dict[str, Any], call_type: str, v: Verdict) -> None:
    anthropic = _is_anthropic(data, call_type)
    msgs = data["messages"]
    level = v.level
    if level == "stop" and anthropic and data.get("stream"):
        if _ANTHROPIC_MOCK_STREAMS:
            data["mock_response"] = STREAM_MARK + stop_text(v)
            return
    elif level == "stop":
        data["mock_response"] = stop_text(v)  # chat/completions streams mocks natively
        return
    if level == "stop":
        # Fallback when the streaming patch couldn't be installed: have the model write the
        # stop message. Tools off, short cap; a text-only reply ends the agent turn too.
        note = stop_text(v) + " Reply to the user in two or three sentences: what you were trying to do and why it is stuck. Do not call tools."
        data["max_tokens"] = min(int(data.get("max_tokens") or 512), 512)
    else:
        note = note_text(v)
    _append_note(msgs, note, anthropic)
    # No tool_choice "none" at the force stage: mtplx answers a tool call under tool_choice none
    # with a canned "no tools are active on this request… connect a coding agent" reply, which
    # confused pi (2026-09-27). The streaming /v1/messages stop fallback still turns tools off.
    if level == "stop":
        data["tool_choice"] = {"type": "none"} if anthropic else "none"


# ----------------------------------------------------------------------------- LiteLLM glue

def _opted_out(data: dict[str, Any]) -> bool:
    md = data.get("metadata") or {}
    if str(md.get("loop_breaker", "")).lower() == "off":
        return True
    for src in (md.get("headers"), (data.get("proxy_server_request") or {}).get("headers")):
        if isinstance(src, dict) and str(src.get("x-loop-breaker", "")).lower() == "off":
            return True
    return False


def conversation_key(data: dict[str, Any]) -> str:
    """Same session + agent key ultron_admit pins on (minus the tier), so the two logs join.
    The breaker itself keeps no counters: every request carries its own history."""
    h = {str(k).lower(): v for k, v in ((data.get("proxy_server_request") or {}).get("headers") or {}).items()}
    if h.get("x-claude-code-session-id"):
        return f"cc:{h['x-claude-code-session-id']}:{h.get('x-claude-code-agent-id') or 'main'}"
    first = next((m.get("content") for m in data.get("messages") or [] if isinstance(m, dict) and m.get("role") == "user"), "")
    if isinstance(first, list):
        first = "\n".join(b.get("text", "") for b in first if isinstance(b, dict) and b.get("type") in ("text", "input_text"))
    return "h:" + hashlib.sha256(str(first or "").encode()).hexdigest()[:24]


def _log(entry: dict[str, Any]) -> None:
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass


def evaluate(data: dict[str, Any], call_type: str, mode: str) -> Verdict:
    """Inspect (and in enforce mode, modify) one request. Never raises."""
    try:
        messages = data.get("messages")
        if mode == "off" or not isinstance(messages, list) or not messages or _opted_out(data):
            return Verdict()
        t0 = time.perf_counter()
        v = detect(extract_steps(messages))
        if v.level != "none":
            if mode == "enforce":
                apply(data, call_type, v)
            _log({
                "ts": time.time(), "mode": mode, "level": v.level, "rule": v.rule, "run": v.run,
                "key": conversation_key(data),
                "tools": v.tools, "model": data.get("model"), "call_type": call_type,
                "messages": len(messages), "ms": round((time.perf_counter() - t0) * 1000, 1),
                "result_preview": v.result_preview[:160],
            })
        return v
    except Exception as exc:  # the proxy must never fail a request because of this hook
        _log({"ts": time.time(), "error": repr(exc)})
        return Verdict()


try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # tests / replay without litellm installed
    CustomLogger = object  # type: ignore[misc,assignment]


def _patch_anthropic_mock_streaming() -> bool:
    """Make LiteLLM's /v1/messages mock honor stream=true.

    litellm 1.102.1 (latest as of 2026-09-27) returns `mock_response(...)` from
    anthropic_messages_handler before any stream handling and never passes `stream` to it,
    so a streaming client gets plain JSON. Chat/completions mocks already stream. LiteLLM
    ships FakeAnthropicMessagesStreamIterator for exactly this (it wraps a full response as
    message_start / content_block_* / message_delta / message_stop SSE); wrap our stop
    replies with it. Only STREAM_MARK-prefixed mocks are affected. Returns False (and the
    hook falls back to a model-written stop message) if the internals moved.
    """
    try:
        from litellm.llms.anthropic.experimental_pass_through.messages import handler
        from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
            FakeAnthropicMessagesStreamIterator,
        )
    except ImportError:
        return False
    original = getattr(handler, "mock_response", None)
    if original is None or getattr(original, "_loop_breaker", False):
        return original is not None

    def mock_response(*args, **kwargs):
        text = kwargs.get("mock_response")
        if isinstance(text, str) and text.startswith(STREAM_MARK):
            kwargs["mock_response"] = text[len(STREAM_MARK):]
            return FakeAnthropicMessagesStreamIterator(original(*args, **kwargs))
        return original(*args, **kwargs)

    mock_response._loop_breaker = True  # type: ignore[attr-defined]
    handler.mock_response = mock_response
    return True


_ANTHROPIC_MOCK_STREAMS = _patch_anthropic_mock_streaming()


MODE_FILE = os.path.expanduser(os.environ.get("LOOP_BREAKER_MODE_FILE", "~/.ultron/loop-breaker-mode"))


def current_mode() -> str:
    """Mode file (flipped live from the Wanda panel) wins over LOOP_BREAKER_MODE from the env."""
    try:
        v = open(MODE_FILE).read().strip().lower()
    except OSError:
        v = ""
    return v if v in ("enforce", "shadow", "off") else os.environ.get("LOOP_BREAKER_MODE", "enforce").lower()


class LoopBreaker(CustomLogger):  # type: ignore[misc,valid-type]
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        evaluate(data, call_type, current_mode())
        return data


proxy_handler_instance = LoopBreaker()
