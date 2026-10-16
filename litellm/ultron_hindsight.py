"""ultron_hindsight: long-term memory from Hindsight for conversations behind ultron:4000.

Optional LiteLLM proxy hook, off by default. Hindsight (vectorize-io) keeps facts across sessions in a memory
bank on a server you run (ULTRON_HINDSIGHT_URL); this hook talks to its HTTP API:

  - recall: on the first request of a human turn, POST /memories/recall with that turn's text. The results become
    a <memory> note appended to that human message.
  - retain: when a turn ends (the reply calls no tool), POST /memories with the turn's human text and final reply
    (async on Hindsight's side: its own LLM extracts the facts).

Prefix caches stay intact. mtplx (local tiers) and the cloud providers cache by prompt prefix, and the client
resends history without our notes. So each note is stored (sqlite, keyed on conversation + human-turn hash) and
put back on the same human message in every later request of the conversation: the prompt only ever grows at
its end. A note is never added to the system prompt, which would change every request's first tokens.

Not hindsight-litellm: in the proxy LiteLLM 1.102.1 only calls its sync log_pre_api_call (a blocking recall on
the event loop), and that edits the logging copy of the messages after the request body is built.

Skipped: requests without tools (side calls: titles,
summaries), Claude Code subagents (x-claude-code-agent-id), loop-breaker / media replies (mock_response),
non-chat calls. Mode: ~/.ultron/hindsight-mode on|off (default off; also off while ULTRON_HINDSIGHT_URL is empty). Per-request opt-out: header
`x-hindsight: off`. Log: ~/.litellm/hindsight.jsonl. Recall fails open (4 s timeout). Stdlib only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # unit tests without litellm
    CustomLogger = object  # type: ignore[misc,assignment]

import ultron_admit

URL = os.environ.get("ULTRON_HINDSIGHT_URL", "").rstrip("/")   # e.g. https://hindsight.example.com:8888
BANK = os.environ.get("ULTRON_HINDSIGHT_BANK", "ultron")
MODE_FILE = os.path.expanduser(os.environ.get("ULTRON_HINDSIGHT_MODE_FILE", "~/.ultron/hindsight-mode"))
NOTES_DB = os.path.expanduser(os.environ.get("ULTRON_HINDSIGHT_DB", "~/.litellm/hindsight-notes.sqlite"))
LOG_PATH = os.path.expanduser(os.environ.get("ULTRON_HINDSIGHT_LOG", "~/.litellm/hindsight.jsonl"))
RECALL_TIMEOUT_S = 4.0  # a recall measured 0.25-0.36 s on the reference setup
DOWN_HOLD_S = 60.0
RETAIN_TIMEOUT_S = 10.0
RECALL_BUDGET = "low"
RECALL_MAX_TOKENS = 1024
NOTE_MAX_FACTS = 8
NOTE_MAX_CHARS = 3000
QUERY_MAX_CHARS = 2000
MIN_QUERY_CHARS = 12  # "ok", "continue", "yes do it": nothing to look up
RETAIN_SIDE_CHARS = 8000  # per side (human text, reply)
NOTES_KEEP_S = 30 * 86400
CHAT_CALLS = ("acompletion", "completion", "anthropic_messages")
NOTE_HEAD = ('<memory source="hindsight">\nPossibly relevant facts from earlier sessions '
             "(they may be outdated; this conversation takes precedence):")
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def mode_on() -> bool:
    if not URL:
        return False
    try:
        return open(MODE_FILE).read().strip().lower() == "on"
    except OSError:
        return False


def _log(entry: dict[str, Any]) -> None:
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass


# ----------------------------------------------------------------------------- messages

def human_text(m: Any) -> str | None:
    """The text of a human turn (system reminders removed), or None when `m` isn't one: a tool result
    (Anthropic: a user message holding tool_result blocks; chat: role tool) or a reminder-only message."""
    if not isinstance(m, dict) or m.get("role") != "user":
        return None
    c = m.get("content")
    if isinstance(c, list):
        if any(isinstance(b, dict) and b.get("type") in ("tool_result", "function_call_output") for b in c):
            return None
        c = "\n".join(str(b.get("text") or "") for b in c if isinstance(b, dict) and b.get("type") in ("text", "input_text"))
    text = REMINDER.sub("", str(c or "")).strip()
    return text or None


def human_turns(messages: Any) -> list[tuple[int, str]]:
    return [(i, t) for i, m in enumerate(messages or []) if (t := human_text(m)) is not None]


def turn_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def conversation(data: dict[str, Any]) -> str:
    """ultron_admit's pin key without the tier: Claude Code session + agent, else a hash of the first user
    message. Computed on the client's own messages, before any note is added."""
    return ultron_admit.pin_identity(data, "-")["key"].rsplit(":", 1)[0]


def with_note(m: dict[str, Any], note: str) -> dict[str, Any]:
    c = m.get("content")
    if isinstance(c, list):
        return {**m, "content": c + [{"type": "text", "text": note}]}
    return {**m, "content": f"{c}\n\n{note}" if c else note}


def format_note(results: Any) -> str:
    lines = []
    for r in results or []:
        if not isinstance(r, dict) or not str(r.get("text") or "").strip():
            continue
        when = str(r.get("occurred_start") or r.get("mentioned_at") or "")[:10]
        lines.append("- " + " ".join(str(r["text"]).split()) + (f" ({when})" if when else ""))
        if len(lines) >= NOTE_MAX_FACTS or sum(len(x) for x in lines) > NOTE_MAX_CHARS:
            break
    return (NOTE_HEAD + "\n" + "\n".join(lines) + "\n</memory>") if lines else ""


def called_tool(response: Any) -> bool:
    """True when the reply asks for a tool (the turn goes on), for a chat ModelResponse or a /v1/messages dict."""
    if response is None:
        return False
    if not isinstance(response, dict):
        dump = getattr(response, "model_dump", None)
        try:
            response = dump() if callable(dump) else dict(response)
        except Exception:
            return False
    if response.get("stop_reason") == "tool_use":
        return True
    if any(isinstance(b, dict) and b.get("type") == "tool_use" for b in response.get("content") or []):
        return True
    for ch in response.get("choices") or []:
        if isinstance(ch, dict) and (ch.get("finish_reason") == "tool_calls" or (ch.get("message") or {}).get("tool_calls")):
            return True
    return False


def skip_reason(data: dict[str, Any], call_type: str) -> str | None:
    if call_type not in CHAT_CALLS or not isinstance(data.get("messages"), list):
        return "call"
    h = ultron_admit._headers(data)
    md = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    if h.get("x-hindsight", "").strip().lower() == "off" or str(md.get("hindsight") or "").lower() == "off":
        return "header"
    if h.get("x-claude-code-agent-id"):
        return "subagent"
    if not data.get("tools"):
        return "no-tools"
    if isinstance(data.get("mock_response"), str):
        return "mocked"
    model = str(data.get("model") or "")
    if model == "ultron/judge" or model.startswith("media/"):
        return "model"
    return None


# ----------------------------------------------------------------------------- Hindsight + note store

def _post(path: str, body: dict[str, Any], timeout: float) -> Any:
    req = urllib.request.Request(f"{URL}/v1/default/banks/{BANK}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def recall(query: str) -> list[Any]:
    out = _post("/memories/recall", {"query": query[:QUERY_MAX_CHARS], "budget": RECALL_BUDGET,
                                     "max_tokens": RECALL_MAX_TOKENS}, RECALL_TIMEOUT_S)
    return out.get("results") or [] if isinstance(out, dict) else []


def retain(content: str, doc_id: str, agent: str, model: str) -> Any:
    item = {"content": content, "document_id": doc_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": f"a conversation with {agent} through ultron's LiteLLM gateway (model {model})",
            "metadata": {"agent": agent, "model": model}, "tags": [agent]}
    return _post("/memories", {"items": [item], "async": True}, RETAIN_TIMEOUT_S)


class Notes:
    """conversation + human-turn hash -> note ('' = recalled, nothing relevant). On disk so a LiteLLM restart
    doesn't drop notes from older turns, which would change those messages and cost a full re-prefill."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or NOTES_DB
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS notes (conv TEXT, turn TEXT, note TEXT, ts REAL, PRIMARY KEY (conv, turn))")
            db.execute("DELETE FROM notes WHERE ts < ?", (time.time() - NOTES_KEEP_S,))

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def get(self, conv: str) -> dict[str, str]:
        with self._db() as db:
            return dict(db.execute("SELECT turn, note FROM notes WHERE conv = ?", (conv,)).fetchall())

    def put(self, conv: str, turn: str, note: str) -> None:
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO notes VALUES (?, ?, ?, ?)", (conv, turn, note, time.time()))


# ----------------------------------------------------------------------------- hook

class UltronHindsight(CustomLogger):  # type: ignore[misc,valid-type]
    def __init__(self) -> None:
        super().__init__()
        self._notes: Notes | None = None
        self.down_until = 0.0  # after a failed recall, no recalls until then (each would wait RECALL_TIMEOUT_S)
        self.pending: OrderedDict[str, dict[str, Any]] = OrderedDict()  # litellm_call_id -> turn to retain

    @property
    def notes(self) -> Notes:
        if self._notes is None:
            self._notes = Notes()
        return self._notes

    async def apply(self, data: dict[str, Any], call_type: str) -> dict[str, Any] | None:
        """Add notes to `data` in place; returns the log entry (None when skipped)."""
        if not mode_on() or skip_reason(data, call_type):
            return None
        messages = data["messages"]
        turns = human_turns(messages)
        if not turns:
            return None
        conv = conversation(data)
        stored = await asyncio.to_thread(self.notes.get, conv)
        last_i, last_text = turns[-1]
        last_turn = turn_hash(last_text)
        entry: dict[str, Any] = {"ts": time.time(), "event": "apply", "cid": data.get("litellm_call_id"), "conv": conv}
        if last_turn not in stored and time.time() >= self.down_until:
            note, t0 = "", time.time()
            try:
                if len(last_text) >= MIN_QUERY_CHARS:
                    results = await asyncio.to_thread(recall, last_text)
                    note = format_note(results)
                    entry.update(event="recall", facts=len(results), ms=round((time.time() - t0) * 1000))
                stored[last_turn] = note
                await asyncio.to_thread(self.notes.put, conv, last_turn, note)
            except Exception as exc:  # fail open; the turn's next request tries again once Hindsight is back
                self.down_until = time.time() + DOWN_HOLD_S
                entry.update(event="recall", error=repr(exc)[:200], ms=round((time.time() - t0) * 1000))
        applied = 0
        for i, text in turns:
            note = stored.get(turn_hash(text))
            if note:
                messages[i] = with_note(messages[i], note)
                applied += 1
        entry.update(notes=applied, turns=len(turns))
        cid = data.get("litellm_call_id")
        if cid:
            self.pending[cid] = {"conv": conv, "turn": last_turn, "text": last_text,
                                 "agent": ultron_admit.client_info(data)["agent"], "model": str(data.get("model") or "")}
            while len(self.pending) > 2048:
                self.pending.popitem(last=False)
        return entry

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            entry = await self.apply(data, str(call_type))
            if entry and (entry["event"] == "recall" or entry.get("notes")):
                _log(entry)
        except Exception as exc:  # never fail a request because of memory
            _log({"ts": time.time(), "error": "pre-call: " + repr(exc)})
        return data

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        p = self.pending.pop(kwargs.get("litellm_call_id") or "", None)
        if not p or called_tool(response_obj):
            return
        try:
            reply = "\n".join(t for t in ultron_admit.reply_texts(response_obj) if t).strip()
            if not reply:
                return
            content = f"User: {p['text'][:RETAIN_SIDE_CHARS]}\n\nAssistant: {reply[:RETAIN_SIDE_CHARS]}"
            t0 = time.time()
            out = await asyncio.to_thread(retain, content, f"{p['conv']}|{p['turn']}", p["agent"], p["model"])
            _log({"ts": time.time(), "event": "retain", "cid": kwargs.get("litellm_call_id"), "conv": p["conv"],
                  "agent": p["agent"], "chars": len(content), "ms": round((time.time() - t0) * 1000),
                  "op": (out or {}).get("operation_id") if isinstance(out, dict) else None})
        except Exception as exc:
            _log({"ts": time.time(), "event": "retain", "conv": p.get("conv"), "error": repr(exc)[:200]})

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.pending.pop(kwargs.get("litellm_call_id") or "", None)


proxy_handler_instance = UltronHindsight()
