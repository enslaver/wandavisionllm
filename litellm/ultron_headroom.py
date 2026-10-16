"""Optional Headroom context compression for every request (off by default; `pip install headroom-ai` into
LiteLLM's environment).

Wraps headroom-ai's HeadroomCallback (https://github.com/headroomlabs-ai/headroom), which compresses
`data["messages"]` in the pre-call phase. Differences from using the class as it ships:

- No token floor: HeadroomCallback stores `min_tokens` but never reads it. Every chat request is
  compressed, local tiers and cloud/* alike. A local tier pays for it in mtplx's prefix cache (an older
  tool result that changes between turns is re-prefilled).
- Always local: the class falls back to the HEADROOM_API_KEY environment variable and then sends the prompt to
  Headroom Cloud. ~/.litellm/env may hold that variable for another proxy, so the key is cleared after init.
- Compression runs in a thread (it is CPU-bound and synchronous; the stock hook runs it on the event loop).
- Mode file ~/.ultron/headroom-mode (on/off, default off, re-read every request), per-request opt-out
  `x-headroom: off`, jsonl log for Wanda's stats.

Runs after ultron_admit (the pin key hashes the client's own first message) and after ultron_hindsight (its
notes ride on user messages, which Headroom leaves alone).

Log: ~/.litellm/headroom.jsonl, one row per compressed request (before/after tokens, ms) or error.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

# Telemetry off before headroom is imported.
for _k, _v in (("HEADROOM_BEACON", "off"), ("DO_NOT_TRACK", "1"), ("HEADROOM_UPDATE_CHECK", "off")):
    os.environ.setdefault(_k, _v)

try:
    from headroom.integrations.litellm_callback import HeadroomCallback
    IMPORT_ERROR = ""
except Exception as exc:  # headroom-ai missing: the hook stays a no-op and says why in the log
    IMPORT_ERROR = repr(exc)
    try:
        from litellm.integrations.custom_logger import CustomLogger as HeadroomCallback  # type: ignore[assignment]
    except Exception:  # unit tests run without litellm
        HeadroomCallback = object  # type: ignore[misc,assignment]

import ultron_admit

MODE_FILE = os.path.expanduser(os.environ.get("ULTRON_HEADROOM_MODE_FILE", "~/.ultron/headroom-mode"))
LOG_PATH = os.path.expanduser(os.environ.get("ULTRON_HEADROOM_LOG", "~/.litellm/headroom.jsonl"))
MODEL_LIMIT = int(os.environ.get("ULTRON_HEADROOM_MODEL_LIMIT", "200000"))  # the largest tier input cap (tiers.conf context)
CHAT_CALLS = ("completion", "acompletion", "anthropic_messages", "aanthropic_messages")


def mode_on() -> bool:
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


def wanted(data: dict[str, Any], call_type: str) -> bool:
    """Whether this request gets compressed: mode on, a chat call, real messages, no x-headroom: off."""
    if call_type not in CHAT_CALLS or not mode_on():
        return False
    if data.get("mock_response") or not isinstance(data.get("messages"), list) or not data["messages"]:
        return False  # loop-breaker / media answered it already
    return ultron_admit._headers(data).get("x-headroom", "").strip().lower() != "off"


class UltronHeadroom(HeadroomCallback):  # type: ignore[misc, valid-type]
    def __init__(self) -> None:
        if IMPORT_ERROR:
            if HeadroomCallback is not object:
                super().__init__()
            self._local_ready = False
            return
        super().__init__(model_limit=MODEL_LIMIT)
        self._api_key = None  # never Headroom Cloud, whatever HEADROOM_API_KEY holds
        self._local_ready = True

    async def async_pre_call_hook(self, user_api_key_dict=None, cache=None, data=None, call_type="", *_a, **_kw):
        if isinstance(cache, dict) and isinstance(data, str):  # older LiteLLM argument order
            data, call_type = cache, data
        if data is None:
            return None
        try:
            if not wanted(data, str(call_type)):
                return data
            if not self._local_ready:
                _log({"ts": time.time(), "error": "headroom-ai not importable: " + IMPORT_ERROR})
                return data
            started = time.time()
            result = await asyncio.to_thread(self._local_compress, data["messages"], str(data.get("model") or ""))
            ms = int((time.time() - started) * 1000)
            saved = int((result or {}).get("tokens_saved") or 0)
            cid = data.get("litellm_call_id") or ""
            if saved > 0 and result and "messages" in result:
                data["messages"] = result["messages"]
                self._total_saved += saved
                recent = _live_recent()
                if cid in recent and "; headroom" not in recent[cid]:  # hook can run twice per /v1/messages
                    recent[cid] += f"; headroom(-{saved})"
            _log({"ts": time.time(), "call_id": cid, "model": data.get("model"),
                  "before": (result or {}).get("tokens_before"), "after": (result or {}).get("tokens_after"),
                  "saved": saved, "ms": ms, "applied": saved > 0})
        except Exception as exc:  # never fail a request because of compression
            _log({"ts": time.time(), "error": repr(exc), "model": (data or {}).get("model")})
        return data


def _live_recent() -> dict:
    """The route-header dict of the admission hook LiteLLM is running. LiteLLM loads each hook from its file path,
    so `import ultron_admit` here would be a second copy of the module with its own (empty) dict."""
    try:
        import litellm
        for cb in getattr(litellm, "callbacks", None) or []:
            admission = getattr(cb, "admission", None)
            if admission is not None and hasattr(admission, "recent"):
                return admission.recent
    except Exception:
        pass
    return ultron_admit.proxy_handler_instance.admission.recent


proxy_handler_instance = UltronHeadroom()
