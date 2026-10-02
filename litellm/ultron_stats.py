"""ultron_stats: per-request stats for every model behind LiteLLM, whatever the backend.

LiteLLM callback, registered FIRST (before loop_breaker/ultron_admit) so the clock starts when
the request arrives; it reads data["model"] lazily, so it still sees admission's final choice
(the hooks share one data dict). Observe-only: chunks pass through untouched, and every handler
swallows its own errors — a stats bug never fails or stalls a request.

  pre-call              request in flight (arrival time, requested model, prompt size estimate)
  streaming iterator    first content chunk (TTFT), live generated-token estimate and tok/s
  success/failure logs  exact usage (prompt, cached, completion tokens) from LiteLLM
  stream end / failure  request done -> one line in LOG_PATH

Aggregate metrics (request/failure counts, TTFT and latency histograms, token counters per model)
come from LiteLLM's stock `prometheus` callback (/metrics, master key). This hook only adds what that
lacks: per-model in-flight requests with live progress, and a per-request log Wanda joins to agents.

Files (Wanda reads both):
  ~/.litellm/stats-live.json      in-flight requests, rewritten at most once a second
  ~/.litellm/ultron-stats.jsonl   one line per finished request
TTFT includes admission wait and the backend's queue; prefill tok/s = uncached prompt / TTFT, so
it is a lower bound when requests queue. Backend internals (memory, prompt-cache bank, MTP
acceptance) aren't visible here; Wanda keeps reading those from the backend where it has them.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import time
from collections import OrderedDict
from typing import Any
from urllib.parse import urlparse

LIVE_PATH = os.path.expanduser(os.environ.get("ULTRON_STATS_LIVE", "~/.litellm/stats-live.json"))
LOG_PATH = os.path.expanduser(os.environ.get("ULTRON_STATS_LOG", "~/.litellm/ultron-stats.jsonl"))
CONFIG = os.path.expanduser(os.environ.get("ULTRON_LITELLM_CONFIG", "~/.litellm/config.yaml"))
GRACE_S = 8.0          # after a stream ends, wait this long for LiteLLM's usage before writing the row
STALE_S = 3600.0       # backstop only: in flight this long with no activity -> written as "lost"
TICK_S = 5.0           # while anything is in flight, check it's still alive this often
LIVE_EVERY_S = 1.0
CHARS_PER_TOKEN = 4.0  # live estimate only; the finished row uses LiteLLM's usage
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


# ----------------------------------------------------------------------------- model -> endpoint

class Endpoints:
    """model group -> "ultron/<tier>" (loopback backend), "cloud/<tier>", or "remote/<model>",
    from the LiteLLM config (re-read when it changes). Wildcards: most specific pattern wins."""

    def __init__(self, path: str) -> None:
        self.path, self.mtime = path, None
        self.exact: dict[str, str] = {}
        self.patterns: list[tuple[str, str]] = []
        self.alias: dict[str, str] = {}

    def _load(self) -> None:
        try:
            m = os.stat(self.path).st_mtime
        except OSError:
            return
        if m == self.mtime:
            return
        import yaml
        cfg = yaml.safe_load(open(self.path)) or {}
        exact, pats = {}, []
        for d in cfg.get("model_list") or []:
            name = str(d.get("model_name") or "")
            p = d.get("litellm_params") or {}
            ep = self.classify(name, str(p.get("model") or ""), str(p.get("api_base") or ""))
            if "*" in name:
                pats.append((name, ep))
            else:
                exact.setdefault(name, ep)
        alias = {}
        for k, v in ((cfg.get("router_settings") or {}).get("model_group_alias") or {}).items():
            alias[str(k)] = str(v.get("model") if isinstance(v, dict) else v)
        pats.sort(key=lambda x: -len(x[0].replace("*", "")))
        self.exact, self.patterns, self.alias, self.mtime = exact, pats, alias, m

    @staticmethod
    def classify(name: str, model: str, api_base: str) -> str:
        bare = model.split("/", 1)[1] if "/" in model else model
        if name.startswith("cloud/"):
            return name
        if (urlparse(api_base).hostname or "") in LOCAL_HOSTS:
            return f"ultron/{bare}"
        return f"remote/{bare}"

    def __call__(self, model: str) -> str:
        try:
            self._load()
        except Exception:
            pass
        m = self.alias.get(model, model)
        if m in self.exact:
            return self.exact[m]
        for pat, ep in self.patterns:
            if fnmatch.fnmatchcase(m, pat):
                return ep
        return f"?/{m}"


ENDPOINTS = Endpoints(CONFIG)


# ----------------------------------------------------------------------------- chunk parsing

def _g(o: Any, k: str) -> Any:
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def _usage_fields(u: Any) -> dict[str, int]:
    """OpenAI (prompt/completion + details) or Anthropic (input/output + cache_read) usage."""
    if not u:
        return {}
    out: dict[str, int] = {}
    for dst, keys in (("prompt", ("prompt_tokens", "input_tokens")), ("gen", ("completion_tokens", "output_tokens"))):
        for k in keys:
            v = _g(u, k)
            if isinstance(v, int) and v > 0:
                out[dst] = v
                break
    cached = _g(_g(u, "prompt_tokens_details") or {}, "cached_tokens")
    if not cached:
        cached = _g(u, "cache_read_input_tokens")
    if isinstance(cached, int) and cached > 0:
        out["cached"] = cached
    # Anthropic input_tokens excludes cache reads; OpenAI prompt_tokens includes cached ones
    if _g(u, "input_tokens") is not None and _g(u, "prompt_tokens") is None and out.get("cached"):
        out["prompt"] = out.get("prompt", 0) + out["cached"] + int(_g(u, "cache_creation_input_tokens") or 0)
    r = _g(_g(u, "completion_tokens_details") or {}, "reasoning_tokens")
    if isinstance(r, int) and r > 0:
        out["reasoning"] = r
    return out


def _event_text(ev: dict[str, Any]) -> tuple[int, dict[str, int]]:
    """(content chars, usage) for one decoded event: an Anthropic stream event or an OpenAI chunk dict."""
    t = ev.get("type")
    if t == "content_block_delta":
        d = ev.get("delta") or {}
        return len(d.get("text") or d.get("thinking") or d.get("partial_json") or ""), {}
    if t == "message_delta":
        return 0, _usage_fields(ev.get("usage"))
    if t == "message_start":
        return 0, _usage_fields((ev.get("message") or {}).get("usage"))
    if "choices" in ev:
        return _choices_chars(ev.get("choices")), _usage_fields(ev.get("usage"))
    return 0, {}


def _choices_chars(choices: Any) -> int:
    n = 0
    for c in choices or []:
        d = _g(c, "delta") or {}
        n += len(_g(d, "content") or "") + len(_g(d, "reasoning_content") or "") + len(_g(d, "reasoning") or "")
        for tc in _g(d, "tool_calls") or []:
            n += len(_g(_g(tc, "function") or {}, "arguments") or "")
    return n


def chunk_info(chunk: Any) -> tuple[int, dict[str, int]]:
    """Content chars and any usage carried by one stream chunk, whatever shape LiteLLM yields."""
    if isinstance(chunk, (bytes, bytearray)):
        chunk = chunk.decode("utf-8", "replace")
    if isinstance(chunk, str):
        chars, usage = 0, {}
        for line in chunk.splitlines():
            if line.startswith("data:"):
                try:
                    ev = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if isinstance(ev, dict):
                    c, u = _event_text(ev)
                    chars += c
                    usage.update(u)
        return chars, usage
    if isinstance(chunk, dict):
        return _event_text(chunk)
    choices = getattr(chunk, "choices", None)
    if choices is not None:
        return _choices_chars(choices), _usage_fields(getattr(chunk, "usage", None))
    if hasattr(chunk, "model_dump"):
        try:
            d = chunk.model_dump(exclude_none=True)
            if isinstance(d, dict):
                return _event_text(d)
        except Exception:
            pass
    return 0, {}


# ----------------------------------------------------------------------------- tracker

class Req:
    __slots__ = ("cid", "t0", "data", "call_type", "stream", "requested", "prompt_est", "t_first", "t_last",
                 "chars", "usage", "ended", "status", "error", "deployment", "api_base", "t_end", "mock", "logged",
                 "llm_start", "llm_first", "llm_end", "task")

    def __init__(self, cid: str, data: dict[str, Any], call_type: str, now: float) -> None:
        self.cid, self.t0, self.data, self.call_type = cid, now, data, call_type
        self.stream = bool(data.get("stream"))
        self.requested = str(data.get("model") or "")
        self.prompt_est = _prompt_chars(data) // int(CHARS_PER_TOKEN)
        self.t_first = self.t_last = self.t_end = None
        self.chars = 0
        self.usage: dict[str, int] = {}
        self.ended = self.logged = False
        self.status = self.error = self.deployment = self.api_base = None
        self.mock = False
        self.llm_start = self.llm_first = self.llm_end = None  # LiteLLM's own clock: call start, upstream first byte, end
        try:  # the ASGI request task: it also streams the body, so done() == the request is over, however it ended
            self.task = asyncio.current_task()
        except RuntimeError:
            self.task = None

    def endpoint(self) -> str:
        if self.mock or self.data.get("mock_response"):
            return "loop-breaker"
        return ENDPOINTS(str(self.data.get("model") or self.requested))

    def gen(self) -> int:
        return self.usage.get("gen") or int(self.chars / CHARS_PER_TOKEN)

    def live(self, now: float) -> dict[str, Any]:
        span = (self.t_last or 0) - (self.t_first or 0)
        g = self.gen()
        return {"cid": self.cid, "t0": round(self.t0, 3), "age": round(now - self.t0, 1), "requested": self.requested,
                "endpoint": self.endpoint(), "call_type": self.call_type, "stream": self.stream,
                "phase": "ending" if self.ended else ("generating" if self.t_first else "waiting"),
                "prompt_est": self.prompt_est, "ttft": round(self.t_first - self.t0, 2) if self.t_first else None,
                "gen_est": g, "tok_s": round(g / span, 1) if span > 0.5 and g > 1 else None}


def _prompt_chars(data: dict[str, Any]) -> int:
    n = 0
    for k in ("messages", "system", "tools", "input", "instructions"):
        v = data.get(k)
        if v:
            try:
                n += len(v) if isinstance(v, str) else len(json.dumps(v, default=str))
            except Exception:
                pass
    return n


def _p(rows: list[float], q: float) -> float | None:
    if not rows:
        return None
    rows = sorted(rows)
    return rows[min(len(rows) - 1, int(q * len(rows)))]


class Tracker:
    def __init__(self, live_path: str = LIVE_PATH, log_path: str = LOG_PATH) -> None:
        self.live_path, self.log_path = live_path, log_path
        self.reqs: OrderedDict[str, Req] = OrderedDict()
        self.done: OrderedDict[str, float] = OrderedDict()  # cid -> finished at; late usage events are ignored
        self.last_live = 0.0
        self.ticking = False
        self.write_live(force=True)  # a restart clears whatever the previous process left in flight

    # -- events
    def start(self, data: dict[str, Any], call_type: str, now: float | None = None) -> None:
        cid = data.get("litellm_call_id")
        if not cid or cid in self.reqs:
            return
        self.reqs[cid] = Req(cid, data, call_type, now or time.time())
        self.write_live(force=True)
        self._tick_soon()

    def chunk(self, cid: str | None, chunk: Any, now: float | None = None) -> None:
        r = self.reqs.get(cid or "")
        if r is None:
            return
        chars, usage = chunk_info(chunk)
        now = now or time.time()
        if chars:
            r.t_first = r.t_first or now
            r.t_last = now
            r.chars += chars
        if usage:
            r.usage.update(usage)
        self.write_live(first=chars and r.t_first == now)

    def stream_end(self, cid: str | None, status: str, error: str | None = None, now: float | None = None) -> None:
        r = self.reqs.get(cid or "")
        if r is None:
            return
        r.ended, r.t_end = True, now or time.time()
        r.status = r.status or status
        r.error = r.error or error
        if r.logged:
            self.finish(r)
        else:
            self._later(GRACE_S)
            self.write_live(force=True)

    def logged(self, kwargs: dict[str, Any], response_obj: Any, ok: bool, now: float | None = None) -> None:
        """LiteLLM success/failure log: exact usage + the deployment it went to. A failure here can be one
        retry attempt, so it records the error but only the proxy failure hook / stream end finishes it."""
        cid = kwargs.get("litellm_call_id")
        r = self.reqs.get(cid or "")
        if r is None:
            return
        slo = kwargs.get("standard_logging_object") or {}
        if ok:
            u = _usage_fields(_g(response_obj, "usage")) if response_obj is not None else {}
            for k, dst in (("prompt_tokens", "prompt"), ("completion_tokens", "gen")):
                if isinstance(slo.get(k), int) and slo[k] > 0 and dst not in u:
                    u[dst] = slo[k]
            r.usage.update({k: v for k, v in u.items() if v})
            r.logged = True
            r.status = r.status if r.status in ("cancelled",) else "ok"
        else:
            r.error = str(slo.get("error_str") or kwargs.get("exception") or "error")[:300]
        r.deployment = slo.get("model") or r.deployment
        r.api_base = slo.get("api_base") or r.api_base
        if ok:
            for attr, k in (("llm_start", "startTime"), ("llm_first", "completionStartTime"), ("llm_end", "endTime")):
                if isinstance(slo.get(k), (int, float)):
                    setattr(r, attr, float(slo[k]))
        if ok:  # streams: LiteLLM logs success only after the last chunk went out
            r.ended, r.t_end = True, r.t_end or (now or time.time())
            self.finish(r)

    def failed(self, cid: str | None, exc: Any, now: float | None = None) -> None:
        r = self.reqs.get(cid or "")
        if r is None:
            return
        r.status, r.error = "error", (f"{type(exc).__name__}: {exc}")[:300]
        r.ended, r.t_end = True, now or time.time()
        self.finish(r)

    def sweep(self, now: float | None = None) -> None:
        now = now or time.time()
        for r in list(self.reqs.values()):
            if r.ended and now - (r.t_end or now) >= GRACE_S:
                self.finish(r)
            elif not r.ended and r.task is not None and r.task.done():
                # the request's task is gone with no stream end / log / failure: the client disconnected
                # before any output (e.g. gave up during prefill or an eviction wait)
                r.status, r.t_end = r.status or ("ok" if r.logged else "cancelled"), now
                self.finish(r)
            elif not r.ended and now - max(r.t0, r.t_last or 0) > STALE_S:
                r.status, r.t_end = "lost", now
                self.finish(r)
        self.write_live(force=True)

    # -- output
    def finish(self, r: Req) -> None:
        if self.reqs.pop(r.cid, None) is None:
            return
        self.done[r.cid] = r.t_end or time.time()
        while len(self.done) > 2048:
            self.done.popitem(last=False)
        self._log(self.row(r))
        self.write_live(force=True)

    def row(self, r: Req) -> dict[str, Any]:
        end = r.t_end or time.time()
        mock = r.mock or bool(r.data.get("mock_response"))
        status = "mock" if mock else (r.status or ("ok" if r.logged else "unknown"))
        u = r.usage
        gen, prompt, cached = u.get("gen"), u.get("prompt"), u.get("cached") or 0
        # Backend timing from LiteLLM's clock (first upstream byte), which matches what mtplx reports; the
        # client's first visible content can be much later when a bridge buffers (the /v1/messages ->
        # /v1/responses path holds back reasoning). Without LiteLLM timing (cancelled), use the chunks.
        seen = (r.t_first - r.t0) if r.t_first else None
        g = gen or (int(r.chars / CHARS_PER_TOKEN) if r.chars else None)
        if r.stream and r.llm_first and r.llm_start and r.llm_end and r.llm_end >= r.llm_first:
            ttft, span = r.llm_first - r.llm_start, r.llm_end - r.llm_first
        else:
            ttft, span = (seen if r.stream else None), ((r.t_last - r.t_first) if r.t_first and r.t_last else 0)
        wait = (r.llm_start - r.t0) if r.llm_start else None
        row = {
            "ts": round(end, 3), "t0": round(r.t0, 3), "call_id": r.cid, "call_type": r.call_type, "stream": r.stream,
            "requested": r.requested, "model": str(r.data.get("model") or r.requested), "endpoint": r.endpoint(),
            "deployment": r.deployment, "status": status, "error": r.error,
            "prompt": prompt, "prompt_est": None if prompt else r.prompt_est, "cached": cached or None, "gen": gen,
            "gen_est": None if gen else g, "reasoning": u.get("reasoning"),
            "wait": round(wait, 3) if wait and wait >= 0.05 else None,       # hooks + admission before the call
            "ttft": round(ttft, 3) if ttft is not None else None,             # backend: call start -> first byte
            "first_seen": round(seen, 3) if seen is not None else None,       # client: arrival -> first content
            "elapsed": round(end - r.t0, 3),
            "decode_tok_s": round(g / span, 1) if g and g > 1 and span > 0.5 else None,
            "prefill_tok_s": round((prompt - cached) / ttft, 1) if prompt and ttft and ttft > 0.2 and prompt > cached else None,
            "e2e_tok_s": round(g / (end - r.t0), 1) if g and end > r.t0 else None,
        }
        return {k: v for k, v in row.items() if v is not None}

    def write_live(self, force: bool = False, first: bool = False) -> None:
        now = time.time()
        if not (force or first) and now - self.last_live < LIVE_EVERY_S:
            return
        self.last_live = now
        body = {"ts": round(now, 3), "pid": os.getpid(), "inflight": [r.live(now) for r in self.reqs.values()]}
        try:
            tmp = self.live_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(body, f)
            os.replace(tmp, self.live_path)
        except OSError:
            pass

    def _log(self, row: dict[str, Any]) -> None:
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except OSError:
            pass

    def _tick_soon(self) -> None:
        if self.ticking:
            return
        try:
            asyncio.get_running_loop().call_later(TICK_S, self._tick)
            self.ticking = True
        except RuntimeError:
            pass

    def _tick(self) -> None:
        self.ticking = False
        try:
            self.sweep()
        finally:
            if self.reqs:
                self._tick_soon()

    def _later(self, delay: float) -> None:
        try:
            asyncio.get_running_loop().call_later(delay + 0.1, self.sweep)
        except RuntimeError:  # no loop (tests): the next event sweeps
            pass


# ----------------------------------------------------------------------------- LiteLLM glue

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # tests without litellm installed
    CustomLogger = object  # type: ignore[misc,assignment]


class UltronStats(CustomLogger):  # type: ignore[misc,valid-type]
    def __init__(self) -> None:
        super().__init__()
        self.t = Tracker()

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            self.t.start(data, str(call_type))
        except Exception:
            pass
        return data

    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        cid = request_data.get("litellm_call_id")
        status, err = "cancelled", None
        try:
            async for chunk in response:
                try:
                    self.t.chunk(cid, chunk)
                except Exception:
                    pass
                yield chunk
            status = "ok"
        except (GeneratorExit, asyncio.CancelledError):
            raise
        except Exception as e:
            status, err = "error", f"{type(e).__name__}: {e}"[:300]
            raise
        finally:
            try:
                self.t.stream_end(cid, status, err)
            except Exception:
                pass

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        try:
            if not data.get("stream"):
                self.t.stream_end(data.get("litellm_call_id"), "ok")
        except Exception:
            pass
        return response

    async def async_post_call_failure_hook(self, request_data, original_exception, user_api_key_dict, traceback_str=None):
        try:
            self.t.failed(request_data.get("litellm_call_id"), original_exception)
        except Exception:
            pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self.t.logged(kwargs, response_obj, True)
        except Exception:
            pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self.t.logged(kwargs, response_obj, False)
        except Exception:
            pass


proxy_handler_instance = UltronStats()
