#!/usr/bin/env python3
"""wanda — realtime panel for the local LLM stack (part of wandavision).

    LiteLLM 0.0.0.0:4000 -> llama-swap 127.0.0.1:8001 -> mtplx / TensorFold tier servers 127.0.0.1:18001+

One background thread samples the stack every second (llama-swap /running, each loaded tier's
/v1/mtplx/snapshot, the LiteLLM hooks' logs). Browsers get the samples over SSE (/api/stream) and
backfill from /api/history. Caddy fronts this on :443 and :80.

Settings (env, set in the LaunchAgent): WANDA_LISTEN (127.0.0.1:8790), WANDA_NAME (panel title,
default: this Mac's short hostname), WANDA_WWW (services.json + icons/), WANDA_OMNIROUTE (OmniRoute
base URL; default: OMNIROUTE_BASE from ~/.litellm/env), WANDA_HOST_NAMES ("ip=name,ip=name" labels
for clients that Tailscale can't name).

Standard library only; runs on any python3 >= 3.9.
"""
import configparser
import json
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HOME = Path.home()
HOST, PORT = os.environ.get("WANDA_LISTEN", "127.0.0.1:8790").rsplit(":", 1)
STATIC = Path(__file__).resolve().parent / "static"
STATE_DIR = HOME / ".wanda"
TOKEN_FILE = STATE_DIR / "token"
WWW = Path(os.environ.get("WANDA_WWW", str(HOME / ".wanda/www")))  # services.json + icons/ (see services.example.json)
NAME = os.environ.get("WANDA_NAME") or (os.uname().nodename.split(".")[0] or "wanda").capitalize()

SWAP = "http://127.0.0.1:8001"
SWAP_CFG = HOME / ".llama-swap/config.yaml"


def _litellm_port():
    """LITELLM_PORT from ~/.litellm/env (start.sh starts LiteLLM on it), else 4000. Read at start."""
    try:
        for line in (HOME / ".litellm/env").read_text().splitlines():
            if line.startswith("LITELLM_PORT="):
                return line.split("=", 1)[1].strip().strip("\"'") or "4000"
    except OSError:
        pass
    return "4000"


LITELLM = f"http://127.0.0.1:{_litellm_port()}"
TIERS_CONF = HOME / ".litellm/tiers.conf"   # the local tiers (litellm/tiers.conf in the repo), in display order
TIER_COLORS = ["#c39bd3", "#88c0d0", "#a3be8c", "#ebcb8b", "#d08770", "#b48ead", "#8fbcbb"]  # = ultron_tiers.COLORS
_tiers = {"mtime": None, "names": ["opus", "sonnet", "haiku"], "colors": {}}


def tier_names():
    """Tier names from ~/.litellm/tiers.conf in file order, re-read when it changes; its color keys
    (or the same defaults ultron_tiers uses) go to tier_colors()."""
    try:
        mtime = TIERS_CONF.stat().st_mtime
    except OSError:
        mtime = None
    if mtime is not None and mtime != _tiers["mtime"]:
        _tiers["mtime"] = mtime
        try:
            cp = configparser.ConfigParser(interpolation=None)  # whole-line comments only, like ultron_tiers
            cp.read_string(TIERS_CONF.read_text())
            names = [s for s in cp.sections() if s != "routing"]
            if names:
                _tiers["names"] = names
                _tiers["colors"] = {n: (cp[n].get("color") or "").strip() for n in names}
        except (configparser.Error, OSError):
            pass
    return _tiers["names"]


def tier_colors():
    return {n: _tiers["colors"].get(n) or TIER_COLORS[i % len(TIER_COLORS)] for i, n in enumerate(tier_names())}

# On/off switches shown as status lamps and in Controls & routing. Each is a file holding "true" or
# "false" plus an append-only log; other tools read the file to decide what to do. Example:
#   {"key": "agents", "label": "Agents", "help": "Remote agents may use this Mac only when this is true",
#    "path": HOME / ".wanda/agents-allowed", "log": HOME / ".wanda/agents-allowed.log"}
FLAGS = []

# Multi-way switches for the LiteLLM hooks (litellm/ in the repo). Each is a one-word file the
# hook re-reads on every request, so a flip applies to the next request with no restart. When a
# file is missing the hook falls back to its env var in ~/.litellm/env.
ULTRON_DIR = HOME / ".ultron"
LITELLM_ENV = HOME / ".litellm/env"
MODE_LOG = ULTRON_DIR / "modes.log"
MODES = [
    {"key": "route", "label": "Route", "path": ULTRON_DIR / "route-mode", "env": None, "default": "auto",
     "options": ["auto", "local-only", "cloud-only"],
     "help": "auto: local tiers first, overflow to OmniRoute combos · local-only: never cloud · "
             "cloud-only: every new conversation to the combos (applies even while admission is in shadow)"},
    {"key": "admit", "label": "Admission", "path": ULTRON_DIR / "admit-mode", "env": "ULTRON_ADMIT_MODE", "default": "shadow",
     "options": ["shadow", "enforce", "off"],
     "help": "enforce: rewrite each new conversation to its local tier / substitute / cloud combo and pin it · "
             "shadow: log what it would do (x-route and cloud-only still apply)"},
    {"key": "loops", "label": "Loop breaker", "path": ULTRON_DIR / "loop-breaker-mode", "env": "LOOP_BREAKER_MODE", "default": "enforce",
     "options": ["shadow", "enforce", "off"],
     "help": "enforce: note at 4 identical call+result repeats, tools off at 6, end the turn at 8 (polls 10/16/20) · "
             "shadow: log only"},
    {"key": "media", "label": "Media", "path": ULTRON_DIR / "media-mode", "env": "ULTRON_MEDIA_MODE", "default": "shadow",
     "options": ["shadow", "enforce", "off"],
     "help": "enforce: image / video / photo-edit / web-search / audio prompts are answered by the cloud endpoint "
             "(files in ~/.ultron/media) instead of a tier · shadow: log the match only · agent requests also need "
             "the helper tier's MEDIA"},
    {"key": "rescue", "label": "Tool-call rescue", "path": ULTRON_DIR / "rescue-mode", "env": "ULTRON_RESCUE_MODE", "default": "enforce",
     "options": ["shadow", "enforce", "off"],
     "help": "enforce: a local tier's reply that ends with the ```bash block it meant to run becomes a real tool call · "
             "shadow: log only"},
    # Shown in the LoRA section ("panel"). on/off are both normal states, so neither is painted as a warning.
    {"key": "trace", "label": "Trace tap", "path": ULTRON_DIR / "trace-mode", "env": None, "default": "off",
     "options": ["off", "on"], "warn": [], "panel": "lora",
     "help": "on: save the latest tool-carrying request of each local-tier conversation to ~/.ultron/traces/ "
             "(training data for lora/; holds tool output) · off: save nothing"},
]
ADMIT_LOG = HOME / ".litellm/ultron-admit.jsonl"
LOOPS_LOG = HOME / ".litellm/loop-breaker.jsonl"
STATS_LOG = HOME / ".litellm/ultron-stats.jsonl"     # ultron_stats hook: one line per finished request
STATS_LIVE = HOME / ".litellm/stats-live.json"       # ultron_stats hook: requests in flight
PINS_DB = HOME / ".litellm/pins.sqlite"
PIN_IDLE_S = {"main": 3600, "sub": 1800}   # same as ultron_admit.py

LOGS = {
    "swap": ("LLAMA-SWAP", None),                      # fetched from llama-swap's /logs
    "litellm": ("LITELLM", HOME / ".litellm/litellm.log"),
    "caddy": ("CADDY", Path("/opt/homebrew/var/log/caddy.log")),
    "modes": ("MODES", HOME / ".ultron/modes.log"),
    "omniroute": ("OMNIROUTE", "omniroute"),                   # calls OmniRoute logged for this stack's key
    "loops": ("LOOPS", HOME / ".litellm/loop-breaker.jsonl"),  # loop_breaker hook
    "admit": ("ADMIT", HOME / ".litellm/ultron-admit.jsonl"),  # local vs OmniRoute routing decisions
    "media": ("MEDIA", HOME / ".litellm/media.jsonl"),  # ultron_media: prompts sent to OmniRoute (or would be, in shadow)
    "rescue": ("RESCUE", HOME / ".litellm/rescue.jsonl"),  # ultron_rescue: text replies turned into tool calls
    "stats": ("REQUESTS", HOME / ".litellm/ultron-stats.jsonl"),  # every request through LiteLLM (ultron_stats)
}
SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_-]{4})[A-Za-z0-9_-]+|((?:Bearer|api[_-]?key[=:]\s*)\s*)[A-Za-z0-9._-]{8,}", re.I)

HISTORY_S = 900              # per-tier samples kept (1 per second -> 15 minutes)
TICK_S = 1.0
# State codes shared with the page. Order matters: the trace paints by code.
UNLOADED, LOADING, IDLE, PREFILL, GENERATING, DOWN = range(6)
STATE_NAME = ["UNLOADED", "LOADING", "IDLE", "PREFILL", "GENERATING", "UNREACHABLE"]


def token():
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    if not TOKEN_FILE.exists():
        TOKEN_FILE.write_text(secrets.token_urlsafe(24))
        TOKEN_FILE.chmod(0o600)
    return TOKEN_FILE.read_text().strip()


def get(url, timeout=2.0, raw=False):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body = r.read()
    return body.decode("utf-8", "replace") if raw else json.loads(body)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------------------------
# Static-ish facts: tier config (ttl, script, model dir) re-read when the llama-swap config changes
# ---------------------------------------------------------------------------------------------
_cfg_cache = {"mtime": None, "tiers": {}}


def tier_config():
    try:
        mtime = SWAP_CFG.stat().st_mtime
    except OSError:
        return {t: {} for t in tier_names()}
    if mtime == _cfg_cache["mtime"] and set(_cfg_cache["tiers"]) == set(tier_names()):
        return _cfg_cache["tiers"]
    out, cur = {t: {} for t in tier_names()}, None
    for line in SWAP_CFG.read_text().splitlines():
        m = re.match(r"^  ([\w-]+):\s*$", line)
        if m:
            cur = m.group(1) if m.group(1) in out else None
            continue
        if cur and (m := re.match(r"^\s+ttl:\s*(\d+)", line)):
            out[cur]["ttl"] = int(m.group(1))
        if cur and (m := re.match(r"^\s+cmd:\s*(\S+)", line)):
            out[cur]["script"] = m.group(1)
    for d in out.values():
        try:
            txt = Path(d["script"]).read_text()
            m = re.search(r'^MODEL="([^"]+)"', txt, re.M)  # a model dir or an HF id: show its last part
            d["model"] = os.path.basename(m.group(1).rstrip("/")) if m else "?"
            d["model_path"] = m.group(1) if m else None
        except Exception:
            d["model"] = "?"
    _cfg_cache.update(mtime=mtime, tiers=out)
    return out


# ---------------------------------------------------------------------------------------------
# Machine: memory, swap, pressure, GPU. Used whether or not any tier is loaded.
# ---------------------------------------------------------------------------------------------
def sh(*cmd, timeout=3):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


MEM_SYSCTLS = ("hw.pagesize", "vm.page_free_count", "vm.page_pageable_external_count",
               "vm.page_speculative_count", "vm.page_purgeable_count")


def machine_stats():
    m = {"load": os.getloadavg()[0]}
    try:
        m["mem_total"] = int(sh("sysctl", "-n", "hw.memsize").strip())
    except ValueError:
        pass
    # Same headroom as ultron_admit.read_mem: free + file-backed + speculative + purgeable. vm_stat's
    # "Pages inactive" overlaps the free count on macOS 27 (free + inactive read 86 GB of 64 GB).
    v = dict(re.findall(r"^([\w.]+): (\d+)$", sh("sysctl", *MEM_SYSCTLS), re.M))
    if "vm.page_free_count" in v:
        m["mem_available"] = sum(int(v.get(k, 0)) for k in MEM_SYSCTLS[1:]) * int(v.get("hw.pagesize", 16384))
    sw = sh("sysctl", "-n", "vm.swapusage")
    if (a := re.search(r"total = ([\d.]+)M\s+used = ([\d.]+)M", sw)):
        m["swap_total"], m["swap_used"] = float(a.group(1)) * 2**20, float(a.group(2)) * 2**20
    try:
        m["pressure"] = int(sh("sysctl", "-n", "kern.memorystatus_vm_pressure_level").strip())
    except ValueError:
        pass
    gpu = sh("ioreg", "-r", "-d", "1", "-c", "IOAccelerator")
    if (g := re.search(r'"Device Utilization %"=(\d+)', gpu)):
        m["gpu_util"] = int(g.group(1))
    if (g := re.search(r'"In use system memory"=(\d+)', gpu)):
        m["gpu_mem"] = int(g.group(1))
    return m


# ---------------------------------------------------------------------------------------------
# Tier snapshot -> the compact shape the page renders
# ---------------------------------------------------------------------------------------------
def slim(s):
    lt, lat, rol = s.get("lifetime") or {}, s.get("latest") or {}, s.get("rolling") or {}
    mem, bank, pr = s.get("mem") or {}, s.get("session_bank") or {}, s.get("prefill_rates") or {}
    sessions = (s.get("sessions") or {}).get("sessions") or []
    last_t = [x.get("last_access_s") or 0 for x in sessions] + [h["t"] for h in rol.get("history") or []]
    inflight = []
    for f in s.get("in_flight") or []:
        p, pf = f.get("last_progress") or {}, f.get("prefill_state") or {}
        inflight.append({
            "age_s": f.get("age_s"), "prompt_tokens": f.get("prompt_tokens"),
            "preview": (f.get("prompt_preview") or "").replace("\n", " ")[:90],
            "gen_tokens": p.get("completion_tokens"), "decode_tok_s": p.get("decode_tok_s"),
            "prefill": ({"done": pf.get("tokens_done"), "total": pf.get("tokens_total"),
                         "tok_s": pf.get("prefill_tok_s"), "phase": pf.get("phase")} if pf else None),
        })
    return {
        "uptime_s": s.get("uptime_s"),
        "ctx": s.get("context_window"),
        "profile": (s.get("profile") or {}).get("name"),
        "sampler": (s.get("profile") or {}).get("sampler"),
        "active": s.get("active_requests") or 0,
        "inflight": inflight,
        "last_activity": max(last_t) if last_t else lt.get("started_at_s"),
        "lifetime": {k: lt.get(k) for k in ("requests_total", "cancelled_total", "prompt_tokens_total",
                                             "completion_tokens_total", "cached_tokens_total")},
        "rolling": {k: rol.get(k) for k in ("count", "mean", "p50", "p95", "max", "sticky_all_time_max")},
        "latest": request_row(lat) if lat else None,
        "mem": {"footprint": mem.get("phys_footprint_bytes"), "weights": mem.get("model_weights_bytes"),
                "active": mem.get("active_memory_bytes"), "peak": mem.get("peak_memory_bytes")},
        "bank": {"entries": bank.get("entries"), "bytes": bank.get("total_nbytes"),
                 "max": bank.get("effective_max_bytes")},
        "prefill_peak": pr.get("peak_tok_s") if pr.get("samples") else None,
        "machine": s.get("machine"),
        "sys_available": s.get("system_available_bytes"),
        "pressure": s.get("memory_pressure_level"),
    }


def request_row(r):
    drafted, acc = r.get("drafted_tokens") or 0, r.get("accepted_drafts") or 0
    return {
        "t": r.get("logged_at_s"),
        "prompt": r.get("prompt_tokens"), "cached": r.get("cached_tokens"), "gen": r.get("completion_tokens"),
        "cache_src": r.get("cache_source"), "decode": r.get("decode_tok_s"), "prefill": r.get("prefill_tok_s"),
        "e2e": r.get("request_tok_s"), "ttft": r.get("ttft_s"), "elapsed": r.get("request_elapsed_s"),
        "depth": r.get("mtp_depth"), "accept": (acc / drafted) if drafted else None,
        "tools": r.get("request_tool_count"), "client": r.get("request_client_label"),
    }


_plain_ids = {}   # port -> (model id, server name) from /v1/models


def plain_backend(tier, port, err):
    """Tier fields for a server without /v1/mtplx/snapshot (TensorFold): up/down from its /health, model id
    from /v1/models, and what it's doing from LiteLLM's in-flight list (ultron_stats)."""
    base = f"http://127.0.0.1:{port}"
    try:
        get(base + "/health", 2.5, raw=True)
    except Exception:
        return {"state": DOWN, "note": str(err)}
    if port not in _plain_ids:
        try:
            ms = get(base + "/v1/models", 2.5).get("data") or []
            m = next((x for x in ms if x.get("id") != tier), ms[0] if ms else {})
            _plain_ids[port] = (m.get("id"), m.get("owned_by"))
        except Exception:
            _plain_ids[port] = (None, None)
    mid, owner = _plain_ids[port]
    live = [r for r in llm_live() if r.get("endpoint") == f"ultron/{tier}" and r.get("phase") != "ending"]
    gen = next((r for r in live if r.get("phase") == "generating"), None)
    out = {"state": GENERATING if gen else PREFILL if live else IDLE, "backend": owner or "no mtplx snapshot",
           "note": f"{owner or 'this server'} has no mtplx snapshot · per-request stats under Models"}
    if mid:
        out["model"] = mid
    if live:
        f = gen or live[0]
        out["live"] = {"n": len(live), "age": f.get("age"), "prompt": f.get("prompt_est"),
                       "gen": f.get("gen_est"), "tok_s": f.get("tok_s")}
    return out


# ---------------------------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------------------------
class Collector:
    def __init__(self):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.seq = 0
        self.payload = {}
        self.history = {}   # tier -> deque of [t, state, tok_s]
        self.events = deque(maxlen=120)       # load/unload/flag flips
        self.requests = deque(maxlen=120)     # completed requests, all tiers
        self._prev = {}     # tier -> last state, request count, load start
        self._slow = {"t": 0}
        self._services = {"t": 0, "list": []}
        self._routing = {"t": 0, "data": None}
        self._omni = {"t": 0, "data": None}
        self._lora = {"t": 0, "data": None}
        self._llm = {"t": 0, "data": None}

    def event(self, kind, text, tier=None):
        self.events.append({"t": time.time(), "kind": kind, "tier": tier, "text": text})

    def sample(self):
        ts = time.time()
        cfg = tier_config()
        swap_ok, running, swap_err = True, {}, None
        try:
            running = {r["model"]: r for r in get(SWAP + "/running", 1.5)["running"]}
        except Exception as e:
            swap_ok, swap_err = False, str(e)

        tiers = {}
        for t in tier_names():
            r = running.get(t)
            d = {"tier": t, "ttl": cfg[t].get("ttl"), "model": cfg[t].get("model"), "state": UNLOADED}
            if not swap_ok:
                d["state"] = DOWN
            elif r:
                d["port"] = int(r["proxy"].rsplit(":", 1)[1])
                if r["state"] == "starting":
                    d["state"] = LOADING
                elif r["state"] != "ready":
                    d["state"], d["note"] = LOADING, r["state"]
                else:
                    try:
                        s = slim(get(f"http://127.0.0.1:{d['port']}/v1/mtplx/snapshot", 2.5))
                        d["snap"] = s
                        if s["inflight"]:
                            f = s["inflight"][0]
                            d["state"] = GENERATING if f.get("gen_tokens") else PREFILL
                        else:
                            d["state"] = IDLE
                    except Exception as e:  # not mtplx (TensorFold has no snapshot endpoint)
                        d.update(plain_backend(t, d["port"], e))
            d["state_name"] = STATE_NAME[d["state"]]
            self._track(t, d, ts)
            tiers[t] = d

        slow = self._slow
        if ts - slow["t"] >= 3:
            slow.update(t=ts, machine=machine_stats(), litellm=self._litellm(), swap_version=self._swap_version())
        if ts - self._routing["t"] >= 3:
            try:
                data = routing_stats()
                data["flow"] = flow_stats()
                self._routing = {"t": ts, "data": data}
            except Exception as e:
                self._routing = {"t": ts, "data": {"error": str(e), "modes": [mode_state(m) for m in MODES]}}
        if ts - self._llm["t"] >= 3:
            try:
                self._llm = {"t": ts, "data": llm_stats()}
            except Exception as e:
                self._llm = {"t": ts, "data": {"error": str(e)}}
        llm = dict(self._llm["data"] or {})
        llm["live"] = llm_live()
        if ts - self._omni["t"] >= 10:
            self._omni = {"t": ts, "data": omniroute_stats()}
        if ts - self._lora["t"] >= 10:
            try:
                self._lora = {"t": ts, "data": lora_stats()}
            except Exception as e:
                self._lora = {"t": ts, "data": {"error": str(e)}}
        if ts - self._services["t"] >= 10:
            self._services = {"t": ts, "list": services_status()}

        payload = {
            "ts": ts,
            "tiers": tiers,
            "tier_colors": tier_colors(),   # also the display order (tiers.conf order)
            "swap": {"ok": swap_ok, "error": swap_err, "version": slow.get("swap_version"),
                     "loaded": sum(1 for d in tiers.values() if d["state"] >= IDLE and d["state"] != DOWN)},
            "litellm": slow.get("litellm"),
            "machine": slow.get("machine"),
            "flags": [flag_state(f) for f in FLAGS],
            "services": self._services["list"],
            "routing": self._routing["data"],
            "omniroute": self._omni["data"],
            "lora": self._lora["data"],
            "llm": llm,
            "events": list(self.events)[-30:],
            "requests": list(self.requests)[-40:],
        }
        with self.cond:
            self.payload = payload
            self.seq += 1
            self.cond.notify_all()

    def _track(self, t, d, ts):
        """History sample, state-change events, and new completed requests for one tier."""
        s = d.get("snap") or {}
        tok = None
        if s.get("inflight"):
            f = s["inflight"][0]
            tok = f.get("decode_tok_s") if d["state"] == GENERATING else (f.get("prefill") or {}).get("tok_s")
        self.history.setdefault(t, deque(maxlen=HISTORY_S)).append([round(ts, 2), d["state"], round(tok, 1) if tok else None])

        p = self._prev.setdefault(t, {"state": None, "reqs": None, "since": None})
        if p["state"] is not None and p["state"] != d["state"]:
            was, now = p["state"], d["state"]
            if now == LOADING and was == UNLOADED:
                self.event("load", "loading", t); p["since"] = ts
            elif now in (IDLE, PREFILL, GENERATING) and was == LOADING:
                took = f" in {ts - p['since']:.0f}s" if p["since"] else ""
                self.event("load", f"loaded{took}", t)
            elif now == UNLOADED and was != DOWN:
                self.event("unload", "unloaded", t)
            elif now == DOWN:
                self.event("err", d.get("note") or "unreachable", t)
        p["state"] = d["state"]

        reqs = (s.get("lifetime") or {}).get("requests_total")
        if reqs is not None:
            if p["reqs"] is not None and reqs > p["reqs"] and s.get("latest"):
                # snapshot only carries the newest request in `latest`; if several finished within one
                # tick, the rest are summarised by count.
                row = dict(s["latest"], tier=t, t=s["latest"].get("t") or ts, batch=reqs - p["reqs"])
                self.requests.append(row)
            p["reqs"] = reqs
        elif d["state"] in (UNLOADED, LOADING):   # counters restart at 0 on the next load
            p["reqs"] = None

    def _litellm(self):
        t0 = time.time()
        try:
            get(LITELLM + "/health/liveliness", 2, raw=True)
            out = {"ok": True, "ms": round((time.time() - t0) * 1000)}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        try:
            r = get(LITELLM + "/health/readiness", 2)
            out["version"] = r.get("litellm_version")
        except Exception:
            pass
        return out

    def _swap_version(self):
        try:
            return get(SWAP + "/api/version", 1.5).get("version")
        except Exception:
            return None

    def run(self):
        while True:
            t0 = time.time()
            try:
                self.sample()
            except Exception as e:  # never let one bad sample kill the feed
                print(f"sample failed: {e!r}", file=sys.stderr, flush=True)
            time.sleep(max(0.05, TICK_S - (time.time() - t0)))


def flag_state(f):
    try:
        raw = f["path"].read_text().strip()
    except OSError:
        raw = None
    last = None
    try:
        lines = f["log"].read_text().strip().splitlines()
        last = lines[-1] if lines else None
    except OSError:
        pass
    return {"key": f["key"], "label": f["label"], "help": f["help"], "value": raw == "true",
            "raw": raw, "last": last}


def set_flag(key, value, who):
    f = next((x for x in FLAGS if x["key"] == key), None)
    if not f:
        raise KeyError(key)
    word = "true" if value else "false"
    tmp = f["path"].with_suffix(".wanda-tmp")
    tmp.write_text(word + "\n")
    os.replace(tmp, f["path"])
    with open(f["log"], "a") as fh:
        fh.write(f"{now_iso()} set {word} by wanda-panel ({who})\n")
    return word


def _env_file_value(name):
    try:
        for line in LITELLM_ENV.read_text().splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').lower()
    except OSError:
        pass
    return None


def mode_state(m):
    try:
        raw = m["path"].read_text().strip().lower()
    except OSError:
        raw = None
    if raw in m["options"]:
        value, source = raw, "panel"
    else:
        env = _env_file_value(m["env"]) if m["env"] else None
        value, source = (env, "env") if env in m["options"] else (m["default"], "default")
    out = {"key": m["key"], "label": m["label"], "help": m["help"], "options": m["options"],
           "value": value, "source": source}
    out.update((k, m[k]) for k in ("warn", "panel") if k in m)
    return out


def set_mode(key, value, who):
    m = next((x for x in MODES if x["key"] == key), None)
    if not m:
        raise KeyError(key)
    if value not in m["options"]:
        raise ValueError(f"{key} must be one of {', '.join(m['options'])}")
    ULTRON_DIR.mkdir(exist_ok=True)
    tmp = m["path"].with_suffix(".wanda-tmp")
    tmp.write_text(value + "\n")
    os.replace(tmp, m["path"])
    with open(MODE_LOG, "a") as fh:
        fh.write(f"{now_iso()} {key} set {value} by wanda-panel ({who})\n")
    return value


class JsonlWindow:
    """Tail of a jsonl file kept in memory for the last `keep_s` seconds; reads only new bytes."""

    def __init__(self, path, keep_s=86400, maxlen=200000):
        self.path, self.keep_s = path, keep_s
        self.rows = deque(maxlen=maxlen)
        self.offset, self.ino = 0, None

    def refresh(self):
        try:
            st = self.path.stat()
        except OSError:
            return self.rows
        if st.st_ino != self.ino or st.st_size < self.offset:   # rotated or truncated
            self.rows.clear()
            self.ino, self.offset = st.st_ino, max(0, st.st_size - 16 * 1024 * 1024)
        if st.st_size > self.offset:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
            cut = chunk.rfind(b"\n") + 1   # leave a half-written last line for next time
            self.offset += cut
            for line in chunk[:cut].splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("ts"):
                    self.rows.append(r)
        horizon = time.time() - self.keep_s
        while self.rows and self.rows[0]["ts"] < horizon:
            self.rows.popleft()
        return self.rows


ADMIT_ROWS = JsonlWindow(ADMIT_LOG)
LOOP_ROWS = JsonlWindow(LOOPS_LOG)
MEDIA_ROWS = JsonlWindow(LOGS["media"][1])
RESCUE_ROWS = JsonlWindow(LOGS["rescue"][1])


def _short_key(k):
    parts = (k or "").split(":")
    if len(parts) >= 4 and parts[0] == "cc":   # cc:<session>:<agent|main>:<tier>
        agent = "main" if parts[2] == "main" else "sub " + parts[2][:6]
        return f"cc {parts[1][:8]} {agent}"
    return (parts[0] + " " + parts[1][:8]) if len(parts) > 1 else k


def routing_stats():
    now = time.time()
    admits = [r for r in ADMIT_ROWS.refresh() if "target" in r]
    errors_24h = sum(1 for r in ADMIT_ROWS.rows if "error" in r)
    h1 = [r for r in admits if r["ts"] >= now - 3600]

    def split(rows):
        cloud = sum(1 for r in rows if str(r["target"]).startswith("cloud/"))
        by_target = {}
        for r in rows:
            by_target[r["target"]] = by_target.get(r["target"], 0) + 1
        return {"total": len(rows), "cloud": cloud, "local": len(rows) - cloud, "by_target": by_target,
                "applied": sum(1 for r in rows if r.get("applied"))}

    new_h1 = [r for r in h1 if r.get("new")]
    rules = {}
    for r in new_h1:
        k = str(r["rule"]).split("(")[0]
        rules[k] = rules.get(k, 0) + 1
    recent = [{"t": r["ts"], "requested": r.get("requested"), "target": r["target"], "rule": r["rule"],
               "applied": r.get("applied"), "main": r.get("main"), "who": _short_key(r.get("key"))}
              for r in list(admits)[::-1] if r.get("new")][:14]

    pins = {"main": {}, "sub": {}, "mode": None}
    try:
        db = sqlite3.connect(f"file:{PINS_DB}?mode=ro", uri=True, timeout=1)
        rows = db.execute("SELECT mode, target, main, COUNT(*) FROM pins WHERE (main=1 AND last_seen>?) OR (main=0 AND last_seen>?)"
                          " GROUP BY mode, target, main", (now - PIN_IDLE_S["main"], now - PIN_IDLE_S["sub"])).fetchall()
        db.close()
        admit_mode = next(mode_state(m)["value"] for m in MODES if m["key"] == "admit")
        pin_mode = "enforce" if admit_mode == "enforce" else "shadow"
        pins["mode"] = pin_mode
        for mode, target, main, n in rows:
            if mode == pin_mode:
                bucket = pins["main" if main else "sub"]
                bucket[target] = bucket.get(target, 0) + n
    except Exception as e:
        pins["error"] = str(e)

    loops = list(LOOP_ROWS.refresh())
    levels = {"warn": 0, "force": 0, "stop": 0, "shadow": 0}   # shadow: logged what it would have done, nothing changed
    for r in loops:
        if r.get("level") in levels:
            levels["shadow" if r.get("mode") == "shadow" else r["level"]] += 1
    recent_loops = [{"t": r["ts"], "level": r.get("level"), "rule": r.get("rule"), "run": r.get("run"),
                     "tools": r.get("tools"), "mode": r.get("mode"), "model": r.get("model"),
                     "who": _short_key(r.get("key"))} for r in loops[::-1] if r.get("level")][:10]

    return {"modes": [mode_state(m) for m in MODES], "h1": split(h1), "d1": split(admits),
            "new_h1": len(new_h1), "rules_h1": rules, "errors_24h": errors_24h, "recent": recent,
            "pins": pins, "loops_24h": levels, "recent_loops": recent_loops,
            "media": media_stats(), "rescue": rescue_stats()}


def media_stats():
    """ultron_media over 24 h: prompts it matched (by kind), what happened to them, and the video jobs."""
    rows = list(MEDIA_ROWS.refresh())
    hits = [r for r in rows if r.get("kind")]
    kinds = {}
    for r in hits:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    events = [r for r in rows if r.get("event")]
    recent = []
    for r in rows[::-1][:14]:
        if r.get("event"):   # a video job finishing in the background
            recent.append({"t": r["ts"], "event": r["event"], "job": r.get("job"), "model": r.get("model"),
                           "error": (r.get("error") or "")[:240]})
        else:
            recent.append({"t": r["ts"], "kind": r.get("kind"), "mode": r.get("mode"), "applied": r.get("applied"),
                           "agent": r.get("agent"), "confirmed": r.get("confirmed"), "skipped": r.get("skipped"),
                           "deduped": r.get("deduped"), "text": (r.get("text") or "")[:120],
                           "file": r.get("file"), "model": r.get("model_used"), "error": (r.get("error") or "")[:240],
                           "ms": r.get("ms"), "who": _short_key(r.get("key")) or "-"})
    return {"hits": len(hits), "kinds": kinds,
            "applied": sum(1 for r in hits if r.get("applied")),
            "would": sum(1 for r in hits if r.get("mode") == "shadow" and not r.get("skipped")),
            "helper_no": sum(1 for r in hits if r.get("skipped") == "helper said no"),
            "errors": sum(1 for r in hits if r.get("error")),
            "video_done": sum(1 for r in events if r["event"] == "video_done"),
            "video_failed": sum(1 for r in events if r["event"] == "video_failed"),
            "recent": recent}


def rescue_stats():
    rows = [r for r in RESCUE_ROWS.refresh() if r.get("tool") or r.get("error")]
    tools = {}
    for r in rows:
        if r.get("tool"):
            tools[r["tool"]] = tools.get(r["tool"], 0) + 1
    return {"total": len(rows), "applied": sum(1 for r in rows if r.get("applied")), "tools": tools,
            "errors": sum(1 for r in rows if r.get("error")),
            "models": sorted({str(r.get("model")) for r in rows if r.get("model")})}


# ---------------------------------------------------------------------------------------------
# Traffic: which agent (from which host) sent what to which endpoint, from the admission log
# ---------------------------------------------------------------------------------------------
# Tailscale's CLI lives in different places depending on how it was installed, and the LaunchAgent's
# PATH has no /usr/local/bin, so try absolute paths.
TS_CLIS = ("/usr/local/bin/tailscale", "/opt/homebrew/bin/tailscale", "/Applications/Tailscale.app/Contents/MacOS/Tailscale")
STATIC_HOSTS = {"127.0.0.1": NAME.lower(), "::1": NAME.lower(), **dict(
    kv.strip().split("=", 1) for kv in os.environ.get("WANDA_HOST_NAMES", "").split(",") if "=" in kv)}
_hosts = {"t": 0, "map": dict(STATIC_HOSTS)}


def ts_names(status):
    """ip -> tailscale name from `tailscale status --json`. The name is the MagicDNS label (DNSName),
    which is what the admin console shows; HostName is the OS name ('localhost' on iOS)."""
    m = {}
    for peer in list((status.get("Peer") or {}).values()) + [status.get("Self") or {}]:
        name = ((peer.get("DNSName") or "").split(".")[0] or (peer.get("HostName") or "").split(".")[0]).lower()
        name = name.replace("’", "").replace(" ", "-")
        if name and name != "localhost":
            for a in peer.get("TailscaleIPs") or []:
                m[a] = name
    return m


def host_name(ip):
    """Display name for a client IP: its tailscale name, else the raw IP. Raw IPs stay in the logs."""
    if not ip:
        return "?"
    now = time.time()
    # refresh every 5 min, or after 30 s when an IP we don't know shows up (new peer, or a failed refresh)
    if now - _hosts["t"] > 300 or (ip not in _hosts["map"] and now - _hosts["t"] > 30):
        _hosts["t"] = now
        for cli in TS_CLIS:
            try:
                names = ts_names(json.loads(sh(cli, "status", "--json", timeout=4) or "{}"))
            except ValueError:
                continue
            if names:  # an empty answer must not wipe the last good map
                _hosts["map"] = {**STATIC_HOSTS, **names}
                break
    return _hosts["map"].get(ip, ip)


def _endpoint(r):
    return r.get("endpoint") or (r["target"] if r.get("applied") else f"ultron/{r.get('tier')}")


def _agent(r):
    a = r.get("agent") or ("claude-code" if str(r.get("key", "")).startswith("cc:") else "other")
    return a + ("+unreal" if r.get("unreal") else "")


def flow_stats():
    now = time.time()
    rows = [r for r in ADMIT_ROWS.rows if "target" in r and r["ts"] >= now - 3600]
    base_to_agent = {}
    for r in rows:
        base_to_agent[str(r.get("key", "")).rsplit(":", 1)[0]] = (_agent(r), host_name(r.get("ip")))
    loops = [l for l in LOOP_ROWS.rows if l.get("level") and l["ts"] >= now - 3600]
    out = {}
    for w in (300, 900, 3600):
        agents, endpoints, links = {}, {}, {}
        for r in rows:
            if r["ts"] < now - w:
                continue
            aid = f"{_agent(r)}@{host_name(r.get('ip'))}"
            ep = _endpoint(r)
            a = agents.setdefault(aid, {"id": aid, "agent": _agent(r), "host": host_name(r.get("ip")), "n": 0, "last": 0,
                                        "sessions": set(), "loops": 0, "ua": r.get("ua")})
            a["n"] += 1
            a["last"] = max(a["last"], r["ts"])
            a["sessions"].add(str(r.get("key", "")).rsplit(":", 1)[0])
            e = endpoints.setdefault(ep, {"n": 0, "last": 0})
            e["n"] += 1
            e["last"] = max(e["last"], r["ts"])
            k = f"{aid}>{ep}"
            l = links.setdefault(k, {"agent": aid, "endpoint": ep, "n": 0, "last": 0})
            l["n"] += 1
            l["last"] = max(l["last"], r["ts"])
        for l in loops:
            if l["ts"] >= now - w:
                ag = base_to_agent.get(str(l.get("key", "")))
                if ag:
                    aid = f"{ag[0]}@{ag[1]}"
                    if aid in agents:
                        agents[aid]["loops"] += 1
        for a in agents.values():
            a["sessions"] = len(a["sessions"])
        out[str(w)] = {"agents": sorted(agents.values(), key=lambda a: -a["n"]), "endpoints": endpoints,
                       "links": list(links.values()), "total": sum(a["n"] for a in agents.values())}
    out["recent"] = [{"t": r["ts"], "agent": f"{_agent(r)}@{host_name(r.get('ip'))}", "endpoint": _endpoint(r)}
                     for r in rows if r["ts"] >= now - 20]
    return out


# ---------------------------------------------------------------------------------------------
# Per-model request stats from the ultron_stats LiteLLM hook — the same for every backend (mtplx,
# TensorFold, OmniRoute). Agent/host come from the admission log, joined on LiteLLM's call id.
# ---------------------------------------------------------------------------------------------
STATS_ROWS = JsonlWindow(STATS_LOG)
_who = {"t": 0, "map": {}}


def who_by_call():
    """call_id -> "agent@host" from the admission log (last hour)."""
    if time.time() - _who["t"] >= 2:
        now = time.time()
        m = {}
        for r in ADMIT_ROWS.refresh():
            if r.get("call_id") and r["ts"] >= now - 3600:
                m[r["call_id"]] = f"{_agent(r)}@{host_name(r.get('ip'))}"
        _who.update(t=now, map=m)
    return _who["map"]


def llm_live():
    try:
        body = json.loads(STATS_LIVE.read_text())
    except (OSError, ValueError):
        return []
    who, now = who_by_call(), time.time()
    out = []
    for r in body.get("inflight") or []:
        r = dict(r)
        r["age"] = round(now - (r.get("t0") or now), 1)
        r["who"] = who.get(r.get("cid"))
        out.append(r)
    return out


def _pct(vals, q):
    vals = sorted(v for v in vals if v is not None)
    return round(vals[min(len(vals) - 1, int(q * len(vals)))], 2) if vals else None


def _agg(rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    st = [r for r in ok if r.get("stream")]
    tok = lambda k: sum(r.get(k) or 0 for r in rows)
    cached, prompt = tok("cached"), tok("prompt")
    errs = [r for r in rows if r.get("status") in ("error", "lost")]
    last_err = max(errs, key=lambda r: r["ts"], default=None)
    return {
        "last_err": last_err and last_err["ts"], "last_err_msg": last_err and str(last_err.get("error") or last_err["status"])[:300],
        "n": len(rows), "ok": len(ok), "err": sum(1 for r in rows if r.get("status") in ("error", "lost")),
        "cancelled": sum(1 for r in rows if r.get("status") == "cancelled"),
        "mock": sum(1 for r in rows if r.get("status") == "mock"),
        "prompt": prompt, "cached": cached, "gen": tok("gen") or tok("gen_est"),
        "cache_pct": round(100 * cached / prompt) if prompt else None,
        "ttft_p50": _pct([r.get("ttft") for r in st], .5), "ttft_p95": _pct([r.get("ttft") for r in st], .95),
        "seen_p50": _pct([r.get("first_seen") for r in st], .5),
        "decode_p50": _pct([r.get("decode_tok_s") for r in ok], .5),
        "prefill_p50": _pct([r.get("prefill_tok_s") for r in ok], .5),
        "elapsed_p50": _pct([r.get("elapsed") for r in ok], .5), "elapsed_p95": _pct([r.get("elapsed") for r in ok], .95),
        "wait_p95": _pct([r.get("wait") or 0 for r in rows], .95),
        "last": max((r["ts"] for r in rows), default=None),
    }


def llm_stats():
    now = time.time()
    rows = list(STATS_ROWS.refresh())
    who = who_by_call()
    windows = {}
    for w in (900, 3600, 86400):
        by = {}
        for r in rows:
            if r["ts"] >= now - w:
                by.setdefault(r.get("endpoint") or "?", []).append(r)
        windows[str(w)] = {ep: _agg(rs) for ep, rs in by.items()}
    keep = ("ts", "status", "endpoint", "requested", "call_type", "prompt", "prompt_est", "cached", "gen", "gen_est",
            "reasoning", "ttft", "first_seen", "elapsed", "decode_tok_s", "prefill_tok_s", "wait", "error")
    recent = [dict({k: r.get(k) for k in keep if r.get(k) is not None}, who=who.get(r.get("call_id")))
              for r in rows[::-1][:40]]
    return {"windows": windows, "recent": recent, "total_24h": len(rows)}


# ---------------------------------------------------------------------------------------------
# OmniRoute (optional): the calls that came through this stack's key, and the overflow combos' health.
# The key is read here from ~/.litellm/env and never sent to the browser.
# ---------------------------------------------------------------------------------------------
OMNI_KEY_NAME = os.environ.get("WANDA_OMNIROUTE_KEY_NAME", "litellm")   # OmniRoute apiKeyName of that key
LITELLM_CFG = HOME / ".litellm/config.yaml"


def overflow_combos():
    """{tier: combo} from the cloud/* deployments in LiteLLM's config (the single source of truth)."""
    out, cur = {}, None
    try:
        for line in LITELLM_CFG.read_text().splitlines():
            m = re.match(r'\s*- model_name:\s*"?cloud/(\w+)"?', line)
            if m:
                cur = m.group(1)
                continue
            m = re.match(r'\s*model:\s*"?openai/([^"\s]+)"?', line)
            if cur and m:
                out[cur], cur = m.group(1), None
    except OSError:
        pass
    return out


def _env_raw(name):
    try:
        for line in LITELLM_ENV.read_text().splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return None


def omni_base():
    """OmniRoute's base URL (no /v1), or "" when no cloud overflow is configured."""
    base = os.environ.get("WANDA_OMNIROUTE") or _env_raw("OMNIROUTE_BASE") or ""
    return re.sub(r"/v1/?$", "", base.rstrip("/"))


def omni_get(path, timeout=8):
    key = _env_raw("OMNIROUTE_KEY")
    if not key:
        raise RuntimeError("OMNIROUTE_KEY not in ~/.litellm/env")
    req = urllib.request.Request(omni_base() + path, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def omniroute_stats():
    combos = overflow_combos()
    out = {"combos": {}, "calls": [], "tiers": combos, "error": None, "disabled": not omni_base()}
    if out["disabled"]:
        return out
    try:
        logs = omni_get("/api/usage/call-logs?limit=300")
        rows = logs if isinstance(logs, list) else (logs.get("logs") or logs.get("data") or [])
        mine = [r for r in rows if r.get("apiKeyName") == OMNI_KEY_NAME or (r.get("comboName") in combos.values())]
        for r in mine[:60]:
            tok = r.get("tokens") or {}
            out["calls"].append({
                "t": r.get("timestamp"), "combo": r.get("comboName"), "requested": r.get("requestedModel"),
                "model": r.get("model"), "provider": r.get("provider"), "status": r.get("status"),
                "ms": r.get("duration"), "in": tok.get("in"), "out": tok.get("out"), "cache": tok.get("cacheRead"),
                "session": r.get("sessionTag"), "path": r.get("path"), "key": r.get("apiKeyName"),
                "error": (str(r.get("error"))[:160] if r.get("error") else None)})
    except Exception as e:
        out["error"] = f"call-logs: {e}"
    try:
        metrics = omni_get("/api/combos/metrics").get("metrics") or {}
        for tier, combo in combos.items():
            m = metrics.get(combo) or {}
            by = m.get("byModel") or {}
            top = sorted(by.items(), key=lambda kv: -(kv[1].get("requests") or 0))[:3]
            n = m.get("totalRequests") or 0
            out["combos"][tier] = {
                "combo": combo, "requests": n, "ok": m.get("totalSuccesses") or 0, "fail": m.get("totalFailures") or 0,
                "fallbacks": m.get("totalFallbacks") or 0, "strategy": m.get("strategy"),
                "avg_ms": round((m.get("totalLatencyMs") or 0) / n) if n else None, "last": m.get("lastUsedAt"),
                "models": [{"model": k, "requests": v.get("requests"), "rate": v.get("successRate"),
                            "avg_ms": v.get("avgLatencyMs"), "status": v.get("lastStatus")} for k, v in top]}
    except Exception as e:
        out["error"] = (out["error"] + " · " if out["error"] else "") + f"combos/metrics: {e}"
    return out


def services_status():
    try:
        services = json.loads((WWW / "services.json").read_text())
    except Exception:
        return []
    out = []
    for s in services:
        row = {k: s.get(k) for k in ("id", "name", "description", "path", "open", "icon", "disabled", "note")}
        row["href"] = s.get("open") or s.get("path")
        if s.get("disabled"):
            row["state"] = "disabled"
        elif s.get("upstream"):
            probe = s.get("health") or s.get("path") or "/"
            if s.get("path") and probe.startswith(s["path"]):
                probe = "/" + probe[len(s["path"]):]
            try:
                urllib.request.urlopen(f"http://{s['upstream']}{probe}", timeout=2).close()
                row["state"] = "up"
            except urllib.error.HTTPError as e:
                row["state"] = "down" if e.code in (502, 503, 504) else "up"
            except Exception:
                row["state"] = "down"
        else:
            row["state"] = "unknown"
        out.append(row)
    return out


# ---------------------------------------------------------------------------------------------
# LoRA (lora/ in the repo, artifacts in ~/lora): captured traces, training runs, fused packs,
# whatever stage of run.sh is running now, and which tier serves a pack. Empty until you use it.
# ---------------------------------------------------------------------------------------------
TRACE_DIR = Path(os.path.expanduser(os.environ.get("ULTRON_TRACE_DIR", str(ULTRON_DIR / "traces"))))
LORA_DIR = HOME / "lora"
LORA_STAGE_RE = re.compile(r"\b(make_dataset|train|valloss|eval|fuse_pack)\.py\b")
LORA_NAME_RE = re.compile(r"/lora/(?:runs|data|packs)/([^/\s]+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _tail(path, n=8192):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - n))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _lora_run(d):
    name, logs = d.name, d.parent
    out = {"name": name, "t": d.stat().st_mtime, "ckpts": len(list(d.glob("[0-9]*_adapters.safetensors")))}
    m = re.match(r"(.+)-ck(\d+)$", name)   # one checkpoint copied out of a run for fuse_pack.py
    if m:
        out["of_run"], out["of_step"] = m.group(1), int(m.group(2))
    try:
        cfg = json.loads((d / "adapter_config.json").read_text())
        out["tier"] = os.path.splitext(os.path.basename(str(cfg.get("config") or "")))[0] or None
        out["iters"] = cfg.get("iters")
        out["rank"] = (cfg.get("lora_parameters") or {}).get("rank")
    except (OSError, ValueError):
        pass
    vl = re.findall(r"^\s*(base|\d+): val loss ([\d.]+)", _tail(logs / f"{name}-valloss.log"), re.M)
    if vl:
        steps = [(s, float(v)) for s, v in vl if s != "base"]
        out["val_base"] = next((float(v) for s, v in vl if s == "base"), None)
        if steps:
            best = min(steps, key=lambda x: x[1])
            out["val_best"], out["val_best_step"] = best[1], int(best[0])
    for line in _tail(logs / f"{name}-eval.log").splitlines():
        m = re.match(r"^(base|adapter) (\{.*\})\s*$", line)
        if m:
            try:
                out["eval_" + m.group(1)] = json.loads(m.group(2))
            except ValueError:
                pass
    for p in (logs / f"{name}.log", logs / f"{name}-valloss.log", logs / f"{name}-eval.log"):
        try:
            out["t"] = max(out["t"], p.stat().st_mtime)
        except OSError:
            pass
    return out


def lora_active():
    """The run.sh stage running now, from the process list (argv only; never echo it to the page)."""
    try:
        ps = subprocess.run(["ps", "-axo", "etime=,command="], capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return None
    for line in ps.splitlines():
        etime, _, cmd = line.strip().partition(" ")
        if "/lora/" not in cmd or not (m := LORA_STAGE_RE.search(cmd)):
            continue
        n = LORA_NAME_RE.search(cmd)
        act = {"stage": m.group(1), "name": n.group(1) if n else None, "etime": etime}
        if act["stage"] == "train" and act["name"]:
            # the log is progress bars: "val ░░ 0% · 0/8 train ██ 90% · 9/10"; take the train bar's
            prog = re.findall(r"train\b[^/\n]*?(\d+)/(\d+)", ANSI_RE.sub("", _tail(LORA_DIR / "runs" / f"{act['name']}.log", 4096)))
            if prog:
                act["step"], act["of"] = int(prog[-1][0]), int(prog[-1][1])
        return act
    return None


def lora_stats():
    names = set(tier_names())
    traces, newest, size = {}, 0, 0
    for p in list(TRACE_DIR.glob("*.json")) + list((LORA_DIR / "traces").glob("*.json")):
        try:
            st = p.stat()
        except OSError:
            continue
        # trace files are named after the conversation key, which ends in its tier
        tier = p.stem.rsplit("_", 1)[-1] if p.parent == TRACE_DIR else "imported"
        tier = tier if tier in names or tier == "imported" else "other"
        traces[tier] = traces.get(tier, 0) + 1
        newest, size = max(newest, st.st_mtime), size + st.st_size
    runs = []
    for d in (LORA_DIR / "runs").glob("*"):
        if d.is_dir():
            try:
                runs.append(_lora_run(d))
            except OSError:
                pass
    runs.sort(key=lambda r: r["t"], reverse=True)
    packs = []
    for d in (LORA_DIR / "packs").glob("*"):
        if d.is_dir():
            try:
                packs.append({"name": d.name, "t": d.stat().st_mtime,
                              "bytes": sum(f.stat().st_size for f in d.iterdir() if f.is_file())})
            except OSError:
                pass
    packs.sort(key=lambda r: r["t"], reverse=True)
    serving = {t: os.path.basename(c["model_path"].rstrip("/")) for t, c in tier_config().items()
               if "/lora/packs/" in (c.get("model_path") or "")}
    return {"traces": traces, "trace_newest": newest or None, "trace_bytes": size,
            "runs": runs[:12], "packs": packs, "serving": serving, "active": lora_active()}


def tail_log(name, lines=300):
    label, path = LOGS[name]
    if path == "omniroute":
        d = COLLECTOR._omni.get("data") or omniroute_stats()
        if d.get("disabled"):
            return "OmniRoute is not configured (set OMNIROUTE_BASE and OMNIROUTE_KEY in ~/.litellm/env)"
        lines_out = [f"OmniRoute {omni_base()} · calls by key '{OMNI_KEY_NAME}' · newest at the bottom · times UTC"
                     + (f" · ERROR {d['error']}" if d.get("error") else "")]
        for c in reversed(d.get("calls") or []):  # the API lists newest first; logs read top to bottom
            t = str(c.get("t") or "")[11:19]
            lines_out.append(f"{t}  {str(c.get('status')):3}  {str(c.get('combo') or '-'):16} {str(c.get('provider') or ''):12} "
                             f"{str(c.get('model') or ''):28} {str(c.get('ms') or ''):>6}ms  in {c.get('in') or 0} out {c.get('out') or 0}"
                             f"{'  session ' + str(c['session'])[:12] if c.get('session') else ''}{'  ERR ' + c['error'] if c.get('error') else ''}")
        return "\n".join(lines_out)
    if path is None:
        text = get(SWAP + "/logs", 3, raw=True)
    else:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 256 * 1024))
            text = fh.read().decode("utf-8", "replace")
    text = "\n".join(text.splitlines()[-lines:])
    if name in ("admit", "loops", "stats", "media", "rescue"):
        text = "\n".join(pretty_jsonl(name, l) for l in text.splitlines())
    return SECRET_RE.sub(lambda m: (m.group(1) or m.group(2) or "") + "****", text)


def pretty_jsonl(name, line):
    try:
        r = json.loads(line)
    except ValueError:
        return line
    t = time.strftime("%H:%M:%S", time.localtime(r.get("ts", 0)))
    g = lambda k: "" if r.get(k) is None else str(r.get(k))
    if name == "stats":
        num = lambda k, f="{:.1f}": "-" if r.get(k) is None else f.format(r[k])
        prompt = g("prompt") or ("~" + g("prompt_est"))
        gen = g("gen") or ("~" + g("gen_est"))
        return (f"{t}  {g('status'):9} {g('endpoint'):14} {g('requested')[:24]:24} {g('call_type')[:10]:10} "
                f"in {prompt:>7} (cached {g('cached') or 0}) out {gen:>6}  ttft {num('ttft', '{:.2f}')}s  "
                f"seen {num('first_seen', '{:.2f}')}s  {num('elapsed')}s  dec {num('decode_tok_s')} pre {num('prefill_tok_s')} t/s"
                f"{'  wait ' + num('wait') + 's' if r.get('wait') else ''}{'  ' + g('error')[:120] if r.get('error') else ''}")
    if name == "media":
        if r.get("event"):
            return f"{t}  {g('event').upper():12} job {g('job')}  {g('model')}{'  ' + g('error')[:300] if r.get('error') else ''}"
        state = ("applied" if r.get("applied") else "skipped: " + g("skipped") if r.get("skipped")
                 else "dedup" if r.get("deduped") else "shadow" if r.get("mode") == "shadow" else "not applied")
        return (f"{t}  {g('kind'):7} {g('mode'):7} {'agent' if r.get('agent') else 'chat ':5} {state:22} {g('ms'):>5}ms "
                f"{_short_key(r.get('key')) or '-':22} {g('model_used') or g('model'):30} {g('file')}  | {g('text')[:100]}"
                f"{'  ERR ' + g('error')[:200] if r.get('error') else ''}")
    if name == "rescue":
        return (f"{t}  {'applied' if r.get('applied') else g('mode'):8} {g('model'):14} {_short_key(r.get('key')) or '-':22} "
                f"{g('tool'):8} {g('input')[:140]}{'  ERR ' + g('error')[:200] if r.get('error') else ''}")
    if "error" in r:
        return f"{t}  ERROR  {r['error']}"
    if name == "admit":
        state = "applied" if r.get("applied") else "shadow"
        load = " ".join(f"{k}:{v.get('state','?')[:5]}/{v.get('active',0)}+{v.get('waiting',0)}" for k, v in (r.get("loaded") or {}).items())
        return (f"{t}  {'NEW ' if r.get('new') else '    '} {g('requested'):26} → {g('target'):13} "
                f"{g('rule'):28} {state:7} {_short_key(r.get('key')):22} mode={g('route_mode')} [{load}]"
                f"{'  overflow=' + g('overflow') if r.get('overflow') else ''}"
                f"{'  mem: ' + g('mem_tight') if r.get('mem_tight') else ''}")
    return (f"{t}  {g('level').upper():5} {g('rule'):10} run={g('run'):<4} "
            f"{','.join(r.get('tools') or [])[:40]:40} {g('mode'):7} {g('model'):20} {_short_key(r.get('key'))}"
            f"  | {g('result_preview')[:80]}")


def swap_call(tier, action):
    if tier not in tier_names():
        raise KeyError(tier)
    if action == "unload":
        req = urllib.request.Request(f"{SWAP}/api/models/unload/{tier}", method="POST", data=b"")
        urllib.request.urlopen(req, timeout=10).close()
        return "unload requested"
    if action == "load":
        # Any request through /upstream/<tier>/ makes llama-swap start it; a cold 27B takes ~30 s,
        # so fire and forget — the page watches the state go LOADING -> IDLE.
        def warm():
            try:
                urllib.request.urlopen(f"{SWAP}/upstream/{tier}/health", timeout=330).close()
            except Exception as e:
                COLLECTOR.event("err", f"load failed: {e}", tier)
        threading.Thread(target=warm, daemon=True).start()
        return "load requested"
    raise KeyError(action)


# ---------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------
COLLECTOR = Collector()
TOKEN = token()


class Handler(BaseHTTPRequestHandler):
    server_version = "wanda/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep launchd's log for errors, not every poll
        pass

    def send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else (
            body.encode() if isinstance(body, str) else json.dumps(body, default=str).encode())
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def client(self):
        return (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        try:
            if p in ("/", "/index.html"):
                html = (STATIC / "index.html").read_text().replace("__WANDA_TOKEN__", TOKEN).replace("__WANDA_NAME__", NAME)
                return self.send(200, html, "text/html; charset=utf-8")
            if p == "/api/status":
                with COLLECTOR.lock:
                    return self.send(200, COLLECTOR.payload)
            if p == "/api/history":
                with COLLECTOR.lock:
                    return self.send(200, {"ts": time.time(), "history": {t: list(h) for t, h in list(COLLECTOR.history.items())},
                                           "events": list(COLLECTOR.events), "requests": list(COLLECTOR.requests)})
            if p == "/api/stream":
                return self.stream()
            if p == "/api/log":
                name = parse_qs(u.query).get("name", [""])[0]
                if name not in LOGS:
                    return self.send(404, {"error": "unknown log"})
                return self.send(200, {"name": name, "text": tail_log(name)})
            if p == "/healthz":
                return self.send(200, "ok", "text/plain")
            if p == "/ultron-img":   # the hub image: drop your own at ~/.wanda/ultron.{png,jpg,webp,svg}; else the bundled one
                for ext, ctype in (("png", "image/png"), ("jpg", "image/jpeg"), ("webp", "image/webp"), ("svg", "image/svg+xml")):
                    f = STATE_DIR / f"ultron.{ext}"
                    if f.is_file():
                        return self.send(200, f.read_bytes(), ctype, {"Cache-Control": "max-age=300"})
                return self.send(200, (STATIC / "ultron.svg").read_bytes(), "image/svg+xml", {"Cache-Control": "max-age=300"})
            if p.startswith("/icons/") and re.match(r"^/icons/[\w.-]+\.svg$", p):
                for f in (WWW / p.lstrip("/"), STATIC / p.lstrip("/")):   # yours first, then the bundled ones
                    if f.is_file():
                        return self.send(200, f.read_bytes(), "image/svg+xml", {"Cache-Control": "max-age=3600"})
            return self.send(404, {"error": "not found"})
        except BrokenPipeError:
            pass
        except Exception as e:
            self.send(500, {"error": str(e)})

    def stream(self):
        """Server-sent events: one `tick` per collector sample. Caddy flushes text/event-stream."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        seen = -1
        try:
            while True:
                with COLLECTOR.cond:
                    COLLECTOR.cond.wait_for(lambda: COLLECTOR.seq != seen, timeout=15)
                    if COLLECTOR.seq == seen:
                        chunk = b": keepalive\n\n"
                    else:
                        seen = COLLECTOR.seq
                        chunk = b"event: tick\ndata: " + json.dumps(COLLECTOR.payload, default=str).encode() + b"\n\n"
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def do_POST(self):
        # A custom header can't be sent cross-origin without a CORS preflight, which we never answer.
        if not secrets.compare_digest(self.headers.get("X-Wanda-Token", ""), TOKEN):
            return self.send(403, {"error": "bad or missing token — reload the page"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            p = urlparse(self.path).path
            if p == "/api/flag":
                if not isinstance(body.get("value"), bool):
                    return self.send(400, {"error": "value must be true or false"})
                word = set_flag(body.get("key"), body["value"], self.client())
                COLLECTOR.event("flag", f"{body.get('key')} set {word} from {self.client()}")
                return self.send(200, {"ok": True, "value": word})
            if p == "/api/mode":
                value = set_mode(body.get("key"), str(body.get("value") or ""), self.client())
                COLLECTOR.event("flag", f"{body.get('key')} mode set {value} from {self.client()}")
                COLLECTOR._routing["t"] = 0   # repaint on the next tick
                return self.send(200, {"ok": True, "value": value})
            if p == "/api/tier":
                msg = swap_call(body.get("tier"), body.get("action"))
                COLLECTOR.event("action", f"{body.get('action')} requested from {self.client()}", body.get("tier"))
                return self.send(200, {"ok": True, "note": msg})
            return self.send(404, {"error": "not found"})
        except KeyError as e:
            return self.send(400, {"error": f"unknown {e}"})
        except ValueError as e:
            return self.send(400, {"error": str(e)})
        except Exception as e:
            return self.send(502, {"error": str(e)})


def main():
    ULTRON_DIR.mkdir(exist_ok=True)
    for f in (MODE_LOG, ADMIT_LOG, LOOPS_LOG, STATS_LOG):
        try:
            f.touch(exist_ok=True)
        except OSError:
            pass
    threading.Thread(target=COLLECTOR.run, daemon=True, name="collector").start()
    srv = ThreadingHTTPServer((HOST, int(PORT)), Handler)
    srv.daemon_threads = True
    print(f"wanda listening on http://{HOST}:{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
