"""ultron_admit: pick local tier vs OmniRoute overflow per conversation, then pin it.

LiteLLM pre-call hook, registered after loop_breaker. Local tiers first; when a request would
need a model load that doesn't fit next to what's loaded (the llama-swap matrix), a NEW
conversation goes to an OmniRoute combo (cloud/<tier>) instead of making llama-swap swap.
Once a conversation has a backend it never changes (pins in ~/.litellm/pins.sqlite), with one
exception: a cloud pin made by overflow (rule 4:overflow*) is soft. Each later request re-runs
decide(), and once a local tier fits it re-pins local (rule repin:<rule>). Local first, cloud only
while the Mac is full, back to local when there's room. Explicit cloud (x-route: cloud,
route-mode cloud-only) stays a hard pin.

Decision, first request of a conversation (and every request on a soft overflow pin):
  1. tier loaded and its queue < max_waiting                        -> ultron/<tier>
  2. not loaded, fits (matrix) next to loaded + main pins, pressure normal -> ultron/<tier> (cold)
  3. substitute tier (tiers.conf `substitute`) loaded, queue ok    -> ultron/<substitute>
  4. cloud allowed (x-route / route-mode, tier has `cloud`)        -> cloud/<tier>
  5. otherwise                                                      -> ultron/<tier> (llama-swap swaps;
     waits for the evicted tier's in-flight work first)
Any local target that needs a load (rule 5, a pinned conversation whose tier was swapped out, a
vision reroute) waits for the evicted tier's in-flight work: up to OVERFLOW_WAIT_S when cloud is
allowed, then that one request goes to cloud/<tier> (pin unchanged); else up to EVICT_WAIT_MAX_S,
then llama-swap swaps anyway. Cloud never falls back to local: overflow is one-way.
A tier a main-thread conversation used within WARM_MAIN_S counts as in-flight: evicting it would
kill a live session, so requests that would evict it overflow to cloud instead (2026-09-29).
Memory guard (2026-09-30; pressure warn no longer trips it, 2026-10-03): while ultron is nearly
swapping (see mem_guard) and cloud is allowed, a new conversation pins cloud/<tier>
(rule 4:overflow:mem, a soft pin) and a request of a conversation pinned local goes to
cloud/<tier> (overflow=mem), that request only: the pin stays, so it comes back once memory
recovers. Held MEM_HOLD_S after the last trigger, then until headroom is back over MEM_CLEAR_GB,
so it doesn't flap. Kernel pressure "warn" doesn't trip it (only critical does): with three tiers
resident a Mac reads warn all day while the tiers sit idle. Nor does a prompt a tier refused to
load: the guard only sees headroom, not the refusal it can't see coming (refused prompts were
admitted at 4-18 GB headroom). The failure hook sends that conversation to cloud for
REFUSED_HOLD_S (overflow=refused, pin unchanged) and trips the guard for MEM_HOLD_S, so the
client's retry lands on cloud instead of being refused again.
Memory gate (2026-10-01): a local request that can't go to cloud (local-only, x-route: private)
waits until no other tier is serving one (wait_for_other_tiers): two tiers prefilling at once ran
Metal out.
Overrides: header `x-route: cloud` -> cloud/<tier>; `x-route: private` -> never cloud;
~/.ultron/route-mode = auto | local-only | cloud-only.

Modes (~/.ultron/admit-mode, set from the Wanda panel and read per request; else the env
ULTRON_ADMIT_MODE): shadow (default) logs what it would do without rewriting, except
the explicit overrides (x-route: cloud, route-mode cloud-only), which always apply so combo
tests work during the shadow period. enforce rewrites data["model"]. off does nothing.
Vision (every mode but off): a tier with `vision = no` in tiers.conf silently ignores images. A
request whose newest turn carries an image and would go to such a tier goes to the [routing]
vision tier instead, this request only (the pin is unchanged). Images in older turns are replaced
with a text note so the model doesn't invent them.
A tiers.conf model with `routed = no` (the image judge) is only ever asked for by name. It has no
cloud twin and isn't pinned; loading it unloads a tier, so in enforce mode its request first waits
for that tier to go idle, then for the memory gate (admit_direct). A loaded one never blocks a fit.
Tiers, their model ids, queue limits, substitutes and cloud overflow: tiers.conf (ultron_tiers.py).
Which agent sent a request (logged for Wanda): agents.conf.
Design notes: docs/architecture.md and litellm/README.md.
"""

from __future__ import annotations

import asyncio
import base64
import configparser
import fnmatch
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from itertools import product
from typing import Any

import ultron_tiers

# The `routed = no` tier (tiers.conf): asked for by name only, and it can run on another host, so
# it never takes part in the memory or eviction decisions below.
JUDGE = "judge"
# What /decide/v1/systemone answers with when the client's body names no model. Empty = ask the tier.
DECISIONS_MODEL = os.environ.get("ULTRON_DECISIONS_MODEL", "")

# Per-tier behaviour comes from tiers.conf and is read where it is used: think_in_content and
# reasoning_content per tier in the request path, `effort` for the mtplx tiers (see EFFORT_BUDGETS).
PIN_IDLE_S = {"main": 3600, "sub": 1800}
EVICT_WAIT_MAX_S = 300  # local only: wait at most this long for a tier to go idle, then swap (< client timeouts)
# Cloud is the last resort: a new conversation joins a local queue up to the tier's max_waiting (tiers.conf,
# default 20), and a reload waits out the evicted tier's work up to OVERFLOW_WAIT_S before one request spills.
OVERFLOW_WAIT_S = 300  # cloud allowed: after this long, this one request goes to cloud/<tier> instead
SOFT_PIN_RULES = ("4:overflow", "route-mode:tier-lock-swap")  # cloud pins that move back to local once it fits (2026-10-02)
PENDING_LOAD_S = 120  # in-process reservation for a cold load that isn't in /running yet
WARM_MAIN_S = 300  # a tier with a main conversation used within this window is live: never evict it (session-bank idle TTL)
MEM_LOW_GB = float(os.environ.get("ULTRON_MEM_LOW_GB", "4"))  # headroom under this = nearly swapping (0: check off)
SWAPOUT_MB_S = float(os.environ.get("ULTRON_SWAPOUT_MB_S", "16"))  # swap-out rate that counts as swapping (0: check off)
MEM_HOLD_S = 20  # stay on cloud at least this long after the last trigger (a fixed 120 s sent 36 of 85 requests in an hour to cloud while every tier sat idle)
MEM_CLEAR_GB = float(os.environ.get("ULTRON_MEM_CLEAR_GB", "5"))  # ...and until headroom is back over this (three idle tiers leave ~4.5-6 GB on 64 GB)
# Kernel pressure level that trips the guard: 4 = critical. Level 2 (warn) is what a 64 GB Mac reads all day with
# three tiers resident (274 of 300 decisions once went to cloud on "pressure 2" while the tiers sat idle), so warn
# only blocks cold loads (decide rule 2), never requests for a tier that is already loaded.
MEM_PRESSURE_TRIP = int(os.environ.get("ULTRON_PRESSURE_TRIP", "4"))
# mtplx refuses a prefill that won't fit (HTTP 507 "insufficient memory") at any headroom: on 2026-10-04..06, 972
# refusals were admitted at 4-18 GB headroom and pressure 1, and 667 of them were the client retrying the same
# conversation within 60 s, local again each time. A refusal sends that conversation to cloud this long.
REFUSED_HOLD_S = float(os.environ.get("ULTRON_REFUSED_HOLD_S", "300"))
# Local -> cloud fallback in the same request (2026-10-08: after a few retries like 429 it should auto
# retry). Admission routes before a request runs, so a 507 mtplx returns mid-prefill went back to the client
# once LiteLLM's own retries (2, with backoff) on that tier ran out: 19 of ~453 local calls in 3 h. config.yaml's
# router_settings.fallbacks now send ultron/<tier> to cloud/<tier> after those retries, for any error the retries
# didn't fix. Never for x-route: private / local-only (admit sets disable_fallbacks). The deployment hook sees the
# cloud attempt, notes "; fallback=cloud/<tier>" in x-ultron-route and holds the conversation on cloud for
# REFUSED_HOLD_S, as a refusal does (log row event: fallback).
# mtplx also degrades without refusing. Its own memory guard projects each prefill's growth (restoring a session copies
# its prefix: ~1.7 GB at 86k tokens on sonnet) against a floor of ~5-6 GB of free + file-backed memory, and when the
# projection falls short it restores less or nothing ("prefill_admission_shed") and prefills the rest cold, or aborts
# the prefill. 2026-10-07 19:30-19:50: sonnet restored 0 of 75-86k cached tokens three turns running (~200 s each)
# and aborted one prefill, at headroom our guard called fine. A shed that threw away SHED_LOST_TOKENS+ of the
# session's cached prefix, a 32k+ prompt restored from nothing, or an aborted prefill (that tier's snapshot, last
# SHED_LOOKBACK_S) sends new requests for THAT tier to cloud for its SHED_HOLD_S (time only: resident-idle headroom
# can sit at 5-6 GB, 2026-10-03). Tuned 2026-10-07 from the first hour (e943f26: 165 of 407 harness calls, 41%,
# overflowed): miss_tokens also counts the turn's new tokens ("74k/87k" tripped), level-4 pressure trims are mtplx
# relieving pressure, not losing work (14 of 30 trips), and one tier's event moved the others' work too.
SHED_HOLD_S = {t: float(v) for t, v in (kv.split("=") for kv in os.environ.get(
    "ULTRON_SHED_HOLD_S", "opus=60,sonnet=30,haiku=30").split(",") if "=" in kv)}  # per tier; "sonnet=0" turns it off there
SHED_LOST_TOKENS = int(os.environ.get("ULTRON_SHED_LOST_TOKENS", "8192"))
SHED_COLD_TOKENS = 32768  # a prompt this big restored from nothing ("none"): the session's cache was dropped (inferred)
SHED_LOOKBACK_S = 30.0
SHED_SEVERE = ("prefill_system_abort", "prefill_shed_before_abort", "allocation_failure_shed")
# mtplx's disk session cache (opus, sonnet: --ssd-session-cache on) saves each finished session in the background and
# holds the bank entry in memory until the save is written. 2026-10-08 02:36: sonnet had 7.5 GB of saves pending (its own
# cap is 2 GB), a 26 GB footprint, and prefilled a 204k prompt at 136 tok/s (normal: 1,100-3,900) for 21 min. Turning
# the cache off would cost the 8.6% of sonnet calls it restores (1,374 of 16,042 over 4 days, ~19k tokens each), so
# instead (keep the cache, gate on backlog), while a tier has more than BACKLOG_GB of saves pending (snapshot
# persistence_pending_bytes) its new requests go to cloud, until the backlog is under half that.
BACKLOG_GB = float(os.environ.get("ULTRON_BACKLOG_GB", "2"))  # 0: off
MEM_WAIT_MAX_S = float(os.environ.get("ULTRON_MEM_WAIT_S", "300"))  # local only: wait at most this long for other tiers (0: off)
MEM_SEND_S = 3.0  # a request the memory gate let through counts as busy this long (after its tier finishes loading), until its backend shows it

SWAP = os.environ.get("ULTRON_SWAP_URL", "http://127.0.0.1:8001")
SWAP_CONFIG = os.path.expanduser(os.environ.get("ULTRON_SWAP_CONFIG", "~/.llama-swap/config.yaml"))
ROUTE_MODE_FILE = os.path.expanduser(os.environ.get("ULTRON_ROUTE_MODE_FILE", "~/.ultron/route-mode"))
PINS_DB = os.path.expanduser(os.environ.get("ULTRON_PINS_DB", "~/.litellm/pins.sqlite"))
LOG_PATH = os.path.expanduser(os.environ.get("ULTRON_ADMIT_LOG", "~/.litellm/ultron-admit.jsonl"))
AGENTS_CONF = os.path.expanduser(os.environ.get(
    "ULTRON_AGENTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents.conf")))


# ----------------------------------------------------------------------------- inputs

def tiers() -> ultron_tiers.Tiers:
    return ultron_tiers.load()


def tier_for(model: str) -> str | None:
    """Tier for a requested model name; None for explicit cloud/* (the client chose)."""
    return tiers().tier_for(model)


def cloud_for(tier: str) -> bool:
    """The tier has a cloud/<tier> overflow: a `cloud` model in tiers.conf, and OMNIROUTE_BASE
    is not set empty in ~/.litellm/env (empty = run local-only)."""
    return tiers().has_cloud(tier) and os.environ.get("OMNIROUTE_BASE", "unset") != ""


def decide_call_id(data: dict[str, Any], call_type: str) -> str | None:
    """The call id of a POST /judge/v1/decide pass-through, else None. Its data is the judge's own JSON body (kind,
    question, ...) beside LiteLLM's logging object: no model, no messages, no litellm_call_id (the logging object has
    it). None is configured now: JEV-27B-VL served that route for a few hours on 2026-10-08 and moved off ultron."""
    if call_type != "pass_through_endpoint" or data.get("kind") not in ("noul", "score", "choice") or "question" not in data:
        return None
    return str(getattr(data.get("litellm_logging_obj"), "litellm_call_id", None) or "")


def default_decisions_model(data: dict[str, Any], call_type: str) -> bool:
    """POST /decide/v1/systemone (config.yaml pass_through_endpoints -> decisions-api.dev, 2026-10-09) forwards the
    client's body as is; one without a `model` gets DECISIONS_MODEL. Known by its shape: a Decisions API body has
    `questions` (typed questions over one `state`), and it is the only pass-through with that field."""
    if (not DECISIONS_MODEL or call_type != "pass_through_endpoint"
            or not isinstance(data.get("questions"), (dict, list)) or data.get("model")):
        return False
    data["model"] = DECISIONS_MODEL
    return True


def _headers(data: dict[str, Any]) -> dict[str, str]:
    h = (data.get("proxy_server_request") or {}).get("headers") or {}
    return {str(k).lower(): str(v) for k, v in h.items()}


def _first_user_text(messages: list[Any]) -> str:
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") in ("text", "input_text"))
            return str(c or "")
    return ""


IMAGE_TYPES = ("image", "image_url", "input_image")
IMAGE_NOTE = "[image omitted: this model cannot see images; it was shown to a vision model in an earlier turn]"


def _has_image(content: Any) -> bool:
    """Image parts in OpenAI (image_url/input_image) or Anthropic (image, also inside tool_result) content."""
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and (b.get("type") in IMAGE_TYPES or _has_image(b.get("content")))
               for b in content)


def _newest_turn_has_image(messages: list[Any]) -> bool:
    """Images in the messages after the last assistant message (the user turn / tool results being answered)."""
    for m in reversed(messages or []):
        if not isinstance(m, dict) or m.get("role") == "assistant":
            return False
        if _has_image(m.get("content")):
            return True
    return False


def _strip_images(content: Any) -> tuple[Any, int]:
    if not isinstance(content, list):
        return content, 0
    out, n = [], 0
    for b in content:
        if isinstance(b, dict) and b.get("type") in IMAGE_TYPES:
            out.append({"type": "text", "text": IMAGE_NOTE}); n += 1
        elif isinstance(b, dict) and isinstance(b.get("content"), list):
            inner, k = _strip_images(b["content"])
            out.append({**b, "content": inner}); n += k
        else:
            out.append(b)
    return out, n


def vision_fix(data: dict[str, Any], target: str) -> tuple[str, str | None]:
    """(target, note). Newest-turn image on a no-vision tier -> the vision tier; older images -> text note."""
    t = tiers()
    if not target.startswith("ultron/") or target.split("/", 1)[1] not in t.no_vision():
        return target, None
    msgs = data.get("messages") or []
    if _newest_turn_has_image(msgs) and t.vision_tier:
        return f"ultron/{t.vision_tier}", f"vision->{t.vision_tier}"
    n = 0
    for m in msgs:
        if isinstance(m, dict):
            m["content"], k = _strip_images(m.get("content"))
            n += k
    return target, (f"stripped {n} old image(s)" if n else None)


_agents_cache: dict[str, Any] = {"key": None, "rules": []}


def agent_rules() -> list[dict[str, Any]]:
    """agents.conf as [{name, ua, ua_prefix, header, tools_any, tools_all, tag}], re-read when it
    changes. Missing or unreadable: no rules (every client is "other")."""
    try:
        stt = os.stat(AGENTS_CONF)
        key = (stt.st_mtime, stt.st_size)
    except OSError:
        return []
    if key != _agents_cache["key"]:
        rules = []
        try:
            cp = configparser.ConfigParser(interpolation=None)
            cp.read(AGENTS_CONF)
            for name in cp.sections():
                s = cp[name]
                lst = lambda k: [x.strip() for x in (s.get(k) or "").split(",") if x.strip()]  # noqa: E731
                rules.append({"name": name, "ua": [x.lower() for x in lst("ua")],
                              "ua_prefix": [x.lower() for x in lst("ua_prefix")],
                              "header": [x.lower() for x in lst("header")],
                              "tools_any": [x.lower() for x in lst("tools_any")], "tools_all": lst("tools_all"),
                              "tag": (s.get("tag") or "").strip().lower() in ("1", "yes", "true", "on")})
        except configparser.Error:
            rules = _agents_cache["rules"]
        _agents_cache.update(key=key, rules=rules)
    return _agents_cache["rules"]


def _rule_matches(r: dict[str, Any], ua: str, h: dict[str, str], names: list[str]) -> bool:
    low = [n.lower() for n in names]
    for hv in r["header"]:
        k, _, v = hv.partition("=")
        if k.strip() in h and (not v or v.strip() in h[k.strip()].lower()):
            return True
    return (any(x in ua for x in r["ua"]) or any(ua.startswith(x) for x in r["ua_prefix"])
            or any(fnmatch.fnmatchcase(n, g) for g in r["tools_any"] for n in low)
            or bool(r["tools_all"]) and set(r["tools_all"]) <= set(names))


def client_info(data: dict[str, Any]) -> dict[str, Any]:
    """Which agent sent this and from where (for Wanda's traffic view). Best effort: the first
    agents.conf section that matches the User-Agent, headers or tool set; tag sections add a label."""
    h = _headers(data)
    ua = h.get("user-agent", "")
    names = []
    for t in data.get("tools") or []:
        if isinstance(t, dict):
            names.append(str(t.get("name") or (t.get("function") or {}).get("name") or ""))
    agent, tags = "other", []
    for r in agent_rules():
        if (r["tag"] or agent == "other") and _rule_matches(r, ua.lower(), h, names):
            if r["tag"]:
                tags.append(r["name"])
            else:
                agent = r["name"]
    md = data.get("metadata") or {}
    lmd = data.get("litellm_metadata") or {}
    ip = str(md.get("requester_ip_address") or lmd.get("requester_ip_address") or "").split(",")[0].strip()
    return {"agent": agent, "tags": tags or None, "ua": ua[:80] or None, "ip": ip or None, "tools": len(names)}


def pin_identity(data: dict[str, Any], tier: str) -> dict[str, Any]:
    """Pin key: Claude Code session + agent id (absent = main thread) + tier; for clients
    without the header (Hermes, pi) a sha256 of the first user message + tier."""
    h = _headers(data)
    session = h.get("x-claude-code-session-id")
    agent = h.get("x-claude-code-agent-id")
    if session:
        base = f"cc:{session}:{agent or 'main'}"
    else:
        session = hashlib.sha256(_first_user_text(data.get("messages")).encode()).hexdigest()[:24]
        base = f"h:{session}"
    return {"key": f"{base}:{tier}", "session": session, "main": not agent}


ADMIT_MODE_FILE = os.path.expanduser(os.environ.get("ULTRON_ADMIT_MODE_FILE", "~/.ultron/admit-mode"))


def admit_mode() -> str:
    """Mode file (flipped live from the Wanda panel) wins over ULTRON_ADMIT_MODE from the env."""
    try:
        v = open(ADMIT_MODE_FILE).read().strip().lower()
    except OSError:
        v = ""
    return v if v in ("enforce", "shadow", "off") else os.environ.get("ULTRON_ADMIT_MODE", "shadow").lower()


TIER_LOCK_FILE = os.path.expanduser(os.environ.get("ULTRON_TIER_LOCK_FILE", "~/.ultron/tier-lock"))
TIER_SWAP_FILE = os.path.expanduser(os.environ.get("ULTRON_TIER_SWAP_FILE", "~/.ultron/tier-lock-swap"))


def _read_word(path: str) -> str:
    try:
        return open(path).read().strip().lower()
    except OSError:
        return ""


def tier_lock() -> str | None:
    """Locked heavy tier (fable|opus) that serves both claude-fable* and claude-opus*; None = off."""
    v = _read_word(TIER_LOCK_FILE)
    return v if v in ("fable", "opus") else None


def locked_tier(tier: str | None) -> str | None:
    """Local tier that serves a request for `tier` under the lock (fable<->opus collapse onto the locked one)."""
    lock = tier_lock()
    return lock if lock and tier in ("fable", "opus") else tier


def lock_swapping(tier: str | None) -> bool:
    """Wanda is swapping the lock: fable/opus requests go to cloud/<their own tier> until the locked tier is up."""
    return tier in ("fable", "opus") and bool(_read_word(TIER_SWAP_FILE))


def route_mode() -> str:
    try:
        mode = open(ROUTE_MODE_FILE).read().strip().lower()
    except OSError:
        return "auto"
    return mode if mode in ("auto", "local-only", "cloud-only") else "auto"


def _get_json(url: str, timeout: float = 1.5) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


MEM_SYSCTLS = ("hw.pagesize", "kern.memorystatus_vm_pressure_level", "vm.page_free_count",
               "vm.page_pageable_external_count", "vm.page_speculative_count", "vm.page_purgeable_count",
               "vm.compressor.swapper.swapouts_total", "vm.compressor_bytes_used", "vm.swapusage")


def read_mem() -> dict[str, Any]:
    """Pressure level, headroom and the swap-out counter. Headroom is what macOS can hand out before
    it has to compress/swap anonymous memory (model weights, KV caches): free + file-backed +
    speculative + purgeable pages. The pressure level alone comes too late: on 2026-09-30 ultron
    swapped ~9 GB in 3 min while it still read 1 (normal)."""
    try:
        out = subprocess.run(["sysctl", *MEM_SYSCTLS], capture_output=True, text=True, timeout=2).stdout
    except Exception:
        out = ""
    v: dict[str, int] = {}
    for line in out.splitlines():  # "name: value"; an unknown oid only drops its own line
        k, _, val = line.partition(":")
        if val.strip().isdigit():
            v[k.strip()] = int(val.strip())
    page = v.get("hw.pagesize", 16384)
    swap = re.search(r"vm\.swapusage:.*?used = ([\d.]+)M", out)  # "total = 6144.00M  used = 5203.62M  free = ..."
    return {"t": time.time(), "page": page, "pressure": v.get("kern.memorystatus_vm_pressure_level", 1),
            "headroom": sum(v.get(k, 0) for k in MEM_SYSCTLS[2:6]) * page if "vm.page_free_count" in v else None,
            "swapouts": v.get("vm.compressor.swapper.swapouts_total"),
            "compressor": v.get("vm.compressor_bytes_used"),  # telemetry only
            "swap_used": int(float(swap.group(1)) * 2**20) if swap else None}


def read_state() -> dict[str, Any]:
    """{'tiers': {tier: {'state', 'active', 'waiting'}}, 'pressure': 1|2|4, 'mem': read_mem()}.
    Blocking; run in a thread."""
    names = tiers().names
    loaded: dict[str, dict[str, Any]] = {}
    for r in _get_json(f"{SWAP}/running").get("running", []):
        tier = r.get("model")
        if tier not in names:
            continue
        info = {"state": r.get("state"), "active": 0, "waiting": 0}
        if r.get("state") == "ready" and r.get("proxy"):
            try:
                snap = _get_json(r["proxy"].rstrip("/") + "/v1/mtplx/snapshot")
                info["active"] = int(snap.get("active_requests") or 0)
                tel = (snap.get("scheduler") or {}).get("telemetry") or {}
                info["waiting"] = int(tel.get("foreground_pending") or 0)
                info["backlog"] = int(tel.get("persistence_pending_bytes") or 0)
                m = snap.get("mem") or {}  # telemetry: process footprint and its part outside MLX (SSD-save copies)
                info["fp"], info["host"] = m.get("phys_footprint_bytes"), m.get("host_overhang_bytes")
                shed = shed_event(snap.get("memory_guard_events"), time.time())
                if shed:
                    info["shed"] = shed
            except Exception:  # not mtplx (e.g. another server): count from ultron_stats' in-flight file
                info["cids"] = None
        loaded[tier] = info
    if any("cids" in i for i in loaded.values()):
        live = litellm_inflight()
        for tier, i in loaded.items():
            if "cids" in i:
                i["cids"] = live.get(tier, [])
    mem = read_mem()
    return {"tiers": loaded, "pressure": mem["pressure"], "mem": mem}


def shed_event(events: Any, now: float) -> dict[str, Any] | None:
    """The newest mtplx memory-guard event (snapshot `memory_guard_events`) in the last SHED_LOOKBACK_S that threw
    work away: an aborted prefill, or an admission shed that restored SHED_LOST_TOKENS+ less than the session had
    cached (kept_session_entries.longest_shared_prefix_tokens) or restored nothing of a SHED_COLD_TOKENS+ prompt."""
    best = None
    for e in events if isinstance(events, list) else []:
        if not isinstance(e, dict) or not isinstance(e.get("ts"), (int, float)) or now - e["ts"] > SHED_LOOKBACK_S:
            continue
        a = e.get("action")
        if a in SHED_SEVERE:
            why = a
        elif a == "prefill_admission_shed":
            reuse, prompt = int(e.get("reusable_prefix_tokens") or 0), int(e.get("prompt_tokens") or 0)
            shared = (e.get("kept_session_entries") or {}).get("longest_shared_prefix_tokens")
            lost = int(shared) - reuse if isinstance(shared, int) else 0
            if lost >= SHED_LOST_TOKENS:
                why = f"shed lost {lost // 1000}k ({reuse // 1000}k/{prompt // 1000}k)"
            elif reuse == 0 and e.get("reusable_prefix_mode") == "none" and prompt >= SHED_COLD_TOKENS:
                why = f"shed cold 0k/{prompt // 1000}k"
            else:
                continue
        else:
            continue
        if best is None or e["ts"] > best["ts"]:
            best = {"ts": e["ts"], "why": why}
    return best


STATS_LIVE = os.path.expanduser(os.environ.get("ULTRON_STATS_LIVE", "~/.litellm/stats-live.json"))


def litellm_inflight() -> dict[str, list[str]]:
    """tier -> call ids LiteLLM has in flight to ultron/<tier> and not yet finishing (ultron_stats)."""
    try:
        body = json.load(open(STATS_LIVE))
    except (OSError, ValueError):
        return {}
    out: dict[str, list[str]] = {}
    for r in body.get("inflight") or []:
        ep = str(r.get("endpoint") or "")
        if ep.startswith("ultron/") and r.get("phase") != "ending":
            out.setdefault(ep.split("/", 1)[1], []).append(str(r.get("cid")))
    return out


def own_view(state: dict[str, Any], cid: str | None, skip: Any = ()) -> dict[str, Any]:
    """Tiers counted from LiteLLM include this request itself (and requests still held at the
    memory gate, `skip`); drop them. Serial backends: one runs, the rest wait."""
    if not any("cids" in i for i in state["tiers"].values()):
        return state
    tiers = {}
    for t, i in state["tiers"].items():
        i = dict(i)
        if "cids" in i:
            n = len([c for c in i.pop("cids") or [] if c != cid and c not in skip])
            i["active"], i["waiting"] = min(n, 1), max(0, n - 1)
        tiers[t] = i
    return {**state, "tiers": tiers}


# ----------------------------------------------------------------------------- matrix

def _expand(expr: str, sets: dict[str, str], vars_: dict[str, str], depth: int = 0) -> list[frozenset[str]]:
    """llama-swap matrix expression -> concrete sets. Grammar: & | () +ref, vars."""
    if depth > 8:
        raise ValueError("matrix +ref recursion")
    tokens = re.findall(r"\+?[A-Za-z0-9._/-]+|[&|()]", expr)
    pos = 0

    def atom() -> list[frozenset[str]]:
        nonlocal pos
        t = tokens[pos]
        pos += 1
        if t == "(":
            r = alt()
            pos += 1  # ')'
            return r
        if t.startswith("+"):
            return _expand(sets[t[1:]], sets, vars_, depth + 1)
        return [frozenset([vars_.get(t, t)])]

    def conj() -> list[frozenset[str]]:
        nonlocal pos
        r = atom()
        while pos < len(tokens) and tokens[pos] == "&":
            pos += 1
            r = [a | b for a, b in product(r, atom())]
        return r

    def alt() -> list[frozenset[str]]:
        nonlocal pos
        r = conj()
        while pos < len(tokens) and tokens[pos] == "|":
            pos += 1
            r = r + conj()
        return r

    return alt()


_matrix_cache: dict[str, Any] = {"mtime": None, "sets": None, "costs": {}}


def matrix() -> tuple[list[frozenset[str]], dict[str, float]]:
    """Allowed co-resident sets from llama-swap's config (the same matrix it enforces)."""
    try:
        mtime = os.path.getmtime(SWAP_CONFIG)
        if mtime != _matrix_cache["mtime"]:
            import yaml
            cfg = yaml.safe_load(open(SWAP_CONFIG)) or {}
            m = (((cfg.get("routing") or {}).get("router") or {}).get("settings") or {}).get("matrix") or {}
            raw, vars_ = m.get("sets") or {}, m.get("vars") or {}
            out: list[frozenset[str]] = []
            for expr in raw.values():
                out += _expand(str(expr), raw, vars_)
            _matrix_cache.update(mtime=mtime, sets=out,
                                 costs={vars_.get(k, k): float(v) for k, v in (m.get("evict_costs") or {}).items()})
    except Exception:
        pass
    if not _matrix_cache["sets"]:  # llama-swap's config unreadable: assume one tier at a time
        return [frozenset({t}) for t in tiers().names], {}
    return _matrix_cache["sets"], _matrix_cache["costs"]


def fits(tier: str, resident: set[str], sets: list[frozenset[str]]) -> bool:
    need = resident | {tier}
    return any(need <= s for s in sets)


def evictees(tier: str, loaded: set[str], sets: list[frozenset[str]], costs: dict[str, float]) -> set[str]:
    """What llama-swap's solver would unload to start `tier` (cheapest set containing it)."""
    cands = [s for s in sets if tier in s] or [frozenset({tier})]
    best = min(cands, key=lambda s: sum(costs.get(m, 1.0) for m in loaded - s))
    return loaded - best


# ----------------------------------------------------------------------------- pins

class Pins:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS pins (mode TEXT, key TEXT, target TEXT, tier TEXT, main INTEGER,"
            " session TEXT, rule TEXT, created REAL, last_seen REAL, requests INTEGER, PRIMARY KEY (mode, key))"
        )

    def expire(self, now: float) -> None:
        self.db.execute("DELETE FROM pins WHERE (main=1 AND last_seen < ?) OR (main=0 AND last_seen < ?)",
                        (now - PIN_IDLE_S["main"], now - PIN_IDLE_S["sub"]))

    def get(self, mode: str, key: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT target, tier, main, session, rule, created, requests FROM pins WHERE mode=? AND key=?",
                              (mode, key)).fetchone()
        if not row:
            return None
        return dict(zip(("target", "tier", "main", "session", "rule", "created", "requests"), row))

    def touch(self, mode: str, key: str, now: float) -> None:
        self.db.execute("UPDATE pins SET last_seen=?, requests=requests+1 WHERE mode=? AND key=?", (now, mode, key))

    def put(self, mode: str, key: str, target: str, tier: str, main: bool, session: str, rule: str, now: float) -> None:
        self.db.execute("INSERT OR REPLACE INTO pins VALUES (?,?,?,?,?,?,?,?,?,1)",
                        (mode, key, target, tier, int(main), session, rule, now, now))

    def main_local_tiers(self, mode: str) -> set[str]:
        """Local tiers that live main-thread conversations are pinned to (memory reservations)."""
        rows = self.db.execute("SELECT DISTINCT target FROM pins WHERE mode=? AND main=1 AND target LIKE 'ultron/%'",
                               (mode,)).fetchall()
        return {r[0].split("/", 1)[1] for r in rows}

    def warm_main_tiers(self, mode: str, now: float) -> set[str]:
        """Tiers a main-thread conversation used within WARM_MAIN_S: evicting one kills a live session."""
        rows = self.db.execute(
            "SELECT DISTINCT target FROM pins WHERE mode=? AND main=1 AND target LIKE 'ultron/%' AND last_seen > ?",
            (mode, now - WARM_MAIN_S)).fetchall()
        return {r[0].split("/", 1)[1] for r in rows}


# ----------------------------------------------------------------------------- decision

def decide(tier: str, state: dict[str, Any], reserved: set[str], x_route: str, mode: str,
           sets: list[frozenset[str]]) -> tuple[str, str]:
    """(target, rule) for a new conversation. Pure: state/reservations are passed in."""
    private = x_route == "private"
    cloud = cloud_for(tier)  # no cloud model for this tier (or none configured): every rule stays local
    if x_route == "cloud" and cloud:
        return f"cloud/{tier}", "x-route:cloud"
    if mode == "cloud-only" and not private and cloud:
        return f"cloud/{tier}", "route-mode:cloud-only"
    tight = state.get("mem_tight")
    if tight and not private and mode != "local-only" and cloud:  # any local start grows a KV cache toward swap
        return f"cloud/{tier}", "4:overflow:mem"
    conf = tiers().tier
    # A loaded routed = no model (the judge) never blocks a fit: llama-swap unloads it for any tier.
    loaded = {t for t, i in state["tiers"].items()
              if i.get("state") in ("ready", "starting") and (conf.get(t) or {}).get("routed", True)}
    reserved = {t for t in reserved if t in loaded}  # a pin only reserves memory while its tier is resident
    info = state["tiers"].get(tier)

    def queue_ok(t: str) -> bool:
        i = state["tiers"].get(t) or {}
        return i.get("state") == "ready" and int(i.get("waiting", 0)) < (conf.get(t) or {}).get("max_waiting", 1)

    if tier in loaded and (info or {}).get("state") == "starting":
        return f"ultron/{tier}", "1:loading"
    if tier in loaded and queue_ok(tier):
        return f"ultron/{tier}", "1:loaded"
    if tier not in loaded and fits(tier, loaded | reserved, sets) and state.get("pressure", 1) <= 1 and not tight:
        return f"ultron/{tier}", "2:cold-fits"
    sub = (conf.get(tier) or {}).get("substitute")
    if sub and sub in loaded and queue_ok(sub):
        return f"ultron/{sub}", f"3:substitute({tier}->{sub})"
    if not private and mode != "local-only" and cloud:
        return f"cloud/{tier}", "4:overflow" + (":queue" if tier in loaded else ":no-fit")
    return f"ultron/{tier}", "5:local-swap"


# ----------------------------------------------------------------------------- hook

class Admission:
    def __init__(self) -> None:
        self.pins: Pins | None = None
        self._lock: asyncio.Lock | None = None
        self.pending: dict[str, float] = {}  # tier -> time a cold load was decided (not yet in /running)
        self.recent: OrderedDict[str, str] = OrderedDict()  # litellm_call_id -> response header value
        self.keys: OrderedDict[str, str] = OrderedDict()  # litellm_call_id -> conversation key (trace file)
        self._state: tuple[float, dict[str, Any]] | None = None
        self._mem_prev: dict[str, Any] | None = None  # last read_mem(), for the swap-out rate
        self._mem_tight: tuple[float, str] | None = None  # (held until, reason)
        self.shed_holds: dict[str, tuple[float, float, str]] = {}  # tier -> (shed ts, held until, why)
        self.backlogs: dict[str, tuple[float, float]] = {}  # tier -> (since, GB of disk saves pending) while over BACKLOG_GB
        self.mem_queue: OrderedDict[str, str] = OrderedDict()  # ticket -> tier, requests held at the memory gate (FIFO)
        self.mem_sent: dict[str, float] = {}  # tier -> when the memory gate last let one of its requests through
        self.refused: dict[str, float] = {}  # conversation key -> until when mtplx's memory refusal keeps it on cloud
        self.mocked: OrderedDict[str, str] = OrderedDict()  # litellm_call_id -> hook that answered it (mock_response)
        self.sessions: OrderedDict[str, str] = OrderedDict()  # mtplx session id -> litellm_call_id last sent on it

    def take_session(self, session: str, cid: str) -> bool:
        """Whether this request may use the conversation's named mtplx session. mtplx answers 409 to a second
        request in flight on one named session, so while another request of the conversation is still in
        flight to a local tier (ultron_stats' in-flight file) this one goes without (anon, as before)."""
        owner = self.sessions.get(session)
        if owner and owner != cid and owner in {c for cids in litellm_inflight().values() for c in cids}:
            return False
        self.sessions[session] = cid
        self.sessions.move_to_end(session)
        while len(self.sessions) > 2048:
            self.sessions.popitem(last=False)
        return True


    @property
    def lock(self) -> asyncio.Lock:
        # Made on first use, inside LiteLLM's running loop: on Python 3.9 a Lock binds to the loop
        # that is current when it's created, and this object is built before that loop runs.
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _pins(self) -> Pins:
        if self.pins is None:
            self.pins = Pins(PINS_DB)
        return self.pins

    async def state(self) -> dict[str, Any]:
        now = time.time()
        if self._state and now - self._state[0] < 1.0:
            return self._state[1]
        s = await asyncio.to_thread(read_state)
        s["mem_tight"] = self.mem_guard(s.get("mem") or {})
        for entry in self.note_backlogs(s["tiers"], now):
            _log(entry)
        for tier, info in s["tiers"].items():
            if self.note_shed(tier, info.get("shed")):  # one row per shed that trips a hold (Wanda ADMIT tab, harness telemetry)
                head = (s.get("mem") or {}).get("headroom")
                _log({"ts": now, "event": "shed", "tier": tier, "why": info["shed"]["why"], "shed_ts": info["shed"]["ts"],
                      "hold_s": SHED_HOLD_S.get(tier, 0), "headroom_gb": None if head is None else round(head / 2**30, 1)})
        self._state = (now, s)
        return s

    def note_shed(self, tier: str, shed: dict[str, Any] | None) -> bool:
        """Start (or extend) `tier`'s hold for a shed newer than the last one seen; True when it did."""
        hold = SHED_HOLD_S.get(tier, 0)
        if not hold or not shed or shed["ts"] <= self.shed_holds.get(tier, (0.0,))[0]:
            return False
        self.shed_holds[tier] = (shed["ts"], shed["ts"] + hold, f"mtplx {tier} {shed['why']}")
        return True

    def note_backlogs(self, tiers: dict[str, dict[str, Any]], now: float) -> list[dict[str, Any]]:
        """Track each tier's pending disk saves against BACKLOG_GB (on over it, off under half); log rows on change."""
        out = []
        for tier in set(tiers) | set(self.backlogs):
            gb = int(tiers.get(tier, {}).get("backlog") or 0) / 2**30
            if tier not in self.backlogs and BACKLOG_GB and gb > BACKLOG_GB:
                self.backlogs[tier] = (now, gb)
                out.append({"ts": now, "event": "backlog", "tier": tier, "pending_gb": round(gb, 1)})
            elif tier in self.backlogs and (not BACKLOG_GB or gb < BACKLOG_GB / 2 or tier not in tiers):
                since = self.backlogs.pop(tier)[0]
                out.append({"ts": now, "event": "backlog_clear", "tier": tier, "pending_gb": round(gb, 1),
                            "held_s": round(now - since)})
            elif tier in self.backlogs:
                self.backlogs[tier] = (self.backlogs[tier][0], gb)
        return out

    def backlog_hold(self, tier: str) -> str | None:
        """Why new requests for `tier` should go to cloud (mtplx's disk saves are backed up), else None."""
        b = self.backlogs.get(tier)
        return f"mtplx {tier} {b[1]:.1f}G saves pending" if b else None

    def shed_hold(self, tier: str, now: float) -> str | None:
        """Why new requests for `tier` should go to cloud right now (a recent mtplx shed on it), else None."""
        h = self.shed_holds.get(tier)
        return h[2] if h and now < h[1] else None

    def mem_guard(self, m: dict[str, Any]) -> str | None:
        """Why ultron is (nearly) swapping, or None: pressure critical (MEM_PRESSURE_TRIP), headroom under
        MEM_LOW_GB, or swap-outs of SWAPOUT_MB_S+ since the last sample (<= 30 s ago). Held for
        MEM_HOLD_S after the last trigger, then until headroom is back over MEM_CLEAR_GB
        (hysteresis, 2026-10-02: a fixed 120 s hold sent 36 of 85 requests in an hour to cloud
        while every tier sat idle with 6+ GB free). mtplx's own sheds are per tier: shed_hold."""
        now = m.get("t") or time.time()
        why = None
        if int(m.get("pressure") or 1) >= MEM_PRESSURE_TRIP:
            why = f"pressure {m['pressure']}"
        elif m.get("headroom") is not None and m["headroom"] < MEM_LOW_GB * 2**30:
            why = f"headroom {m['headroom'] / 2**30:.1f}G"
        prev, self._mem_prev = self._mem_prev, m
        if (not why and SWAPOUT_MB_S and prev and m.get("swapouts") is not None
                and prev.get("swapouts") is not None and 0 < now - prev["t"] <= 30):
            rate = (m["swapouts"] - prev["swapouts"]) * m.get("page", 16384) / (now - prev["t"]) / 2**20
            if rate >= SWAPOUT_MB_S:
                why = f"swapping {rate:.0f}MB/s"
        if why:
            self._mem_tight = (now + MEM_HOLD_S, why)
        if not self._mem_tight:
            return None
        head = m.get("headroom")
        if now < self._mem_tight[0] or (head is not None and head < MEM_CLEAR_GB * 2**30):
            return self._mem_tight[1]
        self._mem_tight = None
        return None

    def note_refusal(self, data: dict[str, Any], exc: Exception) -> dict[str, Any] | None:
        """A local tier answered 507 "insufficient memory" (mtplx's own prefill check): keep this
        conversation on cloud for REFUSED_HOLD_S and trip the memory guard for MEM_HOLD_S."""
        model, msg = str(data.get("model") or ""), str(exc)
        if not model.startswith("ultron/") or "insufficient memory" not in msg:
            return None
        tier, cid, now = model.split("/", 1)[1], data.get("litellm_call_id"), time.time()
        key = self.keys.get(cid or "") or pin_identity(data, tier)["key"]
        self.refused = {k: t for k, t in self.refused.items() if t > now}
        self.refused[key] = now + REFUSED_HOLD_S
        self._mem_tight = (now + MEM_HOLD_S, f"refused {tier}")
        self._state = None
        detail = msg[msg.find("insufficient memory"):][:300]
        return {"ts": now, "event": "refused", "model": model, "tier": tier, "key": key, "call_id": cid,
                "harness_call_id": _headers(data).get("x-harness-call-id"), "hold_s": int(REFUSED_HOLD_S), "detail": detail}

    def note_fallback(self, cid: str, group: str, headers: dict[str, str]) -> dict[str, Any] | None:
        """LiteLLM's router fell back from the local tier admission chose to `group` (cloud/<tier>) after its
        retries failed: say so in x-ultron-route and keep the conversation on cloud for REFUSED_HOLD_S."""
        route = self.recent.get(cid, "")
        if not route.startswith("ultron/") or "; fallback=" in route:  # hook can run twice per /v1/messages
            return None
        tier, now = route.split(";", 1)[0].split("/", 1)[1], time.time()
        key = self.keys.get(cid)
        self.recent[cid] = route + f"; fallback={group}"
        if key:
            self.refused = {k: t for k, t in self.refused.items() if t > now}
            self.refused[key] = now + REFUSED_HOLD_S
        return {"ts": now, "event": "fallback", "tier": tier, "to": group, "key": key, "call_id": cid,
                "harness_call_id": headers.get("x-harness-call-id"), "hold_s": int(REFUSED_HOLD_S) if key else 0}

    async def wait_for_evictees(self, tier: str, sets, costs, cid: str | None = None,
                                max_s: float = EVICT_WAIT_MAX_S, warm: set[str] = ()) -> list[str]:
        """Rule 5 / pinned reloads: never evict a tier with a request in flight or queued,
        or a main-thread conversation used within WARM_MAIN_S (evicting it kills a live session)."""
        deadline = time.time() + max_s
        while True:
            self._state = None
            st = own_view(await self.state(), cid)  # not ourselves: a vision reroute is listed under its original tier
            loaded = {t for t, i in st["tiers"].items() if i.get("state") in ("ready", "starting")}
            if tier in loaded:
                return []
            busy = [t for t in evictees(tier, loaded, sets, costs)
                    if st["tiers"][t].get("active") or st["tiers"][t].get("waiting") or t in warm]
            if not busy or time.time() > deadline:
                return busy
            await asyncio.sleep(1.0)

    async def wait_for_other_tiers(self, tier: str, cid: str | None, max_s: float) -> list[str]:
        """Local only: hold this request while another tier serves one, first come first served.
        A coding agent's main thread on sonnet and its subagent on opus prefilling at once ran Metal
        out of memory (2026-10-01: most requests for 15 min, the Mac swapping 100-400 MB/s). Not only when
        mem_tight: idle headroom is ~16 GB and one 100k sonnet prefill alone peaks ~12 GB over its
        resting size, so the first collision after a quiet spell OOMs before the guard trips. Both
        share one GPU, so taking turns costs little throughput. Returns the tiers still busy when it
        gave up after MEM_WAIT_MAX_S (then it goes anyway)."""
        ticket = cid or f"anon-{id(asyncio.current_task())}"
        self.mem_queue[ticket] = tier
        deadline = time.time() + max_s
        try:
            while True:
                self._state = None
                st = own_view(await self.state(), cid, set(self.mem_queue))
                now = time.time()
                ahead = []
                for k, t in self.mem_queue.items():
                    if k == ticket:
                        break
                    ahead.append(t)
                for t, i in st["tiers"].items():
                    if i.get("state") != "ready":  # loading for a request it let through: busy until MEM_SEND_S after it's up
                        self.mem_sent[t] = now
                busy = {t for t, i in st["tiers"].items() if (i.get("active") or 0) > 0}
                busy |= {t for t, ts in self.mem_sent.items() if now - ts < MEM_SEND_S}
                busy = sorted((busy | set(ahead)) - {tier, tiers().helper})
                if not busy or now > deadline:
                    self.mem_sent[tier] = now
                    return busy
                await asyncio.sleep(1.0)
        finally:
            self.mem_queue.pop(ticket, None)

    async def admit_direct(self, data: dict[str, Any], call_type: str, mode: str, tier: str,
                           cid: str | None = None) -> dict[str, Any]:
        """A routed = no model (the image judge): no cloud twin, no pin, and loading it unloads a tier. In
        enforce mode it waits like a local-only reload: for the tier it evicts to be idle and unused by a main
        conversation for WARM_MAIN_S (else it kills a live session), then for the memory gate. Each wait
        gives up after its limit and the request goes anyway. `cid`: a /judge/v1/decide pass-through's id
        (its data has none)."""
        cid, now = cid or data.get("litellm_call_id"), time.time()
        waited, mem_wait = [], None
        if mode == "enforce":
            sets, costs = matrix()
            waited = await self.wait_for_evictees(tier, sets, costs, cid, EVICT_WAIT_MAX_S,
                                                  self._pins().warm_main_tiers(mode, now))
            if MEM_WAIT_MAX_S > 0:
                t0 = time.time()
                still = await self.wait_for_other_tiers(tier, cid, MEM_WAIT_MAX_S)
                mem_wait = {"s": round(time.time() - t0, 1), "gave_up_on": still or None}
        requested, target = data.get("model"), f"ultron/{tier}"
        if call_type != "pass_through_endpoint":
            data["model"] = target  # a bare name or alias -> the canonical id
        decision = {"ts": now, "mode": mode, "applied": True, "requested": requested, "tier": tier, "target": target,
                    "endpoint": target, **client_info(data), "rule": "direct", "new": False, "call_type": call_type,
                    "busy_evictees_after_wait": waited or None, "mem_wait": mem_wait, "call_id": cid}
        _log(decision)
        if cid:
            self.recent[cid] = (f"{target}; rule=direct" + (f"; evicted-busy={','.join(waited)}" if waited else "")
                                + (f"; mem-wait={mem_wait['s']}s" if mem_wait and mem_wait["s"] >= 1 else ""))
            while len(self.recent) > 512:
                self.recent.popitem(last=False)
        return decision

    async def admit(self, data: dict[str, Any], call_type: str, mode: str) -> dict[str, Any] | None:
        t_in = time.time()
        model = data.get("model")
        tier = tier_for(str(model or ""))
        decide_cid = decide_call_id(data, call_type)
        if decide_cid is not None:  # POST /judge/v1/decide: admitted as for ultron/judge
            return await self.admit_direct(data, call_type, mode, JUDGE, decide_cid)
        if not isinstance(data.get("messages"), list) or data.get("mock_response"):
            return None  # a mock reply (loop_breaker, ultron_media) needs no backend, so no eviction wait
        if str(model or "").startswith("media/"):
            return None  # cloud image/audio/embedding entries: not a tier, and tier_for() would give the default tier
        if tier and not tiers().tier[tier]["routed"]:
            return await self.admit_direct(data, call_type, mode, tier)
        if tier is None:  # the client asked for cloud/<tier> itself: no decision, but log the traffic
            ctier = str(model).split("/", 1)[1] if "/" in str(model) else "?"
            ident, cid = pin_identity(data, ctier), data.get("litellm_call_id")
            _log({"ts": time.time(), "mode": mode, "applied": True, "requested": model, "tier": ctier, "target": model,
                  "endpoint": model, "rule": "explicit", "new": False, "key": ident["key"], "call_id": cid,
                  "harness_call_id": _headers(data).get("x-harness-call-id"), "effort": data.get("reasoning_effort"),
                  "main": ident["main"], "call_type": call_type, **client_info(data)})
            if cid:  # the conversation key, as for admitted requests
                self.keys[cid] = ident["key"]
                while len(self.keys) > 512:
                    self.keys.popitem(last=False)
            return None
        h = _headers(data)
        x_route = h.get("x-route", "").strip().lower()
        rmode = route_mode()
        ident = pin_identity(data, tier)
        req_tier = tier
        swapping = lock_swapping(tier)
        tier = locked_tier(tier)
        now = time.time()
        pins = self._pins()
        pin_mode = "enforce" if mode == "enforce" else "shadow"
        async with self.lock:
            pins.expire(now)
            pin = pins.get(pin_mode, ident["key"])
            if pin is not None and tier != req_tier and pin["target"].startswith("ultron/"):
                pin = dict(pin, target=f"ultron/{tier}")  # lock moved the conversation onto the locked tier
            if pin is not None and swapping and pin["target"].startswith("ultron/"):
                pin = None  # re-decide: the swap sends it to cloud until the locked tier is up
            st = own_view(await self.state(), data.get("litellm_call_id"))
            sets, costs = matrix()
            back = None
            if pin is None or pin["rule"].startswith(SOFT_PIN_RULES):
                self.pending = {t: ts for t, ts in self.pending.items() if now - ts < PENDING_LOAD_S}
                reserved = pins.main_local_tiers(pin_mode) | set(self.pending)
                back = decide(tier, st, reserved, x_route, rmode, sets)
                if swapping and back[0].startswith("ultron/") and x_route != "private":
                    back = (f"cloud/{req_tier}", "route-mode:tier-lock-swap")
                elif back[0].startswith("cloud/"):
                    back = (f"cloud/{req_tier}", back[1])  # OmniRoute twin of what the client asked for
                if pin is not None and not back[0].startswith("ultron/"):
                    back = None  # still full: the overflow pin stands
            if back is None:
                target, rule, new = pin["target"], "pinned:" + pin["rule"], False
                pins.touch(pin_mode, ident["key"], now)
            else:
                target, rule = back
                new = pin is None
                if target.startswith("ultron/") and target.split("/")[1] not in st["tiers"]:
                    self.pending[target.split("/")[1]] = now
                pins.put(pin_mode, ident["key"], target, tier, ident["main"], ident["session"], rule, now)
                if not new:
                    rule = "repin:" + rule
        explicit = rule.startswith(("x-route:", "route-mode:")) or (not new and pin["rule"].startswith(("x-route:", "route-mode:")))
        apply = mode == "enforce" or explicit
        # Vision applies in shadow mode too: a no-vision tier would silently ignore the image.
        effective = target if apply else f"ultron/{tier}"
        vtarget, vision = vision_fix(data, effective)
        if vtarget != effective:
            target, apply = vtarget, True
        waited: list[str] = []
        overflow = shed_why = backlog_why = None
        cloud_ok = x_route != "private" and rmode != "local-only"
        if target.startswith("ultron/") and rmode == "cloud-only" and x_route != "private":
            # The switch moves live conversations too, this request only (the pin stays): decide() covers new ones,
            # and a local pin used to reload the tier the operator had just unloaded (2026-10-08).
            target, apply, overflow = f"cloud/{target.split('/')[1]}", True, "route-mode"
        elif apply and target.startswith("ultron/") and st.get("mem_tight") and cloud_ok:
            target, overflow = f"cloud/{target.split('/')[1]}", "mem"  # this request only; the pin stays local
        elif apply and target.startswith("ultron/") and self.refused.get(ident["key"], 0) > now and cloud_ok:
            target, overflow = f"cloud/{target.split('/')[1]}", "refused"  # mtplx just refused it: same
        elif apply and target.startswith("ultron/") and cloud_ok and self.shed_hold(target.split("/")[1], now):
            shed_why = self.shed_hold(target.split("/")[1], now)
            target, overflow = f"cloud/{target.split('/')[1]}", "shed"  # that tier just threw work away: same
        elif apply and target.startswith("ultron/") and cloud_ok and self.backlog_hold(target.split("/")[1]):
            backlog_why = self.backlog_hold(target.split("/")[1])
            target, overflow = f"cloud/{target.split('/')[1]}", "backlog"  # its disk saves are behind: same
        if apply and target.startswith("ultron/"):
            ttier = target.split("/")[1]
            cloud_ok = cloud_ok and cloud_for(ttier)
            if ttier not in st["tiers"]:
                waited = await self.wait_for_evictees(ttier, sets, costs, data.get("litellm_call_id"),
                                                      OVERFLOW_WAIT_S if cloud_ok else EVICT_WAIT_MAX_S,
                                                      pins.warm_main_tiers(pin_mode, now))
                if waited and cloud_ok:  # still busy: only this request goes to the cloud; the pin stays local
                    target, overflow = f"cloud/{ttier}", "evict-busy"
        mem_wait = None
        if (apply and target.startswith("ultron/") and not cloud_ok and MEM_WAIT_MAX_S > 0
                and target.split("/")[1] != tiers().helper):  # the small helper tier never waits or holds anyone
            t0 = time.time()
            still = await self.wait_for_other_tiers(target.split("/")[1], data.get("litellm_call_id"), MEM_WAIT_MAX_S)
            mem_wait = {"s": round(time.time() - t0, 1), "gave_up_on": still or None}
        if apply:
            data["model"] = target
            if target.startswith("cloud/"):
                data["extra_headers"] = {**(data.get("extra_headers") or {}), "X-Session-Id": ident["session"]}
        mem = st.get("mem") or {}
        headroom = mem.get("headroom")
        gb = lambda b: None if b is None else round(b / 2**30, 1)
        decision = {
            "ts": now, "mode": mode, "applied": apply, "requested": model, "tier": tier, "target": target,
            # where the request really goes: the target when applied, else the configured local route
            "endpoint": target if apply else f"ultron/{tier}",
            **client_info(data),
            "rule": rule, "vision": vision, "new": new, "key": ident["key"], "main": ident["main"], "x_route": x_route or None,
            "route_mode": rmode, "call_type": call_type, "pressure": st.get("pressure"),
            "mem_tight": st.get("mem_tight"),
            "headroom_gb": gb(headroom), "swap_gb": gb(mem.get("swap_used")), "compressor_gb": gb(mem.get("compressor")),
            "loaded": {t: {**{k: i.get(k) for k in ("state", "active", "waiting")},
                           **({"fp_gb": gb(i["fp"]), "host_gb": gb(i.get("host"))} if i.get("fp") else {})}
                       for t, i in st["tiers"].items()},
            "busy_evictees_after_wait": waited or None, "overflow": overflow, "shed": shed_why, "backlog": backlog_why, "mem_wait": mem_wait,
            "call_id": data.get("litellm_call_id"),  # joins ultron_stats rows to the agent/host
            # joins a client's own per-call log (some harnesses send one per model call)
            "harness_call_id": h.get("x-harness-call-id"), "effort": data.get("reasoning_effort"),
            "admit_s": round(time.time() - t_in, 2),  # hook time incl. the lock, eviction wait and memory gate
        }
        _log(decision)
        cid = data.get("litellm_call_id")
        if cid:
            self.recent[cid] = (f"{target}; rule={rule}; {'applied' if apply else 'shadow'}"
                                + (f"; {vision}" if vision else "") + (f"; overflow={overflow}" if overflow else "")
                                + (f"; mem={st['mem_tight']}" if st.get("mem_tight") else "")
                                + (f"; shed={shed_why}" if shed_why else "")
                                + (f"; backlog={backlog_why}" if backlog_why else "")
                                + (f"; mem-wait={mem_wait['s']}s" if mem_wait and mem_wait["s"] >= 1 else ""))
            self.keys[cid] = ident["key"]
            while len(self.recent) > 512:
                self.recent.popitem(last=False)
            while len(self.keys) > 512:
                self.keys.popitem(last=False)
        return decision


def _log(entry: dict[str, Any]) -> None:
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass


try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # tests without litellm installed
    CustomLogger = object  # type: ignore[misc,assignment]


class UltronAdmit(CustomLogger):  # type: ignore[misc,valid-type]
    def __init__(self) -> None:
        super().__init__()
        self.admission = Admission()

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if str(call_type) == "call_mcp_tool":  # MCP gateway (/mcp): a synthetic "Tool: ..." request with model
            return data                         # mcp-tool-call, which tier_for() would admit as sonnet
        if str(call_type) == "pass_through_endpoint":
            try:
                default_decisions_model(data, str(call_type))
            except Exception as exc:
                _log({"ts": time.time(), "error": "decisions: " + repr(exc)})
        try:
            fill_array_items(data.get("tools"))
        except Exception as exc:  # never fail a request because of this
            _log({"ts": time.time(), "error": "items: " + repr(exc)})
        try:
            cid = data.get("litellm_call_id") or ""
            if str(data.get("reasoning_effort") or "").strip().lower() == "max":
                data["reasoning_effort"] = "xhigh"  # our alias for the top level; cloud models only know xhigh
            mock = data.get("mock_response")  # set by loop_breaker / ultron_media, which run before this hook
            if cid and isinstance(mock, str):
                _remember(self.admission.mocked, cid, "loop-breaker" if LOOP_BREAKER_PREFIX in mock else "media")
        except Exception as exc:
            _log({"ts": time.time(), "error": "injected: " + repr(exc)})
        try:  # any admit mode: router_settings.fallbacks would send a failed local request to cloud
            if str(call_type) == "pass_through_endpoint":
                pass  # no router, and this dict is the body sent upstream
            elif _headers(data).get("x-route", "").strip().lower() == "private" or route_mode() == "local-only":
                data["disable_fallbacks"] = True
        except Exception as exc:
            _log({"ts": time.time(), "error": "fallbacks: " + repr(exc)})
        mode = admit_mode()
        if mode != "off":
            try:
                await self.admission.admit(data, call_type, mode)
            except Exception as exc:  # never fail a request because of routing bookkeeping
                _log({"ts": time.time(), "error": repr(exc), "model": data.get("model")})
        return data

    async def async_post_call_failure_hook(self, request_data, original_exception, user_api_key_dict,
                                           traceback_str=None):
        try:
            entry = self.admission.note_refusal(request_data, original_exception)
            if entry:
                _log(entry)
        except Exception as exc:  # never change the client's error because of bookkeeping
            _log({"ts": time.time(), "error": "refused: " + repr(exc)})
        return None

    async def async_post_call_response_headers_hook(self, data, user_api_key_dict, response,
                                                    request_headers=None, litellm_call_info=None):
        cid = data.get("litellm_call_id") or ""
        out = {}
        if self.admission.recent.get(cid):
            out["x-ultron-route"] = self.admission.recent[cid]
        try:
            stages = injected_stages(response, self.admission.mocked.get(cid), bool(data.get("ultron_rescued")))
            if stages:
                out["x-ultron-injected"] = ",".join(stages)
        except Exception as exc:
            _log({"ts": time.time(), "error": "injected: " + repr(exc)})
        return out or None

    async def async_pre_call_deployment_hook(self, kwargs, call_type):
        """Runs on the translated chat messages just before the upstream call (both the
        /v1/messages bridge and plain chat completions)."""
        try:
            t = _local_tier(kwargs)
            tier = t["name"] if t else ""  # the blocks below are upstream's and name the tier; wandavision's t is the conf dict
            if isinstance(kwargs.get("messages"), list):
                kwargs["messages"] = drop_empty_tool_calls(repair_split_tool_calls(kwargs["messages"]),
                                                           kwargs.get("tools"))
            if t and isinstance(kwargs.get("messages"), list) and has_docs(kwargs["messages"]):
                kwargs["messages"] = await asyncio.to_thread(docs_to_text, kwargs["messages"])  # osascript: off the loop
            if t and isinstance(kwargs.get("messages"), list):
                kwargs["messages"] = normalize_history(kwargs["messages"], think_in_content=t["think_in_content"],
                                                       reasoning_content=t["reasoning_content"])
        except Exception as exc:  # never fail a request because of this
            _log({"ts": time.time(), "error": "history: " + repr(exc)})
            return kwargs
        try:
            group = _model_group(kwargs)
            if not tier and group.startswith("cloud/"):
                entry = self.admission.note_fallback(kwargs.get("litellm_call_id") or "", group, _headers(kwargs))
                if entry:
                    _log(entry)
        except Exception as exc:
            _log({"ts": time.time(), "error": "fallback: " + repr(exc)})
        why = None
        try:
            cid = kwargs.get("litellm_call_id") or ""
            why = t and one_shot(kwargs.get("messages"), kwargs.get("tools"), self.admission.recent.get(cid, ""))
            if why:
                kwargs["extra_headers"] = {**(kwargs.get("extra_headers") or {}), **CACHE_BYPASS}
                if cid in self.admission.recent and "bank=off" not in self.admission.recent[cid]:  # hook can run twice per /v1/messages
                    self.admission.recent[cid] += f"; bank=off({why})"
        except Exception as exc:
            _log({"ts": time.time(), "error": "bypass: " + repr(exc)})
        try:
            if t and t["effort"]:  # an mtplx tier: tell its request log which session this request is
                cid = kwargs.get("litellm_call_id") or ""
                ids = {"x-mtplx-request-id": cid} if cid else {}  # mtplx's request log records it as the request id
                if not why:
                    key = self.admission.keys.get(cid) or _fallback_key(kwargs["messages"], t["name"])
                    session = re.sub(r"[^A-Za-z0-9._-]+", "-", key)[:120]  # mtplx names its files after this
                    took = await asyncio.to_thread(self.admission.take_session, session, cid)
                    if took:
                        ids["x-mtplx-session-id"] = session
                    if cid in self.admission.recent and "; sess=" not in self.admission.recent[cid]:
                        self.admission.recent[cid] += f"; sess={session if took else 'anon(busy)'}"
                if ids:
                    kwargs["extra_headers"] = {**(kwargs.get("extra_headers") or {}), **ids}
        except Exception as exc:
            _log({"ts": time.time(), "error": "session: " + repr(exc)})
        try:
            if t and kwargs.get("tools") and trace_on():
                key = self.admission.keys.get(kwargs.get("litellm_call_id") or "") or _fallback_key(kwargs["messages"], t["name"])
                write_trace(key, t["name"], kwargs["messages"], kwargs["tools"])
        except Exception as exc:
            _log({"ts": time.time(), "error": "trace: " + repr(exc)})
        try:
            for field, value in TIER_DEFAULTS.get(tier or "", {}).items():
                if field == "max_tokens":
                    if kwargs.get("max_tokens") is None and kwargs.get("max_completion_tokens") is None:
                        kwargs["max_tokens"] = value
                elif kwargs.get(field) is None:
                    kwargs[field] = value
        except Exception as exc:
            _log({"ts": time.time(), "error": "defaults: " + repr(exc)})
        try:
            effort = str(kwargs.get("reasoning_effort") or "").strip().lower()
            cid = kwargs.get("litellm_call_id") or ""
            cap = EFFORT_CAP.get(tier or "")
            if cap and effort in EFFORT_BUDGETS and EFFORT_BUDGETS[effort] > EFFORT_BUDGETS[cap]:
                effort = kwargs["reasoning_effort"] = cap  # a higher ask is lowered to the tier's ceiling
            if tier in BUDGET_TIERS and effort in EFFORT_BUDGETS:
                kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), "thinking_budget": EFFORT_BUDGETS[effort]}
            if (tier in ECHO_EFFORT_TIERS or tier in BUDGET_TIERS) and effort in EFFORT_BUDGETS and cid in self.admission.recent \
                    and "; effort=" not in self.admission.recent[cid]:  # hook can run twice per /v1/messages
                self.admission.recent[cid] += f"; effort={effort}/{EFFORT_BUDGETS[effort]}"
        except Exception as exc:
            _log({"ts": time.time(), "error": "effort: " + repr(exc)})
        return kwargs


# ----------------------------------------------------------------------------- local tier history
# Local servers render the checkpoint's chat template, and Qwen templates are strict about history:
#  - One system message, first, and no "developer" role. Some servers (llama-server among them) answer
#    500 "System message must be at the beginning." / "Unexpected message role." otherwise.
#  - Qwen3.5 keeps <think> only for assistant turns after the last real user message.
# Claude Code sends a <system-reminder> system message after nearly every tool result (pi sends developer
# messages); by this hook LiteLLM's /v1/messages bridge has already made Claude Code's into user turns that
# hold nothing but reminders. Each one became the "last user message", so the template cut the model's
# reasoning from every earlier tool round of the task, whatever the tool. Appended to the tool result (or
# user message) they follow, the agent round stays one round and its thinking stays visible.
# LiteLLM's hosted_vllm provider also drops reasoning_content/thinking_blocks from assistant turns,
# so for tiers with `think_in_content = yes` in tiers.conf the thinking goes back as a leading <think>
# block in the content, which the Qwen3.5 template reads back out. A template that reads only
# reasoning_content (Qwen3.8) would render such a block twice, so a tier with
# `reasoning_content = yes` instead gets the block taken out of the content and kept on the
# assistant message (patch_hosted_vllm_reasoning). Both are per tier, in tiers.conf.


def _local_tier(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    if SWAP.split("//")[-1] not in str(kwargs.get("api_base") or ""):
        return None
    return tiers().tier.get(str(kwargs.get("model") or "").split("/")[-1])


# mtplx keeps every finished request's KV as a warm session in its session bank. A request that is never
# continued only costs memory there: the helper tier grew ~85 MB per one-shot vision request, past its 5G
# MTPLX_SESSION_BANK_MAX_BYTES cap, until Metal ran out (with this header 80 requests grew it 0.01 GB instead
# of 6 GB). The media hook's helper check sends the same header. Servers other than mtplx ignore it.
CACHE_BYPASS = {"x-mtplx-cache-mode": "bypass"}

# Named mtplx sessions (2026-10-07). Without a session id mtplx files a request under the anon-* session whose
# prompt it matches best (reuse from max(4096, 25% of the prompt) shared tokens), so agent conversations that
# share a long system prompt (agent lanes, ~12.8k tokens) land on one session. A session keeps one live
# cache lease, so each conversation evicted the other's KV and the next turn prefilled cold (opus: 12,288 of
# 49-59k tokens cached, 172-265 s). The deployment hook sends the admission conversation key as
# x-mtplx-session-id (mtplx 2.12.2 engine_session.resolve_session_id), except to one-shots (bank bypass) and
# while another request of the conversation is in flight (Admission.take_session: mtplx 409s that).


def one_shot(messages: Any, tools: Any, route_note: str = "") -> str | None:
    """Why a local request won't be continued on this tier (so mtplx shouldn't bank it), else None.
    single-turn: no tools and no assistant turn yet (Claude Code titles/summaries, suite probes, chat one-offs).
    vision: a newest-turn image rerouted off a no-vision tier; the conversation itself lives on that tier."""
    if "vision->" in route_note:
        return "vision"
    if not tools and not any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages or []):
        return "single-turn"
    return None


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("type") in ("text", "input_text"))
    return str(content or "")


def _append_text(content: Any, text: str) -> Any:
    if isinstance(content, list):
        return content + [{"type": "text", "text": text}]
    base = str(content or "")
    return f"{base}\n\n{text}" if base else text


def _reasoning(m: dict[str, Any]) -> str:
    r = m.get("reasoning_content")
    if isinstance(r, str) and r.strip():
        return r.strip()
    blocks = m.get("thinking_blocks") or []
    return "\n\n".join(str(b["thinking"]).strip() for b in blocks
                       if isinstance(b, dict) and b.get("type") == "thinking" and b.get("thinking")).strip()


def _think_in_content(m: dict[str, Any]) -> dict[str, Any]:
    r = _reasoning(m)
    content = m.get("content")
    if not r or "</think>" in _text(content):
        return m
    out = {k: v for k, v in m.items() if k not in ("reasoning_content", "thinking_blocks")}
    block = f"<think>\n{r}\n</think>\n\n"
    out["content"] = [{"type": "text", "text": block}] + content if isinstance(content, list) else block + str(content or "")
    return out


def _reasoning_content(m: dict[str, Any]) -> dict[str, Any]:
    """Assistant turn with its thinking as a single `reasoning_content` string (thinking_blocks dropped)."""
    r = _reasoning(m)
    if not r:
        return m
    out = {k: v for k, v in m.items() if k != "thinking_blocks"}
    out["reasoning_content"] = r
    return out


def _reminder_only(m: dict[str, Any]) -> bool:
    content = m.get("content")
    if isinstance(content, list) and any(not isinstance(b, dict) or b.get("type") not in ("text", "input_text") for b in content):
        return False
    t = _text(content).strip()
    return t.startswith("<system-reminder>") and t.endswith("</system-reminder>")


def normalize_history(messages: list[Any], think_in_content: bool = False, reasoning_content: bool = False) -> list[Any]:
    """Leading system/developer messages -> one system message. Later ones are appended to the tool
    result or user message they follow (otherwise they become a user message), and so are user messages
    holding only <system-reminder> blocks that follow a tool result. think_in_content moves assistant
    reasoning into a <think> block at the start of the content; reasoning_content keeps it as the
    `reasoning_content` field instead (fable's template)."""
    head, i = [], 0
    while i < len(messages) and isinstance(messages[i], dict) and messages[i].get("role") in ("system", "developer"):
        head.append(_text(messages[i].get("content"))); i += 1
    out: list[Any] = [{"role": "system", "content": "\n\n".join(t for t in head if t)}] if head else []
    for m in messages[i:]:
        role = m.get("role") if isinstance(m, dict) else None
        if role in ("system", "developer"):
            note = _text(m.get("content")).strip()
            if not note:
                continue
            if not note.startswith("<system-reminder>"):
                note = f"<system-reminder>\n{note}\n</system-reminder>"
            prev = out[-1] if out else None
            if isinstance(prev, dict) and prev.get("role") in ("tool", "user"):
                out[-1] = {**prev, "content": _append_text(prev.get("content"), note)}
            else:
                out.append({"role": "user", "content": note})
        elif role == "user" and out and isinstance(out[-1], dict) and out[-1].get("role") == "tool" and _reminder_only(m):
            out[-1] = {**out[-1], "content": _append_text(out[-1].get("content"), _text(m.get("content")).strip())}
        elif role == "assistant" and think_in_content:
            out.append(_think_in_content(m))
        elif role == "assistant" and reasoning_content:
            out.append(_reasoning_content(m))
        else:
            out.append(m)
    return out


# ----------------------------------------------------------------------------- documents
# Claude Code's Read tool returns a PDF as an Anthropic document block. LiteLLM's /v1/messages bridge turns
# it into an image_url part holding a data:application/pdf URL, and mtplx answers 400 "cannot decode image:
# cannot identify image file", on every later request too since the PDF stays in history. For local tiers, a
# non-image data URL (image_url part, or chat "file" part) becomes text: PDFs through macOS PDFKit via
# osascript (the system Python has no PDF library), text/* decoded, anything else a note.
DOC_MAX_CHARS = 200_000  # ~50k tokens; past it the model is told to read fewer pages at a time
DOC_TYPES = ("image_url", "file")
PDF_JXA = r"""
ObjC.import('PDFKit');
function run(argv) {
  var d = $.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath(argv[0]));
  if (d.isNil() || d.isLocked) return '';
  var out = [];
  for (var i = 0; i < d.pageCount; i++) {
    var s = d.pageAtIndex(i).string;
    out.push('--- page ' + (i + 1) + ' ---\n' + (s.isNil() ? '' : s.js));
  }
  return out.join('\n\n');
}
"""
_doc_cache: OrderedDict[str, str] = OrderedDict()  # sha1 of the data URL -> text (the same text each request keeps mtplx's prefix cache)


def _doc_url(part: Any) -> str | None:
    """The data URL of a non-image, non-video attachment part (what mtplx can't take), else None."""
    if not isinstance(part, dict) or part.get("type") not in DOC_TYPES:
        return None
    if part["type"] == "image_url":
        u = part.get("image_url")
        u = u.get("url") if isinstance(u, dict) else u
    else:
        f = part.get("file")
        u = f.get("file_data") if isinstance(f, dict) else None
    if not isinstance(u, str) or not u.startswith("data:") or u.startswith(("data:image/", "data:video/")):
        return None
    return u


def has_docs(messages: Any) -> bool:
    return any(isinstance(m, dict) and isinstance(m.get("content"), list) and any(_doc_url(b) for b in m["content"])
               for m in messages or [])


def pdf_text(data: bytes) -> str:
    """Per-page text of a PDF ('' when it has no text layer, is locked, or this isn't macOS)."""
    fd, path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        r = subprocess.run(["osascript", "-l", "JavaScript", "-e", PDF_JXA, path], capture_output=True, timeout=120)
        return r.stdout.decode("utf-8", "replace").strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    finally:
        os.unlink(path)


def doc_text(url: str) -> str:
    key = hashlib.sha1(url.encode()).hexdigest()
    if key in _doc_cache:
        _doc_cache.move_to_end(key)
        return _doc_cache[key]
    head, _, payload = url.partition(",")
    mime = head[5:].split(";")[0] or "application/octet-stream"
    try:
        data = base64.b64decode(payload) if ";base64" in head else urllib.parse.unquote_to_bytes(payload)
    except ValueError:
        data = b""
    if mime == "application/pdf":
        body = pdf_text(data)
        pages = body.count("--- page ")
        if not re.sub(r"--- page \d+ ---", "", body).strip():
            text = f"[{mime} attachment: no text layer (scanned?) and this model can't read PDFs directly]"
        else:
            text = f"[{mime} attachment, {pages} page(s), converted to text]\n{body}"
    elif mime.startswith("text/") or mime in ("application/json", "application/xml"):
        text = f"[{mime} attachment]\n" + data.decode("utf-8", "replace")
    else:
        text = f"[{mime} attachment omitted: local models read only text and images]"
    if len(text) > DOC_MAX_CHARS:
        text = text[:DOC_MAX_CHARS] + f"\n[truncated at {DOC_MAX_CHARS} characters: read fewer pages at a time]"
    _doc_cache[key] = text
    while len(_doc_cache) > 16:
        _doc_cache.popitem(last=False)
    return text


def docs_to_text(messages: list[Any]) -> list[Any]:
    """Attachment parts mtplx can't decode (see above) -> text parts. Messages without any are kept as is."""
    out = []
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list) and any(_doc_url(b) for b in c):
            m = {**m, "content": [{"type": "text", "text": doc_text(_doc_url(b))} if _doc_url(b) else b for b in c]}
        out.append(m)
    return out


# ----------------------------------------------------------------------------- split tool calls
# mtplx streams an empty delta ({}) between tool-call chunks while it parses the next piece of qwen3_coder
# XML. LiteLLM 1.102.1's /v1/messages bridge (AnthropicStreamWrapper) treats an empty delta as text and
# opens a new block, so the rest of the arguments arrive in a second tool_use block with no name. Claude
# Code stores both halves as {"__unparsedToolInput": {"raw": ...}}, answers the nameless one "No such tool
# available", and mtplx rejects every later request with 400 "assistant tool_call is missing a name"
# (a coding agent on opus and sonnet, 2026-10-01). patch_blank_delta_blocks stops new splits;
# repair_split_tool_calls fixes histories that already hold them.
def _raw_args(args: Any) -> str:
    if not isinstance(args, str):
        args = "" if args is None else json.dumps(args)
    try:
        v = json.loads(args)
    except ValueError:
        return args
    u = v.get("__unparsedToolInput") if isinstance(v, dict) else None
    return u["raw"] if isinstance(u, dict) and isinstance(u.get("raw"), str) else args


def _strip_tool_error(content: Any) -> str:
    return re.sub(r"^\s*<tool_use_error>.*?</tool_use_error>\s*", "", _text(content), flags=re.S).strip()


def repair_split_tool_calls(messages: list[Any]) -> list[Any]:
    """Drop nameless assistant tool calls and their tool results. A nameless call's arguments are the tail
    of the call before it: when the joined text parses as a JSON object, it becomes that call's arguments.
    Text after the dropped result's tool_use_error (system reminders) moves to the tool result before it."""
    dropped: set[str] = set()
    out: list[Any] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        calls = m.get("tool_calls")
        if m.get("role") == "assistant" and isinstance(calls, list) and any(
                isinstance(c, dict) and not (c.get("function") or {}).get("name") for c in calls):
            kept: list[Any] = []
            for c in calls:
                fn = c.get("function") if isinstance(c, dict) else None
                if not isinstance(fn, dict) or fn.get("name"):
                    kept.append(c)
                    continue
                dropped.add(str(c.get("id")))
                prev = kept[-1] if kept and isinstance(kept[-1], dict) else None
                pfn = prev.get("function") if prev else None
                if isinstance(pfn, dict):
                    try:
                        joined = json.loads(_raw_args(pfn.get("arguments")) + _raw_args(fn.get("arguments")))
                    except ValueError:
                        joined = None
                    if isinstance(joined, dict):
                        kept[-1] = {**prev, "function": {**pfn, "arguments": json.dumps(joined)}}
            m = {**m, "tool_calls": kept} if kept else {k: v for k, v in m.items() if k != "tool_calls"}
        elif m.get("role") == "tool" and str(m.get("tool_call_id")) in dropped:
            extra = _strip_tool_error(m.get("content"))
            if extra and out and isinstance(out[-1], dict) and out[-1].get("role") == "tool":
                out[-1] = {**out[-1], "content": _append_text(out[-1].get("content"), extra)}
            continue
        out.append(m)
    return out if dropped else messages


def _no_arguments(call: Any) -> bool:
    """True when a tool call carries nothing: '', '{}', 'null', '[]', whitespace, or a JSON object with no keys."""
    fn = call.get("function") if isinstance(call, dict) else None
    raw = fn.get("arguments") if isinstance(fn, dict) else None
    if raw is None:
        return True
    s = str(raw).strip()
    if s in ("", "{}", "null", "[]"):
        return True
    try:
        return json.loads(s) == {}
    except ValueError:
        return False

def _tools_needing_arguments(tools: Any) -> set[str]:
    """Names of declared tools whose schema has properties or required fields, so a call must carry arguments.

    A zero-argument tool (`murmur_check_messages`, `whoami`, `monolith_guide`) is called with `{}` by design,
    so dropping those calls would delete real work and its results."""
    names: set[str] = set()
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            fn = t if isinstance(t, dict) else {}
        name = fn.get("name")
        params = fn.get("parameters") or fn.get("input_schema") or {}
        if name and isinstance(params, dict) and (params.get("properties") or params.get("required")):
            names.add(name)
    return names

def drop_empty_tool_calls(messages: list[Any], tools: Any = None) -> list[Any]:
    """Drop assistant tool calls with no arguments and their tool results.

    The reference fable server emits `arguments: "{}"` on some calls (2026-10-13: 157 of 253 calls in one 515-message
    trace, 35% of fable traces vs 1.9% opus / 0.1% sonnet). The client answers `Validation failed for tool
    "bash": ... Received arguments: {}`, and a history holding ~25 of those pairs makes the model imitate
    its own empty calls near-deterministically: 3/3 empty at msgs[:300] of that trace, 8/8 correct at msgs[:12].
    So strip the pair before the next request; other calls in the same message and their results stay.

    2026-10-14, after sweeping every tier's traces: the shape is not fable's alone (opus 161, sonnet 125 empty
    calls whose declared schema needs arguments), but 19 opus / 39 sonnet / 2 haiku empties are legitimate
    zero-argument tools (`monolith_guide` 193, `murmur_check_messages` 20, `whoami` 2). A call is dropped only
    when its tool is declared with properties or required; with no `tools` passed, every empty call goes."""
    needs = _tools_needing_arguments(tools)
    dropped: set[str] = set()
    out: list[Any] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        calls = m.get("tool_calls")
        if m.get("role") == "assistant" and isinstance(calls, list):
            kept: list[Any] = []
            empty: list[Any] = []
            for c in calls:
                fn = c.get("function") if isinstance(c, dict) else None
                name = fn.get("name") if isinstance(fn, dict) else None
                stub = _no_arguments(c) and (tools is None or name in needs)
                (empty if stub else kept).append(c)
            if empty:
                dropped.update(str(c.get("id")) for c in empty)
                if kept:
                    m = {**m, "tool_calls": kept}
                elif m.get("content"):
                    m = {k: v for k, v in m.items() if k != "tool_calls"}
                else:
                    continue  # nothing left of the assistant turn
        elif m.get("role") == "tool" and str(m.get("tool_call_id")) in dropped:
            extra = _strip_tool_error(m.get("content"))
            if extra and out and isinstance(out[-1], dict) and out[-1].get("role") == "tool":
                out[-1] = {**out[-1], "content": _append_text(out[-1].get("content"), extra)}
            continue
        out.append(m)
    return out if dropped else messages

def patch_blank_delta_blocks() -> bool:
    """Make AnthropicStreamWrapper ignore empty deltas when deciding to open a block. Idempotent."""
    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
            AnthropicStreamWrapper,
        )
        orig = AnthropicStreamWrapper._should_start_new_content_block
    except Exception:  # tests without litellm, or a LiteLLM that moved it
        return False
    if getattr(orig, "_ultron", False):
        return True

    def _should_start_new_content_block(self, chunk):
        try:
            if self.sent_content_block_start and self._is_blank_delta(chunk):
                return False
        except Exception:
            pass
        return orig(self, chunk)

    _should_start_new_content_block._ultron = True  # type: ignore[attr-defined]
    AnthropicStreamWrapper._should_start_new_content_block = _should_start_new_content_block
    return True


BLANK_DELTA_PATCHED = patch_blank_delta_blocks()


def _reasoning_content_tiers() -> tuple[str, ...]:
    """Tiers that want `reasoning_content` kept on the assistant message (tiers.conf: reasoning_content)."""
    return tuple(name for name, c in tiers().tier.items() if c["reasoning_content"])


def patch_hosted_vllm_reasoning() -> bool:
    """Keep `reasoning_content` on assistant messages for the tiers that ask for it (tiers.conf).
    LiteLLM's hosted_vllm
    _transform_messages pops it (and thinking_blocks) from every assistant turn. Idempotent."""
    try:
        from litellm.llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig
        orig = HostedVLLMChatConfig._transform_messages
    except Exception:  # tests without litellm, or a LiteLLM that moved it
        return False
    if getattr(orig, "_ultron", False):
        return True

    def _transform_messages(self, messages, model, is_async=False):
        keep = []
        try:
            if str(model).split("/")[-1] in _reasoning_content_tiers():
                keep = [(i, m["reasoning_content"]) for i, m in enumerate(messages)
                        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("reasoning_content")]
        except Exception:
            keep = []
        result = orig(self, messages, model, is_async)
        if not keep:
            return result

        def restore(out):
            for i, r in keep:
                try:
                    out[i]["reasoning_content"] = r
                except Exception:
                    pass
            return out

        if asyncio.iscoroutine(result):
            async def waited():
                return restore(await result)
            return waited()
        return restore(result)

    _transform_messages._ultron = True  # type: ignore[attr-defined]
    HostedVLLMChatConfig._transform_messages = _transform_messages
    return True


HOSTED_VLLM_REASONING_PATCHED = patch_hosted_vllm_reasoning()


# ----------------------------------------------------------------------------- OmniRoute /v1/messages bridge
# LiteLLM 1.102.1 sends /v1/messages for an openai/ deployment to the upstream /v1/responses. OmniRoute's
# /v1/responses drops `name` from gemini function_call items, so the client gets a tool_use with the right
# input and name "" (a coding agent via cloud/sonnet, 2026-10-02). It also adds its memory_* tools there in
# chat format, which some upstreams reject (422 "tools[1]: missing field `name`"). /v1/chat/completions has
# neither bug, and it is what every OpenAI-compatible endpoint serves, so /v1/messages to the cloud endpoint
# (OMNIROUTE_BASE) takes LiteLLM's chat bridge. Other openai/ bases keep /v1/responses
# (litellm.use_chat_completions_url_for_anthropic_messages would move all). ULTRON_CHAT_BRIDGE=off: LiteLLM's default.
def chat_bridge(api_base: Any) -> bool:
    if os.environ.get("ULTRON_CHAT_BRIDGE", "on").strip().lower() in ("off", "0", "no", "false"):
        return False
    return cloud_endpoint(api_base)


def cloud_endpoint(api_base: Any) -> bool:
    """api_base is the cloud endpoint (OMNIROUTE_BASE's host)."""
    cloud = urllib.parse.urlsplit(os.environ.get("OMNIROUTE_BASE") or "").netloc
    return bool(cloud) and urllib.parse.urlsplit(str(api_base or "")).netloc == cloud


def patch_omniroute_chat_bridge() -> bool:
    """Send /v1/messages for the cloud endpoint's deployments through chat/completions. Idempotent."""
    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
            LiteLLMMessagesToCompletionTransformationHandler as ChatBridge,
        )
        from litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler import (
            LiteLLMMessagesToResponsesAPIHandler as ResponsesBridge,
        )
        orig = ResponsesBridge.anthropic_messages_handler
    except Exception:  # tests without litellm, or a LiteLLM that moved it
        return False
    if getattr(orig, "_ultron", False):
        return True

    def anthropic_messages_handler(*args, **kwargs):
        if chat_bridge(kwargs.get("api_base")):  # both bridges take the same arguments at the one call site
            return ChatBridge.anthropic_messages_handler(*args, **kwargs)
        return orig(*args, **kwargs)

    anthropic_messages_handler._ultron = True  # type: ignore[attr-defined]
    ResponsesBridge.anthropic_messages_handler = staticmethod(anthropic_messages_handler)
    return True


CHAT_BRIDGE_PATCHED = patch_omniroute_chat_bridge()


def _model_group(kwargs: dict[str, Any]) -> str:
    """The request's model group from metadata (empty when the client named the model directly)."""
    for k in ("metadata", "litellm_metadata"):
        md = kwargs.get(k)
        if isinstance(md, dict) and md.get("model_group"):
            return str(md["model_group"])
    return ""


# reasoning_effort -> thinking budget on the mtplx tiers (2026-10-07). mtplx drops reasoning_effort for the
# Qwen codec, so the tier scripts start it through mtplx/bin/mtplx_effort.py, which maps it per request
# (same table there; "none" = a 1-token budget, not thinking off, so the session-cache key stays put).
# Absent or unknown = the tier's own behaviour. The route header echoes "; effort=high/16384". "max" is
# rewritten to "xhigh" before routing so cloud models get a value they know. fable reads
# reasoning_effort itself (see BUDGET_TIERS).
EFFORT_BUDGETS = {"none": 1, "minimal": 512, "low": 2048, "medium": 8192, "high": 16384, "xhigh": 32768}

def _tiers_with(flag: str) -> tuple[str, ...]:
    """Tier names whose tiers.conf flag is on. Read once at import; no tiers.conf means none."""
    try:
        return tuple(n for n, c in tiers().tier.items() if c.get(flag))
    except Exception:
        return ()

EFFORT_TIERS = _tiers_with("effort")                 # the tier script maps reasoning_effort (mtplx_effort.py)
ECHO_EFFORT_TIERS = EFFORT_TIERS + _tiers_with("reasoning_effort")  # + tiers whose server reads reasoning_effort itself: echo only
# A tier whose server takes a top-level `thinking_budget` instead of reasoning_effort would be named
# here and get the table above as extra_body.thinking_budget. Empty for the reference tiers.
BUDGET_TIERS: tuple = ()
# Per-tier ceiling (tiers.conf: effort_cap): a request above it (xhigh/max) is lowered to the cap.
try:
    EFFORT_CAP = {n: c["effort_cap"] for n, c in tiers().tier.items() if c.get("effort_cap")}
except Exception:
    EFFORT_CAP = {}
# Sampling defaults applied only when the request sets none, so a client's own temperature/max_tokens
# win. The reference fable server sets none of its own (temperature 1.0, unlimited output).
TIER_DEFAULTS = {"fable": {"temperature": 0.6, "max_tokens": 32768, "repetition_penalty": 1.1}}

# ----------------------------------------------------------------------------- x-ultron-injected
# Response header naming every gateway stage that wrote or replaced reply text (2026-10-07):
# a client needs to tell a stack-written reply from the model's). Values, comma-joined:
#   loop-breaker  loop_breaker answered the turn itself (stop)
#   media         ultron_media answered it (image / video / edit / search / transcript)
#   rescue        ultron_rescue turned a trailing ```bash block into a tool call
# A client that wants none of them sends x-loop-breaker: off, x-media: off, x-rescue: off.
LOOP_BREAKER_PREFIX = "[ultron loop-breaker]"  # loop_breaker.NOTE_PREFIX


def _remember(od: OrderedDict, key: str, value: Any, limit: int = 1024) -> None:
    od[key] = value
    od.move_to_end(key)
    while len(od) > limit:
        od.popitem(last=False)


def reply_texts(response: Any) -> list[str]:
    """Assistant text of a non-streaming chat (ModelResponse) or /v1/messages (dict) response."""
    if response is None:
        return []
    if not isinstance(response, dict):
        dump = getattr(response, "model_dump", None)
        try:
            response = dump() if callable(dump) else dict(response)
        except Exception:
            return []
    out = []
    for ch in response.get("choices") or []:
        c = (ch.get("message") or {}).get("content") if isinstance(ch, dict) else None
        if isinstance(c, str):
            out.append(c)
    for b in response.get("content") or []:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(str(b.get("text") or ""))
    return out


def injected_stages(response: Any, mocked: str | None, rescued: bool) -> list[str]:
    stages = []
    if mocked:
        stages.append(mocked)
    if rescued:
        stages.append("rescue")
    return stages


# ----------------------------------------------------------------------------- tool schemas
# LiteLLM's pre-call token count (router enable_pre_call_checks, the max_input_tokens check) raises
# KeyError 'items' on any `"type": "array"` without `items` (litellm token_counter._format_type), and a
# failed count skips the check. Claude Code sends such a tool, so 99% of its /v1/messages requests went
# unchecked. A missing `items` already means "any item", so `{}` changes nothing.
def fill_array_items(node: Any) -> int:
    """Add `"items": {}` to every array schema without one, in place. Returns how many were filled."""
    n = 0
    if isinstance(node, dict):
        t = node.get("type")
        if (t == "array" or (isinstance(t, list) and "array" in t)) and "items" not in node:
            node["items"] = {}
            n += 1
        for v in node.values():
            n += fill_array_items(v)
    elif isinstance(node, list):
        for v in node:
            n += fill_array_items(v)
    return n


# ----------------------------------------------------------------------------- trace tap
# Training data for the LoRA scripts in lora/: the latest request of every tool-carrying conversation
# on a local tier, as the tier sees it (after normalize_history). One file per conversation, overwritten
# each request: every request carries the whole history, so the newest holds every earlier failure
# point. Mode file ~/.ultron/trace-mode: on | off (default off). Traces hold tool output (file contents,
# command output), so treat them like the conversations themselves: they stay on this Mac.
TRACE_MODE_FILE = os.path.expanduser(os.environ.get("ULTRON_TRACE_MODE_FILE", "~/.ultron/trace-mode"))
TRACE_DIR = os.path.expanduser(os.environ.get("ULTRON_TRACE_DIR", "~/.ultron/traces"))


def trace_on() -> bool:
    try:
        return open(TRACE_MODE_FILE).read().strip().lower() == "on"
    except OSError:
        return False


def _fallback_key(messages: list[Any], tier: str) -> str:
    head = json.dumps([_text(m.get("content")) for m in messages[:2] if isinstance(m, dict)], default=str)
    return f"h:{hashlib.sha1(head.encode()).hexdigest()[:16]}:{tier}"


def write_trace(key: str, tier: str, messages: list[Any], tools: Any) -> str:
    os.makedirs(TRACE_DIR, exist_ok=True)
    path = os.path.join(TRACE_DIR, re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[:120] + ".json")
    with open(path + ".tmp", "w") as f:
        json.dump({"ts": time.time(), "key": key, "tier": tier, "messages": messages, "tools": tools}, f, default=str)
    os.replace(path + ".tmp", path)
    return path


proxy_handler_instance = UltronAdmit()
