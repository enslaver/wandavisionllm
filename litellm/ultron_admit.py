"""ultron_admit: pick local tier vs OmniRoute overflow per conversation, then pin it.

LiteLLM pre-call hook, registered after loop_breaker. Local tiers first; when a request would
need a model load that doesn't fit next to what's loaded (the llama-swap matrix), a NEW
conversation goes to an OmniRoute combo (cloud/<tier>) instead of making llama-swap swap.
Once a conversation has a backend it never changes (pins in ~/.litellm/pins.sqlite).

Decision, first request of a conversation only:
  1. tier loaded and its queue < max_waiting                        -> ultron/<tier>
  2. not loaded, fits (matrix) next to loaded + main pins, pressure normal -> ultron/<tier> (cold)
  3. substitute tier (tiers.conf `substitute`) loaded, queue ok    -> ultron/<substitute>
  4. cloud allowed (x-route / route-mode, tier has `cloud`)        -> cloud/<tier>
  5. otherwise                                                      -> ultron/<tier> (llama-swap swaps;
     waits for the evicted tier's in-flight work first)
Any local target that needs a load (rule 5, a pinned conversation whose tier was swapped out, a
vision reroute) waits for the evicted tier's in-flight work: up to OVERFLOW_WAIT_S when cloud is
allowed, then that one request goes to cloud/<tier> (pin unchanged); else up to EVICT_WAIT_MAX_S,
then llama-swap swaps anyway. config.yaml falls cloud/<tier> back to ultron/<tier> on errors.
A tier a main-thread conversation used within WARM_MAIN_S counts as in-flight: evicting it would
kill a live session, so requests that would evict it overflow to cloud instead (2026-09-29).
Memory guard (2026-09-30): while ultron is nearly swapping (see mem_guard) and cloud is allowed, a
new conversation pins cloud/<tier> (rule 4:overflow:mem) and a request of a conversation pinned
local goes to cloud/<tier> (overflow=mem), that request only: the pin stays, so it comes back
once memory recovers. Held MEM_HOLD_S after the last trigger so it doesn't flap.
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
Tiers, their model ids, queue limits, substitutes and cloud overflow: tiers.conf (ultron_tiers.py).
Which agent sent a request (logged for Wanda): agents.conf.
Design notes: docs/architecture.md and litellm/README.md.
"""

from __future__ import annotations

import asyncio
import configparser
import fnmatch
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
import urllib.request
from collections import OrderedDict
from itertools import product
from typing import Any

import ultron_tiers

PIN_IDLE_S = {"main": 3600, "sub": 1800}
EVICT_WAIT_MAX_S = 300  # local only: wait at most this long for a tier to go idle, then swap (< client timeouts)
OVERFLOW_WAIT_S = 60  # cloud allowed: after this long, this one request goes to cloud/<tier> instead
PENDING_LOAD_S = 120  # in-process reservation for a cold load that isn't in /running yet
WARM_MAIN_S = 300  # a tier with a main conversation used within this window is live: never evict it (session-bank idle TTL)
MEM_LOW_GB = float(os.environ.get("ULTRON_MEM_LOW_GB", "4"))  # headroom under this = nearly swapping (0: check off)
SWAPOUT_MB_S = float(os.environ.get("ULTRON_SWAPOUT_MB_S", "16"))  # swap-out rate that counts as swapping (0: check off)
MEM_HOLD_S = 120  # stay on cloud this long after the last trigger (KV caches free slowly; no flapping)
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
               "vm.compressor.swapper.swapouts_total")


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
    return {"t": time.time(), "page": page, "pressure": v.get("kern.memorystatus_vm_pressure_level", 1),
            "headroom": sum(v.get(k, 0) for k in MEM_SYSCTLS[2:6]) * page if "vm.page_free_count" in v else None,
            "swapouts": v.get("vm.compressor.swapper.swapouts_total")}


def read_state() -> dict[str, Any]:
    """{'tiers': {tier: {'state', 'active', 'waiting'}}, 'pressure': 1|2|4, 'mem': read_mem()}.
    Blocking; run in a thread."""
    tiers: dict[str, dict[str, Any]] = {}
    names = tiers().names
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
            except Exception:  # not mtplx (e.g. TensorFold): count from ultron_stats' in-flight file
                info["cids"] = None
        tiers[tier] = info
    if any("cids" in i for i in tiers.values()):
        live = litellm_inflight()
        for tier, i in tiers.items():
            if "cids" in i:
                i["cids"] = live.get(tier, [])
    mem = read_mem()
    return {"tiers": tiers, "pressure": mem["pressure"], "mem": mem}


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
    loaded = {t for t, i in state["tiers"].items() if i.get("state") in ("ready", "starting")}
    reserved = {t for t in reserved if t in loaded}  # a pin only reserves memory while its tier is resident
    info = state["tiers"].get(tier)

    conf = tiers().tier

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
        self.lock = asyncio.Lock()
        self.pending: dict[str, float] = {}  # tier -> time a cold load was decided (not yet in /running)
        self.recent: OrderedDict[str, str] = OrderedDict()  # litellm_call_id -> response header value
        self.keys: OrderedDict[str, str] = OrderedDict()  # litellm_call_id -> conversation key (trace file)
        self._state: tuple[float, dict[str, Any]] | None = None
        self._mem_prev: dict[str, Any] | None = None  # last read_mem(), for the swap-out rate
        self._mem_tight: tuple[float, str] | None = None  # (held until, reason)
        self.mem_queue: OrderedDict[str, str] = OrderedDict()  # ticket -> tier, requests held at the memory gate (FIFO)
        self.mem_sent: dict[str, float] = {}  # tier -> when the memory gate last let one of its requests through

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
        self._state = (now, s)
        return s

    def mem_guard(self, m: dict[str, Any]) -> str | None:
        """Why ultron is (nearly) swapping, or None: pressure warn/critical, headroom under
        MEM_LOW_GB, or swap-outs of SWAPOUT_MB_S+ since the last sample (<= 30 s ago). Held for
        MEM_HOLD_S after the last trigger."""
        now = m.get("t") or time.time()
        why = None
        if int(m.get("pressure") or 1) >= 2:
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
        return self._mem_tight[1] if self._mem_tight and now < self._mem_tight[0] else None

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

    async def admit(self, data: dict[str, Any], call_type: str, mode: str) -> dict[str, Any] | None:
        model = data.get("model")
        tier = tier_for(str(model or ""))
        if not isinstance(data.get("messages"), list) or data.get("mock_response"):
            return None  # a mock reply (loop_breaker, ultron_media) needs no backend, so no eviction wait
        if str(model or "").startswith("media/"):
            return None  # cloud image/audio/embedding entries: not a tier, and tier_for() would give the default tier
        if tier is None:  # the client asked for cloud/<tier> itself: no decision, but log the traffic
            ctier = str(model).split("/", 1)[1] if "/" in str(model) else "?"
            _log({"ts": time.time(), "mode": mode, "applied": True, "requested": model, "tier": ctier, "target": model,
                  "endpoint": model, "rule": "explicit", "new": False, "key": pin_identity(data, ctier)["key"],
                  "call_id": data.get("litellm_call_id"),
                  "main": pin_identity(data, ctier)["main"], "call_type": call_type, **client_info(data)})
            return None
        h = _headers(data)
        x_route = h.get("x-route", "").strip().lower()
        rmode = route_mode()
        ident = pin_identity(data, tier)
        now = time.time()
        pins = self._pins()
        pin_mode = "enforce" if mode == "enforce" else "shadow"
        async with self.lock:
            pins.expire(now)
            pin = pins.get(pin_mode, ident["key"])
            st = own_view(await self.state(), data.get("litellm_call_id"))
            sets, costs = matrix()
            if pin:
                target, rule, new = pin["target"], "pinned:" + pin["rule"], False
                pins.touch(pin_mode, ident["key"], now)
            else:
                self.pending = {t: ts for t, ts in self.pending.items() if now - ts < PENDING_LOAD_S}
                reserved = pins.main_local_tiers(pin_mode) | set(self.pending)
                target, rule = decide(tier, st, reserved, x_route, rmode, sets)
                new = True
                if target.startswith("ultron/") and target.split("/")[1] not in st["tiers"]:
                    self.pending[target.split("/")[1]] = now
                pins.put(pin_mode, ident["key"], target, tier, ident["main"], ident["session"], rule, now)
        explicit = rule.startswith(("x-route:", "route-mode:")) or (not new and pin["rule"].startswith(("x-route:", "route-mode:")))
        apply = mode == "enforce" or explicit
        # Vision applies in shadow mode too: a no-vision tier would silently ignore the image.
        effective = target if apply else f"ultron/{tier}"
        vtarget, vision = vision_fix(data, effective)
        if vtarget != effective:
            target, apply = vtarget, True
        waited: list[str] = []
        overflow = None
        cloud_ok = x_route != "private" and rmode != "local-only"
        if apply and target.startswith("ultron/") and st.get("mem_tight") and cloud_ok and cloud_for(target.split("/")[1]):
            target, overflow = f"cloud/{target.split('/')[1]}", "mem"  # this request only; the pin stays local
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
        headroom = (st.get("mem") or {}).get("headroom")
        decision = {
            "ts": now, "mode": mode, "applied": apply, "requested": model, "tier": tier, "target": target,
            # where the request really goes: the target when applied, else the configured local route
            "endpoint": target if apply else f"ultron/{tier}",
            **client_info(data),
            "rule": rule, "vision": vision, "new": new, "key": ident["key"], "main": ident["main"], "x_route": x_route or None,
            "route_mode": rmode, "call_type": call_type, "pressure": st.get("pressure"),
            "mem_tight": st.get("mem_tight"),
            "headroom_gb": None if headroom is None else round(headroom / 2**30, 1),
            "loaded": {t: {k: i.get(k) for k in ("state", "active", "waiting")} for t, i in st["tiers"].items()},
            "busy_evictees_after_wait": waited or None, "overflow": overflow, "mem_wait": mem_wait,
            "call_id": data.get("litellm_call_id"),  # joins ultron_stats rows to the agent/host
        }
        _log(decision)
        cid = data.get("litellm_call_id")
        if cid:
            self.recent[cid] = (f"{target}; rule={rule}; {'applied' if apply else 'shadow'}"
                                + (f"; {vision}" if vision else "") + (f"; overflow={overflow}" if overflow else "")
                                + (f"; mem={st['mem_tight']}" if st.get("mem_tight") else "")
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
        try:
            fill_array_items(data.get("tools"))
        except Exception as exc:  # never fail a request because of this
            _log({"ts": time.time(), "error": "items: " + repr(exc)})
        mode = admit_mode()
        if mode != "off":
            try:
                await self.admission.admit(data, call_type, mode)
            except Exception as exc:  # never fail a request because of routing bookkeeping
                _log({"ts": time.time(), "error": repr(exc), "model": data.get("model")})
        return data

    async def async_post_call_response_headers_hook(self, data, user_api_key_dict, response,
                                                    request_headers=None, litellm_call_info=None):
        v = self.admission.recent.get(data.get("litellm_call_id") or "")
        return {"x-ultron-route": v} if v else None

    async def async_pre_call_deployment_hook(self, kwargs, call_type):
        """Runs on the translated chat messages just before the upstream call (both the
        /v1/messages bridge and plain chat completions)."""
        try:
            t = _local_tier(kwargs)
            if isinstance(kwargs.get("messages"), list):
                kwargs["messages"] = repair_split_tool_calls(kwargs["messages"])
            if t and isinstance(kwargs.get("messages"), list):
                kwargs["messages"] = normalize_history(kwargs["messages"], think_in_content=t["think_in_content"])
        except Exception as exc:  # never fail a request because of this
            _log({"ts": time.time(), "error": "history: " + repr(exc)})
            return kwargs
        try:
            if t and kwargs.get("tools") and trace_on():
                key = self.admission.keys.get(kwargs.get("litellm_call_id") or "") or _fallback_key(kwargs["messages"], t["name"])
                write_trace(key, t["name"], kwargs["messages"], kwargs["tools"])
        except Exception as exc:
            _log({"ts": time.time(), "error": "trace: " + repr(exc)})
        return kwargs


# ----------------------------------------------------------------------------- local tier history
# Local servers render the checkpoint's chat template, and Qwen templates are strict about history:
#  - One system message, first, and no "developer" role. Some servers (TensorFold, llama-server) answer
#    500 "System message must be at the beginning." / "Unexpected message role." otherwise.
#  - Qwen3.5 keeps <think> only for assistant turns after the last real user message.
# Claude Code sends a <system-reminder> system message after nearly every tool result (pi sends developer
# messages); by this hook LiteLLM's /v1/messages bridge has already made Claude Code's into user turns that
# hold nothing but reminders. Each one became the "last user message", so the template cut the model's
# reasoning from every earlier tool round of the task, whatever the tool. Appended to the tool result (or
# user message) they follow, the agent round stays one round and its thinking stays visible.
# LiteLLM's hosted_vllm provider also drops reasoning_content/thinking_blocks from assistant turns, so for
# tiers with `think_in_content = yes` in tiers.conf the thinking goes back as a leading <think> block in the
# content, which the Qwen3.5 template reads back out. Templates that read only reasoning_content (Qwen3.8)
# would render such a block twice, so it is per tier.


def _local_tier(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    if SWAP.split("//")[-1] not in str(kwargs.get("api_base") or ""):
        return None
    return tiers().tier.get(str(kwargs.get("model") or "").split("/")[-1])


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


def _reminder_only(m: dict[str, Any]) -> bool:
    content = m.get("content")
    if isinstance(content, list) and any(not isinstance(b, dict) or b.get("type") not in ("text", "input_text") for b in content):
        return False
    t = _text(content).strip()
    return t.startswith("<system-reminder>") and t.endswith("</system-reminder>")


def normalize_history(messages: list[Any], think_in_content: bool = False) -> list[Any]:
    """Leading system/developer messages -> one system message. Later ones are appended to the tool
    result or user message they follow (otherwise they become a user message), and so are user messages
    holding only <system-reminder> blocks that follow a tool result. think_in_content moves assistant
    reasoning into a <think> block at the start of the content."""
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
        else:
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
