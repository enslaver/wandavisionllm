"""ultron_tiers: read tiers.conf, the one list of local model tiers.

deploy.py builds the tier parts of llama-swap/config.yaml and litellm/config.yaml from it
(blocks()); the hooks read it on every request (load(), cached until the file changes). Wanda
reads the same file with a few lines of its own. Format and keys: tiers.conf.
Standard library only (deploy.py runs on macOS's /usr/bin/python3).
"""

from __future__ import annotations

import configparser
import fnmatch
import os
import re
from typing import Any

PATH = os.path.expanduser(os.environ.get(
    "ULTRON_TIERS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiers.conf")))
SWAP_API = "http://127.0.0.1:8001/v1"   # llama-swap (launchd/com.llama-swap.plist)
COLORS = ["#c39bd3", "#88c0d0", "#a3be8c", "#ebcb8b", "#d08770", "#b48ead", "#8fbcbb"]

TIER_KEYS = {"script", "ttl", "preload", "evict_cost", "context", "vision", "chat_only", "think_in_content",
             "timeout", "upstream_model", "max_waiting", "substitute", "match", "advertise", "cloud", "color", "routed"}
# routed = no: a model llama-swap serves next to the tiers (the image judge) that is only ever asked
# for by name (ultron/<name>). These keys would route requests to it, so they're refused there.
ROUTING_ONLY_KEYS = ("substitute", "match", "advertise", "cloud")
ROUTING_KEYS = {"default", "resident", "vision", "helper"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


def _list(v: str | None) -> list[str]:
    return [x for x in re.split(r"[\s,]+", v or "") if x]


def _bool(v: str | None, default: bool) -> bool:
    if v is None or not v.strip():
        return default
    return v.strip().lower() in ("1", "yes", "true", "on")


class Tiers:
    """Parsed tiers.conf. `names` keeps the file's order (biggest tier first)."""

    def __init__(self, text: str) -> None:
        # Comments are whole lines only: an inline "#" would eat colors like #88c0d0.
        cp = configparser.ConfigParser(interpolation=None)
        cp.optionxform = str  # keys are already lower case; keep them exact for the error messages
        cp.read_string(text)
        self.problems: list[str] = []
        self.names = [s for s in cp.sections() if s != "routing"]
        self.tier: dict[str, dict[str, Any]] = {}
        for i, name in enumerate(self.names):
            s = cp[name]
            if not NAME_RE.match(name):
                self.problems.append(f"[{name}]: tier names are lower-case letters, digits, - and _")
            for k in s:
                if k not in TIER_KEYS:
                    self.problems.append(f"[{name}] {k}: unknown key")
            try:
                self.tier[name] = {
                    "name": name,
                    "script": s.get("script") or f"tier-{name}.sh",
                    "ttl": int(s.get("ttl") or 600),
                    "preload": _bool(s.get("preload"), False),
                    "evict_cost": float(s.get("evict_cost") or 1),
                    "context": int(s.get("context") or 131072),
                    "vision": _bool(s.get("vision"), True),
                    "chat_only": _bool(s.get("chat_only"), False),
                    "think_in_content": _bool(s.get("think_in_content"), False),
                    "timeout": int(s.get("timeout") or 0),
                    "upstream_model": (s.get("upstream_model") or "").strip(),
                    "max_waiting": int(s.get("max_waiting") or 20),
                    "substitute": (s.get("substitute") or "").strip() or None,
                    "match": [g.lower() for g in _list(s.get("match"))],
                    "advertise": _list(s.get("advertise")),
                    "cloud": (s.get("cloud") or "").strip() or None,
                    "color": (s.get("color") or "").strip() or COLORS[i % len(COLORS)],
                    "routed": _bool(s.get("routed"), True),
                }
            except ValueError as e:
                self.problems.append(f"[{name}]: {e}")
        r = cp["routing"] if cp.has_section("routing") else {}
        for k in r:
            if k not in ROUTING_KEYS:
                self.problems.append(f"[routing] {k}: unknown key")
        routed = self.routed()
        self.default = (r.get("default") or "").strip() or (routed[0] if routed else None)
        self.resident = (r.get("resident") or "").strip() or " | ".join(self.names)
        self.vision_tier = ((r.get("vision") or "").strip()
                            or next((n for n in routed if self.tier[n]["vision"]), None))
        self.helper = (r.get("helper") or "").strip() or (routed[-1] if routed else None)
        self._check()

    def _check(self) -> None:
        if not self.names:
            self.problems.append("no tiers: add at least one [section] besides [routing]")
            return
        known = set(self.names)
        for what, v in (("default", self.default), ("vision", self.vision_tier), ("helper", self.helper)):
            if v and v not in known:
                self.problems.append(f"[routing] {what} = {v}: no such tier")
            elif v and not self.tier.get(v, {}).get("routed", True):
                self.problems.append(f"[routing] {what} = {v}: that tier has routed = no")
        for n, t in self.tier.items():
            if not t["routed"]:
                for k in ROUTING_ONLY_KEYS:
                    if t[k]:
                        self.problems.append(f"[{n}] {k}: not allowed with routed = no")
        for atom in re.findall(r"[A-Za-z0-9._/-]+", self.resident):
            if atom not in known:
                self.problems.append(f"[routing] resident: {atom} is not a tier")
        for n, t in self.tier.items():
            if t["substitute"] and t["substitute"] not in known:
                self.problems.append(f"[{n}] substitute = {t['substitute']}: no such tier")
            elif t["substitute"] and not self.tier.get(t["substitute"], {}).get("routed", True):
                self.problems.append(f"[{n}] substitute = {t['substitute']}: that tier has routed = no")
            if t["substitute"] == n:
                self.problems.append(f"[{n}] substitute: a tier can't substitute for itself")
        seen: dict[str, str] = {}
        for n, t in self.tier.items():
            for mid in [f"ultron/{n}", n] + t["advertise"] + t["match"]:
                if mid.lower() in seen and seen[mid.lower()] != n:
                    self.problems.append(f"[{n}] {mid}: also claimed by [{seen[mid.lower()]}]")
                seen[mid.lower()] = n

    # ------------------------------------------------------------------ lookups used by the hooks

    def tier_for(self, model: str) -> str | None:
        """Tier for a requested model id; None for an explicit cloud/<tier> (the client chose)."""
        m = (model or "").lower()
        if m.startswith("cloud/"):
            return None
        for n, t in self.tier.items():
            if m in (f"ultron/{n}", n) or m in (a.lower() for a in t["advertise"]):
                return n
        globs = [(g, n) for n, t in self.tier.items() for g in t["match"]]
        globs.sort(key=lambda gn: -len(gn[0].replace("*", "").replace("?", "")))  # most specific wins
        for g, n in globs:
            if fnmatch.fnmatchcase(m, g):
                return n
        return self.default

    def routed(self) -> list[str]:
        """Tiers requests can be routed to (every tier but those with routed = no), in file order."""
        return [n for n in self.names if self.tier.get(n, {}).get("routed", True)]

    def no_vision(self) -> list[str]:
        return [n for n, t in self.tier.items() if not t["vision"]]

    def has_cloud(self, tier: str) -> bool:
        return bool((self.tier.get(tier) or {}).get("cloud"))


_cache: dict[str, Any] = {"key": None, "tiers": None}


def load(path: str | None = None) -> Tiers:
    """tiers.conf, re-read when it changes. A broken edit keeps the last good version (hooks never
    fail a request); deploy.py refuses to push one."""
    path = path or PATH
    try:
        st = os.stat(path)
        key = (path, st.st_mtime, st.st_size)
    except OSError:
        key = (path, None, None)
    if key != _cache["key"]:
        try:
            with open(path) as f:
                t = Tiers(f.read())
            if not t.problems or _cache["tiers"] is None:
                _cache["tiers"] = t
        except (OSError, configparser.Error):
            if _cache["tiers"] is None:
                raise
        _cache["key"] = key
    return _cache["tiers"]


# ----------------------------------------------------------------------------- config generation

def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _local_entry(model_name: str, t: dict[str, Any]) -> list[str]:
    what = "tier" if t["routed"] else "model, asked for by name only"
    out = [f"- model_name: {_q(model_name)}",
           "  litellm_params:",
           f"    model: {_q('hosted_vllm/' + t['name'])}",
           f"    api_base: {_q(SWAP_API)}   # llama-swap; loads/unloads tiers",
           '    api_key: "none"                          # tiers are loopback-only, no auth']
    if t["chat_only"]:
        out.append("    use_chat_completions_api: true   # backend has no /v1/responses: LiteLLM converts")
    if t["timeout"]:
        out += [f"    timeout: {t['timeout']}          # whole request; /v1/messages reads this too",
                f"    stream_timeout: {t['timeout']}   # until the first streamed token (a cold long prefill sends nothing)"]
    out += ["  model_info:",
            f"    max_input_tokens: {t['context']}",
            f"    description: {_q('local ' + t['name'] + ' ' + what + ' (llama-swap)')}"]
    return out


def blocks(t: Tiers, home: str = "__HOME__") -> dict[str, str]:
    """Generated config blocks, keyed by the placeholder that marks where each goes."""
    names = t.names
    lm: list[str] = ["# --- local tiers, advertised in /v1/models ---"]
    for n in names:
        lm += _local_entry(f"ultron/{n}", t.tier[n])
    for n in names:
        for mid in t.tier[n]["advertise"]:
            lm += _local_entry(mid, t.tier[n])
    globs = sorted(((g, n) for n in names for g in t.tier[n]["match"]),
                   key=lambda gn: -len(gn[0].replace("*", "").replace("?", "")))  # most specific first
    if globs:
        lm.append("# --- wildcards (never listed): other ids of the same family ---")
        for g, n in globs:
            lm += _local_entry(g, t.tier[n])
    clouds = [n for n in names if t.tier[n]["cloud"]]
    if clouds:
        lm.append("# --- cloud overflow: any OpenAI-compatible endpoint (OMNIROUTE_BASE in ~/.litellm/env) ---")
        for n in clouds:
            c = t.tier[n]["cloud"]
            lm += [f"- model_name: {_q('cloud/' + n)}",
                   "  litellm_params:",
                   f"    model: {_q('openai/' + c)}   # openai/ = OpenAI-compatible; the rest is the endpoint's model id",
                   '    api_base: "os.environ/OMNIROUTE_BASE"',
                   '    api_key: "os.environ/OMNIROUTE_KEY"',
                   "    cache_control_injection_points: [{location: message, role: system}]   # see litellm_settings",
                   "  model_info:",
                   f"    max_input_tokens: {t.tier[n]['context']}",
                   f"    description: {_q('cloud overflow for the ' + n + ' tier (' + c + ')')}"]
    # No cloud -> local fallbacks: overflow is one-way (local -> cloud) and ultron_admit decides it. A
    # cloud/<tier> -> ultron/<tier> fallback re-ran every failed cloud call on the local tier, ignoring
    # route-mode cloud-only: with the endpoint failing (a LiteLLM started without OMNIROUTE_KEY), every
    # cloud request was served locally.
    fb = ["fallbacks: []   # cloud never falls back to local; see ultron_tiers.blocks()"]
    alias = ["model_group_alias:   # bare tier names keep working, not listed in /v1/models"]
    alias += [f"  {n}: {{model: \"ultron/{n}\", hidden: true}}" for n in names]
    slow = [n for n in names if t.tier[n]["timeout"]]
    retry = (["model_group_retry_policy:   # no retry after a timeout: the local server keeps working on the "
              "abandoned request, and a retry queues behind it"]
             + [f"  {_q(g)}: {{TimeoutErrorRetries: 0}}"
                for n in slow for g in [f"ultron/{n}", *t.tier[n]["advertise"], n]]) if slow else []

    sw: list[str] = ["models:"]
    for n in names:
        x = t.tier[n]
        sw += [f"  {n}:",
               f"    cmd: {home}/.mtplx/bin/{x['script']} ${{PORT}}",
               f"    env: [\"TIER={n}\"]",
               "    checkEndpoint: /health",
               f"    ttl: {x['ttl']}" + ("   # never unload" if x["ttl"] == 0 else "")]
        if x["upstream_model"]:
            um = x["upstream_model"]
            um = home + um[1:] if um.startswith("~/") else um  # a model path (mlx_vlm.server loads what the request names)
            sw.append(f"    useModelName: {_q(um)}   # the name the server answers to")
    pre = [n for n in names if t.tier[n]["preload"]]
    hooks = ["hooks:", "  on_startup:", f"    preload: [{', '.join(pre)}]"] if pre else []
    costs = ", ".join(f"{n}: {t.tier[n]['evict_cost']:g}" for n in names)
    routing = ["routing:", "  router:", "    use: matrix", "    settings:", "      matrix:",
               f"        evict_costs: {{{costs}}}   # higher = llama-swap keeps it loaded longer",
               "        sets:",
               f"          main: {_q(t.resident)}   # tiers.conf [routing] resident"]
    return {"__TIERS_MODEL_LIST__": "\n".join(lm), "__TIERS_FALLBACKS__": "\n".join(fb),
            "__TIERS_ALIASES__": "\n".join(alias), "__TIERS_RETRY_POLICY__": "\n".join(retry),
            "__TIERS_SWAP_MODELS__": "\n".join(sw),
            "__TIERS_SWAP_HOOKS__": "\n".join(hooks), "__TIERS_SWAP_ROUTING__": "\n".join(routing)}
