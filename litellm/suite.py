#!/usr/bin/env python3
"""suite.py — live integration suite for the ultron LLM stack.

Runs against the running stack on this Mac (LiteLLM :4000, llama-swap :8001, OmniRoute if configured):

    health          services up, a missing key gets 401, Wanda's memory reading is sane, the helper
                    tier is resident, and every OmniRoute model id the stack names exists there (when
                    OMNIROUTE_BASE is set). No tier loads.
    swap [--live]   tier admission / swapping. Default is a safe decision-table pass against the
                    live matrix + pins DB (no real requests). --live sends real requests and
                    changes which tiers are resident (restores nothing; tiers idle out on TTL).
    vision          image routing for every tier in tiers.conf: a no-vision tier reroutes to the
                    [routing] vision tier, and the tier that answers actually sees the image.
    classify        media-hook classifier positives/negatives (pure, no network).
    confirm         the helper tier's MEDIA/OTHER gate agent requests need (live helper tier, no cloud).
    image           text -> image via the media hook and via the media/image endpoint.
    video [--video-wait S]
                    text -> video via the media hook; polls the render job for the .mp4 / .error.
    baseline [--model a,b] [--tokens N] [--tool] [--ttft]
                    tokens/s per model (and tool-call latency with --tool, TTFT with --ttft).

Run everything (swap is dry):  uvx --with pyyaml python3 suite.py   (health swap vision classify confirm image video)
Run a section:                  uvx --with pyyaml python3 suite.py vision image baseline --model opus,sonnet,haiku --tool

Notes:
  - Needs pyyaml to read llama-swap's matrix (the deploy hook has it inside LiteLLM; a bare
    `/usr/bin/python3` does not), so run it via `uvx --with pyyaml python3 suite.py`.
  - Keys come from ~/.litellm/env (LITELLM_MASTER_KEY, OMNIROUTE_KEY).
  - image/video need ~/.ultron/media-mode=enforce to actually generate; the suite sets it
    temporarily and restores the previous value. It also briefly sets admit-mode=enforce for the
    live swap probes so the warm-tier protection engages. Real traffic in that window is decided
    as enforce; prefer running when the box is quiet.
  - Local baselines force the tier with `x-route: private`, so each baseline evicts the previous
    tier. Run baseline last if you care about which tier stays resident.
  - vision and baseline send `X-Claude-Code-Agent-Id: suite`, so their pins are subagent pins. A main
    pin makes its tier "warm" for 5 min, and in local-only mode the next fable<->opus swap then waits
    up to 300 s for it (baseline once stalled ~150 s twice behind the suite's own requests).
  - Pure stdlib (Python 3.9), matching the deployed-hook convention. Not deployed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zlib

HOME = os.path.expanduser("~")
LITELLM = os.environ.get("ULTRON_LITELLM", "http://127.0.0.1:4000")
LLAMASWAP = os.environ.get("ULTRON_SWAP_URL", "http://127.0.0.1:8001")
ENV = os.path.expanduser("~/.litellm/env")
ADMIT_LOG = os.path.expanduser("~/.litellm/ultron-admit.jsonl")
MEDIA_LOG = os.path.expanduser("~/.litellm/media.jsonl")
MEDIA_DIR = os.path.expanduser("~/.ultron/media")
ADMIT_MODE_FILE = os.path.expanduser("~/.ultron/admit-mode")
MEDIA_MODE_FILE = os.path.expanduser("~/.ultron/media-mode")
PINS_DB = os.path.expanduser("~/.litellm/pins.sqlite")
WANDA = os.environ.get("ULTRON_WANDA", "http://127.0.0.1:8790")
PORTAL = os.environ.get("ULTRON_PORTAL", "http://127.0.0.1/")  # Caddy -> Wanda
LITELLM_CONFIG = os.path.expanduser("~/.litellm/config.yaml")
SUITE_AGENT = {"X-Claude-Code-Agent-Id": "suite"}  # subagent pins: never "warm main", so swaps don't wait on them

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ultron_admit as ua          # pure stdlib; decision logic
import ultron_media as um          # media classifier

RESULTS: list[tuple[str, str, bool, str]] = []


# ----------------------------------------------------------------------------- results

def check(section: str, name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((section, name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


def section(title: str) -> None:
    print(f"\n== {title} ==")


def summarize() -> int:
    print("\n===== SUMMARY =====")
    fails = [r for r in RESULTS if not r[2]]
    by = {}
    for s, n, ok, d in RESULTS:
        by.setdefault(s, [0, 0])[0 if ok else 1] += 1
    for s, (ok, bad) in sorted(by.items()):
        print(f"  {s:10} {ok} passed, {bad} failed")
    if fails:
        for s, n, _, d in fails:
            print(f"    FAIL {s}: {n}")
        return 1
    print("  all passed")
    return 0


# ----------------------------------------------------------------------------- io helpers

def key(name: str) -> str:
    try:
        for line in open(ENV):
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return os.environ.get(name, "")


def request(method, url, body=None, headers=None, timeout=90):
    """(status, headers, json|str, elapsed_s). Errors come back as status 0 / str body."""
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                parsed = json.loads(raw) if raw else None
            except ValueError:
                parsed = raw.decode("utf-8", "replace")
            return r.status, dict(r.headers), parsed, time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()[:600].decode("utf-8", "replace"), time.time() - t0
    except Exception as e:
        return 0, {}, repr(e), time.time() - t0


def chat(model, messages, session=None, max_tokens=150, tools=None, x_route=None, timeout=120,
         extra_headers=None):
    h = {"Authorization": "Bearer " + key("LITELLM_MASTER_KEY"), "Content-Type": "application/json"}
    if session:
        h["X-Claude-Code-Session-Id"] = session
    if x_route:
        h["x-route"] = x_route
    if extra_headers:
        h.update(extra_headers)
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    return request("POST", LITELLM + "/v1/chat/completions", body, h, timeout=timeout)


def tier_states():
    st, _, data, _ = request("GET", LLAMASWAP + "/models", timeout=5)
    out = {}
    if st == 200 and isinstance(data, dict):
        for m in data.get("data", []):
            out[m.get("id")] = (m.get("status") or {}).get("value")
    return out


def content_of(resp):
    """Response content from a (status, headers, data, elapsed) tuple, or None."""
    _, _, data, _ = resp
    if isinstance(data, dict):
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        return msg.get("content")
    return None


def err_of(resp):
    _, _, data, _ = resp
    if isinstance(data, dict) and data.get("error"):
        return json.dumps(data["error"])[:200]
    if isinstance(data, str):
        return data[:200]
    return None


# ----------------------------------------------------------------------------- log tails

def file_lines(path):
    try:
        with open(path) as f:
            return f.read().splitlines()
    except OSError:
        return []


def log_len(path):
    return len(file_lines(path))


def new_decisions(offset, session):
    """Decisions appended to the admit log since offset, matching this session."""
    out = []
    for l in file_lines(ADMIT_LOG)[offset:]:
        try:
            d = json.loads(l)
        except ValueError:
            continue
        if d.get("key", "").startswith("cc:" + session + ":") or session in str(d.get("session", "")):
            out.append(d)
    return out


def new_media(offset, text=None):
    out = []
    for l in file_lines(MEDIA_LOG)[offset:]:
        try:
            d = json.loads(l)
        except ValueError:
            continue
        if text is None or text[:80] in str(d.get("text", "")) or text[:80] in str(d):
            out.append(d)
    return out


# ----------------------------------------------------------------------------- modes

def read_mode(path):
    try:
        v = open(path).read().strip()
        return v if v else None
    except OSError:
        return None


def write_mode(path, value):
    if value is None:
        try:
            os.remove(path)
        except OSError:
            pass
    else:
        with open(path, "w") as f:
            f.write(value + "\n")


class ModeCtx:
    """Set a live mode file for the block, restore the previous value (or absence) after."""

    def __init__(self, path, value):
        self.path, self.value, self.prev = path, value, read_mode(path)

    def __enter__(self):
        write_mode(self.path, self.value)
        return self

    def __exit__(self, *a):
        write_mode(self.path, self.prev)


def make_png(rgb, w=48, h=48):
    """Minimal solid-color PNG (stdlib only)."""
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit, truecolor
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def image_message(question="What color is this image? Answer with one word."):
    data_url = "data:image/png;base64," + base64.b64encode(make_png((255, 0, 0))).decode()
    return [{"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": data_url}}]


# ----------------------------------------------------------------------------- health

def _omniroute_names():
    """{label: [OmniRoute model ids]}: config.yaml entries pointed at OmniRoute, and the media hook's chains."""
    out = {}
    try:
        import yaml
        cfg = yaml.safe_load(open(LITELLM_CONFIG)) or {}
        names = []
        for m in cfg.get("model_list") or []:
            p = m.get("litellm_params") or {}
            if str(p.get("api_base") or "") == "os.environ/OMNIROUTE_BASE":
                names.append(str(p.get("model", "")).split("/", 1)[-1])  # openai/<omniroute id>
        out["config.yaml cloud/* + media/*"] = names
    except Exception as e:  # no pyyaml or no config: report it as a failed check
        out["config.yaml cloud/* + media/*"] = [f"<unreadable: {e}>"]
    for kind in ("image", "edit", "video", "transcribe"):
        out[f"ultron_media {kind} chain"] = list(um.conf()[kind])
    return out


def cmd_health(args):
    section("health: services, auth, dashboard, OmniRoute ids (no tier loads)")
    st, _, _, _ = request("GET", LITELLM + "/health/liveliness", timeout=5)
    check("health", "LiteLLM alive", st == 200, f"status {st}")
    # Without prisma in LiteLLM's env the auth error path crashed and a missing key was a 500 (deploy.py adds it)
    st, _, _, _ = request("GET", LITELLM + "/v1/models", timeout=10)
    check("health", "missing key gets 401", st == 401, f"status {st}")
    st, _, data, _ = request("GET", LITELLM + "/v1/models",
                             headers={"Authorization": "Bearer " + key("LITELLM_MASTER_KEY")}, timeout=10)
    ids = {m.get("id") for m in data.get("data", [])} if isinstance(data, dict) else set()
    want = {f"ultron/{t}" for t in ua.tiers().names}
    check("health", "master key lists every ultron/<tier>", st == 200 and want <= ids,
          f"status {st}, missing {sorted(want - ids)}")

    helper = ua.tiers().helper
    st, _, data, _ = request("GET", LLAMASWAP + "/running", timeout=5)
    running = {r.get("model"): r.get("state") for r in data.get("running", [])} if isinstance(data, dict) else {}
    check("health", f"helper tier {helper} resident", running.get(helper) == "ready", f"running {running}")

    st, _, data, _ = request("GET", WANDA + "/api/status", timeout=10)
    data = data if isinstance(data, dict) else {}
    check("health", "Wanda status: llama-swap and LiteLLM ok",
          st == 200 and (data.get("swap") or {}).get("ok") and (data.get("litellm") or {}).get("ok"), f"status {st}")
    m = data.get("machine") or {}
    avail, total = m.get("mem_available"), m.get("mem_total")
    # vm_stat's free + inactive overlap on macOS 27: the memory lamp once read 86 GB free of 64 GB
    check("health", "Wanda memory reading is within physical RAM", bool(avail and total and 0 < avail <= total),
          f"{(avail or 0) / 2**30:.1f} GB free of {(total or 0) / 2**30:.1f} GB")
    st, _, _, _ = request("GET", PORTAL, timeout=10)
    check("health", f"Caddy serves Wanda ({PORTAL})", st == 200, f"status {st}")

    # every OmniRoute id the stack names must exist there: a provider dropping its ids made every video fail for days
    base = key("OMNIROUTE_BASE")
    if not base:
        print("  (OMNIROUTE_BASE not set: OmniRoute checks skipped)")
        return 0
    st, _, data, _ = request("GET", base.rstrip("/") + "/models",
                             headers={"Authorization": "Bearer " + key("OMNIROUTE_KEY")}, timeout=20)
    have = {m.get("id") for m in data.get("data", [])} if st == 200 and isinstance(data, dict) else set()
    if not check("health", "OmniRoute lists its models", bool(have), f"status {st}, {len(have)} ids"):
        return 0
    for label, names in _omniroute_names().items():
        missing = [n for n in names if n not in have]
        check("health", f"{label} exist on OmniRoute", bool(names) and not missing,
              f"missing {missing}" if missing else f"{len(names)} ids")
    return 0


# ----------------------------------------------------------------------------- swap

DECISION_SCENARIOS = [
    # (name, tier, {tier: (state, active, waiting)}, reserved, x_route, mode, expect)
    ("haiku loaded, queue ok", "haiku", {"haiku": ("ready", 1, 2)}, set(), "", "auto", ("ultron/haiku", "1:loaded")),
    ("opus loaded", "opus", {"opus": ("ready", 1, 0), "haiku": ("ready", 0, 0)}, set(), "", "auto", ("ultron/opus", "1:loaded")),
    ("opus starting", "opus", {"opus": ("starting", 0, 0), "haiku": ("ready", 0, 0)}, set(), "", "auto", ("ultron/opus", "1:loading")),
    ("opus queue full", "opus", {"opus": ("ready", 1, 1), "haiku": ("ready", 0, 0)}, set(), "", "auto", ("cloud/opus", "4:overflow:queue")),
    ("sonnet cold-fits next to haiku", "sonnet", {"haiku": ("ready", 0, 0)}, set(), "", "auto", ("ultron/sonnet", "2:cold-fits")),
    ("sonnet cold-fits beside loaded opus (3-resident)", "sonnet", {"opus": ("ready", 0, 0), "haiku": ("ready", 0, 0)}, set(), "", "auto", ("ultron/sonnet", "2:cold-fits")),
    ("idle sonnet pin does not block opus", "opus", {"haiku": ("ready", 0, 0)}, {"sonnet"}, "", "auto", ("ultron/opus", "2:cold-fits")),
    ("idle opus pin does not block sonnet", "sonnet", {"haiku": ("ready", 0, 0)}, {"opus"}, "", "auto", ("ultron/sonnet", "2:cold-fits")),
    ("memory pressure overflows sonnet", "sonnet", {"haiku": ("ready", 0, 0)}, set(), "", "auto", ("cloud/sonnet", "4:overflow:no-fit")),
    ("haiku substitutes to loaded sonnet", "haiku", {"haiku": ("ready", 1, 3), "sonnet": ("ready", 0, 0)}, set(), "", "auto", ("ultron/sonnet", "3:substitute(haiku->sonnet)")),
    ("private opus queues local", "opus", {"opus": ("ready", 1, 1), "haiku": ("ready", 0, 0)}, set(), "private", "auto", ("ultron/opus", "5:local-swap")),
    ("memory pressure local-only sonnet swaps", "sonnet", {"haiku": ("ready", 0, 0)}, set(), "", "local-only", ("ultron/sonnet", "5:local-swap")),
    ("x-route cloud wins", "haiku", {"haiku": ("ready", 0, 0)}, set(), "cloud", "auto", ("cloud/haiku", "x-route:cloud")),
    ("route-mode cloud-only", "haiku", {"haiku": ("ready", 0, 0)}, set(), "", "cloud-only", ("cloud/haiku", "route-mode:cloud-only")),
    ("private beats cloud-only", "haiku", {"haiku": ("ready", 0, 0)}, set(), "private", "cloud-only", ("ultron/haiku", "1:loaded")),
]
# With the example fable tier (resident = (fable | opus) & sonnet & haiku); skipped when tiers.conf has no fable.
FABLE_SCENARIOS = [
    ("fable cold-fits beside sonnet + haiku", "fable", {"sonnet": ("ready", 0, 0), "haiku": ("ready", 0, 0)}, set(), "", "auto", ("ultron/fable", "2:cold-fits")),
    ("fable never shares the box with opus", "fable", {"opus": ("ready", 0, 0), "sonnet": ("ready", 0, 0), "haiku": ("ready", 0, 0)}, set(), "", "auto", ("cloud/fable", "4:overflow:no-fit")),
    ("private fable swaps opus out", "fable", {"opus": ("ready", 0, 0), "sonnet": ("ready", 0, 0), "haiku": ("ready", 0, 0)}, set(), "private", "auto", ("ultron/fable", "5:local-swap")),
]


def cmd_swap(args):
    section("swap: decision table against the live matrix + pins (dry)")
    st_code, _, live, _ = request("GET", LITELLM + "/health/liveliness", timeout=5)
    sets, costs = ua.matrix()
    pins = ua.Pins(PINS_DB)
    now = time.time()
    print(f"  live matrix sets: {sorted(frozenset(s) for s in sets)}  costs: {costs}")
    if st_code == 200:
        print(f"  live tiers: {tier_states()}")
    print(f"  main pins (enforce): local={sorted(pins.main_local_tiers('enforce'))} "
          f"warm={sorted(pins.warm_main_tiers('enforce', now))}")

    fable = "fable" in ua.tiers().names
    for name, tier, tstate, reserved, xr, mode, exp in DECISION_SCENARIOS + (FABLE_SCENARIOS if fable else []):
        st = {"tiers": {t: {"state": s, "active": a, "waiting": w} for t, (s, a, w) in tstate.items()},
              "pressure": 2 if name.startswith("memory pressure") else 1}
        got = ua.decide(tier, st, set(reserved), xr, mode, sets)
        check("swap", name, got == exp, f"got {got}, expected {exp}")

    # eviction solver sanity on the live matrix (all three fit: nothing should be evicted)
    check("swap", "opus evicts nothing (3-resident)",
          ua.evictees("opus", {"sonnet", "haiku"}, sets, costs) == set(),
          f"{ua.evictees('opus', {'sonnet', 'haiku'}, sets, costs)}")
    check("swap", "sonnet evicts nothing (3-resident)",
          ua.evictees("sonnet", {"opus", "haiku"}, sets, costs) == set(),
          f"{ua.evictees('sonnet', {'opus', 'haiku'}, sets, costs)}")
    if fable:
        check("swap", "fable evicts only opus",
              ua.evictees("fable", {"opus", "sonnet", "haiku"}, sets, costs) == {"opus"},
              f"{ua.evictees('fable', {'opus', 'sonnet', 'haiku'}, sets, costs)}")

    if args.live:
        cmd_swap_live(args)
    return 0


def _tier_loaded(name):
    return tier_states().get(name) == "loaded"


def cmd_swap_live(args):
    print("\n  (live probes; sets admit-mode=enforce, restores after)")
    with ModeCtx(ADMIT_MODE_FILE, "enforce"):
        # P1: sonnet cold-loads on demand (only when opus is not resident, else it overflows)
        if _tier_loaded("opus"):
            check("swap:live", "sonnet cold-load (skipped: opus resident)", True, "would overflow, not load")
        else:
            session = "suite-load-" + uuid.uuid4().hex[:8]
            off = log_len(ADMIT_LOG)
            st_code, _, data, dt = chat("claude-sonnet-5",
                                        [{"role": "user", "content": "reply with the word pong"}],
                                        session=session, max_tokens=8, timeout=150)
            dec = new_decisions(off, session)
            ok_endpoint = bool(dec) and dec[-1].get("endpoint") == "ultron/sonnet"
            ok_content = isinstance(content_of((st_code, {}, data, dt)), str) and "pong" in (content_of((st_code, {}, data, dt)) or "")
            check("swap:live", "sonnet cold-loads locally", ok_endpoint and ok_content,
                  f"endpoint={dec[-1].get('endpoint') if dec else '?'} rule={dec[-1].get('rule') if dec else '?'} "
                  f"content_ok={ok_content} in {dt:.1f}s")
            check("swap:live", "sonnet resident after load", _tier_loaded("sonnet"), str(tier_states()))

        # P2: the 2026-09-29 regression — with the 3-resident matrix, an opus query while sonnet is
        # loaded must load opus ALONGSIDE sonnet, never evicting the live sonnet.
        if not _tier_loaded("sonnet"):
            warm = "suite-warm-" + uuid.uuid4().hex[:8]
            chat("claude-sonnet-5", [{"role": "user", "content": "reply ok"}],
                 session=warm, max_tokens=8, timeout=150)
        if not _tier_loaded("sonnet"):
            check("swap:live", "opus beside warm sonnet (skipped: sonnet not resident)", True,
                  f"tiers={tier_states()}")
            return 0
        warm_session = "suite-warm2-" + uuid.uuid4().hex[:8]
        chat("claude-sonnet-5", [{"role": "user", "content": "reply warm"}],
             session=warm_session, max_tokens=8, timeout=120)  # main-thread pin, recent last_seen
        sonnet_before = _tier_loaded("sonnet")
        off = log_len(ADMIT_LOG)
        opus_session = "suite-opus-" + uuid.uuid4().hex[:8]
        st_code, _, data, dt = chat("claude-opus-5-5",
                                    [{"role": "user", "content": "reply with the word opus"}],
                                    session=opus_session, max_tokens=8, timeout=180)
        dec = new_decisions(off, opus_session)
        endpoint = dec[-1].get("endpoint") if dec else "?"
        rule = dec[-1].get("rule") if dec else "?"
        check("swap:live", "opus loads beside warm sonnet (no eviction)",
              endpoint == "ultron/opus" and rule == "2:cold-fits",
              f"endpoint={endpoint} rule={rule}")
        check("swap:live", "sonnet survives the opus query",
              sonnet_before and _tier_loaded("sonnet"),
              f"before={sonnet_before} after={_tier_loaded('sonnet')}")
        print("  note: sonnet stays resident beside opus")
    return 0


# ----------------------------------------------------------------------------- vision

def cmd_vision(args):
    section("vision: image routing + validation")
    t = ua.tiers()
    vt, blind = t.vision_tier, set(t.no_vision())
    if vt and not _tier_loaded(vt):
        chat(f"ultron/{vt}", [{"role": "user", "content": "warm"}], session="suite-v-" + uuid.uuid4().hex[:8],
             max_tokens=4, timeout=150, extra_headers=SUITE_AGENT)

    def reply_text(resp):
        """Content plus the thinking text (local tiers put the answer in reasoning_content when they run out of tokens)."""
        _, _, data, _ = resp
        if not isinstance(data, dict):
            return ""
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        return str(msg.get("content") or "") + " " + str(msg.get("reasoning_content") or "")

    # every tier from tiers.conf: a vision tier keeps the image and sees it; a no-vision tier reroutes
    # that request to [routing] vision, which sees it. x-route: private keeps both local.
    for name in t.names:
        model = (t.tier[name]["advertise"] or [f"ultron/{name}"])[0]
        want = f"ultron/{vt}" if name in blind else f"ultron/{name}"
        session = f"suite-vis-{name}-" + uuid.uuid4().hex[:8]
        off = log_len(ADMIT_LOG)
        resp = chat(model, [{"role": "user", "content": image_message()}], session=session, max_tokens=400,
                    x_route="private", timeout=300, extra_headers=SUITE_AGENT)
        dec = new_decisions(off, session)
        endpoint = dec[-1].get("endpoint") if dec else "?"
        vision = dec[-1].get("vision") if dec else None
        txt = reply_text(resp)
        if name in blind:
            check("vision", f"image on {name} (no vision) reroutes to {vt}",
                  endpoint == want and vision == f"vision->{vt}", f"endpoint={endpoint} vision={vision}")
        else:
            check("vision", f"image on {name} stays local", endpoint == want, f"endpoint={endpoint}")
        check("vision", f"{want.split('/')[1]} sees the image (asked as {name})", "red" in txt.lower(),
              f"reply: {txt[:60]!r}")
    return 0


# ----------------------------------------------------------------------------- classify

def cmd_classify(args):
    section("classify: media-hook classifier (pure)")

    def turn(text):
        return {"text": text, "images": [], "audio": [], "content": []}

    pos = [
        ("generate an image of a red fox in the snow", "image"),
        ("Can you draw a picture of a castle at sunset?", "image"),
        ("create a photo of a mountain lake, cinematic", "image"),
        ("/imagine a neon city", "image"),
        ("make a video of a rotating cube", "video"),
        ("generate a short clip of ocean waves", "video"),
        ("search the web for litellm 1.103 release notes", "search"),
        ("what's the latest news about the Mars mission? google it", "search"),
    ]
    neg = [
        "generate an image component in React that lazy loads",
        "create a docker image for the api",
        "make the video player fullscreen",
        "write a function that generates an image thumbnail",
        "search the web codebase for TODO",
        "fix the bug where the image tag is missing",
        "make a video element in html",
        "how do I resize an image with imagemagick",
        "explain how to generate images with diffusion models",
        "create a gif of the dashboard for the readme",
        "",
    ]
    for text, kind in pos:
        got = um.classify(turn(text))
        check("classify", f"positive {kind}: {text[:44]!r}", got == kind, f"got {got}")
    for text in neg:
        got = um.classify(turn(text))
        check("classify", f"negative: {text[:44]!r}", got is None, f"got {got}")
    return 0


# ----------------------------------------------------------------------------- confirm

def cmd_confirm(args):
    section("confirm: helper tier MEDIA/OTHER gate for agent requests (live helper tier, no cloud)")
    # Only text the regex already matched reaches the helper, so every case must pass classify() first.
    # With thinking on, this gate once said no to everything (the reasoning used up max_tokens).
    cases = [
        ("Generate an image of a cat", True),
        ("[Workspace::v1: /opt/data/workspace]\nDraw a picture of a cat", True),  # Hermes wraps the turn
        ("can you make me a logo image for my bakery", True),
        ("give me a wallpaper of mountains, 4k", True),
        ("make a short video of waves crashing", True),
        ("search the web for the latest litellm release notes", True),
        ("make a picture-in-picture mode for the player", False),
        ("I want the image to load faster on mobile", False),
        ("browse the web UI and check the login page works", False),
    ]
    # Known misses with a Qwen3.5-4B helper: "show me the photo upload flow", and "change the button
    # color in the image to match our brand" with a screenshot attached, both come back MEDIA.
    for text, want in cases:
        kind = um.classify({"text": text, "images": [], "audio": [], "content": []})
        label = text.split("\n")[-1][:44]
        if kind is None:
            check("confirm", f"regex matches {label!r}", False, "classify() -> None: the helper never sees it")
            continue
        t0 = time.time()
        got = um.helper_confirm(text, kind)
        check("confirm", f"{'MEDIA' if want else 'OTHER'} ({kind}): {label!r}", got == want,
              f"got {'MEDIA' if got else 'OTHER'} in {time.time() - t0:.1f}s")
    return 0


# ----------------------------------------------------------------------------- image

def cmd_image(args):
    section("image: text -> image")
    with ModeCtx(MEDIA_MODE_FILE, "enforce"):
        before_files = sorted(os.listdir(MEDIA_DIR)) if os.path.isdir(MEDIA_DIR) else []
        prompt = f"generate an image of a red fox in the snow (suite {uuid.uuid4().hex[:4]})"
        off = log_len(MEDIA_LOG)
        resp = chat("claude-sonnet-5", [{"role": "user", "content": prompt}],
                    max_tokens=64, timeout=240)
        txt = content_of(resp) or ""
        entry = new_media(off)
        ok_hook = "Image from" in txt and "/media/" in txt
        ok_log = bool(entry) and entry[-1].get("kind") == "image" and entry[-1].get("applied")
        after_files = sorted(os.listdir(MEDIA_DIR)) if os.path.isdir(MEDIA_DIR) else []
        new_files = [f for f in after_files if f not in before_files]
        check("image", "media hook answers the chat prompt", ok_hook, f"reply: {txt[:90]!r}")
        check("image", "media.jsonl logged image applied", ok_log, f"{entry[-1] if entry else 'no entry'}")
        check("image", "an image file was written", bool(new_files), f"{new_files[:3]}")
        if ok_hook:
            m = re.search(r"/media/([^)\s]+)", txt)
            check("image", "reply link points at a real file",
                  bool(m) and m.group(1) in new_files, f"{m.group(1) if m else 'no link'} in {new_files[:3]}")

        # direct endpoint
        h = {"Authorization": "Bearer " + key("LITELLM_MASTER_KEY"), "Content-Type": "application/json"}
        st, _, data, dt = request("POST", LITELLM + "/v1/images/generations",
                                  {"model": "media/image", "prompt": "a green apple on a table", "n": 1,
                                   "size": "1024x1024"}, h, timeout=240)
        item = {}
        if isinstance(data, dict) and isinstance(data.get("data"), list) and data["data"]:
            item = data["data"][0]
        ok_endpoint = st == 200 and (item.get("b64_json") or item.get("url"))
        check("image", "media/image endpoint returns an image", ok_endpoint,
              f"status={st} keys={sorted(item)[:6] if item else err_of((st, {}, data, dt))[:80]} in {dt:.0f}s")

        # veto: a coding sentence must not be intercepted
        off = log_len(MEDIA_LOG)
        resp = chat("claude-sonnet-5", [{"role": "user", "content": "write a function that generates an image thumbnail"}],
                    max_tokens=48, timeout=120)
        new = new_media(off)
        check("image", "coding sentence is not intercepted", not new, f"{[e.get('kind') for e in new]}")
    return 0


# ----------------------------------------------------------------------------- video

def cmd_video(args):
    section("video: text -> video (async render)")
    with ModeCtx(MEDIA_MODE_FILE, "enforce"):
        off = log_len(MEDIA_LOG)
        prompt = f"make a short video of a cat running through a field (suite {uuid.uuid4().hex[:4]})"
        resp = chat("claude-sonnet-5", [{"role": "user", "content": prompt}],
                    max_tokens=64, timeout=60)
        txt = content_of(resp) or ""
        entry = new_media(off)
        ok_reply = "Video started" in txt and "/media/" in txt
        ok_log = bool(entry) and entry[-1].get("kind") == "video" and entry[-1].get("applied")
        check("video", "media hook answers with the render job", ok_reply, f"reply: {txt[:120]!r}")
        check("video", "media.jsonl logged video applied", ok_log, f"{entry[-1] if entry else 'no entry'}")
        m = re.search(r"/([0-9a-f\-]{6,})\.mp4", txt)
        if not ok_reply or not m:
            check("video", "job file appears (no job parsed)", False, "could not parse job from reply")
            return 0
        job = m.group(1)
        deadline = time.time() + args.video_wait
        state = "rendering"
        while time.time() < deadline:
            files = os.listdir(MEDIA_DIR) if os.path.isdir(MEDIA_DIR) else []
            if any(f.startswith(job) and f.endswith(".mp4") for f in files):
                state = "done"
                break
            if any(f.startswith(job) and f.endswith(".error.txt") for f in files):
                state = "failed"
                break
            time.sleep(5)
        if state == "done":
            check("video", "render job completed", True, f"{job}.mp4 written in ~{int(time.time() - (deadline - args.video_wait))}s")
        elif state == "failed":
            errf = [f for f in os.listdir(MEDIA_DIR) if f.startswith(job) and f.endswith(".error.txt")]
            err = open(os.path.join(MEDIA_DIR, errf[0])).read()[:120] if errf else "?"
            check("video", "render job completed", False, f"job failed: {err}")
        else:
            print(f"  note: job {job} still rendering after {args.video_wait}s (re-check ~/.ultron/media)")
            check("video", "render job completed", True, f"still rendering (checked {args.video_wait}s)")
    return 0


# ----------------------------------------------------------------------------- context

def _mem_sampler(stop, result):
    """Background sampler: min free RAM (GiB) and peak swap used (GiB) while a request runs."""
    import subprocess
    while not stop.is_set():
        try:
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3).stdout
            sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=3).stdout
            free = int(re.search(r"Pages free:\s+(\d+)", vm).group(1)) * 16384 / 2**30
            m = re.search(r"used = ([\d.]+)M", sw)
            result["min_free"] = free if result["min_free"] is None else min(result["min_free"], free)
            result["swap_used"] = max(result["swap_used"], float(m.group(1)) / 1024 if m else 0)
        except Exception:
            pass
        time.sleep(1)


def _padding_tokens(n):
    """A long repeated sentence, ~4.8 chars/token, to fill a context budget (Qwen-ish tokenizers)."""
    sent = ("The quick brown fox jumps over the lazy dog while the sun sets behind the hills and "
            "the birds fly south for the winter. ")
    chars = int(n * 4.8)
    return (sent * (chars // len(sent) + 1))[:chars]


def cmd_context(args):
    section(f"context: prompt >{args.context_tokens / 1000:.0f}k tokens per model (memory sampled)")
    print(f"  models: {args.model.split(',')}   target: ~{args.context_tokens / 1000:.0f}k prompt tokens\n")
    for model in [m.strip() for m in args.model.split(",") if m.strip()]:
        xr = None if model.startswith("cloud/") else "private"
        session = "suite-ctx-" + model.replace("/", "-") + "-" + uuid.uuid4().hex[:6]
        prompt = _padding_tokens(args.context_tokens) + "\n\nReply with exactly the word OK."
        stop, mem = threading.Event(), {"min_free": None, "swap_used": 0}
        t = threading.Thread(target=_mem_sampler, args=(stop, mem), daemon=True)
        t.start()
        t0 = time.time()
        resp = chat(model, [{"role": "user", "content": prompt}], session=session,
                    max_tokens=32, x_route=xr, timeout=max(600, args.context_tokens // 20))
        elapsed = time.time() - t0
        stop.set()
        st, _, data, _ = resp
        if isinstance(data, dict) and data.get("usage"):
            u = data["usage"]
            pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
            detail = (f"prompt={pt:,} completion={ct} in {elapsed:.0f}s "
                      f"(prefill {pt / elapsed:,.0f} tok/s) | min free {mem['min_free']:.1f} GiB "
                      f"| swap peak {mem['swap_used']:.1f} GiB")
            ok = pt >= args.context_tokens and ct > 0
            check("context", f"{model} handles {args.context_tokens // 1000}k context", ok, detail)
        else:
            check("context", f"{model} handles {args.context_tokens // 1000}k context", False,
                  f"status={st} {err_of(resp)[:120]} min_free={mem['min_free']:.1f}GiB swap_peak={mem['swap_used']:.1f}GiB")
    return 0


# ----------------------------------------------------------------------------- baseline

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}


def stream_ttft(model, prompt, session, x_route, timeout=120, extra_headers=None):
    """Time to first token via SSE. Returns (ttft_s, status) or (None, status)."""
    h = {"Authorization": "Bearer " + key("LITELLM_MASTER_KEY"), "Content-Type": "application/json"}
    if session:
        h["X-Claude-Code-Session-Id"] = session
    if x_route:
        h["x-route"] = x_route
    if extra_headers:
        h.update(extra_headers)
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 24, "stream": True}).encode()
    req = urllib.request.Request(LITELLM + "/v1/chat/completions", data=body, headers=h, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None, r.status
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except ValueError:
                    continue
                delta = (ev.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):  # thinking tiers reason first
                    return time.time() - t0, 200
            return None, 200
    except Exception:
        return None, 0


def cmd_baseline(args):
    section("baseline: tokens/s per model" + (" + tool-call latency" if args.tool else "") + (" + TTFT" if args.ttft else ""))
    models = [m.strip() for m in (args.model or ",".join(f"ultron/{n}" for n in ua.tiers().names)).split(",") if m.strip()]
    prompt = ("Write a detailed paragraph about the history of computing from the 1940s to today. "
              "Include the people, machines and ideas, and end with a sentence about the present day.")
    tool_ask = "What is the weather in Paris right now? Use the get_weather tool to find out."
    print(f"  models: {models}   tokens: {args.tokens}   header: x-route private forces local tiers\n")

    for model in models:
        xr = None if model.startswith("cloud/") else "private"
        name = model.replace("/", "-")
        # warm the tier (a tiny request; the load time is excluded from the measurement)
        chat(model, [{"role": "user", "content": "hi"}], session="suite-b-" + name + "-" + uuid.uuid4().hex[:6],
             max_tokens=4, x_route=xr, timeout=150, extra_headers=SUITE_AGENT)
        # measured run
        session = "suite-b-" + name + "-" + uuid.uuid4().hex[:6]
        t0 = time.time()
        st, hdr, data, dt = chat(model, [{"role": "user", "content": prompt}], session=session,
                                 max_tokens=args.tokens, x_route=xr, timeout=300, extra_headers=SUITE_AGENT)
        elapsed = time.time() - t0
        usage = (data or {}).get("usage") or {} if isinstance(data, dict) else {}
        pt, ct = usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
        mstats = (data or {}).get("mtplx_stats") if isinstance(data, dict) else {}
        decode = mstats.get("decode_tok_s") if mstats else None
        ok = st == 200 and ct > 0
        summary = (f"decode {decode:.1f} tok/s" if decode else
                   f"{ct / elapsed:.1f} tok/s (completion/elapsed)" if elapsed else "?") + \
                  f"  | tt {elapsed:.1f}s pt={pt} ct={ct} overall={(pt + ct) / elapsed:.1f}/s"
        check("baseline", model, ok, summary)

        if args.tool:
            t0 = time.time()
            # 400: thinking tiers spend ~60 tokens reasoning before the call; 80 cut sonnet off (length)
            st, _, data, dt = chat(model, [{"role": "user", "content": tool_ask}],
                                   session="suite-t-" + name + "-" + uuid.uuid4().hex[:6],
                                   max_tokens=400, tools=[WEATHER_TOOL], x_route=xr, timeout=180, extra_headers=SUITE_AGENT)
            tc = None
            if isinstance(data, dict):
                msg = (data.get("choices") or [{}])[0].get("message") or {}
                tcs = msg.get("tool_calls") or []
                tc = tcs[0].get("function", {}).get("name") if tcs else None
            check("baseline", f"{model} tool call", tc == "get_weather",
                  f"tool={tc} in {time.time() - t0:.1f}s")

        if args.ttft:
            tt, tst = stream_ttft(model, "Count from one to ten.", "suite-tt-" + name + "-" + uuid.uuid4().hex[:6],
                                  xr, timeout=150, extra_headers=SUITE_AGENT)
            check("baseline", f"{model} TTFT (stream)", tt is not None and tst == 200,
                  f"ttft={tt:.2f}s status={tst}" if tt is not None else f"no first token, status={tst}")
    return 0


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Live integration suite for the ultron stack.", add_help=True)
    ap.add_argument("sections", nargs="*", default=[],
                    help="health, swap, vision, classify, confirm, image, video, baseline, context "
                         "(default: all except live-swap, baseline & context)")
    ap.add_argument("--live", action="store_true", help="run swap live probes (changes resident tiers)")
    ap.add_argument("--model", default=None, help="comma list for baseline (default: every local tier)")
    ap.add_argument("--tokens", type=int, default=200, help="max_tokens for the baseline run")
    ap.add_argument("--tool", action="store_true", help="also measure tool-call latency per model")
    ap.add_argument("--ttft", action="store_true", help="also measure time-to-first-token per model")
    ap.add_argument("--video-wait", type=int, default=90, help="seconds to wait for a render job")
    ap.add_argument("--context-tokens", type=int, default=210000,
                    help="prompt-token budget for the context section (default 210k)")
    args = ap.parse_args()

    chosen = set(args.sections) or {"health", "swap", "vision", "classify", "confirm", "image", "video"}
    unknown = chosen - {"health", "swap", "vision", "classify", "confirm", "image", "video", "baseline", "context"}
    if unknown:
        ap.error("unknown section(s): " + ", ".join(sorted(unknown)))

    live = request("GET", LITELLM + "/health/liveliness", timeout=5)
    needs_live = bool(chosen & {"health", "vision", "confirm", "image", "video", "baseline", "context"}) or (
        args.live and "swap" in chosen)
    if live[0] != 200:
        if needs_live:
            print(f"LiteLLM is not reachable at {LITELLM} (status {live[0]}): "
                  f"{err_of(live) or live[3]} — run this on ultron with the stack up.")
            return 2
        print(f"note: LiteLLM not reachable ({live[0]}); running the pure sections only")

    if "health" in chosen:
        cmd_health(args)
    if "swap" in chosen:
        cmd_swap(args)
    if "vision" in chosen:
        cmd_vision(args)
    if "classify" in chosen:
        cmd_classify(args)
    if "confirm" in chosen:
        cmd_confirm(args)
    if "image" in chosen:
        cmd_image(args)
    if "video" in chosen:
        cmd_video(args)
    if "context" in chosen:
        cmd_context(args)
    if "baseline" in chosen or args.tool or args.ttft:
        cmd_baseline(args)

    return summarize()


if __name__ == "__main__":
    sys.exit(main())