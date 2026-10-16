"""Decision table, matrix, pins and shadow/enforce behavior against a mocked llama-swap."""

import os
import re
import asyncio
import json
import time

import pytest

import ultron_admit as ua

SETS = ua._expand("(opus | sonnet) & haiku", {}, {})


def st(pressure=1, mem_tight=None, **tiers):
    """st(haiku=(active, waiting), opus='starting') -> state dict."""
    out = {}
    for t, v in tiers.items():
        out[t] = {"state": "starting", "active": 0, "waiting": 0} if v == "starting" else \
            {"state": "ready", "active": v[0], "waiting": v[1]}
    return {"tiers": out, "pressure": pressure, "mem_tight": mem_tight}


def test_matrix_expansion_and_refs():
    assert set(SETS) == {frozenset({"opus", "haiku"}), frozenset({"sonnet", "haiku"})}
    sets = {"llms": "a | b", "x": "+llms & c"}
    assert set(ua._expand(sets["x"], sets, {})) == {frozenset("ac"), frozenset("bc")}
    assert set(ua._expand("(g | q) & v", {}, {"g": "gemma", "q": "qwen", "v": "vox"})) == \
        {frozenset({"gemma", "vox"}), frozenset({"qwen", "vox"})}


def test_tier_globs():
    assert ua.tier_for("claude-opus-5-5") == "opus"
    assert ua.tier_for("claude-fable-5-1") == "fable"
    assert ua.tier_for("ultron/fable") == "fable" and ua.tier_for("fable") == "fable"
    assert ua.tier_for("ultron/haiku") == "haiku"
    assert ua.tier_for("claude-haiku-4-5-20251001") == "haiku"
    assert ua.tier_for("claude-sonnet-5") == "sonnet"
    assert ua.tier_for("gpt-whatever") == "sonnet"
    assert ua.tier_for("cloud/opus") is None


@pytest.mark.parametrize("tier,state,reserved,x_route,mode,expect", [
    # 1. loaded, queue under max_waiting
    ("haiku", st(haiku=(1, 2)), set(), "", "auto", ("ultron/haiku", "1:loaded")),
    ("opus", st(opus=(1, 0), haiku=(0, 0)), set(), "", "auto", ("ultron/opus", "1:loaded")),
    ("opus", st(opus="starting", haiku=(0, 0)), set(), "", "auto", ("ultron/opus", "1:loading")),
    # queue full -> no substitute for opus -> cloud
    ("opus", st(opus=(1, 20), haiku=(0, 0)), set(), "", "auto", ("cloud/opus", "4:overflow:queue")),
    # 2. cold load that fits next to haiku
    ("sonnet", st(haiku=(0, 0)), set(), "", "auto", ("ultron/sonnet", "2:cold-fits")),
    # the 2026-09-29 bug: opus unloaded, haiku ready, one idle main pin on sonnet -> opus
    # still cold-loads locally (the unloaded sonnet pin must not block it)
    ("opus", st(haiku=(0, 0)), {"sonnet"}, "", "auto", ("ultron/opus", "2:cold-fits")),
    # ...but not under memory pressure -> no substitute loaded -> cloud
    ("sonnet", st(pressure=2, haiku=(0, 0)), set(), "", "auto", ("cloud/sonnet", "4:overflow:no-fit")),
    # ...but kernel pressure "warn" never turns away a tier that is already loaded
    ("opus", st(pressure=2, opus=(0, 0), haiku=(0, 0)), set(), "", "auto", ("ultron/opus", "1:loaded")),
    # sonnet doesn't fit next to opus -> 3. substitute opus (loaded, idle)
    ("sonnet", st(opus=(0, 0), haiku=(0, 0)), set(), "", "auto", ("ultron/opus", "3:substitute(sonnet->opus)")),
    # a main-thread pin on opus, while opus is unloaded, must not block sonnet's cold load
    ("sonnet", st(haiku=(0, 0)), {"opus"}, "", "auto", ("ultron/sonnet", "2:cold-fits")),
    # a pin on the tier next to the cold load (haiku, loaded) is a no-op: haiku is already
    # in `loaded`, so sonnet still fits next to it
    ("sonnet", st(haiku=(0, 0)), {"haiku"}, "", "auto", ("ultron/sonnet", "2:cold-fits")),
    # haiku queue full -> substitute sonnet
    ("haiku", st(haiku=(1, 50), sonnet=(0, 0)), set(), "", "auto", ("ultron/sonnet", "3:substitute(haiku->sonnet)")),
    # 5. private caller never goes to cloud
    ("opus", st(opus=(1, 20), haiku=(0, 0)), set(), "private", "auto", ("ultron/opus", "5:local-swap")),
    ("sonnet", st(opus=(1, 20), haiku=(0, 0)), set(), "", "local-only", ("ultron/sonnet", "5:local-swap")),
    # overrides
    ("haiku", st(haiku=(0, 0)), set(), "cloud", "auto", ("cloud/haiku", "x-route:cloud")),
    ("haiku", st(haiku=(0, 0)), set(), "", "cloud-only", ("cloud/haiku", "route-mode:cloud-only")),
    ("haiku", st(haiku=(0, 0)), set(), "private", "cloud-only", ("ultron/haiku", "1:loaded")),
    # memory guard: nearly swapping -> cloud even when the tier is loaded and idle
    ("opus", st(mem_tight="headroom 2.0G", opus=(0, 0), haiku=(0, 0)), set(), "", "auto", ("cloud/opus", "4:overflow:mem")),
    ("haiku", st(mem_tight="swapping 40MB/s", haiku=(0, 0)), set(), "", "auto", ("cloud/haiku", "4:overflow:mem")),
    # ...but never for private / local-only, and then no cold load: use what's loaded
    ("opus", st(mem_tight="pressure 2", opus=(0, 0), haiku=(0, 0)), set(), "private", "auto", ("ultron/opus", "1:loaded")),
    ("haiku", st(mem_tight="pressure 2", sonnet=(0, 0)), set(), "", "local-only", ("ultron/sonnet", "3:substitute(haiku->sonnet)")),
])
def test_decision_table(tier, state, reserved, x_route, mode, expect):
    assert ua.decide(tier, state, reserved, x_route, mode, SETS) == expect


def test_evictees_follow_solver():
    assert ua.evictees("sonnet", {"opus", "haiku"}, SETS, {"opus": 3, "sonnet": 2, "haiku": 5}) == {"opus"}
    assert ua.evictees("opus", {"sonnet", "haiku"}, SETS, {}) == {"sonnet"}


def req(model="claude-opus-5-5", session="S1", agent=None, first="build the level", x_route=None, cid="c1"):
    h = {}
    if session:
        h["X-Claude-Code-Session-Id"] = session
    if agent:
        h["x-claude-code-agent-id"] = agent
    if x_route:
        h["x-route"] = x_route
    return {"model": model, "litellm_call_id": cid, "proxy_server_request": {"headers": h},
            "messages": [{"role": "user", "content": first}]}


@pytest.fixture
def adm(tmp_path, monkeypatch):
    monkeypatch.setattr(ua, "PINS_DB", str(tmp_path / "pins.sqlite"))
    monkeypatch.setattr(ua, "LOG_PATH", str(tmp_path / "admit.jsonl"))
    monkeypatch.setattr(ua, "ROUTE_MODE_FILE", str(tmp_path / "route-mode"))
    monkeypatch.setattr(ua, "matrix", lambda: (SETS, {"opus": 3, "sonnet": 2, "haiku": 5}))
    a = ua.Admission()
    a.fake = st(opus=(1, 20), haiku=(0, 0))

    async def state():
        return a.fake
    a.state = state
    return a


def run(coro):
    return asyncio.run(coro)


def test_pin_holds_backend_for_the_whole_conversation(adm):
    adm.fake = st(opus=(0, 0), haiku=(0, 0))
    run(adm.admit(req(), "anthropic_messages", "enforce"))
    adm.fake = st(opus=(1, 1), haiku=(0, 0))  # opus busy: a local pin waits its turn, no cloud
    d2 = req(cid="c2")
    run(adm.admit(d2, "anthropic_messages", "enforce"))
    assert d2["model"] == "ultron/opus" and adm.recent["c2"].startswith("ultron/opus; rule=pinned:1:loaded")


def test_overflow_pin_returns_local_once_it_fits(adm):
    d = req()
    run(adm.admit(d, "anthropic_messages", "enforce"))
    assert d["model"] == "cloud/opus" and d["extra_headers"]["X-Session-Id"] == "S1"
    d2 = req(cid="c2")  # opus still busy: the overflow pin stands
    run(adm.admit(d2, "anthropic_messages", "enforce"))
    assert d2["model"] == "cloud/opus" and adm.recent["c2"].startswith("cloud/opus; rule=pinned:4:overflow:queue")
    adm.fake = st(opus=(0, 0), haiku=(0, 0))  # opus frees up: the conversation re-pins local
    d3 = req(cid="c3")
    run(adm.admit(d3, "anthropic_messages", "enforce"))
    assert d3["model"] == "ultron/opus" and "extra_headers" not in d3
    assert adm.recent["c3"].startswith("ultron/opus; rule=repin:1:loaded")
    last = json.loads(open(ua.LOG_PATH).read().splitlines()[-1])
    assert last["rule"] == "repin:1:loaded" and last["new"] is False
    adm.fake = st(opus=(1, 1), haiku=(0, 0))  # now an ordinary local pin: busy opus no longer overflows
    d4 = req(cid="c4")
    run(adm.admit(d4, "anthropic_messages", "enforce"))
    assert d4["model"] == "ultron/opus" and adm.recent["c4"].startswith("ultron/opus; rule=pinned:1:loaded")


def test_explicit_cloud_pin_stays_cloud(adm):
    adm.fake = st(opus=(0, 0), haiku=(0, 0))
    run(adm.admit(req(x_route="cloud"), "anthropic_messages", "enforce"))
    d2 = req(cid="c2")  # header gone, opus idle: the conversation still stays on cloud
    run(adm.admit(d2, "anthropic_messages", "enforce"))
    assert d2["model"] == "cloud/opus" and adm.recent["c2"].startswith("cloud/opus; rule=pinned:x-route:cloud")


def test_mock_replies_are_not_admitted(adm):
    adm.fake = st(opus="starting", haiku=(0, 0))
    d = req()
    before = d["model"]
    d["mock_response"] = "answered by an earlier hook"
    assert run(adm.admit(d, "anthropic_messages", "enforce")) is None
    assert d["model"] == before


def test_media_models_are_not_tiers(adm):
    d = req(model="media/image")
    assert run(adm.admit(d, "image_generation", "enforce")) is None and d["model"] == "media/image"


def test_vision_reroutes_to_sonnet(adm):
    img = [{"type": "text", "text": "what is this?"},
           {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}}]
    adm.fake = st(opus=(0, 0), sonnet=(0, 0), haiku=(0, 0))
    # opus has no vision in the example tiers (text-only pack): a newest-turn image reroutes to sonnet, this request only
    d = req(model="claude-opus-5-5", session="S2")
    d["messages"] = [{"role": "user", "content": "look"}, {"role": "assistant", "content": "ok"},
                     {"role": "user", "content": [img[0], img[1]]}]
    run(adm.admit(d, "anthropic_messages", "enforce"))
    assert d["model"] == "ultron/sonnet"
    assert adm.recent["c1"].endswith("vision->sonnet")
    # sonnet has a vision tower: an image stays on sonnet
    s = req(model="claude-sonnet-5", session="S3", cid="c2")
    s["messages"] = [{"role": "user", "content": [img[0], img[1]]}]
    run(adm.admit(s, "anthropic_messages", "enforce"))
    assert s["model"] == "ultron/sonnet"


def test_subagents_get_their_own_pins_and_do_not_reserve(adm):
    adm.fake = st(haiku=(0, 0))
    main = req(model="claude-sonnet-5")
    run(adm.admit(main, "anthropic_messages", "enforce"))
    assert main["model"] == "ultron/sonnet"
    sub = req(model="claude-haiku-4-5", agent="agent-1")
    run(adm.admit(sub, "anthropic_messages", "enforce"))
    assert sub["model"] == "ultron/haiku"
    pins = adm._pins()
    assert pins.main_local_tiers("enforce") == {"sonnet"}
    # a second session asking for opus: the main pin is on sonnet (unloaded) and the sub pin
    # is on haiku; the unloaded sonnet pin no longer blocks a cold load, so opus stays local
    other = req(model="claude-opus-5-5", session="S2")
    run(adm.admit(other, "anthropic_messages", "enforce"))
    assert other["model"] == "ultron/opus"


def test_hash_key_for_clients_without_session_header(adm):
    adm.fake = st(haiku=(0, 0))
    a = req(model="haiku", session=None, first="hermes turn 1")
    run(adm.admit(a, "acompletion", "enforce"))
    b = req(model="haiku", session=None, first="hermes turn 1", cid="c2")
    b["messages"] += [{"role": "assistant", "content": "x"}, {"role": "user", "content": "more"}]
    run(adm.admit(b, "acompletion", "enforce"))
    lines = [json.loads(l) for l in open(ua.LOG_PATH)]
    assert lines[0]["key"] == lines[1]["key"] and lines[1]["new"] is False


def test_shadow_logs_without_rewriting_but_honors_explicit_overrides(adm, tmp_path):
    d = req()
    run(adm.admit(d, "anthropic_messages", "shadow"))
    assert d["model"] == "claude-opus-5-5" and "extra_headers" not in d
    assert json.loads(open(ua.LOG_PATH).read().splitlines()[-1])["target"] == "cloud/opus"
    c = req(session="S9", x_route="cloud")
    run(adm.admit(c, "anthropic_messages", "shadow"))
    assert c["model"] == "cloud/opus"
    (tmp_path / "route-mode").write_text("cloud-only\n")
    m = req(model="claude-haiku-4-5", session="S10")
    run(adm.admit(m, "acompletion", "shadow"))
    assert m["model"] == "cloud/haiku"
    # a conversation pinned by an override stays on cloud after the mode flips back
    (tmp_path / "route-mode").write_text("auto\n")
    m2 = req(model="claude-haiku-4-5", session="S10", cid="c3")
    run(adm.admit(m2, "acompletion", "shadow"))
    assert m2["model"] == "cloud/haiku"


def test_cloud_only_moves_live_local_conversations_too(adm, tmp_path):
    adm.fake = st(opus=(0, 0), haiku=(0, 0))
    run(adm.admit(req(), "anthropic_messages", "enforce"))  # pinned ultron/opus
    (tmp_path / "route-mode").write_text("cloud-only\n")
    d = req(cid="c2")
    out = run(adm.admit(d, "anthropic_messages", "enforce"))
    assert d["model"] == "cloud/opus" and out["overflow"] == "route-mode" and out["rule"] == "pinned:1:loaded"
    assert adm.recent["c2"].endswith("; overflow=route-mode")
    p = req(session="S7", cid="p1", x_route="private")
    run(adm.admit(p, "anthropic_messages", "enforce"))
    assert p["model"] == "ultron/opus"  # private never goes to cloud
    (tmp_path / "route-mode").write_text("auto\n")
    d3 = req(cid="c3")
    run(adm.admit(d3, "anthropic_messages", "enforce"))
    assert d3["model"] == "ultron/opus"  # the pin was never moved


def test_judge_decide_pass_through_is_admitted_as_the_judge(adm, monkeypatch):
    class Logging:
        litellm_call_id = "j1"
    body = {"kind": "noul", "question": "q?", "state": "s", "litellm_logging_obj": Logging()}
    assert ua.decide_call_id(body, "pass_through_endpoint") == "j1"
    assert ua.decide_call_id(body, "acompletion") is None
    assert ua.decide_call_id({"litellm_logging_obj": Logging()}, "pass_through_endpoint") is None
    out = run(adm.admit(body, "pass_through_endpoint", "shadow"))  # shadow: no waiting for evictees
    assert out["rule"] == "direct" and out["call_id"] == "j1" and adm.recent["j1"].startswith("ultron/judge")
    assert set(body) == {"kind", "question", "state", "litellm_logging_obj"}  # the judge's body goes on unchanged
    assert run(adm.admit({"litellm_logging_obj": Logging()}, "pass_through_endpoint", "enforce")) is None


def test_explicit_cloud_model_is_left_alone(adm):
    d = req(model="cloud/sonnet")
    assert run(adm.admit(d, "acompletion", "enforce")) is None and d["model"] == "cloud/sonnet"


def test_rule5_waits_for_busy_tier_then_swaps(adm, monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(ua.asyncio, "sleep", lambda s: real_sleep(0))
    states = iter([st(sonnet=(1, 0), haiku=(0, 0))] * 3 + [st(sonnet=(0, 0), haiku=(0, 0))] * 10)
    adm.fake = st(sonnet=(1, 1), haiku=(0, 0))

    async def state():
        return next(states) if adm._state is None else adm.fake
    adm.state = state
    d = req(model="claude-opus-5-5", x_route="private")
    run(adm.admit(d, "anthropic_messages", "enforce"))
    assert d["model"] == "ultron/opus"
    last = json.loads(open(ua.LOG_PATH).read().splitlines()[-1])
    assert last["rule"] == "5:local-swap" and last["busy_evictees_after_wait"] is None


def test_client_info_classification():
    cc = req(); cc["proxy_server_request"]["headers"]["user-agent"] = "claude-cli/2.1.283 (external, cli)"
    cc["tools"] = [{"name": "Bash"}, {"name": "mcp__monolith__editor_query"}]
    cc["metadata"] = {"requester_ip_address": "100.64.0.10"}
    i = ua.client_info(cc)
    assert i["agent"] == "claude-code" and i["tags"] == ["unreal"] and i["ip"] == "100.64.0.10"
    herm = {"messages": [], "tools": [{"type": "function", "function": {"name": "browser_vault_fill"}}],
            "proxy_server_request": {"headers": {"user-agent": "OpenAI/Python 2.8.1"}}}
    assert ua.client_info(herm)["agent"] == "hermes"
    pi = {"messages": [], "tools": [{"type": "function", "function": {"name": n}} for n in ("read", "bash", "edit", "write", "ls")]}
    assert ua.client_info(pi)["agent"] == "pi"
    assert ua.client_info({"messages": []})["agent"] == "other"
    cline = {"messages": [], "proxy_server_request": {"headers": {"X-Title": "Cline", "user-agent": "OpenAI/JS 5.0"}}}
    assert ua.client_info(cline)["agent"] == "cline"
    sdk = {"messages": [], "proxy_server_request": {"headers": {"user-agent": "OpenAI/Python 2.8.1"}}}
    assert ua.client_info(sdk)["agent"] == "openai-sdk" and ua.client_info(sdk)["tags"] is None


def test_explicit_cloud_model_is_logged_as_traffic(adm):
    d = req(model="cloud/haiku")
    run(adm.admit(d, "acompletion", "shadow"))
    last = json.loads(open(ua.LOG_PATH).read().splitlines()[-1])
    assert last["endpoint"] == "cloud/haiku" and last["rule"] == "explicit" and last["agent"] == "claude-code"


def test_shadow_endpoint_is_the_configured_local_route(adm):
    d = req()
    run(adm.admit(d, "anthropic_messages", "shadow"))
    last = json.loads(open(ua.LOG_PATH).read().splitlines()[-1])
    assert last["target"] == "cloud/opus" and last["endpoint"] == "ultron/opus"


def test_non_mtplx_tier_counts_from_litellm_inflight(tmp_path, monkeypatch):
    live = tmp_path / "live.json"
    live.write_text(json.dumps({"inflight": [
        {"cid": "me", "endpoint": "ultron/opus", "phase": "waiting"},
        {"cid": "a", "endpoint": "ultron/opus", "phase": "generating"},
        {"cid": "b", "endpoint": "ultron/opus", "phase": "waiting"},
        {"cid": "c", "endpoint": "ultron/opus", "phase": "ending"},
        {"cid": "d", "endpoint": "cloud/opus", "phase": "generating"}]}))
    monkeypatch.setattr(ua, "STATS_LIVE", str(live))
    assert ua.litellm_inflight() == {"opus": ["me", "a", "b"]}
    st = {"tiers": {"opus": {"state": "ready", "active": 0, "waiting": 0, "cids": ["me", "a", "b"]},
                    "haiku": {"state": "ready", "active": 1, "waiting": 0}}, "pressure": 1}
    v = ua.own_view(st, "me")
    assert v["tiers"]["opus"] == {"state": "ready", "active": 1, "waiting": 1}
    assert v["tiers"]["haiku"] == {"state": "ready", "active": 1, "waiting": 0}
    assert "cids" in st["tiers"]["opus"]  # the cached state is not modified
    assert ua.own_view({"tiers": {"haiku": {"state": "ready"}}}, "x") == {"tiers": {"haiku": {"state": "ready"}}}


def _busy_sonnet(adm, monkeypatch, polls):
    """sonnet loaded and busy for `polls` state reads, then idle; the conversation is pinned to opus."""
    real_sleep = asyncio.sleep
    monkeypatch.setattr(ua.asyncio, "sleep", lambda s: real_sleep(0))
    states = iter([st(sonnet=(1, 0), haiku=(0, 0))] * polls + [st(sonnet=(0, 0), haiku=(0, 0))] * 10)
    adm.fake = st(opus=(0, 0), haiku=(0, 0))
    run(adm.admit(req(), "anthropic_messages", "enforce"))  # pins ultron/opus
    adm.fake = st(sonnet=(1, 0), haiku=(0, 0))  # something swapped opus out

    async def state():
        return next(states) if adm._state is None else adm.fake
    adm.state = state


def test_pinned_reload_overflows_to_cloud_when_evictee_stays_busy(adm, monkeypatch):
    _busy_sonnet(adm, monkeypatch, polls=10**6)
    monkeypatch.setattr(ua, "OVERFLOW_WAIT_S", 0)
    d = req(cid="c2")
    run(adm.admit(d, "anthropic_messages", "enforce"))
    assert d["model"] == "cloud/opus" and d["extra_headers"]["X-Session-Id"] == "S1"
    assert adm.recent["c2"].endswith("overflow=evict-busy")
    adm.fake = st(opus=(0, 0), haiku=(0, 0))  # opus is back: the pin itself stayed local

    async def state():
        return adm.fake
    adm.state = state
    d3 = req(cid="c3")
    run(adm.admit(d3, "anthropic_messages", "enforce"))
    assert d3["model"] == "ultron/opus"


def test_pinned_reload_waits_when_evictee_frees_up(adm, monkeypatch):
    _busy_sonnet(adm, monkeypatch, polls=3)
    d = req(cid="c2")
    run(adm.admit(d, "anthropic_messages", "enforce"))
    assert d["model"] == "ultron/opus"


def test_local_only_never_overflows_a_busy_reload(adm, monkeypatch, tmp_path):
    _busy_sonnet(adm, monkeypatch, polls=10**6)
    (tmp_path / "route-mode").write_text("local-only\n")
    monkeypatch.setattr(ua, "EVICT_WAIT_MAX_S", 0)
    monkeypatch.setattr(ua, "MEM_WAIT_MAX_S", 0)  # the memory gate would wait on the busy sonnet too
    d = req(cid="c2")
    run(adm.admit(d, "anthropic_messages", "enforce"))
    assert d["model"] == "ultron/opus"


def test_pinned_opus_overflows_while_sonnet_is_warm(adm, monkeypatch):
    """An opus conversation pinned local must not evict a loaded sonnet a main thread just used."""
    import time
    real_sleep = asyncio.sleep
    monkeypatch.setattr(ua.asyncio, "sleep", lambda s: real_sleep(0))
    monkeypatch.setattr(ua, "OVERFLOW_WAIT_S", 0)
    adm.fake = st(sonnet=(0, 0), haiku=(0, 0))
    p = adm._pins()
    now = time.time()
    p.put("enforce", "cc:S-warm:main:sonnet", "ultron/sonnet", "sonnet", True, "S-warm", "1:loaded", now)
    p.put("enforce", "cc:S-opus:main:opus", "ultron/opus", "opus", True, "S-opus", "3:substitute(sonnet->opus)", now)
    d = req(model="claude-opus-5-5", session="S-opus", cid="c2")
    run(adm.admit(d, "anthropic_messages", "enforce"))
    assert d["model"] == "cloud/opus"  # warm sonnet is not evicted; the pinned opus request overflows
    assert adm.recent["c2"].endswith("overflow=evict-busy")
    # an idle sonnet (no recent main use) is evictable: the same request stays local
    p.db.execute("UPDATE pins SET last_seen=? WHERE key=?", (now - ua.WARM_MAIN_S - 1, "cc:S-warm:main:sonnet"))
    d2 = req(model="claude-opus-5-5", session="S-opus", cid="c3")
    run(adm.admit(d2, "anthropic_messages", "enforce"))
    assert d2["model"] == "ultron/opus"


def test_read_mem_parses_sysctl(monkeypatch):
    out = ("hw.pagesize: 16384\nkern.memorystatus_vm_pressure_level: 1\nvm.page_free_count: 100000\n"
           "vm.page_pageable_external_count: 60000\nvm.page_speculative_count: 1000\nvm.page_purgeable_count: 24\n"
           "vm.compressor.swapper.swapouts_total: 107851626\n")

    class R:
        stdout = out
    monkeypatch.setattr(ua.subprocess, "run", lambda *a, **k: R)
    m = ua.read_mem()
    assert m["pressure"] == 1 and m["swapouts"] == 107851626
    assert m["headroom"] == (100000 + 60000 + 1000 + 24) * 16384
    R.stdout = "hw.pagesize: 16384\nkern.memorystatus_vm_pressure_level: 2\n"  # older macOS: oids missing
    m = ua.read_mem()
    assert m["pressure"] == 2 and m["headroom"] is None and m["swapouts"] is None


def test_shed_event_picks_sheds_that_threw_cache_away():
    now = 10_000.0
    shed = lambda ts, prompt, reuse, mode="live_session", shared=None, **kw: dict(
        {"ts": ts, "action": "prefill_admission_shed", "prompt_tokens": prompt, "reusable_prefix_tokens": reuse,
         "reusable_prefix_mode": mode, "miss_tokens": prompt - reuse},
        **({"kept_session_entries": {"longest_shared_prefix_tokens": shared}} if shared is not None else {}), **kw)
    assert ua.shed_event(None, now) is None and ua.shed_event([], now) is None
    growth = shed(now - 5, 87000, 74000)                      # the turn's new 13k tokens, cache kept: routine
    small_new = shed(now - 5, 17000, 0, mode="none")          # 17k from nothing: may just be a new conversation
    trims = {"ts": now - 5, "action": "pressure_trim", "level": 4}  # mtplx relieving pressure, nothing lost
    assert ua.shed_event([growth, small_new, trims], now) is None
    lost = shed(now - 8, 34411, 9216, mode="near_prefix", shared=34301)
    assert ua.shed_event([growth, lost], now) == {"ts": now - 8, "why": "shed lost 25k (9k/34k)"}
    assert ua.shed_event([shed(now - 8, 34411, 30000, shared=34301)], now) is None  # lost 4k: under the bar
    assert ua.shed_event([shed(now - 3, 79898, 0, mode="none")], now)["why"] == "shed cold 0k/79k"
    abort = {"ts": now - 2, "action": "prefill_system_abort", "reason": "death_signature"}
    assert ua.shed_event([lost, abort], now)["why"] == "prefill_system_abort"  # newest wins
    assert ua.shed_event([dict(abort, ts=now - ua.SHED_LOOKBACK_S - 1)], now) is None  # stale


def test_shed_holds_are_per_tier(adm, monkeypatch):
    monkeypatch.setattr(ua, "SHED_HOLD_S", {"opus": 60.0, "sonnet": 30.0})
    shed = {"ts": 1000.0, "why": "prefill_system_abort"}
    assert adm.note_shed("sonnet", shed) and not adm.note_shed("sonnet", shed)  # same shed once
    assert adm.shed_hold("sonnet", 1029) == "mtplx sonnet prefill_system_abort" and adm.shed_hold("sonnet", 1031) is None
    assert adm.shed_hold("opus", 1001) is None  # another tier's shed doesn't move opus work
    assert not adm.note_shed("haiku", shed)  # no hold configured: off
    adm.fake = st(opus=(0, 0), sonnet=(0, 0), haiku=(0, 0))
    adm.note_shed("sonnet", {"ts": time.time(), "why": "shed lost 25k (9k/34k)"})
    d = req(model="claude-sonnet-4-6", session="S9", cid="s9")
    out = run(adm.admit(d, "acompletion", "enforce"))
    assert d["model"] == "cloud/sonnet" and out["overflow"] == "shed" and out["shed"].startswith("mtplx sonnet shed lost")
    assert adm.recent["s9"].endswith("; overflow=shed; shed=mtplx sonnet shed lost 25k (9k/34k)")
    o = req(model="claude-opus-5-5", session="S9", cid="o9")
    run(adm.admit(o, "acompletion", "enforce"))
    assert o["model"] == "ultron/opus"
    p = req(model="claude-sonnet-4-6", session="S8", cid="p8", x_route="private")
    run(adm.admit(p, "acompletion", "enforce"))
    assert p["model"] == "ultron/sonnet"  # private never goes to cloud


def test_backlog_of_disk_saves_sends_that_tier_to_cloud(adm, monkeypatch):
    monkeypatch.setattr(ua, "BACKLOG_GB", 2.0)
    G = 2**30
    rows = adm.note_backlogs({"sonnet": {"backlog": int(1.5 * G)}, "opus": {"backlog": 0}}, 100.0)
    assert rows == [] and adm.backlog_hold("sonnet") is None
    rows = adm.note_backlogs({"sonnet": {"backlog": int(7.5 * G)}, "opus": {"backlog": 0}}, 101.0)
    assert rows == [{"ts": 101.0, "event": "backlog", "tier": "sonnet", "pending_gb": 7.5}]
    assert adm.note_backlogs({"sonnet": {"backlog": int(1.5 * G)}}, 140.0) == []  # still over half: held
    assert adm.backlog_hold("sonnet") == "mtplx sonnet 1.5G saves pending" and adm.backlog_hold("opus") is None
    adm.fake = st(opus=(0, 0), sonnet=(0, 0), haiku=(0, 0))
    d = req(model="claude-sonnet-4-6", session="B1", cid="b1")
    out = run(adm.admit(d, "acompletion", "enforce"))
    assert d["model"] == "cloud/sonnet" and out["overflow"] == "backlog" and out["backlog"].startswith("mtplx sonnet")
    assert adm.recent["b1"].endswith("; overflow=backlog; backlog=mtplx sonnet 1.5G saves pending")
    o = req(session="B1", cid="o1")
    run(adm.admit(o, "acompletion", "enforce"))
    assert o["model"] == "ultron/opus"  # other tiers unaffected
    p = req(model="claude-sonnet-4-6", session="B2", cid="p1", x_route="private")
    run(adm.admit(p, "acompletion", "enforce"))
    assert p["model"] == "ultron/sonnet"  # private never goes to cloud
    rows = adm.note_backlogs({"sonnet": {"backlog": int(0.5 * G)}}, 160.0)
    assert rows == [{"ts": 160.0, "event": "backlog_clear", "tier": "sonnet", "pending_gb": 0.5, "held_s": 59}]
    assert adm.backlog_hold("sonnet") is None
    adm.note_backlogs({"opus": {"backlog": 3 * G}}, 170.0)
    assert adm.note_backlogs({}, 171.0)[0]["event"] == "backlog_clear"  # tier unloaded: its saves are gone
    monkeypatch.setattr(ua, "BACKLOG_GB", 0.0)
    assert adm.note_backlogs({"opus": {"backlog": 9 * G}}, 180.0) == []  # 0: off


def test_read_state_reports_sheds_per_tier(monkeypatch):
    now = time.time()
    snaps = {
        "http://127.0.0.1:18004": {"active_requests": 0, "memory_guard_events": [
            {"ts": now - 20, "action": "prefill_admission_shed", "prompt_tokens": 60000, "reusable_prefix_tokens": 2816,
             "reusable_prefix_mode": "live_session", "kept_session_entries": {"longest_shared_prefix_tokens": 57000}}]},
        "http://127.0.0.1:18005": {"active_requests": 1, "scheduler": {"telemetry": {"persistence_pending_bytes": 7573005312}},
                                   "mem": {"phys_footprint_bytes": 15 * 2**30, "host_overhang_bytes": 5.4 * 2**30},
                                   "memory_guard_events": [
            {"ts": now - 3, "action": "prefill_system_abort"}, {"ts": now - 1, "action": "pressure_trim", "level": 4}]},
    }

    def get_json(url, timeout=1.5):
        if url.endswith("/running"):
            return {"running": [{"model": "opus", "state": "ready", "proxy": "http://127.0.0.1:18004"},
                                {"model": "sonnet", "state": "ready", "proxy": "http://127.0.0.1:18005"}]}
        return snaps[url.rsplit("/v1/", 1)[0]]
    monkeypatch.setattr(ua, "_get_json", get_json)
    monkeypatch.setattr(ua, "read_mem", lambda: {"t": now, "pressure": 1, "headroom": 9 * 2**30})
    st_ = ua.read_state()
    assert st_["tiers"]["sonnet"]["shed"]["why"] == "prefill_system_abort" and st_["tiers"]["sonnet"]["active"] == 1
    assert st_["tiers"]["opus"]["shed"]["why"] == "shed lost 54k (2k/60k)"
    assert st_["tiers"]["sonnet"]["backlog"] == 7573005312 and st_["tiers"]["opus"]["backlog"] == 0
    assert st_["tiers"]["sonnet"]["fp"] == 15 * 2**30 and st_["tiers"]["opus"]["fp"] is None


def test_admit_rows_carry_tier_footprints(adm):
    adm.fake = st(opus=(0, 0), haiku=(0, 0))
    adm.fake["tiers"]["opus"].update(fp=int(30.2 * 2**30), host=int(2.2 * 2**30))
    out = run(adm.admit(req(cid="f1"), "acompletion", "enforce"))
    assert out["loaded"]["opus"] == {"state": "ready", "active": 0, "waiting": 0, "fp_gb": 30.2, "host_gb": 2.2}
    assert out["loaded"]["haiku"] == {"state": "ready", "active": 0, "waiting": 0}  # no snapshot: no fields


def test_mem_guard_triggers_and_holds():
    a = ua.Admission()
    G = 2**30
    ok = lambda t, so=0, **k: {"t": t, "page": 16384, "pressure": 1, "headroom": 10 * G, "swapouts": so, **k}
    assert a.mem_guard(ok(1000)) is None
    assert a.mem_guard(ok(1001, headroom=3 * G)) == "headroom 3.0G"
    # held MEM_HOLD_S after the last trigger, even with headroom back
    assert a.mem_guard(ok(1001 + ua.MEM_HOLD_S - 1)) == "headroom 3.0G"
    assert a.mem_guard(ok(1001 + ua.MEM_HOLD_S + 1)) is None
    # after the hold: still tight until headroom is back over MEM_CLEAR_GB (hysteresis), then clear
    assert a.mem_guard(ok(1500, headroom=3 * G)) == "headroom 3.0G"
    assert a.mem_guard(ok(1500 + ua.MEM_HOLD_S + 1, headroom=(ua.MEM_CLEAR_GB - 1) * G)) == "headroom 3.0G"
    assert a.mem_guard(ok(1500 + ua.MEM_HOLD_S + 2, headroom=(ua.MEM_CLEAR_GB + 1) * G)) is None
    assert a.mem_guard(ok(1500 + ua.MEM_HOLD_S + 3, headroom=(ua.MEM_CLEAR_GB - 1) * G)) is None  # no re-trip over MEM_LOW_GB
    assert a.mem_guard(ok(2000, pressure=4)) == "pressure 4"
    # kernel "warn" (2) is the normal state with three tiers resident: not a reason to leave local
    assert ua.Admission().mem_guard(ok(2500, pressure=2)) is None
    # swap-outs: 40 MB/s since the previous sample
    b = ua.Admission()
    b.mem_guard(ok(3000, so=0))
    assert b.mem_guard(ok(3002, so=int(80 * 2**20 / 16384))) == "swapping 40MB/s"
    # a tiny trickle is not swapping; a previous sample older than 30 s is not a rate
    c = ua.Admission()
    c.mem_guard(ok(4000, so=0))
    assert c.mem_guard(ok(4005, so=84)) is None
    assert c.mem_guard(ok(4100, so=10**7)) is None


def test_pinned_local_conversation_overflows_while_memory_is_tight(adm):
    adm.fake = st(opus=(0, 0), haiku=(0, 0))
    run(adm.admit(req(), "anthropic_messages", "enforce"))  # pins ultron/opus
    adm.fake = st(mem_tight="headroom 2.0G", opus=(0, 0), haiku=(0, 0))
    d = req(cid="c2")
    run(adm.admit(d, "anthropic_messages", "enforce"))
    assert d["model"] == "cloud/opus" and d["extra_headers"]["X-Session-Id"] == "S1"
    assert adm.recent["c2"].endswith("overflow=mem; mem=headroom 2.0G")
    last = json.loads(open(ua.LOG_PATH).read().splitlines()[-1])
    assert last["overflow"] == "mem" and last["mem_tight"] == "headroom 2.0G"
    # a new conversation while tight pins cloud, and stays there while it's still tight
    n = req(session="S-new", cid="c3")
    run(adm.admit(n, "anthropic_messages", "enforce"))
    assert n["model"] == "cloud/opus" and adm.recent["c3"].startswith("cloud/opus; rule=4:overflow:mem")
    n1 = req(session="S-new", cid="c3b")
    run(adm.admit(n1, "anthropic_messages", "enforce"))
    assert n1["model"] == "cloud/opus" and adm.recent["c3b"].startswith("cloud/opus; rule=pinned:4:overflow:mem")
    # memory back: both return local; the new one re-pins from its overflow pin
    adm.fake = st(opus=(0, 0), haiku=(0, 0))
    d2, n2 = req(cid="c4"), req(session="S-new", cid="c5")
    run(adm.admit(d2, "anthropic_messages", "enforce"))
    run(adm.admit(n2, "anthropic_messages", "enforce"))
    assert d2["model"] == "ultron/opus" and n2["model"] == "ultron/opus"
    assert adm.recent["c5"].startswith("ultron/opus; rule=repin:1:loaded")


REFUSAL = Exception('litellm.APIError: APIError: Hosted_vllmException - {"error":{"message":"insufficient memory: '
                   'the other apps on this Mac leave 6.1 GiB free, and this prompt needs about 0.7 GiB more"}}')


def test_mtplx_refusal_sends_that_conversation_to_cloud(adm, monkeypatch):
    adm.fake = st(opus=(0, 0), haiku=(0, 0))
    d = req()
    run(adm.admit(d, "anthropic_messages", "enforce"))  # pins ultron/opus
    hook = ua.UltronAdmit()
    hook.admission = adm
    run(hook.async_post_call_failure_hook(d, REFUSAL, None))
    last = json.loads(open(ua.LOG_PATH).read().splitlines()[-1])
    assert last["event"] == "refused" and last["key"] == "cc:S1:main:opus" and last["detail"].startswith("insufficient memory")
    assert "target" not in last  # Wanda's routing tiles count rows with a target
    # the guard is tripped for MEM_HOLD_S: mem_guard keeps the refusal as its reason
    G = 2**30
    assert adm.mem_guard({"t": time.time(), "pressure": 1, "headroom": 10 * G}) == "refused opus"
    # the retry (guard since cleared) still goes to cloud; the pin stays local
    adm.fake = st(opus=(0, 0), haiku=(0, 0))
    r = req(cid="c2")
    run(adm.admit(r, "anthropic_messages", "enforce"))
    assert r["model"] == "cloud/opus" and adm.recent["c2"].endswith("overflow=refused")
    # another conversation is unaffected
    o = req(session="S2", cid="c3")
    run(adm.admit(o, "anthropic_messages", "enforce"))
    assert o["model"] == "ultron/opus"
    # after REFUSED_HOLD_S it comes back local
    adm.refused[last["key"]] = time.time() - 1
    b = req(cid="c4")
    run(adm.admit(b, "anthropic_messages", "enforce"))
    assert b["model"] == "ultron/opus" and adm.recent["c4"].startswith("ultron/opus; rule=pinned:1:loaded")


def test_refusal_ignores_other_errors_and_private(adm):
    adm.fake = st(opus=(0, 0), haiku=(0, 0))
    hook = ua.UltronAdmit()
    hook.admission = adm
    d = req()
    run(adm.admit(d, "anthropic_messages", "enforce"))
    run(hook.async_post_call_failure_hook(d, Exception("Connection reset"), None))
    run(hook.async_post_call_failure_hook({**d, "model": "cloud/opus"}, REFUSAL, None))
    assert adm.refused == {}
    p = req(session="S9", x_route="private", cid="p1")
    run(adm.admit(p, "anthropic_messages", "enforce"))
    run(hook.async_post_call_failure_hook(p, REFUSAL, None))
    p2 = req(session="S9", x_route="private", cid="p2")
    run(adm.admit(p2, "anthropic_messages", "enforce"))
    assert p2["model"] == "ultron/opus"  # private never goes to cloud, refused or not


def test_router_fallback_to_cloud_is_noted_and_holds_the_conversation(adm, monkeypatch):
    monkeypatch.setattr(ua, "admit_mode", lambda: "off")  # pre-call hook: only the fallback opt-out here
    adm.fake = st(opus=(0, 0), haiku=(0, 0))
    hook = ua.UltronAdmit()
    hook.admission = adm
    d = req()
    run(adm.admit(d, "anthropic_messages", "enforce"))  # ultron/opus
    run(hook.async_pre_call_hook(None, None, d, "anthropic_messages"))
    assert "disable_fallbacks" not in d  # a failed local request may fall back to cloud
    cloud = {"model": "openai/custom-opus", "api_base": "https://omniroute.example/v1", "litellm_call_id": "c1",
             "metadata": {"model_group": "cloud/opus"}, "messages": [{"role": "user", "content": "build the level"}]}
    run(hook.async_pre_call_deployment_hook(dict(cloud), "acompletion"))
    run(hook.async_pre_call_deployment_hook(dict(cloud), "acompletion"))  # runs twice per /v1/messages
    assert adm.recent["c1"].endswith("; fallback=cloud/opus") and adm.recent["c1"].count("fallback=") == 1
    rows = [json.loads(x) for x in open(ua.LOG_PATH).read().splitlines()]
    fb = [r for r in rows if r.get("event") == "fallback"]
    assert len(fb) == 1 and fb[0]["tier"] == "opus" and fb[0]["key"] == "cc:S1:main:opus" and "target" not in fb[0]
    r = req(cid="c2")  # the conversation's next request goes to cloud for REFUSED_HOLD_S; the pin stays local
    run(adm.admit(r, "anthropic_messages", "enforce"))
    assert r["model"] == "cloud/opus" and "overflow=refused" in adm.recent["c2"]
    run(hook.async_pre_call_deployment_hook({**cloud, "litellm_call_id": "c2"}, "acompletion"))
    assert "fallback=" not in adm.recent["c2"]  # admitted to cloud: not a fallback
    p = req(session="S9", x_route="private", cid="p1")
    run(hook.async_pre_call_hook(None, None, p, "anthropic_messages"))
    assert p["disable_fallbacks"] is True  # private never falls back to cloud
    with open(ua.ROUTE_MODE_FILE, "w") as f:
        f.write("local-only")
    q = req(session="S8", cid="q1")
    run(hook.async_pre_call_hook(None, None, q, "anthropic_messages"))
    assert q["disable_fallbacks"] is True


def test_config_falls_back_from_each_local_tier_to_its_cloud_twin():
    import yaml
    import ultron_tiers
    t = ua.tiers()
    block = ultron_tiers.blocks(t)["__TIERS_FALLBACKS__"]
    fb = {k: v for d in yaml.safe_load(block)["fallbacks"] for k, v in d.items()}
    assert fb == {f"ultron/{n}": [f"cloud/{n}"] for n in t.names if t.has_cloud(n)}
    assert fb and not any(k.startswith("cloud/") for k in fb)  # never cloud -> local
    cfg = open(os.path.join(os.path.dirname(__file__), "config.yaml")).read()
    assert "# __TIERS_FALLBACKS__" in cfg


def test_private_stays_local_while_memory_is_tight(adm):
    adm.fake = st(mem_tight="pressure 2", opus=(0, 0), haiku=(0, 0))
    d = req(x_route="private")
    run(adm.admit(d, "anthropic_messages", "enforce"))
    assert d["model"] == "ultron/opus"


# ----------------------------------------------------------------------------- local tier history

REM = "<system-reminder>\nToday's date is 2026-09-30.\n</system-reminder>"


def _call(i):
    return {"id": f"c{i}", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}


def test_history_folds_interior_system_into_tool_result():
    msgs = [{"role": "system", "content": "a"}, {"role": "developer", "content": [{"type": "text", "text": "b"}]},
            {"role": "user", "content": "do it"}, {"role": "system", "content": REM},
            {"role": "assistant", "content": "", "tool_calls": [_call(1)]},
            {"role": "tool", "tool_call_id": "c1", "content": "out"}, {"role": "system", "content": REM},
            {"role": "assistant", "content": "", "tool_calls": [_call(2)]},
            {"role": "tool", "tool_call_id": "c2", "content": [{"type": "text", "text": "out2"}]},
            {"role": "developer", "content": "plain note"}]
    out = ua.normalize_history(msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert out[0]["content"] == "a\n\nb"
    assert out[1]["content"] == "do it\n\n" + REM
    assert out[3]["content"] == "out\n\n" + REM
    assert out[5]["content"][-1] == {"type": "text", "text": "<system-reminder>\nplain note\n</system-reminder>"}
    assert msgs[5]["content"] == "out"  # input untouched


def test_history_system_after_assistant_becomes_user():
    out = ua.normalize_history([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "done"},
                                {"role": "system", "content": REM}, {"role": "system", "content": ""}])
    assert out[-1] == {"role": "user", "content": REM}
    assert len(out) == 3


def test_history_think_in_content():
    a1 = {"role": "assistant", "content": None, "reasoning_content": " list it first ",
          "thinking_blocks": [{"type": "thinking", "thinking": "list it first", "signature": "x"}], "tool_calls": [_call(1)]}
    a2 = {"role": "assistant", "content": [{"type": "text", "text": "Reading."}],
          "thinking_blocks": [{"type": "thinking", "thinking": "now read"}, {"type": "redacted_thinking", "data": "z"}]}
    a3 = {"role": "assistant", "content": "<think>\nkept\n</think>\n\nok", "reasoning_content": "other"}
    a4 = {"role": "assistant", "content": "plain"}
    out = ua.normalize_history([{"role": "user", "content": "go"}, a1, a2, a3, a4], think_in_content=True)
    assert out[1]["content"] == "<think>\nlist it first\n</think>\n\n"
    assert "reasoning_content" not in out[1] and "thinking_blocks" not in out[1] and out[1]["tool_calls"] == [_call(1)]
    assert out[2]["content"] == [{"type": "text", "text": "<think>\nnow read\n</think>\n\n"}, {"type": "text", "text": "Reading."}]
    assert out[3] is a3 and out[4] is a4
    assert ua.normalize_history([{"role": "user", "content": "go"}, a1])[1] is a1  # think_in_content off: left alone


def test_history_hook_only_for_local_tiers():
    hook = ua.UltronAdmit()
    msgs = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "reasoning_content": "r"},
            {"role": "system", "content": REM}]
    run = lambda model, base: asyncio.run(hook.async_pre_call_deployment_hook(
        {"model": model, "api_base": base, "messages": list(msgs)}, None))["messages"]
    local = "http://127.0.0.1:8001/v1"
    assert run("hosted_vllm/sonnet", local)[1]["content"].startswith("<think>\nr\n</think>")
    assert run("hosted_vllm/haiku", local)[1]["content"].startswith("<think>")
    assert run("hosted_vllm/opus", local)[1]["content"].startswith("<think>")  # Qwen3.6 reads <think> back too
    fable = run("hosted_vllm/fable", local)  # think_in_content = no: Qwen3.8 reads only reasoning_content
    assert fable[1]["content"] == "" and fable[2] == {"role": "user", "content": REM}
    assert run("openai/anthropic/claude-sonnet-5.5", "https://omniroute.example.com/v1") == msgs


def _unparsed(raw):
    return json.dumps({"__unparsedToolInput": {"raw": raw, "len": len(raw)}})


def _fn(i, name, args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": args}}


def test_repair_split_tool_calls():
    # What Claude Code sent back after LiteLLM split opus's parallel calls (a coding agent, 2026-10-01).
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                _fn(1, "Read", _unparsed('{"file_path":"/docs/fr')), _fn(2, "", _unparsed('ontend.md"}')),
                _fn(3, "Bash", _unparsed('{"command":"ls')), _fn(4, "", _unparsed('"}')),
                _fn(5, "Bash", '{"command": "pwd"}'), _fn(6, "", _unparsed("garbage"))]},
            {"role": "tool", "tool_call_id": "c1", "content": "<tool_use_error>InputValidationError</tool_use_error>"},
            {"role": "tool", "tool_call_id": "c2", "content": "<tool_use_error>Error: No such tool available: </tool_use_error>"},
            {"role": "tool", "tool_call_id": "c3", "content": "<tool_use_error>InputValidationError</tool_use_error>"},
            {"role": "tool", "tool_call_id": "c4", "content": "<tool_use_error>Error: No such tool available: </tool_use_error>\n\n" + REM},
            {"role": "tool", "tool_call_id": "c5", "content": "/x"},
            {"role": "tool", "tool_call_id": "c6", "content": "<tool_use_error>Error: No such tool available: </tool_use_error>"}]
    out = ua.repair_split_tool_calls(msgs)
    calls = out[1]["tool_calls"]
    assert [c["id"] for c in calls] == ["c1", "c3", "c5"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"file_path": "/docs/frontend.md"}
    assert json.loads(calls[1]["function"]["arguments"]) == {"command": "ls"}
    assert calls[2]["function"]["arguments"] == '{"command": "pwd"}'  # complete already; "garbage" can't join
    assert [m.get("tool_call_id") for m in out[2:]] == ["c1", "c3", "c5"]
    assert out[3]["content"] == "<tool_use_error>InputValidationError</tool_use_error>\n\n" + REM
    assert msgs[1]["tool_calls"][1]["id"] == "c2"  # input untouched


def test_repair_split_tool_calls_edge_cases():
    clean = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": [_call(1)]},
             {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
    assert ua.repair_split_tool_calls(clean) is clean
    lone = [{"role": "assistant", "content": "hi", "tool_calls": [_fn(1, None, "{}")]},
            {"role": "tool", "tool_call_id": "c1", "content": "x"}, {"role": "user", "content": "next"}]
    assert ua.repair_split_tool_calls(lone) == [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "next"}]


def test_drop_empty_tool_calls():
    # fable/Splash emits `arguments: "{}"`; the client's validation error must not become part of history
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [_fn(1, "bash", "{}")]},
            {"role": "tool", "tool_call_id": "c1", "content": 'Validation failed for tool "bash": - command: must have required properties command'},
            {"role": "assistant", "content": "", "tool_calls": [_fn(2, "bash", '{"command": "ls"}')]},
            {"role": "tool", "tool_call_id": "c2", "content": "ok"}]
    out = ua.drop_empty_tool_calls(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "tool"]
    assert out[1]["tool_calls"][0]["id"] == "c2" and out[2]["tool_call_id"] == "c2"
    assert msgs[1]["tool_calls"][0]["id"] == "c1"  # input untouched
    clean = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "", "tool_calls": [_fn(1, "bash", '{"command": "ls"}')]}]
    assert ua.drop_empty_tool_calls(clean) is clean  # nothing to drop: same list back (prefix cache)
    assert ua.drop_empty_tool_calls([{"role": "assistant", "content": "", "tool_calls": [_fn(1, "bash", "{}")]}]) == []
    # one empty call beside a real one: only the empty one goes
    mixed = [{"role": "user", "content": "go"},
             {"role": "assistant", "content": "", "tool_calls": [_fn(1, "bash", "{}"), _fn(2, "bash", '{"command": "pwd"}')]},
             {"role": "tool", "tool_call_id": "c1", "content": "e"},
             {"role": "tool", "tool_call_id": "c2", "content": "/x"}]
    out = ua.drop_empty_tool_calls(mixed)
    assert [c["id"] for c in out[1]["tool_calls"]] == ["c2"] and len(out) == 3
    # 2026-10-14: gate on the request's own schemas - a zero-argument tool is called with {} by design
    schemas = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
               {"type": "function", "function": {"name": "murmur_check_messages", "parameters": {"type": "object", "properties": {}}}}]
    assert [c["id"] for c in ua.drop_empty_tool_calls(mixed, schemas)[1]["tool_calls"]] == ["c2"]
    zero = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [_fn(1, "murmur_check_messages", "{}")]},
            {"role": "tool", "tool_call_id": "c1", "content": "no new messages"}]
    assert ua.drop_empty_tool_calls(zero, schemas) is zero
    unknown = [{"role": "assistant", "content": "", "tool_calls": [_fn(1, "mcp__thing", "{}")]}]
    assert ua.drop_empty_tool_calls(unknown, schemas) is unknown  # undeclared name: cannot tell, leave it

def test_deployment_hook_keeps_a_declared_zero_argument_call():
    hook = ua.UltronAdmit()
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [_fn(1, "murmur_check_messages", "{}")]},
            {"role": "tool", "tool_call_id": "c1", "content": "no new messages"}]
    tools = [{"type": "function", "function": {"name": "murmur_check_messages", "parameters": {"type": "object", "properties": {}}}}]
    kw = {"model": "hosted_vllm/sonnet", "api_base": "http://127.0.0.1:8001/v1", "messages": list(msgs), "tools": tools}
    out = asyncio.run(hook.async_pre_call_deployment_hook(kw, None))["messages"]
    assert [c["id"] for m in out if m.get("tool_calls") for c in m["tool_calls"]] == ["c1"]

def test_deployment_hook_drops_empty_tool_calls():
    hook = ua.UltronAdmit()
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [_fn(1, "bash", "{}")]},
            {"role": "tool", "tool_call_id": "c1", "content": "e"},
            {"role": "assistant", "content": "", "tool_calls": [_fn(2, "bash", '{"command": "ls"}')]},
            {"role": "tool", "tool_call_id": "c2", "content": "ok"}]
    for model, base in (("hosted_vllm/sonnet", "http://127.0.0.1:8001/v1"), ("hosted_vllm/fable", "http://127.0.0.1:8001/v1"),
                        ("openai/custom-sonnet", "https://omniroute.example.com/v1")):
        out = asyncio.run(hook.async_pre_call_deployment_hook({"model": model, "api_base": base, "messages": list(msgs)}, None))["messages"]
        assert [c["id"] for m in out if m.get("tool_calls") for c in m["tool_calls"]] == ["c2"], model
        assert [m.get("tool_call_id") for m in out if m.get("role") == "tool"] == ["c2"], model

def test_deployment_hook_repairs_split_calls_on_every_deployment():
    hook = ua.UltronAdmit()
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [_fn(1, "Bash", _unparsed('{"command":"ls')), _fn(2, "", _unparsed('"}'))]},
            {"role": "tool", "tool_call_id": "c1", "content": "e"}, {"role": "tool", "tool_call_id": "c2", "content": "e"}]
    for model, base in (("hosted_vllm/sonnet", "http://127.0.0.1:8001/v1"), ("hosted_vllm/fable", "http://127.0.0.1:8001/v1"),
                        ("openai/anthropic/claude-sonnet-5.5", "https://omniroute.example.com/v1")):
        out = asyncio.run(hook.async_pre_call_deployment_hook({"model": model, "api_base": base, "messages": list(msgs)}, None))["messages"]
        assert [c["id"] for c in out[1]["tool_calls"]] == ["c1"] and len(out) == 3, model


def test_blank_delta_patch_keeps_tool_arguments_in_one_block():
    pytest.importorskip("litellm.types.utils")   # the real LiteLLM, not this repo's litellm/ folder
    from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta, ChatCompletionDeltaToolCall, Function
    from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import AnthropicStreamWrapper
    assert ua.patch_blank_delta_blocks()

    def ch(delta, finish=None):
        return ModelResponseStream(id="x", model="opus", choices=[StreamingChoices(index=0, delta=Delta(**delta), finish_reason=finish)])

    def tc(i, args, id=None, name=None):
        return {"tool_calls": [ChatCompletionDeltaToolCall(index=i, id=id, type="function" if id else None,
                                                           function=Function(name=name, arguments=args))]}

    async def chunks():  # mtplx sends {} between tool-call chunks
        for c in (ch({"reasoning_content": "plan"}), ch({}), ch(tc(0, "", "call_a", "Read")), ch(tc(0, '{"file_path":"/fr')),
                  ch({}), ch(tc(0, 'ontend.md"}')), ch({}), ch(tc(1, "", "call_b", "Bash")), ch(tc(1, '{"command":"ls"}')),
                  ch({}, "tool_calls")):
            yield c

    async def run():
        return [ev async for ev in AnthropicStreamWrapper(completion_stream=chunks(), model="opus")]

    evs = asyncio.run(run())
    starts = [e["content_block"] for e in evs if e.get("type") == "content_block_start"]
    assert [(b["type"], b.get("name")) for b in starts] == [("thinking", None), ("tool_use", "Read"), ("tool_use", "Bash")]
    args = "".join(e["delta"]["partial_json"] for e in evs if e.get("type") == "content_block_delta" and e["index"] == 1)
    assert json.loads(args) == {"file_path": "/frontend.md"}


def test_history_folds_reminder_only_user_turn_after_tool():
    rem_list = [{"type": "text", "text": REM}, {"type": "text", "text": "\n" + REM + "\n"}]
    msgs = [{"role": "user", "content": [{"type": "text", "text": REM}, {"type": "text", "text": "fix it"}]},
            {"role": "assistant", "content": "", "tool_calls": [_call(1)]},
            {"role": "tool", "tool_call_id": "c1", "content": "out"}, {"role": "user", "content": rem_list},
            {"role": "assistant", "content": "", "tool_calls": [_call(2)]},
            {"role": "tool", "tool_call_id": "c2", "content": "out2"}, {"role": "user", "content": "now also " + REM},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": REM},
            {"role": "tool", "tool_call_id": "c3", "content": "img"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": REM}]}]
    out = ua.normalize_history(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "assistant", "tool", "user", "assistant", "user", "tool", "user"]
    assert out[2]["content"] == "out\n\n" + REM + "\n\n" + REM
    assert out[0] is msgs[0] and out[5] is msgs[6] and out[7] is msgs[8] and out[9] is msgs[10]


def test_trace_tap(tmp_path, monkeypatch):
    mode = tmp_path / "trace-mode"
    monkeypatch.setattr(ua, "TRACE_MODE_FILE", str(mode))
    monkeypatch.setattr(ua, "TRACE_DIR", str(tmp_path / "traces"))
    hook = ua.UltronAdmit()
    hook.admission.keys["cid-1"] = "cc:abc/def:main"
    tools = [{"type": "function", "function": {"name": "Bash"}}]
    msgs = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "reasoning_content": "r",
             "tool_calls": [_fn(1, "Bash", '{"command": "ls"}')]},
            {"role": "tool", "tool_call_id": "c1", "content": "out"}, {"role": "system", "content": REM}]
    run = lambda **kw: asyncio.run(hook.async_pre_call_deployment_hook(
        {"model": "hosted_vllm/sonnet", "api_base": "http://127.0.0.1:8001/v1", "messages": list(msgs), **kw}, None))
    run(tools=tools, litellm_call_id="cid-1")
    assert not (tmp_path / "traces").exists()  # off by default
    mode.write_text("on\n")
    out = run(tools=tools, litellm_call_id="cid-1")
    tr = json.loads((tmp_path / "traces" / "cc_abc_def_main.json").read_text())
    assert tr["messages"] == out["messages"] and tr["tools"] == tools and tr["tier"] == "sonnet"
    assert tr["messages"][2]["content"].endswith(REM)  # what the tier sees, after normalize_history
    run(tools=tools)  # no admission key -> hashed fallback
    run()             # no tools -> no trace
    assert len(list((tmp_path / "traces").iterdir())) == 2


def test_fill_array_items():
    arr = {"type": "array"}
    tools = [{"name": "A", "input_schema": {"type": "object", "properties": {
                 "a": arr, "b": {"type": "array", "items": {"type": "string"}},
                 "c": {"anyOf": [{"type": ["array", "null"]}, {"type": "null"}]},
                 "d": {"type": "object", "properties": {"e": {"type": "array"}}}}}},
             {"type": "function", "function": {"name": "B", "parameters": {"type": "object", "properties": {"f": {"type": "array"}}}}}]
    assert ua.fill_array_items(tools) == 4
    p = tools[0]["input_schema"]["properties"]
    assert p["a"] == {"type": "array", "items": {}} and p["b"]["items"] == {"type": "string"}
    assert p["c"]["anyOf"][0]["items"] == {} and p["d"]["properties"]["e"]["items"] == {}
    assert tools[1]["function"]["parameters"]["properties"]["f"]["items"] == {}
    assert ua.fill_array_items(tools) == 0 and ua.fill_array_items(None) == 0


def test_decisions_pass_through_gets_the_default_model_and_nothing_else(monkeypatch):
    monkeypatch.setattr(ua, "admit_mode", lambda: "enforce")
    monkeypatch.setattr(ua, "route_mode", lambda: "local-only")   # would set disable_fallbacks on a chat call
    monkeypatch.setattr(ua, "DECISIONS_MODEL", "vendor/model-1")
    hook = ua.UltronAdmit()
    body = {"state": "ticket text", "questions": {"team": {"type": "choice", "instructions": "Who handles it?"}}}
    out = asyncio.run(hook.async_pre_call_hook(None, None, body, "pass_through_endpoint"))
    assert out == {"model": "vendor/model-1", "state": "ticket text", "questions": body["questions"]}  # the body sent upstream
    own = {"model": "vendor/other", "questions": {}}
    asyncio.run(hook.async_pre_call_hook(None, None, own, "pass_through_endpoint"))
    assert own["model"] == "vendor/other"
    chat = {"model": "claude-opus-5-5", "questions": {}}   # only pass-throughs are touched
    assert ua.default_decisions_model(chat, "acompletion") is False and chat["model"] == "claude-opus-5-5"
    assert ua.default_decisions_model({"kind": "noul", "question": "q"}, "pass_through_endpoint") is False


def test_pre_call_hook_fills_items_even_when_admit_off(monkeypatch):
    monkeypatch.setattr(ua, "admit_mode", lambda: "off")
    data = {"model": "claude-opus-5-5", "tools": [{"name": "T", "input_schema": {"type": "object", "properties": {"x": {"type": "array"}}}}]}
    out = asyncio.run(ua.UltronAdmit().async_pre_call_hook(None, None, data, "anthropic_messages"))
    assert out["tools"][0]["input_schema"]["properties"]["x"]["items"] == {}


def test_example_matrix_fable_swaps_only_opus():
    t = ua.tiers()
    sets = ua._expand(t.resident, {}, {})
    costs = {n: t.tier[n]["evict_cost"] for n in t.names}
    assert set(sets) == {frozenset({"opus", "sonnet", "haiku"}), frozenset({"fable", "sonnet", "haiku"}),
                         frozenset({"judge", "sonnet", "haiku"})}
    assert all({"sonnet", "haiku"} <= s for s in sets)  # always resident
    assert ua.evictees("fable", {"opus", "sonnet", "haiku"}, sets, costs) == {"opus"}
    assert ua.evictees("opus", {"fable", "sonnet", "haiku"}, sets, costs) == {"fable"}
    assert ua.evictees("sonnet", {"fable", "haiku"}, sets, costs) == set()
    assert ua.evictees("opus", {"sonnet", "haiku"}, sets, costs) == set()


# ----------------------------------------------------------------------------- memory gate (local only)

def _mem_gate(adm, monkeypatch, tmp_path, busy, mem_tight=None):
    """local-only; opus/sonnet/haiku loaded, each `busy[tier]` requests active."""
    (tmp_path / "route-mode").write_text("local-only\n")
    real_sleep = asyncio.sleep
    monkeypatch.setattr(ua.asyncio, "sleep", lambda s: real_sleep(0))
    polls = []

    async def state():
        polls.append(dict(busy))
        return st(mem_tight=mem_tight, **{t: (n, 0) for t, n in busy.items()})
    adm.state = state
    return polls, real_sleep


def test_mem_gate_holds_a_local_only_request_while_another_tier_serves(adm, monkeypatch, tmp_path):
    """Even with memory fine: one 100k prefill alone spikes ~12 GB of the ~16 GB idle headroom."""
    busy = {"opus": 0, "sonnet": 1, "haiku": 0}
    polls, real_sleep = _mem_gate(adm, monkeypatch, tmp_path, busy)

    async def scenario():
        task = asyncio.create_task(adm.admit(req(cid="c1"), "anthropic_messages", "enforce"))
        for _ in range(10):
            await real_sleep(0)
        assert not task.done()  # opus waits: sonnet is mid-request
        busy["sonnet"] = 0
        return await task
    d = run(scenario())
    assert d["target"] == "ultron/opus" and d["mem_wait"]["gave_up_on"] is None and len(polls) > 3
    assert not adm.mem_queue and adm.mem_sent.keys() == {"opus"}


def test_mem_gate_first_come_first_served(adm, monkeypatch, tmp_path):
    """sonnet queued behind a busy opus; a later opus request waits behind sonnet's turn."""
    busy = {"opus": 1, "sonnet": 0, "haiku": 0}
    polls, real_sleep = _mem_gate(adm, monkeypatch, tmp_path, busy)

    async def spin():
        for _ in range(10):
            await real_sleep(0)

    async def scenario():
        a = asyncio.create_task(adm.admit(req(model="claude-sonnet-5-5", session="S2", cid="a"), "anthropic_messages", "enforce"))
        await spin()
        c = asyncio.create_task(adm.admit(req(session="S3", cid="c"), "anthropic_messages", "enforce"))
        await spin()
        assert not a.done() and not c.done() and list(adm.mem_queue.values()) == ["sonnet", "opus"]
        busy["opus"] = 0  # opus finishes: sonnet goes first
        await spin()
        assert a.done() and not c.done()  # just sent: counts as busy until its backend shows it
        busy["sonnet"] = 1
        adm.mem_sent.clear()
        await spin()
        assert not c.done()
        busy["sonnet"] = 0
        return await a, await c
    da, dc = run(scenario())
    assert (da["target"], dc["target"]) == ("ultron/sonnet", "ultron/opus")


def test_mem_gate_skips_haiku_and_cloud_allowed_requests(adm, monkeypatch, tmp_path):
    busy = {"opus": 0, "sonnet": 1, "haiku": 1}
    _mem_gate(adm, monkeypatch, tmp_path, busy, mem_tight="headroom 2.0G")
    d = run(adm.admit(req(model="claude-haiku-4-5", cid="h"), "anthropic_messages", "enforce"))
    assert d["target"] == "ultron/haiku" and d["mem_wait"] is None  # haiku never waits
    busy["sonnet"] = 0
    d = run(adm.admit(req(cid="o"), "anthropic_messages", "enforce"))
    assert d["mem_wait"]["s"] < 1  # a busy haiku doesn't hold opus
    busy["sonnet"] = 1
    (tmp_path / "route-mode").write_text("auto\n")  # cloud allowed: no gate (the memory guard overflows instead)
    d = run(adm.admit(req(session="S4", cid="o2"), "anthropic_messages", "enforce"))
    assert d["target"] == "cloud/opus" and d["rule"] == "4:overflow:mem" and d["mem_wait"] is None
    busy["sonnet"] = 0
    d = run(adm.admit(req(session="S5", cid="o3", x_route="private"), "anthropic_messages", "enforce"))
    assert d["target"] == "ultron/opus" and d["mem_wait"] is not None  # private stays local, so it goes through the gate


def test_mem_gate_gives_up_after_the_limit(adm, monkeypatch, tmp_path):
    _mem_gate(adm, monkeypatch, tmp_path, {"opus": 0, "sonnet": 1, "haiku": 0})
    monkeypatch.setattr(ua, "MEM_WAIT_MAX_S", 0.01)
    d = run(adm.admit(req(cid="c1"), "anthropic_messages", "enforce"))
    assert d["target"] == "ultron/opus" and d["mem_wait"]["gave_up_on"] == ["sonnet"]


def test_mem_gate_waits_out_a_cold_load(adm, monkeypatch, tmp_path):
    """A tier loading for a request the gate let through shows 0 active: still busy (2026-10-01 live
    test: opus went after 3 s while sonnet was still loading, and both prefilled together)."""
    _, real_sleep = _mem_gate(adm, monkeypatch, tmp_path, {})
    loading = st(opus=(0, 0), haiku=(0, 0), sonnet="starting")

    async def scenario():
        adm.mem_sent["sonnet"] = 0  # let through long ago
        async def state():
            return loading
        adm.state = state
        task = asyncio.create_task(adm.admit(req(cid="c1"), "anthropic_messages", "enforce"))
        for _ in range(10):
            await real_sleep(0)
        assert not task.done()
        loading["tiers"]["sonnet"] = {"state": "ready", "active": 0, "waiting": 0}  # up, request not in yet
        for _ in range(10):
            await real_sleep(0)
        assert not task.done()  # MEM_SEND_S grace after the load
        adm.mem_sent.clear()
        return await task
    d = run(scenario())
    assert d["target"] == "ultron/opus" and d["mem_wait"]["gave_up_on"] is None


def test_upstream_model_renders_use_model_name():
    import ultron_tiers
    t = ultron_tiers.Tiers("[routing]\n[big]\nupstream_model = default_model\n[small]\n")
    assert not t.problems
    sw = ultron_tiers.blocks(t)["__TIERS_SWAP_MODELS__"]
    big, small = sw.split("  small:")
    assert 'useModelName: "default_model"' in big and "useModelName" not in small


def test_example_haiku_is_text_only():
    t = ua.tiers()
    assert "haiku" in t.no_vision() and t.vision_tier == "sonnet"


def test_routed_no_rules():
    import ultron_tiers
    t = ultron_tiers.Tiers("[routing]\n[big]\n[judge]\nrouted = no\nupstream_model = ~/m/judge\n[small]\n")
    assert not t.problems and t.routed() == ["big", "small"]
    assert t.default == "big" and t.helper == "small" and t.vision_tier == "big"  # defaults skip routed = no
    assert t.tier_for("ultron/judge") == "judge" and t.tier_for("judge") == "judge" and t.tier_for("x") == "big"
    sw = ultron_tiers.blocks(t, home="/h")["__TIERS_SWAP_MODELS__"]
    assert 'useModelName: "/h/m/judge"' in sw  # ~/ is the home directory
    lm = ultron_tiers.blocks(t)["__TIERS_MODEL_LIST__"]
    assert '"ultron/judge"' in lm and "cloud/judge" not in lm
    bad = ultron_tiers.Tiers("[routing]\nhelper = j\n[big]\nsubstitute = j\n[j]\nrouted = no\ncloud = x\nmatch = j-*\n")
    assert set(bad.problems) == {"[routing] helper = j: that tier has routed = no",
                                 "[big] substitute = j: that tier has routed = no",
                                 "[j] match: not allowed with routed = no", "[j] cloud: not allowed with routed = no"}


def test_cloud_entries_cache_and_never_fall_back():
    import ultron_tiers
    b = ultron_tiers.blocks(ua.tiers())
    fb = b["__TIERS_FALLBACKS__"]  # local -> cloud only, one entry per tier with a cloud twin
    assert not any('"cloud/' in line.split(":")[0] for line in fb.splitlines())
    assert fb.count("[") == len([n for n in ua.tiers().names if ua.tiers().has_cloud(n)])
    assert b["__TIERS_MODEL_LIST__"].count("cache_control_injection_points") == len(
        [n for n in ua.tiers().names if ua.tiers().has_cloud(n)])


def test_chat_bridge_is_the_cloud_endpoint_only(monkeypatch):
    monkeypatch.setenv("OMNIROUTE_BASE", "https://omniroute.example.com:20128/v1")
    assert ua.chat_bridge("https://omniroute.example.com:20128/v1")
    for base in ("https://api.openai.com/v1", "http://127.0.0.1:8001/v1", "https://omniroute.example.com:8443/v1", None):
        assert not ua.chat_bridge(base), base
    monkeypatch.setenv("ULTRON_CHAT_BRIDGE", "off")
    assert not ua.chat_bridge("https://omniroute.example.com:20128/v1")
    monkeypatch.setenv("OMNIROUTE_BASE", "")
    monkeypatch.delenv("ULTRON_CHAT_BRIDGE")
    assert not ua.chat_bridge("")


def test_cloud_messages_take_chat_bridge(monkeypatch):
    pytest.importorskip("litellm.llms")   # the real LiteLLM, not this repo's litellm/ folder
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        LiteLLMMessagesToCompletionTransformationHandler as ChatBridge,
    )
    from litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler import (
        LiteLLMMessagesToResponsesAPIHandler as ResponsesBridge,
    )
    monkeypatch.setenv("OMNIROUTE_BASE", "https://omniroute.example.com:20128/v1")
    assert ua.patch_omniroute_chat_bridge() and ua.patch_omniroute_chat_bridge()  # idempotent
    calls = []
    monkeypatch.setattr(ChatBridge, "anthropic_messages_handler", staticmethod(lambda **kw: calls.append(kw) or "chat"))
    args = dict(max_tokens=8, messages=[{"role": "user", "content": "hi"}], model="openai/anthropic/claude-sonnet-5.5", _is_async=True)
    assert ResponsesBridge.anthropic_messages_handler(api_base="https://omniroute.example.com:20128/v1", **args) == "chat"
    assert calls and calls[0]["api_base"].startswith("https://omniroute")
    other = ResponsesBridge.anthropic_messages_handler(api_base="https://api.openai.com/v1", **args)  # coroutine, not run
    assert asyncio.iscoroutine(other) and len(calls) == 1
    other.close()


def test_docs_to_text(monkeypatch):
    import base64
    seen = []
    monkeypatch.setattr(ua, "pdf_text", lambda data: seen.append(data) or "--- page 1 ---\nHello\n\n--- page 2 ---\nWorld")
    monkeypatch.setattr(ua, "_doc_cache", ua.OrderedDict())
    pdf = "data:application/pdf;base64," + base64.b64encode(b"%PDF-1.7 x").decode()
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    msgs = [{"role": "user", "content": "summarize the pdf"},
            {"role": "assistant", "content": "", "tool_calls": [_fn(1, "Bash", '{"command": "cat doc.pdf"}')]},
            {"role": "tool", "tool_call_id": "c1", "content": [{"type": "image_url", "image_url": {"url": pdf}}]},
            {"role": "user", "content": [img, {"type": "file", "file": {"file_data": "data:text/plain;base64,"
                                                                       + base64.b64encode(b"notes").decode()}},
                                         {"type": "file", "file": {"file_data": "data:application/zip;base64,UEs="}}]}]
    hook = ua.UltronAdmit()
    out = asyncio.run(hook.async_pre_call_deployment_hook(
        {"model": "hosted_vllm/opus", "api_base": "http://127.0.0.1:8001/v1", "messages": msgs}, None))["messages"]
    assert seen == [b"%PDF-1.7 x"]
    assert out[2]["content"][0]["text"].startswith("[application/pdf attachment, 2 page(s), converted to text]\n--- page 1 ---\nHello")
    assert out[3]["content"][0] is img  # images stay for the vision tiers
    assert out[3]["content"][1]["text"] == "[text/plain attachment]\nnotes"
    assert "application/zip attachment omitted" in out[3]["content"][2]["text"]
    assert out[0] is msgs[0] and msgs[2]["content"][0]["type"] == "image_url"  # input untouched
    ua.docs_to_text(msgs)
    assert len(seen) == 1  # cached: the PDF rides along in every later request of the conversation
    monkeypatch.setattr(ua, "pdf_text", lambda data: "--- page 1 ---\n\n")
    monkeypatch.setattr(ua, "_doc_cache", ua.OrderedDict())
    assert "no text layer" in ua.docs_to_text(msgs)[2]["content"][0]["text"]
    cloud = {"model": "openai/anthropic/claude-opus-5.5", "api_base": "https://omniroute.example.com/v1", "messages": msgs}
    assert asyncio.run(hook.async_pre_call_deployment_hook(cloud, None))["messages"][2] is msgs[2]  # cloud: untouched


# ----------------------------------------------------------------------------- routed = no (the image judge)

JUDGE_SETS = ua._expand("(opus | judge) & sonnet & haiku", {}, {})
JUDGE_COSTS = {"opus": 3, "sonnet": 5, "haiku": 5, "judge": 1}


def test_judge_waits_for_opus_then_goes(adm, monkeypatch):
    """Loading the judge unloads opus (sonnet stays): it waits out opus's request like a local-only
    reload, and is never routed anywhere else (no cloud twin, no pin)."""
    monkeypatch.setattr(ua, "matrix", lambda: (JUDGE_SETS, JUDGE_COSTS))
    real_sleep = asyncio.sleep
    monkeypatch.setattr(ua.asyncio, "sleep", lambda s: real_sleep(0))
    polls = iter([st(opus=(1, 0), sonnet=(0, 0), haiku=(0, 0))] * 3 + [st(opus=(0, 0), sonnet=(0, 0), haiku=(0, 0))] * 10**3)

    async def state():
        return next(polls)
    adm.state = state
    d = req(model="judge", cid="j1")
    out = run(adm.admit(d, "acompletion", "enforce"))
    assert d["model"] == "ultron/judge" and "extra_headers" not in d  # a bare name -> the canonical id
    assert out["rule"] == "direct" and out["requested"] == "judge"
    assert out["busy_evictees_after_wait"] is None and out["mem_wait"]["gave_up_on"] is None
    assert next(polls)["tiers"]["opus"]["active"] == 0  # it polled past the busy reads
    assert adm.recent["j1"] == "ultron/judge; rule=direct"


def test_judge_gives_up_on_a_busy_opus(adm, monkeypatch):
    monkeypatch.setattr(ua, "matrix", lambda: (JUDGE_SETS, JUDGE_COSTS))
    monkeypatch.setattr(ua, "EVICT_WAIT_MAX_S", 0)
    monkeypatch.setattr(ua, "MEM_WAIT_MAX_S", 0)
    adm.fake = st(opus=(1, 0), sonnet=(0, 0), haiku=(0, 0))
    out = run(adm.admit(req(model="ultron/judge", cid="j1"), "acompletion", "enforce"))
    assert out["busy_evictees_after_wait"] == ["opus"] and adm.recent["j1"].endswith("evicted-busy=opus")


def test_judge_shadow_mode_does_not_wait(adm, monkeypatch):
    monkeypatch.setattr(ua, "matrix", lambda: (JUDGE_SETS, JUDGE_COSTS))
    adm.fake = st(opus=(1, 0), sonnet=(0, 0), haiku=(0, 0))
    out = run(adm.admit(req(model="ultron/judge", cid="j1"), "acompletion", "shadow"))
    assert out["rule"] == "direct" and out["mem_wait"] is None and out["busy_evictees_after_wait"] is None


def test_loaded_judge_never_blocks_a_fit(adm, monkeypatch):
    """A new opus conversation while the judge holds opus's place: opus cold-fits (llama-swap
    unloads the judge), it doesn't overflow to cloud."""
    monkeypatch.setattr(ua, "matrix", lambda: (JUDGE_SETS, JUDGE_COSTS))
    adm.fake = st(sonnet=(0, 0), haiku=(0, 0), judge=(0, 0))
    d = req(model="claude-opus-5-5", session="S2", cid="o1")
    out = run(adm.admit(d, "anthropic_messages", "enforce"))
    assert d["model"] == "ultron/opus" and out["rule"] == "2:cold-fits"


def test_opus_reload_waits_for_a_busy_judge(adm, monkeypatch):
    monkeypatch.setattr(ua, "matrix", lambda: (JUDGE_SETS, JUDGE_COSTS))
    monkeypatch.setattr(ua, "OVERFLOW_WAIT_S", 0)
    adm.fake = st(sonnet=(0, 0), haiku=(0, 0), judge=(1, 0))  # a ranking request is mid-flight
    d = req(model="claude-opus-5-5", session="S2", cid="o1")
    run(adm.admit(d, "anthropic_messages", "enforce"))
    assert d["model"] == "cloud/opus" and adm.recent["o1"].endswith("overflow=evict-busy")


def test_one_shot_local_requests_skip_the_mtplx_bank():
    hook = ua.UltronAdmit()
    local = "http://127.0.0.1:8001/v1"
    tools = [{"type": "function", "function": {"name": "Bash"}}]
    first = [{"role": "user", "content": "title this"}]
    later = first + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]

    def run(model="hosted_vllm/haiku", base=local, msgs=first, cid=None, **kw):
        out = asyncio.run(hook.async_pre_call_deployment_hook(
            {"model": model, "api_base": base, "messages": list(msgs), "litellm_call_id": cid, **kw}, None))
        return (out.get("extra_headers") or {}).get("x-mtplx-cache-mode")

    assert run() == "bypass"                                      # single-turn, no tools
    assert run(model="hosted_vllm/sonnet") == "bypass"            # every local tier
    assert run(tools=tools) is None                               # an agent's first turn will be continued
    assert run(msgs=later) is None                                # a conversation already under way
    assert run(model="openai/custom-haiku", base="https://omniroute.example.com/v1") is None
    hook.admission.recent["c1"] = "ultron/haiku; rule=pinned:1:fits; applied; vision->haiku"
    assert run(msgs=later, tools=tools, cid="c1") == "bypass"     # image rerouted off fable: one request on haiku
    assert hook.admission.recent["c1"].endswith("; bank=off(vision)")
    assert run(extra_headers={"X-Other": "1"}) == "bypass"        # merges with headers already set


def test_one_shot():
    assert ua.one_shot([{"role": "user", "content": "x"}], None) == "single-turn"
    assert ua.one_shot([{"role": "user", "content": "x"}], [{"name": "Bash"}]) is None
    assert ua.one_shot([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}], None) is None
    assert ua.one_shot([], [{"name": "Bash"}], "ultron/haiku; vision->haiku") == "vision"


def test_reasoning_effort_is_echoed_for_mtplx_tiers_and_max_becomes_xhigh(monkeypatch):
    monkeypatch.setattr(ua, "admit_mode", lambda: "off")
    hook = ua.UltronAdmit()
    local, cloud = "http://127.0.0.1:8001/v1", "https://omniroute.example.com/v1"
    for cid, model, base, effort, want in (("c1", "hosted_vllm/opus", local, "max", "; effort=xhigh/32768"),
                                           ("c2", "hosted_vllm/sonnet", local, "none", "; effort=none/1"),
                                           ("c3", "hosted_vllm/haiku", local, "bogus", ""),
                                           ("c4", "hosted_vllm/fable", local, "high", "; effort=high/16384"),
                                           ("c5", "openai/custom-opus", cloud, "high", "")):
        data = {"model": "ultron/x", "messages": [{"role": "user", "content": "go"}], "litellm_call_id": cid,
                "reasoning_effort": effort}
        asyncio.run(hook.async_pre_call_hook(None, None, data, "acompletion"))
        assert data["reasoning_effort"] == ("xhigh" if effort == "max" else effort)
        hook.admission.recent[cid] = "pinned"
        kwargs = {"model": model, "api_base": base, "litellm_call_id": cid, "reasoning_effort": data["reasoning_effort"],
                  "messages": [{"role": "user", "content": "go"}], "tools": [{"type": "function", "function": {"name": "Bash"}}]}
        for _ in range(2):  # the /v1/messages bridge can run the hook twice
            asyncio.run(hook.async_pre_call_deployment_hook(dict(kwargs), None))
        assert re.sub(r"; sess=[^;]*", "", hook.admission.recent[cid]) == "pinned" + want, cid  # sess: its own test


def test_fable_gets_thinking_budget_from_reasoning_effort_on_tensorfold(monkeypatch):
    monkeypatch.setattr(ua, "admit_mode", lambda: "off")
    monkeypatch.setattr(ua, "BUDGET_TIERS", ("fable",))  # rollback config: TensorFold takes thinking_budget
    hook = ua.UltronAdmit()
    local = "http://127.0.0.1:8001/v1"
    for model, effort, want in (("hosted_vllm/fable", "low", 2048), ("hosted_vllm/fable", "none", 1),
                                ("hosted_vllm/fable", "bogus", None), ("hosted_vllm/opus", "low", None)):
        kwargs = {"model": model, "api_base": local, "litellm_call_id": "b1", "reasoning_effort": effort,
                  "messages": [{"role": "user", "content": "go"}], "extra_body": {"x": 1}}
        hook.admission.recent["b1"] = "pinned"
        out = asyncio.run(hook.async_pre_call_deployment_hook(kwargs, None))
        assert (out or kwargs).get("extra_body", {}).get("thinking_budget") == want, (model, effort)
        assert (out or kwargs)["extra_body"]["x"] == 1  # the caller's own extra_body survives


def test_fable_on_splash_gets_no_budget_but_sampling_defaults(monkeypatch):
    monkeypatch.setattr(ua, "admit_mode", lambda: "off")
    hook = ua.UltronAdmit()
    local = "http://127.0.0.1:8001/v1"
    base = {"api_base": local, "litellm_call_id": "s1", "messages": [{"role": "user", "content": "go"}]}
    hook.admission.recent["s1"] = "pinned"
    out = asyncio.run(hook.async_pre_call_deployment_hook({**base, "model": "hosted_vllm/fable", "reasoning_effort": "high"}, None))
    assert "thinking_budget" not in (out.get("extra_body") or {})  # Splash reads reasoning_effort natively
    assert out["temperature"] == 0.6 and out["max_tokens"] == 32768 and out["repetition_penalty"] == 1.1
    assert "effort=high/16384" in hook.admission.recent["s1"]
    out = asyncio.run(hook.async_pre_call_deployment_hook(
        {**base, "model": "hosted_vllm/fable", "temperature": 1.0, "max_completion_tokens": 100}, None))
    assert out["temperature"] == 1.0 and "max_tokens" not in out  # the client's own values win
    out = asyncio.run(hook.async_pre_call_deployment_hook({**base, "model": "hosted_vllm/opus"}, None))
    assert "temperature" not in out and "max_tokens" not in out and "repetition_penalty" not in out  # other tiers keep their own server defaults


def test_history_reasoning_content_for_fable():
    a1 = {"role": "assistant", "content": None, "reasoning_content": " list it first ",
          "thinking_blocks": [{"type": "thinking", "thinking": "list it first"}], "tool_calls": [_call(1)]}
    a2 = {"role": "assistant", "content": [{"type": "text", "text": "Reading."}],
          "thinking_blocks": [{"type": "thinking", "thinking": "now read"}]}
    a3 = {"role": "assistant", "content": "plain"}
    out = ua.normalize_history([{"role": "user", "content": "go"}, a1, a2, a3], reasoning_content=True)
    assert out[1]["reasoning_content"] == "list it first" and "thinking_blocks" not in out[1]
    assert out[1]["content"] is None and out[1]["tool_calls"] == [_call(1)]  # content untouched: no <think> block
    assert out[2]["reasoning_content"] == "now read" and out[2]["content"] == a2["content"]
    assert out[3] is a3


def test_hosted_vllm_keeps_reasoning_content_only_for_fable():
    pytest.importorskip("litellm")
    from litellm.llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig
    assert ua.HOSTED_VLLM_REASONING_PATCHED
    cfg = HostedVLLMChatConfig()
    msgs = lambda: [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "x", "reasoning_content": "R",
                     "thinking_blocks": [{"type": "thinking", "thinking": "R"}]}]
    out = cfg._transform_messages(msgs(), "fable")
    assert out[1]["reasoning_content"] == "R" and "thinking_blocks" not in out[1]
    assert "reasoning_content" not in cfg._transform_messages(msgs(), "opus")[1]
    assert asyncio.run(cfg._transform_messages(msgs(), "fable", is_async=True))[1]["reasoning_content"] == "R"


def test_mtplx_tiers_get_a_named_session_per_conversation(monkeypatch):
    inflight = {}
    monkeypatch.setattr(ua, "litellm_inflight", lambda: inflight)
    hook = ua.UltronAdmit()
    local = "http://127.0.0.1:8001/v1"
    tools = [{"type": "function", "function": {"name": "Bash"}}]
    turn = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]

    def headers(cid, model="hosted_vllm/sonnet", msgs=turn, **kw):
        out = asyncio.run(hook.async_pre_call_deployment_hook(
            {"model": model, "api_base": local, "litellm_call_id": cid, "messages": list(msgs), "tools": tools, **kw}, None))
        return out.get("extra_headers") or {}

    hook.admission.keys.update({"c1": "h:8b07a98c:sonnet", "c2": "h:8b07a98c:sonnet", "c3": "h:ffff0000:sonnet"})
    hook.admission.recent.update({"c1": "pinned", "c2": "pinned", "c3": "pinned"})
    assert headers("c1") == {"x-mtplx-request-id": "c1", "x-mtplx-session-id": "h-8b07a98c-sonnet"}
    assert headers("c1")["x-mtplx-session-id"] == "h-8b07a98c-sonnet"  # the /v1/messages bridge's second pass
    assert hook.admission.recent["c1"] == "pinned; sess=h-8b07a98c-sonnet"  # echoed once in x-ultron-route
    inflight["sonnet"] = ["c1"]  # c1 still running: mtplx would 409 a second request on the session
    assert headers("c2") == {"x-mtplx-request-id": "c2"} and hook.admission.recent["c2"] == "pinned; sess=anon(busy)"
    assert headers("c3")["x-mtplx-session-id"] == "h-ffff0000-sonnet"  # another conversation is unaffected
    inflight["sonnet"] = []
    assert headers("c2")["x-mtplx-session-id"] == "h-8b07a98c-sonnet"  # c1 done: the next turn takes it
    assert "x-mtplx-session-id" not in headers("c4", msgs=turn[:1], tools=None)  # one-shot: bank bypass, no session
    assert "x-mtplx-session-id" in headers("c5", model="hosted_vllm/opus")  # no admission key: hashed fallback
    assert headers("c6", model="hosted_vllm/fable") == {} and headers("c7", model="hosted_vllm/judge") == {}


def test_admission_rows_carry_the_harness_call_id_and_memory(adm):
    adm.fake = st(sonnet=(0, 0), haiku=(0, 0))
    adm.fake["mem"] = {"headroom": 8 * 2**30, "swap_used": 5 * 2**30, "compressor": 6 * 2**30}
    d = req(model="claude-sonnet-4-6", session=None, cid="h1")
    d["proxy_server_request"]["headers"]["X-Harness-Call-Id"] = "72bb2dc9"
    d["reasoning_effort"] = "low"
    out = run(adm.admit(d, "acompletion", "enforce"))
    assert out["harness_call_id"] == "72bb2dc9" and out["effort"] == "low" and out["admit_s"] >= 0
    assert (out["headroom_gb"], out["swap_gb"], out["compressor_gb"]) == (8.0, 5.0, 6.0)
    refused = adm.note_refusal(d, Exception("507: insufficient memory for prefill"))
    assert refused["harness_call_id"] == "72bb2dc9"


def test_x_ultron_injected_names_the_stages_that_wrote_the_reply():
    hook = ua.UltronAdmit()
    headers = lambda data, response: asyncio.run(hook.async_post_call_response_headers_hook(data, None, response))
    chat = lambda text: {"choices": [{"message": {"role": "assistant", "content": text}}]}
    assert headers({"litellm_call_id": "c0"}, chat("All done, tests pass.")) is None
    for text in ("[Compressed m00002–m00116 → 1 block(s), ~67962 tokens saved.]\nNo compressible ranges remain",
                 "[Compression FAILED: the call's arguments (9594 chars) were not parseable JSON",
                 "📦 [ACP] folded 3 ranges", "m00001–m00001\nThe quick brown fox",
                 "[acp-proxy: compress call had no valid ranges; nothing compressed]"):
        assert headers({"litellm_call_id": "c1"}, chat(text)) is None, text
    anthropic = {"content": [{"type": "text", "text": "❌ [ACP] compress failed"}], "stop_reason": "end_turn"}
    assert headers({}, anthropic) is None
    for cid, mock, stage in (("c2", "[ultron loop-breaker] Stopped this turn: …", "loop-breaker"),
                             ("c3", "Here's your image: /media/x.png", "media")):
        data = {"model": "ultron/sonnet", "messages": [{"role": "user", "content": "go"}], "litellm_call_id": cid,
                "mock_response": mock}
        asyncio.run(hook.async_pre_call_hook(None, None, data, "acompletion"))
        hook.admission.recent[cid] = "x"
        assert headers(data, chat(mock)) == {"x-ultron-route": "x", "x-ultron-injected": stage}
    assert headers({"ultron_rescued": True}, chat("")) == {"x-ultron-injected": "rescue"}
    assert headers({}, None) is None  # streaming: no body to look at


def test_explicit_cloud_request_keeps_its_conversation_key():
    # admission's conversation key is stable; a client that asks for cloud/<tier> itself gets one too
    hook = ua.UltronAdmit()
    data = {"model": "cloud/sonnet", "messages": [{"role": "user", "content": "go"}], "litellm_call_id": "cid-9"}
    assert asyncio.run(hook.admission.admit(data, "acompletion", "shadow")) is None
    assert hook.admission.keys["cid-9"] == ua.pin_identity(data, "sonnet")["key"]


def test_mcp_tool_call_skips_admission(monkeypatch):
    # LiteLLM's /mcp runs pre-call hooks on a synthetic request (model mcp-tool-call); tier_for() says sonnet
    hook = ua.UltronAdmit()
    called = []
    async def admit(*a, **k):
        called.append(a)
    monkeypatch.setattr(hook.admission, "admit", admit)
    monkeypatch.setattr(ua, "admit_mode", lambda: "enforce")
    data = {"model": "mcp-tool-call", "messages": [{"role": "user", "content": "Tool: search\nArguments: {}"}],
            "mcp_tool_name": "search"}
    assert asyncio.run(hook.async_pre_call_hook(None, None, dict(data), "call_mcp_tool")) == data
    assert called == []
    asyncio.run(hook.async_pre_call_hook(None, None, dict(data), "acompletion"))
    assert len(called) == 1


def test_sonnet_effort_is_capped_at_high(monkeypatch):
    monkeypatch.setattr(ua, "admit_mode", lambda: "off")
    hook = ua.UltronAdmit()
    local = "http://127.0.0.1:8001/v1"
    for cid, model, effort, want in (("k1", "hosted_vllm/sonnet", "xhigh", "high"), ("k2", "hosted_vllm/sonnet", "high", "high"),
                                     ("k3", "hosted_vllm/sonnet", "medium", "medium"), ("k4", "hosted_vllm/opus", "xhigh", "xhigh")):
        hook.admission.recent[cid] = "pinned"
        kwargs = {"model": model, "api_base": local, "litellm_call_id": cid, "reasoning_effort": effort,
                  "messages": [{"role": "user", "content": "go"}]}
        out = asyncio.run(hook.async_pre_call_deployment_hook(kwargs, None))
        assert (out or kwargs)["reasoning_effort"] == want, cid
        assert f"; effort={want}/" in hook.admission.recent[cid], cid
