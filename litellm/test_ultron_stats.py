"""ultron_stats: chunk shapes from both stream paths, the request lifecycle, and endpoint mapping."""

import asyncio
import json
from types import SimpleNamespace as NS

import ultron_stats as us

CONFIG = """
model_list:
  - model_name: "ultron/opus"
    litellm_params: {model: "openai/opus", api_base: "http://127.0.0.1:8001/v1"}
  - model_name: "claude-opus-*"
    litellm_params: {model: "openai/opus", api_base: "http://127.0.0.1:8001/v1"}
  - model_name: "claude-*"
    litellm_params: {model: "openai/sonnet", api_base: "http://127.0.0.1:8001/v1"}
  - model_name: "tf-opus"
    litellm_params: {model: "openai/opus", api_base: "http://localhost:8090/v1"}
  - model_name: "cloud/opus"
    litellm_params: {model: "openai/anthropic/claude-opus-5.5", api_base: "https://omniroute.example:20128/v1"}
router_settings:
  model_group_alias:
    opus: {model: "ultron/opus", hidden: true}
"""


def tracker(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG)
    monkeypatch.setattr(us, "ENDPOINTS", us.Endpoints(str(cfg)))
    return us.Tracker(str(tmp_path / "live.json"), str(tmp_path / "stats.jsonl"))


def rows(tmp_path):
    p = tmp_path / "stats.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def live(tmp_path):
    return json.loads((tmp_path / "live.json").read_text())["inflight"]


def test_endpoint_mapping(tmp_path, monkeypatch):
    tracker(tmp_path, monkeypatch)
    ep = us.ENDPOINTS
    assert ep("ultron/opus") == "ultron/opus"
    assert ep("opus") == "ultron/opus"                 # alias
    assert ep("claude-opus-5-5") == "ultron/opus"      # specific wildcard beats claude-*
    assert ep("claude-sonnet-5") == "ultron/sonnet"
    assert ep("tf-opus") == "ultron/opus"              # any loopback backend is local
    assert ep("cloud/opus") == "cloud/opus"
    assert ep("gpt-x") == "?/gpt-x"


def test_chunk_shapes():
    # OpenAI chat chunk object (ModelResponseStream-like)
    c = NS(choices=[NS(delta=NS(content="hello", reasoning_content=None, reasoning=None, tool_calls=None))], usage=None)
    assert us.chunk_info(c) == (5, {})
    tc = NS(choices=[NS(delta=NS(content=None, reasoning_content="", reasoning=None,
                                 tool_calls=[NS(function=NS(arguments='{"a":1}'))]))], usage=None)
    assert us.chunk_info(tc)[0] == 7
    # Anthropic SSE bytes: several events in one chunk
    sse = (b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"abcd"}}\n\n'
           b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":42,"input_tokens":9000,'
           b'"cache_read_input_tokens":1000}}\n\n')
    chars, usage = us.chunk_info(sse)
    assert chars == 4 and usage == {"gen": 42, "prompt": 10000, "cached": 1000}
    # Anthropic event dict
    assert us.chunk_info({"type": "content_block_delta", "delta": {"partial_json": '{"x"'}})[0] == 4
    # message_start / ping carry no content
    assert us.chunk_info({"type": "message_start", "message": {"usage": {"input_tokens": 5}}}) == (0, {"prompt": 5})
    assert us.chunk_info(b"event: ping\ndata: {\"type\":\"ping\"}\n\n") == (0, {})


def test_openai_usage_with_cache_and_reasoning():
    u = {"prompt_tokens": 1000, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 800},
         "completion_tokens_details": {"reasoning_tokens": 20}}
    assert us._usage_fields(u) == {"prompt": 1000, "gen": 50, "cached": 800, "reasoning": 20}


def test_streaming_lifecycle(tmp_path, monkeypatch):
    t = tracker(tmp_path, monkeypatch)
    data = {"litellm_call_id": "c1", "model": "claude-opus-5-5", "stream": True,
            "messages": [{"role": "user", "content": "x" * 4000}]}
    t.start(data, "anthropic_messages", now=100.0)
    assert live(tmp_path)[0]["phase"] == "waiting" and live(tmp_path)[0]["prompt_est"] > 900
    data["model"] = "cloud/opus"   # admission rewrote it after we saw the request
    t.chunk("c1", {"type": "message_start", "message": {"usage": {"input_tokens": 1}}}, now=101.0)
    t.chunk("c1", {"type": "content_block_delta", "delta": {"text": "a" * 40}}, now=110.0)
    t.chunk("c1", {"type": "content_block_delta", "delta": {"text": "a" * 400}}, now=120.0)
    assert live(tmp_path)[0]["endpoint"] == "cloud/opus"
    t.stream_end("c1", "ok", now=120.5)
    assert rows(tmp_path) == []   # waits for LiteLLM's usage
    t.logged({"litellm_call_id": "c1", "standard_logging_object": {"model": "anthropic/claude-opus-5.5", "prompt_tokens": 9000,
                                                                   "completion_tokens": 200, "startTime": 100.5,
                                                                   "completionStartTime": 101.0, "endTime": 111.0}},
             {"usage": {"prompt_tokens": 9000, "completion_tokens": 200, "prompt_tokens_details": {"cached_tokens": 8000}}},
             True, now=121.0)
    (r,) = rows(tmp_path)
    assert r["status"] == "ok" and r["endpoint"] == "cloud/opus" and r["requested"] == "claude-opus-5-5"
    assert r["prompt"] == 9000 and r["cached"] == 8000 and r["gen"] == 200
    assert r["wait"] == 0.5 and r["ttft"] == 0.5 and r["first_seen"] == 10.0
    assert r["decode_tok_s"] == 20.0 and r["prefill_tok_s"] == 2000.0
    assert r["deployment"] == "anthropic/claude-opus-5.5"
    assert live(tmp_path) == []


def test_cancelled_stream_is_written_after_grace(tmp_path, monkeypatch):
    t = tracker(tmp_path, monkeypatch)
    t.start({"litellm_call_id": "c2", "model": "opus", "stream": True}, "acompletion", now=100.0)
    t.chunk("c2", NS(choices=[NS(delta=NS(content="x" * 80, reasoning_content=None, reasoning=None, tool_calls=None))],
                     usage=None), now=103.0)
    t.stream_end("c2", "cancelled", now=104.0)
    t.sweep(now=104.0 + us.GRACE_S + 1)
    (r,) = rows(tmp_path)
    assert r["status"] == "cancelled" and r["gen_est"] == 20 and "gen" not in r and r["endpoint"] == "ultron/opus"
    assert r["ttft"] == 3.0 and r["first_seen"] == 3.0   # no LiteLLM timing: chunk clock


def test_non_stream_and_failure(tmp_path, monkeypatch):
    t = tracker(tmp_path, monkeypatch)
    t.start({"litellm_call_id": "n1", "model": "claude-sonnet-5"}, "acompletion", now=10.0)
    t.logged({"litellm_call_id": "n1", "standard_logging_object": {"prompt_tokens": 30, "completion_tokens": 7}},
             None, True, now=12.0)
    t.start({"litellm_call_id": "f1", "model": "claude-sonnet-5", "stream": True}, "acompletion", now=20.0)
    t.logged({"litellm_call_id": "f1", "standard_logging_object": {"error_str": "retrying"}}, None, False)
    assert len(rows(tmp_path)) == 1   # a failed attempt alone doesn't finish it
    t.failed("f1", ValueError("context too long"), now=21.0)
    a, b = rows(tmp_path)
    assert a["status"] == "ok" and a["gen"] == 7 and a["endpoint"] == "ultron/sonnet" and "ttft" not in a
    assert b["status"] == "error" and "context too long" in b["error"]


def test_mock_and_stale(tmp_path, monkeypatch):
    t = tracker(tmp_path, monkeypatch)
    t.start({"litellm_call_id": "m1", "model": "opus", "mock_response": "stop"}, "acompletion", now=1.0)
    t.logged({"litellm_call_id": "m1", "standard_logging_object": {}}, None, True, now=1.1)
    t.start({"litellm_call_id": "s1", "model": "opus", "stream": True}, "acompletion", now=1.0)
    t.sweep(now=1.0 + us.STALE_S + 1)
    m, s = rows(tmp_path)
    assert m["status"] == "mock" and m["endpoint"] == "loop-breaker"
    assert s["status"] == "lost"


def test_iterator_hook_passes_chunks_through_and_survives_bad_input(tmp_path, monkeypatch):
    h = us.UltronStats.__new__(us.UltronStats)
    h.t = tracker(tmp_path, monkeypatch)
    data = {"litellm_call_id": "i1", "model": "opus", "stream": True}

    async def run():
        await h.async_pre_call_hook(None, None, data, "acompletion")
        chunks = [object(), b"garbage", {"type": "content_block_delta", "delta": {"text": "hey"}}]

        async def gen():
            for c in chunks:
                yield c
        out = [c async for c in h.async_post_call_streaming_iterator_hook(None, gen(), data)]
        assert out == chunks
    asyncio.run(run())
    assert live(tmp_path)[0]["phase"] == "ending"


def test_client_gone_before_any_output_is_cleared(tmp_path, monkeypatch):
    """No stream, no log, no failure hook: the request's task finishing is what ends it."""
    t = tracker(tmp_path, monkeypatch)
    monkeypatch.setattr(us, "TICK_S", 0.05)

    async def request():
        t.start({"litellm_call_id": "gone", "model": "claude-opus-5-5", "stream": True}, "anthropic_messages")
        await asyncio.sleep(0.01)
        raise asyncio.CancelledError  # client hung up while the backend was still prefilling

    async def run():
        task = asyncio.ensure_future(request())
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert [r["cid"] for r in live(tmp_path)] == ["gone"]
        await asyncio.sleep(0.2)
    asyncio.run(run())
    assert live(tmp_path) == []
    (r,) = rows(tmp_path)
    assert r["status"] == "cancelled" and r["endpoint"] == "ultron/opus"


def test_stream_success_log_finishes_without_stream_end(tmp_path, monkeypatch):
    t = tracker(tmp_path, monkeypatch)
    t.start({"litellm_call_id": "s", "model": "opus", "stream": True}, "acompletion", now=1.0)
    t.logged({"litellm_call_id": "s", "standard_logging_object": {"completion_tokens": 5}}, None, True, now=2.0)
    assert live(tmp_path) == [] and rows(tmp_path)[0]["status"] == "ok"
