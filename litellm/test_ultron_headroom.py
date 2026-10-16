"""Tests for ultron_headroom: mode file, opt-outs, no floor, never Headroom Cloud, fail-open."""
import asyncio
import json

import pytest

import ultron_headroom as uh


def make_hook(result=None, boom=None):
    hook = uh.UltronHeadroom()
    hook._local_ready = True
    hook._total_saved = 0
    calls = []

    def fake_compress(messages, model):
        calls.append((messages, model))
        if boom:
            raise boom
        return result

    hook._local_compress = fake_compress
    hook.calls = calls
    return hook


def run(hook, data, call_type="acompletion"):
    return asyncio.run(hook.async_pre_call_hook(None, None, data, call_type))


def turn(**extra):
    data = {"model": "ultron/sonnet", "messages": [{"role": "user", "content": "hi"}], "litellm_call_id": "c1"}
    data.update(extra)
    return data


def log_rows():
    try:
        return [json.loads(line) for line in open(uh.LOG_PATH)]
    except OSError:
        return []


@pytest.fixture
def on():
    open(uh.MODE_FILE, "w").write("on\n")


SMALL = {"messages": [{"role": "user", "content": "short"}], "tokens_saved": 7, "tokens_before": 20, "tokens_after": 13}


def test_off_by_default_leaves_the_request_alone():
    hook = make_hook(SMALL)
    data = turn()
    assert run(hook, data)["messages"][0]["content"] == "hi"
    assert hook.calls == []


def test_on_compresses_with_no_token_floor(on):
    hook = make_hook(SMALL)
    data = run(hook, turn())  # a 2-token prompt: bili's 34k floor would have skipped it
    assert data["messages"] == SMALL["messages"]
    assert hook.calls and hook.calls[0][1] == "ultron/sonnet"
    row = log_rows()[-1]
    assert row["applied"] is True and row["saved"] == 7 and row["before"] == 20 and row["after"] == 13


def test_x_headroom_off_header_opts_out(on):
    hook = make_hook(SMALL)
    run(hook, turn(proxy_server_request={"headers": {"x-headroom": "off"}}))
    assert hook.calls == []


def test_mock_response_is_skipped(on):
    hook = make_hook(SMALL)
    run(hook, turn(mock_response="[ultron loop-breaker] Stopped"))
    assert hook.calls == []


@pytest.mark.parametrize("call_type", ["embedding", "aembedding", "call_mcp_tool", "image_generation", "moderation"])
def test_non_chat_calls_are_skipped(on, call_type):
    hook = make_hook(SMALL)
    run(hook, turn(), call_type)
    assert hook.calls == []


def test_anthropic_messages_calls_are_compressed(on):
    hook = make_hook(SMALL)
    run(hook, turn(), "anthropic_messages")
    assert len(hook.calls) == 1


def test_nothing_saved_keeps_the_original_messages(on):
    hook = make_hook({"messages": [{"role": "user", "content": "changed"}], "tokens_saved": 0})
    data = run(hook, turn())
    assert data["messages"][0]["content"] == "hi"
    assert log_rows()[-1]["applied"] is False


def test_saved_tokens_show_in_the_route_header(on):
    hook = make_hook(SMALL)
    import ultron_admit
    ultron_admit.proxy_handler_instance.admission.recent["c1"] = "ultron/sonnet; rule=x"
    try:
        run(hook, turn())
        run(hook, turn())  # a /v1/messages request passes the hook twice: one marker only
        assert ultron_admit.proxy_handler_instance.admission.recent["c1"] == "ultron/sonnet; rule=x; headroom(-7)"
    finally:
        ultron_admit.proxy_handler_instance.admission.recent.pop("c1", None)


def test_compress_failure_fails_open_and_logs(on):
    hook = make_hook(boom=RuntimeError("kaboom"))
    data = run(hook, turn())
    assert data["messages"][0]["content"] == "hi"
    assert "kaboom" in log_rows()[-1]["error"]


def test_missing_headroom_package_is_a_logged_noop(on, monkeypatch):
    hook = make_hook(SMALL)
    hook._local_ready = False
    monkeypatch.setattr(uh, "IMPORT_ERROR", "ModuleNotFoundError('headroom')")
    assert run(hook, turn())["messages"][0]["content"] == "hi"
    assert hook.calls == []
    assert "not importable" in log_rows()[-1]["error"]


def test_never_headroom_cloud(monkeypatch):
    """HEADROOM_API_KEY may sit in ~/.litellm/env for another proxy; the class would send prompts to Cloud with it."""
    pytest.importorskip("headroom")
    monkeypatch.setenv("HEADROOM_API_KEY", "hr_test_key")
    assert uh.UltronHeadroom()._api_key is None


def test_route_header_goes_to_the_admission_hook_litellm_is_running(on, monkeypatch):
    """LiteLLM loads each hook from its file path, so `import ultron_admit` is a different copy than the running one."""
    import sys
    import types
    live = types.SimpleNamespace(admission=types.SimpleNamespace(recent={"c1": "cloud/sonnet; rule=x"}))
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(callbacks=[object(), live]))
    run(make_hook(SMALL), turn())
    assert live.admission.recent["c1"] == "cloud/sonnet; rule=x; headroom(-7)"
