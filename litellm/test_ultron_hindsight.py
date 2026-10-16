"""Unit tests for ultron_hindsight: notes go on human turns and come back identically on every later request."""

import asyncio
import json

import pytest

import ultron_hindsight as uh

TOOLS = [{"name": "Bash", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}]
FACT = {"id": "1", "text": "the LiteLLM config lives in the wandavision repo", "occurred_start": "2026-10-08T10:00:00Z"}


def req(messages, **headers):
    return {"model": "claude-opus-4-7", "tools": TOOLS, "litellm_call_id": "c1", "messages": messages,
            "proxy_server_request": {"headers": {"x-claude-code-session-id": "s1", **headers}}}


@pytest.fixture
def hook(tmp_path, monkeypatch):
    (tmp_path / "mode").write_text("on")
    monkeypatch.setattr(uh, "MODE_FILE", str(tmp_path / "mode"))
    monkeypatch.setattr(uh, "URL", "http://hindsight.example")
    monkeypatch.setattr(uh, "LOG_PATH", str(tmp_path / "hindsight.jsonl"))
    monkeypatch.setattr(uh, "NOTES_DB", str(tmp_path / "notes.sqlite"))
    calls = {"recall": [], "retain": []}
    monkeypatch.setattr(uh, "recall", lambda q: calls["recall"].append(q) or [FACT])
    monkeypatch.setattr(uh, "retain", lambda *a: calls["retain"].append(a) or {"operation_id": "op1"})
    h = uh.UltronHindsight()
    h.calls = calls
    return h


def run(h, data, call_type="anthropic_messages"):
    return asyncio.run(h.async_pre_call_hook(None, None, data, call_type))


def test_note_goes_on_the_human_turn_and_stays_identical(hook):
    turn1 = {"role": "user", "content": [{"type": "text", "text": "<system-reminder>x</system-reminder>"},
                                         {"type": "text", "text": "where does the LiteLLM config live?"}]}
    first = run(hook, req([dict(turn1)]))
    note = first["messages"][0]["content"][-1]["text"]
    assert note.startswith('<memory source="hindsight">') and FACT["text"] in note and "(2026-10-08)" in note
    assert hook.calls["recall"] == ["where does the LiteLLM config live?"]  # reminders stripped from the query
    # the tool loop and the next human turn: the client resends history without the note; it comes back as is
    tool_use = {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]}
    tool_res = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}
    loop = run(hook, req([dict(turn1), tool_use, tool_res]))
    assert loop["messages"][0] == first["messages"][0]
    assert len(hook.calls["recall"]) == 1  # no recall mid-turn
    turn2 = {"role": "user", "content": "and the llama-swap config?"}
    nxt = run(hook, req([dict(turn1), tool_use, tool_res, {"role": "assistant", "content": "done"}, turn2]))
    assert nxt["messages"][0] == first["messages"][0] and nxt["messages"][4]["content"].endswith("</memory>")
    assert len(hook.calls["recall"]) == 2


def test_notes_survive_a_restart(hook, tmp_path, monkeypatch):
    m = [{"role": "user", "content": "where does the LiteLLM config live?"}]
    before = run(hook, req([dict(m[0])]))["messages"][0]
    again = uh.UltronHindsight()  # new process: same sqlite, empty memory
    monkeypatch.setattr(uh, "recall", lambda q: pytest.fail("recalled again"))
    assert run(again, req([dict(m[0])]))["messages"][0] == before


def test_skips(hook):
    m = [{"role": "user", "content": "where does the LiteLLM config live?"}]
    assert run(hook, req([dict(m[0])], **{"x-hindsight": "off"}))["messages"] == m
    assert run(hook, req([dict(m[0])], **{"x-claude-code-agent-id": "a1"}))["messages"] == m
    assert run(hook, {**req([dict(m[0])]), "tools": None})["messages"] == m
    assert run(hook, {**req([dict(m[0])]), "mock_response": "[ultron loop-breaker] stop"})["messages"] == m
    assert run(hook, req([dict(m[0])]), call_type="pass_through_endpoint")["messages"] == m
    assert hook.calls["recall"] == []


def test_mode_off_and_short_turns(hook, tmp_path):
    short = [{"role": "user", "content": "continue"}]
    assert run(hook, req([dict(short[0])]))["messages"] == short  # too short to look up
    assert hook.calls["recall"] == []
    (tmp_path / "mode").write_text("off")
    m = [{"role": "user", "content": "where does the LiteLLM config live?"}]
    assert run(hook, req([dict(m[0])]))["messages"] == m


def test_recall_failure_fails_open_and_holds_off(hook, monkeypatch):
    def boom(q):
        raise OSError("down")
    monkeypatch.setattr(uh, "recall", boom)
    m = [{"role": "user", "content": "where does the LiteLLM config live?"}]
    assert run(hook, req([dict(m[0])]))["messages"] == m
    assert hook.down_until > 0
    monkeypatch.setattr(uh, "recall", lambda q: pytest.fail("recalled while down"))
    assert run(hook, req([dict(m[0])]))["messages"] == m


def test_retain_only_when_the_turn_ends(hook):
    run(hook, req([{"role": "user", "content": "where does the LiteLLM config live?"}]))
    tool_reply = {"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}], "stop_reason": "tool_use"}
    asyncio.run(hook.async_log_success_event({"litellm_call_id": "c1"}, tool_reply, 0, 0))
    assert hook.calls["retain"] == []
    run(hook, req([{"role": "user", "content": "where does the LiteLLM config live?"}]))
    final = {"content": [{"type": "text", "text": "In litellm/config.yaml."}], "stop_reason": "end_turn"}
    asyncio.run(hook.async_log_success_event({"litellm_call_id": "c1"}, final, 0, 0))
    content, doc_id, agent, model = hook.calls["retain"][0]
    assert content == "User: where does the LiteLLM config live?\n\nAssistant: In litellm/config.yaml."
    assert doc_id.startswith("cc:s1:main|") and agent == "claude-code" and model == "claude-opus-4-7"
    log = [json.loads(l) for l in open(uh.LOG_PATH)]
    assert log[-1]["event"] == "retain" and log[-1]["op"] == "op1"


def test_chat_completions_shapes(hook):
    data = {**req([{"role": "system", "content": "sys"}, {"role": "user", "content": "where does the LiteLLM config live?"},
                   {"role": "assistant", "content": None, "tool_calls": [{"id": "t", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}]},
                   {"role": "tool", "tool_call_id": "t", "content": "ok"}]),
            "proxy_server_request": {"headers": {"user-agent": "pi-coding-agent"}}}
    out = run(hook, data, call_type="acompletion")["messages"]
    assert out[0]["content"] == "sys" and out[1]["content"].endswith("</memory>") and out[3]["content"] == "ok"
    assert uh.called_tool({"choices": [{"finish_reason": "tool_calls", "message": {}}]})
    assert not uh.called_tool({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})
