"""Unit tests: patterns taken from the real transcripts the thresholds were derived from."""

import json

import loop_breaker as lb


def oa(calls_results, user_nudges=()):
    """OpenAI history: [(name, args_dict, result_text), ...] one call per assistant turn."""
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    for i, (name, args, result) in enumerate(calls_results):
        if i in user_nudges:
            msgs.append({"role": "user", "content": "continue"})
        cid = f"c{i}"
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": result})
    return msgs


def an(calls_results):
    """Anthropic history with Claude Code style system-reminder text next to tool_result."""
    msgs = [{"role": "user", "content": "go"}]
    for i, (name, args, result) in enumerate(calls_results):
        tid = f"toolu_{i}"
        msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": name, "input": args}]})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": [{"type": "text", "text": result}]},
            {"type": "text", "text": "<system-reminder>todo list is empty</system-reminder>"}]})
    return msgs


def level(msgs):
    return lb.detect(lb.extract_steps(msgs)).level


def hermes_web(i):
    return json.dumps({"status": "success", "output": "Web config:\nbackend: ''\n", "tool_calls_made": 0,
                       "duration_seconds": 1.7 + i / 100, "kernel": {"reused": True, "execution_count": 8 + i}})


def test_hermes_execute_code_loop_escalates():
    call = ("execute_code", {"code": "import yaml\nprint(cfg['web'])"})
    assert level(oa([(*call, hermes_web(i)) for i in range(3)])) == "none"
    assert level(oa([(*call, hermes_web(i)) for i in range(4)])) == "warn"
    assert level(oa([(*call, hermes_web(i)) for i in range(6)])) == "force"
    assert level(oa([(*call, hermes_web(i)) for i in range(8)])) == "stop"


def test_user_turn_resets_the_count():
    call = ("bash", {"command": "comfy --json which"})
    assert level(oa([(*call, "same") for _ in range(8)], user_nudges=(3, 6))) == "none"  # 2 since the last user turn
    assert level(oa([(*call, "same") for _ in range(10)], user_nudges=(2,))) == "stop"   # 8 since it
    # Anthropic: tool_result user messages (even with Claude Code's reminder text) are not user turns
    assert level(an([(*call, "same") for _ in range(8)])) == "stop"


def test_bash_description_ignored_but_digits_kept():
    runs = [("Bash", {"command": "git status --short", "description": f"hold check {i}"}, "") for i in range(6)]
    assert level(an(runs)) == "none"  # `git status` is a poll: warns at 10, not 4
    v = lb.detect(lb.extract_steps(an([*runs] * 2)))
    assert v.rule == "poll" and v.run == 12 and v.level == "warn"
    # node ids differing only by a digit are different calls
    reads = [("blueprint_query", {"action": "get_node_details", "params": {"node": f"K2Node_{i}"}}, "{}") for i in range(10)]
    assert level(an(reads)) == "none"


def test_poll_with_progress_is_fine_and_terminal_poll_is_caught():
    progress = [("editor_query", {"action": "poll_pie_smoke", "params": {"session_id": "s1"}},
                 json.dumps({"status": "running", "sample_count": i, "elapsed_seconds": i * 2.5})) for i in range(30)]
    assert level(an(progress)) == "none"
    done = [("editor_query", {"action": "poll_pie_smoke", "params": {"session_id": "s1"}},
             json.dumps({"status": "done", "sample_count": 40, "elapsed_seconds": 90 + i})) for i in range(12)]
    assert level(an(done)) == "warn"
    assert level(an(done * 2)) == "stop"


def test_pie_input_bursts_below_poll_warn():
    burst = [("editor_query", {"action": "pie_inject_input_action", "params": {"action": "IA_Attack"}}, '{"ok":true}')] * 8
    assert level(an(burst)) == "none"


def test_countdown_and_clock_are_masked():
    wake = [("ScheduleWakeup", {"delaySeconds": 3600, "noop": True, "reason": "waiting"},
             f"Scheduled wakeup in {3641 - i}s at 10:{i:02d}:33") for i in range(20)]
    assert level(an(wake)) == "stop"


def test_edits_with_same_result_are_not_a_loop():
    edits = [("Edit", {"file_path": "a.cpp", "old_string": f"x{i}", "new_string": f"y{i}"}, "file updated") for i in range(40)]
    assert level(an(edits)) == "none"


def test_same_error_escalates_fast():
    err = [("curl", {"url": "http://x"}, "Error: blocked by hook") for _ in range(4)]
    assert level(oa(err)) == "force"


def test_ab_cycle():
    cyc = []
    for _ in range(6):
        cyc += [("start_pie", {}, "started"), ("stop_pie", {}, "stopped")]
    assert level(an(cyc)) == "force"
    # a cycle whose results change is progress
    prog = []
    for i in range(10):
        prog += [("ListAgents", {}, f"2 running {i}"), ("Bash", {"command": "git log -1"}, f"commit {i}")]
    assert level(an(prog)) == "none"


def test_parallel_calls_in_one_turn():
    msgs = [{"role": "user", "content": "go"}]
    for i in range(6):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"a{i}", "function": {"name": "read", "arguments": '{"path":"x"}'}},
            {"id": f"b{i}", "function": {"name": "read", "arguments": '{"path":"y"}'}}]})
        msgs.append({"role": "tool", "tool_call_id": f"b{i}", "content": "Y"})
        msgs.append({"role": "tool", "tool_call_id": f"a{i}", "content": "X"})
    assert level(msgs) == "force"


def test_incomplete_last_step_is_ignored():
    msgs = oa([("bash", {"command": "ls"}, "a")] * 8)
    msgs.pop()  # last tool result missing
    assert level(msgs) == "none"


def test_apply_openai_force_and_anthropic_warn_keep_prefix():
    msgs = oa([("bash", {"command": "ls"}, "a")] * 6)
    data = {"model": "haiku", "messages": [dict(m) for m in msgs], "tools": [{"type": "function"}]}
    before = json.dumps(data["messages"])
    v = lb.evaluate(data, "acompletion", "enforce")
    assert v.level == "force" and "tool_choice" not in data
    assert json.dumps(data["messages"][:-1]) == before  # prefix untouched, note appended
    assert data["messages"][-1]["content"].startswith(lb.NOTE_PREFIX)

    amsgs = an([("Bash", {"command": "ls"}, "a")] * 4)
    data = {"model": "claude-sonnet-5", "messages": amsgs}
    lb.evaluate(data, "anthropic_messages", "enforce")
    last = data["messages"][-1]
    assert last["role"] == "user" and last["content"][0]["type"] == "tool_result"
    assert last["content"][-1]["text"].startswith(lb.NOTE_PREFIX)
    assert "tool_choice" not in data


def test_stop_uses_mock_response_and_shadow_changes_nothing():
    msgs = oa([("bash", {"command": "ls"}, "a")] * 8)
    data = {"messages": msgs}
    lb.evaluate(data, "acompletion", "enforce")
    assert data["mock_response"].startswith(lb.NOTE_PREFIX)
    data = {"messages": oa([("bash", {"command": "ls"}, "a")] * 8)}
    snap = json.dumps(data)
    assert lb.evaluate(data, "acompletion", "shadow").level == "stop"
    assert json.dumps(data) == snap


def test_opt_out_and_garbage_never_raise():
    data = {"messages": oa([("bash", {"command": "ls"}, "a")] * 8), "metadata": {"loop_breaker": "off"}}
    assert lb.evaluate(data, "acompletion", "enforce").level == "none"
    for junk in ({}, {"messages": None}, {"messages": [None, 3, {"role": "tool"}]},
                 {"messages": [{"role": "assistant", "tool_calls": [{"function": {"arguments": "{bad"}}]}]}):
        lb.evaluate(junk, "acompletion", "enforce")


def test_busy_pump_is_a_poll():
    pump = [("mcp__monolith__editor_query", {"action": "run_python", "params": {"command": "/x/drive_level.py pump"}},
             '{"ok":true,"output":[{"type":"info","output":"{\\"state\\": \\"busy\\", \\"idx\\": 1}"}]}')] * 12
    assert level(an(pump)) == "warn"  # poll thresholds: 12 < force 16
    busy = [("execute_python_code", {"code": "print(get_status())"}, '{"status": "compiling"}')] * 9
    assert level(an(busy)) == "none"


def test_cycle_with_wait_uses_poll_thresholds():
    pair = [("Bash", {"command": "check_port 9316"}, "BIND FAILED"), ("ScheduleWakeup", {"delaySeconds": 3600}, "scheduled in 3600s")]
    assert level(an(pair * 8)) == "none"  # a plain cycle would already be at stop
    v = lb.detect(lb.extract_steps(an(pair * 10)))
    assert v.rule == "poll_cycle2" and v.level == "warn"


def test_anthropic_streaming_stop_uses_streaming_mock(monkeypatch):
    monkeypatch.setattr(lb, "_ANTHROPIC_MOCK_STREAMS", True)
    data = {"messages": an([("Bash", {"command": "ls"}, "a")] * 8), "stream": True}
    lb.evaluate(data, "anthropic_messages", "enforce")
    assert data["mock_response"].startswith(lb.STREAM_MARK + lb.NOTE_PREFIX)


def test_anthropic_streaming_stop_falls_back_to_forced_text(monkeypatch):
    monkeypatch.setattr(lb, "_ANTHROPIC_MOCK_STREAMS", False)
    data = {"messages": an([("Bash", {"command": "ls"}, "a")] * 8), "stream": True, "max_tokens": 32000}
    lb.evaluate(data, "anthropic_messages", "enforce")
    assert "mock_response" not in data
    assert data["tool_choice"] == {"type": "none"} and data["max_tokens"] == 512
    assert data["messages"][-1]["content"][-1]["text"].startswith(lb.NOTE_PREFIX + " Stopped")


def test_still_building_is_a_poll_but_noop_true_is_not():
    nav = [("mcp__unreal__execute_python_code", {"code": "print('still building:', nav.is_building())"},
            '{ "success": true, "output": "still building: True", "result": "None" }')] * 9
    assert level(an(nav)) == "none"
    noop = [("Bash", {"command": "true"}, "(Bash completed with no output)")] * 6
    assert level(an(noop)) == "force"


def test_log_key_matches_admission_pin_key(tmp_path, monkeypatch):
    import ultron_admit as ua
    monkeypatch.setattr(lb, "LOG_PATH", str(tmp_path / "lb.jsonl"))
    data = {"model": "claude-haiku-4-5", "messages": an([("Bash", {"command": "ls"}, "a")] * 4),
            "proxy_server_request": {"headers": {"X-Claude-Code-Session-Id": "S1", "x-claude-code-agent-id": "a7"}}}
    lb.evaluate(data, "anthropic_messages", "shadow")
    key = json.loads(open(tmp_path / "lb.jsonl").read())["key"]
    assert ua.pin_identity(data, "haiku")["key"] == key + ":haiku"
    oai = {"messages": oa([("bash", {"command": "ls"}, "a")] * 4)}
    lb.evaluate(oai, "acompletion", "shadow")
    key2 = json.loads(open(tmp_path / "lb.jsonl").read().splitlines()[-1])["key"]
    assert ua.pin_identity(oai, "sonnet")["key"] == key2 + ":sonnet"


def test_mode_file_overrides_env(tmp_path, monkeypatch):
    f = tmp_path / "mode"
    monkeypatch.setattr(lb, "MODE_FILE", str(f))
    monkeypatch.setenv("LOOP_BREAKER_MODE", "shadow")
    assert lb.current_mode() == "shadow"
    f.write_text("enforce\n")
    assert lb.current_mode() == "enforce"
    f.write_text("bogus\n")
    assert lb.current_mode() == "shadow"


def test_force_note_names_the_calls_and_never_asks_for_text():
    # A sonnet-tier agent, 2026-09-30: a sync / describe cycle. The old note ("Do not call `Bash` again.
    # Stop and reply to the user now, in text") left it stuck in text-only replies for 6 hours.
    sync = 'git pull --ff-only origin main -- "Scripts/smoke/run_smoke.sh" 2>&1 || echo "Failed: $?"'
    desc = 'git show --stat HEAD -- "Scripts/smoke/run_smoke.sh" 2>&1 | head -10'
    cyc = []
    for i in range(6):
        cyc += [("Bash", {"command": sync, "description": f"sync {i}"}, "Already up to date."),
                ("Bash", {"command": desc, "description": "describe"}, "Scripts/smoke/run_smoke.sh | 2 +-")]
    msgs = an(cyc)
    v = lb.detect(lb.extract_steps(msgs))
    assert v.level == "force"
    note = lb.note_text(v)
    assert f"- Bash `{sync}`" in note and f"- Bash `{desc}`" in note
    assert "in text" not in note and "Do not call `Bash`" not in note
    assert "change course" in note and "next identical call will end this turn" in note
    # long commands are cut, non-shell tools show their args
    assert lb._call_label("Bash", json.dumps({"command": "x" * 300})).endswith("...`")
    assert lb._call_label("Read", json.dumps({"file_path": "/a"})) == "Read `/a`"
