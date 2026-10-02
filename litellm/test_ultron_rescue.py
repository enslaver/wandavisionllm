"""Unit tests: the replies are modelled on ones a sonnet-tier coding agent produced on 2026-09-30."""

import asyncio
import json

import ultron_rescue as ur

CC_TOOLS = [
    {"name": "Bash", "input_schema": {"type": "object", "properties": {
        "command": {"type": "string"}, "description": {"type": "string"}}, "required": ["command"]}},
    {"name": "Read", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}},
    {"name": "ListAgents", "input_schema": {"type": "object", "properties": {"filter": {"type": "string"}}}},
]
CMD = 'cd ~/src/game && git log -1 --format=%H -- "Scripts/smoke/run_smoke.sh" 2>&1 | head -10'
STALL = ("**The smoke test script is synced.** Now I'll describe the file's state to get the commit hash.\n\n"
         "```bash\n" + CMD + "\n```")


def test_stall_becomes_bash_call():
    narration, name, args = ur.tool_call_for(STALL, CC_TOOLS)
    assert name == "Bash" and args == {"command": CMD}
    assert narration.rstrip().endswith("commit hash.")


def test_declared_tool_written_as_command():
    text = "**The subagent is stalled.** I'll call `ListAgents` to see what's running.\n\n```bash\nListAgents {\"filter\": \"running\"}\n```"
    assert ur.tool_call_for(text, CC_TOOLS)[1:] == ("ListAgents", {"filter": "running"})
    # a declared tool with arguments that aren't JSON is not guessed at (and not run in bash)
    assert ur.tool_call_for(text.replace('{"filter": "running"}', "--filter running"), CC_TOOLS) is None


def test_left_alone():
    example = "Here is how you can check it:\n\n```bash\ngit log -1\n```"
    assert ur.tool_call_for(example, CC_TOOLS) is None                           # no announced action
    assert ur.tool_call_for(STALL + "\n\nThat prints the hash.", CC_TOOLS) is None  # text after the block
    assert ur.tool_call_for(STALL.replace("```bash", "```python"), CC_TOOLS) is None   # not a shell block
    two = "Now I'll run both.\n\n```bash\nls\n```\n\n```bash\npwd\n```"
    assert ur.tool_call_for(two, CC_TOOLS) is None                                # more than one block
    assert ur.tool_call_for(STALL, [CC_TOOLS[1]]) is None                         # no shell tool declared


def test_shell_argument_from_schema():
    pi = [{"type": "function", "function": {"name": "bash", "parameters": {
        "type": "object", "properties": {"cmd": {"type": "string"}, "timeout": {"type": "number"}}, "required": ["cmd"]}}}]
    assert ur.tool_call_for(STALL, pi)[1:] == ("bash", {"cmd": CMD})
    console = "Let me check the arch.\n\n```console\n$ uname -m\n```"
    assert ur.tool_call_for(console, CC_TOOLS)[2] == {"command": "uname -m"}


# ----------------------------------------------------------------------------- streaming

def sse_stream(text_deltas, stop="end_turn", tool_use=False):
    evs = [{"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "content": []}},
           {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}]
    evs += [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": d}} for d in text_deltas]
    evs.append({"type": "content_block_stop", "index": 0})
    if tool_use:
        evs += [{"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t", "name": "Read", "input": {}}},
                {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
                {"type": "content_block_stop", "index": 1}]
    evs += [{"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 80}},
            {"type": "message_stop"}]
    return [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in evs]


def run_stream(chunks, enforce=True):
    rw = ur.StreamRewriter(CC_TOOLS, enforce)
    out = []
    for c in chunks:
        out += rw.feed(c)
    out += rw.finish()
    events = [json.loads(line[5:]) for c in out for line in c.decode().splitlines() if line.startswith("data:")]
    text = "".join(e["delta"]["text"] for e in events if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta")
    return rw, events, text


def chop(s, n=7):
    return [s[i:i + n] for i in range(0, len(s), n)]  # splits the fences across deltas too


def test_stream_rescue():
    rw, events, text = run_stream(sse_stream(chop(STALL)))
    assert rw.applied and "```" not in text and text.rstrip().endswith("commit hash.")
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert starts[-1]["content_block"]["type"] == "tool_use" and starts[-1]["index"] == 1
    args = "".join(e["delta"]["partial_json"] for e in events if (e.get("delta") or {}).get("type") == "input_json_delta")
    assert json.loads(args) == {"command": CMD}
    md = [e for e in events if e["type"] == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "tool_use" and md["usage"] == {"output_tokens": 80}
    assert [e["type"] for e in events][-1] == "message_stop"
    # every started block is stopped exactly once
    assert sorted(e["index"] for e in events if e["type"] == "content_block_stop") == [0, 1]


def test_stream_untouched_when_not_rescued():
    prose = STALL + "\n\nThat prints the hash."
    for chunks, original in ((sse_stream(chop(prose)), prose),                   # prose after the block
                             (sse_stream(chop(STALL), tool_use=True), STALL),    # already called a tool
                             (sse_stream(chop(STALL), stop="max_tokens"), STALL)):  # cut off, not finished
        rw, events, text = run_stream(chunks)
        assert not rw.applied and text == original
        assert sum(e["type"] == "content_block_stop" and e["index"] == 0 for e in events) == 1
    rw, events, text = run_stream(sse_stream(chop(STALL)), enforce=False)  # shadow: logged, not changed
    assert rw.rescued and not rw.applied and text == STALL


def test_stream_code_is_released_once_it_cannot_be_rescued():
    rw = ur.StreamRewriter(CC_TOOLS, True)
    chunks = sse_stream(["Let me show it.\n\n```bash\nls\n```\n", "and more prose", " here"])
    out = []
    for c in chunks[:4]:  # up to and including the delta that ends the block with prose after it
        out += rw.feed(c)
    sent = "".join(json.loads(c.decode().split("data:")[1])["delta"].get("text", "") for c in out
                   if b"text_delta" in c)
    assert "```bash\nls\n```" in sent  # not held until the end of the block


def test_non_stream_rescue():
    resp = {"type": "message", "content": [{"type": "text", "text": STALL}], "stop_reason": "end_turn"}
    assert ur.rewrite_response(resp, CC_TOOLS, True)
    assert resp["stop_reason"] == "tool_use"
    assert resp["content"][0]["text"].endswith("commit hash.")
    assert resp["content"][1]["name"] == "Bash" and resp["content"][1]["input"] == {"command": CMD}


def test_applies():
    base = {"model": "ultron/sonnet", "tools": CC_TOOLS}
    assert ur.applies(base)
    assert not ur.applies({**base, "model": "cloud/sonnet"})
    assert not ur.applies({**base, "tools": []})
    assert not ur.applies({**base, "proxy_server_request": {"headers": {"X-Rescue": "off"}}})


def test_hook_streams_through_litellm_glue():
    async def upstream():
        for c in sse_stream(chop(STALL)):
            yield c

    async def collect():
        hook = ur.UltronRescue()
        data = {"model": "ultron/sonnet", "tools": CC_TOOLS, "stream": True}
        return [c async for c in hook.async_post_call_streaming_iterator_hook(None, upstream(), data)]

    ur.LOG_PATH = "/dev/null"
    ur.MODE_FILE = "/nonexistent"
    out = asyncio.run(collect())
    assert any(b'"stop_reason": "tool_use"' in c for c in out)
