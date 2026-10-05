"""Tests for ultron_media: what counts as a media request, and that agent loops are never touched."""

import asyncio
import base64
import json

import pytest

import ultron_media as um

REAL_CONFIRM = um.helper_confirm  # the sandbox fixture stubs um.helper_confirm


def turn(text, images=(), audio=()):
    return {"text": text, "images": list(images), "audio": list(audio), "content": []}


@pytest.mark.parametrize("text,kind", [
    ("generate an image of a red fox in the snow", "image"),
    ("Can you draw a picture of a castle at sunset?", "image"),
    ("create a photo of a mountain lake, cinematic", "image"),
    ("/imagine a neon city", "image"),
    ("make a video of a rotating cube", "video"),
    ("generate a short clip of ocean waves", "video"),
    ("search the web for litellm 1.103 release notes", "search"),
    ("what's the latest news about the Mars mission? google it", "search"),
])
def test_positive(text, kind):
    assert um.classify(turn(text)) == kind


@pytest.mark.parametrize("text", [
    "generate an image component in React that lazy loads",
    "create a docker image for the api",
    "make the video player fullscreen",
    "write a function that generates an image thumbnail",
    "search the web codebase for TODO",
    "fix the bug where the image tag is missing",
    "make a video element in html",
    "how do I resize an image with imagemagick",
    "explain how to generate images with diffusion models",
    "",
])
def test_negative(text):
    assert um.classify(turn(text)) is None


def test_edit_needs_an_image_and_edit_words():
    png = (b"x", "image/png")
    assert um.classify(turn("remove the background from this photo", [png])) == "edit"
    assert um.classify(turn("make the sky in this picture purple", [png])) == "edit"
    assert um.classify(turn("what is in this photo?", [png])) is None
    assert um.classify(turn("remove the background from this photo")) is None  # nothing attached
    assert um.classify(turn("fix the css in this screenshot", [png])) is None


def test_audio_block_is_always_transcribed():
    assert um.classify(turn("", audio=[0])) == "audio"


def test_human_turn_ignores_tool_results_and_reminders():
    fresh = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "<system-reminder>todo</system-reminder>"},
        {"type": "text", "text": "generate an image of a cat"}]}]}
    assert um.human_turn(fresh)["text"] == "generate an image of a cat"
    mid_loop = {"messages": [
        {"role": "user", "content": "generate an image of a cat"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"},
                                     {"type": "text", "text": "generate an image of a cat"}]}]}
    assert um.human_turn(mid_loop) is None
    assert um.human_turn({"messages": [{"role": "assistant", "content": "hi"}]}) is None
    assert um.human_turn({"input": "responses api"}) is None


def test_human_turn_reads_images_and_audio():
    b64 = base64.b64encode(b"\xff\xd8\xff123").decode()
    d = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "edit this"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}]}
    t = um.human_turn(d)
    assert len(t["images"]) == 2 and t["audio"] == [3]


def test_sniff_uses_bytes_not_names():
    assert um.sniff(b"\xff\xd8\xff\xe0abc") == ("jpg", "image/jpeg")
    assert um.sniff(b"\x89PNG\r\n\x1a\nxx")[0] == "png"
    assert um.sniff(b"\x00\x00\x00\x18ftypmp42")[0] == "mp4"


def test_multipart_shape():
    body, ct = um._multipart({"model": "m", "prompt": "p"}, [("image", "in.png", "image/png", b"RAW")])
    boundary = ct.split("boundary=")[1].encode()
    assert body.count(b"--" + boundary) == 4 and b'name="image"; filename="in.png"' in body and b"RAW" in body


# ----------------------------------------------------------------------------- the hook

@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(um, "LOG_PATH", str(tmp_path / "media.jsonl"))
    monkeypatch.setattr(um, "MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setattr(um, "CONF_FILE", str(tmp_path / "none.json"))
    monkeypatch.setattr(um, "MODE_FILE", str(tmp_path / "mode"))
    monkeypatch.setattr(um, "helper_confirm", lambda text, kind: True)
    calls = []
    monkeypatch.setattr(um, "gen_image", lambda prompt: (calls.append(prompt) or ("a.png", "provider/x")))
    return calls


def req(text, tools=None, headers=None, stream=False, model="claude-sonnet-5"):
    d = {"model": model, "messages": [{"role": "user", "content": text}], "stream": stream,
         "proxy_server_request": {"headers": headers or {}}, "litellm_call_id": "c1"}
    if tools:
        d["tools"] = tools
    return d


def run(hook, data, mode="enforce", call_type="acompletion"):
    asyncio.run(hook.handle(data, call_type, mode))
    return data


def test_enforce_answers_image_request_without_a_model(sandbox):
    d = run(um.Media(), req("generate an image of a fox"))
    assert "a.png" in d["mock_response"] and "provider/x" in d["mock_response"] and sandbox == ["generate an image of a fox"]


def test_shadow_and_off_change_nothing(sandbox):
    for mode in ("shadow", "off"):
        d = run(um.Media(), req("generate an image of a fox"), mode=mode)
        assert "mock_response" not in d
    assert sandbox == []


def test_shadow_logs_what_it_would_do(tmp_path):
    run(um.Media(), req("generate an image of a fox"), mode="shadow")
    line = (tmp_path / "media.jsonl").read_text().splitlines()[0]
    assert '"kind": "image"' in line and '"applied": false' in line


def test_agent_request_needs_helper_yes(sandbox, monkeypatch):
    tools = [{"name": "Bash"}]
    monkeypatch.setattr(um, "helper_confirm", lambda text, kind: False)
    assert "mock_response" not in run(um.Media(), req("generate an image of a fox", tools=tools))
    monkeypatch.setattr(um, "helper_confirm", lambda text, kind: True)
    assert "mock_response" in run(um.Media(), req("generate an image of a fox", tools=tools))


def test_agent_mid_loop_and_claude_code_helpers_are_left_alone(sandbox):
    mid = req("x", tools=[{"name": "Bash"}])
    mid["messages"] = [{"role": "user", "content": "generate an image of a fox"},
                       {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}]},
                       {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}]
    assert "mock_response" not in run(um.Media(), mid)
    helper = req("generate an image of a fox", headers={"x-claude-code-session-id": "s1"})  # title generator: no tools
    assert "mock_response" not in run(um.Media(), helper)
    assert sandbox == []


def test_opt_out_header_and_existing_mock(sandbox):
    assert "mock_response" not in run(um.Media(), req("generate an image of a fox", headers={"x-media": "off"}))
    d = req("generate an image of a fox")
    d["mock_response"] = "loop breaker got there first"
    assert run(um.Media(), d)["mock_response"] == "loop breaker got there first" and sandbox == []


def test_same_prompt_is_generated_once(sandbox):
    m = um.Media()
    run(m, req("generate an image of a fox"))
    d2 = run(m, req("generate an image of a fox"))
    assert len(sandbox) == 1 and "a.png" in d2["mock_response"]


def test_failure_falls_through_to_the_model(monkeypatch):
    def boom(prompt):
        raise RuntimeError("omniroute down")
    monkeypatch.setattr(um, "gen_image", boom)
    d = run(um.Media(), req("generate an image of a fox"))
    assert "mock_response" not in d  # the tier answers normally


def test_anthropic_streaming_reply_is_marked_for_sse(sandbox):
    d = req("generate an image of a fox", stream=True)
    run(um.Media(), d, call_type="anthropic_messages")
    if um.lb._ANTHROPIC_MOCK_STREAMS:
        assert d["mock_response"].startswith(um.lb.STREAM_MARK)
    else:
        assert "mock_response" not in d


def test_search_adds_results_and_lets_the_model_answer(monkeypatch):
    monkeypatch.setattr(um, "web_search", lambda q: "1. Title <https://x>\n   snippet")
    d = run(um.Media(), req("search the web for litellm release notes"))
    assert "mock_response" not in d and "https://x" in d["messages"][-1]["content"]


def test_audio_block_becomes_a_transcript(monkeypatch):
    monkeypatch.setattr(um, "transcribe", lambda raw, fmt: ("hello world", "openrouter/whisper"))
    b64 = base64.b64encode(b"RIFFxxxx").decode()
    d = req("")
    d["messages"] = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}]
    run(um.Media(), d)
    assert d["messages"][0]["content"][0]["text"] == "[Transcript of the attached audio: hello world]"


def test_responses_api_and_embeddings_are_ignored(sandbox):
    d = {"model": "m", "input": "generate an image of a fox"}
    run(um.Media(), d)
    run(um.Media(), req("generate an image of a fox"), call_type="aembedding")
    assert sandbox == []


# ----------------------------------------------------------------------------- chat route (chat_prefixes)
# An endpoint that serves some image/audio models only through chat (OmniRoute's gemini/* ids).
CHAT_CONF = {"image": ["gpt-image-1", "gemini/flash-image"], "edit": ["gemini/flash-image", "gpt-image-1"],
             "transcribe": ["gemini/flash", "whisper-1"], "chat_prefixes": ["gemini/"]}


def _chat_conf(monkeypatch, tmp_path):
    f = tmp_path / "media.json"
    f.write_text(json.dumps(CHAT_CONF))
    monkeypatch.setattr(um, "CONF_FILE", str(f))


def _chat_reply(*urls):
    return {"choices": [{"message": {"role": "assistant", "content": [
        {"type": "image_url", "image_url": {"url": u}} for u in urls]}}]}


def test_chat_image_model_goes_through_chat_and_keeps_the_first_image(monkeypatch, tmp_path):
    monkeypatch.undo()  # sandbox's gen_image stub off; real one, with _post faked
    monkeypatch.setattr(um, "MEDIA_DIR", str(tmp_path / "media"))
    _chat_conf(monkeypatch, tmp_path)
    seen = []
    jpg = base64.b64encode(b"\xff\xd8\xff-one").decode()

    def post(path, body, ctype, timeout):
        seen.append(path)
        if path == "/images/generations":
            raise RuntimeError("gpt-image-1 down")
        return _chat_reply("data:image/jpeg;base64," + jpg, "data:image/jpeg;base64," + jpg)
    monkeypatch.setattr(um, "_post", post)
    name, model = um.gen_image("a fox")
    assert model == "gemini/flash-image" and name.endswith(".jpg")
    assert seen == ["/images/generations", "/chat/completions"]


def test_chat_edit_sends_the_image_in_a_chat_message(monkeypatch, tmp_path):
    monkeypatch.undo()
    monkeypatch.setattr(um, "MEDIA_DIR", str(tmp_path / "media"))
    _chat_conf(monkeypatch, tmp_path)
    sent = {}

    def post(path, body, ctype, timeout):
        sent.update(path=path, body=um.json.loads(body))
        return _chat_reply("data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nx").decode())
    monkeypatch.setattr(um, "_post", post)
    name, model = um.edit_image("make it blue", (b"\xff\xd8\xff-in", "image/jpeg"))
    parts = sent["body"]["messages"][0]["content"]
    assert sent["path"] == "/chat/completions" and model.startswith("gemini/") and name.endswith(".png")
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_chat_reply_without_an_image_moves_to_the_next_model(monkeypatch, tmp_path):
    monkeypatch.undo()
    _chat_conf(monkeypatch, tmp_path)
    monkeypatch.setattr(um, "_post", lambda *a: {"choices": [{"message": {"content": "I cannot do that"}}]})
    with pytest.raises(RuntimeError) as e:
        um.gen_image("a fox")
    assert "no image in chat reply" in str(e.value) and "gpt-image-1" in str(e.value)


def test_chat_transcription_uses_chat_audio_block(monkeypatch, tmp_path):
    monkeypatch.undo()
    _chat_conf(monkeypatch, tmp_path)
    sent = {}

    def post(path, body, ctype, timeout):
        sent.update(path=path, body=um.json.loads(body))
        return {"choices": [{"message": {"content": " hello there \n"}}]}
    monkeypatch.setattr(um, "_post", post)
    assert um.transcribe(b"RIFFxxxx", "wav") == ("hello there", "gemini/flash")
    block = sent["body"]["messages"][0]["content"][1]
    assert sent["path"] == "/chat/completions" and block["input_audio"]["format"] == "wav"


def test_native_transcription_still_uses_multipart(monkeypatch, tmp_path):
    monkeypatch.undo()
    _chat_conf(monkeypatch, tmp_path)
    calls = []

    def post(path, body, ctype, timeout):
        calls.append((path, ctype))
        if path == "/chat/completions":
            raise RuntimeError("gemini down")
        return {"text": "hi"}
    monkeypatch.setattr(um, "_post", post)
    assert um.transcribe(b"RIFFxxxx", "wav")[0] == "hi"
    assert calls[1][0] == "/audio/transcriptions" and calls[1][1].startswith("multipart/")


def test_audio_chat_prefix_transcribes_through_chat(monkeypatch, tmp_path):
    # a model that makes images through /images but takes audio only in chat
    monkeypatch.undo()
    f = tmp_path / "media.json"
    f.write_text(json.dumps({"transcribe": ["vendor/flash-lite", "whisper-1"], "audio_chat_prefixes": ["vendor/"]}))
    monkeypatch.setattr(um, "CONF_FILE", str(f))
    sent = []

    def post(path, body, ctype, timeout):
        sent.append(path)
        return {"choices": [{"message": {"content": "hi"}}]}
    monkeypatch.setattr(um, "_post", post)
    assert um.transcribe(b"RIFFxxxx", "wav") == ("hi", "vendor/flash-lite") and sent == ["/chat/completions"]
    assert not um._via_chat("vendor/flash-lite")  # images from the same vendor still use /images


def test_direct_models_are_never_inspected(sandbox):
    # the image judge gets two images and a prompt, which can read like an edit request
    d = run(um.Media(), req("generate an image of a fox", model="ultron/judge"))
    assert "mock_response" not in d and sandbox == []


def test_defaults_are_plain_endpoint_ids():
    assert um.DEFAULTS["video"] and um.DEFAULTS["chat_prefixes"] == []
    # plain model names any OpenAI-compatible endpoint can serve, never one router's provider/model ids
    assert not any("/" in m for k in ("image", "edit", "video", "transcribe") for m in um.DEFAULTS[k])


@pytest.mark.parametrize("reply,want", [("MEDIA", True), ("media.", True), ("OTHER", False), ("", False)])
def test_helper_confirm_asks_without_thinking(monkeypatch, reply, want):
    sent = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": reply}}]}).encode()

    sent_headers = []

    def urlopen(req, timeout):
        sent.append(json.loads(req.data))
        sent_headers.append(dict(req.header_items()))
        return Resp()
    monkeypatch.setattr(um.urllib.request, "urlopen", urlopen)
    assert REAL_CONFIRM("Generate an image of a cat", "image") is want
    # with thinking on, the reasoning used up max_tokens and every reply was empty (= no)
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent_headers[0]["X-mtplx-cache-mode"] == "bypass"  # one-shot: mtplx shouldn't bank it
    assert "MEDIA or OTHER" in sent[0]["messages"][0]["content"]


def test_helper_confirm_fails_closed(monkeypatch):
    def urlopen(req, timeout):
        raise OSError("helper down")
    monkeypatch.setattr(um.urllib.request, "urlopen", urlopen)
    assert REAL_CONFIRM("Generate an image of a cat", "image") is False


def test_video_tries_each_model_then_writes_the_error(sandbox, tmp_path, monkeypatch):
    (tmp_path / "none.json").write_text(json.dumps({"video": ["provider-a/video", "provider-b/video"]}))
    tried = []

    def post(path, body, ctype, timeout):
        tried.append(json.loads(body)["model"])
        raise RuntimeError("The read operation timed out")
    monkeypatch.setattr(um, "_post", post)
    um.gen_video("a paper boat", "job1")
    assert tried == ["provider-a/video", "provider-b/video"]
    err = (tmp_path / "media" / "job1.error.txt").read_text()
    assert all(m in err for m in tried)
    assert '"video_failed"' in (tmp_path / "media.jsonl").read_text()
