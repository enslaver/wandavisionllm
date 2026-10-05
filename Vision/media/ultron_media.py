"""ultron_media: send media requests to the cloud endpoint instead of a chat model.

LiteLLM pre-call hook for ultron, registered after loop_breaker and before ultron_admit. Looks at
the newest HUMAN turn of a chat (/v1/chat/completions or /v1/messages) and, when it asks for media,
does the work on the cloud endpoint (OMNIROUTE_BASE in ~/.litellm/env: any OpenAI-compatible API;
video and search need endpoints OmniRoute has) instead of the local tier:

    image      "generate an image of ..."     -> /v1/images/generations   reply: the image link
    image edit an image attached + "edit ..." -> /v1/images/edits         reply: the image link
    video      "make a video of ..."          -> /v1/videos/generations   reply: link now, file when done
    search     "search the web for ..."       -> /v1/search               results added to the prompt
    audio      an input_audio block           -> /v1/audio/transcriptions transcript replaces the audio
                                                 (or /v1/chat/completions, see audio_chat_prefixes)

Replies (image, edit, video) are returned without a backend call via mock_response, the way
loop_breaker answers a stop. Search and audio only add text; the tier still answers.
Embeddings have no chat form: they are a plain /v1/embeddings entry in config.yaml.

Safety, because a wrong match in a coding agent's loop is worse than a missed one:
  - only the newest turn, and only when a person typed it: a tool_result turn is never inspected
  - requests that carry tools (Claude Code, pi, Hermes) need the local helper tier
    (tiers.conf [routing] helper) to answer MEDIA on top of the regex
    (fail closed: no answer or an error means no interception)
  - Claude Code's helper calls (a session header but no tools: titles, topic checks) are skipped
  - the same prompt in the same conversation is answered once (10 min), so retries don't re-bill
  - a veto list ("docker image", "video player", "svg component", ...) beats every pattern

Mode: ~/.ultron/media-mode = shadow | enforce | off (read per request; default shadow, which only
logs what it would have done). Per-request opt-out: header `x-media: off`. Log: ~/.litellm/media.jsonl.
Models and the public link base: ~/.ultron/media.json (optional; litellm/media.example.json), see
DEFAULTS; the endpoint and link base default to OMNIROUTE_BASE and ULTRON_MEDIA_PUBLIC from ~/.litellm/env.
Files: ~/.ultron/media/, served by Caddy at /media/.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

import loop_breaker as lb
import ultron_tiers

HOME = os.path.expanduser("~")
MODE_FILE = os.path.expanduser(os.environ.get("ULTRON_MEDIA_MODE_FILE", "~/.ultron/media-mode"))
CONF_FILE = os.path.expanduser(os.environ.get("ULTRON_MEDIA_CONF", "~/.ultron/media.json"))
LOG_PATH = os.path.expanduser(os.environ.get("ULTRON_MEDIA_LOG", "~/.litellm/media.jsonl"))
MEDIA_DIR = os.path.expanduser(os.environ.get("ULTRON_MEDIA_DIR", "~/.ultron/media"))

DEFAULTS: dict[str, Any] = {
    "base": os.environ.get("OMNIROUTE_BASE", "http://127.0.0.1:20128/v1"),
    "public": os.environ.get("ULTRON_MEDIA_PUBLIC", "http://localhost/media"),
    # Tried in order; the first that answers wins. Ids are whatever the endpoint serves: these are
    # OpenAI's names. Video can take minutes, so it runs in a thread.
    "image": ["gpt-image-1"],
    "edit": ["gpt-image-1"],
    "video": ["sora-2"],
    "transcribe": ["whisper-1"],
    # Ids the endpoint only serves through /chat/completions (image or audio models that answer in
    # chat), e.g. ["gemini/"] on OmniRoute. They are sent as chat messages instead of /images, /audio.
    "chat_prefixes": [],
    # Ids that make images through /images but transcribe only as a chat input_audio block (chat models
    # that take audio input; OmniRoute's /audio/transcriptions refuses them).
    "audio_chat_prefixes": [],
    "search_results": 5,
    "agents": True,  # act on tool-carrying requests (needs the helper tier's confirmation)
    "helper": "http://127.0.0.1:8001/v1",  # llama-swap: the helper tier answers the yes/no check
}
TIMEOUT = {"image": 150, "edit": 240, "video": 420, "transcribe": 120, "search": 30, "confirm": 8}
DEDUPE_S = 600


def conf() -> dict[str, Any]:
    out = dict(DEFAULTS)
    try:
        with open(CONF_FILE) as f:
            out.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return out


def current_mode() -> str:
    try:
        v = open(MODE_FILE).read().strip().lower()
    except OSError:
        v = ""
    return v if v in ("enforce", "shadow", "off") else os.environ.get("ULTRON_MEDIA_MODE", "shadow").lower()


# ----------------------------------------------------------------------------- reading the prompt

_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
_IDE = re.compile(r"<ide_[a-z_]+>.*?</ide_[a-z_]+>", re.S)


def human_turn(data: dict[str, Any]) -> dict[str, Any] | None:
    """The newest user message as {text, images, audio}, or None when it isn't a fresh human prompt
    (last turn is not a user turn, it carries a tool_result, or the shape is unknown)."""
    msgs = data.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    last = msgs[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return None
    content = last.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return None
    texts: list[str] = []
    images: list[tuple[bytes, str]] = []
    audio: list[int] = []  # indexes into content
    for i, b in enumerate(content):
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "tool_result":
            return None
        if t in ("text", "input_text"):
            texts.append(str(b.get("text") or ""))
        elif t == "image_url":
            url = (b.get("image_url") or {}).get("url") if isinstance(b.get("image_url"), dict) else b.get("image_url")
            img = _data_url(str(url or ""))
            if img:
                images.append(img)
        elif t == "image":
            src = b.get("source") or {}
            if src.get("type") == "base64" and src.get("data"):
                try:
                    images.append((base64.b64decode(src["data"]), src.get("media_type") or "image/png"))
                except ValueError:
                    pass
        elif t == "input_audio" and isinstance(b.get("input_audio"), dict):
            audio.append(i)
    text = _IDE.sub("", _REMINDER.sub("", "\n".join(texts))).strip()
    return {"text": text, "images": images, "audio": audio, "content": content}


def _data_url(url: str) -> tuple[bytes, str] | None:
    m = re.match(r"data:([\w/+.-]+);base64,(.*)$", url, re.S)
    if not m:
        return None
    try:
        return base64.b64decode(m.group(2)), m.group(1)
    except ValueError:
        return None


# ----------------------------------------------------------------------------- classifier

_VERBS = r"(?:generate|create|make|draw|paint|render|produce|design|illustrate|imagine|sketch|give me|show me|i want|i need|can you (?:make|draw|do)|please (?:make|draw))"
IMAGE_RE = re.compile(
    r"(?:^|[\s,;:(])/imagine\b"
    r"|\b" + _VERBS + r"\b[^.\n]{0,60}?\b(?:image|picture|photo|photograph|illustration|artwork|painting|drawing|portrait"
    r"|wallpaper|poster|render|pic|sketch)s?\b",
    re.I,
)
VIDEO_RE = re.compile(
    r"\b(?:generate|create|make|render|produce|animate|give me|show me)\b[^.\n]{0,60}?\b(?:video|clip|animation|footage|short film|movie|gif)s?\b",
    re.I,
)
EDIT_RE = re.compile(
    r"\b(?:edit|retouch|photoshop|inpaint|outpaint|restyle|recolou?r|colou?rize|remove (?:the )?background|cartoonize|upscale)\b"
    r"|\b(?:change|make|turn|convert|modify|replace|add|remove|put)\b[^.\n]{0,60}?\b(?:in|on|from|of|to|into) (?:the|this|that|my) (?:image|photo|picture|pic|portrait|selfie)\b"
    r"|\b(?:change|make|turn|convert) (?:this|the|that) (?:image|photo|picture|pic)\b",
    re.I,
)
SEARCH_RE = re.compile(
    r"\b(?:search|google|browse|look (?:it |this |that )?up|find out)\b[^.\n]{0,20}?\b(?:the )?(?:web|internet|online)\b"
    r"|\bweb search\b|\bgoogle (?:it|this|that|for)\b|\bsearch online\b"
    r"|\b(?:latest|today'?s|breaking|current) news (?:on|about|for)\b",
    re.I,
)
# Words that turn a match into a coding/devops sentence, not an art request.
VETO_RE = re.compile(
    r"\b(?:docker|container|base|disk|vm|iso|ami|qcow2|golden|boot)\s+images?\b|\bimages?\s+(?:tag|build|pull|push|registry|layer|size|name|digest)\b"
    r"|\b(?:component|function|method|class|script|code|snippet|module|endpoint|api|unit test|test|regex|css|html|svg|json|yaml|dockerfile|readme|pr|commit|branch|diff|bug|error)\b"
    r"|\b(?:video|image)\s+(?:player|element|tag|component|loader|processing|pipeline|codec|encoder|decoder|file|format|url|src|element)\b"
    r"|\bffmpeg\b|\bscreen ?(?:shot|recording)\b|\btranscod|\bimagemagick\b|\bpillow\b|\bopencv\b",
    re.I,
)
# A question about media rather than a request for it ("explain how to generate images ...").
INFO_RE = re.compile(r"^\W*(?:explain|describe|how (?:do|does|did|can|could|would|should|to)|why|what(?:'s| is| are| was)|"
                     r"tell me (?:about|how|why)|teach|is it|are there|which|who|when|where)\b", re.I)
SEARCH_VETO_RE = re.compile(r"\b(?:codebase|repo|repository|file|files|grep|project)\b", re.I)


def classify(turn: dict[str, Any]) -> str | None:
    """image | edit | video | search | audio | None. Regex only: cheap, and precision beats recall."""
    if turn["audio"]:
        return "audio"
    text = turn["text"]
    if not text or len(text) > 4000:
        return None
    veto = VETO_RE.search(text) or INFO_RE.search(text)
    if turn["images"]:
        return "edit" if EDIT_RE.search(text) and not veto else None
    if VIDEO_RE.search(text) and not veto:
        return "video"
    if IMAGE_RE.search(text) and not veto:
        return "image"
    if SEARCH_RE.search(text) and not SEARCH_VETO_RE.search(text):
        return "search"
    return None


# ----------------------------------------------------------------------------- cloud endpoint calls

_SSL = ssl.create_default_context()


def _key() -> str:
    return os.environ.get("OMNIROUTE_KEY", "")


def _post(path: str, body: bytes, ctype: str, timeout: float) -> Any:
    req = urllib.request.Request(
        conf()["base"] + path, body, {"Authorization": "Bearer " + _key(), "Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode("utf-8", "replace")
        raise RuntimeError(f"{path} {e.code}: {detail}") from None


def _multipart(fields: dict[str, str], files: list[tuple[str, str, str, bytes]]) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    out = b""
    for k, v in fields.items():
        out += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for name, fn, ct, raw in files:
        out += f'--{b}\r\nContent-Disposition: form-data; name="{name}"; filename="{fn}"\r\nContent-Type: {ct}\r\n\r\n'.encode()
        out += raw + b"\r\n"
    return out + f"--{b}--\r\n".encode(), "multipart/form-data; boundary=" + b


def sniff(raw: bytes) -> tuple[str, str]:
    """(extension, mime) from magic bytes: some providers return JPEG under a .png name, and
    codex rejects an edit whose declared type doesn't match the bytes."""
    if raw[:3] == b"\xff\xd8\xff":
        return "jpg", "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png", "image/png"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp", "image/webp"
    if raw[4:8] == b"ftyp":
        return "mp4", "video/mp4"
    if raw[:4] == b"\x1a\x45\xdf\xa3":
        return "webm", "video/webm"
    if raw[:3] == b"GIF":
        return "gif", "image/gif"
    return "bin", "application/octet-stream"


def _payload_bytes(item: dict[str, Any]) -> bytes:
    """An image/video item from the endpoint: b64_json, a data: URL, or an http(s) URL."""
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    url = str(item.get("url") or item.get("video_url") or "")
    d = _data_url(url)
    if d:
        return d[0]
    if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=120, context=_SSL) as r:
            return r.read()
    raise RuntimeError("no image/video payload in response: keys " + ",".join(sorted(item)))


def _first_item(resp: Any) -> dict[str, Any]:
    if isinstance(resp, dict):
        for k in ("data", "videos", "results", "output"):
            v = resp.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v[0]
        if any(k in resp for k in ("b64_json", "url", "video_url")):
            return resp
    raise RuntimeError("unexpected response shape: " + str(resp)[:200])


def save(raw: bytes, ext: str | None = None) -> str:
    os.makedirs(MEDIA_DIR, exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.{ext or sniff(raw)[0]}"
    with open(os.path.join(MEDIA_DIR, name), "wb") as f:
        f.write(raw)
    return name


def link(name: str) -> str:
    return conf()["public"].rstrip("/") + "/" + name


def _direct_model(model: Any) -> bool:
    """A tiers.conf model with `routed = no` (served only when asked for by name, like the image judge)."""
    try:
        t = ultron_tiers.load()
        name = t.tier_for(str(model or ""))
        return bool(name) and not t.tier[name]["routed"]
    except Exception:
        return False


def _via_chat(model: str) -> bool:
    return model.startswith(tuple(conf()["chat_prefixes"]))


def _chat_image(model: str, prompt: str, image: tuple[bytes, str] | None, timeout: float) -> bytes:
    """Gemini image models answer /chat/completions with image_url parts holding data: URLs (usually two
    near-identical ones); the first is kept."""
    text = prompt[:4000]
    content: Any = text
    if image:
        raw, mime = image
        content = [{"type": "text", "text": text + "\n\nReturn the edited image."},
                   {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(raw).decode()}}]
    resp = _post("/chat/completions", json.dumps(
        {"model": model, "messages": [{"role": "user", "content": content}]}).encode(), "application/json", timeout)
    msg = ((resp.get("choices") or [{}])[0].get("message") or {}) if isinstance(resp, dict) else {}
    parts = msg.get("content") if isinstance(msg.get("content"), list) else []
    for part in parts + list(msg.get("images") or []):
        d = _data_url(str((part.get("image_url") or {}).get("url") or "")) if isinstance(part, dict) else None
        if d:
            return d[0]
    raise RuntimeError("no image in chat reply: " + str(msg.get("content"))[:150])


def gen_image(prompt: str) -> tuple[str, str]:
    """(file name, model). Tries each configured model in order."""
    errs = []
    for model in conf()["image"]:
        try:
            if _via_chat(model):
                return save(_chat_image(model, prompt, None, TIMEOUT["image"])), model
            resp = _post("/images/generations", json.dumps(
                {"model": model, "prompt": prompt[:4000], "n": 1, "size": "1024x1024"}).encode(),
                "application/json", TIMEOUT["image"])
            return save(_payload_bytes(_first_item(resp))), model
        except Exception as e:  # try the next model
            errs.append(f"{model}: {e}")
    raise RuntimeError("; ".join(errs))


def edit_image(prompt: str, image: tuple[bytes, str]) -> tuple[str, str]:
    raw, _declared = image
    ext, mime = sniff(raw)
    errs = []
    for model in conf()["edit"]:
        try:
            if _via_chat(model):
                return save(_chat_image(model, prompt, (raw, mime), TIMEOUT["edit"])), model
            body, ct = _multipart({"model": model, "prompt": prompt[:4000]}, [("image", f"in.{ext}", mime, raw)])
            resp = _post("/images/edits", body, ct, TIMEOUT["edit"])
            return save(_payload_bytes(_first_item(resp))), model
        except Exception as e:
            errs.append(f"{model}: {e}")
    raise RuntimeError("; ".join(errs))


def gen_video(prompt: str, job: str) -> None:
    """Runs in a thread (minutes long). Writes <job>.mp4 when done, else <job>.error.txt."""
    errs = []
    for model in conf()["video"]:
        try:
            resp = _post("/videos/generations", json.dumps({"model": model, "prompt": prompt[:4000]}).encode(),
                         "application/json", TIMEOUT["video"])
            raw = _payload_bytes(_first_item(resp))
            os.makedirs(MEDIA_DIR, exist_ok=True)
            tmp = os.path.join(MEDIA_DIR, job + ".part")
            with open(tmp, "wb") as f:
                f.write(raw)
            os.replace(tmp, os.path.join(MEDIA_DIR, f"{job}.{sniff(raw)[0] if sniff(raw)[0] != 'bin' else 'mp4'}"))
            _log({"ts": time.time(), "event": "video_done", "job": job, "model": model, "bytes": len(raw)})
            return
        except Exception as e:
            errs.append(f"{model}: {e}")
    os.makedirs(MEDIA_DIR, exist_ok=True)
    with open(os.path.join(MEDIA_DIR, job + ".error.txt"), "w") as f:
        f.write("; ".join(errs))
    _log({"ts": time.time(), "event": "video_failed", "job": job, "error": "; ".join(errs)[:500]})


def web_search(query: str) -> str:
    resp = _post("/search", json.dumps({"query": query[:300]}).encode(), "application/json", TIMEOUT["search"])
    lines = []
    for i, r in enumerate((resp.get("results") or [])[: int(conf()["search_results"])], 1):
        snip = re.sub(r"\s+", " ", str(r.get("snippet") or ""))[:400]
        lines.append(f"{i}. {r.get('title', '')} <{r.get('url', '')}>\n   {snip}")
    if not lines:
        raise RuntimeError("no results")
    return "\n".join(lines)


def transcribe(raw: bytes, fmt: str) -> tuple[str, str]:
    errs = []
    for model in conf()["transcribe"]:
        try:
            if _via_chat(model) or model.startswith(tuple(conf()["audio_chat_prefixes"])):
                resp = _post("/chat/completions", json.dumps({"model": model, "max_tokens": 4000, "messages": [{
                    "role": "user", "content": [
                        {"type": "text", "text": "Transcribe this audio verbatim. Output only the transcript."},
                        {"type": "input_audio", "input_audio": {"data": base64.b64encode(raw).decode(), "format": fmt}}]}]}
                ).encode(), "application/json", TIMEOUT["transcribe"])
                text = str(resp["choices"][0]["message"].get("content") or "").strip()
                if not text:
                    raise RuntimeError("empty transcript")
                return text, model
            body, ct = _multipart({"model": model}, [("file", f"in.{fmt}", f"audio/{fmt}", raw)])
            resp = _post("/audio/transcriptions", body, ct, TIMEOUT["transcribe"])
            return str(resp.get("text") or "").strip(), model
        except Exception as e:
            errs.append(f"{model}: {e}")
    raise RuntimeError("; ".join(errs))


CONFIRM_ASK = {
    "image": "a new image (picture, photo, drawing, artwork) to be generated, with the image itself as the answer",
    "video": "a new video clip to be generated, with the video itself as the answer",
    "edit": "the attached image to be edited, with the edited image itself as the answer",
    "search": "a live web search for current information, with the search results as the answer",
}


def helper_confirm(text: str, kind: str) -> bool:
    """Ask the local helper tier (small, always loaded) whether an agent's message really wants
    media made. Fail closed: any error, timeout or unclear answer is a no.
    Thinking off: with it on, a Qwen3.5 helper's reasoning used up max_tokens and the empty reply
    read as no every time (2026-10-01: no agent request was ever confirmed). The old "Answer YES or
    NO" prompt then said no to everything; MEDIA/OTHER scored 19/22 on a hand set (all positives,
    3 false positives)."""
    q = (f"Message: <<<{text[:1500]}>>>\n\nIs this message asking for {CONFIRM_ASK.get(kind, kind)}? Requests to "
         f"write or change code, files, UI, Docker images, HTML or programs are not. Reply with exactly one word: "
         f"MEDIA or OTHER.")
    try:
        helper = ultron_tiers.load().helper
        body = json.dumps({"model": helper, "max_tokens": 4, "temperature": 0,
                           "chat_template_kwargs": {"enable_thinking": False},
                           "messages": [{"role": "user", "content": q}]}).encode()
        req = urllib.request.Request(conf()["helper"] + "/chat/completions", body,
                                     {"Content-Type": "application/json", "Authorization": "Bearer none",
                                      "x-mtplx-cache-mode": "bypass"})  # one-shot: don't bank it (CACHE_BYPASS in ultron_admit)
        with urllib.request.urlopen(req, timeout=TIMEOUT["confirm"]) as r:
            out = json.loads(r.read())["choices"][0]["message"].get("content") or ""
        return out.strip().upper().startswith("MEDIA")
    except Exception:
        return False


# ----------------------------------------------------------------------------- the hook

def _headers(data: dict[str, Any]) -> dict[str, str]:
    h = (data.get("proxy_server_request") or {}).get("headers") or {}
    return {str(k).lower(): str(v) for k, v in h.items()}


def _opted_out(data: dict[str, Any]) -> bool:
    md = data.get("metadata") or {}
    return _headers(data).get("x-media", "").lower() == "off" or str(md.get("media", "")).lower() == "off"


def _log(entry: dict[str, Any]) -> None:
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass


def set_reply(data: dict[str, Any], call_type: str, text: str) -> bool:
    """Answer this request with `text` and no backend call (same route as loop_breaker's stop)."""
    if lb._is_anthropic(data, call_type) and data.get("stream"):
        if not lb._ANTHROPIC_MOCK_STREAMS:
            return False
        data["mock_response"] = lb.STREAM_MARK + text
    else:
        data["mock_response"] = text
    return True


def _reply_text(kind: str, prompt: str, name: str | None, model: str, job: str | None = None) -> str:
    if kind == "video":
        return (f"Video started with `{model}`: {prompt[:120]!r}\n\nIt renders in the cloud and usually takes a few minutes. "
                f"It will appear at {link(job + '.mp4')} (if it never does, the reason is at {link(job + '.error.txt')}).")
    verb = "Edited image" if kind == "edit" else "Image"
    return f"{verb} from `{model}`:\n\n![{prompt[:60]}]({link(name or '')})\n\n{link(name or '')}"


class Media:
    def __init__(self) -> None:
        self.done: dict[str, tuple[float, asyncio.Future]] = {}

    def _dedupe(self, key: str) -> asyncio.Future | None:
        now = time.time()
        for k in [k for k, (t, _) in self.done.items() if now - t > DEDUPE_S]:
            del self.done[k]
        hit = self.done.get(key)
        return hit[1] if hit else None

    async def handle(self, data: dict[str, Any], call_type: str, mode: str) -> None:
        if mode == "off" or data.get("mock_response") or _opted_out(data) or call_type not in (
                "completion", "acompletion", "anthropic_messages", "text_completion"):
            return
        if _direct_model(data.get("model")):
            return  # e.g. the image judge (Vision/judge/rank.py): two images and a prompt can read like an edit request
        turn = human_turn(data)
        if not turn:
            return
        h = _headers(data)
        agent = bool(data.get("tools"))
        if not agent and h.get("x-claude-code-session-id"):
            return  # Claude Code helper call (title / topic): the same text as the real prompt, no tools
        kind = classify(turn)
        if not kind:
            return
        conv = lb.conversation_key(data)
        entry = {"ts": time.time(), "mode": mode, "kind": kind, "agent": agent, "key": conv,
                 "model": data.get("model"), "text": turn["text"][:160], "call_id": data.get("litellm_call_id")}
        t0 = time.perf_counter()
        try:
            if agent:
                if not conf()["agents"] or kind == "audio":
                    entry["skipped"] = "agents off"
                    return
                entry["confirmed"] = await asyncio.to_thread(helper_confirm, turn["text"], kind)
                if not entry["confirmed"]:
                    entry["skipped"] = "helper said no"
                    return
            if mode != "enforce":
                entry["applied"] = False
                return
            if kind in ("image", "edit", "video"):  # search/audio edit this request's own messages: never shared
                dk = f"{conv}:{kind}:{hashlib.sha256(turn['text'].encode()).hexdigest()[:20]}"
                fut = self._dedupe(dk)
                if fut is None:
                    fut = asyncio.ensure_future(self._do(kind, turn, data))
                    self.done[dk] = (time.time(), fut)
                else:
                    entry["deduped"] = True
                result = await asyncio.shield(fut)
            else:
                result = await self._do(kind, turn, data)
            entry.update({"applied": True, **{k: v for k, v in result.items() if k != "reply"}})
            if result.get("reply") is not None:
                if not set_reply(data, call_type, result["reply"]):
                    entry["applied"], entry["skipped"] = False, "no streaming mock patch"
        except Exception as exc:  # the proxy must never fail a request because of this hook
            entry["error"] = repr(exc)[:400]
            entry["applied"] = False
        finally:
            entry["ms"] = round((time.perf_counter() - t0) * 1000)
            _log(entry)

    async def _do(self, kind: str, turn: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        text = turn["text"]
        if kind == "image":
            name, model = await asyncio.to_thread(gen_image, text)
            return {"reply": _reply_text(kind, text, name, model), "model_used": model, "file": name}
        if kind == "edit":
            name, model = await asyncio.to_thread(edit_image, text, turn["images"][-1])
            return {"reply": _reply_text(kind, text, name, model), "model_used": model, "file": name}
        if kind == "video":
            job = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            threading.Thread(target=gen_video, args=(text, job), daemon=True).start()
            return {"reply": _reply_text(kind, text, None, conf()["video"][0], job), "model_used": conf()["video"][0], "file": job}
        if kind == "search":
            results = await asyncio.to_thread(web_search, text)
            note = ("Web search results for the user's request, fetched just now. Use them to answer and cite the "
                    "URLs:\n" + results)
            lb._append_note(data["messages"], note, lb._is_anthropic(data, "acompletion"))
            return {"model_used": "cloud/search"}
        if kind == "audio":
            content = turn["content"]
            model_used = ""
            for i in turn["audio"]:
                blk = content[i]["input_audio"]
                said, model_used = await asyncio.to_thread(
                    transcribe, base64.b64decode(blk.get("data") or ""), str(blk.get("format") or "wav"))
                content[i] = {"type": "text", "text": f"[Transcript of the attached audio: {said}]"}
            return {"model_used": model_used}
        raise RuntimeError("unknown kind " + kind)


try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # tests without litellm installed
    CustomLogger = object  # type: ignore[misc,assignment]


class UltronMedia(CustomLogger):  # type: ignore[misc,valid-type]
    def __init__(self) -> None:
        super().__init__()
        self.media = Media()

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            await self.media.handle(data, call_type, current_mode())
        except Exception as exc:
            _log({"ts": time.time(), "error": repr(exc)})
        return data


proxy_handler_instance = UltronMedia()
