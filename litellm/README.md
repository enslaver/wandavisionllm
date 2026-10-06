# LiteLLM hooks

Five hooks that run inside LiteLLM, so they apply to every client behind `:4000` (Claude
Code, pi, Hermes, opencode, Unreal Engine MCP harnesses, plain OpenAI SDK scripts) on both
`/v1/chat/completions` and `/v1/messages`, with no client changes.

```
client ──► LiteLLM :4000 ──[ultron_stats → loop_breaker → ultron_media → ultron_admit]──► llama-swap :8001 ──► tiers
                                                                                     └──► cloud/<tier> (OmniRoute)
client ◄── [ultron_rescue] ◄── reply
```

They are registered in `config.yaml` under `litellm_settings.callbacks`. The order matters: stats
first so its clock starts on arrival, the loop breaker edits the request, admission picks the
target last. `ultron_rescue` works on the reply instead, on its way back to the client. Everything deployed from here is Python 3.9 standard library only.

| Hook | What it does | Mode switch | Log |
|---|---|---|---|
| `ultron_stats.py` | per-request stats and in-flight list for every backend | always on (observe-only) | `~/.litellm/ultron-stats.jsonl` |
| `loop_breaker.py` | stops agents repeating the same tool call | `~/.ultron/loop-breaker-mode` | `~/.litellm/loop-breaker.jsonl` |
| [`ultron_media.py`](../Vision/media/) | answers "make an image / video", web search, audio via OmniRoute; lives in `Vision/media/`, deployed here | `~/.ultron/media-mode` | `~/.litellm/media.jsonl` |
| `ultron_admit.py` | local tier vs cloud overflow per conversation, then pins it | `~/.ultron/admit-mode`, `~/.ultron/route-mode` | `~/.litellm/ultron-admit.jsonl` |
| `ultron_rescue.py` | turns a ```bash block a local tier wrote instead of a tool call into a tool call | `~/.ultron/rescue-mode` | `~/.litellm/rescue.jsonl` |

Mode files hold one word (`enforce`, `shadow` or `off`) and are re-read on every request, so a flip
from the Wanda panel applies to the next request with no restart. When a file is missing the hook
falls back to its env var in `~/.litellm/env` (`LOOP_BREAKER_MODE`, `ULTRON_MEDIA_MODE`,
`ULTRON_ADMIT_MODE`, `ULTRON_RESCUE_MODE`).

## loop_breaker.py — stop repeated tool calls

Every request carries the agent's full history, so the hook is stateless. It walks the trailing
tool steps and counts how many times in a row the model made the **same call** and got the **same
result**:

- **Same call:** tool name + canonical args. Keys are sorted and whitespace is trimmed; Bash
  `description`/`reason` are ignored. **Digits are kept**, because node ids, offsets and screenshot
  indexes change legitimately.
- **Same result:** the result after masking timestamps, clocks, durations, `execution_count`,
  uuids, hex ids and countdowns (`in 3641s`).
- **What resets the count:** any change in the call or its result, or a real user turn (a message a
  person typed, not tool results and not the hook's own note). If the model loops again after that,
  it is caught again.

| Rule | Warn (note appended) | Force (`tool_choice: none`) | Stop (turn answered by the proxy) |
|---|---|---|---|
| default | 4 | 6 | 8 |
| poll (`poll_*`, `*status*`, PIE input/probe, `pump`, sleep/wait/wakeup, or a busy/running result) | 10 | 16 | 20 |
| same error every time | 3 | 4 | 6 |
| A→B(→C) cycle, unchanged results | 5 | 6 | 8 |
| cycle containing a poll step | 10 | 16 | 20 |

**Where the thresholds come from:** about 430k real tool calls from two Unreal Engine MCP harnesses,
pi, Claude Code and Hermes. The longest run of legitimate identical call + result was 3 for ordinary
tools and 12 for polls. Small local models loop far more often than frontier models (one pi session
repeated the same `bash` call 4,909 times over 9.8 hours), but frontier models loop too.

**How each intervention works:**

- **warn / force** only append to the end of the history, so the backend's prefix cache for 100k+
  prompts survives. mtplx 2.12 keeps the tool schema in the prompt under `tool_choice: none`.
- **force wording** lists the repeated calls (shell tools by their command) and asks for a
  *different* tool call; only if nothing else can work, tell the user. It used to say "Do not call
  `Bash` again. Stop and reply to the user now, in text", which put a sonnet-tier coding agent into
  a 6-hour text-only stall. Replays of that moment: old wording 0/3 tool calls; new wording 3/4
  changed approach, and the 4th stalled in text, which `ultron_rescue.py` turns into a tool call.
- **stop** uses LiteLLM's `mock_response`, so there is no backend call and no HTTP error (clients
  retry errors). LiteLLM's `/v1/messages` mock ignores `stream: true`, so stop replies are wrapped in
  LiteLLM's own `FakeAnthropicMessagesStreamIterator` and streaming clients get proper SSE. If a
  LiteLLM upgrade moves those internals, the hook falls back to a model-written stop message.

**Opt out per request:** header `x-loop-breaker: off`, or metadata `{"loop_breaker": "off"}`.

## ultron_admit.py — local first, cloud overflow, per-conversation pins

On a conversation's first request it picks one of:

1. `ultron/<tier>` — the tier is loaded with a short queue, or it fits cold next to what's loaded
   (llama-swap matrix + main-thread pins).
2. `ultron/<substitute>` — sonnet→opus or haiku→sonnet, when that tier is loaded and not busy.
3. `cloud/<tier>` — the OmniRoute combos in `config.yaml`.

It pins that choice in `~/.litellm/pins.sqlite`, keyed on Claude Code session + agent id + tier (or
a sha256 of the first user message for other clients), so a conversation never switches backend
mid-way (switching throws away the prompt cache and changes the model's voice), with one exception.

**Soft overflow pins.** A `cloud/<tier>` pin made only because the Mac was full (rule
`4:overflow:mem|queue|no-fit`) re-runs the decision on every request. Once a local tier fits, the
conversation re-pins local (logged as `repin:<rule>`, e.g. `repin:1:loaded`) and is then an ordinary
local pin. While still full, the overflow pin stands (`pinned:4:overflow:*`). Without this, one memory
spike sent an agent run to the cloud for the rest of its life (pins expire after an hour idle).
Explicit cloud (`x-route: cloud`, route mode `cloud-only`) stays a hard pin. The first local request
after a re-pin prefills the whole conversation (no prefix cache yet).

**Cloud is the last resort.** A new conversation joins a loaded tier's queue up to its `max_waiting`
(tiers.conf, default 20; haiku 50 in the example) before it overflows, and a request that would make
llama-swap unload a busy tier waits up to `OVERFLOW_WAIT_S` (300 s) before that one request goes to
the cloud. Overflow is one-way: there are no `cloud/<tier>` → `ultron/<tier>` fallbacks in
`config.yaml`. Those re-ran every failed cloud call on the local tier, ignoring route mode
`cloud-only` (with the cloud endpoint failing on a missing key, every cloud request was served
locally).

- **Overrides:** header `x-route: cloud` sends the request to the combo; `x-route: private` keeps it
  local. `~/.ultron/route-mode` is `auto`, `local-only` or `cloud-only`.
- **Modes:** `shadow` logs decisions without rewriting (explicit overrides still apply), `enforce`
  rewrites `data["model"]`, `off` does nothing.
- **Visibility:** every response carries an `x-ultron-route` header; decisions go to Wanda's ADMIT tab.

### Vision routing

Tiers with `vision = no` in `tiers.conf` (fable, opus and haiku in the example set) can't see
images; `[routing] vision` (sonnet) can. A request whose newest turn carries an image and would go
to a no-vision tier is rerouted to the vision tier for that one request (the pin is unchanged);
images in older turns are replaced with a text note so the model doesn't invent them.

A small tier with a vision forge makes the better vision tier: on a 16-check eval (game screenshots
and generated-image review), a Qwen3.5-4B haiku forge scored 15/16 at a 1.6 s median; the 9B sonnet
scored 10/16 at 21 s (5 answers cut off at 4,096 tokens while thinking); the 35B-A3B opus 13/16 at
3.7 s. The example haiku pack is text only; [`mtplx/bin/tier-haiku.sh`](../mtplx/bin/tier-haiku.sh)
says how to forge one with the vision tower, then set `vision = yes` for haiku and `vision = haiku`.

### Eviction safety

With the three-resident matrix (`opus & sonnet & haiku`) requests load alongside what's loaded
instead of evicting it. Two guards stay in case you restrict the matrix (smaller Mac, bigger models):

- A pin only reserves memory while its tier is actually resident.
- A tier a main-thread conversation used within the last `WARM_MAIN_S` (300 s) counts as in-flight,
  so a request that *would* evict it waits, then overflows to `cloud/<tier>` instead of killing a
  live session.

### Memory guard

A long prefill grows KV caches until macOS compresses and swaps model weights, and the kernel's
pressure level is a poor early warning (it read 1 "normal" while ~9 GB was swapped out in three
minutes). The guard trips on any of:

- **headroom < `ULTRON_MEM_LOW_GB`** (default 4). Headroom = free + file-backed + speculative +
  purgeable pages: what macOS can hand out before compressing or swapping.
- **swap-outs ≥ `ULTRON_SWAPOUT_MB_S`** (default 16) since the previous sample.
- **pressure level ≥ `ULTRON_PRESSURE_TRIP`** (default 4, critical). Level 2 (warn) used to trip it,
  but a 64 GB Mac with opus, sonnet and haiku resident reads 2 all day: in one day 274 of 300
  decisions went to the cloud on "pressure 2" while the tiers sat idle. Warn now only blocks a *cold
  load* (rule 2); a request for an already-loaded tier stays local.

While tripped, when cloud is allowed, a **new conversation** pins `cloud/<tier>` (rule
`4:overflow:mem`, a soft pin, so it re-pins local once memory recovers) and a **conversation pinned
local** sends that one request to `cloud/<tier>` (`overflow=mem`); its pin stays. Set
`ULTRON_MEM_LOW_GB` or `ULTRON_SWAPOUT_MB_S` to 0 to switch that check off.

Release is hysteresis, not a timer: the guard clears once `MEM_HOLD_S` (20 s) have passed since the
last trigger **and** headroom is back over `ULTRON_MEM_CLEAR_GB` (default 5; three idle tiers leave
~4.5–6 GB on 64 GB). The old fixed 120 s hold outlasted the spike, because KV caches free within
seconds of a prefill ending: in one hour of an agent run, 54 of 85 requests went to the cloud, 36 of
them while every tier was idle with 6+ GB free. Replaying three hours of logged headroom, the new
release cuts those idle-tier cloud requests from 154 to 20. The cost: two prefills right after a
release can still collide before the next sample trips the guard (the memory gate below covers only
requests that can't go to the cloud).

**Memory gate.** When a request can't go to the cloud, overflow can't help. A coding agent's main
thread on sonnet and its subagent on opus prefilling at once ran Metal out of memory on most requests
for 15 minutes on the reference Mac (headroom 1.7–3.5 GB, swapping 100–400 MB/s). So a request that
can't go to cloud (route mode `local-only`, `x-route: private`) waits at the gate
(`wait_for_other_tiers`) until no *other* tier is serving a request, first come first served across
tiers. This applies even when the guard isn't tripped: idle headroom with opus, sonnet and haiku
loaded is ~16 GB, and one 100k-token sonnet prefill alone peaks ~12 GB over its resting size, so the
first collision after a quiet spell would run out before the guard trips. A tier that is loading, or
that the gate just let a request through to, counts as busy until 3 s (`MEM_SEND_S`) after its
backend shows the request. Same-tier requests don't wait (the tier queues them itself), and the
`[routing] helper` tier neither waits nor holds anyone. After `ULTRON_MEM_WAIT_S` (default 300,
0 = off) it goes anyway. The tiers share one GPU, so taking turns costs little throughput. Logged as
`mem_wait: {s, gave_up_on}`; the `x-ultron-route` header gets `mem-wait=Ns`.

### Helper models: routed = no (the image judge)

A `tiers.conf` section with `routed = no` is a model llama-swap serves next to the tiers but that is
never a routing target. The example is `[judge]`: SkyJM-Gen-4B, the image judge for
[`Vision/judge/rank.py`](../Vision/judge/), on `mlx_vlm.server` (`mtplx/bin/tier-judge.sh`). Clients
reach it only as `ultron/judge` (or `judge`); it has no cloud twin and is never pinned, and it shares
the big tier's slot in `resident` (`(fable | opus | judge) & sonnet & haiku`), so loading it unloads
opus or fable.

- `admit()` sends it to `admit_direct`. In enforce mode that waits like a local-only reload: first
  for the tier it evicts to be idle and unused by a main conversation for `WARM_MAIN_S`
  (`wait_for_evictees`, at most `EVICT_WAIT_MAX_S`), then for the memory gate (at most
  `ULTRON_MEM_WAIT_S`). Then it goes anyway. Logged with `rule: "direct"`.
- `read_state` lists it with the tiers, so an opus/fable reload waits for an in-flight judge request
  (up to `OVERFLOW_WAIT_S`, then that request goes to the cloud) and the memory gate counts it as busy.
- `decide()` drops it from the loaded set before checking fits: otherwise, while the judge holds
  opus's place, every new opus conversation would overflow to the cloud.
- The media hook skips it: two images plus a generation prompt can read like an edit request.

### Cloud /v1/messages bridge

LiteLLM 1.102.1 sends `/v1/messages` for an `openai/` deployment to the upstream `/v1/responses`.
OmniRoute's `/v1/responses` drops `name` from gemini `function_call` items, so a client gets a
`tool_use` with the right input and `name: ""` (a coding agent overflowed to `cloud/sonnet`, and
OmniRoute picked a gemini model). It also adds its `memory_*` tools there in chat format, which some
upstreams reject (422 `tools[1]: missing field 'name'`). Its `/v1/chat/completions` has neither bug,
and chat completions is what every OpenAI-compatible endpoint serves. `patch_omniroute_chat_bridge()`
runs when the module loads and sends `/v1/messages` for deployments on the cloud endpoint
(`OMNIROUTE_BASE`) through LiteLLM's chat-completions bridge. Other `openai/` bases keep
`/v1/responses` (the global `use_chat_completions_url_for_anthropic_messages` would move them all).
`ULTRON_CHAT_BRIDGE=off` in `~/.litellm/env` restores LiteLLM's default.

### Prompt caching

`litellm_settings.enable_anthropic_prompt_caching` adds cache breakpoints for Claude deployments on
`anthropic/`, `bedrock/`, `vertex_ai/` and `azure_ai/` (it stands down when the client marks its own,
as Claude Code does). The generated `cloud/*` entries carry `cache_control_injection_points` on the
system prompt instead, so clients that don't mark (pi, Hermes) still cache it at a Claude model
behind the cloud endpoint. Local tiers need nothing: LiteLLM strips `cache_control` for
`hosted_vllm`, and mtplx and TensorFold cache by prompt prefix.

### One-shot requests skip the mtplx session bank

mtplx keeps every finished request's KV as a warm session. A local request carries no session
header, so mtplx gives it a random `anon-<hex>` session. In mtplx 2.12.0 every committed request also
queues a deferred SSD save; that queue has no size limit, holds the bank entry's arrays after the bank
evicts it, and unique anon ids never coalesce. Under back-to-back one-shot traffic memory climbs past
`MTPLX_SESSION_BANK_MAX_BYTES`: a 200-image eval on haiku grew ~85 MB per request, from 3.5 GB to
55 GB, then Metal ran out (HTTP 507). Sonnet and opus stay bounded because each conversation reuses
one session. mtplx 2.12.1 fixes the queue.

- `one_shot()` in the deployment hook adds `x-mtplx-cache-mode: bypass` (no bank read or write) to a
  local request that won't be continued: **single-turn** (no tools, no assistant turn yet: Claude Code
  titles, suite probes, chat one-offs) or **vision** (a newest-turn image rerouted off a no-vision
  tier, whose conversation lives on that tier). The route header gains `; bank=off(<why>)`.
- The media hook's helper check sends the same header.
- `tier-haiku.sh` runs `--ssd-session-cache off`, which removes the queue path for haiku entirely.
- Measured on 80 eval requests: default headers +6.0 GB, bypass +0.01 GB.

### PDFs and other attachments become text

When Claude Code reads a PDF, it sends an Anthropic `document` block. LiteLLM's `/v1/messages` bridge
turns `image` and `document` blocks alike into OpenAI `image_url` parts, so the local tier receives a
`data:application/pdf;base64,...` "image", and mtplx answers `400 cannot decode image: cannot identify
image file`, on every later turn too, since the PDF stays in the history.

`docs_to_text()` runs in the deployment hook for local tiers only, before `normalize_history`, and
replaces every `image_url` or chat `file` part whose data URL is neither `image/*` nor `video/*`:

- `application/pdf`: per-page text (`--- page N ---`) from macOS PDFKit through
  `osascript -l JavaScript` (the system Python has no PDF library), in a worker thread.
- `text/*`, JSON and XML: the decoded text.
- Anything else, or a PDF with no text layer (a scan): a one-line note that it was omitted.

The text is capped at 200k characters (~50k tokens), with a note telling the model to read fewer pages
at a time. Results are cached by the data URL's hash (last 16), so later turns send identical text and
the prefix cache still hits. `cloud/*` requests keep the original PDF.

### Tool schemas

The pre-call hook adds `"items": {}` to every array schema without one, whatever the admission mode.
LiteLLM's pre-call token count raises `KeyError: 'items'` on such a schema and then skips the
`max_input_tokens` check; Claude Code sends one, so ~99% of its requests went unchecked. A missing
`items` already means "any item", so the fill changes nothing for the model.

### Trace tap (off by default)

With `~/.ultron/trace-mode` = `on` (Wanda: Controls & routing), the deployment hook writes the latest
tool-carrying request of each local-tier conversation, as the tier sees it, to
`~/.ultron/traces/<conversation key>.json`, overwritten per request (each request carries the whole
history). It's the training data for [`lora/`](../lora/README.md). Traces hold tool output (file
contents, command output): treat them like the conversations themselves. `ULTRON_TRACE_DIR` moves them.

### Compression (bili, off by default)

[billion-context](https://github.com/ranxianglei/billion-context) (`bili`, npm, MIT) can compress every
chat request LiteLLM sends upstream, from one install behind LiteLLM: client → LiteLLM `:4000` (auth,
hooks, admission, overflow) → bili `127.0.0.1:8787` → llama-swap or the cloud endpoint. Clients change
nothing.

- **Install:** `npm i -g --prefix ~/.billion-context billion-context`, then
  `launchctl kickstart gui/$(id -u)/com.billion-context.bili`. The `-g` matters: without it the binary
  lands in `node_modules/.bin/`, not at `~/.billion-context/bin/bili` where `start.sh` looks (or set
  `BILI_BIN` in `~/.bili/env`). Without the binary the LaunchAgent logs a hint to `~/.bili/bili.log`
  and exits; launchd doesn't retry it.
- **Switch:** `~/.ultron/bili-mode` = `on` | `off` (Wanda: Controls & routing → Compression; missing =
  off), re-read every request. The deployment hook's last step, after the history fixes, points each
  chat request for a routed tier (and its aliases) or a `cloud/<tier>` overflow at
  `http://127.0.0.1:8787/bili/openai/<upstream base>`. `routed = no` models (the judge), `media/*` and
  embeddings go direct, and so does everything while bili's LaunchAgent has no PID (checked with
  `launchctl` every 5 s, since bili logs every connection). `x-ultron-route` ends in `; bili`.
- **Sessions:** the hook sends the conversation key admission pins on as `x-acp-session`, so bili keeps
  one session per conversation across tiers and restarts.
- **How it compresses:** bili tags every message and adds a `compress` tool plus ~4k tokens of
  instructions to each request (prefix-cached after the first turn). Past ~50k compressible tokens it
  nudges the model to call it; the model writes the summary and bili sends it in place of that range on
  every later request. Until then nothing is saved. Requests with `max_tokens` ≤ 200 pass untouched.
  A small model can call `compress` on a tiny prompt; bili then returns `[Compression FAILED: …]` as
  the reply.
- **Config:** `bili/config.json`, filled in by `deploy.py`: loopback bind, no self-update, each routed
  tier's `context` from `tiers.conf` as its window (bili compresses at 75% of it), each tier's `cloud`
  id at `OMNIROUTE_BASE`, and an upstream timeout no shorter than the longest tier `timeout`.
- **Exposure:** bili's `/bili/` proxy has **no authentication** and forwards to whatever URL is in its
  path, so it stays on loopback. Caddy exposes only the UI at `/__bili/`, read-only (GET/HEAD).
- **Clients:** remove any client-side billion-context plugin (pi, opencode, …) or requests are
  compressed twice.
- **Speed:** on the reference Mac, sonnet decoded 138.8 tok/s through bili vs 139.7 direct.
- **Rollback:** `echo off > ~/.ultron/bili-mode` (next request goes direct, no restart);
  `launchctl bootout gui/$(id -u)/com.billion-context.bili` stops bili entirely.

### Agent-round history

`async_pre_call_deployment_hook` rewrites the chat messages for every local tier (api_base
llama-swap) right before the upstream call; cloud routes are untouched. Qwen chat templates keep an
assistant turn's `<think>` only after the last real user message, and two things broke that:

- **Reminders as user turns.** Claude Code sends a `<system-reminder>` system message after nearly
  every tool result; LiteLLM's `/v1/messages` bridge turns them into user turns holding only
  reminders. Each became the "last user message", so the model never saw its own reasoning from
  earlier tool rounds of the same task, whatever the tool. Now interior system/developer messages,
  and reminder-only user turns that follow a tool result, are appended to that tool result (or the
  user message they follow). Leading system/developer messages merge into one (TensorFold and
  llama-server answer 500 otherwise).
- **Thinking dropped.** LiteLLM's `hosted_vllm` provider pops `reasoning_content`/`thinking_blocks`
  from assistant turns. For tiers with `think_in_content = yes` in `tiers.conf` (Qwen3.5, whose
  template reads `<think>…</think>` back out of assistant content) the reasoning goes in as a
  leading `<think>` block. A Qwen3.8 template reads only `reasoning_content`, so those tiers don't
  get their thinking back yet.

Measured on a 41-message agent replay: 13 of 13 assistant turns keep their thinking (+595 prompt
tokens), and at the point where the agent was stuck it repeated a failed command 3/5 times instead
of 5/5. Errors log as `history: ...` in `ultron-admit.jsonl`; a failure leaves the request unchanged.

### Split tool calls

mtplx streams an empty delta (`{}`) between tool-call chunks. LiteLLM 1.102.1's `/v1/messages` bridge
(`AnthropicStreamWrapper._should_start_new_content_block`) classified an empty delta as text and
opened a block, so any argument chunk after one opened a second `tool_use` block with **no name**.
Claude Code stored both halves as `{"__unparsedToolInput": {"raw": ...}}`, answered the nameless one
"No such tool available", and mtplx rejected every later request of that conversation with 400
`assistant tool_call is missing a name` (a coding agent on opus, then on sonnet: 9 split calls in one
history). Two fixes in `ultron_admit.py`:

- `patch_blank_delta_blocks()` runs when the module loads and wraps that method so an empty delta
  never opens a block (no-op if LiteLLM moves it).
- `repair_split_tool_calls()` runs in the deployment hook for every deployment, local and cloud:
  drops nameless calls and their results, rejoins the split arguments onto the call before when
  they parse as JSON, and keeps any reminder text from the dropped result.

## ultron_media.py — media prompts go to OmniRoute

Kept in [`Vision/media/`](../Vision/media/) with its tests (the Vision half of WandaVision), deployed
to `~/.litellm/` with the other hooks by `./deploy.py push litellm`.

Looks only at the newest **human** turn. When it asks for media, OmniRoute does the work instead of
the local tier:

| Prompt | OmniRoute endpoint | Reply |
|---|---|---|
| "generate an image of …" | `/v1/images/generations` | link to the file |
| image attached + "edit …" | `/v1/images/edits` | link to the file |
| "make a video of …" | `/v1/videos/generations` | link now, file when rendered |
| "search the web for …" | `/v1/search` | results added to the prompt |
| an `input_audio` block | `/v1/audio/transcriptions` (or chat, see below) | transcript replaces the audio |

Files land in `~/.ultron/media/` and Caddy serves them at `/media/` (`ULTRON_MEDIA_PUBLIC`).

A false positive inside a coding agent's loop is worse than a miss, so: tool_result turns are never
inspected; tool-carrying requests also need the local helper tier (thinking off) to answer MEDIA,
not OTHER (fail closed); Claude
Code's helper calls are skipped; the same prompt in one conversation is answered once per 10 min;
and a veto list ("docker image", "video player", "svg component", …) beats every pattern. Model
lists live in `DEFAULTS`; override them in `~/.ultron/media.json`. Models the endpoint serves only
through chat go in `chat_prefixes` (image, edit and audio), or in `audio_chat_prefixes` when only
their transcription has to go through a chat `input_audio` block. Opt out per request with
`x-media: off`.

## ultron_rescue.py — tool calls written as text

Small local models sometimes stop calling tools and write the command they mean to run in a
```bash block ("Now I'll look up its last commit…" + the command). Claude Code sees a text-only
reply and ends the turn; the model then copies that reply on every later turn. A sonnet-tier coding
agent did this for 6 hours after a loop_breaker `force` note. Replays of that transcript: a
corrective note fixed 0/4, mtplx `tool_choice: required` 0/4 (mtplx only adds a prompt hint, no
constrained decoding).

The hook rewrites such replies on the way out: the trailing block becomes a `tool_use` for the
request's shell tool (Bash/bash/terminal…, argument from its schema), or for a declared tool when
the block reads `ToolName {json}`; the block is dropped from the text and `stop_reason` becomes
`tool_use`. It fires only when: local tier, tools declared, no tool call in the reply, it ended
normally, the text ends with its only fenced block (bash/sh/shell/zsh/console/none), and the text
before it announces the action ("I'll", "Let me", "Now I", "I need to"…). Example answers ("here's
the command") are left alone. `/v1/messages` streaming and non-streaming; chat/completions untouched.
The client's own permission checks still apply to the rescued call.

Mode: `~/.ultron/rescue-mode` (`enforce|shadow|off`, default `enforce`). Opt out per request with
header `x-rescue: off`.

## ultron_stats.py — per-request stats

Observe-only and registered first. It writes `~/.litellm/stats-live.json` (in-flight requests with
live tok/s, rewritten at most once a second) and one line per finished request to
`~/.litellm/ultron-stats.jsonl`. Aggregate metrics come from LiteLLM's stock `prometheus` callback
at `/metrics`. Every handler swallows its own errors, so a stats bug never fails a request.

## Files

| Path | Deployed | What |
|---|---|---|
| `config.yaml`, `start.sh` | yes → `~/.litellm/` | LiteLLM config and launcher |
| `env.example` | no | template for `~/.litellm/env` (secrets, runtime settings) |
| `loop_breaker.py`, `ultron_admit.py`, `ultron_stats.py`, `ultron_rescue.py` | yes → `~/.litellm/` | the hooks (`ultron_media.py` is in `../Vision/media/`, deployed to the same place) |
| `test_*.py`, `conftest.py` | no | unit tests (no network, no LiteLLM install needed); `conftest.py` sends the loop breaker's log to a temp file so tests never write to the live one |
| `replay.py` | no | replays pi / Claude Code transcripts through the detector: `python3 replay.py DIR` |
| `suite.py` | no | live integration suite (below) |
| `fake_omniroute.py` | no | stand-in OmniRoute for sandbox tests |

Deploy from the repo root with `./deploy.py push litellm`: it copies what changed, waits until no
request is in flight, and restarts LiteLLM.

## Tests

```bash
uvx --with pyyaml pytest -q                            # unit tests, from this folder
(cd ../Vision/media && uvx --with pyyaml pytest -q)    # the media hook's unit tests
uvx --with pyyaml python3 suite.py                     # live: health + swap (dry) + vision + classify + confirm + image + video
uvx --with pyyaml python3 suite.py health classify confirm   # quick check: no tier loads, no cloud generation
uvx --with pyyaml python3 suite.py swap --live         # real tier loading
uvx --with pyyaml python3 suite.py baseline --tool     # tok/s per model + tool-call latency
```

`health` checks that services answer, a missing key gets 401, Wanda's memory reading is within
physical RAM, the helper tier and every `preload`/`ttl = 0` tier are resident, the loaded tiers are
an allowed `resident` set, and (with `OMNIROUTE_BASE` set) every OmniRoute id in
`config.yaml` and the media hook's chains exists there. `confirm` runs the helper tier's MEDIA/OTHER
gate for agent requests on the live tier. vision and baseline send `X-Claude-Code-Agent-Id: suite`,
so their pins never make a tier "warm main" and stall the next fable/opus swap.

The live suite needs the stack up. It sets `~/.ultron/media-mode=enforce` (and `admit-mode=enforce`
for live swap) for its duration and restores them after. Run it after swapping models or changing
routing or vision settings.

## Removing a hook

1. Delete its entry under `litellm_settings.callbacks` in `config.yaml`.
2. `./deploy.py push litellm`.
