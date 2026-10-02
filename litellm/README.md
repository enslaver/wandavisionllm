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
| `ultron_media.py` | answers "make an image / video", web search, audio via OmniRoute | `~/.ultron/media-mode` | `~/.litellm/media.jsonl` |
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
mid-way (switching throws away the prompt cache and changes the model's voice).

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
- **pressure level ≥ 2** (warn/critical).

While tripped (and for 120 s after the last trigger), when cloud is allowed, a **new conversation**
pins `cloud/<tier>` (rule `4:overflow:mem`) and a **conversation pinned local** sends that one
request to `cloud/<tier>` (`overflow=mem`); its pin stays, so it returns once memory recovers. Set
either env var to 0 to switch that check off.

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

Looks only at the newest **human** turn. When it asks for media, OmniRoute does the work instead of
the local tier:

| Prompt | OmniRoute endpoint | Reply |
|---|---|---|
| "generate an image of …" | `/v1/images/generations` | link to the file |
| image attached + "edit …" | `/v1/images/edits` | link to the file |
| "make a video of …" | `/v1/videos/generations` | link now, file when rendered |
| "search the web for …" | `/v1/search` | results added to the prompt |
| an `input_audio` block | `/v1/audio/transcriptions` | transcript replaces the audio |

Files land in `~/.ultron/media/` and Caddy serves them at `/media/` (`ULTRON_MEDIA_PUBLIC`).

A false positive inside a coding agent's loop is worse than a miss, so: tool_result turns are never
inspected; tool-carrying requests also need the local helper tier (thinking off) to answer MEDIA,
not OTHER (fail closed); Claude
Code's helper calls are skipped; the same prompt in one conversation is answered once per 10 min;
and a veto list ("docker image", "video player", "svg component", …) beats every pattern. Model
lists live in `DEFAULTS`; override them in `~/.ultron/media.json`. Opt out per request with
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
| `loop_breaker.py`, `ultron_admit.py`, `ultron_media.py`, `ultron_stats.py`, `ultron_rescue.py` | yes → `~/.litellm/` | the hooks |
| `test_*.py`, `conftest.py` | no | unit tests (no network, no LiteLLM install needed); `conftest.py` sends the loop breaker's log to a temp file so tests never write to the live one |
| `replay.py` | no | replays pi / Claude Code transcripts through the detector: `python3 replay.py DIR` |
| `suite.py` | no | live integration suite (below) |
| `fake_omniroute.py` | no | stand-in OmniRoute for sandbox tests |

Deploy from the repo root with `./deploy.py push litellm`: it copies what changed, waits until no
request is in flight, and restarts LiteLLM.

## Tests

```bash
uvx --with pyyaml pytest -q                            # unit tests, from this folder
uvx --with pyyaml python3 suite.py                     # live: health + swap (dry) + vision + classify + confirm + image + video
uvx --with pyyaml python3 suite.py health classify confirm   # quick check: no tier loads, no cloud generation
uvx --with pyyaml python3 suite.py swap --live         # real tier loading
uvx --with pyyaml python3 suite.py baseline --tool     # tok/s per model + tool-call latency
```

`health` checks that services answer, a missing key gets 401, Wanda's memory reading is within
physical RAM, the helper tier is resident, and (with `OMNIROUTE_BASE` set) every OmniRoute id in
`config.yaml` and the media hook's chains exists there. `confirm` runs the helper tier's MEDIA/OTHER
gate for agent requests on the live tier. vision and baseline send `X-Claude-Code-Agent-Id: suite`,
so their pins never make a tier "warm main" and stall the next fable/opus swap.

The live suite needs the stack up. It sets `~/.ultron/media-mode=enforce` (and `admit-mode=enforce`
for live swap) for its duration and restores them after. Run it after swapping models or changing
routing or vision settings.

## Removing a hook

1. Delete its entry under `litellm_settings.callbacks` in `config.yaml`.
2. `./deploy.py push litellm`.
