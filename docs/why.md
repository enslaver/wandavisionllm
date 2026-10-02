# Why this exists

WandaVision started as one question: *can the coding agents I already use run on a Mac I own,
without changing how I use them?* The answer was yes, but only after solving five problems that a
single `llama-server` behind an API key doesn't. Each piece of the stack exists because of one of them.

## 1. Agents think in Claude tiers

Claude Code picks a model per task: a big one for planning, a fast one for edits, a small one for
titles, topic checks and subagents, often several at once. It asks for `claude-opus-*`,
`claude-sonnet-*` and `claude-haiku-*`.

So the stack serves a local tier under each of those names (plus `claude-fable-*` in the example set). LiteLLM speaks the Anthropic
Messages API that Claude Code uses (and the OpenAI API for everything else), and routes by wildcard,
so a new Claude model id routes to the right tier with no edit. `/model opus` in any Claude Code on
the network lands on the local opus tier. Tier names never change; swapping the model behind one is
a one-line edit in its launch script.

## 2. 64 GB is not infinite

A 27B model is ~22 GB of weights before any context. At 128k tokens its KV cache pushes the process
to ~47 GB; at 200k, ~52 GB. Three tiers at full context don't fit, and macOS doesn't fail cleanly
when memory runs out — it compresses, then swaps model weights, and throughput collapses (an early
test with three larger tiers loaded fell to 13% free memory, 15 of 16 GB swap, and opus dropped to
26–42 tok/s).

LiteLLM can route but can't start or stop processes, so **llama-swap** sits behind it: it starts a
tier on its first request, unloads it after an idle TTL, and enforces a memory *matrix* (which tiers
may be resident together). Each tier script caps its runtime's cache sizes, because by default each
mtplx process plans for a ~43 GB session bank.

## 3. Small models loop, and nobody catches it

Local 4B–27B models get stuck repeating tool calls. From real sessions:

- One agent made the **identical `bash` call 4,909 times over 9.8 hours**. It survived 36 context
  compactions and a human typing "continue".
- Another made the same `execute_code` call **210 times in 34 minutes**; its context grew from 42k
  to 88k tokens until each turn took ~130 s to first token.
- A third produced five 16k-token runaway replies in a row, up to 555 s each.
- Frontier models loop too: 89 identical `ScheduleWakeup` calls; `git status` 95 times over 7 hours.

The runtimes' repetition guards work inside one reply, but each looping turn is a short, valid, fresh
tool call, so they never fire. Client-side guards failed because results differed in trivia
(`execution_count`, durations). The only place every client passes through is the proxy, so the
**loop breaker** is a LiteLLM hook. Its thresholds come from ~430k real tool calls, including where
repetition is legitimate (polling a build, paging through a file), so it warns, then forces a text
answer, then ends the turn — without breaking the backend's prompt cache.

## 4. Sometimes the Mac should say no

Evicting a model to load another kills whatever session was using it; swapping to disk makes
everything slow. When a new conversation would do either, it is better to send that conversation to
a cloud model and keep the Mac healthy.

The **admission** hook decides once per conversation — local tier, a loaded substitute tier, or a
cloud combo — and **pins** the choice, so a conversation never switches models mid-task (which would
throw away its prompt cache and change its voice). A **memory guard** watches free + file-backed
pages and swap-out rate (the kernel's "pressure" level read *normal* while 9 GB was being swapped
out) and sends new work to the cloud until memory recovers. All of it can run in *shadow* mode first,
logging what it would do.

## 5. You can't run what you can't see

With three models loading, unloading, prefilling and generating for several agents on several
machines, `tail -f` stopped being enough. Questions that needed answers at a glance:

- Which tiers are loaded, and what is each one doing right now, at how many tokens per second?
- Which agent on which machine sent that request, and did it go local or to the cloud? Why?
- Is anything looping? Is the Mac about to swap?

**Wanda** answers those on one page, updated every second, and puts the hooks' switches (route
mode, admission, loop breaker) one click away with no restart.

## And one rule for changing it

Every piece above is a config file or script in a different place on the Mac. Editing them in place
led to drift and to "which version is live?" questions. So this repo is the source of truth and
`deploy.py` is the only way to change the Mac: it copies what changed, waits until no request is in
flight before restarting anything, and refuses to overwrite an edit that was made in place.
