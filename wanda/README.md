# Wanda — the panel

Realtime dashboard for the stack: one page that shows what every tier is doing right now, who is
sending requests, where they were routed and why, and lets you flip the hooks' modes without a
restart. Standard-library Python, no build step, no external assets.

```
LiteLLM 0.0.0.0:4000 → llama-swap 127.0.0.1:8001 → tiers 127.0.0.1:18001+   (from tiers.conf, e.g. fable / opus / sonnet / haiku)
Wanda 127.0.0.1:8790 behind Caddy :443 / :80
```

## What's on the page

- **Status lamps** — LiteLLM liveness, llama-swap, memory / swap / pressure, GPU, routing modes,
  and any on/off flags you define.
- **Tier cards** — live state (UNLOADED / LOADING / IDLE / PREFILL / GENERATING), decode or prefill
  tok/s, TTFT, MTP acceptance, cache hit, 5-minute stats, lifetime tokens, memory + session bank,
  idle → unload countdown, the request in flight, Load / Unload buttons.
- **Throughput trace** — 10 minutes per tier; each completed request is a bar over its real span.
- **Traffic flow** — agents (Claude Code, pi, Hermes, opencode, …, labelled by Tailscale name) →
  LiteLLM → local tiers or cloud combos, over the last hour.
- **Models** — per-model request stats from the `ultron_stats` hook, local and cloud.
- **Controls & routing** — one switch per hook mode file in `~/.ultron` (Route, Admission, Loop
  breaker, Media, Tool-call rescue), 1 h / 24 h routing, loop, media and rescue tiles, OmniRoute combo
  health, and lists of new conversations, OmniRoute calls, loop-breaker actions and media prompts
  (applied images link to `/media/<file>`). Loop entries logged in shadow mode count as
  "shadow-only", not stops.
- **LoRA** — the Trace tap switch (`~/.ultron/trace-mode`), trace counts per tier, the `lora/run.sh`
  stage running now (train step x/y), which tier serves a pack, and `~/lora/runs` (checkpoints,
  held-out val loss base → best, repeat rate base → adapter) and `~/lora/packs`. Empty until you use
  [`lora/`](../lora/README.md).
- **Requests & events** — every completed request plus load / unload / mode-change events.
- **Logs** — llama-swap, LiteLLM, Caddy, mode changes, OmniRoute calls, LOOPS, ADMIT, MEDIA,
  RESCUE, REQUESTS (secrets masked).
- **Services** — links to everything else behind Caddy, from `services.json`, with health dots.
- **Bottom bar** — the whole stack at a glance; click a segment to jump to its section.

Anything that changes state (Unload, a mode flip) needs a second press within 5 s.

## Files

| Path | What |
|---|---|
| `server.py` | samples the stack every 1 s; serves `/`, `/api/*`, `/icons/*` |
| `static/index.html` | the page |
| `static/ultron.svg`, `static/icons/` | bundled hub image and service icons |
| `install.sh` | runs after each push: installs the LaunchAgent and restarts Wanda |
| `com.wanda.portal.plist` | LaunchAgent (`__HOME__` filled in by `deploy.py`) |
| `services.example.json` | template for `~/.wanda/www/services.json` (not deployed) |

Deploy with `./deploy.py push wanda` from the repo root. The folder is copied to `~/wanda`.

## Settings

Environment variables, set in `com.wanda.portal.plist`:

| Variable | Default | What |
|---|---|---|
| `WANDA_LISTEN` | `127.0.0.1:8790` | bind address; keep it on loopback behind Caddy |
| `WANDA_NAME` | this Mac's short hostname | panel title and hub label |
| `WANDA_WWW` | `~/.wanda/www` | holds `services.json` and your own `icons/*.svg` |
| `WANDA_OMNIROUTE` | `OMNIROUTE_BASE` from `~/.litellm/env` | OmniRoute base URL; empty = cloud panels show "not configured" |
| `WANDA_OMNIROUTE_KEY_NAME` | `litellm` | the OmniRoute API key name whose calls the OMNIROUTE log shows |
| `WANDA_HOST_NAMES` | — | `ip=name,ip=name` labels for clients Tailscale can't name |

Your own hub image: drop `ultron.png` / `.jpg` / `.webp` / `.svg` into `~/.wanda/`.

## API

| Call | What |
|---|---|
| `GET /api/status` | one sample (JSON) |
| `GET /api/stream` | server-sent events, a `tick` every second |
| `GET /api/history` | 15 minutes of samples per tier, plus events and requests |
| `GET /api/log?name=swap\|litellm\|caddy\|modes\|omniroute\|loops\|admit\|media\|rescue\|stats` | log tail, secrets masked |
| `POST /api/mode {key, value}` | set a hook mode (`route`, `admit`, `loops`, `media`, `rescue`, `trace`) |
| `POST /api/flag {key, value}` | set an on/off flag |
| `POST /api/tier {tier, action: load\|unload}` | load or unload a tier through llama-swap |

POSTs need the `X-Wanda-Token` header. The token lives in `~/.wanda/token` (created on first run)
and is injected into the page, so anyone who can load the page can use the buttons: keep Wanda on
your LAN / tailnet, not the open internet.

## Extending

- **A service link:** add a route to `caddy/Caddyfile`, then an entry to `~/.wanda/www/services.json`
  (see `services.example.json`: `upstream` + `health` give it a live dot).
- **An on/off flag:** add an entry to `FLAGS` in `server.py` (a file holding `true`/`false` that
  other tools read); it shows up as a lamp and a switch.
- **A hook mode:** add an entry to `MODES` in `server.py` and have the hook read the file
  (`warn` = the options painted amber, `panel: "lora"` = show it in the LoRA section).

## Ops

```bash
launchctl kickstart -k gui/$(id -u)/com.wanda.portal   # restart
tail -f ~/.wanda/wanda.log
curl -s 127.0.0.1:8790/api/status | python3 -m json.tool | head
```
