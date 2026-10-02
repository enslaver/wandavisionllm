# Configuration

Three kinds of settings, kept apart on purpose:

| Kind | Where | Changed by | Takes effect |
|---|---|---|---|
| Deploy-time | `wandavision.conf` in the repo (git-ignored) | you | next `./deploy.py push` |
| Secrets and runtime | `~/.litellm/env` on the Mac | you | LiteLLM restart (`./deploy.py push litellm` or `launchctl kickstart -k gui/$(id -u)/com.litellm.proxy`) |
| Live switches | `~/.ultron/*-mode` files | the Wanda panel (or `echo`) | next request, no restart |

## wandavision.conf

| Key | Default | What |
|---|---|---|
| `WANDAVISION_HOSTNAME` | `localhost` | fills `__HOSTNAME__` in `caddy/Caddyfile`: the HTTPS site name |
| `WANDAVISION_HOST` | empty | ssh host to deploy to; empty = this machine |
| `WANDAVISION_REMOTE_ROOT` | this repo's path | where the repo lives on `WANDAVISION_HOST` |

Environment variables with the same names override the file. `__HOME__` is always your home
directory. `./deploy.py render DIR` writes the filled-in files to `DIR` so you can review them.

## ~/.litellm/env

Template: [`litellm/env.example`](../litellm/env.example).

| Key | What |
|---|---|
| `LITELLM_MASTER_KEY` | the API key clients send; required |
| `LITELLM_PORT` | default 4000 |
| `OMNIROUTE_BASE`, `OMNIROUTE_KEY` | cloud overflow endpoint (`…/v1`) and key; also used by the media hook and Wanda |
| `ULTRON_MEDIA_PUBLIC` | public base URL for generated media (`https://your-mac…/media`) |
| `ULTRON_ADMIT_MODE`, `LOOP_BREAKER_MODE`, `ULTRON_MEDIA_MODE`, `ULTRON_RESCUE_MODE` | fallbacks when the mode files are missing |
| `ULTRON_MEM_LOW_GB`, `ULTRON_SWAPOUT_MB_S` | memory guard thresholds; 0 turns a check off |
| `ULTRON_MEM_WAIT_S` | memory gate: how long a local-only request waits for other tiers to finish (300; 0 = off) |

## Live switches

| File | Values | Default | What |
|---|---|---|---|
| `~/.ultron/route-mode` | `auto`, `local-only`, `cloud-only` | `auto` | where new conversations may go |
| `~/.ultron/admit-mode` | `enforce`, `shadow`, `off` | `shadow` | admission rewrites the model (`enforce`) or only logs (`shadow`) |
| `~/.ultron/loop-breaker-mode` | `enforce`, `shadow`, `off` | `enforce` | loop breaker acts or only logs |
| `~/.ultron/media-mode` | `enforce`, `shadow`, `off` | `shadow` | media prompts go to OmniRoute or only log |
| `~/.ultron/rescue-mode` | `enforce`, `shadow`, `off` | `enforce` | a tool call written as a ```bash block becomes a real tool call, or only logs |
| `~/.ultron/trace-mode` | `on`, `off` | `off` | save each local-tier agent conversation's latest request to `~/.ultron/traces/` (for `lora/`) |

Wanda's Controls & routing section has a switch for each; the trace tap's is in its LoRA section.

Start with admission and media in `shadow`, watch the ADMIT and MEDIA tabs in Wanda for a day, then
switch to `enforce`.

## Per-request overrides

| Header / metadata | Effect |
|---|---|
| `x-route: cloud` | this conversation goes to `cloud/<tier>` |
| `x-route: private` | never leaves the Mac |
| `x-loop-breaker: off` or metadata `{"loop_breaker": "off"}` | skip the loop breaker |
| `x-media: off` | skip the media hook |

Every response carries `x-ultron-route` with the decision and its reason.

## Model names

| Requested model | Tier |
|---|---|
| `ultron/fable`, `claude-fable-*`, `fable` | fable |
| `ultron/opus`, `claude-opus-*`, `opus` | opus |
| `ultron/sonnet`, `claude-sonnet-*`, `claude-*` (anything else Claude), `sonnet` | sonnet |
| `ultron/haiku`, `claude-haiku-*`, `haiku` | haiku |
| `cloud/<tier>` for each tier with `cloud =` | straight to the cloud overflow model |
| `media/image`, `media/image-edit`, `media/embed` | OmniRoute image, edit, embedding endpoints |

`/v1/models` lists the `ultron/*`, `cloud/*` and `media/*` ids and one current Claude id per tier;
the wildcards and bare names route but aren't listed. All of it comes from the `match` and `advertise`
keys in `litellm/tiers.conf`; the table shows the example tiers.

## Swapping a model

Tier names never change, so clients and LiteLLM don't either.

1. Edit `MODEL` (and sampling flags if the model card says so) in `mtplx/bin/tier-<name>.sh`.
2. In `litellm/tiers.conf`, update that tier's `context`, `vision`, `think_in_content` and
   `chat_only` to what the new model and runtime support (the script's header lists them).
3. If memory changes, recheck `resident` in `tiers.conf` and the cache caps in the script.
4. `./deploy.py push`, then unload the tier from Wanda so the next request loads the new model.
5. `cd litellm && uvx --with pyyaml python3 suite.py` to validate routing and vision.

## Adding a tier or a hook

- **A new deployed file:** put it in the matching folder and add it to `COMPONENTS` in `deploy.py`.
- **A hook:** write it as a LiteLLM `CustomLogger` with a `proxy_handler_instance`, add it under
  `litellm_settings.callbacks` in `config.yaml` (order matters), add a test, and list it in
  `COMPONENTS`. If it needs a live switch, add an entry to `MODES` in `wanda/server.py`.
- **A tier:** add a `[section]` to `litellm/tiers.conf` (its keys are documented at the top of the
  file), give it a script `mtplx/bin/tier-<name>.sh`, add it to `resident`, and `./deploy.py push`.
  deploy.py builds the llama-swap and LiteLLM entries from `tiers.conf`; the hooks and Wanda read it
  directly. Other runtimes: [`mtplx/bin/examples/`](../mtplx/bin/examples/README.md).

## Running without a cloud provider

Leave `OMNIROUTE_BASE` empty in `~/.litellm/env`, and set route mode to `local-only` and media mode
to `off`. The `cloud/*` and `media/*` entries stay in `config.yaml` but nothing routes to them; Wanda
shows the cloud panels as "not configured". When a
tier can't load, llama-swap swaps (evicting another tier) instead of overflowing.
