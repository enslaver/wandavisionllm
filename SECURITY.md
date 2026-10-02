# Security

## Reporting a vulnerability

Please use GitHub's **private vulnerability reporting** (the repository's *Security* tab → *Report a
vulnerability*) rather than a public issue. Expect a first reply within a week.

## What the stack exposes, and how to run it safely

| Surface | Exposure | Protection |
|---|---|---|
| LiteLLM `:4000` | all interfaces | `LITELLM_MASTER_KEY`; `start.sh` refuses to start without a real one |
| Caddy `:80` / `:443` | all interfaces | none of its own: it fronts Wanda and LiteLLM |
| Wanda `:8790` | loopback, behind Caddy | read-only pages are open; state changes need `X-Wanda-Token` |
| llama-swap `:8001`, tiers `:18001+` | loopback only | no auth; never bind them elsewhere |

- **Keep the Mac on a private network.** Wanda has no login: its POST token is injected into the page,
  so anyone who can load the page can unload tiers and flip modes, and the page shows request
  metadata (agents, hosts, models, prompt sizes). Serve it on your LAN or a tailnet, not the internet.
  If you must expose it, put authentication in Caddy (`basic_auth` or forward auth) in front of `/`.
- **The master key is the only thing between the network and your models** (and your cloud budget,
  when overflow is on). Use a long random key; rotate it by editing `~/.litellm/env` and restarting
  LiteLLM.
- **Secrets stay on the Mac:** `~/.litellm/env` (mode 600) and `~/.wanda/token`. Log tabs mask keys,
  but review logs before sharing them. `bin/backup-stack.sh` copies secrets, so keep its destination
  private.
- **Media and cloud overflow send prompts to third parties.** Use route mode `local-only`, header
  `x-route: private`, or media mode `off` for anything that must not leave the machine.
- **Uncensored models.** The example tier scripts use standard community packs. If you swap in an
  abliterated ("uncensored") model, its safety tuning is gone; don't serve it to people who haven't
  opted in.

## Supported versions

Only the latest release gets fixes.
