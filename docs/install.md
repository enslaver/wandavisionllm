# Install

About an hour on a fresh Mac, most of it downloading models. Every command runs on the Mac that will
serve the models unless it says otherwise.

## 1. Prerequisites

- Apple Silicon Mac; 64 GB of memory for the reference tiers ([hardware.md](hardware.md) for less).
- [Homebrew](https://brew.sh) and [uv](https://docs.astral.sh/uv/).
- Optional: [Tailscale](https://tailscale.com) with HTTPS certificates enabled for your tailnet, so
  Caddy can serve the panel on `https://your-mac.your-tailnet.ts.net`.

```bash
brew install caddy python
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. The runtimes

| What | Where it must end up | How |
|---|---|---|
| LiteLLM | `~/.local/bin/litellm` | `uv tool install 'litellm[proxy]==1.102.1' --with prometheus_client` |
| llama-swap | `~/.local/bin/llama-swap` | download the macOS arm64 binary from [llama-swap releases](https://github.com/mostlygeek/llama-swap/releases) (v260 or newer: the config uses the `matrix` router and startup preload) |
| mtplx | `~/.mtplx/bin/mtplx` | install the MTPLX app; it provides the CLI shim at that path (tested with 2.12.0; 2.12.1 fixes an SSD-cache leak the haiku script works around) |
| TensorFold | `~/.tensorfold/venv/bin/tensorfold` | install [TensorFold](https://github.com/ashhart/TensorFold) into a venv at `~/.tensorfold/venv` (tested with 0.3.4.1) |
| mlx-vlm (optional) | `~/.local/bin/mlx_vlm.server` | `uv tool install mlx-vlm==0.6.13`; only the image judge uses it |

Only want mtplx? Delete the `[fable]` section from `litellm/tiers.conf` (and `fable |` from its
`resident =` line); TensorFold only runs the fable tier, because it decodes the dense 27B about twice
as fast.

## 3. The models

The tier scripts in `mtplx/bin/` name the models. Get them before the first request: llama-swap
gives a tier 300 s to become healthy, which isn't enough for a first download.

| Tier | Model | Get it |
|---|---|---|
| fable | `Vontra/Qwen3.8-27B-MLX-4bit` + drafter `z-lab/Qwen3.8-27B-DFlash2` | `~/.tensorfold/venv/bin/tensorfold pull Vontra/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2` (the script refuses to start without the drafter) |
| opus | `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed` | `uvx --from huggingface_hub hf download Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed --local-dir ~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed` |
| sonnet | `Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed` | the same, with that id |
| haiku | `Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed` | the same, with that id |
| judge (optional) | `skylenage-ai/SkyJM-Gen-4B`, converted to MLX 8-bit at `~/models/SkyJM-Gen-4B-mlx-8bit` | the three commands in the header of `mtplx/bin/tier-judge.sh`; or delete the `[judge]` section from `litellm/tiers.conf` (and `\| judge` from `resident =`) |

`mtplx forge discover` searches Hugging Face for ready-made MTPLX builds if you'd rather download
than forge. Any model works; change `MODEL` in the tier script and see
[configuration.md](configuration.md#swapping-a-model).

## 4. Settings

```bash
git clone https://github.com/enslaver/wandavisionllm.git && cd wandavisionllm

cp wandavision.conf.example wandavision.conf
$EDITOR wandavision.conf          # WANDAVISION_HOSTNAME: your Tailscale name, or leave localhost

mkdir -p ~/.litellm
cp litellm/env.example ~/.litellm/env && chmod 600 ~/.litellm/env
python3 -c "import secrets; print('sk-' + secrets.token_urlsafe(32))"   # paste as LITELLM_MASTER_KEY
$EDITOR ~/.litellm/env
```

LiteLLM refuses to start while `LITELLM_MASTER_KEY` is empty or still `sk-change-me`: it listens on
all interfaces.

**No cloud provider?** Leave `OMNIROUTE_BASE` empty and, after step 5, set Wanda's
*Route* to `local-only` and the media mode to `off` (or `echo local-only > ~/.ultron/route-mode`
and `echo off > ~/.ultron/media-mode`).

## 5. Deploy

```bash
./deploy.py push --dry-run        # lists every file it would write
./deploy.py push
```

On a first install this writes the tier scripts, llama-swap and LiteLLM configs, loads the two
LaunchAgents (LiteLLM, llama-swap), installs and starts Wanda, and starts Caddy with
`brew services`. Rerunning is safe: a push only writes files whose content changed.

## 6. Check it

```bash
curl -s 127.0.0.1:4000/health/liveliness                 # LiteLLM: "I'm alive!"
curl -s 127.0.0.1:8001/running                           # llama-swap: haiku preloads
open http://localhost/                                   # Wanda
./deploy.py status                                       # everything "in sync"
```

First requests to fable, opus and sonnet take 15–30 s while the tier loads; Wanda shows it as LOADING.

## 7. Point your agents at it

**Claude Code** (any machine that can reach the Mac):

```bash
export ANTHROPIC_BASE_URL=http://your-mac:4000       # or https://your-mac.your-tailnet.ts.net/llm
export ANTHROPIC_AUTH_TOKEN=<LITELLM_MASTER_KEY>
claude                                               # /model opus | sonnet | haiku
```

**OpenAI-compatible clients** (pi, Hermes, opencode, SDK scripts): base URL
`http://your-mac:4000/v1`, API key `<LITELLM_MASTER_KEY>`, model `ultron/opus`, `ultron/sonnet` or
`ultron/haiku`. `GET /v1/models` lists them.

## 8. Optional extras

- **Service links in Wanda:** `mkdir -p ~/.wanda/www && cp wanda/services.example.json ~/.wanda/www/services.json`.
- **Backups:** `~/bin/backup-stack.sh ~/some/private/folder` copies configs, state and model manifests
  (it includes `~/.litellm/env`, so keep the destination private).
- **Deploy from a laptop:** set `WANDAVISION_HOST` and `WANDAVISION_REMOTE_ROOT` in
  `wandavision.conf` on the laptop; `deploy.py` reruns itself on the Mac over ssh.
- **Validate:** `cd litellm && uvx --with pyyaml python3 suite.py` runs the live integration suite.
- **Image and video generation on a GPU box, image ranking, Open WebUI:** see [Vision](../Vision/README.md).
- **Image LoRAs for haiku** (trained with Unsloth on a CUDA PC): [lora/README.md](../lora/README.md).

## Uninstall

```bash
for l in com.litellm.proxy com.llama-swap com.wanda.portal; do
  launchctl bootout gui/$(id -u)/$l; rm -f ~/Library/LaunchAgents/$l.plist
done
brew services stop caddy
rm -rf ~/wanda ~/.wanda ~/.llama-swap ~/.ultron ~/.wandavision   # keeps ~/.litellm (your env) and the models
```
