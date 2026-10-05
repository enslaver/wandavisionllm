# Vision — image and video generation

Everything in the stack that makes pictures and clips, or judges them, lives here. All of it is optional: the
chat tiers (fable, opus, sonnet, haiku) run without it, and their configs stay in the other folders.

```
chat "make a video of ..."  -> LiteLLM :4000 [ultron_media hook] -> cloud endpoint (OMNIROUTE_BASE): image / edit / video / search / transcribe
Open WebUI (any machine)    -> ComfyUI on the GPU box, behind its Caddy   z-image-turbo, flux2-klein edit, MiniMax H3 video
generated images            -> judge/rank.py -> ultron/judge (SkyJM-Gen-4B on the Mac) -> a ranking
```

| Folder | What | Runs on | Deployed by |
|---|---|---|---|
| [`media/`](media/) | `ultron_media.py`, the LiteLLM hook that answers "generate an image / make a video / edit this photo / search the web / attached audio" from the cloud endpoint (`OMNIROUTE_BASE` in `~/.litellm/env`) instead of a tier, plus its tests. Details: [../litellm/README.md](../litellm/README.md) | the Mac, `~/.litellm/` | `./deploy.py push litellm` |
| [`comfyui/`](comfyui/) | ComfyUI on the GPU box: API-format workflows, the pinned custom nodes and models, the service scripts | the GPU box | [`gpu-box/deploy.py`](../gpu-box/README.md) |
| [`open-webui/`](open-webui/README.md) | Open WebUI's image and video settings and the `generate_video` tool | the Open WebUI host (any machine) | by hand (README) |
| [`judge/`](judge/README.md) | `rank.py`: pairwise ranking of generated images with `ultron/judge` | any machine, calls the Mac | — |

Shared config stays with the component that owns the file: `litellm/config.yaml` keeps the `media/*` model
entries, `litellm/tiers.conf` the `[judge]` section, and `caddy/Caddyfile` serves `/media/` (the files the media
hook writes to `~/.ultron/media/`). The media hook's model ids come from `~/.ultron/media.json` on the Mac (its
defaults are OpenAI's names: `gpt-image-1`, `sora-2`, `whisper-1`; use whatever your endpoint serves).

## media/

The hook runs on the Mac with the other hooks: `./deploy.py push litellm` copies `ultron_media.py` into
`~/.litellm/` next to `loop_breaker.py` and `ultron_tiers.py`, which it imports. It starts in `shadow` mode
(logs what it would do); switch it in Wanda or with `~/.ultron/media-mode`. Its tests need those two modules,
so `conftest.py` puts `../../litellm` on the path.

## comfyui/

| Path | What |
|---|---|
| `workflows/*.api.json` | `zimage-turbo-generate` (text -> image), `flux2-klein-edit` (image edit), `minimax-h3-t2v` (text -> video with sound). API format ("Export (API)"), the only format Open WebUI accepts |
| `nodes.json` | custom nodes pinned to the commits the reference GPU box runs. `core` (ComfyUI-Manager) is always installed; the workflows above need only ComfyUI's built-in nodes. `extra` = 29 more from that install, optional (`push nodes --all`) |
| `models.json` | 11 models, 76 GB, in two groups (`image` 36 GB, `video` 39 GB) with size, source URL and the workflow that needs each. URLs come from ComfyUI's own workflow templates; `flux-2-klein-9b-fp8` is gated on Hugging Face (set `HF_TOKEN`) |
| `run_service.bat`, `comfy_idle_unload.ps1`, `extra_model_paths.yaml` | how ComfyUI runs on the GPU box: loopback :8188 behind Caddy, restart loop, models unloaded after 15 min idle, models under `~\models\ComfyUI` |

What fits in 12 GB (an RTX 4080 Laptop GPU, measured 2026-10-01, see [open-webui](open-webui/README.md)): image
generate 37 s cold, edit 29 s cold, a 3 s video clip 103 s.

## Install and refresh the GPU box

[`gpu-box/deploy.py`](../gpu-box/README.md) brings the GPU box in line with this folder: Caddy, the ComfyUI
service files, the pinned custom nodes, the scheduled tasks, the models, and Unsloth for training image LoRAs.
Run it on the GPU box, from a clone of this repo.

```powershell
python gpu-box\deploy.py status                    # what differs on the GPU box
python gpu-box\deploy.py push                      # caddy, comfyui files, custom nodes (core), tasks
python gpu-box\deploy.py push models --group image # fetch model files (big)
```

After changing a workflow here, Open WebUI needs it re-uploaded (Admin -> Settings -> Images); the GPU box needs
nothing.

## Tests

```bash
cd Vision/media && uvx --with pyyaml pytest -q   # media hook
cd gpu-box && uvx pytest -q                      # gpu-box/deploy.py helpers
cd litellm && uvx --with pyyaml python3 suite.py image video   # live: text -> image / video through the stack
```
