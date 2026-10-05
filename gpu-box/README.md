# gpu-box

Optional. The GPU box is a Windows 11 PC with an NVIDIA GPU, next to the Mac on the same network. It runs ComfyUI
for [Vision](../Vision/README.md) (image generation, image edit, video with sound) and can train image LoRAs with
Unsloth ([lora](../lora/README.md#train-on-the-gpu-box-unsloth)). The Mac doesn't need it: the stack runs without
one, and the media hook uses the cloud endpoint either way.

`gpu-box/deploy.py` is the GPU box's twin of `../deploy.py`: this repo is the source of truth, and the script installs
and refreshes what the GPU box needs from it. Clone the repo on the GPU box and run the script from the clone.
Python 3.9+, stdlib only.

```powershell
python gpu-box\deploy.py status                       # per component: in sync, or what differs
python gpu-box\deploy.py push --dry-run               # what a push would change
python gpu-box\deploy.py push                         # prereq, caddy, comfyui, nodes (core), tasks
python gpu-box\deploy.py push nodes --all             # also the 29 extra custom nodes
python gpu-box\deploy.py push models --group image    # model files: image | video (default both, 76 GB)
python gpu-box\deploy.py push unsloth                 # Unsloth Studio, if its venv is missing
python gpu-box\deploy.py push comfyui --restart       # restart ComfyUI once its queue is empty
```

A bare `push` skips `models` and `unsloth`: they move gigabytes. Exit code is 1 when a component needs attention.

| Component | What it manages | Source in repo |
|---|---|---|
| `prereq` | checks only: NVIDIA driver, git, python; reports Tailscale (optional) and the host name in use | — |
| `caddy` | `~\caddy\Caddyfile` and `start_caddy.bat` (the GPU box's front door: `/comfy/` and `/` -> ComfyUI :8188, on the LAN and on your HTTPS name); downloads `caddy.exe` when missing; validates, then `caddy reload` | `gpu-box/caddy/` |
| `comfyui` | `run_service.bat`, `comfy_idle_unload.ps1`, `extra_model_paths.yaml` in `~\ComfyUI\ComfyUI-Easy-Install` | `Vision/comfyui/` |
| `nodes` | ComfyUI custom nodes: clone when missing, `checkout` the pinned commit, `pip install -r` into the embedded python. Skips a node with local changes | `Vision/comfyui/nodes.json` |
| `tasks` | scheduled tasks `Caddy-GPU-Box` and `ComfyUI-GPU-Box` (at boot, restart on failure) and `ComfyUI Idle Unload` (every 5 min) | `TASKS` in `deploy.py` |
| `models` | downloads missing files into `~\models\ComfyUI\models\<dir>`, resuming `.part` files, checking the pinned size; also finds files already in ComfyUI's own `models\` | `Vision/comfyui/models.json` |
| `unsloth` | runs Unsloth's installer (`irm https://unsloth.ai/install.ps1 \| iex`) when `~\.unsloth\studio` has no venv; creates `~\lora` for training output | — |

The services run from the copies, not from the clone, so the clone can move or be updated without stopping them.
Edits made in place are not protected the way `../deploy.py` protects the Mac's: `push` overwrites a differing
managed file. Change the repo, then push. To update: `git pull`, then `python gpu-box\deploy.py push`.

## Host name

`caddy/Caddyfile` serves HTTPS on `__HOSTNAME__`, which `deploy.py` fills in from `GPU_BOX_HOSTNAME`: the
environment first, then `wandavision.conf` at the repo root (the same KEY=VALUE file `../deploy.py` reads; it is
git-ignored). Unset, it is `localhost`, and the LAN address `http://gpu-box.local` (or the box's IP) is the one to use.

```powershell
Add-Content wandavision.conf "GPU_BOX_HOSTNAME=gpu-box.example.ts.net"   # your Tailscale MagicDNS name
python gpu-box\deploy.py push caddy
```

On a Tailscale MagicDNS name (`*.ts.net`) Caddy gets the certificate from Tailscale, so the GPU box must be on
the tailnet.

## Fresh GPU box

1. Install the NVIDIA driver, Git for Windows (`winget install Git.Git`), Python 3.9+
   (`winget install Python.Python.3.12`), and Tailscale if you want the `*.ts.net` name. Clone the repo and check:
   ```powershell
   git clone https://github.com/enslaver/wandavisionllm.git $env:USERPROFILE\wandavision
   cd $env:USERPROFILE\wandavision
   python gpu-box\deploy.py status prereq
   ```
2. Install ComfyUI with **ComfyUI-Easy-Install** (Pixaroma Community Edition 3.14.3 is what the reference box runs):
   unzip it into `~\ComfyUI`, run `ComfyUI-Easy-Install.bat`. It is interactive and builds
   `~\ComfyUI\ComfyUI-Easy-Install` with its own `python_embeded` (Python 3.12, torch 2.13+cu130, flash-attn,
   triton-windows). The deploy script refuses to go on without that tree. ComfyUI v0.37.0-27 (`93810483`) is what
   the pinned nodes were run against.
3. Set the host name (above), then, in an **elevated** PowerShell (boot-trigger tasks need it):
   `python gpu-box\deploy.py push`.
4. `python gpu-box\deploy.py push models` (or `--group image`). `flux-2-klein-9b-fp8` is gated: accept the licence
   at huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8 and set `HF_TOKEN` (or `hf auth login`) first.
5. Optional, for LoRA training: `python gpu-box\deploy.py push unsloth`, then the steps in
   [../lora/README.md](../lora/README.md#train-on-the-gpu-box-unsloth).
6. Check: `http://gpu-box.local/` (or `https://gpu-box.example.ts.net/`) shows ComfyUI. Then point Open WebUI at it:
   [../Vision/open-webui/README.md](../Vision/open-webui/README.md).

The reference GPU box is an RTX 4080 Laptop GPU with 12 GB. On it, on 2026-10-03, `status` and `push --dry-run`
reported every component in sync, the task XML installed (the idle-unload task was created and deleted; the
boot-trigger ones were refused without elevation, as expected), and the resumable download passed its unit test.
Not run: a from-scratch install, a real model download, cloning a node, `push unsloth`. Running from a clone and
the `__HOSTNAME__` placeholder are covered by the unit tests only.

## Not managed

- **Runtime state:** `~\ComfyUI\ComfyUI-Easy-Install\comfy_*.log`, `~\caddy\caddy.log`, ComfyUI's `user\` and
  `output\`, `~\lora\` artifacts, `~\.unsloth\`.
- **Your own ComfyUI graphs** (`~\ComfyUI\workflows` or wherever you keep them). Only the three API-format
  workflows in `Vision/comfyui/workflows/` are part of the stack.
- **Hugging Face tokens and Tailscale auth:** never in the repo.
- **LoRA scripts** (`lora\unsloth_vision.py`, `lora\unsloth_queue.sh`) run straight from the clone, so there is
  nothing to copy.

## Tests

```bash
cd gpu-box && uvx pytest -q          # deploy.py helpers; run anywhere, no Windows needed
```
