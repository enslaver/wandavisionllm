# open-webui

Part of [Vision](../README.md). Not deployed by `deploy.py`. [Open WebUI](https://github.com/open-webui/open-webui)
runs on whatever machine you like (the Open WebUI host); these are the image and video settings it needs to make
pictures and clips through the stack, kept here so they can be re-applied by hand.

| File | What |
|---|---|
| `../comfyui/workflows/zimage-turbo-generate.api.json` | ComfyUI text -> image, Z-Image-Turbo (8 steps, cfg 1) |
| `../comfyui/workflows/flux2-klein-edit.api.json` | ComfyUI image edit, Flux.2 Klein 9B distilled (4 steps, keeps the input's size at 1 MP) |
| `../comfyui/workflows/minimax-h3-t2v.api.json` | ComfyUI text -> video with sound, MiniMax H3 + 8-step turbo LoRA |
| `video_tool.py` | Open WebUI tool `generate_video` that runs the H3 workflow on the GPU box and attaches the MP4 |

ComfyUI runs on the GPU box (`http://gpu-box.example.ts.net`, or `http://gpu-box.local` on the LAN);
[`gpu-box/deploy.py`](../../gpu-box/README.md) installs it and the models these workflows need. Open WebUI only
accepts **API-format** workflows ("Export (API)" in ComfyUI); a normal save gives the UI format and fails with
`400: Something went wrong :/` (`KeyError` on a node id).

Measured on the reference GPU box (RTX 4080 Laptop, 12 GB), 2026-10-01: generate 36.5 s cold, edit 28.8 s cold,
video 3 s clip 103 s / 2 s clip 89 s.

## Admin -> Settings -> Images

**Image generation** through LiteLLM on the Mac, which passes it to the cloud endpoint (`OMNIROUTE_BASE`). No GPU
box needed:

| Setting | Value |
|---|---|
| Engine | OpenAI |
| API base URL | `https://your-mac.example.ts.net/llm/v1` (or `http://your-mac:4000/v1`) |
| API key | the LiteLLM key (`LITELLM_MASTER_KEY`, the same one as the Mac's connection under Connections) |
| Model | `media/image` (an entry in `litellm/config.yaml`; set its `model` to an image model your endpoint serves: the example is `openai/gpt-image-1`) |
| Size | `1024x1024` |

To use ComfyUI instead: Engine ComfyUI, base URL `http://gpu-box.example.ts.net`, workflow
`comfyui/workflows/zimage-turbo-generate.api.json`, steps 8, and these nodes (no Model node, so the workflow's own
model is used):

| Field | Key | Node |
|---|---|---|
| Prompt | `text` | `27` |
| Width | `width` | `13` |
| Height | `height` | `13` |
| n | `batch_size` | `13` |
| Steps | `steps` | `3` |
| Seed | `seed` | `3` |

**Image edit** (ComfyUI): base URL `http://gpu-box.example.ts.net`, workflow
`comfyui/workflows/flux2-klein-edit.api.json`, nodes:

| Field | Key | Node |
|---|---|---|
| Image | `image` | `76` |
| Prompt | `text` | `74` |
| Seed | `noise_seed` | `73` |

Leave width/height/steps unmapped for edit: Open WebUI passes `None` for any it doesn't have, and the workflow
sizes the output from the input image.

## Video

Open WebUI 0.11.4 has no video setting. Workspace -> Tools -> + -> paste `Vision/open-webui/video_tool.py` -> save,
then enable the tool on the chat models (Workspace -> Models -> edit -> Tools). Valves (the tool's settings):
`COMFYUI_URL` (default `http://gpu-box.example.ts.net`: set your own), default/max seconds, aspect ratio,
megapixels (0.4 = 864x480; higher is much slower on 12 GB), timeout.

Clips land in ComfyUI's `output\video\` on the GPU box and as a file on the chat message. Without the GPU box,
a chat that goes through LiteLLM gets video from the cloud endpoint instead: with the media hook in `enforce`
mode, "make a video of ..." is answered by `ultron_media` (see [../README.md](../README.md)).
