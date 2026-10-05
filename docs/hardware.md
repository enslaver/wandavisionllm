# Hardware

## The reference machine

| | |
|---|---|
| Machine | Mac Studio (2026, `Mac17,14`) |
| Chip | Apple M5 Max |
| CPU | 18 cores: 6 super + 12 performance |
| GPU | 40 cores |
| Memory | 64 GB unified |
| Storage | 1 TB internal SSD |
| OS | macOS 27.0 |

Supporting pieces, both optional:

- **Cloud overflow:** a small Linux VPS running OmniRoute (an OpenAI-compatible multi-provider router),
  reached over Tailscale. Any OpenAI-compatible endpoint can stand in.
- **Repo share:** this repo lives on a NAS share that the Mac and the laptops mount at the same path,
  so `deploy.py` can be run from any of them (`WANDAVISION_HOST`). A plain clone on the Mac works too.

## Software versions it was tested with

| Component | Version |
|---|---|
| LiteLLM | 1.102.1 |
| llama-swap | v260 |
| mtplx | 2.12.0 |
| TensorFold | 0.3.4.1 |
| Caddy | 2.11.4 |
| Python (system) | 3.9.6 |

## Tiers and memory

The example tier scripts in `mtplx/bin/`:

| Tier | Model | Runtime | Weights | Cache cap | Context | Vision |
|---|---|---|---|---|---|---|
| fable | Qwen3.8-27B, MLX 4-bit (`Vontra/Qwen3.8-27B-MLX-4bit`) | TensorFold (DFlash2 draft trees) | ~16 GB + ~3.9 GB drafter | 4 GiB prompt + 6 GiB MLX | 200k input (262k window) | no |
| opus | Qwen3.6-35B-A3B MoE, MTPLX Optimized-Speed (`Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed`) | mtplx (MTP) | ~21 GB | 6 GB session bank | 256k | no |
| sonnet | Qwen3.5-9B, MTPLX Optimized-Speed (`Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed`) | mtplx (MTP) | ~8.7 GB | 6 GB session bank | 262k | yes |
| haiku | Qwen3.5-4B, MTPLX Optimized-Speed (`Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed`) | mtplx (MTP) | ~2.6 GB | 5 GB session bank | 262k | no |

On 64 GB, either big tier fits resident with sonnet and haiku (~30 or ~32 GB, plus ~15 + ~8 GB, with
every cache at its cap; less in practice), but not both: `resident = (fable | opus) & sonnet & haiku`,
so loading one unloads the other. Opus is a MoE with 3B active parameters and a small KV cache (~20
KiB/token, ~5 GiB at 256k). Fable still wants the machine to itself past ~128k tokens of context.
Measured on the reference Mac with a 4-bit Qwen3.8-27B checkpoint on TensorFold:

| Fable context | Decode | Peak memory |
|---|---|---|
| 128k | 41 tok/s | 47 GB |
| 200k | 28 tok/s | 52 GB |
| 250k | 19 tok/s | swaps |

The memory guard in `ultron_admit` sends new conversations to the cloud before that point.

## Measured speed

The 27B (fable tier), 4-bit, same client, thinking off, temperature 0.6, median of 3 runs (tok/s):

| Runtime | code | html | prose | file edit |
|---|---|---|---|---|
| mtplx, mixed-precision MTPLX pack, MTP depth 3 | 67.7 | 67.5 | 40.7 | 178.7 |
| **TensorFold, uniform 4-bit (the fable script)** | **153.5** | **142.2** | **67.0** | **494.6** |

TensorFold's output is byte-identical to serial decoding; its speedup comes from draft trees. Haiku
(mtplx, MTP, the Qwen3.5-4B pack above) runs ~226 tok/s end to end on a 2k-token code answer, with
parallel tool calls.

Run `cd litellm && uvx --with pyyaml python3 suite.py baseline --tool` to measure your own.

## Ready-made settings by RAM

The numbers that matter are weights + cache caps per tier, plus ~8–12 GB for macOS and everything
else. Each row names a model per tier and the `resident =` line for the `[routing]` section of
`tiers.conf`. That line uses llama-swap's matrix syntax: `&` = loaded together, `|` = one or the other,
parentheses group. When a request would need a load that doesn't fit the line, the admission hook
(`ultron_admit`) sends a new conversation to cloud overflow instead of making llama-swap swap (route
mode `local-only` swaps instead).

| Unified memory | fable | opus | sonnet | haiku | `resident =` |
|---|---|---|---|---|---|
| 16–24 GB | — | Qwen3.5-9B MTPLX (~8.7 GB) | Qwen3.5-4B MTPLX (~2.6 GB) | Qwen3.5-2B GGUF on llama-server (`unsloth/Qwen3.5-2B-GGUF:Q4_K_M`, ~1.3 GB) | `opus \| (sonnet & haiku)` |
| 32 GB | — | Qwen3.5-9B MTPLX (~8.7 GB) | Qwen3.5-4B MTPLX (~2.6 GB) | Qwen3.5-2B GGUF on llama-server (~1.3 GB) | `opus & sonnet & haiku` |
| 48 GB | — | Qwen3.6-35B-A3B MTPLX (~21 GB) | Qwen3.5-9B MTPLX (~8.7 GB) | Qwen3.5-4B MTPLX (~2.6 GB) | `(opus \| sonnet) & haiku` |
| 64 GB | Qwen3.8-27B 4-bit on TensorFold (~16 GB + drafter) | Qwen3.6-35B-A3B MTPLX | Qwen3.5-9B MTPLX | Qwen3.5-4B MTPLX | `(fable \| opus) & sonnet & haiku` |
| 128 GB | Qwen3.8-27B 4-bit on TensorFold, full 200k input | Qwen3.6-35B-A3B MTPLX | Qwen3.5-9B MTPLX | Qwen3.5-4B MTPLX | `(fable \| opus) & sonnet & haiku` |

Below 64 GB, delete the `[fable]` and `[judge]` sections from `tiers.conf`, and let sonnet unload
(`ttl = 600`, `preload = no`, `evict_cost = 2` in `[sonnet]`). From 64 GB, sonnet and haiku stay
loaded and the image judge (~5.7 GB, optional) takes the big tier's slot:
`(fable | opus | judge) & sonnet & haiku`. Per row, beyond the models:

- **16–24 GB:** opus alone, or the two small tiers together. Session-bank caps 2–3 GB, context 64k–128k.
  The 9B pack has a vision tower, so opus gets `vision = yes`. Send heavy work to the cloud.
- **32 GB:** the same models, all resident (~13 GB of weights). Caps 3–4 GB each. TensorFold's 27B needs
  more than its default 22.4 GiB memory budget on a 32 GB Mac, so there is no fable tier here.
- **48 GB:** opus and sonnet take turns next to an always-loaded haiku. In `tier-opus.sh` lower the
  session bank (`MTPLX_SESSION_BANK_MAX_BYTES=3G`) and the context (add `--context-window 131072`, and
  set `context` in `tiers.conf`).
- **64 GB:** the example scripts as they are.
- **128 GB:** raise fable's caps (`--prompt-cache-gib 8 --mlx-cache-gib 8` in `tier-fable.sh`) so long
  conversations stay cached.

Runtimes other than mtplx and TensorFold: `mtplx/bin/examples/` has llama-server and mlx_lm.server
tier scripts. To change a tier, edit its script (`MODEL`, flags) and its keys in `tiers.conf` (`vision`,
`context`, …); see [configuration.md](configuration.md#swapping-a-model). Intel Macs and other operating
systems are not supported: mtplx, TensorFold and mlx-lm are MLX-based, and the service layer is launchd.
