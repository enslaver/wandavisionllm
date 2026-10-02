#!/usr/bin/env python3
"""fuse_pack.py — fold a LoRA adapter into an MTPLX pack's quantized body -> a new pack mtplx serves.

mtplx can't load a body adapter at serve time (only --mtp-adapter), so the adapter goes into the
weights: for every adapted linear, W + scale * (lora_a @ lora_b)^T in fp32, quantized with the pack's
own settings (4-bit affine g64, or a per-layer override from config.json). Everything else (other
tensors, MTP head, vision tower, tokenizer, template) is kept; mtplx_runtime.json gets a "lora" block.
Single-file packs (sonnet: model.safetensors) and sharded ones (opus: model-0000N-of-0000M + index)
both work: only shards holding an adapted layer are rewritten, the rest are hard-linked.

Use --bf16 (the full-precision checkpoint the pack was forged from). Without it W is the pack's own
dequantized 4-bit weight: those values sit on the quantization grid, so a small delta rounds straight
back and only ~20% of it survives (measured on the smoke adapter). From bf16 the rounding keeps the
delta on average, and the script first checks that quantizing bf16 alone reproduces the pack.

    ~/lora/.venv/bin/python fuse_pack.py PACK ADAPTER_DIR OUT --bf16 BF16_DIR
Then point MODEL= in mtplx/bin/tier-<tier>.sh at OUT (README: "Ship an adapter").
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("adapter"); ap.add_argument("out")
ap.add_argument("--bf16", help="full-precision checkpoint dir (HF layout: model.language_model.*)")
a = ap.parse_args()
pack, adapter, out = (Path(p).expanduser() for p in (a.pack, a.adapter, a.out))
if out.exists():
    sys.exit(f"{out} exists")
lora_cfg = json.load(open(adapter / "adapter_config.json"))
scale = float(lora_cfg["lora_parameters"]["scale"])
quant = json.load(open(pack / "config.json"))["quantization"]
base_q = dict(group_size=quant["group_size"], bits=quant["bits"], mode=quant.get("mode", "affine"))


def qparams(name):
    """The pack's quantization for one module (per-layer overrides, e.g. opus' 8-bit router)."""
    o = quant.get(name)
    return {**base_q, **{k: o[k] for k in ("group_size", "bits", "mode") if k in o}} if isinstance(o, dict) else base_q


index = pack / "model.safetensors.index.json"
if index.exists():
    wmap = json.load(open(index))["weight_map"]
else:
    wmap = {k: "model.safetensors" for k in mx.load(str(pack / "model.safetensors"))}

ad = mx.load(str(adapter / "adapters.safetensors"))
bases = sorted(k[: -len(".lora_a")] for k in ad if k.endswith(".lora_a"))
if missing := [b for b in bases if b + ".weight" not in wmap]:
    sys.exit(f"adapter layers not in the pack: {missing[:5]}")

full = {}  # pack name -> (bf16 shard path, hf name)
if a.bf16:
    bf16 = Path(a.bf16).expanduser()
    if (bf16 / "model.safetensors.index.json").exists():  # opus: 16 shards, 72 GB; don't open them all to list names
        names = json.load(open(bf16 / "model.safetensors.index.json"))["weight_map"].items()
    else:
        names = [(hf, shard) for shard in sorted(glob.glob(str(bf16 / "model*.safetensors"))) for hf in mx.load(shard)]
    for hf, shard in names:
        full[hf.replace("model.language_model.", "language_model.model.", 1)] = (str(bf16 / Path(shard).name), hf)
    if missing := [b for b in bases if b + ".weight" not in full]:
        sys.exit(f"adapter layers not in the bf16 checkpoint: {missing[:5]}")

# A layer's .weight, .scales and .biases can sit in different pack shards (opus: layer 22's in_proj_qkv),
# so look each tensor up on its own. mx.load is lazy: open every touched shard once, write each once.
SUFFIXES = (".weight", ".scales", ".biases")
touched = sorted({wmap[b + x] for b in bases for x in SUFFIXES})
shards = {n: mx.load(str(pack / n), return_metadata=True) for n in touched}  # name -> (tensors, metadata)


def get(name):
    return shards[wmap[name]][0][name]


def put(name, value):
    shards[wmap[name]][0][name] = value


by_src = {}  # bf16 shard -> adapted layers, so each bf16 file is opened once
for b in bases:
    by_src.setdefault(full[b + ".weight"][0] if a.bf16 else None, []).append(b)

out.mkdir(parents=True)
t0, kept, same = time.time(), [], []
for src_path, names in sorted(by_src.items(), key=lambda kv: kv[0] or ""):
    src = mx.load(src_path) if src_path else None
    for b in names:
        q = qparams(b)
        w, s, z = (get(b + x) for x in SUFFIXES)
        if src is not None:
            dense = src[full[b + ".weight"][1]].astype(mx.float32)
            if len(same) < 8:  # quantizing bf16 alone must give the pack back
                same.append(mx.mean(mx.quantize(dense, **q)[0] == w).item())
        else:
            dense = mx.dequantize(w, s, z, **q).astype(mx.float32)
        delta = scale * (ad[b + ".lora_b"].astype(mx.float32).T @ ad[b + ".lora_a"].astype(mx.float32).T)
        nw, ns, nz = mx.quantize(dense + delta, **q)
        old = mx.dequantize(w, s, z, **q).astype(mx.float32)
        new = mx.dequantize(nw, ns, nz, **q).astype(mx.float32)
        kept.append((mx.sum((new - old) * delta) / mx.maximum(mx.sum(delta * delta), 1e-12)).item())
        nw, ns, nz = nw, ns.astype(s.dtype), nz.astype(z.dtype)
        mx.eval(nw, ns, nz)
        for x, v in zip(SUFFIXES, (nw, ns, nz)):
            put(b + x, v)
    del src
    if same and min(same) < 0.98:
        shutil.rmtree(out)
        sys.exit(f"quantizing the bf16 checkpoint doesn't reproduce the pack (identical words {min(same):.3f}); "
                 "wrong checkpoint or a different recipe")
    print(f"  {Path(src_path).name if src_path else 'pack'}: {len(names)} layers fused", flush=True)
for n in touched:
    tensors, meta = shards.pop(n)
    mx.save_safetensors(str(out / n), tensors, metadata=meta)
    print(f"  wrote {n}", flush=True)
    del tensors

for f in pack.iterdir():
    if f.name in touched or f.name == "mtplx_runtime.json" or f.is_dir():
        continue
    try:
        os.link(f, out / f.name)
    except OSError:
        shutil.copy2(f, out / f.name)
kept.sort()
rt = json.load(open(pack / "mtplx_runtime.json"))
rt["lora"] = {"base_pack": pack.name, "adapter": str(adapter), "from": "bf16" if a.bf16 else "pack-4bit",
              "fused_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "layers": len(bases),
              "rank": lora_cfg["lora_parameters"].get("rank"), "scale": scale,
              "delta_kept": {"min": round(kept[0], 3), "median": round(kept[len(kept) // 2], 3)},
              "bf16_reproduces_pack": round(min(same), 4) if same else None,
              "note": "MTP head unchanged (draft acceptance may drop)"}
json.dump(rt, open(out / "mtplx_runtime.json", "w"), indent=2)
print(f"{out}: {len(bases)} layers from {'bf16' if a.bf16 else 'the 4-bit pack'} in {time.time() - t0:.0f}s; "
      f"delta kept {kept[len(kept) // 2]:.2f} median (min {kept[0]:.2f})"
      + (f"; bf16 alone reproduces the pack: {min(same):.4f}" if same else ""))
