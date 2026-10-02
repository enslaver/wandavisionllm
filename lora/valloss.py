#!/usr/bin/env python3
"""valloss.py — loss on a dataset's final turns for the base and each saved adapter checkpoint.
    ~/lora/.venv/bin/python valloss.py ~/lora/data/v0/valid.jsonl ~/lora/runs/v0 [--base BASE_VIEW]
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402,F401  (chunked loss/delta patches)
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_lm import load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("run")
ap.add_argument("--base", default=os.environ.get("LORA_BASE", "~/lora/base/sonnet-4bit"))  # text-only view of the pack (README)
a = ap.parse_args()
BASE = os.path.expanduser(a.base)
data, run = Path(a.data).expanduser(), Path(a.run).expanduser()
rows = [json.loads(l) for l in open(data)]


def loss(model, tok):
    tot, n = 0.0, 0
    for r in rows:
        full = tok.apply_chat_template(r["messages"], tools=r.get("tools"), return_dict=False)
        off = len(tok.apply_chat_template(r["messages"][:-1], tools=r.get("tools"), add_generation_prompt=True, return_dict=False))
        x = mx.array(full)[None]
        logits = model(x[:, :-1])[:, off - 1:]
        ce = nn.losses.cross_entropy(logits.astype(mx.float32), x[:, off:]).sum()
        tot += ce.item(); n += len(full) - off
    return tot / n


ckpts = [("base", None)] + [(p.name.split("_")[0].lstrip("0"), p) for p in sorted(run.glob("*_adapters.safetensors"))]
for name, ck in ckpts:
    if ck is None:
        model, tok = load(BASE)
    else:
        d = Path(tempfile.mkdtemp())
        shutil.copy(run / "adapter_config.json", d / "adapter_config.json")
        shutil.copy(ck, d / "adapters.safetensors")
        model, tok = load(BASE, adapter_path=str(d))
    model.eval()
    print(f"{name:>6}: val loss {loss(model, tok):.3f}", flush=True)
    del model
