#!/usr/bin/env python3
"""Peak memory of one LoRA forward+backward at several sequence lengths.
    ~/lora/.venv/bin/python memprobe.py [model_dir | config.yaml] [lengths, e.g. 2048,4096,8192]
A config.yaml (sonnet.yaml, opus.yaml, haiku.yaml) supplies the lora_parameters (rank, scale, keys); the model is
LORA_BASE, else ~/lora/base/<config name>-4bit.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402  (applies chunked_delta + chunked_loss)
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.tuner.trainer import grad_checkpoint  # noqa: E402
from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(os.environ.get("LORA_BASE", "~/lora/base/sonnet-4bit"))
lengths = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2048,4096,8192").split(",")]
lora_cfg = {"rank": 16, "dropout": 0.0, "scale": 20.0}
if path.endswith((".yaml", ".yml")):
    import yaml
    cfg = yaml.safe_load(open(path))
    lora_cfg = dict(cfg.get("lora_parameters") or lora_cfg, dropout=0.0)
    path = os.path.expanduser(os.environ.get("LORA_BASE") or f"~/lora/base/{os.path.basename(path).rsplit('.', 1)[0]}-4bit")
model, tok = load(path)
model.freeze()
linear_to_lora_layers(model, len(model.layers), lora_cfg)
n_lora = sum(v.size for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.trainable_parameters()))
print(f"{path}: {n_lora / 1e6:.1f}M trainable", flush=True)
model.train()
grad_checkpoint(model.layers[0])
vg = nn.value_and_grad(model, train.chunked_loss)
for T in lengths:
    mx.clear_cache(); mx.reset_peak_memory()
    batch = mx.random.randint(0, 1000, (1, T + 1))
    lengths_ = mx.array([[T - 512, T]])
    t = time.time()
    (loss, n), g = vg(model, batch, lengths_)
    mx.eval(loss, g)
    print(f"T={T:6d}  peak {mx.get_peak_memory() / 2**30:5.1f} GB  {time.time() - t:5.1f}s  loss {loss.item():.2f}", flush=True)
