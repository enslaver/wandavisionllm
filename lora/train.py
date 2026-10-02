#!/usr/bin/env python3
"""mlx_lm.lora with two memory fixes for long agent prompts on Qwen3.5. Same CLI:
    ~/lora/.venv/bin/python train.py --config sonnet.yaml [--iters N --data DIR --adapter-path DIR]

- chunked_delta: the linear-attention layers train in 64-token chunks instead of a per-token loop.
- chunked_loss: the 248k-vocab logits are computed 1024 positions at a time under mx.checkpoint,
  so an 8k-token example holds ~1 GB of logits instead of ~8 GB (plus as much again for gradients).
- no mx.compile around the train step: compiled, an 8k example peaked at 43 GB (the checkpointed
  activations stay alive); uncompiled it is ~20 GB.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chunked_delta  # noqa: E402,F401  (patches mlx_lm on import)
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_lm import lora  # noqa: E402
from mlx_lm.tuner import trainer  # noqa: E402

LOSS_CHUNK = 1024


def chunked_loss(model, batch, lengths):
    inputs, targets = batch[:, :-1], batch[:, 1:]
    lm = getattr(model, "language_model", model)
    h = lm.model(inputs)
    head = lm.model.embed_tokens.as_linear if lm.args.tie_word_embeddings else lm.lm_head  # frozen

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    total = mx.array(0.0)
    for s in range(0, targets.shape[1], LOSS_CHUNK):
        tc, mc = targets[:, s:s + LOSS_CHUNK], mask[:, s:s + LOSS_CHUNK]  # constants: only hc is differentiated
        piece = mx.checkpoint(lambda hc, tc=tc, mc=mc: (nn.losses.cross_entropy(head(hc), tc) * mc).astype(mx.float32).sum())
        total = total + piece(h[:, s:s + LOSS_CHUNK])
    ntoks = mask.sum()
    return total / ntoks, ntoks


for fn in (trainer.train, trainer.evaluate):  # their loss= default was bound at definition
    fn.__defaults__ = tuple(chunked_loss if d is trainer.default_loss else d for d in fn.__defaults__)
trainer.default_loss = chunked_loss


class _NoCompile:
    """trainer.mx with compile() as a no-op; everything else is mlx.core."""
    def __getattr__(self, name):
        return getattr(mx, name)

    @staticmethod
    def compile(fun=None, **_):
        return fun if fun is not None else (lambda f: f)


trainer.mx = _NoCompile()
# Freed buffers otherwise stay cached for reuse and count against Metal's wired limit (~48 GB of 64,
# shared with the serving tiers).
mx.set_cache_limit(int(float(__import__("os").environ.get("LORA_CACHE_GB", "2")) * 2**30))

if __name__ == "__main__":
    lora.main()
