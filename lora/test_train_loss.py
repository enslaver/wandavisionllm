"""train.chunked_loss: only answer tokens count, never the pad after <|im_end|>\\n.
    ~/lora/.venv/bin/python -m pytest -q test_train_loss.py
"""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

import train

V, D = 50, 16


class Body:
    """lm.model: hidden states from the embedding, plus the embed_tokens chunked_loss reads."""
    def __init__(self):
        self.embed_tokens = nn.Embedding(V, D)

    def __call__(self, x):
        return self.embed_tokens(x)


def test_mask_stops_at_last_real_token(monkeypatch):
    mx.random.seed(0)
    emb = Body()
    m = SimpleNamespace(model=emb, lm_head=nn.Linear(D, V, bias=False), args=SimpleNamespace(tie_word_embeddings=False))

    # Two rows padded with 0 to the batch width, as iterate_batches builds them.
    rows, offs = [[5, 6, 7, 8, 9, 10], [11, 12, 13, 14]], [3, 2]
    P = 9
    batch = mx.array([r + [0] * (P - len(r)) for r in rows])
    lengths = mx.array([[o, len(r)] for r, o in zip(rows, offs)])
    monkeypatch.setattr(train, "LOSS_CHUNK", 3)   # several chunks, one boundary inside an answer
    loss, ntoks = train.chunked_loss(m, batch, lengths)

    logits = m.lm_head(emb(batch[:, :-1]))
    ce = nn.losses.cross_entropy(logits, batch[:, 1:])
    want, n = 0.0, 0
    for i, (r, o) in enumerate(zip(rows, offs)):
        for j in range(o, len(r)):          # targets r[o] .. r[-1]; never batch[i, len(r)] (the pad)
            want += ce[i, j - 1].item()
            n += 1
    assert ntoks.item() == n == (6 - 3) + (4 - 2)
    assert abs(loss.item() - want / n) < 1e-4
