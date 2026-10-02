"""Chunked gated delta rule for training Qwen3.5 / Qwen3-Next in mlx_lm.

mlx_lm runs the gated delta rule (the linear-attention layers, 3 of every 4 blocks) with its Metal
kernel only at inference; in training it falls back to a Python loop over every token. At 8k tokens
that graph holds more Metal buffers than the 499000 limit ("[metal::malloc] Resource limit
exceeded"). This is the chunked (WY) form, as in transformers' torch_chunk_gated_delta_rule: 64-token
chunks, so 128 sequential steps at 8k, all plain mx ops (differentiable). The in-chunk triangular
solve (I - A)^-1 is a blocked forward substitution (64 -> 32 -> 16 -> 8, rows below that). The
shortcut (I + A)(I + A^2)...(I + A^32) is exact in theory but A^32 reaches ~1e17 when a chunk's keys
are correlated, and it returned NaN on real activations.

Importing this module patches mlx_lm's gated_delta_update for training calls (use_kernel=False, no
padding mask); inference keeps the Metal kernel. test_chunked_delta.py checks it against the loop.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import gated_delta as _gd

CHUNK = 64


def chunk_gated_delta(q, k, v, gl, beta, state, chunk=CHUNK):
    """q, k [B,T,Hk,Dk] (q pre-scaled by normalize_qk), v [B,T,H,Dv], gl log decay [B,T,H],
    beta [B,T,H], state [B,H,Dv,Dk]. Returns (y [B,T,H,Dv], state)."""
    B, T, Hk, Dk = q.shape
    H, Dv = v.shape[-2:]
    if (r := H // Hk) > 1:
        q, k = mx.repeat(q, r, -2), mx.repeat(k, r, -2)
    out_dtype = q.dtype
    q, k, v = (x.astype(mx.float32).transpose(0, 2, 1, 3) for x in (q, k, v))  # [B,H,T,D]
    gl, beta = (x.astype(mx.float32).transpose(0, 2, 1) for x in (gl, beta))     # [B,H,T]
    pad = (-T) % chunk
    if pad:  # zero beta and zero log decay: padded steps change nothing
        q, k, v = (mx.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)]) for x in (q, k, v))
        gl, beta = (mx.pad(x, [(0, 0), (0, 0), (0, pad)]) for x in (gl, beta))
    n = (T + pad) // chunk
    q, k, v = (x.reshape(B, H, n, chunk, x.shape[-1]) for x in (q, k, v))
    beta = beta.reshape(B, H, n, chunk)
    g = gl.reshape(B, H, n, chunk).cumsum(-1)

    lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
    strict = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_), -1)
    diff = g[..., :, None] - g[..., None, :]
    decay = mx.where(lower, mx.exp(mx.where(lower, diff, 0.0)), 0.0)  # [B,H,n,C,C], exp(g_i - g_j) for j <= i

    kb, vb = k * beta[..., None], v * beta[..., None]
    a = mx.where(strict, -(kb @ k.swapaxes(-1, -2)) * decay, 0.0)
    tm = _unit_lower_inverse(a)
    u = tm @ vb                                   # [B,H,n,C,Dv]
    w = tm @ (kb * mx.exp(g)[..., None])          # [B,H,n,C,Dk]
    attn = (q @ k.swapaxes(-1, -2)) * decay       # includes the diagonal: output after the update

    s = state.astype(mx.float32).swapaxes(-1, -2)  # [B,H,Dk,Dv]
    ys = []
    for i in range(n):
        v_new = u[:, :, i] - w[:, :, i] @ s
        ys.append((q[:, :, i] * mx.exp(g[:, :, i])[..., None]) @ s + attn[:, :, i] @ v_new)
        g_end = g[:, :, i, -1]
        s = s * mx.exp(g_end)[..., None, None] + \
            (k[:, :, i] * mx.exp(g_end[..., None] - g[:, :, i])[..., None]).swapaxes(-1, -2) @ v_new
    y = mx.stack(ys, 2).reshape(B, H, n * chunk, Dv)[:, :, :T].transpose(0, 2, 1, 3)
    return y.astype(out_dtype), s.swapaxes(-1, -2)


def _unit_lower_inverse(a):
    """(I - a)^-1 for strictly lower triangular a [..., m, m] (m a power of two)."""
    m = a.shape[-1]
    if m <= 8:  # forward substitution: row i = e_i + sum_{j<i} a[i, j] * row j
        eye = mx.eye(m, dtype=a.dtype)
        rows = [mx.broadcast_to(eye[0], a.shape[:-2] + (m,))]
        for i in range(1, m):
            rows.append(eye[i] + (a[..., i, :i, None] * mx.stack(rows, -2)).sum(-2))
        return mx.stack(rows, -2)
    h = m // 2
    t11 = _unit_lower_inverse(a[..., :h, :h])
    t22 = _unit_lower_inverse(a[..., h:, h:])
    t21 = t22 @ a[..., h:, :h] @ t11
    zero = mx.zeros(a.shape[:-2] + (h, h), dtype=a.dtype)
    return mx.concatenate([mx.concatenate([t11, zero], -1), mx.concatenate([t21, t22], -1)], -2)


_original = _gd.gated_delta_update


def gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *,
                       use_kernel=True, lower_bound=None, allow_neg_eigval=False):
    if use_kernel or mask is not None:
        return _original(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel,
                         lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)
    beta = mx.sigmoid(b) * (2.0 if allow_neg_eigval else 1.0)
    if lower_bound is None:  # log of compute_g / compute_lower_bound_g
        gl = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    else:
        gl = lower_bound * mx.sigmoid(mx.exp(A_log.astype(mx.float32)) * (a.astype(mx.float32) + dt_bias))
    if state is None:
        B, _, _, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    return chunk_gated_delta(q, k, v, gl, beta, state)


def patch():
    import importlib
    for name in ("qwen3_5", "qwen3_next"):
        try:
            mod = importlib.import_module(f"mlx_lm.models.{name}")
        except ImportError:
            continue
        if getattr(mod, "gated_delta_update", None) is _original:
            mod.gated_delta_update = gated_delta_update
    _gd.gated_delta_update = gated_delta_update


patch()
