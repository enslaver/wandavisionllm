"""chunked_delta vs mlx_lm's per-token loop: outputs, final state and gradients.
    ~/lora/.venv/bin/python -m pytest -q test_chunked_delta.py
Exactness runs on the CPU: Metal's fp32 matmul is itself ~1e-3 relative off an elementwise sum, so on
the GPU the two agree only to that (test_gpu_close).
"""
import mlx.core as mx
import pytest
from mlx_lm.models import gated_delta as gd

import chunked_delta as cd


@pytest.fixture(autouse=True)
def cpu(request):
    if "gpu" in request.node.name:
        yield
        return
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def inputs(T, B=1, Hk=2, H=4, Dk=32, Dv=32, seed=0):
    mx.random.seed(seed)
    q, k = mx.random.normal((B, T, Hk, Dk)), mx.random.normal((B, T, Hk, Dk))
    q, k = gd.normalize_qk(q, k, inv_scale=Dk ** -0.5, eps=1e-6)
    v = mx.random.normal((B, T, H, Dv))
    a, b = mx.random.normal((B, T, H)), mx.random.normal((B, T, H))
    A_log, dt_bias = mx.random.normal((H,)) * 0.5, mx.random.normal((H,)) * 0.5
    state = mx.random.normal((B, H, Dv, Dk)) * 0.1
    return q, k, v, a, b, A_log, dt_bias, state


def loop(q, k, v, a, b, A_log, dt_bias, state):
    return gd.gated_delta_ops(q, k, v, gd.compute_g(A_log, a, dt_bias), mx.sigmoid(b), state)


@pytest.mark.parametrize("T", [1, 63, 64, 150, 257])
def test_forward_matches_loop(T):
    x = inputs(T)
    y0, s0 = loop(*x)
    y1, s1 = cd.gated_delta_update(*x, use_kernel=False)
    assert mx.allclose(y0, y1, atol=1e-4, rtol=1e-3).item()
    assert mx.allclose(s0, s1, atol=1e-4, rtol=1e-3).item()


def test_gradients_match_loop():
    x = inputs(130, seed=1)

    def f(fn):
        def loss(q, k, v, a, b):
            y, s = fn(q, k, v, a, b, *x[5:])
            return (y * mx.cos(mx.arange(y.size).reshape(y.shape))).sum() + s.sum()
        return mx.grad(loss, argnums=(0, 1, 2, 3, 4))(*x[:5])

    g0 = f(loop)
    g1 = f(lambda *z: cd.gated_delta_update(*z, use_kernel=False))
    for a, b in zip(g0, g1):
        assert mx.allclose(a, b, atol=1e-3, rtol=1e-2).item()


def test_inference_path_untouched():
    x = inputs(70)
    y0, _ = gd.gated_delta_kernel(x[0], x[1], x[2], gd.compute_g(x[5], x[3], x[6]), mx.sigmoid(x[4]), x[7])
    y1, _ = cd.gated_delta_update(*x)  # use_kernel=True -> original
    assert mx.allclose(y0, y1).item()


def test_gpu_close():
    x = inputs(300, seed=2)
    y0, s0 = loop(*x)
    y1, s1 = cd.gated_delta_update(*x, use_kernel=False)
    assert (mx.abs(y0 - y1).max() / mx.abs(y0).max()).item() < 5e-3
    assert (mx.abs(s0 - s1).max() / mx.abs(s0).max()).item() < 5e-3


def test_correlated_keys_stay_finite():
    """Near-identical keys, beta ~1, almost no decay: the worst case for the in-chunk solve."""
    q, k, v, a, b, A_log, dt_bias, state = inputs(200, seed=3)
    base = mx.random.normal((1, 1, k.shape[2], k.shape[3]))
    k = base + 0.01 * k
    q, k = gd.normalize_qk(q, k, inv_scale=k.shape[-1] ** -0.5, eps=1e-6)
    b = mx.full(b.shape, 6.0)          # beta = sigmoid(6) ~ 1
    A_log = mx.full(A_log.shape, -8.0)  # decay ~ 1
    y0, s0 = loop(q, k, v, a, b, A_log, dt_bias, state)
    y1, s1 = cd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, use_kernel=False)
    assert mx.isfinite(y1).all().item()
    assert (mx.abs(y0 - y1).max() / mx.abs(y0).max()).item() < 1e-4
    assert (mx.abs(s0 - s1).max() / mx.abs(s0).max()).item() < 1e-4


def test_unit_lower_inverse():
    a = mx.tril(mx.random.normal((3, 64, 64)), -1) * 0.5
    t = cd._unit_lower_inverse(a)
    assert mx.allclose(t @ (mx.eye(64) - a), mx.broadcast_to(mx.eye(64), (3, 64, 64)), atol=1e-3).item()
