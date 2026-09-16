import math

import numpy as np
import torch

from shllm.autograd import Value
from shllm.numpy_lm import backward, forward, sgd_step, softmax


def test_value_matches_torch_autograd():
    # 같은 식을 Value 와 torch 로 계산해 미분값이 같은지
    a, b, c = Value(1.5), Value(-2.0), Value(0.7)
    out = ((a * b + c) ** 2.0).tanh() * 3 + (a / b).exp() - c.log()
    out.backward()

    ta, tb, tc = (torch.tensor(v, requires_grad=True) for v in (1.5, -2.0, 0.7))
    tout = torch.tanh((ta * tb + tc) ** 2) * 3 + torch.exp(ta / tb) - torch.log(tc)
    tout.backward()

    assert math.isclose(out.data, tout.item(), rel_tol=1e-6)
    for v, t in ((a, ta), (b, tb), (c, tc)):
        assert math.isclose(v.grad, t.grad.item(), rel_tol=1e-5), (v, t.grad)


def test_value_accumulates_when_reused():
    x = Value(3.0)
    y = x * x + x  # x 를 세 번 썼다 → dy/dx = 2x + 1 = 7
    y.backward()
    assert math.isclose(x.grad, 7.0)


def test_relu_blocks_negative():
    x = Value(-1.0)
    y = x.relu() * 5
    y.backward()
    assert y.data == 0 and x.grad == 0


def test_numpy_bigram_gradient_matches_torch():
    rng = np.random.default_rng(0)
    V, N = 7, 20
    W = rng.normal(size=(V, V)).astype(np.float64)
    x = rng.integers(0, V, N)
    y = rng.integers(0, V, N)
    loss, p = forward(W, x, y)
    dW = backward(p, x, y, V)

    tW = torch.tensor(W, requires_grad=True)
    tloss = torch.nn.functional.cross_entropy(tW[torch.tensor(x)], torch.tensor(y))
    tloss.backward()
    assert math.isclose(loss, tloss.item(), rel_tol=1e-6)
    assert np.allclose(dW, tW.grad.numpy(), atol=1e-8)


def test_softmax_rows_sum_to_one_and_sgd_reduces_loss():
    rng = np.random.default_rng(1)
    V = 5
    x = np.array([0, 1, 2, 0, 1, 2] * 10)
    y = np.array([1, 2, 0, 1, 2, 0] * 10)  # 결정적 바이그램: 0→1, 1→2, 2→0
    W = np.zeros((V, V))
    assert np.allclose(softmax(rng.normal(size=(3, V))).sum(axis=1), 1)
    first, _ = forward(W, x, y)
    for _ in range(200):
        _, p = forward(W, x, y)
        W = sgd_step(W, backward(p, x, y, V), lr=5.0)
    last, p = forward(W, x, y)
    assert first == math.log(V)  # 0 행렬 = 균등분포
    assert last < 0.1  # 결정적 규칙은 거의 완벽히 배운다
    assert p[0].argmax() == 1 and p[2].argmax() == 0
