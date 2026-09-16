"""NumPy 로 손으로 쓴 신경 바이그램 (4장) — autograd 없이 순전파와 역전파를 직접 계산한다.

모델: logits = W[x]  (W: (V, V), x 의 행이 "다음 토큰 점수" 벡터)
      p = softmax(logits),  loss = -mean(log p[정답])

1장의 확률표를 "세어서" 만든 것과 달리, 여기서는 랜덤 W 에서 출발해 loss 의 기울기 방향으로 W 를 조금씩
고친다. 충분히 돌리면 1장의 표와 같은 곳에 도착한다 — 학습이란 세는 것의 일반화다.

역전파의 핵심 한 줄: softmax + cross-entropy 를 합치면  d loss / d logits = (p - onehot(y)) / N.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """행마다 softmax. 최댓값을 빼는 것은 exp 가 overflow 하지 않게 하는 관례 (결과는 같다)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(W: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """loss 와 확률 p 를 돌려준다. x, y: (N,) 정수. W: (V, V)."""
    logits = W[x]  # (N, V)  — 룩업 = 원-핫 × W (3장)
    p = softmax(logits)  # (N, V)
    loss = -np.log(
        p[np.arange(len(y)), y]
    ).mean()  # 정답 칸의 log 확률 평균 (1장의 loss 와 같은 식)
    return float(loss), p


def backward(p: np.ndarray, x: np.ndarray, y: np.ndarray, vocab_size: int) -> np.ndarray:
    """d loss / d W. 룩업의 역전파는 "해당 행에 기울기를 더하기" 다 (np.add.at — 같은 행이 여러 번 나오면 누적)."""
    n = len(y)
    dlogits = p.copy()  # (N, V)
    dlogits[np.arange(n), y] -= 1.0  # p - onehot(y)
    dlogits /= n  # mean 의 미분
    dW = np.zeros((vocab_size, vocab_size), dtype=p.dtype)
    np.add.at(dW, x, dlogits)  # dW[x[i]] += dlogits[i]
    return dW


def sgd_step(W: np.ndarray, dW: np.ndarray, lr: float) -> np.ndarray:
    """경사하강 한 걸음: 기울기의 반대 방향으로 lr 만큼."""
    return W - lr * dW
