"""스칼라 자동미분 (4장) — PyTorch 의 autograd 가 하는 일을 숫자 하나짜리 버전으로 직접 만든다.

Value 는 숫자 하나(data)와 "이 값이 최종 결과에 얼마나 영향을 주는가"(grad)를 가진다.
연산을 할 때마다 어떤 값들로부터 어떤 연산으로 만들어졌는지(_prev, _backward)를 기억해 두고,
backward() 는 그 그래프를 거꾸로 따라가며 연쇄법칙으로 grad 를 채운다.

    x = Value(2.0); y = Value(3.0)
    z = x * y + x            # z = 2·3 + 2 = 8
    z.backward()             # dz/dx = y + 1 = 4,  dz/dy = x = 2

torch.Tensor 도 원리는 같다 — 텐서 단위로, C++ 로, 수백 가지 연산에 대해 구현돼 있을 뿐이다.
(Andrej Karpathy 의 micrograd 를 이 교재의 용어로 다시 쓴 것.)
"""

from __future__ import annotations

import math


class Value:
    """숫자 하나 + 그 숫자가 만들어진 이력. 연산자를 오버로드해 파이썬 산술식이 그래프를 만들게 한다."""

    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0  # 최종 출력에 대한 편미분. backward() 가 채운다
        self._backward = lambda: None  # 이 노드의 grad 를 자식들에게 나눠 주는 함수
        self._prev = set(_children)
        self._op = _op

    # --- 연산: 각 연산은 "값 계산" 과 "국소 미분 규칙(_backward)" 두 가지를 정의한다 ---

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(a+b)/da = 1, d(a+b)/db = 1 → 위에서 온 grad 를 그대로 흘린다. += 인 이유: 같은 값을 여러 번 쓰면 기여가 쌓인다
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # d(a·b)/da = b, d(a·b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k: float) -> Value:
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward() -> None:
            self.grad += k * self.data ** (k - 1) * out.grad  # d(a^k)/da = k·a^(k-1)

        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad  # d(e^a)/da = e^a

        out._backward = _backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad  # d(ln a)/da = 1/a

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - t * t) * out.grad  # d tanh/da = 1 - tanh²

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad  # 양수면 통과, 음수면 차단

        out._backward = _backward
        return out

    # 파이썬이 a - b, a / b, 3 * a 같은 식을 위 연산으로 풀 수 있게 하는 보조 정의
    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other if isinstance(other, Value) else -other)

    def __truediv__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1.0

    def __radd__(self, other: float) -> Value:
        return self + other

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __rsub__(self, other: float) -> Value:
        return Value(other) - self

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    # --- 역전파 ---

    def backward(self) -> None:
        """이 값을 최종 출력으로 보고 모든 조상의 grad 를 채운다 (위상 정렬 → 역순으로 _backward 호출)."""
        order: list[Value] = []
        seen: set[Value] = set()

        def visit(v: Value) -> None:  # 자식을 먼저 다 넣고 자신을 넣는다 → 마지막이 출력
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    visit(child)
                order.append(v)

        visit(self)
        self.grad = 1.0  # d(out)/d(out) = 1
        for v in reversed(order):  # 출력에서 입력 쪽으로
            v._backward()
