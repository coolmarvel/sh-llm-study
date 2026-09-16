"""n-gram 통계 언어모델 (1장). "다음 글자 확률표" 하나로 텍스트를 생성한다.

언어모델은 결국 P(다음 토큰 | 지금까지의 토큰들) 을 계산하는 함수다 (0장).
이 모듈은 그 함수를 가장 단순한 방법, 즉 **세어서 나누기** 로 만든다.

    BigramModel  직전 토큰 1개만 본다. 확률표가 (V, V) 텐서 하나라서 눈으로 볼 수 있다.
    NGramModel   직전 토큰 n-1 개를 본다. 표가 V^(n-1) 행이 되어 텐서로 못 들고 dict 로 든다.
                 n=1 이면 유니그램(문맥 없음), n=2 면 BigramModel 과 같은 모델이다.

두 클래스는 같은 인터페이스를 가진다:
    fit(ids)                    정수 토큰 열로 횟수를 센다
    next_token_probs(context)   (V,) 확률 벡터
    generate(start, n)          확률대로 하나씩 뽑아 이어 붙인다
    loss(ids)                   평균 음의 로그가능도 (작을수록 좋다, 단위 nat)
    perplexity(ids)             exp(loss). "매 순간 몇 개 중에 하나를 고르는 셈인가"

4장 이후의 신경망 모델도 같은 세 가지(확률 계산·생성·손실)를 하며, 달라지는 것은
"세어서 나누기" 대신 "파라미터를 학습해서" 확률표를 만든다는 점뿐이다.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch


class BigramModel:
    """바이그램 모델: counts[a, b] = 토큰 a 바로 다음에 토큰 b 가 나온 횟수.

    smoothing(가산 스무딩) 은 모든 칸에 미리 더해 두는 가짜 횟수다. 0 이면 한 번도 안 나온
    쌍의 확률이 0 이 되어 loss 가 무한대로 터진다, 학습에 없던 조합이 검증 텍스트에는 있기 때문.
    교과서의 "라플라스 스무딩" 은 1 을 더하지만, 어휘가 수천 자면 행마다 가짜 횟수가 수천 개 생겨
    실제 횟수를 묻어 버린다 (노트북 01 에서 확인). 그래서 기본값은 0.01 이다.
    """

    def __init__(self, vocab_size: int, smoothing: float = 0.01) -> None:
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.counts = torch.zeros(vocab_size, vocab_size, dtype=torch.int64)  # (V, V)
        self._probs: torch.Tensor | None = None  # 확률표 캐시. counts 가 바뀌면 버린다

    def fit(self, ids: list[int] | torch.Tensor) -> BigramModel:
        """토큰 열에서 (앞, 뒤) 쌍을 모두 센다. 파이썬 루프 없이 텐서 연산 한 번으로 끝낸다."""
        ids = torch.as_tensor(ids, dtype=torch.int64)
        prev, nxt = ids[:-1], ids[1:]  # 같은 열을 한 칸 어긋나게 겹치면 모든 인접 쌍이 나온다
        # counts[prev[i], nxt[i]] += 1 을 i 전체에 대해. accumulate=True 가 없으면 중복 쌍이 1 로 덮인다
        self.counts.index_put_((prev, nxt), torch.ones_like(prev), accumulate=True)
        self._probs = None
        return self

    @property
    def probs(self) -> torch.Tensor:
        """(V, V) 확률표. 행 = 현재 토큰, 열 = 다음 토큰. 각 행의 합은 1."""
        if self._probs is None:
            p = self.counts.float() + self.smoothing
            empty = p.sum(dim=1) == 0  # smoothing=0 이고 한 번도 앞자리에 안 나온 토큰 → 0/0 방지
            p[empty] = 1.0  # 그런 행은 균등분포로 둔다
            self._probs = p / p.sum(dim=1, keepdim=True)  # (V, V) / (V, 1) → 행마다 나눔
        return self._probs

    def next_token_probs(self, context: list[int] | int) -> torch.Tensor:
        """직전 토큰 하나만 보고 다음 토큰 확률 (V,) 을 돌려준다. context 의 앞부분은 무시된다."""
        token = context if isinstance(context, int) else context[-1]
        return self.probs[token]

    def generate(
        self, start: list[int] | int, max_new_tokens: int, generator: torch.Generator | None = None
    ) -> list[int]:
        """start 뒤에 max_new_tokens 개를 확률대로 뽑아 이어 붙인다 (start 포함해서 돌려준다)."""
        out = [start] if isinstance(start, int) else list(start)
        for _ in range(max_new_tokens):
            p = self.next_token_probs(out)  # (V,)
            # multinomial: 확률 벡터를 주면 그 분포대로 인덱스 하나를 뽑는다 (주사위 던지기)
            out.append(int(torch.multinomial(p, num_samples=1, generator=generator)))
        return out

    def loss(self, ids: list[int] | torch.Tensor) -> float:
        """평균 음의 로그가능도: -(1/N) Σ log P(ids[i+1] | ids[i]). 7장의 cross-entropy 와 같은 양이다."""
        ids = torch.as_tensor(ids, dtype=torch.int64)
        logp = torch.log(self.probs[ids[:-1], ids[1:]])  # (N-1,) 실제 나온 쌍마다 log 확률
        return float(-logp.mean())

    def perplexity(self, ids: list[int] | torch.Tensor) -> float:
        return math.exp(self.loss(ids))


class NGramModel:
    """n-gram 모델: 직전 n-1 개 토큰(문맥)마다 "다음 토큰 횟수" 를 따로 센다.

    counts[k][문맥][다음 토큰] = 횟수. k 는 문맥 길이(0 ~ n-1), 문맥은 튜플이라 dict 키로 쓸 수 있다
    (자바의 Map<List<Integer>, Map<Integer, Integer>>). 가능한 문맥이 V^(n-1) 가지라 n=3 만 돼도
    텐서로는 못 든다, 실제로 등장한 문맥만 dict 에 담는다. 짧은 문맥의 표도 전부 같이 센다(표가 n 개).

    스무딩은 BigramModel 과 다르다. 균등분포(아무 글자나) 대신 **한 단계 짧은 문맥의 분포** 를 섞는다:

        P_k(t | 문맥 k개) = ( count_k(문맥, t) + α · P_{k-1}(t | 문맥 k-1개) ) / ( total_k(문맥) + α )
        P_0(t)            = ( count_0(t) + α / V ) / ( N + α )                 ← 유니그램은 균등분포와 섞는다

    긴 문맥을 한 번도 못 봤으면(total=0) 자동으로 짧은 문맥의 답이 되고, 한두 번 본 문맥이면 그 횟수와
    짧은 문맥의 답이 α 만큼의 비율로 섞인다. 균등분포에 섞으면 어휘가 수천 자일 때 실제 횟수가
    묻히지만(노트북 01), 이 방식은 어휘 크기와 무관하다. α = "짧은 문맥을 가짜 횟수 몇 개만큼 믿을지".
    """

    def __init__(self, n: int, vocab_size: int, smoothing: float = 1.0) -> None:
        if n < 1:
            raise ValueError("n 은 1 이상이어야 합니다 (1 = 유니그램)")
        self.n = n
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        # defaultdict: 없는 키를 읽으면 기본값을 만들어 넣는다. Map.computeIfAbsent 와 같다
        self.counts: list[dict[tuple[int, ...], dict[int, int]]] = [
            defaultdict(lambda: defaultdict(int)) for _ in range(n)
        ]
        self.totals: list[dict[tuple[int, ...], int]] = [defaultdict(int) for _ in range(n)]

    def fit(self, ids: list[int] | torch.Tensor) -> NGramModel:
        ids = [int(i) for i in ids]
        for k in range(self.n):  # 문맥 길이 0, 1, ..., n-1 을 각각 센다
            counts, totals = self.counts[k], self.totals[k]
            for i in range(k, len(ids)):
                ctx = tuple(ids[i - k : i])  # 직전 k 개. k=0 이면 빈 튜플 = 문맥 없음
                counts[ctx][ids[i]] += 1
                totals[ctx] += 1
        return self

    @property
    def n_contexts(self) -> int:
        """가장 긴 문맥(n-1 개)이 실제로 등장한 가짓수. V^(n-1) 과 비교하면 표가 얼마나 비었는지 보인다."""
        return len(self.totals[self.n - 1])

    def _ctx(self, tokens: list[int], k: int) -> tuple[int, ...]:
        return tuple(tokens[len(tokens) - k :]) if k else ()

    def _probs_vec(self, tokens: list[int], k: int) -> torch.Tensor:
        """문맥 k 개짜리 확률 벡터 (V,). 위 식을 k=0 까지 재귀로 내려가며 계산한다."""
        a = self.smoothing
        base = (
            torch.full((self.vocab_size,), 1 / self.vocab_size)
            if k == 0
            else self._probs_vec(tokens, k - 1)
        )
        p = a * base  # (V,) 가짜 횟수 α 개를 짧은 문맥의 분포 모양으로 나눠 준다
        ctx = self._ctx(tokens, k)
        for tok, c in self.counts[k].get(ctx, {}).items():
            p[tok] += c
        return p / (self.totals[k].get(ctx, 0) + a)

    def _prob(self, tokens: list[int], k: int, target: int) -> float:
        """_probs_vec 과 같은 식을 토큰 하나(target)에 대해서만, loss 계산용 (벡터를 만들면 느리다)."""
        a = self.smoothing
        base = 1 / self.vocab_size if k == 0 else self._prob(tokens, k - 1, target)
        ctx = self._ctx(tokens, k)
        c = self.counts[k].get(ctx, {}).get(target, 0)
        return (c + a * base) / (self.totals[k].get(ctx, 0) + a)

    def next_token_probs(self, context: list[int] | int) -> torch.Tensor:
        """(V,) 확률 벡터. 문맥이 n-1 개보다 짧으면 있는 만큼만 쓴다."""
        tokens = [context] if isinstance(context, int) else list(context)
        return self._probs_vec(tokens, min(self.n - 1, len(tokens)))

    def generate(
        self, start: list[int] | int, max_new_tokens: int, generator: torch.Generator | None = None
    ) -> list[int]:
        out = [start] if isinstance(start, int) else list(start)
        for _ in range(max_new_tokens):
            p = self.next_token_probs(out)
            out.append(int(torch.multinomial(p, num_samples=1, generator=generator)))
        return out

    def loss(self, ids: list[int] | torch.Tensor) -> float:
        """평균 음의 로그가능도 -(1/(N-1)) Σ log P(ids[i] | 앞 토큰들). 첫 토큰은 문맥이 없어 뺀다."""
        ids = [int(i) for i in ids]
        total = 0.0
        for i in range(1, len(ids)):
            tokens = ids[max(0, i - self.n + 1) : i]
            total -= math.log(self._prob(tokens, len(tokens), ids[i]))
        return total / (len(ids) - 1)

    def perplexity(self, ids: list[int] | torch.Tensor) -> float:
        return math.exp(self.loss(ids))
