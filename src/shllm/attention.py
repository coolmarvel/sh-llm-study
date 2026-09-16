"""셀프 어텐션 (5장) — "문맥의 어디를 볼지" 를 내용으로 정한다.

    attention(q, k, v, mask)   공식 그 자체: softmax(Q Kᵀ / √d) V — 루프 없는 텐서 식
    AttentionHead              헤드 하나: 입력 (B, T, C) → Q, K, V 를 만들어 attention 을 적용 → (B, T, head_size)
    CausalSelfAttention        여러 헤드를 한 번에(행렬 하나로 Q·K·V 를 뽑아 헤드 축으로 쪼갬) + 출력 투영. 6장 GPT 가 쓰는 판

"causal"(인과) 마스크: 자리 t 는 t 이하만 본다. 미래 토큰을 보면 "다음 토큰 예측" 이 컨닝이 되기 때문이다.
어텐션은 위치를 모른다 — 순서 정보는 3장 위치 임베딩이 입력에 이미 더해져 있다.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k, v: (..., T, d). 돌려주는 것: (출력 (..., T, d), 어텐션 가중치 (..., T, T)).

    1. scores[i, j] = q_i · k_j / √d   — "자리 i 가 자리 j 를 얼마나 볼까" 의 원점수. √d 로 나누는 이유:
       d 가 크면 내적의 분산이 d 에 비례해 커져 softmax 가 한 곳에 몰린다(기울기 소실).
    2. mask 가 False 인 칸은 -inf → softmax 후 0 (못 본다)
    3. weights = softmax(scores)       — 행마다 합이 1 인 "시선 배분"
    4. out_i = Σ_j weights[i, j] · v_j  — 본 만큼 값을 섞는다
    """
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)  # (..., T, T)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)  # (..., T, T)
    return weights @ v, weights  # (..., T, d), (..., T, T)


def causal_mask(T: int) -> torch.Tensor:
    """(T, T) 하삼각 True. mask[i, j] = (j <= i) — 자리 i 는 자기 자신과 과거만 본다."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


class AttentionHead(nn.Module):
    """헤드 하나. 입력 벡터를 세 가지 역할로 투영한다: 질문(Q)·꼬리표(K)·내용(V)."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # register_buffer: 파라미터는 아니지만 모델과 함께 저장·이동되는 텐서 (마스크는 학습 대상이 아니다)
        self.register_buffer("mask", causal_mask(block_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)  # 각 (B, T, head_size)
        out, weights = attention(
            q, k, v, self.mask[:T, :T]
        )  # 마스크는 (T, T) → 배치로 브로드캐스팅
        return self.dropout(out), weights


class CausalSelfAttention(nn.Module):
    """멀티헤드 어텐션 (GPT-2 방식). n_head 개의 헤드가 각각 C/n_head 차원을 맡고, 결과를 이어 붙여 다시 C 로 투영한다.

    헤드를 여러 개 두는 이유: 한 헤드는 "직전 토큰" 을, 다른 헤드는 "같은 조사" 를 보는 식으로 서로 다른 관계를 병렬로 본다.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd={n_embd} 는 n_head={n_head} 로 나누어떨어져야 합니다")
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(
            n_embd, 3 * n_embd
        )  # Q, K, V 를 행렬 하나로 한 번에 (세 번 곱하는 것과 같지만 빠르다)
        self.proj = nn.Linear(n_embd, n_embd)  # 헤드들을 이어 붙인 뒤 섞는 출력 투영
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("mask", causal_mask(block_size))
        self.last_weights: torch.Tensor | None = (
            None  # 시각화용: 마지막 forward 의 (B, n_head, T, T)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)  # (B, T, 3C) → 세 개의 (B, T, C)
        # 헤드 축을 만든다: (B, T, C) → (B, T, n_head, head_size) → (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        out, weights = attention(
            q, k, v, self.mask[:T, :T]
        )  # (B, n_head, T, head_size), (B, n_head, T, T)
        self.last_weights = weights.detach()
        out = self.attn_dropout(out)
        out = out.transpose(1, 2).reshape(
            B, T, C
        )  # 헤드를 다시 이어 붙인다 (B, T, n_head·head_size = C)
        return self.resid_dropout(self.proj(out))
