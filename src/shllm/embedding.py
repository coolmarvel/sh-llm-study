"""임베딩 (3장), 정수 토큰 id 를 신경망이 다룰 수 있는 벡터로.

TokenEmbedding              (V, C) 룩업 테이블. 원-핫 벡터 × 행렬과 같지만 행을 바로 꺼낸다
sinusoidal_positions        학습하지 않는 위치 벡터 (원조 Transformer 방식)
LearnedPositionalEmbedding  학습하는 위치 벡터 (GPT-2 방식), (T_max, C)
GPTEmbedding                토큰 임베딩 + 위치 임베딩 → (B, T, C). 6장 GPT 의 입구

cooccurrence_matrix · ppmi · svd_embeddings · nearest
    학습 없이 "세어서" 벡터를 만드는 고전 방법. 1장의 (V, V) 확률표를 C 차원으로 압축한 것이
    임베딩이라는 감각을 얻기 위한 도구다. 실제 GPT 의 임베딩은 4장부터 경사하강으로 학습된다.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    """토큰 id → 벡터. weight[id] 한 줄이 그 토큰의 벡터다. nn.Embedding 을 직접 구현한 것."""

    def __init__(self, vocab_size: int, n_embd: int) -> None:
        super().__init__()
        # nn.Parameter: "이 텐서는 학습 대상" 표시. optimizer 가 찾아서 갱신한다 (4장)
        self.weight = nn.Parameter(torch.randn(vocab_size, n_embd) * 0.02)  # (V, C)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) 정수 → (B, T, C). 정수 텐서로 인덱싱하면 각 id 의 행을 그 자리에 꽂아 준다
        return self.weight[idx]


def one_hot_lookup(idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """같은 계산을 원-핫 × 행렬로. TokenEmbedding 이 하는 일이 정확히 이것임을 보이는 용도 (느리다)."""
    one_hot = F.one_hot(idx, num_classes=weight.shape[0]).to(weight.dtype)  # (B, T, V)
    return one_hot @ weight  # (B, T, V) @ (V, C) → (B, T, C)


def sinusoidal_positions(n_positions: int, n_embd: int) -> torch.Tensor:
    """(T, C) 위치 벡터. 열마다 주기가 다른 sin/cos, 짧은 주기는 이웃 위치를, 긴 주기는 먼 위치를 구분한다."""
    pos = torch.arange(n_positions, dtype=torch.float32)[:, None]  # (T, 1)
    i = torch.arange(0, n_embd, 2, dtype=torch.float32)  # (C/2,) 짝수 열 번호
    freq = torch.exp(
        -math.log(10000.0) * i / n_embd
    )  # 1 → 1/10000 로 기하급수적으로 낮아지는 주파수
    angle = pos * freq  # (T, C/2)
    out = torch.zeros(n_positions, n_embd)
    out[:, 0::2] = torch.sin(angle)  # 짝수 열
    out[:, 1::2] = torch.cos(angle)  # 홀수 열
    return out


class LearnedPositionalEmbedding(nn.Module):
    """위치 0..T_max-1 마다 벡터 하나를 학습한다 (GPT-2). 문맥 길이 T_max 를 넘는 입력은 처리할 수 없다."""

    def __init__(self, block_size: int, n_embd: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.weight = nn.Parameter(torch.randn(block_size, n_embd) * 0.02)  # (T_max, C)

    def forward(self, n_positions: int) -> torch.Tensor:
        if n_positions > self.block_size:
            raise ValueError(
                f"길이 {n_positions} > block_size {self.block_size}: 문맥 길이를 넘었습니다"
            )
        return self.weight[:n_positions]  # (T, C)


class GPTEmbedding(nn.Module):
    """GPT 의 입구: x = 토큰 임베딩 + 위치 임베딩.

    같은 토큰이라도 몇 번째 자리에 있느냐에 따라 다른 벡터가 되게 한다. 더하기(연결이 아니라)로 합쳐도
    C 차원이 충분히 크면 두 정보가 섞이지 않고 공존한다, 6장에서 이 벡터가 Transformer 블록으로 들어간다.
    """

    def __init__(self, vocab_size: int, block_size: int, n_embd: int) -> None:
        super().__init__()
        self.tok = TokenEmbedding(vocab_size, n_embd)
        self.pos = LearnedPositionalEmbedding(block_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        tok = self.tok(idx)  # (B, T, C)
        pos = self.pos(T)  # (T, C), 배치마다 같으니 B 축 없이 두고 브로드캐스팅으로 더한다
        return tok + pos  # (B, T, C)


# ---------- 세어서 만드는 벡터 (학습 없음) ----------


def cooccurrence_matrix(ids: torch.Tensor, vocab_size: int, window: int = 2) -> torch.Tensor:
    """(V, V) 동시출현 횟수. counts[a, b] = a 의 앞뒤 window 칸 안에 b 가 나온 횟수 (대칭)."""
    ids = torch.as_tensor(ids, dtype=torch.int64)
    counts = torch.zeros(vocab_size, vocab_size)
    for d in range(1, window + 1):
        a, b = ids[:-d], ids[d:]  # 1장 바이그램과 같은 "어긋나게 겹치기" 를 거리 d 로
        ones = torch.ones(len(a))
        counts.index_put_((a, b), ones, accumulate=True)
        counts.index_put_((b, a), ones, accumulate=True)
    return counts


def ppmi(counts: torch.Tensor) -> torch.Tensor:
    """양의 점별 상호정보량. PMI(a,b) = log P(a,b) / (P(a) P(b)). "우연보다 얼마나 더 자주 같이 나오나".

    횟수를 그대로 쓰면 '는'·',' 같은 흔한 토큰이 모든 행을 지배한다. PMI 는 각 토큰의 빈도로 나눠 그 영향을 뺀다.
    음수(우연보다 덜 같이 나옴)는 정보가 불안정해 0 으로 자른다 (Positive PMI).
    """
    total = counts.sum()
    p_ab = counts / total  # (V, V)
    p_a = counts.sum(dim=1, keepdim=True) / total  # (V, 1)
    p_b = counts.sum(dim=0, keepdim=True) / total  # (1, V)
    pmi = torch.log(p_ab.clamp(min=1e-12) / (p_a * p_b).clamp(min=1e-12))
    pmi[counts == 0] = 0.0
    return pmi.clamp(min=0)


def svd_embeddings(matrix: torch.Tensor, n_embd: int) -> torch.Tensor:
    """(V, V) 행렬을 SVD 로 (V, C) 벡터로 압축. 행렬 ≈ E @ E'ᵀ 가 되게 하는 E, 상위 C 개 방향만 남긴다."""
    U, S, _ = torch.svd_lowrank(matrix, q=n_embd, niter=4)  # 전체 SVD 보다 훨씬 빠른 근사
    return U * S.sqrt()  # (V, C)


def nearest(emb: torch.Tensor, index: int, k: int = 5) -> list[tuple[int, float]]:
    """코사인 유사도로 index 토큰과 가장 가까운 k 개 (자기 자신 제외). [(id, 유사도), ...]"""
    unit = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-8)  # 각 행을 길이 1 로
    sims = unit @ unit[index]  # (V,) 내적 = 코사인
    top = torch.topk(sims, k + 1).indices.tolist()
    return [(j, float(sims[j])) for j in top if j != index][:k]
