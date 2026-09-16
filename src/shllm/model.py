"""Transformer 블록 → GPT (6장). nanoGPT / GPT-2 의 구조를 그대로, 읽히게.

    GPTConfig   모델 크기를 정하는 숫자들 (어휘·문맥·층·헤드·폭·드롭아웃)
    MLP         위치별 2층 신경망 (C → 4C → C). 어텐션이 "모은" 정보를 자리마다 가공한다
    Block       x = x + attn(ln1(x));  x = x + mlp(ln2(x))   — 잔차(residual) + Pre-LayerNorm
    GPT         임베딩(3장) → Block × n_layer → LayerNorm → 어휘 점수. forward(idx, targets) → (logits, loss)

파라미터 수 대부분은 (a) 토큰 임베딩 V·C 와 (b) 블록당 12·C² (어텐션 4C² + MLP 8C²) 에 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from shllm.attention import CausalSelfAttention
from shllm.embedding import GPTEmbedding


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 256  # 최대 문맥 길이 T_max
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256  # C
    dropout: float = 0.0


class MLP(nn.Module):
    """자리마다 독립으로 적용되는 2층 신경망. 4장 MLP 와 같은 모양이지만 문맥을 이어 붙이지 않는다 — 문맥은 어텐션이 이미 섞었다."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)  # 4배로 넓혔다가
        self.proj = nn.Linear(4 * n_embd, n_embd)  # 다시 C 로
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))  # GELU: ReLU 의 부드러운 판 (GPT-2)


class Block(nn.Module):
    """Transformer 블록 하나. 입력과 출력 shape 이 같아(B, T, C) 몇 개든 쌓을 수 있다."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg.n_embd, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 잔차 연결: 블록은 x 를 "고치는 양" 만 배운다. 층이 깊어도 기울기가 x 를 타고 곧장 흐른다
        x = x + self.attn(self.ln1(x))  # 토큰들 사이의 소통
        x = x + self.mlp(self.ln2(x))  # 토큰 각자의 계산
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = GPTEmbedding(cfg.vocab_size, cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # (C → V) 어휘 점수
        # 가중치 묶기(weight tying): 입력 임베딩 (V, C) 과 출력 행렬 (V, C) 를 같은 텐서로. 파라미터 V·C 절약 + 성능 소폭 향상
        self.lm_head.weight = self.embed.tok.weight
        self.apply(self._init_weights)
        # GPT-2: 잔차로 들어가는 투영은 층 수에 맞춰 더 작게 초기화 (깊이가 더해질수록 분산이 커지는 것을 막는다)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def n_params(self, non_embedding: bool = True) -> int:
        """파라미터 수. non_embedding=True 면 위치 임베딩을 뺀다 (토큰 임베딩은 lm_head 와 묶여 있어 그대로 센다 — GPT-2 관례)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.pos.weight.numel()
        return n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"입력 길이 {T} > block_size {self.cfg.block_size}")
        x = self.drop(self.embed(idx))  # (B, T, C)
        for block in self.blocks:
            x = block(x)  # (B, T, C) 유지
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V) — 모든 자리에서 다음 토큰 점수를 한 번에
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def attention_maps(self) -> list[torch.Tensor]:
        """마지막 forward 의 층별 어텐션 가중치 [(B, n_head, T, T), ...] — 시각화·대시보드용."""
        return [b.attn.last_weights for b in self.blocks]
