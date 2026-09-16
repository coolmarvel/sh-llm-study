"""MLP 언어모델 (4장) — Bengio et al. 2003. 어텐션 이전, 신경망으로 "다음 토큰 확률" 을 만든 첫 모델.

    NeuralBigram      W (V, V) 하나. numpy_lm 의 PyTorch 판 — 1장의 확률표를 학습으로 얻는다
    MLPLanguageModel  직전 block_size 개 토큰을 임베딩해 이어 붙이고 → 은닉층(tanh) → 어휘 점수
                      1장 n-gram 과 달리 문맥이 dict 키가 아니라 벡터라, 비슷한 문맥끼리 정보를 공유한다

두 모델 다 forward(idx, targets) → (logits, loss). 6장의 GPT 도 같은 서명을 가진다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralBigram(nn.Module):
    """logits = table[현재 토큰]. 파라미터 V×V — 세어서 만든 1장의 표를 경사하강으로 찾는다."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.table = nn.Embedding(
            vocab_size, vocab_size
        )  # (V, V): 행 = 현재 토큰, 열 = 다음 토큰 점수

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.table(idx)  # (B, T) → (B, T, V)
        loss = None
        if targets is not None:
            # cross_entropy 는 (N, V) 점수와 (N,) 정답을 받는다 → (B, T) 를 펼친다
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class MLPLanguageModel(nn.Module):
    """직전 block_size 토큰 → 다음 토큰. 문맥 길이가 고정이고, 자리마다 다른 가중치를 쓴다(위치별로 따로 배운다)."""

    def __init__(self, vocab_size: int, block_size: int, n_embd: int, n_hidden: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.emb = nn.Embedding(vocab_size, n_embd)  # (V, C)
        self.hidden = nn.Linear(block_size * n_embd, n_hidden)  # 이어 붙인 문맥 (T·C) → 은닉 H
        self.out = nn.Linear(n_hidden, vocab_size)  # H → 어휘 점수 V

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx: (B, T) 이고 T == block_size. 마지막 자리 하나의 다음 토큰만 예측한다
        B, T = idx.shape
        x = self.emb(idx)  # (B, T, C)
        x = x.view(B, T * self.emb.embedding_dim)  # (B, T·C) — 문맥 벡터들을 한 줄로 이어 붙인다
        h = torch.tanh(self.hidden(x))  # (B, H)
        logits = self.out(h)  # (B, V)
        loss = None if targets is None else F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """idx: (1, ≥block_size). 마지막 block_size 토큰으로 다음을 뽑아 이어 붙인다."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size :])
            probs = F.softmax(logits, dim=-1)  # (1, V)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def train_steps(
    model: nn.Module,
    get_batch,
    steps: int,
    lr: float = 1e-3,
    log_every: int = 100,
) -> list[float]:
    """가장 단순한 학습 루프. 7장에서 스케줄·체크포인트·평가가 붙은 완성판으로 다시 쓴다.

    get_batch() → (x, y). 매 스텝: 순전파 → loss → 기울기 0 초기화 → 역전파 → 파라미터 갱신.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)  # 기울기는 누적되므로(Value 의 += 처럼) 매번 비운다
        loss.backward()  # autograd 가 모든 파라미터의 .grad 를 채운다
        opt.step()  # 기울기 반대 방향으로 한 걸음
        losses.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"step {step:>5}  loss {loss.item():.3f}")
    return losses
