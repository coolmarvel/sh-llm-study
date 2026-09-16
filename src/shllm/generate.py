"""텍스트 생성 (8장) — 확률에서 토큰을 뽑는 방법들.

    generate(model, idx, max_new_tokens, temperature, top_k, top_p, repetition_penalty)
        GPT 의 forward 를 반복해 한 토큰씩 붙인다. 6장 노트북부터 미리 쓰고 8장에서 뜯어본다.
    next_token_distribution(model, idx, ...)   한 스텝의 (조정된) 확률 — 대시보드 "토큰별 확률" 뷰용
    generate_text(model, tok, prompt, ...)     문자열 → 문자열 편의 함수

모든 조정은 logits(점수) 단계에서 한다. 순서: 반복 억제 → temperature → top-k → top-p → softmax → multinomial.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def adjust_logits(
    logits: torch.Tensor,
    context: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """(B, V) 점수를 조정해 돌려준다. 잘려 나간 토큰은 -inf (softmax 후 0)."""
    logits = logits.clone()
    if repetition_penalty != 1.0:
        # 이미 나온 토큰의 점수를 깎는다 (양수면 나누고 음수면 곱해서 항상 "덜 뽑히게")
        for b in range(logits.size(0)):
            seen = torch.unique(context[b])
            s = logits[b, seen]
            logits[b, seen] = torch.where(s > 0, s / repetition_penalty, s * repetition_penalty)
    if temperature <= 0:
        # temperature 0 = 탐욕(greedy): 최고 점수만 남긴다
        best = logits.argmax(dim=-1, keepdim=True)
        out = torch.full_like(logits, float("-inf"))
        return out.scatter(1, best, 0.0)
    logits = logits / temperature  # <1 이면 분포가 뾰족해지고(안전), >1 이면 평평해진다(모험)
    if top_k is not None and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k).values[:, -1, None]  # k 번째 점수
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        # 누적 확률이 top_p 를 넘는 꼬리를 자른다 (nucleus sampling). 확률 순으로 정렬해 계산
        sorted_logits, order = torch.sort(logits, descending=True)
        cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        drop = (
            cum - F.softmax(sorted_logits, dim=-1) > top_p
        )  # 자기 자신을 넣기 전 누적이 이미 p 를 넘으면 버린다
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, order, sorted_logits)
    return logits


@torch.no_grad()
def next_token_distribution(
    model: torch.nn.Module,
    idx: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """idx (B, T) 의 다음 토큰 확률 (B, V). block_size 를 넘는 앞부분은 잘라 낸다."""
    block = getattr(model, "cfg", None).block_size if hasattr(model, "cfg") else idx.size(1)
    logits, _ = model(idx[:, -block:])
    logits = logits[:, -1, :]  # 마지막 자리만 (B, V)
    return F.softmax(
        adjust_logits(logits, idx, temperature, top_k, top_p, repetition_penalty), dim=-1
    )


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """idx (B, T) 뒤에 max_new_tokens 개를 붙여 (B, T+n) 을 돌려준다."""
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        probs = next_token_distribution(model, idx, temperature, top_k, top_p, repetition_penalty)
        nxt = torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)
        idx = torch.cat([idx, nxt], dim=1)
    if was_training:
        model.train()
    return idx


def generate_text(model: torch.nn.Module, tok, prompt: str, max_new_tokens: int = 100, **kw) -> str:
    idx = torch.tensor([tok.encode(prompt)])
    return tok.decode(generate(model, idx, max_new_tokens, **kw)[0].tolist())
