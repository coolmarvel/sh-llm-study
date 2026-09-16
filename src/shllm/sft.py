"""지시 파인튜닝 미니 시연 (10장) — base 모델을 "질문 → 답" 형식으로 추가 학습한다.

8장 모델은 이어 쓰기만 한다. 대화형 모델은 같은 모델에 **형식이 있는 데이터**를 조금 더 학습시킨 것이다(SFT,
supervised fine-tuning). 여기서는 코퍼스의 작품 목록(작가·제목)으로 질문-답 쌍을 만들어 시연한다 — 진짜 대화
데이터는 없지만, "형식을 배운다" 는 것이 무엇인지와 그 한계(모르는 것도 그럴듯하게 답함 = 환각)를 보기엔 충분하다.

    PROMPT / ANSWER   형식 토큰. 모델이 "여기부터 답" 을 알아보게 하는 약속
    build_examples    (작가, 제목) 목록 → 질문-답 문자열들
    finetune          base 모델을 예제로 몇 스텝 더 학습 (답 부분에만 loss)
    answer            질문을 형식에 맞춰 넣고 줄바꿈까지 생성
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from shllm.generate import generate

PROMPT = "질문: "
ANSWER = "\n답: "
END = "\n"

TEMPLATES = [
    ("{title}의 작가는 누구인가?", "{author}"),
    ("{title}은 누가 썼나?", "{author}"),
    ("{author}이 쓴 작품을 하나 말해 보라.", "{title}"),
    ("{title}을 쓴 사람은?", "{author}"),
]


def build_examples(works: list[tuple[str, str]]) -> list[str]:
    """works = [(작가, 제목), ...] → "질문: …\\n답: …\\n" 문자열 목록."""
    out = []
    for author, title in works:
        for q, a in TEMPLATES:
            out.append(
                PROMPT
                + q.format(title=title, author=author)
                + ANSWER
                + a.format(title=title, author=author)
                + END
            )
    return out


def _encode_example(tok, text: str) -> tuple[list[int], list[int]]:
    """토큰 ids 와 마스크(답 부분 = 1). 질문 부분에는 loss 를 주지 않는다 — 질문을 외우는 게 목적이 아니다."""
    q_end = text.index(ANSWER) + len(ANSWER)
    q_ids = tok.encode(text[:q_end])
    a_ids = tok.encode(text[q_end:])
    return q_ids + a_ids, [0] * len(q_ids) + [1] * len(a_ids)


def finetune(
    model: torch.nn.Module,
    tok,
    examples: list[str],
    steps: int = 200,
    lr: float = 1e-4,
    batch_size: int = 8,
    seed: int = 0,
    log_every: int = 50,
) -> list[float]:
    """예제를 무작위로 뽑아 답 부분의 cross-entropy 로 base 모델을 이어 학습한다. 돌려주는 것: 스텝별 loss."""
    encoded = [_encode_example(tok, e) for e in examples]
    block = model.cfg.block_size
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    model.train()
    losses = []
    for step in range(steps):
        picks = torch.randint(0, len(encoded), (batch_size,), generator=g).tolist()
        T = min(block, max(len(encoded[i][0]) for i in picks))
        x = torch.zeros(batch_size, T, dtype=torch.long)
        y = torch.full(
            (batch_size, T), -100, dtype=torch.long
        )  # -100 = cross_entropy 가 무시하는 자리
        for row, i in enumerate(picks):
            ids, mask = encoded[i]
            ids, mask = ids[:T], mask[:T]
            x[row, : len(ids)] = torch.tensor(ids)
            # 자리 t 의 정답은 t+1 토큰, 답 부분만 남긴다
            for t in range(len(ids) - 1):
                if mask[t + 1]:
                    y[row, t] = ids[t + 1]
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"step {step:>4}  loss {loss.item():.3f}")
    model.eval()
    return losses


@torch.no_grad()
def answer(model: torch.nn.Module, tok, question: str, max_new_tokens: int = 20) -> str:
    """형식에 맞춰 질문을 넣고 greedy 로 생성, 첫 줄바꿈까지가 답."""
    prompt = PROMPT + question + ANSWER
    idx = torch.tensor([tok.encode(prompt)])
    out = generate(model, idx, max_new_tokens, temperature=0.0)
    text = tok.decode(out[0, idx.size(1) :].tolist())
    return text.split("\n")[0].strip()
