import torch

from shllm.generate import adjust_logits, generate, next_token_distribution
from shllm.model import GPT, GPTConfig


def test_temperature_and_greedy():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    ctx = torch.zeros(1, 1, dtype=torch.long)
    hot = torch.softmax(adjust_logits(logits, ctx, temperature=0.5), -1)
    flat = torch.softmax(adjust_logits(logits, ctx, temperature=2.0), -1)
    assert hot[0, 0] > flat[0, 0]  # 낮은 temperature → 1등에 더 몰린다
    greedy = torch.softmax(adjust_logits(logits, ctx, temperature=0.0), -1)
    assert torch.allclose(greedy, torch.tensor([[1.0, 0.0, 0.0]]))


def test_top_k_and_top_p():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    ctx = torch.zeros(1, 1, dtype=torch.long)
    p = torch.softmax(adjust_logits(logits, ctx, top_k=2), -1)
    assert p[0, 2] == 0 and p[0, 3] == 0 and abs(p[0, 0] - 0.625) < 1e-5
    p = torch.softmax(adjust_logits(logits, ctx, top_p=0.7), -1)  # 0.5 + 0.3 = 0.8 ≥ 0.7 → 두 개만
    assert (p[0, :2] > 0).all() and p[0, 2] == 0 and p[0, 3] == 0
    p = torch.softmax(
        adjust_logits(logits, ctx, top_p=0.9), -1
    )  # 0.5+0.3 = 0.8 < 0.9 ≤ 0.95 → 세 개
    assert p[0, 2] > 0 and p[0, 3] == 0


def test_repetition_penalty_lowers_seen_tokens():
    logits = torch.tensor([[1.0, 1.0, -1.0]])
    ctx = torch.tensor([[0, 2]])  # 토큰 0 과 2 는 이미 나왔다
    out = adjust_logits(logits, ctx, repetition_penalty=2.0)
    assert out[0, 0] == 0.5 and out[0, 1] == 1.0 and out[0, 2] == -2.0


def test_generate_shapes_and_determinism():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, n_embd=16))
    idx = torch.randint(0, 20, (2, 3))
    out = generate(model, idx, 10, top_k=5, generator=torch.Generator().manual_seed(0))
    assert out.shape == (2, 13) and torch.equal(out[:, :3], idx)
    out2 = generate(model, idx, 10, top_k=5, generator=torch.Generator().manual_seed(0))
    assert torch.equal(out, out2)
    # 문맥이 block_size 를 넘어도 생성이 된다 (앞을 잘라 쓴다)
    long = generate(model, idx, 20, generator=torch.Generator().manual_seed(0))
    assert long.shape == (2, 23)
    probs = next_token_distribution(model, long)
    assert probs.shape == (2, 20) and torch.allclose(probs.sum(-1), torch.ones(2))
