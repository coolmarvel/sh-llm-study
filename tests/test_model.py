import torch

from shllm.model import GPT, Block, GPTConfig


def small_cfg(**kw):
    base = dict(vocab_size=50, block_size=8, n_layer=2, n_head=2, n_embd=16)
    base.update(kw)
    return GPTConfig(**base)


def test_block_keeps_shape():
    blk = Block(small_cfg())
    x = torch.randn(3, 5, 16)
    assert blk(x).shape == (3, 5, 16)


def test_gpt_forward_loss_and_param_count():
    torch.manual_seed(0)
    cfg = small_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 6))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 6, cfg.vocab_size)
    assert 3.0 < loss.item() < 5.0  # 초기엔 대략 log V = 3.9 근처
    C, V, T, L = cfg.n_embd, cfg.vocab_size, cfg.block_size, cfg.n_layer
    per_block = (
        (C * 3 * C + 3 * C) + (C * C + C) + 2 * (2 * C) + (C * 4 * C + 4 * C) + (4 * C * C + C)
    )
    expected = (
        V * C + T * C + L * per_block + 2 * C
    )  # 임베딩 + 블록 + ln_f (lm_head 는 묶여 있어 0)
    assert sum(p.numel() for p in model.parameters()) == expected
    assert model.n_params() == expected - T * C


def test_weight_tying():
    model = GPT(small_cfg())
    assert model.lm_head.weight.data_ptr() == model.embed.tok.weight.data_ptr()


def test_causal_property_end_to_end():
    torch.manual_seed(0)
    model = GPT(small_cfg())
    model.eval()
    idx = torch.randint(0, 50, (1, 6))
    logits1, _ = model(idx)
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % 50
    logits2, _ = model(idx2)
    assert torch.allclose(
        logits1[0, :5], logits2[0, :5], atol=1e-5
    )  # 마지막 토큰 변경은 앞자리 예측에 영향 없음
    assert not torch.allclose(logits1[0, 5], logits2[0, 5])


def test_gpt_learns_simple_pattern():
    torch.manual_seed(0)
    cfg = small_cfg(vocab_size=10, block_size=8, n_layer=1, n_head=2, n_embd=32)
    model = GPT(cfg)
    seq = torch.tensor([i % 10 for i in range(200)])  # 0,1,2,…,9,0,1,… 다음 = 현재+1
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(0)
    for _ in range(150):
        s = torch.randint(0, len(seq) - 9, (16,), generator=g)
        x = torch.stack([seq[i : i + 8] for i in s])
        y = torch.stack([seq[i + 1 : i + 9] for i in s])
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2
    maps = model.attention_maps()
    assert len(maps) == 1 and maps[0].shape == (16, 2, 8, 8)


def test_block_size_exceeded():
    model = GPT(small_cfg(block_size=4))
    try:
        model(torch.zeros(1, 5, dtype=torch.long))
    except ValueError:
        return
    raise AssertionError("block_size 초과는 ValueError")
