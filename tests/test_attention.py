import torch

from shllm.attention import AttentionHead, CausalSelfAttention, attention, causal_mask


def test_attention_formula_matches_loop():
    torch.manual_seed(0)
    T, d = 5, 4
    q, k, v = torch.randn(T, d), torch.randn(T, d), torch.randn(T, d)
    out, w = attention(q, k, v, causal_mask(T))
    assert out.shape == (T, d) and w.shape == (T, T)
    assert torch.allclose(w.sum(dim=-1), torch.ones(T))  # 행마다 합 1
    assert torch.equal(torch.triu(w, diagonal=1), torch.zeros(T, T))  # 미래는 0
    # 루프로 같은 계산
    for i in range(T):
        scores = torch.tensor([float(q[i] @ k[j]) / d**0.5 for j in range(i + 1)])
        wi = torch.softmax(scores, dim=0)
        assert torch.allclose(out[i], sum(wi[j] * v[j] for j in range(i + 1)), atol=1e-5)


def test_attention_is_order_invariant_without_positions():
    # 마스크 없이, 입력 순서를 섞어도 각 토큰의 출력은 같다 (어텐션 자체는 순서를 모른다)
    torch.manual_seed(0)
    x = torch.randn(6, 8)
    perm = torch.randperm(6)
    out, _ = attention(x, x, x)
    out_p, _ = attention(x[perm], x[perm], x[perm])
    assert torch.allclose(out[perm], out_p, atol=1e-6)


def test_causal_head_future_does_not_affect_past():
    torch.manual_seed(0)
    head = AttentionHead(n_embd=8, head_size=4, block_size=10)
    x = torch.randn(1, 6, 8)
    out1, _ = head(x)
    x2 = x.clone()
    x2[0, 5] = torch.randn(8)  # 마지막 토큰만 바꿈
    out2, _ = head(x2)
    assert torch.allclose(out1[0, :5], out2[0, :5])  # 앞자리 출력은 그대로
    assert not torch.allclose(out1[0, 5], out2[0, 5])


def test_multihead_shapes_and_weights():
    torch.manual_seed(0)
    attn = CausalSelfAttention(n_embd=16, n_head=4, block_size=8)
    x = torch.randn(2, 5, 16)
    y = attn(x)
    assert y.shape == (2, 5, 16)
    w = attn.last_weights
    assert w.shape == (2, 4, 5, 5)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 5))
    assert (torch.triu(w, diagonal=1) == 0).all()
    # 파라미터 수: qkv (C·3C + 3C) + proj (C·C + C)
    C = 16
    assert sum(p.numel() for p in attn.parameters()) == C * 3 * C + 3 * C + C * C + C


def test_multihead_rejects_bad_head_count():
    try:
        CausalSelfAttention(n_embd=10, n_head=4, block_size=4)
    except ValueError:
        return
    raise AssertionError("n_embd 가 n_head 로 안 나눠지면 ValueError 여야 한다")
