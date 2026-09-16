import pytest
import torch

from shllm.embedding import (
    GPTEmbedding,
    LearnedPositionalEmbedding,
    TokenEmbedding,
    cooccurrence_matrix,
    nearest,
    one_hot_lookup,
    ppmi,
    sinusoidal_positions,
    svd_embeddings,
)


def test_token_embedding_is_one_hot_matmul():
    torch.manual_seed(0)
    emb = TokenEmbedding(vocab_size=10, n_embd=4)
    idx = torch.tensor([[1, 2, 3], [9, 0, 1]])  # (B=2, T=3)
    out = emb(idx)
    assert out.shape == (2, 3, 4)
    assert torch.allclose(out, one_hot_lookup(idx, emb.weight))
    assert torch.equal(out[0, 0], emb.weight[1])  # id 1 의 행 그대로


def test_sinusoidal_positions_properties():
    pe = sinusoidal_positions(16, 8)
    assert pe.shape == (16, 8)
    assert pe.abs().max() <= 1.0
    assert torch.allclose(pe[0, 0::2], torch.zeros(4))  # sin(0)=0
    assert torch.allclose(pe[0, 1::2], torch.ones(4))  # cos(0)=1
    # 서로 다른 위치는 서로 다른 벡터
    assert not torch.allclose(pe[3], pe[4])


def test_learned_positions_respect_block_size():
    pos = LearnedPositionalEmbedding(block_size=8, n_embd=4)
    assert pos(5).shape == (5, 4)
    with pytest.raises(ValueError):
        pos(9)


def test_gpt_embedding_shape_params_and_grad():
    V, T_max, C = 50, 8, 16
    emb = GPTEmbedding(V, T_max, C)
    assert sum(p.numel() for p in emb.parameters()) == V * C + T_max * C
    idx = torch.randint(0, V, (3, 6))  # (B=3, T=6)
    x = emb(idx)
    assert x.shape == (3, 6, C)
    # 같은 토큰이 다른 자리에 있으면 다른 벡터
    same = torch.full((1, 2), 7)
    y = emb(same)
    assert not torch.allclose(y[0, 0], y[0, 1])
    # 역전파가 두 표 모두에 닿는다
    x.sum().backward()
    assert emb.tok.weight.grad is not None and emb.pos.weight.grad is not None


def test_cooccurrence_counts():
    ids = torch.tensor([0, 1, 2, 0, 1])
    c = cooccurrence_matrix(ids, vocab_size=3, window=1)
    assert torch.equal(c, c.T)  # 대칭
    assert c[0, 1] == 2 and c[1, 2] == 1 and c[2, 0] == 1 and c[0, 2] == 1
    assert c[0, 0] == 0
    c2 = cooccurrence_matrix(ids, vocab_size=3, window=2)
    assert c2[0, 2] == 1 + 1  # 거리 1: (2,0) 한 번 + 거리 2: (0,2) 한 번
    assert c2[0, 1] == 2 + 1  # 거리 1: (0,1) 두 번 + 거리 2: (1,0) 한 번


def test_ppmi_nonnegative_and_zero_where_unseen():
    counts = torch.tensor([[0.0, 5.0, 1.0], [5.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    m = ppmi(counts)
    assert (m >= 0).all()
    assert m[0, 0] == 0 and m[2, 2] == 0  # 한 번도 같이 안 나온 쌍은 0
    # (0,1): 5/14 ÷ (6/14·6/14) ≈ 1.94 배,  (1,2): 1/14 ÷ (6/14·2/14) ≈ 1.17 배 → 우연 대비 더 자주 붙는 쌍이 크다
    assert m[0, 1] > m[1, 2] > 0


def test_svd_embeddings_find_distributional_neighbors():
    # a 와 c 는 항상 b 앞에 온다 (같은 문맥) → 가장 가까운 이웃이어야 한다. d 는 e 와만 붙어 다닌다
    torch.manual_seed(0)
    seq = [0, 1, 2, 1] * 30 + [3, 4] * 30
    counts = cooccurrence_matrix(torch.tensor(seq), vocab_size=5, window=1)
    emb = svd_embeddings(ppmi(counts), n_embd=3)
    assert emb.shape == (5, 3)
    assert nearest(emb, 0, k=1)[0][0] == 2
    assert nearest(emb, 2, k=1)[0][0] == 0
    assert all(j != 0 for j, _ in nearest(emb, 0, k=4))  # 자기 자신은 제외
