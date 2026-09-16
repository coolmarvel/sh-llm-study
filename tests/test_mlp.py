import torch

from shllm.data import get_batch
from shllm.mlp import MLPLanguageModel, NeuralBigram, train_steps


def test_get_batch_shapes_and_shift():
    data = torch.arange(100)
    g = torch.Generator().manual_seed(0)
    x, y = get_batch(data, block_size=8, batch_size=4, generator=g)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])  # y 는 x 를 한 칸 민 것
    assert torch.equal(y[:, -1], x[:, -1] + 1)


def test_neural_bigram_learns_deterministic_rule():
    torch.manual_seed(0)
    V = 4
    data = torch.tensor([0, 1, 2, 3] * 50)  # 0→1→2→3→0
    model = NeuralBigram(V)
    g = torch.Generator().manual_seed(0)
    losses = train_steps(model, lambda: get_batch(data, 4, 16, g), steps=150, lr=0.1, log_every=0)
    assert losses[-1] < 0.1 < losses[0]
    logits, _ = model(torch.tensor([[2]]))
    assert logits[0, 0].argmax() == 3


def test_mlp_forward_shapes_and_training():
    torch.manual_seed(0)
    V, T = 10, 4
    model = MLPLanguageModel(V, block_size=T, n_embd=8, n_hidden=32)
    x = torch.randint(0, V, (5, T))
    logits, loss = model(x, torch.randint(0, V, (5,)))
    assert logits.shape == (5, V) and loss.ndim == 0
    # 문맥의 마지막 두 토큰 합 mod V 가 다음 토큰인 규칙 — 바이그램은 못 풀고 문맥 2개면 풀린다
    seq = [1, 2]
    for _ in range(600):
        seq.append((seq[-1] + seq[-2]) % V)
    data = torch.tensor(seq)
    g = torch.Generator().manual_seed(0)

    def batch():
        bx, by = get_batch(data, T, 32, g)
        return bx, by[:, -1]  # MLP 는 마지막 자리의 다음 토큰만 맞힌다

    losses = train_steps(model, batch, steps=400, lr=1e-2, log_every=0)
    assert losses[-1] < 0.3
    out = model.generate(
        data[None, :T], max_new_tokens=5, generator=torch.Generator().manual_seed(1)
    )
    assert out.shape == (1, T + 5)
    assert out[0, T:].tolist() == seq[T : T + 5]  # 규칙대로 이어 쓴다
