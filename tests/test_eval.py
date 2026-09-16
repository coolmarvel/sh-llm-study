import math

import torch

from shllm.eval import bits_per_char, evaluate_loss, perplexity, scaling_experiment
from shllm.model import GPT, GPTConfig


def test_perplexity_and_bits_per_char():
    assert perplexity(0.0) == 1.0
    assert math.isclose(perplexity(math.log(10)), 10.0)
    # 토큰당 2 nat, 글자당 토큰 0.5 개 → 글자당 1 nat = 1.4427 bit
    assert math.isclose(bits_per_char(2.0, n_tokens=50, n_chars=100), 1 / math.log(2), rel_tol=1e-9)


def test_evaluate_loss_is_deterministic_and_near_log_v_at_init():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=30, block_size=8, n_layer=1, n_head=2, n_embd=16))
    data = torch.randint(0, 30, (500,))
    a = evaluate_loss(model, data, 8, batch_size=4, n_batches=3)
    b = evaluate_loss(model, data, 8, batch_size=4, n_batches=3)
    assert a == b
    assert abs(a - math.log(30)) < 0.5


def test_scaling_experiment_bigger_is_better_on_pattern(tmp_path):
    seq = torch.tensor([(i * 7) % 13 for i in range(3000)])
    rows = scaling_experiment(
        seq[:2500],
        seq[2500:],
        sizes=[{"n_layer": 1, "n_embd": 8, "n_head": 1}, {"n_layer": 2, "n_embd": 32, "n_head": 2}],
        steps=60,
        vocab_size=13,
        block_size=8,
        batch_size=16,
        lr=1e-2,
        runs_dir=tmp_path / "r",
        ckpt_dir=tmp_path / "c",
        verbose=False,
    )
    assert [r["params"] for r in rows] == sorted(r["params"] for r in rows)
    assert rows[1]["val_loss"] < rows[0]["val_loss"]
    assert set(rows[0]) >= {"n_layer", "n_embd", "params", "val_loss", "seconds"}
