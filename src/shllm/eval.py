"""평가 (9장) — "잘 됐는지 어떻게 아나".

    evaluate_loss(model, data, block_size, batch_size, n_batches)   val 토큰당 평균 loss (nat)
    perplexity(loss)                                                 exp(loss) — "매 순간 몇 개 중 하나를 찍는 셈인가"
    bits_per_char(loss_per_token, n_tokens, n_chars)                 토큰당 loss 를 글자당 bit 로 환산 — 토크나이저가 달라도 비교 가능
    scaling_experiment(...)                                          모델 크기별로 짧게 학습해 (파라미터 수, val loss) 표를 만든다

토큰당 loss 는 토크나이저가 다르면 비교할 수 없다 (2장). 같은 텍스트를 글자 단위로 환산하면 공정하다:
    bits/char = loss_per_token · (토큰 수 / 글자 수) / ln 2
"""

from __future__ import annotations

import math
import time

import torch

from shllm.data import get_batch
from shllm.model import GPT, GPTConfig
from shllm.train import TrainConfig, Trainer


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    data: torch.Tensor,
    block_size: int,
    batch_size: int = 32,
    n_batches: int = 20,
    seed: int = 0,
) -> float:
    """무작위 배치 n_batches 개의 평균 loss. 시드를 고정해 모델끼리 같은 배치로 비교한다."""
    was_training = model.training
    model.eval()
    g = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, g)
        _, loss = model(x, y)
        total += loss.item()
    if was_training:
        model.train()
    return total / n_batches


def perplexity(loss: float) -> float:
    return math.exp(loss)


def bits_per_char(loss_per_token: float, n_tokens: int, n_chars: int) -> float:
    """토큰당 nat → 글자당 bit. 압축률 관점: 이 모델로 텍스트를 부호화하면 글자당 몇 비트인가."""
    return loss_per_token * (n_tokens / n_chars) / math.log(2)


def scaling_experiment(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    sizes: list[dict],
    steps: int,
    vocab_size: int,
    block_size: int = 64,
    batch_size: int = 32,
    lr: float = 1e-3,
    runs_dir=None,
    ckpt_dir=None,
    verbose: bool = True,
) -> list[dict]:
    """sizes = [{"n_layer": 1, "n_embd": 64, "n_head": 2}, ...] 마다 같은 스텝 수로 학습해 결과 표를 돌려준다."""
    rows = []
    for size in sizes:
        cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size, **size)
        tcfg = TrainConfig(
            run_name=f"scaling-L{cfg.n_layer}-C{cfg.n_embd}",
            batch_size=batch_size,
            max_steps=steps,
            lr=lr,
            min_lr=lr / 10,
            warmup_steps=max(10, steps // 20),
            eval_every=max(50, steps),  # 중간 평가는 생략 (시간 절약) — 처음과 끝만
            eval_batches=8,
            ckpt_every=steps,
        )
        torch.manual_seed(0)
        model = GPT(cfg)
        kw = {}
        if runs_dir is not None:
            kw["runs_dir"] = runs_dir
        if ckpt_dir is not None:
            kw["ckpt_dir"] = ckpt_dir
        trainer = Trainer(model, train_data, val_data, tcfg, **kw)
        t0 = time.perf_counter()
        trainer.train(verbose=False)
        val = evaluate_loss(model, val_data, block_size, batch_size, n_batches=20)
        row = {
            **size,
            "params": model.n_params(),
            "val_loss": round(val, 4),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        rows.append(row)
        if verbose:
            print(
                f"L={cfg.n_layer} C={cfg.n_embd}: {row['params']:>10,} 파라미터  val {val:.3f}  {row['seconds']:.0f}s"
            )
    return rows
